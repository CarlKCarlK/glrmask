//! Publish exact partitioned tokenizers with retained terminal/runtime coordinates.

use crate::automata::lexer::tokenizer::Lexer;
use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::dfa::DFA;
use crate::automata::lexer::tokenizer::{
    TerminalExclusionCertificate,
    TerminalExclusionResidualState,
    TerminalResidualCoordinates,
    Tokenizer,
};
use crate::ds::bitset::BitSet;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;
use super::{Regex, compile_with_plan, compile_with_plan_internal_options};
use super::deferred::{DeferredDfa, try_compile_with_plan_deferred_dense_min_pair_cells};
use super::dfa_analysis::{single_group_dfa_max_remaining, single_group_dfa_state_is_live};
use super::partition::isolate_component_nullable_start;
use super::plan::{
    NestedGroupOpCache,
    build_exclusion_compile_plan_with_labels_and_cache,
    materialize_repeated_subexpression_dfas,
    prewarm_shared_duplicate_nested_group_ops,
    shared_duplicate_nested_group_op_cache,
};
use super::product::build_product_dfa;
use super::settings::adaptive_lexer_enabled;
use rayon::prelude::*;

/// Compile-time tokenizer construction from already-compiled single-terminal
/// DFAs. Besides the exact tokenizer, retain for every raw
/// tokenizer state the product coordinates `(terminal, terminal_dfa_state)`.
///
/// This deliberately bypasses adaptive component determinization: the retained
/// coordinate is the point of this path, and adaptive products would need to
/// carry the same trace explicitly rather than reconstructing it afterward.
pub fn build_partitioned_tokenizer_from_precompiled_terminal_dfas(
    exprs: &[Expr],
    partitions: &[u32],
    retained_exprs: Arc<[Expr]>,
) -> Option<Tokenizer> {
    if exprs.len() != partitions.len() || exprs.len() != retained_exprs.len() {
        return None;
    }
    let terminal_dfas = exprs
        .iter()
        .map(|expr| match expr {
            Expr::Dfa(dfa) => Some(Arc::clone(dfa)),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;

    let mut grouped = BTreeMap::<u32, Vec<usize>>::new();
    for (terminal, &partition) in partitions.iter().enumerate() {
        grouped.entry(partition).or_default().push(terminal);
    }

    struct ComponentWithCoordinates {
        terminal_ids: Vec<usize>,
        dfa: DFA,
        rows: Vec<Vec<(u32, u32)>>,
    }

    let components = grouped
        .into_iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(_, terminal_ids)| -> Option<ComponentWithCoordinates> {
            if terminal_ids.len() == 1 {
                let terminal = terminal_ids[0];
                let dfa = terminal_dfas.get(terminal)?.as_ref().clone();
                let rows = (0..dfa.num_states())
                    .map(|state| vec![(terminal as u32, state as u32)])
                    .collect();
                return Some(ComponentWithCoordinates {
                    terminal_ids,
                    dfa,
                    rows,
                });
            }

            // The traced builder historically placed every terminal in the
            // product independently. Large schema imports contain many exactly
            // identical single-terminal DFAs, so that needlessly multiplies the
            // product coordinate even though those terminals can never diverge.
            // DFA equality is exact structural equality (including group
            // metadata), making it safe to share one product coordinate and
            // fan its observation/residual state back out to every alias.
            let mut unique_dfas = Vec::<Arc<DFA>>::new();
            let mut unique_members = Vec::<Vec<(usize, usize)>>::new();
            let mut unique_by_dfa = FxHashMap::<Arc<DFA>, usize>::default();
            for (local_group, &terminal) in terminal_ids.iter().enumerate() {
                let terminal_dfa = Arc::clone(terminal_dfas.get(terminal)?);
                let unique = if let Some(&unique) = unique_by_dfa.get(&terminal_dfa) {
                    unique
                } else {
                    let unique = unique_dfas.len();
                    unique_by_dfa.insert(Arc::clone(&terminal_dfa), unique);
                    unique_dfas.push(Arc::clone(terminal_dfas.get(terminal)?));
                    unique_members.push(Vec::new());
                    unique
                };
                unique_members[unique].push((local_group, terminal));
            }
            if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
                eprintln!(
                    "[glrmask/profile][precompiled_terminal_dfa_dedup] terminals={} unique={} saved={}",
                    terminal_ids.len(),
                    unique_dfas.len(),
                    terminal_ids.len().saturating_sub(unique_dfas.len()),
                );
            }
            let local_exprs = unique_dfas
                .iter()
                .map(|dfa| Expr::Dfa(Arc::clone(dfa)))
                .collect::<Vec<_>>();
            let empty_ops = BTreeMap::<u32, BTreeSet<u32>>::new();
            let (mut dfa, group_ops_applied, trace) = build_product_dfa(
                &local_exprs,
                None,
                local_exprs.len(),
                &empty_ops,
                &empty_ops,
                true,
                true,
                false,
                false,
            );
            if group_ops_applied {
                return None;
            }
            // `build_product_dfa` labels states by unique-DFA coordinate. Expand
            // those exact labels back to the original terminal-local groups.
            // This preserves the tokenizer's public terminal observations even
            // though identical terminals share transition topology.
            let compact_metadata = (0..dfa.num_states() as u32)
                .map(|state| {
                    (
                        dfa.finalizers(state).clone(),
                        dfa.possible_future_group_ids(state).clone(),
                    )
                })
                .collect::<Vec<_>>();
            dfa.ensure_group_capacity(terminal_ids.len());
            for (state, (compact_finalizers, compact_futures)) in
                compact_metadata.into_iter().enumerate()
            {
                let mut finalizers = BitSet::new(terminal_ids.len());
                for unique in compact_finalizers.iter() {
                    for &(local_group, _) in unique_members.get(unique)? {
                        finalizers.set(local_group);
                    }
                }
                let mut futures = BitSet::new(terminal_ids.len());
                for unique in compact_futures.iter() {
                    for &(local_group, _) in unique_members.get(unique)? {
                        futures.set(local_group);
                    }
                }
                dfa.overwrite_state_metadata(state as u32, finalizers, futures);
            }
            for (local_group, &terminal) in terminal_ids.iter().enumerate() {
                dfa.set_group_u8set(
                    local_group as u32,
                    *terminal_dfas[terminal].group_id_to_u8set(0),
                );
            }
            let trace = trace?;
            if trace.state_tuples.len() != dfa.num_states() {
                return None;
            }
            let mut rows = Vec::with_capacity(dfa.num_states());
            for state in 0..dfa.num_states() {
                let tuple = trace.state_tuples.tuple(state);
                let mut row = Vec::with_capacity(tuple.len());
                for (coordinate, residual_state) in tuple {
                    for &(_, terminal) in unique_members.get(coordinate as usize)? {
                        row.push((terminal as u32, residual_state));
                    }
                }
                row.sort_unstable();
                rows.push(row);
            }
            Some(ComponentWithCoordinates {
                terminal_ids,
                dfa,
                rows,
            })
        })
        .collect::<Option<Vec<_>>>()?;

    let mut combined = DFA::new(1);
    combined.ensure_group_capacity(exprs.len());
    let mut root_futures = BitSet::new(exprs.len());
    let mut rows = vec![Vec::<(u32, u32)>::new()];
    for component in components {
        for local_group in component.dfa.possible_future_group_ids(0).iter() {
            root_futures.set(component.terminal_ids[local_group]);
        }
        let offset = combined.append_rebased_component(component.dfa, &component.terminal_ids);
        if offset as usize != rows.len() {
            return None;
        }
        combined.add_epsilon_transition(0, offset);
        rows.extend(component.rows);
    }
    combined.set_possible_future_group_ids(0, root_futures);

    let mut tokenizer = Regex { dfa: combined }.into_tokenizer(
        exprs.len() as u32,
        Some(retained_exprs),
    );
    tokenizer.set_terminal_residual_coordinates(
        TerminalResidualCoordinates::from_rows_and_dfas(rows, terminal_dfas),
    );
    Some(tokenizer)
}

/// Build the direct-L1 sidecar from the same product components used by normal
/// partition compilation. This avoids both recompiling every terminal DFA and
/// projecting terminal residual coordinates back out of the finished tokenizer
/// after the fact.
pub fn build_partitioned_tokenizer_with_product_trace_terminal_residuals(
    exprs: &[Expr],
    visible_labels: Option<&[String]>,
    partitions: &[u32],
    residual_isolation_classes: Option<&[Option<u32>]>,
    retained_exprs: Arc<[Expr]>,
    adaptive: Option<bool>,
    collapse_traced_duplicate_coordinates: bool,
) -> Option<Tokenizer> {
    if exprs.len() != partitions.len() || exprs.len() != retained_exprs.len() || exprs.is_empty() {
        return None;
    }
    if adaptive.unwrap_or_else(adaptive_lexer_enabled) {
        // Cross-partition adaptive products need their own trace composition.
        // The default lexer path is non-adaptive; fail closed for explicit
        // adaptive diagnostics rather than silently changing that topology.
        return None;
    }
    if let Some(labels) = visible_labels
        && labels.len() != exprs.len()
    {
        return None;
    }
    if let Some(classes) = residual_isolation_classes
        && classes.len() != exprs.len()
    {
        return None;
    }

    let started_at = Instant::now();
    let rewritten_exprs = materialize_repeated_subexpression_dfas(exprs);
    let exprs = rewritten_exprs.as_deref().unwrap_or(exprs);
    let mut grouped = BTreeMap::<u32, Vec<usize>>::new();
    for (terminal, &partition) in partitions.iter().enumerate() {
        grouped.entry(partition).or_default().push(terminal);
    }
    if let Some(classes) = residual_isolation_classes {
        for terminal_ids in grouped.values() {
            let mut class = None;
            let mut has_unprotected = false;
            for &terminal in terminal_ids {
                match classes[terminal] {
                    Some(current) => match class {
                        Some(previous) if previous != current => return None,
                        Some(_) => {}
                        None => class = Some(current),
                    },
                    None => has_unprotected = true,
                }
            }
            if class.is_some() && has_unprotected {
                return None;
            }
        }
    }
    let shared_duplicates = shared_duplicate_nested_group_op_cache(exprs, &grouped);
    if let Some(shared_duplicates) = &shared_duplicates {
        prewarm_shared_duplicate_nested_group_ops(shared_duplicates);
    }
    struct TracedComponent {
        terminal_ids: Vec<usize>,
        dfa: Arc<DFA>,
        rows: Vec<Vec<(u32, u32)>>,
        sources: Vec<(usize, Arc<DFA>, u32, Option<Arc<TerminalExclusionCertificate>>)>,
    }

    struct ComplexTerminalSource {
        source: Arc<DFA>,
        mapping: Vec<u32>,
        exclusion: Option<Arc<TerminalExclusionCertificate>>,
    }

    let components = grouped
        .into_iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(_, terminal_ids)| -> Option<TracedComponent> {
            let local_exprs = terminal_ids
                .iter()
                .map(|&terminal| exprs[terminal].clone())
                .collect::<Vec<_>>();
            let local_labels = visible_labels.map(|labels| {
                terminal_ids
                    .iter()
                    .map(|&terminal| labels[terminal].clone())
                    .collect::<Vec<_>>()
            });
            let mut nested_group_op_cache = NestedGroupOpCache {
                shared_duplicates: shared_duplicates.clone(),
                ..NestedGroupOpCache::default()
            };
            let plan = build_exclusion_compile_plan_with_labels_and_cache(
                &local_exprs,
                local_labels.as_deref(),
                &mut nested_group_op_cache,
            );
            let visible_groups = plan.visible_groups;
            if visible_groups != terminal_ids.len() {
                return None;
            }
            let mut complex = vec![false; visible_groups];
            for &group in plan.exclusions.keys().chain(plan.intersections.keys()) {
                if let Some(slot) = complex.get_mut(group as usize) {
                    *slot = true;
                }
            }
            let complex_group_count = complex.iter().filter(|&&is_complex| is_complex).count();

            let compiled_group_count = plan.compiled_exprs.len();
            // Preserve only the logical group-op wiring for the rare complex
            // visible groups. After product compilation we rebuild their
            // standalone expression from the trace's already-compiled DFA
            // coordinates, avoiding any recursive recompilation of the source
            // expressions.
            let mut complex_ops = vec![None::<(Vec<u32>, Vec<u32>)>; visible_groups];
            for local_group in 0..visible_groups {
                if !complex[local_group] {
                    continue;
                }
                let exclusions = plan
                    .exclusions
                    .get(&(local_group as u32))
                    .into_iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>();
                let intersections = plan
                    .intersections
                    .get(&(local_group as u32))
                    .into_iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>();
                complex_ops[local_group] = Some((exclusions, intersections));
            }

            let (dfa, trace) = compile_with_plan_internal_options(
                plan,
                true,
                true,
                collapse_traced_duplicate_coordinates,
            );
            let dfa = Arc::new(dfa);
            if visible_groups == 1 {
                let terminal = terminal_ids[0];
                let rows = (0..dfa.num_states() as u32)
                    .map(|state| {
                        (dfa.finalizers(state).contains(0)
                            || dfa.possible_future_group_ids(state).contains(0))
                            .then_some(vec![(terminal as u32, state)])
                            .unwrap_or_default()
                    })
                    .collect();
                return Some(TracedComponent {
                    terminal_ids,
                    dfa: Arc::clone(&dfa),
                    rows,
                    sources: vec![(terminal, dfa, 0, None)],
                });
            }

            let trace = trace?;
            if trace.state_tuples.len() != dfa.num_states() {
                return None;
            }
            let mut coordinate_for_logical_group = vec![usize::MAX; compiled_group_count];
            for (coordinate, groups) in trace.coordinate_groups.iter().enumerate() {
                for &group in groups {
                    if group >= compiled_group_count {
                        continue;
                    }
                    if coordinate_for_logical_group[group] != usize::MAX {
                        return None;
                    }
                    coordinate_for_logical_group[group] = coordinate;
                }
            }
            if coordinate_for_logical_group
                .iter()
                .any(|&coordinate| coordinate == usize::MAX)
            {
                return None;
            }
            // Materialize the sparse product trace once as a dense
            // state×physical-coordinate table. Complex visible terminals below
            // are themselves small products of these same coordinates, so this
            // lets us translate a partition state directly into the standalone
            // product tuple without replaying every byte edge for every terminal.
            let partition_coordinate_states = if complex_group_count == 0 {
                None
            } else {
                let mut states = vec![vec![u32::MAX; trace.components.len()]; dfa.num_states()];
                for state in 0..dfa.num_states() {
                    for (coordinate, residual_state) in trace.state_tuples.tuple(state) {
                        let slot = states.get_mut(state)?.get_mut(coordinate as usize)?;
                        *slot = residual_state;
                    }
                }
                Some(states)
            };
            let mut simple_sources = Vec::<Option<Arc<DFA>>>::with_capacity(visible_groups);
            for local_group in 0..visible_groups {
                let coordinate = coordinate_for_logical_group[local_group];
                let source = (!complex[local_group])
                    .then(|| trace.components[coordinate].terminal_residual_dfa_arc())
                    .flatten();
                simple_sources.push(source);
            }

            // A visible terminal affected by an exclusion/intersection cannot
            // use one raw product coordinate directly: its language is the
            // group operation over several coordinates. Compile only those rare
            // exceptional terminals independently, then propagate their exact
            // residual state through the already-built partition DFA. This is
            // deliberately strict: every reachable partition state must agree
            // on the standalone residual coordinate and on terminal liveness.
            // Any disagreement fails closed to the established projected path.
            let mut complex_sources = (0..visible_groups)
                .into_par_iter()
                .map(|local_group| -> Option<Option<ComplexTerminalSource>> {
                if simple_sources[local_group].is_some() {
                    return Some(None);
                }
                let (exclusions, intersections) = complex_ops[local_group].as_ref()?;
                let source_compile_started = Instant::now();
                // The already-built partition product contains every component
                // coordinate needed by this exclusion/intersection terminal.
                // Project only those coordinates, deduplicate the resulting
                // tuples, then minimize that much smaller residual DFA. This
                // avoids recompiling the group expression and avoids running
                // Hopcroft once per terminal over the whole partition DFA.
                let mut relevant_coordinates = SmallVec::<[usize; 4]>::new();
                for logical_group in std::iter::once(local_group)
                    .chain(exclusions.iter().map(|&group| group as usize))
                    .chain(intersections.iter().map(|&group| group as usize))
                {
                    let coordinate = *coordinate_for_logical_group.get(logical_group)?;
                    if !relevant_coordinates.contains(&coordinate) {
                        relevant_coordinates.push(coordinate);
                    }
                }
                relevant_coordinates.sort_unstable();

                let partition_live = |state: u32| {
                    dfa.finalizers(state).contains(local_group)
                        || dfa.possible_future_group_ids(state).contains(local_group)
                };
                if !partition_live(0) {
                    // The group operation is the empty language. Keep an
                    // exact canonical dead residual instead of discarding the
                    // successful traced partition and rebuilding via the
                    // fallback lexer.
                    let mut source = DFA::new(1);
                    source.ensure_group_capacity(1);
                    source.set_group_u8set(0, *dfa.group_id_to_u8set(local_group as u32));
                    return Some(Some(ComplexTerminalSource {
                        source: Arc::new(source),
                        mapping: vec![u32::MAX; dfa.num_states()],
                        exclusion: None,
                    }));
                }

                let mut raw_state_by_key = FxHashMap::<SmallVec<[u32; 4]>, u32>::default();
                let mut raw_representatives = Vec::<u32>::new();
                let mut partition_to_raw = vec![u32::MAX; dfa.num_states()];
                let coordinate_states = partition_coordinate_states.as_ref()?;
                for partition_state in 0..dfa.num_states() {
                    if !partition_live(partition_state as u32) {
                        continue;
                    }
                    let key = relevant_coordinates
                        .iter()
                        .map(|&coordinate| coordinate_states[partition_state][coordinate])
                        .collect::<SmallVec<[u32; 4]>>();
                    let raw_state = if let Some(&existing) = raw_state_by_key.get(&key) {
                        existing
                    } else {
                        let next = raw_representatives.len() as u32;
                        raw_state_by_key.insert(key, next);
                        raw_representatives.push(partition_state as u32);
                        next
                    };
                    partition_to_raw[partition_state] = raw_state;
                }
                if partition_to_raw[0] != 0 {
                    return None;
                }

                let mut raw = DFA::new(raw_representatives.len());
                raw.ensure_group_capacity(1);
                raw.set_group_u8set(0, *dfa.group_id_to_u8set(local_group as u32));
                for (raw_state, &representative) in raw_representatives.iter().enumerate() {
                    let mut transitions = Vec::<(u8, u32)>::new();
                    for (byte, target) in dfa.transitions(representative) {
                        let mapped = partition_to_raw[target as usize];
                        if mapped != u32::MAX {
                            transitions.push((byte, mapped));
                        }
                    }
                    raw.set_transitions_from_sorted_entries(raw_state as u32, transitions);
                    let mut finalizers = BitSet::new(1);
                    if dfa.finalizers(representative).contains(local_group) {
                        finalizers.set(0);
                    }
                    raw.overwrite_state_metadata(raw_state as u32, finalizers, BitSet::new(1));
                }
                raw.recompute_possible_futures();
                // The projection is already an exact deterministic residual DFA.
                // Minimizing it here is unnecessary for the proof sidecar and, on
                // the large exclusion families that dominate the dynamic-build
                // tail, does not remove any states. Keep the exact projection and
                // its direct partition-state mapping instead of paying another
                // Hopcroft pass for every exceptional terminal.
                let source = Arc::new(raw);
                let mapping = partition_to_raw;
                let source_compile_ms = source_compile_started.elapsed().as_secs_f64() * 1000.0;
                if source.has_epsilon_transitions() {
                    return None;
                }

                let source_live = |state: u32| {
                    source.finalizers(state).contains(0)
                        || source.possible_future_group_ids(state).contains(0)
                };
                if mapping.len() != dfa.num_states()
                    || partition_live(0) != source_live(*mapping.first()?)
                {
                    return None;
                }

                let mapping_started = Instant::now();
                for partition_state in 0..dfa.num_states() {
                    let source_state = *mapping.get(partition_state)?;
                    let partition_is_live = partition_live(partition_state as u32);
                    let source_is_live = source_state != u32::MAX && source_live(source_state);
                    if partition_is_live != source_is_live {
                        return None;
                    }
                }

                // Optional Layer-2 certificate for the narrow, common shape
                // `Exclude(left, finite_right)`. The exact product trace gives
                // us left/right residual coordinates for every representative
                // of the standalone terminal residual DFA. Failure here drops
                // only the optimization sidecar; tokenizer correctness does not
                // depend on this certificate.
                let exclusion = (|| -> Option<Arc<TerminalExclusionCertificate>> {
                    if exclusions.len() != 1 || !intersections.is_empty() {
                        return None;
                    }
                    let terminal = *terminal_ids.get(local_group)?;
                    if !matches!(retained_exprs.get(terminal)?, Expr::Exclude { .. }) {
                        return None;
                    }
                    let left_coordinate = *coordinate_for_logical_group.get(local_group)?;
                    let right_group = *exclusions.first()? as usize;
                    let right_coordinate = *coordinate_for_logical_group.get(right_group)?;
                    if left_coordinate == right_coordinate {
                        return None;
                    }
                    let left_dfa = trace.components.get(left_coordinate)?.terminal_residual_dfa_arc()?;
                    let right_dfa = trace.components.get(right_coordinate)?.terminal_residual_dfa_arc()?;
                    let right_max_remaining = single_group_dfa_max_remaining(right_dfa.as_ref())?;
                    let coordinate_states = partition_coordinate_states.as_ref()?;
                    let mut residual_states = Vec::with_capacity(raw_representatives.len());
                    for &representative in &raw_representatives {
                        let coordinate_row = coordinate_states.get(representative as usize)?;
                        let left_state = *coordinate_row.get(left_coordinate)?;
                        if left_state == u32::MAX
                            || !single_group_dfa_state_is_live(left_dfa.as_ref(), left_state)
                        {
                            return None;
                        }
                        let raw_right_state = *coordinate_row.get(right_coordinate)?;
                        let (right_state, max_remaining) = if raw_right_state == u32::MAX
                            || !single_group_dfa_state_is_live(right_dfa.as_ref(), raw_right_state)
                        {
                            (u32::MAX, u32::MAX)
                        } else {
                            let remaining = *right_max_remaining
                                .get(raw_right_state as usize)?
                                .as_ref()?;
                            (raw_right_state, remaining)
                        };
                        residual_states.push(TerminalExclusionResidualState {
                            left_state,
                            right_state,
                            right_max_remaining: max_remaining,
                        });
                    }
                    Some(Arc::new(TerminalExclusionCertificate::new(
                        left_dfa,
                        residual_states,
                    )?))
                })();

                if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
                    eprintln!(
                        "[glrmask/profile][product_trace_exceptional_terminal] local_group={} exclusions={} intersections={} source_states={} source_transitions={} source_compile_ms={:.3} mapping_ms={:.3}",
                        local_group,
                        exclusions.len(),
                        intersections.len(),
                        source.num_states(),
                        source.transition_count(),
                        source_compile_ms,
                        mapping_started.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                Some(Some(ComplexTerminalSource {
                    source,
                    mapping,
                    exclusion,
                }))
            })
            .collect::<Option<Vec<_>>>()?;

            let mut rows = Vec::with_capacity(dfa.num_states());
            for state in 0..dfa.num_states() {
                let tuple = trace.state_tuples.tuple(state);
                let mut row = Vec::<(u32, u32)>::with_capacity(
                    tuple.len().saturating_add(complex_group_count),
                );
                if complex_group_count != visible_groups {
                    for &(coordinate, residual_state) in &tuple {
                        for &local_group in trace.coordinate_groups.get(coordinate as usize)? {
                            if local_group >= visible_groups || simple_sources[local_group].is_none() {
                                continue;
                            }
                            row.push((terminal_ids[local_group] as u32, residual_state));
                        }
                    }
                }
                if complex_group_count != 0 {
                    for local_group in 0..visible_groups {
                        if simple_sources[local_group].is_some() {
                            continue;
                        }
                        let complex_source = complex_sources[local_group].as_ref()?;
                        let residual_state = complex_source.mapping[state];
                        if residual_state != u32::MAX {
                            row.push((terminal_ids[local_group] as u32, residual_state));
                        }
                    }
                }
                if complex_group_count != visible_groups {
                    row.sort_unstable_by_key(|&(terminal, _)| terminal);
                }
                rows.push(row);
            }

            let mut sources = Vec::with_capacity(visible_groups);
            for local_group in 0..visible_groups {
                let terminal = terminal_ids[local_group];
                if let Some(source) = simple_sources[local_group].take() {
                    sources.push((terminal, source, 0, None));
                } else {
                    let complex_source = complex_sources[local_group].take()?;
                    sources.push((
                        terminal,
                        complex_source.source,
                        0,
                        complex_source.exclusion,
                    ));
                }
            }
            Some(TracedComponent {
                terminal_ids,
                dfa,
                rows,
                sources,
            })
        })
        .collect::<Option<Vec<_>>>()?;

    let total_states = 1usize
        + components
            .iter()
            .map(|component| component.dfa.num_states())
            .sum::<usize>();
    let mut combined = DFA::new(1);
    combined.ensure_group_capacity(exprs.len());
    let mut root_futures = BitSet::new(exprs.len());
    let mut rows = Vec::<Vec<(u32, u32)>>::with_capacity(total_states);
    rows.push(Vec::new());
    let mut terminal_dfas = vec![None::<Arc<DFA>>; exprs.len()];
    let mut terminal_groups = vec![u32::MAX; exprs.len()];
    let mut exclusion_certificates = vec![None::<Arc<TerminalExclusionCertificate>>; exprs.len()];
    let mut offset = 1u32;
    for component in components {
        for (terminal, source, group, exclusion) in component.sources {
            if terminal >= exprs.len() || terminal_dfas[terminal].is_some() {
                return None;
            }
            terminal_dfas[terminal] = Some(source);
            terminal_groups[terminal] = group;
            exclusion_certificates[terminal] = exclusion;
        }
        for local_group in component.dfa.possible_future_group_ids(0).iter() {
            root_futures.set(component.terminal_ids[local_group]);
        }
        combined.add_epsilon_transition(0, offset);
        let actual_offset = combined.append_rebased_component_ref(
            component.dfa.as_ref(),
            &component.terminal_ids,
        );
        if actual_offset != offset {
            return None;
        }
        rows.extend(component.rows);
        offset += component.dfa.num_states() as u32;
    }
    combined.set_possible_future_group_ids(0, root_futures);
    let terminal_dfas = terminal_dfas.into_iter().collect::<Option<Vec<_>>>()?;
    if terminal_groups.iter().any(|&group| group == u32::MAX) || rows.len() != combined.num_states() {
        return None;
    }
    let mut tokenizer = Regex { dfa: combined }.into_tokenizer(exprs.len() as u32, Some(retained_exprs));
    let coordinates = TerminalResidualCoordinates::from_rows_and_dfa_groups(
        rows,
        terminal_dfas,
        terminal_groups,
    )
    .with_exclusion_certificates(exclusion_certificates)?;
    tokenizer.set_terminal_residual_coordinates(coordinates);
    if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
        eprintln!(
            "[glrmask/profile][product_trace_terminal_residual_coordinates] terminals={} states={} elapsed_ms={:.3}",
            exprs.len(),
            tokenizer.num_states(),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(tokenizer)
}

/// Build an exact partitioned tokenizer while keeping very large pure binary
/// intersection transition tables in their compressed runtime form.
///
/// Unlike the structural compile/runtime pair, this path does not construct a
/// synthesized tokenizer or a full-to-synthesized state map. It is intended
/// for consumers, such as direct dynamic masking, that need only the exact
/// runtime tokenizer.
pub fn build_exact_partitioned_runtime_tokenizer(
    exprs: &[Expr],
    visible_labels: Option<&[String]>,
    partitions: &[u32],
    residual_isolation_classes: &[Option<u32>],
) -> Tokenizer {
    // Below this size, the ordinary materialized product is already cheap and
    // has better locality. Large products keep class transitions compressed.
    const MIN_COMPRESSED_RUNTIME_PAIR_CELLS: usize = 1_000_000;

    assert_eq!(exprs.len(), partitions.len());
    assert_eq!(exprs.len(), residual_isolation_classes.len());
    if let Some(labels) = visible_labels {
        assert_eq!(exprs.len(), labels.len());
    }

    let rewritten_exprs = materialize_repeated_subexpression_dfas(exprs);
    let compile_exprs = rewritten_exprs.as_deref().unwrap_or(exprs);
    let mut grouped = BTreeMap::<u32, Vec<usize>>::new();
    for (terminal, &partition) in partitions.iter().enumerate() {
        grouped.entry(partition).or_default().push(terminal);
    }

    for terminal_ids in grouped.values() {
        let mut class = None;
        let mut has_unprotected = false;
        for &terminal in terminal_ids {
            match residual_isolation_classes[terminal] {
                Some(current) => match class {
                    Some(previous) => assert_eq!(
                        previous, current,
                        "one lexer partition cannot contain distinct protected residual classes",
                    ),
                    None => class = Some(current),
                },
                None => has_unprotected = true,
            }
        }
        assert!(
            class.is_none() || !has_unprotected,
            "one lexer partition cannot mix protected and unprotected residual coordinates",
        );
    }

    let shared_duplicates = shared_duplicate_nested_group_op_cache(compile_exprs, &grouped);
    if let Some(shared_duplicates) = &shared_duplicates {
        prewarm_shared_duplicate_nested_group_ops(shared_duplicates);
    }

    let components = grouped
        .into_iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(_partition, terminal_ids)| {
            let local_exprs = terminal_ids
                .iter()
                .map(|&terminal| compile_exprs[terminal].clone())
                .collect::<Vec<_>>();
            let local_labels = visible_labels.map(|labels| {
                terminal_ids
                    .iter()
                    .map(|&terminal| labels[terminal].clone())
                    .collect::<Vec<_>>()
            });
            let mut nested_group_op_cache = NestedGroupOpCache {
                shared_duplicates: shared_duplicates.clone(),
                ..NestedGroupOpCache::default()
            };
            let plan = build_exclusion_compile_plan_with_labels_and_cache(
                &local_exprs,
                local_labels.as_deref(),
                &mut nested_group_op_cache,
            );
            let full = match try_compile_with_plan_deferred_dense_min_pair_cells(
                plan,
                MIN_COMPRESSED_RUNTIME_PAIR_CELLS,
                true,
            ) {
                Ok((mut full, trace)) => {
                    if let Some(trace) = trace {
                        full.attach_dense_runtime_trace(trace)
                            .expect("deferred exact runtime tokenizer must accept its state trace");
                    }
                    full
                }
                Err(plan) => DeferredDfa::Ready(compile_with_plan(plan)),
            };
            (terminal_ids, full)
        })
        .collect::<Vec<_>>();

    let mut combined = DFA::new(1);
    combined.ensure_group_capacity(exprs.len());
    let mut root_futures = BitSet::new(exprs.len());
    let mut compressed_segments = Vec::new();
    for (terminal_ids, full) in components {
        let (mut component_dfa, compressed) = full.finish_runtime();
        if compressed.is_none() {
            // Isolate nullable roots while this component is still small and
            // independent. The final tokenizer is a disjoint epsilon union of
            // these components, so this is equivalent to isolating the full
            // union but avoids rescanning every state in unrelated huge
            // compressed components.
            component_dfa = isolate_component_nullable_start(
                component_dfa,
                terminal_ids.len(),
            )
            .0;
        }
        debug_assert_eq!(component_dfa.num_groups(), terminal_ids.len());
        for local_group in component_dfa.possible_future_group_ids(0).iter() {
            root_futures.set(terminal_ids[local_group]);
        }
        let offset = combined.append_rebased_component(component_dfa, &terminal_ids);
        combined.add_epsilon_transition(0, offset);
        if let Some(mut segment) = compressed {
            segment.state_offset = offset;
            compressed_segments.push(segment);
        }
    }
    combined.set_possible_future_group_ids(0, root_futures);
    Tokenizer::from_parts_with_compressed_transitions(
        combined,
        exprs.len() as u32,
        Some(Arc::from(exprs.to_vec().into_boxed_slice())),
        compressed_segments,
    )
}
