//! Prepare smaller mask components and prove their relationship to full runtime components.

use crate::automata::lexer::tokenizer::Lexer;
use crate::Vocab;
use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::dfa::DFA;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::ds::bitset::BitSet;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use super::Regex;
use super::deferred::DeferredDfa;
use super::mapping::{
    CompiledTerminalExpressionPair,
    StructuralPairProof,
    compile_terminal_expression_pair_with_structural_map_and_proof,
    prepare_terminal_expression_pair_with_structural_map_inner,
};
use super::partition::{
    CompiledPartitionedExpressionPair,
    DeferredPartitionedRegex,
    LexerComponent,
    LexerComponentPair,
    PreparedPartitionedExpressionPair,
    adaptively_determinize_components,
    combine_lexer_components_under_epsilon_root,
    combine_synthesized_component_pairs_under_epsilon_root,
    compile_terminal_ids_with_shared_duplicate_cache,
    compose_partitioned_component_state_maps,
    isolate_component_nullable_start,
};
use super::plan::{
    expr_profile_summary,
    materialize_repeated_subexpression_dfas,
    prewarm_shared_duplicate_nested_group_ops,
    shared_duplicate_nested_group_op_cache,
};
use super::repeat_horizon::VocabularyRepeatHorizonCache;
use super::settings::adaptive_lexer_state_limit;
use rayon::prelude::*;

pub struct ExtractedDispatchComponent {
    pub terminal_ids: Vec<usize>,
    pub source_states: Vec<u32>,
    pub dfa: DFA,
}

pub struct PrecompiledFurtherSynthesisPairs {
    pairs: Mutex<BTreeMap<usize, CompiledTerminalExpressionPair>>,
    pub build_ms: f64,
}

impl PrecompiledFurtherSynthesisPairs {
    fn take(&self, terminal: usize) -> Option<CompiledTerminalExpressionPair> {
        self.pairs
            .lock()
            .expect("precompiled synthesis-pair cache poisoned")
            .remove(&terminal)
    }
}

pub fn precompile_further_synthesis_pairs(
    source_expressions: &[Expr],
    synthesized_expressions: &[Expr],
    protected_terminal_ids: &[u32],
    vocab: &Vocab,
    repeat_horizons: &VocabularyRepeatHorizonCache,
    max_token_len: usize,
    relevant_bytes: &[u8],
) -> Option<Arc<PrecompiledFurtherSynthesisPairs>> {
    if source_expressions.len() != synthesized_expressions.len() {
        return None;
    }
    let protected = protected_terminal_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let changed = source_expressions
        .iter()
        .zip(synthesized_expressions)
        .enumerate()
        .filter_map(|(terminal, (source, synthesized))| {
            (source != synthesized).then_some(terminal)
        })
        .collect::<Vec<_>>();
    if changed.is_empty()
        || changed
            .iter()
            .any(|&terminal| !protected.contains(&(terminal as u32)))
    {
        return None;
    }
    let started_at = Instant::now();
    let pairs = changed
        .par_iter()
        .map(|&terminal| {
            compile_terminal_expression_pair_with_structural_map_and_proof(
                &source_expressions[terminal],
                &synthesized_expressions[terminal],
                vocab,
                repeat_horizons,
                max_token_len,
                relevant_bytes,
                StructuralPairProof::VocabularyTokenQuotient,
            )
            .map(|pair| (terminal, pair))
        })
        .collect::<Option<BTreeMap<_, _>>>()?;
    Some(Arc::new(PrecompiledFurtherSynthesisPairs {
        pairs: Mutex::new(pairs),
        build_ms: started_at.elapsed().as_secs_f64() * 1000.0,
    }))
}

fn extract_dispatch_component_from_states(
    tokenizer: &Tokenizer,
    root: u32,
    mut source_states: Vec<u32>,
) -> Option<ExtractedDispatchComponent> {
    source_states.sort_unstable();
    let root_position = source_states.iter().position(|&state| state == root)?;
    source_states.swap(0, root_position);

    let terminal_ids = tokenizer
        .dfa
        .possible_future_group_ids(root)
        .iter()
        .collect::<Vec<_>>();
    if terminal_ids.is_empty() {
        return None;
    }
    let mut local_group_by_terminal = vec![usize::MAX; tokenizer.num_terminals as usize];
    for (local_group, &terminal) in terminal_ids.iter().enumerate() {
        *local_group_by_terminal.get_mut(terminal)? = local_group;
    }
    let mut local_state_by_source = vec![u32::MAX; tokenizer.dfa.num_states()];
    for (local_state, &source_state) in source_states.iter().enumerate() {
        local_state_by_source[source_state as usize] = local_state as u32;
    }

    let mut dfa = DFA::new(source_states.len());
    dfa.ensure_group_capacity(terminal_ids.len());
    for (local_group, &terminal) in terminal_ids.iter().enumerate() {
        dfa.set_group_u8set(
            local_group as u32,
            *tokenizer.dfa.group_id_to_u8set(terminal as u32),
        );
    }
    for (local_state, &source_state) in source_states.iter().enumerate() {
        let source_dfa_state = tokenizer.dfa.states().get(source_state as usize)?;
        let transitions = source_dfa_state
            .transitions
            .iter()
            .map(|(byte, &target)| {
                let target = *local_state_by_source.get(target as usize)?;
                (target != u32::MAX).then_some((byte, target))
            })
            .collect::<Option<Vec<_>>>()?;
        dfa.set_transitions_from_sorted_entries(local_state as u32, transitions);
        for &target in &source_dfa_state.epsilon_transitions {
            let target = *local_state_by_source.get(target as usize)?;
            if target == u32::MAX {
                return None;
            }
            dfa.add_epsilon_transition(local_state as u32, target);
        }

        let mut finalizers = BitSet::new(terminal_ids.len());
        let mut futures = BitSet::new(terminal_ids.len());
        for terminal in tokenizer.dfa.finalizers(source_state).iter() {
            let local_group = *local_group_by_terminal.get(terminal)?;
            if local_group == usize::MAX {
                return None;
            }
            finalizers.set(local_group);
        }
        for terminal in tokenizer
            .dfa
            .possible_future_group_ids(source_state)
            .iter()
        {
            let local_group = *local_group_by_terminal.get(terminal)?;
            if local_group == usize::MAX {
                return None;
            }
            futures.set(local_group);
        }
        dfa.overwrite_state_metadata(local_state as u32, finalizers, futures);
    }
    Some(ExtractedDispatchComponent {
        terminal_ids,
        source_states,
        dfa,
    })
}

pub fn extract_dispatch_components(
    tokenizer: &Tokenizer,
) -> Option<Vec<ExtractedDispatchComponent>> {
    let roots = tokenizer.deterministic_dispatch_roots()?;
    let components = tokenizer.disjoint_dispatch_components()?;
    if roots.len() != components.len() {
        return None;
    }
    roots
        .iter()
        .copied()
        .zip(components)
        .map(|(root, source_states)| {
            extract_dispatch_component_from_states(tokenizer, root, source_states)
        })
        .collect()
}

/// Extract singleton dispatch components together with externally entered
/// residual states that were cloned outside the live root-reachable component
/// graph by protected residual synthesis.
pub fn extract_augmented_singleton_dispatch_components(
    tokenizer: &Tokenizer,
) -> Option<Vec<ExtractedDispatchComponent>> {
    let profile = std::env::var_os("GLRMASK_DEFINITION_PHYSICAL_REPORT").is_some();
    let roots = tokenizer.deterministic_dispatch_roots()?.to_vec();
    let components = tokenizer.disjoint_dispatch_components()?;
    if roots.len() != components.len() {
        return None;
    }
    let component_count = components.len();
    let state_count = tokenizer.dfa.num_states();
    let mut owner = vec![usize::MAX; state_count];
    owner[tokenizer.initial_state_id() as usize] = roots.len();
    let mut terminal_to_singleton_component = vec![usize::MAX; tokenizer.num_terminals as usize];
    let mut singleton_terminal_by_component = vec![usize::MAX; roots.len()];
    let mut singleton_components = 0usize;
    for (component, (&root, states)) in roots.iter().zip(&components).enumerate() {
        for &state in states {
            owner[state as usize] = component;
        }
        let terminals = tokenizer
            .dfa
            .possible_future_group_ids(root)
            .iter()
            .collect::<Vec<_>>();
        if terminals.len() == 1 {
            terminal_to_singleton_component[terminals[0]] = component;
            singleton_terminal_by_component[component] = terminals[0];
            singleton_components += 1;
        }
    }

    // Appended protected residuals retain exact final/future terminal metadata.
    // Use that label as the initial owner, then admit only states whose complete
    // outgoing graph remains inside the same singleton component.
    let initially_unowned = owner.iter().filter(|&&owner| owner == usize::MAX).count();
    let mut label_assigned = 0usize;
    for state in 0..state_count {
        if owner[state] != usize::MAX {
            continue;
        }
        let mut terminals = tokenizer.dfa.finalizers(state as u32).iter().collect::<Vec<_>>();
        terminals.extend(tokenizer.dfa.possible_future_group_ids(state as u32).iter());
        terminals.sort_unstable();
        terminals.dedup();
        if terminals.len() == 1 {
            let component = terminal_to_singleton_component[terminals[0]];
            if component != usize::MAX {
                owner[state] = component;
                label_assigned += 1;
            }
        }
    }
    let mut forward_propagated = 0usize;
    let mut queue = VecDeque::<u32>::new();
    for state in 0..state_count {
        if owner[state] < roots.len()
            && singleton_terminal_by_component[owner[state]] != usize::MAX
            && !components[owner[state]].contains(&(state as u32))
        {
            queue.push_back(state as u32);
        }
    }
    while let Some(state) = queue.pop_front() {
        let component = owner[state as usize];
        let terminal = singleton_terminal_by_component[component];
        for target in tokenizer.dfa.states()[state as usize]
            .transitions
            .iter()
            .map(|(_, &target)| target)
            .chain(
                tokenizer.dfa.states()[state as usize]
                    .epsilon_transitions
                    .iter()
                    .copied(),
            )
        {
            let target_slot = &mut owner[target as usize];
            if *target_slot == component {
                continue;
            }
            if *target_slot != usize::MAX {
                continue;
            }
            let mut target_terminals = tokenizer
                .dfa
                .finalizers(target)
                .iter()
                .collect::<Vec<_>>();
            target_terminals.extend(tokenizer.dfa.possible_future_group_ids(target).iter());
            target_terminals.sort_unstable();
            target_terminals.dedup();
            if target_terminals.is_empty()
                || (target_terminals.len() == 1 && target_terminals[0] == terminal)
            {
                *target_slot = component;
                forward_propagated += 1;
                queue.push_back(target);
            }
        }
    }
    let mut propagated = 0usize;
    let mut changed = true;
    while changed {
        changed = false;
        for state in 0..state_count {
            if owner[state] != usize::MAX {
                continue;
            }
            let dfa_state = &tokenizer.dfa.states()[state];
            let mut candidate = usize::MAX;
            let mut valid = true;
            for target in dfa_state
                .transitions
                .iter()
                .map(|(_, &target)| target)
                .chain(dfa_state.epsilon_transitions.iter().copied())
            {
                let target_owner = owner[target as usize];
                if target_owner == usize::MAX || target_owner == roots.len() {
                    valid = false;
                    break;
                }
                if candidate == usize::MAX {
                    candidate = target_owner;
                } else if candidate != target_owner {
                    valid = false;
                    break;
                }
            }
            if valid
                && candidate != usize::MAX
                && terminal_to_singleton_component.contains(&candidate)
            {
                owner[state] = candidate;
                changed = true;
                propagated += 1;
            }
        }
    }

    let mut augmented = components;
    for state in 0..state_count {
        let component = owner[state];
        if component < roots.len() && !augmented[component].contains(&(state as u32)) {
            augmented[component].push(state as u32);
        }
    }
    let remaining_unowned = owner.iter().filter(|&&owner| owner == usize::MAX).count();
    let mut build_failures = 0usize;
    let extracted = roots
        .into_iter()
        .zip(augmented)
        .filter_map(|(root, states)| {
            let terminals = tokenizer
                .dfa
                .possible_future_group_ids(root)
                .iter()
                .collect::<Vec<_>>();
            if terminals.len() != 1 {
                return None;
            }
            match extract_dispatch_component_from_states(tokenizer, root, states) {
                Some(component) => Some(component),
                None => {
                    build_failures += 1;
                    None
                }
            }
        })
        .collect::<Vec<_>>();
    if profile {
        eprintln!(
            "[glrmask/profile][definition_dispatch_augmented] states={} roots={} singleton_components={} initially_unowned={} label_assigned={} forward_propagated={} propagated={} remaining_unowned={} extracted_singletons={} build_failures={} augmented_states={}",
            state_count,
            component_count,
            singleton_components,
            initially_unowned,
            label_assigned,
            forward_propagated,
            propagated,
            remaining_unowned,
            extracted.len(),
            build_failures,
            extracted.iter().map(|component| component.source_states.len()).sum::<usize>(),
        );
    }
    Some(extracted)
}

fn augment_component_from_verified_prefix(
    source: &DFA,
    rebuilt: &DFA,
    synthesized: &mut DFA,
    rebuilt_to_synthesized: &[u32],
) -> Option<Vec<u32>> {
    if source.num_groups() != rebuilt.num_groups()
        || source.num_groups() != synthesized.num_groups()
        || rebuilt_to_synthesized.len() != rebuilt.num_states()
        || rebuilt.num_states() > source.num_states()
    {
        return None;
    }
    for state in 0..rebuilt.num_states() {
        let state = state as u32;
        if source.finalizers(state) != rebuilt.finalizers(state)
            || source.possible_future_group_ids(state)
                != rebuilt.possible_future_group_ids(state)
            || source.states()[state as usize].epsilon_transitions
                != rebuilt.states()[state as usize].epsilon_transitions
            || source.states()[state as usize].transitions
                != rebuilt.states()[state as usize].transitions
        {
            return None;
        }
    }

    let mut source_to_synthesized = vec![u32::MAX; source.num_states()];
    source_to_synthesized[..rebuilt.num_states()].copy_from_slice(rebuilt_to_synthesized);
    for source_state in rebuilt.num_states()..source.num_states() {
        source_to_synthesized[source_state] = synthesized.add_state();
    }
    for source_state in rebuilt.num_states()..source.num_states() {
        let source_state_u32 = source_state as u32;
        let target_state = source_to_synthesized[source_state];
        let source_dfa_state = &source.states()[source_state];
        if !source_dfa_state.epsilon_transitions.is_empty() {
            return None;
        }
        let transitions = source_dfa_state
            .transitions
            .iter()
            .map(|(byte, &target)| {
                Some((byte, *source_to_synthesized.get(target as usize)?))
            })
            .collect::<Option<Vec<_>>>()?;
        if transitions.iter().any(|&(_, target)| target == u32::MAX) {
            return None;
        }
        synthesized.set_transitions_from_sorted_entries(target_state, transitions);
        synthesized.overwrite_state_metadata(
            target_state,
            source.finalizers(source_state_u32).clone(),
            source.possible_future_group_ids(source_state_u32).clone(),
        );
    }
    Some(source_to_synthesized)
}

/// Further synthesize protected singleton components of an already-built
/// partitioned tokenizer without recompiling its ordinary components.
///
/// Every unchanged dispatch component is cloned exactly. Changed protected
/// components use the same structural terminal-pair proof as the global
/// synthesis, then clone any externally-entered residual states appended to
/// the actual source component. The returned map covers the actual source
/// tokenizer's raw-state domain.
pub fn compile_further_synthesized_tokenizer_with_structural_map(
    source: &Tokenizer,
    source_expressions: &[Expr],
    synthesized_expressions: &[Expr],
    protected_terminal_ids: &[u32],
    vocab: &Vocab,
    repeat_horizons: &VocabularyRepeatHorizonCache,
    max_token_len: usize,
    relevant_bytes: &[u8],
    precompiled_pairs: Option<&PrecompiledFurtherSynthesisPairs>,
) -> Option<(Tokenizer, Vec<u32>)> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some();
    let started_at = profile.then(Instant::now);
    let reject = |stage: &str| {
        if profile {
            eprintln!(
                "[glrmask/profile][partition_local_component_reuse] selected=false stage={} elapsed_ms={:.3}",
                stage,
                started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            );
        }
        None
    };
    if source_expressions.len() != synthesized_expressions.len()
        || source_expressions.len() != source.num_terminals as usize
    {
        return reject("input_shape");
    }
    let protected = protected_terminal_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let changed = source_expressions
        .iter()
        .zip(synthesized_expressions)
        .map(|(source, synthesized)| source != synthesized)
        .collect::<Vec<_>>();
    if !changed.iter().any(|&changed| changed) {
        return reject("unchanged");
    }

    let Some(extracted) = extract_dispatch_components(source) else {
        return reject("extract_dispatch");
    };
    let mut output_components = Vec::<LexerComponent>::with_capacity(extracted.len());
    let mut component_maps = Vec::<(Vec<u32>, Vec<u32>)>::with_capacity(extracted.len());
    let mut handled_changed = vec![false; changed.len()];
    let mut effective_synthesized_expressions = synthesized_expressions.to_vec();

    for component in extracted {
        let changed_terminals = component
            .terminal_ids
            .iter()
            .copied()
            .filter(|&terminal| changed.get(terminal).copied().unwrap_or(false))
            .collect::<Vec<_>>();
        if changed_terminals.is_empty() {
            let state_count = component.dfa.num_states() as u32;
            component_maps.push((
                component.source_states,
                (0..state_count).collect::<Vec<_>>(),
            ));
            output_components.push(LexerComponent {
                terminal_ids: component.terminal_ids,
                dfa: component.dfa,
                protected_residual: false,
            });
            continue;
        }
        if changed_terminals.len() != 1
            || component.terminal_ids.len() != 1
            || !protected.contains(&(changed_terminals[0] as u32))
        {
            return reject("changed_component_shape");
        }
        let terminal = changed_terminals[0];
        handled_changed[terminal] = true;
        let Some(pair) = precompiled_pairs
            .and_then(|pairs| pairs.take(terminal))
            .or_else(|| {
                compile_terminal_expression_pair_with_structural_map_and_proof(
                    &source_expressions[terminal],
                    &synthesized_expressions[terminal],
                    vocab,
                    repeat_horizons,
                    max_token_len,
                    relevant_bytes,
                    StructuralPairProof::VocabularyTokenQuotient,
                )
            })
        else {
            return reject("protected_pair");
        };
        effective_synthesized_expressions[terminal] = pair.synthesized_expression.clone();
        let (mut synthesized, synthesized_nullable) =
            isolate_component_nullable_start(pair.synthesized.dfa, 1);
        let (rebuilt, rebuilt_nullable) = isolate_component_nullable_start(pair.full.dfa, 1);
        if !synthesized_nullable.is_empty() || !rebuilt_nullable.is_empty() {
            return reject("protected_nullable");
        }
        let Some(source_to_synthesized) = augment_component_from_verified_prefix(
            &component.dfa,
            &rebuilt,
            &mut synthesized,
            &pair.full_to_synthesized,
        ) else {
            if profile {
                eprintln!(
                    "[glrmask/profile][partition_local_component_reuse_detail] terminal={} source_states={} rebuilt_states={} synthesized_states={}",
                    terminal,
                    component.dfa.num_states(),
                    rebuilt.num_states(),
                    synthesized.num_states(),
                );
            }
            return reject("protected_prefix");
        };
        // This lane consumes complete entries from `vocab`, not arbitrary
        // one-byte continuations from every raw source state. The structural
        // pair above is therefore certified in the vocabulary-token quotient
        // coordinate. Requiring a raw byte homomorphism here is strictly
        // stronger and rejects valid bounded-repeat quotients.
        component_maps.push((component.source_states, source_to_synthesized));
        output_components.push(LexerComponent {
            terminal_ids: component.terminal_ids,
            dfa: synthesized,
            protected_residual: true,
        });
    }
    if changed
        .iter()
        .enumerate()
        .any(|(terminal, &changed)| changed && !handled_changed[terminal])
    {
        return reject("unhandled_changed");
    }

    let mut source_to_synthesized = vec![u32::MAX; source.dfa.num_states()];
    source_to_synthesized[0] = 0;
    let mut offset = 1u32;
    for (component, (source_states, local_map)) in output_components.iter().zip(&component_maps) {
        if source_states.len() != local_map.len() {
            return reject("component_map_length");
        }
        for (&source_state, &local_state) in source_states.iter().zip(local_map) {
            source_to_synthesized[source_state as usize] = offset + local_state;
        }
        offset = offset.checked_add(component.dfa.num_states() as u32)?;
    }
    let mut dfa = combine_lexer_components_under_epsilon_root(
        output_components,
        source_expressions.len(),
    );
    // Nullable-start isolation and prior structural augmentation may retain
    // unreachable raw states outside the live dispatch components. They remain
    // part of the compiler's raw-state coordinate, so clone them after the live
    // component layout is fixed and redirect every edge through the completed
    // source-state map.
    let extra_source_states = source_to_synthesized
        .iter()
        .enumerate()
        .filter_map(|(state, &mapped)| (mapped == u32::MAX).then_some(state))
        .collect::<Vec<_>>();
    for &source_state in &extra_source_states {
        source_to_synthesized[source_state] = dfa.add_state();
    }
    for source_state in extra_source_states {
        let target_state = source_to_synthesized[source_state];
        let source_dfa_state = &source.dfa.states()[source_state];
        let transitions = source_dfa_state
            .transitions
            .iter()
            .map(|(byte, &target)| (byte, source_to_synthesized[target as usize]))
            .collect::<Vec<_>>();
        dfa.set_transitions_from_sorted_entries(target_state, transitions);
        for &target in &source_dfa_state.epsilon_transitions {
            dfa.add_epsilon_transition(target_state, source_to_synthesized[target as usize]);
        }
        dfa.overwrite_state_metadata(
            target_state,
            source.dfa.finalizers(source_state as u32).clone(),
            source
                .dfa
                .possible_future_group_ids(source_state as u32)
                .clone(),
        );
    }
    let tokenizer = Regex { dfa }.into_tokenizer(
        source_expressions.len() as u32,
        Some(Arc::from(
            effective_synthesized_expressions.into_boxed_slice(),
        )),
    );
    if profile {
        eprintln!(
            "[glrmask/profile][partition_local_component_reuse] selected=true source_states={} synthesized_states={} elapsed_ms={:.3}",
            source.dfa.num_states(),
            tokenizer.dfa.num_states(),
            started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
        );
    }
    Some((tokenizer, source_to_synthesized))
}

/// Compile independently protected terminal partitions as exact/synthesized
/// pairs while compiling every unchanged partition only once. The returned
/// state map is structural: the global epsilon root maps to the global root,
/// unchanged components map identically, and protected components use their
/// certified local product maps. Adaptive prefix determinization remains
/// available for the ordinary components but never crosses a protected
/// residual coordinate.
fn prepare_partitioned_expression_pair_with_proof(
    full_exprs: &[Expr],
    synthesized_exprs: &[Expr],
    visible_labels: Option<&[String]>,
    partitions: &[u32],
    residual_isolation_classes: &[Option<u32>],
    adaptive: bool,
    vocab: &Vocab,
    repeat_horizons: &VocabularyRepeatHorizonCache,
    max_token_len: usize,
    relevant_bytes: &[u8],
    proof: StructuralPairProof,
) -> Option<PreparedPartitionedExpressionPair> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    if full_exprs.len() != synthesized_exprs.len()
        || full_exprs.len() != partitions.len()
        || full_exprs.len() != residual_isolation_classes.len()
        || visible_labels.is_some_and(|labels| labels.len() != full_exprs.len())
        || full_exprs.is_empty()
    {
        return None;
    }

    let changed = full_exprs
        .iter()
        .zip(synthesized_exprs)
        .zip(residual_isolation_classes)
        .map(|((full, synthesized), isolation_class)| {
            isolation_class.is_some() && full != synthesized
        })
        .collect::<Vec<_>>();
    if !changed.iter().any(|&changed| changed) {
        return None;
    }

    let mut grouped = BTreeMap::<u32, Vec<usize>>::new();
    for (terminal, &partition) in partitions.iter().enumerate() {
        grouped.entry(partition).or_default().push(terminal);
    }

    // Only ordinary terminals participate in repeated-subexpression sharing.
    // Replacing changed expressions with epsilon avoids compiling their large
    // exact bodies merely to prepare a cache they cannot use.
    let mut ordinary_exprs = synthesized_exprs.to_vec();
    for (terminal, &is_changed) in changed.iter().enumerate() {
        if is_changed {
            ordinary_exprs[terminal] = Expr::Epsilon;
        }
    }
    let rewritten_ordinary = materialize_repeated_subexpression_dfas(&ordinary_exprs);
    let ordinary_exprs = rewritten_ordinary.as_deref().unwrap_or(&ordinary_exprs);
    let ordinary_groups = grouped
        .iter()
        .filter_map(|(&partition, terminals)| {
            terminals
                .iter()
                .all(|&terminal| !changed[terminal])
                .then_some((partition, terminals.clone()))
        })
        .collect::<BTreeMap<_, _>>();
    let shared_duplicates = shared_duplicate_nested_group_op_cache(ordinary_exprs, &ordinary_groups);
    if let Some(shared_duplicates) = &shared_duplicates {
        prewarm_shared_duplicate_nested_group_ops(shared_duplicates);
    }

    let compiled = grouped
        .into_iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(_partition, terminal_ids)| {
            let changed_terminals = terminal_ids
                .iter()
                .copied()
                .filter(|&terminal| changed[terminal])
                .collect::<Vec<_>>();
            if changed_terminals.is_empty() {
                let dfa = compile_terminal_ids_with_shared_duplicate_cache(
                    ordinary_exprs,
                    visible_labels,
                    &terminal_ids,
                    shared_duplicates.as_ref(),
                );
                let minimize_started_at = profile.then(Instant::now);
                let before_states = dfa.num_states();
                let dfa = dfa.minimize();
                if profile {
                    let labels = visible_labels
                        .map(|labels| {
                            terminal_ids
                                .iter()
                                .filter_map(|&terminal| labels.get(terminal))
                                .take(4)
                                .cloned()
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    eprintln!(
                        "[glrmask/profile][tokenizer] paired_ordinary_minimize terminals={} terminal_ids={:?} labels={:?} states_before={} states_after={} elapsed_ms={:.3}",
                        terminal_ids.len(),
                        terminal_ids,
                        labels,
                        before_states,
                        dfa.num_states(),
                        minimize_started_at
                            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                    );
                }
                let state_count = dfa.num_states() as u32;
                return Some((
                    LexerComponentPair {
                        terminal_ids,
                        synthesized: dfa.clone(),
                        full: DeferredDfa::Ready(dfa),
                        full_to_synthesized: (0..state_count).collect(),
                        protected_residual: false,
                    },
                    None,
                ));
            }

            if changed_terminals.len() != 1 || terminal_ids.len() != 1 {
                if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
                    eprintln!(
                        "[glrmask/profile][tokenizer] structural_partition_pair_rejected reason=non_singleton_partition terminals={:?} changed={:?}",
                        terminal_ids,
                        changed_terminals,
                    );
                }
                return None;
            }
            let terminal = changed_terminals[0];
            residual_isolation_classes[terminal]?;
            let pair = prepare_terminal_expression_pair_with_structural_map_inner(
                &full_exprs[terminal],
                &synthesized_exprs[terminal],
                vocab,
                repeat_horizons,
                max_token_len,
                relevant_bytes,
                true,
                proof,
            );
            let Some(pair) = pair else {
                if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
                    eprintln!(
                        "[glrmask/profile][tokenizer] structural_partition_pair_rejected reason=local_pair_failed terminal={} full_expr={:?} synthesized_expr={:?}",
                        terminal,
                        expr_profile_summary(&full_exprs[terminal]),
                        expr_profile_summary(&synthesized_exprs[terminal]),
                    );
                }
                return None;
            };
            let (synthesized, synthesized_nullable) =
                isolate_component_nullable_start(pair.synthesized.dfa, terminal_ids.len());
            let mut full = pair.full;
            let full_nullable = if full.initial_is_nonnullable_without_epsilon() {
                BTreeSet::new()
            } else {
                let (dfa, nullable) =
                    isolate_component_nullable_start(full.finish(), terminal_ids.len());
                full = DeferredDfa::Ready(dfa);
                nullable
            };
            if !synthesized_nullable.is_empty() || !full_nullable.is_empty() {
                if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
                    eprintln!(
                        "[glrmask/profile][tokenizer] structural_partition_pair_rejected reason=nullable_protected_terminal terminal={} synthesized_nullable={:?} full_nullable={:?}",
                        terminal, synthesized_nullable, full_nullable,
                    );
                }
                return None;
            }
            Some((
                LexerComponentPair {
                    terminal_ids,
                    synthesized,
                    full,
                    full_to_synthesized: pair.full_to_synthesized,
                    protected_residual: true,
                },
                Some((terminal, pair.synthesized_expression)),
            ))
        })
        .collect::<Option<Vec<_>>>()?;

    let mut protected = Vec::new();
    let mut ordinary = Vec::new();
    let mut effective_synthesized_expressions = synthesized_exprs.to_vec();
    for (pair, effective_expression) in compiled {
        if let Some((terminal, expression)) = effective_expression {
            effective_synthesized_expressions[terminal] = expression;
        }
        if pair.protected_residual {
            protected.push(pair);
        } else {
            let dfa = match pair.full {
                DeferredDfa::Ready(dfa) => dfa,
                DeferredDfa::ReadyCompressed { .. } | DeferredDfa::DenseBinary(_) => {
                    unreachable!("ordinary lexer components are never deferred")
                }
            };
            ordinary.push(LexerComponent {
                terminal_ids: pair.terminal_ids,
                dfa,
                protected_residual: false,
            });
        }
    }

    let ordinary = if adaptive && ordinary.len() >= 2 {
        adaptively_determinize_components(ordinary, adaptive_lexer_state_limit())
    } else {
        ordinary
    };
    let mut pairs = ordinary
        .into_iter()
        .map(|component| {
            let terminal_ids = component.terminal_ids;
            let (dfa, _) = isolate_component_nullable_start(component.dfa, terminal_ids.len());
            let state_count = dfa.num_states() as u32;
            LexerComponentPair {
                terminal_ids,
                synthesized: dfa.clone(),
                full: DeferredDfa::Ready(dfa),
                full_to_synthesized: (0..state_count).collect(),
                protected_residual: false,
            }
        })
        .collect::<Vec<_>>();
    pairs.append(&mut protected);
    pairs.sort_unstable_by_key(|pair| {
        pair.terminal_ids.first().copied().unwrap_or(usize::MAX)
    });

    let (synthesized, synthesized_offsets) =
        combine_synthesized_component_pairs_under_epsilon_root(&pairs, full_exprs.len());
    let full_to_synthesized =
        compose_partitioned_component_state_maps(&pairs, &synthesized_offsets)?;

    Some(PreparedPartitionedExpressionPair {
        synthesized: Regex { dfa: synthesized },
        full: DeferredPartitionedRegex {
            components: pairs,
            total_groups: full_exprs.len(),
        },
        full_to_synthesized,
        synthesized_expressions: effective_synthesized_expressions,
    })
}

pub fn prepare_partitioned_expression_pair_with_structural_map(
    full_exprs: &[Expr],
    synthesized_exprs: &[Expr],
    visible_labels: Option<&[String]>,
    partitions: &[u32],
    residual_isolation_classes: &[Option<u32>],
    adaptive: bool,
    vocab: &Vocab,
    repeat_horizons: &VocabularyRepeatHorizonCache,
    max_token_len: usize,
    relevant_bytes: &[u8],
) -> Option<PreparedPartitionedExpressionPair> {
    prepare_partitioned_expression_pair_with_proof(
        full_exprs,
        synthesized_exprs,
        visible_labels,
        partitions,
        residual_isolation_classes,
        adaptive,
        vocab,
        repeat_horizons,
        max_token_len,
        relevant_bytes,
        StructuralPairProof::RawHomomorphism,
    )
}

pub fn prepare_partitioned_expression_pair_with_vocabulary_token_quotient(
    full_exprs: &[Expr],
    synthesized_exprs: &[Expr],
    visible_labels: Option<&[String]>,
    partitions: &[u32],
    residual_isolation_classes: &[Option<u32>],
    adaptive: bool,
    vocab: &Vocab,
    repeat_horizons: &VocabularyRepeatHorizonCache,
    max_token_len: usize,
    relevant_bytes: &[u8],
) -> Option<PreparedPartitionedExpressionPair> {
    prepare_partitioned_expression_pair_with_proof(
        full_exprs,
        synthesized_exprs,
        visible_labels,
        partitions,
        residual_isolation_classes,
        adaptive,
        vocab,
        repeat_horizons,
        max_token_len,
        relevant_bytes,
        StructuralPairProof::VocabularyTokenQuotient,
    )
}

pub fn compile_partitioned_expression_pair_with_structural_map(
    full_exprs: &[Expr],
    synthesized_exprs: &[Expr],
    visible_labels: Option<&[String]>,
    partitions: &[u32],
    residual_isolation_classes: &[Option<u32>],
    adaptive: bool,
    vocab: &Vocab,
    repeat_horizons: &VocabularyRepeatHorizonCache,
    max_token_len: usize,
    relevant_bytes: &[u8],
) -> Option<CompiledPartitionedExpressionPair> {
    prepare_partitioned_expression_pair_with_structural_map(
        full_exprs,
        synthesized_exprs,
        visible_labels,
        partitions,
        residual_isolation_classes,
        adaptive,
        vocab,
        repeat_horizons,
        max_token_len,
        relevant_bytes,
    )
    .map(PreparedPartitionedExpressionPair::finish_full)
}
