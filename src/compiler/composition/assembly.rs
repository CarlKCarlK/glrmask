//! Assemble and finalize runtime artifacts from reconciled component data.

use crate::automata::lexer::Lexer;
use super::{
    Action, AnalyzedGrammar, Arc, BTreeMap, BTreeSet, BitSet, CompiledSubgrammarInput,
    ComposedTable, Constraint, ConstraintComposition, ConstraintRuntimeBackend, DWA, EOF,
    FxHashMap, FxHashSet, Instant, InternalIdMap, ManyToOneIdMap, MappedArtifact, OnceLock,
    PossibleMatches, SpecialTokenTerminal, Tokenizer, Vocab, Weight, build_composition_templates,
    build_internal_token_bytes_from_groups, compose_profile_enabled,
    install_published_static_boundary_shards, maybe_install_runtime_lexer_product,
    prepare_concrete_boundary_delta_plan, runtime_dynamic_vocab_for_vocab,
};

pub(crate) fn merged_terminal_display_names(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
) -> Vec<String> {
    let mut names = parent.terminal_display_names.clone();
    for (index, child) in children.iter().enumerate() {
        names.extend(
            child
                .constraint
                .terminal_display_names
                .iter()
                .map(|name| format!("subgrammar{index}::{name}")),
        );
    }
    names
}

pub(crate) fn merged_leaf_terminal_display_names(leaves: &[&Constraint]) -> Vec<String> {
    let mut names = Vec::new();
    for (index, leaf) in leaves.iter().enumerate() {
        if index == 0 {
            names.extend(leaf.terminal_display_names.iter().cloned());
        } else {
            names.extend(
                leaf.terminal_display_names
                    .iter()
                    .map(|name| format!("subgrammar{index}::{name}")),
            );
        }
    }
    names
}

/// Leaf-slice version of `component_ignores_are_globally_erasable` for nested
/// static links, where the link spans recursively expanded intact leaves
/// rather than direct components.
pub(crate) fn leaf_ignores_are_globally_erasable(leaves: &[&Constraint]) -> bool {
    if leaves.iter().any(|leaf| !leaf.table.skip_terminals.is_empty()) {
        return false;
    }
    let Some(first) = leaves.first() else {
        return true;
    };
    match first.ignore_terminal {
        None => leaves.iter().all(|leaf| leaf.ignore_terminal.is_none()),
        Some(_) => {
            let Some(expected) = constraint_ignore_expr(first) else {
                return false;
            };
            leaves.iter().skip(1).all(|leaf| {
                let Some(_ignore) = leaf.ignore_terminal else {
                    return false;
                };
                constraint_ignore_expr(leaf).is_some_and(|actual| actual == expected)
            })
        }
    }
}

/// Leaf-slice version of `merged_ignore_terminals` for nested static links.
pub(crate) fn merged_leaf_ignore_terminals(
    leaves: &[&Constraint],
    leaf_offsets: &[u32],
    globally_erasable: bool,
) -> MergedIgnoreTerminals {
    let ignores = leaves
        .iter()
        .enumerate()
        .filter_map(|(leaf_index, leaf)| {
            leaf.ignore_terminal
                .map(|terminal| leaf_offsets[leaf_index] + terminal)
        })
        .collect::<Vec<_>>();
    let canonical = globally_erasable
        .then(|| ignores.first().copied())
        .flatten();
    let canonical_expr = canonical.and_then(|_| constraint_ignore_expr(leaves[0]).cloned());
    let mut all = BitSet::new(
        leaf_offsets
            .iter()
            .copied()
            .zip(leaves.iter())
            .map(|(offset, leaf)| offset + leaf.tokenizer.num_terminals())
            .max()
            .unwrap_or(0) as usize,
    );
    for &ignore in &ignores {
        all.set(ignore as usize);
    }
    for (leaf_index, leaf) in leaves.iter().enumerate() {
        let terminal_offset = leaf_offsets[leaf_index] as usize;
        for &skip in &leaf.table.skip_terminals {
            all.set(terminal_offset + skip as usize);
        }
    }
    let (global, scoped) = if canonical.is_some() {
        (all.clone(), BitSet::new(all.len()))
    } else {
        (BitSet::new(all.len()), all.clone())
    };
    let aliases = canonical
        .map(|canonical| {
            ignores
                .into_iter()
                .filter(|&ignore| ignore != canonical)
                .collect()
        })
        .unwrap_or_default();
    MergedIgnoreTerminals {
        canonical,
        canonical_expr,
        all,
        scoped,
        global,
        aliases,
    }
}

/// Recursively clear boundary shards inside a nested component selected for
/// the conservative legacy replacement lane.
///
/// That lane builds the outer shard over full vocabulary with per-leaf
/// ownership, so it explicitly subsumes the replaced block's inner repair.
/// Certified-static blocks do not use this helper: their inner static shards
/// remain load-bearing and coexist with the summary-restricted outer block
/// shard. Operates on this composition's own overlay copies (`Arc::make_mut`
/// detaches shared children), never on the caller's input constraints.
pub(crate) fn clear_nested_segmented_boundary_shards(constraint: &mut Constraint) {
    let Some(overlay) = constraint.static_dynamic_overlay.as_mut() else {
        return;
    };
    for component in &mut overlay.segmented_parser_components {
        let inner = std::sync::Arc::make_mut(&mut component.constraint);
        if inner.static_dynamic_overlay.is_none() {
            continue;
        }
        if let Err(error) =
            install_published_static_boundary_shards(inner.static_dynamic_overlay.as_mut().expect(
                "nested segmented component requires overlay for shard clearing",
            ), Vec::new())
        {
            debug_assert!(false, "nested shard clearing must not fail: {error}");
        }
        clear_nested_segmented_boundary_shards(inner);
    }
}

/// Clear retained nested boundary shards only for top-level components whose
/// outer walk deliberately used the legacy leaf-crossing predicate. Components
/// compiled under true block ownership retain their already-static inner
/// shards; those retained contributions are the proof that lets the outer
/// repair omit block-internal crossings.
pub(crate) fn clear_selected_nested_segmented_boundary_shards(
    constraint: &mut Constraint,
    selected: &BitSet,
) -> Result<(), String> {
    if selected.is_zero() {
        return Ok(());
    }
    let overlay = constraint
        .static_dynamic_overlay
        .as_mut()
        .ok_or_else(|| "nested shard clearing requires a segmented overlay".to_string())?;
    for index in selected.iter() {
        let component = overlay
            .segmented_parser_components
            .get_mut(index)
            .ok_or_else(|| format!("nested shard clearing references component {index} outside overlay"))?;
        let inner = std::sync::Arc::make_mut(&mut component.constraint);
        if inner.static_dynamic_overlay.is_none() {
            continue;
        }
        install_published_static_boundary_shards(
            inner
                .static_dynamic_overlay
                .as_mut()
                .expect("nested segmented component requires overlay for shard clearing"),
            Vec::new(),
        )?;
        clear_nested_segmented_boundary_shards(inner);
    }
    Ok(())
}

pub(super) fn merged_original_token_ids(
    vocab: &Vocab,
    special_token_terminals: &[SpecialTokenTerminal],
) -> Vec<u32> {
    let extras = special_token_terminals
        .iter()
        .map(|special| special.token_id)
        .collect::<BTreeSet<_>>();
    let mut extras = extras.into_iter().peekable();
    let mut merged = Vec::with_capacity(vocab.entries_map().len() + extras.size_hint().0);
    for &token in vocab.entries_map().keys() {
        while extras.peek().is_some_and(|extra| *extra < token) {
            merged.push(extras.next().unwrap());
        }
        if extras.peek().is_some_and(|extra| *extra == token) {
            extras.next();
        }
        merged.push(token);
    }
    merged.extend(extras);
    merged
}

pub(super) fn merged_special_token_terminals(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    terminal_offsets: &[u32],
    table: &crate::compiler::glr::table::GLRTable,
    control_terminals: &BTreeSet<u32>,
) -> Vec<SpecialTokenTerminal> {
    fn consumes_terminal(action: &Action) -> bool {
        match action {
            Action::Reduce(_, _) => false,
            Action::Split { shift, accept, .. } => shift.is_some() || *accept,
            Action::Shift(_, _)
            | Action::StackShifts(_)
            | Action::GuardedStackShifts(_)
            | Action::Accept
            | Action::ReplaceShifts(_)
            | Action::Skip => true,
        }
    }

    let mut merged = Vec::new();
    for (component_index, constraint) in std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .enumerate()
    {
        let terminal_offset = terminal_offsets[component_index];
        merged.extend(
            constraint
                .special_token_terminals
                .iter()
                .filter_map(|special| {
                    let terminal_id = terminal_offset + special.terminal_id;
                    if control_terminals.contains(&terminal_id) {
                        return None;
                    }
                    // Placeholder terminals and placeholders retained inside an
                    // already-composed child are dead after table splicing. Do
                    // not keep their exact-token metadata: runtime special-token
                    // masking is driven by this list rather than the byte lexer.
                    table
                        .action
                        .iter()
                        .any(|row| row.get(&terminal_id).is_some_and(consumes_terminal))
                        .then_some(SpecialTokenTerminal {
                            terminal_id,
                            token_id: special.token_id,
                        })
                }),
        );
    }
    merged.sort_unstable_by_key(|special| (special.token_id, special.terminal_id));
    merged.dedup_by_key(|special| (special.token_id, special.terminal_id));
    merged
}

#[derive(Debug, Clone)]
pub(crate) struct MergedIgnoreTerminals {
    pub(crate) canonical: Option<u32>,
    pub(crate) canonical_expr: Option<crate::automata::regex::Expr>,
    pub(crate) all: BitSet,
    /// Ignore terminals whose identity effect depends on the active parser
    /// scope. These remain visible to the boundary terminal/parser DWA.
    pub(crate) scoped: BitSet,
    /// Equivalent component-local ignore terminals which can be erased before
    /// parser interpretation. They are canonicalized to `canonical` in the
    /// final composed tokenizer/artifacts.
    pub(crate) global: BitSet,
    pub(crate) aliases: Vec<u32>,
}


pub(super) fn build_static_dynamic_overlay_metadata(
    composed_table: &ComposedTable,
    components: &[&Constraint],
    merged_ignores: &MergedIgnoreTerminals,
    terminal_display_names: &[String],
    tokenizer_state_offsets: &[u32],
) -> Result<
    (
        crate::runtime::StaticDynamicOverlayMetadata,
        Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
    ),
    String,
> {
    let delta_started_at = Instant::now();
    let augmented_start = composed_table
        .table
        .rules
        .first()
        .map(|rule| rule.lhs)
        .ok_or_else(|| "composed table contains no augmented-start rule".to_string())?;
    let analyzed = AnalyzedGrammar::from_composed_rules(
        composed_table.table.rules.clone(),
        composed_table.table.num_terminals,
        terminal_display_names.to_vec(),
        composed_table.table.nonterminal_display_names.clone(),
        augmented_start,
    );

    // Only the parent grammar is rewritten at this link. Child rule graphs are
    // embedded unchanged, so ordinary child terminal templates transport
    // identically. Scoped ignores are the sole child-side exception.
    let parent_terminal_end = composed_table
        .terminal_offsets
        .get(1)
        .copied()
        .unwrap_or(analyzed.num_terminals) as usize;
    let mut delta_terminals = vec![false; analyzed.num_terminals as usize];
    let selected_parent_end = parent_terminal_end.min(delta_terminals.len());
    delta_terminals[..selected_parent_end].fill(true);
    for terminal in merged_ignores.scoped.iter() {
        if let Some(selected) = delta_terminals.get_mut(terminal) {
            *selected = true;
        }
    }

    let (templates, template_dfas_by_terminal, templates_ms) =
        build_composition_templates(&composed_table.table, &analyzed, &delta_terminals);
    let plan_started_at = Instant::now();
    let plan = prepare_concrete_boundary_delta_plan(
        composed_table,
        components,
        &delta_terminals,
        &templates,
        analyzed.num_terminals,
    );
    let plan_ms = plan_started_at.elapsed().as_secs_f64() * 1000.0;
    let mut repair_terminals = vec![false; analyzed.num_terminals as usize];
    for &terminal in plan.by_global_terminal.keys() {
        if let Some(repair) = repair_terminals.get_mut(terminal as usize) {
            *repair = true;
        }
    }
    for &terminal in &plan.unsafe_terminals {
        if let Some(repair) = repair_terminals.get_mut(terminal as usize) {
            *repair = true;
        }
    }
    let mut parent_states = vec![false; composed_table.table.num_states as usize];
    let mut child_states = vec![false; composed_table.table.num_states as usize];
    for (component_index, relation) in composed_table.state_relations.iter().enumerate() {
        for targets in relation {
            for &state in targets {
                let slot = if component_index == 0 {
                    parent_states.get_mut(state as usize)
                } else {
                    child_states.get_mut(state as usize)
                };
                if let Some(slot) = slot {
                    *slot = true;
                }
            }
        }
    }
    let non_parent_only_parser_states = parent_states
        .iter()
        .zip(&child_states)
        .map(|(&parent, &child)| child && !parent)
        .collect::<Vec<_>>();
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_component_only_delta_metadata] changed={} unsafe={} templates_ms={templates_ms:.3} plan_ms={plan_ms:.3} total_ms={:.3}",
            plan.by_global_terminal.len(),
            plan.unsafe_terminals.len(),
            delta_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok((
        crate::runtime::StaticDynamicOverlayMetadata {
            terminal_offsets: composed_table.terminal_offsets.clone(),
            tokenizer_state_offsets: tokenizer_state_offsets.to_vec(),
            repair_terminals,
            non_parent_only_parser_states,
            segmented_parser_components: Vec::new(),
            segmented_parser_links: Vec::new(),
            segmented_parser_state_offsets: Vec::new(),
            recursive_parser_layout: Default::default(),
            recursive_compiler_table: Default::default(),
            recursive_tokenizer_internal_tsids: Default::default(),
            recursive_virtual_tokenizer_states: Default::default(),
            segmented_mask_authoritative: false,
            segmented_static_baseline: false,
            segmented_component_union_root_dispatch: Vec::new(),
            segmented_boundary_shards: Vec::new(),
            segmented_boundary_parser: None,
            segmented_boundary_terminal_trie: None,
        },
        template_dfas_by_terminal,
    ))
}

pub(super) fn constraint_ignore_expr(constraint: &Constraint) -> Option<&crate::automata::regex::Expr> {
    constraint.ignore_expr.as_ref().or_else(|| {
        constraint
            .ignore_terminal
            .and_then(|terminal| constraint.retained_terminal_expr(terminal))
    })
}

/// Whether every parser scope accepts the same ignore language globally.
///
/// Existing scoped `Skip` actions are proof that a component already contains
/// non-global ignore ownership, so such a component cannot be flattened back
/// into one global ignore merely because its public `ignore_terminal` happens
/// to match another component. Explicit control terminals themselves are not a
/// problem: adjacent/nested children can retain explicit calls while sharing a
/// single globally erased ignore.
pub(crate) fn component_ignores_are_globally_erasable(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
) -> bool {
    let components = std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .collect::<Vec<_>>();
    if components
        .iter()
        .any(|component| !component.table.skip_terminals.is_empty())
    {
        return false;
    }
    let Some(first) = components.first() else {
        return true;
    };
    match first.ignore_terminal {
        None => components
            .iter()
            .all(|component| component.ignore_terminal.is_none()),
        Some(first_ignore) => {
            let Some(expected) = constraint_ignore_expr(first) else {
                return false;
            };
            components.iter().skip(1).all(|component| {
                let Some(_ignore) = component.ignore_terminal else {
                    return false;
                };
                constraint_ignore_expr(component).is_some_and(|actual| actual == expected)
            })
        }
    }
}

pub(super) fn components_have_no_explicit_controls(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
) -> bool {
    std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .all(|component| component.table.control_terminals.is_empty())
}

pub(super) fn components_have_no_compiled_eof_stack_rewrites(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
) -> bool {
    std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .all(|component| {
            component.table.action.iter().all(|row| {
                !matches!(
                    row.get(&EOF),
                    Some(
                        Action::Shift(..)
                            | Action::StackShifts(_)
                            | Action::GuardedStackShifts(_)
                            | Action::ReplaceShifts(_)
                            | Action::Skip
                            | Action::Split { shift: Some(_), .. }
                    )
                )
            })
        })
}

/// The legacy splice identifies child start/accept states with parent
/// caller/continuation states. That optimization is not equivalent when one
/// subgrammar call can directly follow another without consuming a real parent
/// terminal: the first return and second entry are both erased, so a token
/// containing only the second child's first terminal has no correct transported
/// component effect.
///
/// Use the grammar's exact ever-follow relation to reject the optimization for
/// direct, nullable-mediated, or nonterminal-mediated call adjacency. The
/// explicit-control linker remains the reference path for those cases.
pub(super) fn legacy_splice_has_only_byte_terminal_continuations(
    parent: &Constraint,
    parent_rules: &[crate::grammar::flat::Rule],
    children: &[CompiledSubgrammarInput<'_>],
) -> bool {
    if children.is_empty() {
        return true;
    }
    let placeholders = children
        .iter()
        .flat_map(CompiledSubgrammarInput::placeholder_terminals)
        .collect::<BTreeSet<_>>();
    let boundary_controlled_followers = placeholders
        .iter()
        .copied()
        .chain(
            parent
                .special_token_terminals
                .iter()
                .map(|special| special.terminal_id),
        )
        .collect::<BTreeSet<_>>();
    let Some(augmented_start) = parent_rules.first().map(|rule| rule.lhs) else {
        return false;
    };
    let analyzed = AnalyzedGrammar::from_composed_rules(
        parent_rules.to_vec(),
        parent.table.num_terminals,
        parent.terminal_display_names.clone(),
        parent.table.nonterminal_display_names.clone(),
        augmented_start,
    );
    let disallowed = crate::compiler::pipeline::compute_disallowed_follows(&analyzed);
    for &left in &placeholders {
        for &right in &boundary_controlled_followers {
            let is_disallowed = disallowed
                .get(&left)
                .is_some_and(|blocked| blocked.contains(right as usize));
            if !is_disallowed {
                return false;
            }
        }
    }
    true
}

pub(crate) fn merged_ignore_terminals(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    terminal_offsets: &[u32],
    globally_erasable: bool,
) -> MergedIgnoreTerminals {
    let ignores = std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .enumerate()
        .filter_map(|(component_index, constraint)| {
            constraint
                .ignore_terminal
                .map(|terminal| terminal_offsets[component_index] + terminal)
        })
        .collect::<Vec<_>>();
    let canonical = globally_erasable
        .then(|| ignores.first().copied())
        .flatten();
    let canonical_expr = canonical.and_then(|_| constraint_ignore_expr(parent).cloned());
    let mut all = BitSet::new(
        terminal_offsets
            .iter()
            .copied()
            .zip(std::iter::once(parent).chain(children.iter().map(|child| child.constraint)))
            .map(|(offset, constraint)| offset + constraint.tokenizer.num_terminals())
            .max()
            .unwrap_or(0) as usize,
    );
    for &ignore in &ignores {
        all.set(ignore as usize);
    }
    for (component_index, component) in std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .enumerate()
    {
        let terminal_offset = terminal_offsets[component_index] as usize;
        for &skip in &component.table.skip_terminals {
            all.set(terminal_offset + skip as usize);
        }
    }
    let (global, scoped) = if canonical.is_some() {
        (all.clone(), BitSet::new(all.len()))
    } else {
        (BitSet::new(all.len()), all.clone())
    };
    let aliases = canonical
        .map(|canonical| {
            ignores
                .into_iter()
                .filter(|&ignore| ignore != canonical)
                .collect()
        })
        .unwrap_or_default();
    MergedIgnoreTerminals {
        canonical,
        canonical_expr,
        all,
        scoped,
        global,
        aliases,
    }
}

pub(super) fn merged_terminal_live_states(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    terminal_offsets: &[u32],
    tokenizer_state_offsets: &[u32],
    num_terminals: usize,
) -> Vec<Vec<u32>> {
    let component_constraints = std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .collect::<Vec<_>>();
    if !component_constraints.iter().all(|component| {
        component.terminal_live_states.len()
            == component.tokenizer.num_terminals() as usize
    }) {
        return Vec::new();
    }
    let mut merged = vec![Vec::<u32>::new(); num_terminals];
    for (component_index, component) in component_constraints.iter().enumerate() {
        let terminal_offset = terminal_offsets[component_index] as usize;
        let state_offset = tokenizer_state_offsets[component_index];
        let local_start = component.tokenizer.start_state();
        for (local_terminal, states) in component.terminal_live_states.iter().enumerate() {
            let destination = &mut merged[terminal_offset + local_terminal];
            destination.extend(states.iter().map(|&state| state_offset + state));
            if states.binary_search(&local_start).is_ok() {
                destination.push(0);
            }
        }
    }
    for states in &mut merged {
        states.sort_unstable();
        states.dedup();
    }
    merged
}

pub(super) fn canonicalize_terminal_live_states(
    states: &mut [Vec<u32>],
    canonical: Option<u32>,
    aliases: &[u32],
) {
    let Some(canonical) = canonical else {
        return;
    };
    let canonical = canonical as usize;
    for &alias in aliases {
        let alias = alias as usize;
        if alias >= states.len() || canonical >= states.len() {
            continue;
        }
        let alias_states = std::mem::take(&mut states[alias]);
        states[canonical].extend(alias_states);
    }
    states[canonical].sort_unstable();
    states[canonical].dedup();
}

pub(super) fn canonicalize_possible_matches(
    possible_matches: &mut PossibleMatches,
    canonical: Option<u32>,
    aliases: &[u32],
) {
    let Some(canonical) = canonical else {
        return;
    };
    for &alias in aliases {
        let Some(alias_weight) = possible_matches.remove(&alias) else {
            continue;
        };
        possible_matches
            .entry(canonical)
            .and_modify(|weight| *weight = weight.union(&alias_weight))
            .or_insert(alias_weight);
    }
}

pub(super) fn canonicalize_parser_artifact_ignore(
    artifact: MappedArtifact<(DWA, PossibleMatches)>,
    canonical: Option<u32>,
    aliases: &[u32],
) -> MappedArtifact<(DWA, PossibleMatches)> {
    let ((dwa, mut possible_matches), id_map) = artifact.into_parts();
    canonicalize_possible_matches(&mut possible_matches, canonical, aliases);
    MappedArtifact::new((dwa, possible_matches), id_map)
}

pub(super) fn merged_terminal_live_states_owned_parent(
    parent: &mut Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    terminal_offsets: &[u32],
    tokenizer_state_offsets: &[u32],
    num_terminals: usize,
    preserve_parent_runtime: bool,
) -> Vec<Vec<u32>> {
    if parent.terminal_live_states.len() != parent.tokenizer.num_terminals() as usize
        || children.iter().any(|child| {
            child.constraint.terminal_live_states.len()
                != child.constraint.tokenizer.num_terminals() as usize
        })
    {
        return Vec::new();
    }
    // An authoritative segmented composition retains the source component as
    // an executable backend. Dynamic masking consumes `terminal_live_states`,
    // so do not hollow out that source merely to seed the outer merged index.
    // Static/non-segmented owned composition can keep the zero-copy move.
    let mut merged = if preserve_parent_runtime {
        parent.terminal_live_states.clone()
    } else {
        std::mem::take(&mut parent.terminal_live_states)
    };
    merged.resize_with(num_terminals, Vec::new);
    for (child_index, child) in children.iter().enumerate() {
        let component_index = child_index + 1;
        let terminal_offset = terminal_offsets[component_index] as usize;
        let state_offset = tokenizer_state_offsets[component_index];
        let local_start = child.constraint.tokenizer.start_state();
        for (local_terminal, states) in child
            .constraint
            .terminal_live_states
            .iter()
            .enumerate()
        {
            let destination = &mut merged[terminal_offset + local_terminal];
            destination.extend(states.iter().map(|&state| state_offset + state));
            if states.binary_search(&local_start).is_ok() {
                destination.push(0);
            }
            destination.sort_unstable();
            destination.dedup();
        }
    }
    merged
}

pub(super) fn build_composed_constraint_unfinalized(
    composed_table: ComposedTable,
    tokenizer: impl Into<Arc<Tokenizer>>,
    tokenizer_state_offsets: Vec<u32>,
    parser_dwa: DWA,
    parser_state_domain_labels: Vec<i32>,
    possible_matches: PossibleMatches,
    internal_ids: InternalIdMap,
    template_dfas_by_terminal: Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
    special_token_terminals: Vec<SpecialTokenTerminal>,
    embedded_end_token_ids: Vec<u32>,
    terminal_display_names: Vec<String>,
    ignore_terminal: Option<u32>,
    ignore_expr: Option<crate::automata::regex::Expr>,
    terminal_live_states: Vec<Vec<u32>>,
    tokenizer_fast_transitions: crate::runtime::FastTokenizerTransitions,
    defer_internal_token_bytes: bool,
    defer_dynamic_mask_vocab: bool,
    vocab: &Vocab,
) -> ConstraintComposition {
    let tokenizer = tokenizer.into();
    let total_started_at = Instant::now();
    let phase_started_at = Instant::now();
    let terminal_offsets = composed_table.terminal_offsets.clone();
    let parser_state_relations = composed_table.state_relations.clone();
    let relation_clone_ms = phase_started_at.elapsed().as_secs_f64() * 1000.0;
    let InternalIdMap {
        tokenizer_states,
        vocab_tokens,
        deferred_vocab_singleton_original_ids,
    } = internal_ids;
    debug_assert!(deferred_vocab_singleton_original_ids.is_none());
    let internal_token_bytes = if defer_internal_token_bytes {
        BTreeMap::new()
    } else {
        build_internal_token_bytes_from_groups(vocab, &vocab_tokens.internal_to_originals)
    };
    let ManyToOneIdMap {
        original_to_internal: state_to_internal_tsid,
        internal_to_originals: internal_tsid_to_states,
        representative_original_ids: _,
    } = tokenizer_states;
    debug_assert!(state_to_internal_tsid.iter().all(|&tsid| tsid != u32::MAX));
    // Direct component coordinates partition the composed raw tokenizer states:
    // each state has exactly one runtime TSID. Materialize the flat relation
    // directly instead of letting generic finalization allocate 1.4 million
    // temporary SmallVec rows and rediscover the same partition.
    // Sentinel `[u32::MAX]` means the relation is exactly the singleton
    // `state_to_internal_tsid` map. Runtime lookup already falls back to that
    // map; the sentinel prevents generic cache finalization from rebuilding two
    // redundant million-entry CSR vectors.
    let state_internal_tsid_offsets = vec![u32::MAX];
    let state_internal_tsids = Vec::new();
    let ManyToOneIdMap {
        original_to_internal: original_token_to_internal,
        internal_to_originals: internal_token_to_tokens,
        representative_original_ids: _,
    } = vocab_tokens;
    let tokenizer_has_epsilon_transitions = tokenizer.has_epsilon_transitions();
    let mut table = composed_table.table;
    table.set_embedded_end_token_ids(&embedded_end_token_ids);
    let num_terminals = table.num_terminals as usize;
    let phase_started_at = Instant::now();
    let dynamic_mask_vocab = if defer_dynamic_mask_vocab {
        // The explicit segmented result is a static coordinator: local masks
        // are authoritative in retained A components and cross-component masks
        // are authoritative in B. Its own direct-dynamic vocabulary is not on
        // the normal mask path. Keep it lazy just like a loaded static
        // constraint; if a future fallback ever requests it,
        // `dynamic_mask_vocab_for_runtime()` can reconstruct it from
        // `token_bytes` on demand.
        crate::runtime::DynamicMaskVocab::default()
    } else {
        runtime_dynamic_vocab_for_vocab(vocab)
    };
    let dynamic_vocab_ms = phase_started_at.elapsed().as_secs_f64() * 1000.0;
    let phase_started_at = Instant::now();
    let token_bytes = vocab.entries_arc();
    let token_bytes_ms = phase_started_at.elapsed().as_secs_f64() * 1000.0;
    let constraint = Constraint {
            end_tokens: std::sync::Arc::from([]),
        runtime_backend: ConstraintRuntimeBackend::Static,
        static_dynamic_overlay: None,
        boundary_trigger: crate::runtime::BoundaryTrigger::None,
        boundary_candidate_summary: std::sync::OnceLock::new(),
        late_grammar_slots: Vec::new(),
        late_bind_vocab: OnceLock::from(vocab.clone()),
        scoped_ignore_only_tokens: Vec::new(),
        scoped_ignore_prefix_fusions: Vec::new(),
        parser_dwa,
        packed_parser_dwa: None,
        parser_start_final_override: None,
        parser_top_accept: BTreeMap::new(),
        parser_top_accept_parts: BTreeMap::new(),
        direct_regular_l1_complete_by_terminal: BTreeMap::new(),
        packed_non_dwa_weights: None,
        direct_regular_wide_frontier_acceptance: Vec::new(),
        direct_regular_dynamic_hot_frontiers: Vec::new(),
        direct_regular_parser_state_acceptance: Vec::new(),
        direct_regular_automaton: None,
        table,
        terminal_display_names,
        tokenizer,
        boundary_completion_index: None,
        tokenizer_has_epsilon_transitions,
        ignore_terminal,
        special_token_terminals,
        dynamic_mask_vocab,
        lazy_dynamic_mask_vocab: OnceLock::new(),
        // Constraint composition is not allowed to rely on the legacy dynamic
        // possible-matches fallback. This is the exact transported and
        // reconciled table from every compiled component.
        // DO NOT REMOVE OR WEAKEN THIS COMMENT.
        possible_matches,
        possible_matches_complete: true,
        state_to_internal_tsid,
        internal_tsid_to_states,
        deferred_internal_tsid_to_states: Default::default(),
        composition_reset_tokens_by_terminal: Vec::new(),
        unbound_grammar_placeholders: BTreeMap::new(),
        composition_parser_templates_by_terminal: Vec::new(),
        composition_parser_characterizations_by_terminal: Vec::new(),
        composition_grammar_summary: None,
        terminal_live_states,
        state_internal_tsid_offsets,
        state_internal_tsids,
        runtime_source_state_offset: None,
        runtime_product_source_offsets: Vec::new(),
        runtime_product_source_states: Vec::new(),
        runtime_product_exact_source_states: Vec::new(),
        runtime_product_state_by_source_subset: FxHashMap::default(),
        template_dfas_by_terminal,
        fast_template_dfas_by_terminal: Vec::new(),
        original_token_to_internal,
        packed_original_token_to_internal: None,
        deferred_original_token_to_internal: OnceLock::new(),
        internal_token_to_tokens,
        deferred_internal_token_to_tokens: OnceLock::new(),
        token_bytes,
        packed_token_bytes: Some(crate::compiler::compile::vocab_packed_token_bytes(vocab)),
        internal_token_bytes,
        token_bytes_dense: Vec::new(),
        internal_token_buf_masks: Vec::new(),
        word_group_buf_masks: Vec::new(),
        pair_word_group_buf_masks: Default::default(),
        quad_word_group_buf_masks: Default::default(),
        super_word_group_buf_masks: Default::default(),
        mega_word_group_buf_masks: Default::default(),
        giga_word_group_buf_masks: Default::default(),
        word_group_sparse_masks: Vec::new(),
        word_group_prefix_buf_masks: Default::default(),
        word_group_sparse_prefix_entries: Vec::new(),
        quad_group_sparse_masks: Vec::new(),
        quad_group_dense_masks: Vec::new(),
        byte_group_sparse_masks: Vec::new(),
        byte_group_dense_masks: Vec::new(),
        word_group_sparse_total_entries: 0,
        word_group_sparse_max_entries: 0,
        all_tokens_buf_mask: Box::new([]),
        internal_token_dense_words: 0,
        weight_token_dense_masks: FxHashMap::default(),
        packed_dwa_token_dense_masks: Default::default(),
        weight_token_buf_masks: FxHashMap::default(),
        weight_token_sparse_buf_masks: FxHashMap::default(),
        range_final_token_sets: FxHashSet::default(),
        seed_terminal_dense: FxHashMap::default(),
        seed_terminal_dense_fallback: Default::default(),
        seed_universe_dense: Arc::<[u64]>::from(Vec::<u64>::new().into_boxed_slice()),
        dwa_fast_transitions: Default::default(),
        parser_runtime_caches_prebuilt: false,
        indexed_dag_dense_transitions: Vec::new(),
        indexed_dag_dense_finals: Vec::new(),
        tokenizer_fast_transitions,
        heavy_token_dense_masks: Vec::new(),
        heavy_token_indices: Vec::new(),
            internal_token_buf_flat: Box::new([]),
            backed_internal_token_buf_flat: None,
        internal_token_buf_offsets: Box::new([]),
        total_internal_buf_cost: 0,
        heavy_total_cost: 0,
        light_avg_cost_x256: 0,
        internal_token_buf_op_costs: Vec::new(),
        word_group_buf_op_costs: Vec::new(),
        final_mask_mapping: crate::runtime::mask_mapping::FinalMaskMapping::default(),
        parser_state_domain_labels,
        ignore_expr,
        serialized_artifact_cache: None,
        deferred_terminal_exprs_blob: None,
        deferred_terminal_exprs: Default::default(),
        deferred_composition_metadata_blob: None,
        composition_link_metadata_materialized: true,
        deferred_table_rules_blob: None,
        deferred_table_rules: Default::default(),
    };
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_unfinalized_build] relation_clone_ms={relation_clone_ms:.3} dynamic_vocab_ms={dynamic_vocab_ms:.3} token_bytes_ms={token_bytes_ms:.3} total_ms={:.3}",
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    ConstraintComposition {
        constraint,
        terminal_offsets,
        tokenizer_state_offsets,
        parser_state_relations,
    }
}

pub(super) fn finalize_composed_constraint(
    composed_table: ComposedTable,
    tokenizer: Tokenizer,
    tokenizer_state_offsets: Vec<u32>,
    parser_artifacts: MappedArtifact<(DWA, PossibleMatches)>,
    parser_top_accept: BTreeMap<i32, Weight>,
    parser_state_domain_labels: Vec<i32>,
    template_dfas_by_terminal: Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
    special_token_terminals: Vec<SpecialTokenTerminal>,
    embedded_end_token_ids: Vec<u32>,
    terminal_display_names: Vec<String>,
    ignore_terminal: Option<u32>,
    ignore_expr: Option<crate::automata::regex::Expr>,
    terminal_live_states: Vec<Vec<u32>>,
    tokenizer_fast_transitions: crate::runtime::FastTokenizerTransitions,
    structural_terminal_aliases: usize,
    components_have_no_runtime_product: bool,
    vocab: &Vocab,
) -> ConstraintComposition {
    let ((parser_dwa, possible_matches), internal_ids) = parser_artifacts.into_parts();
    let mut composition = build_composed_constraint_unfinalized(
        composed_table,
        tokenizer,
        tokenizer_state_offsets,
        parser_dwa,
        parser_state_domain_labels,
        possible_matches,
        internal_ids,
        template_dfas_by_terminal,
        special_token_terminals,
        embedded_end_token_ids,
        terminal_display_names,
        ignore_terminal,
        ignore_expr,
        terminal_live_states,
        tokenizer_fast_transitions,
        false,
        false,
        vocab,
    );
    composition.constraint.parser_top_accept = parser_top_accept;
    let lexer_product_started_at = Instant::now();
    let lexer_product_report = maybe_install_runtime_lexer_product(
        &mut composition.constraint,
        structural_terminal_aliases,
        components_have_no_runtime_product,
    );
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_runtime_lexer_product] attempted={} selected={} parser_overlap={} terminal_aliases={} source_states={} product_states={} source_transitions={} product_transitions={} multi_tsid_product_states={} total_ms={:.3}",
            lexer_product_report.attempted,
            lexer_product_report.selected,
            lexer_product_report.parser_overlap,
            structural_terminal_aliases,
            lexer_product_report.source_states,
            lexer_product_report.product_states,
            lexer_product_report.source_transitions,
            lexer_product_report.product_transitions,
            lexer_product_report.multi_tsid_product_states,
            lexer_product_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    composition.constraint.rebuild_runtime_caches();
    composition
}

pub(super) fn finalize_owned_composed_constraint_runtime(
    constraint: &mut Constraint,
    structural_terminal_aliases: usize,
    components_have_no_runtime_product: bool,
) -> f64 {
    let finalize_started_at = Instant::now();
    let lexer_product_started_at = Instant::now();
    let lexer_product_report = maybe_install_runtime_lexer_product(
        constraint,
        structural_terminal_aliases,
        components_have_no_runtime_product,
    );
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_runtime_lexer_product] attempted={} selected={} parser_overlap={} terminal_aliases={} source_states={} product_states={} source_transitions={} product_transitions={} multi_tsid_product_states={} total_ms={:.3}",
            lexer_product_report.attempted,
            lexer_product_report.selected,
            lexer_product_report.parser_overlap,
            structural_terminal_aliases,
            lexer_product_report.source_states,
            lexer_product_report.product_states,
            lexer_product_report.source_transitions,
            lexer_product_report.product_transitions,
            lexer_product_report.multi_tsid_product_states,
            lexer_product_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    constraint.rebuild_runtime_caches();
    finalize_started_at.elapsed().as_secs_f64() * 1000.0
}

pub(super) fn finalize_owned_composed_constraint_runtime_single_thread(
    constraint: &mut Constraint,
    structural_terminal_aliases: usize,
    components_have_no_runtime_product: bool,
) -> f64 {
    static FINALIZE_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    let pool = FINALIZE_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .thread_name(|index| format!("glrmask-finalize-{index}"))
            .build()
            .expect("build single-thread composition finalizer pool")
    });
    pool.install(|| {
        finalize_owned_composed_constraint_runtime(
            constraint,
            structural_terminal_aliases,
            components_have_no_runtime_product,
        )
    })
}

pub(super) fn materialized_constraint_for_composition(source: &Constraint) -> Result<Option<Constraint>, String> {
    if source.packed_parser_dwa.is_none() && source.deferred_composition_metadata_blob.is_none() {
        return Ok(None);
    }
    let mut materialized = source.clone();
    materialized.materialize_parser_dwa_for_compilation()?;
    materialized.materialize_non_dwa_weights_for_compilation()?;
    materialized.materialize_composition_metadata_for_compilation()?;
    Ok(Some(materialized))
}

/// Prepare a source for segmented linking without changing its local masking
/// backend. Dynamic constraints deliberately have no parser DWA: rebuilding
/// one from their retained grammar would turn a dynamic A component into a
/// static one before it is stored in `SegmentedParserComponent`.
pub(super) fn prepared_constraint_for_segmented_composition(
    source: &Constraint,
) -> Result<Option<Constraint>, String> {
    let recursive_compiler_table_missing = source.uses_compact_segmented_parser_runtime()
        && (source.table.num_states == 0 || source.table.action.is_empty());
    let recursive_compiler_tokenizer_missing = source
        .recursive_parser_layout()?
        .is_some_and(|layout| {
            source.tokenizer.num_states() != layout.total_tokenizer_states
                || source.state_to_internal_tsid.len() != layout.total_tokenizer_states as usize
        });
    if (source.deferred_composition_metadata_blob.is_none()
        || source.composition_link_metadata_materialized)
        && !recursive_compiler_table_missing
        && !recursive_compiler_tokenizer_missing
    {
        return Ok(None);
    }
    let mut prepared = source.clone();
    if source.deferred_composition_metadata_blob.is_some()
        && !source.composition_link_metadata_materialized
    {
        prepared.materialize_composition_link_metadata_for_compilation()?;
    }
    prepared.prepare_recursive_compiler_table_for_composition()?;
    prepared.prepare_recursive_compiler_tokenizer_for_composition()?;
    // Segmented A retains the component's own runtime backend. In particular,
    // a loaded static component's packed parser DWA is already a complete
    // runtime representation; unpacking it here merely to link B duplicates
    // work and memory without changing the component semantics. Dynamic A
    // likewise stays dynamic and deliberately has no parser DWA to rebuild.
    debug_assert_eq!(prepared.uses_dynamic_runtime(), source.uses_dynamic_runtime());
    Ok(Some(prepared))
}


pub(super) fn recursive_component_tokenizer_span(constraint: &Constraint) -> Result<u32, String> {
    Ok(constraint
        .recursive_parser_layout()?
        .map_or(constraint.tokenizer.num_states(), |layout| layout.total_tokenizer_states))
}

pub(super) fn merged_special_token_terminals_recursive_fast(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    terminal_offsets: &[u32],
) -> Vec<SpecialTokenTerminal> {
    let bound_parent_slots = children
        .iter()
        .flat_map(CompiledSubgrammarInput::placeholder_terminals)
        .collect::<BTreeSet<_>>();
    let mut merged = parent
        .special_token_terminals
        .iter()
        .filter(|special| {
            !bound_parent_slots.contains(&special.terminal_id)
                && !parent.is_late_grammar_placeholder_terminal(special.terminal_id)
        })
        .cloned()
        .collect::<Vec<_>>();
    for (child_index, child) in children.iter().enumerate() {
        let offset = terminal_offsets[child_index + 1];
        merged.extend(
            child
                .constraint
                .special_token_terminals
                .iter()
                .filter(|special| {
                    !child
                        .constraint
                        .is_late_grammar_placeholder_terminal(special.terminal_id)
                })
                .map(|special| SpecialTokenTerminal {
                    terminal_id: offset + special.terminal_id,
                    token_id: special.token_id,
                }),
        );
    }
    merged.sort_unstable_by_key(|special| (special.token_id, special.terminal_id));
    merged.dedup_by_key(|special| (special.token_id, special.terminal_id));
    merged
}
