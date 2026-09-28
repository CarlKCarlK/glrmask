//! Entry points for immutable linking and recursive component reuse.

use super::{
    Arc,
    BTreeMap,
    BTreeSet,
    BitSet,
    BoundaryRepair,
    BoundaryRuntimeCandidate,
    CompiledSubgrammarInput,
    ComposedTable,
    Constraint,
    ConstraintComposition,
    DWA,
    DirectComponentCoordinateMaps,
    FxHashSet,
    Instant,
    InternalIdMap,
    ManyToOneIdMap,
    OnceLock,
    ParserDwaComponent,
    PositiveBoundaryParser,
    PreparedOwnedComponentArtifacts,
    PublishedBoundaryRuntime,
    ScopedWeightOpCache,
    StructuralSharingReport,
    SubgrammarTableInput,
    Tokenizer,
    VecDeque,
    Vocab,
    WalkStaticLinkInputs,
    Weight,
    append_dynamic_direct_boundary_shards_for_unselected,
    boundary_id_map_for_selected_tokens,
    build_boundary_refinement_plan,
    build_commit_templates_from_raw_templates,
    build_composed_constraint_unfinalized,
    build_direct_component_state_coordinates,
    build_direct_component_token_coordinates,
    build_parser_default_domain_plan,
    build_recursive_tokenizer_internal_tsid_relation,
    build_segmented_parser_links,
    build_segmented_runtime_metadata,
    build_walk_static_boundary_link,
    canonicalize_possible_matches,
    canonicalize_terminal_live_states,
    clear_selected_nested_segmented_boundary_shards,
    component_ignores_are_globally_erasable,
    component_tokenizer_state_layout_owned_parent,
    components_have_no_compiled_eof_stack_rewrites,
    compose_profile_enabled,
    compose_subgrammar_tables_explicit_with_rules,
    dynamic_fallback_walk_link_output,
    ensure_positive_runtime_parser_dwa,
    finalize_owned_composed_constraint_runtime,
    finalize_owned_composed_constraint_runtime_single_thread,
    install_dynamic_direct_boundary_shards,
    install_published_static_boundary_shards,
    install_segmented_boundary_shards,
    legacy_splice_has_only_byte_terminal_continuations,
    macro_join,
    macro_parallelism_disabled,
    merged_ignore_terminals,
    merged_original_token_ids,
    merged_retained_terminal_exprs,
    merged_special_token_terminals,
    merged_special_token_terminals_recursive_fast,
    merged_terminal_display_names,
    merged_terminal_live_states_owned_parent,
    prebuild_segmented_token_mask_caches,
    prepare_deferred_component_artifacts,
    prepare_unmapped_component_possible_matches,
    prepared_constraint_for_segmented_composition,
    publish_boundary_parser_candidate_for_state_count,
    publish_real_boundary_parser_work,
    publish_static_boundary_shard_work,
    recursive_component_tokenizer_span,
    report_macro_item_timings,
    segmented_boundary_state_to_private_tsid,
};
use rayon::prelude::*;

pub(super) fn compose_dynamic_recursive_shared_fast(
    mut parent: Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    shared_children: &[Arc<Constraint>],
    vocab: &Vocab,
) -> Result<ConstraintComposition, String> {
    let started_at = Instant::now();
    if children.len() != shared_children.len() {
        return Err("dynamic recursive shared child/component count mismatch".into());
    }
    for (index, (input, shared)) in children.iter().zip(shared_children).enumerate() {
        if !std::ptr::eq(input.constraint, shared.as_ref()) {
            return Err(format!(
                "dynamic recursive shared child {index} does not match borrowed composition input"
            ));
        }
    }
    if children.is_empty() {
        return Err("constraint composition requires at least one child".into());
    }
    parent.materialize_composition_link_metadata_for_compilation()?;
    let components = std::iter::once(&parent)
        .chain(children.iter().map(|child| child.constraint))
        .collect::<Vec<_>>();
    let component_count = components.len();
    let vocab_check_started_at = Instant::now();
    for (component_index, constraint) in components.iter().enumerate() {
        if !constraint.token_bytes_match_vocab(vocab) {
            return Err(format!(
                "component {component_index} was not compiled for the supplied vocabulary",
            ));
        }
    }
    let vocab_check_ms = vocab_check_started_at.elapsed().as_secs_f64() * 1000.0;
    let placeholder_started_at = Instant::now();
    let component_end_token_ids = components
        .iter()
        .flat_map(|constraint| constraint.table.embedded_end_token_ids())
        .collect::<BTreeSet<_>>();
    validate_compiled_subgrammar_placeholders(
        &parent,
        children,
        vocab,
        &component_end_token_ids,
    )?;
    let placeholder_ms = placeholder_started_at.elapsed().as_secs_f64() * 1000.0;

    let terminal_checks_started_at = Instant::now();
    let mut terminal_offsets = Vec::with_capacity(components.len());
    let mut next_terminal = 0u32;
    for component in &components {
        terminal_offsets.push(next_terminal);
        next_terminal = next_terminal
            .checked_add(component.table.num_terminals)
            .ok_or_else(|| "dynamic recursive terminal coordinate overflow".to_owned())?;
    }
    let global_ignores = component_ignores_are_globally_erasable(&parent, children);
    let merged_ignores = merged_ignore_terminals(
        &parent,
        children,
        &terminal_offsets,
        global_ignores,
    );
    if merged_ignores.canonical.is_some() {
        return Err(
            "dynamic recursive fast path does not yet support globally-erased ignore aliases"
                .to_owned(),
        );
    }
    if children
        .iter()
        .any(|child| child.constraint.composition_start_nullable().unwrap_or(true))
    {
        return Err("dynamic recursive fast path does not yet support nullable children".into());
    }
    let terminal_checks_ms = terminal_checks_started_at.elapsed().as_secs_f64() * 1000.0;

    // Reuse checked component-prepared proper-prefix vocabularies. Unknown
    // metadata stays unknown: never run the static boundary compiler here and
    // never treat an unavailable proof as an empty candidate set.
    let candidate_started = Instant::now();
    let mut dynamic_candidate_ids = components.iter().map(|component| {
        if component.boundary_candidate_summary.get().is_some_and(|summary| summary.is_known()) {
            crate::compiler::composition::boundary::candidates::boundary_candidate_ids(component, vocab).0
        } else { None }
    }).collect::<Vec<_>>();
    let no_ignores = components.iter().all(|component| component.ignore_terminal.is_none()
        && component.table.skip_terminals.is_empty());
    if (!global_ignores || no_ignores) && dynamic_candidate_ids[0].is_some() {
        let entered = children.iter().map(|child| child.constraint).collect::<Vec<_>>();
        let calls = children.iter().flat_map(|child| std::iter::once(child.placeholder_terminal)
            .chain(child.additional_placeholder_terminals.iter().copied())).collect::<Vec<_>>();
        if let Ok(refined) = crate::compiler::composition::boundary::tail::build_root_call_candidates(
            &parent, &entered, &calls, vocab,
        ) {
            dynamic_candidate_ids[0].as_mut().expect("checked above")
                .retain(|id| refined.candidate_ids.binary_search(id).is_ok());
        }
    }
    if compose_profile_enabled() {
        eprintln!("[glrmask/profile][dynamic_prepared_boundary_vocabularies] counts={:?} total_ms={:.3}",
            dynamic_candidate_ids.iter().map(|ids| ids.as_ref().map(Vec::len)).collect::<Vec<_>>(),
            candidate_started.elapsed().as_secs_f64()*1000.0);
    }

    let tokenizer_span_started_at = Instant::now();
    let mut tokenizer_state_offsets = Vec::with_capacity(components.len());
    let mut next_tokenizer_state = 0u32;
    for component in &components {
        tokenizer_state_offsets.push(next_tokenizer_state);
        next_tokenizer_state = next_tokenizer_state
            .checked_add(recursive_component_tokenizer_span(component)?)
            .ok_or_else(|| "dynamic recursive tokenizer coordinate overflow".to_owned())?;
    }
    let tokenizer_span_ms = tokenizer_span_started_at.elapsed().as_secs_f64() * 1000.0;
    let links_started_at = Instant::now();
    let segmented_parser_links = build_segmented_parser_links(children)?;
    let links_ms = links_started_at.elapsed().as_secs_f64() * 1000.0;
    let validate_layout_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    if segmented_parser_links.is_empty() {
        return Err("dynamic recursive fast path requires at least one linker control".into());
    }

    let metadata_started_at = Instant::now();
    let terminal_display_names = merged_terminal_display_names(&parent, children);
    debug_assert_eq!(terminal_display_names.len(), next_terminal as usize);
    let special_token_terminals = merged_special_token_terminals_recursive_fast(
        &parent,
        children,
        &terminal_offsets,
    );
    let live_special_token_ids = special_token_terminals
        .iter()
        .map(|special| special.token_id)
        .collect::<BTreeSet<_>>();
    let embedded_end_token_ids = component_end_token_ids
        .intersection(&live_special_token_ids)
        .copied()
        .collect::<Vec<_>>();
    let metadata_ms = metadata_started_at.elapsed().as_secs_f64() * 1000.0;

    let shell_started_at = Instant::now();
    // The live recursive runtime never executes this outer table. Keep only
    // enough root grammar metadata for nullable/end-token facts and the global
    // terminal coordinate; exact LR behavior remains in the retained leaves.
    let shell_rules = parent.table.rules.first().cloned().into_iter().collect::<Vec<_>>();
    let shell_table = crate::compiler::glr::table::GLRTable {
        action: Vec::new(),
        goto: Vec::new(),
        num_states: 0,
        num_terminals: next_terminal,
        num_rules: shell_rules.len() as u32,
        rules: shell_rules,
        nonterminal_display_names: parent.table.nonterminal_display_names.clone(),
        embedded_start: parent.table.embedded_start.clone(),
        construction: parent.table.construction,
        admission_policy: parent.table.admission_policy,
        advance: Vec::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts: FxHashSet::default(),
        control_terminals: BTreeSet::new(),
        skip_terminals: BTreeSet::new(),
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers: Vec::new(),
    };
    let composed_table = ComposedTable {
        table: shell_table,
        terminal_offsets: terminal_offsets.clone(),
        placeholder_terminals: children
            .iter()
            .flat_map(CompiledSubgrammarInput::placeholder_terminals)
            .collect(),
        placeholder_component_indices: children
            .iter()
            .enumerate()
            .flat_map(|(index, child)| {
                std::iter::repeat_n(index + 1, 1 + child.additional_placeholder_terminals.len())
            })
            .collect(),
        state_relations: vec![Vec::new(); component_count],
        boundary_nonterminals: BTreeSet::new(),
        control_terminals: BTreeSet::new(),
        appended_parent_action_terminals: BTreeSet::new(),
    };
    let shell_ms = shell_started_at.elapsed().as_secs_f64() * 1000.0;

    let coordinator_started_at = Instant::now();
    let root_tokenizer = parent.tokenizer.clone();
    let root_fast_transitions = parent.tokenizer_fast_transitions.clone();
    let id_map = InternalIdMap {
        // DynamicDirect never consumes the coordinator TSID quotient. Keep a
        // one-class placeholder so serialized recursive compatibility metadata
        // can use a compact constant relation without building the old global
        // state/token partition.
        tokenizer_states: ManyToOneIdMap {
            original_to_internal: vec![0],
            internal_to_originals: Vec::new(),
            representative_original_ids: Vec::new(),
        },
        vocab_tokens: ManyToOneIdMap::empty(),
        deferred_vocab_singleton_original_ids: None,
    };
    let mut result = build_composed_constraint_unfinalized(
        composed_table,
        root_tokenizer,
        tokenizer_state_offsets.clone(),
        DWA::new(1, 0),
        Vec::new(),
        Default::default(),
        id_map,
        vec![None; next_terminal as usize],
        special_token_terminals,
        embedded_end_token_ids,
        terminal_display_names,
        None,
        None,
        Vec::new(),
        root_fast_transitions,
        true,
        true,
        vocab,
    );
    let coordinator_ms = coordinator_started_at.elapsed().as_secs_f64() * 1000.0;

    let publish_started_at = Instant::now();
    drop(components);
    let mut segmented_components = Vec::with_capacity(component_count);
    segmented_components.push(crate::runtime::SegmentedParserComponent {
        constraint: Arc::new(parent),
        boundary: None,
        tokenizer_state_offset: tokenizer_state_offsets[0],
        terminal_offset: terminal_offsets[0],
        global_terminal_aliases: Vec::new(),
        local_tsid_to_global_tsids: Vec::new(),
        root_disallowed_terminal: None,
        global_to_local_parser_state: Vec::new(),
    });
    for (child_index, child) in shared_children.iter().enumerate() {
        let component_index = child_index + 1;
        segmented_components.push(crate::runtime::SegmentedParserComponent {
            constraint: Arc::clone(child),
            boundary: None,
            tokenizer_state_offset: tokenizer_state_offsets[component_index],
            terminal_offset: terminal_offsets[component_index],
            global_terminal_aliases: Vec::new(),
            local_tsid_to_global_tsids: Vec::new(),
            root_disallowed_terminal: None,
            global_to_local_parser_state: Vec::new(),
        });
    }
    let overlay = result
        .constraint
        .static_dynamic_overlay
        .get_or_insert_with(Default::default);
    overlay.terminal_offsets = terminal_offsets;
    overlay.tokenizer_state_offsets = tokenizer_state_offsets;
    overlay.segmented_parser_components = segmented_components;
    overlay.segmented_parser_links = segmented_parser_links;
    overlay.segmented_parser_state_offsets.clear();
    overlay.segmented_mask_authoritative = true;
    overlay.segmented_static_baseline = false;
    overlay.segmented_component_union_root_dispatch.clear();
    overlay.segmented_boundary_parser = None;
    overlay.segmented_boundary_terminal_trie = None;
    install_dynamic_direct_boundary_shards(overlay, None);
    for (component, ids) in overlay.segmented_parser_components.iter_mut().zip(dynamic_candidate_ids) {
        if let Some(shard) = component.boundary.as_mut() {
            shard.candidate_tokens = ids.map(Arc::<[u32]>::from);
        }
    }
    overlay.segmented_boundary_shards = overlay.segmented_parser_components.iter()
        .filter_map(|component| component.boundary.clone()).collect();

    // Empty bytes are an explicit "provider-native only" marker. Dynamic
    // recomposition consumes the retained component tree directly. Static or
    // legacy compiler views may reconstruct from the provider in a later path.
    let _ = overlay
        .recursive_compiler_table
        .set(Arc::<[u8]>::from(Vec::<u8>::new().into_boxed_slice()));
    let publish_ms = publish_started_at.elapsed().as_secs_f64() * 1000.0;

    let layout_started_at = Instant::now();
    let layout = result
        .constraint
        .recursive_parser_layout_for_pending_root()?
        .ok_or_else(|| "dynamic recursive fast path failed to derive recursive layout".to_owned())?;
    let layout_ms = layout_started_at.elapsed().as_secs_f64() * 1000.0;
    let relation_started_at = Instant::now();
    // Authoritative DynamicDirect execution never consumes an outer recursive
    // TSID quotient. Leave it entirely absent in the live coordinator; the
    // serializer emits legacy compatibility rows only if/when save() is called.
    let relation_ms = relation_started_at.elapsed().as_secs_f64() * 1000.0;
    result.constraint.clear_recursive_legacy_boundary_start_states();
    result.constraint.clear_recursive_legacy_parser_state_projections();
    result.constraint.serialized_artifact_cache = None;

    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_dynamic_recursive_shared_fast] components={} terminals={} scoped_tokenizer_states={} vocab_check_ms={vocab_check_ms:.3} placeholder_ms={placeholder_ms:.3} terminal_checks_ms={terminal_checks_ms:.3} tokenizer_span_ms={tokenizer_span_ms:.3} links_ms={links_ms:.3} validate_layout_ms={validate_layout_ms:.3} metadata_ms={metadata_ms:.3} shell_ms={shell_ms:.3} coordinator_ms={coordinator_ms:.3} publish_ms={publish_ms:.3} layout_ms={layout_ms:.3} relation_ms={relation_ms:.3} total_ms={:.3}",
            component_count,
            next_terminal,
            layout.total_tokenizer_states,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(result)
}

pub(super) fn detach_recursive_component_compiler_views(constraint: &mut Constraint) -> Result<(), String> {
    if !constraint.uses_compact_segmented_parser_runtime() {
        return Ok(());
    }
    let Some(overlay) = constraint.static_dynamic_overlay.as_mut() else {
        return Ok(());
    };
    for component in &mut overlay.segmented_parser_components {
        let source = Arc::make_mut(&mut component.constraint);
        source.detach_recursive_outer_table()?;
        source.detach_recursive_outer_tokenizer()?;
    }
    Ok(())
}


/// Compose already-compiled parent and child constraints. The component
/// lexers, parse tables, parser DWAs, and possible-match tables are transported
/// and reused; only the restricted cross-component boundary repair is compiled
/// from the merged artifacts.
pub(super) fn validate_compiled_subgrammar_placeholders(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    vocab: &Vocab,
    component_end_token_ids: &BTreeSet<u32>,
) -> Result<(), String> {
    for (child_index, child) in children.iter().enumerate() {
        for placeholder_terminal in child.placeholder_terminals() {
            let has_byte_token_matches = parent
                .possible_matches
                .get(&placeholder_terminal)
                .is_some_and(|weight| !weight.is_empty());
            let has_exact_token_match = parent.special_token_terminals.iter().any(|special| {
                special.terminal_id == placeholder_terminal
                    && vocab.entries_map().contains_key(&special.token_id)
            });
            if has_byte_token_matches || has_exact_token_match {
                return Err(format!(
                    "subgrammar placeholder terminal {placeholder_terminal} for child {child_index} matches one or more model-vocabulary tokens in the compiled parent; placeholders must be non-vocabulary sentinels (for example an @token(...) id outside the supplied vocabulary)",
                ));
            }
            for special in parent
                .special_token_terminals
                .iter()
                .filter(|special| special.terminal_id == placeholder_terminal)
            {
                if component_end_token_ids.contains(&special.token_id) {
                    return Err(format!(
                        "subgrammar placeholder terminal {placeholder_terminal} for child {child_index} uses token ID {}, which is also configured as a grammar-level end token; every replaced placeholder must use a unique sentinel token ID",
                        special.token_id,
                    ));
                }
            }
        }
    }
    Ok(())
}

#[deprecated(
    note = "legacy compiled-artifact flattened composition is unsupported; use explicit segmented composition or the grammar-level inline reference"
)]
#[allow(unreachable_code)]
pub(crate) fn compose_constraints(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    vocab: &Vocab,
) -> Result<ConstraintComposition, String> {
    Err("legacy compiled-artifact flattened composition is unsupported; use explicit segmented composition or the grammar-level inline reference".into())
}

/// Fast consuming composition path. The parent remains the logical and physical
/// base of the returned ordinary `Constraint`; child tokenizer states are
/// appended to it, so the million-state parent is neither cloned nor rebased.
#[deprecated(
    note = "Legacy owned-parent composition is unsupported; use explicit segmented composition or compose_constraints for flattened validation"
)]
#[allow(unreachable_code)]
pub(crate) fn compose_constraints_owned_parent(
    parent: Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    vocab: &Vocab,
) -> Result<ConstraintComposition, String> {
    Err("legacy owned-parent composition is unsupported; use explicit segmented composition or compose_constraints for flattened validation".into())
}

/// Boundary policy for the exact segmented composition runtime. Component
/// masks remain independently backed by their source constraints in both
/// modes. `StaticParserDwa` compiles crossing shards; `Dynamic` stores no
/// required B artifact and uses the exact composed full-vocabulary walker.
/// Boundary triggers remain available as dormant acceleration metadata rather
/// than restricting correctness traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SegmentedBoundaryBackend {
    StaticParserDwa,
    Dynamic,
}

pub(crate) fn compose_constraints_owned_parent_segmented(
    parent: Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    vocab: &Vocab,
    boundary_backend: SegmentedBoundaryBackend,
) -> Result<ConstraintComposition, String> {
    compose_constraints_owned_parent_impl(
        parent,
        children,
        None,
        boundary_backend,
        None,
        vocab,
    )
}

/// Compose with a per-starting-component boundary policy. `static_components`
/// is indexed in component order (parent first, then supplied children). A set
/// bit compiles and uses that component's static boundary shard; an unset bit
/// uses the zero-data exact dynamic boundary walker, gated by the component's
/// reusable trigger metadata.
pub(crate) fn compose_constraints_owned_parent_segmented_hybrid(
    parent: Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    vocab: &Vocab,
    static_components: &BitSet,
) -> Result<ConstraintComposition, String> {
    if static_components.len() != children.len() + 1 {
        return Err(format!(
            "hybrid boundary policy has {} components for a {}-component composition",
            static_components.len(),
            children.len() + 1,
        ));
    }
    compose_constraints_owned_parent_impl(
        parent,
        children,
        None,
        SegmentedBoundaryBackend::StaticParserDwa,
        Some(static_components),
        vocab,
    )
}

pub(crate) fn compose_constraints_owned_parent_segmented_shared(
    parent: Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    shared_children: &[Arc<Constraint>],
    vocab: &Vocab,
    boundary_backend: SegmentedBoundaryBackend,
) -> Result<ConstraintComposition, String> {
    if shared_children.len() != children.len() {
        return Err("shared child/component count mismatch".into());
    }
    for (index, (input, shared)) in children.iter().zip(shared_children).enumerate() {
        if !std::ptr::eq(input.constraint, shared.as_ref()) {
            return Err(format!("shared child {index} does not match borrowed composition input"));
        }
    }
    compose_constraints_owned_parent_impl(
        parent,
        children,
        Some(shared_children),
        boundary_backend,
        None,
        vocab,
    )
}

#[deprecated(
    note = "Legacy owned-parent composition is unsupported; use explicit segmented composition or compose_constraints for flattened validation"
)]
#[allow(unreachable_code)]
pub(crate) fn compose_constraints_owned_parent_shared(
    parent: Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    shared_children: &[Arc<Constraint>],
    vocab: &Vocab,
) -> Result<ConstraintComposition, String> {
    Err("legacy owned-parent composition is unsupported; use explicit segmented composition or compose_constraints for flattened validation".into())
}

/// Link with an explicit supported boundary backend.
///
/// Retired flattened-splice entry points fail before reaching this pipeline;
/// there is deliberately no `None`/implicit backend and no dormant selection
/// through legacy experiment variables.
fn compose_constraints_owned_parent_impl(
    mut parent: Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    shared_children: Option<&[Arc<Constraint>]>,
    boundary_backend: SegmentedBoundaryBackend,
    static_boundary_components: Option<&BitSet>,
    vocab: &Vocab,
) -> Result<ConstraintComposition, String> {
    let direct_dynamic_boundary =
        boundary_backend == SegmentedBoundaryBackend::Dynamic;
    // Ownership and deferred link metadata are input preparation, not reasons
    // to rebuild a recursive component's flattened compiler table/tokenizer.
    // Retain already-prepared shared inputs unchanged; otherwise acquire the
    // ownership we must retain anyway and decode only the small link section.
    // This work stays INSIDE the binding call/timer. Borrowed/shared source
    // constraints are never mutated, and their static masking backend stays
    // packed. Nullable/global-ignore cases still take the exact general linker.
    let prepared_dynamic_children = if direct_dynamic_boundary
        && (shared_children.is_none()
            || children.iter().any(|child| {
                child.constraint.deferred_composition_metadata_blob.is_some()
                    && !child.constraint.composition_link_metadata_materialized
            }))
    {
        Some(children.iter().enumerate().map(|(index, input)| {
            let mut child = shared_children.map_or_else(
                || Arc::new(input.constraint.clone()),
                |shared| Arc::clone(&shared[index]),
            );
            if child.deferred_composition_metadata_blob.is_some()
                && !child.composition_link_metadata_materialized
            {
                Arc::make_mut(&mut child)
                    .materialize_composition_link_metadata_for_compilation()?;
            }
            Ok(child)
        }).collect::<Result<Vec<_>, String>>()?)
    } else {
        None
    };
    let prepared_dynamic_inputs = prepared_dynamic_children.as_ref().map(|owned| {
        children.iter().zip(owned).map(|(input, child)| CompiledSubgrammarInput {
            placeholder_terminal: input.placeholder_terminal,
            additional_placeholder_terminals: input.additional_placeholder_terminals,
            constraint: child.as_ref(),
        }).collect::<Vec<_>>()
    });
    let children = prepared_dynamic_inputs.as_deref().unwrap_or(children);
    let shared_children = prepared_dynamic_children.as_deref().or(shared_children);
    if direct_dynamic_boundary {
        if let Some(shared_children) = shared_children {
            let children_link_ready = shared_children.iter().all(|child| {
                child.deferred_composition_metadata_blob.is_none()
                    || child.composition_link_metadata_materialized
            });
            let nullable_child = children
                .iter()
                .any(|child| child.constraint.composition_start_nullable().unwrap_or(true));
            let terminal_offsets = {
                let mut offsets = Vec::with_capacity(children.len() + 1);
                let mut next = 0u32;
                for constraint in std::iter::once(&parent)
                    .chain(children.iter().map(|child| child.constraint))
                {
                    offsets.push(next);
                    next = match next.checked_add(constraint.table.num_terminals) {
                        Some(next) => next,
                        None => return Err("dynamic recursive terminal coordinate overflow".into()),
                    };
                }
                offsets
            };
            let global_ignores = component_ignores_are_globally_erasable(&parent, children);
            let has_global_ignore_alias = merged_ignore_terminals(
                &parent,
                children,
                &terminal_offsets,
                global_ignores,
            )
            .canonical
            .is_some();
            if children_link_ready && !nullable_child && !has_global_ignore_alias {
                return compose_dynamic_recursive_shared_fast(
                    parent,
                    children,
                    shared_children,
                    vocab,
                );
            }
            if compose_profile_enabled() {
                eprintln!(
                    "[glrmask/profile][constraint_dynamic_recursive_shared_fast] declined=true children_link_ready={} nullable_child={} global_ignore_alias={}",
                    children_link_ready,
                    nullable_child,
                    has_global_ignore_alias,
                );
            }
        } else if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_dynamic_recursive_shared_fast] declined=true reason=borrowed_child_requires_owned_arc"
            );
        }
    }
    let outer_started_at = Instant::now();
    let phase_started_at = Instant::now();
    {}
    let parent_parser_materialize_ms = phase_started_at.elapsed().as_secs_f64() * 1000.0;
    let phase_started_at = Instant::now();
    if direct_dynamic_boundary {
        parent.materialize_composition_link_metadata_for_compilation()?;
    } else {
        parent.materialize_composition_metadata_for_compilation()?;
    }
    {
        parent.prepare_recursive_compiler_table_for_composition()?;
        parent.prepare_recursive_compiler_tokenizer_for_composition()?;
    }
    let parent_metadata_materialize_ms = phase_started_at.elapsed().as_secs_f64() * 1000.0;
    let phase_started_at = Instant::now();
    let materialized_children = children
        .iter()
        .map(|child| {
            {
                prepared_constraint_for_segmented_composition(child.constraint)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let child_prepare_ms = phase_started_at.elapsed().as_secs_f64() * 1000.0;
    let phase_started_at = Instant::now();
    let materialized_any_child = materialized_children.iter().any(Option::is_some);
    let normalized_children = children
        .iter()
        .enumerate()
        .map(|(index, child)| CompiledSubgrammarInput {
            placeholder_terminal: child.placeholder_terminal,
            additional_placeholder_terminals: child.additional_placeholder_terminals,
            constraint: materialized_children[index]
                .as_ref()
                .unwrap_or(child.constraint),
        })
        .collect::<Vec<_>>();
    let normalize_inputs_ms = phase_started_at.elapsed().as_secs_f64() * 1000.0;
    let children = normalized_children.as_slice();
    // Compute the reusable bounded interface-tail envelope while the semantic
    // parent/child graph is still explicit. This is intentionally before any
    // flattening: child summaries compose through typed calls, so known parent
    // postambles survive without a raw lexer-state × parser-history product.
    let boundary_tail_prepared = {
        let bindings = children
            .iter()
            .flat_map(|child| {
                child
                    .placeholder_terminals()
                    .map(move |slot| (slot, child.constraint))
            })
            .collect::<Vec<_>>();
        match crate::compiler::composition::boundary::tail::build_composition_boundary_tail_r2(
            &parent,
            &bindings,
            vocab,
        ) {
            Ok(probe) => {
                if compose_profile_enabled() {
                    eprintln!(
                        "[glrmask/profile][boundary_tail_prepare] level=2 candidates={} exit_last1={} exit_pairs={} fp_iters={} widened={} child_ms={:.3} summary_ms={:.3} map_ms={:.3}",
                        probe.candidate_ids.len(),
                        probe.exit_last1_count,
                        probe.exit_pair_count,
                        probe.fixed_point_iterations,
                        probe.fixed_point_widened,
                        probe.child_summary_ms,
                        probe.summary_ms,
                        probe.map_ms,
                    );
                }
                Some((probe.candidate_ids, probe.fixed_point_widened, 2u8))
            }
            Err(r2_error) => match crate::compiler::composition::boundary::tail::build_composition_boundary_tail_r1(
                &parent,
                &bindings,
                vocab,
            ) {
                Ok(probe) => {
                    if compose_profile_enabled() {
                        eprintln!(
                            "[glrmask/profile][boundary_tail_prepare] level=1 candidates={} exit_bytes={} fp_iters={} widened={} summary_ms={:.3} map_ms={:.3} r2_unavailable={r2_error:?}",
                            probe.candidate_ids.len(),
                            probe.exit_byte_count,
                            probe.fixed_point_iterations,
                            probe.fixed_point_widened,
                            probe.summary_ms,
                            probe.map_ms,
                        );
                    }
                    Some((probe.candidate_ids, probe.fixed_point_widened, 1u8))
                }
                Err(r1_error) => {
                    if compose_profile_enabled() {
                        eprintln!(
                            "[glrmask/profile][boundary_tail_prepare] unavailable r2={r2_error:?} r1={r1_error:?}"
                        );
                    }
                    None
                }
            },
        }
    };
    // A packed child must be materialized into a compiler-owned clone, so the
    // shared-Arc optimization can no longer refer to the exact compiler input.
    let shared_children = if materialized_any_child {
        None
    } else {
        shared_children
    };

    parent.serialized_artifact_cache = None;
    {}
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_composition_owned_parent_prelude] parent_parser_materialize_ms={parent_parser_materialize_ms:.3} parent_metadata_materialize_ms={parent_metadata_materialize_ms:.3} child_prepare_ms={child_prepare_ms:.3} normalize_inputs_ms={normalize_inputs_ms:.3} total_ms={:.3}",
            outer_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    let total_started_at = Instant::now();
    let segmented_runtime_requested = true;
    let two_dwa_runtime_requested = false;
    let segmented_skip_requested = true;

    if children.is_empty() {
        return Err("constraint composition requires at least one child".into());
    }
    let parent_rules = parent.retained_table_rules()?;
    let child_rules = children
        .iter()
        .map(|child| child.constraint.retained_table_rules())
        .collect::<Result<Vec<_>, String>>()?;
    let segmented_parser_links = build_segmented_parser_links(children)?;
    let segmented_parser_state_offsets: Vec<u32> = Vec::new();
    let components_have_no_runtime_product = std::iter::once(&parent)
        .chain(children.iter().map(|child| child.constraint))
        .all(|constraint| constraint.runtime_source_state_offset().is_none());
    let component_end_token_ids = std::iter::once(&parent)
        .chain(children.iter().map(|child| child.constraint))
        .flat_map(|constraint| constraint.table.embedded_end_token_ids())
        .collect::<BTreeSet<_>>();
    for (component_index, constraint) in std::iter::once(&parent)
        .chain(children.iter().map(|child| child.constraint))
        .enumerate()
    {
        if !constraint.token_bytes_match_vocab(vocab) {
            return Err(format!(
                "component {component_index} was not compiled for the supplied vocabulary",
            ));
        }
    }
    validate_compiled_subgrammar_placeholders(
        &parent,
        children,
        vocab,
        &component_end_token_ids,
    )?;

    let global_ignores = component_ignores_are_globally_erasable(&parent, children);

    if compose_profile_enabled() {
        eprintln!("[glrmask/profile][constraint_linker_inputs] global_ignores={} parent_controls={} child_controls={:?} eof_rewrites_clean={} all_children_nonnullable={} legacy_follow_safe={} child_count={}",
            global_ignores,
            parent.table.control_terminals.len(),
            children.iter().map(|child| child.constraint.table.control_terminals.len()).collect::<Vec<_>>(),
            components_have_no_compiled_eof_stack_rewrites(&parent, children),
            children.iter().all(|child| !child.constraint.table.embedded_start_nullable()),
            legacy_splice_has_only_byte_terminal_continuations(&parent, parent_rules, children),
            children.len());
    }
    let table_inputs = children
        .iter()
        .map(|child| SubgrammarTableInput {
            placeholder_terminal: child.placeholder_terminal,
            additional_placeholder_terminals: child.additional_placeholder_terminals,
            table: &child.constraint.table,
            ignore_terminal: (!global_ignores)
                .then_some(child.constraint.ignore_terminal)
                .flatten(),
            start_nullable: child.constraint.table.embedded_start_nullable(),
        })
        .collect::<Vec<_>>();
    let all_children_nonnullable = children
        .iter()
        .all(|child| !child.constraint.table.embedded_start_nullable());
    // The authoritative segmented A+B runtime keeps linker call/return actions
    // as true zero-width parser controls for both static and dynamic boundary
    // policy. This is the representation in which a
    // child may return before the first terminal of the next model token and
    // the unchanged parent A factor can immediately own that token.  The
    // legacy splice erases that zero-width boundary into lookahead-dependent LR
    // rows, which in turn forces B to enumerate arbitrary parent-local suffixes.
    let retain_linker_controls = true;
    let use_legacy_splice = false;
    let table_started_at = Instant::now();
   let composed_table = compose_subgrammar_tables_explicit_with_rules(
                    &parent.table,
                    parent_rules,
                    (!global_ignores)
                        .then_some(parent.ignore_terminal)
                        .flatten(),
                    &table_inputs,
                    &child_rules,
                )?;
    let structural_started_at = Instant::now();
    let structural_states_before = composed_table.table.num_states as usize;
    // The quotient merges duplicate child LR regions, which breaks the
    // functional global-to-local parser-state relations the segmented runtime
    // requires (observed as "non-functional LR-state relation" on multi-child
    // links). Both segmented backends (StaticParserDwa and Dynamic) build the
    // same `state_relations` inverse, so the quotient must be skipped whenever
    // an explicit segmented boundary is requested. It is only a table-size
    // optimization; only the non-segmented (flattened) path keeps it.
    let attempt_structural_sharing = false;
    let structural_report = {
        StructuralSharingReport {
            nonterminals_before: composed_table.table.nonterminal_display_names.len(),
            nonterminal_classes: composed_table.table.nonterminal_display_names.len(),
            states_before: composed_table.table.num_states as usize,
            states_after: composed_table.table.num_states as usize,
            ..StructuralSharingReport::default()
        }
    };
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_structural_sharing] enabled={} terminal_aliases={} terminal_structural_matches={} terminal_exact_checks={} terminal_exact_unknown={} nonterminals_before={} nonterminal_classes={} contextual_candidate_groups={} contextual_saved_states={} states_before={} states_after={} saved_states={} total_ms={:.3}",
            attempt_structural_sharing,
            structural_report.terminal_aliases,
            structural_report.terminal_structural_matches,
            structural_report.terminal_exact_checks,
            structural_report.terminal_exact_unknown,
            structural_report.nonterminals_before,
            structural_report.nonterminal_classes,
            structural_report.contextual_candidate_groups,
            structural_report.contextual_states_saved,
            structural_report.states_before,
            structural_report.states_after,
            structural_report.states_before.saturating_sub(structural_report.states_after),
            structural_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    let table_ms = table_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!("[glrmask/profile][constraint_component_terminal_offsets] offsets={:?}", composed_table.terminal_offsets);
    }

    let metadata_started_at = Instant::now();
    let component_constraints = std::iter::once(&parent)
        .chain(children.iter().map(|child| child.constraint))
        .collect::<Vec<_>>();
    let composed_grammar_summary = {
        None
    };
    let (expected_tokenizer_state_offsets, merged_tokenizer_state_count) =
        component_tokenizer_state_layout_owned_parent(&component_constraints);
    let tokenizer_inputs = component_constraints
        .iter()
        .enumerate()
        .map(|(index, constraint)| {
            (constraint.composition_tokenizer(), composed_table.terminal_offsets[index])
        })
        .collect::<Vec<_>>();

    let component_views_ms = metadata_started_at.elapsed().as_secs_f64() * 1000.0;
    let specials_started_at = Instant::now();
    let control_elimination_report: Option<crate::compiler::glr::table::ControlEliminationReport> = {
        None
    };
    let control_elimination_ms = control_elimination_report
        .as_ref()
        .map(|report| report.elapsed_ms)
        .unwrap_or(0.0);
    let special_token_terminals = merged_special_token_terminals(
        &parent,
        children,
        &composed_table.terminal_offsets,
        &composed_table.table,
        &composed_table.control_terminals,
    );
    let parser_components = component_constraints
        .iter()
        .enumerate()
        .map(|(index, constraint)| ParserDwaComponent {
            constraint,
            parser_state_relation: &composed_table.state_relations[index],
            tokenizer_state_offset: expected_tokenizer_state_offsets[index],
            terminal_offset: composed_table.terminal_offsets[index],
            composed_table: Some(&composed_table.table),
        })
        .collect::<Vec<_>>();
    if compose_profile_enabled() {
        for (component_index, relation) in composed_table.state_relations.iter().enumerate() {
            let empty = relation.iter().filter(|targets| targets.is_empty()).count();
            let singleton = relation.iter().filter(|targets| targets.len() == 1).count();
            let multi = relation.len().saturating_sub(empty + singleton);
            let mut deltas = BTreeMap::<i64, usize>::new();
            for (local, targets) in relation.iter().enumerate() {
                if let [target] = targets.as_slice() {
                    *deltas.entry(*target as i64 - local as i64).or_default() += 1;
                }
            }
            let dominant = deltas
                .iter()
                .max_by_key(|(_, count)| **count)
                .map(|(delta, count)| (*delta, *count));
            eprintln!(
                "[glrmask/profile][constraint_state_relation_shape] component={} local_states={} empty={} singleton={} multi={} dominant_affine={:?} distinct_deltas={} total_targets={}",
                component_index,
                relation.len(),
                empty,
                singleton,
                multi,
                dominant,
                deltas.len(),
                relation.iter().map(Vec::len).sum::<usize>(),
            );
        }
    }
    let parser_default_domains = build_parser_default_domain_plan(
        &parser_components,
        composed_table.table.num_states,
    );
    if compose_profile_enabled() {
        let selected = parser_default_domains
            .component_domains
            .iter()
            .flatten()
            .count();
        let domain_states = parser_default_domains
            .component_domains
            .iter()
            .flatten()
            .map(|domain| domain.states.count_ones())
            .sum::<usize>();
        let per_component = parser_default_domains
            .component_domains
            .iter()
            .enumerate()
            .filter_map(|(index, domain)| {
                domain.as_ref().map(|domain| {
                    format!("{index}:{}:{}", domain.states.count_ones(), domain.predicted_saved_edges)
                })
            })
            .collect::<Vec<_>>()
            .join(",");
        eprintln!(
            "[glrmask/profile][constraint_parser_default_domains] selected={} domain_states={} predicted_saved_edges={} per_component=[{}]",
            selected,
            domain_states,
            parser_default_domains.predicted_saved_edges,
            per_component,
        );
    }
    let live_special_token_ids = special_token_terminals
        .iter()
        .map(|special| special.token_id)
        .collect::<BTreeSet<_>>();
    let embedded_end_token_ids = component_end_token_ids
        .intersection(&live_special_token_ids)
        .copied()
        .collect::<Vec<_>>();
    let specials_ms = specials_started_at.elapsed().as_secs_f64() * 1000.0;
    let token_ids_started_at = Instant::now();
    let original_token_ids = merged_original_token_ids(vocab, &special_token_terminals);
    let token_ids_ms = token_ids_started_at.elapsed().as_secs_f64() * 1000.0;
    let names_started_at = Instant::now();
    let terminal_display_names = merged_terminal_display_names(&parent, children);
    let merged_ignores = merged_ignore_terminals(
        &parent,
        children,
        &composed_table.terminal_offsets,
        global_ignores,
    );
    let ignore_terminal = merged_ignores.canonical;
    let names_ms = names_started_at.elapsed().as_secs_f64() * 1000.0;
    let metadata_ms = metadata_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_composition_metadata] component_views_ms={component_views_ms:.3} specials_ms={specials_ms:.3} token_ids_ms={token_ids_ms:.3} names_ms={names_ms:.3} total_ms={metadata_ms:.3}",
        );
    }
    // Publish state coordinates and the exact boundary-token coordinate as
    // soon as each becomes available. Component weight remapping then overlaps
    // boundary template/terminal/parser construction instead of sitting on the
    // serial parser-union path. When pre-table terminal construction already
    // built the exact raw-state partition, reuse it instead of reconstructing
    // the same partition in the coordinate lane.

    let state_map_cell = OnceLock::<Result<ManyToOneIdMap, String>>::new();
    let selected_boundary_tokens_cell =
        OnceLock::<Result<Option<Vec<u32>>, String>>::new();
    let walk_static_boundary_cell = OnceLock::<
        Result<Option<crate::compiler::composition::boundary::walk::WalkStaticLinkOutput>, String>,
    >::new();
    let skip_boundary_for_floor =
        false;
    let preparation_started_at = Instant::now();
    let ((tokenizer_result, tokenizer_ms), (prepared_components_result, (boundary_result, boundary_ms))) =
        macro_join(
            "compose_owned_tokenizer_and_component_boundary_prepare",
            || {
                let started_at = Instant::now();
                let parent_tokenizer = parent.composition_tokenizer().clone();
                let child_tokenizers = children
                    .iter()
                    .enumerate()
                    .map(|(index, child)| {
                        (
                            child.constraint.composition_tokenizer(),
                            composed_table.terminal_offsets[index + 1],
                        )
                    })
                    .collect::<Vec<_>>();
                let result = Tokenizer::disjoint_union_with_owned_parent(
                    parent_tokenizer,
                    composed_table.terminal_offsets[0],
                    &child_tokenizers,
                );
                (result, started_at.elapsed().as_secs_f64() * 1000.0)
            },
            || macro_join(
                "compose_component_coordinates_and_boundary",
            || {
                let ((state_result, component_state_ms), ((token_coordinate_result, token_coordinate_ms), (possible_matches_result, possible_matches_extract_ms))) =
                    macro_join(
                        "compose_component_state_and_token_pm_coordinates",
                        || {
                            let started_at = Instant::now();
                            let result = {
                                build_direct_component_state_coordinates(
                                    &parser_components,
                                    merged_tokenizer_state_count,
                                )
                            };
                            let published_state_map = result
                                .as_ref()
                                .map(|coordinates| coordinates.tokenizer_states.clone())
                                .map_err(Clone::clone);
                            assert!(
                                state_map_cell.set(published_state_map).is_ok(),
                                "component state map published twice",
                            );
                            (result, started_at.elapsed().as_secs_f64() * 1000.0)
                        },
                        || macro_join(
                            "compose_component_token_and_possible_matches_coordinates",
                        || {
                            let started_at = Instant::now();
                            let result = build_direct_component_token_coordinates(
                                &parser_components,
                                &original_token_ids,
                            );
                            (result, started_at.elapsed().as_secs_f64() * 1000.0)
                        },
                        || {
                            let started_at = Instant::now();
                            let result = prepare_unmapped_component_possible_matches(
                                &parser_components,
                                &composed_table.terminal_offsets,
                            );
                            (result, started_at.elapsed().as_secs_f64() * 1000.0)
                        },
                    ));
                let state_coordinates = state_result?;
                if compose_profile_enabled() {
                    eprintln!(
                        "[glrmask/profile][constraint_owned_component_coordinates] state_ms={component_state_ms:.3} token_ms={token_coordinate_ms:.3} possible_matches_ms={possible_matches_extract_ms:.3}",
                    );
                }
                let (
                    vocab_tokens,
                    local_to_global_tokens,
                    component_token_coordinate_is_singleton,
                ) = token_coordinate_result?;
                let component_maps = state_coordinates
                    .local_to_global_tsids
                    .into_iter()
                    .zip(local_to_global_tokens)
                    .map(|(local_to_global_tsids, local_to_global_tokens)| {
                        DirectComponentCoordinateMaps {
                            local_to_global_tsids,
                            local_to_global_tokens,
                        }
                    })
                    .collect::<Vec<_>>();
                let component_id_map = InternalIdMap {
                    tokenizer_states: state_coordinates.tokenizer_states,
                    vocab_tokens,
                    deferred_vocab_singleton_original_ids: None,
                };
                let selected_wait_started_at = Instant::now();
                let selected_boundary_tokens = loop {
                    if let Some(selected) = selected_boundary_tokens_cell.get() {
                        break selected.as_ref().map_err(Clone::clone)?.clone();
                    }
                    if rayon::yield_now().is_none() {
                        std::thread::yield_now();
                    }
                };
                let selected_wait_ms = selected_wait_started_at.elapsed().as_secs_f64() * 1000.0;
                let possible_matches_by_component = possible_matches_result?;
                let prepared = if let Some(selected_boundary_tokens) = selected_boundary_tokens {
                    let boundary_id_map_started_at = Instant::now();
                    let boundary_id_map = boundary_id_map_for_selected_tokens(
                        &component_id_map.tokenizer_states,
                        &selected_boundary_tokens,
                    )?;
                    let boundary_id_map_ms =
                        boundary_id_map_started_at.elapsed().as_secs_f64() * 1000.0;
                    let refinement_started_at = Instant::now();
                    let plan = build_boundary_refinement_plan(
                        component_id_map,
                        &boundary_id_map,
                        component_token_coordinate_is_singleton,
                    )
                    .ok_or_else(|| {
                        "component coordinate map does not cover boundary repair".to_string()
                    })?;
                    let refinement_ms = refinement_started_at.elapsed().as_secs_f64() * 1000.0;
                    let (automata_maps, possible_matches, remap_ms) =
                        prepare_deferred_component_artifacts(
                            possible_matches_by_component,
                            component_maps,
                            plan.component_token_map.as_deref(),
                            plan.common_map.num_tsids() as usize,
                        )?;
                    let (token_mask_caches, token_mask_cache_ms) =
                        prebuild_segmented_token_mask_caches(&plan.common_map);
                    if compose_profile_enabled() {
                        eprintln!(
                            "[glrmask/profile][constraint_owned_component_prepare_detail] selected_tokens={} selected_wait_ms={selected_wait_ms:.3} boundary_id_map_ms={boundary_id_map_ms:.3} refinement_ms={refinement_ms:.3} remap_ms={remap_ms:.3} token_mask_cache_ms={token_mask_cache_ms:.3}",
                            selected_boundary_tokens.len(),
                        );
                    }
                    PreparedOwnedComponentArtifacts {
                        automata_maps,
                        possible_matches,
                        id_map: plan.common_map,
                        boundary_tsid_map: Some(plan.boundary_tsid_map),
                        boundary_token_map: Some(plan.boundary_token_map),
                        token_mask_caches,
                        token_mask_cache_ms,
                        remap_ms,
                    }
                } else {
                    let (automata_maps, possible_matches, remap_ms) =
                        prepare_deferred_component_artifacts(
                            possible_matches_by_component,
                            component_maps,
                            None,
                            component_id_map.num_tsids() as usize,
                        )?;
                    let (token_mask_caches, token_mask_cache_ms) =
                        prebuild_segmented_token_mask_caches(&component_id_map);
                    if compose_profile_enabled() {
                        eprintln!(
                            "[glrmask/profile][constraint_owned_component_prepare_detail] selected_tokens=0 selected_wait_ms={selected_wait_ms:.3} boundary_id_map_ms=0.000 refinement_ms=0.000 remap_ms={remap_ms:.3} token_mask_cache_ms={token_mask_cache_ms:.3}",
                        );
                    }
                    PreparedOwnedComponentArtifacts {
                        automata_maps,
                        possible_matches,
                        id_map: component_id_map,
                        boundary_tsid_map: None,
                        boundary_token_map: None,
                        token_mask_caches,
                        token_mask_cache_ms,
                        remap_ms,
                    }
                };
                Ok::<_, String>((
                    prepared,
                    component_state_ms.max(token_coordinate_ms),
                    possible_matches_extract_ms,
                ))
            },
            || {
                if skip_boundary_for_floor || direct_dynamic_boundary {
                    let _ = selected_boundary_tokens_cell.set(Ok(None));
                    return (Ok(None), 0.0);
                }
                {
                    // Production static backend (Phase 2b): boundary shards
                    // come from the standard crossing-filtered walk, not from
                    // witness discovery. The walk publishes its crossing
                    // candidates as the selected boundary tokens so the
                    // component lane refines the shared token coordinate to
                    // cover them (like discovery publication). A
                    // static-requested link never silently succeeds as
                    // dynamic: the explicit env kill-switch is the only quiet
                    // all-dynamic route; nested and virtual-residual cases a
                    // static shard cannot serve are loud errors.
                    let started_at = Instant::now();
                    let num_components = children.len() + 1;
                    let requested_static = |index: usize| {
                        static_boundary_components.is_none_or(|bits| bits.contains(index))
                    };
                    let link = if std::env::var_os("GLRMASK_DISABLE_STATIC_BOUNDARY_SHARDS")
                        .is_some()
                    {
                        Ok(dynamic_fallback_walk_link_output(num_components))
                    } else {
                        let mut unsupported = None;
                        for index in 0..num_components {
                            if !requested_static(index) {
                                continue;
                            }
                            let component: &Constraint = if index == 0 {
                                &parent
                            } else {
                                children[index - 1].constraint
                            };
                            // Nested blocks check their whole leaf subtree for
                            // virtual residual runtimes inside the walk link
                            // (leaf precision); only flat components are
                            // checked here.
                            if component.has_recursive_segmented_parser_tree() {
                                continue;
                            }
                            if component.tokenizer.has_virtual_residual_runtime() {
                                unsupported = Some(index);
                                break;
                            }
                        }
                        if let Some(index) = unsupported {
                            Err(format!(
                                "walk static link component {index} requested a static shard but has a virtual residual runtime; leave it unselected (hybrid) or use the Dynamic boundary backend",
                            ))
                        } else {
                            build_walk_static_boundary_link(&WalkStaticLinkInputs {
                                parent: &parent,
                                children,
                                vocab,
                                static_components: static_boundary_components,
                                expected_terminal_offsets: &composed_table.terminal_offsets,
                            })
                        }
                    };
                    let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
                    if compose_profile_enabled() {
                        match &link {
                            Ok(output) => eprintln!(
                                "[glrmask/profile][constraint_walk_static_link] shards={} dynamic={} ms={elapsed_ms:.3}",
                                output.published_shards.len(),
                                output.all_dynamic,
                            ),
                            Err(error) => eprintln!(
                                "[glrmask/profile][constraint_walk_static_link] error={error} ms={elapsed_ms:.3}",
                            ),
                        }
                    }
                    return match link {
                        Err(error) => {
                            let _ = selected_boundary_tokens_cell.set(Err(error.clone()));
                            (Err(error), elapsed_ms)
                        }
                        Ok(output) => {
                            // Publish the walk's crossing candidates as the
                            // selected boundary tokens so the component lane
                            // refines the shared token coordinate to cover
                            // them (exactly like discovery publication). The
                            // runtime shard gate consults the outer token map;
                            // without refinement, crossing-only tokens have no
                            // outer internal id and shard-accepted tokens are
                            // silently dropped from masks. Empty/all-dynamic
                            // links publish None (nothing to cover).
                            let mut selected: Vec<u32> = output
                                .boundary_tokens_by_start_component
                                .iter()
                                .flatten()
                                .copied()
                                .collect();
                            selected.sort_unstable();
                            selected.dedup();
                            let publication = if output.all_dynamic || selected.is_empty() {
                                None
                            } else {
                                Some(selected)
                            };
                            let _ = selected_boundary_tokens_cell.set(Ok(publication));
                            let _ = walk_static_boundary_cell.set(Ok(Some(output)));
                            (Ok(None::<BoundaryRepair>), elapsed_ms)
                        }
                    };
                }
            },
        ));
    let (prepared_components, coordinate_ms, possible_matches_extract_ms) = prepared_components_result?;
    let boundary_repair = boundary_result?;
    let preparation_ms = preparation_started_at.elapsed().as_secs_f64() * 1000.0;
    let num_parser_states = composed_table.table.num_states;
    let num_terminals = composed_table.table.num_terminals as usize;
    let (
        mut boundary_work,
        static_boundary_shard_work,
        mut template_dfas_by_terminal,
        composition_parser_templates_by_terminal,
        commit_templates_deferred,
        boundary_tokens_by_start_component,
    ) = match boundary_repair {
        Some(boundary) => {
            debug_assert!(boundary.active_terminals.iter().any(|&active| active));
            (
                Some(boundary.parser),
                boundary.static_boundary_shards,
                boundary.template_dfas_by_terminal,
                boundary.composition_parser_templates_by_terminal,
                boundary.commit_templates_deferred,
                boundary.boundary_tokens_by_start_component,
            )
        }
        None => (
            None,
            None,
            vec![None; num_terminals],
            Vec::new(),
            false,
            None,
        ),
    };
    // A present partitioned-static work set is a complete composition-specific
    // B representation, including the intentionally empty-shard case. Once it
    // exists, building the legacy/global B is redundant: live masking and v24
    // serialization use the component-owned shards. Keep global publication
    // only for legacy/oracle paths where partitioned work is unavailable.
    let partitioned_static_boundary_complete = static_boundary_shard_work.is_some();

    // Build the signed/resolved/hash-consed positive boundary parser as soon as
    // boundary repair has produced its terminal automaton and templates.  This
    // half of B does not depend on the LR table itself; only the table-construction
    // mode controls grouped cancellation.  Starting here lets it overlap terminal
    // live-state merging, tokenizer assembly, coordinate canonicalization, and
    // unfinalized constraint construction.
    let early_boundary_positive_requested =
        std::env::var_os("GLRMASK_EXPERIMENT_EARLY_BOUNDARY_POSITIVE").is_some()
            && std::env::var_os("GLRMASK_EXPERIMENT_RUNTIME_BOUNDARY_TERMINAL_TRIE").is_none()
            && std::env::var_os("GLRMASK_EXPERIMENT_SKIP_BOUNDARY_PARSER_BUILD").is_none();
    let mut early_boundary_positive = None;
    if (segmented_skip_requested || two_dwa_runtime_requested)
        && !macro_parallelism_disabled()
        && !partitioned_static_boundary_complete
        && early_boundary_positive_requested
        && std::env::var_os("GLRMASK_EXPERIMENT_EARLY_BOUNDARY_PUBLISH").is_none()
        && let Some(work) = boundary_work.take()
    {
        let allow_grouped_cancellation = composed_table.table.construction
            == crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged;
        early_boundary_positive = Some(std::thread::spawn(move || {
            let candidate = work.materialize_positive_parser_without_table(
                allow_grouped_cancellation,
                num_parser_states,
            )?;
            publish_boundary_parser_candidate_for_state_count(candidate, num_parser_states)
        }));
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_early_boundary_positive] started=true table_clone_ms=0.000"
            );
        }
    }

    // Coordinate/boundary publication is complete. Release the immutable
    // parent borrows before consuming the owned-parent tokenizer below;
    // parser automata are intentionally materialized later, inside the final
    // parser-union window.
    drop(parser_components);
    drop(component_constraints);

    let terminal_live_started_at = Instant::now();
    // Explicit segmented composition retains the parent as recursive leaf 0.
    // That leaf owns a live tokenizer just like every retained child, so its
    // fast-transition facade must survive construction of the temporary flat
    // compiler tokenizer. The coordinator drops that flat tokenizer again
    // before publication.
    let preserve_parent_runtime = true;
    let mut terminal_live_states = merged_terminal_live_states_owned_parent(
        &mut parent,
        children,
        &composed_table.terminal_offsets,
        &expected_tokenizer_state_offsets,
        composed_table.table.num_terminals as usize,
        preserve_parent_runtime,
    );
    canonicalize_terminal_live_states(
        &mut terminal_live_states,
        merged_ignores.canonical,
        &merged_ignores.aliases,
    );
    let terminal_live_ms = terminal_live_started_at.elapsed().as_secs_f64() * 1000.0;

    let parent_fast_transitions = {
        parent.tokenizer_fast_transitions.clone()
    };
    let child_fast_transitions = children
        .iter()
        .enumerate()
        .map(|(index, child)| {
            (
                &child.constraint.tokenizer_fast_transitions,
                expected_tokenizer_state_offsets[index + 1],
            )
        })
        .collect::<Vec<_>>();
    let tokenizer_fast_transitions = parent_fast_transitions
        .append_rebased_children(&child_fast_transitions)
        .unwrap_or_default();
    let merged_exprs_for_restore = (parent.tokenizer.terminal_exprs().is_none()
        || children
            .iter()
            .any(|child| child.constraint.tokenizer.terminal_exprs().is_none()))
    .then(|| {
        let components = std::iter::once(&parent)
            .chain(children.iter().map(|child| child.constraint))
            .collect::<Vec<_>>();
        merged_retained_terminal_exprs(
            &components,
            &composed_table.terminal_offsets,
            composed_table.table.num_terminals,
        )
    })
    .flatten();
    let (mut tokenizer, tokenizer_state_offsets) = tokenizer_result;
    if tokenizer.terminal_exprs().is_none()
        && let Some(exprs) = merged_exprs_for_restore
    {
        tokenizer.restore_terminal_exprs(Some(exprs))?;
    }
    if let Some(canonical) = merged_ignores.canonical {
        tokenizer.canonicalize_terminal_aliases(canonical, &merged_ignores.aliases);
    }
    assert_eq!(
        tokenizer_state_offsets, expected_tokenizer_state_offsets,
        "predicted composed tokenizer state offsets differ from materialized union",
    );


    let PreparedOwnedComponentArtifacts {
        automata_maps,
        mut possible_matches,
        id_map,
        boundary_tsid_map,
        boundary_token_map,
        token_mask_caches,
        token_mask_cache_ms,
        remap_ms: component_remap_ms,
    } = prepared_components;
    canonicalize_possible_matches(
        &mut possible_matches,
        merged_ignores.canonical,
        &merged_ignores.aliases,
    );
    let segmented_boundary_state_to_tsid = {
        boundary_tsid_map
            .as_deref()
            .map(|map| segmented_boundary_state_to_private_tsid(&id_map.tokenizer_states, map))
            .transpose()?
    };
    let id_num_tsids = id_map.num_tsids();
    let id_max_internal_token = id_map.max_internal_token_id();
    let walk_link_ran = walk_static_boundary_cell
        .get()
        .is_some_and(|result| result.as_ref().is_ok_and(|output| output.is_some()));
    if boundary_work.is_none()
        && early_boundary_positive.is_none()
        && (boundary_tsid_map.is_some() || boundary_token_map.is_some())
        && !walk_link_ran
    {
        return Err(
            "prepared component artifacts retained boundary maps without a boundary repair"
                .to_string(),
        );
    }

    let union_started_at = Instant::now();
    let runtime_template_advance_requested =
        std::env::var_os("GLRMASK_ENABLE_TEMPLATE_DFA_ADVANCE").is_some()
            || std::env::var_os("GLRMASK_VALIDATE_TEMPLATE_DFA_ADVANCE").is_some();
    if commit_templates_deferred && runtime_template_advance_requested {
        let commit_template_started_at = Instant::now();
        template_dfas_by_terminal = build_commit_templates_from_raw_templates(
            &composition_parser_templates_by_terminal,
        );
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_deferred_commit_templates] entries={} ms={:.3}",
                template_dfas_by_terminal.iter().flatten().count(),
                commit_template_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
    } else if commit_templates_deferred && compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_deferred_commit_templates] entries=0 skipped=true reason=table_commit_runtime"
        );
    }

    // The complete boundary parser depends only on already-owned boundary
    // work plus the composed LR table.  Start it before the remaining runtime
    // artifact assembly so deterministic B publication can overlap that work.
    let early_boundary_publish_requested =
        std::env::var_os("GLRMASK_EXPERIMENT_EARLY_BOUNDARY_PUBLISH").is_some()
            && std::env::var_os("GLRMASK_EXPERIMENT_RUNTIME_BOUNDARY_TERMINAL_TRIE").is_none()
            && std::env::var_os("GLRMASK_EXPERIMENT_SKIP_BOUNDARY_PARSER_BUILD").is_none();
    let mut early_boundary_publish = None;
    if (segmented_skip_requested || two_dwa_runtime_requested)
        && !macro_parallelism_disabled()
        && !partitioned_static_boundary_complete
        && early_boundary_publish_requested
        && let Some(work) = boundary_work.take()
    {
        let clone_started_at = Instant::now();
        let table = composed_table.table.clone();
        let clone_ms = clone_started_at.elapsed().as_secs_f64() * 1000.0;
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_early_boundary_publish] started=true table_clone_ms={clone_ms:.3}"
            );
        }
        early_boundary_publish = Some(std::thread::spawn(move || {
            publish_real_boundary_parser_work(work, &table)
        }));
    }

    // Segmented runtime fast path: there is intentionally no flattened
    // component parser union to borrow from `parent`. Finishing in this branch
    // lets us move the already-owned parent directly into runtime segment zero
    // instead of cloning a multi-megabyte parser artifact just to keep it alive.
    {
        let unfinalized_started_at = Instant::now();
        let mut result = build_composed_constraint_unfinalized(
            composed_table,
            tokenizer,
            tokenizer_state_offsets,
            DWA::new(id_num_tsids, id_max_internal_token),
            parser_default_domains.parser_state_labels.clone(),
            possible_matches,
            id_map,
            template_dfas_by_terminal,
            special_token_terminals,
            embedded_end_token_ids,
            terminal_display_names,
            ignore_terminal,
            merged_ignores.canonical_expr.clone(),
            terminal_live_states,
            tokenizer_fast_transitions,
            true,
            true,
            vocab,
        );
        let unfinalized_ms = unfinalized_started_at.elapsed().as_secs_f64() * 1000.0;
        result.constraint.composition_parser_templates_by_terminal =
            composition_parser_templates_by_terminal;
        result.constraint.composition_grammar_summary = composed_grammar_summary.clone();
        if let Some(caches) = token_mask_caches {
            caches.install(&mut result.constraint);
            if compose_profile_enabled() {
                eprintln!(
                    "[glrmask/profile][constraint_segmented_token_mask_prebuild] installed=true build_ms={token_mask_cache_ms:.3}"
                );
            }
        }

        // Child preservation and positive boundary-parser construction are
        // independent.  Keep B as a positive NWA at this boundary: standalone
        // determinization is a publication/final-union concern, not part of
        // constructing B's parser language.
        let segment_prepare_started_at = Instant::now();
        let (child_clone_result, boundary_positive_result) = macro_join(
            "compose_child_preservation_and_boundary_positive",
            || {
                let started_at = Instant::now();
                let sources = if let Some(shared_children) = shared_children {
                    shared_children
                        .iter()
                        .map(|source| (Arc::clone(source), None))
                        .collect::<Vec<_>>()
                } else {
                    children
                        .iter()
                        .map(|child| child.constraint.clone())
                        .map(|source| (Arc::new(source), None))
                        .collect::<Vec<_>>()
                };
                let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
                (sources, elapsed_ms)
            },
            || -> Result<Option<BoundaryRuntimeCandidate>, String> {
                if partitioned_static_boundary_complete {
                    return Ok(None);
                }
                if early_boundary_positive.is_some() || early_boundary_publish.is_some() {
                    return Ok(None);
                }
                if std::env::var_os("GLRMASK_EXPERIMENT_SKIP_BOUNDARY_PARSER_BUILD").is_some() {
                    return Ok(None);
                }
                let Some(work) = boundary_work else {
                    return Ok(None);
                };
                let started_at = Instant::now();
                if boundary_backend == SegmentedBoundaryBackend::Dynamic
                    || (false
                        && std::env::var_os(
                            "GLRMASK_EXPERIMENT_RUNTIME_BOUNDARY_TERMINAL_TRIE",
                        )
                        .is_some())
                {
                    let (trie, boundary_id_map, template_cache) =
                        work.materialize_terminal_trie()?;
                    let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
                    if compose_profile_enabled() {
                        eprintln!(
                            "[glrmask/profile][constraint_segmented_boundary_positive] representation=terminal_trie nodes={} live_tsids={} token_classes={} build_ms={elapsed_ms:.3}",
                            trie.nodes.len(),
                            trie.root_by_tsid.iter().filter(|&&root| root != u32::MAX).count(),
                            boundary_id_map.num_internal_tokens(),
                        );
                    }
                    Ok(Some(BoundaryRuntimeCandidate::TerminalTrie {
                        trie,
                        id_map: boundary_id_map,
                        template_cache,
                        build_ms: elapsed_ms,
                    }))
                } else {
                    let (positive, boundary_id_map, template_cache) =
                        work.materialize_positive_parser(&result.constraint.table)?;
                    positive.ensure_positive()?;
                    let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
                    if compose_profile_enabled() {
                        match &positive {
                            PositiveBoundaryParser::Nwa(nwa) => eprintln!(
                                "[glrmask/profile][constraint_segmented_boundary_positive] representation=nwa states={} transitions={} positive_only=true build_ms={elapsed_ms:.3}",
                                nwa.num_states(),
                                nwa.num_transitions(),
                            ),
                            PositiveBoundaryParser::Dwa(dwa) => eprintln!(
                                "[glrmask/profile][constraint_segmented_boundary_positive] representation=dwa states={} transitions={} positive_only=true build_ms={elapsed_ms:.3}",
                                dwa.num_states(),
                                dwa.num_transitions(),
                            ),
                        }
                    }
                    Ok(Some(BoundaryRuntimeCandidate::Parser {
                        positive,
                        id_map: boundary_id_map,
                        template_cache,
                        build_ms: elapsed_ms,
                    }))
                }
            },
        );
        let segment_prepare_ms = segment_prepare_started_at.elapsed().as_secs_f64() * 1000.0;
        let (mut source_constraints, child_clone_ms) = child_clone_result;
        let boundary_candidate = boundary_positive_result?;
        // Static shard parser work must be compiled in the same recursive leaf
        // coordinate used by the live provider. Keep the terminal-side work
        // pending until segmented wrapper metadata has been installed below.
        let mut pending_static_boundary_shards = static_boundary_shard_work.unwrap_or_default();
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_segment_clone] cloned_children={} parser_states={} overlapped=true total_ms={child_clone_ms:.3}",
                source_constraints.len(),
                source_constraints
                    .iter()
                    .map(|(constraint, _)| constraint.parser_dwa.num_states() as usize)
                    .sum::<usize>(),
            );
        }
        source_constraints.insert(0, (Arc::new(parent), None));

        let global_state_count = result.constraint.table.num_states as usize;
        if source_constraints.len() != result.parser_state_relations.len()
            || source_constraints.len() != result.tokenizer_state_offsets.len()
            || source_constraints.len() != result.terminal_offsets.len()
        {
            return Err("segmented parser source/relation count mismatch".into());
        }

        // Runtime DWA publication and component-union certification are
        // independent.  In the table-free early-B path, ordinary constraint
        // finalization is independent as well: it never reads segmented A/B
        // metadata. Overlap that cache rebuild with the remaining deterministic
        // B work, then attach the overlay only after both complete.
        let overlap_finalize_with_boundary = std::env::var_os(
            "GLRMASK_EXPERIMENT_OVERLAP_FINALIZE_WITH_BOUNDARY",
        )
        .is_some()
            && early_boundary_positive.is_some();
        let mut overlapped_finalize_ms = None::<f64>;
        let segment_publish_started_at = Instant::now();
        let (boundary_runtime_result, segmented_result) = if overlap_finalize_with_boundary {
            let segmented_result = build_segmented_runtime_metadata(
                source_constraints,
                &result.parser_state_relations,
                &result.tokenizer_state_offsets,
                &result.terminal_offsets,
                &automata_maps,
                global_state_count,
                two_dwa_runtime_requested,
                &parser_default_domains,
                id_num_tsids,
                merged_ignores.canonical,
                &merged_ignores.aliases,
            );
            let handle = early_boundary_positive
                .take()
                .expect("overlapped finalization requires early deterministic B handle");
            let (boundary_runtime_result, finalize_ms) = std::thread::scope(|scope| {
                let finalize_handle = scope.spawn(|| {
                    finalize_owned_composed_constraint_runtime_single_thread(
                        &mut result.constraint,
                        structural_report.terminal_aliases,
                        components_have_no_runtime_product,
                    )
                });
                let boundary_runtime_result = handle
                    .join()
                    .map_err(|_| "early deterministic boundary parser thread panicked".to_string())?
                    .map(Some);
                let finalize_ms = finalize_handle
                    .join()
                    .expect("single-thread composition finalizer panicked");
                Ok::<_, String>((boundary_runtime_result, finalize_ms))
            })?;
            overlapped_finalize_ms = Some(finalize_ms);
            if compose_profile_enabled() {
                eprintln!(
                    "[glrmask/profile][constraint_finalize_overlap] enabled=true finalize_ms={finalize_ms:.3}"
                );
            }
            (boundary_runtime_result, segmented_result)
        } else {
            macro_join(
                "compose_boundary_publish_and_segmented_metadata",
            || -> Result<Option<PublishedBoundaryRuntime>, String> {
                if let Some(handle) = early_boundary_positive.take() {
                    let joined = handle
                        .join()
                        .map_err(|_| "early deterministic boundary parser thread panicked".to_string())??;
                    return Ok(Some(joined));
                }
                if let Some(handle) = early_boundary_publish.take() {
                    let joined = handle
                        .join()
                        .map_err(|_| "early boundary parser publication thread panicked".to_string())??;
                    return Ok(Some(joined));
                }
                let Some(candidate) = boundary_candidate else {
                    return Ok(None);
                };
                match candidate {
                    BoundaryRuntimeCandidate::Parser {
                        mut positive,
                        id_map: mut boundary_id_map,
                        template_cache,
                        build_ms: positive_build_ms,
                    } => {
                        let tsid_quotient =
                            positive.quotient_boundary_tsids(&mut boundary_id_map);
                        let normalize_started_at = Instant::now();
                        let parser_dwa = if std::env::var_os(
                            "GLRMASK_EXPERIMENT_SMALL_BOUNDARY_WEIGHT_DETERMINIZER",
                        )
                        .is_some()
                            && boundary_id_map.num_tsids() as usize <= 16
                            && boundary_id_map.num_internal_tokens() as usize <= 64
                        {
                            positive.into_runtime_dwa_small_boundary(
                                &result.constraint.table,
                                boundary_id_map.num_tsids() as usize,
                                boundary_id_map.num_internal_tokens() as usize,
                                tsid_quotient.as_deref(),
                            )
                        } else {
                            positive.into_runtime_dwa(&result.constraint.table)
                        };
                        ensure_positive_runtime_parser_dwa(&parser_dwa)?;
                        let normalize_ms =
                            normalize_started_at.elapsed().as_secs_f64() * 1000.0;
                        Ok(Some(PublishedBoundaryRuntime::Parser {
                            parser_dwa,
                            id_map: boundary_id_map,
                            template_cache,
                            positive_build_ms,
                            normalize_ms,
                            tsid_quotient,
                        }))
                    }
                    BoundaryRuntimeCandidate::TerminalTrie {
                        trie,
                        id_map,
                        template_cache,
                        build_ms,
                    } => Ok(Some(PublishedBoundaryRuntime::TerminalTrie {
                        trie,
                        id_map,
                        template_cache,
                        build_ms,
                    })),
                }
            },
            || build_segmented_runtime_metadata(
                source_constraints,
                &result.parser_state_relations,
                &result.tokenizer_state_offsets,
                &result.terminal_offsets,
                &automata_maps,
                global_state_count,
                two_dwa_runtime_requested,
                &parser_default_domains,
                id_num_tsids,
                merged_ignores.canonical,
                &merged_ignores.aliases,
            ),
            )
        };
        let segment_publish_wall_ms = segment_publish_started_at.elapsed().as_secs_f64() * 1000.0;
        let boundary_runtime = boundary_runtime_result?;
        let (segmented_components, deterministic_root_dispatch, segment_publish_ms) = segmented_result?;

        let overlay = result.constraint.static_dynamic_overlay.get_or_insert_with(|| {
            crate::runtime::StaticDynamicOverlayMetadata {
                terminal_offsets: result.terminal_offsets.clone(),
                tokenizer_state_offsets: result.tokenizer_state_offsets.clone(),
                repair_terminals: Vec::new(),
                non_parent_only_parser_states: vec![false; global_state_count],
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
            }
        });
        overlay.segmented_parser_components = segmented_components;
        overlay.segmented_parser_links = segmented_parser_links.clone();
        overlay.segmented_parser_state_offsets = segmented_parser_state_offsets.clone();
        overlay.segmented_mask_authoritative = true;
        if let Some(dispatch) = deterministic_root_dispatch {
            overlay.segmented_component_union_root_dispatch = dispatch;
        }
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_segmented_parser_runtime] components={} exact_singleton_relations=true parent_moved=true deterministic_union_view={} publish_ms={segment_publish_ms:.3}",
                overlay.segmented_parser_components.len(),
                !overlay.segmented_component_union_root_dispatch.is_empty(),
            );
        }

        if let Some(boundary_runtime) = boundary_runtime {
            match boundary_runtime {
                PublishedBoundaryRuntime::Parser {
                    parser_dwa,
                    id_map: boundary_id_map,
                    template_cache,
                    positive_build_ms,
                    normalize_ms: final_union_normalize_ms,
                    tsid_quotient,
                } => {
                    if let Some(template_cache) = template_cache {
                        result.constraint.composition_parser_templates_by_terminal =
                            template_cache;
                    }
                    let boundary_num_tsids = boundary_id_map.num_tsids() as usize;
                    let mut tokenizer_state_to_tsid = if std::env::var_os(
                        "GLRMASK_EXPERIMENT_EARLY_BOUNDARY_TSID_QUOTIENT",
                    )
                    .is_some()
                    {
                        boundary_id_map.tokenizer_states.original_to_internal.clone()
                    } else {
                        segmented_boundary_state_to_tsid
                            .clone()
                            .unwrap_or_else(|| boundary_id_map.tokenizer_states.original_to_internal.clone())
                    };
                    if let Some(old_to_new) = tsid_quotient.as_ref() {
                        for tsid in &mut tokenizer_state_to_tsid {
                            if *tsid == u32::MAX {
                                continue;
                            }
                            *tsid = old_to_new
                                .get(*tsid as usize)
                                .copied()
                                .unwrap_or(u32::MAX);
                        }
                    }
                    debug_assert_eq!(
                        tokenizer_state_to_tsid
                            .iter()
                            .copied()
                            .filter(|&value| value != u32::MAX)
                            .max()
                            .map_or(0, |value| value as usize + 1),
                        boundary_num_tsids,
                        "segmented boundary raw-state map must cover the private TSID coordinate",
                    );
                    let internal_token_to_originals =
                        boundary_id_map.vocab_tokens.internal_to_originals;
                    let overlay = result.constraint.static_dynamic_overlay.as_mut().expect(
                        "segmented component metadata must exist before boundary metadata",
                    );
                    if compose_profile_enabled() {
                        eprintln!(
                            "[glrmask/profile][constraint_segmented_boundary_parser] states={} transitions={} deterministic=true tsids={} token_classes={} positive_only=true positive_build_ms={positive_build_ms:.3} final_union_normalize_ms={final_union_normalize_ms:.3}",
                            parser_dwa.num_states(),
                            parser_dwa.num_transitions(),
                            tokenizer_state_to_tsid
                                .iter()
                                .copied()
                                .filter(|&v| v != u32::MAX)
                                .max()
                                .map_or(0, |v| v as usize + 1),
                            internal_token_to_originals.len(),
                        );
                    }
                    let boundary = Arc::new(crate::runtime::SegmentedBoundaryParser {
                            parser_dwa,
                            compact_parser_dwa: None,
                            recursive_parser_dwa: None,
                            uses_composed_tsid_coordinate: false,
                            tokenizer_state_to_tsid,
                            internal_token_to_originals,
                        });
                    overlay.segmented_boundary_parser = Some(Arc::clone(&boundary));
                    if !partitioned_static_boundary_complete {
                        install_segmented_boundary_shards(
                            overlay,
                            crate::runtime::SegmentedBoundaryShardBackend::StaticParser(boundary),
                            boundary_tokens_by_start_component.as_deref(),
                        );
                    }
                }
                PublishedBoundaryRuntime::CompactParser {
                    parser_dwa,
                    id_map: boundary_id_map,
                    template_cache,
                    positive_build_ms,
                    normalize_ms: final_union_normalize_ms,
                    tsid_quotient,
                } => {
                    if let Some(template_cache) = template_cache {
                        result.constraint.composition_parser_templates_by_terminal = template_cache;
                    }
                    let boundary_num_tsids = boundary_id_map.num_tsids() as usize;
                    let mut tokenizer_state_to_tsid = if std::env::var_os(
                        "GLRMASK_EXPERIMENT_EARLY_BOUNDARY_TSID_QUOTIENT",
                    )
                    .is_some()
                    {
                        boundary_id_map.tokenizer_states.original_to_internal.clone()
                    } else {
                        segmented_boundary_state_to_tsid
                            .clone()
                            .unwrap_or_else(|| boundary_id_map.tokenizer_states.original_to_internal.clone())
                    };
                    if let Some(old_to_new) = tsid_quotient.as_ref() {
                        for tsid in &mut tokenizer_state_to_tsid {
                            if *tsid == u32::MAX {
                                continue;
                            }
                            *tsid = old_to_new
                                .get(*tsid as usize)
                                .copied()
                                .unwrap_or(u32::MAX);
                        }
                    }
                    debug_assert_eq!(
                        tokenizer_state_to_tsid
                            .iter()
                            .copied()
                            .filter(|&value| value != u32::MAX)
                            .max()
                            .map_or(0, |value| value as usize + 1),
                        boundary_num_tsids,
                        "segmented compact boundary raw-state map must cover the private TSID coordinate",
                    );
                    let internal_token_to_originals = boundary_id_map.vocab_tokens.internal_to_originals;
                    if compose_profile_enabled() {
                        eprintln!(
                            "[glrmask/profile][constraint_segmented_boundary_parser] states={} transitions={} deterministic=true compact_weights={} tsids={} token_classes={} positive_only=true positive_build_ms={positive_build_ms:.3} final_union_normalize_ms={final_union_normalize_ms:.3}",
                            parser_dwa.num_states(),
                            parser_dwa.num_transitions(),
                            parser_dwa.weights.len(),
                            boundary_num_tsids,
                            internal_token_to_originals.len(),
                        );
                    }
                    let overlay = result.constraint.static_dynamic_overlay.as_mut().expect(
                        "segmented component metadata must exist before compact boundary metadata",
                    );
                    let boundary = Arc::new(crate::runtime::SegmentedBoundaryParser {
                            parser_dwa: DWA::new(0, 0),
                            compact_parser_dwa: Some(parser_dwa),
                            recursive_parser_dwa: None,
                            uses_composed_tsid_coordinate: false,
                            tokenizer_state_to_tsid,
                            internal_token_to_originals,
                        });
                    overlay.segmented_boundary_parser = Some(Arc::clone(&boundary));
                    if !partitioned_static_boundary_complete {
                        install_segmented_boundary_shards(
                            overlay,
                            crate::runtime::SegmentedBoundaryShardBackend::StaticParser(boundary),
                            boundary_tokens_by_start_component.as_deref(),
                        );
                    }
                }
                PublishedBoundaryRuntime::TerminalTrie {
                    trie,
                    id_map: boundary_id_map,
                    template_cache,
                    build_ms,
                } => {
                    if let Some(template_cache) = template_cache {
                        result.constraint.composition_parser_templates_by_terminal =
                            template_cache;
                    }
                    let tokenizer_state_to_tsid = segmented_boundary_state_to_tsid
                        .clone()
                        .unwrap_or(boundary_id_map.tokenizer_states.original_to_internal);
                    let internal_token_to_originals =
                        boundary_id_map.vocab_tokens.internal_to_originals;
                    if compose_profile_enabled() {
                        eprintln!(
                            "[glrmask/profile][constraint_segmented_boundary_terminal_nwa] states={} token_classes={} build_ms={build_ms:.3}",
                            trie.symbolic_nwa.as_ref().map_or(0, |nwa| nwa.nodes.len()),
                            internal_token_to_originals.len(),
                        );
                    }
                    let overlay = result.constraint.static_dynamic_overlay.as_mut().expect(
                        "segmented component metadata must exist before boundary metadata",
                    );
                    let boundary = Arc::new(crate::runtime::SegmentedBoundaryTerminalTrie {
                            nodes: trie.nodes,
                            root_by_tsid: trie.root_by_tsid,
                            tokenizer_state_to_tsid,
                            internal_token_to_originals,
                            symbolic_nwa: trie.symbolic_nwa,
                        });
                    overlay.segmented_boundary_terminal_trie = Some(Arc::clone(&boundary));
                    install_segmented_boundary_shards(
                        overlay,
                        crate::runtime::SegmentedBoundaryShardBackend::DynamicTerminalTrie(boundary),
                        boundary_tokens_by_start_component.as_deref(),
                    );
                }
            }
        }

        if direct_dynamic_boundary {
            let overlay = result.constraint.static_dynamic_overlay.as_mut().expect(
                "segmented component metadata must exist before dynamic boundary metadata",
            );
            // Dynamic composition has no composition-specific B artifact. The
            // zero-cost trigger level is conservative, so every starting
            // component initially falls back to one exact direct dynamic walk.
            install_dynamic_direct_boundary_shards(
                overlay,
                boundary_tokens_by_start_component.as_deref(),
            );
            overlay.segmented_boundary_parser = None;
            overlay.segmented_boundary_terminal_trie = None;
        }

        let published_static_boundary_shards = if partitioned_static_boundary_complete {
            let recursive_table = result
                .constraint
                .recursive_control_eliminated_parser_table()?
                .expect("partitioned static boundary requires recursive provider table");
            for work in &mut pending_static_boundary_shards {
                work.parser
                    .set_parser_table_override(Arc::clone(&recursive_table))?;
            }
            let static_shard_publish_started_at = Instant::now();
            let publish = |work| {
                    publish_static_boundary_shard_work(
                        work,
                        &result.constraint.table,
                        id_num_tsids as usize,
                    )
            };
            let published = if macro_parallelism_disabled() {
                let mut timings = Vec::with_capacity(pending_static_boundary_shards.len());
                let result = pending_static_boundary_shards
                    .into_iter()
                    .map(|work| {
                        let started = Instant::now();
                        let result = publish(work);
                        timings.push(started.elapsed().as_secs_f64() * 1000.0);
                        result
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                report_macro_item_timings("compose_static_boundary_shard_publish", &timings);
                result
            } else {
                pending_static_boundary_shards
                    .into_par_iter()
                    .map(publish)
                    .collect::<Result<Vec<_>, String>>()?
            };
            let static_shard_publish_ms =
                static_shard_publish_started_at.elapsed().as_secs_f64() * 1000.0;
            if compose_profile_enabled() {
                eprintln!(
                    "[glrmask/profile][constraint_static_boundary_shards] shards={} parser_states={} parser_transitions={} publish_ms={static_shard_publish_ms:.3}",
                    published.len(),
                    published
                        .iter()
                        .filter_map(|shard| shard.boundary.recursive_parser_dwa.as_ref())
                        .map(|dwa| dwa.num_states() as usize)
                        .sum::<usize>(),
                    published
                        .iter()
                        .filter_map(|shard| shard.boundary.recursive_parser_dwa.as_ref())
                        .map(|dwa| dwa.num_transitions() as usize)
                        .sum::<usize>(),
                );
            }
            published
        } else {
            Vec::new()
        };

        if partitioned_static_boundary_complete {
            let overlay = result.constraint.static_dynamic_overlay.as_mut().expect(
                "segmented component metadata must exist before static boundary shards",
            );
            overlay.segmented_boundary_parser = None;
            overlay.segmented_boundary_terminal_trie = None;
            install_published_static_boundary_shards(overlay, published_static_boundary_shards)?;
            if let Some(static_components) = static_boundary_components {
                append_dynamic_direct_boundary_shards_for_unselected(
                    overlay,
                    static_components,
                    boundary_tokens_by_start_component.as_deref(),
                );
            }
        }

        if let Some(walk_result) = walk_static_boundary_cell.get() {
            let walk = walk_result.as_ref().map_err(Clone::clone)?;
            if let Some(walk) = walk {
                let overlay = result.constraint.static_dynamic_overlay.as_mut().expect(
                    "segmented component metadata must exist before walk boundary shards",
                );
                overlay.segmented_boundary_parser = None;
                overlay.segmented_boundary_terminal_trie = None;
                if walk.all_dynamic {
                    // Exact dynamic link (explicit static-shards kill-switch):
                    // identical shard shape to a dynamic link.
                    install_dynamic_direct_boundary_shards(overlay, None);
                } else {
                    install_published_static_boundary_shards(
                        overlay,
                        walk.published_shards.clone(),
                    )?;
                    if walk.effective_static_components.iter().count()
                        != overlay.segmented_parser_components.len()
                    {
                        append_dynamic_direct_boundary_shards_for_unselected(
                            overlay,
                            &walk.effective_static_components,
                            Some(&walk.boundary_tokens_by_start_component),
                        );
                    }
                    clear_selected_nested_segmented_boundary_shards(
                        &mut result.constraint,
                        &walk.clear_nested_boundary_components,
                    )?;
                }
            }
        }

        // Production gate: every requested static component must actually
        // hold a StaticParser shard (or no shard when nothing crosses from
        // it). A requested-static component silently installed as dynamic
        // is a loud error, never a green link.
        if let Some(Ok(Some(walk))) = walk_static_boundary_cell.get() {
            if !walk.all_dynamic {
                let overlay = result
                    .constraint
                    .static_dynamic_overlay
                    .as_ref()
                    .expect("walk-static link requires overlay for backend gate");
                for index in 0..overlay.segmented_parser_components.len() {
                    let requested = static_boundary_components
                        .is_none_or(|bits| bits.contains(index));
                    if !requested {
                        continue;
                    }
                    let has_candidates = walk
                        .boundary_tokens_by_start_component
                        .get(index)
                        .is_some_and(|tokens| !tokens.is_empty());
                    let backend = overlay.segmented_parser_components[index]
                        .boundary
                        .as_ref()
                        .map(|shard| &shard.backend);
                    let satisfied = match backend {
                        Some(crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)) => true,
                        None => !has_candidates,
                        _ => false,
                    };
                    if !satisfied {
                        return Err(format!(
                            "walk static link component {index} requested a static shard but has no StaticParser backend (candidates={has_candidates}); declining",
                        ));
                    }
                }
            }
        }

        if result.constraint.uses_compact_segmented_parser_runtime() {
            // Derive and cache the authoritative recursive leaf layout before
            // discarding historical per-component projections. This is needed
            // for dynamic-only compositions too because a compiled child can
            // itself be bound beneath another recursive wrapper.
            result
                .constraint
                .recursive_parser_layout_for_pending_root()?
                .expect("compact segmented runtime must have a recursive parser layout");
            // Pin a walk-static install against the authoritative leaf layout:
            // the walk's private TSID maps assume expanded leaves (flat links:
            // direct components; nested links: recursively expanded intact
            // leaves) packed back-to-back in link order. On any mismatch,
            // decline loudly rather than misrouting queries or silently
            // succeeding as dynamic. This must never fire; if it does, it is
            // LOUD on purpose.
            if let Some(Ok(Some(walk))) = walk_static_boundary_cell.get() {
                if !walk.all_dynamic && !walk.published_shards.is_empty() {
                    let layout = result
                        .constraint
                        .recursive_parser_layout()?
                        .expect("walk-static link requires a recursive parser layout");
                    if layout.leaf_tokenizer_state_offsets
                        != walk.expected_leaf_tokenizer_offsets
                        || layout.total_tokenizer_states
                            != walk.expected_total_tokenizer_states
                        || layout.leaf_terminal_offsets
                            != walk.expected_leaf_terminal_offsets
                        || layout.total_leaf_terminals
                            != walk.expected_total_leaf_terminals
                    {
                        eprintln!(
                            "[glrmask/profile][constraint_walk_static_link_layout_mismatch] expected_offsets={:?} actual_offsets={:?} expected_total={} actual_total={} expected_term_offsets={:?} actual_term_offsets={:?} expected_term_total={} actual_term_total={} action=decline",
                            walk.expected_leaf_tokenizer_offsets,
                            layout.leaf_tokenizer_state_offsets,
                            walk.expected_total_tokenizer_states,
                            layout.total_tokenizer_states,
                            walk.expected_leaf_terminal_offsets,
                            layout.leaf_terminal_offsets,
                            walk.expected_total_leaf_terminals,
                            layout.total_leaf_terminals,
                        );
                        return Err(format!(
                            "walk static link leaf layout mismatch: expected offsets {:?} total {} terms {:?} total {}, runtime has offsets {:?} total {} terms {:?} total {}; declining static shards",
                            walk.expected_leaf_tokenizer_offsets,
                            walk.expected_total_tokenizer_states,
                            walk.expected_leaf_terminal_offsets,
                            walk.expected_total_leaf_terminals,
                            layout.leaf_tokenizer_state_offsets,
                            layout.total_tokenizer_states,
                            layout.leaf_terminal_offsets,
                            layout.total_leaf_terminals,
                        ));
                    }
                }
            }
            let recursive_tokenizer_tsids = build_recursive_tokenizer_internal_tsid_relation(
                &result.constraint,
                &automata_maps,
            )?;
            result
                .constraint
                .install_recursive_tokenizer_internal_tsids(recursive_tokenizer_tsids)?;
            // Recursive wrapper intervals are now the authoritative component
            // ownership coordinate. The materialized composed-state dispatch
            // was useful only to certify the transitional deterministic-union
            // representation; do not retain it in the live runtime.
            result
                .constraint
                .static_dynamic_overlay
                .as_mut()
                .expect("compact segmented runtime requires overlay")
                .segmented_component_union_root_dispatch
                .clear();
            result.constraint.clear_recursive_legacy_boundary_start_states();
            result.constraint.clear_recursive_legacy_parser_state_projections();
        }

        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_segmented_union_phases] unfinalized_ms={unfinalized_ms:.3} segment_prepare_ms={segment_prepare_ms:.3} segment_publish_wall_ms={segment_publish_wall_ms:.3}"
            );
            eprintln!(
                "[glrmask/profile][constraint_segmented_parser_flatten] skipped=true component_materialize_ms=0 deferred_component_remap_ms=0 direct_ms=0"
            );
        }
        let union_ms = union_started_at.elapsed().as_secs_f64() * 1000.0;
        let parser_runtime_cache_ms = 0.0;
        let token_cache_prebuild_ms = token_mask_cache_ms;

        let finalize_ms = overlapped_finalize_ms.unwrap_or_else(|| {
            finalize_owned_composed_constraint_runtime(
                &mut result.constraint,
                structural_report.terminal_aliases,
                components_have_no_runtime_product,
            )
        });
        if let Some((candidate_ids, widened, level)) = boundary_tail_prepared.as_ref() {
            if let Err(error) = crate::compiler::composition::boundary::candidates::install_precomputed_boundary_candidate_ids(
                &mut result.constraint,
                vocab,
                candidate_ids,
                *widened,
            ) {
                if compose_profile_enabled() {
                    eprintln!("[glrmask/profile][boundary_tail_install] level={level} skipped={error:?}");
                }
            } else if compose_profile_enabled() {
                eprintln!(
                    "[glrmask/profile][boundary_tail_install] level={level} candidates={} installed=true",
                    candidate_ids.len(),
                );
            }
        }
        detach_recursive_component_compiler_views(&mut result.constraint)?;
        result.constraint.detach_recursive_outer_table()?;
        result.constraint.detach_recursive_outer_tokenizer()?;
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_composition_owned_parent] components={} table_ms={table_ms:.3} control_elimination_ms={control_elimination_ms:.3} tokenizer_ms={tokenizer_ms:.3} coordinate_ms={coordinate_ms:.3} parser_extract_ms=0.000 boundary_ms={boundary_ms:.3} preparation_ms={preparation_ms:.3} terminal_live_ms={terminal_live_ms:.3} union_ms={union_ms:.3} parser_runtime_cache_ms={parser_runtime_cache_ms:.3} token_cache_prebuild_ms={token_cache_prebuild_ms:.3} finalize_ms={finalize_ms:.3} total_ms={:.3}",
                children.len() + 1,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return Ok(result);
    }

    }

/// Original model tokens accepted anywhere in an acyclic terminal DWA.
///
/// Fixpoint-free single topological pass; matches the fixpoint propagation
/// on acyclic inputs. Used for shard candidate-token triggers and gates.
/// Exact union of the correlated weights of all accepting paths.
/// This is the ordinary accepted-token summary before projecting away TSIDs.
pub(crate) fn accepted_weight_support(dwa: &DWA) -> Weight {
    assert!(dwa.is_acyclic(), "accepted-token summary expects acyclic DWA");
    let n = dwa.num_states() as usize;
    let mut indegree = vec![0usize; n];
    for state in dwa.states() {
        for &(target, _) in state.transitions.values() {
            indegree[target as usize] += 1;
        }
    }
    let mut queue = VecDeque::new();
    for (state, &degree) in indegree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(state as u32);
        }
    }
    let mut topo = Vec::with_capacity(n);
    while let Some(source) = queue.pop_front() {
        topo.push(source);
        for &(target, _) in dwa.states()[source as usize].transitions.values() {
            indegree[target as usize] -= 1;
            if indegree[target as usize] == 0 {
                queue.push_back(target);
            }
        }
    }
    assert_eq!(topo.len(), n);

    let mut reach = vec![Weight::empty(); n];
    reach[dwa.start_state() as usize] = Weight::all();
    let mut accepted = Weight::empty();
    let mut ops = ScopedWeightOpCache::default();
    for source in topo {
        let source_support = reach[source as usize].clone();
        if source_support.is_empty() {
            continue;
        }
        let state = &dwa.states()[source as usize];
        if let Some(final_weight) = state.final_weight.as_ref() {
            let support = ops.intersection(&source_support, final_weight);
            accepted = ops.union(&accepted, &support);
        }
        for &(target, ref edge_weight) in state.transitions.values() {
            let support = ops.intersection(&source_support, edge_weight);
            if support.is_empty() {
                continue;
            }
            reach[target as usize] = ops.union(&reach[target as usize], &support);
        }
    }

    accepted
}

pub(crate) fn accepted_original_tokens(
    dwa: &DWA,
    id_map: &InternalIdMap,
) -> BTreeSet<u32> {
    let profile = std::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some();
    let started = profile.then(Instant::now);
    let accepted = accepted_weight_support(dwa);
    let support_ms = started.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0);
    let mut originals = BTreeSet::new();
    for (_, internal_tokens) in accepted.raw_range_values() {
        for range in internal_tokens.ranges() {
            for internal_token in range {
                if let Some(ids) = id_map
                    .vocab_tokens
                    .internal_to_originals
                    .get(internal_token as usize)
                {
                    originals.extend(ids.iter().copied());
                }
            }
        }
    }
    if let Some(started) = started {
        eprintln!("[glrmask/profile][accepted_token_support] states={} edges={} tokens={} support_ms={support_ms:.3} total_ms={:.3}",
            dwa.num_states(), dwa.num_transitions(), originals.len(), started.elapsed().as_secs_f64() * 1000.0);
    }
    originals
}

/// Load a `vocab_dump.bin` cache file (test/bench helper).
pub(crate) fn load_vocab(path: &str) -> Vocab {
    use std::fs;
    let bytes = fs::read(path).expect("read vocab dump");
    fn read_u32(bytes: &[u8], offset: &mut usize) -> u32 {
        let end = *offset + 4;
        let value = u32::from_le_bytes(bytes[*offset..end].try_into().unwrap());
        *offset = end;
        value
    }
    let mut offset = 0usize;
    let count = read_u32(&bytes, &mut offset) as usize;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let id = read_u32(&bytes, &mut offset);
        let len = read_u32(&bytes, &mut offset) as usize;
        let end = offset + len;
        entries.push((id, bytes[offset..end].to_vec()));
        offset = end;
    }
    assert_eq!(offset, bytes.len());
    Vocab::new(entries)
}
