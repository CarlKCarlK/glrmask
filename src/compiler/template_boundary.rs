//! Static boundary queries over already-compiled table-free parser components.
//!
//! Reuse the ordinary boundary lexical walk, signed-template substitution,
//! exact cancellation/normalization, and StaticParser runtime publication.
//! The only parser input is the persisted finite relation inventory.
use std::{collections::BTreeMap, sync::Arc};
use crate::{Constraint, Error, Result, Vocab};
use crate::automata::lexer::tokenizer::{Lexer, Tokenizer};
use crate::automata::weighted_u32::terminal_automaton::TerminalAutomaton;
use crate::compiler::boundary_walk::{BoundaryShardLinkInputs, BoundaryShardWalkPlan};
use crate::compiler::constraint_compose::{WalkBoundaryShardWork, merged_retained_terminal_exprs};
use crate::compiler::stages::id_map_and_terminal_dwa::scope::ImmediateComponentId;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap, MappedArtifact};
use crate::compiler::constraint_possible_matches as pm;
use crate::ds::bitset::BitSet;
use crate::runtime::{ConstraintRuntimeBackend, SegmentedBoundaryShard, SegmentedBoundaryShardBackend};

fn fail(message: impl Into<String>) -> Error { Error::Compilation(message.into()) }

/// PM deliberately leaves tokens/states with no delayed-terminal matches
/// unmapped. Static boundary outputs still need those coordinates: complete
/// each dimension with one exact empty-PM class, keeping absent vocabulary IDs
/// unmapped. This refines PM, not the independent lexical boundary quotient.
fn complete_possible_match_coordinate(
    source: &InternalIdMap,
    tokenizer_states: u32,
    vocab: &Vocab,
) -> Result<InternalIdMap> {
    fn complete(mut map: Vec<u32>, present: impl IntoIterator<Item = usize>) -> Result<ManyToOneIdMap> {
        let empty = map.iter().copied().filter(|&id| id != u32::MAX).max()
            .map_or(Ok(0), |id| id.checked_add(1).ok_or_else(|| fail("PM class overflow")))?;
        let mut needs_empty = false;
        for original in present {
            let value = map.get_mut(original).ok_or_else(|| fail("PM source coordinate is incomplete"))?;
            if *value == u32::MAX { *value = empty; needs_empty = true; }
        }
        let count = empty.checked_add(u32::from(needs_empty)).ok_or_else(|| fail("PM class overflow"))?;
        Ok(ManyToOneIdMap::from_original_to_internal_allowing_unmapped(map, count))
    }
    let mut states = source.tokenizer_states.original_to_internal.clone();
    states.resize(tokenizer_states as usize, u32::MAX);
    let mut tokens = source.vocab_tokens.original_to_internal.clone();
    let token_slots = vocab.entries_map().keys().copied().max().map_or(Ok(0), |id|
        usize::try_from(id).ok().and_then(|id| id.checked_add(1)).ok_or_else(|| fail("vocabulary coordinate overflow")))?;
    tokens.resize(token_slots, u32::MAX);
    Ok(InternalIdMap {
        tokenizer_states: complete(states, 0..tokenizer_states as usize)?,
        vocab_tokens: complete(tokens, vocab.entries_map().keys().map(|&id| id as usize))?,
        deferred_vocab_singleton_original_ids: None,
    })
}

fn local_boundaries_are_static(constraint: &Constraint) -> bool {
    !constraint.uses_dynamic_runtime() && constraint.static_dynamic_overlay.as_ref().is_some_and(|overlay|
        overlay.segmented_parser_components.iter().all(|component| match component.boundary.as_ref() {
            Some(shard) => matches!(shard.backend, SegmentedBoundaryShardBackend::StaticParser(_)),
            None => true,
        }))
}

fn needs_static_boundaries(constraint: &Constraint) -> bool {
    constraint.uses_compact_segmented_parser_runtime() && (!local_boundaries_are_static(constraint)
        || constraint.static_dynamic_overlay.as_ref().is_some_and(|overlay|
            overlay.segmented_parser_components.iter().any(|component| needs_static_boundaries(&component.constraint))))
}

/// A same-owner child word needs B only if it can cross RETURN/CALL without
/// consuming a caller lexeme. Prove absence from exact local template domains,
/// for every concrete lower suffix, rather than from reachable grammar states.
/// Unknown output tops, nullable links and nested components retain the word.
fn retain_same_owner_paths(constraint: &Constraint, component: usize, bounded: bool) -> Result<bool> {
    if !bounded { return Ok(true); }
    if component == 0 { return Ok(false); }
    let overlay = constraint.static_dynamic_overlay.as_ref().unwrap();
    let parent = overlay.segmented_parser_components[0].constraint.as_ref();
    let child = overlay.segmented_parser_components[component].constraint.as_ref();
    if parent.template_parser.as_ref().unwrap().composition.is_some()
        || child.template_parser.as_ref().unwrap().composition.is_some() { return Ok(true); }
    let slots = overlay.segmented_parser_links.iter().filter(|link|
        link.parent_component == 0 && link.child_component as usize == component)
        .map(|link| link.slot_terminal as usize).collect::<std::collections::BTreeSet<_>>();
    if slots.is_empty() { return Ok(true); }
    // The original caller relation excludes CALL's appended child frame.
    // Exact RETURN removes that frame, exposing precisely these output tops.
    let views = slots.into_iter().map(|terminal|
        crate::runtime::parser_backend::scoped_program::ScopedProgram::terminal(parent, terminal))
        .collect::<Vec<_>>();
    let excluded = super::template_follow_support::disallowed_scoped(&views, views.len(), 1_000_000)
        .map_err(fail)?;
    Ok(!excluded.iter().all(|row| row.len() == views.len()))
}

// The native linker already ran the same necessary root-CALL proof for
// this actual flat, non-nullable binding. Reuse its link-local result, never
// a source component's broad certificate or a previous static cut subset.
fn prepared_root_call_candidates(constraint:&Constraint)->Option<Arc<[u32]>> {
    let overlay=constraint.static_dynamic_overlay.as_ref()?;
    if !overlay.segmented_parser_components.iter().all(|component|
        component.constraint.template_parser.as_ref().is_some_and(|parser|parser.composition.is_none()))
        || overlay.segmented_parser_links.iter().any(|link|link.child_start_nullable) {return None;}
    let shard=overlay.segmented_parser_components.first()?.boundary.as_ref()?;
    if shard.start_component!=0 || !matches!(shard.backend,SegmentedBoundaryShardBackend::DynamicDirect) {return None;}
    shard.candidate_tokens.as_ref().map(Arc::clone)
}

pub(crate) fn install(constraint: &mut Constraint, vocab: &Vocab) -> Result<()> {
    if !constraint.uses_compact_segmented_parser_runtime() { return Ok(()); }
    if !constraint.has_template_parser() { return Err(fail("static template boundary requires a table-free parser")); }
    if !needs_static_boundaries(constraint) { return Ok(()); }
    // A nested reusable component keeps its literal structure. Upgrade only
    // its boundary implementation, leaving parser/lexer coordinates unchanged.
    if let Some(overlay) = constraint.static_dynamic_overlay.as_mut() {
        for component in &mut overlay.segmented_parser_components {
            if needs_static_boundaries(&component.constraint) {
                install(Arc::make_mut(&mut component.constraint), vocab)?;
            }
        }
    }
    // Conversion to templates preserves every local parser/lexer coordinate.
    // An existing static B therefore remains a complete exact program. Only
    // changed child serialization needs invalidation when a descendant's B
    // was upgraded; no parser, lexer or boundary graph is rebuilt here.
    constraint.serialized_artifact_cache = None;
    if local_boundaries_are_static(constraint) { return Ok(()); }
    // A loaded component's packed non-DWA weights use its old PM coordinates.
    // Static publication replaces those coordinates and possible_matches, so
    // first preserve the other weights and retire the old packed ID map.
    constraint.materialize_non_dwa_weights_for_compilation().map_err(fail)?;
    let layout = constraint.recursive_parser_layout().map_err(fail)?
        .ok_or_else(|| fail("static template boundary has no scoped layout"))?;
    let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
    let started = std::time::Instant::now();
    if profile { eprintln!("[glrmask/profile][static_template_boundary] phase=begin leaves={} links={} stack_symbols={}",
        layout.leaves.len(), layout.links.len(), layout.total_states); }
    let certificate = if layout.links.iter().any(|link| link.child_start_nullable) {
        None
    } else {
        Some(super::boundary_transfer::certify_bounded_closure(&layout.links).map_err(fail)?)
    };
    let parser = constraint.template_parser.as_ref().ok_or_else(|| fail("missing template parser"))?;
    let composition = parser.composition.as_ref().ok_or_else(|| fail("missing composed template inventory"))?;
    if composition.control_start != layout.total_leaf_terminals {
        return Err(fail("static template lexical/parser terminal coordinates disagree"));
    }
    let component_count = constraint.static_dynamic_overlay.as_ref().unwrap().segmented_parser_components.len();
    let mut candidate_tokens_by_component = constraint.static_dynamic_overlay.as_ref().unwrap()
        .segmented_parser_components.iter().map(|component| {
            crate::compiler::boundary_candidates::persisted_boundary_candidate_ids(
                &component.constraint,
                vocab,
            )
        }).collect::<std::result::Result<Vec<_>, _>>().map_err(fail)?;
    let leaves = layout.leaves.iter().map(|leaf|
        constraint.constraint_at_recursive_component_path(&leaf.component_path)
            .ok_or_else(|| fail("static template leaf path is invalid")))
        .collect::<Result<Vec<_>>>()?;
    let global_ignores = super::constraint_compose::leaf_ignores_are_globally_erasable(&leaves);
    let no_ignores = leaves.iter().all(|leaf| leaf.ignore_terminal.is_none()
        && leaf.parser_skip_terminals().is_empty());
    if (!global_ignores || no_ignores) && certificate.is_some() {
        let components = &constraint.static_dynamic_overlay.as_ref().unwrap().segmented_parser_components;
        let parent = components[0].constraint.as_ref();
        let children = components.iter().skip(1).map(|component| component.constraint.as_ref()).collect::<Vec<_>>();
        let calls = constraint.static_dynamic_overlay.as_ref().unwrap().segmented_parser_links.iter()
            .filter(|link| link.parent_component == 0).map(|link| link.slot_terminal).collect::<Vec<_>>();
        if let Some(existing) = candidate_tokens_by_component[0].as_mut() {
            let before=existing.len();
            if let Some(prepared)=prepared_root_call_candidates(constraint) {
                existing.retain(|id|prepared.binary_search(id).is_ok());
                if profile {eprintln!("[glrmask/profile][boundary_root_call_candidates] reused_link_proof=true reusable={before} filtered={}",existing.len());}
            } else if let Ok(refined)=super::boundary_tail::build_root_call_candidates(parent,&children,&calls,vocab) {
                existing.retain(|id|refined.candidate_ids.binary_search(id).is_ok());
                if profile {eprintln!("[glrmask/profile][boundary_root_call_candidates] reused_link_proof=false reusable={before} filtered={} summary_ms={:.3} map_ms={:.3}",existing.len(),refined.summary_ms,refined.map_ms);}
            }
        }
    }
    let candidate_tokens_by_component = candidate_tokens_by_component.iter()
        .any(Option::is_some).then_some(candidate_tokens_by_component);
    let projected = leaves.iter().any(|leaf| leaf.tokenizer.has_any_virtual_runtime());
    let observation = projected.then(|| crate::runtime::static_observation::RecursiveStaticObservation::prepare(
        &leaves, vocab.max_token_byte_len())).transpose().map_err(fail)?;
    let views = leaves.iter().enumerate().map(|(index, leaf)| {
        observation.as_ref().map_or(Some(leaf.tokenizer.as_ref()), |o| o.leaf_view(index, leaf))
            .ok_or_else(|| fail("missing finite boundary observation view"))
    }).collect::<Result<Vec<_>>>()?;
    let tokenizer_inputs = views.iter().zip(&layout.leaf_terminal_offsets)
        .map(|(&view, &offset)| (view, offset)).collect::<Vec<_>>();
    let (mut merged, tokenizer_offsets) = Tokenizer::disjoint_union_with_terminal_offsets(&tokenizer_inputs);
    // The lexical union inserts one synthetic root before the intact leaves.
    // That root is useful during compilation but is never a live recursive
    // tokenizer state. Preserve the exact +1 injection rather than treating
    // the compiler union and runtime leaf coordinates as identical.
    if !projected && (tokenizer_offsets.iter().zip(&layout.leaf_tokenizer_state_offsets)
        .any(|(&compiled, &runtime)| runtime.checked_add(1) != Some(compiled))
        || layout.total_tokenizer_states.checked_add(1) != Some(merged.num_states())) {
        return Err(fail("static lexical observation changed the recursive state coordinate"));
    }
    if let Some(observation) = &observation {
        if observation.leaf_offsets[..leaves.len()] != tokenizer_offsets
            || observation.leaf_offsets.last() != Some(&merged.num_states()) {
            return Err(fail("finite boundary observation does not match the lexical union"));
        }
    }
    if merged.terminal_exprs().is_none() {
        if let Some(exprs) = merged_retained_terminal_exprs(&leaves, &layout.leaf_terminal_offsets,
            layout.total_leaf_terminals) {
            merged.restore_terminal_exprs(Some(exprs)).map_err(fail)?;
        }
    }
    if profile {for (index,leaf) in leaves.iter().enumerate() {
        eprintln!("[glrmask/profile][compiler_effect_metadata] leaf={index} summary={}",leaf.parser_backend_report()["compiler_effect_metadata"]);
    }}
    let state_counts = views.iter().map(|view| view.num_states()).collect::<Vec<_>>();
    let owners = layout.leaves.iter().map(|leaf| ImmediateComponentId(leaf.top_component)).collect::<Vec<_>>();
    let mut transparent = BitSet::new(layout.total_leaf_terminals as usize);
    for (leaf, &offset) in leaves.iter().zip(&layout.leaf_terminal_offsets) {
        for terminal in leaf.parser_skip_terminals().iter().copied().chain(leaf.ignore_terminal) {
            transparent.set((offset + terminal) as usize);
        }
    }
    let mut plans = Vec::with_capacity(component_count);
    for component in 0..component_count {
        let mut starts = vec![false; merged.num_states() as usize];
        for (leaf_index, leaf) in layout.leaves.iter().enumerate() {
            if leaf.top_component as usize != component { continue; }
            let offset = tokenizer_offsets[leaf_index] as usize;
            starts[offset..offset + state_counts[leaf_index] as usize].fill(true);
        }
        if starts.iter().any(|value| *value) {
            let retain_non_crossing_paths = retain_same_owner_paths(constraint, component, certificate.is_some())?;
            if profile { eprintln!("[glrmask/profile][static_template_boundary] phase=owner_paths component={component} retain_same_owner={retain_non_crossing_paths}"); }
            plans.push(BoundaryShardWalkPlan { start_component: component,
                crossing_owner: ImmediateComponentId(component as u32), commit_states: starts,
                // A nullable call can enable a terminal word entirely within
                // its caller's lexical scope. A cannot recognize the unresolved
                // placeholder, so B must keep these same-owner paths as well.
                // The exact control-star program below still decides viability.
                // RETURN followed by another CALL to the same component can
                // cross its interface without consuming a caller lexeme.
                // A complete local-domain proof may exclude that crossing;
                // otherwise the exact scoped control-star decides viability.
                retain_non_crossing_paths });
        }
    }
    let metadata = parser.link_grammar.as_deref()
        .filter(|grammar| grammar.terminal_count == layout.total_leaf_terminals);
    let context = metadata.map_or_else(|| crate::template_parser::static_compile::lexical_context(
        layout.total_leaf_terminals), |grammar| grammar.analyze(constraint.terminal_display_names.clone()));
    let follow_started = std::time::Instant::now();
    let follows = if metadata.is_some() {
        super::pipeline::compute_disallowed_follows(&context)
    } else {
        let proof_rows = super::template_follow_support::disallowed_scoped(&composition.views,
            layout.total_leaf_terminals as usize, 32_000_000).map_err(fail)?;
        proof_rows.into_iter().enumerate().filter_map(|(terminal, exclusions)| {
            if exclusions.is_empty() { return None; }
            let mut row = BitSet::new(layout.total_leaf_terminals as usize);
            for excluded in exclusions { row.set(excluded as usize); }
            Some((terminal as u32, row))
        }).collect::<BTreeMap<_, _>>()
    };
    let excluded_pairs = follows.values().map(BitSet::count_ones).sum::<usize>();
    let scoped_adjacent = if !global_ignores
        && super::boundary_env::enabled("GLRMASK_BOUNDARY_SCOPED_ADJACENCY") {
        metadata.and_then(|grammar| super::boundary_scoped_follow::optimized_scoped_follow_relation(
            &context, &grammar.component_nonterminals, &grammar.scoped_ignores))
    } else { None };
    let prepared_first = (!projected).then(|| leaves.iter().enumerate().map(|(index, leaf)|
        super::boundary_precomputed_completion::PreparedSourceSpan::for_component(leaf,
            tokenizer_offsets[index], layout.leaf_terminal_offsets[index])).collect::<Vec<_>>());
    if profile { eprintln!("[glrmask/profile][static_template_boundary] phase=follow_support rows={} excluded_pairs={} elapsed_ms={:.3}",
        follows.len(), excluded_pairs, follow_started.elapsed().as_secs_f64() * 1000.0); }
    if profile { eprintln!("[glrmask/profile][static_template_boundary] phase=scoped_adjacency metadata={} rows={} excluded_pairs={}",
        metadata.is_some(), scoped_adjacent.as_ref().map_or(0, BTreeMap::len),
        scoped_adjacent.as_ref().map_or(0, |rows| rows.values().map(BitSet::count_ones).sum::<usize>())); }
    let inputs = BoundaryShardLinkInputs {
        merged_tokenizer: &merged, vocab, grammar: &context, disallowed_follows: &follows,
        ignore_terminal: None, follow_transparent_ignores: Some(&transparent),
        terminal_offsets: &layout.leaf_terminal_offsets, leaf_to_immediate: Some(&owners),
        tokenizer_offsets: &tokenizer_offsets, component_state_counts: &state_counts,
        candidate_tokens_by_component: candidate_tokens_by_component.as_deref(), retain_parent_non_crossing_paths: certificate.is_none(),
        walk_plans: Some(plans),
    };
    let early_adjacency = super::boundary_env::enabled("GLRMASK_BOUNDARY_SCOPED_ADJACENCY_EARLY")
        .then_some(scoped_adjacent.as_ref()).flatten();
    let (mut walks, _) = super::boundary_walk::build_boundary_shard_walks_with_support(&inputs,
        early_adjacency, prepared_first.as_deref())
        .ok_or_else(|| fail("static template boundary lexical walk could not certify its scope"))?;
    if let Some(relation) = &scoped_adjacent {
        for walk in &mut walks {
            super::boundary_walk::apply_scoped_adjacency(walk, relation, layout.total_leaf_terminals);
        }
    }
    if profile { eprintln!("[glrmask/profile][static_template_boundary] phase=lexical_done shards={} elapsed_ms={:.3}",
        walks.len(), started.elapsed().as_secs_f64() * 1000.0); }
    // A remembers a delayed lexical decision using exact leaf terminal IDs.
    // Materialize its exclusions now, independently of the B quotient. The
    // shared static evaluator must never fall back to a vocabulary walk merely
    // because a remembered exclusion first appears after a token boundary.
    let prepared_possible=if projected {None} else {
        super::constraint_compose::prepared_leaf_possible_matches(&leaves,&tokenizer_offsets,
            &layout.leaf_terminal_offsets,merged.num_states(),vocab).map_err(fail)?
    };
    let reused_possible=prepared_possible.is_some();
    let possible=match prepared_possible {
        Some(possible)=>possible,
        None=>{
            let computed=pm::compute_constraint_possible_matches_for_vocab(&merged,vocab,
                pm::ConstraintPossibleMatchesConfig::EAGER);
            if !computed.complete {return Err(fail("static template boundary has incomplete exclusions"));}
            computed.mapped_possible_matches
        }
    };
    if profile {eprintln!("[glrmask/profile][boundary_possible_matches] reused_leaf_relations={reused_possible}");}
    let mut common = complete_possible_match_coordinate(possible.id_map(),
        merged.num_states(), vocab)?;
    if projected {
        // One observation per finite source state: neither A nor PM alone can
        // justify collapsing states which B may distinguish after a crossing.
        common.tokenizer_states = ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
            (0..merged.num_states()).collect(), merged.num_states());
    }
    let possible_matches = possible.remap_into_existing_common(&common)
        .into_artifact().into_iter().map(|(terminal, weight)| {
            let runtime_terminal = layout.outer_terminal_count.checked_add(terminal)
                .ok_or_else(|| fail("scoped exclusion terminal overflow"))?;
            Ok((runtime_terminal, weight))
        }).collect::<Result<BTreeMap<_, _>>>()?;
    let controls = (composition.control_start..composition.programs.len() as u32).collect::<Vec<_>>();
    let mut selected = controls.iter().copied().collect::<std::collections::BTreeSet<_>>();
    let mut ending_selected=std::collections::BTreeSet::new();
    for walk in &walks {
        for row in walk.output.dwa.states() {
            for (terminal, target, weight) in row.transitions.entries() {
                if weight.is_empty() { continue; }
                let terminal = u32::try_from(terminal).map_err(|_| fail("negative lexical terminal"))?;
                if terminal >= composition.control_start {
                    return Err(fail("lexical boundary terminal lies outside the ordinary inventory"));
                }
                if walk.output.dwa.states()[target as usize].final_weight.as_ref()
                    .is_some_and(|final_weight|weight.is_subset(final_weight)) {
                    ending_selected.insert(terminal);
                } else {
                    selected.insert(terminal);
                }
            }
        }
    }
    if profile { eprintln!("[glrmask/profile][static_template_boundary] phase=programs_selected selected={} total={} elapsed_ms={:.3}",
        selected.len(), composition.programs.len(), started.elapsed().as_secs_f64() * 1000.0); }
    let (templates, mut classes) = crate::template_parser::static_compile::prepare_scoped_boundary_programs(
        &composition.views, parser.state_count, &selected)?;
    let admissions=crate::template_parser::static_compile::prepare_scoped_boundary_admissions(
        &composition.views,&ending_selected,&mut classes)?;
    let predecessor=metadata.and_then(|metadata|metadata.stack_effects.as_ref()).filter(|summary|summary.states==parser.state_count)
        .and_then(|summary|super::boundary_stack_support::from_compiler_effects(summary).ok());
    let read_context=predecessor.as_ref().and_then(|certificate|certificate.native_context())
        .filter(|_|std::env::var_os("GLRMASK_PROFILE_BOUNDARY_NO_READ_SUPPORT").is_none());
    if profile { eprintln!("[glrmask/profile][static_template_boundary] phase=programs_ready templates={} admissions={} pop_classes={} predecessor={} elapsed_ms={:.3}",
        templates.len(),admissions.len(), classes.len(),read_context.is_some(), started.elapsed().as_secs_f64() * 1000.0); }
    let prepared=(!walks.is_empty()).then(|| super::boundary_transfer::template_program::prepare_classed_programs(
        &templates,Some(&admissions),&classes).expect("shared template constructor refused its checked contract; redundant expanded-builder fallback is disabled"));
    let mut published = Vec::with_capacity(walks.len());
    for walk in walks {
        let mut output = super::boundary_transfer::template_program::compile_classed_with_admissions(
            &templates,&admissions,&controls,certificate.as_ref(),&walk.output.dwa,&classes,read_context.as_ref(),prepared.as_ref()).map_err(fail)?;
        let mut id_map = walk.output.id_map;
        if projected {
            let target = InternalIdMap {
                tokenizer_states: common.tokenizer_states.clone(),
                vocab_tokens: id_map.vocab_tokens.clone(),
                deferred_vocab_singleton_original_ids: id_map.deferred_vocab_singleton_original_ids.clone(),
            };
            output.parser_dwa = MappedArtifact::new(output.parser_dwa, id_map)
                .remap_into_existing_common(&target).into_artifact();
            id_map = target;
        }
        let work = WalkBoundaryShardWork { start_component: walk.start_component as u32,
            terminal_automaton: TerminalAutomaton::Dwa(walk.output.dwa), id_map,
            candidate_tokens: Arc::from(walk.candidate_tokens.into_iter().collect::<Vec<_>>()) };
        let (mut shard, _) = super::boundary_transfer::publish_signed_shard(work, output,
            &tokenizer_offsets, &state_counts).map_err(fail)?;
        if projected {
            let boundary = Arc::make_mut(&mut shard.boundary);
            boundary.uses_composed_tsid_coordinate = true;
            boundary.tokenizer_state_to_tsid.clear();
        }
        if profile { eprintln!("[glrmask/profile][static_template_boundary] phase=shard_done component={} elapsed_ms={:.3}",
            shard.start_component, started.elapsed().as_secs_f64() * 1000.0); }
        published.push(shard);
    }
    let physical_tsids = if let Some(observation) = &observation {
        let mut rows = Vec::with_capacity(layout.total_tokenizer_states as usize);
        for (index, leaf) in leaves.iter().enumerate() {
            for local in 0..leaf.tokenizer.num_states() {
                rows.push(vec![observation.leaf_key(index, leaf, local)
                    .ok_or_else(|| fail("physical lexer state has no finite observation"))?]);
            }
        }
        rows
    } else {
        common.tokenizer_states.original_to_internal.iter().skip(1).map(|&tsid| vec![tsid]).collect()
    };
    let root_tsids = physical_tsids.iter().take(constraint.tokenizer.num_states() as usize)
        .map(|row| row[0]).collect();
    let inverse_tsids = if projected {
        (0..merged.num_states()).map(|id| vec![id]).collect()
    } else {
        common.tokenizer_states.internal_to_originals_vecs().into_iter()
            .map(|states| states.into_iter().filter_map(|state| state.checked_sub(1)).collect()).collect()
    };
    let overlay = constraint.static_dynamic_overlay.as_mut().unwrap();
    for component in &mut overlay.segmented_parser_components { component.boundary = None; }
    let mut shards = Vec::with_capacity(published.len());
    for published in published {
        let shard = SegmentedBoundaryShard {
            start_component: published.start_component,
            accepts_empty_stack: published.start_component == 0,
            start_parser_states: BitSet::new(0),
            candidate_tokens: Some(published.candidate_tokens),
            mask_vocabulary: Default::default(),
            backend: SegmentedBoundaryShardBackend::StaticParser(published.boundary),
        };
        overlay.segmented_parser_components[shard.start_component as usize].boundary = Some(shard.clone());
        shards.push(shard);
    }
    overlay.segmented_boundary_shards = shards;
    // B keeps its private quotient. The coordinator coordinate is independently
    // exact for correlated A exclusions and for mapping every B output token.
    overlay.recursive_tokenizer_internal_tsids = Default::default();
    overlay.recursive_tokenizer_internal_tsids.set(Arc::new(physical_tsids))
        .map_err(|_| fail("static template TSID coordinate initialized twice"))?;
    overlay.recursive_static_observation = observation.map(Arc::new);
    constraint.state_to_internal_tsid = root_tsids;
    constraint.internal_tsid_to_states = inverse_tsids;
    constraint.deferred_internal_tsid_to_states = Default::default();
    constraint.state_internal_tsid_offsets = vec![u32::MAX];
    constraint.state_internal_tsids.clear();
    constraint.packed_original_token_to_internal = None;
    constraint.deferred_original_token_to_internal = Default::default();
    constraint.internal_token_to_tokens = common.vocab_tokens.internal_to_originals_vecs();
    constraint.deferred_internal_token_to_tokens = Default::default();
    constraint.internal_token_bytes = pm::build_internal_token_bytes_from_groups(vocab,
        &common.vocab_tokens.internal_to_originals);
    constraint.original_token_to_internal = common.vocab_tokens.original_to_internal;
    constraint.possible_matches = possible_matches;
    constraint.possible_matches_complete = true;
    constraint.seed_universe_dense = Arc::from([]);
    constraint.seed_terminal_dense.clear();
    constraint.parser_runtime_caches_prebuilt = false;
    constraint.runtime_backend = ConstraintRuntimeBackend::Static;
    constraint.serialized_artifact_cache = None;
    constraint.validate_template_composition_layout().map_err(fail)?;
    constraint.rebuild_runtime_caches();
    if let Some(observation) = constraint.static_dynamic_overlay.as_ref()
        .and_then(|overlay| overlay.recursive_static_observation.as_ref()) {
        observation.validate(constraint).map_err(fail)?;
    }
    if profile { eprintln!("[glrmask/profile][static_template_boundary] phase=done elapsed_ms={:.3}",
        started.elapsed().as_secs_f64() * 1000.0); }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BuildOptions, Grammar, Optimization};

    #[test]
    fn prepared_root_call_proof_matches_recomputation_and_static_runtime_after_reload() {
        let vocab=Vocab::new(vec![(0,b"x".to_vec()),(1,b"a".to_vec()),(2,b"y".to_vec()),
            (3,b"xay".to_vec()),(4,b"xa".to_vec()),(5,b"ay".to_vec()),(6,b"xb".to_vec())]);
        for nullable in [false,true] {
            let child=Grammar::from_ebnf(if nullable {r#"start ::= "a" | """#} else {r#"start ::= "a""#})
                .compile(&vocab).unwrap();
            let dynamic=Grammar::from_glrm(r#"glrm 1; extern grammar child; start root; nt root = "x" child "y";"#)
                .compile_unlinked(&vocab).unwrap().bind("child",&child).unwrap()
                .link_with(BuildOptions::default().optimization(Optimization::FastBuild)).unwrap();
            let prepared=prepared_root_call_candidates(&dynamic);
            assert_eq!(prepared.is_none(),nullable);
            if let Some(prepared)=prepared {
                let overlay=dynamic.static_dynamic_overlay.as_ref().unwrap();
                let parent=overlay.segmented_parser_components[0].constraint.as_ref();
                let children=overlay.segmented_parser_components.iter().skip(1)
                    .map(|component|component.constraint.as_ref()).collect::<Vec<_>>();
                let calls=overlay.segmented_parser_links.iter().map(|link|link.slot_terminal).collect::<Vec<_>>();
                let refined=super::super::boundary_tail::build_root_call_candidates(parent,&children,&calls,&vocab).unwrap();
                let mut independent=super::super::boundary_candidates::persisted_boundary_candidate_ids(parent,&vocab)
                    .unwrap().unwrap();independent.retain(|id|refined.candidate_ids.binary_search(id).is_ok());
                assert_eq!(prepared.as_ref(),independent.as_slice());
            }
            let mut candidate=dynamic;install(&mut candidate,&vocab).unwrap();
            let reference=Grammar::from_ebnf(if nullable {r#"start ::= "x" ("a" | "") "y""#}
                else {r#"start ::= "x" "a" "y""#})
                .compile_with(&vocab,BuildOptions::default().optimization(Optimization::FastRuntime)).unwrap();
            let loaded=Constraint::load(&candidate.save()).unwrap();
            let mut words=vec![Vec::new()];
            for depth in 0..=4 {
                let mut next=Vec::new();
                for word in words {
                    for actual in [&candidate,&loaded] {
                        let mut a=actual.start();let mut b=reference.start();
                        for prefix in 0..=word.len() {
                            let mut actual_mask=vec![0;actual.mask_len()];let mut expected_mask=vec![0;reference.mask_len()];
                            a.fill_mask(&mut actual_mask);b.fill_mask(&mut expected_mask);
                            assert_eq!(actual_mask,expected_mask,"nullable={nullable} word={word:?} prefix={prefix}");
                            assert_eq!(a.is_accepting(),b.is_accepting());assert_eq!(a.is_rejected(),b.is_rejected());
                            if prefix<word.len() {assert_eq!(a.commit_bytes(&word[prefix..prefix+1]).is_ok(),
                                b.commit_bytes(&word[prefix..prefix+1]).is_ok());}
                        }
                    }
                    if depth<4 {for byte in [b'x',b'a',b'y',b'b'] {let mut following=word.clone();following.push(byte);next.push(following);}}
                }
                words=next;
            }
        }
    }

    #[test]
    fn prepared_leaf_possible_matches_equal_eager_union_for_every_cell_and_reload() {
        let vocab=Vocab::new(vec![(1,Vec::new()),(3,b"a".to_vec()),(7,b"ab".to_vec()),
            (11,b"a".to_vec()),(15,b"!".to_vec()),(19,b"?".to_vec()),(23,b"b".to_vec()),
            (27,b"aa!".to_vec()),(31,b"xa".to_vec()),(37,vec![0]),(41,b" ".to_vec()),(43,b"abab".to_vec())]);
        let left=Grammar::from_glrm(r#"glrm 1; start root; t WORD = /[a-z]+/; nt root = WORD "!";"#)
            .compile_with(&vocab,BuildOptions::default().optimization(Optimization::FastRuntime)).unwrap();
        let right=Grammar::from_glrm(r#"glrm 1; start root; t TEXT = /a[ab]*/; nt root = TEXT "?";"#)
            .compile_with(&vocab,BuildOptions::default().optimization(Optimization::FastRuntime)).unwrap();
        let loaded_left=Constraint::load(&left.save()).unwrap();let loaded_right=Constraint::load(&right.save()).unwrap();
        for leaves in [vec![&left,&right],vec![&loaded_left,&loaded_right],vec![&left,&loaded_right]] {
            let terminals=vec![0,leaves[0].tokenizer.num_terminals()];
            let inputs=leaves.iter().zip(&terminals).map(|(&leaf,&offset)|(leaf.tokenizer.as_ref(),offset)).collect::<Vec<_>>();
            let (merged,offsets)=Tokenizer::disjoint_union_with_terminal_offsets(&inputs);
            for (index,leaf) in leaves.iter().enumerate() {
                assert!(leaf.possible_matches_complete,"leaf={index} incomplete PM");
                assert!(!leaf.tokenizer.has_any_virtual_runtime(),"leaf={index} projected tokenizer");
                assert!(leaf.state_internal_tsid_offsets.is_empty() || leaf.state_internal_tsid_offsets.as_slice()==[u32::MAX],
                    "leaf={index} multivalued TSID");
                assert_eq!(leaf.state_to_internal_tsid.len(),leaf.tokenizer.num_states() as usize,"leaf={index} state coordinate");
            }
            let prepared=super::super::constraint_compose::prepared_leaf_possible_matches(&leaves,&offsets,
                &terminals,merged.num_states(),&vocab).unwrap().expect("complete intact leaf PM must be reusable");
            let expected=pm::compute_constraint_possible_matches_for_vocab(&merged,&vocab,pm::ConstraintPossibleMatchesConfig::EAGER);
            assert!(expected.complete);
            let contains=|mapped:&MappedArtifact<BTreeMap<u32,crate::ds::weight::Weight>>,state:u32,token:u32,terminal:u32| {
                let tsid=mapped.id_map().tokenizer_states.original_to_internal[state as usize];
                let itoken=mapped.id_map().vocab_tokens.original_to_internal[token as usize];
                tsid!=u32::MAX && itoken!=u32::MAX && mapped.artifact().get(&terminal)
                    .is_some_and(|weight|weight.tokens_for_tsid(tsid).contains(itoken))
            };
            for state in 0..merged.num_states() {for &token in vocab.entries_map().keys() {
                for terminal in 0..merged.num_terminals() {
                    assert_eq!(contains(&prepared,state,token,terminal),contains(&expected.mapped_possible_matches,state,token,terminal),
                        "state={state} token={token} terminal={terminal}");
                }
            }}
        }
        let mut incomplete=loaded_left.clone();incomplete.possible_matches_complete=false;
        let leaves=vec![&incomplete,&loaded_right];let terminals=vec![0,incomplete.tokenizer.num_terminals()];
        let (merged,offsets)=Tokenizer::disjoint_union_with_terminal_offsets(&[(incomplete.tokenizer.as_ref(),0),
            (loaded_right.tokenizer.as_ref(),terminals[1])]);
        assert!(super::super::constraint_compose::prepared_leaf_possible_matches(&leaves,&offsets,&terminals,
            merged.num_states(),&vocab).unwrap().is_none());
    }

    #[test]
    fn exact_caller_domains_distinguish_one_call_from_adjacent_child_calls() {
        let vocab = Vocab::new(vec![(0,b"x".to_vec()),(1,b"a".to_vec()),
            (2,b"aa".to_vec()),(3,b"xa".to_vec()),(4,b"xaa".to_vec()),(5,b"aaa".to_vec())]);
        let child = Grammar::from_ebnf(r#"start ::= "a""#).compile(&vocab).unwrap();
        for (source, inline, retain) in [
            (r#"start root; extern grammar child; nt root ::= "x" child;"#, r#"start ::= "x" "a""#, false),
            (r#"start root; extern grammar child; nt root ::= "x" child child;"#, r#"start ::= "x" "a" "a""#, true),
        ] {
            let options = BuildOptions::default().optimization(Optimization::FastRuntime);
            let unlinked = Grammar::from_glrm(source).compile_unlinked(&vocab).unwrap();
            let mut candidate = unlinked.bind("child", &child).unwrap().link().unwrap();
            assert_eq!(retain_same_owner_paths(&candidate, 1, true).unwrap(), retain);
            install(&mut candidate, &vocab).unwrap();
            let reference = Grammar::from_ebnf(inline).compile_with(&vocab, options).unwrap();
            for text in [b"x".as_slice(),b"xa",b"xaa",b"xaaa"] {
                let mut actual = candidate.start(); let mut expected = reference.start();
                for prefix in 0..=text.len() {
                    let mut a = vec![0; candidate.mask_len()]; let mut b = vec![0; reference.mask_len()];
                    actual.fill_mask(&mut a); expected.fill_mask(&mut b);
                    assert_eq!(a,b,"retain={retain} prefix={prefix} text={text:?}");
                    assert_eq!(actual.is_accepting(),expected.is_accepting());
                    assert_eq!(actual.is_rejected(),expected.is_rejected());
                    if prefix < text.len() {
                        assert_eq!(actual.commit_bytes(&text[prefix..prefix+1]).is_ok(),
                            expected.commit_bytes(&text[prefix..prefix+1]).is_ok());
                    }
                }
            }
        }
    }

    #[test]
    fn existing_static_component_bodies_are_retained_without_cloning_or_rebuilding() {
        let vocab = Vocab::new(vec![(0, b"x".to_vec()), (1, b"a".to_vec()), (2, b"y".to_vec()), (3, b"xay".to_vec())]);
        let child = Grammar::from_ebnf(r#"start ::= "a""#).compile(&vocab).unwrap();
        let mut c = Grammar::from_glrm(r#"glrm 1; start root; extern grammar child; nt root = "x" child "y";"#)
            .compile_unlinked(&vocab).unwrap().bind("child", &child).unwrap()
            .link_with(BuildOptions::default().optimization(Optimization::FastRuntime)).unwrap();
        c.install_template_parser().unwrap();
        let bodies = c.static_dynamic_overlay.as_ref().unwrap().segmented_parser_components.iter()
            .map(|component| Arc::clone(&component.constraint)).collect::<Vec<_>>();
        let bytes = c.save();
        assert!(!needs_static_boundaries(&c));
        install(&mut c, &vocab).unwrap();
        for (body, component) in bodies.iter().zip(&c.static_dynamic_overlay.as_ref().unwrap().segmented_parser_components) {
            assert!(Arc::ptr_eq(body, &component.constraint));
        }
        assert_eq!(bytes, c.save());
        let mut state = c.start(); state.commit_token(3).unwrap(); assert!(state.is_accepting());
    }
}
