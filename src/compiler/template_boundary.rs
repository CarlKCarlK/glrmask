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
    let leaves = layout.leaves.iter().map(|leaf|
        constraint.constraint_at_recursive_component_path(&leaf.component_path)
            .ok_or_else(|| fail("static template leaf path is invalid")))
        .collect::<Result<Vec<_>>>()?;
    let projected = leaves.iter().any(|leaf| leaf.tokenizer.has_any_virtual_runtime());
    let views = leaves.iter().map(|leaf| {
        if leaf.tokenizer.has_any_virtual_runtime() {
            let view = leaf.dynamic_mask_vocab.mask_projection_tokenizer()
                .ok_or_else(|| fail("virtual component requires a persisted finite lexical observation projection"))?;
            if view.has_any_virtual_runtime() { return Err(fail("virtual lexical observation projection is not finite")); }
            Ok(view)
        } else { Ok(leaf.tokenizer.as_ref()) }
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
    let observation = projected.then(|| {
        let mut leaf_offsets = tokenizer_offsets.clone(); leaf_offsets.push(merged.num_states());
        crate::runtime::static_observation::RecursiveStaticObservation { leaf_offsets }
    });
    if merged.terminal_exprs().is_none() {
        if let Some(exprs) = merged_retained_terminal_exprs(&leaves, &layout.leaf_terminal_offsets,
            layout.total_leaf_terminals) {
            merged.restore_terminal_exprs(Some(exprs)).map_err(fail)?;
        }
    }
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
            plans.push(BoundaryShardWalkPlan { start_component: component,
                crossing_owner: ImmediateComponentId(component as u32), commit_states: starts,
                // A nullable call can enable a terminal word entirely within
                // its caller's lexical scope. A cannot recognize the unresolved
                // placeholder, so B must keep these same-owner paths as well.
                // The exact control-star program below still decides viability.
                retain_non_crossing_paths: certificate.is_none() });
        }
    }
    let context = crate::template_parser::static_compile::lexical_context(layout.total_leaf_terminals);
    let follow_started = std::time::Instant::now();
    let proof_rows = super::template_follow_support::disallowed(&composition.programs,
        layout.total_leaf_terminals as usize, 32_000_000).map_err(fail)?;
    let mut follows = BTreeMap::new();
    let mut excluded_pairs = 0usize;
    for (terminal, exclusions) in proof_rows.into_iter().enumerate() {
        if exclusions.is_empty() { continue; }
        excluded_pairs += exclusions.len();
        let mut row = BitSet::new(layout.total_leaf_terminals as usize);
        for excluded in exclusions { row.set(excluded as usize); }
        follows.insert(terminal as u32, row);
    }
    if profile { eprintln!("[glrmask/profile][static_template_boundary] phase=follow_support rows={} excluded_pairs={} elapsed_ms={:.3}",
        follows.len(), excluded_pairs, follow_started.elapsed().as_secs_f64() * 1000.0); }
    let inputs = BoundaryShardLinkInputs {
        merged_tokenizer: &merged, vocab, grammar: &context, disallowed_follows: &follows,
        ignore_terminal: None, follow_transparent_ignores: Some(&transparent),
        terminal_offsets: &layout.leaf_terminal_offsets, leaf_to_immediate: Some(&owners),
        tokenizer_offsets: &tokenizer_offsets, component_state_counts: &state_counts,
        candidate_tokens_by_component: None, retain_parent_non_crossing_paths: certificate.is_none(),
        walk_plans: Some(plans),
    };
    let (walks, _) = super::boundary_walk::build_boundary_shard_walks(&inputs)
        .ok_or_else(|| fail("static template boundary lexical walk could not certify its scope"))?;
    if profile { eprintln!("[glrmask/profile][static_template_boundary] phase=lexical_done shards={} elapsed_ms={:.3}",
        walks.len(), started.elapsed().as_secs_f64() * 1000.0); }
    // A remembers a delayed lexical decision using exact leaf terminal IDs.
    // Materialize its exclusions now, independently of the B quotient. The
    // shared static evaluator must never fall back to a vocabulary walk merely
    // because a remembered exclusion first appears after a token boundary.
    let possible = pm::compute_constraint_possible_matches_for_vocab(&merged, vocab,
        pm::ConstraintPossibleMatchesConfig::EAGER);
    if !possible.complete { return Err(fail("static template boundary has incomplete exclusions")); }
    let mut common = complete_possible_match_coordinate(possible.mapped_possible_matches.id_map(),
        merged.num_states(), vocab)?;
    if projected {
        // One observation per finite source state: neither A nor PM alone can
        // justify collapsing states which B may distinguish after a crossing.
        common.tokenizer_states = ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
            (0..merged.num_states()).collect(), merged.num_states());
    }
    let possible_matches = possible.mapped_possible_matches.remap_into_existing_common(&common)
        .into_artifact().into_iter().map(|(terminal, weight)| {
            let runtime_terminal = layout.outer_terminal_count.checked_add(terminal)
                .ok_or_else(|| fail("scoped exclusion terminal overflow"))?;
            Ok((runtime_terminal, weight))
        }).collect::<Result<BTreeMap<_, _>>>()?;
    let controls = (composition.control_start..composition.programs.len() as u32).collect::<Vec<_>>();
    let mut selected = controls.iter().copied().collect::<std::collections::BTreeSet<_>>();
    for walk in &walks {
        for row in walk.output.dwa.states() {
            for (terminal, _, weight) in row.transitions.entries() {
                if weight.is_empty() { continue; }
                let terminal = u32::try_from(terminal).map_err(|_| fail("negative lexical terminal"))?;
                if terminal >= composition.control_start {
                    return Err(fail("lexical boundary terminal lies outside the ordinary inventory"));
                }
                selected.insert(terminal);
            }
        }
    }
    if profile { eprintln!("[glrmask/profile][static_template_boundary] phase=programs_selected selected={} total={} elapsed_ms={:.3}",
        selected.len(), composition.programs.len(), started.elapsed().as_secs_f64() * 1000.0); }
    let (templates, classes) = crate::template_parser::static_compile::prepare_classed_boundary_programs(
        &composition.programs, parser.state_count, &selected)?;
    if profile { eprintln!("[glrmask/profile][static_template_boundary] phase=programs_ready templates={} pop_classes={} elapsed_ms={:.3}",
        templates.len(), classes.len(), started.elapsed().as_secs_f64() * 1000.0); }
    let mut published = Vec::with_capacity(walks.len());
    for walk in walks {
        let mut output = super::boundary_transfer::template_program::compile_classed(
            &templates, &controls, certificate.as_ref(), &walk.output.dwa, &classes).map_err(fail)?;
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
