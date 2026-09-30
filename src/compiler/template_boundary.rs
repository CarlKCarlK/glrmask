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
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
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

pub(crate) fn install(constraint: &mut Constraint, vocab: &Vocab) -> Result<()> {
    if !constraint.uses_compact_segmented_parser_runtime() { return Ok(()); }
    if !constraint.has_template_parser() { return Err(fail("static template boundary requires a table-free parser")); }
    // A nested reusable component keeps its literal structure. Upgrade only
    // its boundary implementation, leaving parser/lexer coordinates unchanged.
    if let Some(overlay) = constraint.static_dynamic_overlay.as_mut() {
        for component in &mut overlay.segmented_parser_components {
            if component.constraint.uses_compact_segmented_parser_runtime() {
                install(Arc::make_mut(&mut component.constraint), vocab)?;
            }
        }
    }
    let layout = constraint.recursive_parser_layout().map_err(fail)?
        .ok_or_else(|| fail("static template boundary has no scoped layout"))?;
    let certificate = super::boundary_transfer::certify_bounded_closure(&layout.links).map_err(fail)?;
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
    if leaves.iter().any(|leaf| leaf.tokenizer.has_any_virtual_runtime()) {
        return Err(fail("static template boundary needs a finite lexical observation projection for virtual lexers"));
    }
    let tokenizer_inputs = leaves.iter().zip(&layout.leaf_terminal_offsets)
        .map(|(leaf, &offset)| (leaf.tokenizer.as_ref(), offset)).collect::<Vec<_>>();
    let (mut merged, tokenizer_offsets) = Tokenizer::disjoint_union_with_terminal_offsets(&tokenizer_inputs);
    // The lexical union inserts one synthetic root before the intact leaves.
    // That root is useful during compilation but is never a live recursive
    // tokenizer state. Preserve the exact +1 injection rather than treating
    // the compiler union and runtime leaf coordinates as identical.
    if tokenizer_offsets.iter().zip(&layout.leaf_tokenizer_state_offsets)
        .any(|(&compiled, &runtime)| runtime.checked_add(1) != Some(compiled))
        || layout.total_tokenizer_states.checked_add(1) != Some(merged.num_states()) {
        return Err(fail("static lexical observation changed the recursive state coordinate"));
    }
    if merged.terminal_exprs().is_none() {
        if let Some(exprs) = merged_retained_terminal_exprs(&leaves, &layout.leaf_terminal_offsets,
            layout.total_leaf_terminals) {
            merged.restore_terminal_exprs(Some(exprs)).map_err(fail)?;
        }
    }
    let state_counts = leaves.iter().map(|leaf| leaf.tokenizer.num_states()).collect::<Vec<_>>();
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
                retain_non_crossing_paths: false });
        }
    }
    let context = crate::template_parser::static_compile::lexical_context(layout.total_leaf_terminals);
    let follows = BTreeMap::new();
    let inputs = BoundaryShardLinkInputs {
        merged_tokenizer: &merged, vocab, grammar: &context, disallowed_follows: &follows,
        ignore_terminal: None, follow_transparent_ignores: Some(&transparent),
        terminal_offsets: &layout.leaf_terminal_offsets, leaf_to_immediate: Some(&owners),
        tokenizer_offsets: &tokenizer_offsets, component_state_counts: &state_counts,
        candidate_tokens_by_component: None, retain_parent_non_crossing_paths: false,
        walk_plans: Some(plans),
    };
    let (walks, _) = super::boundary_walk::build_boundary_shard_walks(&inputs)
        .ok_or_else(|| fail("static template boundary lexical walk could not certify its scope"))?;
    // A remembers a delayed lexical decision using exact leaf terminal IDs.
    // Materialize its exclusions now, independently of the B quotient. The
    // shared static evaluator must never fall back to a vocabulary walk merely
    // because a remembered exclusion first appears after a token boundary.
    let possible = pm::compute_constraint_possible_matches_for_vocab(&merged, vocab,
        pm::ConstraintPossibleMatchesConfig::EAGER);
    if !possible.complete { return Err(fail("static template boundary has incomplete exclusions")); }
    let common = complete_possible_match_coordinate(possible.mapped_possible_matches.id_map(),
        merged.num_states(), vocab)?;
    let possible_matches = possible.mapped_possible_matches.remap_into_existing_common(&common)
        .into_artifact().into_iter().map(|(terminal, weight)| {
            let runtime_terminal = layout.outer_terminal_count.checked_add(terminal)
                .ok_or_else(|| fail("scoped exclusion terminal overflow"))?;
            Ok((runtime_terminal, weight))
        }).collect::<Result<BTreeMap<_, _>>>()?;
    let templates = crate::template_parser::static_compile::prepare_static_templates(
        &composition.programs, parser.state_count)?;
    let controls = (composition.control_start..composition.programs.len() as u32).collect::<Vec<_>>();
    let mut published = Vec::with_capacity(walks.len());
    for walk in walks {
        let output = super::boundary_transfer::template_program::compile(
            &templates, &controls, &certificate, &walk.output.dwa, parser.state_count).map_err(fail)?;
        let work = WalkBoundaryShardWork { start_component: walk.start_component as u32,
            terminal_automaton: TerminalAutomaton::Dwa(walk.output.dwa), id_map: walk.output.id_map,
            candidate_tokens: Arc::from(walk.candidate_tokens.into_iter().collect::<Vec<_>>()) };
        let (shard, _) = super::boundary_transfer::publish_signed_shard(work, output,
            &tokenizer_offsets, &state_counts).map_err(fail)?;
        published.push(shard);
    }
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
    overlay.recursive_tokenizer_internal_tsids.set(Arc::new(common.tokenizer_states.original_to_internal
        .iter().skip(1).map(|&tsid| vec![tsid]).collect()))
        .map_err(|_| fail("static template TSID coordinate initialized twice"))?;
    constraint.state_to_internal_tsid = common.tokenizer_states.original_to_internal
        [1..1 + constraint.tokenizer.num_states() as usize].to_vec();
    constraint.internal_tsid_to_states = common.tokenizer_states.internal_to_originals_vecs()
        .into_iter().map(|states| states.into_iter().filter_map(|state| state.checked_sub(1)).collect()).collect();
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
    Ok(())
}
