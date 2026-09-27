//! Composition of already-compiled constraint artifacts.
//!
//! The expensive component parser DWAs are reused exactly. Their private
//! `(tokenizer-state class, vocabulary-token class)` coordinates are first
//! reconciled through the merged raw tokenizer and the shared original
//! vocabulary. Parser-state labels are transported through the table splice's
//! one-to-many relation. Default parser labels retain wildcard semantics during
//! the overlap-local union: explicit positive labels override them, unmatched
//! positive labels fall through to them, and negative labels never do.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, OnceLock};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;
use smallvec::SmallVec;

use range_set_blaze::RangeSetBlaze;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::automata::lexer::tokenizer::Lexer;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::compile::compile_terminal_expr_dfa;
use crate::automata::weighted_u32::determinize::determinize;
use crate::automata::weighted_u32::minimize::{minimize_owned, reverse_hashcons_owned};
use crate::automata::weighted_u32::equivalence::find_difference;
use crate::automata::weighted_u32::dwa::{DWA, DWAState};
use crate::automata::weighted_u32::nwa::{NWA, NWAState};
use crate::automata::weighted_u32::terminal_automaton::TerminalAutomaton;
use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::unweighted_u32::minimize_acyclic::minimize_acyclic as minimize_unweighted_dfa;
use crate::compiler::glr::analysis::{AnalyzedGrammar, EOF};
use crate::compiler::glr::labels::{
    DEFAULT_LABEL, encode_negative_label, encode_positive_label, is_negative_label,
    negative_to_positive_label,
};
use crate::compiler::stages::equiv_types::{
    InternalIdMap, ManyToOneIdMap, MappedArtifact,
};
use crate::compiler::stages::mapped_artifact::{WeightRefs, remap_weights_with_maps};
use crate::compiler::stages::id_map_and_terminal_dwa::classify::vocab_tokens_with_adjacent_pairs;
use crate::compiler::stages::id_map_and_terminal_dwa::grammar_helpers::compute_ever_allowed_follows;
use crate::compiler::stages::id_map_and_terminal_dwa::types::TerminalColoring;
use crate::compiler::stages::parser_dwa::{
    LazyBooleanParserDomains, SharedBooleanParserDomains, build_boolean_terminal_bundle_nwa,
    build_parser_nwa_from_terminal_dwa_with_precomputed_templates,
    build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count,
    build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count_no_table,
    build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count_no_table_with_bundle_cache,
    build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count_nondeterministic_bundles,
    build_parser_dwa_from_terminal_dwa_with_precomputed_templates,
    build_prebuilt_terminal_bundle_preimage_domain_dwa_direct_profiled,
    prebuild_parser_bundle_cache_excluding_terminals, PrebuiltParserBundleCache,
    build_terminal_bundle_preimage_domain_nwa,
    universal_parser_stack_domain_dwa,
    normalize_parser_stack_domain_nwa_preserving_explicit,
    normalize_weighted_parser_stack_nwa,
    normalize_weighted_parser_stack_nwa_small_boundary,
    normalize_weighted_parser_stack_nwa_small_boundary_compact_for_parser_state_count,
    normalize_weighted_parser_stack_nwa_small_boundary_for_parser_state_count,
    normalize_weighted_parser_stack_nwa_small_boundary_with_tsid_map, SmallBoundaryDwa,
    normalize_signed_weighted_parser_stack_nwa_small_boundary,
    normalize_signed_weighted_parser_stack_nwa_small_boundary_for_parser_state_count,
    resolve_negative_codes_small_boundary,
};
use glrmask_parser_dwa::__private::resolve_negatives::resolve_negative_codes_in_nwa;
use crate::compiler::stages::templates::characterize::{
    characterize_finish_probe, characterize_selected_terminals,
    characterize_selected_terminals_for_terminal_count,
    characterize_terminal_action_state_seeds_for_terminal_count,
    characterize_terminal_nt_predecessor_seeds_for_terminal_count,
};
use crate::compiler::stages::templates::compile_dfa::{
    specialize_template_dfa_defaults_for_commit_split_input,
    try_split_commit_template_dfas,
};
use crate::compiler::stages::templates::Templates;
use crate::compiler::constraint_possible_matches::{
    build_internal_token_bytes_from_groups, runtime_dynamic_vocab_for_vocab,
};
use crate::compiler::glr::table::{
    Action, ComposedTable, ControlEliminationReport, SubgrammarTableInput,
    compose_subgrammar_tables_explicit_with_rules,
};
use crate::grammar::flat::Symbol;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::ds::weight::{ScopedWeightOpCache, SharedTokenSet, Weight};
use crate::runtime::{
    CompositionGrammarSummary, Constraint, ConstraintRuntimeBackend, SpecialTokenTerminal,
};
use crate::Vocab;
use crate::compiler::composition::boundary::walk::{
    WalkStaticLinkInputs, build_walk_static_boundary_link, dynamic_fallback_walk_link_output,
};
use crate::compiler::{macro_join, macro_parallelism_disabled, report_macro_item_timings};

mod structural_sharing;
mod runtime_lexer_product;
use runtime_lexer_product::maybe_install_runtime_lexer_product;
use structural_sharing::StructuralSharingReport;

#[inline]
pub(crate) fn compose_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
}

/// The generic parser-DWA builder currently skips weighted minimization unless
/// explicitly overridden. Boundary overlays are different: leaving their large
/// intermediate DWA unminimized makes the subsequent component union explode.
fn parser_builder_skips_internal_minimization() -> bool {
    std::env::var("GLRMASK_SKIP_PARSER_DWA_MINIMIZE")
        .ok()
        .map(|value| {
            let trimmed = value.trim();
            !(trimmed.is_empty()
                || trimmed == "0"
                || trimmed.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true)
}

fn boundary_parser_minimize_min_states() -> u32 {
    std::env::var("GLRMASK_BOUNDARY_PARSER_MINIMIZE_MIN_STATES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(64)
}

pub(crate) fn eliminate_composed_runtime_controls(
    composed: &mut ComposedTable,
) -> Result<Option<ControlEliminationReport>, String> {
    if composed.control_terminals.is_empty() {
        debug_assert!(composed.table.control_terminals.is_empty());
        return Ok(None);
    }
    debug_assert_eq!(composed.control_terminals, composed.table.control_terminals);
    let report = composed.table.eliminate_control_terminals_exact()?;
    composed.control_terminals.clear();
    debug_assert!(composed.table.control_terminals.is_empty());
    Ok(Some(report))
}

mod coordinates;
pub(crate) use coordinates::{
    CompiledSubgrammarInput, ConstraintComposition, ParserDwaComponent,
    build_segmented_parser_links, install_published_static_boundary_shards,
};
use coordinates::{
    DirectComponentCoordinateMaps, ParserDefaultDomain,
    append_dynamic_direct_boundary_shards_for_unselected, build_direct_component_coordinate_maps,
    build_direct_component_state_coordinates,
    build_direct_component_token_coordinates, build_parser_default_domain_plan,
    build_recursive_tokenizer_internal_tsid_relation,
    build_segmented_runtime_metadata, component_parser_nwa,
    install_dynamic_direct_boundary_shards, install_segmented_boundary_shards,
    strip_unscoped_ignore_identity, tokenizer_tsid_relation_is_singleton,
};

mod publication;
pub(crate) use publication::{
    PublishedStaticBoundaryShard, WalkBoundaryShardWork, WalkShardPublishProfile,
    publish_signed_boundary_shard_work,
};
use publication::{
    BoundaryParserWork, BoundaryRepair, BoundaryRuntimeCandidate, BoundaryShardWork,
    PositiveBoundaryParser, PossibleMatches, PublishedBoundaryRuntime,
    build_commit_templates_from_raw_templates, defer_boundary_commit_templates,
    ensure_positive_runtime_parser_dwa, load_boundary_terminal_capture,
    publish_boundary_parser_candidate_for_state_count, publish_real_boundary_parser_work,
    publish_static_boundary_shard_work,
};

mod discovery;
pub(crate) use discovery::{build_exact_component_boundary_trigger, merged_retained_terminal_exprs};
use discovery::{
    BoundaryTokenDiscovery, BoundaryTokenNodeKey, ExprByteSummary, PreTableBoundaryBaseDiscovery,
    add_boundary_special_token_paths, add_control_loops_to_terminal_artifact,
    boundary_id_map_for_selected_tokens, boundary_interface_adjacent_pair_candidates,
    boundary_tokens_by_start_component, build_static_boundary_shard_work,
    collect_one_byte_seed_relations, collect_one_byte_seed_relations_components,
    component_state_coordinate_map,
    component_tokenizer_state_layout_owned_parent,
    compose_nonnullable_grammar_adjacency_summaries, direct_boundary_terminal_automaton,
    disallowed_follows_from_allowed_rows, discover_boundary_token_paths, expr_byte_summary,
    extend_boundary_interfaces_through_stack_neutral_lr_actions,
    replace_boundary_discovery_tokens, visible_boundary_interface_pairs,
};

mod parser_union;
use parser_union::{
    PreparedOwnedComponentArtifacts,
    boundary_discovery_good_signature, build_boundary_refinement_plan,
    explicit_parser_nwa, prebuild_segmented_token_mask_caches,
    prepare_deferred_component_artifacts,
    prepare_unmapped_component_possible_matches, reverse_hashcons_positive_acyclic_nwa,
    reverse_hashcons_positive_acyclic_nwa_fast_with_reverse_topo,
    reverse_hashcons_signed_acyclic_nwa_fast, segmented_boundary_state_to_private_tsid,
};

mod templates;
pub(crate) use templates::{transport_composition_template_dfa_with_skeleton};
use templates::{
    ConcreteBoundaryDeltaPlan, boundary_delta_reset_relations,
    build_complete_composed_parser_template_cache, build_composition_templates,
    changed_parent_template_candidate_terminals, factor_one_terminal_seed_relations,
    finish_eager_changed_parent_templates, finish_eager_concrete_boundary_delta_plan,
    install_concrete_boundary_delta_templates, merge_one_terminal_relations,
    prepare_concrete_boundary_delta_plan, profile_current_boundary_template_delta,
    try_build_cached_composition_templates,
    try_build_cached_composition_templates_for_terminal_count,
    try_build_changed_parent_templates_for_terminal_count,
    try_rebuild_cached_transported_component_templates,
};

mod boundary_repair;

mod assembly;
pub(crate) use assembly::{
    MergedIgnoreTerminals, clear_selected_nested_segmented_boundary_shards,
    component_ignores_are_globally_erasable, leaf_ignores_are_globally_erasable,
    merged_ignore_terminals, merged_leaf_ignore_terminals, merged_leaf_terminal_display_names,
    merged_terminal_display_names,
};
use assembly::{
    build_composed_constraint_unfinalized, canonicalize_possible_matches,
    canonicalize_terminal_live_states, components_have_no_compiled_eof_stack_rewrites,
    finalize_owned_composed_constraint_runtime,
    finalize_owned_composed_constraint_runtime_single_thread,
    legacy_splice_has_only_byte_terminal_continuations,
    merged_original_token_ids, merged_special_token_terminals,
    merged_special_token_terminals_recursive_fast,
    merged_terminal_live_states_owned_parent, prepared_constraint_for_segmented_composition,
    recursive_component_tokenizer_span,
};

mod link;
#[allow(deprecated)]
pub(crate) use link::{
    SegmentedBoundaryBackend, accepted_original_tokens, accepted_weight_support,
    compose_constraints_owned_parent, compose_constraints_owned_parent_segmented,
    compose_constraints_owned_parent_segmented_shared,
};

#[cfg(test)]
mod tests;

pub(crate) mod boundary;

#[cfg(test)]
use coordinates::{NO_PARSER_DOMAIN_LABEL, add_component_parser_state_transitions, mapped_labels};
#[cfg(test)]
pub(crate) use publication::{publish_walk_boundary_shard_work};
#[cfg(test)]
use parser_union::{full_nwa_topological_order, reverse_hashcons_positive_acyclic_nwa_fast};
#[cfg(test)]
use templates::{
    rebuild_transported_component_templates, transport_composition_template_dfa,
    trim_unweighted_dfa_productive, try_prepare_cached_composition_template_plan,
    unweighted_dfa_difference, unweighted_dfa_language_is_empty, unweighted_dfa_shortest_word,
};

#[cfg(test)]
#[allow(deprecated)]
pub(crate) use link::{
    compose_constraints, compose_constraints_owned_parent_segmented_hybrid,
    compose_constraints_owned_parent_shared, load_vocab,
};
