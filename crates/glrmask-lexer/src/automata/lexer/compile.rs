//! Lexer compilation entry points. Implementation phases keep expression algebra,
//! finite-state construction, vocabulary proofs, and runtime publication separate.

use crate::Vocab;
use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::dfa::DFA;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::ds::u8set::U8Set;
use dfa_analysis::dfa_transition_count;
use component::compile_product_component_dfa_direct;
use nfa::build_regex_nfa;
pub(crate) use nfa::expr_u8set;
use partition::compile_terminal_partitions;
use plan::{
    ExclusionCompilePlan,
    build_exclusion_compile_plan,
    build_exclusion_compile_plan_with_labels,
};
use product::{ProductBuildTrace, build_product_dfa};
use settings::adaptive_lexer_enabled;
use std::sync::Arc;
use std::time::Instant;

pub use deferred::expression_contains_large_bounded_repeat;
#[cfg(any(test, feature = "internal-api"))]
pub use deferred::expression_supports_bounded_code_residual_runtime;
#[cfg(feature = "internal-api")]
pub use deferred::{
    expression_may_support_bounded_code_residual_runtime,
    expression_supports_deferred_dense_runtime,
};
#[cfg(feature = "internal-api")]
pub use expression_graph::compile_expression_labeled_nfa;
#[cfg(feature = "internal-api")]
pub use factor::{
    factor_regex_expr,
    factor_regex_expr_with_shared_cache,
};
pub use component::compile_terminal_expr_dfa;
#[cfg(feature = "internal-api")]
pub use mapping::{
    direct_bounded_suffix_state_count_estimate,
    max_direct_bounded_suffix_state_count_estimate,
};
#[cfg(feature = "internal-api")]
pub use mapping::{
    CertifiedVocabularyExactStateCandidates,
    CompiledTerminalExpressionPair,
    VocabularyExactStateCertifier,
    compile_terminal_expression_pair_with_structural_map,
    compile_terminal_expression_pair_with_vocabulary_token_quotient,
    install_vocabulary_exact_state_certifier,
    structural_pair_component_count,
};
#[cfg(feature = "internal-api")]
pub use partition::{
    CompiledPartitionedExpressionPair,
    DeferredPartitionedRegex,
    PreparedPartitionedExpressionPair,
};
#[cfg(feature = "internal-api")]
pub use partition_runtime::{
    build_exact_partitioned_runtime_tokenizer,
    build_partitioned_tokenizer_with_product_trace_terminal_residuals,
};
#[cfg(feature = "internal-api")]
pub use partition_runtime::build_partitioned_tokenizer_from_precompiled_terminal_dfas;
pub use repeat_horizon::VocabularyRepeatHorizonCache;
#[cfg(feature = "internal-api")]
pub use repeat_horizon::vocabulary_repeat_boundary_horizon;
#[cfg(feature = "internal-api")]
pub use synthesis::{
    ExtractedDispatchComponent,
    PrecompiledFurtherSynthesisPairs,
    compile_further_synthesized_tokenizer_with_structural_map,
    compile_partitioned_expression_pair_with_structural_map,
    extract_augmented_singleton_dispatch_components,
    extract_dispatch_components,
    precompile_further_synthesis_pairs,
    prepare_partitioned_expression_pair_with_structural_map,
    prepare_partitioned_expression_pair_with_vocabulary_token_quotient,
};
pub use virtual_repeat::{
    virtual_binary_bounded_repeat_intersection_descriptor,
    virtual_large_bounded_repeat_descriptor,
    virtual_unit_repeat_descriptor,
    virtual_zero_min_unit_repeat_fits_state_ids,
};
#[cfg(any(test, feature = "internal-api"))]
pub use virtual_repeat::{
    build_virtual_unit_repeat_tokenizer,
    build_virtual_zero_min_unit_repeat_tokenizer,
};
#[cfg(feature = "internal-api")]
pub use virtual_repeat::{
    large_top_level_bounded_repeat_bound,
    virtual_zero_min_unit_repeat_descriptor,
};
use serde::Serialize;
use serde::Deserialize;

mod bounded_repeat;
mod deferred;

mod dfa_analysis;
mod expression_graph;

mod factor;
mod component;

mod mapping;
mod nfa;

mod partition;
mod partition_runtime;

mod plan;
mod product;

mod repeat_horizon;
mod repeat_suffix;

mod settings;
mod synthesis;

mod virtual_repeat;

#[cfg(test)]
mod tests;

#[doc(hidden)]
pub fn build_bounded_code_mask_component_for_vocab(
    expr: &Expr,
    vocab: &Vocab,
    max_token_len: usize,
    repeat_horizons: &VocabularyRepeatHorizonCache,
) -> Option<(DFA, u32)> {
    super::runtime_residual::build_bounded_code_mask_component_for_vocab(
        expr,
        vocab,
        max_token_len,
        repeat_horizons,
    )
}

#[doc(hidden)]
pub struct PreparedBoundedCodeMaskComponent(
    super::runtime_residual::PreparedBoundedCodeMaskComponent,
);

impl PreparedBoundedCodeMaskComponent {
    pub fn finish_for_vocab(
        self,
        vocab: &Vocab,
        max_token_len: usize,
        repeat_horizons: &VocabularyRepeatHorizonCache,
    ) -> Option<(DFA, u32)> {
        self.0
            .finish_for_vocab(vocab, max_token_len, repeat_horizons)
    }

    pub fn finish_for_vocab_conservative(
        self,
        max_token_len: usize,
    ) -> Option<(DFA, u32)> {
        self.0.finish_for_vocab_conservative(max_token_len)
    }
}

#[doc(hidden)]
pub fn prepare_bounded_code_mask_component(
    expr: &Expr,
) -> Option<PreparedBoundedCodeMaskComponent> {
    super::runtime_residual::prepare_bounded_code_mask_component(expr)
        .map(PreparedBoundedCodeMaskComponent)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Regex {
    dfa: DFA,
}

impl Regex {
    pub fn into_tokenizer(self, num_terminals: u32, exprs: Option<std::sync::Arc<[Expr]>>) -> Tokenizer {
        Tokenizer {
            dfa: self.dfa,
            num_terminals,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs,
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: std::sync::OnceLock::new(),
            matched_terminals_cache: std::sync::OnceLock::new(),
            initial_byte_frontiers: std::sync::OnceLock::new(),
            all_self_loop_bytes_cache: std::sync::OnceLock::new(),
            transition_count_cache: std::sync::OnceLock::new(),
            forced_minimized_state_count_cache: std::sync::OnceLock::new(),
            scalar_deterministic_dispatch_cache: std::sync::OnceLock::new(),
            sorted_dispatch_roots_cache: std::sync::OnceLock::new(),
            state_first_bytes_cache: std::sync::OnceLock::new(),
        }
    }

    pub fn num_states(&self) -> usize {
        self.dfa.num_states()
    }

    pub fn num_transitions(&self) -> usize {
        dfa_transition_count(&self.dfa)
    }

    pub fn step(&self, state: u32, byte: u8) -> Option<u32> {
        self.dfa.step(state, byte)
    }

    pub fn get_u8set(&self, state: u32) -> U8Set {
        self.dfa.get_u8set(state)
    }
}

impl Expr {
    pub fn build(self) -> Regex {
        build_regex(&[self])
    }
}

/// Compile multiple expressions into a single multi-group [`Regex`].
///
/// Each expression's index becomes its group ID in the resulting DFA.
fn compile_single_expr_dfa(expr: &Expr) -> DFA {
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let direct_started_at = profile_timing.then(Instant::now);
    if let Some((mut dfa, needs_future_recompute)) = compile_product_component_dfa_direct(expr) {
        let direct_ms = direct_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        dfa.ensure_group_capacity(1);
        dfa.set_group_u8set(0, expr_u8set(expr));
        if needs_future_recompute {
            dfa.recompute_possible_futures();
        }
        if profile_timing {
            eprintln!(
                "[glrmask/profile][tokenizer] single_expr_path path=direct states={} transitions={} direct_ms={:.3}",
                dfa.num_states(),
                dfa_transition_count(&dfa),
                direct_ms,
            );
        }
        return dfa;
    }

    let direct_ms = direct_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let nfa_started_at = profile_timing.then(Instant::now);
    let mut nfa = build_regex_nfa(std::slice::from_ref(expr));
    let nfa_ms = nfa_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let condense_started_at = profile_timing.then(Instant::now);
    nfa.condense_epsilon_sccs();
    let condense_ms =
        condense_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let determinize_started_at = profile_timing.then(Instant::now);
    // Minimize over the NFA's byte-equivalence classes before expanding the
    // surviving transitions back to bytes.  Large JSON string/property regexes
    // can determinize to thousands of states with ~150k byte transitions only
    // to collapse to a few hundred states immediately afterward.  The class
    // alphabet path is language-equivalent and avoids materializing that large
    // transient byte graph.
    let dfa = nfa.to_minimized_dfa();
    if profile_timing {
        eprintln!(
            "[glrmask/profile][tokenizer] single_expr_path path=nfa_dfa states={} transitions={} direct_attempt_ms={:.3} nfa_ms={:.3} condense_ms={:.3} determinize_ms={:.3}",
            dfa.num_states(),
            dfa_transition_count(&dfa),
            direct_ms,
            nfa_ms,
            condense_ms,
            determinize_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
        );
    }
    dfa
}

fn compile_with_plan(plan: ExclusionCompilePlan) -> DFA {
    compile_with_plan_internal(plan, false).0
}

fn compile_with_plan_internal(
    plan: ExclusionCompilePlan,
    capture_product_trace: bool,
) -> (DFA, Option<ProductBuildTrace>) {
    compile_with_plan_internal_options(plan, capture_product_trace, true, false)
}

fn compile_with_plan_internal_options(
    plan: ExclusionCompilePlan,
    capture_product_trace: bool,
    virtual_fixed_sequences: bool,
    collapse_traced_duplicate_coordinates: bool,
) -> (DFA, Option<ProductBuildTrace>) {
    let profile_trace = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TRACE").is_some();
    let profile_detail = profile_trace
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some();
    let profile_timing = profile_detail
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    // Product construction compiles many one-group component DFAs internally.
    // Timing/detail mode should describe the enclosing lexer build, not emit a
    // line (and take timestamps) for every nested leaf compile. Exhaustive trace
    // mode remains available when that low-level view is explicitly requested.
    let profile_plan = profile_trace || (profile_timing && plan.compiled_exprs.len() > 1);
    let profile_started_at = Instant::now();
    let group_set_started_at = Instant::now();
    let group_sets: Vec<U8Set> = plan
        .compiled_exprs
        .iter()
        .map(expr_u8set)
        .collect();
    let group_set_ms = profile_plan
        .then(|| group_set_started_at.elapsed().as_secs_f64() * 1000.0);
    let used_product_dfa = plan.compiled_exprs.len() > 1;
    let dfa_build_started_at = Instant::now();
    let (mut dfa, product_group_ops_applied, mut product_trace) = if plan.compiled_exprs.is_empty() {
        // A grammar can lower to the empty language, for example when a const
        // literal conflicts with sibling assertions. Keep a single non-final
        // start state so tokenizer users can still query/step the DFA safely.
        (DFA::new(1), false, None)
    } else if used_product_dfa {
        build_product_dfa(
            &plan.compiled_exprs,
            plan.profile_labels.as_deref(),
            plan.visible_groups,
            &plan.exclusions,
            &plan.intersections,
            capture_product_trace,
            virtual_fixed_sequences,
            plan.local_small_product,
            collapse_traced_duplicate_coordinates,
        )
    } else {
        (compile_single_expr_dfa(&plan.compiled_exprs[0]), false, None)
    };
    let dfa_build_ms = profile_plan
        .then(|| dfa_build_started_at.elapsed().as_secs_f64() * 1000.0);

    let metadata_started_at = Instant::now();
    let output_group_count = if product_group_ops_applied {
        plan.visible_groups
    } else {
        group_sets.len()
    };
    dfa.ensure_group_capacity(output_group_count);
    for (group_id, set) in group_sets.into_iter().take(output_group_count).enumerate() {
        dfa.set_group_u8set(group_id as u32, set);
    }
    let metadata_ms = profile_plan
        .then(|| metadata_started_at.elapsed().as_secs_f64() * 1000.0);

    let group_ops_started_at = Instant::now();
    let mut group_ops_changed = false;
    if !product_group_ops_applied && !plan.exclusions.is_empty() {
        group_ops_changed |= dfa.apply_group_exclusions(&plan.exclusions);
    }
    if !product_group_ops_applied && !plan.intersections.is_empty() {
        group_ops_changed |= dfa.apply_group_intersections(&plan.intersections);
    }
    let project_in_place = !product_group_ops_applied
        && plan.visible_groups < plan.compiled_exprs.len()
        && std::env::var_os("GLRMASK_DISABLE_IN_PLACE_GROUP_PROJECTION").is_none();
    if group_ops_changed && !project_in_place {
        dfa.recompute_possible_futures();
    }
    let group_ops_ms = profile_plan
        .then(|| group_ops_started_at.elapsed().as_secs_f64() * 1000.0);

    let project_started_at = Instant::now();
    let dfa = if project_in_place {
        dfa.project_groups_in_place(plan.visible_groups);
        if group_ops_changed {
            // Reachability commutes with projection: after applying hidden
            // exclusion/intersection predicates to finalizers, recomputing the
            // visible futures directly is equivalent to recomputing all groups
            // and then projecting, while using narrower bitsets throughout.
            dfa.recompute_possible_futures();
        }
        dfa
    } else if !product_group_ops_applied && plan.visible_groups < plan.compiled_exprs.len() {
        dfa.project_groups(plan.visible_groups)
    } else {
        dfa
    };
    let project_ms = profile_plan
        .then(|| project_started_at.elapsed().as_secs_f64() * 1000.0);

    let pre_minimize_states = profile_plan.then(|| dfa.num_states());
    let pre_minimize_transitions = profile_plan.then(|| dfa_transition_count(&dfa));
    let force_tokenizer_minimize = std::env::var_os("GLRMASK_FORCE_TOKENIZER_MINIMIZE").is_some();
    let minimize_started_at = Instant::now();
    let final_dfa = if used_product_dfa && !force_tokenizer_minimize {
        dfa
    } else {
        product_trace = None;
        dfa.minimize()
    };
    let minimize_ms = profile_plan
        .then(|| minimize_started_at.elapsed().as_secs_f64() * 1000.0);
    if profile_detail && profile_plan {
        let minimized_states = if used_product_dfa && !force_tokenizer_minimize {
            "not_run".to_string()
        } else {
            final_dfa.num_states().to_string()
        };
        eprintln!(
            "[glrmask/profile][tokenizer] combined groups={} visible_groups={} product_dfa={} pre_minimize_states={} pre_minimize_transitions={} final_states={} final_transitions={} minimized_states={}",
            plan.compiled_exprs.len(),
            plan.visible_groups,
            used_product_dfa,
            pre_minimize_states.unwrap_or_default(),
            pre_minimize_transitions.unwrap_or_default(),
            final_dfa.num_states(),
            dfa_transition_count(&final_dfa),
            minimized_states,
        );
    }
    if profile_plan {
        eprintln!(
            "[glrmask/profile][tokenizer] compile_plan groups={} visible_groups={} product_dfa={} group_set_ms={:.3} dfa_build_ms={:.3} metadata_ms={:.3} group_ops_ms={:.3} project_ms={:.3} minimize_ms={:.3} total_ms={:.3}",
            plan.compiled_exprs.len(),
            plan.visible_groups,
            used_product_dfa,
            group_set_ms.unwrap_or_default(),
            dfa_build_ms.unwrap_or_default(),
            metadata_ms.unwrap_or_default(),
            group_ops_ms.unwrap_or_default(),
            project_ms.unwrap_or_default(),
            minimize_ms.unwrap_or_default(),
            profile_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    (final_dfa, product_trace)
}

pub fn build_regex(exprs: &[Expr]) -> Regex {
    build_regex_monolithic(exprs)
}

/// Compile a small auxiliary product lexer while keeping its fine-grained
/// construction work on the calling worker. This is intended for callers that
/// already execute inside a parallel compiler DAG, where nested Rayon fan-out
/// can otherwise turn a few milliseconds of local work into scheduler tail.
pub fn build_regex_local_small_product(exprs: &[Expr]) -> Regex {
    let mut plan = build_exclusion_compile_plan(exprs);
    plan.local_small_product = true;
    Regex {
        dfa: compile_with_plan(plan),
    }
}

/// Compile all expressions into one traditional deterministic lexer.
pub fn build_regex_monolithic(exprs: &[Expr]) -> Regex {
    Regex {
        dfa: compile_with_plan(build_exclusion_compile_plan(exprs)),
    }
}

pub fn build_regex_with_profile_labels(exprs: &[Expr], visible_labels: &[String]) -> Regex {
    Regex {
        dfa: compile_with_plan(build_exclusion_compile_plan_with_labels(
            exprs,
            Some(visible_labels),
        )),
    }
}

/// Compile terminals in caller-selected deterministic partitions, then join
/// those partitions with epsilon edges from one global start state. Terminals
/// sharing a partition may be jointly determinized; terminals in different
/// partitions can never cause a cross-partition subset/product blow-up.
pub fn build_regex_partitioned(exprs: &[Expr], partitions: &[u32]) -> Regex { build_regex_partitioned_with_options(exprs, partitions, PartitionOptions::default()) }



pub fn build_regex_partitioned_with_adaptive(
    exprs: &[Expr],
    partitions: &[u32],
    adaptive: bool,
) -> Regex { build_regex_partitioned_with_options(exprs, partitions, PartitionOptions { adaptive: Some(adaptive), ..PartitionOptions::default() }) }











/// Internal policy for a partitioned lexer build.
///
/// No adaptive override means the existing environment/default policy, not
/// `false`. Labels affect diagnostics; residual classes prevent combining
/// components with incompatible residual ownership.
#[derive(Clone, Copy, Default)]
pub struct PartitionOptions<'a> {
    pub profile_labels: Option<&'a [String]>,
    pub residual_isolation_classes: Option<&'a [Option<u32>]>,
    pub adaptive: Option<bool>,
}

pub fn build_regex_partitioned_with_options(
    exprs: &[Expr],
    partitions: &[u32],
    options: PartitionOptions<'_>,
) -> Regex {
    Regex {
        dfa: compile_terminal_partitions(
            exprs,
            options.profile_labels,
            partitions,
            options.residual_isolation_classes,
            options.adaptive.unwrap_or_else(adaptive_lexer_enabled),
        ),
    }
}

#[cfg(test)]
mod options_tests;
