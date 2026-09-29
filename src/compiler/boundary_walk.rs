//! Production boundary-shard terminal-DWA construction (static-link redesign).
//!
//! For a composition with a merged tokenizer and a provider-materialized
//! exact boundary table (live leaf coordinate), the boundary shard for start
//! component `i` is built by running the STANDARD terminal-DWA trie walk on
//! the merged tokenizer from `Commit_i` (component `i`'s token-start states),
//! keeping only crossing paths at the NWA level, then standard templates +
//! parser-DWA construction (see `build_boundary_shard` in step 4). Exact by
//! construction: the walk is what a monolithic compile would run on the same
//! merged DFA, restricted to `i` starts; non-crossing paths are already
//! covered at runtime by `A_i`.
//!
//! Plan note: `prepared-static-linker-architecture-2026-09-17.md` §2 (rev 4).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

use smallvec::SmallVec;

use crate::automata::lexer::tokenizer::{Lexer, Tokenizer};
use crate::automata::weighted_u32::dwa::DWA;
use crate::automata::weighted_u32::minimize_acyclic::minimize_acyclic_owned;
use crate::automata::weighted_u32::terminal_automaton::TerminalAutomaton;
use crate::compiler::constraint_compose::{
    CompiledSubgrammarInput, PublishedStaticBoundaryShard, WalkBoundaryShardWork,
    accepted_original_tokens, build_segmented_parser_links, compose_profile_enabled,
    component_ignores_are_globally_erasable, eliminate_composed_runtime_controls,
    leaf_ignores_are_globally_erasable, merged_ignore_terminals,
    merged_leaf_ignore_terminals, merged_leaf_terminal_display_names,
    merged_retained_terminal_exprs, merged_terminal_display_names,
};
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::parser::{
    DisjointComponentActionProvider, ParserComponentTableSource, ScopedParserSymbol,
    ScopedSubgrammarLink, materialize_control_eliminated_scoped_provider_table,
};
use crate::compiler::glr::table::{
    ComposedTable, GLRTable, SubgrammarTableInput, compose_subgrammar_tables_with_rules,
};
use crate::compiler::pipeline::compute_disallowed_follows;
use crate::compiler::stages::equiv_types::InternalIdMap;
use crate::compiler::stages::id_map_and_terminal_dwa as tdwa;
use crate::ds::bitset::BitSet;
use crate::grammar::flat::{Symbol, TerminalID};
use crate::runtime::Constraint;
use crate::Vocab;

/// Test-only: enumerate bounded accepted terminal words of a boundary terminal
/// DWA, keeping only paths whose edge-weight intersection with the final weight
/// is non-empty. Enabled by `GLRMASK_DEBUG_BOUNDARY_WORDS`.
#[cfg(test)]
fn trace_boundary_terminal_dwa_words(label: &str, dwa: &DWA, num_terminals: usize) {
    if std::env::var_os("GLRMASK_DEBUG_BOUNDARY_WORDS").is_none() {
        return;
    }
    const MAX_LEN: usize = 8;
    const MAX_WORDS: usize = 256;
    let mut accepted = Vec::<Vec<u32>>::new();
    let mut stack = vec![(
        dwa.start_state(),
        crate::ds::weight::Weight::all(),
        Vec::<u32>::new(),
    )];
    while let Some((state_id, path_weight, word)) = stack.pop() {
        let Some(state) = dwa.states().get(state_id as usize) else {
            continue;
        };
        if let Some(final_weight) = state.final_weight.as_ref()
            && !path_weight.intersection(final_weight).is_empty()
        {
            accepted.push(word.clone());
            if accepted.len() >= MAX_WORDS {
                break;
            }
        }
        if word.len() >= MAX_LEN {
            continue;
        }
        for (transition_label, target, edge_weight) in state.transitions.entries() {
            if transition_label < 0 {
                continue;
            }
            let next_weight = path_weight.intersection(edge_weight);
            if next_weight.is_empty() {
                continue;
            }
            let mut next_word = word.clone();
            next_word.push(transition_label as u32);
            stack.push((target, next_weight, next_word));
        }
    }
    accepted.sort();
    accepted.dedup();
    eprintln!(
        "[boundary-words {label}] num_terminals={num_terminals} accepted_terminal_words(len<={MAX_LEN})={accepted:?}"
    );
}

/// Inputs for one boundary-shard terminal-DWA build.
pub(crate) struct BoundaryWalkInputs<'a> {
    /// Merged (disjoint-union) tokenizer of the composition being linked.
    pub merged_tokenizer: &'a Tokenizer,
    /// Original-ID model-token candidate subset for this immediate component.
    /// Unknown summaries are widened by the caller to the full vocabulary.
    pub vocab: &'a Vocab,
    /// Analyzed composed grammar (for always-allowed follows + the walk's
    /// terminal observation scope).
    pub grammar: &'a AnalyzedGrammar,
    /// Pairwise disallowed terminal follows of the composed table.
    pub disallowed_follows: &'a BTreeMap<u32, BitSet>,
    /// Canonical ignore terminal of the merged tokenizer, if any.
    pub ignore_terminal: Option<u32>,
    /// Composed-ids of scoped-ignore terminals that must be transparent to the
    /// within-token follow-pair pruning. Their labels are retained; scope is
    /// enforced by the parser-shard scoped identity transfers. The canonical
    /// global ignore is tracked separately through `ignore_terminal`.
    pub follow_transparent_ignores: Option<&'a BitSet>,
    /// Checked initial/reset/ownership scope for this boundary shard.
    pub scope: &'a tdwa::scope::BoundaryAnalysisScope,
    /// Retain non-crossing paths (bypass the NWA-level crossing filter).
    /// The link sets this only for the parent shard when a child can be
    /// traversed zero-width (nullable start): the composed parser then
    /// accepts parent-internal paths the parent alone rejects, and only the
    /// unfiltered walk covers them. Child shards always filter (their
    /// component mask covers every dropped path); see
    /// `BoundaryShardLinkInputs::retain_parent_non_crossing_paths`.
    pub retain_non_crossing_paths: bool,
    /// Prebuilt flat transition table for the merged tokenizer, shared across
    /// shards. Built on demand when `None`.
    pub flat_trans: Option<&'a Arc<[u32]>>,
}

/// Per-stage timings for one shard build (milliseconds wall).
/// `walk_ms` is the INCLUSIVE wall of the whole L2P terminal-DWA build call;
/// `terminal_dwa_ms` is an as-reported sub-accumulation that itself INCLUDES
/// `determinize_ms` + `minimize_ms` (see the `TerminalDwaPhaseProfile`
/// construction: `terminal_dwa_ms = vocab_tree + … + determinize +
/// minimize + …`). Never sum all seven fields: that double-counts
/// determinize/minimize. The exclusive leaves are setup + walk_wall-excess
/// (unattributed inside the call) + id_map + compact + (terminal_dwa minus
/// determinize minus minimize) + determinize + minimize.
#[derive(Debug, Clone, Default)]
pub(crate) struct BoundaryWalkProfile {
    pub input_tokens: usize,
    pub candidate_tokens: usize,
    pub initial_states: usize,
    pub continuation_reset_states: usize,
    pub tokenizer_classes: usize,
    pub token_classes: usize,
    pub lexical_accepted_tokens: usize,
    /// Set only by an exact accepted-token enumeration against this query's
    /// complete vocabulary. False means unknown, not that a token is absent.
    pub all_input_tokens_accepted: bool,
    pub setup_ms: f64,
    pub walk_ms: f64,
    pub id_map_ms: f64,
    pub terminal_dwa_ms: f64,
    pub compact_ms: f64,
    pub determinize_ms: f64,
    pub minimize_ms: f64,
}

/// Output of one boundary-shard terminal-DWA build.
pub(crate) struct BoundaryWalkOutput {
    /// Minimized terminal DWA: crossing-only by default (empty language iff
    /// `X_i` is empty), full walk output when non-crossing paths are retained.
    pub dwa: DWA,
    /// Per-shard ID map retaining the shared link-wide equivalence coordinates.
    pub id_map: InternalIdMap,
    pub profile: BoundaryWalkProfile,
}

/// Token-start states of one component: all merged-tokenizer raw states owned
/// by component `i` (`tokenizer_offsets[i] .. + num_states_of_i`).
///
/// This is a sound superset of the runtime commit states (every token the
/// runtime starts in `i` starts in one of these states), hence sound for
/// crossing detection; it is pessimistic for cost (narrowing to the true
/// commit set is future work).
pub(crate) fn commit_states_for_component(
    tokenizer_offsets: &[u32],
    num_states_of_component: u32,
    index: usize,
    total_states: usize,
) -> Vec<bool> {
    let mut keep = vec![false; total_states];
    let start = tokenizer_offsets[index] as usize;
    let end = start + num_states_of_component as usize;
    keep[start..end].fill(true);
    keep
}

/// Build the crossing terminal DWA for one start component.
///
/// Runs the standard L2P walk (`build_l2p_id_map_and_terminal_dwa_mode`) on
/// the merged tokenizer with `seed_state_filter = Commit_i`, the link-shared
/// equivalence (TI discovery skipped), the NWA-level crossing filter for
/// `component_index` (bypassed when `retain_non_crossing_paths` is set), and
/// core compaction skipped (the shard parser indexes (parser-stack x tsid),
/// which needs the exact Step-1 coordinate).
/// Returns `None` only when the vocab is empty.
pub(crate) fn build_boundary_terminal_dwa(
    inputs: &BoundaryWalkInputs,
) -> Option<BoundaryWalkOutput> {
    build_boundary_terminal_dwa_with_shared(inputs, None)
}

fn build_boundary_terminal_dwa_with_shared(
    inputs: &BoundaryWalkInputs,
    shared: Option<&tdwa::ScopedBoundarySharedContext<'_>>,
) -> Option<BoundaryWalkOutput> {
    build_boundary_terminal_dwa_with_prepared(inputs, shared, None)
}

fn build_boundary_terminal_dwa_with_prepared(
    inputs: &BoundaryWalkInputs,
    shared: Option<&tdwa::ScopedBoundarySharedContext<'_>>,
    prepared: Option<&super::boundary_precomputed_completion::PreparedSourceSpan>,
) -> Option<BoundaryWalkOutput> {
    let tile_size = crate::compiler::boundary_env::bounded_usize(
        "GLRMASK_BOUNDARY_QUERY_TILE_SIZE", 32, 1, 4096,
    );
    let terminal_support=crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_QUERY_TERMINAL_SUPPORT");
    let query_view=crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_QUERY_VIEW");
    let identity = crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_IDENTITY_REFINEMENT");
    let first_started = Instant::now();
    let use_first = crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_FIRST_COMPLETION_REFINEMENT");
    let precomputed = (use_first
        && crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_PRECOMPUTED_COMPLETION"))
        .then(|| prepared.and_then(|span| span.refine(inputs.merged_tokenizer, inputs.vocab, inputs.scope)))
        .flatten();
    let precomputed_selected = precomputed.is_some();
    let first = precomputed.or_else(|| use_first.then(|| inputs.flat_trans.and_then(|flat|
        super::boundary_first_completion::prepare(inputs.merged_tokenizer, inputs.vocab, inputs.scope, flat))).flatten());
    if compose_profile_enabled() && prepared.is_some() {
        eprintln!("[glrmask/profile][boundary_precomputed_completion] component={} selected={precomputed_selected}", inputs.scope.start_component().0);
    }
    let narrowed = first.as_ref().and_then(|plan|
        inputs.scope.intersect_initial_support(&plan.representatives));
    let scoped = BoundaryWalkInputs {
        merged_tokenizer:inputs.merged_tokenizer,vocab:inputs.vocab,
        grammar:inputs.grammar,disallowed_follows:inputs.disallowed_follows,
        ignore_terminal:inputs.ignore_terminal,
        follow_transparent_ignores:inputs.follow_transparent_ignores,
        scope:narrowed.as_ref().unwrap_or(inputs.scope),
        flat_trans:inputs.flat_trans,retain_non_crossing_paths:inputs.retain_non_crossing_paths,
    };
    let first_ms=first_started.elapsed().as_secs_f64()*1000.0;
    let allow_identity_tiles = std::env::var_os("GLRMASK_BOUNDARY_IDENTITY_TILES").is_some();
    let synthesized = (identity && scoped.scope.require_crossing()
        && std::env::var_os("GLRMASK_BOUNDARY_FINITE_CONTEXT_LEXER").is_some())
        .then(|| build_boundary_terminal_dwa_synthesized(&scoped,
            std::env::var_os("GLRMASK_BOUNDARY_FINITE_CONTEXT_FULL").is_none()))
        .flatten();
    let mut output=synthesized.or_else(|| build_boundary_terminal_dwa_query(&scoped,shared,
        if identity && !allow_identity_tiles { None } else { tile_size },
        terminal_support && !identity,query_view,identity))?;
    if let (Some(plan),Some(_))=(&first,&narrowed) {
        let old=&output.id_map.tokenizer_states;
        let mut lifted=old.original_to_internal.clone();
        for(q,&keep)in inputs.scope.initial_states().keep_raw().iter().enumerate(){
            if keep{lifted[q]=old.original_to_internal[plan.original_to_rep[q]as usize];}
        }
        output.id_map.tokenizer_states=crate::compiler::stages::equiv_types::ManyToOneIdMap::from_original_to_internal_allowing_unmapped(lifted,old.num_internal_ids());
        output.profile.initial_states=plan.before;
        output.profile.setup_ms+=first_ms;
        if compose_profile_enabled(){eprintln!("[glrmask/profile][boundary_first_completion] component={} before={} after={} prefixes={} steps={} cells={} ms={first_ms:.3}",inputs.scope.start_component().0,plan.before,plan.after,plan.prefixes,plan.steps,plan.signature_cells);}
    }
    if (first.is_some() || identity || tile_size.is_some() || terminal_support)
        && std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_QUERY_TILES").is_some()
    {
        // This reference bypasses tiling, terminal support and compact views.
        let reference=build_boundary_terminal_dwa_on_view(inputs,shared)?;
        tdwa::l2p::compare_terminal_dwa_artifacts(
            &tdwa::types::LocalIdMapTerminalDwa{dwa:reference.dwa,id_map:reference.id_map,profile:Default::default()},
            &tdwa::types::LocalIdMapTerminalDwa{dwa:output.dwa.clone(),id_map:output.id_map.clone(),profile:Default::default()},
        ).expect("query tiles changed original-coordinate weighted terminal language");
        eprintln!("[glrmask/validate][boundary_query_tiles] component={} exact=true",inputs.scope.start_component().0);
    }
    Some(output)
}

fn build_boundary_terminal_dwa_synthesized(
    inputs: &BoundaryWalkInputs,
    completion_only: bool,
) -> Option<BoundaryWalkOutput> {
    if !inputs.scope.require_crossing() { return None; }
    let started = Instant::now();
    let prepared = super::boundary_finite_lexer::prepare(
        inputs.merged_tokenizer, inputs.vocab, inputs.scope.initial_states().keep_raw(),
        inputs.scope.ownership(), inputs.scope.start_component(), completion_only,
    )?;
    let scope = tdwa::scope::BoundaryAnalysisScope::new(
        tdwa::scope::InitialStateDomain::from_mask(prepared.seeds.len(), prepared.seeds.clone()).ok()?,
        prepared.tokenizer.deterministic_reset_states().into_vec(),
        Arc::new(inputs.scope.ownership().clone()), inputs.scope.start_component(), true,
        inputs.scope.follow_transparent().cloned(),
    ).ok()?;
    let preparation_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut output = build_boundary_terminal_dwa_on_view_policy(&BoundaryWalkInputs {
        merged_tokenizer: &prepared.tokenizer, vocab: inputs.vocab, grammar: inputs.grammar,
        disallowed_follows: inputs.disallowed_follows, ignore_terminal: inputs.ignore_terminal,
        follow_transparent_ignores: inputs.follow_transparent_ignores, scope: &scope,
        retain_non_crossing_paths: false, flat_trans: None,
    }, None, None, true)?;
    // Only token-entry states are lifted. RESET/factor coordinates are private
    // continuation states and must not be confused with raw runtime states.
    let current = &output.id_map.tokenizer_states;
    let mut original = vec![u32::MAX; prepared.original_to_synth.len()];
    for (q, &keep) in inputs.scope.initial_states().keep_raw().iter().enumerate() {
        if !keep { continue; }
        let synthetic = *prepared.original_to_synth.get(q)?;
        if synthetic == u32::MAX { return None; }
        original[q] = *current.original_to_internal.get(synthetic as usize)?;
    }
    output.id_map.tokenizer_states = crate::compiler::stages::equiv_types::ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
        original, current.num_internal_ids(),
    );
    // The published shard is addressed by every raw runtime lexer ID, not
    // just the certified token-entry set. As in the ordinary induced view,
    // give omitted IDs one fresh zero-language class. A symbolic full weight
    // or support reaching that new coordinate cannot certify this operation.
    let dead_class = output.id_map.tokenizer_states.num_internal_ids();
    if !proves_fresh_dead_class(&output.dwa, dead_class,
        crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_ROOT_SUPPORT_BOUND")) {
        return None;
    }
    output.id_map.tokenizer_states = output.id_map.tokenizer_states.fill_unmapped_with_new_class();
    output.profile.tokenizer_classes = output.id_map.tokenizer_states.num_internal_ids() as usize;
    output.profile.initial_states = inputs.scope.initial_states().len();
    output.profile.setup_ms += preparation_ms;
    output.profile.walk_ms = started.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!("[glrmask/profile][boundary_finite_context] component={} selected=true completion_only={completion_only} total_ms={:.3} preparation_ms={preparation_ms:.3} profile={:?} tdwa_states={} tdwa_edges={}",
            inputs.scope.start_component().0, output.profile.walk_ms, prepared.profile,
            output.dwa.num_states(), output.dwa.num_transitions());
    }
    Some(output)
}

#[cfg(test)]
mod finite_context_gate {
    use super::*;
    use crate::automata::lexer::ast::{bytes, choice, plus};
    use crate::automata::lexer::compile::{build_regex_monolithic, build_regex_partitioned};
    use crate::grammar::flat::{Rule, Symbol};

    #[test]
    fn finite_context_matches_independent_original_terminal_languages() {
        let local_expr = vec![choice(vec![bytes(b"abcdef"), bytes(b"xydef"), bytes(b"ab")]),
            plus(choice(vec![bytes(b"a"), bytes(b"bc")])), plus(bytes(b" "))];
        let foreign_expr = vec![bytes(b"!def"), choice(vec![bytes(b"b"), bytes(b"c"), bytes(b"a!")])];
        let foreign = build_regex_partitioned(&foreign_expr, &[0,1]).into_tokenizer(2, None);
        let mut random = 3347u64;
        let mut next = || { random = random.wrapping_mul(6364136223846793005).wrapping_add(1); (random >> 32) as usize };
        let mut comparisons = 0;
        for partitioned in [false, true] {
            let local = if partitioned { build_regex_partitioned(&local_expr, &[0,1,2]) }
                else { build_regex_monolithic(&local_expr) }.into_tokenizer(3, None);
            let (mut tokenizer, _) = Tokenizer::disjoint_union_with_terminal_offsets(&[(&local,0),(&foreign,3)]);
            let mut expressions = local_expr.clone(); expressions.extend(foreign_expr.clone());
            tokenizer.restore_terminal_exprs_without_virtual_runtime(Some(expressions)).unwrap();
            let rules = vec![Rule { lhs:0, rhs:vec![Symbol::Nonterminal(1)] }].into_iter()
                .chain((0..5).flat_map(|t| [Rule { lhs:1, rhs:vec![Symbol::Terminal(t)] },
                    Rule { lhs:1, rhs:vec![Symbol::Terminal(t),Symbol::Nonterminal(1)] }])).collect();
            let grammar = AnalyzedGrammar::from_composed_rules(rules,5,
                (0..5).map(|t|format!("t{t}")).collect(),vec!["start".into(),"body".into()],0);
            let owners = Arc::new(tdwa::scope::BoundaryOwnership::flat(&[0,3],5).unwrap());
            let flat = Arc::from(tdwa::l1::build_flat_transition_table(&tokenizer));
            for case in 0..32 {
                let mut words = vec![b"def!".to_vec(),b"ab!".to_vec(),b"abc".to_vec(),b"a b".to_vec(),b"!defa".to_vec()];
                for _ in 0..10 { let len=1+next()%6; words.push((0..len).map(|_|b"abcdef! "[next()%8]).collect()); }
                let vocab = Vocab::new(words.iter().enumerate().map(|(i,w)|((i*7)as u32,w.clone())).collect());
                let mut selected = (0..tokenizer.num_states()).map(|_| next()%3 != 0).collect::<Vec<_>>();
                selected[0] = true; // Exercise a mixed-owner epsilon dispatcher too.
                let mut follows = BTreeMap::new();
                if case%3 != 0 { let mut row=BitSet::new(5); row.set(3); follows.insert(1,row); }
                let ignore = (case%3 == 0).then_some(2);
                let scope = tdwa::scope::BoundaryAnalysisScope::new(
                    tdwa::scope::InitialStateDomain::from_mask(selected.len(),selected).unwrap(),
                    tokenizer.deterministic_reset_states().into_vec(), Arc::clone(&owners),
                    tdwa::scope::ImmediateComponentId(0), true, None,
                ).unwrap();
                let inputs = BoundaryWalkInputs { merged_tokenizer:&tokenizer, vocab:&vocab,
                    grammar:&grammar, disallowed_follows:&follows, ignore_terminal:ignore,
                    follow_transparent_ignores:None, scope:&scope, retain_non_crossing_paths:false,
                    flat_trans:Some(&flat) };
                // The reference directly invokes the ordinary family path:
                // no FIRST classes, context synthesis, packing, or query view.
                let reference = build_boundary_terminal_dwa_on_view(&inputs,None).unwrap();
                for completion_only in [false,true] {
                    let candidate = build_boundary_terminal_dwa_synthesized(&inputs,completion_only).unwrap();
                    assert!(candidate.id_map.tokenizer_states.original_to_internal.iter().all(|&id| id != u32::MAX),
                        "published synthesized view must map even omitted raw states to a proven dead class");
                    tdwa::l2p::compare_terminal_dwa_artifacts(
                        &tdwa::types::LocalIdMapTerminalDwa { dwa:reference.dwa.clone(), id_map:reference.id_map.clone(), profile:Default::default() },
                        &tdwa::types::LocalIdMapTerminalDwa { dwa:candidate.dwa, id_map:candidate.id_map, profile:Default::default() },
                    ).unwrap_or_else(|error|panic!("case={case} partitioned={partitioned} completion={completion_only}: {error:?}"));
                    comparisons += 1;
                }
            }
        }
        assert_eq!(comparisons,128);
    }
}

fn build_boundary_terminal_dwa_query(
    inputs: &BoundaryWalkInputs,
    shared: Option<&tdwa::ScopedBoundarySharedContext<'_>>,
    tile_size:Option<usize>, terminal_support:bool, query_view:bool, identity:bool,
) -> Option<BoundaryWalkOutput> {
    if identity && query_view && crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_BORROWED_QUERY"){
        if let Some(candidate)=build_boundary_terminal_dwa_borrowed_query(inputs){
            if std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_BORROWED_QUERY").is_some(){
                let reference=build_boundary_terminal_dwa_query_reference(inputs,shared,tile_size,terminal_support,query_view,identity)?;
                assert_eq!(candidate.dwa.start_state(),reference.dwa.start_state());assert_eq!(candidate.dwa.states(),reference.dwa.states());
                for (a,b)in [(&candidate.id_map.tokenizer_states,&reference.id_map.tokenizer_states),(&candidate.id_map.vocab_tokens,&reference.id_map.vocab_tokens)]{
                    assert_eq!(a.original_to_internal,b.original_to_internal);assert_eq!(a.internal_to_originals,b.internal_to_originals);assert_eq!(a.representative_original_ids,b.representative_original_ids);
                }
                eprintln!("[glrmask/validate][borrowed_query] exact_graph=true exact_maps=true tokens={}",inputs.vocab.len());
            }
            return Some(candidate);
        }
    }
    build_boundary_terminal_dwa_query_reference(inputs,shared,tile_size,terminal_support,query_view,identity)
}

fn build_boundary_terminal_dwa_query_reference(
    inputs: &BoundaryWalkInputs,
    shared: Option<&tdwa::ScopedBoundarySharedContext<'_>>,
    tile_size:Option<usize>, terminal_support:bool, query_view:bool, identity:bool,
) -> Option<BoundaryWalkOutput> {
    if query_view
        && let Some(prepared)=super::boundary_query_view::prepare_with_policy(inputs.merged_tokenizer,inputs.vocab,inputs.scope,inputs.flat_trans.map(|flat|flat.as_ref()))
    {
        let view=&prepared.view;
        let mut output=build_boundary_terminal_dwa_query_jobs(&BoundaryWalkInputs{
            merged_tokenizer:&view.tokenizer,vocab:inputs.vocab,grammar:inputs.grammar,
            disallowed_follows:inputs.disallowed_follows,ignore_terminal:inputs.ignore_terminal,
            follow_transparent_ignores:inputs.follow_transparent_ignores,scope:&prepared.scope,
            retain_non_crossing_paths:inputs.retain_non_crossing_paths,flat_trans:None,
        },None,tile_size,terminal_support,identity)?;
        let current=&output.id_map.tokenizer_states;
        let mut originals=vec![u32::MAX;view.original_to_view.len()];
        let mut groups=vec![Vec::new();current.num_internal_ids() as usize];
        for (raw,&keep) in inputs.scope.initial_states().keep_raw().iter().enumerate(){if keep{
            let compact=view.original_to_view[raw];assert_ne!(compact,u32::MAX);
            let class=current.original_to_internal[compact as usize];originals[raw]=class;
            if class!=u32::MAX{groups[class as usize].push(raw as u32);}
        }}
        let representatives=groups.iter().map(|g|g.first().copied().unwrap_or(u32::MAX)).collect();
        output.id_map.tokenizer_states=crate::compiler::stages::equiv_types::ManyToOneIdMap{
            original_to_internal:originals,internal_to_originals:groups,representative_original_ids:representatives,
        };
        // Runtime publication requires a total raw-state map even though this
        // shard can be queried only from its certified initial domain. Add one
        // fresh zero-language class, never reuse a productive continuation ID.
        // Prove that no accepted lexical point mentions the new ID before
        // assigning all omitted states to it; symbolic ALL is not a finite row.
        let dead_class=output.id_map.tokenizer_states.num_internal_ids();
        if !proves_fresh_dead_class(&output.dwa,dead_class,
            crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_ROOT_SUPPORT_BOUND")){
            return build_boundary_terminal_dwa_on_view(inputs,shared);
        }
        output.id_map.tokenizer_states=output.id_map.tokenizer_states.fill_unmapped_with_new_class();
        output.profile.setup_ms+=prepared.footprint_ms+prepared.materialize_ms;
        output.profile.initial_states=inputs.scope.initial_states().len();
        if compose_profile_enabled(){eprintln!("[glrmask/profile][boundary_query_view] component={} raw_states={} view_states={} first_states={} reset_states={} steps={} footprint_ms={:.3} materialize_ms={:.3} compile_ms={:.3}",
            inputs.scope.start_component().0,inputs.merged_tokenizer.num_states(),view.tokenizer.num_states(),
            prepared.first_states,prepared.reset_states,prepared.state_steps,prepared.footprint_ms,
            prepared.materialize_ms,output.profile.walk_ms);}
        if std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_QUERY_VIEW").is_some(){
            let reference=build_boundary_terminal_dwa_on_view(inputs,shared)?;
            tdwa::l2p::compare_terminal_dwa_artifacts(
                &tdwa::types::LocalIdMapTerminalDwa{dwa:reference.dwa,id_map:reference.id_map,profile:Default::default()},
                &tdwa::types::LocalIdMapTerminalDwa{dwa:output.dwa.clone(),id_map:output.id_map.clone(),profile:Default::default()},
            ).expect("query observation view changed original-coordinate terminal language");
            eprintln!("[glrmask/validate][boundary_query_view] component={} exact=true",inputs.scope.start_component().0);
        }
        return Some(output);
    }
    build_boundary_terminal_dwa_query_jobs(inputs,shared,tile_size,terminal_support,identity)
}

/// A disjoint partition of original model tokens is a union of independent
/// weighted-language queries. All local maps are reconciled in ORIGINAL
/// coordinates by the existing compiler merger, never by concatenating IDs.
fn build_boundary_terminal_dwa_query_jobs(
    inputs:&BoundaryWalkInputs,shared:Option<&tdwa::ScopedBoundarySharedContext<'_>>,
    tile_size:Option<usize>,terminal_support:bool,identity:bool,
)->Option<BoundaryWalkOutput>{
    use crate::compiler::stages::mapped_artifact::MappedArtifact;
    use rayon::prelude::*;
    let started=Instant::now();
    let compile_one=|inputs:&BoundaryWalkInputs,shared:Option<&tdwa::ScopedBoundarySharedContext<'_>>| {
        let began=Instant::now();
        let support=terminal_support.then(||super::boundary_query_terminals::query_support(
            inputs.merged_tokenizer,inputs.vocab,inputs.scope.initial_states().keep_raw(),
        )).flatten().map(|(mut keep,work)|{
            if let Some(t)=inputs.ignore_terminal{if let Some(k)=keep.get_mut(t as usize){*k=true;}}
            if let Some(ts)=inputs.scope.follow_transparent(){for t in ts.iter(){if let Some(k)=keep.get_mut(t){*k=true;}}}
            if compose_profile_enabled(){eprintln!("[glrmask/profile][boundary_query_terminals] component={} tokens={} terminals={} kept={} work={} ms={:.3}",
                inputs.scope.start_component().0,inputs.vocab.len(),keep.len(),keep.iter().filter(|&&x|x).count(),work,began.elapsed().as_secs_f64()*1000.0);}
            keep
        });
        let elapsed=began.elapsed().as_secs_f64()*1000.0;
        let mut result=build_boundary_terminal_dwa_on_view_policy(inputs,shared,support.as_deref(),identity)?;
        result.profile.setup_ms+=elapsed;Some(result)
    };
    let Some(size)=tile_size.filter(|_|inputs.scope.require_crossing()
        && inputs.vocab.len()<=4096 && !inputs.vocab.entries_map().values().any(Vec::is_empty))
    else{return compile_one(inputs,shared);};
    let mut routed=BTreeMap::<u8,Vec<(u32,Vec<u8>)>>::new();
    for (&id,bytes) in inputs.vocab.entries_map(){
        routed.entry(tdwa::classify::classify_vocab_char_type(bytes)).or_default().push((id,bytes.clone()));
    }
    let mut groups=Vec::new();
    for mut entries in routed.into_values(){
        entries.sort_unstable_by(|a,b|a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
        for chunk in entries.chunks(size){groups.push(Vocab::new(chunk.to_vec()));}
    }
    if groups.len()<2 || groups.len()>256{return compile_one(inputs,shared);}
    let flat:Arc<[u32]>=inputs.flat_trans.cloned().unwrap_or_else(||Arc::from(tdwa::l1::build_flat_transition_table(inputs.merged_tokenizer)));
    let outputs=groups.par_iter().map(|vocab|{
        let began=Instant::now();
        let scoped=tdwa::scope::crossing_prefix_seed_support(
            inputs.merged_tokenizer,vocab,&flat,inputs.scope.ownership(),inputs.scope.start_component(),
        ).and_then(|(support,_)|inputs.scope.intersect_initial_support(&support));
        // A declined or empty prefilter is not treated as a proof of an empty
        // language. The mature compiler remains the conservative fallback.
        let local=BoundaryWalkInputs{
            merged_tokenizer:inputs.merged_tokenizer,vocab,grammar:inputs.grammar,
            disallowed_follows:inputs.disallowed_follows,ignore_terminal:inputs.ignore_terminal,
            follow_transparent_ignores:inputs.follow_transparent_ignores,
            scope:scoped.as_ref().unwrap_or(inputs.scope),
            retain_non_crossing_paths:inputs.retain_non_crossing_paths,flat_trans:Some(&flat),
        };
        let output=build_boundary_terminal_dwa_query(&local,shared,None,terminal_support,true,identity)?;
        if compose_profile_enabled(){eprintln!("[glrmask/profile][boundary_query_tile] component={} tokens={} initial={} ms={:.3}",
            inputs.scope.start_component().0,vocab.len(),local.scope.initial_states().len(),began.elapsed().as_secs_f64()*1000.0);}
        Some(output)
    }).collect::<Option<Vec<_>>>()?;
    let mut profile=BoundaryWalkProfile{input_tokens:inputs.vocab.len(),candidate_tokens:inputs.vocab.len(),
        initial_states:inputs.scope.initial_states().len(),continuation_reset_states:inputs.scope.reset_states().len(),..Default::default()};
    for output in &outputs {
        profile.id_map_ms+=output.profile.id_map_ms;profile.terminal_dwa_ms+=output.profile.terminal_dwa_ms;
        profile.compact_ms+=output.profile.compact_ms;profile.determinize_ms+=output.profile.determinize_ms;
        profile.minimize_ms+=output.profile.minimize_ms;
    }
    let merge_started=Instant::now();
    let mapped=tdwa::merge::merge_mapped_dwas(outputs.into_iter().map(|o|MappedArtifact::new(o.dwa,o.id_map)).collect(),
        inputs.merged_tokenizer.num_states() as usize,*inputs.vocab.entries_map().keys().max()?);
    let (dwa,id_map)=mapped.into_parts();let dwa=minimize_acyclic_owned(dwa);
    let merge_ms=merge_started.elapsed().as_secs_f64()*1000.0;
    profile.walk_ms=started.elapsed().as_secs_f64()*1000.0;
    profile.tokenizer_classes=id_map.num_tsids() as usize;profile.token_classes=id_map.num_internal_tokens() as usize;
    profile.lexical_accepted_tokens=terminal_accepted_original_tokens(&dwa,&id_map).len();
    if compose_profile_enabled(){eprintln!("[glrmask/profile][boundary_query_tiles] component={} groups={} tokens={} size={} merge_ms={merge_ms:.3} total_ms={:.3} states={} transitions={}",
        inputs.scope.start_component().0,groups.len(),inputs.vocab.len(),size,profile.walk_ms,dwa.num_states(),dwa.num_transitions());}
    Some(BoundaryWalkOutput{dwa,id_map,profile})
}

fn build_boundary_terminal_dwa_on_view(
    inputs:&BoundaryWalkInputs,shared:Option<&tdwa::ScopedBoundarySharedContext<'_>>,
)->Option<BoundaryWalkOutput>{
    build_boundary_terminal_dwa_on_view_filtered(inputs,shared,None)
}

fn build_boundary_terminal_dwa_on_view_filtered(
    inputs:&BoundaryWalkInputs,shared:Option<&tdwa::ScopedBoundarySharedContext<'_>>,
terminal_filter:Option<&[bool]>,
)->Option<BoundaryWalkOutput>{
    build_boundary_terminal_dwa_on_view_policy(inputs,shared,terminal_filter,false)
}

fn build_boundary_terminal_dwa_on_view_policy(
    inputs:&BoundaryWalkInputs,shared:Option<&tdwa::ScopedBoundarySharedContext<'_>>,
    terminal_filter:Option<&[bool]>,identity:bool,
)->Option<BoundaryWalkOutput>{
    let setup_started = Instant::now();
    let tokenizer = inputs.merged_tokenizer;
    let num_terms = inputs.grammar.num_terminals as usize;
    let coloring = tdwa::types::TerminalColoring::identity(num_terms);
    let owned_flat;
    let flat: &Arc<[u32]> = match inputs.flat_trans {
        Some(flat) => flat,
        None => {
            owned_flat = Arc::from(tdwa::l1::build_flat_transition_table(tokenizer));
            &owned_flat
        }
    };
    let setup_ms = setup_started.elapsed().as_secs_f64() * 1000.0;
    let walk_started = Instant::now();
    let refined = identity.then(|| tdwa::build_scoped_boundary_identity_with_certificate(
        tokenizer, inputs.vocab, inputs.ignore_terminal, inputs.grammar,
        inputs.disallowed_follows, Arc::clone(flat), inputs.scope,
    )).flatten();
    let (mapped, tdwa_profile, native_fixed_point) = refined.unwrap_or_else(|| {
        let (mapped, profile)=tdwa::build_scoped_boundary_id_map_and_terminal_dwa_with_filter(
        tokenizer,
        inputs.vocab,
        &coloring,
        inputs.ignore_terminal,
        inputs.grammar,
        inputs.disallowed_follows,
        Arc::clone(flat),
        inputs.scope,
        shared,
        terminal_filter,
        );
        (mapped,profile,None)
    });
    finalize_scoped_terminal_output(inputs,mapped,tdwa_profile,native_fixed_point,setup_ms,walk_started)
}

fn finalize_scoped_terminal_output(
 inputs:&BoundaryWalkInputs,mapped:crate::compiler::stages::mapped_artifact::MappedArtifact<TerminalAutomaton>,tdwa_profile:tdwa::types::TerminalDwaPhaseProfile,
 native_fixed_point:Option<tdwa::NativeMinimizationFixedPoint>,setup_ms:f64,walk_started:Instant,
)->Option<BoundaryWalkOutput>{
    let (automaton, id_map) = mapped.into_parts();
    let mut dwa = match automaton {
        TerminalAutomaton::Dwa(dwa) => dwa,
        TerminalAutomaton::TokenDeterministicNwa(_) | TerminalAutomaton::EpsilonNwa(_) => {
            unreachable!("scoped boundary family builder publishes one DWA")
        }
    };
    // The scoped ordinary-family pipeline can retain structurally live
    // terminal paths whose token/TSID weights have no complete original-token
    // witness.  Such an automaton denotes the empty boundary language even
    // though it still has transitions.  Canonicalize it here so callers do
    // not carry dead terminal structure into template/parser construction and
    // so the empty-shard representation remains stable across family choices.
    let certified = native_fixed_point.is_some()
        && crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_REUSE_NATIVE_MINIMUM");
    let summarize_after=crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_SUMMARIZE_AFTER_MIN");
    let reference=(certified && std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_NATIVE_MINIMUM").is_some())
        .then(||finalize_boundary_terminal_dwa(dwa.clone(),&id_map,summarize_after));
    let (finalized, accepted_tokens, final_minimize_ms) = if certified {
        let accepted=if let Some(support)=native_fixed_point.as_ref().and_then(|proof|proof.accepted_weight()){
            if std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_CERTIFIED_ROOT_SUPPORT").is_some(){
                assert_eq!(support,&super::constraint_compose::accepted_weight_support(&dwa),
                    "certified root support changed correlated output weights");
            }
            super::boundary_terminal_summary::project_accepted_tokens(support,&id_map)
        }else{terminal_accepted_original_tokens(&dwa,&id_map)};
        let graph=if accepted.is_empty(){DWA::new(id_map.num_tsids(),id_map.max_internal_token_id())}else{dwa};
        (graph,accepted,0.0)
    }else{finalize_boundary_terminal_dwa(dwa,&id_map,summarize_after)};
    if let Some((reference,accepted,_))=reference {
        assert_eq!(accepted_tokens,accepted,"native minimum changed original accepted token IDs");
        assert_eq!(finalized.start_state(),reference.start_state());
        assert_eq!(finalized.states(),reference.states(),"native fixed-point certificate did not preserve exact graph");
        eprintln!("[glrmask/validate][reuse_native_minimum] exact_graph=true tokens={}",inputs.vocab.len());
    }
    dwa = finalized;
    let lexical_accepted_tokens = accepted_tokens.len();
    let all_input_tokens_accepted = accepted_tokens.len() == inputs.vocab.len()
        && inputs.vocab.entries_map().keys().all(|token| accepted_tokens.contains(token));
    let walk_ms = walk_started.elapsed().as_secs_f64() * 1000.0;
    let profile = BoundaryWalkProfile {
        input_tokens: inputs.vocab.len(),
        candidate_tokens: inputs.vocab.len(),
        initial_states: inputs.scope.initial_states().len(),
        continuation_reset_states: inputs.scope.reset_states().len(),
        tokenizer_classes: id_map.num_tsids() as usize,
        token_classes: id_map.num_internal_tokens() as usize,
        lexical_accepted_tokens,
        all_input_tokens_accepted,
        setup_ms,
        walk_ms,
        id_map_ms: tdwa_profile.id_map_ms,
        terminal_dwa_ms: tdwa_profile.terminal_dwa_ms + final_minimize_ms,
        compact_ms: tdwa_profile.compact_ms,
        determinize_ms: tdwa_profile.determinize_ms,
        minimize_ms: tdwa_profile.minimize_ms + final_minimize_ms,
    };
    Some(BoundaryWalkOutput {
        dwa,
        id_map,
        profile,
    })
}

/// Existential token support is invariant under exact weighted-language
/// minimization. Computing it afterwards avoids scanning an unminimized
/// thousands-state graph solely to rediscover the same accepted token set.
/// The old ordering remains independently callable by differential tests.
fn finalize_boundary_terminal_dwa(
    mut dwa: DWA,
    id_map: &InternalIdMap,
    summarize_after: bool,
) -> (DWA, BTreeSet<u32>, f64) {
    let mut minimize_ms = 0.0;
    if summarize_after && dwa.num_states() > 1 && dwa.is_acyclic() {
        let started = Instant::now();
        dwa = minimize_acyclic_owned(dwa);
        minimize_ms += started.elapsed().as_secs_f64() * 1000.0;
    }
    let accepted = terminal_accepted_original_tokens(&dwa, id_map);
    if accepted.is_empty() {
        dwa = DWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
    } else if !summarize_after && dwa.num_states() > 1 && dwa.is_acyclic() {
        let started = Instant::now();
        dwa = minimize_acyclic_owned(dwa);
        minimize_ms += started.elapsed().as_secs_f64() * 1000.0;
    }
    (dwa, accepted, minimize_ms)
}

/// Sufficient absence proof, not a substitute for exact accepted-language
/// enumeration: every accepted path either stops at the root or traverses a
/// root edge. Its coefficient is a subset of that final/edge coefficient.
/// Unknown or overly broad root weights use the original complete proof.
fn proves_fresh_dead_class(dwa: &DWA, dead_class: u32, use_root_bound: bool) -> bool {
    let below = |weight: &crate::ds::weight::Weight| {
        !weight.is_full() && weight.range_entries().all(|(_, hi, _)| hi < dead_class)
    };
    if use_root_bound {
        if let Some(root) = dwa.states().get(dwa.start_state() as usize) {
            if root.final_weight.as_ref().is_none_or(&below)
                && root.transitions.values().all(|(_, weight)| below(weight)) {
                return true;
            }
        }
    }
    let support = crate::compiler::constraint_compose::accepted_weight_support(dwa);
    below(&support)
}

#[cfg(test)]
mod support_finalize_gate {
    use super::*;
    use crate::automata::weighted_u32::{dwa::DWAState, equivalence::find_difference};
    use crate::compiler::stages::equiv_types::ManyToOneIdMap;
    use crate::ds::weight::Weight;
    use range_set_blaze::RangeSetBlaze;

    fn identity(n: u32) -> ManyToOneIdMap {
        ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
            (0..n).collect(), (0..n).collect())
    }
    fn weight(row: u32, lo: u32, hi: u32) -> Weight {
        Weight::from_uniform(row..=row, RangeSetBlaze::from_iter([lo..=hi]))
    }

    #[test]
    fn summary_order_and_root_absence_match_independent_complete_scan() {
        let map = InternalIdMap { tokenizer_states:identity(4), vocab_tokens:identity(16),
            deferred_vocab_singleton_original_ids:None };
        let weights = [Weight::empty(), Weight::all(), weight(0,0,3), weight(1,2,5),
            weight(2,7,9), weight(0,0,3).union(&weight(3,8,12))];
        let mut random = 581344u64;
        let mut next = || { random=random.wrapping_mul(6364136223846793005).wrapping_add(1); (random>>32)as usize };
        let mut empty_cases = 0;
        for case in 0..256 {
            let n=4+next()%8;
            let mut states=vec![DWAState::default();n];
            for i in 0..n {
                if case%11 != 0 && next()%3 == 0 {
                    states[i].final_weight=Some(weights[next()%weights.len()].clone());
                }
                for label in 0..4 {
                    if i+1<n && next()%3 != 0 {
                        let target=i+1+next()%(n-i-1);
                        states[i].transitions.insert(label,(target as u32,weights[next()%weights.len()].clone()));
                    }
                }
            }
            let original = DWA::from_parts(states,0);
            let expected = accepted_original_tokens(&original,&map);
            empty_cases += usize::from(expected.is_empty());
            let (old,old_tokens,_) = finalize_boundary_terminal_dwa(original.clone(),&map,false);
            let (new,new_tokens,_) = finalize_boundary_terminal_dwa(original.clone(),&map,true);
            assert_eq!(old_tokens,expected,"old case {case}");
            assert_eq!(new_tokens,expected,"new case {case}");
            assert_eq!(find_difference(&old,&new).unwrap(),None,"case {case}");
            if expected.is_empty() {
                assert_eq!(new.num_states(),1);
                assert_eq!(new.num_transitions(),0);
            }
            for bound in [0,1,2,4,100] {
                assert_eq!(proves_fresh_dead_class(&original,bound,true),
                    proves_fresh_dead_class(&original,bound,false),"case {case} bound {bound}");
            }
        }
        assert!(empty_cases>=20,"empty-language branch must actually execute");
    }

    #[test]
    fn root_absence_keeps_exact_fallback_and_ignores_unreachable_weights() {
        let mut states=vec![DWAState::default();4];
        states[0].transitions.insert(1,(1,weight(0,0,2)));
        states[1].final_weight=Some(Weight::all());
        states[2].final_weight=Some(weight(99,0,15)); // Unreachable is irrelevant.
        let source=DWA::from_parts(states.clone(),0);
        assert!(proves_fresh_dead_class(&source,1,true));
        assert!(!proves_fresh_dead_class(&source,0,true));
        states[0].transitions.insert(1,(1,Weight::all()));
        states[1].final_weight=Some(weight(0,0,2));
        // Root ALL cannot prove absence, but the full continuation can.
        assert!(proves_fresh_dead_class(&DWA::from_parts(states.clone(),0),1,true));
        states[1].final_weight=Some(weight(1,0,2));
        assert!(!proves_fresh_dead_class(&DWA::from_parts(states,0),1,true));
    }
}

/// Original model tokens accepted by a shard terminal DWA (candidate-token
/// trigger + gate helper). The shard DWAs are acyclic (asserted).
pub(crate) fn boundary_accepted_tokens(dwa: &DWA, id_map: &InternalIdMap) -> BTreeSet<u32> {
    terminal_accepted_original_tokens(dwa, id_map)
}

// Terminal-only selection: parser support calculations retain their original
// implementation. The candidate reads this graph without changing it.
fn terminal_accepted_original_tokens(dwa: &DWA, id_map: &InternalIdMap) -> BTreeSet<u32> {
    if crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_BATCHED_TOKEN_SUPPORT")
        && let Some(tokens) = super::boundary_terminal_summary::accepted_tokens(dwa, id_map)
    {
        return tokens;
    }
    accepted_original_tokens(dwa, id_map)
}

/// Distinct terminal labels emitted by a terminal DWA (template selection).
pub(crate) fn boundary_emitted_terminals(dwa: &DWA, num_terminals: usize) -> Vec<bool> {
    let mut selected = vec![false; num_terminals];
    for state in dwa.states() {
        for &label in state.transitions.keys() {
            if label >= 0
                && let Some(slot) = selected.get_mut(label as usize)
            {
                *slot = true;
            }
        }
    }
    selected
}

/// Terminal offsets of a composed table (re-exported shape for shard builds).
pub(crate) fn composed_terminal_offsets(composed_table: &ComposedTable) -> &[u32] {
    composed_table.terminal_offsets.as_slice()
}

/// Link-level inputs for building every boundary shard.
pub(crate) struct BoundaryShardLinkInputs<'a> {
    pub merged_tokenizer: &'a Tokenizer,
    pub vocab: &'a Vocab,
    pub grammar: &'a AnalyzedGrammar,
    pub disallowed_follows: &'a BTreeMap<u32, BitSet>,
    pub ignore_terminal: Option<u32>,
    /// Composed-ids of scoped-ignore terminals that must be transparent to the
    /// terminal-DWA follow-pair pruning. Their labels are retained; scope is
    /// enforced by the parser-shard scoped identity transfers. The canonical
    /// global ignore is tracked separately through `ignore_terminal`.
    pub follow_transparent_ignores: Option<&'a BitSet>,
    pub terminal_offsets: &'a [u32],
    /// Optional leaf-to-immediate ownership for nested links. `None` means the
    /// terminal offset blocks are themselves the immediate components.
    pub leaf_to_immediate: Option<&'a [tdwa::scope::ImmediateComponentId]>,
    pub tokenizer_offsets: &'a [u32],
    pub component_state_counts: &'a [u32],
    /// Per-immediate-component original token whitelist. `None` for one entry
    /// means summary unavailable => full-vocabulary widening; `Some(empty)` is
    /// a proved empty candidate domain. `None` for the outer option preserves
    /// the full-vocabulary reference path used by low-level tests.
    pub candidate_tokens_by_component: Option<&'a [Option<Vec<u32>>]>,
    /// Retain non-crossing paths in the PARENT shard (component 0). Child
    /// shards always filter. The filter drops exactly the single-component
    /// paths, and a dropped path gaps the mask only if the composed table
    /// accepts its terminal sequence while no component mask admits the
    /// token. For a child-C-internal sequence at a C-topped stack, the
    /// boundary table's C rows are C's own rows, so composed acceptance
    /// implies C acceptance and C's mask covers it. For a parent-internal
    /// sequence, the parent mask covers it unless a zero-width empty-child
    /// derivation enables a parent-alone rejection — possible only when a
    /// linked child is start-nullable. Multi-component paths are crossing
    /// from every seed's view and are always kept. Nullable bound children
    /// decline loudly at signed-context certification, so retention only
    /// fires on embedded-nullable-but-effectively-nonnullable links.
    pub retain_parent_non_crossing_paths: bool,
    /// Explicit per-start-component walk plans (nested links). When `None`,
    /// one plan per component is derived from `component_state_counts` /
    /// `tokenizer_offsets` exactly as before (flat behavior unchanged).
    pub walk_plans: Option<Vec<BoundaryShardWalkPlan>>,
}

/// One built shard walk with a nonempty crossing set.
pub(crate) struct BuiltBoundaryShardWalk {
    pub start_component: usize,
    pub output: BoundaryWalkOutput,
    pub candidate_tokens: BTreeSet<u32>,
}

/// Walk plan for one start component: which leaf index the crossing filter
/// views from, the full merged-tokenizer commit set, and whether non-crossing
/// paths are retained. Flat links use one plan per direct component with the
/// component's own commit states; nested links use one plan per top-level
/// component where a block plan views from the block-root leaf with the union
/// of the block leaves' commit states (every block-internal crossing touches
/// a leaf outside the block root, so the filter keeps exactly the
/// block-crossing gaps, mirroring the proven nested fixture).
pub(crate) struct BoundaryShardWalkPlan {
    pub start_component: usize,
    /// Ownership class used only by the lexical crossing predicate. This is
    /// deliberately distinct from `start_component`: staged nested fallback
    /// may use a concrete root-leaf owner while publication still targets one
    /// immediate/top-level component.
    pub crossing_owner: tdwa::scope::ImmediateComponentId,
    pub commit_states: Vec<bool>,
    pub retain_non_crossing_paths: bool,
}

pub(crate) fn compute_conservative_follow_disallowed(
    disallowed_follows: &BTreeMap<u32, BitSet>,
    follow_transparent_ignores: Option<&BitSet>,
) -> Option<BTreeMap<u32, BitSet>> {
    let transparent = follow_transparent_ignores?;
    if transparent.is_zero() {
        return None;
    }
    let mut conservative = disallowed_follows.clone();
    for terminal in transparent.iter() {
        conservative.remove(&(terminal as u32));
    }
    for bits in conservative.values_mut() {
        for terminal in transparent.iter() {
            if terminal < bits.len() {
                bits.clear(terminal);
            }
        }
    }
    conservative.retain(|_, bits| !bits.is_zero());
    Some(conservative)
}

/// Link-level shard profile: shared-once cost + per-shard profiles.
pub(crate) struct BoundaryShardLinkProfile {
    pub shared_id_map_ms: f64,
    pub shared_wall_ms: f64,
    pub shared_tsids: usize,
    pub shared_itokens: usize,
    pub flat_ms: f64,
    pub per_shard: Vec<(usize, BoundaryWalkProfile)>,
}

/// Opt-in (`GLRMASK_PROFILE_COMPOSE`/`GLRMASK_PROFILE_COMPILE`) coherent
/// breakdown of one flat static-link call. `total_ms`, `setup`,
/// `link_context`, `walks`, `transfer_prepare`, and `shard_pipeline` are
/// INCLUSIVE monotonic-`Instant` walls. `walks_ms` is the wall of the
/// shard-walk fan-out, which runs via `rayon` `par_iter` unless
/// `GLRMASK_DISABLE_MACRO_PARALLELISM` is set — so per-shard rows may sum
/// above it from genuine thread overlap. Separately, per-shard
/// `BoundaryWalkProfile` rows are as-reported profile accumulation (NOT
/// per-worker elapsed walls): `walk_ms` is the inclusive L2P-call wall and
/// `terminal_dwa_ms` itself includes `determinize_ms` + `minimize_ms`, so the
/// printed subfields must never be summed into a "per-shard walk total".
/// The per-shard line therefore prints the raw subfields with an explicit
/// inclusive labeling — do not sum them and do not treat any sum as elapsed
/// time. Per-shard template/parser/publish fields are work diagnostics and may
/// overlap; they are not part of `accounted_ms`. `residual_ms` is `total -
/// accounted` using only non-overlapping stage walls.
pub(crate) struct WalkStaticLinkBreakdown {
    pub total_ms: f64,
    pub nested: bool,
    pub num_components: usize,
    pub setup_ms: f64,
    pub link_context_ms: f64,
    pub tokenizer_merge_ms: f64,
    pub grammar_analysis_ms: f64,
    pub candidate_summary_ms: f64,
    pub shared_flat_ms: f64,
    /// End-to-end wall from per-component lexical walk through publication.
    /// Component items may overlap under macro parallelism.
    pub component_pipeline_wall_ms: f64,
    pub transfer_prepare_ms: f64,
    /// Sum of per-shard work durations; not a critical-path wall.
    pub templates_ms: f64,
    /// Sum of per-shard work durations; not a critical-path wall.
    pub parser_dwa_ms: f64,
    /// Sum of per-shard work durations; not a critical-path wall.
    pub publish_ms: f64,
    pub accounted_ms: f64,
    pub residual_ms: f64,
    pub per_shard: Vec<WalkStaticLinkShardBreakdown>,
}

/// Per-shard row of [`WalkStaticLinkBreakdown`]: the walk-construction
/// profile is the existing `BoundaryWalkProfile` reported as-is (its
/// `walk_ms` is the inclusive L2P-call wall and `terminal_dwa_ms` includes
/// `determinize_ms` + `minimize_ms` — never sum the row), the parser-DWA
/// templates/parser/publish fields are true per-shard `Instant` work
/// durations and may overlap with other rows. `slot` names the grammar placeholder bound at this link
/// (`link.parent_component -> link.child_component` via that slot terminal).
pub(crate) struct WalkStaticLinkShardBreakdown {
    pub start_component: usize,
    pub parent_component: u32,
    pub child_component: u32,
    pub slot_terminal: u32,
    pub walk: BoundaryWalkProfile,
    pub summary_ms: f64,
    pub summary_history_positions: usize,
    pub summary_initial_frontier_pairs: usize,
    pub summary_byte_steps: usize,
    pub summary_peak_frontier_pairs: usize,
    pub summary_widened_subtrees: usize,
    pub templates_ms: f64,
    pub parser_dwa_ms: f64,
    pub publish_ms: f64,
}

fn boundary_analysis_json_enabled() -> bool {
    std::env::var_os("GLRMASK_BOUNDARY_ANALYSIS_JSON").is_some()
}

fn emit_boundary_analysis_json(breakdown: &WalkStaticLinkBreakdown) {
    if !boundary_analysis_json_enabled() {
        return;
    }
    let shards = breakdown
        .per_shard
        .iter()
        .map(|row| {
            serde_json::json!({
                "start_component": row.start_component,
                "parent_component": row.parent_component,
                "child_component": row.child_component,
                "slot_terminal": row.slot_terminal,
                "tokens": {
                    "original": row.walk.input_tokens,
                    "reusable_summary_candidates": row.walk.candidate_tokens,
                    "lexical_accepted": row.walk.lexical_accepted_tokens,
                },
                "states": {
                    "initial_raw": row.walk.initial_states,
                    "continuation_reset": row.walk.continuation_reset_states,
                    "tokenizer_classes": row.walk.tokenizer_classes,
                    "token_classes": row.walk.token_classes,
                },
                "summary": {
                    "ms": row.summary_ms,
                    "history_positions": row.summary_history_positions,
                    "initial_frontier_pairs": row.summary_initial_frontier_pairs,
                    "byte_steps": row.summary_byte_steps,
                    "peak_frontier_pairs": row.summary_peak_frontier_pairs,
                    "widened_subtrees": row.summary_widened_subtrees,
                },
                "timing_ms": {
                    "walk_setup": row.walk.setup_ms,
                    "walk_inclusive": row.walk.walk_ms,
                    "id_map": row.walk.id_map_ms,
                    "terminal_dwa_inclusive": row.walk.terminal_dwa_ms,
                    "compact": row.walk.compact_ms,
                    "determinize": row.walk.determinize_ms,
                    "minimize": row.walk.minimize_ms,
                    "templates": row.templates_ms,
                    "parser_dwa": row.parser_dwa_ms,
                    "publish": row.publish_ms,
                }
            })
        })
        .collect::<Vec<_>>();
    let record = serde_json::json!({
        "schema": "glrmask.boundary-analysis.v1",
        "nested": breakdown.nested,
        "components": breakdown.num_components,
        "timing_ms": {
            "total": breakdown.total_ms,
            "setup": breakdown.setup_ms,
            "link_context": breakdown.link_context_ms,
            "tokenizer_merge": breakdown.tokenizer_merge_ms,
            "grammar_analysis": breakdown.grammar_analysis_ms,
            "candidate_summary": breakdown.candidate_summary_ms,
            "shared_flat": breakdown.shared_flat_ms,
            "component_pipeline_wall": breakdown.component_pipeline_wall_ms,
            "transfer_prepare": breakdown.transfer_prepare_ms,
            "templates_work": breakdown.templates_ms,
            "parser_dwa_work": breakdown.parser_dwa_ms,
            "publish_work": breakdown.publish_ms,
            "accounted": breakdown.accounted_ms,
            "residual": breakdown.residual_ms,
        },
        "shards": shards,
    });
    eprintln!(
        "[glrmask/boundary-analysis] {}",
        serde_json::to_string(&record).expect("boundary analysis record serialization")
    );
}

/// Build every boundary shard walk for a link through the ordinary terminal
/// partition/family pipeline, independently scoped per immediate component.
/// Components with a proved empty candidate domain or empty crossing set are
/// skipped (the runtime skips missing shards). Returns `None` only when the
/// model vocabulary itself is empty.
fn map_boundary_shard_walks_with<R, F>(
    inputs: &BoundaryShardLinkInputs,
    consume: &F,
    early_adjacency: Option<&BTreeMap<u32,BitSet>>,
    prepared_first: Option<&[Option<super::boundary_precomputed_completion::PreparedSourceSpan>]>,
) -> Result<Option<(Vec<(usize, R)>, BoundaryShardLinkProfile)>, String>
where
    R: Send,
    F: Fn(BuiltBoundaryShardWalk) -> Result<R, String> + Sync,
{
    use rayon::prelude::*;

    let flat_started = Instant::now();
    let make_original_flat=|| Arc::from(tdwa::l1::build_flat_transition_table(inputs.merged_tokenizer));
    let use_direct_flat=crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_DIRECT_FLAT_ARC");
    let prepared_flat=use_direct_flat.then(|| super::boundary_flat_transitions::build(inputs.merged_tokenizer,true)).flatten();
    let direct_flat_selected=prepared_flat.is_some();
    let flat: Arc<[u32]> = prepared_flat.unwrap_or_else(make_original_flat);
    if direct_flat_selected && std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_DIRECT_FLAT_ARC").is_some() {
        let reference:Arc<[u32]>=make_original_flat();
        assert_eq!(&*flat,&*reference,"direct Arc transition table differs from ordinary scalar rows");
        eprintln!("[glrmask/validate][boundary_direct_flat_arc] exact=true cells={}",flat.len());
    }
    if use_direct_flat && compose_profile_enabled() {
        eprintln!("[glrmask/profile][boundary_direct_flat_arc] selected={direct_flat_selected}");
    }
    let flat_ms = flat_started.elapsed().as_secs_f64() * 1000.0;
    // Reuse the ordinary immutable direct table. Scalar rows take one lookup;
    // epsilon source/target rows preserve the exact original NFA operation.
    // Construction and every metadata query stay inside this link timer.
    let support_setup_started = Instant::now();
    let support_transitions = (crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_TOKEN_LIVENESS")
        && crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_TOKEN_FOLLOWS"))
        .then(|| super::boundary_token_support::PreparedSupportTransitions::new(
            inputs.merged_tokenizer, &flat)).flatten();
    if compose_profile_enabled() {
        eprintln!("[glrmask/profile][boundary_support_transition_setup] states={} ms={:.3}",
            inputs.merged_tokenizer.num_states(), support_setup_started.elapsed().as_secs_f64() * 1000.0);
    }
    let shared = crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_SHARED_COMPILE_CONTEXT")
        .then(|| tdwa::ScopedBoundarySharedContext::new(inputs.merged_tokenizer, inputs.grammar));
    let num_terms = inputs.grammar.num_terminals as usize;
    #[cfg(test)]
    if std::env::var_os("GLRMASK_DEBUG_L2P_TOKEN_CLASSES").is_some() {
        let always_allowed = tdwa::grammar_helpers::compute_always_allowed_follows(inputs.grammar);
        eprintln!(
            "[boundary-walk-setup] num_terms={} ignore_terminal={:?} follow_transparent_ignores={:?} always_allowed={:?} disallowed={:?}",
            num_terms,
            inputs.ignore_terminal,
            inputs.follow_transparent_ignores,
            always_allowed,
            inputs.disallowed_follows,
        );
    }
    let link_profile = BoundaryShardLinkProfile {
        shared_id_map_ms: 0.0,
        shared_wall_ms: 0.0,
        shared_tsids: 0,
        shared_itokens: 0,
        flat_ms,
        per_shard: Vec::new(),
    };
    let num_components = inputs.component_state_counts.len();
    let ownership = Arc::new(match inputs.leaf_to_immediate {
        Some(leaf_to_immediate) => {
            let owner_count = leaf_to_immediate
                .iter()
                .map(|owner| owner.0)
                .max()
                .map_or(0, |owner| owner.saturating_add(1));
            tdwa::scope::BoundaryOwnership::from_leaf_layout(
                inputs.terminal_offsets,
                inputs.grammar.num_terminals,
                leaf_to_immediate,
                owner_count,
            )
        }
        None => tdwa::scope::BoundaryOwnership::flat(
            inputs.terminal_offsets,
            inputs.grammar.num_terminals,
        ),
    }
    .expect("checked boundary link layout must define complete terminal ownership"));
    // Walk plans: explicit (nested) or derived per component (flat, unchanged).
    let plans: Vec<BoundaryShardWalkPlan> = match &inputs.walk_plans {
        Some(plans) => plans
            .iter()
            .map(|plan| BoundaryShardWalkPlan {
                start_component: plan.start_component,
                crossing_owner: plan.crossing_owner,
                commit_states: plan.commit_states.clone(),
                retain_non_crossing_paths: plan.retain_non_crossing_paths,
            })
            .collect(),
        None => (0..num_components)
            .map(|index| BoundaryShardWalkPlan {
                start_component: index,
                crossing_owner: tdwa::scope::ImmediateComponentId(index as u32),
                commit_states: commit_states_for_component(
                    inputs.tokenizer_offsets,
                    inputs.component_state_counts[index],
                    index,
                    inputs.merged_tokenizer.num_states() as usize,
                ),
                retain_non_crossing_paths: inputs.retain_parent_non_crossing_paths
                    && index == 0,
            })
            .collect(),
    };
    if let Some(candidate_sets) = inputs.candidate_tokens_by_component {
        assert_eq!(
            candidate_sets.len(),
            plans.len(),
            "boundary candidate sets must cover every planned start component",
        );
    }
    // `None` means "empty crossing set, skip"; each component lane owns its
    // scoped equivalence/terminal construction independently.
    let build_one = |plan: &BoundaryShardWalkPlan| -> Option<BuiltBoundaryShardWalk> {
        let candidate_storage;
        let candidate_vocab = match inputs
            .candidate_tokens_by_component
            .and_then(|sets| sets.get(plan.start_component))
        {
            Some(Some(ids)) => {
                if ids.is_empty() {
                    return None;
                }
                let entries = ids
                    .iter()
                    .filter_map(|&id| {
                        inputs
                            .vocab
                            .entries_map()
                            .get(&id)
                            .map(|bytes| (id, bytes.clone()))
                    })
                    .collect::<Vec<_>>();
                if entries.len() != ids.len() {
                    panic!("boundary candidate summary contains token IDs outside vocabulary");
                }
                candidate_storage = Vocab::new(entries);
                &candidate_storage
            }
            Some(None) | None => inputs.vocab,
        };
        let mut seed_mask = plan.commit_states.clone();
        if !plan.retain_non_crossing_paths
            && std::env::var_os("GLRMASK_DISABLE_BOUNDARY_PREFIX_SEEDS").is_none()
        {
            let started = Instant::now();
            let prepared_support = crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_PRECOMPUTED_PREFIX_SEEDS")
                .then_some(prepared_first).flatten()
                .and_then(|spans| spans.get(plan.start_component))
                .and_then(Option::as_ref)
                .and_then(|span| span.prefix_seed_support(
                    inputs.merged_tokenizer,candidate_vocab,&seed_mask,&ownership,plan.crossing_owner,
                ));
            let precomputed = prepared_support.is_some();
            if let Some(support) = &prepared_support
                && std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_PRECOMPUTED_PREFIX_SEEDS").is_some()
            {
                let (reference, _) = tdwa::scope::crossing_prefix_seed_support(
                    inputs.merged_tokenizer,candidate_vocab,&flat,&ownership,plan.crossing_owner,
                ).expect("original prefix-seed validator must run for this covered input");
                let expected = reference.iter().zip(&seed_mask).map(|(a,b)|*a&&*b).collect::<Vec<_>>();
                assert_eq!(support,&expected,"prepared prefix-seed query changed original source bits");
            }
            // No reverse-trie observation states are visited by the prepared
            // query. Zero is marked as not collected in the profile below.
            let supported = prepared_support.map(|support| (support,0)).or_else(||
                tdwa::scope::crossing_prefix_seed_support(
                    inputs.merged_tokenizer,candidate_vocab,&flat,&ownership,plan.crossing_owner,
                ));
            if let Some((support, observation_states)) = supported {
                let before = seed_mask.iter().filter(|&&keep| keep).count();
                for (keep, supported) in seed_mask.iter_mut().zip(support) { *keep &= supported; }
                let after = seed_mask.iter().filter(|&&keep| keep).count();
                if compose_profile_enabled() {
                    eprintln!("[glrmask/profile][boundary_prefix_seed_support] component={} tokens={} seeds_before={} seeds_after={} observation_states={} observation_states_collected={} precomputed={} elapsed_ms={:.3}",
                        plan.start_component, candidate_vocab.len(), before, after,
                        observation_states, !precomputed, precomputed, started.elapsed().as_secs_f64() * 1000.0);
                }
                if after == 0 { return None; }
            }
        }
        let initial_states = tdwa::scope::InitialStateDomain::from_mask(
            inputs.merged_tokenizer.num_states() as usize,
            seed_mask,
        )
        .expect("boundary walk plan must contain a nonempty checked initial-state domain");
        let scope = tdwa::scope::BoundaryAnalysisScope::new(
            initial_states,
            inputs
                .merged_tokenizer
                .deterministic_reset_states()
                .into_iter()
                .collect(),
            Arc::clone(&ownership),
            plan.crossing_owner,
            !plan.retain_non_crossing_paths,
            inputs.follow_transparent_ignores.cloned(),
        )
        .expect("boundary walk scope must agree with the checked merged layout");
        let original_candidate_vocab = candidate_vocab;
        let mut live_vocab = None;
        if !plan.retain_non_crossing_paths
            && crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_TOKEN_LIVENESS")
        {
            let started = Instant::now();
            let follow_aware = crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_TOKEN_FOLLOWS");
            let cut = (follow_aware && crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_CUT_SUPPORT"))
                .then(|| {
                    let prepared=crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_PREPARED_CUT")
                        .then_some(prepared_first).flatten().and_then(|spans|super::boundary_precomputed_completion::cut_support(
                            spans,inputs.merged_tokenizer,candidate_vocab,scope.initial_states().keep_raw(),
                            &ownership,plan.crossing_owner,inputs.disallowed_follows,
                            inputs.ignore_terminal,inputs.follow_transparent_ignores,early_adjacency,
                        ));
                    let ordinary=|| super::boundary_cut_support::crossing_support_with_transitions(
                        inputs.merged_tokenizer,candidate_vocab,scope.initial_states().keep_raw(),
                        &ownership,plan.crossing_owner,inputs.disallowed_follows,
                        inputs.ignore_terminal,inputs.follow_transparent_ignores,early_adjacency,support_transitions.as_ref(),
                    );
                    if let Some(candidate)=prepared.as_ref(){
                        if std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_PREPARED_CUT").is_some(){
                            let reference=ordinary().expect("ordinary necessary-cut reference must finish");
                            assert_eq!(candidate.tokens,reference.tokens,"prepared cut changed original token IDs");
                            eprintln!("[glrmask/validate][boundary_prepared_cut] component={} exact_ids=true",plan.start_component);
                        }
                    }
                    prepared.or_else(ordinary)
                }).flatten();
            let used_cut = cut.is_some();
            let support = if let Some(cut) = cut {
                if std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_CUT_SUPPORT").is_some() {
                    let reference = super::boundary_token_support::crossing_token_support_with_prepared_transitions(
                        inputs.merged_tokenizer, candidate_vocab, scope.initial_states().keep_raw(),
                        &ownership, plan.crossing_owner, inputs.disallowed_follows,
                        inputs.ignore_terminal, inputs.follow_transparent_ignores, early_adjacency,
                        support_transitions.as_ref(),
                    ).expect("independent support reference must finish in validation mode");
                    assert_eq!(cut.tokens, reference.tokens, "cut support differs from raw frontier observer");
                    eprintln!("[glrmask/validate][boundary_cut_support] component={} exact=true",plan.start_component);
                }
                if compose_profile_enabled() {
                    eprintln!("[glrmask/profile][boundary_cut_support] component={} before={} after={} profile={:?}",
                        plan.start_component,candidate_vocab.len(),cut.tokens.len(),cut.profile);
                }
                Some(super::boundary_token_support::CrossingTokenSupport {
                    tokens:cut.tokens, byte_prefixes:0, state_steps:cut.profile.raw_step_work, max_frontier:0,
                })
            } else if follow_aware {
                crate::compiler::boundary_token_support::crossing_token_support_with_prepared_transitions(
                    inputs.merged_tokenizer, candidate_vocab, scope.initial_states().keep_raw(),
                    &ownership, plan.crossing_owner, inputs.disallowed_follows,
                    inputs.ignore_terminal, inputs.follow_transparent_ignores, early_adjacency,
                    support_transitions.as_ref(),
                )
            } else {
                crate::compiler::boundary_token_support::crossing_token_support(
                    inputs.merged_tokenizer, candidate_vocab, scope.initial_states().keep_raw(),
                    &ownership, plan.crossing_owner,
                )
            };
            if let Some(support) = support {
                if compose_profile_enabled() {
                    // Cut mode reports its own counters above; zero legacy
                    // prefix/frontier counters mean not collected in that mode.
                    eprintln!("[glrmask/profile][boundary_token_liveness] component={} follows={follow_aware} cut={used_cut} before={} after={} byte_prefixes={} state_steps={} max_frontier={} elapsed_ms={:.3}",
                        plan.start_component, candidate_vocab.len(), support.tokens.len(),
                        support.byte_prefixes, support.state_steps, support.max_frontier,
                        started.elapsed().as_secs_f64() * 1000.0);
                }
                if !support.tokens.is_empty() && support.tokens.len() < candidate_vocab.len() {
                    live_vocab = Some(Vocab::new(support.tokens.into_iter().map(|id| {
                        (id, candidate_vocab.entries_map()[&id].clone())
                    }).collect()));
                }
                // The empty case retains the existing exact builder until its
                // empty-language result has been independently checked.
            }
        }
        let candidate_vocab = live_vocab.as_ref().unwrap_or(candidate_vocab);
        if let Some(directory) = std::env::var_os("GLRMASK_DUMP_BOUNDARY_INPUTS") {
            let directory = std::path::PathBuf::from(directory);
            std::fs::create_dir_all(&directory).expect("create boundary input diagnostics");
            let payload = serde_json::json!({
                "start_component": plan.start_component,
                "crossing_owner": plan.crossing_owner.0,
                "require_crossing": !plan.retain_non_crossing_paths,
                "raw_states": inputs.merged_tokenizer.num_states(),
                "num_terminals": inputs.grammar.num_terminals,
                "owners": (0..inputs.grammar.num_terminals).map(|t| ownership.owner_of_terminal(t).map(|owner| owner.0)).collect::<Vec<_>>(),
                "initial_raw_states": scope.initial_states().keep_raw().iter().enumerate().filter_map(|(q,&keep)| keep.then_some(q as u32)).collect::<Vec<_>>(),
                "reset_states": scope.reset_states(),
                "original_vocab": original_candidate_vocab.entries_map().iter().map(|(&id,bytes)| (id,bytes)).collect::<Vec<_>>(),
                "filtered_vocab": candidate_vocab.entries_map().iter().map(|(&id,bytes)| (id,bytes)).collect::<Vec<_>>(),
                "ignore_terminal": inputs.ignore_terminal,
                "follow_transparent": inputs.follow_transparent_ignores.map(|m| m.iter().collect::<Vec<_>>()),
                "extra_adjacency": early_adjacency.map(|rows| rows.iter().map(|(&t,m)| (t,m.iter().collect::<Vec<_>>())).collect::<Vec<_>>()),
                "disallowed": inputs.disallowed_follows.iter().map(|(&t,m)| (t,m.iter().collect::<Vec<_>>())).collect::<Vec<_>>(),
                "rules": inputs.grammar.rules.iter().map(|r| (r.lhs, r.rhs.iter().map(|s| match s {
                    crate::grammar::flat::Symbol::Terminal(t) => (0u8,*t),
                    crate::grammar::flat::Symbol::Nonterminal(n) => (1u8,*n),
                }).collect::<Vec<_>>())).collect::<Vec<_>>(),
                "terminal_names": inputs.grammar.terminal_display_names,
                "nonterminal_names": inputs.grammar.nonterminal_display_names,
            });
            std::fs::write(directory.join(format!("component-{}.json", plan.start_component)), payload.to_string())
                .expect("write boundary input diagnostics");
            std::fs::write(directory.join(format!("component-{}-tokenizer.bin", plan.start_component)),
                crate::automata::lexer::tokenizer::artifact_serde::to_fast_bytes(inputs.merged_tokenizer))
                .expect("write boundary input tokenizer");
        }
        let mut output = build_boundary_terminal_dwa_with_prepared(&BoundaryWalkInputs {
            merged_tokenizer: inputs.merged_tokenizer,
            vocab: candidate_vocab,
            grammar: inputs.grammar,
            disallowed_follows: inputs.disallowed_follows,
            ignore_terminal: inputs.ignore_terminal,
            follow_transparent_ignores: inputs.follow_transparent_ignores,
            scope: &scope,
            flat_trans: Some(&flat),
            retain_non_crossing_paths: plan.retain_non_crossing_paths,
        }, shared.as_ref(), prepared_first.and_then(|spans| spans.get(plan.start_component)).and_then(Option::as_ref))
        .expect("nonempty candidate-vocab shard walks must produce a DWA");
        if live_vocab.is_some()
            && std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_TOKEN_LIVENESS").is_some()
        {
            let mut reference = build_boundary_terminal_dwa(&BoundaryWalkInputs {
                merged_tokenizer: inputs.merged_tokenizer, vocab: original_candidate_vocab,
                grammar: inputs.grammar, disallowed_follows: inputs.disallowed_follows,
                ignore_terminal: inputs.ignore_terminal,
                follow_transparent_ignores: inputs.follow_transparent_ignores,
                scope: &scope, flat_trans: Some(&flat),
                retain_non_crossing_paths: plan.retain_non_crossing_paths,
            }).expect("reference boundary vocabulary is nonempty");
            let mut candidate = output.dwa.clone();
            if let Some(adjacency)=early_adjacency {
                // The new filter proves absence from the explicitly scoped
                // language, not from the old deliberately widened language.
                // Apply the SAME exact product to both independently built
                // full-vocab/filtered-vocab artifacts before comparing them.
                reference.dwa=minimize_acyclic_owned(tdwa::l2p::apply_explicit_follow_constraints(
                    &reference.dwa,adjacency,inputs.grammar.num_terminals as usize,None).dwa);
                candidate=minimize_acyclic_owned(tdwa::l2p::apply_explicit_follow_constraints(
                    &candidate,adjacency,inputs.grammar.num_terminals as usize,None).dwa);
            }
            tdwa::l2p::compare_terminal_dwa_artifacts(
                &tdwa::types::LocalIdMapTerminalDwa { dwa: reference.dwa, id_map: reference.id_map,
                    profile: Default::default() },
                &tdwa::types::LocalIdMapTerminalDwa { dwa: candidate, id_map: output.id_map.clone(),
                    profile: Default::default() },
            ).expect("boundary token support changed completed terminal weighted language");
            eprintln!("[glrmask/validate][boundary_token_liveness] component={} exact=true",plan.start_component);
        }
        if let Some(directory) = std::env::var_os("GLRMASK_DUMP_BOUNDARY_INPUTS") {
            let directory = std::path::PathBuf::from(directory);
            let mut id_map = output.id_map.clone();
            id_map.materialize_deferred_vocab_singletons();
            let t = &id_map.tokenizer_states;
            let v = &id_map.vocab_tokens;
            let bytes = bincode::serialize(&(&output.dwa,
                (&t.original_to_internal, &t.internal_to_originals, &t.representative_original_ids),
                (&v.original_to_internal, &v.internal_to_originals, &v.representative_original_ids)))
                .expect("serialize exact boundary terminal output");
            std::fs::write(directory.join(format!("component-{}-terminal.bin", plan.start_component)), bytes)
                .expect("write exact boundary terminal output");
            let expressions = inputs.merged_tokenizer.terminal_exprs().map(|exprs| exprs.to_vec());
            std::fs::write(directory.join(format!("component-{}-expressions.bin", plan.start_component)),
                bincode::serialize(&expressions).expect("serialize boundary expression observations"))
                .expect("write boundary expression observations");
        }
        // `build_boundary_terminal_dwa` sees the already restricted model-token
        // subset.  Preserve both sides of the funnel in the link-level profile:
        // full original model vocabulary before reusable-summary restriction,
        // then the exact candidate subset presented to the ordinary family
        // pipeline for this component.
        output.profile.input_tokens = inputs.vocab.len();
        output.profile.candidate_tokens = candidate_vocab.len();
        let reuse_inventory = output.profile.all_input_tokens_accepted
            && crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_REUSE_TOKEN_INVENTORY");
        let candidate_tokens = if reuse_inventory {
            candidate_vocab.entries_map().keys().copied().collect()
        } else { boundary_accepted_tokens(&output.dwa, &output.id_map) };
        if reuse_inventory && std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_TOKEN_INVENTORY").is_some() {
            assert_eq!(candidate_tokens, boundary_accepted_tokens(&output.dwa, &output.id_map),
                "cached all-input coverage must reproduce the untouched exact enumeration");
        }
        if compose_profile_enabled() {
            eprintln!("[glrmask/profile][boundary_token_inventory] component={} reused={reuse_inventory} tokens={}",
                plan.start_component, candidate_tokens.len());
        }
        output.profile.lexical_accepted_tokens = candidate_tokens.len();
        #[cfg(test)]
        if std::env::var_os("GLRMASK_DEBUG_A_WITNESS").is_some() {
            let emitted = boundary_emitted_terminals(
                &output.dwa,
                inputs.grammar.num_terminals as usize,
            );
            let emitted_ids = emitted
                .iter()
                .enumerate()
                .filter_map(|(terminal, &present)| present.then_some(terminal))
                .collect::<Vec<_>>();
            eprintln!(
                "[A-walk] start_component={} retain_non_crossing={} candidate_count={} token0_present={} token3_present={} candidates={:?} emitted_terminals={:?}",
                plan.start_component,
                plan.retain_non_crossing_paths,
                candidate_tokens.len(),
                candidate_tokens.contains(&0),
                candidate_tokens.contains(&3),
                candidate_tokens,
                emitted_ids,
            );
        }
        if candidate_tokens.is_empty() {
            return None;
        }
        Some(BuiltBoundaryShardWalk {
            start_component: plan.start_component,
            output,
            candidate_tokens,
        })
    };
    let build_and_consume = |plan: &BoundaryShardWalkPlan| -> Result<
        Option<(usize, R, BoundaryWalkProfile)>,
        String,
    > {
        let lane_started = Instant::now();
        let Some(shard) = build_one(plan) else {
            return Ok(None);
        };
        let start_component = shard.start_component;
        let profile = shard.output.profile.clone();
        let build_ms = lane_started.elapsed().as_secs_f64() * 1000.0;
        let consume_started = Instant::now();
        let value = consume(shard)?;
        if compose_profile_enabled() {
            eprintln!("[glrmask/profile][boundary_component_lane] component={start_component} build_ms={build_ms:.3} consume_ms={:.3} total_ms={:.3}",
                consume_started.elapsed().as_secs_f64() * 1000.0,
                lane_started.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(Some((start_component, value, profile)))
    };
    let mut completed = if crate::compiler::macro_parallelism_disabled() {
        let mut timings = Vec::with_capacity(plans.len());
        let built = plans
            .iter()
            .map(|plan| {
                let started = Instant::now();
                let result = build_and_consume(plan);
                timings.push(started.elapsed().as_secs_f64() * 1000.0);
                result
            })
            .collect::<Result<Vec<_>, String>>()?;
        crate::compiler::report_macro_item_timings("boundary_walk_component_pipelines", &timings);
        built.into_iter().flatten().collect::<Vec<_>>()
    } else {
        plans
            .par_iter()
            .map(build_and_consume)
            .collect::<Result<Vec<_>, String>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
    };
    completed.sort_by_key(|(start_component, _, _)| *start_component);
    let mut link_profile = link_profile;
    for (start_component, _, profile) in &completed {
        link_profile.per_shard.push((*start_component, profile.clone()));
    }
    let values = completed
        .into_iter()
        .map(|(start_component, value, _)| (start_component, value))
        .collect();
    Ok(Some((values, link_profile)))
}

/// Reference/test wrapper that stops after the scoped lexical walk. Production
/// uses `map_boundary_shard_walks_with` so each component can continue into its
/// parser pipeline immediately on the same worker.
pub(crate) fn build_boundary_shard_walks(
    inputs: &BoundaryShardLinkInputs,
) -> Option<(Vec<BuiltBoundaryShardWalk>, BoundaryShardLinkProfile)> {
    map_boundary_shard_walks_with(inputs, &|shard| Ok(shard), None, None)
        .expect("identity boundary-walk consumer cannot fail")
        .map(|(built, profile)| {
            (
                built.into_iter().map(|(_, shard)| shard).collect(),
                profile,
            )
        })
}

/// Inputs for a production static boundary link over one composition.
pub(crate) struct WalkStaticLinkInputs<'a> {
    pub parent: &'a Constraint,
    pub children: &'a [CompiledSubgrammarInput<'a>],
    pub vocab: &'a Vocab,
    /// Hybrid selection (parent first, then supplied children): a set bit
    /// requests a static shard. `None` requests static shards everywhere.
    pub static_components: Option<&'a BitSet>,
    /// Terminal offsets of the coordinator's own composed table. The walk link
    /// builds its own control-free splice; the offsets must agree (both lay
    /// out parent-then-children) and are pinned here so a layout change fails
    /// loudly instead of misrouting terminal ownership.
    pub expected_terminal_offsets: &'a [u32],
}

/// Output of a production static boundary link: published static shards plus
/// the per-component candidate metadata the install site needs.
pub(crate) struct WalkStaticLinkOutput {
    pub published_shards: Vec<PublishedStaticBoundaryShard>,
    pub boundary_tokens_by_start_component: Vec<Vec<u32>>,
    /// Effective static selection after virtual-residual exclusions (indexed
    /// in component order). Components not selected need exact dynamic shards.
    pub effective_static_components: BitSet,
    /// True when no component takes a static shard: the caller installs exact
    /// dynamic shards for every component instead.
    pub all_dynamic: bool,
    /// Expected runtime scoped leaf packing (link-component order) and total.
    /// The caller pins the installing runtime's layout against these after
    /// install; a mismatch (nesting the up-front check missed, a layout
    /// change) is a loud decline, never a silent dynamic swap.
    pub expected_leaf_tokenizer_offsets: Vec<u32>,
    pub expected_total_tokenizer_states: u32,
    /// Expected runtime scoped leaf terminal packing (same preorder) and
    /// total. Pinned alongside the tokenizer layout after install.
    pub expected_leaf_terminal_offsets: Vec<u32>,
    pub expected_total_leaf_terminals: u32,
    /// Top-level component indices whose nested retained boundary coverage is
    /// not statically certified. Their outer shard deliberately uses the
    /// legacy root-leaf crossing predicate, so it covers block-inner crossings
    /// and the installing runtime must clear those inner shards. Components
    /// absent from this set use true block ownership and retain their inner
    /// static shards as part of the coverage proof.
    pub clear_nested_boundary_components: BitSet,
}

/// One visited composed node during leaf expansion: its overlay plus the
/// per-direct-component leaf ranges assigned depth-first. Visits are keyed by
/// POSITION (visit order), never by constraint identity: DAG reuse of one
/// child under several slots expands once per use with distinct ranges, which
/// is valid and must not be rejected.
struct ExpansionVisit<'a> {
    overlay: &'a crate::runtime::StaticDynamicOverlayMetadata,
    /// Composed terminal domain of the visited node (its table's count).
    node_terminals: u32,
    subs: Vec<ExpansionSubtree>,
}

/// Leaf range assigned to one direct component of a visited node.
struct ExpansionSubtree {
    root: usize,
    first: usize,
    /// Exclusive end; ranges are contiguous depth-first by construction.
    last: usize,
    /// Visit id of the sub-expansion, or None when the component is intact.
    visit: Option<usize>,
}

/// An inner-overlay link before slot resolution: `slot_terminal` still names
/// a terminal in the PARENT COMPONENT's composed coordinate, not in a leaf.
/// Resolution descends to the owning leaf (see `resolve_expansion_slot`).
struct RawNestedLink {
    visit: usize,
    parent_component: usize,
    slot_terminal: u32,
    child_component: usize,
    child_start: u32,
    return_pop: u32,
    child_start_nullable: bool,
}

struct NestedExpansionRecorder<'a> {
    leaves: Vec<&'a Constraint>,
    visits: Vec<ExpansionVisit<'a>>,
    raw_links: Vec<RawNestedLink>,
}

/// Validate one overlay's terminal layout against the composer coordinate
/// contract (ascending contiguous offsets; each component's interval covers
/// exactly its terminal domain; dual-stored offsets agree). Any violation is
/// a loud mapping error, never an assumption.
fn validate_overlay_terminal_layout(
    overlay: &crate::runtime::StaticDynamicOverlayMetadata,
    node_terminals: u32,
) -> Result<(), String> {
    let components = &overlay.segmented_parser_components;
    let offsets = &overlay.terminal_offsets;
    if offsets.len() != components.len() {
        return Err(format!(
            "nested link overlay has {} terminal offsets for {} components",
            offsets.len(),
            components.len(),
        ));
    }
    if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err("nested link overlay terminal offsets are not ascending".to_string());
    }
    // Both splice producers pack the parent block first at terminal 0
    // (`compose_subgrammar_tables_with_rules` and its prepass twin push 0
    // before any child), and overlays only ever copy those offsets (or faithful
    // serde round-trips), so a nonzero first offset is corruption, not a new
    // layout: decline loudly instead of mapping slots against a shifted frame.
    if offsets.first() != Some(&0) {
        return Err("nested link overlay terminal offsets must start at 0".to_string());
    }
    for (index, component) in components.iter().enumerate() {
        let start = offsets[index];
        let end = if index + 1 < offsets.len() {
            offsets[index + 1]
        } else {
            node_terminals
        };
        if end < start {
            return Err(format!(
                "nested link overlay component {index} has an inverted terminal interval"
            ));
        }
        if component.terminal_offset != start {
            return Err(format!(
                "nested link overlay component {index} offset {} disagrees with overlay offsets {start}",
                component.terminal_offset,
            ));
        }
        let domain = component.constraint.table.num_terminals;
        if end.checked_sub(start) != Some(domain) {
            return Err(format!(
                "nested link overlay component {index} interval size {} differs from its terminal domain {domain}",
                end.saturating_sub(start),
            ));
        }
    }
    Ok(())
}

/// Map a composed-space slot terminal to its owning direct component plus the
/// component-local terminal. Ownership follows the composer contract (explicit
/// terminal aliases, else ascending-offset intervals via partition point, the
/// same rule as `boundary_tokens_by_start_component`). Checked arithmetic
/// throughout; zero or ambiguous owners are loud errors.
fn map_overlay_slot_terminal(
    overlay: &crate::runtime::StaticDynamicOverlayMetadata,
    slot: u32,
) -> Result<(usize, u32), String> {
    let components = &overlay.segmented_parser_components;
    let mut found: Option<(usize, u32)> = None;
    let mut ambiguous = false;
    for (index, component) in components.iter().enumerate() {
        for &(global, local) in &component.global_terminal_aliases {
            if global != slot {
                continue;
            }
            if local >= component.constraint.table.num_terminals {
                return Err(format!(
                    "nested link terminal alias maps slot {slot} outside component {index} domain",
                ));
            }
            match found {
                None => found = Some((index, local)),
                Some(previous) if previous == (index, local) => {}
                Some(_) => ambiguous = true,
            }
        }
    }
    let offsets = &overlay.terminal_offsets;
    let owner = offsets
        .partition_point(|&offset| offset <= slot)
        .checked_sub(1)
        .filter(|&owner| owner < components.len())
        .ok_or_else(|| {
            format!("nested link slot terminal {slot} lies outside the composed terminal domain")
        })?;
    let start = offsets[owner];
    let local = slot.checked_sub(start).ok_or_else(|| {
        format!("nested link slot terminal {slot} underflows component {owner} offset {start}")
    })?;
    if local >= components[owner].constraint.table.num_terminals {
        return Err(format!(
            "nested link slot terminal {slot} lies outside component {owner} terminal domain",
        ));
    }
    if ambiguous {
        return Err(format!(
            "nested link slot terminal {slot} has ambiguous owners (aliases disagree)"
        ));
    }
    match found {
        None => Ok((owner, local)),
        Some(previous) if previous == (owner, local) => Ok(previous),
        Some(_) => Err(format!(
            "nested link slot terminal {slot} has ambiguous owners (alias and interval disagree)"
        )),
    }
}

/// Resolve a slot in a direct component's terminal space to its owning leaf
/// plus the leaf-local terminal, descending composed subtrees. Intact
/// components resolve to their single leaf with a domain check.
fn resolve_expansion_slot(
    visits: &[ExpansionVisit<'_>],
    leaves: &[&Constraint],
    visit: usize,
    component: usize,
    slot: u32,
) -> Result<(usize, u32), String> {
    let record = visits
        .get(visit)
        .ok_or_else(|| format!("nested link references unknown expansion visit {visit}"))?;
    let sub = record.subs.get(component).ok_or_else(|| {
        format!("nested link references unknown inner component {component}")
    })?;
    match sub.visit {
        None => {
            if sub.last != sub.first + 1 {
                return Err(
                    "nested link intact sub-expansion covers multiple leaves".to_string(),
                );
            }
            let leaf = leaves.get(sub.first).ok_or_else(|| {
                format!("nested link leaf index {} is out of range", sub.first)
            })?;
            if slot >= leaf.table.num_terminals {
                return Err(format!(
                    "nested link slot terminal {slot} lies outside leaf table domain {}",
                    leaf.table.num_terminals,
                ));
            }
            Ok((sub.first, slot))
        }
        Some(sub_visit) => resolve_expansion_node_slot(visits, leaves, sub_visit, slot),
    }
}

/// Resolve a slot in a visited node's composed terminal space: map to the
/// owning direct component, then descend.
fn resolve_expansion_node_slot(
    visits: &[ExpansionVisit<'_>],
    leaves: &[&Constraint],
    visit: usize,
    slot: u32,
) -> Result<(usize, u32), String> {
    let overlay = &visits
        .get(visit)
        .ok_or_else(|| format!("nested link references unknown expansion visit {visit}"))?
        .overlay;
    let (sub, local) = map_overlay_slot_terminal(overlay, slot)?;
    resolve_expansion_slot(visits, leaves, visit, sub, local)
}

/// A link expanded to intact leaves: the parent block (leaves `0..P`, where an
/// intact parent is exactly leaf 0) plus every direct child, where composed
/// children AND a composed parent recurse into their overlays' intact
/// component constraints. Links are remapped to leaf coordinates: every link's
/// parent-local slot is resolved to its owning leaf plus a leaf-local slot, so
/// inner-overlay links stay valid at any nesting depth. The runtime recursive
/// layout visits leaves in this same pre-order, so the walk's leaf packing
/// matches the installing runtime's.
pub(crate) struct NestedLinkExpansion<'a> {
    /// Intact leaf constraints in link-component order (parent block first).
    pub leaves: Vec<&'a Constraint>,
    /// Full link set in leaf coordinates (outer + remapped inner links); every
    /// slot is local to its parent leaf's table.
    pub links: Vec<ScopedSubgrammarLink>,
    /// Back-to-back leaf terminal offsets plus total.
    pub leaf_terminal_offsets: Vec<u32>,
    pub num_terminals: u32,
    /// Leaf index of each top-level component's root (top 0 -> leaf 0).
    pub top_root_leaves: Vec<u32>,
    /// Leaf indices per top-level component (for block commit unions and
    /// virtual-residual checks).
    pub top_leaf_ranges: Vec<Vec<usize>>,
    /// Top-level terminal offsets/sizes (coordinator layout pin).
    pub top_terminal_offsets: Vec<u32>,
    pub top_terminal_sizes: Vec<u32>,
    /// True when any top-level component (parent block included) expanded to
    /// more than one leaf.
    pub nested: bool,
}

fn expand_recorded_subtree_leaves<'a>(
    constraint: &'a Constraint,
    recorder: &mut NestedExpansionRecorder<'a>,
) -> Result<ExpansionSubtree, String> {
    if !constraint.has_recursive_segmented_parser_tree() {
        let index = recorder.leaves.len();
        recorder.leaves.push(constraint);
        return Ok(ExpansionSubtree {
            root: index,
            first: index,
            last: index + 1,
            visit: None,
        });
    }
    let overlay = constraint.static_dynamic_overlay.as_ref().ok_or_else(|| {
        "nested link subtree reports a segmented tree but has no overlay".to_string()
    })?;
    if overlay.segmented_parser_components.is_empty() {
        return Err("nested link subtree has no segmented components".to_string());
    }
    validate_overlay_terminal_layout(overlay, constraint.table.num_terminals)?;
    let first = recorder.leaves.len();
    let mut subs = Vec::with_capacity(overlay.segmented_parser_components.len());
    for component in &overlay.segmented_parser_components {
        subs.push(expand_recorded_subtree_leaves(
            &component.constraint,
            recorder,
        )?);
    }
    // Depth-first sequential expansion assigns contiguous ranges; verify
    // rather than assume, since slot resolution depends on it.
    let mut cursor = first;
    for sub in &subs {
        if sub.first != cursor {
            return Err("nested link leaf ranges are not contiguous".to_string());
        }
        cursor = sub.last;
    }
    let visit = recorder.visits.len();
    recorder.visits.push(ExpansionVisit {
        overlay,
        node_terminals: constraint.table.num_terminals,
        subs,
    });
    for link in &overlay.segmented_parser_links {
        let parent_component = usize::try_from(link.parent_component)
            .map_err(|_| "nested link inner parent index overflow".to_string())?;
        let child_component = usize::try_from(link.child_component)
            .map_err(|_| "nested link inner child index overflow".to_string())?;
        if parent_component >= recorder.visits[visit].subs.len()
            || child_component >= recorder.visits[visit].subs.len()
        {
            return Err(format!(
                "nested link references unknown inner component (parent {}, child {})",
                link.parent_component, link.child_component,
            ));
        }
        recorder.raw_links.push(RawNestedLink {
            visit,
            parent_component,
            slot_terminal: link.slot_terminal,
            child_component,
            child_start: link.child_start,
            return_pop: link.return_pop,
            child_start_nullable: link.child_start_nullable,
        });
    }
    let root = recorder.visits[visit]
        .subs
        .first()
        .map(|sub| sub.root)
        .ok_or_else(|| "nested link subtree has no segmented components".to_string())?;
    Ok(ExpansionSubtree {
        root,
        first,
        last: cursor,
        visit: Some(visit),
    })
}

/// Expand one production link to intact leaves with leaf-coordinate links.
/// Composed parents expand exactly like composed children; every inner and
/// outer slot is resolved from its component-local coordinate to the owning
/// leaf plus a leaf-local slot, so links stay valid at any nesting depth.
pub(crate) fn expand_nested_link_leaves<'a>(
    parent: &'a Constraint,
    children: &'a [CompiledSubgrammarInput<'a>],
) -> Result<NestedLinkExpansion<'a>, String> {
    let mut recorder = NestedExpansionRecorder {
        leaves: Vec::new(),
        visits: Vec::new(),
        raw_links: Vec::new(),
    };
    let parent_expansion = expand_recorded_subtree_leaves(parent, &mut recorder)?;
    if parent_expansion.first != 0 || parent_expansion.root != 0 {
        return Err(
            "nested link parent expansion must occupy leaves from index 0".to_string(),
        );
    }
    let mut nested = parent_expansion.last - parent_expansion.first != 1;
    let mut top_root_leaves = vec![u32::try_from(parent_expansion.root)
        .map_err(|_| "nested link root leaf index overflow".to_string())?];
    let mut top_leaf_ranges =
        vec![(parent_expansion.first..parent_expansion.last).collect::<Vec<usize>>()];
    // Outer links over direct children (leaf remap happens below).
    let outer_links = build_segmented_parser_links(children)?;
    for child in children.iter() {
        let expansion = expand_recorded_subtree_leaves(child.constraint, &mut recorder)?;
        let range: Vec<usize> = (expansion.first..expansion.last).collect();
        if range.len() != 1 {
            nested = true;
        }
        let root_leaf = u32::try_from(expansion.root)
            .map_err(|_| "nested link root leaf index overflow".to_string())?;
        top_root_leaves.push(root_leaf);
        top_leaf_ranges.push(range);
    }
    let NestedExpansionRecorder {
        leaves,
        visits,
        raw_links,
    } = recorder;
    let mut links: Vec<ScopedSubgrammarLink> = Vec::new();
    // Remap inner-overlay links: slot coordinates are parent-component-local
    // and must be resolved to the owning leaf at any depth.
    for raw in &raw_links {
        let (parent_leaf, local_slot) = resolve_expansion_slot(
            &visits,
            &leaves,
            raw.visit,
            raw.parent_component,
            raw.slot_terminal,
        )?;
        let child_root = visits
            .get(raw.visit)
            .and_then(|record| record.subs.get(raw.child_component))
            .map(|sub| sub.root)
            .ok_or_else(|| {
                format!(
                    "nested link references unknown inner child {}",
                    raw.child_component
                )
            })?;
        links.push(ScopedSubgrammarLink {
            parent_component: u32::try_from(parent_leaf)
                .map_err(|_| "nested link leaf index overflow".to_string())?,
            slot_terminal: local_slot,
            child_component: u32::try_from(child_root)
                .map_err(|_| "nested link leaf index overflow".to_string())?,
            child_start: raw.child_start,
            return_pop: raw.return_pop,
            child_start_nullable: raw.child_start_nullable,
        });
    }
    // Remap outer links (parent block, child i+1) to leaf coordinates. Slots
    // name terminals in the parent block's composed coordinate; intact parents
    // keep the existing identity mapping, composed parents resolve per slot.
    for link in outer_links {
        let child_top = link.child_component as usize;
        if child_top == 0 || child_top > children.len() {
            return Err(format!(
                "nested link outer reference targets unknown top component {child_top}"
            ));
        }
        let (parent_leaf, local_slot) = match parent_expansion.visit {
            None => (0usize, link.slot_terminal),
            Some(visit) => {
                resolve_expansion_node_slot(&visits, &leaves, visit, link.slot_terminal)?
            }
        };
        links.push(ScopedSubgrammarLink {
            parent_component: u32::try_from(parent_leaf)
                .map_err(|_| "nested link leaf index overflow".to_string())?,
            slot_terminal: local_slot,
            child_component: top_root_leaves[child_top],
            child_start: link.child_start,
            return_pop: link.return_pop,
            child_start_nullable: link.child_start_nullable,
        });
    }
    let mut leaf_terminal_offsets = Vec::with_capacity(leaves.len());
    let mut total = 0u32;
    for leaf in &leaves {
        leaf_terminal_offsets.push(total);
        total = total
            .checked_add(leaf.table.num_terminals)
            .ok_or_else(|| "nested link terminal domain overflow".to_string())?;
    }
    let mut top_terminal_offsets = Vec::with_capacity(children.len() + 1);
    let mut top_terminal_sizes = Vec::with_capacity(children.len() + 1);
    for range in &top_leaf_ranges {
        let start = leaf_terminal_offsets[range[0]];
        let size: u32 = range
            .iter()
            .map(|&leaf| leaves[leaf].table.num_terminals)
            .try_fold(0u32, |acc, size| {
                acc.checked_add(size)
                    .ok_or_else(|| "nested link block size overflow".to_string())
            })?;
        top_terminal_offsets.push(start);
        top_terminal_sizes.push(size);
    }
    Ok(NestedLinkExpansion {
        leaves,
        links,
        leaf_terminal_offsets,
        num_terminals: total,
        top_root_leaves,
        top_leaf_ranges,
        top_terminal_offsets,
        top_terminal_sizes,
        nested,
    })
}

/// Whole-link exact-dynamic output: no static shards, every component dynamic.
pub(crate) fn dynamic_fallback_walk_link_output(num_components: usize) -> WalkStaticLinkOutput {
    WalkStaticLinkOutput {
        published_shards: Vec::new(),
        boundary_tokens_by_start_component: vec![Vec::new(); num_components],
        effective_static_components: BitSet::new(num_components),
        all_dynamic: true,
        expected_leaf_tokenizer_offsets: Vec::new(),
        expected_total_tokenizer_states: 0,
        expected_leaf_terminal_offsets: Vec::new(),
        expected_total_leaf_terminals: 0,
        clear_nested_boundary_components: BitSet::new(num_components),
    }
}

/// Effective static selection: the hybrid request minus components whose
/// tokenizer has a virtual residual runtime. Those can query lazily-allocated
/// tokenizer states at runtime that the link-time private TSID map cannot
/// cover, so their shards stay on the exact dynamic walker.
fn effective_static_selection(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    static_components: Option<&BitSet>,
) -> BitSet {
    let num_components = children.len() + 1;
    let mut effective = BitSet::new(num_components);
    for index in 0..num_components {
        let component =
            if index == 0 { parent } else { children[index - 1].constraint };
        let selected = static_components.is_none_or(|bits| bits.contains(index));
        if selected && !component.tokenizer.has_virtual_residual_runtime() {
            effective.set(index);
        }
    }
    effective
}

/// Unbound subgrammar slots of a link as composed-terminal IDs: special-token
/// terminals that are recorded late-grammar slots or whose backing token has
/// no bytes (out-of-vocab linker sentinels), minus the placeholders bound by
/// this link. Bound slots are spliced (gone from actions and rules); unbound
/// slots keep live placeholder shifts that would admit phantom terminal paths
/// through grammars that contribute nothing, so the caller empties them in
/// its private boundary-table copy.
pub(crate) fn unbound_link_slot_terminals(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    terminal_offsets: &[u32],
) -> Result<Vec<TerminalID>, String> {
    let bound: BTreeSet<u32> = children
        .iter()
        .flat_map(|child| {
            std::iter::once(child.placeholder_terminal)
                .chain(child.additional_placeholder_terminals.iter().copied())
        })
        .collect();
    let mut unbound = Vec::new();
    for special in &parent.special_token_terminals {
        if bound.contains(&special.terminal_id) {
            continue;
        }
        if parent.is_late_grammar_placeholder_terminal(special.terminal_id)
            || parent.token_bytes_for_id(special.token_id).is_none()
        {
            unbound.push(terminal_offsets[0].checked_add(special.terminal_id).ok_or_else(
                || "unbound parent slot terminal offset overflow".to_string(),
            )?);
        }
    }
    for (child_index, child) in children.iter().enumerate() {
        let constraint = child.constraint;
        for special in &constraint.special_token_terminals {
            if constraint.is_late_grammar_placeholder_terminal(special.terminal_id)
                || constraint.token_bytes_for_id(special.token_id).is_none()
            {
                unbound.push(
                    terminal_offsets[child_index + 1]
                        .checked_add(special.terminal_id)
                        .ok_or_else(|| {
                            format!(
                                "unbound slot terminal offset overflow for child {child_index}"
                            )
                        })?,
                );
            }
        }
    }
    unbound.sort_unstable();
    unbound.dedup();
    Ok(unbound)
}

/// Intact component tables for one flat link, with per-component standalone
/// ignore terminals (mirroring the splice inputs).
struct LinkComponentTables<'a> {
    tables: Vec<&'a GLRTable>,
    ignores: Vec<Option<TerminalID>>,
}

impl ParserComponentTableSource for LinkComponentTables<'_> {
    fn component_count(&self) -> usize {
        self.tables.len()
    }

    fn component_table(&self, component: u32) -> Option<&GLRTable> {
        self.tables.get(component as usize).copied()
    }

    fn component_ignore_terminal(&self, component: u32) -> Option<TerminalID> {
        // Same idempotence contract as the splice composer: a component table
        // that already owns this ignore in `skip_terminals` keeps its own
        // rows; only a raw component needs provider-level Identity.
        let table = self.tables.get(component as usize)?;
        self.ignores
            .get(component as usize)
            .copied()
            .flatten()
            .filter(|ignore| !table.skip_terminals.contains(ignore))
    }
}

/// Exact boundary table for one flat link (AUDIT/PIN ROUTE ONLY): the
/// disjoint-component provider over the intact component tables (per-link
/// Call/Return with call-specific parent targets), materialized and
/// control-eliminated. Production static linking no longer uses this object
/// (see `build_walk_static_boundary_link`); it remains for the called-frame
/// regression pins that compare provider-exact against splice-inexact
/// publication. Its state alphabet is the live leaf coordinate (components
/// back-to-back, local ids intact), and call-site provenance survives
/// elimination by construction (Call pushes the call-specific parent target
/// beneath the shared child start; Finish exposes it), so divergent
/// multi-site placeholders are exact. A single flattened shared start row
/// cannot express that provenance and must not be used here.
fn link_provider_boundary_table(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    terminal_offsets: &[u32],
    num_terminals: u32,
    global_ignores: bool,
) -> Result<GLRTable, String> {
    let mut tables = Vec::with_capacity(children.len() + 1);
    let mut ignores = Vec::with_capacity(children.len() + 1);
    tables.push(&*parent.table);
    ignores.push((!global_ignores).then_some(parent.ignore_terminal).flatten());
    for child in children {
        tables.push(&*child.constraint.table);
        ignores.push(
            (!global_ignores)
                .then_some(child.constraint.ignore_terminal)
                .flatten(),
        );
    }
    for (index, table) in tables.iter().enumerate() {
        if !table.control_terminals.is_empty() {
            return Err(format!(
                "walk static link component {index} carries linker controls; control-bearing components need dynamic shards",
            ));
        }
    }
    let source = LinkComponentTables { tables, ignores };
    let links = build_segmented_parser_links(children)?;
    let provider = DisjointComponentActionProvider::new(&source, &links)?;
    let mut terminal_symbols = Vec::with_capacity(num_terminals as usize);
    for global in 0..num_terminals {
        let owner = terminal_offsets
            .iter()
            .rposition(|&offset| offset <= global)
            .ok_or_else(|| {
                format!("composed terminal {global} has no owning component")
            })?;
        let local = global.checked_sub(terminal_offsets[owner]).ok_or_else(|| {
            format!("composed terminal {global} underflows its component offset")
        })?;
        let mut symbols = SmallVec::<[ScopedParserSymbol; 4]>::new();
        symbols.push(ScopedParserSymbol::Terminal {
            component: owner as u32,
            terminal: local,
        });
        terminal_symbols.push(symbols);
    }
    let table =
        materialize_control_eliminated_scoped_provider_table(&provider, &terminal_symbols)?;
    if table.num_terminals != num_terminals {
        return Err(format!(
            "provider boundary table terminal domain {} differs from composed {num_terminals}",
            table.num_terminals,
        ));
    }
    if !table.control_terminals.is_empty() {
        return Err("provider boundary table unexpectedly retains controls".to_string());
    }
    Ok(table)
}

/// Build every static boundary shard for one production link: a packed splice
/// (rules + terminal layout only), the signed-transfer link context over
/// intact local tables, a merged tokenizer, one independently scoped ordinary
/// terminal-DWA lane per component, then per-shard signed-NWA
/// compilation (scoped ordinary transfers + Entry/Finish exports, exact
/// negative resolution, table-free normalization) and publication. This
/// mirrors the gate's `link_and_time` walks; no provider-materialized table
/// and no exact_control_elimination exists on this route.
pub(crate) fn build_walk_static_boundary_link(
    inputs: &WalkStaticLinkInputs,
) -> Result<WalkStaticLinkOutput, String> {
    let parent = inputs.parent;
    let children = inputs.children;
    let link_total_started = Instant::now();
    // Expand a composed parent block and nested children to intact leaves
    // (flat links: leaves are exactly the direct components, so the flat path
    // below is unchanged).
    let expansion = expand_nested_link_leaves(parent, children)?;
    if expansion.nested {
        return build_walk_static_boundary_link_nested(inputs, &expansion);
    }
    let vocab = inputs.vocab;
    let num_components = children.len() + 1;
    let effective =
        effective_static_selection(parent, children, inputs.static_components);

    // The signed-transfer route links intact local tables and never invokes
    // control elimination. Reject control-bearing components first, before any
    // elimination-adjacent call, so even declined inputs cannot trigger it.
    for component in std::iter::once(parent).chain(children.iter().map(|child| child.constraint)) {
        if !component.table.control_terminals.is_empty() {
            return Err(
                "walk static link requires control-free component tables".to_string(),
            );
        }
    }

    // Packed splice: rules + terminal layout only. The packed shape is the
    // exact production construction (per-caller overlays); it feeds the
    // grammar/follows analysis and the coordinator-layout pin, never the
    // boundary parser. The boundary parser is compiled by the signed-transfer
    // compiler below from intact local tables; no provider-materialized table
    // and no exact_control_elimination exists on this route.
    let global_ignores = component_ignores_are_globally_erasable(parent, children);
    let parent_rules = parent.retained_table_rules()?;
    let child_rules = children
        .iter()
        .map(|child| child.constraint.retained_table_rules())
        .collect::<Result<Vec<_>, String>>()?;
    let table_inputs: Vec<SubgrammarTableInput> = children
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
        .collect();
    let mut composed = compose_subgrammar_tables_with_rules(
        &parent.table,
        parent_rules,
        (!global_ignores).then_some(parent.ignore_terminal).flatten(),
        &table_inputs,
        child_rules.as_slice(),
    )?;
    eliminate_composed_runtime_controls(&mut composed)?;
    if !composed.table.control_terminals.is_empty() || !composed.control_terminals.is_empty()
    {
        return Err(
            "walk static link requires a control-free spliced table".to_string(),
        );
    }
    if composed.terminal_offsets.as_slice() != inputs.expected_terminal_offsets {
        return Err(format!(
            "walk static link terminal layout differs from coordinator table: walk {:?} vs coordinator {:?} (leaf_terms {:?}, top_sizes {:?})",
            composed.terminal_offsets,
            inputs.expected_terminal_offsets,
            expansion.leaves.iter().map(|leaf| leaf.table.num_terminals).collect::<Vec<_>>(),
            expansion.top_terminal_sizes,
        ));
    }
    // Signed-transfer link context: intact local tables, provider-layout
    // injections, validated Entry/Finish contracts.
    let link_context_started = Instant::now();
    let unbound = unbound_link_slot_terminals(parent, children, &composed.terminal_offsets)?;
    for &terminal in &unbound {
        if terminal as usize >= composed.table.num_terminals as usize {
            return Err(format!(
                "unbound slot terminal {terminal} lies outside the composed terminal domain",
            ));
        }
    }
    let signed_context = crate::compiler::boundary_transfer::build_signed_link_context(
        parent,
        children,
        &composed.terminal_offsets,
        composed.table.num_terminals,
        global_ignores,
        unbound.into_iter().collect(),
    )?;
    let link_context_ms = link_context_started.elapsed().as_secs_f64() * 1000.0;

    // Merged tokenizer + ignores (mirror the gate's low_level_compose).
    let link_setup_started = Instant::now();
    let tokenizer_merge_started = Instant::now();
    let terminal_names = merged_terminal_display_names(parent, children);
    let tokenizer_inputs: Vec<(&Tokenizer, u32)> = std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .enumerate()
        .map(|(index, constraint)| {
            (constraint.composition_tokenizer(), composed.terminal_offsets[index])
        })
        .collect();
    let (mut merged, tokenizer_offsets) =
        Tokenizer::disjoint_union_with_terminal_offsets(&tokenizer_inputs);
    if tokenizer_offsets.first() != Some(&1) {
        return Err(
            "merged tokenizer must place the fresh reset fan-out at state 0".to_string(),
        );
    }
    // These spans are established at the actual disjoint-union boundary, from
    // the same immutable component allocations and its returned raw offsets.
    // The following Expr-sidecar restoration changes no byte/label topology.
    let prepared_first = crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_PRECOMPUTED_COMPLETION")
        .then(|| std::iter::once(parent)
            .chain(children.iter().map(|child| child.constraint))
            .zip(tokenizer_offsets.iter().copied())
            .zip(composed.terminal_offsets.iter().copied())
            .map(|((component, offset), terminal_offset)| super::boundary_precomputed_completion::PreparedSourceSpan::for_component(component, offset, terminal_offset))
            .collect::<Vec<_>>());
    if merged.terminal_exprs().is_none() {
        let all: Vec<&Constraint> = std::iter::once(parent)
            .chain(children.iter().map(|child| child.constraint))
            .collect();
        if let Some(exprs) = merged_retained_terminal_exprs(
            &all,
            &composed.terminal_offsets,
            composed.table.num_terminals,
        ) {
            merged.restore_terminal_exprs(Some(exprs))?;
        }
    }
    let ignores =
        merged_ignore_terminals(parent, children, &composed.terminal_offsets, global_ignores);
    let tokenizer_merge_ms = tokenizer_merge_started.elapsed().as_secs_f64() * 1000.0;

    // Grammar + pairwise follows over the non-emptied composed rules.
    let grammar_analysis_started = Instant::now();
    let augmented_start = composed
        .table
        .rules
        .first()
        .map(|rule| rule.lhs)
        .ok_or_else(|| "composed table contains no augmented-start rule".to_string())?;
    let grammar = AnalyzedGrammar::from_composed_rules(
        composed.table.rules.clone(),
        composed.table.num_terminals,
        terminal_names,
        composed.table.nonterminal_display_names.clone(),
        augmented_start,
    );
    let disallowed = compute_disallowed_follows(&grammar);
    let grammar_analysis_ms = grammar_analysis_started.elapsed().as_secs_f64() * 1000.0;
    let component_state_counts: Vec<u32> = tokenizer_inputs
        .iter()
        .map(|(tokenizer, _)| tokenizer.num_states())
        .collect();

    // Expected runtime scoped leaf packing (link-component order).
    let mut expected_leaf_tokenizer_offsets = Vec::with_capacity(num_components);
    let mut expected_total = 0u32;
    for &count in &component_state_counts {
        expected_leaf_tokenizer_offsets.push(expected_total);
        expected_total = expected_total
            .checked_add(count)
            .ok_or_else(|| "scoped tokenizer state count overflow".to_string())?;
    }

    let empty_output = || WalkStaticLinkOutput {
        published_shards: Vec::new(),
        boundary_tokens_by_start_component: vec![Vec::new(); num_components],
        effective_static_components: effective.clone(),
        all_dynamic: false,
        expected_leaf_tokenizer_offsets: expected_leaf_tokenizer_offsets.clone(),
        expected_total_tokenizer_states: expected_total,
        expected_leaf_terminal_offsets: expansion.leaf_terminal_offsets.clone(),
        expected_total_leaf_terminals: expansion.num_terminals,
        clear_nested_boundary_components: BitSet::new(num_components),
    };
    let retain_parent_non_crossing_paths = children
        .iter()
        .any(|child| child.constraint.table.embedded_start_nullable());
    let candidate_summary_started = Instant::now();
    let candidate_summary_profiles = std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .map(|component| {
            let started = Instant::now();
            let (ids, stats) =
                crate::compiler::boundary_candidates::boundary_candidate_ids(component, vocab);
            (ids, stats, started.elapsed().as_secs_f64() * 1000.0)
        })
        .collect::<Vec<_>>();
    let mut candidate_tokens_by_component = candidate_summary_profiles
        .iter()
        .map(|(ids, _, _)| ids.clone())
        .collect::<Vec<_>>();
    // A flat root has no outer RETURN, and its first crossing must enter one
    // of these children. This additional proof is link-aware, so compute and
    // apply it INSIDE the link timer and never install it on a reusable child
    // or parent artifact. Global ignores and nullable-child non-crossing
    // publication deliberately retain the existing conservative route.
    let no_ignores = std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .all(|component| component.ignore_terminal.is_none()
            && component.table.skip_terminals.is_empty());
    if (!global_ignores || no_ignores) && !retain_parent_non_crossing_paths {
        if let Some(existing) = candidate_tokens_by_component[0].as_mut() {
            let entered = children.iter().map(|child| child.constraint).collect::<Vec<_>>();
            let calls = children.iter().flat_map(|child| {
                std::iter::once(child.placeholder_terminal)
                    .chain(child.additional_placeholder_terminals.iter().copied())
            }).collect::<Vec<_>>();
            if let Ok(refined) = crate::compiler::boundary_tail::build_root_call_candidates(
                parent, &entered, &calls, vocab,
            ) {
                let reusable_count = existing.len();
                existing.retain(|id| refined.candidate_ids.binary_search(id).is_ok());
                if compose_profile_enabled() {
                    eprintln!(
                        "[glrmask/profile][boundary_root_call_candidates] reusable={} filtered={} exit_bytes={} entry_prefixes={:?} summary_ms={:.3} map_ms={:.3}",
                        reusable_count, existing.len(), refined.exit_byte_count,
                        refined.entry_prefix_count, refined.summary_ms, refined.map_ms,
                    );
                }
            }
        }
    }
    let candidate_summary_ms = candidate_summary_started.elapsed().as_secs_f64() * 1000.0;
    let scoped_adjacent = if crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_SCOPED_ADJACENCY")
        && !global_ignores
    {
        let started = Instant::now();
        let components = std::iter::once(parent).chain(children.iter().map(|child| child.constraint)).collect::<Vec<_>>();
        let counts = components.iter().map(|component| component.table.nonterminal_display_names.len()).collect::<Vec<_>>();
        let labels = components.iter().enumerate().map(|(owner, component)| {
            component.ignore_terminal.into_iter().chain(component.table.skip_terminals.iter().copied())
                .map(|terminal| terminal + composed.terminal_offsets[owner]).collect::<BTreeSet<_>>().into_iter().collect::<Vec<_>>()
        }).collect::<Vec<_>>();
        let reference_relation = || {
            let mut relation = crate::compiler::boundary_scoped_follow::scoped_follow_relation(&grammar,&counts,&labels);
            // Preserve the exact pre-existing ignore-incident projection.
            if let Some(relation) = relation.as_mut() {
                let ignored=labels.iter().flatten().copied().collect::<BTreeSet<_>>();
                for (&previous,blocked) in relation.iter_mut() {
                    if !ignored.contains(&previous) {
                        for next in 0..grammar.num_terminals {
                            if !ignored.contains(&next) { blocked.clear(next as usize); }
                        }
                    }
                }
                relation.retain(|_,blocked|!blocked.is_zero());
            }
            relation
        };
        let use_delta=crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_SCOPED_FOLLOW_DELTA");
        let candidate=use_delta.then(|| super::boundary_scoped_follow_delta::scoped_ignore_follow_relation(&grammar,&counts,&labels)).flatten();
        let delta_selected=candidate.is_some();
        let relation=if let Some(candidate)=candidate {
            if std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_SCOPED_FOLLOW_DELTA").is_some() {
                assert_eq!(Some(&candidate),reference_relation().as_ref(),"ignore-delta differs from unchanged padded grammar relation");
                eprintln!("[glrmask/validate][boundary_scoped_follow_delta] exact=true");
            }
            Some(candidate)
        } else { reference_relation() };
        if use_delta && compose_profile_enabled() {
            eprintln!("[glrmask/profile][boundary_scoped_follow_delta] selected={delta_selected}");
        }
        if compose_profile_enabled() {
            eprintln!("[glrmask/profile][boundary_scoped_adjacency_setup] selected={} ms={:.3}",relation.is_some(),started.elapsed().as_secs_f64()*1000.0);
        }
        relation
    } else { None };
    let link_setup_ms = link_setup_started.elapsed().as_secs_f64() * 1000.0;
    let transfer_cache = crate::compiler::boundary_transfer::FragmentTransferCache::new(
        &signed_context,
    )?;
    let transfer_prepare_ms = transfer_cache.prepare_ms;
    struct ProcessedShard {
        start_component: usize,
        candidates: Vec<u32>,
        published: Option<PublishedStaticBoundaryShard>,
        row: Option<WalkStaticLinkShardBreakdown>,
    }
    let process_shard = |mut shard: BuiltBoundaryShardWalk| -> Result<ProcessedShard, String> {
        let start_component = shard.start_component;
        let adjacency_reference = (scoped_adjacent.is_some()
            && std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_SCOPED_ADJACENCY").is_some()).then(|| shard.output.dwa.clone());
        if let Some(relation) = &scoped_adjacent {
            let started = Instant::now();
            let before_states = shard.output.dwa.num_states();
            let before_tokens = shard.candidate_tokens.len();
            let product = if crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_FOLLOW_ROW_QUOTIENT"){
                tdwa::l2p::apply_boundary_follow_constraints
            }else{tdwa::l2p::apply_explicit_follow_constraints};
            let filtered = product(&shard.output.dwa,relation,grammar.num_terminals as usize,None).dwa;
            let filtered = minimize_acyclic_owned(filtered);
            let candidates = boundary_accepted_tokens(&filtered,&shard.output.id_map);
            // Keep the empty-case reference until its parser emptiness proof is
            // explicitly checked; never silently erase an unsupported shard.
            if !candidates.is_empty() {
                shard.output.dwa = filtered;
                shard.candidate_tokens = candidates;
            }
            let elapsed = started.elapsed().as_secs_f64()*1000.0;
            // A new follow product has a new language; do not reuse the
            // preceding query's coverage certificate after this mutation.
            shard.output.profile.all_input_tokens_accepted = false;
            shard.output.profile.walk_ms += elapsed;
            shard.output.profile.lexical_accepted_tokens = shard.candidate_tokens.len();
            if compose_profile_enabled() {
                eprintln!("[glrmask/profile][boundary_scoped_adjacency] component={start_component} states_before={before_states} states_after={} tokens_before={before_tokens} tokens_after={} ms={elapsed:.3}",shard.output.dwa.num_states(),shard.candidate_tokens.len());
            }
        }
        let candidates = shard.candidate_tokens.iter().copied().collect::<Vec<_>>();
        if !effective.contains(start_component) {
            return Ok(ProcessedShard {
                start_component,
                candidates,
                published: None,
                row: None,
            });
        }
        let (slot_parent, slot_child, slot_terminal) = signed_context
            .links
            .iter()
            .find(|link| link.child_component as usize == start_component)
            .map(|link| (link.parent_component, link.child_component, link.slot_terminal))
            .unwrap_or((u32::MAX, start_component as u32, u32::MAX));
        let walk_profile = shard.output.profile.clone();
        let (_, summary_stats, summary_ms) = &candidate_summary_profiles[start_component];
        let emitted = boundary_emitted_terminals(
            &shard.output.dwa,
            composed.table.num_terminals as usize,
        );
        let templates_started = Instant::now();
        let library = crate::compiler::boundary_transfer::build_fragment_library_cached(
            &signed_context,
            &transfer_cache,
            &emitted,
            start_component as u32,
        )?;
        let shard_templates_ms = templates_started.elapsed().as_secs_f64() * 1000.0;
        #[cfg(test)]
        trace_boundary_terminal_dwa_words(
            &format!("component{start_component}"),
            &shard.output.dwa,
            composed.table.num_terminals as usize,
        );
        let parser_started = Instant::now();
        let compiled = crate::compiler::boundary_transfer::compile_signed_shard_parser(
            &signed_context,
            &library,
            &shard.output.dwa,
            &shard.output.id_map,
            start_component as u32,
        )?;
        let shard_parser_ms = parser_started.elapsed().as_secs_f64() * 1000.0;
        if let Some(reference) = adjacency_reference {
            let terms = boundary_emitted_terminals(&reference, grammar.num_terminals as usize);
            let reference_library = crate::compiler::boundary_transfer::build_fragment_library_cached(
                &signed_context,&transfer_cache,&terms,start_component as u32,
            )?;
            let reference = crate::compiler::boundary_transfer::compile_signed_shard_parser(
                &signed_context,&reference_library,&reference,&shard.output.id_map,start_component as u32,
            )?;
            let comparison=glrmask_parser_dwa::__private::parser_equivalence::compare_parser_mask_prefix_languages(
                &reference.parser_dwa,&compiled.parser_dwa,signed_context.total_scoped_states,500_000,
            )?;
            if let Some(difference)=comparison.difference {
                return Err(format!("scoped adjacency changed arbitrary-stack parser mask: {difference:?}"));
            }
            eprintln!("[glrmask/validate][boundary_scoped_adjacency] component={start_component} exact=true pairs={} branches={}",comparison.product_states,comparison.compared_branches);
        }

        let work = WalkBoundaryShardWork {
            start_component: start_component as u32,
            terminal_automaton: TerminalAutomaton::Dwa(shard.output.dwa),
            id_map: shard.output.id_map,
            candidate_tokens: Arc::from(candidates.clone().into_boxed_slice()),
        };
        let publish_started = Instant::now();
        let (published, publish_profile) = crate::compiler::boundary_transfer::publish_signed_shard(
            work,
            compiled,
            &tokenizer_offsets,
            &component_state_counts,
        )?;
        let shard_publish_ms = publish_started.elapsed().as_secs_f64() * 1000.0;
        let _ = publish_profile;
        Ok(ProcessedShard {
            start_component,
            candidates,
            published: Some(published),
            row: Some(WalkStaticLinkShardBreakdown {
                start_component,
                parent_component: slot_parent,
                child_component: slot_child,
                slot_terminal,
                walk: walk_profile,
                summary_ms: *summary_ms,
                summary_history_positions: summary_stats.history_positions,
                summary_initial_frontier_pairs: summary_stats.initial_frontier_pairs,
                summary_byte_steps: summary_stats.byte_steps,
                summary_peak_frontier_pairs: summary_stats.peak_frontier_pairs,
                summary_widened_subtrees: summary_stats.widened_subtrees,
                templates_ms: shard_templates_ms,
                parser_dwa_ms: shard_parser_ms,
                publish_ms: shard_publish_ms,
            }),
        })
    };
    let component_pipeline_started = Instant::now();
    let Some((processed, link_profile)) = map_boundary_shard_walks_with(
        &BoundaryShardLinkInputs {
            merged_tokenizer: &merged,
            vocab,
            grammar: &grammar,
            disallowed_follows: &disallowed,
            ignore_terminal: ignores.canonical,
            follow_transparent_ignores: Some(&ignores.scoped),
            terminal_offsets: &composed.terminal_offsets,
            leaf_to_immediate: None,
            tokenizer_offsets: &tokenizer_offsets,
            component_state_counts: &component_state_counts,
            candidate_tokens_by_component: Some(&candidate_tokens_by_component),
            retain_parent_non_crossing_paths,
            walk_plans: None,
        },
        &process_shard,
        crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_SCOPED_ADJACENCY_EARLY").then_some(scoped_adjacent.as_ref()).flatten(),
        prepared_first.as_deref(),
    )? else {
        return Ok(empty_output());
    };
    let component_pipeline_wall_ms =
        component_pipeline_started.elapsed().as_secs_f64() * 1000.0;
    let mut processed = processed
        .into_iter()
        .map(|(_, item)| item)
        .collect::<Vec<_>>();
    processed.sort_by_key(|item| item.start_component);
    let mut published = Vec::new();
    let mut tokens_by_component: Vec<Vec<u32>> = vec![Vec::new(); num_components];
    let mut shard_rows = Vec::<WalkStaticLinkShardBreakdown>::new();
    for item in processed {
        tokens_by_component[item.start_component] = item.candidates;
        if let Some(one) = item.published {
            published.push(one);
        }
        if let Some(row) = item.row {
            shard_rows.push(row);
        }
    }
    published.sort_by_key(|shard| shard.start_component);
    let templates_ms = shard_rows.iter().map(|row| row.templates_ms).sum::<f64>();
    let parser_dwa_ms = shard_rows.iter().map(|row| row.parser_dwa_ms).sum::<f64>();
    let publish_ms = shard_rows.iter().map(|row| row.publish_ms).sum::<f64>();
    let total_ms = link_total_started.elapsed().as_secs_f64() * 1000.0;
    let accounted_ms = link_setup_ms
        + link_context_ms
        + transfer_prepare_ms
        + component_pipeline_wall_ms;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][walk_static_link_breakdown] nested=false components={} total_ms={total_ms:.3} setup_ms={link_setup_ms:.3} link_context_ms={link_context_ms:.3} component_pipeline_wall_ms={component_pipeline_wall_ms:.3} shared_flat_ms={:.3} transfer_prepare_ms={transfer_prepare_ms:.3} templates_work_ms={templates_ms:.3} parser_dwa_work_ms={parser_dwa_ms:.3} publish_work_ms={publish_ms:.3} accounted_ms={accounted_ms:.3} residual_ms={residual_ms:.3}",
            num_components,
            link_profile.flat_ms,
            residual_ms = total_ms - accounted_ms,
        );
        for row in &shard_rows {
            eprintln!(
                "[glrmask/profile][walk_static_link_shard] start_component={} parent_component={} child_component={} slot_terminal={} walk_setup={:.3} walk_walk_inclusive={:.3} walk_idmap={:.3} walk_termdwa_includes_det_min={:.3} walk_compact={:.3} walk_determinize={:.3} walk_minimize={:.3} templates_ms={:.3} parser_dwa_ms={:.3} publish_ms={:.3}",
                row.start_component,
                row.parent_component,
                row.child_component,
                row.slot_terminal,
                row.walk.setup_ms,
                row.walk.walk_ms,
                row.walk.id_map_ms,
                row.walk.terminal_dwa_ms,
                row.walk.compact_ms,
                row.walk.determinize_ms,
                row.walk.minimize_ms,
                row.templates_ms,
                row.parser_dwa_ms,
                row.publish_ms,
            );
        }
        // Assembled only when profiling is on: with all profile env off the
        // per-shard row vec is still pushed (pointer clones, no weight work)
        // but never moved into the breakdown struct.
        let breakdown = WalkStaticLinkBreakdown {
            total_ms,
            nested: false,
            num_components,
            setup_ms: link_setup_ms,
            link_context_ms,
            tokenizer_merge_ms,
            grammar_analysis_ms,
            candidate_summary_ms,
            shared_flat_ms: link_profile.flat_ms,
            component_pipeline_wall_ms,
            transfer_prepare_ms,
            templates_ms,
            parser_dwa_ms,
            publish_ms,
            accounted_ms,
            residual_ms: total_ms - accounted_ms,
            per_shard: shard_rows,
        };
        emit_boundary_analysis_json(&breakdown);
    } else if boundary_analysis_json_enabled() {
        let breakdown = WalkStaticLinkBreakdown {
            total_ms,
            nested: false,
            num_components,
            setup_ms: link_setup_ms,
            link_context_ms,
            tokenizer_merge_ms,
            grammar_analysis_ms,
            candidate_summary_ms,
            shared_flat_ms: link_profile.flat_ms,
            component_pipeline_wall_ms,
            transfer_prepare_ms,
            templates_ms,
            parser_dwa_ms,
            publish_ms,
            accounted_ms,
            residual_ms: total_ms - accounted_ms,
            per_shard: shard_rows,
        };
        emit_boundary_analysis_json(&breakdown);
    }
    Ok(WalkStaticLinkOutput {
        published_shards: published,
        boundary_tokens_by_start_component: tokens_by_component,
        effective_static_components: effective,
        all_dynamic: false,
        expected_leaf_tokenizer_offsets,
        expected_total_tokenizer_states: expected_total,
        expected_leaf_terminal_offsets: expansion.leaf_terminal_offsets.clone(),
        expected_total_leaf_terminals: expansion.num_terminals,
        clear_nested_boundary_components: BitSet::new(num_components),
    })
}

/// Direct child block roots of one leaf in the link tree (sorted, dedup'd:
/// multi-slot binds contribute one link per slot for the same child).
/// Bounds-checked loudly; the splice and the DFS oracle share this so the two
/// traversals cannot drift apart.
fn link_tree_child_roots(
    expansion: &NestedLinkExpansion<'_>,
    parent_leaf: usize,
) -> Result<Vec<usize>, String> {
    let mut child_roots: Vec<usize> = Vec::new();
    for link in &expansion.links {
        if link.parent_component as usize != parent_leaf {
            continue;
        }
        let child = usize::try_from(link.child_component)
            .map_err(|_| "nested link child leaf index overflow".to_string())?;
        if child >= expansion.leaves.len() {
            return Err(format!(
                "nested link child leaf {child} is out of range for {} leaves",
                expansion.leaves.len(),
            ));
        }
        child_roots.push(child);
    }
    child_roots.sort_unstable();
    child_roots.dedup();
    Ok(child_roots)
}

/// Splice occurrence order. A shared child is visited once per parent path:
/// analysis rules may be duplicated, but their terminals map back to the
/// same runtime leaf. Reject cycles before invoking the recursive splice.
pub(crate) fn nested_splice_leaf_order(
    expansion: &NestedLinkExpansion<'_>,
) -> Result<Vec<usize>, String> {
    if expansion.leaves.is_empty() {
        return Err("nested link expansion has no leaves".to_string());
    }
    for link in &expansion.links {
        if link.parent_component as usize >= expansion.leaves.len()
            || link.child_component as usize >= expansion.leaves.len()
        {
            return Err("nested link endpoint lies outside leaf domain".to_string());
        }
    }
    enum Visit {
        Enter(usize),
        Exit(usize),
    }
    let mut order = Vec::with_capacity(expansion.leaves.len());
    let mut state = vec![0u8; expansion.leaves.len()];
    let mut stack = vec![Visit::Enter(0)];
    while let Some(frame) = stack.pop() {
        match frame {
            Visit::Enter(leaf) => {
                if leaf >= state.len() {
                    return Err(format!("nested link leaf {leaf} is out of range"));
                }
                if state[leaf] == 1 {
                    return Err(format!(
                        "nested static link unsupported: link-level control cycle through leaf {leaf} (use the Dynamic backend)",
                    ));
                }
                state[leaf] = 1;
                order.push(leaf);
                stack.push(Visit::Exit(leaf));
                // Push reversed so pops visit children ascending (splice order).
                for &child in link_tree_child_roots(expansion, leaf)?.iter().rev() {
                    stack.push(Visit::Enter(child));
                }
            }
            Visit::Exit(leaf) => {
                state[leaf] = 2;
            }
        }
    }
    let reached = state.iter().filter(|&&state| state == 2).count();
    if reached != expansion.leaves.len() {
        return Err(format!(
            "nested link forest leaves {} of {} reachable from leaf 0",
            reached,
            expansion.leaves.len(),
        ));
    }
    Ok(order)
}

/// Recursive packed splice for one nested block: inner blocks first, then
/// this level (rules + terminal layout only, mirroring the proven nested
/// fixture). Only the returned rules/offsets feed grammar analysis; no parser
/// behavior is ever derived from the packed shape.
///
/// The splice packs link-tree DFS pre-order, which interleaves with leaf
/// (overlay-storage) order whenever a late bind extends a non-root subtree:
/// e.g. siblings [root, A, B] plus C bound inside A packs [root, A, C, B]
/// while leaves stay [root, A, B, C]. The recorded visit order lets the caller
/// remap the packed rules explicitly into leaf coordinate (see
/// `remap_spliced_terminals_to_leaf_order`); traversal order is pinned against
/// the independent `nested_splice_leaf_order` oracle there.
fn splice_nested_block(
    root_leaf: usize,
    expansion: &NestedLinkExpansion<'_>,
    global_ignores: bool,
    visit_order: &mut Vec<usize>,
) -> Result<ComposedTable, String> {
    let root = expansion.leaves[root_leaf];
    visit_order.push(root_leaf);
    let child_roots = link_tree_child_roots(expansion, root_leaf)?;
    let mut child_tables = Vec::with_capacity(child_roots.len());
    let mut slot_lists: Vec<Vec<u32>> = Vec::with_capacity(child_roots.len());
    let mut nullables = Vec::with_capacity(child_roots.len());
    for &child_root in &child_roots {
        child_tables.push(splice_nested_block(
            child_root,
            expansion,
            global_ignores,
            visit_order,
        )?);
        let mut slots: Vec<u32> = expansion
            .links
            .iter()
            .filter(|link| {
                link.parent_component as usize == root_leaf
                    && link.child_component as usize == child_root
            })
            .map(|link| link.slot_terminal)
            .collect();
        slots.sort_unstable();
        slots.dedup();
        if slots.is_empty() {
            return Err(format!(
                "nested link block {root_leaf} has a child block with no slot terminal"
            ));
        }
        let nullable = expansion
            .links
            .iter()
            .find(|link| {
                link.parent_component as usize == root_leaf
                    && link.child_component as usize == child_root
            })
            .map(|link| link.child_start_nullable)
            .unwrap_or(false);
        slot_lists.push(slots);
        nullables.push(nullable);
    }
    let child_rules: Vec<&[crate::grammar::flat::Rule]> = child_tables
        .iter()
        .map(|tabled: &ComposedTable| tabled.table.rules.as_slice())
        .collect();
    let table_inputs: Vec<SubgrammarTableInput> = child_roots
        .iter()
        .zip(child_tables.iter())
        .zip(slot_lists.iter())
        .zip(nullables.iter())
        .map(|(((&child_root, table), slots), &nullable)| {
            let (placeholder, additionals) = slots
                .split_first()
                .expect("nested slot list is nonempty");
            let child_leaf = expansion.leaves[child_root];
            SubgrammarTableInput {
                placeholder_terminal: *placeholder,
                additional_placeholder_terminals: additionals,
                table: &table.table,
                ignore_terminal: (!global_ignores)
                    .then_some(child_leaf.ignore_terminal)
                    .flatten(),
                start_nullable: nullable,
            }
        })
        .collect();
    let mut composed = compose_subgrammar_tables_with_rules(
        &root.table,
        root.retained_table_rules()?,
        (!global_ignores).then_some(root.ignore_terminal).flatten(),
        &table_inputs,
        child_rules.as_slice(),
    )?;
    eliminate_composed_runtime_controls(&mut composed)?;
    if !composed.table.control_terminals.is_empty() || !composed.control_terminals.is_empty() {
        return Err(
            "walk static link requires a control-free spliced table".to_string(),
        );
    }
    Ok(composed)
}

/// Remap a recursively spliced block table from splice (link-tree DFS
/// occurrence) terminal coordinate into leaf coordinate.
///
/// The splice packs occurrence pre-order (a shared child once per parent
/// path) while every downstream consumer speaks leaf order: the grammar's
/// display names are leaf-ordered, the walks take leaf offsets, and the
/// install pin checks the runtime leaf layout. When occurrence order
/// interleaves with leaf order (late binds into non-root subtrees, or any
/// shared child), each packed occurrence maps onto its shared runtime leaf
/// explicitly (many-to-one for repeats, never a permutation assumption).
///
/// This only rewrites `Symbol::Terminal` ids in the composed rules. That is
/// complete: downstream reads exactly the rules, the (order-free) terminal
/// domain size, and the nonterminal names (`AnalyzedGrammar::from_composed_rules`
/// derives nullable/first/follow from the rules alone); nonterminal ids are an
/// internally consistent packing under any leaf order, so they are untouched.
/// The root-frame offsets are pinned against DFS-derived expectations first
/// (the splice must have packed what the traversal recorded), then rewritten
/// to the leaf-coordinate block-root offsets the rules now speak.
fn remap_spliced_terminals_to_leaf_order(
    expansion: &NestedLinkExpansion<'_>,
    dfs_order: &[usize],
    composed: &mut ComposedTable,
) -> Result<(), String> {
    if dfs_order.len() < expansion.leaves.len() {
        return Err(format!(
            "nested splice visit order covers {} of {} leaves",
            dfs_order.len(),
            expansion.leaves.len(),
        ));
    }
    // Every runtime leaf must occur; shared leaves may occur more than once.
    {
        let mut seen = vec![false; expansion.leaves.len()];
        for &leaf in dfs_order {
            let slot = seen.get_mut(leaf).ok_or_else(|| {
                format!("nested splice visit order names unknown leaf {leaf}")
            })?;
            *slot = true;
        }
        if seen.contains(&false) {
            return Err("nested splice visit order omits a runtime leaf".to_string());
        }
    }
    // Offsets belong to occurrences, not leaf identities: a shared child's
    // later copy has a different source offset but the same destination.
    let mut dfs_offsets = Vec::with_capacity(dfs_order.len());
    let mut cursor = 0u32;
    for &leaf in dfs_order {
        let size = expansion
            .leaves
            .get(leaf)
            .ok_or_else(|| format!("nested splice visit order names unknown leaf {leaf}"))?
            .table
            .num_terminals;
        dfs_offsets.push(cursor);
        cursor = cursor
            .checked_add(size)
            .ok_or_else(|| "nested splice terminal domain overflow".to_string())?;
    }
    if cursor != composed.table.num_terminals {
        return Err(format!(
            "nested splice DFS domain {cursor} differs from packed domain {}",
            composed.table.num_terminals,
        ));
    }
    // Root-frame pin: the splice packs the root leaf plus its direct child
    // blocks depth-first from this order, so each packed offset must equal the
    // DFS offset of its block root. A mismatch means the splice did not pack
    // what the traversal recorded: decline loudly, never remap blindly.
    let mut frame = vec![0usize];
    frame.extend(link_tree_child_roots(expansion, 0)?);
    let mut block_sizes = vec![0u32; expansion.leaves.len()];
    for &leaf in dfs_order.iter().rev() {
        let mut size = expansion.leaves[leaf].table.num_terminals;
        for child in link_tree_child_roots(expansion, leaf)? {
            size = size.checked_add(block_sizes[child])
                .ok_or_else(|| "nested splice block domain overflow".to_string())?;
        }
        block_sizes[leaf] = size;
    }
    let mut expected = vec![0];
    let mut frame_cursor = expansion.leaves[0].table.num_terminals;
    for &child in frame.iter().skip(1) {
        expected.push(frame_cursor);
        frame_cursor = frame_cursor.checked_add(block_sizes[child])
            .ok_or_else(|| "nested splice frame domain overflow".to_string())?;
    }
    if frame_cursor != cursor {
        return Err("nested splice frame domain differs from occurrence domain".to_string());
    }
    if composed.terminal_offsets.as_slice() != expected.as_slice() {
        return Err(format!(
            "nested walk splice layout {:?} differs from DFS block layout {:?}",
            composed.terminal_offsets, expected,
        ));
    }
    // Map every packed occurrence into the shared runtime leaf domain.
    let mut map = vec![0u32; cursor as usize];
    for (&leaf, &offset) in dfs_order.iter().zip(&dfs_offsets) {
        let size = expansion.leaves[leaf].table.num_terminals;
        for local in 0..size {
            let from = offset
                .checked_add(local)
                .ok_or_else(|| "nested splice remap offset overflow".to_string())?;
            let to = expansion.leaf_terminal_offsets[leaf]
                .checked_add(local)
                .ok_or_else(|| "nested splice remap offset overflow".to_string())?;
            let slot = map.get_mut(from as usize).ok_or_else(|| {
                format!("nested splice terminal {from} lies outside the DFS domain")
            })?;
            *slot = to;
        }
    }
    for rule in &mut composed.table.rules {
        for symbol in &mut rule.rhs {
            if let Symbol::Terminal(id) = symbol {
                *id = *map.get(*id as usize).ok_or_else(|| {
                    format!(
                        "nested splice rule terminal {id} lies outside the DFS domain {}",
                        map.len(),
                    )
                })?;
            }
        }
    }
    // The rules now speak leaf coordinate; keep the frame offsets describing
    // them (block-root leaf offsets, ascending by construction).
    composed.terminal_offsets = frame
        .iter()
        .map(|&root| expansion.leaf_terminal_offsets[root])
        .collect();
    composed.table.num_terminals = expansion.num_terminals;
    Ok(())
}

/// Unbound subgrammar slots in leaf coordinates: a slot is bound iff some
/// link fills it; the rest follows the flat helper's rule per leaf
/// (late-grammar placeholders or out-of-vocab linker sentinels).
fn nested_unbound_slot_terminals(
    expansion: &NestedLinkExpansion<'_>,
) -> Result<Vec<TerminalID>, String> {
    let bound: BTreeSet<u32> = expansion
        .links
        .iter()
        .map(|link| {
            expansion.leaf_terminal_offsets[link.parent_component as usize]
                .checked_add(link.slot_terminal)
                .ok_or_else(|| "bound slot terminal offset overflow".to_string())
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    let mut unbound = Vec::new();
    for (leaf_index, leaf) in expansion.leaves.iter().enumerate() {
        for special in &leaf.special_token_terminals {
            let global = expansion.leaf_terminal_offsets[leaf_index]
                .checked_add(special.terminal_id)
                .ok_or_else(|| {
                    format!("slot terminal offset overflow for leaf {leaf_index}")
                })?;
            if bound.contains(&global) {
                continue;
            }
            if leaf.is_late_grammar_placeholder_terminal(special.terminal_id)
                || leaf.token_bytes_for_id(special.token_id).is_none()
            {
                unbound.push(global);
            }
        }
    }
    unbound.sort_unstable();
    unbound.dedup();
    Ok(unbound)
}

/// Whether every retained boundary contribution below this component is
/// already static (or absent because no repair is needed). Only such a block
/// may omit its internal leaf crossings from a new outer shard: those paths
/// are then covered by the retained component runtime itself. A dynamic inner
/// shard fails this proof and forces the staged legacy crossing predicate at
/// the outer link.
fn retained_internal_static_boundary_coverage(component: &Constraint) -> bool {
    let Some(overlay) = component.static_dynamic_overlay.as_ref() else {
        return true;
    };
    overlay.segmented_parser_components.iter().all(|child| {
        let local_static = child.boundary.as_ref().is_none_or(|shard| {
            matches!(
                shard.backend,
                crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
            )
        });
        local_static && retained_internal_static_boundary_coverage(&child.constraint)
    })
}

/// Nested production static link: the flat pipeline over recursively expanded
/// intact leaves with the full multi-level link set (see the flat function
/// for the pipeline stages). One walk per top-level component; publication
/// per top-level component with leaf tokenizer coordinates.
fn build_walk_static_boundary_link_nested(
    inputs: &WalkStaticLinkInputs,
    expansion: &NestedLinkExpansion<'_>,
) -> Result<WalkStaticLinkOutput, String> {
    let link_total_started = Instant::now();
    let vocab = inputs.vocab;
    let num_components = inputs.children.len() + 1;
    let leaves = &expansion.leaves;

    // The signed-transfer route links intact local tables and never invokes
    // control elimination. Reject control-bearing leaves first.
    for leaf in leaves.iter() {
        if !leaf.table.control_terminals.is_empty() {
            return Err(
                "walk static link requires control-free component tables".to_string(),
            );
        }
    }

    // Effective static selection over top-level components: a requested block
    // needs every leaf in its subtree free of virtual residual runtimes, whose
    // lazily-allocated tokenizer states the link-time coordinate cannot cover.
    // A requested-but-blocked block is a loud error, never a silent dynamic
    // swap; unrequested blocks stay on the exact dynamic walker (hybrid).
    let mut effective = BitSet::new(num_components);
    for top in 0..num_components {
        let requested = inputs
            .static_components
            .is_none_or(|bits| bits.contains(top));
        if !requested {
            continue;
        }
        let mut blocked = None;
        for &leaf_index in &expansion.top_leaf_ranges[top] {
            if leaves[leaf_index].tokenizer.has_virtual_residual_runtime() {
                blocked = Some(leaf_index);
                break;
            }
        }
        if let Some(leaf_index) = blocked {
            return Err(format!(
                "walk static link component {top} requested a static shard but leaf {leaf_index} has a virtual residual runtime; leave it unselected (hybrid) or use the Dynamic boundary backend",
            ));
        }
        effective.set(top);
    }

    let global_ignores = leaf_ignores_are_globally_erasable(leaves);

    // Signed-transfer context over leaves first: the bounded-closure
    // certificate declines nullable bound children and link-level cycles
    // loudly here, before any walks run.
    let unbound = nested_unbound_slot_terminals(expansion)?;
    for &terminal in &unbound {
        if terminal >= expansion.num_terminals {
            return Err(format!(
                "unbound slot terminal {terminal} lies outside the composed terminal domain",
            ));
        }
    }
    let pre_context_setup_ms = link_total_started.elapsed().as_secs_f64() * 1000.0;
    let link_context_started = Instant::now();
    let signed_context = crate::compiler::boundary_transfer::build_signed_link_context_from_parts(
        leaves.iter().map(|leaf| &*leaf.table).collect(),
        leaves.iter().map(|leaf| leaf.ignore_terminal).collect(),
        expansion.links.clone(),
        &expansion.leaf_terminal_offsets,
        expansion.num_terminals,
        global_ignores,
        unbound.into_iter().collect(),
    )?.with_component_template_sources(&leaves)?;
    let link_context_ms = link_context_started.elapsed().as_secs_f64() * 1000.0;
    let post_context_setup_started = Instant::now();

    // Packed recursive splice (rules + layout only) for grammar analysis and
    // the coordinator-layout pin. The splice packs link-tree DFS order, which
    // interleaves with leaf order on late non-root extensions; pin the splice
    // traversal against the independent DFS oracle, then remap the packed
    // rules explicitly into leaf coordinate (never assume the orders coincide).
    let dfs_order = nested_splice_leaf_order(expansion)?;
    let mut splice_order = Vec::with_capacity(expansion.leaves.len());
    let mut composed = splice_nested_block(0, expansion, global_ignores, &mut splice_order)?;
    if splice_order != dfs_order {
        return Err(format!(
            "nested walk splice visit order {splice_order:?} differs from link-tree DFS order {dfs_order:?}",
        ));
    }
    remap_spliced_terminals_to_leaf_order(expansion, &dfs_order, &mut composed)?;
    if composed.table.num_terminals != expansion.num_terminals {
        return Err(format!(
            "nested walk terminal domain {} differs from leaf domain {}",
            composed.table.num_terminals, expansion.num_terminals,
        ));
    }
    // Coordinator pin on the TOP frame: the splice packs the root block's
    // direct children (root frame), which coincides with tops only when top 0
    // is a single leaf. The leaf-expansion top starts must match the
    // coordinator's per-top starts exactly.
    if expansion.top_terminal_offsets.as_slice() != inputs.expected_terminal_offsets {
        return Err(format!(
            "nested walk top-level layout differs from coordinator table: walk {:?} vs coordinator {:?} (leaf_terms {:?}, top_sizes {:?})",
            expansion.top_terminal_offsets,
            inputs.expected_terminal_offsets,
            expansion.leaves.iter().map(|leaf| leaf.table.num_terminals).collect::<Vec<_>>(),
            expansion.top_terminal_sizes,
        ));
    }

    // Merged tokenizer + names + ignores over leaves.
    let tokenizer_merge_started = Instant::now();
    let terminal_names = merged_leaf_terminal_display_names(leaves);
    let tokenizer_inputs: Vec<(&Tokenizer, u32)> = leaves
        .iter()
        .enumerate()
        .map(|(index, leaf)| {
            (
                leaf.composition_tokenizer(),
                expansion.leaf_terminal_offsets[index],
            )
        })
        .collect();
    let (mut merged, tokenizer_offsets) =
        Tokenizer::disjoint_union_with_terminal_offsets(&tokenizer_inputs);
    if tokenizer_offsets.first() != Some(&1) {
        return Err(
            "merged tokenizer must place the fresh reset fan-out at state 0".to_string(),
        );
    }
    if merged.terminal_exprs().is_none() {
        if let Some(exprs) = merged_retained_terminal_exprs(
            leaves,
            &expansion.leaf_terminal_offsets,
            expansion.num_terminals,
        ) {
            merged.restore_terminal_exprs(Some(exprs))?;
        }
    }
    let ignores =
        merged_leaf_ignore_terminals(leaves, &expansion.leaf_terminal_offsets, global_ignores);
    let tokenizer_merge_ms = tokenizer_merge_started.elapsed().as_secs_f64() * 1000.0;

    // Grammar + pairwise follows over the spliced rules.
    let grammar_analysis_started = Instant::now();
    let augmented_start = composed
        .table
        .rules
        .first()
        .map(|rule| rule.lhs)
        .ok_or_else(|| "composed table contains no augmented-start rule".to_string())?;
    let grammar = AnalyzedGrammar::from_composed_rules(
        composed.table.rules.clone(),
        composed.table.num_terminals,
        terminal_names,
        composed.table.nonterminal_display_names.clone(),
        augmented_start,
    );
    let disallowed = compute_disallowed_follows(&grammar);
    let grammar_analysis_ms = grammar_analysis_started.elapsed().as_secs_f64() * 1000.0;
    let leaf_state_counts: Vec<u32> = tokenizer_inputs
        .iter()
        .map(|(tokenizer, _)| tokenizer.num_states())
        .collect();

    // Expected runtime scoped leaf packing (leaf order).
    let mut expected_leaf_tokenizer_offsets = Vec::with_capacity(leaves.len());
    let mut expected_total = 0u32;
    for &count in &leaf_state_counts {
        expected_leaf_tokenizer_offsets.push(expected_total);
        expected_total = expected_total
            .checked_add(count)
            .ok_or_else(|| "scoped tokenizer state count overflow".to_string())?;
    }

    let top_components = std::iter::once(inputs.parent)
        .chain(inputs.children.iter().map(|child| child.constraint))
        .collect::<Vec<_>>();
    debug_assert_eq!(top_components.len(), num_components);
    let retained_static_coverage = top_components
        .iter()
        .enumerate()
        .map(|(top, component)| {
            expansion.top_leaf_ranges[top].len() <= 1
                || retained_internal_static_boundary_coverage(component)
        })
        .collect::<Vec<_>>();
    let candidate_summary_started = Instant::now();
    let candidate_summary_profiles = top_components
        .iter()
        .map(|component| {
            let started = Instant::now();
            let (ids, stats) =
                crate::compiler::boundary_candidates::boundary_candidate_ids(component, vocab);
            (ids, stats, started.elapsed().as_secs_f64() * 1000.0)
        })
        .collect::<Vec<_>>();
    let mut candidate_tokens_by_component = candidate_summary_profiles
        .iter()
        .map(|(ids, _, _)| ids.clone())
        .collect::<Vec<_>>();
    let candidate_summary_ms = candidate_summary_started.elapsed().as_secs_f64() * 1000.0;
    for top in 0..num_components {
        if expansion.top_leaf_ranges[top].len() > 1 && !retained_static_coverage[top] {
            // Staged legacy fallback: the outer shard is replacing an inner
            // dynamic/uncertified repair, so it must cover that block's leaf-
            // internal crossings as well as its outward interface. W_C is only
            // an outward-interface summary and therefore cannot prune this
            // replacement lane (e.g. token `mg` crosses mid -> grandchild but
            // never exits the block). Widen to the full vocabulary and use the
            // legacy per-leaf ownership predicate below.
            candidate_tokens_by_component[top] = None;
        }
    }
    let mut clear_nested_boundary_components = BitSet::new(num_components);
    for (top, (&covered, range)) in retained_static_coverage
        .iter()
        .zip(&expansion.top_leaf_ranges)
        .enumerate()
    {
        if range.len() > 1 && !covered {
            clear_nested_boundary_components.set(top);
        }
    }

    let empty_output = || WalkStaticLinkOutput {
        published_shards: Vec::new(),
        boundary_tokens_by_start_component: vec![Vec::new(); num_components],
        effective_static_components: effective.clone(),
        all_dynamic: false,
        expected_leaf_tokenizer_offsets: expected_leaf_tokenizer_offsets.clone(),
        expected_total_tokenizer_states: expected_total,
        expected_leaf_terminal_offsets: expansion.leaf_terminal_offsets.clone(),
        expected_total_leaf_terminals: expansion.num_terminals,
        clear_nested_boundary_components: clear_nested_boundary_components.clone(),
    };
    // Nullable starts are checked over non-parent tops only: with a composed
    // parent, top 0 spans several leaves and a leaf-0 skip would wrongly probe
    // parent-block leaves.
    let retain_parent_non_crossing_paths = expansion
        .top_leaf_ranges
        .iter()
        .skip(1)
        .flatten()
        .any(|&leaf_index| leaves[leaf_index].table.embedded_start_nullable());
    // One walk plan per top-level component. A block with retained static
    // internal coverage gets one immediate-component ownership class for all
    // of its leaves. Otherwise keep each leaf in its own class (the legacy
    // conservative predicate), so the new outer shard continues to cover the
    // block's internal crossings before those uncertified inner shards are
    // cleared at install time.
    let total_states = merged.num_states() as usize;
    let mut walk_plans = Vec::with_capacity(num_components);
    let mut leaf_to_immediate = vec![tdwa::scope::ImmediateComponentId(u32::MAX); leaves.len()];
    for top in 0..num_components {
        let mut commit = vec![false; total_states];
        let root_leaf = expansion.top_root_leaves[top] as usize;
        let crossing_owner = tdwa::scope::ImmediateComponentId(root_leaf as u32);
        for &leaf_index in &expansion.top_leaf_ranges[top] {
            leaf_to_immediate[leaf_index] = if retained_static_coverage[top] {
                crossing_owner
            } else {
                tdwa::scope::ImmediateComponentId(leaf_index as u32)
            };
            let start = tokenizer_offsets[leaf_index] as usize;
            let end = start + leaf_state_counts[leaf_index] as usize;
            if end > commit.len() {
                return Err(
                    "nested block commit range lies outside the merged tokenizer".to_string(),
                );
            }
            commit[start..end].fill(true);
        }
        walk_plans.push(BoundaryShardWalkPlan {
            start_component: top,
            crossing_owner,
            commit_states: commit,
            // Immediate terminal ownership is not, by itself, a proof that a
            // token path stayed inside a nested block. The block may complete
            // through zero-width returns and re-enter at another call site
            // before the next visible terminal (e.g. g -> RETURN -> CALL -> m).
            // The component summary is exactly the proper-prefix outward-event
            // witness for that case. For those candidate tokens, keep the
            // lexical non-crossing paths and let the signed parser decide
            // whether an actual outer repair exists. A proved-empty summary
            // can still use the narrower block-ownership predicate; Unknown
            // widens, never prunes.
            retain_non_crossing_paths: (retain_parent_non_crossing_paths && top == 0)
                || (retained_static_coverage[top]
                    && expansion.top_leaf_ranges[top].len() > 1
                    && candidate_tokens_by_component[top]
                        .as_ref()
                        .is_none_or(|tokens| !tokens.is_empty())),
        });
    }
    if leaf_to_immediate
        .iter()
        .any(|owner| owner.0 == u32::MAX)
    {
        return Err("nested boundary ownership did not cover every leaf".to_string());
    }
    let setup_ms = pre_context_setup_ms
        + post_context_setup_started.elapsed().as_secs_f64() * 1000.0;
    let transfer_cache = crate::compiler::boundary_transfer::FragmentTransferCache::new(
        &signed_context,
    )?;
    let transfer_prepare_ms = transfer_cache.prepare_ms;
    struct ProcessedNestedShard {
        start_component: usize,
        candidates: Vec<u32>,
        published: Option<PublishedStaticBoundaryShard>,
        row: Option<WalkStaticLinkShardBreakdown>,
    }
    let process_shard = |shard: BuiltBoundaryShardWalk| -> Result<ProcessedNestedShard, String> {
        let start_component = shard.start_component;
        let candidates = shard.candidate_tokens.iter().copied().collect::<Vec<_>>();
        if !effective.contains(start_component) {
            return Ok(ProcessedNestedShard {
                start_component,
                candidates,
                published: None,
                row: None,
            });
        }
        let (slot_parent, slot_child, slot_terminal) = signed_context
            .links
            .iter()
            .find(|link| {
                expansion
                    .top_leaf_ranges
                    .get(start_component)
                    .is_some_and(|range| range.contains(&(link.child_component as usize)))
            })
            .map(|link| (link.parent_component, link.child_component, link.slot_terminal))
            .unwrap_or((u32::MAX, start_component as u32, u32::MAX));
        let walk_profile = shard.output.profile.clone();
        let (_, summary_stats, summary_ms) = &candidate_summary_profiles[start_component];
        let emitted = boundary_emitted_terminals(
            &shard.output.dwa,
            composed.table.num_terminals as usize,
        );
        let templates_started = Instant::now();
        let library = crate::compiler::boundary_transfer::build_fragment_library_cached(
            &signed_context,
            &transfer_cache,
            &emitted,
            start_component as u32,
        )?;
        let templates_ms = templates_started.elapsed().as_secs_f64() * 1000.0;
        #[cfg(test)]
        trace_boundary_terminal_dwa_words(
            &format!("component{start_component}"),
            &shard.output.dwa,
            composed.table.num_terminals as usize,
        );
        let parser_started = Instant::now();
        let compiled = crate::compiler::boundary_transfer::compile_signed_shard_parser(
            &signed_context,
            &library,
            &shard.output.dwa,
            &shard.output.id_map,
            start_component as u32,
        )?;
        let parser_dwa_ms = parser_started.elapsed().as_secs_f64() * 1000.0;
        let work = WalkBoundaryShardWork {
            start_component: start_component as u32,
            terminal_automaton: TerminalAutomaton::Dwa(shard.output.dwa),
            id_map: shard.output.id_map,
            candidate_tokens: Arc::from(candidates.clone().into_boxed_slice()),
        };
        let publish_started = Instant::now();
        let (published, _publish_profile) =
            crate::compiler::boundary_transfer::publish_signed_shard(
                work,
                compiled,
                &tokenizer_offsets,
                &leaf_state_counts,
            )?;
        let publish_ms = publish_started.elapsed().as_secs_f64() * 1000.0;
        Ok(ProcessedNestedShard {
            start_component,
            candidates,
            published: Some(published),
            row: Some(WalkStaticLinkShardBreakdown {
                start_component,
                parent_component: slot_parent,
                child_component: slot_child,
                slot_terminal,
                walk: walk_profile,
                summary_ms: *summary_ms,
                summary_history_positions: summary_stats.history_positions,
                summary_initial_frontier_pairs: summary_stats.initial_frontier_pairs,
                summary_byte_steps: summary_stats.byte_steps,
                summary_peak_frontier_pairs: summary_stats.peak_frontier_pairs,
                summary_widened_subtrees: summary_stats.widened_subtrees,
                templates_ms,
                parser_dwa_ms,
                publish_ms,
            }),
        })
    };
    let component_pipeline_started = Instant::now();
    let Some((processed, link_profile)) = map_boundary_shard_walks_with(
        &BoundaryShardLinkInputs {
            merged_tokenizer: &merged,
            vocab,
            grammar: &grammar,
            disallowed_follows: &disallowed,
            ignore_terminal: ignores.canonical,
            follow_transparent_ignores: Some(&ignores.scoped),
            terminal_offsets: &expansion.leaf_terminal_offsets,
            leaf_to_immediate: Some(&leaf_to_immediate),
            tokenizer_offsets: &tokenizer_offsets,
            component_state_counts: &leaf_state_counts,
            candidate_tokens_by_component: Some(&candidate_tokens_by_component),
            retain_parent_non_crossing_paths,
            walk_plans: Some(walk_plans),
        },
        &process_shard,
        None,
        None, // Nested-coordinate transport retains the exact on-the-fly path.
    )? else {
        return Ok(empty_output());
    };
    let component_pipeline_wall_ms =
        component_pipeline_started.elapsed().as_secs_f64() * 1000.0;
    let mut processed = processed
        .into_iter()
        .map(|(_, item)| item)
        .collect::<Vec<_>>();
    processed.sort_by_key(|item| item.start_component);
    let mut published = Vec::new();
    let mut tokens_by_component: Vec<Vec<u32>> = vec![Vec::new(); num_components];
    let mut shard_rows = Vec::<WalkStaticLinkShardBreakdown>::new();
    for item in processed {
        tokens_by_component[item.start_component] = item.candidates;
        if let Some(shard) = item.published {
            published.push(shard);
        }
        if let Some(row) = item.row {
            shard_rows.push(row);
        }
    }
    published.sort_by_key(|shard| shard.start_component);
    shard_rows.sort_by_key(|row| row.start_component);
    let templates_ms = shard_rows.iter().map(|row| row.templates_ms).sum::<f64>();
    let parser_dwa_ms = shard_rows.iter().map(|row| row.parser_dwa_ms).sum::<f64>();
    let publish_ms = shard_rows.iter().map(|row| row.publish_ms).sum::<f64>();
    let total_ms = link_total_started.elapsed().as_secs_f64() * 1000.0;
    let accounted_ms = setup_ms
        + link_context_ms
        + transfer_prepare_ms
        + component_pipeline_wall_ms;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][walk_static_link_breakdown] nested=true components={} total_ms={total_ms:.3} setup_ms={setup_ms:.3} link_context_ms={link_context_ms:.3} component_pipeline_wall_ms={component_pipeline_wall_ms:.3} shared_flat_ms={:.3} transfer_prepare_ms={transfer_prepare_ms:.3} templates_work_ms={templates_ms:.3} parser_dwa_work_ms={parser_dwa_ms:.3} publish_work_ms={publish_ms:.3} accounted_ms={accounted_ms:.3} residual_ms={:.3}",
            num_components,
            link_profile.flat_ms,
            total_ms - accounted_ms,
        );
    }
    if compose_profile_enabled() || boundary_analysis_json_enabled() {
        let breakdown = WalkStaticLinkBreakdown {
            total_ms,
            nested: true,
            num_components,
            setup_ms,
            link_context_ms,
            tokenizer_merge_ms,
            grammar_analysis_ms,
            candidate_summary_ms,
            shared_flat_ms: link_profile.flat_ms,
            component_pipeline_wall_ms,
            transfer_prepare_ms,
            templates_ms,
            parser_dwa_ms,
            publish_ms,
            accounted_ms,
            residual_ms: total_ms - accounted_ms,
            per_shard: shard_rows,
        };
        emit_boundary_analysis_json(&breakdown);
    }
    Ok(WalkStaticLinkOutput {
        published_shards: published,
        boundary_tokens_by_start_component: tokens_by_component,
        effective_static_components: effective,
        all_dynamic: false,
        expected_leaf_tokenizer_offsets,
        expected_total_tokenizer_states: expected_total,
        expected_leaf_terminal_offsets: expansion.leaf_terminal_offsets.clone(),
        expected_total_leaf_terminals: expansion.num_terminals,
        clear_nested_boundary_components,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::weighted_u32::terminal_automaton::TerminalAutomaton;
    use crate::compiler::constraint_compose::{
        CompiledSubgrammarInput, SegmentedBoundaryBackend,
        component_ignores_are_globally_erasable, compose_constraints_owned_parent_segmented,
        eliminate_composed_runtime_controls, install_published_static_boundary_shards,
        load_vocab, merged_ignore_terminals, merged_retained_terminal_exprs,
        merged_terminal_display_names, publish_walk_boundary_shard_work, WalkBoundaryShardWork,
    };
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::glr::table::{
        GLRTable, SubgrammarTableInput, compose_subgrammar_tables, control_elimination_budget_exhausted,
        control_elimination_run_count, empty_terminals_in_composed_table,
    };
    use crate::compiler::pipeline::compute_disallowed_follows;
    use crate::grammar::flat::TerminalID;
    use crate::runtime::Constraint;

    /// Process-env guard for strict-reference tests. Saves the prior value and
    /// restores it on drop (callers must hold `crate::TEST_ENV_LOCK`).
    struct EnvVarGuard {
        key: &'static str,
        original: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let original = std::env::var_os(key);
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, original }
        }

        fn unset(key: &'static str) -> Self {
            let original = std::env::var_os(key);
            unsafe {
                std::env::remove_var(key);
            }
            Self { key, original }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.original {
                Some(value) => unsafe {
                    std::env::set_var(self.key, value);
                },
                None => unsafe {
                    std::env::remove_var(self.key);
                },
            }
        }
    }

    struct LowLevelComposed {
        table: ComposedTable,
        tokenizer: Tokenizer,
        tokenizer_offsets: Vec<u32>,
        terminal_names: Vec<String>,
        ignore_canonical: Option<u32>,
    }

    fn terminal_id(constraint: &Constraint, name: &str) -> TerminalID {
        constraint
            .terminal_display_names
            .iter()
            .position(|candidate| candidate == name)
            .unwrap() as u32
    }

    // Existing composition code only: table splice + control elimination +
    // disjoint-union tokenizer. Mirrors the phase1 probe helper.
    fn low_level_compose(
        parent: &Constraint,
        children: &[CompiledSubgrammarInput<'_>],
    ) -> LowLevelComposed {
        let global_ignores = component_ignores_are_globally_erasable(parent, children);
        let table_inputs: Vec<SubgrammarTableInput> = children
            .iter()
            .map(|child| SubgrammarTableInput {
                placeholder_terminal: child.placeholder_terminal,
                additional_placeholder_terminals: &[],
                table: &child.constraint.table,
                ignore_terminal: (!global_ignores)
                    .then_some(child.constraint.ignore_terminal)
                    .flatten(),
                start_nullable: child.constraint.table.embedded_start_nullable(),
            })
            .collect();
        let mut composed = compose_subgrammar_tables(
            &parent.table,
            (!global_ignores).then_some(parent.ignore_terminal).flatten(),
            &table_inputs,
        )
        .expect("compose tables");
        eliminate_composed_runtime_controls(&mut composed).expect("eliminate controls");
        let terminal_names = merged_terminal_display_names(parent, children);
        let mut tokenizer_inputs: Vec<(&Tokenizer, u32)> =
            Vec::with_capacity(children.len() + 1);
        tokenizer_inputs.push((&parent.tokenizer, composed.terminal_offsets[0]));
        for (index, child) in children.iter().enumerate() {
            tokenizer_inputs
                .push((&child.constraint.tokenizer, composed.terminal_offsets[index + 1]));
        }
        let (mut merged, tokenizer_offsets) =
            Tokenizer::disjoint_union_with_terminal_offsets(&tokenizer_inputs);
        if merged.terminal_exprs().is_none() {
            let all: Vec<&Constraint> = std::iter::once(parent)
                .chain(children.iter().map(|child| child.constraint))
                .collect();
            if let Some(exprs) = merged_retained_terminal_exprs(
                &all,
                &composed.terminal_offsets,
                composed.table.num_terminals,
            ) {
                merged.restore_terminal_exprs(Some(exprs)).expect("restore merged exprs");
            }
        }
        let ignores = merged_ignore_terminals(
            parent,
            children,
            &composed.terminal_offsets,
            global_ignores,
        );
        assert_eq!(
            tokenizer_offsets[0], 1,
            "merged state 0 must be the fresh reset fan-out"
        );
        LowLevelComposed {
            table: composed,
            tokenizer: merged,
            tokenizer_offsets,
            terminal_names,
            ignore_canonical: ignores.canonical,
        }
    }

    fn analyzed_grammar(table: &GLRTable, names: &[String]) -> AnalyzedGrammar {
        let augmented_start =
            table.rules.first().expect("composed table has augmented start").lhs;
        AnalyzedGrammar::from_composed_rules(
            table.rules.clone(),
            table.num_terminals,
            names.to_vec(),
            table.nonterminal_display_names.clone(),
            augmented_start,
        )
    }

    #[test]
    fn scoped_witnessed_ti_candidates_are_owner_and_policy_local() {
        let _env_lock = crate::TEST_ENV_LOCK.lock().expect("test env lock poisoned");
        let _strict = EnvVarGuard::set(
            "GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_STRICT_REFERENCE",
            "1",
        );
        let _no_small_skip = EnvVarGuard::set("GLRMASK_P0_TI_SKIP_MAX_TOKENIZER_STATES", "0");

        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"c".to_vec()),
            (3, b"x".to_vec()),
            (4, b"ax".to_vec()),
            (5, b"bx".to_vec()),
            (6, b"xa".to_vec()),
            (7, b"xc".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start doc;
                t A ::= "a";
                t B ::= "b";
                t C ::= "c";
                t X ::= "x";
                nt doc ::= A X | B X | X C;
            "#,
            &vocab,
        )
        .unwrap();
        let grammar = analyzed_grammar(&constraint.table, &constraint.terminal_display_names);
        let disallowed = compute_disallowed_follows(&grammar);
        let a = terminal_id(&constraint, "A");
        let b = terminal_id(&constraint, "B");
        let c = terminal_id(&constraint, "C");
        assert_eq!([a, b, c], [0, 1, 2], "fixture relies on declared terminal order");

        // Deliberately split the otherwise identical C into a second immediate
        // owner.  The TI pre-certificate must never cross that ownership
        // boundary, even though the lexical expression is byte-identical.
        let ownership = Arc::new(
            tdwa::scope::BoundaryOwnership::flat(&[0, 2], grammar.num_terminals)
                .expect("two-owner TI fixture"),
        );
        let initial = tdwa::scope::InitialStateDomain::from_mask(
            constraint.tokenizer.num_states() as usize,
            vec![true; constraint.tokenizer.num_states() as usize],
        )
        .expect("full test initial domain");
        let mut resets = constraint
            .tokenizer
            .deterministic_reset_states()
            .into_iter()
            .collect::<Vec<_>>();
        resets.sort_unstable();
        resets.dedup();
        let scope = tdwa::scope::BoundaryAnalysisScope::new(
            initial,
            resets,
            ownership,
            tdwa::scope::ImmediateComponentId(0),
            true,
            None,
        )
        .expect("TI fixture scope");
        let active = vec![true; grammar.num_terminals as usize];
        let groups = tdwa::l2p::scoped_terminal_interchangeability_candidate_groups(
            &constraint.tokenizer,
            &active,
            &grammar,
            &disallowed,
            constraint.ignore_terminal,
            &scope,
        );
        assert_eq!(groups, vec![vec![a, b]]);

        // Exercise the actual scoped family pipeline.  If the witnessed A/B
        // merge activates, strict-reference mode rebuilds this exact scope with
        // TI suppressed and compares the completed weighted language in
        // original tokenizer-state/token coordinates.
        let flat: Arc<[u32]> =
            Arc::from(tdwa::l1::build_flat_transition_table(&constraint.tokenizer));
        let coloring = crate::compiler::stages::id_map_and_terminal_dwa::types::TerminalColoring::identity(
            grammar.num_terminals as usize,
        );
        let (_artifact, _profile) = tdwa::build_scoped_boundary_id_map_and_terminal_dwa(
            &constraint.tokenizer,
            &vocab,
            &coloring,
            constraint.ignore_terminal,
            &grammar,
            &disallowed,
            flat,
            &scope,
        );
    }

    /// Step-4c boundary parser table: the control-free spliced composed table
    /// (the Phase-1 object — never the dynamic path's control-bearing
    /// recursive table) with every unbound slot terminal emptied. Controls
    /// are a runtime device of the dynamic path; they must not appear in the
    /// static boundary parser's table at all, so a nonzero control count is
    /// a hard error, not a fallback. `unbound_slots` holds
    /// (component_index, component-local terminal) pairs; they are resolved
    /// to composed IDs through the composed terminal offsets.
    fn prepare_spliced_boundary_table(
        name: &str,
        composed: &LowLevelComposed,
        unbound_slots: &[(usize, TerminalID)],
    ) -> (Arc<GLRTable>, f64) {
        assert!(
            composed.table.table.control_terminals.is_empty(),
            "{name}: spliced composed table must be control-free, has {:?}",
            composed.table.table.control_terminals,
        );
        assert!(
            composed.table.control_terminals.is_empty(),
            "{name}: composed control set must be empty",
        );
        let started = Instant::now();
        let mut table = composed.table.table.clone();
        let emptied: Vec<TerminalID> = unbound_slots
            .iter()
            .map(|(component, local)| {
                composed.table.terminal_offsets[*component]
                    .checked_add(*local)
                    .expect("slot terminal offset overflow")
            })
            .collect();
        for &terminal in &emptied {
            assert!(
                (terminal as usize) < table.num_terminals as usize,
                "{name}: unbound slot terminal {terminal} out of range",
            );
        }
        empty_terminals_in_composed_table(&mut table, &emptied);
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "BOUNDARY_TABLE name={name} states={} terms={} controls={} emptied={} prep_ms={ms:.3}",
            table.num_states,
            table.num_terminals,
            table.control_terminals.len(),
            emptied.len(),
        );
        (Arc::new(table), ms)
    }

    /// Pin the installing runtime's scoped tokenizer packing: leaves packed
    /// back-to-back from 0 in link-component order (parent first). The walk
    /// shard's private TSID map is indexed by exactly this coordinate; a
    /// layout change here must update `publish_walk_boundary_shard_work`.
    fn assert_scoped_tokenizer_packing(
        name: &str,
        installed: &Constraint,
        component_state_counts: &[u32],
    ) {
        let layout = installed
            .recursive_parser_layout()
            .expect("recursive layout")
            .expect("recursive layout present");
        let mut expected = Vec::with_capacity(component_state_counts.len());
        let mut next = 0u32;
        for &count in component_state_counts {
            expected.push(next);
            next = next.checked_add(count).expect("scoped total overflow");
        }
        assert_eq!(
            layout.leaf_tokenizer_state_offsets, expected,
            "{name}: runtime leaf packing must match link-component order",
        );
        assert_eq!(
            layout.total_tokenizer_states, next,
            "{name}: runtime scoped total must match link components",
        );
    }

    /// Component-local IDs of `TOOL_ARGS_SLOT_*` terminals in `constraint`
    /// except the bound ones. The sweep parents/children follow the
    /// `TOOL_ARGS_SLOT_{index}` slot convention; the shared `slot`-named
    /// JSON-string terminals (`subgrammar2::"..."`) never match the prefix.
    fn unbound_slot_terminals(
        constraint: &Constraint,
        bound_names: &[&str],
    ) -> Vec<TerminalID> {
        constraint
            .terminal_display_names
            .iter()
            .enumerate()
            .filter(|(_, name)| {
                name.starts_with("TOOL_ARGS_SLOT_") && !bound_names.contains(&name.as_str())
            })
            .map(|(id, _)| id as TerminalID)
            .collect()
    }

    #[test]
    fn toy_two_component_crossing_sets() {
        // Parent `doc ::= PA SUB PB` with child `item ::= CC CD` bound at SUB.
        // Hand-verified crossing sets (trace the merged lexer from each
        // component's states; the composed table's pairwise follows prune the
        // rest):
        // - "ac": PA[parent] then CC[child], pair (PA,CC) allowed -> X_parent.
        // - "db": CD[child] then PB[parent], pair (CD,PB) allowed -> X_child.
        // - "ca": pair (CC,PA) never adjacent -> pruned everywhere.
        // - "bd": pair (PB,CD) never adjacent (PB is last) -> pruned.
        // - "ab", "cd": single-component paths -> internal, not crossing.
        // - single bytes: no reset strictly inside -> never crossing.
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"c".to_vec()),
            (3, b"d".to_vec()),
            (4, b"ac".to_vec()),
            (5, b"db".to_vec()),
            (6, b"ca".to_vec()),
            (7, b"bd".to_vec()),
            (8, b"ab".to_vec()),
            (9, b"cd".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start doc;
                t PA ::= "a";
                t PB ::= "b";
                t SUB ::= @token(999);
                nt doc ::= PA SUB PB;
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start item;
                t CC ::= "c";
                t CD ::= "d";
                nt item ::= CC CD;
            "#,
            &vocab,
        )
        .unwrap();
        let children = [CompiledSubgrammarInput {
            placeholder_terminal: terminal_id(&parent, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &child,
        }];
        let composed = low_level_compose(&parent, &children);
        let grammar = analyzed_grammar(&composed.table.table, &composed.terminal_names);
        let disallowed = compute_disallowed_follows(&grammar);
        let flat: Arc<[u32]> =
            Arc::from(tdwa::l1::build_flat_transition_table(&composed.tokenizer));
        let ownership = Arc::new(
            tdwa::scope::BoundaryOwnership::flat(
                &composed.table.terminal_offsets,
                grammar.num_terminals,
            )
            .expect("toy terminal ownership"),
        );
        if std::env::var_os("GLRMASK_DEBUG_TOY_SCOPE").is_some() {
            for raw in composed.tokenizer_offsets[0]
                ..composed.tokenizer_offsets[0] + parent.tokenizer.num_states()
            {
                eprintln!(
                    "TOY_SCOPE raw={} matched={:?} future={:?}",
                    raw,
                    composed
                        .tokenizer
                        .matched_terminals_iter(raw)
                        .collect::<Vec<_>>(),
                    composed
                        .tokenizer
                        .possible_future_terminals_iter(raw)
                        .collect::<Vec<_>>(),
                );
                for token_id in [2u32, 3, 4, 5, 9] {
                    let bytes = vocab.get(token_id).unwrap();
                    let exec = composed.tokenizer.execute_from_state(bytes, raw);
                    eprintln!(
                        "TOY_SCOPE raw={} token={} bytes={:?} matches={:?} end={:?}",
                        raw,
                        token_id,
                        bytes,
                        exec.matches,
                        exec.end_state,
                    );
                }
            }
        }
        for (index, (num_states, expected)) in
            [(parent.tokenizer.num_states(), vec![4u32]), (child.tokenizer.num_states(), vec![5u32])]
                .into_iter()
                .enumerate()
        {
            let commit = commit_states_for_component(
                &composed.tokenizer_offsets,
                num_states,
                index,
                composed.tokenizer.num_states() as usize,
            );
            let scope = tdwa::scope::BoundaryAnalysisScope::new(
                tdwa::scope::InitialStateDomain::from_mask(
                    composed.tokenizer.num_states() as usize,
                    commit.clone(),
                )
                .expect("toy initial-state domain"),
                composed
                    .tokenizer
                    .deterministic_reset_states()
                    .into_iter()
                    .collect(),
                Arc::clone(&ownership),
                tdwa::scope::ImmediateComponentId(index as u32),
                true,
                None,
            )
            .expect("toy boundary scope");
            let output = build_boundary_terminal_dwa(&BoundaryWalkInputs {
                merged_tokenizer: &composed.tokenizer,
                vocab: &vocab,
                grammar: &grammar,
                disallowed_follows: &disallowed,
                ignore_terminal: composed.ignore_canonical,
                follow_transparent_ignores: None,
                scope: &scope,
                flat_trans: Some(&flat),
                retain_non_crossing_paths: false,
            })
            .expect("toy shard walk must produce a DWA");
            if std::env::var_os("GLRMASK_DEBUG_TOY_SCOPE").is_some() {
                eprintln!(
                    "TOY_SCOPE shard={} idmap_o2i={:?} idmap_classes={:?} reps={:?}",
                    index,
                    output.id_map.tokenizer_states.original_to_internal,
                    output.id_map.tokenizer_states.internal_to_originals,
                    output.id_map.tokenizer_states.representative_original_ids,
                );
            }
            assert!(output.dwa.is_acyclic(), "toy shard {index} DWA must be acyclic");
            let tokens = boundary_accepted_tokens(&output.dwa, &output.id_map);
            assert_eq!(
                tokens.into_iter().collect::<Vec<_>>(),
                expected,
                "toy shard {index} crossing set",
            );
        }
    }

    fn restore_component(component: &mut Constraint, label: &str) {
        if component.tokenizer.terminal_exprs().is_none()
            && let Some(exprs) = component.retained_terminal_exprs().map(|exprs| exprs.to_vec())
        {
            Arc::make_mut(&mut component.tokenizer).restore_terminal_exprs(Some(exprs))
                .expect("restore component terminal exprs");
        }
        let inline_rules = component.table.rules.len();
        let retained = component.retained_table_rules().expect("decode retained rules").len();
        if inline_rules != retained {
            component.table.rules =
                component.retained_table_rules().expect("decode retained rules").to_vec();
        }
        component
            .materialize_composition_metadata_for_compilation()
            .expect("materialize composition metadata");
        eprintln!("BOUNDARY_WALK restore {label} inline_rules={inline_rules} retained_rules={retained}");
    }

    struct Selected10Outer {
        vocab: Vocab,
        core: Constraint,
        dispatch: Constraint,
        composed: LowLevelComposed,
        grammar: AnalyzedGrammar,
        disallowed: BTreeMap<u32, BitSet>,
    }

    fn load_selected10_outer() -> Selected10Outer {
        let root = phase1_root();
        let vocab_path = phase1_vocab_path();
        let vocab = load_vocab(&vocab_path);
        assert_eq!(
            vocab.entries_map().len(),
            128_256,
            "selected10 expects the CFA Llama-3.1 vocabulary",
        );
        let mut core =
            Constraint::load(&std::fs::read(root.join("core.bin")).expect("read core.bin"))
                .expect("load core");
        let dispatch_name = std::env::var("PHASE1_DISPATCH")
            .unwrap_or_else(|_| "dispatch-literal.bin".to_string());
        let mut dispatch =
            Constraint::load(&std::fs::read(root.join(&dispatch_name)).expect("read dispatch"))
                .expect("load dispatch");
        restore_component(&mut core, "core");
        restore_component(&mut dispatch, "dispatch");
        let children = [CompiledSubgrammarInput {
            placeholder_terminal: terminal_id(&core, "PROGRAMMATIC_TOOL_SUFFIX"),
            additional_placeholder_terminals: &[],
            constraint: &dispatch,
        }];
        let composed = low_level_compose(&core, &children);
        let grammar = analyzed_grammar(&composed.table.table, &composed.terminal_names);
        let disallowed = compute_disallowed_follows(&grammar);
        Selected10Outer { vocab, core, dispatch, composed, grammar, disallowed }
    }

    fn require_release_selected10_gate(name: &str) {
        assert!(
            !cfg!(debug_assertions),
            "{name} is a selected10 integration/performance gate and must be run with an optimized test binary; rerun with `cargo test --release ...`"
        );
    }

    /// Production-path selected10 crossing flow shared by the plain gate and the
    /// two-stage strict gate below: one shared equivalence, one standard walk
    /// per component, NWA-level crossing filter. Asserts the 143-token dispatch
    /// crossing set (MINBOUND oracle dump) and the 26-state true-minimal
    /// crossing DWA, plus the empty core shard.
    fn run_selected10_boundary_walk_crossing() {
        require_release_selected10_gate("selected10_boundary_walk_crossing");
        let fixture = load_selected10_outer();
        let vocab = &fixture.vocab;
        let core = &fixture.core;
        let dispatch = &fixture.dispatch;
        let composed = &fixture.composed;
        let grammar = &fixture.grammar;
        let disallowed = &fixture.disallowed;
        let flat: Arc<[u32]> =
            Arc::from(tdwa::l1::build_flat_transition_table(&composed.tokenizer));
        let ownership = Arc::new(
            tdwa::scope::BoundaryOwnership::flat(
                &composed.table.terminal_offsets,
                grammar.num_terminals,
            )
            .expect("selected10 terminal ownership"),
        );
        // Frozen output of the independent Phase-1 MINBOUND B-A oracle for
        // this exact selected10 fixture. The original artifact is preserved
        // in the 2026-09-18 linker compaction with SHA-256
        // 689820493b69f3a603c1c38235a74b4b91aa2a6449f22cddbdbc5a5e3d60bf6d.
        // Keep the oracle in-repo so this regression never depends on
        // historical /tmp state or a machine-local PHASE1_DUMP_DIR.
        let oracle_text = include_str!("testdata/selected10_minbound_tokens.txt");
        let oracle: BTreeSet<u32> =
            oracle_text.split_whitespace().map(|value| value.parse::<u32>().unwrap()).collect();
        assert_eq!(oracle.len(), 143, "oracle dump must hold the 143 MINBOUND tokens");

        let component_states = [core.tokenizer.num_states(), dispatch.tokenizer.num_states()];
        for (index, num_states) in component_states.into_iter().enumerate() {
            let commit = commit_states_for_component(
                &composed.tokenizer_offsets,
                num_states,
                index,
                composed.tokenizer.num_states() as usize,
            );
            let scope = tdwa::scope::BoundaryAnalysisScope::new(
                tdwa::scope::InitialStateDomain::from_mask(
                    composed.tokenizer.num_states() as usize,
                    commit.clone(),
                )
                .expect("selected10 initial-state domain"),
                composed
                    .tokenizer
                    .deterministic_reset_states()
                    .into_iter()
                    .collect(),
                Arc::clone(&ownership),
                tdwa::scope::ImmediateComponentId(index as u32),
                true,
                None,
            )
            .expect("selected10 boundary scope");
            let output = build_boundary_terminal_dwa(&BoundaryWalkInputs {
                merged_tokenizer: &composed.tokenizer,
                vocab,
                grammar,
                disallowed_follows: disallowed,
                ignore_terminal: composed.ignore_canonical,
                follow_transparent_ignores: None,
                scope: &scope,
                flat_trans: Some(&flat),
                retain_non_crossing_paths: false,
            })
            .expect("selected10 shard walk must produce a DWA");
            assert!(output.dwa.is_acyclic(), "shard {index} DWA must be acyclic");
            let tokens = boundary_accepted_tokens(&output.dwa, &output.id_map);
            let emitted =
                boundary_emitted_terminals(&output.dwa, grammar.num_terminals as usize);
            let emitted_count = emitted.iter().filter(|slot| **slot).count();
            let profile = &output.profile;
            eprintln!(
                "BOUNDARY_WALK shard={index} states={} trans={} tokens={} emitted_terms={} \
                 setup_ms={:.3} walk_ms={:.3} id_map_ms={:.3} dwa_ms={:.3} det_ms={:.3} min_ms={:.3} compact_ms={:.3}",
                output.dwa.num_states(),
                output.dwa.num_transitions(),
                tokens.len(),
                emitted_count,
                profile.setup_ms,
                profile.walk_ms,
                profile.id_map_ms,
                profile.terminal_dwa_ms,
                profile.determinize_ms,
                profile.minimize_ms,
                profile.compact_ms,
            );
            if index == 0 {
                assert!(tokens.is_empty(), "core shard must be empty, got {}", tokens.len());
                assert_eq!(output.dwa.num_states(), 1, "empty core shard DWA shape");
            } else {
                if tokens != oracle {
                    let missing = oracle.difference(&tokens).copied().collect::<Vec<_>>();
                    let extra = tokens.difference(&oracle).copied().collect::<Vec<_>>();
                    eprintln!(
                        "BOUNDARY_WALK oracle_diff missing={missing:?} extra={extra:?}"
                    );
                }
                assert_eq!(tokens.len(), 143, "dispatch shard token count");
                assert_eq!(tokens, oracle, "dispatch shard must match the oracle set");
                assert_eq!(
                    output.dwa.num_states(),
                    26,
                    "dispatch crossing DWA must be the true-minimal 26-state form"
                );
            }
        }
    }

    /// Production-path selected10 crossing gate (no strict reference).
    #[test]
    #[ignore]
    fn selected10_boundary_walk_crossing() {
        run_selected10_boundary_walk_crossing();
    }

    /// Exact weighted-language gate for scoped vocabulary partitioning. The
    /// full-vocabulary all-L2P build is the conservative reference. Reconcile
    /// both private TSID/token quotients into one common coordinate before
    /// asking the acyclic weighted-DWA equivalence checker for a distinguishing
    /// terminal word.
    #[test]
    #[ignore]
    fn selected10_partitioned_scoped_matches_full_weighted_relation() {
        require_release_selected10_gate(
            "selected10_partitioned_scoped_matches_full_weighted_relation",
        );
        let _env_lock = crate::TEST_ENV_LOCK.lock().unwrap();
        let fixture = load_selected10_outer();
        let component_state_counts = [
            fixture.core.tokenizer.num_states(),
            fixture.dispatch.tokenizer.num_states(),
        ];
        let build = || {
            build_boundary_shard_walks(&BoundaryShardLinkInputs {
                merged_tokenizer: &fixture.composed.tokenizer,
                vocab: &fixture.vocab,
                grammar: &fixture.grammar,
                disallowed_follows: &fixture.disallowed,
                ignore_terminal: fixture.composed.ignore_canonical,
                follow_transparent_ignores: None,
                terminal_offsets: &fixture.composed.table.terminal_offsets,
                leaf_to_immediate: None,
                tokenizer_offsets: &fixture.composed.tokenizer_offsets,
                component_state_counts: &component_state_counts,
                candidate_tokens_by_component: None,
                retain_parent_non_crossing_paths: false,
                walk_plans: None,
            })
            .expect("selected10 scoped boundary build")
            .0
        };

        let full = {
            let _full = EnvVarGuard::set("GLRMASK_DEBUG_SCOPED_FORCE_FULL_VOCAB", "1");
            build()
        };
        let partitioned = {
            let _partitioned = EnvVarGuard::unset("GLRMASK_DEBUG_SCOPED_FORCE_FULL_VOCAB");
            build()
        };
        assert_eq!(full.len(), partitioned.len(), "nonempty shard set differs");
        for (full, partitioned) in full.into_iter().zip(partitioned) {
            let start_component = full.start_component;
            assert_eq!(
                start_component, partitioned.start_component,
                "shard publication order differs",
            );
            let full = crate::compiler::stages::mapped_artifact::MappedArtifact::new(
                full.output.dwa,
                full.output.id_map,
            );
            let partitioned = crate::compiler::stages::mapped_artifact::MappedArtifact::new(
                partitioned.output.dwa,
                partitioned.output.id_map,
            );
            let reconciled: crate::compiler::stages::mapped_artifact::MappedArtifact<(DWA, DWA)> =
                (full, partitioned).into();
            let ((full, partitioned), _) = reconciled.into_parts();
            match crate::automata::weighted_u32::equivalence::find_difference(
                &full,
                &partitioned,
            ) {
                Ok(None) => {}
                Ok(Some(word)) => panic!(
                    "partitioned scoped shard {start_component} changed weighted language at terminal word {word:?}",
                ),
                Err(error) => panic!("weighted equivalence checker failed: {error:?}"),
            }
        }
    }

    /// Durable two-stage strict acceptance gate for boundary shards. Arms the
    /// factor strict reference for BOTH the shard label (`boundary_shard`,
    /// stage-2 completed-artifact compare in each walk) and the
    /// shared-equivalence label (`boundary_shard_test`, stage-1
    /// factored-vs-exact class assert), then runs the production crossing
    /// flow. The selector assertions below prove both stages are ENABLED, not
    /// that they execute (they share the test's hardcoded label names, so a
    /// construction-side label rename would still pass them while disabling a
    /// stage). Execution is proven by the stage-1 and stage-2 `differs=false`
    /// markers, which the gate runner script checks in the captured output.
    #[test]
    #[ignore]
    fn selected10_boundary_shard_two_stage_strict_reference() {
        let _env_lock = crate::TEST_ENV_LOCK.lock().unwrap();
        let _strict = EnvVarGuard::set(
            "GLRMASK_L2P_FIRST_BYTE_VOCAB_FACTOR_STRICT_REFERENCE",
            "boundary_shard,boundary_shard_test",
        );
        // The stage-2 completed-artifact marker only prints when L2P timing
        // profiling is on (`l2p_timing_profile_enabled`); arm it in-process
        // so the gate markers appear without relying on ambient env.
        let _timing = EnvVarGuard::set("GLRMASK_PROFILE_L2P_TIMING", "1");
        assert!(
            tdwa::l2p::l2p_first_byte_vocab_factor_strict_reference_enabled_for_partition(
                "boundary_shard",
            ),
            "stage-2 shard strict reference must be armed for boundary_shard",
        );
        assert!(
            tdwa::l2p::l2p_first_byte_vocab_factor_strict_reference_enabled_for_partition(
                "boundary_shard_test",
            ),
            "stage-1 shared-equivalence strict reference must be armed for boundary_shard_test",
        );
        run_selected10_boundary_walk_crossing();
    }

    /// Convergence-bound regression gate (Phase 2 step 4): the selected10
    /// outer recursive table's control elimination provably diverges (cyclic
    /// reduce graph defeats the DFS visiting set), so it must decline via
    /// the convergence budget quickly instead of stalling the link.
    #[test]
    #[ignore]
    fn recursive_table_convergence_bound_fails_fast() {
        let fixture = load_selected10_outer();
        let dyn_children = [CompiledSubgrammarInput {
            placeholder_terminal: terminal_id(&fixture.core, "PROGRAMMATIC_TOOL_SUFFIX"),
            additional_placeholder_terminals: &[],
            constraint: &fixture.dispatch,
        }];
        let dynamic = compose_constraints_owned_parent_segmented(
            fixture.core.clone(),
            &dyn_children,
            &fixture.vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("dynamic compose")
        .constraint;
        let started = Instant::now();
        let error = dynamic
            .recursive_control_eliminated_parser_table()
            .expect_err("outer recursive table must decline via the convergence budget");
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        eprintln!("BOUNDARY_INSTALL convergence_decline_ms={elapsed_ms:.3} error={error}");
        assert!(
            control_elimination_budget_exhausted(&error),
            "decline must be budget exhaustion, got: {error}"
        );
        assert!(elapsed_ms < 15_000.0, "decline must fail fast, took {elapsed_ms:.0} ms");
    }

    fn phase1_root() -> std::path::PathBuf {
        use std::path::Path;

        let root = std::env::var("PHASE1_DIR").unwrap_or_else(|_| {
            "/Users/isaacbreen/Projects2/temp/2026-09/glrmask-selected10-cache-v29".to_string()
        });
        Path::new(&root).to_path_buf()
    }

    /// Phase-1 cache vocab path, reusing the existing `load_selected10_outer`
    /// pattern (PHASE1_VOCAB override, else PHASE1_DIR/vocab_dump.bin).
    /// No new hardcoded fallbacks beyond the established default root.
    fn phase1_vocab_path() -> String {
        std::env::var("PHASE1_VOCAB")
            .unwrap_or_else(|_| phase1_root().join("vocab_dump.bin").to_string_lossy().into_owned())
    }

    fn phase1_vocab() -> Vocab {
        load_vocab(&phase1_vocab_path())
    }

    /// Setup manifest for a prepared install pair: strict structured JSON
    /// emitted AFTER all pins pass. Never written on failure (no vacuous
    /// pass). Schema `install-prepare/2`: schema/test/shards (structured
    /// start_component + candidate_tokens arrays), component_state_counts,
    /// scenario byte strings, scenario_lengths. Parsed + validated by the
    /// window test (never treated as an opaque string).
    fn write_install_prepare_manifest(
        dir: &std::path::Path,
        test_name: &str,
        shards: &[(u32, usize)],
        component_state_counts: &[u32],
        vocab: &Vocab,
    ) {
        assert!(!shards.is_empty(), "{test_name}: refusing empty-shard manifest");
        let manifest = serde_json::json!({
            "schema": "install-prepare/2",
            "test": test_name,
            "shards": shards.iter().map(|(component, candidates)| {
                serde_json::json!({"start_component": component, "candidate_tokens": candidates})
            }).collect::<Vec<_>>(),
            "component_state_counts": component_state_counts,
            "scenarios": INSTALL_DIFFERENTIAL_PREFIXES.iter().map(|prefix| {
                String::from_utf8_lossy(prefix).into_owned()
            }).collect::<Vec<_>>(),
            "scenario_lengths": INSTALL_DIFFERENTIAL_PREFIXES.iter().map(|prefix| {
                prefix.len() + 1
            }).collect::<Vec<_>>(),
            "vocab_entries": vocab.entries_map().len(),
        });
        std::fs::write(
            dir.join("MANIFEST.json"),
            serde_json::to_string_pretty(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");
        std::fs::write(dir.join("PREPARE_OK"), b"ok").expect("write marker");
    }

    /// Save + reload round-trip for an installed pair: persists static and
    /// dynamic with the caller's vocab, reloads both, reasserts the
    /// backend/packing pins on the RELOADED static, compares fresh-vs-reload
    /// initial full masks per backend, and returns the reloaded pair.
    /// Stops (no vacuous pass) if serialization loses any pin.
    fn save_reload_install_pair(
        test_name: &str,
        dir: &std::path::Path,
        dynamic: &Constraint,
        static_comp: &Constraint,
        component_state_counts: &[u32],
        packing_name: &str,
        vocab: &Vocab,
    ) -> (Constraint, Constraint) {
        let raw_static = static_comp.save();
        let raw_dynamic = dynamic.save();
        assert!(!raw_static.is_empty(), "{test_name}: refusing empty static.bin");
        assert!(!raw_dynamic.is_empty(), "{test_name}: refusing empty dynamic.bin");
        std::fs::write(dir.join("static.bin"), &raw_static).expect("write static.bin");
        std::fs::write(dir.join("dynamic.bin"), &raw_dynamic).expect("write dynamic.bin");
        // Same-vocabulary reload with the caller's fixture vocab.
        let re_static = Constraint::load_with_vocab(&raw_static, vocab).expect("reload static");
        let re_dynamic = Constraint::load_with_vocab(&raw_dynamic, vocab).expect("reload dynamic");
        // Backend pins on the RELOADED static (serialization must not lose them).
        assert!(
            re_static.uses_compact_segmented_parser_runtime(),
            "{test_name}: reloaded static must stay on the compact runtime",
        );
        let mut re_shards = 0usize;
        for (index, component) in re_static
            .static_dynamic_overlay
            .as_ref()
            .expect("reloaded overlay")
            .segmented_parser_components
            .iter()
            .enumerate()
        {
            if let Some(shard) = component.boundary.as_ref() {
                re_shards += 1;
                assert!(
                    matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ),
                    "{test_name}: reloaded component {index} must be a StaticParser shard",
                );
            }
        }
        assert!(re_shards > 0, "{test_name}: reloaded static must carry shards");
        assert_scoped_tokenizer_packing(packing_name, &re_static, component_state_counts);
        // Fresh-vs-reload initial full masks per backend (exact vectors).
        assert_eq!(
            static_comp.start().mask(),
            re_static.start().mask(),
            "{test_name}: static initial mask must survive reload",
        );
        assert_eq!(
            dynamic.start().mask(),
            re_dynamic.start().mask(),
            "{test_name}: dynamic initial mask must survive reload",
        );
        (re_dynamic, re_static)
    }

    /// Optional prepare-output dir for install-gate tests: when the named env
    /// var is set, its value is the output directory (created fresh; refuses
    /// non-empty dirs so stale bins can never pass).
    fn install_prepare_dir(env_key: &str) -> Option<std::path::PathBuf> {
        std::env::var_os(env_key).map(|dir| {
            let dir = std::path::PathBuf::from(dir);
            if dir.exists() {
                let count = std::fs::read_dir(&dir).expect("read prepare dir").count();
                assert_eq!(count, 0, "{env_key}: refusing non-empty prepare dir");
            } else {
                std::fs::create_dir_all(&dir).expect("create prepare dir");
            }
            dir
        })
    }

    /// Step-4 install gate: build walk shards through the link orchestration,
    /// publish against the spliced control-free boundary table (unbound
    /// slots emptied), install by replacing the dynamic composition's
    /// dynamic shards, and require mask-for-mask equality with the dynamic
    /// backend over a byte-prefix corpus. No fallback: a nonzero control
    /// count or a publish failure is a hard gate failure.
    #[test]
    #[ignore]
    fn selected10_walk_shard_install_matches_dynamic() {
        require_release_selected10_gate("selected10_walk_shard_install_matches_dynamic");
        let fixture = load_selected10_outer();
        let dyn_children = [CompiledSubgrammarInput {
            placeholder_terminal: terminal_id(&fixture.core, "PROGRAMMATIC_TOOL_SUFFIX"),
            additional_placeholder_terminals: &[],
            constraint: &fixture.dispatch,
        }];
        let compose_started = Instant::now();
        let dynamic = compose_constraints_owned_parent_segmented(
            fixture.core.clone(),
            &dyn_children,
            &fixture.vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("dynamic compose")
        .constraint;
        eprintln!(
            "BOUNDARY_INSTALL dynamic_compose_ms={:.3}",
            compose_started.elapsed().as_secs_f64() * 1000.0,
        );

        let component_state_counts =
            [fixture.core.tokenizer.num_states(), fixture.dispatch.tokenizer.num_states()];
        let link_started = Instant::now();
        let (built, link_profile) = build_boundary_shard_walks(&BoundaryShardLinkInputs {
            merged_tokenizer: &fixture.composed.tokenizer,
            vocab: &fixture.vocab,
            grammar: &fixture.grammar,
            disallowed_follows: &fixture.disallowed,
            ignore_terminal: fixture.composed.ignore_canonical,
            follow_transparent_ignores: None,
            terminal_offsets: &fixture.composed.table.terminal_offsets,
            leaf_to_immediate: None,
            tokenizer_offsets: &fixture.composed.tokenizer_offsets,
            component_state_counts: &component_state_counts,
            candidate_tokens_by_component: None,
            // Core and dispatch schemas are never start-nullable.
            retain_parent_non_crossing_paths: false,
            walk_plans: None,
        })
        .expect("walk shards");
        eprintln!(
            "BOUNDARY_INSTALL walks shared_ms={:.3} flat_ms={:.3} shards={:?} link_ms={:.3}",
            link_profile.shared_wall_ms,
            link_profile.flat_ms,
            built
                .iter()
                .map(|shard| (
                    shard.start_component,
                    shard.candidate_tokens.len(),
                    shard.output.profile.walk_ms,
                ))
                .collect::<Vec<_>>(),
            link_started.elapsed().as_secs_f64() * 1000.0,
        );
        assert_eq!(built.len(), 1, "only the dispatch shard is nonempty");
        assert_eq!(built[0].start_component, 1);
        assert_eq!(built[0].candidate_tokens.len(), 143);

        // Outer unbound slots: dispatch's TOOL_ARGS_SLOT_* (component 1, none
        // bound — the 10-way segmented dispatch link is unavailable) plus any
        // unbound parent slots (core binds its only slot; expect none).
        let mut unbound: Vec<(usize, TerminalID)> = unbound_slot_terminals(&fixture.core, &[])
            .into_iter()
            .map(|local| (0usize, local))
            .collect();
        unbound.extend(
            unbound_slot_terminals(&fixture.dispatch, &[]).into_iter().map(|local| (1usize, local)),
        );
        let (boundary_table, table_ms) =
            prepare_spliced_boundary_table("outer", &fixture.composed, &unbound);
        eprintln!("BOUNDARY_INSTALL table_ms={table_ms:.3}");
        let mut published = Vec::with_capacity(built.len());
        for shard in built {
            let work = WalkBoundaryShardWork {
                start_component: shard.start_component as u32,
                terminal_automaton: TerminalAutomaton::Dwa(shard.output.dwa),
                id_map: shard.output.id_map,
                candidate_tokens: shard.candidate_tokens.into_iter().collect::<Vec<_>>().into(),
            };
            let (shard, profile) = publish_walk_boundary_shard_work(
                work,
                &boundary_table,
                &fixture.composed.tokenizer_offsets,
                &component_state_counts,
            )
            .expect("publish walk shard");
            eprintln!(
                "BOUNDARY_INSTALL shard={} parser_states={} parser_trans={} candidates={} uses_composed={} terms={} templates_ms={:.3} materialize_ms={:.3} normalize_ms={:.3}",
                shard.start_component,
                profile.parser_states,
                profile.parser_trans,
                shard.candidate_tokens.len(),
                shard.boundary.uses_composed_tsid_coordinate,
                profile.terms,
                profile.templates_ms,
                profile.materialize_ms,
                profile.normalize_ms,
            );
            published.push(shard);
        }

        let mut static_comp = dynamic.clone();
        install_published_static_boundary_shards(
            static_comp.static_dynamic_overlay.as_mut().expect("overlay"),
            published,
        )
        .expect("install");
        assert!(
            static_comp.uses_compact_segmented_parser_runtime(),
            "installed composition must stay on the compact runtime",
        );
        assert_scoped_tokenizer_packing("outer", &static_comp, &component_state_counts);
        // Optional prepare: save/reload round-trip + pins, then return BEFORE
        // the in-process differential (the prepared-pair differential test
        // owns windowed mask comparison). Default path unchanged.
        if let Some(dir) = install_prepare_dir("BOUNDARY_INSTALL_PREPARE_OUT_DIR") {
            let shard_ids: Vec<(u32, usize)> = static_comp
                .static_dynamic_overlay
                .as_ref()
                .expect("overlay")
                .segmented_parser_components
                .iter()
                .filter_map(|component| {
                    component.boundary.as_ref().map(|shard| {
                        (
                            shard.start_component,
                            shard.candidate_tokens.as_ref().map(|tokens| tokens.len()).unwrap_or(0),
                        )
                    })
                })
                .collect();
            save_reload_install_pair(
                "selected10_walk_shard_install_matches_dynamic",
                &dir,
                &dynamic,
                &static_comp,
                &component_state_counts,
                "outer",
                &fixture.vocab,
            );
            write_install_prepare_manifest(
                &dir,
                "selected10_walk_shard_install_matches_dynamic",
                &shard_ids,
                &component_state_counts,
                &fixture.vocab,
            );
            return;
        }
        run_install_differential(&dynamic, &mut static_comp, false);
    }

    /// Byte-prefix corpus for the install differential. The expected coverage
    /// is 248 mask positions (initial + one per committed byte across all 12
    /// scenarios); the windowed prepared-pair test derives its actual count
    /// from this list and reports any deviation instead of forcing 248.
    const INSTALL_DIFFERENTIAL_PREFIXES: [&[u8]; 12] = [
        b"",
        b"const x = tools",
        b"const x = tools.tool_0(",
        b"const x = tools.tool_3({",
        b"const x = tools.tool_3({\"p\": ",
        b"const x = tools.tool_9({\"outer\": {\"inner\": [1, ",
        b"function f(a, b) { return ",
        b"for (let i = 0; i < ",
        b"const s = \"hello",
        b"const x = { a: ",
        b"const x = tools.",
        b"tools",
    ];

    /// Byte-prefix mask differential between a dynamic composition and its
    /// walk-shard-installed twin. `fallback` labels runs where convergence
    /// decline left the dynamic shards in place (vacuous by construction).
    fn run_install_differential(
        dynamic: &Constraint,
        static_comp: &mut Constraint,
        fallback: bool,
    ) {

        let prefixes: Vec<&[u8]> = INSTALL_DIFFERENTIAL_PREFIXES.to_vec();
        let mut positions = 0usize;
        let mut checksum: u64 = 0xcbf29ce484222325;
        for (scenario, prefix) in prefixes.iter().enumerate() {
            let scenario_started = Instant::now();
            let mut st_dyn = dynamic.start();
            let mut st_static = static_comp.start();
            let mut check = |st_dyn: &mut crate::runtime::ConstraintState<'_>,
                             st_static: &mut crate::runtime::ConstraintState<'_>,
                             scenario: usize,
                             consumed: usize| {
                let mask_dyn = st_dyn.mask();
                let mask_static = st_static.mask();
                assert_eq!(
                    mask_static,
                    mask_dyn,
                    "mask mismatch scenario={scenario} consumed={consumed}",
                );
                for (index, &word) in mask_dyn.iter().enumerate() {
                    checksum ^= (word as u64).wrapping_add(index as u64);
                    checksum = checksum.wrapping_mul(0x100000001b3);
                }
                positions += 1;
            };
            check(&mut st_dyn, &mut st_static, scenario, 0);
            for (consumed, &byte) in prefix.iter().enumerate() {
                let r_dyn = st_dyn.commit_bytes(&[byte]);
                let r_static = st_static.commit_bytes(&[byte]);
                assert_eq!(
                    r_dyn.is_ok(),
                    r_static.is_ok(),
                    "commit divergence scenario={scenario} consumed={consumed}",
                );
                if r_dyn.is_err() {
                    break;
                }
                check(&mut st_dyn, &mut st_static, scenario, consumed + 1);
            }
            eprintln!(
                "BOUNDARY_INSTALL scenario={scenario} len={} ms={:.3}",
                prefix.len(),
                scenario_started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        eprintln!(
            "BOUNDARY_INSTALL fallback={fallback} positions={positions} checksum={checksum:016x}"
        );
        assert!(positions > 100, "corpus must cover real positions");
    }

    /// Windowed differential over a PREPARED pair: reloads both backends from
    /// `GLRMASK_INSTALL_DIFF_PREPARE_DIR`, validates the structured prepare
    /// manifest (schema `install-prepare/2`, known test name, PREPARE_OK,
    /// 12 scenario identities, nonempty shards exactly matching the reloaded
    /// component IDs/candidate counts, packing against manifest counts),
    /// reasserts pins, then in ONE pass commits through `start+count-1`,
    /// comparing complete masks and collecting hashes at wanted positions
    /// (initial position 0 plus each committed byte — exactly the positions
    /// `run_install_differential` checks). Commit behavior is STRICTER than
    /// the original: the old `fallback` arg labels convergence-decline runs
    /// (where both-error breaks the byte loop); here every commit in the
    /// validated fixture must succeed. One strict JSON object plus the
    /// `INSTALL_WINDOW_OK` marker are emitted only AFTER full-vector
    /// equality. Selected by env: prepare dir (required), scenario index
    /// (required), window start/count (required, 1..=8). No new corpus.
    /// The global 248 total is asserted from the shared prefix list.
    #[test]
    #[ignore]
    fn prepared_install_pair_windowed_differential() {
        let _env_lock = crate::TEST_ENV_LOCK.lock().unwrap();
        let dir = std::env::var("GLRMASK_INSTALL_DIFF_PREPARE_DIR")
            .expect("GLRMASK_INSTALL_DIFF_PREPARE_DIR must name a prepared pair dir");
        let dir = std::path::PathBuf::from(dir);
        let scenario: usize = std::env::var("GLRMASK_INSTALL_DIFF_SCENARIO")
            .expect("GLRMASK_INSTALL_DIFF_SCENARIO required")
            .parse()
            .expect("scenario integer");
        let start: usize = std::env::var("GLRMASK_INSTALL_DIFF_START")
            .expect("GLRMASK_INSTALL_DIFF_START required")
            .parse()
            .expect("start integer");
        let count: usize = std::env::var("GLRMASK_INSTALL_DIFF_COUNT")
            .expect("GLRMASK_INSTALL_DIFF_COUNT required")
            .parse()
            .expect("count integer");
        assert!((1..=8).contains(&count), "window admits 1..=8 comparisons");
        let prefix = INSTALL_DIFFERENTIAL_PREFIXES
            .get(scenario)
            .expect("scenario indexes the original 12-prefix corpus");
        assert_eq!(
            INSTALL_DIFFERENTIAL_PREFIXES
                .iter()
                .map(|prefix| prefix.len() + 1)
                .sum::<usize>(),
            248,
            "global corpus must total 248 positions",
        );
        let end = start.checked_add(count).expect("window end overflow");
        assert!(
            end <= prefix.len() + 1,
            "window [{start}..{end}) out of bounds (scenario len {})",
            prefix.len() + 1,
        );
        let want: Vec<usize> = (start..end).collect();
        // Structured manifest validation (parsed, never opaque).
        let manifest_raw =
            std::fs::read_to_string(dir.join("MANIFEST.json")).expect("prepare manifest present");
        let manifest: serde_json::Value =
            serde_json::from_str(&manifest_raw).expect("manifest must be valid JSON");
        assert_eq!(
            manifest.get("schema").and_then(|value| value.as_str()),
            Some("install-prepare/2"),
            "manifest schema must be install-prepare/2",
        );
        let pair_test = manifest
            .get("test")
            .and_then(|value| value.as_str())
            .expect("manifest test name");
        assert!(
            [
                "selected10_walk_shard_install_matches_dynamic",
                "prepared_transfer_row918_no_global_elimination"
            ]
            .contains(&pair_test),
            "manifest test name must be a known install gate, got {pair_test:?}",
        );
        assert_eq!(
            std::fs::read(dir.join("PREPARE_OK")).expect("PREPARE_OK present"),
            b"ok",
            "PREPARE_OK marker must read ok",
        );
        let manifest_scenarios = manifest
            .get("scenarios")
            .and_then(|value| value.as_array())
            .expect("manifest scenarios array");
        assert_eq!(
            manifest_scenarios.len(),
            12,
            "manifest must carry the 12 scenario identities",
        );
        for (index, expected) in INSTALL_DIFFERENTIAL_PREFIXES.iter().enumerate() {
            let actual = manifest_scenarios[index]
                .as_str()
                .expect("scenario identity must be a string");
            assert_eq!(
                actual,
                String::from_utf8_lossy(expected),
                "manifest scenario {index} must match the corpus",
            );
        }
        let manifest_counts = manifest
            .get("component_state_counts")
            .and_then(|value| value.as_array())
            .expect("manifest component_state_counts");
        // Reload both backends with the shared phase-1 vocab; reassert pins.
        let vocab = phase1_vocab();
        let raw_static = std::fs::read(dir.join("static.bin")).expect("read static.bin");
        let raw_dynamic = std::fs::read(dir.join("dynamic.bin")).expect("read dynamic.bin");
        assert!(!raw_static.is_empty() && !raw_dynamic.is_empty(), "refusing empty bins");
        let static_comp =
            Constraint::load_with_vocab(&raw_static, &vocab).expect("reload static");
        let dynamic =
            Constraint::load_with_vocab(&raw_dynamic, &vocab).expect("reload dynamic");
        assert!(
            static_comp.uses_compact_segmented_parser_runtime(),
            "reloaded static must stay on the compact runtime",
        );
        let mut actual_shards: Vec<(u32, usize)> = Vec::new();
        for (index, component) in static_comp
            .static_dynamic_overlay
            .as_ref()
            .expect("overlay")
            .segmented_parser_components
            .iter()
            .enumerate()
        {
            if let Some(shard) = component.boundary.as_ref() {
                assert!(
                    matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ),
                    "reloaded component {index} must be a StaticParser shard",
                );
                actual_shards.push((
                    shard.start_component,
                    shard.candidate_tokens.as_ref().map(|tokens| tokens.len()).unwrap_or(0),
                ));
            }
        }
        assert!(!actual_shards.is_empty(), "reloaded static must carry shards");
        let manifest_shards = manifest
            .get("shards")
            .and_then(|value| value.as_array())
            .expect("manifest shards array");
        let mut manifest_ids: Vec<(u32, usize)> = manifest_shards
            .iter()
            .map(|entry| {
                (
                    entry
                        .get("start_component")
                        .and_then(|value| value.as_u64())
                        .expect("shard start_component u64") as u32,
                    entry
                        .get("candidate_tokens")
                        .and_then(|value| value.as_u64())
                        .expect("shard candidate_tokens u64") as usize,
                )
            })
            .collect();
        manifest_ids.sort();
        actual_shards.sort();
        assert_eq!(
            manifest_ids, actual_shards,
            "manifest shards must exactly match reloaded component IDs/candidate counts",
        );
        let manifest_counts: Vec<u32> = manifest_counts
            .iter()
            .map(|value| value.as_u64().expect("component count u64") as u32)
            .collect();
        assert_scoped_tokenizer_packing("windowed", &static_comp, &manifest_counts);
        // Single pass: commit through start+count-1, compare full vectors
        // and collect hashes at wanted positions only.
        let mut st_dyn = dynamic.start();
        let mut st_static = static_comp.start();
        let mut positions: Vec<serde_json::Value> = Vec::with_capacity(count);
        let hash_one = |mask: &[u32]| {
            let mut hash: u64 = 0xcbf29ce484222325;
            for (index, &word) in mask.iter().enumerate() {
                hash ^= (word as u64).wrapping_add(index as u64);
                hash = hash.wrapping_mul(0x100000001b3);
            }
            format!("{hash:016x}")
        };
        let record = |consumed: usize,
                          st_dyn: &mut crate::runtime::ConstraintState<'_>,
                          st_static: &mut crate::runtime::ConstraintState<'_>,
                          positions: &mut Vec<serde_json::Value>| {
            let mask_dyn = st_dyn.mask();
            let mask_static = st_static.mask();
            assert_eq!(
                mask_static, mask_dyn,
                "mask mismatch scenario={scenario} consumed={consumed}",
            );
            assert!(!mask_dyn.is_empty(), "refusing empty-mask hash");
            let static_hash = hash_one(&mask_static);
            let dynamic_hash = hash_one(&mask_dyn);
            assert_eq!(static_hash.len(), 16);
            assert_eq!(dynamic_hash.len(), 16);
            assert!(
                static_hash.chars().all(|char| char.is_ascii_hexdigit()),
                "static hash must be 16 hex digits",
            );
            assert!(
                dynamic_hash.chars().all(|char| char.is_ascii_hexdigit()),
                "dynamic hash must be 16 hex digits",
            );
            assert_eq!(static_hash, dynamic_hash);
            positions.push(serde_json::json!({
                "index": consumed,
                "static_hash": static_hash,
                "dynamic_hash": dynamic_hash,
            }));
        };
        if want.contains(&0) {
            record(0, &mut st_dyn, &mut st_static, &mut positions);
        }
        for (consumed, &byte) in prefix.iter().enumerate() {
            let r_dyn = st_dyn.commit_bytes(&[byte]);
            let r_static = st_static.commit_bytes(&[byte]);
            assert_eq!(
                r_dyn.is_ok(),
                r_static.is_ok(),
                "commit divergence scenario={scenario} consumed={consumed}",
            );
            // Strict: every commit in the validated fixture must succeed
            // (stronger than the original both-error break).
            assert!(
                r_dyn.is_ok(),
                "strict window: commit must succeed scenario={scenario} consumed={consumed}",
            );
            if want.contains(&(consumed + 1)) {
                record(consumed + 1, &mut st_dyn, &mut st_static, &mut positions);
            }
            if consumed + 1 >= end {
                break;
            }
        }
        assert_eq!(positions.len(), count, "every requested position compared");
        // One strict JSON object + marker, only AFTER full-vector equality.
        let doc = serde_json::json!({
            "schema": "install-window/1",
            "pair_test": pair_test,
            "scenario": scenario,
            "prefix": String::from_utf8_lossy(prefix).into_owned(),
            "start": start,
            "count": count,
            "total_positions": prefix.len() + 1,
            "positions": positions,
        });
        println!("{}", serde_json::to_string(&doc).expect("serialize window"));
        eprintln!(
            "INSTALL_WINDOW_OK scenario={scenario} start={start} count={count} total={}",
            prefix.len() + 1,
        );
    }

    /// Dispatch-parent grammar source for the inner-link sweep. Mirrors
    /// `dispatcher_literal_names_parent_source` in
    /// `examples/static_link_measure.rs` (10 `.tool_i(` slots over the same
    /// placeholder token IDs).
    fn dispatch_parent_source() -> String {
        let mut source = String::from("start suffix;\n");
        for index in 0..10 {
            source.push_str(&format!(
                "t TOOL_ARGS_SLOT_{index} ::= @token({});\n",
                128_320 + 10 + index
            ));
        }
        source.push_str("nt suffix ::=\n    ");
        for index in 0..10 {
            if index != 0 {
                source.push_str("\n  | ");
            }
            source.push_str(&format!(r#"".tool_{index}(" TOOL_ARGS_SLOT_{index} ")""#));
        }
        source.push_str(";\n");
        source
    }

    /// Per-link stage timing record for the sweep table.
    struct LinkTiming {
        name: String,
        dyn_compose_ms: f64,
        low_level_ms: f64,
        shared_ms: f64,
        flat_ms: f64,
        walk_ms: f64,
        det_ms: f64,
        min_ms: f64,
        shards_built: usize,
        table_ms: f64,
        templates_ms: f64,
        parser_ms: f64,
        install_ms: f64,
        installed: usize,
        diff_positions: usize,
        diff_mismatches: usize,
        diff_checksum: u64,
        diff_ms: f64,
    }

    impl LinkTiming {
        fn total_link_ms(&self) -> f64 {
            self.shared_ms
                + self.walk_ms
                + self.table_ms
                + self.templates_ms
                + self.parser_ms
                + self.install_ms
        }
    }

    /// Seeded RNG token-walk differential between two compositions (dynamic
    /// reference vs installed/fallback twin). Language-agnostic: drives on
    /// the dynamic masks. Returns (positions, mismatches, checksum).
    fn rng_choose_token(mask: &[u32], rng: &mut u64) -> Option<u32> {
        let allowed: usize = mask.iter().map(|word| word.count_ones() as usize).sum();
        if allowed == 0 {
            return None;
        }
        *rng ^= *rng >> 12;
        *rng ^= *rng << 25;
        *rng ^= *rng >> 27;
        let draw = rng.wrapping_mul(0x2545_f491_4f6c_dd1d);
        let mut rank = draw as usize % allowed;
        for (word_index, &word) in mask.iter().enumerate() {
            let count = word.count_ones() as usize;
            if rank >= count {
                rank -= count;
                continue;
            }
            let mut live = word;
            for _ in 0..rank {
                live &= live - 1;
            }
            let bit = live.trailing_zeros() as usize;
            return Some((word_index * 32 + bit) as u32);
        }
        None
    }

    fn rng_differential(
        dynamic: &Constraint,
        other: &Constraint,
        steps: usize,
        seed: u64,
    ) -> (usize, usize, u64, Option<String>) {
        fn mask_bits(mask: &[u32]) -> Vec<u32> {
            let mut out = Vec::new();
            for (word_index, &word) in mask.iter().enumerate() {
                let mut live = word;
                while live != 0 {
                    let bit = live.trailing_zeros() as usize;
                    out.push((word_index * 32 + bit) as u32);
                    live &= live - 1;
                }
            }
            out
        }
        let mut rng = seed;
        let mut checksum: u64 = 0xcbf29ce484222325;
        let mut positions = 0usize;
        let mut mismatches = 0usize;
        let mut first: Option<String> = None;
        let mut st_dyn = dynamic.start();
        let mut st_other = other.start();
        let mut check = |st_dyn: &mut crate::runtime::ConstraintState<'_>,
                         st_other: &mut crate::runtime::ConstraintState<'_>,
                         step_index: usize,
                         first: &mut Option<String>| {
            let mask_dyn = st_dyn.mask();
            let mask_other = st_other.mask();
            for (index, &word) in mask_dyn.iter().enumerate() {
                checksum ^= (word as u64).wrapping_add(index as u64);
                checksum = checksum.wrapping_mul(0x100000001b3);
            }
            positions += 1;
            if mask_dyn != mask_other {
                mismatches += 1;
                if first.is_none() {
                    let dyn_bits = mask_bits(&mask_dyn);
                    let other_bits = mask_bits(&mask_other);
                    let dyn_set: BTreeSet<u32> = dyn_bits.into_iter().collect();
                    let other_set: BTreeSet<u32> = other_bits.into_iter().collect();
                    let dyn_only: Vec<u32> =
                        dyn_set.difference(&other_set).copied().take(12).collect();
                    let other_only: Vec<u32> =
                        other_set.difference(&dyn_set).copied().take(12).collect();
                    *first = Some(format!(
                        "mask step={step_index} dyn_admits={} other_admits={} dyn_only={dyn_only:?} other_only={other_only:?}",
                        dyn_set.len(),
                        other_set.len(),
                    ));
                }
            }
        };
        check(&mut st_dyn, &mut st_other, 0, &mut first);
        let mut path: Vec<u32> = Vec::new();
        for step in 0..steps {
            let mask_dyn = st_dyn.mask();
            let Some(token) = rng_choose_token(&mask_dyn, &mut rng) else {
                break;
            };
            path.push(token);
            let r_dyn = st_dyn.commit_token(token);
            let r_other = st_other.commit_token(token);
            if r_dyn.is_ok() != r_other.is_ok() {
                mismatches += 1;
                if first.is_none() {
                    first = Some(format!(
                        "commit step={} token={token} dyn_ok={} other_ok={}",
                        step + 1,
                        r_dyn.is_ok(),
                        r_other.is_ok(),
                    ));
                }
                break;
            }
            if r_dyn.is_err() {
                break;
            }
            check(&mut st_dyn, &mut st_other, step + 1, &mut first);
        }
        if mismatches > 0 {
            eprintln!("BOUNDARY_LINK_PATH positions={positions} path={path:?}");
        }
        (positions, mismatches, checksum, first)
    }

    /// Run the full walk-shard link for one composition and time every stage.
    /// Publication uses the production signed-transfer route (same fragment
    /// library, composer, and publisher as `build_walk_static_boundary_link`),
    /// so this sweep validates what production installs. There is no table
    /// stage on this route (`table_ms` is exactly 0).
    fn link_and_time(
        name: &str,
        parent: &Constraint,
        slots: &[(String, &Constraint)],
        vocab: &Vocab,
        diff_steps: usize,
    ) -> LinkTiming {
        let inputs: Vec<CompiledSubgrammarInput> = slots
            .iter()
            .map(|(slot, child)| CompiledSubgrammarInput {
                placeholder_terminal: terminal_id(parent, slot),
                additional_placeholder_terminals: &[],
                constraint: child,
            })
            .collect();
        let compose_started = Instant::now();
        let dynamic = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &inputs,
            vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("dynamic compose")
        .constraint;
        let dyn_compose_ms = compose_started.elapsed().as_secs_f64() * 1000.0;
        let low_level_started = Instant::now();
        let composed = low_level_compose(parent, &inputs);
        let grammar = analyzed_grammar(&composed.table.table, &composed.terminal_names);
        let disallowed = compute_disallowed_follows(&grammar);
        let low_level_ms = low_level_started.elapsed().as_secs_f64() * 1000.0;
        let counts: Vec<u32> = std::iter::once(parent.tokenizer.num_states())
            .chain(inputs.iter().map(|child| child.constraint.tokenizer.num_states()))
            .collect();
        let retain_parent_non_crossing_paths = inputs
            .iter()
            .any(|child| child.constraint.table.embedded_start_nullable());
        let (built, profile) = build_boundary_shard_walks(&BoundaryShardLinkInputs {
            merged_tokenizer: &composed.tokenizer,
            vocab,
            grammar: &grammar,
            disallowed_follows: &disallowed,
            ignore_terminal: composed.ignore_canonical,
            follow_transparent_ignores: None,
            terminal_offsets: &composed.table.terminal_offsets,
            leaf_to_immediate: None,
            tokenizer_offsets: &composed.tokenizer_offsets,
            component_state_counts: &counts,
            candidate_tokens_by_component: None,
            retain_parent_non_crossing_paths,
            walk_plans: None,
        })
        .expect("walk shards");
        let shards_built = built.len();
        let (mut walk_ms, mut det_ms, mut min_ms) = (0.0, 0.0, 0.0);
        for (_, shard_profile) in &profile.per_shard {
            walk_ms += shard_profile.walk_ms;
            det_ms += shard_profile.determinize_ms;
            min_ms += shard_profile.minimize_ms;
        }
        // Signed-transfer shard publication (production route): scoped ordinary
        // transfers + Entry/Finish exports, exact resolution, table-free
        // normalization. No composed/provider table exists here; unbound slots
        // use the same empty-language definition as production
        // (`unbound_link_slot_terminals`). There is no table stage anymore, so
        // `table_ms` below is exactly 0.
        use crate::compiler::boundary_transfer::{
            build_fragment_library, build_signed_link_context, compile_signed_shard_parser,
            publish_signed_shard,
        };
        let global_ignores = component_ignores_are_globally_erasable(parent, &inputs);
        let unbound_set: std::collections::BTreeSet<TerminalID> =
            unbound_link_slot_terminals(parent, &inputs, &composed.table.terminal_offsets)
                .expect("unbound slots")
                .into_iter()
                .collect();
        let signed_context = build_signed_link_context(
            parent,
            &inputs,
            &composed.table.terminal_offsets,
            composed.table.table.num_terminals,
            global_ignores,
            unbound_set,
        )
        .expect("signed link context");
        let table_ms = 0.0;
        let (mut templates_ms, mut parser_ms, mut install_ms) = (0.0, 0.0, 0.0);
        let mut static_comp = dynamic.clone();
        let mut published = Vec::with_capacity(built.len());
        for shard in built {
            let emitted = boundary_emitted_terminals(
                &shard.output.dwa,
                composed.table.table.num_terminals as usize,
            );
            let library = build_fragment_library(
                &signed_context,
                &emitted,
                shard.start_component as u32,
            )
            .expect("fragment library");
            templates_ms += library.templates_ms;
            let compiled = compile_signed_shard_parser(
                &signed_context,
                &library,
                &shard.output.dwa,
                &shard.output.id_map,
                shard.start_component as u32,
            )
            .expect("signed shard compile");
            parser_ms += compiled.compose_ms + compiled.resolve_ms + compiled.normalize_ms;
            let work = WalkBoundaryShardWork {
                start_component: shard.start_component as u32,
                terminal_automaton: TerminalAutomaton::Dwa(shard.output.dwa),
                id_map: shard.output.id_map,
                candidate_tokens: shard
                    .candidate_tokens
                    .into_iter()
                    .collect::<Vec<_>>()
                    .into(),
            };
            let (one, _publish_profile) = publish_signed_shard(
                work,
                compiled,
                &composed.tokenizer_offsets,
                &counts,
            )
            .expect("publish signed shard");
            published.push(one);
        }
        let installed = published.len();
        if !published.is_empty() {
            let install_started = Instant::now();
            install_published_static_boundary_shards(
                static_comp.static_dynamic_overlay.as_mut().expect("overlay"),
                published,
            )
            .expect("install");
            install_ms = install_started.elapsed().as_secs_f64() * 1000.0;
        }
        assert!(
            static_comp.uses_compact_segmented_parser_runtime(),
            "{name}: installed composition must stay on the compact runtime",
        );
        assert_scoped_tokenizer_packing(name, &static_comp, &counts);
        let diff_started = Instant::now();
        let (diff_positions, diff_mismatches, diff_checksum, diff_first) =
            rng_differential(&dynamic, &static_comp, diff_steps, 0x9e37_79b9_7f4a_7c15);
        let diff_ms = diff_started.elapsed().as_secs_f64() * 1000.0;
        let timing = LinkTiming {
            name: name.to_string(),
            dyn_compose_ms,
            low_level_ms,
            shared_ms: profile.shared_wall_ms,
            flat_ms: profile.flat_ms,
            walk_ms,
            det_ms,
            min_ms,
            shards_built,
            table_ms,
            templates_ms,
            parser_ms,
            install_ms,
            installed,
            diff_positions,
            diff_mismatches,
            diff_checksum,
            diff_ms,
        };
        eprintln!(
            "BOUNDARY_LINK name={} dyn={:.1} low={:.1} shared={:.1} flat={:.1} walk={:.1} det={:.1} min={:.1} shards={} table={:.1} templates={:.1} parser={:.1} install={:.1} installed={} diff_pos={} diff_mm={} diffck={:016x} diff_ms={:.0} total_link={:.1}",
            timing.name,
            timing.dyn_compose_ms,
            timing.low_level_ms,
            timing.shared_ms,
            timing.flat_ms,
            timing.walk_ms,
            timing.det_ms,
            timing.min_ms,
            timing.shards_built,
            timing.table_ms,
            timing.templates_ms,
            timing.parser_ms,
            timing.install_ms,
            timing.installed,
            timing.diff_positions,
            timing.diff_mismatches,
            timing.diff_checksum,
            timing.diff_ms,
            timing.total_link_ms(),
        );
        if let Some(detail) = &diff_first {
            eprintln!("BOUNDARY_LINK_FIRST_MISMATCH name={name} {detail}");
        }
        timing
    }

    /// Step-4 link timing sweep: full walk-shard link over every available
    /// composition (10× dispatch-parent+schema + 1× core+dispatch outer)
    /// with per-stage max/median and >3×-median outlier flags.
    #[test]
    #[ignore]
    fn link_timing_all_compositions() {
        let root = phase1_root();
        let vocab = phase1_vocab();
        let parent =
            Constraint::from_glrm_grammar(&dispatch_parent_source(), &vocab).expect("parent");
        let mut schema_paths: Vec<_> = std::fs::read_dir(&root)
            .expect("read cache dir")
            .filter_map(|entry| {
                let name = entry.ok()?.file_name().to_string_lossy().into_owned();
                name.starts_with("schema-").then_some(name)
            })
            .collect();
        schema_paths.sort();
        assert_eq!(schema_paths.len(), 10, "cache must hold the 10 schema constraints");
        let mut schemas = Vec::with_capacity(10);
        for name in &schema_paths {
            let mut schema =
                Constraint::load(&std::fs::read(root.join(name)).expect("read schema"))
                    .expect("load schema");
            restore_component(&mut schema, name);
            schemas.push(schema);
        }
        let mut core =
            Constraint::load(&std::fs::read(root.join("core.bin")).expect("read core.bin"))
                .expect("load core");
        let dispatch_name = std::env::var("PHASE1_DISPATCH")
            .unwrap_or_else(|_| "dispatch-literal.bin".to_string());
        let mut dispatch =
            Constraint::load(&std::fs::read(root.join(&dispatch_name)).expect("read dispatch"))
                .expect("load dispatch");
        restore_component(&mut core, "core");
        restore_component(&mut dispatch, "dispatch");

        let mut timings = Vec::new();
        for (index, schema) in schemas.iter().enumerate() {
            timings.push(link_and_time(
                &format!("inner-{index}"),
                &parent,
                &[(format!("TOOL_ARGS_SLOT_{index}"), schema)],
                &vocab,
                64,
            ));
        }
        // NOTE: no 10-way segmented dispatch link — N-way segmented compose
        // rejects it ("component 2 has a non-functional LR-state relation").
        // The architecture composes dispatch flat and segments only the outer
        // link, so the sweep covers the 10 synthetic inner links + outer.
        timings.push(link_and_time(
            "outer",
            &core,
            &[("PROGRAMMATIC_TOOL_SUFFIX".to_string(), &dispatch)],
            &vocab,
            64,
        ));

        let stages: &[(&str, fn(&LinkTiming) -> f64)] = &[
            ("shared", |t| t.shared_ms),
            ("walk", |t| t.walk_ms),
            ("det", |t| t.det_ms),
            ("min", |t| t.min_ms),
            ("templates", |t| t.templates_ms),
            ("parser", |t| t.parser_ms),
            ("table", |t| t.table_ms),
            ("install", |t| t.install_ms),
            ("total_link", |t| t.total_link_ms()),
        ];
        for (stage, get) in stages {
            let mut values: Vec<f64> = timings.iter().map(get).collect();
            values.sort_by(f64::total_cmp);
            let median = values[values.len() / 2];
            let max = values[values.len() - 1];
            let argmax = timings.iter().map(get).enumerate().max_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let flagged = max > 3.0 * median;
            eprintln!(
                "BOUNDARY_STAGE stage={stage} n={} median_ms={median:.1} max_ms={max:.1} max_link={} flagged_3x={flagged}",
                values.len(),
                argmax.map(|(i, _)| timings[i].name.as_str()).unwrap_or("?"),
            );
        }
        let total_mismatches: usize = timings.iter().map(|timing| timing.diff_mismatches).sum();
        let failing: Vec<&str> = timings
            .iter()
            .filter(|timing| timing.diff_mismatches > 0)
            .map(|timing| timing.name.as_str())
            .collect();
        eprintln!("BOUNDARY_SWEEP links={} total_mismatches={total_mismatches} failing={failing:?}", timings.len());
        for timing in &timings {
            assert_eq!(
                timing.installed,
                timing.shards_built,
                "link {} must install every built shard",
                timing.name,
            );
        }
        assert_eq!(total_mismatches, 0, "walk-shard sweep must match DynamicDirect on every link");
    }

    /// Called-frame non-equivalence pin (Phase 2b reconciliation audit).
    ///
    /// The compact segmented runtime evaluates boundary shards against live
    /// GSS stacks in the leaf/provider coordinate, where an in-progress call
    /// appears as Call-pushed frames `[.., parent_target, child_start, ..]`.
    /// The spliced composed table speaks a different calling convention (child
    /// start overlaid on callers, returns via reduce+goto through caller
    /// states, fresh appended child-state identities), so its template labels
    /// never match called frames: a splice-published shard systematically
    /// under-admits wherever the live stack holds an active call. The
    /// provider-materialized table speaks the live coordinate with call-site
    /// guards and is exact here. The Phase 2 sweep stayed green only because
    /// no corpus position needed a child-shard admission (the full sweep
    /// passes with every child shard removed, outer with zero shards).
    ///
    /// Fixture: one placeholder called from two states with divergent
    /// continuations (`L SUB x | R SUB y`, child `a`), fused exit tokens
    /// `ax`/`ay` discriminated by call-site guards. Asserts DynamicDirect
    /// admits each fused token at its own call site, the provider-published
    /// shard matches DynamicDirect exactly, and the splice-published shard
    /// misses both fused tokens. If the splice side ever starts admitting
    /// them, the coordinate story changed: revisit this pin and the audit.
    #[test]
    fn splice_boundary_shard_underadmits_called_frames_divergent() {
        fn admits(mask: &[u32], token: u32) -> bool {
            mask.get(token as usize / 32)
                .is_some_and(|word| word & (1u32 << (token % 32)) != 0)
        }

        let vocab = Vocab::new(vec![
            (0, b"ax".to_vec()),
            (1, b"ay".to_vec()),
            (2, b"L".to_vec()),
            (3, b"R".to_vec()),
            (4, b"a".to_vec()),
            (5, b"x".to_vec()),
            (6, b"y".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "L" SUB "x" | "R" SUB "y";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let inputs = [CompiledSubgrammarInput {
            placeholder_terminal: terminal_id(&parent, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &child,
        }];
        let dynamic = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("dynamic compose")
        .constraint;

        // Publish the alternative shards into a real static coordinator. A
        // DynamicDirect coordinator deliberately has no outer static seed-mask
        // universe; swapping only its shard enum is not a backend conversion.
        let static_shell = compose_constraints_owned_parent_segmented(
            parent.clone(), &inputs, &vocab, SegmentedBoundaryBackend::StaticParserDwa,
        ).expect("static coordinator for publication fixture").constraint;

        let composed = low_level_compose(&parent, &inputs);
        let grammar = analyzed_grammar(&composed.table.table, &composed.terminal_names);
        let disallowed = compute_disallowed_follows(&grammar);
        let counts: Vec<u32> = vec![
            parent.tokenizer.num_states(),
            child.tokenizer.num_states(),
        ];
        let (built, _) = build_boundary_shard_walks(&BoundaryShardLinkInputs {
            merged_tokenizer: &composed.tokenizer,
            vocab: &vocab,
            grammar: &grammar,
            disallowed_follows: &disallowed,
            ignore_terminal: composed.ignore_canonical,
            follow_transparent_ignores: None,
            terminal_offsets: &composed.table.terminal_offsets,
            leaf_to_immediate: None,
            tokenizer_offsets: &composed.tokenizer_offsets,
            component_state_counts: &counts,
            candidate_tokens_by_component: None,
            retain_parent_non_crossing_paths: inputs
                .iter()
                .any(|child| child.constraint.table.embedded_start_nullable()),
            walk_plans: None,
        })
        .expect("walk shards");
        assert_eq!(built.len(), 1, "only the child shard crosses here");
        assert_eq!(built[0].start_component, 1);
        assert!(
            unbound_slot_terminals(&parent, &["SUB"]).is_empty()
                && unbound_slot_terminals(&child, &[]).is_empty(),
            "fixture has no unbound slots; emptying must not be involved",
        );
        let (splice_table, _) =
            prepare_spliced_boundary_table("divergent", &composed, &[]);
        let global_ignores = component_ignores_are_globally_erasable(&parent, &inputs);
        let provider_table = Arc::new(
            link_provider_boundary_table(
                &parent,
                &inputs,
                &composed.table.terminal_offsets,
                composed.table.table.num_terminals,
                global_ignores,
            )
            .expect("provider boundary table"),
        );

        let publish_against = |table: &Arc<GLRTable>| {
            let shard = &built[0];
            let work = WalkBoundaryShardWork {
                start_component: shard.start_component as u32,
                terminal_automaton: TerminalAutomaton::Dwa(shard.output.dwa.clone()),
                id_map: shard.output.id_map.clone(),
                candidate_tokens: shard
                    .candidate_tokens
                    .iter()
                    .copied()
                    .collect::<Vec<_>>()
                    .into(),
            };
            let (one, _) =
                publish_walk_boundary_shard_work(work, table, &composed.tokenizer_offsets, &counts)
                    .expect("publish walk shard");
            let mut installed = static_shell.clone();
            install_published_static_boundary_shards(
                installed.static_dynamic_overlay.as_mut().expect("overlay"),
                vec![one],
            )
            .expect("install");
            assert!(
                installed.uses_compact_segmented_parser_runtime(),
                "installed composition must stay on the compact runtime",
            );
            installed
        };
        let splice_static = publish_against(&splice_table);
        let provider_static = publish_against(&provider_table);

        // After L: fused `ax` (token 0) must be admitted; after R: `ay` (1).
        for (commit, fused, sibling, site) in [(2u32, 0u32, 1u32, "L"), (3u32, 1u32, 0u32, "R")] {
            let mut st_dyn = dynamic.start();
            let mut st_splice = splice_static.start();
            let mut st_provider = provider_static.start();
            st_dyn.commit_token(commit).unwrap();
            st_splice.commit_token(commit).unwrap();
            st_provider.commit_token(commit).unwrap();
            let mask_dyn = st_dyn.mask();
            let mask_splice = st_splice.mask();
            let mask_provider = st_provider.mask();
            assert!(
                admits(&mask_dyn, fused),
                "dynamic must admit fused token {fused} after {site}",
            );
            assert!(
                !admits(&mask_dyn, sibling),
                "dynamic must reject sibling fused token {sibling} after {site}",
            );
            assert_eq!(
                mask_provider, mask_dyn,
                "provider shard must match DynamicDirect after {site}",
            );
            assert!(
                !admits(&mask_splice, fused),
                "splice shard under-admits fused token {fused} after {site} \
                 (called-frame labels unmatched); if this now admits, revisit the audit pin",
            );
        }
    }

    /// Production-outer decline pin (Phase 2b reconciliation audit).
    ///
    /// The provider-materialized boundary table's `exact_control_elimination`
    /// diverges on the selected10 outer (core+dispatch) link at the known
    /// row-918 / terminal-3 cell (`state=918 terminal=3`, guard-differentiated
    /// cyclic fan-out over a ~10-state reduce cycle — the identical cell the
    /// old segmented-static path hung on), so current production static
    /// composition DECLINES the outer link loudly via the convergence cap
    /// instead of linking it. Phase 2 linked outer exactly via the splice in
    /// ~7.6 s; that object is inexact on called frames (see the divergent pin
    /// above), so neither table object currently serves outer statically.
    ///
    /// If this test observes convergence (Ok), the elimination pathology is
    /// fixed and the production contract can be revisited; the panic message
    /// says so. Fast decline (well under the 10 s wall cap) is part of the
    /// pin: the budget must fail fast, never hang.
    #[test]
    #[ignore]
    fn provider_boundary_table_diverges_on_selected10_outer() {
        let fixture = load_selected10_outer();
        let children = [CompiledSubgrammarInput {
            placeholder_terminal: terminal_id(&fixture.core, "PROGRAMMATIC_TOOL_SUFFIX"),
            additional_placeholder_terminals: &[],
            constraint: &fixture.dispatch,
        }];
        let global_ignores = component_ignores_are_globally_erasable(&fixture.core, &children);
        let started = Instant::now();
        let result = link_provider_boundary_table(
            &fixture.core,
            &children,
            &fixture.composed.table.terminal_offsets,
            fixture.composed.table.table.num_terminals,
            global_ignores,
        );
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        match result {
            Ok(table) => panic!(
                "provider boundary table unexpectedly converged on outer: states={} terms={} \
                 controls={} ms={ms:.1}; the elimination pathology is fixed — update this pin \
                 and revisit the production static-link contract",
                table.num_states,
                table.num_terminals,
                table.control_terminals.len(),
            ),
            Err(error) => {
                eprintln!("BOUNDARY_PROVIDER_DIVERGENCE ms={ms:.1} error={error}");
                assert!(
                    error.contains("state=918")
                        && error.contains("terminal=3")
                        && error.contains("stack_effect_visits"),
                    "outer decline must be the row-918 convergence cap, got: {error}",
                );
                assert!(
                    ms < 60_000.0,
                    "convergence cap must fail fast, took {ms:.1} ms",
                );
            }
        }
    }

    /// Prepared-transfer outer link through the NEW signed-transfer production
    /// Row-918 local transfer/entry/finish characterization (no link, no
    /// install): links agree, core terminal-3 transfer finite, slot Entry
    /// shape valid + finite + child-start-pushed, dispatch Finish finite +
    /// no local EOF effects. Extracted so the cheap asserts run as their own
    /// <=20s diagnostic; the original row918 test calls this helper.
    fn assert_row918_local_transfers(
        fixture: &Selected10Outer,
        links: &[crate::runtime::SegmentedParserLink],
    ) {
        use crate::compiler::boundary_transfer::{
            StateInjection, diagnose_transfer, instantiate_entry, instantiate_finish,
            scope_characterization, validate_slot_entry_shape,
        };
        use crate::compiler::stages::templates::characterize::characterize_selected_terminals_for_terminal_count;

        assert_eq!(links.len(), 1, "outer link has one child link");
        let link = &links[0];
        let parent_injection = StateInjection { offset: 0 };
        let child_injection = StateInjection {
            offset: fixture.core.table.num_states,
        };
        // Ordinary local transfer for core terminal 3 (the row-918 terminal).
        let mut selected = vec![false; fixture.core.table.num_terminals as usize];
        selected[3] = true;
        let characterized = characterize_selected_terminals_for_terminal_count(
            &fixture.core.table,
            fixture.core.table.num_terminals,
            &selected,
        );
        let local_t3 = characterized.get(&3).expect("core terminal 3 characterizes");
        let scoped_t3 =
            scope_characterization(local_t3, &parent_injection).expect("scope core terminal 3");
        let diag_t3 = diagnose_transfer(&scoped_t3);
        eprintln!(
            "BOUNDARY_TRANSFER t3 escapes={} reduces={} nt_escapes={} nt_rereduces={} read={} push={} cycle={:?} kinds={:?}",
            diag_t3.num_escapes,
            diag_t3.num_reduces,
            diag_t3.num_nt_escapes,
            diag_t3.num_nt_rereduces,
            diag_t3.max_input_read,
            diag_t3.max_push_len,
            diag_t3.cycle_status,
            diag_t3.endpoint_kinds,
        );
        assert!(
            diag_t3.cycle_status.is_none(),
            "core terminal-3 local transfer must be finite",
        );
        // Slot Entry: validate the provider-supported shape, then instantiate.
        validate_slot_entry_shape(&fixture.core.table, link.slot_terminal)
            .expect("outer slot supports static Entry");
        let mut slot_selected = vec![false; fixture.core.table.num_terminals as usize];
        slot_selected[link.slot_terminal as usize] = true;
        let slot_characterized = characterize_selected_terminals_for_terminal_count(
            &fixture.core.table,
            fixture.core.table.num_terminals,
            &slot_selected,
        );
        let local_slot = slot_characterized
            .get(&link.slot_terminal)
            .expect("slot terminal characterizes");
        let scoped_slot =
            scope_characterization(local_slot, &parent_injection).expect("scope slot");
        let scoped_child_start = child_injection
            .scope_state(link.child_start)
            .expect("scope child start");
        let entry = instantiate_entry(scoped_slot, scoped_child_start, 0);
        eprintln!(
            "BOUNDARY_TRANSFER entry escapes={} reduces={} read={} push={} cycle={:?} kinds={:?}",
            entry.diagnostics.num_escapes,
            entry.diagnostics.num_reduces,
            entry.diagnostics.max_input_read,
            entry.diagnostics.max_push_len,
            entry.diagnostics.cycle_status,
            entry.diagnostics.endpoint_kinds,
        );
        assert!(
            entry.diagnostics.cycle_status.is_none(),
            "slot Entry transfer must be finite",
        );
        assert!(
            entry
                .characterization
                .escapes
                .iter()
                .all(|escape| escape.pushes.last() == Some(&scoped_child_start)),
            "every Entry escape must carry the scoped child-start push",
        );
        // Finish export for the dispatch child under this link's policy.
        let (finish, has_local_eof_effects) =
            instantiate_finish(&fixture.dispatch.table, link, &child_injection)
                .expect("dispatch finish transfer");
        eprintln!(
            "BOUNDARY_TRANSFER finish return_pop={} nullable={} escapes={} reduces={} nt_escapes={} nt_rereduces={} read={} push={} cycle={:?} kinds={:?} local_eof={has_local_eof_effects}",
            link.return_pop,
            link.child_start_nullable,
            finish.diagnostics.num_escapes,
            finish.diagnostics.num_reduces,
            finish.diagnostics.num_nt_escapes,
            finish.diagnostics.num_nt_rereduces,
            finish.diagnostics.max_input_read,
            finish.diagnostics.max_push_len,
            finish.diagnostics.cycle_status,
            finish.diagnostics.endpoint_kinds,
        );
        assert!(
            finish.diagnostics.cycle_status.is_none(),
            "dispatch finish transfer must be finite",
        );
        assert!(
            !has_local_eof_effects,
            "bounded flat prototype requires canonical EOF completion (reductions + pure Accept); local EOF work needs outer control-choice points",
        );
    }

    /// Row-918 transfer/entry/finish characterization WITHOUT the expensive
    /// link: runs the extracted helper only (<=20s diagnostic). The counter
    /// proof stays in the fresh setup case, not here.
    #[test]
    #[ignore]
    fn row918_local_transfers_without_link() {
        use crate::compiler::boundary_transfer::validate_shared_child_links;
        use crate::compiler::constraint_compose::build_segmented_parser_links;

        let _env_lock = crate::TEST_ENV_LOCK.lock().unwrap();
        let fixture = load_selected10_outer();
        let children = [CompiledSubgrammarInput {
            placeholder_terminal: terminal_id(&fixture.core, "PROGRAMMATIC_TOOL_SUFFIX"),
            additional_placeholder_terminals: &[],
            constraint: &fixture.dispatch,
        }];
        let links = build_segmented_parser_links(&children).expect("segmented links");
        validate_shared_child_links(&links).expect("incoming links agree");
        assert_row918_local_transfers(&fixture, &links);
    }

    /// route (advisor v3 Prototype 1, milestones F + 9).
    ///
    /// Links only LOCAL tables: the ordinary core terminal-3 transfer plus
    /// the dispatch Finish export under this link's child-start/return-pop
    /// policy, with the slot Entry shape validated. Records local
    /// characterization cycle status / read bound / push bound / sizes for
    /// both transfers, then compiles, installs, and differentially validates
    /// the full outer link. This path never calls `exact_control_elimination`
    /// by construction (proven by the run counter); there is no global table
    /// object here at all.
    #[test]
    #[ignore]
    fn prepared_transfer_row918_no_global_elimination() {
        use crate::compiler::boundary_transfer::validate_shared_child_links;
        use crate::compiler::constraint_compose::build_segmented_parser_links;

        let fixture = load_selected10_outer();
        let children = [CompiledSubgrammarInput {
            placeholder_terminal: terminal_id(&fixture.core, "PROGRAMMATIC_TOOL_SUFFIX"),
            additional_placeholder_terminals: &[],
            constraint: &fixture.dispatch,
        }];
        let links = build_segmented_parser_links(&children).expect("segmented links");
        validate_shared_child_links(&links).expect("incoming links agree");
        assert_row918_local_transfers(&fixture, &links);
        // Full outer link through the NEW signed-transfer production route
        // (milestone 9): local transfers, Ready-port composition, exact
        // resolution, table-free normalization — and no control elimination.
        let elim_before = control_elimination_run_count();
        let link_output = build_walk_static_boundary_link(&WalkStaticLinkInputs {
            parent: &fixture.core,
            children: &children,
            vocab: &fixture.vocab,
            static_components: None,
            expected_terminal_offsets: &fixture.composed.table.terminal_offsets,
        })
        .expect("outer link compiles through the signed-transfer route");
        assert_eq!(
            control_elimination_run_count(),
            elim_before,
            "signed-transfer outer link must not invoke control elimination",
        );
        assert!(
            !link_output.all_dynamic,
            "outer link must take the static route",
        );
        for shard in &link_output.published_shards {
            assert!(
                link_output
                    .effective_static_components
                    .contains(shard.start_component as usize),
                "published shard {} must be effectively static",
                shard.start_component,
            );
        }
        eprintln!(
            "BOUNDARY_TRANSFER outer published={:?} tokens={:?}",
            link_output
                .published_shards
                .iter()
                .map(|shard| shard.start_component)
                .collect::<Vec<_>>(),
            link_output.boundary_tokens_by_start_component,
        );
        // Install the new-route shards over a dynamic composition and require
        // mask-for-mask equality: the outer shard must WORK, not just compile.
        let dyn_children = [CompiledSubgrammarInput {
            placeholder_terminal: terminal_id(&fixture.core, "PROGRAMMATIC_TOOL_SUFFIX"),
            additional_placeholder_terminals: &[],
            constraint: &fixture.dispatch,
        }];
        let dynamic = compose_constraints_owned_parent_segmented(
            fixture.core.clone(),
            &dyn_children,
            &fixture.vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("dynamic compose")
        .constraint;
        let mut static_comp = dynamic.clone();
        install_published_static_boundary_shards(
            static_comp.static_dynamic_overlay.as_mut().expect("overlay"),
            link_output.published_shards,
        )
        .expect("install");
        assert!(
            static_comp.uses_compact_segmented_parser_runtime(),
            "installed composition must stay on the compact runtime",
        );
        for (index, component) in static_comp
            .static_dynamic_overlay
            .as_ref()
            .expect("overlay")
            .segmented_parser_components
            .iter()
            .enumerate()
        {
            if let Some(shard) = component.boundary.as_ref() {
                assert!(
                    matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ),
                    "installed outer component {index} must be a StaticParser shard",
                );
            }
        }
        let component_state_counts =
            [fixture.core.tokenizer.num_states(), fixture.dispatch.tokenizer.num_states()];
        assert_scoped_tokenizer_packing("outer-signed", &static_comp, &component_state_counts);
        // Optional prepare: save/reload round-trip + pins, then return BEFORE
        // the in-process differential. Default path unchanged.
        if let Some(dir) = install_prepare_dir("ROW918_PREPARE_OUT_DIR") {
            let shard_ids: Vec<(u32, usize)> = static_comp
                .static_dynamic_overlay
                .as_ref()
                .expect("overlay")
                .segmented_parser_components
                .iter()
                .filter_map(|component| {
                    component.boundary.as_ref().map(|shard| {
                        (
                            shard.start_component,
                            shard.candidate_tokens.as_ref().map(|tokens| tokens.len()).unwrap_or(0),
                        )
                    })
                })
                .collect();
            save_reload_install_pair(
                "prepared_transfer_row918_no_global_elimination",
                &dir,
                &dynamic,
                &static_comp,
                &component_state_counts,
                "outer-signed",
                &fixture.vocab,
            );
            write_install_prepare_manifest(
                &dir,
                "prepared_transfer_row918_no_global_elimination",
                &shard_ids,
                &component_state_counts,
                &fixture.vocab,
            );
            return;
        }
        run_install_differential(&dynamic, &mut static_comp, false);
    }

    /// Divergent called-frame fixture through the PRODUCTION static link
    /// (milestones E/G/H at fixture scale).
    ///
    /// Nullable bound children are OUTSIDE the signed-transfer static
    /// linker's supported class (bounded flat closure certificate): silent
    /// Entry/Return episodes have no uniform bound, so the link declines
    /// loudly here instead of under-admitting. This pins that decline.
    #[test]
    fn nullable_static_link_declines_loudly() {
        let vocab = Vocab::new(vec![
            (0, b"L".to_vec()),
            (1, b"a".to_vec()),
            (2, b"x".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "L" SUB "x";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt item ::= "a";
                nt child ::= item?;
            "#,
            &vocab,
        )
        .unwrap();
        assert!(
            child.table.embedded_start_nullable(),
            "pin fixture must stay effectively nullable",
        );
        let inputs = [CompiledSubgrammarInput {
            placeholder_terminal: terminal_id(&parent, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &child,
        }];
        let composed = low_level_compose(&parent, &inputs);
        let error = match build_walk_static_boundary_link(&WalkStaticLinkInputs {
            parent: &parent,
            children: &inputs,
            vocab: &vocab,
            static_components: None,
            expected_terminal_offsets: &composed.table.terminal_offsets,
        }) {
            Err(error) => error,
            Ok(_) => panic!("nullable static links must decline loudly (general C* is future work)"),
        };
        assert!(
            error.contains("nullable"),
            "decline must name nullability, got: {error}",
        );
    }

    /// Divergent called-frame fixture through the PRODUCTION static link
    /// (milestones E/G/H at fixture scale).
    ///
    /// `build_walk_static_boundary_link` must install a real static shard that
    /// matches DynamicDirect at both call sites (caller-sensitive fused
    /// tokens), with no `DynamicDirect` backend anywhere in the installed
    /// overlay. Removing the child shard (ablation) must lose the fused
    /// tokens — a nonempty shard that never fires is not evidence. With
    /// `GLRMASK_STRICT_STATIC_TRAP_DYNAMIC=1`, masking the installed static
    /// composition must not trip the DynamicDirect trap.
    #[test]
    #[ignore]
    fn walk_static_link_divergent_fixture_matches_dynamic_and_ablates() {
        if crate::isolate_environment_test(true) { return; }
        fn admits(mask: &[u32], token: u32) -> bool {
            mask.get(token as usize / 32)
                .is_some_and(|word| word & (1u32 << (token % 32)) != 0)
        }

        // Panic-safe env restore: Drop removes the trap var even if an assert
        // below fails, so the trap cannot leak into other tests. The unsafe
        // blocks are sound because the caller holds TEST_ENV_LOCK, serializing
        // all process-env mutation in the test build.
        struct TrapGuard;
        impl TrapGuard {
            fn set() -> Self {
                unsafe {
                    std::env::set_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", "1");
                }
                Self
            }
        }
        impl Drop for TrapGuard {
            fn drop(&mut self) {
                unsafe {
                    std::env::remove_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC");
                }
            }
        }

        let vocab = Vocab::new(vec![
            (0, b"ax".to_vec()),
            (1, b"ay".to_vec()),
            (2, b"L".to_vec()),
            (3, b"R".to_vec()),
            (4, b"a".to_vec()),
            (5, b"x".to_vec()),
            (6, b"y".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "L" SUB "x" | "R" SUB "y";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let inputs = [CompiledSubgrammarInput {
            placeholder_terminal: terminal_id(&parent, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &child,
        }];
        let composed = low_level_compose(&parent, &inputs);
        // No control elimination anywhere on the new parser-side path: the
        // signed-transfer compiler links local tables only.
        let elim_before = control_elimination_run_count();
        let link_output = build_walk_static_boundary_link(&WalkStaticLinkInputs {
            parent: &parent,
            children: &inputs,
            vocab: &vocab,
            static_components: None,
            expected_terminal_offsets: &composed.table.terminal_offsets,
        })
        .expect("production static link on the divergent fixture");
        assert_eq!(
            control_elimination_run_count(),
            elim_before,
            "signed-transfer link must not invoke control elimination",
        );
        assert!(
            !link_output.all_dynamic,
            "fixture link must be static, not all-dynamic",
        );
        assert_eq!(
            link_output.published_shards.len(),
            1,
            "only the child shard crosses here",
        );
        assert_eq!(link_output.published_shards[0].start_component, 1);

        let dynamic = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("dynamic compose")
        .constraint;
        let mut installed = dynamic.clone();
        install_published_static_boundary_shards(
            installed.static_dynamic_overlay.as_mut().expect("overlay"),
            link_output
                .published_shards
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
        )
        .expect("install");
        // No DynamicDirect backend anywhere in the installed overlay.
        for (index, component) in installed
            .static_dynamic_overlay
            .as_ref()
            .expect("overlay")
            .segmented_parser_components
            .iter()
            .enumerate()
        {
            if let Some(shard) = component.boundary.as_ref() {
                assert!(
                    matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ),
                    "installed component {index} must be a StaticParser shard",
                );
            }
        }
        // Call-site-exact admission at both sites.
        for (commit, fused, sibling, site) in [(2u32, 0u32, 1u32, "L"), (3u32, 1u32, 0u32, "R")] {
            let mut st_dyn = dynamic.start();
            let mut st_static = installed.start();
            st_dyn.commit_token(commit).unwrap();
            st_static.commit_token(commit).unwrap();
            let mask_dyn = st_dyn.mask();
            let mask_static = st_static.mask();
            assert!(admits(&mask_dyn, fused), "dynamic admits fused {fused} after {site}");
            assert!(!admits(&mask_dyn, sibling), "dynamic rejects sibling {sibling} after {site}");
            assert_eq!(mask_static, mask_dyn, "static matches dynamic after {site}");
        }
        // Strict-static trap: no DynamicDirect evaluation on this path.
        // Masks are collected under the env lock, then the guard is dropped
        // (removing the var) before asserting, so a failure cannot leak the
        // trap into other tests.
        let trapped: Vec<(u32, Vec<u32>)> = {
            let _env_lock = crate::TEST_ENV_LOCK.lock().unwrap();
            let _trap = TrapGuard::set();
            let mut trapped = Vec::new();
            for (commit, fused, _) in [(2u32, 0u32, "L"), (3u32, 1u32, "R")] {
                let mut st = installed.start();
                st.commit_token(commit).unwrap();
                trapped.push((fused, st.mask()));
            }
            trapped
        };
        for (fused, mask) in &trapped {
            assert!(admits(mask, *fused), "static admits fused {fused} under the trap");
        }
        // Ablation: with the child shard removed, the fused tokens must go
        // missing (the shard is load-bearing, not redundant).
        let mut ablated = dynamic.clone();
        install_published_static_boundary_shards(
            ablated.static_dynamic_overlay.as_mut().expect("overlay"),
            Vec::new(),
        )
        .expect("install empty");
        for (commit, fused, site) in [(2u32, 0u32, "L"), (3u32, 1u32, "R")] {
            let mut st = ablated.start();
            st.commit_token(commit).unwrap();
            assert!(
                !admits(&st.mask(), fused),
                "ablated composition must lose fused token {fused} after {site}",
            );
        }
    }

    /// Strict-harness scope regression: the L2P strict baseline must inherit the
    /// caller's shard scope (seed filter, shared equivalence, crossing filter and
    /// TI/compaction options). A baseline rebuilt without them compares a
    /// shard-scoped candidate against a full-scope artifact and spuriously
    /// mismatches (selected10: state=0 token=0 word=[81], the first coordinate
    /// in deterministic enumeration order). Runs the divergent fixture's
    /// production link with the factor strict reference armed: without the
    /// scope inheritance this panics inside the shard-build strict compare;
    /// with it the link succeeds.
    #[test]
    fn strict_baseline_inherits_shard_scope_on_divergent_fixture() {
        let vocab = Vocab::new(vec![
            (0, b"ax".to_vec()),
            (1, b"ay".to_vec()),
            (2, b"L".to_vec()),
            (3, b"R".to_vec()),
            (4, b"a".to_vec()),
            (5, b"x".to_vec()),
            (6, b"y".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "L" SUB "x" | "R" SUB "y";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let inputs = [CompiledSubgrammarInput {
            placeholder_terminal: terminal_id(&parent, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &child,
        }];
        let composed = low_level_compose(&parent, &inputs);
        let link_output = {
            let _env_lock = crate::TEST_ENV_LOCK.lock().unwrap();
            let _strict = EnvVarGuard::set(
                "GLRMASK_L2P_FIRST_BYTE_VOCAB_FACTOR_STRICT_REFERENCE",
                "boundary_shard",
            );
            build_walk_static_boundary_link(&WalkStaticLinkInputs {
                parent: &parent,
                children: &inputs,
                vocab: &vocab,
                static_components: None,
                expected_terminal_offsets: &composed.table.terminal_offsets,
            })
            .expect("shard-scoped strict link on the divergent fixture")
        };
        assert!(
            !link_output.all_dynamic,
            "fixture link must be static, not all-dynamic",
        );
        assert_eq!(
            link_output.published_shards.len(),
            1,
            "only the child shard crosses here",
        );
    }

    /// Nested depth-2 end-to-end fixture through the nested signed-transfer
    /// link (production walk/compile/publish/install, strict trap, curated
    /// differential, light ablation).
    ///
    /// Shape: P ::= "L" SUB SUB "x" | "R" SUB SUB "y" (two adjacent call
    /// sites into the same mid child M); M ::= "m" SUB2 (leading visible
    /// terminal, trailing slot); G ::= "g". All links effectively
    /// nonnullable, acyclic, canonical. Fused vocabulary tokens span every
    /// adjacent cross-leaf terminal pair ("Lm", "Rm", "mg", "gm", "gx",
    /// "gy"); without them no walk path can cross components (single bytes
    /// never cross) and the shards would be vacuous.
    ///
    /// This fixture is end-to-end integration only: the 2H certificate
    /// (depth 2, bound 4) is asserted, but the R,R,E,E load-bearing proof
    /// lives in `boundary_transfer::nested_ready_depth_four_is_load_bearing`.
    /// Here the arbiter is a curated reachable-prefix differential of the
    /// installed StaticParser composition against DynamicDirect, run under
    /// the strict-static trap; a light empty-shard ablation confirms the
    /// installed shards are genuinely used.
    ///
    /// Link architecture under test: two top-level walks (parent commit with
    /// the parent-leaf crossing filter; whole-block commit with the mid-leaf
    /// crossing filter — a path survives iff it touches a terminal outside
    /// the filter leaf, so the block walk keeps every leaf-crossing gap
    /// including g→m), each compiled with the full 3-leaf signed context and
    /// installed on a nested composition whose mid+grandchild block is already
    /// static. The retained inner shard covers genuinely block-internal repair;
    /// the outer block shard covers outward completion/re-entry candidates and
    /// must coexist with that inner static coverage under the strict trap.
    #[test]
    fn nested_static_link_depth_two_chain_matches_dynamic_and_ablates() {
        if crate::isolate_environment_test(false) { return; }
        use crate::compiler::glr::parser::ScopedSubgrammarLink;

        fn admits(mask: &[u32], token: u32) -> bool {
            mask.get(token as usize / 32)
                .is_some_and(|word| word & (1u32 << (token % 32)) != 0)
        }

        // Panic-safe trap guard (mirrors the flat divergent fixture): Drop
        // removes the var even on failure. The whole test holds TEST_ENV_LOCK
        // so no foreign trap can leak into the dynamic reference phase.
        struct TrapGuard(bool);
        impl TrapGuard {
            fn set() -> Self {
                unsafe {
                    std::env::set_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", "1");
                }
                Self(true)
            }
            fn clear(&mut self) {
                if self.0 {
                    unsafe {
                        std::env::remove_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC");
                    }
                    self.0 = false;
                }
            }
        }
        impl Drop for TrapGuard {
            fn drop(&mut self) {
                self.clear();
            }
        }

        fn compile_publish_shard(
            ctx: &crate::compiler::boundary_transfer::SignedLinkContext<'_>,
            dwa: &DWA,
            id_map: &InternalIdMap,
            start: u32,
            num_terminals: u32,
            tok_offsets: &[u32],
            leaf_counts: &[u32],
        ) -> crate::compiler::constraint_compose::PublishedStaticBoundaryShard {
            let emitted = boundary_emitted_terminals(dwa, num_terminals as usize);
            let library =
                crate::compiler::boundary_transfer::build_fragment_library(ctx, &emitted, start)
                    .expect("nested fragment library");
            let compiled = crate::compiler::boundary_transfer::compile_signed_shard_parser(
                ctx, &library, dwa, id_map, start,
            )
            .expect("nested shard compile");
            let candidates: Arc<[u32]> = boundary_accepted_tokens(dwa, id_map)
                .into_iter()
                .collect::<Vec<_>>()
                .into();
            let work = WalkBoundaryShardWork {
                start_component: start,
                terminal_automaton: TerminalAutomaton::Dwa(dwa.clone()),
                id_map: id_map.clone(),
                candidate_tokens: candidates,
            };
            crate::compiler::boundary_transfer::publish_signed_shard(
                work,
                compiled,
                tok_offsets,
                leaf_counts,
            )
            .expect("nested shard publish")
            .0
        }

        fn install_nested(
            base: &Constraint,
            top_shards: Vec<crate::compiler::constraint_compose::PublishedStaticBoundaryShard>,
        ) -> Constraint {
            let mut installed = base.clone();
            assert_eq!(
                installed
                    .static_dynamic_overlay
                    .as_ref()
                    .expect("overlay")
                    .segmented_parser_components
                    .len(),
                2,
                "nested install needs the 2-component top overlay",
            );
            install_published_static_boundary_shards(
                installed.static_dynamic_overlay.as_mut().expect("overlay"),
                top_shards,
            )
            .expect("install nested shards");
            installed
        }

        // Token ids: 0 L, 1 R, 2 x, 3 y, 4 m, 5 g, 6 Lm, 7 Rm, 8 mg, 9 gm,
        // 10 gx, 11 gy.
        const L: u32 = 0;
        const R: u32 = 1;
        const X: u32 = 2;
        const Y: u32 = 3;
        const M: u32 = 4;
        const G: u32 = 5;
        const LM: u32 = 6;
        const RM: u32 = 7;
        const MG: u32 = 8;
        const GM: u32 = 9;
        const GX: u32 = 10;
        const GY: u32 = 11;

        let _env_lock = crate::TEST_ENV_LOCK.lock().unwrap();
        let vocab = Vocab::new(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
            (4, b"m".to_vec()),
            (5, b"g".to_vec()),
            (6, b"Lm".to_vec()),
            (7, b"Rm".to_vec()),
            (8, b"mg".to_vec()),
            (9, b"gm".to_vec()),
            (10, b"gx".to_vec()),
            (11, b"gy".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "L" SUB SUB "x" | "R" SUB SUB "y";
            "#,
            &vocab,
        )
        .unwrap();
        let mid = Constraint::from_glrm_grammar(
            r#"
                start m;
                t SUB2 ::= @token(998);
                nt m ::= "m" SUB2;
            "#,
            &vocab,
        )
        .unwrap();
        let grandchild = Constraint::from_glrm_grammar(
            r#"
                start g;
                nt g ::= "g";
            "#,
            &vocab,
        )
        .unwrap();
        assert!(
            !mid.table.embedded_start_nullable(),
            "nested fixture mid must stay effectively nonnullable",
        );
        assert!(
            !grandchild.table.embedded_start_nullable(),
            "nested fixture grandchild must stay effectively nonnullable",
        );
        assert!(parent.ignore_terminal.is_none());
        assert!(mid.ignore_terminal.is_none());
        assert!(grandchild.ignore_terminal.is_none());
        let sub_p = terminal_id(&parent, "SUB");
        let sub2_m = terminal_id(&mid, "SUB2");

        // Reference nesting through the production dynamic composer (nested
        // dynamic masking is proven by existing tests; it is the arbiter).
        let mid_inputs = [CompiledSubgrammarInput {
            placeholder_terminal: sub2_m,
            additional_placeholder_terminals: &[],
            constraint: &grandchild,
        }];
        let mid_dyn = compose_constraints_owned_parent_segmented(
            mid.clone(),
            &mid_inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("inner dynamic compose")
        .constraint;
        let mid_static = compose_constraints_owned_parent_segmented(
            mid.clone(),
            &mid_inputs,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("inner static compose")
        .constraint;
        assert!(
            mid_static
                .static_dynamic_overlay
                .as_ref()
                .is_some_and(|overlay| !overlay.segmented_boundary_shards.is_empty()),
            "inner static compose must install its boundary repair before nesting",
        );
        let outer_inputs = [CompiledSubgrammarInput {
            placeholder_terminal: sub_p,
            additional_placeholder_terminals: &[],
            constraint: &mid_dyn,
        }];
        let outer_dyn = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &outer_inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("outer dynamic compose")
        .constraint;
        let outer_mixed_inputs = [CompiledSubgrammarInput {
            placeholder_terminal: sub_p,
            additional_placeholder_terminals: &[],
            constraint: &mid_static,
        }];
        let outer_with_static_inner = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &outer_mixed_inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("outer dynamic compose over static inner block")
        .constraint;
        assert!(
            outer_with_static_inner
                .static_dynamic_overlay
                .as_ref()
                .and_then(|overlay| overlay.segmented_parser_components.get(1))
                .and_then(|component| component.constraint.static_dynamic_overlay.as_ref())
                .is_some_and(|overlay| !overlay.segmented_boundary_shards.is_empty()),
            "outer dynamic wrapper must preserve the child's static internal boundary repair",
        );
        assert!(
            outer_dyn.uses_compact_segmented_parser_runtime(),
            "nested dynamic composition must stay on the compact runtime",
        );
        assert!(
            mid_dyn.uses_compact_segmented_parser_runtime(),
            "nested inner composition must stay on the compact runtime",
        );

        // Leaf layout pins from the real compose machinery: leaves
        // [[0],[1,0],[1,1]] with back-to-back terminal ranges.
        let layout = outer_dyn
            .recursive_parser_layout()
            .expect("layout")
            .expect("compact layout");
        let paths: Vec<Vec<u32>> = layout
            .leaves
            .iter()
            .map(|leaf| leaf.component_path.clone())
            .collect();
        assert_eq!(
            paths,
            vec![vec![0], vec![1, 0], vec![1, 1]],
            "nested runtime leaf order must be parent, mid, grandchild",
        );
        let n_p = parent.table.num_terminals;
        let n_m = mid.table.num_terminals;
        let n_g = grandchild.table.num_terminals;
        let outer_overlay = outer_dyn.static_dynamic_overlay.as_ref().expect("overlay");
        assert_eq!(outer_overlay.segmented_parser_components.len(), 2);
        let mid_overlay = mid_dyn.static_dynamic_overlay.as_ref().expect("inner overlay");
        assert_eq!(mid_overlay.segmented_parser_components.len(), 2);
        let t0 = outer_overlay.segmented_parser_components[0].terminal_offset;
        let t1 = outer_overlay.segmented_parser_components[1].terminal_offset;
        assert_eq!((t0, t1), (0, n_p), "outer terminal layout must pack parent first");
        let leaf_offsets = vec![0, n_p, n_p + n_m];
        let num_terminals = n_p + n_m + n_g;

        // Expanded links with live return pops and nullability.
        let rp_mid = mid_dyn.composition_child_return_pop().expect("mid return pop");
        let rp_g = grandchild
            .composition_child_return_pop()
            .expect("grandchild return pop");
        assert!(!mid_dyn.composition_start_nullable().expect("mid null"));
        assert!(!grandchild.composition_start_nullable().expect("g null"));
        let links = vec![
            ScopedSubgrammarLink {
                parent_component: 0,
                slot_terminal: sub_p,
                child_component: 1,
                child_start: 0,
                return_pop: rp_mid,
                child_start_nullable: false,
            },
            ScopedSubgrammarLink {
                parent_component: 1,
                slot_terminal: sub2_m,
                child_component: 2,
                child_start: 0,
                return_pop: rp_g,
                child_start_nullable: false,
            },
        ];

        // Recursive machine splice (layout-only, never parser behavior):
        // inner splice first, then the inner composed table as one child.
        let mut splice1 = crate::compiler::glr::table::compose_subgrammar_tables_with_rules(
            &mid.table,
            mid.retained_table_rules().expect("mid rules"),
            None,
            &[SubgrammarTableInput {
                placeholder_terminal: sub2_m,
                additional_placeholder_terminals: &[],
                table: &grandchild.table,
                ignore_terminal: None,
                start_nullable: false,
            }],
            &[grandchild.retained_table_rules().expect("g rules")],
        )
        .expect("inner splice");
        eliminate_composed_runtime_controls(&mut splice1).expect("inner eliminate");
        let mut splice2 = crate::compiler::glr::table::compose_subgrammar_tables_with_rules(
            &parent.table,
            parent.retained_table_rules().expect("parent rules"),
            None,
            &[SubgrammarTableInput {
                placeholder_terminal: sub_p,
                additional_placeholder_terminals: &[],
                table: &splice1.table,
                ignore_terminal: None,
                start_nullable: false,
            }],
            &[splice1.table.rules.as_slice()],
        )
        .expect("outer splice");
        eliminate_composed_runtime_controls(&mut splice2).expect("outer eliminate");
        assert!(
            splice2.table.control_terminals.is_empty(),
            "nested splice must be control-free",
        );
        assert_eq!(
            splice1.terminal_offsets.as_slice(),
            &[0, n_m],
            "inner splice layout must pack mid then grandchild",
        );
        assert_eq!(
            splice2.terminal_offsets.as_slice(),
            &[0, n_p],
            "outer splice layout must pack parent then the mid block",
        );
        assert_eq!(
            splice2.table.num_terminals, num_terminals,
            "nested terminal domain must be exactly the three leaf ranges",
        );

        // Merged tokenizer over intact leaf tokenizers (production union).
        let tokenizer_inputs = vec![
            (parent.composition_tokenizer(), leaf_offsets[0]),
            (mid.composition_tokenizer(), leaf_offsets[1]),
            (grandchild.composition_tokenizer(), leaf_offsets[2]),
        ];
        let (mut merged, tok_offsets) =
            Tokenizer::disjoint_union_with_terminal_offsets(&tokenizer_inputs);
        assert_eq!(
            tok_offsets.first(),
            Some(&1),
            "merged tokenizer must place the fresh reset fan-out at state 0",
        );
        let c_p = parent.composition_tokenizer().num_states();
        let c_m = mid.composition_tokenizer().num_states();
        let c_g = grandchild.composition_tokenizer().num_states();
        let leaf_counts = vec![c_p, c_m, c_g];
        if merged.terminal_exprs().is_none() {
            let all = vec![&parent, &mid, &grandchild];
            if let Some(exprs) = merged_retained_terminal_exprs(
                &all,
                &leaf_offsets,
                num_terminals,
            ) {
                merged.restore_terminal_exprs(Some(exprs)).expect("restore exprs");
            }
        }

        // Grammar + follows over the spliced rules (production analysis).
        let mut terminal_names = parent.terminal_display_names.clone();
        terminal_names.extend(mid.terminal_display_names.iter().cloned());
        terminal_names.extend(grandchild.terminal_display_names.iter().cloned());
        assert_eq!(terminal_names.len(), num_terminals as usize);
        let augmented_start = splice2
            .table
            .rules
            .first()
            .map(|rule| rule.lhs)
            .expect("spliced rules");
        let grammar = AnalyzedGrammar::from_composed_rules(
            splice2.table.rules.clone(),
            num_terminals,
            terminal_names,
            splice2.table.nonterminal_display_names.clone(),
            augmented_start,
        );
        let disallowed = compute_disallowed_follows(&grammar);

        // Two top-level walks under immediate-component ownership: the parent
        // owns leaf 0 and the already-composed block owns leaves 1+2. Pure
        // m↔g crossings are internal to the block and must not become outer
        // boundary candidates; g→x/g→y still crosses the block boundary.
        let flat: Arc<[u32]> = Arc::from(tdwa::l1::build_flat_transition_table(&merged));
        let ownership = Arc::new(
            tdwa::scope::BoundaryOwnership::from_leaf_layout(
                &leaf_offsets,
                num_terminals,
                &[
                    tdwa::scope::ImmediateComponentId(0),
                    tdwa::scope::ImmediateComponentId(1),
                    tdwa::scope::ImmediateComponentId(1),
                ],
                2,
            )
            .expect("nested block ownership"),
        );
        let total_states = merged.num_states() as usize;
        let run_walk = |component_index: usize, commit: &[bool], retain_non_crossing: bool| {
            let scope = tdwa::scope::BoundaryAnalysisScope::new(
                tdwa::scope::InitialStateDomain::from_mask(
                    total_states,
                    commit.to_vec(),
                )
                .expect("nested initial-state domain"),
                merged
                    .deterministic_reset_states()
                    .into_iter()
                    .collect(),
                Arc::clone(&ownership),
                tdwa::scope::ImmediateComponentId(component_index as u32),
                !retain_non_crossing,
                None,
            )
            .expect("nested boundary scope");
            build_boundary_terminal_dwa(&BoundaryWalkInputs {
                merged_tokenizer: &merged,
                vocab: &vocab,
                grammar: &grammar,
                disallowed_follows: &disallowed,
                ignore_terminal: None,
                follow_transparent_ignores: None,
                scope: &scope,
                retain_non_crossing_paths: retain_non_crossing,
                flat_trans: Some(&flat),
            })
            .expect("nonempty-vocab nested walks must produce a DWA")
        };
        let commit_p = commit_states_for_component(&tok_offsets, c_p, 0, total_states);
        let mut commit_block = commit_states_for_component(&tok_offsets, c_m, 1, total_states);
        let commit_g = commit_states_for_component(&tok_offsets, c_g, 2, total_states);
        for (slot, extra) in commit_block.iter_mut().zip(commit_g.iter()) {
            *slot |= *extra;
        }
        let walk_p = run_walk(0, &commit_p, false);
        // The block can complete after `g` and immediately re-enter at the
        // parent's second SUB call before `m`. Both visible terminals are
        // block-owned, so the lexical ownership filter alone would miss GM.
        // Production narrows this conservative lane to the block's summary
        // candidates; this fixture uses the full tiny vocabulary deliberately.
        let walk_b = run_walk(1, &commit_block, true);
        let cand_p = boundary_accepted_tokens(&walk_p.dwa, &walk_p.id_map);
        let cand_b = boundary_accepted_tokens(&walk_b.dwa, &walk_b.id_map);
        assert!(
            !cand_p.is_empty(),
            "parent walk must cross (fused Lm/Rm expected)",
        );
        assert!(
            !cand_b.is_empty(),
            "block walk must cross (fused mg/gm/gx expected)",
        );
        assert!(
            cand_b.contains(&GM),
            "block walk must retain zero-width completion/re-entry token GM, got {cand_b:?}",
        );
        assert!(
            cand_b.contains(&GX) || cand_b.contains(&GY),
            "block walk must cover an outer-boundary fused token, got {cand_b:?}",
        );

        // Signed-transfer compilation over the full 3-leaf context. The 2H
        // certificate is asserted: depth-2 nonnullable links allow exactly 4
        // control events per gap.
        // Nested unbound-slot scan over intact leaves: a slot is bound iff
        // some link fills it, located by (parent leaf offset + local slot).
        // The flat helper assumes children are leaves (no bound check for
        // child specials), so the nested link generalizes it here.
        let bound_global: std::collections::BTreeSet<u32> = links
            .iter()
            .map(|link| {
                leaf_offsets[link.parent_component as usize]
                    .checked_add(link.slot_terminal)
                    .expect("bound slot offset overflow")
            })
            .collect();
        let mut unbound = std::collections::BTreeSet::new();
        for (constraint, base) in
            [&parent, &mid, &grandchild].into_iter().zip(leaf_offsets.iter())
        {
            for special in &constraint.special_token_terminals {
                let global = base
                    .checked_add(special.terminal_id)
                    .expect("slot terminal offset overflow");
                if bound_global.contains(&global) {
                    continue;
                }
                if constraint.is_late_grammar_placeholder_terminal(special.terminal_id)
                    || constraint.token_bytes_for_id(special.token_id).is_none()
                {
                    unbound.insert(global);
                }
            }
        }
        assert!(
            unbound.is_empty(),
            "nested fixture must bind every slot, got {unbound:?}",
        );
        let example_children = [CompiledSubgrammarInput {
            placeholder_terminal: sub_p,
            additional_placeholder_terminals: &[],
            constraint: &mid_dyn,
        }];
        let global_ignores = component_ignores_are_globally_erasable(&parent, &example_children);
        let signed_context =
            crate::compiler::boundary_transfer::build_signed_link_context_from_parts(
                vec![&parent.table, &mid.table, &grandchild.table],
                vec![None, None, None],
                links,
                &leaf_offsets,
                num_terminals,
                global_ignores,
                unbound,
            )
            .expect("nested signed context");
        assert_eq!(signed_context.closure.max_nesting_depth, 2);
        assert_eq!(signed_context.closure.max_controls_per_gap, 4);
        let shard_p = compile_publish_shard(
            &signed_context,
            &walk_p.dwa,
            &walk_p.id_map,
            0,
            num_terminals,
            &tok_offsets,
            &leaf_counts,
        );
        let shard_b = compile_publish_shard(
            &signed_context,
            &walk_b.dwa,
            &walk_b.id_map,
            1,
            num_terminals,
            &tok_offsets,
            &leaf_counts,
        );
        assert_eq!(
            (shard_p.start_component, shard_b.start_component),
            (0, 1),
            "nested link must publish exactly the parent and block shards",
        );

        // The manually published/ablated static shards need a genuine static
        // coordinator, including its seed-mask universe. The dynamic reference
        // above must stay provider-only rather than paying for unused static data.
        let outer_static_shell = compose_constraints_owned_parent_segmented(
            parent.clone(), &outer_mixed_inputs, &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        ).expect("static coordinator for nested publication fixture").constraint;
        let installed = install_nested(&outer_static_shell, vec![shard_p, shard_b]);
        for (index, component) in installed
            .static_dynamic_overlay
            .as_ref()
            .expect("overlay")
            .segmented_parser_components
            .iter()
            .enumerate()
        {
            let shard = component.boundary.as_ref().unwrap_or_else(|| {
                panic!("installed top component {index} must carry a static shard")
            });
            assert!(
                matches!(
                    shard.backend,
                    crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                ),
                "installed component {index} must be a StaticParser shard",
            );
        }
        let mid_installed = &installed
            .static_dynamic_overlay
            .as_ref()
            .expect("overlay")
            .segmented_parser_components[1]
            .constraint
            .static_dynamic_overlay
            .as_ref()
            .expect("mid overlay");
        assert!(
            mid_installed
                .segmented_parser_components
                .iter()
                .all(|component| component.boundary.as_ref().is_none_or(|shard| matches!(
                    shard.backend,
                    crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                ))),
            "mid overlay must not acquire a dynamic internal boundary shard",
        );
        assert!(
            !mid_installed.segmented_boundary_shards.is_empty(),
            "mid overlay static shard list must survive the outer install",
        );
        assert!(
            mid_installed.segmented_boundary_shards.iter().all(|shard| matches!(
                shard.backend,
                crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
            )),
            "every retained inner shard must stay static",
        );

        // Curated reachable-prefix corpus (fast): prefixes of the two
        // complete L/R strings plus fused-spelling entry points. Only fully
        // mask-admitted paths are kept. This covers root entry, mid entry,
        // grandchild entry, both returns, and both call sites without an
        // exhaustive state search.
        struct Node {
            path: Vec<u32>,
            mask: Vec<u32>,
        }
        let candidates: Vec<Vec<u32>> = vec![
            vec![],
            vec![L],
            vec![R],
            vec![L, M],
            vec![R, M],
            vec![LM],
            vec![RM],
            vec![L, M, G],
            vec![R, M, G],
            vec![LM, G],
            vec![RM, G],
            vec![L, M, G, M],
            vec![R, M, G, M],
            vec![L, M, G, M, G],
            vec![R, M, G, M, G],
            vec![L, M, G, M, G, X],
            vec![R, M, G, M, G, Y],
        ];
        let mut nodes = Vec::new();
        for path in &candidates {
            let mut st = outer_dyn.start();
            let mut ok = true;
            for &token in path {
                // Mask-gated stepping: the runtime treats committing a
                // non-admitted token whose bytes still advance the lexer as
                // a loud mismatch, so only mask-admitted steps extend a
                // corpus prefix (mirrors the production differential walk).
                if !admits(&st.mask(), token) {
                    ok = false;
                    break;
                }
                if st.commit_token(token).is_err() {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            let mask = st.mask();
            nodes.push(Node { path: path.clone(), mask });
        }
        assert!(
            nodes.len() >= 10,
            "nested corpus must cover real positions, got {}",
            nodes.len()
        );
        // The corpus must contain the deep gap sources on both sites.
        for key_path in [
            vec![L, M, G],
            vec![R, M, G],
            vec![L, M, G, M, G],
            vec![R, M, G, M, G],
        ] {
            assert!(
                nodes.iter().any(|node| node.path == key_path),
                "nested corpus must visit gap source {key_path:?}",
            );
        }

        // Phase 2 (trap armed): every recorded path replays identically on
        // the installed static composition. Any hidden dynamic fallback on
        // the static side panics here instead of contributing exact masks.
        let mut trap = TrapGuard::set();
        let mut mismatches = 0usize;
        for node in &nodes {
            let mut st = installed.start();
            for &token in &node.path {
                st.commit_token(token).expect("static replay");
            }
            let actual = st.mask();
            if actual != node.mask {
                eprintln!(
                    "[glrmask/test][nested_static_mismatch] path={:?} dynamic={:?} static={:?}",
                    node.path, node.mask, actual,
                );
                mismatches += 1;
            }
        }
        trap.clear();
        assert_eq!(mismatches, 0, "nested static must match dynamic on every path");

        // Readable call-site discrimination through two nesting levels. Note
        // the fused spellings overlap single-byte prefixes: gx (bytes g,x)
        // is admissible where g is still unconsumed, i.e. after Lmgm, not
        // after Lmgmg; gm is admissible after Lm; mg is admissible after L.
        for (prefix, fused, sibling, site) in [
            (vec![L, M, G, M], GX, GY, "L"),
            (vec![R, M, G, M], GY, GX, "R"),
        ] {
            let mut st = installed.start();
            for &token in &prefix {
                st.commit_token(token).unwrap();
            }
            let mask = st.mask();
            assert!(admits(&mask, fused), "nested admits {fused} after {site}mgm");
            assert!(!admits(&mask, sibling), "nested rejects {sibling} after {site}mgm");
        }
        {
            let mut st = installed.start();
            st.commit_token(L).unwrap();
            assert!(admits(&st.mask(), M), "nested admits m after L");
            assert!(admits(&st.mask(), MG), "nested admits mg after L");
            assert!(!admits(&st.mask(), RM), "nested rejects Rm after L");
        }
        {
            let mut st = installed.start();
            st.commit_token(LM).unwrap();
            assert!(admits(&st.mask(), GM), "nested admits gm after Lm");
            assert!(!admits(&st.mask(), GX), "nested rejects gx after Lm");
        }

        // Light ablation: empty shards must lose admissions (confirms the
        // installed shards are genuinely used, not redundant). Removing the
        // boundary can only remove admissions (component masks are
        // unchanged), so every ablated mask is a subset; at least one must
        // be strict or the fixture is vacuous. (The R,R,E,E load-bearing
        // control proof lives in
        // `boundary_transfer::nested_ready_depth_four_is_load_bearing`.)
        let ablated = install_nested(&outer_static_shell, Vec::new());
        // Lockstep walk: at every shared prefix the shard-less mask must be
        // a subset of the dynamic mask; a prefix where the reference admits
        // a token the ablated mask lacks (or a final-mask difference) proves
        // the installed shards are genuinely used. Mask-gated stepping keeps
        // the runtime commit oracle satisfied on the shard-less side.
        fn is_subset(a: &[u32], b: &[u32]) -> bool {
            a.iter().zip(b.iter()).all(|(x, y)| x & !y == 0) && a.len() <= b.len()
        }
        let mut strict = 0usize;
        for node in &nodes {
            let mut st_dyn = outer_dyn.start();
            let mut st_abl = ablated.start();
            assert!(
                is_subset(&st_abl.mask(), &st_dyn.mask()),
                "ablated masks must be subsets at path {:?}",
                Vec::<u32>::new(),
            );
            let mut diverged = false;
            for &token in &node.path {
                assert!(
                    admits(&st_dyn.mask(), token),
                    "corpus path must stay dynamic-admitted at {:?}",
                    node.path,
                );
                if !admits(&st_abl.mask(), token) {
                    diverged = true;
                    break;
                }
                st_dyn.commit_token(token).expect("dynamic replay");
                st_abl.commit_token(token).expect("ablated replay");
                assert!(
                    is_subset(&st_abl.mask(), &st_dyn.mask()),
                    "ablated masks must be subsets at path {:?}",
                    node.path,
                );
            }
            if !diverged {
                assert_eq!(
                    st_dyn.mask(),
                    node.mask,
                    "dynamic replay must reproduce the recorded mask",
                );
                if st_abl.mask() != node.mask {
                    diverged = true;
                }
            }
            if diverged {
                strict += 1;
            }
        }
        assert!(
            strict > 0,
            "nested ablation must lose admissions somewhere (fixture vacuous otherwise)",
        );
    }

    #[test]
    fn test_compute_conservative_follow_disallowed() {
        let mut disallowed = BTreeMap::new();
        let mut set0 = BitSet::new(5);
        set0.set(1);
        set0.set(2);
        disallowed.insert(0, set0);

        let mut set1 = BitSet::new(5);
        set1.set(2);
        disallowed.insert(1, set1);

        let mut transparent = BitSet::new(5);
        transparent.set(1);

        let conservative = compute_conservative_follow_disallowed(&disallowed, Some(&transparent)).unwrap();
        // terminal 1 was transparent, so key 1 is removed, and bit 1 is cleared from key 0
        assert!(!conservative.contains_key(&1));
        assert!(conservative.contains_key(&0));
        let set0_after = &conservative[&0];
        assert!(!set0_after.contains(1));
        assert!(set0_after.contains(2));
    }

    /// TEMPORARY diagnostic (untracked-candidate, pending cleanup/integration):
    /// count alive ORIGINAL model tokens on the final installed static
    /// boundary parser DWAs stored in an existing composed artifact.
    ///
    /// Reads the artifact path from `PARSER_ALIVE_BIN` (no recomposition),
    /// walks `static_dynamic_overlay.segmented_boundary_shards` (plus each
    /// retained component's own `boundary` shard, printed distinctly), and for
    /// every `StaticParser` backend with a `recursive_parser_dwa` runs the
    /// exact forward-support traversal of `accepted_original_tokens` over the
    /// FINAL parser DWA with that shard's OWN `internal_token_to_originals`
    /// map (never the host constraint map). Prints one JSON line per shard to
    /// stdout: `{bin_sha, start_component, scope, backend, parser_states,
    /// parser_trans, acyclic, unique_internals, unique_originals,
    /// candidate_tokens_len}`.
    #[test]
    #[ignore = "diagnostic: requires existing parser artifacts"]
    fn parser_alive_installed_shards() {
        use crate::ds::weight::{ScopedWeightOpCache, Weight};
        use std::collections::{BTreeSet, VecDeque};

        fn alive_support(
            dwa: &DWA,
            internal_to_originals: &[Vec<u32>],
        ) -> (BTreeSet<u32>, BTreeSet<u32>) {
            assert!(
                dwa.is_acyclic(),
                "parser-alive diagnostic expects an acyclic parser DWA"
            );
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
            assert_eq!(topo.len(), n, "parser DWA topo walk must visit every state");
            // Full (tsid, token) weights propagate through every edge/final
            // intersection before any original expansion: start/final/empty
            // edges and unreachable states handled by construction.
            let mut reach = vec![Weight::empty(); n];
            if n > 0 {
                reach[dwa.start_state() as usize] = Weight::all();
            }
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
            let mut internals = BTreeSet::new();
            let mut originals = BTreeSet::new();
            for (_, internal_tokens) in accepted.raw_range_values() {
                for range in internal_tokens.ranges() {
                    for internal_token in range {
                        if let Some(ids) =
                            internal_to_originals.get(internal_token as usize)
                        {
                            if ids.is_empty() {
                                continue;
                            }
                            internals.insert(internal_token);
                            originals.extend(ids.iter().copied());
                        }
                    }
                }
            }
            (internals, originals)
        }

        let path = std::env::var("PARSER_ALIVE_BIN")
            .expect("PARSER_ALIVE_BIN must name an existing composed .static.bin artifact");
        let bytes = std::fs::read(&path).expect("read artifact bytes");
        let digest = blake3::hash(&bytes);
        let bin_sha = digest.to_hex().to_string();
        let constraint = Constraint::load(bytes).expect("load composed artifact");
        let overlay = constraint
            .static_dynamic_overlay
            .as_ref()
            .expect("composed artifact must carry a static/dynamic overlay");
        let mut rows: Vec<String> = Vec::new();
        let mut report = |scope: String,
                          start_component: u32,
                          backend: &str,
                          parser: Option<&DWA>,
                          ito: Option<&[Vec<u32>]>,
                          candidates: Option<&[u32]>| {
            let (states, trans, acyclic, internals, originals) = match (parser, ito) {
                (Some(dwa), Some(map)) => {
                    let (i, o) = alive_support(dwa, map);
                    (
                        dwa.num_states(),
                        dwa.num_transitions(),
                        dwa.is_acyclic(),
                        i.len(),
                        o.len(),
                    )
                }
                _ => (0, 0, false, 0, 0),
            };
            rows.push(format!(
                "{{\"bin_sha\":\"{bin_sha}\",\"scope\":\"{scope}\",\"start_component\":{start_component},\"backend\":\"{backend}\",\"parser_states\":{states},\"parser_trans\":{trans},\"acyclic\":{acyclic},\"unique_internals\":{internals},\"unique_originals\":{originals},\"candidate_tokens_len\":{}}}",
                candidates.map_or(-1i64, |c| c.len() as i64),
            ));
        };
        for shard in &overlay.segmented_boundary_shards {
            let (backend, parser, ito) = match &shard.backend {
                crate::runtime::SegmentedBoundaryShardBackend::StaticParser(b) => (
                    "StaticParser",
                    b.recursive_parser_dwa.as_ref().or(Some(&b.parser_dwa)),
                    Some(b.internal_token_to_originals.as_slice()),
                ),
                crate::runtime::SegmentedBoundaryShardBackend::DynamicTerminalTrie(_) => {
                    ("DynamicTerminalTrie", None, None)
                }
                crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect => {
                    ("DynamicDirect", None, None)
                }
            };
            report(
                "overlay".to_string(),
                shard.start_component,
                backend,
                parser,
                ito,
                shard.candidate_tokens.as_deref(),
            );
        }
        for (index, component) in overlay.segmented_parser_components.iter().enumerate() {
            if let Some(shard) = component.boundary.as_ref() {
                let (backend, parser, ito) = match &shard.backend {
                    crate::runtime::SegmentedBoundaryShardBackend::StaticParser(b) => (
                        "StaticParser",
                        b.recursive_parser_dwa.as_ref().or(Some(&b.parser_dwa)),
                        Some(b.internal_token_to_originals.as_slice()),
                    ),
                    crate::runtime::SegmentedBoundaryShardBackend::DynamicTerminalTrie(_) => {
                        ("DynamicTerminalTrie", None, None)
                    }
                    crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect => {
                        ("DynamicDirect", None, None)
                    }
                };
                report(
                    format!("component[{index}]"),
                    shard.start_component,
                    backend,
                    parser,
                    ito,
                    shard.candidate_tokens.as_deref(),
                );
            }
        }
        for row in &rows {
            println!("{row}");
        }
        assert!(!rows.is_empty(), "artifact must carry at least one boundary shard");
    }
}

fn build_boundary_terminal_dwa_borrowed_query(inputs:&BoundaryWalkInputs)->Option<BoundaryWalkOutput>{
    let flat=inputs.flat_trans?;
    let prepared=super::boundary_query_view::prepare_borrowed(inputs.merged_tokenizer,inputs.vocab,inputs.scope,flat)?;
    let began=Instant::now();
    let (mapped,profile,certificate)=tdwa::build_scoped_boundary_borrowed_identity_with_certificate(
        inputs.merged_tokenizer,inputs.vocab,inputs.ignore_terminal,inputs.grammar,inputs.disallowed_follows,
        Arc::clone(flat),&prepared.scope,&prepared.original_to_view,&prepared.view_to_original)?;
    let local=BoundaryWalkInputs{merged_tokenizer:inputs.merged_tokenizer,vocab:inputs.vocab,grammar:inputs.grammar,
        disallowed_follows:inputs.disallowed_follows,ignore_terminal:inputs.ignore_terminal,
        follow_transparent_ignores:inputs.follow_transparent_ignores,scope:&prepared.scope,
        retain_non_crossing_paths:inputs.retain_non_crossing_paths,flat_trans:Some(flat)};
    let mut output=finalize_scoped_terminal_output(&local,mapped,profile,certificate,0.,began)?;
        let current=&output.id_map.tokenizer_states;
        let mut originals=vec![u32::MAX;prepared.original_to_view.len()];
        let mut groups=vec![Vec::new();current.num_internal_ids() as usize];
        for (raw,&keep) in inputs.scope.initial_states().keep_raw().iter().enumerate(){if keep{
            let compact=prepared.original_to_view[raw];assert_ne!(compact,u32::MAX);
            let class=current.original_to_internal[compact as usize];originals[raw]=class;
            if class!=u32::MAX{groups[class as usize].push(raw as u32);}
        }}
        let representatives=groups.iter().map(|g|g.first().copied().unwrap_or(u32::MAX)).collect();
        output.id_map.tokenizer_states=crate::compiler::stages::equiv_types::ManyToOneIdMap{
            original_to_internal:originals,internal_to_originals:groups,representative_original_ids:representatives,
        };
        // Runtime publication requires a total raw-state map even though this
        // shard can be queried only from its certified initial domain. Add one
        // fresh zero-language class, never reuse a productive continuation ID.
        // Prove that no accepted lexical point mentions the new ID before
        // assigning all omitted states to it; symbolic ALL is not a finite row.
        let dead_class=output.id_map.tokenizer_states.num_internal_ids();
        if !proves_fresh_dead_class(&output.dwa,dead_class,
            crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_ROOT_SUPPORT_BOUND")){
            return None;
        }
        output.id_map.tokenizer_states=output.id_map.tokenizer_states.fill_unmapped_with_new_class();
        output.profile.setup_ms+=prepared.footprint_ms+prepared.materialize_ms;
        output.profile.initial_states=inputs.scope.initial_states().len();
        if compose_profile_enabled(){eprintln!("[glrmask/profile][boundary_query_view] component={} raw_states={} view_states={} first_states={} reset_states={} steps={} footprint_ms={:.3} materialize_ms={:.3} compile_ms={:.3}",
            inputs.scope.start_component().0,inputs.merged_tokenizer.num_states(),prepared.view_to_original.len(),
            prepared.first_states,prepared.reset_states,prepared.state_steps,prepared.footprint_ms,
            prepared.materialize_ms,output.profile.walk_ms);}
    Some(output)
}
