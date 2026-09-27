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
use crate::compiler::composition::{
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
    prepared: Option<&crate::compiler::composition::boundary::precomputed_completion::PreparedSourceSpan>,
) -> Option<BoundaryWalkOutput> {
    let tile_size = crate::compiler::composition::boundary::env::bounded_usize(
        "GLRMASK_BOUNDARY_QUERY_TILE_SIZE", 32, 1, 4096,
    );
    let terminal_support=crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_QUERY_TERMINAL_SUPPORT");
    let query_view=crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_QUERY_VIEW");
    let identity = crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_IDENTITY_REFINEMENT");
    let first_started = Instant::now();
    let use_first = crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_FIRST_COMPLETION_REFINEMENT");
    let precomputed = (use_first
        && crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_PRECOMPUTED_COMPLETION"))
        .then(|| prepared.and_then(|span| span.refine(inputs.merged_tokenizer, inputs.vocab, inputs.scope)))
        .flatten();
    let precomputed_selected = precomputed.is_some();
    let first = precomputed.or_else(|| use_first.then(|| inputs.flat_trans.and_then(|flat|
        crate::compiler::composition::boundary::first_completion::prepare(inputs.merged_tokenizer, inputs.vocab, inputs.scope, flat))).flatten());
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
    let prepared = crate::compiler::composition::boundary::finite_lexer::prepare(
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
        crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_ROOT_SUPPORT_BOUND")) {
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
    if identity && query_view && crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_BORROWED_QUERY"){
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
        && let Some(prepared)=crate::compiler::composition::boundary::query_view::prepare_with_policy(inputs.merged_tokenizer,inputs.vocab,inputs.scope,inputs.flat_trans.map(|flat|flat.as_ref()))
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
            crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_ROOT_SUPPORT_BOUND")){
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
        let support=terminal_support.then(||crate::compiler::composition::boundary::query_terminals::query_support(
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
        && crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_REUSE_NATIVE_MINIMUM");
    let summarize_after=crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_SUMMARIZE_AFTER_MIN");
    let reference=(certified && std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_NATIVE_MINIMUM").is_some())
        .then(||finalize_boundary_terminal_dwa(dwa.clone(),&id_map,summarize_after));
    let (finalized, accepted_tokens, final_minimize_ms) = if certified {
        let accepted=if let Some(support)=native_fixed_point.as_ref().and_then(|proof|proof.accepted_weight()){
            if std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_CERTIFIED_ROOT_SUPPORT").is_some(){
                assert_eq!(support,&crate::compiler::composition::accepted_weight_support(&dwa),
                    "certified root support changed correlated output weights");
            }
            crate::compiler::composition::boundary::terminal_summary::project_accepted_tokens(support,&id_map)
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
    let support = crate::compiler::composition::accepted_weight_support(dwa);
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
    if crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_BATCHED_TOKEN_SUPPORT")
        && let Some(tokens) = crate::compiler::composition::boundary::terminal_summary::accepted_tokens(dwa, id_map)
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
    prepared_first: Option<&[Option<crate::compiler::composition::boundary::precomputed_completion::PreparedSourceSpan>]>,
) -> Result<Option<(Vec<(usize, R)>, BoundaryShardLinkProfile)>, String>
where
    R: Send,
    F: Fn(BuiltBoundaryShardWalk) -> Result<R, String> + Sync,
{
    use rayon::prelude::*;

    let flat_started = Instant::now();
    let make_original_flat=|| Arc::from(tdwa::l1::build_flat_transition_table(inputs.merged_tokenizer));
    let use_direct_flat=crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_DIRECT_FLAT_ARC");
    let prepared_flat=use_direct_flat.then(|| crate::compiler::composition::boundary::flat_transitions::build(inputs.merged_tokenizer,true)).flatten();
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
    let support_transitions = (crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_TOKEN_LIVENESS")
        && crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_TOKEN_FOLLOWS"))
        .then(|| crate::compiler::composition::boundary::token_support::PreparedSupportTransitions::new(
            inputs.merged_tokenizer, &flat)).flatten();
    if compose_profile_enabled() {
        eprintln!("[glrmask/profile][boundary_support_transition_setup] states={} ms={:.3}",
            inputs.merged_tokenizer.num_states(), support_setup_started.elapsed().as_secs_f64() * 1000.0);
    }
    let shared = crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_SHARED_COMPILE_CONTEXT")
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
            let prepared_support = crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_PRECOMPUTED_PREFIX_SEEDS")
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
            && crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_TOKEN_LIVENESS")
        {
            let started = Instant::now();
            let follow_aware = crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_TOKEN_FOLLOWS");
            let cut = (follow_aware && crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_CUT_SUPPORT"))
                .then(|| {
                    let prepared=crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_PREPARED_CUT")
                        .then_some(prepared_first).flatten().and_then(|spans|crate::compiler::composition::boundary::precomputed_completion::cut_support(
                            spans,inputs.merged_tokenizer,candidate_vocab,scope.initial_states().keep_raw(),
                            &ownership,plan.crossing_owner,inputs.disallowed_follows,
                            inputs.ignore_terminal,inputs.follow_transparent_ignores,early_adjacency,
                        ));
                    let ordinary=|| crate::compiler::composition::boundary::cut_support::crossing_support_with_transitions(
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
                    let reference = crate::compiler::composition::boundary::token_support::crossing_token_support_with_prepared_transitions(
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
                Some(crate::compiler::composition::boundary::token_support::CrossingTokenSupport {
                    tokens:cut.tokens, byte_prefixes:0, state_steps:cut.profile.raw_step_work, max_frontier:0,
                })
            } else if follow_aware {
                crate::compiler::composition::boundary::token_support::crossing_token_support_with_prepared_transitions(
                    inputs.merged_tokenizer, candidate_vocab, scope.initial_states().keep_raw(),
                    &ownership, plan.crossing_owner, inputs.disallowed_follows,
                    inputs.ignore_terminal, inputs.follow_transparent_ignores, early_adjacency,
                    support_transitions.as_ref(),
                )
            } else {
                crate::compiler::composition::boundary::token_support::crossing_token_support(
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
            && crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_REUSE_TOKEN_INVENTORY");
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
    tables.push(&parent.table);
    ignores.push((!global_ignores).then_some(parent.ignore_terminal).flatten());
    for child in children {
        tables.push(&child.constraint.table);
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
    let signed_context = crate::compiler::composition::boundary::transfer::build_signed_link_context(
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
    let prepared_first = crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_PRECOMPUTED_COMPLETION")
        .then(|| std::iter::once(parent)
            .chain(children.iter().map(|child| child.constraint))
            .zip(tokenizer_offsets.iter().copied())
            .zip(composed.terminal_offsets.iter().copied())
            .map(|((component, offset), terminal_offset)| crate::compiler::composition::boundary::precomputed_completion::PreparedSourceSpan::for_component(component, offset, terminal_offset))
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
                crate::compiler::composition::boundary::candidates::boundary_candidate_ids(component, vocab);
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
            if let Ok(refined) = crate::compiler::composition::boundary::tail::build_root_call_candidates(
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
    let scoped_adjacent = if crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_SCOPED_ADJACENCY")
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
            let mut relation = crate::compiler::composition::boundary::scoped_follow::scoped_follow_relation(&grammar,&counts,&labels);
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
        let use_delta=crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_SCOPED_FOLLOW_DELTA");
        let candidate=use_delta.then(|| crate::compiler::composition::boundary::scoped_follow_delta::scoped_ignore_follow_relation(&grammar,&counts,&labels)).flatten();
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
    let transfer_cache = crate::compiler::composition::boundary::transfer::FragmentTransferCache::new(
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
            let product = if crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_FOLLOW_ROW_QUOTIENT"){
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
        let library = crate::compiler::composition::boundary::transfer::build_fragment_library_cached(
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
        let compiled = crate::compiler::composition::boundary::transfer::compile_signed_shard_parser(
            &signed_context,
            &library,
            &shard.output.dwa,
            &shard.output.id_map,
            start_component as u32,
        )?;
        let shard_parser_ms = parser_started.elapsed().as_secs_f64() * 1000.0;
        if let Some(reference) = adjacency_reference {
            let terms = boundary_emitted_terminals(&reference, grammar.num_terminals as usize);
            let reference_library = crate::compiler::composition::boundary::transfer::build_fragment_library_cached(
                &signed_context,&transfer_cache,&terms,start_component as u32,
            )?;
            let reference = crate::compiler::composition::boundary::transfer::compile_signed_shard_parser(
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
        let (published, publish_profile) = crate::compiler::composition::boundary::transfer::publish_signed_shard(
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
        crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_SCOPED_ADJACENCY_EARLY").then_some(scoped_adjacent.as_ref()).flatten(),
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
    let signed_context = crate::compiler::composition::boundary::transfer::build_signed_link_context_from_parts(
        leaves.iter().map(|leaf| &leaf.table).collect(),
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
                crate::compiler::composition::boundary::candidates::boundary_candidate_ids(component, vocab);
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
    let transfer_cache = crate::compiler::composition::boundary::transfer::FragmentTransferCache::new(
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
        let library = crate::compiler::composition::boundary::transfer::build_fragment_library_cached(
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
        let compiled = crate::compiler::composition::boundary::transfer::compile_signed_shard_parser(
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
            crate::compiler::composition::boundary::transfer::publish_signed_shard(
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
mod tests ;

fn build_boundary_terminal_dwa_borrowed_query(inputs:&BoundaryWalkInputs)->Option<BoundaryWalkOutput>{
    let flat=inputs.flat_trans?;
    let prepared=crate::compiler::composition::boundary::query_view::prepare_borrowed(inputs.merged_tokenizer,inputs.vocab,inputs.scope,flat)?;
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
            crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_ROOT_SUPPORT_BOUND")){
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
