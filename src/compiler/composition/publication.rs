//! Validate and publish boundary work as runnable parser shards.

use crate::compiler::stages::mapped_artifact::WeightRefs;
use super::{
    AnalyzedGrammar, Arc, BTreeMap, DWA, FxHashMap, Instant, InternalIdMap, ManyToOneIdMap,
    MappedArtifact, NWA, PrebuiltParserBundleCache, SmallBoundaryDwa, SmallVec, Templates,
    TerminalAutomaton, UnweightedDfa, VecDeque, Vocab, Weight,
    boundary_parser_minimize_min_states,
    build_parser_dwa_from_terminal_dwa_with_precomputed_templates,
    build_parser_nwa_from_terminal_dwa_with_precomputed_templates,
    build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count,
    build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count_no_table,
    build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count_no_table_with_bundle_cache,
    build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count_nondeterministic_bundles,
    characterize_selected_terminals_for_terminal_count, compose_profile_enabled, find_difference,
    is_negative_label, macro_parallelism_disabled, minimize_owned,
    normalize_signed_weighted_parser_stack_nwa_small_boundary,
    normalize_signed_weighted_parser_stack_nwa_small_boundary_for_parser_state_count,
    normalize_weighted_parser_stack_nwa, normalize_weighted_parser_stack_nwa_small_boundary,
    normalize_weighted_parser_stack_nwa_small_boundary_compact_for_parser_state_count,
    normalize_weighted_parser_stack_nwa_small_boundary_for_parser_state_count,
    normalize_weighted_parser_stack_nwa_small_boundary_with_tsid_map,
    parser_builder_skips_internal_minimization, remap_weights_with_maps,
    report_macro_item_timings, resolve_negative_codes_in_nwa,
    resolve_negative_codes_small_boundary, reverse_hashcons_owned,
    reverse_hashcons_positive_acyclic_nwa,
    reverse_hashcons_positive_acyclic_nwa_fast_with_reverse_topo,
    reverse_hashcons_signed_acyclic_nwa_fast,
    specialize_template_dfa_defaults_for_commit_split_input, try_split_commit_template_dfas,
};
use rayon::prelude::*;

pub(super) type PossibleMatches = BTreeMap<u32, Weight>;

pub(super) struct BoundaryRepair {
    pub(super) parser: BoundaryParserWork,
    /// Complete exact static partition of B by the component in which model-
    /// token execution starts. `Some` means the partition is authoritative;
    /// `None` means this compiler path did not construct one and runtime must
    /// retain the established global/fallback boundary representation.
    pub(super) static_boundary_shards: Option<Vec<BoundaryShardWork>>,
    pub(super) template_dfas_by_terminal: Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
    pub(super) commit_templates_deferred: bool,
    pub(super) composition_parser_templates_by_terminal: Vec<Option<UnweightedDfa>>,
    pub(super) active_terminals: Vec<bool>,
    /// Conservative first-internal-crossing token summaries indexed by the
    /// component in which model-token execution starts.  These are derived
    /// from the same exact boundary witness discovery used to build B; they
    /// are accelerator metadata only and are never required for correctness.
    pub(super) boundary_tokens_by_start_component: Option<Vec<Vec<u32>>>,
}

pub(super) struct BoundaryShardWork {
    pub(super) start_component: u32,
    pub(super) candidate_tokens: Arc<[u32]>,
    pub(super) parser: BoundaryParserWork,
}

pub(super) fn build_commit_templates_from_raw_templates(
    raw_templates: &[Option<UnweightedDfa>],
) -> Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>> {
    let build = |(terminal, dfa): (usize, &Option<UnweightedDfa>)| {
            let dfa = dfa.as_ref()?;
            let commit_dfa = specialize_template_dfa_defaults_for_commit_split_input(dfa);
            try_split_commit_template_dfas(&commit_dfa)
                .map(|split| (terminal, Arc::new(split)))
    };
    let split_templates = if macro_parallelism_disabled() {
        let mut timings = Vec::new();
        let result = raw_templates
            .iter()
            .enumerate()
            .filter_map(|item| {
                let started = Instant::now();
                let result = build(item);
                timings.push(started.elapsed().as_secs_f64() * 1000.0);
                result
            })
            .collect::<Vec<_>>();
        report_macro_item_timings("compose_commit_template_splits", &timings);
        result
    } else {
        raw_templates
            .par_iter()
            .enumerate()
            .filter_map(build)
            .collect::<Vec<_>>()
    };
    let mut result = vec![None; raw_templates.len()];
    for (terminal, split) in split_templates {
        result[terminal] = Some(split);
    }
    result
}

pub(super) fn defer_boundary_commit_templates() -> bool {
    std::env::var_os("GLRMASK_DISABLE_DEFER_BOUNDARY_COMMIT_TEMPLATES").is_none()
}

pub(super) enum BoundaryParserWork {
    Materialized(MappedArtifact<DWA>),
    Deferred {
        terminal_automaton: TerminalAutomaton,
        id_map: InternalIdMap,
        analyzed: AnalyzedGrammar,
        templates: Templates,
    },
    /// Exact capture/precomputed boundary path after the composed GLR table is
    /// already available. Parser-NWA construction needs only the terminal
    /// domain size, not nullable/FIRST/FOLLOW analysis.
    DeferredTerminalCount {
        terminal_automaton: TerminalAutomaton,
        id_map: InternalIdMap,
        num_terminals: u32,
        templates: Templates,
        prebuilt_bundle_cache: Option<PrebuiltParserBundleCache>,
        /// Optional exact parser-table view used only to compile this static
        /// boundary accelerator.  Authoritative compositions keep linker
        /// controls in their live table, but a static B can compile against an
        /// exact control-eliminated clone without changing the runtime parser
        /// coordinate or semantics.
        parser_table_override: Option<Arc<crate::compiler::glr::table::GLRTable>>,
    },
}

pub(super) enum PositiveBoundaryParser {
    Dwa(DWA),
    Nwa(NWA),
}


pub(super) struct BoundaryTerminalTrieWork {
    pub(super) nodes: Vec<crate::runtime::BoundaryTerminalTrieNode>,
    pub(super) root_by_tsid: Vec<u32>,
    pub(super) symbolic_nwa: Option<crate::runtime::BoundaryTerminalNwa>,
}

pub(super) enum BoundaryRuntimeCandidate {
    Parser {
        positive: PositiveBoundaryParser,
        id_map: InternalIdMap,
        template_cache: Option<Vec<Option<UnweightedDfa>>>,
        build_ms: f64,
    },
    TerminalTrie {
        trie: BoundaryTerminalTrieWork,
        id_map: InternalIdMap,
        template_cache: Option<Vec<Option<UnweightedDfa>>>,
        build_ms: f64,
    },
}

pub(super) enum PublishedBoundaryRuntime {
    Parser {
        parser_dwa: DWA,
        id_map: InternalIdMap,
        template_cache: Option<Vec<Option<UnweightedDfa>>>,
        positive_build_ms: f64,
        normalize_ms: f64,
        tsid_quotient: Option<Vec<u32>>,
    },
    CompactParser {
        parser_dwa: SmallBoundaryDwa,
        id_map: InternalIdMap,
        template_cache: Option<Vec<Option<UnweightedDfa>>>,
        positive_build_ms: f64,
        normalize_ms: f64,
        tsid_quotient: Option<Vec<u32>>,
    },
    TerminalTrie {
        trie: BoundaryTerminalTrieWork,
        id_map: InternalIdMap,
        template_cache: Option<Vec<Option<UnweightedDfa>>>,
        build_ms: f64,
    },
}

#[derive(Clone)]
pub(crate) struct PublishedStaticBoundaryShard {
    pub(crate) start_component: u32,
    pub(crate) candidate_tokens: Arc<[u32]>,
    pub(crate) boundary: Arc<crate::runtime::SegmentedBoundaryParser>,
}

pub(super) fn publish_static_boundary_shard_work(
    work: BoundaryShardWork,
    table: &crate::compiler::glr::table::GLRTable,
    composed_tsid_count: usize,
) -> Result<PublishedStaticBoundaryShard, String> {
    let effective_table = work.parser.parser_table_override().cloned();
    let (positive, id_map, _template_cache) = work.parser.materialize_positive_parser(table)?;
    positive.ensure_positive()?;
    if id_map.num_tsids() as usize != composed_tsid_count {
        return Err(format!(
            "static boundary shard {} TSID coordinate differs from composed constraint: shard={} composed={composed_tsid_count}",
            work.start_component,
            id_map.num_tsids(),
        ));
    }
    // Keep the component/common TSID coordinate exactly. Per-shard TSID
    // quotienting would require one large raw-state -> private-TSID vector per
    // shard; the composed constraint already owns the authoritative scalar map.
    let parser_dwa = positive.into_runtime_dwa(effective_table.as_deref().unwrap_or(table));
    ensure_positive_runtime_parser_dwa(&parser_dwa)?;
    Ok(PublishedStaticBoundaryShard {
        start_component: work.start_component,
        candidate_tokens: work.candidate_tokens,
        boundary: Arc::new(crate::runtime::SegmentedBoundaryParser {
            // Provider-native v25 static shards are compiled directly in the
            // recursive leaf coordinate. Keep the legacy/materialized slot
            // empty; v25 wire serialization persists `recursive_parser_dwa`.
            parser_dwa: DWA::new(0, 0),
            compact_parser_dwa: None,
            recursive_parser_dwa: Some(parser_dwa),
            uses_composed_tsid_coordinate: true,
            tokenizer_state_to_tsid: Vec::new(),
            internal_token_to_originals: id_map.vocab_tokens.internal_to_originals,
        }),
    })
}

/// Walk-built boundary shard work: a crossing terminal DWA over shard-local
/// TSIDs, published late (after the overlay exists) against the spliced
/// control-free boundary table. Produced by `boundary_walk` (link-shared
/// equivalence + seeded standard walks + NWA crossing filter), one per start
/// component with a nonempty crossing set. Components with empty crossings
/// get no shard (the runtime skips missing shards).
pub(crate) struct WalkBoundaryShardWork {
    pub(crate) start_component: u32,
    pub(crate) terminal_automaton: TerminalAutomaton,
    pub(crate) id_map: InternalIdMap,
    pub(crate) candidate_tokens: Arc<[u32]>,
}

/// Per-shard publish profile: template characterization + parser-DWA
/// materialization + runtime normalization, with output sizes.
pub(crate) struct WalkShardPublishProfile {
    pub(crate) templates_ms: f64,
    pub(crate) materialize_ms: f64,
    pub(crate) normalize_ms: f64,
    pub(crate) terms: usize,
    pub(crate) parser_states: usize,
    pub(crate) parser_trans: usize,
}

/// Publish one walk-built shard as a `StaticParser` boundary shard with
/// shard-local TSIDs (`uses_composed_tsid_coordinate = false` + the walk
/// id_map's private raw-state and token maps).
///
/// Templates are characterized fresh over the provider-materialized exact
/// boundary table (live leaf coordinate, unbound slots emptied — never the
/// dynamic path's control-bearing recursive table) for exactly the terminals
/// the crossing DWA emits (the same construction the discovery-built path
/// applies via `set_parser_table_override`); the parser is built with the
/// standard count-only constructor + runtime normalization.
///
/// Precondition: live state keys stay within the link-time union ranges (no
/// lazily-allocated virtual-residual tokenizer states — the private map only
/// covers link-time states). The caller falls back to dynamic shards when a
/// component tokenizer has a virtual residual runtime.
///
/// Coordinate contract: the installing runtime is compact segmented, whose
/// live tokenizer keys are leaf-packed scoped states (leaves back-to-back
/// from 0 with NO reset state) — not merged-union states (which insert a
/// fresh reset fan-out at 0). The private map is therefore indexed by the
/// scoped coordinate: `scoped(leaf i, local l)` maps to the TSID of
/// `merged[tokenizer_offsets[i] + l]`, with leaves in link-component order.
/// Indexing it by merged states instead shifts every lookup by the reset
/// state and silently misroutes queries (inner-6 over-admission +
/// inner-7 under-admission were both this bug).
pub(crate) fn publish_walk_boundary_shard_work(
    work: WalkBoundaryShardWork,
    boundary_table: &Arc<crate::compiler::glr::table::GLRTable>,
    tokenizer_offsets: &[u32],
    component_state_counts: &[u32],
) -> Result<(PublishedStaticBoundaryShard, WalkShardPublishProfile), String> {
    let num_terminals = boundary_table.num_terminals;
    let TerminalAutomaton::Dwa(ref crossing) = work.terminal_automaton else {
        return Err(format!(
            "walk boundary shard {} must carry a DWA terminal automaton",
            work.start_component,
        ));
    };
    let mut selected = vec![false; num_terminals as usize];
    for state in crossing.states() {
        for &label in state.transitions.keys() {
            if label >= 0
                && let Some(slot) = selected.get_mut(label as usize)
            {
                *slot = true;
            }
        }
    }
    let templates_started_at = Instant::now();
    let characterizations = characterize_selected_terminals_for_terminal_count(
        boundary_table,
        num_terminals,
        &selected,
    );
    let templates = Templates::from_characterizations(&characterizations);
    let templates_ms = templates_started_at.elapsed().as_secs_f64() * 1000.0;
    let parser_work = BoundaryParserWork::DeferredTerminalCount {
        terminal_automaton: work.terminal_automaton,
        id_map: work.id_map,
        num_terminals,
        templates,
        prebuilt_bundle_cache: None,
        parser_table_override: Some(Arc::clone(boundary_table)),
    };
    let materialize_started_at = Instant::now();
    let (positive, id_map, _template_cache) =
        parser_work.materialize_positive_parser(boundary_table)?;
    let materialize_ms = materialize_started_at.elapsed().as_secs_f64() * 1000.0;
    positive.ensure_positive()?;
    let normalize_started_at = Instant::now();
    let parser_dwa = positive.into_runtime_dwa(boundary_table);
    let normalize_ms = normalize_started_at.elapsed().as_secs_f64() * 1000.0;
    ensure_positive_runtime_parser_dwa(&parser_dwa)?;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_walk_shard_publish] start_component={} terms={} parser_states={} parser_trans={} templates_ms={templates_ms:.3} materialize_ms={materialize_ms:.3} normalize_ms={normalize_ms:.3}",
            work.start_component,
            selected.iter().filter(|slot| **slot).count(),
            parser_dwa.num_states(),
            parser_dwa.num_transitions(),
        );
    }
    finish_walk_shard_publication(
        work.start_component,
        work.candidate_tokens,
        parser_dwa,
        &id_map,
        WalkShardPublishProfile {
            templates_ms,
            materialize_ms,
            normalize_ms,
            terms: selected.iter().filter(|slot| **slot).count(),
            parser_states: 0,
            parser_trans: 0,
        },
        tokenizer_offsets,
        component_state_counts,
    )
}

/// Publish one signed-transfer boundary shard built WITHOUT any composed or
/// provider table: the caller supplies the already positively-normalized
/// parser DWA (control-aware signed-NWA composition + exact negative
/// resolution + table-free normalization). Publication into the runtime
/// `StaticParser` format (shard-local TSID map in the scoped tokenizer
/// coordinate) is shared with the table-built path.
pub(crate) fn publish_signed_boundary_shard_work(
    work: WalkBoundaryShardWork,
    parser_dwa: DWA,
    profile: WalkShardPublishProfile,
    tokenizer_offsets: &[u32],
    component_state_counts: &[u32],
) -> Result<(PublishedStaticBoundaryShard, WalkShardPublishProfile), String> {
    ensure_positive_runtime_parser_dwa(&parser_dwa)?;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_signed_shard_publish] start_component={} terms={} parser_states={} parser_trans={} templates_ms={:.3} materialize_ms={:.3} normalize_ms={:.3}",
            work.start_component,
            profile.terms,
            parser_dwa.num_states(),
            parser_dwa.num_transitions(),
            profile.templates_ms,
            profile.materialize_ms,
            profile.normalize_ms,
        );
    }
    finish_walk_shard_publication(
        work.start_component,
        work.candidate_tokens,
        parser_dwa,
        &work.id_map,
        profile,
        tokenizer_offsets,
        component_state_counts,
    )
}

/// Shared tail of walk-shard publication: builds the scoped-coordinate TSID
/// map and wraps the parser DWA as a `StaticParser` boundary shard.
pub(super) fn finish_walk_shard_publication(
    start_component: u32,
    candidate_tokens: Arc<[u32]>,
    parser_dwa: DWA,
    id_map: &InternalIdMap,
    mut profile: WalkShardPublishProfile,
    tokenizer_offsets: &[u32],
    component_state_counts: &[u32],
) -> Result<(PublishedStaticBoundaryShard, WalkShardPublishProfile), String> {
    // The private TSID map is indexed by the installing runtime's scoped
    // tokenizer coordinate (leaves back-to-back from 0, no reset state) and
    // must cover every scoped link-time state exactly once; gaps would
    // silently drop boundary contributions at runtime.
    if tokenizer_offsets.len() != component_state_counts.len() {
        return Err(format!(
            "walk boundary shard {start_component} tokenizer layout mismatch: {} offsets vs {} components",
            tokenizer_offsets.len(),
            component_state_counts.len(),
        ));
    }
    let total_scoped: usize =
        component_state_counts.iter().map(|&count| count as usize).sum();
    let mut tokenizer_state_to_tsid = Vec::with_capacity(total_scoped);
    for (component, &count) in component_state_counts.iter().enumerate() {
        let base = tokenizer_offsets[component] as usize;
        for local in 0..count as usize {
            let merged = base.checked_add(local).ok_or_else(|| {
                format!(
                    "walk boundary shard {start_component} scoped state (component {component}, local {local}) overflows",
                )
            })?;
            let tsid = id_map
                .tokenizer_states
                .original_to_internal
                .get(merged)
                .copied()
                .unwrap_or(u32::MAX);
            if tsid == u32::MAX {
                return Err(format!(
                    "walk boundary shard {start_component} scoped state (component {component}, local {local}) has no TSID (merged {merged})",
                ));
            }
            tokenizer_state_to_tsid.push(tsid);
        }
    }
    profile.parser_states = parser_dwa.num_states() as usize;
    profile.parser_trans = parser_dwa.num_transitions() as usize;
    Ok((
        PublishedStaticBoundaryShard {
            start_component,
            candidate_tokens,
            boundary: Arc::new(crate::runtime::SegmentedBoundaryParser {
                parser_dwa: DWA::new(0, 0),
                compact_parser_dwa: None,
                recursive_parser_dwa: Some(parser_dwa),
                uses_composed_tsid_coordinate: false,
                tokenizer_state_to_tsid,
                internal_token_to_originals: id_map.vocab_tokens.internal_to_originals.clone(),
            }),
        },
        profile,
    ))
}


pub(super) fn publish_real_boundary_parser_work(
    work: BoundaryParserWork,
    table: &crate::compiler::glr::table::GLRTable,
) -> Result<PublishedBoundaryRuntime, String> {
    let positive_started_at = Instant::now();
    let (mut positive, mut boundary_id_map, template_cache) =
        work.materialize_positive_parser(table)?;
    positive.ensure_positive()?;
    let positive_build_ms = positive_started_at.elapsed().as_secs_f64() * 1000.0;
    let tsid_quotient = positive.quotient_boundary_tsids(&mut boundary_id_map);
    let normalize_started_at = Instant::now();
    let parser_dwa = if std::env::var_os(
        "GLRMASK_EXPERIMENT_SMALL_BOUNDARY_WEIGHT_DETERMINIZER",
    )
    .is_some()
        && boundary_id_map.num_tsids() as usize <= 16
        && boundary_id_map.num_internal_tokens() as usize <= 64
    {
        positive.into_runtime_dwa_small_boundary(
            table,
            boundary_id_map.num_tsids() as usize,
            boundary_id_map.num_internal_tokens() as usize,
            tsid_quotient.as_deref(),
        )
    } else {
        positive.into_runtime_dwa(table)
    };
    ensure_positive_runtime_parser_dwa(&parser_dwa)?;
    let normalize_ms = normalize_started_at.elapsed().as_secs_f64() * 1000.0;
    Ok(PublishedBoundaryRuntime::Parser {
        parser_dwa,
        id_map: boundary_id_map,
        template_cache,
        positive_build_ms,
        normalize_ms,
        tsid_quotient,
    })
}


pub(super) fn publish_boundary_parser_candidate_for_state_count(
    candidate: BoundaryRuntimeCandidate,
    num_parser_states: u32,
) -> Result<PublishedBoundaryRuntime, String> {
    let BoundaryRuntimeCandidate::Parser {
        mut positive,
        id_map: mut boundary_id_map,
        template_cache,
        build_ms: positive_build_ms,
    } = candidate
    else {
        return Err("real boundary parser publication received non-parser candidate".into());
    };
    positive.ensure_positive()?;
    let tsid_quotient = positive.quotient_boundary_tsids(&mut boundary_id_map);
    let normalize_started_at = Instant::now();
    let small_coordinate = std::env::var_os(
        "GLRMASK_EXPERIMENT_SMALL_BOUNDARY_WEIGHT_DETERMINIZER",
    )
    .is_some()
        && boundary_id_map.num_tsids() as usize <= 16
        && boundary_id_map.num_internal_tokens() as usize <= 64;
    if !small_coordinate {
        // The early fully-published path is intended for the compact boundary
        // coordinate. Leave other representations on the established table-
        // based publication path rather than silently changing their policy.
        return Err("state-count-only boundary publication requires small-boundary determinizer".into());
    }

    if std::env::var_os("GLRMASK_EXPERIMENT_COMPACT_BOUNDARY_RUNTIME_DWA").is_some() {
        let parser_dwa = positive
            .compact_runtime_dwa_for_parser_state_count(
                num_parser_states,
                boundary_id_map.num_tsids() as usize,
                boundary_id_map.num_internal_tokens() as usize,
                tsid_quotient.as_deref(),
            )
            .ok_or_else(|| "compact boundary runtime DWA rejected small coordinate".to_string())?;
        if std::env::var_os("GLRMASK_VALIDATE_COMPACT_BOUNDARY_RUNTIME_DWA").is_some() {
            let materialized = parser_dwa.to_generic_dwa();
            ensure_positive_runtime_parser_dwa(&materialized)?;
            let reference = match &positive {
                PositiveBoundaryParser::Dwa(dwa) => dwa.clone(),
                PositiveBoundaryParser::Nwa(nwa) => {
                    normalize_weighted_parser_stack_nwa_small_boundary_for_parser_state_count(
                        num_parser_states,
                        nwa,
                        boundary_id_map.num_tsids() as usize,
                        boundary_id_map.num_internal_tokens() as usize,
                        tsid_quotient.as_deref(),
                    )
                }
            };
            let forward = find_difference(&materialized, &reference)
                .map_err(|error| format!("compact boundary DWA forward validation failed: {error}"))?;
            let reverse = find_difference(&reference, &materialized)
                .map_err(|error| format!("compact boundary DWA reverse validation failed: {error}"))?;
            assert!(forward.is_none() && reverse.is_none(), "compact boundary runtime DWA changed weighted language");
            eprintln!(
                "[glrmask/validate][compact_boundary_runtime_dwa] exact=true states={} transitions={} weights={}",
                parser_dwa.num_states(),
                parser_dwa.num_transitions(),
                parser_dwa.weights.len(),
            );
        }
        let normalize_ms = normalize_started_at.elapsed().as_secs_f64() * 1000.0;
        return Ok(PublishedBoundaryRuntime::CompactParser {
            parser_dwa,
            id_map: boundary_id_map,
            template_cache,
            positive_build_ms,
            normalize_ms,
            tsid_quotient,
        });
    }

    let parser_dwa = positive.into_runtime_dwa_small_boundary_for_parser_state_count(
        num_parser_states,
        boundary_id_map.num_tsids() as usize,
        boundary_id_map.num_internal_tokens() as usize,
        tsid_quotient.as_deref(),
    );
    ensure_positive_runtime_parser_dwa(&parser_dwa)?;
    let normalize_ms = normalize_started_at.elapsed().as_secs_f64() * 1000.0;
    Ok(PublishedBoundaryRuntime::Parser {
        parser_dwa,
        id_map: boundary_id_map,
        template_cache,
        positive_build_ms,
        normalize_ms,
        tsid_quotient,
    })
}

pub(super) fn compute_boundary_tsid_behavior_quotient(
    nwa: &NWA,
    old_tsid_count: usize,
    token_classes: usize,
) -> Option<(Vec<u32>, usize)> {
    if old_tsid_count <= 1 || token_classes == 0 || token_classes > 64 {
        return None;
    }
    let mut unique_ids = FxHashMap::<usize, u16>::default();
    let mut unique_weights = Vec::<Weight>::new();
    let mut register = |weight: &Weight| {
        if weight.is_full() || weight.is_empty() {
            return;
        }
        let key = weight.ptr_key();
        if unique_ids.contains_key(&key) {
            return;
        }
        let Ok(index) = u16::try_from(unique_weights.len()) else {
            return;
        };
        unique_ids.insert(key, index);
        unique_weights.push(weight.clone());
    };
    for state in nwa.states() {
        if let Some(weight) = state.final_weight.as_ref() {
            register(weight);
        }
        for (_, weight) in &state.epsilons {
            register(weight);
        }
        for branches in state.transitions.values() {
            for (_, weight) in branches {
                register(weight);
            }
        }
    }
    if unique_weights.len() > u16::MAX as usize {
        return None;
    }
    type Signature = SmallVec<[(u16, u64); 4]>;
    let mut signatures = vec![Signature::new(); old_tsid_count];
    for (weight_index, weight) in unique_weights.iter().enumerate() {
        for (start, end, tokens) in weight.range_entries() {
            if end as usize >= old_tsid_count {
                return None;
            }
            let mut mask = 0u64;
            for range in tokens.ranges() {
                for token in range {
                    if token as usize >= token_classes || token >= 64 {
                        return None;
                    }
                    mask |= 1u64 << token;
                }
            }
            if mask == 0 {
                continue;
            }
            for tsid in start..=end {
                signatures[tsid as usize].push((weight_index as u16, mask));
            }
        }
    }
    let mut classes = FxHashMap::<Signature, u32>::default();
    let mut old_to_new = vec![0u32; old_tsid_count];
    for (old_tsid, signature) in signatures.into_iter().enumerate() {
        let next = classes.len() as u32;
        let class = *classes.entry(signature).or_insert(next);
        old_to_new[old_tsid] = class;
    }
    Some((old_to_new, unique_weights.len()))
}



pub(super) fn compute_boundary_dwa_tsid_behavior_quotient(
    dwa: &DWA,
    old_tsid_count: usize,
    token_classes: usize,
) -> Option<(Vec<u32>, usize)> {
    if old_tsid_count <= 1 || token_classes == 0 || token_classes > 64 {
        return None;
    }
    let mut unique_ids = FxHashMap::<usize, u16>::default();
    let mut unique_weights = Vec::<Weight>::new();
    for weight in dwa.weight_refs() {
        if weight.is_full() || weight.is_empty() {
            continue;
        }
        let key = weight.ptr_key();
        if unique_ids.contains_key(&key) {
            continue;
        }
        let Ok(index) = u16::try_from(unique_weights.len()) else {
            return None;
        };
        unique_ids.insert(key, index);
        unique_weights.push(weight.clone());
    }
    type Signature = SmallVec<[(u16, u64); 4]>;
    let mut signatures = vec![Signature::new(); old_tsid_count];
    for (weight_index, weight) in unique_weights.iter().enumerate() {
        for (start, end, tokens) in weight.range_entries() {
            if end as usize >= old_tsid_count {
                return None;
            }
            let mut mask = 0u64;
            for range in tokens.ranges() {
                for token in range {
                    if token as usize >= token_classes || token >= 64 {
                        return None;
                    }
                    mask |= 1u64 << token;
                }
            }
            if mask == 0 {
                continue;
            }
            for tsid in start..=end {
                signatures[tsid as usize].push((weight_index as u16, mask));
            }
        }
    }
    let mut classes = FxHashMap::<Signature, u32>::default();
    let mut old_to_new = vec![0u32; old_tsid_count];
    for (old_tsid, signature) in signatures.into_iter().enumerate() {
        let next = classes.len() as u32;
        let class = *classes.entry(signature).or_insert(next);
        old_to_new[old_tsid] = class;
    }
    Some((old_to_new, unique_weights.len()))
}

pub(super) fn apply_boundary_tsid_quotient_to_dwa_and_id_map(
    dwa: &mut DWA,
    id_map: &mut InternalIdMap,
    old_to_new: &[u32],
) -> Option<()> {
    let old_tsid_count = id_map.num_tsids() as usize;
    if old_to_new.len() != old_tsid_count {
        return None;
    }
    let new_tsid_count = old_to_new.iter().copied().max().map_or(0, |v| v as usize + 1);
    if new_tsid_count == 0 || new_tsid_count >= old_tsid_count {
        return None;
    }
    let token_count = id_map.num_internal_tokens() as usize;
    let tsid_map = old_to_new.iter().map(|&class| vec![class]).collect::<Vec<_>>();
    let token_map = (0..token_count as u32).map(|token| vec![token]).collect::<Vec<_>>();
    {
        let mut weights = dwa.weight_refs_mut();
        remap_weights_with_maps(&mut weights, &tsid_map, &token_map, new_tsid_count);
    }
    let old_groups = std::mem::take(&mut id_map.tokenizer_states.internal_to_originals);
    let old_representatives = std::mem::take(&mut id_map.tokenizer_states.representative_original_ids);
    let mut new_groups = vec![Vec::<u32>::new(); new_tsid_count];
    let mut new_representatives = vec![u32::MAX; new_tsid_count];
    for old_tsid in 0..old_tsid_count {
        let class = old_to_new[old_tsid] as usize;
        if let Some(group) = old_groups.get(old_tsid) {
            new_groups[class].extend(group.iter().copied());
        }
        if new_representatives[class] == u32::MAX {
            new_representatives[class] = old_representatives
                .get(old_tsid)
                .copied()
                .or_else(|| new_groups[class].first().copied())
                .unwrap_or(old_tsid as u32);
        }
    }
    for group in &mut new_groups {
        group.sort_unstable();
        group.dedup();
    }
    for internal in &mut id_map.tokenizer_states.original_to_internal {
        if *internal != u32::MAX {
            *internal = *old_to_new.get(*internal as usize)?;
        }
    }
    id_map.tokenizer_states.internal_to_originals = new_groups;
    id_map.tokenizer_states.representative_original_ids = new_representatives;
    Some(())
}

pub(super) fn apply_boundary_tsid_quotient_to_nwa_and_id_map(
    nwa: &mut NWA,
    id_map: &mut InternalIdMap,
    old_to_new: &[u32],
) -> Option<()> {
    let old_tsid_count = id_map.num_tsids() as usize;
    if old_to_new.len() != old_tsid_count {
        return None;
    }
    let new_tsid_count = old_to_new.iter().copied().max().map_or(0, |v| v as usize + 1);
    if new_tsid_count == 0 || new_tsid_count >= old_tsid_count {
        return None;
    }
    let token_count = id_map.num_internal_tokens() as usize;
    let tsid_map = old_to_new.iter().map(|&class| vec![class]).collect::<Vec<_>>();
    let token_map = (0..token_count as u32).map(|token| vec![token]).collect::<Vec<_>>();
    {
        let mut weights = nwa.weight_refs_mut();
        remap_weights_with_maps(&mut weights, &tsid_map, &token_map, new_tsid_count);
    }

    // This quotient runs before component/boundary coordinate reconciliation,
    // so retain every raw representative belonging to the merged class. The
    // later refinement planner needs them to map every common TSID back to the
    // compact boundary class.
    let old_groups = std::mem::take(&mut id_map.tokenizer_states.internal_to_originals);
    let old_representatives =
        std::mem::take(&mut id_map.tokenizer_states.representative_original_ids);
    let mut new_groups = vec![Vec::<u32>::new(); new_tsid_count];
    let mut new_representatives = vec![u32::MAX; new_tsid_count];
    for old_tsid in 0..old_tsid_count {
        let class = old_to_new[old_tsid] as usize;
        if let Some(group) = old_groups.get(old_tsid) {
            new_groups[class].extend(group.iter().copied());
        }
        if new_representatives[class] == u32::MAX {
            new_representatives[class] = old_representatives
                .get(old_tsid)
                .copied()
                .or_else(|| new_groups[class].first().copied())
                .unwrap_or(old_tsid as u32);
        }
    }
    for group in &mut new_groups {
        group.sort_unstable();
        group.dedup();
    }
    for internal in &mut id_map.tokenizer_states.original_to_internal {
        if *internal != u32::MAX {
            *internal = *old_to_new.get(*internal as usize)?;
        }
    }
    id_map.tokenizer_states.internal_to_originals = new_groups;
    id_map.tokenizer_states.representative_original_ids = new_representatives;
    Some(())
}

impl PositiveBoundaryParser {
    pub(super) fn into_runtime_dwa(self, table: &crate::compiler::glr::table::GLRTable) -> DWA {
        match self {
            Self::Dwa(dwa) => dwa,
            Self::Nwa(nwa) => normalize_weighted_parser_stack_nwa(table, &nwa),
        }
    }

    /// Quotient the private boundary TSID coordinate by its exact
    /// behavior on every weight in the already-positive boundary NWA.
    ///
    /// Two TSIDs are equivalent iff every source weight assigns them the same
    /// boundary-token set. All subsequent parser-DWA normalization operations
    /// are pointwise union/intersection/difference over weights, so this is a
    /// congruence: normalizing after the quotient is exactly the quotient of
    /// normalizing before it.
    pub(super) fn quotient_boundary_tsids(
        &mut self,
        id_map: &mut InternalIdMap,
    ) -> Option<Vec<u32>> {
        if std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_TSID_QUOTIENT").is_none() {
            return None;
        }
        let Self::Nwa(nwa) = self else {
            return None;
        };
        let old_tsid_count = id_map.num_tsids() as usize;
        let token_classes = id_map.num_internal_tokens() as usize;
        if old_tsid_count <= 1 || token_classes == 0 || token_classes > 64 {
            return None;
        }
        let started_at = Instant::now();

        let signature_started_at = Instant::now();
        let (old_to_new, unique_source_weight_count) =
            compute_boundary_tsid_behavior_quotient(nwa, old_tsid_count, token_classes)?;
        let signature_ms = signature_started_at.elapsed().as_secs_f64() * 1000.0;
        let grouping_ms = 0.0;
        let new_tsid_count = old_to_new.iter().copied().max().map_or(0, |v| v as usize + 1);
        if new_tsid_count >= old_tsid_count {
            return None;
        }

        let remap_started_at = Instant::now();
        let defer_weight_remap = std::env::var_os(
            "GLRMASK_EXPERIMENT_SMALL_BOUNDARY_WEIGHT_DETERMINIZER",
        )
        .is_some();
        if !defer_weight_remap {
            let tsid_map = old_to_new
                .iter()
                .map(|&class| vec![class])
                .collect::<Vec<_>>();
            let token_map = (0..token_classes as u32)
                .map(|token| vec![token])
                .collect::<Vec<_>>();
            let mut weights = nwa.weight_refs_mut();
            remap_weights_with_maps(
                &mut weights,
                &tsid_map,
                &token_map,
                new_tsid_count,
            );
        }
        let remap_ms = remap_started_at.elapsed().as_secs_f64() * 1000.0;

        // The segmented runtime carries the exact raw-state -> private-TSID map
        // separately. The boundary InternalIdMap therefore needs only one raw
        // representative per quotient class here, not the union of all 16k
        // source representatives.
        let old_representatives =
            std::mem::take(&mut id_map.tokenizer_states.representative_original_ids);
        let mut new_representatives = vec![u32::MAX; new_tsid_count];
        for old_tsid in 0..old_tsid_count {
            let class = old_to_new[old_tsid] as usize;
            if new_representatives[class] == u32::MAX {
                new_representatives[class] = old_representatives
                    .get(old_tsid)
                    .copied()
                    .unwrap_or(old_tsid as u32);
            }
        }
        for internal in &mut id_map.tokenizer_states.original_to_internal {
            if *internal != u32::MAX {
                let Some(&class) = old_to_new.get(*internal as usize) else {
                    return None;
                };
                *internal = class;
            }
        }
        id_map.tokenizer_states.internal_to_originals = new_representatives
            .iter()
            .map(|&representative| vec![representative])
            .collect();
        id_map.tokenizer_states.representative_original_ids = new_representatives;

        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_boundary_tsid_quotient] old_tsids={} new_tsids={} unique_source_weights={} signature_ms={signature_ms:.3} grouping_ms={grouping_ms:.3} remap_ms={remap_ms:.3} total_ms={:.3}",
                old_tsid_count,
                new_tsid_count,
                unique_source_weight_count,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Some(old_to_new)
    }

    pub(super) fn into_runtime_dwa_small_boundary(
        self,
        table: &crate::compiler::glr::table::GLRTable,
        num_tsids: usize,
        num_tokens: usize,
        source_tsid_map: Option<&[u32]>,
    ) -> DWA {
        match self {
            Self::Dwa(dwa) => dwa,
            Self::Nwa(nwa) => {
                if let Some(source_tsid_map) = source_tsid_map {
                    normalize_weighted_parser_stack_nwa_small_boundary_with_tsid_map(
                        table,
                        &nwa,
                        num_tsids,
                        num_tokens,
                        source_tsid_map,
                    )
                } else {
                    normalize_weighted_parser_stack_nwa_small_boundary(
                        table,
                        &nwa,
                        num_tsids,
                        num_tokens,
                    )
                }
            }
        }
    }

    pub(super) fn into_runtime_dwa_small_boundary_for_parser_state_count(
        self,
        num_parser_states: u32,
        num_tsids: usize,
        num_tokens: usize,
        source_tsid_map: Option<&[u32]>,
    ) -> DWA {
        match self {
            Self::Dwa(dwa) => dwa,
            Self::Nwa(nwa) => normalize_weighted_parser_stack_nwa_small_boundary_for_parser_state_count(
                num_parser_states,
                &nwa,
                num_tsids,
                num_tokens,
                source_tsid_map,
            ),
        }
    }

    pub(super) fn compact_runtime_dwa_for_parser_state_count(
        &self,
        num_parser_states: u32,
        num_tsids: usize,
        num_tokens: usize,
        source_tsid_map: Option<&[u32]>,
    ) -> Option<SmallBoundaryDwa> {
        let Self::Nwa(nwa) = self else {
            return None;
        };
        normalize_weighted_parser_stack_nwa_small_boundary_compact_for_parser_state_count(
            nwa,
            num_parser_states,
            num_tsids,
            num_tokens,
            source_tsid_map,
        )
    }

    pub(super) fn ensure_positive(&self) -> Result<(), String> {
        match self {
            Self::Dwa(dwa) => ensure_positive_runtime_parser_dwa(dwa),
            Self::Nwa(nwa) => {
                for (source, state) in nwa.states().iter().enumerate() {
                    if let Some(&label) = state
                        .transitions
                        .keys()
                        .find(|&&label| is_negative_label(label))
                    {
                        return Err(format!(
                            "runtime boundary parser NWA contains negative/PUSH label {label} at state {source}"
                        ));
                    }
                }
                Ok(())
            }
        }
    }

}


pub(super) fn build_boundary_terminal_trie_work(
    terminal_automaton: &TerminalAutomaton,
    id_map: &InternalIdMap,
) -> Result<BoundaryTerminalTrieWork, String> {
    // Keep the acyclic weighted terminal machine compact. Expanding it into a
    // terminal-prefix trie is not viable in general: a small converging DAG can
    // encode exponentially many terminal sequences. Runtime evaluation takes
    // the product with the *current* parser frontier instead, where grammar
    // state prunes almost all of those paths; exceptional wide products fall
    // back to the unified dynamic evaluator.
    let owned_nwa;
    let nwa = match terminal_automaton {
        TerminalAutomaton::Dwa(dwa) => {
            owned_nwa = dwa.to_nwa();
            &owned_nwa
        }
        TerminalAutomaton::TokenDeterministicNwa(nwa)
        | TerminalAutomaton::EpsilonNwa(nwa) => nwa,
    };
    if id_map.num_internal_tokens() == 0 {
        return Err("runtime boundary terminal NWA requires a private token class".into());
    }
    if !nwa.is_acyclic() {
        return Err("runtime boundary terminal NWA requires an acyclic source automaton".into());
    }

    let state_count = nwa.states().len();
    let mut indegree = vec![0usize; state_count];
    for state in nwa.states() {
        for (&label, branches) in &state.transitions {
            if label < 0 {
                return Err(format!(
                    "boundary terminal NWA encountered non-terminal label {label}"
                ));
            }
            for (target, _) in branches {
                let Some(slot) = indegree.get_mut(*target as usize) else {
                    return Err(format!(
                        "boundary terminal NWA source references invalid state {target}"
                    ));
                };
                *slot += 1;
            }
        }
        for (target, _) in &state.epsilons {
            let Some(slot) = indegree.get_mut(*target as usize) else {
                return Err(format!(
                    "boundary terminal NWA source references invalid state {target}"
                ));
            };
            *slot += 1;
        }
    }
    for &start in nwa.start_states() {
        if start as usize >= state_count {
            return Err(format!(
                "boundary terminal NWA source references invalid start state {start}"
            ));
        }
    }
    let mut queue = VecDeque::new();
    for (state, &degree) in indegree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(state as u32);
        }
    }
    let mut topological_order = Vec::with_capacity(state_count);
    while let Some(state_id) = queue.pop_front() {
        topological_order.push(state_id);
        let state = &nwa.states()[state_id as usize];
        for branches in state.transitions.values() {
            for (target, _) in branches {
                indegree[*target as usize] -= 1;
                if indegree[*target as usize] == 0 {
                    queue.push_back(*target);
                }
            }
        }
        for (target, _) in &state.epsilons {
            indegree[*target as usize] -= 1;
            if indegree[*target as usize] == 0 {
                queue.push_back(*target);
            }
        }
    }
    if topological_order.len() != state_count {
        return Err("runtime boundary terminal NWA requires an acyclic source automaton".into());
    }

    let nodes = nwa
        .states()
        .iter()
        .map(|state| {
            let mut transitions = Vec::new();
            for (&label, branches) in &state.transitions {
                for (target, weight) in branches {
                    transitions.push(crate::runtime::BoundaryTerminalNwaTransition {
                        terminal: label as u32,
                        target: *target,
                        weight: weight.clone(),
                    });
                }
            }
            crate::runtime::BoundaryTerminalNwaNode {
                final_weight: state.final_weight.clone(),
                transitions,
                epsilons: state.epsilons.clone(),
            }
        })
        .collect::<Vec<_>>();
    if compose_profile_enabled() {
        let labeled_edges = nodes.iter().map(|node| node.transitions.len()).sum::<usize>();
        let epsilon_edges = nodes.iter().map(|node| node.epsilons.len()).sum::<usize>();
        eprintln!(
            "[glrmask/profile][boundary_terminal_nwa] states={} starts={} labeled_edges={} epsilon_edges={} tsids={} token_classes={}",
            nodes.len(),
            nwa.start_states().len(),
            labeled_edges,
            epsilon_edges,
            id_map.num_tsids(),
            id_map.num_internal_tokens(),
        );
    }
    Ok(BoundaryTerminalTrieWork {
        // v21 artifacts used an expanded trie. New in-memory builds carry only
        // the compact weighted DAG; the legacy fields remain for compatibility.
        nodes: Vec::new(),
        root_by_tsid: vec![u32::MAX; id_map.num_tsids() as usize],
        symbolic_nwa: Some(crate::runtime::BoundaryTerminalNwa {
            nodes,
            start_states: nwa.start_states().to_vec(),
            topological_order,
        }),
    })
}


impl BoundaryParserWork {
    pub(super) fn set_parser_table_override(
        &mut self,
        table: Arc<crate::compiler::glr::table::GLRTable>,
    ) -> Result<(), String> {
        match self {
            Self::DeferredTerminalCount {
                num_terminals,
                templates,
                parser_table_override,
                ..
            } => {
                if table.num_terminals != *num_terminals {
                    return Err(format!(
                        "parser-table override terminal domain mismatch: table={} boundary={}",
                        table.num_terminals, *num_terminals,
                    ));
                }
                // Parser templates encode stack effects, so they belong to the
                // parser-state coordinate just as much as the table itself.
                // Re-characterize exactly the terminal subset already selected
                // by boundary discovery; reusing materialized-table templates
                // with a recursive provider table would mix two stack alphabets.
                let mut selected = vec![false; *num_terminals as usize];
                for &terminal in templates.by_terminal.keys() {
                    let Some(slot) = selected.get_mut(terminal as usize) else {
                        return Err(format!(
                            "boundary template terminal {terminal} lies outside parser domain {num_terminals}"
                        ));
                    };
                    *slot = true;
                }
                let characterizations = characterize_selected_terminals_for_terminal_count(
                    &table,
                    *num_terminals,
                    &selected,
                );
                *templates = Templates::from_characterizations(&characterizations);
                *parser_table_override = Some(table);
                Ok(())
            }
            _ => Err(
                "parser-table override requires deferred count-only boundary work".to_owned(),
            ),
        }
    }

    pub(super) fn parser_table_override(
        &self,
    ) -> Option<&Arc<crate::compiler::glr::table::GLRTable>> {
        match self {
            Self::DeferredTerminalCount {
                parser_table_override,
                ..
            } => parser_table_override.as_ref(),
            _ => None,
        }
    }

    pub(super) fn materialize_terminal_trie(
        self,
    ) -> Result<(
        BoundaryTerminalTrieWork,
        InternalIdMap,
        Option<Vec<Option<UnweightedDfa>>>,
    ), String> {
        match self {
            Self::DeferredTerminalCount {
                terminal_automaton,
                id_map,
                num_terminals,
                templates,
                prebuilt_bundle_cache: _,
                parser_table_override: _,
            } => {
                let trie = build_boundary_terminal_trie_work(&terminal_automaton, &id_map)?;
                let mut template_cache = vec![None; num_terminals as usize];
                for (terminal, dfa) in templates.by_terminal {
                    if let Some(slot) = template_cache.get_mut(terminal as usize) {
                        *slot = Some(dfa);
                    }
                }
                Ok((trie, id_map, Some(template_cache)))
            }
            _ => Err("runtime boundary terminal trie requires deferred count-only boundary work".into()),
        }
    }

    pub(super) fn materialize_positive_parser(
        self,
        table: &crate::compiler::glr::table::GLRTable,
    ) -> Result<(
        PositiveBoundaryParser,
        InternalIdMap,
        Option<Vec<Option<UnweightedDfa>>>,
    ), String> {
        match self {
            Self::Materialized(parser) => {
                let (dwa, id_map) = parser.into_parts();
                Ok((PositiveBoundaryParser::Dwa(dwa), id_map, None))
            }
            Self::Deferred {
                terminal_automaton,
                id_map,
                analyzed,
                templates,
            } => {
                let Some(mut parser_nwa) = build_parser_nwa_from_terminal_dwa_with_precomputed_templates(
                    &terminal_automaton,
                    &analyzed,
                    &templates,
                    table,
                ) else {
                    // No parser stack can realize any discovered terminal path:
                    // the exact boundary repair language is empty, not invalid.
                    return Ok((
                        PositiveBoundaryParser::Dwa(DWA::new(
                            id_map.num_tsids(),
                            id_map.max_internal_token_id(),
                        )),
                        id_map,
                        None,
                    ));
                };
                resolve_negative_codes_in_nwa(
                    &mut parser_nwa,
                    table.construction
                        == crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
                );
                let parser_dwa = normalize_weighted_parser_stack_nwa(table, &parser_nwa);
                Ok((PositiveBoundaryParser::Dwa(parser_dwa), id_map, None))
            }
            Self::DeferredTerminalCount {
                mut terminal_automaton,
                mut id_map,
                num_terminals,
                templates,
                prebuilt_bundle_cache: _,
                parser_table_override,
            } => {
                let table = parser_table_override.as_deref().unwrap_or(table);
                if std::env::var_os("GLRMASK_EXPERIMENT_EARLY_BOUNDARY_TSID_QUOTIENT").is_some()
                    && let TerminalAutomaton::Dwa(dwa) = &mut terminal_automaton
                {
                    let old_tsids = id_map.num_tsids() as usize;
                    let token_classes = id_map.num_internal_tokens() as usize;
                    let started_at = Instant::now();
                    if let Some((old_to_new, unique_weights)) =
                        compute_boundary_dwa_tsid_behavior_quotient(dwa, old_tsids, token_classes)
                    {
                        let new_tsids = old_to_new.iter().copied().max().map_or(0, |v| v as usize + 1);
                        if new_tsids < old_tsids {
                            apply_boundary_tsid_quotient_to_dwa_and_id_map(dwa, &mut id_map, &old_to_new)
                                .expect("early boundary TSID quotient must apply to matching DWA/id-map coordinate");
                            if compose_profile_enabled() {
                                eprintln!(
                                    "[glrmask/profile][constraint_boundary_early_tsid_quotient] old_tsids={} new_tsids={} unique_terminal_weights={} total_ms={:.3}",
                                    old_tsids, new_tsids, unique_weights, started_at.elapsed().as_secs_f64() * 1000.0,
                                );
                            }
                        }
                    }
                }
                let build_started_at = Instant::now();
                let nondeterministic_bundles_requested = std::env::var_os(
                    "GLRMASK_EXPERIMENT_COMPILE_NONDETERMINISTIC_BUNDLES",
                )
                .is_some();
                // Preserving bundle nondeterminism avoids repeated local DFA
                // union work on wide boundary alphabets, but on small boundary
                // coordinates it creates more NWA support for the final
                // publication pass than it saves during bundle construction.
                // Key this on the boundary token coordinate itself: it is a
                // representation-size policy, not a grammar/layout heuristic.
                let nondeterministic_bundle_min_tokens = std::env::var(
                    "GLRMASK_COMPILE_NONDETERMINISTIC_BUNDLES_MIN_TOKENS",
                )
                .ok()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or(64);
                let compile_nondeterministic_bundles = nondeterministic_bundles_requested
                    && id_map.num_internal_tokens() >= nondeterministic_bundle_min_tokens;
                let parser_nwa = if compile_nondeterministic_bundles {
                    build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count_nondeterministic_bundles(
                        &terminal_automaton,
                        num_terminals,
                        &templates,
                        table,
                    )
                } else {
                    build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count(
                        &terminal_automaton,
                        num_terminals,
                        &templates,
                        table,
                    )
                };
                let Some(mut parser_nwa) = parser_nwa else {
                    let mut template_cache = vec![None; num_terminals as usize];
                    for (terminal, dfa) in templates.by_terminal {
                        if let Some(slot) = template_cache.get_mut(terminal as usize) {
                            *slot = Some(dfa);
                        }
                    }
                    return Ok((
                        PositiveBoundaryParser::Dwa(DWA::new(
                            id_map.num_tsids(),
                            id_map.max_internal_token_id(),
                        )),
                        id_map,
                        Some(template_cache),
                    ));
                };
                let build_ms = build_started_at.elapsed().as_secs_f64() * 1000.0;
                if std::env::var_os("GLRMASK_EXPERIMENT_SMALL_BOUNDARY_SIGNED_FUSED").is_some()
                    && id_map.num_tsids() as usize <= 16
                    && id_map.num_internal_tokens() as usize <= 64
                {
                    let fused_started_at = Instant::now();
                    let parser_dwa = normalize_signed_weighted_parser_stack_nwa_small_boundary(
                        table,
                        &parser_nwa,
                        id_map.num_tsids() as usize,
                        id_map.num_internal_tokens() as usize,
                    )
                    .ok_or_else(|| "small-boundary fused signed publication rejected boundary coordinate".to_string())?;
                    let fused_ms = fused_started_at.elapsed().as_secs_f64() * 1000.0;
                    if compose_profile_enabled() {
                        eprintln!(
                            "[glrmask/profile][constraint_segmented_boundary_parser_phases] build_nwa_ms={build_ms:.3} fused_signed_publish_ms={fused_ms:.3} deterministic_runtime=true compile_nondeterministic_bundles={compile_nondeterministic_bundles}",
                        );
                    }
                    let mut template_cache = vec![None; num_terminals as usize];
                    for (terminal, dfa) in templates.by_terminal {
                        if let Some(slot) = template_cache.get_mut(terminal as usize) {
                            *slot = Some(dfa);
                        }
                    }
                    return Ok((
                        PositiveBoundaryParser::Dwa(parser_dwa),
                        id_map,
                        Some(template_cache),
                    ));
                }
                let resolve_started_at = Instant::now();
                let allow_grouped_cancellation = table.construction
                    == crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged;
                let resolved_reverse_topo = if std::env::var_os(
                    "GLRMASK_EXPERIMENT_SMALL_BOUNDARY_SIGNED_RESOLUTION",
                )
                .is_some()
                    && id_map.num_tsids() as usize <= 16
                    && id_map.num_internal_tokens() as usize <= 64
                {
                    parser_nwa = resolve_negative_codes_small_boundary(
                        &parser_nwa,
                        id_map.num_tsids() as usize,
                        id_map.num_internal_tokens() as usize,
                    )
                    .ok_or_else(|| {
                        "small-boundary signed resolver rejected boundary coordinate".to_string()
                    })?;
                    None
                } else {
                    resolve_negative_codes_in_nwa(&mut parser_nwa, allow_grouped_cancellation)
                };
                let resolve_negative_ms =
                    resolve_started_at.elapsed().as_secs_f64() * 1000.0;
                let hashcons_started_at = Instant::now();
                let before_hashcons_states = parser_nwa.num_states();
                let before_hashcons_transitions = parser_nwa.num_transitions();
                if std::env::var_os("GLRMASK_EXPERIMENT_FAST_HASHCONS_RESOLVED_BOUNDARY_NWA")
                    .is_some()
                {
                    parser_nwa = reverse_hashcons_positive_acyclic_nwa_fast_with_reverse_topo(
                        parser_nwa,
                        resolved_reverse_topo,
                    );
                } else if std::env::var_os("GLRMASK_EXPERIMENT_HASHCONS_RESOLVED_BOUNDARY_NWA")
                    .is_some()
                {
                    parser_nwa = reverse_hashcons_positive_acyclic_nwa(parser_nwa);
                }
                let hashcons_ms = hashcons_started_at.elapsed().as_secs_f64() * 1000.0;
                if compose_profile_enabled() {
                    eprintln!(
                        "[glrmask/profile][constraint_segmented_boundary_parser_phases] build_nwa_ms={build_ms:.3} resolve_negative_ms={resolve_negative_ms:.3} hashcons_ms={hashcons_ms:.3} hashcons_states={}=>{} hashcons_transitions={}=>{} determinize_dwa_ms=deferred deterministic_runtime=false positive_nwa=true compile_nondeterministic_bundles={compile_nondeterministic_bundles}",
                        before_hashcons_states,
                        parser_nwa.num_states(),
                        before_hashcons_transitions,
                        parser_nwa.num_transitions(),
                    );
                }
                let mut template_cache = vec![None; num_terminals as usize];
                for (terminal, dfa) in templates.by_terminal {
                    if let Some(slot) = template_cache.get_mut(terminal as usize) {
                        *slot = Some(dfa);
                    }
                }
                Ok((
                    PositiveBoundaryParser::Nwa(parser_nwa),
                    id_map,
                    Some(template_cache),
                ))
            }
        }
    }

    pub(super) fn materialize_positive_parser_without_table(
        self,
        allow_grouped_cancellation: bool,
        num_parser_states: u32,
    ) -> Result<BoundaryRuntimeCandidate, String> {
        let Self::DeferredTerminalCount {
            mut terminal_automaton,
            mut id_map,
            num_terminals,
            templates,
            prebuilt_bundle_cache,
            parser_table_override,
        } = self
        else {
            return Err(
                "table-free boundary parser construction requires deferred count-only work".into(),
            );
        };
        let started_at = Instant::now();
        if let Some(table) = parser_table_override {
            let work = Self::DeferredTerminalCount {
                terminal_automaton,
                id_map,
                num_terminals,
                templates,
                prebuilt_bundle_cache,
                parser_table_override: None,
            };
            let (positive, id_map, template_cache) =
                work.materialize_positive_parser(table.as_ref())?;
            return Ok(BoundaryRuntimeCandidate::Parser {
                positive,
                id_map,
                template_cache,
                build_ms: started_at.elapsed().as_secs_f64() * 1000.0,
            });
        }
        let fused_signed_requested =
            std::env::var_os("GLRMASK_EXPERIMENT_SMALL_BOUNDARY_SIGNED_FUSED").is_some();
        if fused_signed_requested
            && std::env::var_os("GLRMASK_EXPERIMENT_EARLY_BOUNDARY_TSID_QUOTIENT").is_some()
            && let TerminalAutomaton::Dwa(dwa) = &mut terminal_automaton
        {
            let old_tsids = id_map.num_tsids() as usize;
            let token_classes = id_map.num_internal_tokens() as usize;
            let quotient_started = Instant::now();
            if let Some((old_to_new, unique_weights)) =
                compute_boundary_dwa_tsid_behavior_quotient(dwa, old_tsids, token_classes)
            {
                let new_tsids = old_to_new.iter().copied().max().map_or(0, |v| v as usize + 1);
                if new_tsids < old_tsids {
                    apply_boundary_tsid_quotient_to_dwa_and_id_map(
                        dwa,
                        &mut id_map,
                        &old_to_new,
                    )
                    .ok_or_else(|| "early table-free boundary TSID quotient failed".to_string())?;
                    if compose_profile_enabled() {
                        eprintln!(
                            "[glrmask/profile][constraint_boundary_early_tsid_quotient_no_table] old_tsids={} new_tsids={} unique_terminal_weights={} total_ms={:.3}",
                            old_tsids,
                            new_tsids,
                            unique_weights,
                            quotient_started.elapsed().as_secs_f64() * 1000.0,
                        );
                    }
                }
            }
        }
        let nondeterministic_bundles_requested = std::env::var_os(
            "GLRMASK_EXPERIMENT_COMPILE_NONDETERMINISTIC_BUNDLES",
        )
        .is_some();
        let nondeterministic_bundle_min_tokens = std::env::var(
            "GLRMASK_COMPILE_NONDETERMINISTIC_BUNDLES_MIN_TOKENS",
        )
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(64);
        let compile_nondeterministic_bundles = nondeterministic_bundles_requested
            && id_map.num_internal_tokens() >= nondeterministic_bundle_min_tokens;
        let parser_nwa = if !compile_nondeterministic_bundles
            && let Some(cache) = prebuilt_bundle_cache.as_ref()
        {
            build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count_no_table_with_bundle_cache(
                &terminal_automaton,
                num_terminals,
                &templates,
                cache,
            )
        } else {
            build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count_no_table(
                &terminal_automaton,
                num_terminals,
                &templates,
                compile_nondeterministic_bundles,
            )
        };
        let Some(mut parser_nwa) = parser_nwa else {
            let mut template_cache = vec![None; num_terminals as usize];
            for (terminal, dfa) in templates.by_terminal {
                if let Some(slot) = template_cache.get_mut(terminal as usize) {
                    *slot = Some(dfa);
                }
            }
            return Ok(BoundaryRuntimeCandidate::Parser {
                positive: PositiveBoundaryParser::Dwa(DWA::new(
                    id_map.num_tsids(),
                    id_map.max_internal_token_id(),
                )),
                id_map,
                template_cache: Some(template_cache),
                build_ms: started_at.elapsed().as_secs_f64() * 1000.0,
            });
        };
        if fused_signed_requested
            && id_map.num_tsids() as usize <= 16
            && id_map.num_internal_tokens() as usize <= 64
        {
            let fused_started = Instant::now();
            let parser_dwa =
                normalize_signed_weighted_parser_stack_nwa_small_boundary_for_parser_state_count(
                    num_parser_states,
                    &parser_nwa,
                    id_map.num_tsids() as usize,
                    id_map.num_internal_tokens() as usize,
                )
                .ok_or_else(|| {
                    "table-free small-boundary fused signed publication rejected coordinate"
                        .to_string()
                })?;
            let mut template_cache = vec![None; num_terminals as usize];
            for (terminal, dfa) in templates.by_terminal {
                if let Some(slot) = template_cache.get_mut(terminal as usize) {
                    *slot = Some(dfa);
                }
            }
            let build_ms = started_at.elapsed().as_secs_f64() * 1000.0;
            if compose_profile_enabled() {
                eprintln!(
                    "[glrmask/profile][constraint_boundary_fused_signed_no_table] signed_states={} output_states={} output_transitions={} fused_ms={:.3} total_ms={build_ms:.3}",
                    parser_nwa.num_states(),
                    parser_dwa.num_states(),
                    parser_dwa.num_transitions(),
                    fused_started.elapsed().as_secs_f64() * 1000.0,
                );
            }
            return Ok(BoundaryRuntimeCandidate::Parser {
                positive: PositiveBoundaryParser::Dwa(parser_dwa),
                id_map,
                template_cache: Some(template_cache),
                build_ms,
            });
        }
        if std::env::var_os("GLRMASK_EXPERIMENT_HASHCONS_SIGNED_BOUNDARY_NWA").is_some() {
            parser_nwa = reverse_hashcons_signed_acyclic_nwa_fast(parser_nwa);
        }
        let resolved_reverse_topo =
            resolve_negative_codes_in_nwa(&mut parser_nwa, allow_grouped_cancellation);
        if std::env::var_os("GLRMASK_EXPERIMENT_FAST_HASHCONS_RESOLVED_BOUNDARY_NWA").is_some() {
            parser_nwa = reverse_hashcons_positive_acyclic_nwa_fast_with_reverse_topo(
                parser_nwa,
                resolved_reverse_topo,
            );
        } else if std::env::var_os("GLRMASK_EXPERIMENT_HASHCONS_RESOLVED_BOUNDARY_NWA").is_some() {
            parser_nwa = reverse_hashcons_positive_acyclic_nwa(parser_nwa);
        }
        let mut template_cache = vec![None; num_terminals as usize];
        for (terminal, dfa) in templates.by_terminal {
            if let Some(slot) = template_cache.get_mut(terminal as usize) {
                *slot = Some(dfa);
            }
        }
        let build_ms = started_at.elapsed().as_secs_f64() * 1000.0;
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_boundary_positive_no_table] states={} transitions={} build_ms={build_ms:.3} compile_nondeterministic_bundles={compile_nondeterministic_bundles}",
                parser_nwa.num_states(),
                parser_nwa.num_transitions(),
            );
        }
        Ok(BoundaryRuntimeCandidate::Parser {
            positive: PositiveBoundaryParser::Nwa(parser_nwa),
            id_map,
            template_cache: Some(template_cache),
            build_ms,
        })
    }

    pub(super) fn materialize_unminimized(
        self,
        table: &crate::compiler::glr::table::GLRTable,
        vocab: &Vocab,
    ) -> MappedArtifact<DWA> {
        match self {
            Self::Materialized(parser) => parser,
            Self::Deferred {
                terminal_automaton,
                id_map,
                analyzed,
                templates,
            } => MappedArtifact::new(
                build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
                    table,
                    &analyzed,
                    &terminal_automaton,
                    &templates,
                    vocab,
                    &id_map,
                    false,
                ),
                id_map,
            ),
            Self::DeferredTerminalCount { .. } => panic!(
                "count-only boundary parser work is supported only by the segmented NWA runtime"
            ),
        }
    }

    pub(super) fn materialize(self, table: &crate::compiler::glr::table::GLRTable, vocab: &Vocab) -> MappedArtifact<DWA> {
        match self {
            Self::Materialized(parser) => parser,
            Self::Deferred {
                terminal_automaton,
                id_map,
                analyzed,
                templates,
            } => {
                let parser_dwa = build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
                    table,
                    &analyzed,
                    &terminal_automaton,
                    &templates,
                    vocab,
                    &id_map,
                    false,
                );
                let parser_dwa = if parser_dwa.num_states() >= boundary_parser_minimize_min_states()
                    && parser_builder_skips_internal_minimization()
                    && std::env::var_os("GLRMASK_DISABLE_BOUNDARY_PARSER_MINIMIZE").is_none()
                {
                    minimize_owned(reverse_hashcons_owned(parser_dwa))
                } else {
                    parser_dwa
                };
                MappedArtifact::new(parser_dwa, id_map)
            }
            Self::DeferredTerminalCount { .. } => panic!(
                "count-only boundary parser work is supported only by the segmented NWA runtime"
            ),
        }
    }
}

/// Runtime parser segments are allowed to remain nondeterministic, but their
/// alphabet is strictly the positive parser-stack read alphabet (plus DEFAULT).
/// Negative labels encode compile-time PUSH effects and must never cross this
/// boundary. Keep this as a release-mode check so future compiler fast paths
/// cannot accidentally reintroduce runtime signed-stack interpretation.
pub(super) fn ensure_positive_runtime_parser_dwa(dwa: &DWA) -> Result<(), String> {
    for (source, state) in dwa.states().iter().enumerate() {
        if let Some(&label) = state.transitions.keys().find(|&&label| is_negative_label(label)) {
            return Err(format!(
                "runtime boundary parser DWA contains negative/PUSH label {label} at state {source}"
            ));
        }
    }
    Ok(())
}

/// Load the small terminal-DWA capture format used by the exact composition
/// oracle. This is intentionally an experiment-only bridge: it lets the real
/// linker consume a mathematically exact precomputed boundary while we measure
/// and optimize everything after boundary discovery. Production composition
/// must eventually construct the same object directly rather than depend on a
/// filesystem capture.
pub(super) fn load_boundary_terminal_capture(path: &str) -> Result<MappedArtifact<DWA>, String> {
    fn read_u32(bytes: &[u8], offset: &mut usize) -> Result<u32, String> {
        let end = offset
            .checked_add(4)
            .ok_or_else(|| "boundary capture u32 offset overflow".to_string())?;
        let raw = bytes
            .get(*offset..end)
            .ok_or_else(|| "truncated boundary capture u32".to_string())?;
        *offset = end;
        Ok(u32::from_le_bytes(raw.try_into().unwrap()))
    }
    fn read_u64(bytes: &[u8], offset: &mut usize) -> Result<u64, String> {
        let end = offset
            .checked_add(8)
            .ok_or_else(|| "boundary capture u64 offset overflow".to_string())?;
        let raw = bytes
            .get(*offset..end)
            .ok_or_else(|| "truncated boundary capture u64".to_string())?;
        *offset = end;
        Ok(u64::from_le_bytes(raw.try_into().unwrap()))
    }
    fn read_vec(bytes: &[u8], offset: &mut usize) -> Result<Vec<u32>, String> {
        let len = read_u32(bytes, offset)? as usize;
        (0..len).map(|_| read_u32(bytes, offset)).collect()
    }
    fn read_vec_vec(bytes: &[u8], offset: &mut usize) -> Result<Vec<Vec<u32>>, String> {
        let len = read_u32(bytes, offset)? as usize;
        (0..len).map(|_| read_vec(bytes, offset)).collect()
    }
    fn read_map(bytes: &[u8], offset: &mut usize) -> Result<ManyToOneIdMap, String> {
        Ok(ManyToOneIdMap {
            original_to_internal: read_vec(bytes, offset)?,
            internal_to_originals: read_vec_vec(bytes, offset)?,
            representative_original_ids: read_vec(bytes, offset)?,
        })
    }

    let bytes = std::fs::read(path)
        .map_err(|error| format!("read boundary terminal capture {path}: {error}"))?;
    if bytes.get(..8) != Some(&b"GLRMTD1\0"[..]) {
        return Err(format!("boundary terminal capture {path} has invalid magic"));
    }
    let mut offset = 8usize;
    let name_count = read_u32(&bytes, &mut offset)? as usize;
    for _ in 0..name_count {
        let len = read_u32(&bytes, &mut offset)? as usize;
        offset = offset
            .checked_add(len)
            .filter(|&end| end <= bytes.len())
            .ok_or_else(|| "truncated boundary capture terminal name".to_string())?;
    }
    let tokenizer_states = read_map(&bytes, &mut offset)?;
    let vocab_tokens = read_map(&bytes, &mut offset)?;
    let dwa_len = usize::try_from(read_u64(&bytes, &mut offset)?)
        .map_err(|_| "boundary capture DWA length exceeds usize".to_string())?;
    let end = offset
        .checked_add(dwa_len)
        .filter(|&end| end == bytes.len())
        .ok_or_else(|| "boundary capture DWA payload length mismatch".to_string())?;
    let dwa: DWA = bincode::deserialize(&bytes[offset..end])
        .map_err(|error| format!("deserialize boundary terminal DWA: {error}"))?;
    Ok(MappedArtifact::new(
        dwa,
        InternalIdMap {
            tokenizer_states,
            vocab_tokens,
            deferred_vocab_singleton_original_ids: None,
        },
    ))
}
