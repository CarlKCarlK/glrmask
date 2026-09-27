//! Construct the parser overlay for paths crossing component boundaries.

use crate::automata::lexer::Lexer;
use super::{
    Action, AnalyzedGrammar, Arc, BTreeMap, BTreeSet, BitSet, BoundaryParserWork, BoundaryRepair,
    BoundaryTokenDiscovery, BoundaryTokenNodeKey, CompiledSubgrammarInput, ComposedTable,
    ConcreteBoundaryDeltaPlan, Constraint, DEFAULT_LABEL, DWA, DWAState, FxHashMap, FxHashSet,
    Instant, InternalIdMap, LazyBooleanParserDomains, ManyToOneIdMap, MappedArtifact,
    MergedIgnoreTerminals, NWA, OnceLock, PreTableBoundaryBaseDiscovery,
    PrebuiltParserBundleCache, RangeSetBlaze, ScopedWeightOpCache, SharedBooleanParserDomains,
    SharedTokenSet, SpecialTokenTerminal, Templates, TerminalAutomaton, TerminalColoring,
    Tokenizer, VecDeque, Vocab, Weight, add_boundary_special_token_paths,
    add_control_loops_to_terminal_artifact, boundary_delta_reset_relations,
    boundary_discovery_good_signature, boundary_interface_adjacent_pair_candidates,
    boundary_parser_minimize_min_states, boundary_tokens_by_start_component,
    build_boolean_terminal_bundle_nwa, build_complete_composed_parser_template_cache,
    build_composition_templates, build_parser_dwa_from_terminal_dwa_with_precomputed_templates,
    build_prebuilt_terminal_bundle_preimage_domain_dwa_direct_profiled,
    build_static_boundary_shard_work, build_terminal_bundle_preimage_domain_nwa,
    changed_parent_template_candidate_terminals, collect_one_byte_seed_relations,
    collect_one_byte_seed_relations_components, component_state_coordinate_map,
    component_tokenizer_state_layout_owned_parent,
    compose_nonnullable_grammar_adjacency_summaries, compose_profile_enabled,
    compute_ever_allowed_follows, defer_boundary_commit_templates, determinize,
    direct_boundary_terminal_automaton, disallowed_follows_from_allowed_rows,
    discover_boundary_token_paths, explicit_parser_nwa,
    extend_boundary_interfaces_through_stack_neutral_lr_actions,
    factor_one_terminal_seed_relations, find_difference, finish_eager_changed_parent_templates,
    finish_eager_concrete_boundary_delta_plan, install_concrete_boundary_delta_templates,
    load_boundary_terminal_capture, macro_join, macro_parallelism_disabled,
    merge_one_terminal_relations, merged_ignore_terminals, minimize_owned,
    normalize_parser_stack_domain_nwa_preserving_explicit, normalize_weighted_parser_stack_nwa,
    parser_builder_skips_internal_minimization, prebuild_parser_bundle_cache_excluding_terminals,
    prepare_concrete_boundary_delta_plan, profile_current_boundary_template_delta,
    replace_boundary_discovery_tokens, report_macro_item_timings, reverse_hashcons_owned,
    try_build_cached_composition_templates,
    try_build_cached_composition_templates_for_terminal_count,
    try_build_changed_parent_templates_for_terminal_count,
    try_rebuild_cached_transported_component_templates, universal_parser_stack_domain_dwa,
    visible_boundary_interface_pairs,
};
use rayon::prelude::*;

/// Exact weighted determinization of shared lazy parser-stack predicates.
///
/// The lazy arena's DEFAULT read is an additive wildcard. For each output row
/// we therefore compute every concrete explicit derivative with wildcard
/// branches included, and a separate wildcard-only derivative used as the
/// runtime DWA DEFAULT fallback. This converts symbolic additive DEFAULT
/// semantics to ordinary DWA fallback semantics without materializing a parser
/// NWA or normalizing each supported root separately.
/// Partition weighted parser-domain roots into exact disjoint support atoms.
///
/// Each input Weight is a set of `(TSID, token)` points. For every TSID we
/// sweep token-range endpoints and intern the exact set of parser-domain roots
/// active on each token interval. Points with the same membership signature
/// are accumulated into one Weight. The resulting Weights are pairwise
/// disjoint, so later prefix-finality subtraction never has to split an atom.
pub(super) fn atomize_weighted_parser_roots(
    arena: &mut SharedBooleanParserDomains,
    roots: &[(u32, Weight)],
) -> Vec<(u32, Weight)> {
    let started_at = Instant::now();
    let mut by_tsid = BTreeMap::<u32, Vec<(u32, SharedTokenSet)>>::new();
    for (root, weight) in roots {
        if weight.is_empty() {
            continue;
        }
        if weight.is_full() {
            // Boundary support is vocabulary/TSID bounded. A full sentinel
            // would require an explicit finite universe to atomize, so leave
            // that rare shape to the ordinary weighted combiner.
            return roots.to_vec();
        }
        for (range, tokens) in weight.raw_range_values() {
            for tsid in *range.start()..=*range.end() {
                by_tsid
                    .entry(tsid)
                    .or_default()
                    .push((*root, Arc::clone(tokens)));
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct TokenEvent {
        pos: u32,
        root: u32,
        add: bool,
    }

    let mut signature_ids = FxHashMap::<Vec<u32>, usize>::default();
    let mut signatures = Vec::<Vec<u32>>::new();
    let mut entries_by_signature = Vec::<Vec<(u32, RangeSetBlaze<u32>)>>::new();
    let mut token_events_total = 0usize;
    let mut token_segments_total = 0usize;

    for (tsid, root_sets) in by_tsid {
        let mut events = Vec::<TokenEvent>::new();
        for (root, tokens) in root_sets {
            for range in tokens.ranges() {
                events.push(TokenEvent {
                    pos: *range.start(),
                    root,
                    add: true,
                });
                if let Some(pos) = range.end().checked_add(1) {
                    events.push(TokenEvent {
                        pos,
                        root,
                        add: false,
                    });
                }
            }
        }
        token_events_total += events.len();
        events.sort_unstable_by_key(|event| (event.pos, event.add, event.root));
        if events.is_empty() {
            continue;
        }

        let mut active = BTreeSet::<u32>::new();
        let mut ranges_by_signature = BTreeMap::<usize, Vec<std::ops::RangeInclusive<u32>>>::new();
        let mut cursor = events[0].pos;
        let mut index = 0usize;
        while index < events.len() {
            let pos = events[index].pos;
            if cursor < pos && !active.is_empty() {
                let signature = active.iter().copied().collect::<Vec<_>>();
                let signature_id = if let Some(&existing) = signature_ids.get(&signature) {
                    existing
                } else {
                    let id = signatures.len();
                    signatures.push(signature.clone());
                    signature_ids.insert(signature, id);
                    entries_by_signature.push(Vec::new());
                    id
                };
                ranges_by_signature
                    .entry(signature_id)
                    .or_default()
                    .push(cursor..=pos - 1);
                token_segments_total += 1;
            }

            let bucket_start = index;
            while index < events.len() && events[index].pos == pos {
                index += 1;
            }
            for event in &events[bucket_start..index] {
                if !event.add {
                    active.remove(&event.root);
                }
            }
            for event in &events[bucket_start..index] {
                if event.add {
                    active.insert(event.root);
                }
            }
            cursor = pos;
        }
        debug_assert!(active.is_empty());

        for (signature_id, ranges) in ranges_by_signature {
            let tokens = RangeSetBlaze::from_iter(ranges);
            if !tokens.is_empty() {
                entries_by_signature[signature_id].push((tsid, tokens));
            }
        }
    }

    let raw_atoms = signatures.len();
    let atoms = signatures
        .into_iter()
        .zip(entries_by_signature)
        .filter_map(|(signature, entries)| {
            let domain = arena.union_all(signature);
            if domain == SharedBooleanParserDomains::EMPTY {
                return None;
            }
            let weight = Weight::from_per_tsid_token_sets(entries);
            (!weight.is_empty()).then_some((domain, weight))
        })
        .collect::<Vec<_>>();
    let unique_domains = atoms
        .iter()
        .map(|(domain, _)| *domain)
        .collect::<FxHashSet<_>>()
        .len();
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_domain_atomize] input_roots={} raw_atoms={} output_atoms={} unique_domains={} token_events={} token_segments={} total_ms={:.3}",
            roots.len(),
            raw_atoms,
            atoms.len(),
            unique_domains,
            token_events_total,
            token_segments_total,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    atoms
}

/// Determinize parser-domain support after exact `(TSID, token)` atomization.
/// Atom weights are pairwise disjoint and never change. A state therefore needs
/// only `(atom_id, parser-domain-root)` pairs; prefix finality drops complete
/// atoms instead of computing Weight differences. Concrete edge/final weights
/// are materialized once per distinct atom-id support set.
pub(super) fn combine_disjoint_weighted_shared_parser_atoms(
    arena: &mut SharedBooleanParserDomains,
    atoms: &[(u32, Weight)],
) -> DWA {
    let started_at = Instant::now();
    let initial = atoms
        .iter()
        .enumerate()
        .filter_map(|(atom, (root, weight))| {
            (!weight.is_empty() && !arena.is_empty_root(*root))
                .then_some((atom as u32, *root))
        })
        .collect::<Vec<_>>();
    if initial.is_empty() {
        return DWA::new(0, 0);
    }

    let mut states = vec![DWAState::default()];
    let mut lanes_by_state = vec![initial.clone()];
    let mut ids = FxHashMap::<Vec<(u32, u32)>, u32>::default();
    ids.insert(initial, 0);
    let mut queue = VecDeque::from([0u32]);
    let mut support_weights = FxHashMap::<Vec<u32>, Weight>::default();
    let mut weight_ops = ScopedWeightOpCache::default();
    let mut support_hits = 0usize;
    let mut support_misses = 0usize;
    let mut max_lanes = 0usize;

    fn materialize_support(
        atom_ids: Vec<u32>,
        atoms: &[(u32, Weight)],
        cache: &mut FxHashMap<Vec<u32>, Weight>,
        weight_ops: &mut ScopedWeightOpCache,
        hits: &mut usize,
        misses: &mut usize,
    ) -> Weight {
        if atom_ids.is_empty() {
            return Weight::empty();
        }
        if let Some(existing) = cache.get(&atom_ids) {
            *hits += 1;
            return existing.clone();
        }
        *misses += 1;
        let weights = atom_ids
            .iter()
            .map(|&atom| &atoms[atom as usize].1)
            .collect::<Vec<_>>();
        let weight = weight_ops.union_all(weights);
        cache.insert(atom_ids, weight.clone());
        weight
    }

    while let Some(state_id) = queue.pop_front() {
        let lanes = lanes_by_state[state_id as usize].clone();
        max_lanes = max_lanes.max(lanes.len());

        let final_atoms = lanes
            .iter()
            .filter_map(|&(atom, root)| arena.is_universal_root(root).then_some(atom))
            .collect::<Vec<_>>();
        let final_weight = materialize_support(
            final_atoms,
            atoms,
            &mut support_weights,
            &mut weight_ops,
            &mut support_hits,
            &mut support_misses,
        );
        if !final_weight.is_empty() {
            states[state_id as usize].final_weight = Some(final_weight);
        }

        // Atomic support makes prefix normalization exact without any Weight
        // subtraction: an atom is either accepted by its whole current domain
        // or not accepted at all.
        let live = lanes
            .into_iter()
            .filter(|&(_, root)| !arena.is_universal_root(root))
            .collect::<Vec<_>>();
        if live.is_empty() {
            continue;
        }

        let mut default_next = Vec::<(u32, u32)>::new();
        let mut overrides_by_label = BTreeMap::<i32, Vec<(u32, u32)>>::new();
        for &(atom, root) in &live {
            let default = arena.advance_default(root);
            if !arena.is_empty_root(default) {
                default_next.push((atom, default));
            }
            for &(label, derivative) in arena.explicit_derivatives(root).iter() {
                debug_assert!(label >= 0 && label != DEFAULT_LABEL);
                overrides_by_label
                    .entry(label)
                    .or_default()
                    .push((atom, derivative));
            }
        }

        let mut emit = |label: i32,
                        next: Vec<(u32, u32)>,
                        states: &mut Vec<DWAState>,
                        lanes_by_state: &mut Vec<Vec<(u32, u32)>>,
                        ids: &mut FxHashMap<Vec<(u32, u32)>, u32>,
                        queue: &mut VecDeque<u32>| {
            if next.is_empty() {
                return;
            }
            let atom_ids = next.iter().map(|&(atom, _)| atom).collect::<Vec<_>>();
            let edge_weight = materialize_support(
                atom_ids,
                atoms,
                &mut support_weights,
                &mut weight_ops,
                &mut support_hits,
                &mut support_misses,
            );
            if edge_weight.is_empty() {
                return;
            }
            let target = if let Some(&target) = ids.get(&next) {
                target
            } else {
                let target = states.len() as u32;
                states.push(DWAState::default());
                lanes_by_state.push(next.clone());
                ids.insert(next, target);
                queue.push_back(target);
                target
            };
            states[state_id as usize]
                .transitions
                .insert(label, (target, edge_weight));
        };

        emit(
            DEFAULT_LABEL,
            default_next.clone(),
            &mut states,
            &mut lanes_by_state,
            &mut ids,
            &mut queue,
        );

        // A concrete derivative differs from DEFAULT only for atoms listed in
        // this label's sparse override row. Merge the sorted override atoms into
        // the sorted DEFAULT lane vector instead of rescanning every live atom.
        for (label, overrides) in overrides_by_label {
            let mut next = Vec::with_capacity(default_next.len() + overrides.len());
            let mut default_index = 0usize;
            let mut override_index = 0usize;
            while default_index < default_next.len() || override_index < overrides.len() {
                match (
                    default_next.get(default_index),
                    overrides.get(override_index),
                ) {
                    (Some(&(default_atom, default_root)), Some(&(override_atom, override_root))) => {
                        if default_atom < override_atom {
                            next.push((default_atom, default_root));
                            default_index += 1;
                        } else if override_atom < default_atom {
                            next.push((override_atom, override_root));
                            override_index += 1;
                        } else {
                            next.push((override_atom, override_root));
                            default_index += 1;
                            override_index += 1;
                        }
                    }
                    (Some(&lane), None) => {
                        next.push(lane);
                        default_index += 1;
                    }
                    (None, Some(&lane)) => {
                        next.push(lane);
                        override_index += 1;
                    }
                    (None, None) => break,
                }
            }
            emit(
                label,
                next,
                &mut states,
                &mut lanes_by_state,
                &mut ids,
                &mut queue,
            );
        }
    }

    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_domain_atom_determinize] atoms={} states={} transitions={} max_lanes={} derivative_rows_cached={} support_cache_entries={} support_hits={} support_misses={} total_ms={:.3}",
            atoms.len(),
            states.len(),
            states.iter().map(|state| state.transitions.len()).sum::<usize>(),
            max_lanes,
            arena.derivative_row_cache_len(),
            support_weights.len(),
            support_hits,
            support_misses,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    DWA::from_parts(states, 0)
}

pub(super) fn combine_weighted_shared_parser_roots(
    arena: &mut SharedBooleanParserDomains,
    roots: &[(u32, Weight)],
) -> DWA {
    #[derive(Clone)]
    struct Lane {
        root: u32,
        weight: Weight,
    }

    fn canonicalize_lanes(
        lanes: impl IntoIterator<Item = Lane>,
        weight_ops: &mut ScopedWeightOpCache,
    ) -> Vec<Lane> {
        let mut grouped = BTreeMap::<u32, Vec<Weight>>::new();
        for lane in lanes {
            if !lane.weight.is_empty() {
                grouped.entry(lane.root).or_default().push(lane.weight);
            }
        }
        grouped
            .into_iter()
            .filter_map(|(root, weights)| {
                let weight = weight_ops.union_all(weights.iter());
                (!weight.is_empty()).then_some(Lane { root, weight })
            })
            .collect()
    }

    fn lane_key(lanes: &[Lane]) -> Vec<(u32, usize)> {
        lanes
            .iter()
            .map(|lane| (lane.root, lane.weight.ptr_key()))
            .collect()
    }

    fn union_lane_weights(
        lanes: &[Lane],
        weight_ops: &mut ScopedWeightOpCache,
    ) -> Weight {
        weight_ops.union_all(lanes.iter().map(|lane| &lane.weight))
    }

    let mut weight_ops = ScopedWeightOpCache::default();
    let keep_final_lanes =
        std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_DOMAIN_KEEP_FINAL_LANES").is_some();
    let initial = canonicalize_lanes(roots.iter().filter_map(|(root, weight)| {
        (!weight.is_empty() && !arena.is_empty_root(*root)).then(|| Lane {
            root: *root,
            weight: weight.clone(),
        })
    }), &mut weight_ops);
    if initial.is_empty() {
        return DWA::new(0, 0);
    }

    let mut states = vec![DWAState::default()];
    let mut lanes_by_state = vec![initial.clone()];
    let mut ids = FxHashMap::<Vec<(u32, usize)>, u32>::default();
    ids.insert(lane_key(&initial), 0);
    let mut queue = VecDeque::from([0u32]);
    let mut labels_by_root = FxHashMap::<u32, Arc<Vec<i32>>>::default();

    while let Some(state_id) = queue.pop_front() {
        let lanes = lanes_by_state[state_id as usize].clone();
        let mut final_parts = Vec::<Weight>::new();
        for lane in &lanes {
            if arena.is_universal_root(lane.root) {
                final_parts.push(lane.weight.clone());
            }
        }
        let final_weight = weight_ops.union_all(final_parts.iter());
        if !final_weight.is_empty() {
            states[state_id as usize].final_weight = Some(final_weight.clone());
        }

        let live = if keep_final_lanes {
            lanes
        } else {
            lanes
                .into_iter()
                .filter_map(|lane| {
                    let residual = weight_ops.difference(&lane.weight, &final_weight);
                    (!residual.is_empty()).then_some(Lane {
                        root: lane.root,
                        weight: residual,
                    })
                })
                .collect::<Vec<_>>()
        };
        if live.is_empty() {
            continue;
        }

        let mut explicit_labels = BTreeSet::<i32>::new();
        for lane in &live {
            let labels = labels_by_root
                .entry(lane.root)
                .or_insert_with(|| Arc::new(arena.explicit_labels(lane.root)));
            explicit_labels.extend(labels.iter().copied());
        }

        let mut emit = |label: i32,
                        next: Vec<Lane>,
                        states: &mut Vec<DWAState>,
                        lanes_by_state: &mut Vec<Vec<Lane>>,
                        ids: &mut FxHashMap<Vec<(u32, usize)>, u32>,
                        queue: &mut VecDeque<u32>| {
            let next = canonicalize_lanes(next, &mut weight_ops);
            if next.is_empty() {
                return;
            }
            let edge_weight = union_lane_weights(&next, &mut weight_ops);
            if edge_weight.is_empty() {
                return;
            }
            let key = lane_key(&next);
            let target = if let Some(&target) = ids.get(&key) {
                target
            } else {
                let target = states.len() as u32;
                states.push(DWAState::default());
                lanes_by_state.push(next);
                ids.insert(key, target);
                queue.push_back(target);
                target
            };
            states[state_id as usize]
                .transitions
                .insert(label, (target, edge_weight));
        };

        // Runtime DEFAULT fallback: derivative contributed only by symbolic
        // wildcard reads, for every positive parser-state label not explicitly
        // represented below.
        let default_next = live
            .iter()
            .filter_map(|lane| {
                let root = arena.advance_default(lane.root);
                (!arena.is_empty_root(root)).then(|| Lane {
                    root,
                    weight: lane.weight.clone(),
                })
            })
            .collect::<Vec<_>>();
        emit(
            DEFAULT_LABEL,
            default_next,
            &mut states,
            &mut lanes_by_state,
            &mut ids,
            &mut queue,
        );

        for label in explicit_labels {
            debug_assert!(label >= 0 && label != DEFAULT_LABEL);
            let next = live
                .iter()
                .filter_map(|lane| {
                    let root = arena.advance(lane.root, label as u32);
                    (!arena.is_empty_root(root)).then(|| Lane {
                        root,
                        weight: lane.weight.clone(),
                    })
                })
                .collect::<Vec<_>>();
            emit(
                label,
                next,
                &mut states,
                &mut lanes_by_state,
                &mut ids,
                &mut queue,
            );
        }
    }

    DWA::from_parts(states, 0)
}

pub(super) fn profile_direct_boundary_terminal_dwa_domain_dp(
    table: &crate::compiler::glr::table::GLRTable,
    templates: &Templates,
    terminal_dwa: &DWA,
) {
    if std::env::var_os("GLRMASK_PROFILE_DIRECT_BOUNDARY_DOMAIN_DP").is_none()
        || !terminal_dwa.is_acyclic()
    {
        return;
    }
    let total_started_at = Instant::now();
    let n = terminal_dwa.states().len();
    let mut indegree = vec![0usize; n];
    for state in terminal_dwa.states() {
        for &(target, _) in state.transitions.values() {
            indegree[target as usize] += 1;
        }
    }
    let mut topo_queue = VecDeque::new();
    for (state, &degree) in indegree.iter().enumerate() {
        if degree == 0 {
            topo_queue.push_back(state as u32);
        }
    }
    let mut topo = Vec::with_capacity(n);
    while let Some(source) = topo_queue.pop_front() {
        topo.push(source);
        for &(target, _) in terminal_dwa.states()[source as usize].transitions.values() {
            indegree[target as usize] -= 1;
            if indegree[target as usize] == 0 {
                topo_queue.push_back(target);
            }
        }
    }
    assert_eq!(topo.len(), n);

    #[derive(Clone)]
    struct Group {
        target: u32,
        weight: Weight,
        terminals: Vec<u32>,
    }
    let mut groups_by_state = Vec::<Vec<Group>>::with_capacity(n);
    let mut terminal_sets = BTreeSet::<Vec<u32>>::new();
    for state in terminal_dwa.states() {
        let mut groups = BTreeMap::<(u32, usize), (Weight, Vec<u32>)>::new();
        for (&label, (target, weight)) in &state.transitions {
            if label < 0 {
                eprintln!("[glrmask/profile][direct_boundary_domain_dp] skipped=true reason=negative_terminal_label");
                return;
            }
            groups
                .entry((*target, weight.ptr_key()))
                .or_insert_with(|| (weight.clone(), Vec::new()))
                .1
                .push(label as u32);
        }
        let groups = groups
            .into_iter()
            .map(|((target, _), (weight, mut terminals))| {
                terminals.sort_unstable();
                terminals.dedup();
                terminal_sets.insert(terminals.clone());
                Group { target, weight, terminals }
            })
            .collect::<Vec<_>>();
        groups_by_state.push(groups);
    }
    let bundle_started_at = Instant::now();
    let build_bundle = |terminals: &Vec<u32>| {
            build_boolean_terminal_bundle_nwa(templates, terminals)
                .map(|bundle| (terminals.clone(), Arc::new(bundle)))
        };
    let bundles = if macro_parallelism_disabled() {
        let mut timings = Vec::with_capacity(terminal_sets.len());
        let result = terminal_sets
            .iter()
            .filter_map(|terminals| {
                let started = Instant::now();
                let result = build_bundle(terminals);
                timings.push(started.elapsed().as_secs_f64() * 1000.0);
                result
            })
            .collect::<FxHashMap<_, _>>();
        report_macro_item_timings("compose_direct_boundary_bundles", &timings);
        result
    } else {
        terminal_sets
            .par_iter()
            .filter_map(build_bundle)
            .collect::<FxHashMap<_, _>>()
    };
    let bundle_ms = bundle_started_at.elapsed().as_secs_f64() * 1000.0;
    if bundles.len() != terminal_sets.len() {
        eprintln!(
            "[glrmask/profile][direct_boundary_domain_dp] skipped=true reason=missing_bundle built={} requested={}",
            bundles.len(),
            terminal_sets.len(),
        );
        return;
    }

    type BooleanDomainKey = (u32, Vec<(bool, Vec<(i32, u32, bool)>)>);
    let boolean_domain_key = |domain: &DWA| -> BooleanDomainKey {
        let states = domain
            .states()
            .iter()
            .map(|state| {
                let final_accept = state
                    .final_weight
                    .as_ref()
                    .is_some_and(|weight| !weight.is_empty());
                let transitions = state
                    .transitions
                    .iter()
                    .map(|(&label, &(target, ref weight))| {
                        debug_assert!(weight.is_full() || weight.is_empty());
                        (label, target, !weight.is_empty())
                    })
                    .collect::<Vec<_>>();
                (final_accept, transitions)
            })
            .collect::<Vec<_>>();
        (domain.start_state(), states)
    };

    #[derive(Clone)]
    struct Lane {
        domain: Arc<DWA>,
        support: Weight,
    }
    let universal = Arc::new(universal_parser_stack_domain_dwa());
    let mut domain_interner = FxHashMap::<BooleanDomainKey, Arc<DWA>>::default();
    domain_interner.insert(boolean_domain_key(&universal), Arc::clone(&universal));
    let mut structural_hits = 0usize;
    let mut structural_misses = 1usize;
    let mut domains = vec![FxHashMap::<usize, Lane>::default(); n];
    let mut preimage_cache = FxHashMap::<(Vec<u32>, usize), Arc<DWA>>::default();
    let mut weight_ops = ScopedWeightOpCache::default();
    let mut cache_hits = 0usize;
    let mut cache_misses = 0usize;
    let mut direct_none = 0usize;
    let mut preimage_ms = 0.0f64;
    let mut normalize_ms = 0.0f64;
    let mut concat_ms = 0.0f64;
    let mut max_lanes = 0usize;
    let mut unique_domain_ptrs = FxHashSet::<usize>::default();
    unique_domain_ptrs.insert(Arc::as_ptr(&universal) as usize);

    for &source in topo.iter().rev() {
        let mut lanes = FxHashMap::<usize, Lane>::default();
        if let Some(final_weight) = &terminal_dwa.states()[source as usize].final_weight {
            let ptr = Arc::as_ptr(&universal) as usize;
            lanes.insert(ptr, Lane {
                domain: Arc::clone(&universal),
                support: final_weight.clone(),
            });
        }
        for group in &groups_by_state[source as usize] {
            let bundle = &bundles[&group.terminals];
            for target_lane in domains[group.target as usize].values() {
                let support = weight_ops.intersection(&group.weight, &target_lane.support);
                if support.is_empty() {
                    continue;
                }
                let target_ptr = Arc::as_ptr(&target_lane.domain) as usize;
                let key = (group.terminals.clone(), target_ptr);
                let domain = if let Some(existing) = preimage_cache.get(&key) {
                    cache_hits += 1;
                    Arc::clone(existing)
                } else {
                    cache_misses += 1;
                    let (result, profile) =
                        build_prebuilt_terminal_bundle_preimage_domain_dwa_direct_profiled(
                            table,
                            bundle,
                            &target_lane.domain,
                        );
                    preimage_ms += profile.total_ms;
                    normalize_ms += profile.normalize_ms;
                    concat_ms += profile.concatenate_ms;
                    let Some(result) = result else {
                        direct_none += 1;
                        continue;
                    };
                    let structural_key = boolean_domain_key(&result);
                    let result = if let Some(existing) = domain_interner.get(&structural_key) {
                        structural_hits += 1;
                        Arc::clone(existing)
                    } else {
                        structural_misses += 1;
                        let result = Arc::new(result);
                        domain_interner.insert(structural_key, Arc::clone(&result));
                        result
                    };
                    unique_domain_ptrs.insert(Arc::as_ptr(&result) as usize);
                    preimage_cache.insert(key, Arc::clone(&result));
                    result
                };
                let ptr = Arc::as_ptr(&domain) as usize;
                if let Some(existing) = lanes.get_mut(&ptr) {
                    existing.support = weight_ops.union(&existing.support, &support);
                } else {
                    lanes.insert(ptr, Lane { domain, support });
                }
            }
        }
        max_lanes = max_lanes.max(lanes.len());
        domains[source as usize] = lanes;
    }

    let start = &domains[terminal_dwa.start_state() as usize];
    let unique_domain_states = preimage_cache
        .values()
        .map(|domain| domain.num_states() as usize)
        .sum::<usize>()
        + universal.num_states() as usize;
    let unique_domain_transitions = preimage_cache
        .values()
        .map(|domain| domain.num_transitions())
        .sum::<usize>()
        + universal.num_transitions();
    let start_domain_states = start
        .values()
        .map(|lane| lane.domain.num_states() as usize)
        .sum::<usize>();
    let start_domain_transitions = start
        .values()
        .map(|lane| lane.domain.num_transitions())
        .sum::<usize>();

    if std::env::var_os("GLRMASK_PROFILE_DIRECT_BOUNDARY_DOMAIN_FLATTEN").is_some() {
        let flatten_started_at = Instant::now();
        let mut arena = NWA::new(0, 0);
        let global_start = arena.add_state();
        arena.set_start_states(vec![global_start]);
        let mut appended_states = 1usize;
        for lane in start.values() {
            let body = arena.append_with_body(&lane.domain.to_nwa());
            appended_states += lane.domain.num_states() as usize;
            for target in body.start_states {
                arena.add_epsilon(global_start, target, lane.support.clone());
            }
        }
        debug_assert_eq!(arena.num_states() as usize, appended_states);
        let append_ms = flatten_started_at.elapsed().as_secs_f64() * 1000.0;
        let normalize_started_at = Instant::now();
        let flattened = normalize_weighted_parser_stack_nwa(table, &arena);
        let normalize_ms = normalize_started_at.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[glrmask/profile][direct_boundary_domain_flatten] start_domains={} input_states={} input_transitions={} output_states={} output_transitions={} append_ms={append_ms:.3} normalize_ms={normalize_ms:.3} total_ms={:.3}",
            start.len(),
            arena.num_states(),
            arena.num_transitions(),
            flattened.num_states(),
            flattened.num_transitions(),
            flatten_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    eprintln!(
        "[glrmask/profile][direct_boundary_domain_dp] terminal_states={} terminal_transitions={} groups={} terminal_sets={} start_domains={} max_lanes={} preimage_cache_entries={} cache_hits={} cache_misses={} direct_none={} structural_hits={} structural_misses={} interned_domains={} unique_domain_ptrs={} aggregate_domain_states={} aggregate_domain_transitions={} start_domain_states={} start_domain_transitions={} bundle_ms={bundle_ms:.3} preimage_cpu_ms={preimage_ms:.3} concatenate_cpu_ms={concat_ms:.3} normalize_cpu_ms={normalize_ms:.3} total_ms={:.3}",
        terminal_dwa.num_states(),
        terminal_dwa.num_transitions(),
        groups_by_state.iter().map(Vec::len).sum::<usize>(),
        terminal_sets.len(),
        start.len(),
        max_lanes,
        preimage_cache.len(),
        cache_hits,
        cache_misses,
        direct_none,
        structural_hits,
        structural_misses,
        domain_interner.len(),
        unique_domain_ptrs.len(),
        unique_domain_states,
        unique_domain_transitions,
        start_domain_states,
        start_domain_transitions,
        total_started_at.elapsed().as_secs_f64() * 1000.0,
    );
}

pub(super) fn validate_lazy_boundary_terminal_dwa_preimages(
    table: &crate::compiler::glr::table::GLRTable,
    templates: &Templates,
    terminal_dwa: &DWA,
) {
    if std::env::var_os("GLRMASK_VALIDATE_LAZY_BOUNDARY_PREIMAGE_DP").is_none()
        || !terminal_dwa.is_acyclic()
    {
        return;
    }
    let started_at = Instant::now();
    let n = terminal_dwa.states().len();
    let mut indegree = vec![0usize; n];
    for state in terminal_dwa.states() {
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
        for &(target, _) in terminal_dwa.states()[source as usize].transitions.values() {
            indegree[target as usize] -= 1;
            if indegree[target as usize] == 0 {
                queue.push_back(target);
            }
        }
    }
    assert_eq!(topo.len(), n);

    #[derive(Clone)]
    struct Group {
        target: u32,
        weight: Weight,
        terminals: Vec<u32>,
    }
    let mut groups_by_state = Vec::<Vec<Group>>::with_capacity(n);
    let mut terminal_sets = BTreeSet::<Vec<u32>>::new();
    for state in terminal_dwa.states() {
        let mut groups = BTreeMap::<(u32, usize), (Weight, Vec<u32>)>::new();
        for (&label, (target, weight)) in &state.transitions {
            assert!(label >= 0);
            groups
                .entry((*target, weight.ptr_key()))
                .or_insert_with(|| (weight.clone(), Vec::new()))
                .1
                .push(label as u32);
        }
        let groups = groups
            .into_iter()
            .map(|((target, _), (weight, mut terminals))| {
                terminals.sort_unstable();
                terminals.dedup();
                terminal_sets.insert(terminals.clone());
                Group { target, weight, terminals }
            })
            .collect::<Vec<_>>();
        groups_by_state.push(groups);
    }
    let build_bundle = |terminals: &Vec<u32>| {
            let bundle = build_boolean_terminal_bundle_nwa(templates, terminals)
                .expect("lazy validation bundle must build");
            (terminals.clone(), Arc::new(bundle))
    };
    let bundles = if macro_parallelism_disabled() {
        let mut timings = Vec::with_capacity(terminal_sets.len());
        let result = terminal_sets
            .iter()
            .map(|terminals| {
                let started = Instant::now();
                let result = build_bundle(terminals);
                timings.push(started.elapsed().as_secs_f64() * 1000.0);
                result
            })
            .collect::<FxHashMap<_, _>>();
        report_macro_item_timings("compose_lazy_validation_bundles", &timings);
        result
    } else {
        terminal_sets.par_iter().map(build_bundle).collect::<FxHashMap<_, _>>()
    };

    let mut arena = LazyBooleanParserDomains::new();
    let mut domains = vec![BTreeMap::<u32, Weight>::new(); n];
    let mut weight_ops = ScopedWeightOpCache::default();
    let mut preimage_cache = FxHashMap::<(Vec<u32>, u32), u32>::default();
    let mut checked = 0usize;
    let mut cache_hits = 0usize;
    for &source in topo.iter().rev() {
        let mut lanes = BTreeMap::<u32, Weight>::new();
        if let Some(final_weight) = &terminal_dwa.states()[source as usize].final_weight {
            lanes.insert(LazyBooleanParserDomains::UNIVERSAL, final_weight.clone());
        }
        for group in &groups_by_state[source as usize] {
            let bundle = &bundles[&group.terminals];
            for (&target_root, target_weight) in &domains[group.target as usize] {
                let support = weight_ops.intersection(&group.weight, target_weight);
                if support.is_empty() {
                    continue;
                }
                let key = (group.terminals.clone(), target_root);
                let root = if let Some(&root) = preimage_cache.get(&key) {
                    cache_hits += 1;
                    root
                } else {
                    let root = arena
                        .preimage_bundle(bundle, target_root)
                        .expect("lazy preimage must exist");
                    let target_nwa = arena.to_nwa(target_root);
                    let oracle_nwa = build_terminal_bundle_preimage_domain_nwa(
                        table,
                        templates,
                        &group.terminals,
                        &target_nwa,
                    )
                    .expect("oracle preimage must exist");
                    let oracle = normalize_parser_stack_domain_nwa_preserving_explicit(
                        table,
                        &oracle_nwa,
                    );
                    let lazy_nwa = arena.to_nwa(root);
                    let lazy = normalize_parser_stack_domain_nwa_preserving_explicit(
                        table,
                        &lazy_nwa,
                    );
                    let difference = find_difference(&lazy, &oracle)
                        .expect("lazy/oracle parser domains should be acyclic");
                    if let Some(word) = difference.as_ref() {
                        eprintln!(
                            "[glrmask/validate][lazy_boundary_preimage_mismatch] terminals={:?} target_root={} root={} witness={:?} lazy={} oracle={}",
                            group.terminals,
                            target_root,
                            root,
                            word,
                            lazy.eval_word(word),
                            oracle.eval_word(word),
                        );
                        panic!("lazy parser-domain preimage differs from exact NWA oracle");
                    }
                    checked += 1;
                    preimage_cache.insert(key, root);
                    root
                };
                if root == LazyBooleanParserDomains::EMPTY {
                    continue;
                }
                if let Some(existing) = lanes.get_mut(&root) {
                    *existing = weight_ops.union(existing, &support);
                } else {
                    lanes.insert(root, support);
                }
            }
        }
        domains[source as usize] = lanes;
    }
    eprintln!(
        "[glrmask/validate][lazy_boundary_preimage_dp] exact=true checked={} cache_hits={} expr_nodes={} start_roots={} total_ms={:.3}",
        checked,
        cache_hits,
        arena.node_count(),
        domains[terminal_dwa.start_state() as usize].len(),
        started_at.elapsed().as_secs_f64() * 1000.0,
    );
}


pub(super) fn build_boundary_parser_from_weighted_terminal_paths(
    table: &crate::compiler::glr::table::GLRTable,
    templates: &Templates,
    terminal_dwa: &DWA,
) -> Option<DWA> {
    if std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_TERMINAL_PATH_PREIMAGE").is_none()
        || !terminal_dwa.is_acyclic()
    {
        return None;
    }
    let total_started_at = Instant::now();
    let enumerate_started_at = Instant::now();
    let mut weight_ops = ScopedWeightOpCache::default();
    let mut stack = vec![(terminal_dwa.start_state(), Vec::<u32>::new(), Weight::all())];
    let mut sequences = BTreeMap::<Vec<u32>, Weight>::new();
    let mut visits = 0usize;
    while let Some((state_id, path, support)) = stack.pop() {
        visits += 1;
        if visits >= 1_000_000 {
            return None;
        }
        let state = &terminal_dwa.states()[state_id as usize];
        if let Some(final_weight) = state.final_weight.as_ref() {
            let accepted = weight_ops.intersection(&support, final_weight);
            if !accepted.is_empty() {
                sequences
                    .entry(path.clone())
                    .and_modify(|existing| *existing = weight_ops.union(existing, &accepted))
                    .or_insert(accepted);
            }
        }
        for (&label, (target, weight)) in &state.transitions {
            if label < 0 {
                return None;
            }
            let next_support = weight_ops.intersection(&support, weight);
            if next_support.is_empty() {
                continue;
            }
            let mut next_path = path.clone();
            next_path.push(label as u32);
            stack.push((*target, next_path, next_support));
        }
    }
    let enumerate_ms = enumerate_started_at.elapsed().as_secs_f64() * 1000.0;

    let bundle_started_at = Instant::now();
    let terminals = sequences
        .keys()
        .flat_map(|sequence| sequence.iter().copied())
        .collect::<BTreeSet<_>>();
    let build_bundle = |&terminal: &u32| {
            build_boolean_terminal_bundle_nwa(templates, &[terminal])
                .map(|bundle| (terminal, Arc::new(bundle)))
    };
    let bundles = if macro_parallelism_disabled() {
        let mut timings = Vec::with_capacity(terminals.len());
        let result = terminals
            .iter()
            .filter_map(|terminal| {
                let started = Instant::now();
                let result = build_bundle(terminal);
                timings.push(started.elapsed().as_secs_f64() * 1000.0);
                result
            })
            .collect::<FxHashMap<_, _>>();
        report_macro_item_timings("compose_terminal_path_bundles", &timings);
        result
    } else {
        terminals
            .par_iter()
            .filter_map(build_bundle)
            .collect::<FxHashMap<_, _>>()
    };
    if bundles.len() != terminals.len() {
        return None;
    }
    let bundle_ms = bundle_started_at.elapsed().as_secs_f64() * 1000.0;

    if std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_TERMINAL_PATH_PARALLEL_LOCAL").is_some() {
        let local_started_at = Instant::now();
        let local_domains = sequences
            .par_iter()
            .filter_map(|(sequence, support)| {
                let mut arena = SharedBooleanParserDomains::new();
                let mut root = SharedBooleanParserDomains::UNIVERSAL;
                for &terminal in sequence.iter().rev() {
                    let bundle = bundles.get(&terminal)?;
                    root = arena.preimage_bundle(bundle, root)?;
                    if root == SharedBooleanParserDomains::EMPTY {
                        return None;
                    }
                }
                Some((arena.to_dwa(root), support.clone()))
            })
            .collect::<Vec<_>>();
        let local_ms = local_started_at.elapsed().as_secs_f64() * 1000.0;

        let append_started_at = Instant::now();
        let mut combined = NWA::new(0, 0);
        let start = combined.add_state();
        combined.set_start_states(vec![start]);
        let mut domain_states = 0usize;
        let mut domain_transitions = 0usize;
        for (domain, support) in &local_domains {
            domain_states += domain.num_states() as usize;
            domain_transitions += domain.num_transitions();
            let body = combined.append_with_body(&domain.to_nwa());
            for target in body.start_states {
                combined.add_epsilon(start, target, support.clone());
            }
        }
        let append_ms = append_started_at.elapsed().as_secs_f64() * 1000.0;
        let normalize_started_at = Instant::now();
        let candidate = normalize_weighted_parser_stack_nwa(table, &combined);
        let normalize_ms = normalize_started_at.elapsed().as_secs_f64() * 1000.0;
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_boundary_terminal_path_parallel_local] sequences={} domains={} domain_states={} domain_transitions={} combined_states={} combined_transitions={} output_states={} output_transitions={} local_ms={local_ms:.3} append_ms={append_ms:.3} normalize_ms={normalize_ms:.3} total_ms={:.3}",
                sequences.len(), local_domains.len(), domain_states, domain_transitions,
                combined.num_states(), combined.num_transitions(), candidate.num_states(),
                candidate.num_transitions(), local_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return Some(candidate);
    }

    let preimage_started_at = Instant::now();
    let mut arena = SharedBooleanParserDomains::new();
    let mut preimage_cache = FxHashMap::<(u32, u32), u32>::default();
    let mut preimage_calls = 0usize;
    let mut preimage_hits = 0usize;
    let mut weighted_roots = BTreeMap::<u32, Weight>::new();
    for (sequence, support) in &sequences {
        let mut root = SharedBooleanParserDomains::UNIVERSAL;
        for &terminal in sequence.iter().rev() {
            let key = (terminal, root);
            root = if let Some(&cached) = preimage_cache.get(&key) {
                preimage_hits += 1;
                cached
            } else {
                let bundle = bundles.get(&terminal)?;
                let computed = arena.preimage_bundle(bundle, root)?;
                preimage_cache.insert(key, computed);
                computed
            };
            preimage_calls += 1;
            if root == SharedBooleanParserDomains::EMPTY {
                break;
            }
        }
        if root == SharedBooleanParserDomains::EMPTY {
            continue;
        }
        weighted_roots
            .entry(root)
            .and_modify(|existing| *existing = weight_ops.union(existing, support))
            .or_insert_with(|| support.clone());
    }
    let preimage_ms = preimage_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        let (explicit_total, defaults, zero, one, two, max_explicit) = arena.row_shape_stats();
        eprintln!(
            "[glrmask/profile][constraint_boundary_terminal_path_domain_rows] nodes={} explicit_total={} avg_explicit={:.3} defaults={} zero={} one={} two={} max_explicit={}",
            arena.node_count(), explicit_total, explicit_total as f64 / arena.node_count().max(1) as f64,
            defaults, zero, one, two, max_explicit,
        );
    }

    let combine_started_at = Instant::now();
    let mut roots = weighted_roots.into_iter().collect::<Vec<_>>();
    let use_atoms = std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_TERMINAL_PATH_ATOMS").is_some();
    let atomize_started_at = Instant::now();
    if use_atoms {
        roots = atomize_weighted_parser_roots(&mut arena, &roots);
    }
    let atomize_ms = atomize_started_at.elapsed().as_secs_f64() * 1000.0;
    let candidate = if use_atoms {
        combine_disjoint_weighted_shared_parser_atoms(&mut arena, &roots)
    } else {
        combine_weighted_shared_parser_roots(&mut arena, &roots)
    };
    let combine_ms = combine_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_terminal_path_preimage] terminal_states={} terminal_transitions={} visits={} sequences={} terminals={} graph_nodes={} weighted_roots={} preimage_calls={} preimage_hits={} cache_entries={} output_states={} output_transitions={} enumerate_ms={enumerate_ms:.3} bundle_ms={bundle_ms:.3} preimage_ms={preimage_ms:.3} atomize_ms={atomize_ms:.3} combine_ms={combine_ms:.3} total_ms={:.3}",
            terminal_dwa.num_states(),
            terminal_dwa.num_transitions(),
            visits,
            sequences.len(),
            terminals.len(),
            arena.node_count(),
            roots.len(),
            preimage_calls,
            preimage_hits,
            preimage_cache.len(),
            candidate.num_states(),
            candidate.num_transitions(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(candidate)
}

pub(super) fn build_boundary_parser_from_weighted_terminal_dwa(
    table: &crate::compiler::glr::table::GLRTable,
    templates: &Templates,
    terminal_dwa: &DWA,
) -> Option<DWA> {
    if std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_TERMINAL_DWA_DOMAIN_DP").is_none() {
        return None;
    }
    if !terminal_dwa.is_acyclic() {
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_boundary_terminal_domain_dp] skipped=true reason=cyclic terminal_states={}",
                terminal_dwa.num_states(),
            );
        }
        return None;
    }

    let total_started_at = Instant::now();
    let n = terminal_dwa.states().len();
    let mut indegree = vec![0usize; n];
    for state in terminal_dwa.states() {
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
        for &(target, _) in terminal_dwa.states()[source as usize].transitions.values() {
            indegree[target as usize] -= 1;
            if indegree[target as usize] == 0 {
                queue.push_back(target);
            }
        }
    }
    debug_assert_eq!(topo.len(), n);

    #[derive(Clone)]
    struct TerminalGroup {
        target: u32,
        edge_weight: Weight,
        terminals: Vec<u32>,
        skip_allowed_states: Option<Arc<Vec<u32>>>,
    }
    let use_direct_skip =
        std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_DIRECT_SKIP_PREIMAGE").is_some();
    let mut seen_terminal = vec![false; table.num_terminals as usize];
    let mut non_skip_terminal = vec![false; table.num_terminals as usize];
    let mut skip_states_by_terminal = vec![Vec::<u32>::new(); table.num_terminals as usize];
    if use_direct_skip {
        for (source, row) in table.action.iter().enumerate() {
            for (terminal, action) in row {
                let Some(seen) = seen_terminal.get_mut(terminal as usize) else {
                    continue;
                };
                *seen = true;
                match action {
                    Action::Skip => skip_states_by_terminal[terminal as usize].push(source as u32),
                    _ => non_skip_terminal[terminal as usize] = true,
                }
            }
        }
    }
    let pure_skip = (0..table.num_terminals as usize)
        .map(|terminal| {
            use_direct_skip && seen_terminal[terminal] && !non_skip_terminal[terminal]
        })
        .collect::<Vec<_>>();

    let group_started_at = Instant::now();
    let mut groups_by_state = Vec::<Vec<TerminalGroup>>::with_capacity(n);
    let mut all_terminal_sets = BTreeSet::<Vec<u32>>::new();
    let mut skip_group_count = 0usize;
    for state in terminal_dwa.states() {
        let mut groups = BTreeMap::<(u32, usize, bool), (Weight, Vec<u32>)>::new();
        for (&label, (target, weight)) in &state.transitions {
            if label < 0 {
                return None;
            }
            let terminal = label as u32;
            let is_skip = pure_skip.get(terminal as usize).copied().unwrap_or(false);
            let entry = groups
                .entry((*target, weight.ptr_key(), is_skip))
                .or_insert_with(|| (weight.clone(), Vec::new()));
            entry.1.push(terminal);
        }
        let groups = groups
            .into_iter()
            .map(|((target, _, is_skip), (edge_weight, mut terminals))| {
                terminals.sort_unstable();
                terminals.dedup();
                let skip_allowed_states = if is_skip {
                    skip_group_count += 1;
                    let mut allowed = terminals
                        .iter()
                        .flat_map(|&terminal| {
                            skip_states_by_terminal[terminal as usize].iter().copied()
                        })
                        .collect::<Vec<_>>();
                    allowed.sort_unstable();
                    allowed.dedup();
                    Some(Arc::new(allowed))
                } else {
                    all_terminal_sets.insert(terminals.clone());
                    None
                };
                TerminalGroup {
                    target,
                    edge_weight,
                    terminals,
                    skip_allowed_states,
                }
            })
            .collect::<Vec<_>>();
        groups_by_state.push(groups);
    }
    let groups_ms = group_started_at.elapsed().as_secs_f64() * 1000.0;

    let bundle_started_at = Instant::now();
    let build_bundle = |terminals: &Vec<u32>| {
            build_boolean_terminal_bundle_nwa(templates, terminals)
                .map(|bundle| (terminals.clone(), Arc::new(bundle)))
    };
    let bundles = if macro_parallelism_disabled() {
        let mut timings = Vec::with_capacity(all_terminal_sets.len());
        let result = all_terminal_sets
            .iter()
            .filter_map(|terminals| {
                let started = Instant::now();
                let result = build_bundle(terminals);
                timings.push(started.elapsed().as_secs_f64() * 1000.0);
                result
            })
            .collect::<FxHashMap<_, _>>();
        report_macro_item_timings("compose_terminal_domain_bundles", &timings);
        result
    } else {
        all_terminal_sets
            .par_iter()
            .filter_map(build_bundle)
            .collect::<FxHashMap<_, _>>()
    };
    let bundle_ms = bundle_started_at.elapsed().as_secs_f64() * 1000.0;
    if bundles.len() != all_terminal_sets.len() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_terminal_domain_dp] skipped=true reason=missing_bundle built={} requested={}",
            bundles.len(),
            all_terminal_sets.len(),
        );
        return None;
    }

    let dp_started_at = Instant::now();
    let mut arena = SharedBooleanParserDomains::new();
    let mut domains = vec![BTreeMap::<u32, Weight>::new(); n];
    let mut preimage_cache = FxHashMap::<(Vec<u32>, u32), u32>::default();
    let mut preimage_calls = 0usize;
    let mut preimage_cache_hits = 0usize;
    let mut max_lanes = 0usize;
    let mut weight_ops = ScopedWeightOpCache::default();

    fn merge_lane(
        lanes: &mut BTreeMap<u32, Weight>,
        root: u32,
        weight: Weight,
        weight_ops: &mut ScopedWeightOpCache,
    ) {
        if weight.is_empty() {
            return;
        }
        if let Some(existing) = lanes.get_mut(&root) {
            *existing = weight_ops.union(existing, &weight);
        } else {
            lanes.insert(root, weight);
        }
    }

    for &source in topo.iter().rev() {
        let source_index = source as usize;
        let state = &terminal_dwa.states()[source_index];
        let mut lanes = BTreeMap::<u32, Weight>::new();
        if let Some(final_weight) = state.final_weight.as_ref() {
            merge_lane(
                &mut lanes,
                SharedBooleanParserDomains::UNIVERSAL,
                final_weight.clone(),
                &mut weight_ops,
            );
        }
        for group in &groups_by_state[source_index] {
            let target_lanes = &domains[group.target as usize];
            if target_lanes.is_empty() {
                continue;
            }
            let bundle = group
                .skip_allowed_states
                .is_none()
                .then(|| {
                    bundles
                        .get(&group.terminals)
                        .expect("every non-skip terminal-DWA group bundle must be prebuilt")
                });
            for (&target_root, target_support) in target_lanes {
                let support = weight_ops.intersection(&group.edge_weight, target_support);
                if support.is_empty() {
                    continue;
                }
                let key = (group.terminals.clone(), target_root);
                let source_root = if let Some(&cached) = preimage_cache.get(&key) {
                    preimage_cache_hits += 1;
                    cached
                } else {
                    let root = if let Some(allowed) = group.skip_allowed_states.as_deref() {
                        arena.preimage_identity_skip(target_root, allowed)
                    } else {
                        arena.preimage_bundle(bundle.expect("non-skip bundle must exist"), target_root)?
                    };
                    if std::env::var_os("GLRMASK_VALIDATE_SHARED_BOUNDARY_PREIMAGE").is_some()
                        && group.skip_allowed_states.is_none()
                    {
                        let target_nwa = arena.to_nwa(target_root);
                        let oracle_nwa = build_terminal_bundle_preimage_domain_nwa(
                            table,
                            templates,
                            &group.terminals,
                            &target_nwa,
                        )
                        .expect("oracle preimage must exist when shared preimage exists");
                        let oracle = normalize_parser_stack_domain_nwa_preserving_explicit(
                            table,
                            &oracle_nwa,
                        );
                        let shared_nwa = arena.to_nwa(root);
                        let shared = normalize_parser_stack_domain_nwa_preserving_explicit(
                            table,
                            &shared_nwa,
                        );
                        let difference = find_difference(&shared, &oracle)
                            .expect("parser-domain preimages should be acyclic");
                        if let Some(word) = difference.as_ref() {
                            let first_derivative = word.first().and_then(|&label| {
                                (label >= 0).then(|| {
                                    let derivative = arena.advance(root, label as u32);
                                    (
                                        derivative,
                                        arena.is_universal_root(derivative),
                                        arena.advance_default(root),
                                        arena.explicit_derivatives(root)
                                            .iter()
                                            .find(|(candidate, _)| *candidate == label)
                                            .copied(),
                                    )
                                })
                            });
                            let bundle_debug = bundle.map(|bundle| {
                                bundle
                                    .states()
                                    .iter()
                                    .enumerate()
                                    .map(|(state, node)| {
                                        (
                                            state,
                                            node.final_weight.as_ref().is_some_and(|w| !w.is_empty()),
                                            node.transitions
                                                .iter()
                                                .map(|(&label, targets)| {
                                                    (label, targets.iter().map(|(target, _)| *target).collect::<Vec<_>>())
                                                })
                                                .collect::<Vec<_>>(),
                                            node.epsilons.iter().map(|(target, _)| *target).collect::<Vec<_>>(),
                                        )
                                    })
                                    .collect::<Vec<_>>()
                            });
                            eprintln!(
                                "[glrmask/validate][shared_boundary_preimage_mismatch] terminals={:?} target_root={} shared_root={} witness={:?} shared={} oracle={} first_derivative={:?} bundle={:?}",
                                group.terminals,
                                target_root,
                                root,
                                word,
                                shared.eval_word(word),
                                oracle.eval_word(word),
                                first_derivative,
                                bundle_debug,
                            );
                            panic!("shared parser-domain preimage differs from exact NWA oracle");
                        }
                    }
                    preimage_cache.insert(key, root);
                    root
                };
                preimage_calls += 1;
                if source_root != SharedBooleanParserDomains::EMPTY {
                    merge_lane(&mut lanes, source_root, support, &mut weight_ops);
                }
            }
        }
        max_lanes = max_lanes.max(lanes.len());
        domains[source_index] = lanes;
    }
    let dp_ms = dp_started_at.elapsed().as_secs_f64() * 1000.0;

    if compose_profile_enabled() {
        let start_outer_ranges = domains[terminal_dwa.start_state() as usize]
            .values()
            .map(|weight| weight.raw_range_values().count())
            .sum::<usize>();
        let start_inner_ranges = domains[terminal_dwa.start_state() as usize]
            .values()
            .flat_map(|weight| weight.raw_range_values().map(|(_, tokens)| tokens.ranges().count()))
            .sum::<usize>();
        let start_tsid_cells = domains[terminal_dwa.start_state() as usize]
            .values()
            .flat_map(|weight| weight.raw_range_values().map(|(range, _)| {
                (*range.end() as usize + 1).saturating_sub(*range.start() as usize)
            }))
            .sum::<usize>();
        let start_inner_range_tsid_cells = domains[terminal_dwa.start_state() as usize]
            .values()
            .flat_map(|weight| weight.raw_range_values().map(|(range, tokens)| {
                let width = (*range.end() as usize + 1).saturating_sub(*range.start() as usize);
                width.saturating_mul(tokens.ranges().count())
            }))
            .sum::<usize>();
        eprintln!(
            "[glrmask/profile][constraint_boundary_terminal_domain_dp_phase] phase=dp terminal_states={} groups={} skip_groups={} terminal_sets={} graph_nodes={} start_roots={} start_outer_ranges={} start_inner_ranges={} start_tsid_cells={} start_inner_range_tsid_cells={} max_state_lanes={} preimage_calls={} preimage_cache_hits={} preimage_cache_entries={} groups_ms={groups_ms:.3} bundle_ms={bundle_ms:.3} dp_ms={dp_ms:.3}",
            terminal_dwa.num_states(),
            groups_by_state.iter().map(Vec::len).sum::<usize>(),
            skip_group_count,
            all_terminal_sets.len(),
            arena.node_count(),
            domains[terminal_dwa.start_state() as usize].len(),
            start_outer_ranges,
            start_inner_ranges,
            start_tsid_cells,
            start_inner_range_tsid_cells,
            max_lanes,
            preimage_calls,
            preimage_cache_hits,
            preimage_cache.len(),
        );
    }
    if let Ok(spec) = std::env::var("GLRMASK_DEBUG_BOUNDARY_DOMAIN_LANE") {
        let parts = spec
            .split(':')
            .filter_map(|part| part.parse::<u32>().ok())
            .collect::<Vec<_>>();
        if let [tsid, token, parser_state] = parts.as_slice() {
            let mut carrying = Vec::new();
            for (&root, weight) in &domains[terminal_dwa.start_state() as usize] {
                if weight.tokens_for_tsid(*tsid).contains(*token) {
                    let derivative = arena.advance(root, *parser_state);
                    carrying.push((
                        root,
                        derivative,
                        arena.is_universal_root(derivative),
                        arena.advance_default(root),
                        arena.explicit_derivatives(root)
                            .iter()
                            .find(|(label, _)| *label == *parser_state as i32)
                            .copied(),
                    ));
                }
            }
            eprintln!(
                "[glrmask/debug][constraint_boundary_domain_lane] tsid={} token={} parser_state={} carrying_roots={} rows={:?}",
                tsid,
                token,
                parser_state,
                carrying.len(),
                carrying,
            );
        }
    }
    let direct_started_at = Instant::now();
    let root_weights = domains[terminal_dwa.start_state() as usize]
        .iter()
        .map(|(&root, weight)| (root, weight.clone()))
        .collect::<Vec<_>>();
    let use_atoms = std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_DOMAIN_DISJOINT_ATOMS")
        .is_some();
    let root_weights = if use_atoms
        || std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_DOMAIN_ATOMIZE").is_some()
    {
        atomize_weighted_parser_roots(&mut arena, &root_weights)
    } else {
        root_weights
    };
    let candidate = if use_atoms {
        combine_disjoint_weighted_shared_parser_atoms(&mut arena, &root_weights)
    } else {
        combine_weighted_shared_parser_roots(&mut arena, &root_weights)
    };
    let direct_ms = direct_started_at.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[glrmask/profile][constraint_boundary_terminal_domain_dp] terminal_states={} terminal_transitions={} groups={} skip_groups={} terminal_sets={} graph_nodes={} weighted_start_roots={} max_state_lanes={} preimage_calls={} preimage_cache_hits={} preimage_cache_entries={} output_states={} output_transitions={} groups_ms={groups_ms:.3} bundle_ms={bundle_ms:.3} dp_ms={dp_ms:.3} direct_ms={direct_ms:.3} total_ms={:.3}",
        terminal_dwa.num_states(),
        terminal_dwa.num_transitions(),
        groups_by_state.iter().map(Vec::len).sum::<usize>(),
        skip_group_count,
        all_terminal_sets.len(),
        arena.node_count(),
        root_weights.len(),
        max_lanes,
        preimage_calls,
        preimage_cache_hits,
        preimage_cache.len(),
        candidate.num_states(),
        candidate.num_transitions(),
        total_started_at.elapsed().as_secs_f64() * 1000.0,
    );
    Some(candidate)
}

pub(super) fn build_full_boundary_lazy_direct_parser(
    table: &crate::compiler::glr::table::GLRTable,
    templates: &Templates,
    discovery: &BoundaryTokenDiscovery,
    globally_erasable_ignore_terminals: &BitSet,
    component_state_map: &ManyToOneIdMap,
    id_map: &InternalIdMap,
    seed_relations: &BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>,
) -> Option<DWA> {
    if std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_LAZY_DIRECT_PARSER").is_none() {
        return None;
    }

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct CanonicalKey {
        accepting: bool,
        transitions: Vec<(u32, usize)>,
        epsilons: Vec<usize>,
    }
    #[derive(Debug)]
    struct CanonicalNode {
        accepting: bool,
        transitions: Vec<(u32, usize)>,
        epsilons: Vec<usize>,
    }

    let total_started_at = Instant::now();
    let canonicalize_started_at = Instant::now();
    let mut canonical_by_key = BTreeMap::<CanonicalKey, usize>::new();
    let mut canonical_nodes = Vec::<CanonicalNode>::new();
    let mut witness_starts = Vec::<usize>::with_capacity(discovery.witnesses.len());
    for witness in &discovery.witnesses {
        let mut local_to_canonical = vec![usize::MAX; witness.nodes.len()];
        let mut good_nodes = witness
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(local, node)| witness.good[local].then_some((local, node.key.offset)))
            .collect::<Vec<_>>();
        good_nodes.sort_unstable_by(|left, right| right.1.cmp(&left.1));
        for (local, _) in good_nodes {
            let mut transitions = Vec::new();
            let mut epsilons = Vec::new();
            for edge in witness.nodes[local]
                .outgoing
                .iter()
                .filter(|edge| witness.good[edge.target])
            {
                let target = local_to_canonical[edge.target];
                debug_assert_ne!(target, usize::MAX);
                if globally_erasable_ignore_terminals.contains(edge.terminal as usize) {
                    epsilons.push(target);
                } else {
                    transitions.push((edge.terminal, target));
                }
            }
            transitions.sort_unstable();
            transitions.dedup();
            epsilons.sort_unstable();
            epsilons.dedup();
            let key = CanonicalKey {
                accepting: witness.accepting[local],
                transitions,
                epsilons,
            };
            let canonical = if let Some(&existing) = canonical_by_key.get(&key) {
                existing
            } else {
                let canonical = canonical_nodes.len();
                debug_assert!(key.transitions.iter().all(|(_, target)| *target < canonical));
                debug_assert!(key.epsilons.iter().all(|target| *target < canonical));
                canonical_nodes.push(CanonicalNode {
                    accepting: key.accepting,
                    transitions: key.transitions.clone(),
                    epsilons: key.epsilons.clone(),
                });
                canonical_by_key.insert(key, canonical);
                canonical
            };
            local_to_canonical[local] = canonical;
        }
        witness_starts.push(local_to_canonical[0]);
    }
    let canonicalize_ms = canonicalize_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!("[glrmask/profile][constraint_boundary_lazy_phase] phase=canonicalize canonical_nodes={} witnesses={} ms={canonicalize_ms:.3}", canonical_nodes.len(), witness_starts.len());
    }

    let mut terminal_sets = BTreeSet::<Vec<u32>>::new();
    for node in &canonical_nodes {
        let mut by_target = BTreeMap::<usize, Vec<u32>>::new();
        for &(terminal, target) in &node.transitions {
            by_target.entry(target).or_default().push(terminal);
        }
        for (_, mut terminals) in by_target {
            terminals.sort_unstable();
            terminals.dedup();
            terminal_sets.insert(terminals);
        }
    }
    let bundle_started_at = Instant::now();
    let build_bundle = |terminals: &Vec<u32>| {
            build_boolean_terminal_bundle_nwa(templates, terminals)
                .map(|bundle| (terminals.clone(), Arc::new(bundle)))
    };
    let prebuilt_bundles = if macro_parallelism_disabled() {
        let mut timings = Vec::with_capacity(terminal_sets.len());
        let result = terminal_sets
            .iter()
            .filter_map(|terminals| {
                let started = Instant::now();
                let result = build_bundle(terminals);
                timings.push(started.elapsed().as_secs_f64() * 1000.0);
                result
            })
            .collect::<FxHashMap<_, _>>();
        report_macro_item_timings("compose_boundary_lazy_bundles", &timings);
        result
    } else {
        terminal_sets
            .par_iter()
            .filter_map(build_bundle)
            .collect::<FxHashMap<_, _>>()
    };
    let bundle_ms = bundle_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!("[glrmask/profile][constraint_boundary_lazy_phase] phase=bundles terminal_sets={} built={} ms={bundle_ms:.3}", terminal_sets.len(), prebuilt_bundles.len());
    }
    if prebuilt_bundles.len() != terminal_sets.len() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_lazy_direct_parser] skipped=true reason=missing_bundle built={} requested={}",
            prebuilt_bundles.len(),
            terminal_sets.len(),
        );
        return None;
    }

    let dp_started_at = Instant::now();
    let mut arena = SharedBooleanParserDomains::new();
    let mut roots = vec![SharedBooleanParserDomains::EMPTY; canonical_nodes.len()];
    let mut preimage_calls = 0usize;
    let mut preimage_cache_hits = 0usize;
    let mut preimage_cache = FxHashMap::<(Vec<u32>, u32), u32>::default();
    for (node_id, node) in canonical_nodes.iter().enumerate() {
        if node.accepting {
            roots[node_id] = SharedBooleanParserDomains::UNIVERSAL;
            continue;
        }
        let mut terminals_by_target = BTreeMap::<usize, Vec<u32>>::new();
        for &(terminal, target) in &node.transitions {
            terminals_by_target.entry(target).or_default().push(terminal);
        }
        let mut branches = Vec::<u32>::new();
        for (target, mut terminals) in terminals_by_target {
            terminals.sort_unstable();
            terminals.dedup();
            let bundle = prebuilt_bundles
                .get(&terminals)
                .expect("every lazy-domain terminal bundle must be prebuilt");
            let target_root = roots[target];
            let cache_key = (terminals, target_root);
            let root = if let Some(&cached) = preimage_cache.get(&cache_key) {
                preimage_cache_hits += 1;
                cached
            } else {
                let root = arena.preimage_bundle(bundle, target_root)?;
                preimage_cache.insert(cache_key, root);
                root
            };
            preimage_calls += 1;
            branches.push(root);
        }
        branches.extend(node.epsilons.iter().map(|&target| roots[target]));
        roots[node_id] = arena.union_all(branches);
    }
    let dp_ms = dp_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!("[glrmask/profile][constraint_boundary_lazy_phase] phase=dp expr_nodes={} preimage_calls={} cache_hits={} cache_entries={} ms={dp_ms:.3}", arena.node_count(), preimage_calls, preimage_cache_hits, preimage_cache.len());
    }

    // Batch support by lazy root before constructing Weight objects. Thousands
    // of model-token witnesses routinely share one parser predicate; repeated
    // singleton Weight unions would recreate the old boundary-weight pathology.
    let support_started_at = Instant::now();
    let mut tokens_by_root_tsid = BTreeMap::<u32, BTreeMap<u32, BTreeSet<u32>>>::new();
    for (witness, &start) in discovery.witnesses.iter().zip(&witness_starts) {
        let root = roots[start];
        if root == SharedBooleanParserDomains::EMPTY {
            continue;
        }
        let Some(internal_token) = id_map.internal_token_for_original(witness.token_id) else {
            continue;
        };
        let by_tsid = tokens_by_root_tsid.entry(root).or_default();
        for &raw_state in &witness.start_states {
            let Some(&tsid) = component_state_map.original_to_internal.get(raw_state as usize) else {
                continue;
            };
            if tsid != u32::MAX {
                by_tsid.entry(tsid).or_default().insert(internal_token);
            }
        }
    }

    let mut seed_bundle_cache = FxHashMap::<u32, Arc<NWA>>::default();
    for (sequence, by_state) in seed_relations {
        debug_assert_eq!(sequence.len(), 1);
        let terminal = sequence[0];
        let bundle = if let Some(bundle) = seed_bundle_cache.get(&terminal) {
            Arc::clone(bundle)
        } else {
            let bundle = Arc::new(build_boolean_terminal_bundle_nwa(templates, &[terminal])?);
            seed_bundle_cache.insert(terminal, Arc::clone(&bundle));
            bundle
        };
        let root = arena.preimage_bundle(&bundle, SharedBooleanParserDomains::UNIVERSAL)?;
        if root == SharedBooleanParserDomains::EMPTY {
            continue;
        }
        let by_tsid = tokens_by_root_tsid.entry(root).or_default();
        for (&raw_state, originals) in by_state {
            let Some(&tsid) = component_state_map.original_to_internal.get(raw_state as usize) else {
                continue;
            };
            if tsid == u32::MAX {
                continue;
            }
            let tokens = by_tsid.entry(tsid).or_default();
            tokens.extend(
                originals
                    .iter()
                    .filter_map(|&original| id_map.internal_token_for_original(original)),
            );
        }
    }
    let support_ms = support_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!("[glrmask/profile][constraint_boundary_lazy_phase] phase=support roots={} ms={support_ms:.3}", tokens_by_root_tsid.len());
    }

    let root_started_at = Instant::now();
    let root_weights = tokens_by_root_tsid
        .into_iter()
        .filter_map(|(root, by_tsid)| {
            let weight = Weight::from_per_tsid_token_sets(by_tsid.into_iter().map(
                |(tsid, tokens)| (tsid, tokens.into_iter().collect::<RangeSetBlaze<_>>()),
            ));
            (!weight.is_empty()).then_some((root, weight))
        })
        .collect::<Vec<_>>();
    let candidate = combine_weighted_shared_parser_roots(&mut arena, &root_weights);
    let root_ms = root_started_at.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[glrmask/profile][constraint_boundary_lazy_direct_parser] canonical_nodes={} witnesses={} terminal_sets={} expr_nodes={} preimage_calls={} preimage_cache_hits={} preimage_cache_entries={} weighted_roots={} dwa_states={} dwa_transitions={} canonicalize_ms={canonicalize_ms:.3} bundle_ms={bundle_ms:.3} dp_ms={dp_ms:.3} support_ms={support_ms:.3} direct_ms={root_ms:.3} total_ms={:.3}",
        canonical_nodes.len(),
        witness_starts.len(),
        terminal_sets.len(),
        arena.node_count(),
        preimage_calls,
        preimage_cache_hits,
        preimage_cache.len(),
        root_weights.len(),
        candidate.num_states(),
        candidate.num_transitions(),
        total_started_at.elapsed().as_secs_f64() * 1000.0,
    );
    Some(candidate)
}

pub(super) fn try_prepare_pre_table_boundary_base_discovery(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    global_ignores: bool,
    vocab: &Vocab,
    authoritative_dynamic_boundary: bool,
) -> Option<PreTableBoundaryBaseDiscovery> {
    // Authoritative dynamic B now has an exact pre-table grammar splice and a
    // small terminal factor, so doing this work concurrently with LR linking
    // shortens the critical path. Older experimental composition modes retain
    // their explicit opt-in gate.
    let pretable_requested = authoritative_dynamic_boundary
        || std::env::var_os("GLRMASK_EXPERIMENT_PRETABLE_BASE_DISCOVERY").is_some();
    let experimental_enabled = pretable_requested
        && (authoritative_dynamic_boundary
            || (std::env::var_os("GLRMASK_EXPERIMENT_CROSS_ONLY_BOUNDARY").is_some()
                && std::env::var_os("GLRMASK_EXPERIMENT_COMPONENT_GRAMMAR_SPLICE").is_some()
                && std::env::var_os("GLRMASK_EXPERIMENT_SEGMENTED_PARSER_RUNTIME").is_some()));
    if !experimental_enabled
        || std::env::var_os("GLRMASK_DISABLE_DEFER_BOUNDARY_PARSER_TO_FINAL_UNION").is_some()
        || std::env::var_os("GLRMASK_COMPOSE_GENERIC_BOUNDARY_REFERENCE").is_some()
        || children
            .iter()
            .any(|child| child.constraint.table.embedded_start_nullable())
    {
        return None;
    }

    let started_at = Instant::now();
    let components = std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .collect::<Vec<_>>();
    let mut terminal_offsets = Vec::with_capacity(components.len());
    let mut next_terminal = 0u32;
    for component in &components {
        terminal_offsets.push(next_terminal);
        next_terminal = next_terminal.checked_add(component.table.num_terminals)?;
    }
    let mut placeholder_terminals = Vec::new();
    let mut placeholder_component_indices = Vec::new();
    for (child_index, child) in children.iter().enumerate() {
        for terminal in child.placeholder_terminals() {
            placeholder_terminals.push(terminal);
            placeholder_component_indices.push(child_index + 1);
        }
    }
    let (summary, base_interface_pairs) = compose_nonnullable_grammar_adjacency_summaries(
        &components,
        &terminal_offsets,
        &placeholder_terminals,
        &placeholder_component_indices,
    )
    .ok()?;
    let (tokenizer_state_offsets, merged_tokenizer_state_count) =
        component_tokenizer_state_layout_owned_parent(&components);
    let merged_ignores = merged_ignore_terminals(
        parent,
        children,
        &terminal_offsets,
        global_ignores,
    );
    let mut discovery_interface_pairs = base_interface_pairs.clone();
    if authoritative_dynamic_boundary && !global_ignores {
        // The explicit linker installs a component's scoped ignore as an
        // identity skip throughout that component before linking call/return
        // controls. Therefore whenever a base interface `left -> right` enters
        // a component with scoped ignore `n`, the concrete lexical relation
        // also contains `left -> n -> right`. This does not depend on the
        // finished merged LR state numbering and can be derived while the table
        // is being constructed.
        let base = discovery_interface_pairs.clone();
        for (left, right) in base {
            let owner = terminal_offsets
                .partition_point(|&offset| offset <= right)
                .saturating_sub(1);
            let Some(component) = components.get(owner) else {
                continue;
            };
            let Some(ignore) = component.ignore_terminal else {
                continue;
            };
            let neutral = terminal_offsets[owner] + ignore;
            discovery_interface_pairs.insert((left, neutral));
            discovery_interface_pairs.insert((neutral, right));
        }
    }
    if std::env::var_os(
        "GLRMASK_EXPERIMENT_PRETABLE_CONSERVATIVE_NEUTRAL_INTERFACES",
    )
    .is_some()
    {
        let mut neutral_terminals = BTreeSet::<u32>::new();
        for (component_index, component) in components.iter().enumerate() {
            let offset = terminal_offsets[component_index];
            neutral_terminals.extend(
                component
                    .table
                    .skip_terminals
                    .iter()
                    .map(|terminal| offset + terminal),
            );
            if !global_ignores
                && let Some(ignore) = component.ignore_terminal
            {
                neutral_terminals.insert(offset + ignore);
            }
        }
        for &(left, right) in &base_interface_pairs {
            for &neutral in &neutral_terminals {
                discovery_interface_pairs.insert((left, neutral));
                discovery_interface_pairs.insert((neutral, right));
            }
        }
    }
    if authoritative_dynamic_boundary {
        let owner = |terminal: u32| {
            terminal_offsets
                .partition_point(|&offset| offset <= terminal)
                .saturating_sub(1)
        };
        discovery_interface_pairs.retain(|&(left, right)| owner(left) != owner(right));
    }
    let seed_terminals = vec![false; next_terminal as usize];
    let mut context_terminals = BitSet::new(next_terminal as usize);
    let mut follow_transparent_terminals = BitSet::new(next_terminal as usize);
    for (component_index, component) in components.iter().enumerate() {
        let offset = terminal_offsets[component_index];
        let mut local_neutral = component
            .table
            .skip_terminals
            .iter()
            .map(|terminal| offset + terminal)
            .collect::<BTreeSet<_>>();
        if authoritative_dynamic_boundary
            && !global_ignores
            && let Some(ignore) = component.ignore_terminal
        {
            local_neutral.insert(offset + ignore);
        }
        for terminal in local_neutral {
            if discovery_interface_pairs
                .iter()
                .any(|&(left, right)| left == terminal || right == terminal)
            {
                context_terminals.set(terminal as usize);
            }
            if std::env::var_os("GLRMASK_EXPERIMENT_STRICT_BOUNDARY_FOLLOWS").is_none() {
                follow_transparent_terminals.set(terminal as usize);
            }
        }
    }
    let disallowed_follows = (authoritative_dynamic_boundary
        || std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_GRAMMAR_FOLLOWS").is_some())
        .then(|| {
            disallowed_follows_from_allowed_rows(
                &summary.allowed_follows,
                next_terminal as usize,
            )
        });
    let prebuild_terminal = authoritative_dynamic_boundary
        || std::env::var_os("GLRMASK_EXPERIMENT_PRETABLE_TERMINAL_DWA").is_some();
    let (discovery, discovery_ms, state_map_ms, component_state_map, terminal_artifact, terminal_ms) = if prebuild_terminal {
        let ((discovery, discovery_ms), (component_state_map, state_map_ms)) = macro_join(
            "compose_pretable_discovery_and_state_map",
            || {
                let phase = Instant::now();
                let discovery = discover_boundary_token_paths(
                    vocab,
                    &components,
                    &tokenizer_state_offsets,
                    &terminal_offsets,
                    &seed_terminals,
                    &merged_ignores.global,
                    &discovery_interface_pairs,
                    &context_terminals,
                    &follow_transparent_terminals,
                    disallowed_follows.as_ref(),
                    authoritative_dynamic_boundary,
                    None,
                );
                (discovery, phase.elapsed().as_secs_f64() * 1000.0)
            },
            || {
                let phase = Instant::now();
                let map = component_state_coordinate_map(
                    &components,
                    &tokenizer_state_offsets,
                    merged_tokenizer_state_count,
                );
                (map, phase.elapsed().as_secs_f64() * 1000.0)
            },
        );
        let component_state_map = component_state_map.ok()?;
        let terminal_started_at = Instant::now();
        let terminal_artifact = if !discovery.token_ids.is_empty() {
            let plan = ConcreteBoundaryDeltaPlan {
                original_num_terminals: next_terminal,
                synthetic_num_terminals: next_terminal,
                by_global_terminal: BTreeMap::new(),
                compared_terminals: BTreeSet::new(),
                unsafe_terminals: BTreeSet::new(),
            };
            direct_boundary_terminal_automaton(
                merged_tokenizer_state_count,
                Some(&component_state_map),
                vocab,
                &discovery.token_ids,
                BTreeMap::new(),
                0.0,
                &discovery,
                &merged_ignores.global,
                &BTreeSet::new(),
                &terminal_offsets,
                &tokenizer_state_offsets,
                Some(&plan),
                None,
                authoritative_dynamic_boundary,
            )
            .ok()
        } else {
            None
        };
        let terminal_ms = terminal_started_at.elapsed().as_secs_f64() * 1000.0;
        (
            discovery,
            discovery_ms,
            state_map_ms,
            Some(component_state_map),
            terminal_artifact,
            terminal_ms,
        )
    } else {
        let phase = Instant::now();
        let discovery = discover_boundary_token_paths(
            vocab,
            &components,
            &tokenizer_state_offsets,
            &terminal_offsets,
            &seed_terminals,
            &merged_ignores.global,
            &discovery_interface_pairs,
            &context_terminals,
            &follow_transparent_terminals,
            disallowed_follows.as_ref(),
            authoritative_dynamic_boundary,
            None,
        );
        let discovery_ms = phase.elapsed().as_secs_f64() * 1000.0;
        (discovery, discovery_ms, 0.0, None, None, 0.0)
    };
    let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_pre_table_base_discovery] base_interfaces={} discovery_interfaces={} tokens={} witnesses={} discovery_ms={discovery_ms:.3} state_map_ms={state_map_ms:.3} terminal_ms={terminal_ms:.3} terminal_prebuilt={} total_ms={elapsed_ms:.3}",
            base_interface_pairs.len(),
            discovery_interface_pairs.len(),
            discovery.token_ids.len(),
            discovery.witnesses.len(),
            terminal_artifact.is_some(),
        );
    }
    Some(PreTableBoundaryBaseDiscovery {
        terminal_offsets,
        prefer_cross_interface_only: authoritative_dynamic_boundary,
        tokenizer_state_offsets,
        summary,
        base_interface_pairs,
        discovery_interface_pairs,
        discovery,
        component_state_map,
        terminal_artifact,
        elapsed_ms,
    })
}

pub(super) fn build_boundary_repair(
    composed_table: &ComposedTable,
    merged_tokenizer: Option<&Tokenizer>,
    merged_tokenizer_state_count: usize,
    terminal_display_names: Vec<String>,
    ignore_terminals: &MergedIgnoreTerminals,
    vocab: &Vocab,
    special_token_terminals: &[SpecialTokenTerminal],
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    precomputed_component_state_map: Option<&ManyToOneIdMap>,
    deferred_component_state_map: Option<&OnceLock<Result<ManyToOneIdMap, String>>>,
    pre_table_base_discovery: Option<&PreTableBoundaryBaseDiscovery>,
    selected_boundary_tokens: Option<&OnceLock<Result<Option<Vec<u32>>, String>>>,
    authoritative_segmented_boundary: bool,
    dynamic_boundary_backend: bool,
    static_boundary_components: Option<&BitSet>,
) -> Result<Option<BoundaryRepair>, String> {
    let total_started_at = Instant::now();
    if std::env::var_os("GLRMASK_PROFILE_COMPONENT_GRAMMAR_ANALYSIS").is_some() {
        for (component_index, component) in components.iter().enumerate() {
            let started_at = Instant::now();
            let augmented_start = component
                .table
                .rules
                .first()
                .map(|rule| rule.lhs)
                .ok_or_else(|| format!("component {component_index} has no augmented-start rule"))?;
            let analysis = AnalyzedGrammar::from_composed_rules(
                component.table.rules.clone(),
                component.table.num_terminals,
                component.terminal_display_names.clone(),
                component.table.nonterminal_display_names.clone(),
                augmented_start,
            );
            eprintln!(
                "[glrmask/profile][component_grammar_analysis] component={} states={} rules={} nonterminals={} terminals={} nullable={} ms={:.3}",
                component_index,
                component.table.num_states,
                analysis.rules.len(),
                analysis.num_nonterminals,
                analysis.num_terminals,
                analysis.nullable.len(),
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    if let Ok(capture_path) = std::env::var("GLRMASK_EXPERIMENT_BOUNDARY_TERMINAL_CAPTURE") {
        if std::env::var_os("GLRMASK_EXPERIMENT_SEGMENTED_PARSER_RUNTIME").is_none() {
            return Err(
                "count-only exact boundary capture requires the segmented parser runtime"
                    .to_string(),
            );
        }
        let capture_started_at = Instant::now();
        let capture = load_boundary_terminal_capture(&capture_path)?;
        let capture_load_ms = capture_started_at.elapsed().as_secs_f64() * 1000.0;
        let (terminal_dwa, id_map) = capture.into_parts();
        let num_terminals = composed_table.table.num_terminals;
        let mut active_terminals = vec![false; num_terminals as usize];
        for state in terminal_dwa.states() {
            for &label in state.transitions.keys() {
                if label >= 0
                    && let Some(active) = active_terminals.get_mut(label as usize)
                {
                    *active = true;
                }
            }
        }
        let template_started_at = Instant::now();
        let (templates, template_dfas_by_terminal, templates_ms) =
            try_build_cached_composition_templates_for_terminal_count(
                composed_table,
                components,
                num_terminals,
                &active_terminals,
                &ignore_terminals.scoped,
            )
            .ok_or_else(|| {
                "exact boundary capture requires fully cached count-only composition templates"
                    .to_string()
            })?;
        let template_wall_ms = template_started_at.elapsed().as_secs_f64() * 1000.0;
        // The segmented boundary builder borrows these templates and then
        // moves their DFAs into the recomposition cache after the NWA has been
        // constructed. Avoid cloning all 119 DFAs here.
        let composition_parser_templates_by_terminal = Vec::new();
        if let Some(publication) = selected_boundary_tokens {
            // A segmented parser keeps this boundary's private weight
            // coordinate, so component coordinates need not be refined by its
            // token classes. Exact-capture mode is segmented-only, so there is
            // no flattened consumer that needs an O(|vocab|) support scan.
            let _ = publication.set(Ok(None));
        }
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_boundary_capture] path={} states={} transitions={} active_terminals={} private_token_classes={} load_ms={:.3} templates_reported_ms={templates_ms:.3} templates_wall_ms={template_wall_ms:.3} total_ms={:.3}",
                capture_path,
                terminal_dwa.num_states(),
                terminal_dwa.num_transitions(),
                active_terminals.iter().filter(|&&active| active).count(),
                id_map.num_internal_tokens(),
                capture_load_ms,
                capture_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return Ok(Some(BoundaryRepair {
            parser: BoundaryParserWork::DeferredTerminalCount {
                terminal_automaton: TerminalAutomaton::Dwa(terminal_dwa),
                id_map,
                num_terminals,
                templates,
                prebuilt_bundle_cache: None,
                parser_table_override: None,
            },
            static_boundary_shards: None,
            template_dfas_by_terminal,
            commit_templates_deferred: defer_boundary_commit_templates(),
            composition_parser_templates_by_terminal,
            active_terminals,
            boundary_tokens_by_start_component: None,
        }));
    }

    let cross_only_boundary = authoritative_segmented_boundary
        || std::env::var_os("GLRMASK_EXPERIMENT_CROSS_ONLY_BOUNDARY").is_some();
    // When dynamic A+B retains the explicit linker controls in the composed
    // LR table, those controls are parser-internal epsilon transitions.  They
    // must never become labels in the lexical B automaton; exact B parser
    // advancement closes over them through `advance_control_closed_stacks`.
    let retain_runtime_controls =
        authoritative_segmented_boundary && !composed_table.control_terminals.is_empty();
    let boundary_lexical_control_terminals = if retain_runtime_controls {
        BTreeSet::new()
    } else {
        composed_table.control_terminals.clone()
    };
    // Static B is an accelerator over the same composed LR-state coordinate,
    // but its parser-DWA compiler consumes ordinary terminal effects rather
    // than executing linker controls dynamically.  Compile static B against
    // an exact control-eliminated *view* of the table while keeping the live
    // composed table unchanged and explicitly controlled.  Control
    // elimination preserves LR state IDs, so the resulting parser predicate
    // is still evaluated directly on the authoritative composed GSS.
    let static_boundary_parser_table = if retain_runtime_controls && !dynamic_boundary_backend {
        let started_at = Instant::now();
        let mut table = composed_table.table.clone();
        table.eliminate_control_terminals_exact()?;
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_static_boundary_control_compile_view] states={} controls={} eliminated=true ms={:.3}",
                table.num_states,
                composed_table.control_terminals.len(),
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Some(Arc::new(table))
    } else {
        None
    };
    let boundary_parser_table = static_boundary_parser_table
        .as_deref()
        .unwrap_or(&composed_table.table);
    let fast_component_grammar_splice = cross_only_boundary
        && (authoritative_segmented_boundary
            || (std::env::var_os("GLRMASK_EXPERIMENT_COMPONENT_GRAMMAR_SPLICE").is_some()
                && std::env::var_os("GLRMASK_EXPERIMENT_SEGMENTED_PARSER_RUNTIME").is_some()))
        && std::env::var_os("GLRMASK_DISABLE_DEFER_BOUNDARY_PARSER_TO_FINAL_UNION").is_none()
        && std::env::var_os("GLRMASK_COMPOSE_GENERIC_BOUNDARY_REFERENCE").is_none();
    // In the cross-only factor every accepted repair word has a parser-visible
    // component ownership switch.  The transported component union A contains
    // only component-local words, so no per-terminal `New \ Old` factor is
    // required for B.  Keep an explicit *empty* delta plan (rather than omitting
    // the plan) so the terminal-factor builder mechanically retains only its
    // `CrossedFull` lane; with no changed/unsafe terminals both local lanes are
    // unproductive by construction.
    let cross_only_trivial_delta = fast_component_grammar_splice;
    let analyzed_started_at = Instant::now();
    let (analyzed, spliced_allowed_follows, spliced_base_interface_pairs) =
        if fast_component_grammar_splice {
            let (summary, pairs) = if let Some(precomputed) = pre_table_base_discovery
                .filter(|precomputed| {
                    precomputed.terminal_offsets == composed_table.terminal_offsets
                        && precomputed.tokenizer_state_offsets == tokenizer_state_offsets
                })
            {
                (
                    precomputed.summary.clone(),
                    precomputed.base_interface_pairs.clone(),
                )
            } else {
                compose_nonnullable_grammar_adjacency_summaries(
                    components,
                    &composed_table.terminal_offsets,
                    &composed_table.placeholder_terminals,
                    &composed_table.placeholder_component_indices,
                )?
            };
            // Downstream fast-path code needs only terminal-domain metadata.
            // FIRST/FOLLOW/rule analysis has been replaced exactly by the
            // algebraic component splice above; parser construction is returned
            // through DeferredTerminalCount before any generic consumer can
            // inspect these intentionally-empty grammar-analysis fields.
            let stub = AnalyzedGrammar {
                rules: Vec::new(),
                num_terminals: composed_table.table.num_terminals,
                terminal_display_names: terminal_display_names.clone(),
                protected_shift_terminals: BitSet::new(composed_table.table.num_terminals as usize),
                num_nonterminals: 0,
                nonterminal_display_names: Vec::new(),
                residual_isolation_classes: BTreeMap::new(),
                requires_global_terminal_observation: true,
                direct_regular_automaton: None,
                nullable: BTreeSet::new(),
                first: Vec::new(),
                follow: Vec::new(),
                rules_by_lhs: Vec::new(),
            };
            (stub, Some(summary.allowed_follows), Some(pairs))
        } else {
            let augmented_start = composed_table
                .table
                .rules
                .first()
                .map(|rule| rule.lhs)
                .ok_or_else(|| "composed table contains no augmented-start rule".to_string())?;
            (
                AnalyzedGrammar::from_composed_rules(
                    composed_table.table.rules.clone(),
                    composed_table.table.num_terminals,
                    terminal_display_names.clone(),
                    composed_table.table.nonterminal_display_names.clone(),
                    augmented_start,
                ),
                None,
                None,
            )
        };
    let analyzed_ms = analyzed_started_at.elapsed().as_secs_f64() * 1000.0;

    // A transported component parser DWA already covers paths wholly inside
    // that component. Boundary repair is required whenever the explicit
    // linker can take a zero-width call/return before the next lexical
    // terminal. For a child root N, those lexical beginnings are exactly:
    //
    //   FIRST(N)  â€” paths which begin the child; and
    //   FOLLOW(N) â€” paths which begin the parent continuation.
    //
    // Compute this over the fully composed rule graph, rather than reading the
    // direct child-start/continuation rows. FIRST/FOLLOW propagates through
    // nullable suffixes, adjacent children, and children whose first visible
    // syntax belongs to another nested child.
    let mut seed_terminals = vec![false; composed_table.table.num_terminals as usize];
    if !fast_component_grammar_splice {
        for &nonterminal in &composed_table.boundary_nonterminals {
            let Some(first) = analyzed.first.get(nonterminal as usize) else {
                return Err(format!(
                    "boundary nonterminal {nonterminal} lies outside composed FIRST analysis",
                ));
            };
            let Some(follow) = analyzed.follow.get(nonterminal as usize) else {
                return Err(format!(
                    "boundary nonterminal {nonterminal} lies outside composed FOLLOW analysis",
                ));
            };
            for terminal in first.iter().chain(follow.iter()) {
                if terminal < seed_terminals.len() {
                    seed_terminals[terminal] = true;
                }
            }
        }
    }
    let interface_started_at = Instant::now();
    let base_interface_pairs = spliced_base_interface_pairs.unwrap_or_else(|| {
        visible_boundary_interface_pairs(
            &analyzed,
            &composed_table.boundary_nonterminals,
            &composed_table.control_terminals,
        )
    });
    let interface_pairs = if std::env::var_os("GLRMASK_EXPERIMENT_BASE_BOUNDARY_INTERFACES_ONLY")
        .is_some()
    {
        base_interface_pairs.clone()
    } else {
        extend_boundary_interfaces_through_stack_neutral_lr_actions(
            &composed_table.table,
            &base_interface_pairs,
        )
    };
    let interface_ms = interface_started_at.elapsed().as_secs_f64() * 1000.0;
    let mut follow_transparent_terminals = BitSet::new(seed_terminals.len());
    if std::env::var_os("GLRMASK_EXPERIMENT_STRICT_BOUNDARY_FOLLOWS").is_none() {
        for &terminal in &composed_table.table.skip_terminals {
            follow_transparent_terminals.set(terminal as usize);
        }
    }
    let mut boundary_context_terminals = BitSet::new(seed_terminals.len());
    for &terminal in &composed_table.table.skip_terminals {
        let participates = seed_terminals
            .get(terminal as usize)
            .copied()
            .unwrap_or(false)
            || interface_pairs
                .iter()
                .any(|&(left, right)| left == terminal || right == terminal);
        if participates {
            boundary_context_terminals.set(terminal as usize);
        }
    }
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_interface_pairs] base_pairs={} extended_pairs={} stack_neutral_terminals={}",
            base_interface_pairs.len(),
            interface_pairs.len(),
            composed_table.table.skip_terminals.len(),
        );
    }
    if compose_profile_enabled() {
        let owner = |terminal: u32| {
            composed_table
                .terminal_offsets
                .partition_point(|&offset| offset <= terminal)
                .saturating_sub(1)
        };
        let mut base_owner_counts = BTreeMap::<(usize, usize), usize>::new();
        for &(left, right) in &base_interface_pairs {
            *base_owner_counts.entry((owner(left), owner(right))).or_default() += 1;
        }
        let mut extended_owner_counts = BTreeMap::<(usize, usize), usize>::new();
        for &(left, right) in &interface_pairs {
            *extended_owner_counts.entry((owner(left), owner(right))).or_default() += 1;
        }
        eprintln!("[glrmask/profile][constraint_boundary_interface_owners] base={base_owner_counts:?} extended={extended_owner_counts:?}");
        if std::env::var_os("GLRMASK_PROFILE_BOUNDARY_INTERFACE_PAIRS").is_some() {
            let rows = base_interface_pairs
                .iter()
                .map(|&(left, right)| {
                    (
                        left,
                        analyzed.terminal_display_name(left).to_string(),
                        right,
                        analyzed.terminal_display_name(right).to_string(),
                    )
                })
                .collect::<Vec<_>>();
            eprintln!("[glrmask/profile][constraint_boundary_interface_pairs_detail] {rows:?}");
        }
    }
    // LR-inserted stack-neutral terminals are not global boundary-discovery
    // seeds: that would make every token containing trivia look like a boundary
    // token. They still need their ordinary one-terminal language compiled so a
    // token consisting solely of such a terminal receives the LR `Skip` /
    // state-refining template. Keep that support set separate from FIRST/FOLLOW.
    if compose_profile_enabled() {
        let context = boundary_context_terminals
            .iter()
            .map(|terminal| format!("{}:{}", terminal, analyzed.terminal_display_name(terminal as u32)))
            .collect::<Vec<_>>();
        eprintln!("[glrmask/profile][constraint_boundary_context_terminals] {context:?}");
    }
    let mut one_terminal_support_terminals = if dynamic_boundary_backend {
        // Retained static component DWAs expose ordinary one-terminal
        // component-local tokens. They do *not* expose a parent continuation
        // whose lookahead first performs a zero-width child EOF reduction:
        // while the child is still on the composed stack, no parent-local A
        // projection exists yet. The table linker records exactly those
        // parent-domain lookaheads in `appended_parent_action_terminals`; B
        // owns those return terminals. Dynamic interfaces additionally retain
        // both endpoints because either side may require the dynamic
        // lexer/parser walker rather than a static A contribution. Do not
        // infer return support from every grammar-adjacency endpoint: the LR
        // table already gives the exact continuation set.
        let mut selected = vec![false; seed_terminals.len()];
        for &(left, right) in &interface_pairs {
            let owner = |terminal: u32| {
                composed_table
                    .terminal_offsets
                    .partition_point(|&offset| offset <= terminal)
                    .saturating_sub(1)
            };
            let left_owner = owner(left);
            let right_owner = owner(right);
            let dynamic_interface = components
                .get(left_owner)
                .is_some_and(|component| component.uses_dynamic_runtime())
                || components
                    .get(right_owner)
                    .is_some_and(|component| component.uses_dynamic_runtime());
            if dynamic_interface {
                for terminal in [left, right] {
                    if let Some(slot) = selected.get_mut(terminal as usize) {
                        *slot = true;
                    }
                }
            }
        }
        // Context terminals remain available to multi-terminal boundary-path
        // discovery, but a standalone token consisting only of such a terminal
        // does not cross component ownership and therefore belongs to A, not B.
        selected
    } else if cross_only_boundary {
        // A one-terminal path cannot change component ownership. In segmented
        // mode those paths remain the responsibility of the retained component
        // parser segments; the boundary factor contains only paths with an
        // actual ownership crossing. The exact selected10 B-A oracle has zero
        // one-terminal support, and the general statement follows directly
        // from the cross-component factor definition.
        vec![false; seed_terminals.len()]
    } else {
        seed_terminals.clone()
    };
    if !cross_only_boundary {
        for &terminal in &composed_table.table.skip_terminals {
            if let Some(selected) = one_terminal_support_terminals.get_mut(terminal as usize) {
                *selected = true;
            }
        }
    }
    // The authoritative dynamic cross-boundary factor may safely discard a
    // lexical terminal path whose adjacent parser-visible terminals can never
    // be adjacent in the composed grammar. This is only a necessary-condition
    // filter: the composed LR table remains the authority for actual parser
    // admission, including state-dependent identity/scope-refining terminals.
    // Stack-neutral skip/ignore terminals are deliberately transparent below,
    // because those LR-injected identities need not occur in a grammar RHS.
    // Boundary-path discovery and exact one-byte seed analysis are independent
    // read-only passes over the tokenizer.  Running them serially made boundary
    // repair pay two full million-state/vocabulary scans back-to-back.
    let follow_started_at = Instant::now();
    let boundary_disallowed_follows = if authoritative_segmented_boundary
        || std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_GRAMMAR_FOLLOWS").is_some()
    {
        if let Some(allowed) = spliced_allowed_follows.as_ref() {
            Some(disallowed_follows_from_allowed_rows(
                allowed,
                analyzed.num_terminals as usize,
            ))
        } else {
            Some(crate::compiler::pipeline::compute_disallowed_follows(
                &analyzed,
            ))
        }
    } else {
        None
    };
    let follow_ms = follow_started_at.elapsed().as_secs_f64() * 1000.0;
    if std::env::var_os("GLRMASK_VALIDATE_COMPONENT_GRAMMAR_SPLICE").is_some() {
        let started_at = Instant::now();
        let (spliced_summary, spliced_pairs) = compose_nonnullable_grammar_adjacency_summaries(
            components,
            &composed_table.terminal_offsets,
            &composed_table.placeholder_terminals,
            &composed_table.placeholder_component_indices,
        )?;
        let spliced_follows = &spliced_summary.allowed_follows;
        let reference = compute_ever_allowed_follows(&analyzed);
        let mut differing_rows = Vec::new();
        for terminal in 0..analyzed.num_terminals as usize {
            let mut expected = BitSet::new(analyzed.num_terminals as usize);
            if let Some(row) = reference.get(terminal) {
                for &follow in row {
                    if (follow as usize) < analyzed.num_terminals as usize {
                        expected.set(follow as usize);
                    }
                }
            }
            let actual = spliced_follows
                .get(terminal)
                .cloned()
                .unwrap_or_else(|| BitSet::new(analyzed.num_terminals as usize));
            if actual != expected {
                let actual_only = actual.difference(&expected).iter().take(16).collect::<Vec<_>>();
                let expected_only = expected.difference(&actual).iter().take(16).collect::<Vec<_>>();
                differing_rows.push((terminal, actual_only, expected_only));
                if differing_rows.len() == 16 {
                    break;
                }
            }
        }
        let pair_actual_only = spliced_pairs
            .difference(&base_interface_pairs)
            .copied()
            .take(32)
            .collect::<Vec<_>>();
        let pair_expected_only = base_interface_pairs
            .difference(&spliced_pairs)
            .copied()
            .take(32)
            .collect::<Vec<_>>();
        eprintln!(
            "[glrmask/validate][component_grammar_splice] follow_rows_equal={} differing_rows={differing_rows:?} pairs_equal={} spliced_pairs={} reference_pairs={} pair_actual_only={pair_actual_only:?} pair_expected_only={pair_expected_only:?} ms={:.3}",
            differing_rows.is_empty(),
            pair_actual_only.is_empty() && pair_expected_only.is_empty(),
            spliced_pairs.len(),
            base_interface_pairs.len(),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_grammar_prep] analyzed_ms={analyzed_ms:.3} interface_ms={interface_ms:.3} follow_ms={follow_ms:.3}"
        );
    }
    let eager_all_templates = !fast_component_grammar_splice
        && std::env::var_os("GLRMASK_COMPOSE_SELECTED_TEMPLATES_ONLY").is_none();
    let all_terminals = vec![true; analyzed.num_terminals as usize];
    let eager_changed_parent_enabled = fast_component_grammar_splice
        && std::env::var_os("GLRMASK_EXPERIMENT_EAGER_CHANGED_PARENT_TEMPLATES").is_some();
    let pretransport_active = if std::env::var_os(
        "GLRMASK_EXPERIMENT_EAGER_PRETRANSPORT_TEMPLATES",
    )
    .is_some()
    {
        pre_table_base_discovery
            .filter(|precomputed| {
                fast_component_grammar_splice
                    && precomputed.terminal_offsets == composed_table.terminal_offsets
                    && precomputed.tokenizer_state_offsets == tokenizer_state_offsets
                    && precomputed.discovery_interface_pairs == interface_pairs
            })
            .map(|precomputed| {
                let mut active = one_terminal_support_terminals.clone();
                for terminal in precomputed.discovery.terminals.iter() {
                    if let Some(slot) = active.get_mut(terminal) {
                        *slot = true;
                    }
                }
                for &terminal in &boundary_lexical_control_terminals {
                    if let Some(slot) = active.get_mut(terminal as usize) {
                        *slot = true;
                    }
                }
                active
            })
    } else {
        None
    };
    let (
        (pretransported_templates, prebuilt_bundle_cache, pretransport_ms),
        (
            (eager_templates, eager_changed_parent),
            ((boundary_paths, discovery_ms), (seed_relations, one_byte_ms), (early_posttable_terminal, early_posttable_terminal_ms)),
        ),
    ) = macro_join(
        "compose_boundary_pretransport_and_eager_work",
        || {
            let started_at = Instant::now();
            let transported = pretransport_active.as_ref().and_then(|active| {
                try_rebuild_cached_transported_component_templates(
                    composed_table,
                    components,
                    active,
                )
            });
            let prebuilt_bundle_cache = if std::env::var_os(
                "GLRMASK_EXPERIMENT_EAGER_PREBUILD_UNCHANGED_BUNDLES",
            )
            .is_some()
            {
                transported.as_ref().and_then(|templates| {
                    let terminal_automaton = pre_table_base_discovery
                        .and_then(|precomputed| precomputed.terminal_artifact.as_ref())
                        .map(|artifact| artifact.artifact())?;
                    let candidates = changed_parent_template_candidate_terminals(
                        composed_table,
                        components[0],
                        analyzed.num_terminals,
                    );
                    let mut excluded = vec![false; analyzed.num_terminals as usize];
                    for terminal in candidates.iter() {
                        if let Some(slot) = excluded.get_mut(terminal) {
                            *slot = true;
                        }
                    }
                    Some(prebuild_parser_bundle_cache_excluding_terminals(
                        terminal_automaton,
                        analyzed.num_terminals,
                        templates,
                        &excluded,
                    ))
                })
            } else {
                None
            };
            (
                transported,
                prebuilt_bundle_cache,
                started_at.elapsed().as_secs_f64() * 1000.0,
            )
        },
        || macro_join(
            "compose_boundary_templates_and_discovery",
            || {
                (
                    eager_all_templates.then(|| {
                        build_composition_templates(
                            &composed_table.table,
                            &analyzed,
                            &all_terminals,
                        )
                    }),
                    eager_changed_parent_enabled.then(|| {
                        try_build_changed_parent_templates_for_terminal_count(
                            composed_table,
                            components,
                            analyzed.num_terminals,
                            &ignore_terminals.scoped,
                            !cross_only_trivial_delta,
                        )
                    }).flatten(),
                )
            },
            || {
                let (boundary_result, seed_result) = macro_join(
                    "compose_boundary_paths_and_seed_relations",
                    || {
                        let started_at = Instant::now();
                        let usable_precomputed = pre_table_base_discovery.filter(|precomputed| {
                            fast_component_grammar_splice
                                && precomputed.prefer_cross_interface_only == authoritative_segmented_boundary
                                && precomputed.terminal_offsets == composed_table.terminal_offsets
                                && precomputed.tokenizer_state_offsets == tokenizer_state_offsets
                                && precomputed.base_interface_pairs == base_interface_pairs
                        });
                        let boundary_paths = if let Some(precomputed) = usable_precomputed {
                            let final_discovery_interface_pairs = if precomputed.prefer_cross_interface_only {
                                let owner = |terminal: u32| {
                                    composed_table
                                        .terminal_offsets
                                        .partition_point(|&offset| offset <= terminal)
                                        .saturating_sub(1)
                                };
                                interface_pairs
                                    .iter()
                                    .copied()
                                    .filter(|&(left, right)| owner(left) != owner(right))
                                    .collect::<BTreeSet<_>>()
                            } else {
                                interface_pairs.clone()
                            };
                            let new_interface_pairs = final_discovery_interface_pairs
                                .difference(&precomputed.discovery_interface_pairs)
                                .copied()
                                .collect::<BTreeSet<_>>();
                            let mut combined = precomputed.discovery.clone();
                            let precomputed_interfaces_exact = precomputed.discovery_interface_pairs
                                == final_discovery_interface_pairs;
                            if !precomputed_interfaces_exact && !new_interface_pairs.is_empty() {
                                // Any path that becomes accepting only after LR bridge
                                // interfaces are added contains at least one newly-added
                                // adjacent terminal pair. Since lexer terminals are
                                // non-nullable, the split contributes the adjacent byte
                                // pair (last(left), first(right)) inside the same model
                                // token. Restrict the incremental exact scan to those
                                // tokens, then replace their base witnesses with the full
                                // extended-interface witnesses.
                                let incremental_candidates =
                                    boundary_interface_adjacent_pair_candidates(
                                        vocab,
                                        components,
                                        &composed_table.terminal_offsets,
                                        &new_interface_pairs,
                                    )
                                    .into_iter()
                                    .collect::<BTreeSet<_>>();
                                if !incremental_candidates.is_empty() {
                                    let replacement = discover_boundary_token_paths(
                                        vocab,
                                        components,
                                        tokenizer_state_offsets,
                                        &composed_table.terminal_offsets,
                                        &seed_terminals,
                                        &ignore_terminals.global,
                                        &interface_pairs,
                                        &boundary_context_terminals,
                                        &follow_transparent_terminals,
                                        boundary_disallowed_follows.as_ref(),
                                        authoritative_segmented_boundary,
                                        Some(&incremental_candidates),
                                    );
                                    combined = replace_boundary_discovery_tokens(
                                        combined,
                                        replacement,
                                    );
                                }
                                if compose_profile_enabled() {
                                    eprintln!(
                                        "[glrmask/profile][constraint_boundary_incremental_discovery] base_tokens={} new_pairs={} incremental_candidates={} combined_tokens={}",
                                        precomputed.discovery.token_ids.len(),
                                        new_interface_pairs.len(),
                                        incremental_candidates.len(),
                                        combined.token_ids.len(),
                                    );
                                }
                            } else if compose_profile_enabled() {
                                eprintln!(
                                    "[glrmask/profile][constraint_boundary_incremental_discovery] base_tokens={} new_pairs={} incremental_candidates=0 combined_tokens={} precomputed_interfaces_exact={}",
                                    precomputed.discovery.token_ids.len(),
                                    new_interface_pairs.len(),
                                    combined.token_ids.len(),
                                    precomputed_interfaces_exact,
                                );
                            }

                            if std::env::var_os(
                                "GLRMASK_VALIDATE_PRETABLE_BOUNDARY_DISCOVERY",
                            )
                            .is_some()
                            {
                                let reference = discover_boundary_token_paths(
                                    vocab,
                                    components,
                                    tokenizer_state_offsets,
                                    &composed_table.terminal_offsets,
                                    &seed_terminals,
                                    &ignore_terminals.global,
                                    &interface_pairs,
                                    &boundary_context_terminals,
                                    &follow_transparent_terminals,
                                    boundary_disallowed_follows.as_ref(),
                                    authoritative_segmented_boundary,
                                    None,
                                );
                                assert_eq!(
                                    combined.token_ids, reference.token_ids,
                                    "pre-table incremental boundary token set differs from full discovery",
                                );
                                assert_eq!(
                                    combined.terminals, reference.terminals,
                                    "pre-table incremental boundary terminal set differs from full discovery",
                                );
                                let combined_signature =
                                    boundary_discovery_good_signature(&combined);
                                let reference_signature =
                                    boundary_discovery_good_signature(&reference);
                                if combined_signature != reference_signature {
                                    let brief = |entry: &(
                                        u32,
                                        Vec<u32>,
                                        Vec<(
                                            BoundaryTokenNodeKey,
                                            bool,
                                            Vec<(u32, BoundaryTokenNodeKey)>,
                                        )>,
                                    )| (entry.0, entry.1.clone(), entry.2.len());
                                    let actual_only = combined_signature
                                        .difference(&reference_signature)
                                        .next()
                                        .map(brief);
                                    let expected_only = reference_signature
                                        .difference(&combined_signature)
                                        .next()
                                        .map(brief);
                                    panic!(
                                        "pre-table incremental boundary witnesses differ from full discovery: actual_only={actual_only:?} expected_only={expected_only:?}"
                                    );
                                }
                                eprintln!(
                                    "[glrmask/validate][pretable_boundary_discovery] exact=true tokens={} witnesses={}",
                                    combined.token_ids.len(),
                                    combined.witnesses.len(),
                                );
                            }
                            combined
                        } else {
                            discover_boundary_token_paths(
                                vocab,
                                components,
                                tokenizer_state_offsets,
                                &composed_table.terminal_offsets,
                                &seed_terminals,
                                &ignore_terminals.global,
                                &interface_pairs,
                                &boundary_context_terminals,
                                &follow_transparent_terminals,
                                boundary_disallowed_follows.as_ref(),
                                authoritative_segmented_boundary,
                                None,
                            )
                        };
                        (boundary_paths, started_at.elapsed().as_secs_f64() * 1000.0)
                    },
                    || {
                        let started_at = Instant::now();
                        let relations = if cross_only_boundary && !dynamic_boundary_backend {
                            BTreeMap::new()
                        } else {
                            collect_one_byte_seed_relations_components(
                                components,
                                tokenizer_state_offsets,
                                &composed_table.terminal_offsets,
                                vocab,
                                &one_terminal_support_terminals,
                            )
                        };
                        if std::env::var_os(
                            "GLRMASK_VALIDATE_COMPOSE_COMPONENT_BOUNDARY_VIEW",
                        )
                        .is_some()
                        {
                            let mut reference = BTreeMap::<
                                Vec<u32>,
                                BTreeMap<u32, BTreeSet<u32>>,
                            >::new();
                            let tokenizer = merged_tokenizer.expect(
                                "component boundary-view validation requires a materialized tokenizer",
                            );
                            let all_states = (0..tokenizer.num_states()).collect::<Vec<_>>();
                            collect_one_byte_seed_relations(
                                tokenizer,
                                vocab,
                                &one_terminal_support_terminals,
                                &all_states,
                                &mut reference,
                            );
                            assert_eq!(
                                relations, reference,
                                "component-view one-byte relation differs from merged tokenizer"
                            );
                            eprintln!(
                                "[glrmask/validate][compose_component_one_byte_view] relation_rows={} exact=true",
                                relations.len(),
                            );
                        }
                        (relations, started_at.elapsed().as_secs_f64() * 1000.0)
                    },
                );
                let early_started_at = Instant::now();
                let early_terminal = if std::env::var_os(
                    "GLRMASK_EXPERIMENT_POSTTABLE_TERMINAL_OVERLAP",
                )
                .is_some()
                    && cross_only_trivial_delta
                    && boundary_lexical_control_terminals.is_empty()
                    && !special_token_terminals.iter().any(|special| {
                        boundary_result
                            .0
                            .terminals
                            .contains(special.terminal_id as usize)
                    })
                    && !boundary_result.0.token_ids.is_empty()
                {
                    let component_state_map = if let Some(map) = precomputed_component_state_map {
                        Some(map)
                    } else if let Some(deferred) = deferred_component_state_map {
                        loop {
                            if let Some(result) = deferred.get() {
                                break result.as_ref().ok();
                            }
                            if rayon::yield_now().is_none() {
                                std::thread::yield_now();
                            }
                        }
                    } else {
                        None
                    };
                    component_state_map.and_then(|component_state_map| {
                        let plan = ConcreteBoundaryDeltaPlan {
                            original_num_terminals: analyzed.num_terminals,
                            synthetic_num_terminals: analyzed.num_terminals,
                            by_global_terminal: BTreeMap::new(),
                            compared_terminals: BTreeSet::new(),
                            unsafe_terminals: BTreeSet::new(),
                        };
                        direct_boundary_terminal_automaton(
                            merged_tokenizer_state_count,
                            Some(component_state_map),
                            vocab,
                            &boundary_result.0.token_ids,
                            BTreeMap::new(),
                            0.0,
                            &boundary_result.0,
                            &ignore_terminals.global,
                            &boundary_lexical_control_terminals,
                            &composed_table.terminal_offsets,
                            tokenizer_state_offsets,
                            Some(&plan),
                            None,
                            false,
                        )
                        .ok()
                    })
                } else {
                    None
                };
                let early_ms = early_started_at.elapsed().as_secs_f64() * 1000.0;
                if compose_profile_enabled() && early_terminal.is_some() {
                    eprintln!(
                        "[glrmask/profile][constraint_boundary_posttable_terminal_overlap] states={} transitions={} ms={early_ms:.3}",
                        early_terminal.as_ref().unwrap().artifact().num_states(),
                        early_terminal.as_ref().unwrap().artifact().stats().transitions,
                    );
                }
                (boundary_result, seed_result, (early_terminal, early_ms))
            },
        ));
    if compose_profile_enabled() && pretransported_templates.is_some() {
        eprintln!(
            "[glrmask/profile][constraint_eager_pretransport] templates={} prebuilt_bundles={} ms={pretransport_ms:.3}",
            pretransported_templates.as_ref().map_or(0, |templates| templates.by_terminal.len()),
            prebuilt_bundle_cache.as_ref().map_or(0, PrebuiltParserBundleCache::len),
        );
    }
    let boundary_tokens_by_start_component = boundary_tokens_by_start_component(
        &boundary_paths,
        &composed_table.terminal_offsets,
        tokenizer_state_offsets,
        components.len(),
    );
    let discovered_boundary_terminals = boundary_paths.terminals.clone();
    let mut active_terminals = one_terminal_support_terminals.clone();
    for terminal in discovered_boundary_terminals.iter() {
        active_terminals[terminal] = true;
    }
    if !retain_runtime_controls {
        for &terminal in &composed_table.control_terminals {
            if let Some(active) = active_terminals.get_mut(terminal as usize) {
                *active = true;
            }
        }
    }
    // Reset-complete one-terminal model tokens must be part of the published
    // boundary token coordinate before the owned path prepares its token map.
    // This is the arbitrary-length companion to the one-byte seed relation, so
    // it must use exactly the same one-terminal support set. Multi-terminal
    // cross-boundary paths are already represented by `boundary_paths`; using
    // the whole active-terminal superset here needlessly drags every terminal
    // touched by those paths into one-terminal B support.
    let mut seed_relations = seed_relations;
    if !cross_only_boundary || dynamic_boundary_backend {
        merge_one_terminal_relations(
            &mut seed_relations,
            boundary_delta_reset_relations(
                components,
                &composed_table.terminal_offsets,
                vocab,
                &one_terminal_support_terminals,
                &boundary_lexical_control_terminals,
            ),
        );
    }

    let mut boundary_special_token_terminals = special_token_terminals
        .iter()
        .copied()
        .filter(|special| {
            active_terminals
                .get(special.terminal_id as usize)
                .copied()
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    boundary_special_token_terminals
        .sort_unstable_by_key(|special| (special.token_id, special.terminal_id));
    boundary_special_token_terminals
        .dedup_by_key(|special| (special.token_id, special.terminal_id));
    if !active_terminals.iter().any(|&active| active) {
        if let Some(selected_boundary_tokens) = selected_boundary_tokens {
            let _ = selected_boundary_tokens.set(Ok(None));
        }
        return Ok(None);
    }
    if compose_profile_enabled() {
        let selected_count = active_terminals.iter().filter(|&&active| active).count();
        eprintln!(
            "[glrmask/profile][constraint_boundary_terminals] begin={} one_terminal_support={} discovered={} boundary_tokens={} active={}",
            seed_terminals.iter().filter(|&&selected| selected).count(),
            one_terminal_support_terminals.iter().filter(|&&selected| selected).count(),
            discovered_boundary_terminals.count_ones(),
            boundary_paths.token_ids.len(),
            selected_count,
        );
        if std::env::var_os("GLRMASK_PROFILE_COMPOSE_VERBOSE").is_some() {
            let selected = active_terminals
                .iter()
                .enumerate()
                .filter_map(|(terminal, &active)| {
                    active.then(|| format!("{}:{}", terminal, analyzed.terminal_display_name(terminal as u32)))
                })
                .collect::<Vec<_>>();
            eprintln!("[glrmask/profile][constraint_boundary_terminal_names] selected={selected:?}");
        }
    }

    let selected_original_tokens = seed_relations
        .values()
        .flat_map(|by_state| by_state.values())
        .flat_map(|tokens| tokens.iter().copied())
        .chain(boundary_paths.token_ids.iter().copied())
        .chain(
            boundary_special_token_terminals
                .iter()
                .map(|special| special.token_id),
        )
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if selected_original_tokens.is_empty() {
        if let Some(selected_boundary_tokens) = selected_boundary_tokens {
            let _ = selected_boundary_tokens.set(Ok(None));
        }
        // Internal linker control terminals are zero-width parser actions, not
        // model-token language. An explicit-control composition can therefore
        // need control closure while having no lexical boundary witnesses at
        // all. In that shape there is simply no B terminal artifact: A is
        // evaluated on the control-closed composed parser frontier.
        return Ok(None);
    }
    if let Some(selected_boundary_tokens) = selected_boundary_tokens {
        let _ = selected_boundary_tokens.set(Ok(Some(selected_original_tokens.clone())));
    }

    let final_discovery_interface_pairs = if authoritative_segmented_boundary {
        let owner = |terminal: u32| {
            composed_table
                .terminal_offsets
                .partition_point(|&offset| offset <= terminal)
                .saturating_sub(1)
        };
        interface_pairs
            .iter()
            .copied()
            .filter(|&(left, right)| owner(left) != owner(right))
            .collect::<BTreeSet<_>>()
    } else {
        interface_pairs.clone()
    };
    let prebuilt_terminal_artifact = pre_table_base_discovery
        .filter(|precomputed| {
            fast_component_grammar_splice
                && cross_only_trivial_delta
                && precomputed.prefer_cross_interface_only == authoritative_segmented_boundary
                && precomputed.terminal_offsets == composed_table.terminal_offsets
                && precomputed.tokenizer_state_offsets == tokenizer_state_offsets
                && precomputed.discovery_interface_pairs == final_discovery_interface_pairs
                && precomputed.discovery.token_ids == selected_original_tokens
                && boundary_special_token_terminals.is_empty()
                && boundary_lexical_control_terminals.is_empty()
        })
        .and_then(|precomputed| precomputed.terminal_artifact.as_ref())
        .cloned();
    let prebuilt_terminal_artifact = early_posttable_terminal
        .filter(|_| {
            cross_only_trivial_delta
                && boundary_special_token_terminals.is_empty()
                && boundary_lexical_control_terminals.is_empty()
                && boundary_paths.token_ids == selected_original_tokens
        })
        .or(prebuilt_terminal_artifact);
    let _ = early_posttable_terminal_ms;

    let post_discovery_started_at = Instant::now();
    let mut eager_delta_precompute = None;
    let eager_templates = if let Some(templates) = eager_templates {
        Some(templates)
    } else if let Some(eager) = eager_changed_parent {
        finish_eager_changed_parent_templates(
            eager,
            composed_table,
            components,
            &active_terminals,
            pretransported_templates,
        )
        .map(|(templates, commit_templates, ms, delta)| {
            eager_delta_precompute = Some(delta);
            (templates, commit_templates, ms)
        })
    } else {
        None
    };
    // Any eagerly transported/cached template was characterized against the
    // live explicit-control table. Static B must instead use the exact
    // control-eliminated compile view above, so discard those parser-template
    // accelerators here rather than mixing two table semantics.
    let eager_templates = if static_boundary_parser_table.is_some() {
        eager_delta_precompute = None;
        None
    } else {
        eager_templates
    };
    let eager_finish_wall_ms = post_discovery_started_at.elapsed().as_secs_f64() * 1000.0;

    let state_map_started_at = Instant::now();
    let owned_component_state_map = if precomputed_component_state_map.is_none()
        && deferred_component_state_map.is_none()
    {
        Some(component_state_coordinate_map(
            components,
            tokenizer_state_offsets,
            merged_tokenizer_state_count,
        )?)
    } else {
        None
    };
    let deferred_component_state_map = if let Some(deferred) = deferred_component_state_map {
        let prepared = loop {
            if let Some(prepared) = deferred.get() {
                break prepared;
            }
            if rayon::yield_now().is_none() {
                std::thread::yield_now();
            }
        };
        Some(prepared.as_ref().map_err(Clone::clone)?)
    } else {
        None
    };
    let component_state_map = precomputed_component_state_map
        .or(deferred_component_state_map)
        .or(owned_component_state_map.as_ref())
        .expect("component state map must be available");
    let state_map_ms = state_map_started_at.elapsed().as_secs_f64() * 1000.0;

    let use_concrete_delta =
        std::env::var_os("GLRMASK_DISABLE_CONCRETE_BOUNDARY_TEMPLATE_DELTA").is_none()
            && std::env::var_os("GLRMASK_COMPOSE_GENERIC_BOUNDARY_REFERENCE").is_none();
    let lazy_seed_relations = std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_LAZY_DIRECT_PARSER")
        .is_some()
        .then(|| seed_relations.clone());

    let ((templates, template_dfas_by_terminal, templates_ms), terminal_dwa, concrete_delta_plan) =
        if use_concrete_delta {
            // Delta construction depends on the composed templates, so this
            // concrete-delta path intentionally serializes template construction
            // before the boundary terminal graph. The ordinary path retains
            // the existing rayon overlap below.
            let (mut templates, mut template_dfas_by_terminal, templates_ms) =
                eager_templates.unwrap_or_else(|| {
                    if static_boundary_parser_table.is_some() {
                        build_composition_templates(
                            boundary_parser_table,
                            &analyzed,
                            &active_terminals,
                        )
                    } else if fast_component_grammar_splice {
                        try_build_cached_composition_templates_for_terminal_count(
                            composed_table,
                            components,
                            analyzed.num_terminals,
                            &active_terminals,
                            &ignore_terminals.scoped,
                        )
                        .unwrap_or_else(|| {
                            if dynamic_boundary_backend {
                                // The terminal-trie boundary evaluator advances the
                                // authoritative composed GLR table directly. It needs
                                // the exact cross-boundary terminal language, but no
                                // parser-template DFA. A dynamic component therefore
                                // need not be compiled merely to populate a cache used
                                // only by the static boundary-DWA publisher.
                                (
                                    Templates::default(),
                                    vec![None; analyzed.num_terminals as usize],
                                    0.0,
                                )
                            } else {
                                build_composition_templates(
                                    &composed_table.table,
                                    &analyzed,
                                    &active_terminals,
                                )
                            }
                        })
                    } else {
                        try_build_cached_composition_templates(
                            composed_table,
                            components,
                            &analyzed,
                            &active_terminals,
                            &ignore_terminals.scoped,
                        )
                        .unwrap_or_else(|| {
                            build_composition_templates(
                                &composed_table.table,
                                &analyzed,
                                &active_terminals,
                            )
                        })
                    }
                });
            if eager_all_templates {
                for (terminal, slot) in template_dfas_by_terminal.iter_mut().enumerate() {
                    if !active_terminals[terminal] {
                        *slot = None;
                    }
                }
            }
            let delta_plan_started_at = Instant::now();
            let plan = if cross_only_trivial_delta {
                ConcreteBoundaryDeltaPlan {
                    original_num_terminals: analyzed.num_terminals,
                    synthetic_num_terminals: analyzed.num_terminals,
                    by_global_terminal: BTreeMap::new(),
                    compared_terminals: BTreeSet::new(),
                    unsafe_terminals: BTreeSet::new(),
                }
            } else if let Some(precomputed) = eager_delta_precompute.take() {
                finish_eager_concrete_boundary_delta_plan(
                    precomputed,
                    &active_terminals,
                    analyzed.num_terminals,
                )
            } else {
                prepare_concrete_boundary_delta_plan(
                    composed_table,
                    components,
                    &active_terminals,
                    &templates,
                    analyzed.num_terminals,
                )
            };
            let delta_plan_ms = delta_plan_started_at.elapsed().as_secs_f64() * 1000.0;
            let delta_install_started_at = Instant::now();
            if !cross_only_trivial_delta {
                install_concrete_boundary_delta_templates(&mut templates, &plan);
            }
            let delta_install_ms = delta_install_started_at.elapsed().as_secs_f64() * 1000.0;
            let seed_relations = if std::env::var_os(
                "GLRMASK_DISABLE_FACTOR_ONE_TERMINAL_SEEDS",
            )
            .is_some()
            {
                seed_relations
            } else {
                factor_one_terminal_seed_relations(
                    seed_relations,
                    &plan,
                    components,
                    tokenizer_state_offsets,
                    &composed_table.terminal_offsets,
                )
            };
            let seed_relations = if std::env::var_os("GLRMASK_EXPERIMENT_DROP_ONE_TERMINAL_BOUNDARY").is_some() {
                BTreeMap::new()
            } else {
                seed_relations
            };
            let started_at = Instant::now();
            let result = if let Some(prebuilt) = prebuilt_terminal_artifact.clone() {
                if compose_profile_enabled() {
                    eprintln!(
                        "[glrmask/profile][constraint_boundary_terminal_prebuilt] reused=true states={} transitions={}",
                        prebuilt.artifact().num_states(),
                        prebuilt.artifact().stats().transitions,
                    );
                }
                Ok(prebuilt)
            } else {
                direct_boundary_terminal_automaton(
                    merged_tokenizer_state_count,
                    Some(component_state_map),
                    vocab,
                    &selected_original_tokens,
                    seed_relations,
                    one_byte_ms,
                    &boundary_paths,
                    &ignore_terminals.global,
                    &boundary_lexical_control_terminals,
                    &composed_table.terminal_offsets,
                    tokenizer_state_offsets,
                    Some(&plan),
                    None,
                    dynamic_boundary_backend,
                )
            };
            if compose_profile_enabled() {
                eprintln!(
                    "[glrmask/profile][constraint_boundary_delta_phases] plan_ms={delta_plan_ms:.3} install_ms={delta_install_ms:.3}"
                );
            }
            (
                (templates, template_dfas_by_terminal, templates_ms),
                (result, started_at.elapsed().as_secs_f64() * 1000.0),
                Some(plan),
            )
        } else {
            let (template_result, terminal_result) = macro_join(
                "compose_boundary_templates_and_terminal_dwa",
                || {
                    let (templates, mut template_dfas_by_terminal, templates_ms) =
                        eager_templates.unwrap_or_else(|| {
                            if static_boundary_parser_table.is_some() {
                                build_composition_templates(
                                    boundary_parser_table,
                                    &analyzed,
                                    &active_terminals,
                                )
                            } else if fast_component_grammar_splice {
                                try_build_cached_composition_templates_for_terminal_count(
                                    composed_table,
                                    components,
                                    analyzed.num_terminals,
                                    &active_terminals,
                                    &ignore_terminals.scoped,
                                )
                                .unwrap_or_else(|| {
                                    if dynamic_boundary_backend {
                                        (
                                            Templates::default(),
                                            vec![None; analyzed.num_terminals as usize],
                                            0.0,
                                        )
                                    } else {
                                        build_composition_templates(
                                            &composed_table.table,
                                            &analyzed,
                                            &active_terminals,
                                        )
                                    }
                                })
                            } else {
                                try_build_cached_composition_templates(
                                    composed_table,
                                    components,
                                    &analyzed,
                                    &active_terminals,
                                    &ignore_terminals.scoped,
                                )
                                .unwrap_or_else(|| {
                                    build_composition_templates(
                                        &composed_table.table,
                                        &analyzed,
                                        &active_terminals,
                                    )
                                })
                            }
                        });
                    if eager_all_templates {
                        for (terminal, slot) in template_dfas_by_terminal.iter_mut().enumerate() {
                            if !active_terminals[terminal] {
                                *slot = None;
                            }
                        }
                    }
                    (templates, template_dfas_by_terminal, templates_ms)
                },
                || {
                    let started_at = Instant::now();
                    let result = if std::env::var_os(
                        "GLRMASK_COMPOSE_GENERIC_BOUNDARY_REFERENCE",
                    )
                    .is_some()
                    {
                        match merged_tokenizer {
                            Some(tokenizer) => {
                                let canonicalized_tokenizer = ignore_terminals.canonical.map(|canonical| {
                                    let mut canonicalized = tokenizer.clone();
                                    canonicalized.canonicalize_terminal_aliases(
                                        canonical,
                                        &ignore_terminals.aliases,
                                    );
                                    canonicalized
                                });
                                let tokenizer = canonicalized_tokenizer.as_ref().unwrap_or(tokenizer);
                                let flat_trans: Arc<[u32]> = Arc::from(
                                    crate::compiler::stages::id_map_and_terminal_dwa::l1::
                                        build_flat_transition_table(tokenizer),
                                );
                                let global_max_length_state_map =
                                    crate::compiler::stages::id_map_and_terminal_dwa::
                                        build_global_max_length_state_map(tokenizer, vocab, &flat_trans);
                                let coloring =
                                    TerminalColoring::identity(analyzed.num_terminals as usize);
                                let artifact =
                                    crate::compiler::stages::id_map_and_terminal_dwa::
                                        build_restricted_id_map_and_terminal_dwa_with_precomputed_global_max_length(
                                            tokenizer,
                                            vocab,
                                            &coloring,
                                            false,
                                            ignore_terminals.canonical,
                                            &analyzed,
                                            &BTreeMap::new(),
                                            flat_trans,
                                            &global_max_length_state_map,
                                            None,
                                            Some(&active_terminals),
                                        )
                                        .0;
                                Ok(add_control_loops_to_terminal_artifact(
                                    artifact,
                                    &boundary_lexical_control_terminals,
                                ))
                            }
                            None => Err(
                                "generic boundary reference requires a materialized tokenizer"
                                    .to_string(),
                            ),
                        }
                    } else {
                        direct_boundary_terminal_automaton(
                            merged_tokenizer_state_count,
                            Some(component_state_map),
                            vocab,
                            &selected_original_tokens,
                            seed_relations,
                            one_byte_ms,
                            &boundary_paths,
                            &ignore_terminals.global,
                            &boundary_lexical_control_terminals,
                            &composed_table.terminal_offsets,
                            tokenizer_state_offsets,
                            None,
                            None,
                            dynamic_boundary_backend,
                        )
                    };
                    (result, started_at.elapsed().as_secs_f64() * 1000.0)
                },
            );
            (template_result, terminal_result, None)
        };
    let (terminal_dwa, terminal_ms) = terminal_dwa;
    let profile_delta_selected = if std::env::var_os(
        "GLRMASK_PROFILE_BOUNDARY_SEED_TEMPLATE_DELTA",
    )
    .is_some()
    {
        &seed_terminals
    } else {
        &active_terminals
    };
    profile_current_boundary_template_delta(
        composed_table,
        components,
        profile_delta_selected,
        &templates,
        &boundary_paths,
        tokenizer_state_offsets,
    );
    let template_publish_started_at = Instant::now();
    let mut composition_parser_templates_by_terminal =
        vec![None; analyzed.num_terminals as usize];
    for (&terminal, dfa) in &templates.by_terminal {
        if terminal < analyzed.num_terminals
            && active_terminals
                .get(terminal as usize)
                .copied()
                .unwrap_or(false)
        {
            composition_parser_templates_by_terminal[terminal as usize] = Some(dfa.clone());
        }
    }
    let template_publish_ms = template_publish_started_at.elapsed().as_secs_f64() * 1000.0;
    if std::env::var_os("GLRMASK_COMPOSE_BUILD_FULL_TEMPLATE_CACHE").is_some() {
        let cache_started_at = Instant::now();
        composition_parser_templates_by_terminal = build_complete_composed_parser_template_cache(
            composed_table,
            components,
            &analyzed,
            &ignore_terminals.scoped,
        );
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_composition_parser_template_cache] entries={} states={} transitions={} ms={:.3}",
                composition_parser_templates_by_terminal.iter().flatten().count(),
                composition_parser_templates_by_terminal
                    .iter()
                    .flatten()
                    .map(|dfa| dfa.states.len())
                    .sum::<usize>(),
                composition_parser_templates_by_terminal
                    .iter()
                    .flatten()
                    .flat_map(|dfa| dfa.states.iter())
                    .map(|state| state.transitions.len())
                    .sum::<usize>(),
                cache_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    // Static authoritative B is partitioned before parser-DWA construction.
    // The existing global terminal automaton remains available as a temporary
    // save/load oracle while the partitioned wire format is introduced, but
    // live masking uses these independently compiled start-component shards.
    // Exact-special paths remain on the established global representation for
    // now: an atomic special token has no byte-internal crossing position, and
    // mixing that legacy repair path into the first shard format would blur
    // the proper-internal boundary invariant.
    let static_boundary_shards = if authoritative_segmented_boundary
        && !dynamic_boundary_backend
        && std::env::var_os("GLRMASK_DISABLE_STATIC_BOUNDARY_SHARDS").is_none()
        && boundary_special_token_terminals.is_empty()
        && let Some(plan) = concrete_delta_plan.as_ref()
    {
        Some(build_static_boundary_shard_work(
            merged_tokenizer_state_count,
            component_state_map,
            vocab,
            &boundary_paths,
            &ignore_terminals.global,
            &boundary_lexical_control_terminals,
            &composed_table.terminal_offsets,
            tokenizer_state_offsets,
            plan,
            &boundary_tokens_by_start_component,
            static_boundary_components,
            plan.synthetic_num_terminals,
            &templates,
            static_boundary_parser_table.clone(),
        )?)
    } else {
        None
    };
    let terminal_dwa = terminal_dwa?;
    let special_source_state = merged_tokenizer
        .map(Tokenizer::initial_state_id)
        .unwrap_or(0);
    let special_paths_started_at = Instant::now();
    let terminal_dwa = add_boundary_special_token_paths(
        terminal_dwa,
        &boundary_special_token_terminals,
        special_source_state,
        Some(component_state_map),
        &boundary_lexical_control_terminals,
    )?;
    let special_paths_ms = special_paths_started_at.elapsed().as_secs_f64() * 1000.0;

    let parser_started_at = Instant::now();
    let (terminal_automaton, id_map) = terminal_dwa.into_parts();
    if let TerminalAutomaton::Dwa(dwa) = &terminal_automaton {
        if std::env::var_os("GLRMASK_PROFILE_BOUNDARY_WEIGHTED_TERMINAL_PATHS").is_some() {
            let started_at = Instant::now();
            let mut stack = vec![(dwa.start_state(), Vec::<i32>::new(), Weight::all())];
            let mut accepted_paths = 0usize;
            let mut unique_sequences = BTreeSet::<Vec<i32>>::new();
            let mut max_len = 0usize;
            let mut total_len = 0usize;
            let mut final_support_outer_ranges = 0usize;
            let mut visits = 0usize;
            while let Some((state_id, path, support)) = stack.pop() {
                visits += 1;
                assert!(visits < 1_000_000, "boundary terminal path profiler exceeded acyclic visit bound");
                let state = &dwa.states()[state_id as usize];
                if let Some(final_weight) = state.final_weight.as_ref() {
                    let accepted = support.intersection(final_weight);
                    if !accepted.is_empty() {
                        accepted_paths += 1;
                        max_len = max_len.max(path.len());
                        total_len += path.len();
                        final_support_outer_ranges += accepted.outer_range_count();
                        unique_sequences.insert(path.clone());
                    }
                }
                for (&label, (target, weight)) in &state.transitions {
                    let next_support = support.intersection(weight);
                    if next_support.is_empty() {
                        continue;
                    }
                    let mut next_path = path.clone();
                    next_path.push(label);
                    stack.push((*target, next_path, next_support));
                }
            }
            eprintln!(
                "[glrmask/profile][boundary_weighted_terminal_paths] states={} transitions={} visits={} accepted_paths={} unique_sequences={} avg_len={:.2} max_len={} support_outer_ranges={} total_ms={:.3}",
                dwa.num_states(), dwa.num_transitions(), visits, accepted_paths, unique_sequences.len(),
                if accepted_paths == 0 { 0.0 } else { total_len as f64 / accepted_paths as f64 },
                max_len, final_support_outer_ranges, started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        profile_direct_boundary_terminal_dwa_domain_dp(
            &composed_table.table,
            &templates,
            dwa,
        );
        validate_lazy_boundary_terminal_dwa_preimages(
            &composed_table.table,
            &templates,
            dwa,
        );
    }
    let mut parser_analyzed = analyzed.clone();
    if let Some(plan) = concrete_delta_plan.as_ref() {
        debug_assert_eq!(plan.original_num_terminals, analyzed.num_terminals);
        parser_analyzed.num_terminals = plan.synthetic_num_terminals;
        parser_analyzed
            .terminal_display_names
            .resize(plan.synthetic_num_terminals as usize, "<boundary-delta>".to_string());
    }
    let use_direct_parser =
        std::env::var_os("GLRMASK_EXPERIMENT_USE_BOUNDARY_LAZY_DIRECT_PARSER").is_some()
            || std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_TERMINAL_PATH_PREIMAGE").is_some();
    let validate_direct_parser =
        std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_LAZY_DIRECT_PARSER").is_some();
    if std::env::var_os("GLRMASK_DISABLE_DEFER_BOUNDARY_PARSER_TO_FINAL_UNION").is_none()
        && !use_direct_parser
        && !validate_direct_parser
    {
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_boundary_hidden_phases] eager_finish_wall_ms={eager_finish_wall_ms:.3} state_map_ms={state_map_ms:.3} template_publish_ms={template_publish_ms:.3} special_paths_ms={special_paths_ms:.3}"
            );
            eprintln!(
                "[glrmask/profile][constraint_boundary_build] active={} begin_active={} discovered_active={} boundary_tokens={} boundary_special_tokens={} discovery_ms={discovery_ms:.3} one_byte_ms={one_byte_ms:.3} terminal_ms={terminal_ms:.3} templates_ms={templates_ms:.3} parser_deferred=true parser_ms={:.3} total_ms={:.3}",
                active_terminals.iter().filter(|&&active| active).count(),
                seed_terminals.iter().filter(|&&active| active).count(),
                discovered_boundary_terminals.count_ones(),
                boundary_paths.token_ids.len(),
                boundary_special_token_terminals.len(),
                parser_started_at.elapsed().as_secs_f64() * 1000.0,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        let parser = if fast_component_grammar_splice {
            BoundaryParserWork::DeferredTerminalCount {
                terminal_automaton,
                id_map,
                num_terminals: parser_analyzed.num_terminals,
                templates,
                prebuilt_bundle_cache,
                parser_table_override: static_boundary_parser_table.clone(),
            }
        } else {
            BoundaryParserWork::Deferred {
                terminal_automaton,
                id_map,
                analyzed: parser_analyzed,
                templates,
            }
        };
        return Ok(Some(BoundaryRepair {
            parser,
            static_boundary_shards,
            template_dfas_by_terminal,
            commit_templates_deferred: defer_boundary_commit_templates(),
            composition_parser_templates_by_terminal,
            active_terminals,
            boundary_tokens_by_start_component: Some(boundary_tokens_by_start_component),
        }));
    }
    let terminal_path_candidate = match &terminal_automaton {
        TerminalAutomaton::Dwa(dwa) => {
            build_boundary_parser_from_weighted_terminal_paths(&composed_table.table, &templates, dwa)
        }
        _ => None,
    };
    let terminal_domain_candidate = match &terminal_automaton {
        TerminalAutomaton::Dwa(dwa) => build_boundary_parser_from_weighted_terminal_dwa(
            &composed_table.table,
            &templates,
            dwa,
        ),
        _ => None,
    };
    let lazy_parser_candidate = if concrete_delta_plan.is_none()
        && boundary_special_token_terminals.is_empty()
        && composed_table.control_terminals.is_empty()
    {
        lazy_seed_relations.as_ref().and_then(|seed_relations| {
            build_full_boundary_lazy_direct_parser(
                &composed_table.table,
                &templates,
                &boundary_paths,
                &ignore_terminals.global,
                component_state_map,
                &id_map,
                seed_relations,
            )
        })
    } else {
        None
    };
    let direct_parser_candidate = terminal_path_candidate
        .or(terminal_domain_candidate)
        .or(lazy_parser_candidate);
    let mut generic_parser_dwa = if !use_direct_parser
        || validate_direct_parser
        || direct_parser_candidate.is_none()
    {
        Some(build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
            &composed_table.table,
            &parser_analyzed,
            &terminal_automaton,
            &templates,
            vocab,
            &id_map,
            false,
        ))
    } else {
        None
    };
    if validate_direct_parser {
        if let Some(candidate) = direct_parser_candidate.as_ref() {
            let generic = generic_parser_dwa
                .as_ref()
                .expect("direct-parser validation requires generic reference");
            let mut extra_positive_labels = candidate
                .states()
                .iter()
                .chain(generic.states())
                .flat_map(|state| state.transitions.keys().copied())
                .filter(|&label| {
                    label >= composed_table.table.num_states as i32 && label != DEFAULT_LABEL
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            extra_positive_labels.sort_unstable();
            let candidate_explicit = determinize(&explicit_parser_nwa(
                candidate,
                composed_table.table.num_states,
                &extra_positive_labels,
            ))
            .map_err(|error| error.to_string())?;
            let generic_explicit = determinize(&explicit_parser_nwa(
                generic,
                composed_table.table.num_states,
                &extra_positive_labels,
            ))
            .map_err(|error| error.to_string())?;
            let difference = find_difference(&candidate_explicit, &generic_explicit)
                .map_err(|error| error.to_string())?;
            if let Some(word) = difference.as_ref() {
                let candidate_weight = candidate_explicit.eval_word(word);
                let generic_weight = generic_explicit.eval_word(word);
                let candidate_only = candidate_weight.difference(&generic_weight);
                let generic_only = generic_weight.difference(&candidate_weight);
                let summarize = |weight: &Weight| {
                    weight
                        .raw_range_values()
                        .take(6)
                        .map(|(range, tokens)| {
                            let ranges = tokens.ranges().take(8).collect::<Vec<_>>();
                            ((*range.start(), *range.end()), ranges)
                        })
                        .collect::<Vec<_>>()
                };
                eprintln!(
                    "[glrmask/validate][constraint_boundary_lazy_direct_parser_mismatch] word={word:?} candidate_only={:?} generic_only={:?}",
                    summarize(&candidate_only),
                    summarize(&generic_only),
                );
            }
            assert_eq!(
                difference, None,
                "lazy direct boundary parser differs from generic parser DWA",
            );
            eprintln!(
                "[glrmask/validate][constraint_boundary_lazy_direct_parser] exact=true candidate_states={} generic_states={}",
                candidate.num_states(),
                generic.num_states(),
            );
        }
    }
    let parser_dwa = if use_direct_parser {
        direct_parser_candidate
            .or_else(|| generic_parser_dwa.take())
            .expect("boundary parser construction produced no parser DWA")
    } else {
        generic_parser_dwa
            .take()
            .expect("generic boundary parser must be built when direct path is disabled")
    };
    let pre_hashcons_states = parser_dwa.num_states();
    let pre_hashcons_transitions = parser_dwa.num_transitions();
    let mut hashcons_ms = 0.0;
    let mut boundary_minimize_ms = 0.0;
    let parser_dwa = if pre_hashcons_states >= boundary_parser_minimize_min_states()
        && parser_builder_skips_internal_minimization()
        && std::env::var_os("GLRMASK_DISABLE_BOUNDARY_PARSER_MINIMIZE").is_none()
    {
        let hashcons_started_at = Instant::now();
        let hashconsed = reverse_hashcons_owned(parser_dwa);
        hashcons_ms = hashcons_started_at.elapsed().as_secs_f64() * 1000.0;
        let minimize_started_at = Instant::now();
        let minimized = minimize_owned(hashconsed);
        boundary_minimize_ms = minimize_started_at.elapsed().as_secs_f64() * 1000.0;
        minimized
    } else {
        parser_dwa
    };
    let post_minimize_states = parser_dwa.num_states();
    let post_minimize_transitions = parser_dwa.num_transitions();
    let parser_ms = parser_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_build] active={} begin_active={} discovered_active={} boundary_tokens={} boundary_special_tokens={} discovery_ms={discovery_ms:.3} one_byte_ms={one_byte_ms:.3} terminal_ms={terminal_ms:.3} templates_ms={templates_ms:.3} parser_pre_states={} parser_pre_transitions={} hashcons_ms={hashcons_ms:.3} boundary_minimize_ms={boundary_minimize_ms:.3} parser_post_states={} parser_post_transitions={} parser_ms={parser_ms:.3} total_ms={:.3}",
            active_terminals.iter().filter(|&&active| active).count(),
            seed_terminals.iter().filter(|&&active| active).count(),
            discovered_boundary_terminals.count_ones(),
            boundary_paths.token_ids.len(),
            boundary_special_token_terminals.len(),
            pre_hashcons_states,
            pre_hashcons_transitions,
            post_minimize_states,
            post_minimize_transitions,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(Some(BoundaryRepair {
        parser: BoundaryParserWork::Materialized(MappedArtifact::new(parser_dwa, id_map)),
        static_boundary_shards,
        template_dfas_by_terminal,
        commit_templates_deferred: defer_boundary_commit_templates(),
        composition_parser_templates_by_terminal,
        active_terminals,
        boundary_tokens_by_start_component: Some(boundary_tokens_by_start_component),
    }))
}
