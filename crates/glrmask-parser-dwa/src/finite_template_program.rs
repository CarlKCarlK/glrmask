//! Integer-weight assembly of an ordinary acyclic template-substitution program.
//!
//! This changes representation, not stack semantics. An instance copies its
//! template's topology, stamps every edge with one coefficient, redirects
//! each present final to one continuation port, and connects the entry ports
//! by identity epsilons. It is the same operation as the ordinary builder,
//! without first allocating an expanded range-weight NWA.
use super::*;
use std::ops::Range;

#[derive(Clone, Debug)]
pub struct FiniteTemplateInstance {
    pub template: usize,
    pub coefficient: usize,
    pub continuation: u32,
    pub entries: Range<u32>,
}

pub struct FiniteTemplateProgram<'a> {
    pub templates: &'a [&'a NWA],
    /// Already certified, globally faithful finite-coordinate coefficients.
    pub coefficients: &'a [Weight],
    pub port_finals: &'a [Option<usize>],
    pub starts: &'a [u32],
    pub instances: &'a [FiniteTemplateInstance],
}

#[derive(Debug, Default)]
pub struct FiniteTemplateProgramProfile {
    pub input_states: usize,
    pub input_edges: usize,
    pub positive_states: usize,
    pub positive_edges: usize,
    pub assembly_ms: f64,
    pub early_top_ms: f64,
    pub early_trim_ms: f64,
    pub resolve_ms: f64,
    pub support_ms: f64,
    pub trim_ms: f64,
    pub weight_support_ms: f64,
    pub requotient_ms:f64,
    pub reexpand_ms:f64,
    pub normalize_ms: f64,
}

fn selected_for_state_count(states: usize) -> bool {
    std::env::var_os("GLRMASK_DISABLE_PARSER_SUPPORT_NORMALIZE_SINGLETONS").is_none()
        && (std::env::var_os("GLRMASK_PARSER_SUPPORT_NORMALIZE_SINGLETONS").is_some()
            || states >= std::env::var("GLRMASK_PARSER_SUPPORT_NORMALIZE_SINGLETON_MIN_NWA_STATES")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(4096))
        && (std::env::var_os("GLRMASK_PARSER_SUPPORT_NORMALIZE_SUBSETS").is_none()
            || std::env::var_os("GLRMASK_DISABLE_PARSER_SUPPORT_NORMALIZE_SUBSETS").is_some())
}

fn empty_state() -> FastBoundaryNwaState {
    FastBoundaryNwaState { epsilons: Vec::new(), transitions: Vec::new(), final_weight: 0 }
}

pub(super) fn build(
    program: &FiniteTemplateProgram<'_>, alphabet: u32,
    interner: &mut FastBoundaryWeightInterner, limits: FiniteCompileLimits,
) -> Option<(Vec<FastBoundaryNwaState>, usize, Vec<u32>)> {
    let ports = program.port_finals.len();
    if ports == 0 || ports > limits.states || alphabet == 0
        || alphabet as usize > limits.states || alphabet >= DEFAULT_LABEL as u32
        || program.starts.iter().any(|&q| q as usize >= ports)
        || program.coefficients.len() > limits.weights
    { return None; }
    let mut template_sizes = Vec::with_capacity(program.templates.len());
    for template in program.templates {
        let n = template.states().len();
        if n == 0 || n > limits.states || template.start_states().iter().any(|&q| q as usize >= n)
        { return None; }
        let mut edges = 0usize;
        for row in template.states() {
            edges = edges.checked_add(row.epsilons.len())?
                .checked_add(usize::from(row.final_weight.is_some()))?;
            if row.epsilons.iter().any(|&(q, _)| q as usize >= n) { return None; }
            for (&label, branches) in &row.transitions {
                let valid = label == DEFAULT_LABEL || (0..alphabet as i32).contains(&label)
                    || (is_negative_label(label)
                        && (0..alphabet as i32).contains(&negative_to_positive_label(label)));
                if !valid || branches.iter().any(|&(q, _)| q as usize >= n) { return None; }
                edges = edges.checked_add(branches.len())?;
            }
            if edges > limits.edges { return None; }
        }
        template_sizes.push((n, edges));
    }
    let mut total_states = ports;
    let mut total_edges = 0usize;
    for instance in program.instances {
        let &(states, edges) = template_sizes.get(instance.template)?;
        if instance.coefficient >= program.coefficients.len()
            || instance.continuation as usize >= ports
            || instance.entries.start > instance.entries.end
            || instance.entries.end as usize > ports
        { return None; }
        total_states = total_states.checked_add(states)?;
        total_edges = total_edges.checked_add(edges)?.checked_add(
            (instance.entries.end - instance.entries.start) as usize
                * program.templates[instance.template].start_states().len())?;
        if total_states > limits.states || total_edges > limits.edges { return None; }
    }
    let mut source_ids = FxHashMap::default();
    let coefficients = program.coefficients.iter().map(|w|
        interner.source_weight_id(w, &mut source_ids, None)).collect::<Option<Vec<_>>>()?;
    let mut states = Vec::with_capacity(total_states);
    for weight in program.port_finals {
        let final_weight = match weight { Some(id) => *coefficients.get(*id)?, None => 0 };
        states.push(FastBoundaryNwaState { final_weight, ..empty_state() });
    }
    for instance in program.instances {
        let template = program.templates[instance.template];
        let offset = states.len() as u32;
        let weight = coefficients[instance.coefficient];
        for row in template.states() {
            let mut epsilons = Vec::with_capacity(row.epsilons.len() + usize::from(row.final_weight.is_some()));
            epsilons.extend(row.epsilons.iter().map(|&(q, _)| (offset + q, weight)));
            if row.final_weight.is_some() { epsilons.push((instance.continuation, weight)); }
            let transitions = row.transitions.iter().map(|(&label, branches)| {
                let mut result = SmallVec::with_capacity(branches.len());
                if weight != 0 { result.extend(branches.iter().map(|&(q, _)| (offset + q, weight))); }
                // Match the ordinary signed-native conversion's zero-edge
                // marker convention; empty original keys remain distinct.
                if result.is_empty() && !branches.is_empty() { result.push((0, 0)); }
                (label, result)
            }).collect();
            states.push(FastBoundaryNwaState { epsilons, transitions, final_weight: 0 });
        }
        for port in instance.entries.clone() {
            states[port as usize].epsilons.extend(template.start_states().iter()
                .map(|&q| (offset + q, interner.all_id())));
        }
    }
    if !interner.allow_work(0, states.len(), total_edges) { return None; }
    // The continuation/entry graph is supplied by the caller. Check the whole
    // instantiated program rather than assuming acyclic templates suffice.
    let topology = fast_boundary_topological_order(&states)?;
    Some((states, total_edges, topology))
}

/// Injective positive reachability compaction. Keep every explicit label key,
/// including a zero-weight guard. No two productive targets become equal.
pub(super) fn trim(states: Vec<FastBoundaryNwaState>, starts: &[u32])
    -> Option<(Vec<FastBoundaryNwaState>, Vec<u32>)>
{
    let n = states.len();
    let mut live = vec![false; n];
    let mut todo = starts.to_vec();
    while let Some(q) = todo.pop() {
        let row = states.get(q as usize)?;
        if std::mem::replace(&mut live[q as usize], true) { continue; }
        for &(target, weight) in row.epsilons.iter().chain(row.transitions.iter().flat_map(|(_, b)| b.iter())) {
            if target as usize >= n { return None; }
            if weight != 0 { todo.push(target); }
        }
    }
    let mut map = vec![u32::MAX; n];
    let mut count = 0u32;
    for (q, &keep) in live.iter().enumerate() { if keep { map[q] = count; count += 1; } }
    let mut output = Vec::with_capacity(count as usize);
    for (q, mut row) in states.into_iter().enumerate() {
        if !live[q] { continue; }
        row.epsilons.retain(|&(_, weight)| weight != 0);
        for (target, _) in &mut row.epsilons { *target = map[*target as usize]; }
        for (_, branches) in &mut row.transitions {
            let had_branches = !branches.is_empty();
            branches.retain(|(_, weight)| *weight != 0);
            for (target, _) in branches.iter_mut() { *target = map[*target as usize]; }
            if had_branches && branches.is_empty() { branches.push((0, 0)); }
        }
        output.push(row);
    }
    Some((output, starts.iter().map(|&q| map[q as usize]).collect()))
}

pub fn normalize_finite_template_program(
    program: &FiniteTemplateProgram<'_>, parser_states: u32, rows: usize,
    read_context: Option<&FiniteParserReadSupport>, trim_positive: bool,
) -> Option<(FiniteBoundaryDwa, FiniteTemplateProgramProfile)> {
    let started = Instant::now();
    let limits = FiniteCompileLimits::default();
    let mut interner = FastBoundaryWeightInterner::new(rows, 64)?;
    interner.limits = Some(limits);
    let reuse_topology=std::env::var_os("GLRMASK_BOUNDARY_REUSE_PROGRAM_TOPOLOGY").is_some()
        && std::env::var_os("GLRMASK_BOUNDARY_EARLY_TOP_TRIM").is_none();
    // Virtualize storage only when the existing positive-trim policy applies.
    // Unsupported early-support/diagnostic modes retain the eager reference.
    let virtual_selected=crate::optimized_env_flag("GLRMASK_BOUNDARY_VIRTUAL_TEMPLATE_GRAPH")
        && trim_positive
        && crate::optimized_env_flag("GLRMASK_BOUNDARY_MEMO_CANCELLATIONS")
        && ["GLRMASK_BOUNDARY_EARLY_WEIGHTED_TOP","GLRMASK_BOUNDARY_EARLY_TOP_SUPPORT",
            "GLRMASK_BOUNDARY_EARLY_TOP_TRIM","GLRMASK_VALIDATE_BOUNDARY_MEMO_CANCELLATIONS"]
            .iter().all(|name|std::env::var_os(name).is_none());
    let(mut states,edges,topology_order,mut active_starts,logical_states,assembly_ms,virtual_resolve_ms)=if virtual_selected {
        let(graph,order)=finite_signed_graph::VirtualSignedGraph::build(program,parser_states,&mut interner,limits)?;
        if !selected_for_state_count(graph.profile.logical_states) { return None; }
        let assembly_ms=elapsed_ms(started);
        let phase=Instant::now();
        let filter=crate::optimized_env_flag("GLRMASK_BOUNDARY_CANCELLATION_READ_FILTER");
        let result=graph.resolve_positive_reachable(&mut interner,&order,reuse_topology,filter,program.starts)?;
        let(states,resolve_profile,order,starts)=(result.states,result.profile,result.topology,result.starts);
        let edges=graph.profile.logical_edges;
        let logical_states=graph.profile.logical_states;
        if compile_profile_enabled(){
            eprintln!("[glrmask/profile][virtual_signed_graph] prepare_ms={assembly_ms:.3} graph={:?} resolve={resolve_profile:?}",graph.profile);
        }
        drop(graph);
        (states,edges,order,starts,logical_states,assembly_ms,Some(elapsed_ms(phase)))
    } else {
        let(states,edges,order)=build(program,parser_states,&mut interner,limits)?;
        let logical_states=states.len();
        (states,edges,order,program.starts.to_vec(),logical_states,elapsed_ms(started),None)
    };
    let topology=if reuse_topology{Some(CheckedNativeTopology::from_order(topology_order)?)}else{None};
    if !selected_for_state_count(states.len()) { return None; }
    let mut profile = FiniteTemplateProgramProfile {
        input_states: logical_states, input_edges: edges,
        assembly_ms, ..Default::default()
    };
    if std::env::var_os("GLRMASK_BOUNDARY_EARLY_WEIGHTED_TOP").is_some() {
        // Keep attribution and the old Boolean experiment separate. Failure
        // discards this private graph; the outer caller runs the exact fallback.
        if std::env::var_os("GLRMASK_BOUNDARY_EARLY_TOP_SUPPORT").is_some() { return None; }
        let phase=Instant::now();
        let result=finite_weighted_top::restrict(&mut states,program.starts,parser_states,&mut interner);
        if result.is_none() && compile_profile_enabled() {
            eprintln!("[glrmask/profile][boundary_weighted_top_declined] states={} alphabet={} native_failed={} native_weights={} native_work={} ms={:.3}",
                states.len(),parser_states,interner.failed,interner.values.len(),interner.work,elapsed_ms(phase));
        }
        let statistics=result?;
        profile.early_top_ms=elapsed_ms(phase);
        if compile_profile_enabled() {eprintln!("[glrmask/profile][boundary_weighted_top_support] ms={:.3} stats={statistics:?}",profile.early_top_ms);}
        if std::env::var_os("GLRMASK_BOUNDARY_EARLY_TOP_TRIM").is_some() {
            let phase=Instant::now();
            (states,active_starts)=trim(states,&active_starts)?;
            profile.early_trim_ms=elapsed_ms(phase);
            if compile_profile_enabled(){eprintln!("[glrmask/profile][boundary_weighted_top_trim] states={} ms={:.3}",states.len(),profile.early_trim_ms);}
        }
    }
    if std::env::var_os("GLRMASK_BOUNDARY_EARLY_TOP_SUPPORT").is_some(){
        let phase=Instant::now();
        let statistics=finite_top_support::restrict(&mut states,program.starts,parser_states)?;
        profile.early_top_ms=elapsed_ms(phase);
        if compile_profile_enabled(){eprintln!("[glrmask/profile][boundary_early_top_support] ms={:.3} stats={statistics:?}",profile.early_top_ms);}
        if std::env::var_os("GLRMASK_BOUNDARY_EARLY_TOP_TRIM").is_some(){
            let phase=Instant::now();
            // The same injective compactor is label-agnostic; signed labels
            // are retained unchanged. Every live target keeps a distinct ID.
            (states,active_starts)=trim(states,&active_starts)?;
            profile.early_trim_ms=elapsed_ms(phase);
            if compile_profile_enabled(){eprintln!("[glrmask/profile][boundary_early_top_trim] states={} ms={:.3}",states.len(),profile.early_trim_ms);}
        }
    }
    if let Some(resolve_ms)=virtual_resolve_ms {
        profile.resolve_ms=resolve_ms;
    } else {
        let phase = Instant::now();
        fast_boundary_resolve_negative_codes_with_topology(&mut states, &mut interner, topology.as_ref())?;
        profile.resolve_ms = elapsed_ms(phase);
    }
    let phase = Instant::now();
    if let Some(context) = read_context {
        finite_read_support::restrict_with_topology(&mut states, &active_starts, context, topology.as_ref())?;
    }
    profile.support_ms = elapsed_ms(phase);
    let phase = Instant::now();
    let owned_starts;
    let starts = if trim_positive {
        (states, owned_starts) = trim(states, &active_starts)?;
        owned_starts.as_slice()
    } else { &active_starts };
    profile.trim_ms = elapsed_ms(phase);
    profile.positive_states = states.len();
    profile.positive_edges = states.iter().map(|s| s.epsilons.len()
        + s.transitions.iter().map(|(_, b)| b.len()).sum::<usize>()).sum();
    if !selected_for_state_count(states.len())
        || !interner.allow_work(0, states.len(), profile.positive_edges) { return None; }
    if let Ok(mode)=std::env::var("GLRMASK_BOUNDARY_POSITIVE_WEIGHT_SUPPORT"){
        let phase=Instant::now();
        let stats=finite_weight_support::restrict(&mut states,starts,&mut interner,
            mode!="backward",mode!="forward")?;
        profile.weight_support_ms=elapsed_ms(phase);
        if compile_profile_enabled(){eprintln!("[glrmask/profile][boundary_positive_weight_support] mode={mode} stats={stats:?} ms={:.3}",profile.weight_support_ms);}
    }
    let decoder=if std::env::var_os("GLRMASK_BOUNDARY_REQUOTIENT_POSITIVE_WEIGHTS").is_some(){
        let phase=Instant::now();let decoder=finite_requotient::apply(&mut states,&mut interner)?;
        profile.requotient_ms=elapsed_ms(phase);
        if compile_profile_enabled(){eprintln!("[glrmask/profile][boundary_requotient] selected={} old_points={} new_classes={} rows={} ms={:.3}",decoder.is_some(),decoder.as_ref().map_or(0,|d|d.old_points),decoder.as_ref().map_or(0,|d|d.classes()),interner.tsid_count,profile.requotient_ms);}
        decoder
    }else{None};
    let phase = Instant::now();
    let result = determinize_preconverted_small_boundary_output(
        &states, starts, parser_states, &mut interner, program.coefficients.len(),
        elapsed_ms(started), started, false, true,
    )?;
    profile.normalize_ms = elapsed_ms(phase);
    match result {
        SmallBoundaryDeterminizeOutput::Finite(result) => {
            let phase=Instant::now();let result=if let Some(d)=decoder{d.decode(result)}else{result};
            profile.reexpand_ms=elapsed_ms(phase);Some((result,profile))
        },
        _ => unreachable!("finite output requested"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(program: &FiniteTemplateProgram<'_>) -> NWA {
        let mut result = NWA::new(0, 0);
        for final_id in program.port_finals {
            let q = result.add_state();
            if let Some(id) = final_id { result.set_final_weight(q, program.coefficients[*id].clone()); }
        }
        result.set_start_states(program.starts.to_vec());
        for instance in program.instances {
            let template = program.templates[instance.template];
            let weight = &program.coefficients[instance.coefficient];
            let offset = result.states().len();
            let body = result.append_with_body(template);
            for q in offset..result.states().len() {
                let row = &mut result.states_mut()[q];
                for (_, w) in row.epsilons.iter_mut().chain(row.transitions.values_mut().flatten()) { *w = weight.clone(); }
                if row.final_weight.take().is_some() { result.add_epsilon(q as u32, instance.continuation, weight.clone()); }
            }
            for q in instance.entries.clone() { for &entry in &body.start_states {
                result.add_epsilon(q, entry, Weight::all());
            }}
        }
        result
    }

    #[test]
    fn direct_template_program_matches_ordinary_node_for_node() {
        let mut seed = 349u64;
        let mut next = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); (seed >> 32) as usize };
        for case in 0..256 {
            let mut templates = Vec::new();
            for _ in 0..3 {
                let n = 3 + next() % 6;
                let mut t = NWA::new(0, 0); for _ in 0..n { t.add_state(); } t.set_start_states(vec![0]);
                for q in 0..n {
                    if next() % 3 == 0 || q + 1 == n { t.set_final_weight(q as u32, Weight::empty()); }
                    for target in q + 1..n {
                        let label = [0, 1, DEFAULT_LABEL, crate::compiler::glr::labels::encode_negative_label(1)][next() % 4];
                        if next() % 4 == 0 { t.add_epsilon(q as u32, target as u32, Weight::all()); }
                        else { t.add_transition(q as u32, label, target as u32, Weight::empty()); }
                    }
                    if next() % 3 == 0 { t.states_mut()[q].transitions.entry(2).or_default(); }
                }
                templates.push(t);
            }
            let refs = templates.iter().collect::<Vec<_>>();
            let coefficients = [Weight::all(), Weight::from_token_set_for_tsid(0, [1,3,7].into_iter().collect())];
            let finals = [None, None, Some(1)];
            let instances = [FiniteTemplateInstance { template: 0, coefficient: 1, continuation: 1, entries: 0..1 },
                FiniteTemplateInstance { template: 1, coefficient: 0, continuation: 2, entries: 1..2 },
                FiniteTemplateInstance { template: 2, coefficient: 1, continuation: 2, entries: 0..2 }];
            let program = FiniteTemplateProgram { templates: &refs, coefficients: &coefficients,
                port_finals: &finals, starts: &[0], instances: &instances };
            let expected = reference(&program);
            let mut interner = FastBoundaryWeightInterner::new(1, 64).unwrap();
            let (actual, _, _) = build(&program, 3, &mut interner, Default::default()).unwrap();
            assert_eq!(actual.len(), expected.states().len());
            for (q, (a, b)) in actual.iter().zip(expected.states()).enumerate() {
                assert_eq!(interner.to_weight(a.final_weight), b.final_weight.clone().unwrap_or_else(Weight::empty), "final case={case} q={q}");
                assert_eq!(a.epsilons.iter().map(|&(t,w)|(t,interner.to_weight(w))).collect::<Vec<_>>(), b.epsilons, "eps case={case} q={q}");
                assert_eq!(a.transitions.iter().map(|(l,bs)|(*l,bs.iter().map(|&(t,w)|(t,interner.to_weight(w))).collect::<Vec<_>>())).collect::<BTreeMap<_,_>>(), b.transitions, "row case={case} q={q}");
            }
        }
    }

    #[test]
    fn direct_template_program_declines_cycles_bad_ports_and_limits() {
        let mut t = NWA::new(0, 0); t.add_state(); t.set_start_states(vec![0]); t.set_final_weight(0, Weight::all());
        let coefficients = [Weight::all()]; let refs = [&t]; let finals = [None];
        let instances = [FiniteTemplateInstance { template: 0, coefficient: 0, continuation: 0, entries: 0..1 }];
        let p = FiniteTemplateProgram { templates: &refs, coefficients: &coefficients, port_finals: &finals,
            starts: &[0], instances: &instances };
        let mut i = FastBoundaryWeightInterner::new(1, 64).unwrap();
        assert!(build(&p, 4, &mut i, Default::default()).is_none());
        assert!(build(&p, 4, &mut i, FiniteCompileLimits { states: 1, ..Default::default() }).is_none());
        let mut instances = instances.clone(); instances[0].continuation = 8;
        let p = FiniteTemplateProgram { instances: &instances, ..p };
        assert!(build(&p, 4, &mut i, Default::default()).is_none());
    }

    #[test]
    fn direct_template_program_trim_preserves_native_normalized_prefix_masks() {
        let mut seed = 881u64;
        let mut next = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); (seed >> 32) as usize };
        let mut removed = 0usize;
        for case in 0..512 {
            let n = 3 + next() % 12;
            let mut interner = FastBoundaryWeightInterner::new(1, 64).unwrap();
            interner.limits = Some(Default::default());
            let mut sources = FxHashMap::default();
            let mut weights = vec![0, 1];
            for pattern in [1, 2, 3, 5, 7] {
                let w = Weight::from_token_set_for_tsid(0,
                    (0..4u32).filter(|&b| pattern & (1 << b) != 0).collect());
                weights.push(interner.source_weight_id(&w, &mut sources, None).unwrap());
            }
            let mut states = Vec::new();
            for q in 0..n {
                let mut row = empty_state();
                row.final_weight = weights[next() % weights.len()];
                for label in [0, 1, 2, 3, DEFAULT_LABEL] {
                    let mut branches = SmallVec::new();
                    for target in q + 1..n {
                        if next() % 3 == 0 {
                            branches.push((target as u32, weights[next() % weights.len()]));
                        }
                    }
                    if !branches.is_empty() || next() % 4 == 0 { row.transitions.push((label, branches)); }
                }
                for target in q + 1..n {
                    if next() % 4 == 0 { row.epsilons.push((target as u32, weights[next() % weights.len()])); }
                }
                states.push(row);
            }
            let left = match determinize_preconverted_small_boundary_output(
                &states, &[0], 4, &mut interner, weights.len(), 0.0, Instant::now(), false, true).unwrap() {
                SmallBoundaryDeterminizeOutput::Finite(x) => x.to_generic_dwa(),
                _ => unreachable!(),
            };
            let (states, starts) = trim(states, &[0]).unwrap();
            removed += n - states.len();
            let right = match determinize_preconverted_small_boundary_output(
                &states, &starts, 4, &mut interner, weights.len(), 0.0, Instant::now(), false, true).unwrap() {
                SmallBoundaryDeterminizeOutput::Finite(x) => x.to_generic_dwa(),
                _ => unreachable!(),
            };
            let result = crate::parser_equivalence::compare_parser_mask_prefix_languages(
                &left, &right, 4, 100_000).unwrap();
            assert!(result.difference.is_none(), "case={case}: {:?}", result.difference);
        }
        assert!(removed > 0, "fixtures must exercise actual row removal");
    }
}
