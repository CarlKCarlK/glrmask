//! Shared finite control-gap assembly for static parser queries.
//!
//! The lexical graph chooses ordinary terminals and their correlated token
//! weights. A bounded control certificate supplies the zero-width gap depth.
//! No LR table, parser action row, or grammar production is consulted here.
use super::*;

pub(super) fn assemble(
    templates: &Templates,
    controls: &[u32],
    max_controls_per_gap: u32,
    lexical: &DWA,
    state_budget: Option<usize>,
) -> Result<(NWA, usize, usize), String> {
    assemble_impl(&templates.by_terminal_nwa, controls, Some(max_controls_per_gap), lexical, state_budget)
}

fn assemble_impl(
    templates: &BTreeMap<u32, NWA>,
    controls: &[u32],
    bounded_depth: Option<u32>,
    lexical: &DWA,
    state_budget: Option<usize>,
) -> Result<(NWA, usize, usize), String> {
    let depths = match bounded_depth {
        Some(depth) => usize::try_from(depth).ok().and_then(|n| n.checked_add(1))
            .ok_or("static template control depth overflow")?,
        // Exact C*: only the after-consuming port 0 may publish a lexical
        // final. The after-control port 1 loops through arbitrary controls.
        // Two ports do not bound the number of zero-width parser advances.
        None => 2,
    };
    let port_count = lexical.states().len().checked_mul(depths)
        .filter(|&n| n <= u32::MAX as usize).ok_or("static template port count overflow")?;
    let mut arena = NWA::new(0, 0);
    let mut ordinary_states = 0usize;
    let mut control_states = 0usize;
    let check_growth = |current: usize, extra: usize| -> Result<(), String> {
        let next = current.checked_add(extra).filter(|&n| n <= u32::MAX as usize)
            .ok_or("static template graph coordinate overflow")?;
        if state_budget.is_some_and(|limit| next > limit) {
            return Err("static template graph exceeds representation budget; no relation was truncated".into());
        }
        Ok(())
    };
    check_growth(0, port_count)?;
    let mut ports = Vec::with_capacity(port_count);
    for (index, row) in lexical.states().iter().enumerate() {
        for _ in 0..depths { ports.push(arena.add_state()); }
        // Token admission observes no trailing control closure: a lexical
        // final can publish only immediately after its consuming transition.
        if let Some(weight) = &row.final_weight {
            if !weight.is_empty() { arena.set_final_weight(ports[index * depths], weight.clone()); }
        }
    }
    let start = *ports.get(lexical.start_state() as usize * depths)
        .ok_or("static template query has no start port")?;
    arena.set_start_states(vec![start]);
    for (index, row) in lexical.states().iter().enumerate() {
        for (terminal, target, weight) in row.transitions.entries() {
            if terminal < 0 { return Err("static lexical query contains a negative terminal label".into()); }
            if weight.is_empty() { continue; }
            let template = templates.get(&(terminal as u32))
                .ok_or_else(|| format!("static query has no template for terminal {terminal}"))?;
            let continuation = *ports.get(target as usize * depths)
                .ok_or("static lexical query edge leaves its graph")?;
            check_growth(arena.states().len(), template.states().len())?;
            let body = append_weighted_fragment(&mut arena, template, weight, continuation)?;
            ordinary_states += template.states().len();
            for depth in 0..depths {
                for &entry in &body.start_states {
                    arena.add_epsilon(ports[index * depths + depth], entry, Weight::all());
                }
            }
        }
    }
    for index in 0..lexical.states().len() {
        let control_layers = if bounded_depth.is_some() { depths - 1 } else { 1 };
        for depth in 0..control_layers {
            for &control in controls {
                let template = templates.get(&control)
                    .ok_or_else(|| format!("static query has no template for control {control}"))?;
                check_growth(arena.states().len(), template.states().len())?;
                let body = append_weighted_fragment(&mut arena, template, &Weight::all(),
                    ports[index * depths + depth + 1])?;
                control_states += template.states().len();
                for &entry in &body.start_states {
                    arena.add_epsilon(ports[index * depths + depth], entry, Weight::all());
                    if bounded_depth.is_none() {
                        arena.add_epsilon(ports[index * depths + 1], entry, Weight::all());
                    }
                }
            }
        }
    }
    Ok((arena, ordinary_states, control_states))
}

/// Complete static compilation from persisted template programs. In contrast
/// to an LR-reachability optimizer, the final normalizer is exact on every
/// finite concrete stack suffix, including an empty suffix after a final POP.
pub(crate) fn compile(
    templates: &Templates,
    controls: &[u32],
    certificate: &ClosureCertificate,
    lexical: &DWA,
    symbol_count: u32,
) -> Result<SignedShardOutput, String> {
    compile_impl(&templates.by_terminal_nwa, controls, Some(certificate.max_controls_per_gap), lexical, symbol_count, None)
}

/// Exact unbounded control closure for nullable components. This finite cyclic
/// signed NWA is solved by the shared weighted cancellation and normalization
/// fixed point, not by truncating a control path to a chosen depth.
pub(crate) fn compile_saturated(
    templates: &Templates,
    controls: &[u32],
    lexical: &DWA,
    symbol_count: u32,
) -> Result<SignedShardOutput, String> {
    compile_impl(&templates.by_terminal_nwa, controls, None, lexical, symbol_count, None)
}

pub(crate) fn compile_classed(
    templates: &BTreeMap<u32, NWA>,
    controls: &[u32],
    certificate: Option<&ClosureCertificate>,
    lexical: &DWA,
    classes: &glrmask_parser_dwa::__private::pop_classes::PopLabelClasses,
) -> Result<SignedShardOutput, String> {
    compile_impl(templates, controls, certificate.map(|c| c.max_controls_per_gap),
        lexical, classes.symbol_count(), Some(classes))
}

fn compile_impl(
    templates: &BTreeMap<u32, NWA>,
    controls: &[u32],
    bounded_depth: Option<u32>,
    lexical: &DWA,
    symbol_count: u32,
    classes: Option<&glrmask_parser_dwa::__private::pop_classes::PopLabelClasses>,
) -> Result<SignedShardOutput, String> {
    if !lexical.is_acyclic() { return Err("static template lexical query is cyclic".into()); }
    let start = Instant::now();
    let (mut program, _, _) = assemble_impl(templates, controls,
        bounded_depth, lexical, Some(1_000_000))?;
    if bounded_depth.is_some() && !program.is_acyclic() {
        return Err("static template program violates its finite control certificate".into());
    }
    let signed_states = program.states().len();
    let signed_transitions = program.num_transitions();
    if signed_transitions > 8_000_000 {
        return Err("static template program exceeds its transition budget".into());
    }
    let compose_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    if let Some(classes) = classes {
        glrmask_parser_dwa::__private::resolve_negatives::resolve_negative_codes_in_nwa_with_pop_classes(
            &mut program, classes)?;
        let resolve_ms = start.elapsed().as_secs_f64() * 1000.0;
        let normalize_started = Instant::now();
        let parser_dwa = classes.compile_positive(program, 8_000_000)?;
        return Ok(SignedShardOutput { parser_dwa, templates_ms: 0.0, compose_ms, resolve_ms,
            normalize_ms: normalize_started.elapsed().as_secs_f64()*1000.0,
            signed_states, signed_transitions, terms: templates.len().saturating_sub(controls.len()) });
    } else {
        resolve_negative_codes_in_nwa(&mut program, false);
    }
    let resolve_ms = start.elapsed().as_secs_f64() * 1000.0;
    if program.states().iter().any(|row| row.transitions.keys().any(|&label| is_negative_label(label))) {
        return Err("static template cancellation left a negative stack action".into());
    }
    let start = Instant::now();
    let parser_dwa = crate::compiler::stages::parser_dwa::normalize_weighted_stack_predicate_for_symbol_count(
        symbol_count, &program);
    Ok(SignedShardOutput { parser_dwa, templates_ms: 0.0, compose_ms, resolve_ms,
        normalize_ms: start.elapsed().as_secs_f64() * 1000.0,
        signed_states, signed_transitions,
        terms: templates.len().saturating_sub(controls.len()) })
}
