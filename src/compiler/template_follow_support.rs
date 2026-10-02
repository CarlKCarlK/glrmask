//! Conservative terminal adjacency derived only from finite stack relations.
//!
//! Ignoring input guards can add possible output tops, never remove one.
//! Closing that upper bound under controls and testing the next relation's
//! input-domain top certificate therefore proves only impossible pairs.
use std::{collections::BTreeSet, sync::Arc};
use glrmask_artifact::CommitTemplateDfas;
use glrmask_parser_dwa::__private::templates::admissibility::{TemplateDomain, TopAdmission};
use glrmask_glr::__private::glr::labels::{is_negative_label, negative_to_positive_label};

#[derive(Clone, Debug, PartialEq, Eq)]
enum Tops {
    // Includes an unknown lower symbol and the empty concrete stack.
    Any,
    Known(BTreeSet<u32>),
}

fn spend(work: &mut usize, amount: usize) -> bool {
    match work.checked_sub(amount) {
        Some(left) => { *work = left; true }
        None => { *work = 0; false }
    }
}

fn outputs(program: &CommitTemplateDfas, work: &mut usize) -> Result<Tops, String> {
    // Phases 2/3 share the PUSH graph, distinguishing a zero-PUSH arrival from
    // a path which has already written a concrete output top. Incoming edges
    // into an accepting PUSH node supply every possible final written label.
    let mut todo = vec![(0u8, program.pop.start_state)];
    let mut seen = BTreeSet::new();
    let mut result = BTreeSet::new();
    while let Some((phase, state)) = todo.pop() {
        if !spend(work, 1) { return Ok(Tops::Any); }
        if !seen.insert((phase, state)) { continue; }
        let graph = match phase { 0 => &program.pop, 1 => &program.read, _ => &program.push };
        let row = graph.states.get(state as usize).ok_or("template support state leaves its graph")?;
        if row.is_accepting && phase != 3 { return Ok(Tops::Any); }
        if !spend(work, row.transitions.len()) { return Ok(Tops::Any); }
        match phase {
            0 => {
                if let Some(target) = program.pop_to_read.get(state as usize).copied().flatten() {
                    todo.push((1, target));
                }
                if let Some(target) = program.pop_to_push.get(state as usize).copied().flatten() {
                    todo.push((2, target));
                }
                todo.extend(row.transitions.values().map(|&target| (0, target)));
            }
            1 => {
                if let Some(target) = program.read_to_push.get(state as usize).copied().flatten() {
                    todo.push((2, target));
                }
                todo.extend(row.transitions.values().map(|&target| (1, target)));
            }
            _ => {
                for (&label, &target) in &row.transitions {
                    if !is_negative_label(label) { return Err("PUSH support has a non-PUSH label".into()); }
                    let next = graph.states.get(target as usize).ok_or("PUSH support edge leaves its graph")?;
                    if next.is_accepting { result.insert(negative_to_positive_label(label) as u32); }
                    todo.push((3, target));
                }
            }
        }
    }
    Ok(Tops::Known(result))
}

fn close(mut tops: Tops, controls: usize, output: &[Tops], domains: &[TemplateDomain],
    work: &mut usize) -> Tops {
    loop {
        let Tops::Known(values) = &mut tops else { return Tops::Any; };
        let before = values.len();
        for control in controls..output.len() {
            if !spend(work, values.len() + 1) { return Tops::Any; }
            if !values.iter().any(|&top| domains[control].classify_top(top) != TopAdmission::Never) { continue; }
            match &output[control] {
                Tops::Any => return Tops::Any,
                Tops::Known(additions) => { values.extend(additions); }
            }
        }
        if values.len() == before { return tops; }
    }
}

/// Each returned row lists *proved impossible* next terminal IDs. An empty
/// row means no exclusions, not an empty language. A work-budget exhaustion
/// widens the analysis; it never truncates a control closure or a token path.
pub(crate) fn disallowed(programs: &[Option<Arc<CommitTemplateDfas>>], terminal_count: usize,
    budget: usize) -> Result<Vec<Vec<u32>>, String> {
    if terminal_count > programs.len() { return Err("invalid ordinary/control split".into()); }
    let mut work = budget;
    let mut rows = vec![Vec::new(); terminal_count];
    let programs = programs.iter().map(|p| p.as_deref().ok_or("missing template program"))
        .collect::<Result<Vec<_>, _>>()?;
    let mut output = Vec::with_capacity(programs.len());
    let mut domains = Vec::with_capacity(programs.len());
    for program in programs {
        if !spend(&mut work, program.pop.states.len() + program.read.states.len() + program.push.states.len()) {
            return Ok(rows);
        }
        output.push(outputs(program, &mut work)?);
        domains.push(TemplateDomain::compile(program)?);
    }
    for first in 0..terminal_count {
        let Tops::Known(tops) = close(output[first].clone(), terminal_count, &output, &domains, &mut work)
            else { continue; };
        for second in 0..terminal_count {
            if !spend(&mut work, tops.len() + 1) { return Ok(rows); }
            if !tops.iter().any(|&top| domains[second].classify_top(top) != TopAdmission::Never) {
                rows[first].push(second as u32);
            }
        }
    }
    Ok(rows)
}

pub(crate) trait ScopedFollowProgram {
    fn source(&self) -> &CommitTemplateDfas;
    fn offset(&self) -> u32;
    fn append_push(&self) -> Option<u32>;
    fn classify_top(&self, top: u32) -> TopAdmission;
}

/// The same conservative proof using component-local domains and translated
/// output tops. No owned relocated terminal graph is constructed.
pub(crate) fn disallowed_scoped<P: ScopedFollowProgram>(
    programs: &[P],
    terminal_count: usize,
    budget: usize,
) -> Result<Vec<Vec<u32>>, String> {
    if terminal_count > programs.len() { return Err("invalid ordinary/control split".into()); }
    let mut work = budget; let mut rows = vec![Vec::new(); terminal_count];
    let mut output = Vec::with_capacity(programs.len());
    for view in programs {
        let mut tops = outputs(view.source(), &mut work)?;
        if let Tops::Known(values) = &mut tops {
            *values = values.iter().map(|symbol| symbol + view.offset()).collect();
        }
        if let Some(symbol) = view.append_push() { tops = Tops::Known(BTreeSet::from([symbol])); }
        output.push(tops);
    }
    for first in 0..terminal_count {
        let mut tops = output[first].clone();
        loop {
            let Tops::Known(values) = &mut tops else { break; };
            let before = values.len();
            for control in terminal_count..programs.len() {
                if !spend(&mut work, values.len() + 1) { tops = Tops::Any; break; }
                if !values.iter().any(|&top| programs[control].classify_top(top) != TopAdmission::Never) { continue; }
                match &output[control] {
                    Tops::Any => { tops = Tops::Any; break; },
                    Tops::Known(additions) => values.extend(additions),
                }
            }
            if matches!(&tops, Tops::Known(values) if values.len() == before) { break; }
        }
        let Tops::Known(values) = tops else { continue; };
        for second in 0..terminal_count {
            if !spend(&mut work, values.len() + 1) { return Ok(rows); }
            if !values.iter().any(|&top| programs[second].classify_top(top) != TopAdmission::Never) {
                rows[first].push(second as u32);
            }
        }
    }
    Ok(rows)
}
