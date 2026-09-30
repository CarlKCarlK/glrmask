//! Compile-time operations on the same finite stack programs used at runtime.
//! DEFAULT is a per-source-state fallback, not an ordinary union alphabet
//! symbol: explicit dead edges continue to shadow it during determinization.
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use crate::automata::unweighted_u32::{dfa::DFA, nfa::NFA};
use crate::compiler::glr::labels::{DEFAULT_LABEL, encode_negative_label, negative_to_positive_label};
use super::CommitTemplateDfas;
use glrmask_parser_dwa::__private::templates::compile_dfa::{
    recombine_split_commit_template_language, try_split_commit_template_dfas,
};

const MAX_WORK: usize = 32_000_000;
const MAX_STATES: usize = 131_072;
const MAX_EDGES: usize = 1_048_576;

fn spend(work: &mut usize, amount: usize) -> Result<(), String> {
    *work = work.checked_sub(amount).ok_or("template link program exceeds its work budget")?;
    Ok(())
}

fn closure(nfa: &NFA, seeds: impl IntoIterator<Item = u32>, work: &mut usize) -> Result<Vec<u32>, String> {
    let mut seen = BTreeSet::new(); let mut todo = seeds.into_iter().collect::<Vec<_>>();
    while let Some(id) = todo.pop() {
        spend(work, 1)?;
        if seen.insert(id) {
            let state = nfa.states.get(id as usize).ok_or("template link epsilon leaves its graph")?;
            todo.extend(state.epsilons.iter().copied());
        }
    }
    Ok(seen.into_iter().collect())
}

/// Reify the relation, not necessarily the original signed action-word
/// language. Consecutive READ edges inspect the same concrete top. Collapse
/// their tests to one pop-and-restore before entering the output language;
/// otherwise the one-READ compiler normal form would reject valid providers.
/// Zero-READ acceptance and links remain separate, including on empty stacks.
pub(crate) fn action_nfa(template: &CommitTemplateDfas) -> Result<NFA, String> {
    let mut nfa = recombine_split_commit_template_language(template);
    if template.read.states.iter().all(|row| row.transitions.values().all(|&target|
        template.read.states.get(target as usize).is_some_and(|next| next.transitions.is_empty()))) {
        return Ok(nfa);
    }
    let read_offset = template.pop.states.len();
    let push_offset = read_offset.checked_add(template.read.states.len())
        .ok_or("template READ coordinate overflow")?;
    let fixed = push_offset.checked_add(template.push.states.len())
        .filter(|&n| n <= MAX_STATES).ok_or("template READ normalization exceeds state budget")?;
    // Discard only the old READ pop/restore intermediates. The fixed POP,
    // READ-entry and PUSH nodes retain their identities and incoming links.
    nfa.states.truncate(fixed);
    let mut work = MAX_WORK;
    let mut edges = 0usize;
    for (id, row) in template.read.states.iter().enumerate() {
        let from = read_offset + id;
        nfa.states[from].transitions.clear();
        nfa.states[from].epsilons.clear();
        if let Some(target) = template.read_to_push.get(id).copied().flatten() {
            if target as usize >= template.push.states.len() { return Err("invalid READ-to-PUSH link".into()); }
            nfa.add_epsilon(from as u32, push_offset as u32 + target);
        }
        for (&label, &first) in &row.transitions {
            if label < 0 || label == DEFAULT_LABEL { return Err("READ requires a concrete stack symbol".into()); }
            let mut at = first; let mut seen = BTreeSet::new();
            let mut outputs = BTreeSet::new(); let mut accepts = false;
            loop {
                spend(&mut work, 1)?;
                if !seen.insert(at) { return Err("cyclic READ relation".into()); }
                let current = template.read.states.get(at as usize).ok_or("READ edge leaves its graph")?;
                accepts |= current.is_accepting;
                if let Some(target) = template.read_to_push.get(at as usize).copied().flatten() {
                    if target as usize >= template.push.states.len() { return Err("invalid READ-to-PUSH link".into()); }
                    outputs.insert(push_offset as u32 + target);
                }
                let Some(&next) = current.transitions.get(&label) else { break; };
                at = next;
            }
            if !accepts && outputs.is_empty() { continue; }
            edges = edges.saturating_add(2 + outputs.len());
            if nfa.states.len() > MAX_STATES - 2 || edges > MAX_EDGES {
                return Err("template READ normalization exceeds graph budget".into());
            }
            let restore = nfa.add_state(); let output = nfa.add_state();
            nfa.states[output as usize].is_accepting = accepts;
            nfa.states[output as usize].epsilons.extend(outputs);
            nfa.add_transition(from as u32, label, restore);
            nfa.add_transition(restore, encode_negative_label(label as u32), output);
        }
    }
    Ok(nfa)
}

/// Inject concrete stack labels and restrict the *initial* top to its owner.
/// Later DEFAULT pops remain unrestricted, exactly like the existing provider.
/// A zero-input branch gains a read-and-restore of the initial owner top; a
/// consuming branch uses that same first pop directly. No stack is enumerated.
pub(crate) fn scoped(template: &CommitTemplateDfas, offset: u32, count: u32) -> Result<NFA, String> {
    let end = offset.checked_add(count).filter(|&end| end < DEFAULT_LABEL as u32)
        .ok_or("template link stack alphabet overflow")?;
    let mut nfa = action_nfa(template)?;
    if nfa.states.len() > MAX_STATES { return Err("template scoping exceeds state budget".into()); }
    let mut work = MAX_WORK;
    for state in &mut nfa.states {
        let mut transitions = BTreeMap::new();
        for (&label, targets) in &state.transitions {
            let label = if label == DEFAULT_LABEL { label } else {
                let local = if label < 0 { negative_to_positive_label(label) as u32 } else { label as u32 };
                if local >= count { return Err("template link label lies outside component alphabet".into()); }
                let global = offset + local;
                if label < 0 { encode_negative_label(global) } else { global as i32 }
            };
            transitions.insert(label, targets.clone());
        }
        state.transitions = transitions;
    }
    let initial = closure(&nfa, nfa.start_states.clone(), &mut work)?;
    let untouched = nfa.add_state();
    for &id in &initial {
        if nfa.states[id as usize].is_accepting { nfa.states[untouched as usize].is_accepting = true; }
        let pushes = nfa.states[id as usize].transitions.iter()
            .filter(|(label, _)| **label < 0).map(|(&label, targets)| (label, targets.clone())).collect::<Vec<_>>();
        for (label, targets) in pushes { nfa.states[untouched as usize].transitions.entry(label).or_default().extend(targets); }
    }
    let has_untouched = nfa.states[untouched as usize].is_accepting || !nfa.states[untouched as usize].transitions.is_empty();
    let start = nfa.add_state(); nfa.start_states = vec![start];
    for top in offset..end {
        let mut targets = BTreeSet::new();
        for &id in &initial {
            spend(&mut work, 1)?;
            let row = &nfa.states[id as usize].transitions;
            if let Some(next) = row.get(&(top as i32)).or_else(|| row.get(&DEFAULT_LABEL)) { targets.extend(next.iter().copied()); }
        }
        if targets.is_empty() && !has_untouched { continue; }
        if nfa.states.len() >= MAX_STATES { return Err("template owner guard exceeds state budget".into()); }
        let branch = nfa.add_state();
        nfa.states[branch as usize].epsilons.extend(targets);
        if has_untouched { nfa.add_transition(branch, encode_negative_label(top), untouched); }
        nfa.add_transition(start, top as i32, branch);
    }
    Ok(nfa)
}

pub(crate) fn append_push(nfa: &mut NFA, symbol: u32) {
    let end = nfa.add_state(); nfa.set_accepting(end);
    for id in 0..end {
        if nfa.states[id as usize].is_accepting {
            nfa.states[id as usize].is_accepting = false;
            nfa.add_transition(id, encode_negative_label(symbol), end);
        }
    }
}

pub(crate) fn nullable_return(symbol: u32) -> NFA {
    let mut nfa = NFA::new(); let end = nfa.add_state(); nfa.set_accepting(end);
    nfa.add_transition(0, symbol as i32, end); nfa
}

fn union(programs: &[NFA]) -> Result<NFA, String> {
    let mut result = NFA::new_empty(); let mut edges = 0usize;
    for program in programs {
        let base = u32::try_from(result.states.len()).map_err(|_| "template union state overflow")?;
        if result.states.len().saturating_add(program.states.len()) > MAX_STATES { return Err("template union exceeds state budget".into()); }
        for state in &program.states {
            edges = edges.saturating_add(state.epsilons.len() + state.transitions.values().map(Vec::len).sum::<usize>());
            if edges > MAX_EDGES { return Err("template union exceeds edge budget".into()); }
            let mut state = state.clone();
            for target in state.transitions.values_mut().flatten().chain(state.epsilons.iter_mut()) { *target += base; }
            result.states.push(state);
        }
        result.start_states.extend(program.start_states.iter().map(|&id| base + id));
    }
    Ok(result)
}

#[cfg(test)]
mod oracle_tests {
    use super::*;
    use crate::compiler::glr::{accumulator::TerminalsDisallowed, parser::ParserGSS};
    use crate::runtime::commit::template_advance::advance_with_prepared_template;

    fn program(pops: &[i32], reads: &[i32], pushes: &[u32]) -> CommitTemplateDfas {
        let mut pop = DFA::new(); let mut read = DFA::new(); let mut push = DFA::new();
        let mut at = pop.start_state;
        for &label in pops { let next = pop.add_state(); pop.add_transition(at, label, next); at = next; }
        let mut pop_to_read = vec![None; pop.states.len()]; pop_to_read[at as usize] = Some(0);
        at = read.start_state;
        for &label in reads { let next = read.add_state(); read.add_transition(at, label, next); at = next; }
        let mut read_to_push = vec![None; read.states.len()]; read_to_push[at as usize] = Some(0);
        at = push.start_state;
        for &symbol in pushes { let next = push.add_state(); push.add_transition(at, encode_negative_label(symbol), next); at = next; }
        push.set_accepting(at, true);
        CommitTemplateDfas { pop, read, push, pop_to_read, pop_to_push: vec![], read_to_push }
    }

    // Independent reference: retain the three original phase graphs and use
    // their ordinary executor. Only rename concrete labels; do not recombine,
    // union, determinize, or insert the new compiler's ownership guard.
    fn rename(program: &CommitTemplateDfas, offset: u32) -> CommitTemplateDfas {
        let mut result = program.clone();
        for graph in [&mut result.pop, &mut result.read] {
            for state in &mut graph.states {
                state.transitions = state.transitions.iter().map(|(&label, &target)|
                    (if label == DEFAULT_LABEL { label } else { label + offset as i32 }, target)).collect();
            }
        }
        for state in &mut result.push.states {
            state.transitions = state.transitions.iter().map(|(&label, &target)|
                (encode_negative_label(negative_to_positive_label(label) as u32 + offset), target)).collect();
        }
        result
    }

    fn stacks() -> Vec<Vec<u32>> {
        let mut all = vec![Vec::new()]; let mut layer = all.clone();
        for _ in 0..4 {
            let mut next = Vec::new();
            for prefix in layer { for top in 0..6 { let mut word = prefix.clone(); word.push(top); next.push(word); } }
            all.extend(next.iter().cloned()); layer = next;
        }
        all
    }

    fn evaluate(program: &CommitTemplateDfas, stack: &[u32]) -> BTreeSet<Vec<u32>> {
        let input = ParserGSS::from_single_stack(stack.to_vec(), TerminalsDisallowed::new());
        advance_with_prepared_template(program, input, None).to_stacks(256).unwrap()
            .into_iter().map(|(stack, _)| stack).collect()
    }

    fn cases() -> Vec<CommitTemplateDfas> {
        let mut choices = vec![program(&[], &[], &[]), program(&[], &[], &[1, 2]),
            program(&[], &[1, 1], &[2]), program(&[DEFAULT_LABEL, DEFAULT_LABEL], &[1], &[2]),
            program(&[1], &[], &[])];
        let mut dead = program(&[DEFAULT_LABEL], &[], &[2]);
        let target = dead.pop.add_state(); dead.pop.add_transition(0, 0, target); choices.push(dead);
        let mut early = program(&[DEFAULT_LABEL], &[], &[1]); early.pop.set_accepting(0, true); choices.push(early);
        let mut deep = program(&[1, DEFAULT_LABEL], &[], &[]);
        let target = deep.pop.add_state(); deep.pop.add_transition(1, 0, target); choices.push(deep);
        choices
    }

    #[test]
    fn scoped_programs_match_phase_executor_over_all_small_stacks() {
        for (case, original) in cases().into_iter().enumerate() {
            let reference = rename(&original, 2);
            let linked = compile(&[scoped(&original, 2, 3).unwrap()]).unwrap();
            let domain = super::super::compile_domain(&linked).unwrap();
            for stack in stacks() {
                let expected = if stack.last().is_some_and(|top| (2..5).contains(top)) {
                    evaluate(&reference, &stack)
                } else { BTreeSet::new() };
                assert_eq!(evaluate(&linked, &stack), expected, "case={case}, stack={stack:?}");
                assert_eq!(domain.matches_top_first(stack.iter().rev().copied()), !expected.is_empty());
            }
        }
    }

    #[test]
    fn default_union_preserves_each_source_rows_explicit_dead_shadow() {
        let mut a = program(&[DEFAULT_LABEL], &[], &[1]);
        let dead = a.pop.add_state(); a.pop.add_transition(0, 0, dead);
        let b = program(&[0], &[], &[2]);
        let linked = compile(&[recombine_split_commit_template_language(&a),
            recombine_split_commit_template_language(&b)]).unwrap();
        for stack in stacks() {
            let mut expected = evaluate(&a, &stack); expected.extend(evaluate(&b, &stack));
            assert_eq!(evaluate(&linked, &stack), expected, "stack={stack:?}");
        }
    }

    #[test]
    fn appended_call_start_follows_every_output_including_epsilon_branches() {
        for original in cases() {
            let mut nfa = action_nfa(&original).unwrap();
            append_push(&mut nfa, 5);
            let linked = compile(&[nfa]).unwrap();
            for stack in stacks() {
                let expected = evaluate(&original, &stack).into_iter().map(|mut word| { word.push(5); word }).collect();
                assert_eq!(evaluate(&linked, &stack), expected, "stack={stack:?}");
            }
        }
    }

    #[test]
    fn malformed_coordinates_and_cycles_fail_before_a_program_is_published() {
        assert!(scoped(&program(&[2], &[], &[]), 0, 2).is_err());
        assert!(scoped(&program(&[], &[], &[]), u32::MAX, 2).is_err());
        let mut cyclic = NFA::new(); cyclic.set_accepting(0); cyclic.add_transition(0, 0, 0);
        assert!(compile(&[cyclic]).is_err());
    }
}

pub(crate) fn compile(programs: &[NFA]) -> Result<CommitTemplateDfas, String> {
    let nfa = union(programs)?;
    let mut work = MAX_WORK;
    let initial = closure(&nfa, nfa.start_states.clone(), &mut work)?;
    let mut dfa = DFA::new();
    let mut known = BTreeMap::from([(initial.clone(), 0u32)]);
    let mut queue = VecDeque::from([(0u32, initial)]);
    let mut edges = 0usize;
    while let Some((id, subset)) = queue.pop_front() {
        dfa.states[id as usize].is_accepting = subset.iter().any(|&s| nfa.states[s as usize].is_accepting);
        let labels = subset.iter().flat_map(|&s| nfa.states[s as usize].transitions.keys().copied()).collect::<BTreeSet<_>>();
        for label in labels {
            let mut targets = Vec::new();
            for &source in &subset {
                spend(&mut work, 1)?;
                let row = &nfa.states[source as usize].transitions;
                let next = if label >= 0 && label != DEFAULT_LABEL { row.get(&label).or_else(|| row.get(&DEFAULT_LABEL)) }
                    else { row.get(&label) };
                if let Some(next) = next { targets.extend(next.iter().copied()); }
            }
            let target_set = closure(&nfa, targets, &mut work)?;
            // Keep explicit dead transitions: dropping them would expose the
            // source state's DEFAULT fallback and enlarge the relation.
            let target = if let Some(&target) = known.get(&target_set) { target } else {
                if dfa.states.len() >= MAX_STATES { return Err("template determinization exceeds state budget".into()); }
                let target = dfa.add_state(); known.insert(target_set.clone(), target);
                queue.push_back((target, target_set)); target
            };
            edges += 1; if edges > MAX_EDGES { return Err("template determinization exceeds edge budget".into()); }
            dfa.add_transition(id, label, target);
        }
    }
    if !dfa.compute_is_acyclic() { return Err("template link produced a cyclic stack relation".into()); }
    let result = try_split_commit_template_dfas(&dfa).ok_or("linked relation is not a finite POP/READ/PUSH program")?;
    super::compile_domain(&result).map_err(|error| error.to_string())?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::{accumulator::TerminalsDisallowed, parser::ParserGSS};
    use crate::runtime::commit::template_advance::advance_with_prepared_template;

    fn rewrite(pop_labels: &[i32], pushes: &[u32]) -> CommitTemplateDfas {
        let mut pop = DFA::new(); let mut cursor = 0;
        for &label in pop_labels { let target = pop.add_state(); pop.add_transition(cursor, label, target); cursor = target; }
        let mut pop_to_push = vec![None; pop.states.len()]; pop_to_push[cursor as usize] = Some(0);
        let mut push = DFA::new(); let mut cursor = 0;
        for &symbol in pushes { let target = push.add_state(); push.add_transition(cursor, encode_negative_label(symbol), target); cursor = target; }
        push.set_accepting(cursor, true);
        CommitTemplateDfas { pop, read: DFA::new(), push, pop_to_read: vec![], pop_to_push, read_to_push: vec![] }
    }

    fn outputs(program: &CommitTemplateDfas, source: &[u32]) -> BTreeSet<Vec<u32>> {
        let stack = ParserGSS::from_single_stack(source.to_vec(), TerminalsDisallowed::new());
        advance_with_prepared_template(program, stack, None).to_stacks(128)
            .expect("small exact test frontier").into_iter().map(|(stack, _)| stack).collect()
    }

    #[test]
    fn union_preserves_per_branch_explicit_dead_default_shadowing() {
        let mut a = rewrite(&[DEFAULT_LABEL], &[2]);
        let dead = a.pop.add_state(); a.pop.add_transition(0, 0, dead);
        let b = rewrite(&[0], &[1]);
        let joined = compile(&[recombine_split_commit_template_language(&a), recombine_split_commit_template_language(&b)]).unwrap();
        assert_eq!(outputs(&joined, &[0]), BTreeSet::from([vec![1]]));
        assert_eq!(outputs(&joined, &[1]), BTreeSet::from([vec![2]]));
        assert_eq!(outputs(&joined, &[9, 0]), BTreeSet::from([vec![9, 1]]));
        let mut called = recombine_split_commit_template_language(&joined); append_push(&mut called, 8);
        let called = compile(&[called]).unwrap();
        assert_eq!(outputs(&called, &[0]), BTreeSet::from([vec![1, 8]]));
        assert_eq!(outputs(&called, &[1]), BTreeSet::from([vec![2, 8]]));
    }

    #[test]
    fn input_free_outputs_gain_only_the_initial_owner_guard() {
        let source = rewrite(&[], &[1]);
        let result = compile(&[scoped(&source, 10, 3).unwrap()]).unwrap();
        assert_eq!(outputs(&result, &[99, 10]), BTreeSet::from([vec![99, 10, 11]]));
        assert!(outputs(&result, &[99, 9]).is_empty());
        assert!(outputs(&result, &[]).is_empty());
    }

    #[test]
    fn deeper_default_pops_keep_the_existing_provider_semantics() {
        let source = rewrite(&[DEFAULT_LABEL, DEFAULT_LABEL], &[]);
        let result = compile(&[scoped(&source, 10, 3).unwrap()]).unwrap();
        assert_eq!(outputs(&result, &[99, 11]), BTreeSet::from([vec![]]));
        assert_eq!(outputs(&result, &[88, 99, 11]), BTreeSet::from([vec![88]]));
        assert!(outputs(&result, &[99, 9]).is_empty());
        assert!(outputs(&result, &[11]).is_empty());
    }

    #[test]
    fn identity_and_consuming_branches_survive_the_same_scoped_input() {
        let mut source = rewrite(&[DEFAULT_LABEL], &[]);
        source.pop.set_accepting(0, true);
        let result = compile(&[scoped(&source, 10, 3).unwrap()]).unwrap();
        assert_eq!(outputs(&result, &[99, 11]), BTreeSet::from([vec![99], vec![99, 11]]));
        assert!(outputs(&result, &[99, 9]).is_empty());
    }
}

#[cfg(test)]
#[path = "link_program/literal_oracle.rs"]
mod literal_oracle;
