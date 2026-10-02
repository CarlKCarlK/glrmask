//! Compile a data-only parser into the ordinary static masking artifact.
//!
//! The lexer stage is deliberately conservative about parser properties: no
//! terminal colouring, no rule-derived follow exclusions and no inferred LR
//! admission facts. The supplied stack relations then make the parser product
//! exact. Possible-match and terminal equivalence are reconciled independently.

use super::*;
use crate::automata::unweighted_u32::nfa::NFA;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::constraint_possible_matches as pm;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::stages::equiv_types::MappedArtifact;
use crate::compiler::stages::id_map_and_terminal_dwa as tdwa;
use crate::compiler::stages::parser_dwa::{
    build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count_no_table,
    normalize_weighted_stack_predicate_for_symbol_count,
};
use crate::compiler::stages::templates::Templates;
use crate::ds::bitset::BitSet;
use crate::runtime::ConstraintRuntimeBackend;
use glrmask_parser_dwa::__private::resolve_negatives::resolve_negative_codes_in_nwa;
use glrmask_parser_dwa::__private::pop_classes::PopLabelClasses;
use crate::automata::weighted::nwa::{NWA, NWAState};
use crate::compiler::glr::labels::negative_to_positive_label;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// Bound representation growth, not the accepted language. Exceeding any
/// bound is an explicit build error, never a truncation or runtime fallback.
#[derive(Clone, Default)]
struct ExpansionBudget {
    states: usize,
    edges: usize,
    subset_members: usize,
    work: usize,
}

impl ExpansionBudget {
    fn charge(&mut self, states: usize, edges: usize, members: usize, work: usize) -> Result<()> {
        self.states = self.states.saturating_add(states);
        self.edges = self.edges.saturating_add(edges);
        self.subset_members = self.subset_members.saturating_add(members);
        self.work = self.work.saturating_add(work);
        if self.states > 131_072
            || self.edges > 1_048_576
            || self.subset_members > 2_097_152
            || self.work > 33_554_432
        {
            return Err(Error::Compilation(format!(
                "static template expansion exceeded its representation/work budget; the parser relation was not truncated (states={}, edges={}, subset_members={}, work={})",
                self.states, self.edges, self.subset_members, self.work,
            )));
        }
        Ok(())
    }
}

/// Convert a phase graph to an ordinary action-word NFA. DEFAULT must be
/// resolved *before* any epsilon/subset union: merging default branches first
/// can let another branch's concrete edge shadow the wrong alternative.
fn concrete_action_nfa(
    split: &CommitTemplateDfas,
    symbols: u32,
    budget: &mut ExpansionBudget,
) -> Result<NFA> {
    action_nfa(split, symbols, budget, None, None)
}

fn action_nfa(
    split: &CommitTemplateDfas,
    symbols: u32,
    budget: &mut ExpansionBudget,
    mut classes: Option<&mut PopLabelClasses>,
    scope: Option<&crate::runtime::parser_backend::scoped_program::ScopedProgram>,
) -> Result<NFA> {
    let global = |local: u32| scope.map_or(local, |view| view.offset + local);
    let domain = scope.map_or(0..symbols, |view| view.offset..view.offset + view.symbols);
    let read_offset = split.pop.states.len();
    let push_offset = read_offset + split.read.states.len();
    let fixed = push_offset + split.push.states.len();
    budget.charge(fixed, 0, 0, fixed)?;
    let mut nfa = NFA::new_empty();
    nfa.states.resize_with(fixed, Default::default);
    nfa.start_states.push(split.pop.start_state);
    for (id, state) in split.pop.states.iter().enumerate() {
        nfa.states[id].is_accepting = state.is_accepting;
        for (&label, &target) in &state.transitions {
            if label == DEFAULT_LABEL {
                // Even a concrete edge into a dead state shadows DEFAULT.
                if let Some(classes) = classes.as_deref_mut() {
                    budget.charge(0, 0, 0, state.transitions.len())?;
                    if let Some(label) = classes.intern_scoped_complement(domain.clone(), state.transitions.keys()
                        .copied().filter(|&label| label != DEFAULT_LABEL).map(|label| global(label as u32)))
                        .map_err(Error::Compilation)?
                    {
                        budget.charge(0, 1, 0, 0)?;
                        // The class already records a global component range;
                        // its symbolic code is not a local stack symbol.
                        nfa.add_transition(id as u32, label, target);
                    }
                } else {
                    for local in 0..domain.end - domain.start {
                        let symbol = global(local);
                        budget.charge(0, 0, 0, 1)?;
                        if !state.transitions.contains_key(&(local as i32)) {
                            budget.charge(0, 1, 0, 0)?;
                            nfa.add_transition(id as u32, symbol as i32, target);
                        }
                    }
                }
            } else {
                budget.charge(0, 1, 0, 1)?;
                nfa.add_transition(id as u32, global(label as u32) as i32, target);
            }
        }
        for (links, offset) in [
            (&split.pop_to_read, read_offset),
            (&split.pop_to_push, push_offset),
        ] {
            if let Some(target) = links.get(id).copied().flatten() {
                budget.charge(0, 1, 0, 1)?;
                nfa.add_epsilon(id as u32, offset as u32 + target);
            }
        }
    }
    for (id, state) in split.read.states.iter().enumerate() {
        let from = (read_offset + id) as u32;
        nfa.states[from as usize].is_accepting = state.is_accepting;
        for (&label, &target) in &state.transitions {
            budget.charge(1, 2, 0, 1)?;
            let restore = nfa.add_state();
            nfa.add_transition(from, global(label as u32) as i32, restore);
            nfa.add_transition(
                restore,
                encode_negative_label(global(label as u32)),
                read_offset as u32 + target,
            );
        }
        if let Some(target) = split.read_to_push.get(id).copied().flatten() {
            budget.charge(0, 1, 0, 1)?;
            nfa.add_epsilon(from, push_offset as u32 + target);
        }
    }
    for (id, state) in split.push.states.iter().enumerate() {
        let from = (push_offset + id) as u32;
        nfa.states[from as usize].is_accepting = state.is_accepting;
        for (&label, &target) in &state.transitions {
            budget.charge(0, 1, 0, 1)?;
            nfa.add_transition(from, encode_negative_label(global(negative_to_positive_label(label) as u32)), push_offset as u32 + target);
        }
    }
    if let Some(view) = scope {
        if let Some(symbol) = view.append_push {
            budget.charge(1, 0, 0, 1)?;
            let end = nfa.add_state(); nfa.set_accepting(end);
            for id in 0..end {
                if nfa.states[id as usize].is_accepting {
                    budget.charge(0, 1, 0, 1)?;
                    nfa.states[id as usize].is_accepting = false;
                    nfa.add_transition(id, encode_negative_label(symbol), end);
                }
            }
        }
        // Consuming paths are already guarded by their local READ/POP labels.
        // An input-free success additionally requires a concrete owner top.
        if view.guard_owner && view.domain.start() == glrmask_parser_dwa::__private::templates::admissibility::DomainProbe::Accept {
            budget.charge(view.symbols as usize + 1, view.symbols as usize * 2, 0, view.symbols as usize)?;
            let starts = std::mem::take(&mut nfa.start_states);
            let entry = nfa.add_state(); nfa.start_states.push(entry);
            for top in view.offset..view.offset + view.symbols {
                let restore = nfa.add_state();
                nfa.add_transition(entry, top as i32, restore);
                for &start in &starts { nfa.add_transition(restore, encode_negative_label(top), start); }
            }
        }
    }
    Ok(nfa)
}

/// Exact subset construction with explicit accounting before growing its
/// buckets, graph and retained subset keys. The ordinary unweighted compiler
/// assumes trusted finite inputs; the public data-only entry point must also
/// handle a small NFA whose deterministic representation is exponential.
fn bounded_determinize(nfa: &NFA, budget: &mut ExpansionBudget) -> Result<DFA> {
    fn closure(nfa: &NFA, seeds: &[u32], budget: &mut ExpansionBudget) -> Result<Vec<u32>> {
        let mut seen = BTreeSet::new();
        let mut pending = Vec::new();
        for &q in seeds {
            budget.charge(0, 0, 0, 1)?;
            if seen.insert(q) {
                pending.push(q);
            }
        }
        while let Some(q) = pending.pop() {
            for &to in &nfa.states[q as usize].epsilons {
                budget.charge(0, 0, 0, 1)?;
                if seen.insert(to) {
                    pending.push(to);
                }
            }
        }
        Ok(seen.into_iter().collect())
    }
    let start = closure(nfa, &nfa.start_states, budget)?;
    budget.charge(1, 0, start.len(), 0)?;
    let mut dfa = DFA::new();
    let mut known = FxHashMap::from_iter([(start.clone(), 0u32)]);
    let mut pending = VecDeque::from([(0u32, start)]);
    while let Some((id, subset)) = pending.pop_front() {
        let mut targets = BTreeMap::<i32, Vec<u32>>::new();
        for q in subset {
            dfa.states[id as usize].is_accepting |= nfa.states[q as usize].is_accepting;
            for (&label, destinations) in &nfa.states[q as usize].transitions {
                budget.charge(0, 0, 0, destinations.len())?;
                targets
                    .entry(label)
                    .or_default()
                    .extend_from_slice(destinations);
            }
        }
        for (label, seeds) in targets {
            let key = closure(nfa, &seeds, budget)?;
            let target = if let Some(&target) = known.get(&key) {
                target
            } else {
                budget.charge(1, 0, key.len(), 0)?;
                let target = dfa.add_state();
                known.insert(key.clone(), target);
                pending.push_back((target, key));
                target
            };
            budget.charge(0, 1, 0, 1)?;
            dfa.add_transition(id, label, target);
        }
    }
    Ok(dfa)
}

/// Compatibility context for the existing lexical compiler. It contains no
/// grammar rules, nonterminals, productions or parser automaton. Empty follow
/// certificates and global observation prohibit grammar-specific shortcuts;
/// all real parser semantics come exclusively from the supplied templates.
pub(crate) fn lexical_context(terminals: u32) -> AnalyzedGrammar {
    let mut protected = BitSet::new(terminals as usize);
    for terminal in 0..terminals {
        protected.set(terminal as usize);
    }
    AnalyzedGrammar {
        rules: Vec::new(),
        num_terminals: terminals,
        terminal_display_names: (0..terminals).map(|t| format!("terminal_{t}")).collect(),
        protected_shift_terminals: protected,
        num_nonterminals: 0,
        nonterminal_display_names: Vec::new(),
        residual_isolation_classes: BTreeMap::new(),
        requires_global_terminal_observation: true,
        direct_regular_automaton: None,
        nullable: BTreeSet::new(),
        first: Vec::new(),
        follow: Vec::new(),
        rules_by_lhs: Vec::new(),
    }
}

/// Shared exact action-word preparation. Resolve DEFAULT priority inside
/// each source row before any epsilon/subset union. No LR facts are required.
pub(crate) fn prepare_static_templates(
    programs: &[Option<Arc<CommitTemplateDfas>>], symbol_count: u32,
) -> Result<Templates> {
    prepare_static_templates_for_terminals(programs, symbol_count,
        &(0..programs.len() as u32).collect())
}

/// Compile exactly the inventory observed by a lexical query, with IDs kept
/// in the caller's coordinate. Unobserved terminals cannot contribute to this
/// product; expanding their DEFAULT rows wastes the shared resource budget.
/// Every selected relation remains complete, including all of its stack paths.
pub(crate) fn prepare_static_templates_for_terminals(
    programs: &[Option<Arc<CommitTemplateDfas>>], symbol_count: u32,
    selected: &BTreeSet<u32>,
) -> Result<Templates> {
    let mut budget = ExpansionBudget::default();
    let mut terminal_templates = BTreeMap::new();
    for &terminal in selected {
        let split = programs.get(terminal as usize).and_then(Option::as_deref)
            .ok_or_else(|| Error::Compilation(format!("missing selected terminal relation {terminal}")))?;
        let result = (|| {
            let nfa = concrete_action_nfa(split, symbol_count, &mut budget)?;
            bounded_determinize(&nfa, &mut budget)
        })().map_err(|error| Error::Compilation(format!("terminal {terminal}: {error}")))?;
        terminal_templates.insert(terminal, result);
    }
    Ok(Templates::from_terminal_dfas(terminal_templates))
}

fn boundary_action_program(nfa: NFA, budget: &mut ExpansionBudget) -> NWA {
    // Reuse the ordinary action-word constructor before instantiation. Class
    // labels are distinct consuming symbols here; replacing them with their
    // finite local languages later preserves this exact subset construction.
    // A refused representation keeps the complete original phase program.
    let mut candidate_budget = budget.clone();
    if let Ok(dfa) = bounded_determinize(&nfa, &mut candidate_budget) {
        // Ordinary compiler templates use this same full action-language
        // minimizer after determinization. Scope complements have already
        // absorbed every explicit dead shadow, so they remain exact when
        // the ordinary constructor removes a dead literal branch.
        let dfa = crate::automata::unweighted_u32::minimize_acyclic::minimize_acyclic(&dfa);
        *budget = candidate_budget;
        let states = dfa.states.into_iter().map(|state| NWAState {
            final_weight: state.is_accepting.then(crate::ds::weight::Weight::all),
            transitions: state.transitions.into_iter().map(|(label, target)|
                (label, vec![(target, crate::ds::weight::Weight::all())])).collect(),
            epsilons: Vec::new(),
        }).collect();
        return NWA::from_parts(states, vec![dfa.start_state]);
    }
    let states = nfa.states.into_iter().map(|state| NWAState {
        final_weight: state.is_accepting.then(crate::ds::weight::Weight::all),
        transitions: state.transitions.into_iter().map(|(label, targets)|
            (label, targets.into_iter().map(|target|
                (target, crate::ds::weight::Weight::all())).collect())).collect(),
        epsilons: state.epsilons.into_iter().map(|target|
            (target, crate::ds::weight::Weight::all())).collect(),
    }).collect();
    NWA::from_parts(states, nfa.start_states)
}

/// Bounded ordinary action-word programs for a boundary query. POP complements stay
/// symbolic until the common cancellation solver knows which concrete PUSH
/// symbols reach them. Neither a template DFA nor the global stack alphabet is
/// eagerly expanded merely to build a construction-only intermediate graph.
pub(crate) fn prepare_classed_boundary_programs(
    programs: &[Option<Arc<CommitTemplateDfas>>], symbol_count: u32,
    selected: &BTreeSet<u32>,
) -> Result<(BTreeMap<u32, NWA>, PopLabelClasses)> {
    if std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some() {
        let mut states = 0usize;
        let mut edges = 0usize;
        let mut largest = (0usize, 0u32);
        for &id in selected {
            if let Some(program) = programs.get(id as usize).and_then(Option::as_deref) {
                let n = program.pop.states.len() + program.read.states.len() + program.push.states.len();
                states += n;
                edges += [&program.pop, &program.read, &program.push].iter()
                    .flat_map(|graph| &graph.states).map(|row| row.transitions.len()).sum::<usize>();
                largest = largest.max((n, id));
            }
        }
        eprintln!("[glrmask/profile][classed_template_inventory] templates={} source_states={states} source_edges={edges} largest_states={} largest_terminal={}",
            selected.len(), largest.0, largest.1);
    }
    let mut budget = ExpansionBudget::default();
    let mut classes = PopLabelClasses::new(symbol_count).map_err(Error::Compilation)?;
    let mut result = BTreeMap::new();
    for &terminal in selected {
        let split = programs.get(terminal as usize).and_then(Option::as_deref)
            .ok_or_else(|| Error::Compilation(format!("missing selected terminal relation {terminal}")))?;
        let nfa = action_nfa(split, symbol_count, &mut budget, Some(&mut classes), None)
            .map_err(|error| Error::Compilation(format!("terminal {terminal}: {error}")))?;
        result.insert(terminal, boundary_action_program(nfa, &mut budget));
    }
    Ok((result, classes))
}

pub(crate) fn prepare_scoped_boundary_programs(
    programs: &[crate::runtime::parser_backend::scoped_program::ScopedProgram], symbol_count: u32,
    selected: &BTreeSet<u32>,
) -> Result<(BTreeMap<u32, NWA>, PopLabelClasses)> {
    let mut budget = ExpansionBudget::default();
    let mut classes = PopLabelClasses::new(symbol_count).map_err(Error::Compilation)?;
    let mut result = BTreeMap::new();
    for &terminal in selected {
        let view = programs.get(terminal as usize).ok_or_else(||
            Error::Compilation(format!("missing scoped relation {terminal}")))?;
        view.validate_coordinate(symbol_count)?;
        let nfa = action_nfa(&view.source, symbol_count, &mut budget, Some(&mut classes), Some(view))?;
        result.insert(terminal, boundary_action_program(nfa, &mut budget));
    }
    Ok((result, classes))
}

pub(crate) fn prepare_scoped_boundary_admissions(
    programs:&[crate::runtime::parser_backend::scoped_program::ScopedProgram],
    selected:&BTreeSet<u32>,classes:&mut PopLabelClasses,
)->Result<BTreeMap<u32,NWA>> {
    selected.iter().map(|&terminal| {
        let view=programs.get(terminal as usize).ok_or_else(||
            Error::Compilation(format!("missing scoped admission relation {terminal}")))?;
        view.validate_coordinate(classes.symbol_count())?;
        let graph=view.domain.scoped_prefix_program(view.offset,view.symbols,view.guard_owner,classes)
            .map_err(Error::Compilation)?;
        Ok((terminal,graph))
    }).collect()
}

#[cfg(test)]
mod selected_inventory_tests {
    use super::*;

    fn pop(label: i32) -> Arc<CommitTemplateDfas> {
        let mut dfa = DFA::new();
        let end = dfa.add_state();
        dfa.set_accepting(end, true);
        dfa.add_transition(dfa.start_state, label, end);
        Arc::new(CommitTemplateDfas { pop: dfa, read: DFA::new(), push: DFA::new(),
            pop_to_read: vec![], pop_to_push: vec![], read_to_push: vec![] })
    }

    #[test]
    fn selected_inventory_preserves_original_ids_and_exact_programs() {
        let programs = vec![Some(pop(0)), Some(pop(DEFAULT_LABEL)), Some(pop(2))];
        let full = prepare_static_templates(&programs, 4).unwrap();
        let selected = prepare_static_templates_for_terminals(&programs, 4, &BTreeSet::from([1, 2])).unwrap();
        assert_eq!(selected.by_terminal.keys().copied().collect::<Vec<_>>(), vec![1, 2]);
        for id in [1, 2] {
            assert_eq!(bincode::serialize(&selected.by_terminal[&id]).unwrap(),
                bincode::serialize(&full.by_terminal[&id]).unwrap());
        }
    }

    #[test]
    fn unused_inventory_is_not_expanded_but_missing_selected_program_is_an_error() {
        let programs = vec![None, None, Some(pop(1))];
        let selected = prepare_static_templates_for_terminals(&programs, 4, &BTreeSet::from([2])).unwrap();
        assert_eq!(selected.by_terminal.len(), 1);
        for id in [0, 3] {
            let error = prepare_static_templates_for_terminals(&programs, 4, &BTreeSet::from([id])).unwrap_err();
            assert!(error.to_string().contains("missing selected terminal relation"));
        }
    }

    #[test]
    fn scoped_admission_projection_matches_full_transfer_for_all_stack_words() {
        use crate::runtime::parser_backend::scoped_program::ScopedProgram;
        use crate::ds::weight::Weight;
        let mut pop=DFA::new();for _ in 0..3 {pop.add_state();}
        pop.set_accepting(2,true);pop.add_transition(0,DEFAULT_LABEL,1);pop.add_transition(0,3,3);
        pop.add_transition(1,DEFAULT_LABEL,2);pop.add_transition(1,2,3);
        let mut read=DFA::new();let end=read.add_state();read.set_accepting(end,true);read.add_transition(0,1,end);
        let mut push=DFA::new();let end=push.add_state();push.set_accepting(end,true);
        push.add_transition(0,encode_negative_label(0),end);
        let source=Arc::new(CommitTemplateDfas{pop,read,push,pop_to_read:vec![None,None,Some(0),None],
            pop_to_push:vec![],read_to_push:vec![None,Some(0)]});
        for source in [source,self::pop(0),{
            let mut p=DFA::new();p.set_accepting(0,true);
            Arc::new(CommitTemplateDfas{pop:p,read:DFA::new(),push:DFA::new(),pop_to_read:vec![],pop_to_push:vec![],read_to_push:vec![]})
        }] {
            let mut view=ScopedProgram::prepare(source,4).unwrap().relocated(2).unwrap();
            view.append_push=Some(7);
            let mut classes=PopLabelClasses::new(8).unwrap();
            let raw=action_nfa(&view.source,8,&mut ExpansionBudget::default(),Some(&mut classes),Some(&view)).unwrap();
            let mut full=boundary_action_program(raw,&mut ExpansionBudget{work:33_554_432,..Default::default()});
            glrmask_parser_dwa::__private::resolve_negatives::resolve_negative_codes_in_nwa_with_pop_classes(&mut full,&classes).unwrap();
            let reference=classes.compile_positive(full,10000).unwrap();
            let checker=view.domain.scoped_prefix_program(view.offset,view.symbols,view.guard_owner,&mut classes).unwrap();
            assert!(!checker.states().iter().any(|row|row.transitions.keys().any(|&label|label<0)));
            let candidate=classes.compile_positive(checker,10000).unwrap();
            let comparison=glrmask_parser_dwa::__private::parser_equivalence::compare_parser_mask_prefix_languages(
                &reference,&candidate,8,10000).unwrap();
            assert!(comparison.difference.is_none(),"{:?}",comparison.difference);
            assert!(candidate.states()[candidate.start_state() as usize].final_weight.as_ref().is_none_or(Weight::is_empty));
        }
    }

    #[test]
    fn scoped_action_word_constructor_matches_raw_phases_before_and_after_cancellation() {
        use crate::runtime::parser_backend::scoped_program::ScopedProgram;
        use crate::ds::weight::Weight;
        let mut pop = DFA::new();
        for _ in 0..3 { pop.add_state(); }
        pop.set_accepting(2, true);
        pop.add_transition(0, DEFAULT_LABEL, 1);
        pop.add_transition(0, 3, 3); // A dead literal shadows the first POP.
        pop.add_transition(1, DEFAULT_LABEL, 2);
        pop.add_transition(1, 2, 3); // The second POP has its own shadow.
        let mut read = DFA::new();
        let end = read.add_state(); read.set_accepting(end, true);
        read.add_transition(0, 1, end);
        let mut push = DFA::new();
        let end = push.add_state(); push.set_accepting(end, true);
        push.add_transition(0, encode_negative_label(0), end);
        let source = Arc::new(CommitTemplateDfas { pop, read, push,
            pop_to_read: vec![None, None, Some(0), None],
            pop_to_push: vec![], read_to_push: vec![None, Some(0)] });
        let mut view = ScopedProgram::prepare(source, 4).unwrap().relocated(2).unwrap();
        view.append_push = Some(7); // The concrete CALL frame stays typed.
        let mut classes = PopLabelClasses::new(8).unwrap();
        let nfa = action_nfa(&view.source, 8, &mut ExpansionBudget::default(),
            Some(&mut classes), Some(&view)).unwrap();
        let normalized = boundary_action_program(nfa.clone(), &mut ExpansionBudget::default());
        // Refuse the subset budget deliberately: the fallback must retain
        // every original phase state, epsilon, final and concrete/class key.
        let mut refused = ExpansionBudget { work: 33_554_432, ..Default::default() };
        let raw = boundary_action_program(nfa.clone(), &mut refused);
        assert_eq!(raw.states().len(), nfa.states.len());
        assert_eq!(refused.work, 33_554_432);
        assert!(normalized.states().iter().all(|row| row.epsilons.is_empty()));
        assert!(normalized.states().len() < raw.states().len());
        // Independently interpret the original NFA against the deterministic
        // action graph. Compare finals at every product, including after an
        // earlier final, so this proves the full signed action-word language.
        let closure = |seeds: Vec<u32>| {
            let mut seen = BTreeSet::new(); let mut todo = seeds;
            while let Some(q) = todo.pop() {
                if seen.insert(q) { todo.extend(nfa.states[q as usize].epsilons.iter().copied()); }
            }
            seen.into_iter().collect::<Vec<_>>()
        };
        let initial = (closure(nfa.start_states.clone()), Some(normalized.start_states()[0]));
        let mut seen = BTreeSet::from([initial.clone()]);
        let mut todo = VecDeque::from([initial]);
        while let Some((subset, state)) = todo.pop_front() {
            assert_eq!(subset.iter().any(|&q| nfa.states[q as usize].is_accepting),
                state.is_some_and(|q| normalized.states()[q as usize].final_weight.is_some()));
            let labels = subset.iter().flat_map(|&q| nfa.states[q as usize].transitions.keys().copied())
                .chain(state.into_iter().flat_map(|q| normalized.states()[q as usize].transitions.keys().copied()))
                .collect::<BTreeSet<_>>();
            for label in labels {
                let seeds = subset.iter().flat_map(|&q| nfa.states[q as usize].transitions.get(&label)
                    .into_iter().flatten().copied()).collect();
                let next = (closure(seeds), state.and_then(|q| normalized.states()[q as usize]
                    .transitions.get(&label).map(|edges| edges[0].0)));
                if seen.insert(next.clone()) { todo.push_back(next); }
            }
            assert!(seen.len() < 10000);
        }
        // Put a concrete PUSH before the scoped program so cancellation also
        // exercises local class membership and each literal-dead exception.
        for top in 0..8 {
            let compile = |mut graph: NWA| {
                let entry = graph.add_state();
                let starts = graph.start_states().to_vec();
                graph.set_start_states(vec![entry]);
                for start in starts {
                    graph.add_transition(entry, encode_negative_label(top), start, Weight::all());
                }
                glrmask_parser_dwa::__private::resolve_negatives::resolve_negative_codes_in_nwa_with_pop_classes(
                    &mut graph, &classes).unwrap();
                classes.compile_positive(graph, 10000).unwrap()
            };
            let reference = compile(raw.clone()); let candidate = compile(normalized.clone());
            let comparison = glrmask_parser_dwa::__private::parser_equivalence::compare_parser_mask_prefix_languages(
                &reference, &candidate, 8, 10000).unwrap();
            assert!(comparison.difference.is_none(), "PUSH {top}: {:?}", comparison.difference);
        }
    }
}

pub(super) fn compile(
    program: &ParserProgram,
    tokenizer: crate::automata::lexer::tokenizer::Tokenizer,
    ignore_terminal: Option<u32>,
    vocab: &Vocab,
    specials: &[crate::runtime::SpecialTokenTerminal],
) -> Result<crate::runtime::Constraint> {
    let context = lexical_context(program.parser.terminal_count);
    let templates = prepare_static_templates(&program.templates, program.parser.state_count)?;
    let (terminal, _, _) = tdwa::build_id_map_and_terminal_dwa(
        &tokenizer,
        vocab,
        &tdwa::types::TerminalColoring::identity(context.num_terminals as usize),
        false,
        ignore_terminal,
        &context,
        &BTreeMap::new(),
        None,
    );
    let (terminal_dwa, terminal_ids) = terminal.into_parts();
    let parser_dwa = match build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count_no_table(
        &terminal_dwa, context.num_terminals, &templates, true,
    ) {
        Some(mut signed) => {
            resolve_negative_codes_in_nwa(&mut signed, false);
            normalize_weighted_stack_predicate_for_symbol_count(program.parser.state_count, &signed)
        }
        None => DWA::new(terminal_ids.num_tsids(), terminal_ids.max_internal_token_id()),
    };
    let possible = pm::compute_constraint_possible_matches_for_vocab(
        &tokenizer,
        vocab,
        pm::ConstraintPossibleMatchesConfig::EAGER,
    );
    let mut mapped = MappedArtifact::from((
        MappedArtifact::new(parser_dwa, terminal_ids),
        possible.mapped_possible_matches,
    ));
    mapped.compact_dimensions();
    let ((parser_dwa, possible_matches), ids) = mapped.into_parts();
    let mut inner =
        crate::dynamic_constraint::DynamicConstraint::from_template_runtime_parts_unfinalized(
            tokenizer,
            context.terminal_display_names,
            ignore_terminal,
            program.templates.iter().cloned().collect(),
            Arc::clone(&program.parser),
            vocab,
            possible.runtime_dynamic_vocab.vocab,
        );
    inner.runtime_backend = ConstraintRuntimeBackend::Static;
    inner.special_token_terminals = specials.to_vec();
    inner.parser_dwa = parser_dwa.share_exact_transition_rows_owned();
    inner.possible_matches = possible_matches;
    inner.possible_matches_complete = possible.complete;
    inner.state_to_internal_tsid = ids.tokenizer_states.original_to_internal.clone();
    inner.internal_tsid_to_states = ids.tokenizer_states.internal_to_originals_vecs();
    inner.state_internal_tsid_offsets = vec![u32::MAX];
    inner.original_token_to_internal = ids.vocab_tokens.original_to_internal.clone();
    inner.internal_token_to_tokens = ids.vocab_tokens.internal_to_originals_vecs();
    inner.internal_token_bytes =
        pm::build_internal_token_bytes_from_groups(vocab, &ids.vocab_tokens.internal_to_originals);
    inner.rebuild_runtime_caches();
    assert!(
        !inner.table.is_present(),
        "custom static compilation retained an LR table"
    );
    Ok(inner)
}
