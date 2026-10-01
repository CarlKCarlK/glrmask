//! Bounded-worklist escape to an exact shared input/output phase dataflow.
//!
//! The ordinary template interpreter is cheap for sparse frontiers. But a POP
//! chain over an ambiguous shared GSS can revisit the same automaton state for
//! exponentially many *paths*, even after PUSH output sharing is preserved.
//! When its pending frontier grows, merge all inputs to a phase state before
//! evaluating that state. POP, READ and PUSH distribute over language union.
//! A topological schedule is valid because every phase is acyclic and epsilon
//! links only go POP -> READ/PUSH or READ -> PUSH.
//!
//! This is the same parser-advance primitive over the authoritative GSS, not a
//! second token/lexer/mask engine. The caller supplies only uniform annotations;
//! correlated annotations retain the established ordered interpreter.
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rustc_hash::FxHashMap;

use crate::automata::unweighted_u32::dfa::DFA;
use crate::compiler::glr::labels::{DEFAULT_LABEL, is_negative_label, negative_to_positive_label};
use crate::compiler::glr::parser::ParserGSS;
use crate::runtime::CommitTemplateDfas;
use super::template_advance::Phase;

#[derive(Clone, Debug)]
pub(crate) struct PreparedPhaseDag {
    ranks: [Box<[u32]>; 3],
}

fn ranks(graph: &DFA) -> Option<Box<[u32]>> {
    let n = graph.states.len();
    graph.states.get(graph.start_state as usize)?;
    let mut incoming = vec![0usize; n];
    for state in &graph.states {
        for &target in state.transitions.values() {
            *incoming.get_mut(target as usize)? += 1;
        }
    }
    let mut order: Vec<_> = incoming.iter().enumerate()
        .filter_map(|(i, &degree)| (degree == 0).then_some(i)).collect();
    let mut head = 0;
    while head < order.len() {
        let id = order[head]; head += 1;
        for &target in graph.states[id].transitions.values() {
            incoming[target as usize] -= 1;
            if incoming[target as usize] == 0 { order.push(target as usize); }
        }
    }
    if order.len() != n { return None; }
    ranks_from_order(&order)
}

fn ranks_from_order(order: &[usize]) -> Option<Box<[u32]>> {
    let mut rank = vec![0u32; order.len()];
    for (i, &id) in order.iter().enumerate() { rank[id] = u32::try_from(i).ok()?; }
    Some(rank.into_boxed_slice())
}

impl PreparedPhaseDag {
    pub(crate) fn prepare(template: &CommitTemplateDfas) -> Option<Self> {
        for state in &template.pop.states {
            if state.transitions.keys().any(|&label| is_negative_label(label)) { return None; }
        }
        for state in &template.read.states {
            if state.transitions.keys().any(|&label| label == DEFAULT_LABEL || is_negative_label(label)) {
                return None;
            }
        }
        for state in &template.push.states {
            if state.transitions.keys().any(|&label| !is_negative_label(label)) { return None; }
        }
        for target in template.pop_to_read.iter().flatten() { template.read.states.get(*target as usize)?; }
        for target in template.pop_to_push.iter().chain(&template.read_to_push).flatten() {
            template.push.states.get(*target as usize)?;
        }
        Some(Self { ranks: [ranks(&template.pop)?, ranks(&template.read)?, ranks(&template.push)?] })
    }

    pub(crate) fn from_prepared(prepared: &super::template_prepare::TemplatePreparation<'_>) -> Option<Self> {
        let orders = prepared.orders();
        // Preserve this index's existing empty-phase eligibility rule.
        if orders.iter().any(Vec::is_empty) { return None; }
        Some(Self { ranks: [ranks_from_order(&orders[0])?,
            ranks_from_order(&orders[1])?, ranks_from_order(&orders[2])?] })
    }

    fn phase_number(phase: Phase) -> u8 {
        match phase { Phase::Pop => 0, Phase::Read => 1, Phase::Push => 2 }
    }

    /// Evaluate the exact residual work of a partially completed uniform
    /// interpreter. Previously emitted output is retained. No initial input
    /// clone or new work is required on the ordinary small-frontier path.
    pub(super) fn apply(
        &self,
        template: &CommitTemplateDfas,
        work: impl IntoIterator<Item = (Phase, u32, ParserGSS)>,
        mut output: ParserGSS,
    ) -> ParserGSS {
        let mut incoming = FxHashMap::<(u8, u32), ParserGSS>::default();
        let mut queue = BinaryHeap::<Reverse<(u8, u32, u32)>>::new();
        let enqueue = |incoming: &mut FxHashMap<(u8, u32), ParserGSS>,
                       queue: &mut BinaryHeap<Reverse<(u8, u32, u32)>>,
                       phase: u8, id: u32, stacks: ParserGSS| {
            if stacks.is_empty() { return; }
            match incoming.entry((phase, id)) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let merged = entry.get().merge(&stacks); entry.insert(merged);
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(stacks);
                    queue.push(Reverse((phase, self.ranks[phase as usize][id as usize], id)));
                }
            }
        };
        for (phase, id, stacks) in work {
            enqueue(&mut incoming, &mut queue, Self::phase_number(phase), id, stacks);
        }
        while let Some(Reverse((phase, _, id))) = queue.pop() {
            let stacks = incoming.remove(&(phase, id))
                .expect("a scheduled phase state owns all its incoming languages");
            let graph = match phase { 0 => &template.pop, 1 => &template.read, _ => &template.push };
            let state = &graph.states[id as usize];
            if state.is_accepting { output = output.merge(&stacks); }
            if phase == 2 {
                for (&label, &target) in &state.transitions {
                    enqueue(&mut incoming, &mut queue, phase, target,
                        stacks.push(negative_to_positive_label(label) as u32));
                }
            } else {
                for top in stacks.peek_values() {
                    // Every explicit POP edge, including a dead blocker,
                    // takes priority over DEFAULT. READ never uses DEFAULT.
                    let target = state.transitions.get(&(top as i32)).or_else(||
                        if phase == 0 { state.transitions.get(&DEFAULT_LABEL) } else { None });
                    if let Some(&target) = target {
                        let advanced = if phase == 0 { stacks.pop_top_value(&top) }
                            else { stacks.isolate(Some(top)) };
                        enqueue(&mut incoming, &mut queue, phase, target, advanced);
                    }
                }
            }
            // Epsilon phase links do not require a current stack symbol. In
            // particular a READ -> PUSH link can act on an empty concrete stack.
            if phase == 0 {
                if let Some(Some(target)) = template.pop_to_read.get(id as usize) {
                    enqueue(&mut incoming, &mut queue, 1, *target, stacks.clone());
                }
                if let Some(Some(target)) = template.pop_to_push.get(id as usize) {
                    enqueue(&mut incoming, &mut queue, 2, *target, stacks);
                }
            } else if phase == 1 {
                if let Some(Some(target)) = template.read_to_push.get(id as usize) {
                    enqueue(&mut incoming, &mut queue, 2, *target, stacks);
                }
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::{accumulator::TerminalsDisallowed, labels::encode_negative_label};

    fn literal(template: &CommitTemplateDfas,
               work: &[(Phase,u32,ParserGSS)], emitted:&ParserGSS) -> ParserGSS {
        let mut complete=emitted.to_stacks(4096).unwrap();
        for (phase,entry,input) in work {
            for (stack,acc) in input.to_stacks(4096).unwrap() {
                let mut pending=vec![(*phase,*entry,stack)];let mut budget=1_000_000;
                while let Some((phase,id,stack))=pending.pop() {
                    budget-=1;assert!(budget>0,"finite oracle fixture exceeds its independent bound");
                    let graph=match phase {Phase::Pop=>&template.pop,Phase::Read=>&template.read,Phase::Push=>&template.push};
                    let state=&graph.states[id as usize];
                    if state.is_accepting {complete.push((stack.clone(),acc.clone()));}
                    match phase {
                        Phase::Pop => {
                            if let Some(top)=stack.last() {
                                if let Some(&target)=state.transitions.get(&(*top as i32)).or_else(||state.transitions.get(&DEFAULT_LABEL)) {
                                    let mut next=stack.clone();next.pop();pending.push((Phase::Pop,target,next));
                                }
                            }
                            if let Some(Some(target))=template.pop_to_read.get(id as usize){pending.push((Phase::Read,*target,stack.clone()));}
                            if let Some(Some(target))=template.pop_to_push.get(id as usize){pending.push((Phase::Push,*target,stack));}
                        }
                        Phase::Read => {
                            if let Some(top)=stack.last(){if let Some(&target)=state.transitions.get(&(*top as i32)){pending.push((Phase::Read,target,stack.clone()));}}
                            if let Some(Some(target))=template.read_to_push.get(id as usize){pending.push((Phase::Push,*target,stack));}
                        }
                        Phase::Push => {
                            for (&label,&target) in &state.transitions {
                                let mut next=stack.clone();next.push(negative_to_positive_label(label)as u32);pending.push((phase,target,next));
                            }
                        }
                    }
                }
            }
        }
        ParserGSS::from_stacks(&complete)
    }

    #[test]
    fn residual_phase_dataflow_matches_literal_relations_and_uniform_annotations() {
        let mut seed=0x608894u64;
        let mut next=||{seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;seed};
        for case in 0..256 {
            let mut graphs=[DFA::new(),DFA::new(),DFA::new()];
            for (phase,graph) in graphs.iter_mut().enumerate() {
                for _ in 1..6 {graph.add_state();}
                for i in 0..6u32 {
                    graph.set_accepting(i,next()%4==0);
                    if i==5 {continue;}
                    for symbol in 0..4u32 {if next()%3==0 {
                        let label=if phase==2 {encode_negative_label(symbol)}else{symbol as i32};
                        graph.add_transition(i,label,i+1+(next()%(5-i)as u64)as u32);
                    }}
                    if phase==0 && next()%2==0 {graph.add_transition(i,DEFAULT_LABEL,i+1+(next()%(5-i)as u64)as u32);}
                }
            }
            let [pop,read,push]=graphs;
            let mut links=|| (0..6).map(|_|if next()%3==0{Some((next()%6)as u32)}else{None}).collect::<Vec<_>>();
            let t=CommitTemplateDfas{pop,read,push,pop_to_read:links(),pop_to_push:links(),read_to_push:links()};
            let plan=PreparedPhaseDag::prepare(&t).unwrap();
            for acc in [TerminalsDisallowed::new(),TerminalsDisallowed::new().with_insert(0,3).with_insert(2,1)] {
                let inputs=[ParserGSS::empty(),ParserGSS::from_single_stack(Vec::new(),acc.clone()),
                    ParserGSS::from_single_stack(vec![0,2,1],acc.clone()),
                    ParserGSS::from_stacks(&[(vec![],acc.clone()),(vec![0,1],acc.clone()),(vec![0,2,3],acc.clone()),(vec![0,3,1],acc.clone())])];
                for input in inputs {
                    // Start at arbitrary phase cursors simultaneously, as a
                    // real overflowing path worklist does. Include duplicate
                    // arrivals and an already emitted accepted language.
                    let work=vec![(Phase::Push,2,input.clone()),(Phase::Pop,0,input.clone()),
                        (Phase::Read,1,input.clone()),(Phase::Pop,2,input.clone()),(Phase::Read,1,input.clone())];
                    let emitted=ParserGSS::from_single_stack(vec![0,2],acc.clone());
                    let expected=literal(&t,&work,&emitted);
                    let actual=plan.apply(&t,work,emitted);
                    assert_eq!(actual.semantically_eq(&expected,65_536),Some(true),"case={case}");
                    for (_,annotation) in actual.to_stacks(65_536).unwrap(){assert_eq!(annotation,acc,"case={case}");}
                }
            }
        }
    }

    #[test]
    fn phase_index_declines_cycles_invalid_targets_and_phase_labels() {
        let t=CommitTemplateDfas{pop:DFA::new(),read:DFA::new(),push:DFA::new(),
            pop_to_read:vec![Some(0)],pop_to_push:vec![],read_to_push:vec![]};
        assert!(PreparedPhaseDag::prepare(&t).is_some());
        let mut cycle=t.clone();cycle.pop.add_transition(0,DEFAULT_LABEL,0);
        assert!(PreparedPhaseDag::prepare(&cycle).is_none());
        let mut bad=t.clone();bad.read.states[0].transitions.insert(0,10);
        assert!(PreparedPhaseDag::prepare(&bad).is_none());
        let mut bad=t.clone();bad.push.states[0].transitions.insert(0,0);
        assert!(PreparedPhaseDag::prepare(&bad).is_none());
        let mut bad=t.clone();bad.pop_to_read=vec![Some(10)];
        assert!(PreparedPhaseDag::prepare(&bad).is_none());
    }
}
