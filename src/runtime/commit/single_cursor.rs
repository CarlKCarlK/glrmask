//! Exact deterministic input-prefix cursor for acyclic stack programs.
//!
//! This is only a parser-advance primitive. A uniform shared GSS can expose a
//! common Segment prefix through VirtualStack. Both POP and READ are
//! deterministic for a known input top, even when phase links yield multiple
//! output alternatives. Follow that input prefix once and emit only residual
//! PUSH programs to the existing shared suffix/DAG evaluator. At an unseen
//! branching floor or the bounded work limit, retain the exact remaining
//! POP/READ program; neither output branches nor parser state are truncated.
use crate::automata::unweighted_u32::dfa::DFA;
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::labels::{DEFAULT_LABEL, is_negative_label};
use crate::compiler::glr::parser::ParserGSS;
use crate::ds::leveled_gss::VirtualStack;
use crate::runtime::CommitTemplateDfas;
use smallvec::SmallVec;

const MAX_CURSOR_STEPS: usize = 64;

#[derive(Clone, Debug)]
pub(crate) struct PreparedInputCursor {
    productive: [Box<[bool]>; 3],
}

pub(super) struct InputCursorResult {
    pub(super) output: ParserGSS,
    pub(super) residual: SmallVec<[(usize, u32, ParserGSS); 8]>,
}

fn order(graph: &DFA) -> Option<Vec<usize>> {
    if graph.states.is_empty() {
        return (graph.start_state == 0).then(Vec::new);
    }
    graph.states.get(graph.start_state as usize)?;
    let mut incoming = vec![0usize; graph.states.len()];
    for row in &graph.states {
        for &target in row.transitions.values() { *incoming.get_mut(target as usize)? += 1; }
    }
    let mut order = incoming.iter().enumerate().filter_map(|(i,&n)|(n==0).then_some(i)).collect::<Vec<_>>();
    let mut index=0;
    while index < order.len() {
        for &target in graph.states[order[index]].transitions.values() {
            incoming[target as usize]-=1;
            if incoming[target as usize]==0 { order.push(target as usize); }
        }
        index+=1;
    }
    (order.len()==graph.states.len()).then_some(order)
}

fn forward_links(template: &CommitTemplateDfas, phase: usize, id: usize) -> [Option<(usize, u32)>; 2] {
    let read = (phase == 0).then(|| template.pop_to_read.get(id).copied().flatten())
        .flatten().map(|target| (1, target));
    let push = match phase {
        0 => template.pop_to_push.get(id), 1 => template.read_to_push.get(id), _ => None,
    }.copied().flatten().map(|target| (2, target));
    [read, push]
}

impl PreparedInputCursor {
    pub(crate) fn prepare(template: &CommitTemplateDfas) -> Option<Self> {
        let graphs=[&template.pop,&template.read,&template.push];
        let orders=[order(graphs[0])?,order(graphs[1])?,order(graphs[2])?];
        for (phase,graph) in graphs.iter().enumerate() {
            if graph.states.iter().any(|row|row.transitions.keys().any(|&label|match phase {
                0=>is_negative_label(label),1=>is_negative_label(label)||label==DEFAULT_LABEL,
                _=>!is_negative_label(label),
            })) {return None;}
        }
        for (links,source,target) in [(&template.pop_to_read,0,1),(&template.pop_to_push,0,2),(&template.read_to_push,1,2)] {
            if links.len()>graphs[source].states.len() || links.iter().flatten().any(|&i|i as usize>=graphs[target].states.len()) { return None; }
        }
        Some(Self::from_orders(template, &orders))
    }

    pub(crate) fn from_prepared(prepared: &super::template_prepare::TemplatePreparation<'_>) -> Self {
        Self::from_orders(prepared.template(), prepared.orders())
    }

    fn from_orders(template: &CommitTemplateDfas, orders: &[Vec<usize>; 3]) -> Self {
        let graphs = [&template.pop, &template.read, &template.push];
        let mut productive:[Box<[bool]>;3]=std::array::from_fn(|phase|vec![false;graphs[phase].states.len()].into_boxed_slice());
        for phase in (0..3).rev() {
            for &id in orders[phase].iter().rev() {
                let state=&graphs[phase].states[id];
                // This deliberately overapproximates feasibility: READ paths
                // may test incompatible labels. Only global unproductivity is
                // used to discard work; uncertain branches force fallback.
                let mut live=state.is_accepting || state.transitions.values().any(|&target|productive[phase][target as usize]);
                for (target_phase,target) in forward_links(template,phase,id).into_iter().flatten() {
                    live |= productive[target_phase][target as usize];
                }
                productive[phase][id]=live;
            }
        }
        Self { productive }
    }

    pub(super) fn apply(&self, template: &CommitTemplateDfas, input: &ParserGSS) -> Option<InputCursorResult> {
        if !self.productive[0].get(template.pop.start_state as usize).copied()? {
            return Some(InputCursorResult { output: ParserGSS::empty(), residual: SmallVec::new() });
        }
        let mut stack: VirtualStack<u32, TerminalsDisallowed> = input.try_virtual_stack()?;
        let mut id = template.pop.start_state;
        let mut remaining = MAX_CURSOR_STEPS;
        let mut popped = false;
        let mut output = ParserGSS::empty();
        let mut residual = SmallVec::new();
        loop {
            let mut materialized = None;
            let mut base = || materialized.get_or_insert_with(|| {
                if popped { stack.clone().into_gss() } else { input.clone() }
            }).clone();
            // The program is deterministic only over the visible Segment
            // prefix. Do not inspect or enumerate the branching floor here.
            if remaining == 0 || (stack.top().is_none() && stack.has_hidden_floor_values()) {
                residual.push((0, id, base()));
                break;
            }
            remaining -= 1;
            let state = &template.pop.states[id as usize];
            let mut kept = false;
            if state.is_accepting {
                let unchanged = base();
                output = if output.is_empty() { unchanged } else { output.merge(&unchanged) };
                kept = true;
            }
            if let Some(target) = template.pop_to_push.get(id as usize).copied().flatten() {
                if self.productive[2][target as usize] { residual.push((2, target, base())); }
            }
            if let Some(mut read_id) = template.pop_to_read.get(id as usize).copied().flatten() {
                if self.productive[1][read_id as usize] {
                    loop {
                        if remaining == 0 {
                            residual.push((1, read_id, base()));
                            break;
                        }
                        remaining -= 1;
                        let read = &template.read.states[read_id as usize];
                        if read.is_accepting && !kept {
                            let unchanged = base();
                            output = if output.is_empty() { unchanged } else { output.merge(&unchanged) };
                            kept = true;
                        }
                        if let Some(target) = template.read_to_push.get(read_id as usize).copied().flatten() {
                            if self.productive[2][target as usize] { residual.push((2, target, base())); }
                        }
                        // READ edges inspect the SAME top at every state.
                        let Some(&top) = stack.top() else { break; };
                        let Some(&next) = read.transitions.get(&(top as i32)) else { break; };
                        if !self.productive[1][next as usize] { break; }
                        read_id = next;
                    }
                }
            }
            // Select explicit edges before testing productivity; an explicit
            // rejecting edge still shadows a productive DEFAULT alternative.
            let Some(&top) = stack.top() else { break; };
            let Some(&next) = state.transitions.get(&(top as i32)).or_else(|| state.transitions.get(&DEFAULT_LABEL)) else { break; };
            if !self.productive[0][next as usize] { break; }
            drop(base);
            // A known visible top guarantees one available POP. No later
            // failure is allowed to discard the output seeds already emitted.
            let unpopped = stack.pop(1);
            debug_assert_eq!(unpopped, 0);
            popped = true;
            id = next;
        }
        Some(InputCursorResult { output, residual })
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::labels::encode_negative_label;
    use std::collections::BTreeSet;

    fn literal(template:&CommitTemplateDfas, phase:usize,id:u32, input:&ParserGSS)->ParserGSS {
        let mut all=Vec::new();
        for (stack,acc) in input.to_stacks(8192).unwrap() {
            let mut pending=vec![(phase,id,stack)];let mut visited=BTreeSet::new();
            while let Some((phase,id,stack))=pending.pop() {
                if !visited.insert((phase,id,stack.clone())) {continue;}
                let graph=[&template.pop,&template.read,&template.push][phase];
                let state=&graph.states[id as usize];
                if state.is_accepting {all.push((stack.clone(),acc.clone()));}
                if phase==2 {
                    for (&label,&target) in &state.transitions {
                        let mut next=stack.clone();next.push(crate::compiler::glr::labels::negative_to_positive_label(label)as u32);
                        pending.push((2,target,next));
                    }
                } else {
                    if let Some(&top)=stack.last() {
                        if let Some(&target)=state.transitions.get(&(top as i32)).or_else(||if phase==0{state.transitions.get(&DEFAULT_LABEL)}else{None}) {
                            let mut next=stack.clone();if phase==0{next.pop();}pending.push((phase,target,next));
                        }
                    }
                    if phase==0 {
                        if let Some(Some(target))=template.pop_to_read.get(id as usize){pending.push((1,*target,stack.clone()));}
                        if let Some(Some(target))=template.pop_to_push.get(id as usize){pending.push((2,*target,stack));}
                    } else if let Some(Some(target))=template.read_to_push.get(id as usize){pending.push((2,*target,stack));}
                }
            }
        }
        ParserGSS::from_stacks(&all)
    }

    fn check(template:&CommitTemplateDfas,input:&ParserGSS)->bool {
        let plan=PreparedInputCursor::prepare(template).unwrap();
        let original=input.to_stacks(8192).unwrap();
        let Some(result)=plan.apply(template,input) else {assert_eq!(original,input.to_stacks(8192).unwrap());return false;};
        let mut actual = result.output;
        for (phase, state, input) in result.residual {
            actual = actual.merge(&literal(template, phase, state, &input));
        }
        let expected=literal(template,0,template.pop.start_state,input);
        assert_eq!(actual.semantically_eq(&expected,65_536),Some(true));
        assert_eq!(original,input.to_stacks(8192).unwrap(),"the input must survive transactional speculation");
        true
    }

    #[test]
    fn input_cursor_preserves_epsilon_read_acceptance_and_default_dead_edges() {
        let mut pop=DFA::new();pop.add_state();pop.add_transition(0,0,1);
        let mut read=DFA::new();read.set_accepting(0,true);
        let mut t=CommitTemplateDfas{pop,read,push:DFA::new(),pop_to_read:vec![None,Some(0)],pop_to_push:vec![],read_to_push:vec![]};
        let input=ParserGSS::from_single_stack(vec![0],TerminalsDisallowed::new());
        assert!(check(&t,&input),"READ epsilon must accept an emptied concrete stack");
        t.read.set_accepting(0,false);t.push.add_state();t.push.add_transition(0,encode_negative_label(1),1);t.push.set_accepting(1,true);t.read_to_push=vec![Some(0)];
        assert!(check(&t,&input),"READ to PUSH epsilon must work without any READ edge or top");
        let dead=t.pop.add_state();t.pop.states[0].transitions.clear();t.pop.add_transition(0,0,dead);t.pop.add_transition(0,DEFAULT_LABEL,1);
        t.pop_to_read=vec![Some(0)];
        let other=t.push.add_state();let end=t.push.add_state();
        t.push.add_transition(other,encode_negative_label(2),end);t.push.set_accepting(end,true);
        t.pop_to_push=vec![None,Some(other)];
        assert!(check(&t,&input),"dead explicit POP must not activate DEFAULT");
        let mut malformed=t.clone();malformed.pop.add_transition(dead,DEFAULT_LABEL,dead);assert!(PreparedInputCursor::prepare(&malformed).is_none());
        let mut malformed=t.clone();malformed.pop_to_read=vec![Some(99)];assert!(PreparedInputCursor::prepare(&malformed).is_none());
    }

    #[test]
    fn input_cursor_retains_exact_residuals_at_work_budget_and_output_branches() {
        let mut pop = DFA::new(); let mut read = DFA::new(); let mut push = DFA::new();
        for _ in 0..150 { pop.add_state(); read.add_state(); }
        let end = push.add_state(); push.add_transition(0, encode_negative_label(2), end); push.set_accepting(end, true);
        for i in 0..150 {
            pop.add_transition(i, DEFAULT_LABEL, i+1);
            pop.set_accepting(i, i%7 == 0);
            read.add_transition(i, 1, i+1);
            read.set_accepting(i, i%9 == 0);
        }
        pop.set_accepting(150, true); read.set_accepting(150, true);
        let mut template=CommitTemplateDfas { pop, read, push,
            pop_to_read:vec![Some(0);151],pop_to_push:vec![Some(0);151],read_to_push:vec![Some(0);151] };
        let input=ParserGSS::from_single_stack(vec![1;151],TerminalsDisallowed::new());
        assert!(check(&template,&input), "a long READ chain must retain both its residual and the POP continuation");
        template.pop_to_read.clear();
        assert!(check(&template,&input), "a long POP chain must retain all accepted prefixes and residuals");
        let floor=ParserGSS::from_stacks(&[(vec![0,2],TerminalsDisallowed::new()),(vec![0,3],TerminalsDisallowed::new())]);
        assert!(check(&template,&floor.push(1).push(1)), "an unknown shared floor resumes the exact program");
    }

    #[test]
    fn input_cursor_matches_generated_literal_relations_and_shared_floors() {
        let mut seed=0x608894u64;
        let mut next=||{seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;seed};
        let mut accepted=0;let mut declined=0;
        for _case in 0..512 {
            let mut graphs=[DFA::new(),DFA::new(),DFA::new()];
            for (phase,graph) in graphs.iter_mut().enumerate() {
                for _ in 1..5 {graph.add_state();}
                for id in 0..5u32 {
                    graph.set_accepting(id,next()%4==0);if id==4{continue;}
                    for symbol in 0..4u32 {if next()%3==0 {
                        let label=if phase==2{encode_negative_label(symbol)}else{symbol as i32};
                        graph.add_transition(id,label,id+1+(next()%(4-id)as u64)as u32);
                    }}
                    if phase==0&&next()%2==0{graph.add_transition(id,DEFAULT_LABEL,id+1);}
                }
            }
            let [pop,read,push]=graphs;
            let mut links=||(0..5).map(|_|if next()%3==0{Some((next()%5)as u32)}else{None}).collect::<Vec<_>>();
            let template=CommitTemplateDfas{pop,read,push,pop_to_read:links(),pop_to_push:links(),read_to_push:links()};
            for acc in [TerminalsDisallowed::new(),TerminalsDisallowed::new().with_insert(1,2)] {
                let floor=ParserGSS::from_stacks(&[(vec![0,1],acc.clone()),(vec![0,2],acc.clone())]);
                let inputs=[ParserGSS::empty(),ParserGSS::from_single_stack(vec![],acc.clone()),
                    ParserGSS::from_single_stack(vec![0,1,2],acc.clone()),floor.push(3),floor.push(3).push(1)];
                for input in inputs {if check(&template,&input){accepted+=1;}else{declined+=1;}}
            }
        }
        assert!(accepted>100&&declined>100,"test must exercise both exact specialization and conservative decline");
    }
}
