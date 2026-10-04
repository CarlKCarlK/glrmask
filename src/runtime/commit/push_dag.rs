//! Graph-native application of large acyclic PUSH languages.
//!
//! A path enumerator loses sharing when several labels converge on the same
//! automaton state. Instead, for one entry stack language B, maintain
//! L[q] = union { B · w | entry --w--> q }. Process states in topological order
//! and propagate `push(label)` over the union once. Append distributes over
//! union, so all accepting L[q] together are exactly the original relation.
//! The existing GSS owns the result and shares all common lower prefixes.
//!
//! This index applies only to uniform annotations. The established evaluator
//! retains its ordered operations for correlated path annotations.
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use rustc_hash::FxHashMap;
use crate::compiler::glr::labels::{is_negative_label, negative_to_positive_label};
use crate::compiler::glr::parser::ParserGSS;
use crate::runtime::CommitTemplateDfas;

const SMALL_OUTPUT_PATHS: usize = 64;

#[derive(Clone, Debug)]
struct State {
    accepting: bool,
    /// Concrete pushed label and the target's topological rank. Edges sharing
    /// a target are adjacent, so sibling languages can use the existing exact
    /// bulk GSS constructor rather than repeatedly copying a growing union.
    edges: Box<[(u32, u32)]>,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedPushDag {
    states: Box<[State]>,
    /// Only genuine phase entries with large output languages select this
    /// evaluator. Small ordinary parser actions retain their cheaper path.
    entries: Box<[Option<u32>]>,
}

impl PreparedPushDag {
    pub(crate) fn prepare(template: &CommitTemplateDfas) -> Option<Self> {
        Self::compile(template, SMALL_OUTPUT_PATHS)
    }

    pub(crate) fn from_prepared(prepared: &super::template_prepare::TemplatePreparation<'_>) -> Option<Self> {
        Self::from_order(prepared.template(), SMALL_OUTPUT_PATHS, prepared.push_order())
    }

    fn compile(template: &CommitTemplateDfas, minimum_paths: usize) -> Option<Self> {
        let graph = &template.push;
        let n = graph.states.len();
        let mut indegree = vec![0usize; n];
        for state in &graph.states {
            for (&label, &target) in &state.transitions {
                if !is_negative_label(label) { return None; }
                *indegree.get_mut(target as usize)? += 1;
            }
        }
        let mut order: Vec<_> = indegree.iter().enumerate()
            .filter_map(|(i, &degree)| (degree == 0).then_some(i)).collect();
        let mut head = 0;
        while head < order.len() {
            let source = order[head]; head += 1;
            for &target in graph.states[source].transitions.values() {
                indegree[target as usize] -= 1;
                if indegree[target as usize] == 0 { order.push(target as usize); }
            }
        }
        if order.len() != n { return None; }
        Self::from_order(template, minimum_paths, &order)
    }

    fn from_order(template: &CommitTemplateDfas, minimum_paths: usize, order: &[usize]) -> Option<Self> {
        let graph = &template.push;
        let n = graph.states.len();
        let mut counts = vec![0usize; n];
        for &source in order.iter().rev() {
            counts[source] = usize::from(graph.states[source].is_accepting);
            for &target in graph.states[source].transitions.values() {
                counts[source] = counts[source].saturating_add(counts[target as usize])
                    .min(minimum_paths.saturating_add(1));
            }
        }
        let mut rank = vec![0u32; n];
        for (i, &source) in order.iter().enumerate() { rank[source] = u32::try_from(i).ok()?; }
        let mut entries = vec![None; n];
        for &target in template.pop_to_push.iter().chain(&template.read_to_push).flatten() {
            if *counts.get(target as usize)? > minimum_paths {
                entries[target as usize] = Some(rank[target as usize]);
            }
        }
        if entries.iter().all(Option::is_none) { return None; }
        let states = order.iter().copied().map(|source| {
            let mut edges: Vec<_> = graph.states[source].transitions.iter()
                .filter(|(_, target)| counts[**target as usize] != 0)
                .map(|(&label, &target)| (negative_to_positive_label(label) as u32, rank[target as usize]))
                .collect();
            edges.sort_unstable_by_key(|&(_, target)| target);
            State {
                accepting: graph.states[source].is_accepting,
                edges: edges.into_boxed_slice(),
            }
        }).collect();
        Some(Self { states, entries: entries.into_boxed_slice() })
    }

    pub(crate) fn apply(&self, entry: u32, base: &ParserGSS) -> Option<ParserGSS> {
        let entry = self.entries.get(entry as usize).copied().flatten()?;
        if base.is_empty() { return Some(ParserGSS::empty()); }
        base.single_interface_lower_id()?;
        // Allocate scratch for reachable cursors only. A late phase entry must
        // not allocate a slot for every state in a million-state provider.
        let mut frontier = FxHashMap::<u32, ParserGSS>::default();
        let mut pending = BinaryHeap::new();
        frontier.insert(entry, base.clone()); pending.push(Reverse(entry));
        let mut output = ParserGSS::empty();
        while let Some(Reverse(rank)) = pending.pop() {
            let input = frontier.remove(&rank).expect("each pending PUSH state owns its incoming language");
            let state = &self.states[rank as usize];
            if state.accepting { output = output.merge(&input); }
            let mut first = 0;
            while first < state.edges.len() {
                let target = state.edges[first].1;
                let mut end = first + 1;
                while end < state.edges.len() && state.edges[end].1 == target { end += 1; }
                let siblings = &state.edges[first..end];
                debug_assert!(target > rank, "validated PUSH graph must be acyclic");
                let pushed = if siblings.len() == 1 {
                    input.push(siblings[0].0)
                } else {
                    // This evaluator already requires one shared annotation.
                    // Append distributes over union. The bulk primitive retains
                    // the same prefix and all concrete labels; an unsupported
                    // intermediate GSS layout keeps the original exact fold.
                    input.apply_shared_pop_push_single_branches(
                        0, siblings.iter().map(|(label, _)| label),
                    ).unwrap_or_else(|| {
                        let mut language = ParserGSS::empty();
                        for &(label, _) in siblings { language = language.merge(&input.push(label)); }
                        language
                    })
                };
                match frontier.entry(target) {
                    std::collections::hash_map::Entry::Occupied(mut slot) => {
                        let merged = slot.get().merge(&pushed); slot.insert(merged);
                    }
                    std::collections::hash_map::Entry::Vacant(slot) => {
                        slot.insert(pushed); pending.push(Reverse(target));
                    }
                }
                first = end;
            }
        }
        Some(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::unweighted_u32::dfa::DFA;
    use crate::compiler::glr::{accumulator::TerminalsDisallowed, labels::encode_negative_label};

    fn template(push: DFA, entry: u32) -> CommitTemplateDfas {
        CommitTemplateDfas { pop: DFA::new(), read: DFA::new(), push,
            pop_to_read: Vec::new(), pop_to_push: vec![Some(entry)], read_to_push: Vec::new() }
    }
    fn literal(graph: &DFA, entry: u32, input: &ParserGSS) -> ParserGSS {
        let mut out = Vec::new();
        for (stack, acc) in input.to_stacks(100).unwrap() {
            let mut pending = vec![(entry, stack)];
            while let Some((i, stack)) = pending.pop() {
                let state = &graph.states[i as usize];
                if state.is_accepting { out.push((stack.clone(), acc.clone())); }
                for (&label, &target) in &state.transitions {
                    let mut next = stack.clone(); next.push(negative_to_positive_label(label) as u32);
                    pending.push((target, next));
                }
            }
        }
        ParserGSS::from_stacks(&out)
    }

    #[test]
    fn large_sibling_fanout_batches_each_target_without_losing_words() {
        for count in [65u32, 200, 400] {
            let mut graph = DFA::new();
            let left = graph.add_state(); let right = graph.add_state();
            graph.set_accepting(left, true); graph.set_accepting(right, true);
            for label in 0..count {
                graph.add_transition(0, encode_negative_label(label), if label % 2 == 0 { left } else { right });
            }
            let t = template(graph, 0);
            let plan = PreparedPushDag::prepare(&t).unwrap();
            let input = ParserGSS::from_single_stack(vec![0, 1], TerminalsDisallowed::new());
            let actual = plan.apply(0, &input).unwrap();
            let expected = literal(&t.push, 0, &input);
            assert_eq!(actual.semantically_eq(&expected, 65_536), Some(true));
            assert_eq!(actual.top_value_count(), count as usize);
        }
    }

    #[test]
    fn graph_native_push_matches_literal_words_for_random_dags_and_multiple_entries() {
        let mut seed=0x608894u64;
        let mut next=||{ seed^=seed<<13; seed^=seed>>7; seed^=seed<<17; seed };
        let mut covered=0;
        for _ in 0..256 {
            let mut graph=DFA::new(); for _ in 1..8 { graph.add_state(); }
            for i in 0..8u32 {
                graph.set_accepting(i, next()%3==0);
                if i==7 { continue; }
                for label in 0..4u32 {
                    if next()%3==0 { graph.add_transition(i,encode_negative_label(label),i+1+(next()%(7-i) as u64) as u32); }
                }
            }
            let mut t=template(graph,0); t.read_to_push=vec![Some(2),Some(5)];
            let Some(plan)=PreparedPushDag::compile(&t,0) else { continue; };
            let acc=TerminalsDisallowed::new().with_insert(1,3);
            for entry in [0,2,5] {
                for input in [ParserGSS::empty(),
                    ParserGSS::from_single_stack(Vec::new(),acc.clone()),
                    ParserGSS::from_single_stack(vec![0,6],acc.clone()),
                    ParserGSS::from_stacks(&[(vec![0,1],acc.clone()),(vec![0,2,3],acc.clone())]),
                ] {
                    let expected=literal(&t.push,entry,&input);
                    if let Some(actual)=plan.apply(entry,&input) {
                        covered+=1;
                        assert_eq!(actual.semantically_eq(&expected,65_536),Some(true));
                    } else { assert!(expected.is_empty()); }
                }
            }
        }
        assert!(covered>1_000,"test must exercise reachable accepting entry languages");
    }

    #[test]
    fn exponentially_many_outputs_remain_a_linear_shared_graph() {
        for depth in [8u32,16,24,64,128] {
            let mut graph=DFA::new();for _ in 0..depth {graph.add_state();}
            for i in 0..depth { for label in [1,2] {graph.add_transition(i,encode_negative_label(label),i+1);} }
            graph.set_accepting(depth,true);let t=template(graph,0);
            let plan=PreparedPushDag::prepare(&t).unwrap();
            let input=ParserGSS::from_single_stack(vec![0],TerminalsDisallowed::new());
            let actual=plan.apply(0,&input).unwrap();
            assert_eq!(actual.max_depth(),depth+1);
            assert!(actual.node_count_at_most(1024)<=depth as usize+5,
                "2^depth output words must share each DFA layer, not be enumerated: {:?}",actual.summary());
            assert_eq!(actual.top_value_count(),2);
            if depth==8 {assert_eq!(actual.semantically_eq(&literal(&t.push,0,&input),1024),Some(true));}
        }
    }

    #[test]
    fn correlated_annotations_decline_and_malformed_graphs_do_not_get_a_plan() {
        let mut graph=DFA::new();let end=graph.add_state();graph.set_accepting(end,true);
        graph.add_transition(0,encode_negative_label(1),end);
        let mut t=template(graph,0);
        let plan=PreparedPushDag::compile(&t,0).unwrap();
        let input=ParserGSS::from_stacks(&[(vec![0,1],TerminalsDisallowed::new()),
            (vec![0,2],TerminalsDisallowed::new().with_insert(0,3))]);
        assert!(plan.apply(0,&input).is_none());
        t.push.add_transition(end,encode_negative_label(1),0);assert!(PreparedPushDag::compile(&t,0).is_none());
        t.push.states[end as usize].transitions.clear();
        t.push.states[0].transitions.insert(1,end);assert!(PreparedPushDag::compile(&t,0).is_none());
    }
}
