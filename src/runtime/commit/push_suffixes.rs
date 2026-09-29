//! Bounded exact output-language materialization for acyclic PUSH programs.
//!
//! This is a derived acceleration index, never a replacement for the complete
//! template. Large languages stay as DAGs; no exponential path enumeration is
//! performed. The usual shared GSS branch constructor owns all stack storage.
use crate::compiler::glr::labels::{is_negative_label, negative_to_positive_label};
use crate::compiler::glr::parser::ParserGSS;
use crate::runtime::CommitTemplateDfas;

const MAX_PATHS: usize = 64;
const MAX_LENGTH: usize = 8;
const MAX_TOTAL_SYMBOLS: usize = 4_096;

#[derive(Clone, Debug)]
pub(crate) struct PreparedPushSuffixes {
    accepts_empty: bool,
    suffixes: Box<[Box<[u32]>]>,
    single_symbols: bool,
}

impl PreparedPushSuffixes {
    pub(crate) fn apply(&self, base: &ParserGSS) -> Option<ParserGSS> {
        // Reordering output construction is valid only for one shared path
        // annotation. Correlated inputs retain the authoritative DAG walk.
        if base.is_empty() { return Some(ParserGSS::empty()); }
        base.single_interface_lower_id()?;
        // An empty list would mean the empty relation, not the identity
        // returned by the GSS helper for an empty branch iterator.
        let nonempty = if self.suffixes.is_empty() {
            ParserGSS::empty()
        } else if self.single_symbols {
            base.apply_shared_pop_push_single_branches(0, self.suffixes.iter().map(|s| &s[0]))?
        } else {
            base.apply_shared_pop_push_branches(0, self.suffixes.iter().map(|s| s.as_ref()))?
        };
        Some(if self.accepts_empty { base.merge(&nonempty) } else { nonempty })
    }
}

pub(crate) fn prepare(template: &CommitTemplateDfas) -> Vec<Option<PreparedPushSuffixes>> {
    let graph = &template.push;
    let n = graph.states.len();
    let empty = || (0..n).map(|_| None).collect::<Vec<_>>();
    // Only actual phase entry points need prepared suffix languages. Intermediate
    // PUSH cursors remain in the generic path when an entry exceeds this budget.
    let mut entry = vec![false; n];
    for target in template.pop_to_push.iter().chain(&template.read_to_push).flatten() {
        let Some(slot) = entry.get_mut(*target as usize) else { return empty(); };
        *slot = true;
    }
    let mut indegree = vec![0usize; n];
    for state in &graph.states {
        for (&label, &target) in &state.transitions {
            if !is_negative_label(label) { return empty(); }
            let Some(degree) = indegree.get_mut(target as usize) else { return empty(); };
            *degree += 1;
        }
    }
    let mut order: Vec<_> = indegree.iter().enumerate()
        .filter_map(|(i, &d)| (d == 0).then_some(i)).collect();
    let mut head = 0;
    while head < order.len() {
        let source = order[head]; head += 1;
        for &target in graph.states[source].transitions.values() {
            indegree[target as usize] -= 1;
            if indegree[target as usize] == 0 { order.push(target as usize); }
        }
    }
    if order.len() != n { return empty(); }
    let mut counts = vec![0usize; n];
    let mut lengths = vec![0usize; n];
    for &i in order.iter().rev() {
        counts[i] = usize::from(graph.states[i].is_accepting);
        for &target in graph.states[i].transitions.values() {
            let child = target as usize;
            counts[i] = (counts[i] + counts[child]).min(MAX_PATHS + 1);
            if counts[child] != 0 { lengths[i] = lengths[i].max((lengths[child]+1).min(MAX_LENGTH+1)); }
        }
    }
    let mut plans = empty();
    let mut used = 0usize;
    for start in 0..n {
        if !entry[start] || counts[start] < 4 || counts[start] > MAX_PATHS || lengths[start] > MAX_LENGTH { continue; }
        if used.saturating_add(counts[start] * lengths[start]) > MAX_TOTAL_SYMBOLS { continue; }
        let mut paths = Vec::with_capacity(counts[start]);
        let mut pending = vec![(start, Vec::<u32>::new())];
        let mut accepts_empty = false;
        while let Some((at, path)) = pending.pop() {
            let state = &graph.states[at];
            if state.is_accepting {
                if path.is_empty() { accepts_empty = true; }
                else { used += path.len(); paths.push(path.clone().into_boxed_slice()); }
            }
            for (&label, &target) in &state.transitions {
                if counts[target as usize] == 0 { continue; }
                let mut next = path.clone(); next.push(negative_to_positive_label(label) as u32);
                pending.push((target as usize, next));
            }
        }
        debug_assert_eq!(paths.len() + usize::from(accepts_empty), counts[start]);
        plans[start] = Some(PreparedPushSuffixes {
            accepts_empty,
            single_symbols: paths.iter().all(|s| s.len() == 1),
            suffixes: paths.into_boxed_slice(),
        });
    }
    plans
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::unweighted_u32::dfa::DFA;
    use crate::compiler::glr::{accumulator::TerminalsDisallowed, labels::encode_negative_label};

    fn template(push: DFA) -> CommitTemplateDfas {
        CommitTemplateDfas { pop: DFA::new(), read: DFA::new(), push,
            pop_to_read: Vec::new(), pop_to_push: vec![Some(0)], read_to_push: Vec::new() }
    }
    fn literal(graph: &DFA, start: u32, input: &ParserGSS) -> ParserGSS {
        let mut out = Vec::new();
        for (stack, acc) in input.to_stacks(100).unwrap() {
            let mut pending = vec![(start, stack)];
            while let Some((i, stack)) = pending.pop() {
                let state = &graph.states[i as usize];
                if state.is_accepting { out.push((stack.clone(), acc.clone())); }
                for (&label, &target) in &state.transitions {
                    let mut next=stack.clone(); next.push(negative_to_positive_label(label) as u32);
                    pending.push((target,next));
                }
            }
        }
        ParserGSS::from_stacks(&out)
    }

    #[test]
    fn prepared_suffixes_match_independent_literal_interpreter() {
        let mut seed=608894u64;
        let mut next=||{seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;seed};
        let mut covered=0;
        for _ in 0..512 {
            let mut push=DFA::new(); for _ in 1..8 { push.add_state(); }
            for i in 0..8u32 {
                push.set_accepting(i,next()%3==0);
                if i==7 {continue;}
                for label in 0..5u32 {
                    if next()%3==0 { push.add_transition(i,encode_negative_label(label),i+1+(next()%(7-i) as u64) as u32); }
                }
            }
            let t=template(push);
            let plans=prepare(&t);
            let Some(plan)=plans[0].as_ref() else {continue;};
            covered+=1;
            for input in [
                ParserGSS::empty(),
                ParserGSS::from_single_stack(Vec::new(),TerminalsDisallowed::new()),
                ParserGSS::from_single_stack(vec![0,6,191],TerminalsDisallowed::new()),
                ParserGSS::from_stacks(&[(vec![0,1],TerminalsDisallowed::new()),(vec![0,2,3],TerminalsDisallowed::new())]),
            ] {
                let expected=literal(&t.push,0,&input);
                match plan.apply(&input) {
                    Some(actual) => assert_eq!(actual.semantically_eq(&expected,16_384),Some(true)),
                    None => assert!(input.try_virtual_stack().is_none(),
                        "only an unsupported storage shape may decline this exact shortcut"),
                }
            }
        }
        assert!(covered>50,"generated tests must actually exercise prepared multi-output languages");
    }

    #[test]
    fn twenty_seven_targets_share_the_same_lower_prefix() {
        let mut push=DFA::new();let end=push.add_state();push.set_accepting(end,true);
        for top in 216..243 {push.add_transition(0,encode_negative_label(top),end);}
        let t=template(push);let plans=prepare(&t);let plan=plans[0].as_ref().unwrap();
        let base=ParserGSS::from_single_stack(vec![0,6],TerminalsDisallowed::new());
        let out=plan.apply(&base).unwrap();
        assert_eq!(out.top_value_count(),27);
        assert_eq!(out.to_stacks(100).unwrap().len(),27);
        assert_eq!(out.semantically_eq(&literal(&t.push,0,&base),100),Some(true));
        assert!(out.summary().total_unique_nodes<=5,"do not duplicate the shared lower prefix per output");
    }

    #[test]
    fn exponential_or_deep_languages_decline_without_enumeration() {
        for (depth,fan) in [(20,2),(10,1)] {
            let mut push=DFA::new();for _ in 0..depth {push.add_state();}
            for i in 0..depth {for label in 0..fan {push.add_transition(i,encode_negative_label(label),i+1);}}
            push.set_accepting(depth,true);let t=template(push);
            assert!(prepare(&t).iter().all(Option::is_none));
        }
    }

    #[test]
    fn correlated_annotations_decline_and_phase_validation_is_not_relaxed() {
        let mut push=DFA::new();let end=push.add_state();push.set_accepting(end,true);
        for top in 0..5 {push.add_transition(0,encode_negative_label(top),end);}
        let mut t=template(push);let plans=prepare(&t);
        let input=ParserGSS::from_stacks(&[
            (vec![0,1],TerminalsDisallowed::new()),
            (vec![0,2],TerminalsDisallowed::new().with_insert(0,3)),
        ]);
        assert!(plans[0].as_ref().unwrap().apply(&input).is_none());
        t.push.states[0].transitions.insert(0,end);
        assert!(prepare(&t).iter().all(Option::is_none));
        t.push.states[0].transitions.remove(&0);
        t.push.add_transition(end,encode_negative_label(7),0);
        assert!(prepare(&t).iter().all(Option::is_none));
    }
}
