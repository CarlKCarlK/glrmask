//! One validation/topological pass for the derived views of an immutable
//! acyclic stack program. This is preparation-only scratch, not another
//! runtime parser, a persisted index, or a global cache.
use crate::automata::unweighted_u32::dfa::DFA;
use crate::compiler::glr::labels::{DEFAULT_LABEL, is_negative_label};
use crate::runtime::CommitTemplateDfas;

pub(crate) struct TemplatePreparation<'a> {
    template: &'a CommitTemplateDfas,
    orders: [Vec<usize>; 3],
}

fn validated_order(graph: &DFA, phase: usize) -> Option<Vec<usize>> {
    if graph.states.is_empty() {
        return (graph.start_state == 0).then(Vec::new);
    }
    graph.states.get(graph.start_state as usize)?;
    let mut incoming = vec![0usize; graph.states.len()];
    for row in &graph.states {
        for (&label, &target) in &row.transitions {
            let legal = match phase {
                0 => !is_negative_label(label),
                1 => !is_negative_label(label) && label != DEFAULT_LABEL,
                2 => is_negative_label(label),
                _ => return None,
            };
            if !legal {
                return None;
            }
            let degree = incoming.get_mut(target as usize)?;
            *degree = degree.checked_add(1)?;
        }
    }
    let mut order = incoming
        .iter()
        .enumerate()
        .filter_map(|(id, &degree)| (degree == 0).then_some(id))
        .collect::<Vec<_>>();
    let mut head = 0;
    while head < order.len() {
        let source = order[head];
        head += 1;
        for &target in graph.states[source].transitions.values() {
            incoming[target as usize] -= 1;
            if incoming[target as usize] == 0 {
                order.push(target as usize);
            }
        }
    }
    (order.len() == graph.states.len()).then_some(order)
}

impl<'a> TemplatePreparation<'a> {
    pub(crate) fn new(template: &'a CommitTemplateDfas) -> Option<Self> {
        let graphs = [&template.pop, &template.read, &template.push];
        for (links, source, target) in [
            (&template.pop_to_read, 0, 1),
            (&template.pop_to_push, 0, 2),
            (&template.read_to_push, 1, 2),
        ] {
            if links.len() > graphs[source].states.len()
                || links
                    .iter()
                    .flatten()
                    .any(|&id| id as usize >= graphs[target].states.len())
            {
                return None;
            }
        }
        let orders = [
            validated_order(graphs[0], 0)?,
            validated_order(graphs[1], 1)?,
            validated_order(graphs[2], 2)?,
        ];
        Some(Self { template, orders })
    }

    pub(crate) fn template(&self) -> &'a CommitTemplateDfas {
        self.template
    }
    pub(crate) fn orders(&self) -> &[Vec<usize>; 3] {
        &self.orders
    }
    pub(crate) fn push_order(&self) -> &[usize] {
        &self.orders[2]
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        phase_dag::PreparedPhaseDag, push_dag::PreparedPushDag, push_suffixes,
        single_cursor::PreparedInputCursor,
    };
    use super::*;
    use crate::compiler::glr::labels::encode_negative_label;

    fn assert_same(template: &CommitTemplateDfas) {
        let prepared = TemplatePreparation::new(template).unwrap();
        // The old entry points validate/prepare independently. Compare every
        // derived field, not only behavior on one exercised stack.
        assert_eq!(
            format!("{:?}", PreparedInputCursor::prepare(template)),
            format!("{:?}", Some(PreparedInputCursor::from_prepared(&prepared)))
        );
        assert_eq!(
            format!("{:?}", PreparedPhaseDag::prepare(template)),
            format!("{:?}", PreparedPhaseDag::from_prepared(&prepared))
        );
        assert_eq!(
            format!("{:?}", push_suffixes::prepare(template)),
            format!("{:?}", push_suffixes::from_prepared(&prepared))
        );
        assert_eq!(
            format!("{:?}", PreparedPushDag::prepare(template)),
            format!("{:?}", PreparedPushDag::from_prepared(&prepared))
        );
    }

    #[test]
    fn shared_validation_builds_identical_views_for_generated_programs() {
        let mut seed = 0x6088941134u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..512 {
            let mut t = CommitTemplateDfas {
                pop: DFA::new(),
                read: DFA::new(),
                push: DFA::new(),
                pop_to_read: Vec::new(),
                pop_to_push: Vec::new(),
                read_to_push: Vec::new(),
            };
            let n = 3 + (next() % 8) as usize;
            for (phase, graph) in [&mut t.pop, &mut t.read, &mut t.push]
                .into_iter()
                .enumerate()
            {
                for _ in 1..n {
                    graph.add_state();
                }
                let mut order = (0..n).collect::<Vec<_>>();
                for i in (1..n).rev() {
                    let j = next() as usize % (i + 1);
                    order.swap(i, j);
                }
                graph.start_state = order[0] as u32;
                for at in 0..n {
                    let id = order[at];
                    graph.set_accepting(id as u32, next() % 3 == 0);
                    if at + 1 == n {
                        continue;
                    }
                    for symbol in 0..5 {
                        if next() % 3 == 0 {
                            continue;
                        }
                        let target = order[at + 1 + next() as usize % (n - at - 1)];
                        let label = match phase {
                            0 if symbol == 4 => DEFAULT_LABEL,
                            2 => encode_negative_label(symbol),
                            _ => symbol as i32,
                        };
                        graph.add_transition(id as u32, label, target as u32);
                    }
                }
            }
            for links in [&mut t.pop_to_read, &mut t.pop_to_push, &mut t.read_to_push] {
                *links = (0..n)
                    .map(|_| {
                        if next() % 3 == 0 {
                            None
                        } else {
                            Some((next() % n as u64) as u32)
                        }
                    })
                    .collect();
                while links.last() == Some(&None) {
                    links.pop();
                }
            }
            assert_same(&t);
        }
        assert_same(&CommitTemplateDfas::default());
        for phase in 0..3 {
            let mut t = CommitTemplateDfas {
                pop: DFA::new(),
                read: DFA::new(),
                push: DFA::new(),
                ..CommitTemplateDfas::default()
            };
            match phase {
                0 => t.pop = DFA::default(),
                1 => t.read = DFA::default(),
                _ => t.push = DFA::default(),
            }
            assert_same(&t);
        }
    }

    #[test]
    fn shared_preparation_rejects_invalid_unreachable_graph_data() {
        let base = CommitTemplateDfas {
            pop: DFA::new(),
            read: DFA::new(),
            push: DFA::new(),
            ..CommitTemplateDfas::default()
        };
        let mut t = base.clone();
        t.pop.add_state();
        t.pop.add_transition(1, DEFAULT_LABEL, 1);
        assert!(TemplatePreparation::new(&t).is_none());
        let mut t = base.clone();
        t.read.add_transition(0, DEFAULT_LABEL, 0);
        assert!(TemplatePreparation::new(&t).is_none());
        let mut t = base.clone();
        t.push.add_transition(0, 1, 0);
        assert!(TemplatePreparation::new(&t).is_none());
        let mut t = base.clone();
        t.pop_to_push = vec![Some(100)];
        assert!(TemplatePreparation::new(&t).is_none());
        let mut t = base.clone();
        t.read_to_push = vec![None, None];
        assert!(TemplatePreparation::new(&t).is_none());
        let mut t = base;
        t.pop.start_state = 1;
        assert!(TemplatePreparation::new(&t).is_none());
    }
}
