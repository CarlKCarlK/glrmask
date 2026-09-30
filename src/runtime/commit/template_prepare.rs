//! Shared, immutable validation used by both input domains and fast views.
//! The borrowed proof owns only temporary topological orders; all runtime
//! indices retain their existing representations and independent fallbacks.
pub(crate) use glrmask_parser_dwa::__private::templates::admissibility::ValidatedTemplate as TemplatePreparation;

#[cfg(test)]
mod tests {
    use super::super::{
        phase_dag::PreparedPhaseDag, push_dag::PreparedPushDag, push_suffixes,
        single_cursor::PreparedInputCursor,
    };
    use super::*;
    use crate::automata::unweighted_u32::dfa::DFA;
    use crate::compiler::glr::labels::DEFAULT_LABEL;
    use crate::runtime::CommitTemplateDfas;
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
        assert!(TemplatePreparation::new(&t).is_err());
        let mut t = base.clone();
        t.read.add_transition(0, DEFAULT_LABEL, 0);
        assert!(TemplatePreparation::new(&t).is_err());
        let mut t = base.clone();
        t.push.add_transition(0, 1, 0);
        assert!(TemplatePreparation::new(&t).is_err());
        let mut t = base.clone();
        t.pop_to_push = vec![Some(100)];
        assert!(TemplatePreparation::new(&t).is_err());
        let mut t = base.clone();
        t.read_to_push = vec![None, None];
        assert!(TemplatePreparation::new(&t).is_err());
        let mut t = base;
        t.pop.start_state = 1;
        assert!(TemplatePreparation::new(&t).is_err());
    }
}
