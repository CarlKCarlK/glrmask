//! Conservative action-to-stack-effect projection.
//!
//! The native table path first records borrowed per-source effects. Sorting and
//! deduplication occur before allocating owned PUSH vectors. Entry effects and
//! EOF facts are captured in the same action traversal.
//!
//! These are conservative compiler facts, never executable parser rows.

use std::collections::{BTreeMap, BTreeSet};

use crate::compiler::glr::analysis::EOF;
use crate::compiler::glr::table::{Action, GLRTable};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub(crate) struct Effect {
    pub source: u32,
    pub pop: usize,
    pub pushes: Vec<u32>,
}

pub(super) type Link = (u32, u32, u32, u32, u32, bool);

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct CompilerEffects {
    pub(crate) states: u32,
    pub(crate) terminals: u32,
    pub(crate) effects: Vec<Effect>,
    pub(crate) entries: BTreeMap<u32, Vec<Effect>>,
    pub(crate) accepting: Vec<u32>,
}

impl CompilerEffects {
    pub(crate) fn validate(&self) -> Result<(), String> {
        let valid = |e: &Effect| {
            e.source < self.states && e.pop <= 1024
                && e.pushes.len() <= 1024
                && e.pushes.iter().all(|&q| q < self.states)
        };
        let entries = self.entries.values().map(Vec::len).sum::<usize>();
        if self.states == 0 || self.states > 50_000 || self.effects.len() > 200_000
            || entries > 200_000 || self.effects.iter().any(|e| !valid(e))
            || self.accepting.len() > self.states as usize
            || self.accepting.iter().any(|&q| q >= self.states)
            || self.entries.iter().any(|(&t, effects)| {
                t >= self.terminals || effects.iter().any(|e| {
                    !valid(e) || e.pop > 1 || e.pushes.len() != 1
                })
            })
        {
            return Err("invalid compiler stack-effect metadata".into());
        }
        Ok(())
    }

    pub(crate) fn from_table(
        table: &GLRTable, slots: &BTreeSet<u32>,
    ) -> Result<Self, String> {
        let summary = project_table(table, slots)?;
        crate::compiler::boundary_transfer::validate_slot_entry_shapes(table, slots)?;
        summary.validate()?;
        Ok(summary)
    }

    pub(crate) fn from_index(
        index: &glrmask_parser_dwa::__private::templates::native::NativeTableIndex<'_>,
    ) -> Result<Self, String> {
        let summary = project_table(index.table(), index.valid_slot_entry_terminals())?;
        summary.validate()?;
        Ok(summary)
    }

    pub(crate) fn from_regular_programs(
        programs: &[Option<std::sync::Arc<crate::runtime::CommitTemplateDfas>>],
        completion: &crate::runtime::CommitTemplateDfas,
        states: u32,
    ) -> Result<Self, String> {
        use crate::compiler::glr::labels::{DEFAULT_LABEL, negative_to_positive_label};
        let mut effects = BTreeSet::new();
        let mut entries = BTreeMap::new();
        for (terminal, program) in programs.iter().enumerate() {
            let program = program.as_deref().ok_or("missing regular effect program")?;
            let mut row = Vec::new();
            for (&top, &middle) in &program.pop.states[program.pop.start_state as usize].transitions {
                if top < 0 || top == DEFAULT_LABEL {
                    return Err("nonliteral regular POP effect".into());
                }
                let start = program.pop_to_push.get(middle as usize).copied().flatten()
                    .ok_or("regular effect lacks its PUSH phase")?;
                for (&label, &end) in &program.push.states[start as usize].transitions {
                    if label >= 0 || !program.push.states[end as usize].is_accepting
                        || !program.push.states[end as usize].transitions.is_empty()
                    {
                        return Err("nonunit regular PUSH effect".into());
                    }
                    let effect = Effect {
                        source: top as u32,
                        pop: 1,
                        pushes: vec![negative_to_positive_label(label) as u32],
                    };
                    effects.insert(effect.clone());
                    row.push(effect);
                }
            }
            entries.insert(terminal as u32, row);
        }
        let accepting = completion.pop.states[completion.pop.start_state as usize]
            .transitions.iter().filter_map(|(&label, &end)| {
                completion.pop.states[end as usize].is_accepting.then_some(label as u32)
            }).collect();
        let summary = Self {
            states,
            terminals: programs.len() as u32,
            effects: effects.into_iter().collect(),
            entries,
            accepting,
        };
        summary.validate()?;
        Ok(summary)
    }

    pub(crate) fn compose(
        sources: &[&Self],
        offsets: &[u32],
        terminal_offsets: &[u32],
        links: &[Link],
        states: u32,
        terminals: u32,
    ) -> Result<Self, String> {
        if sources.is_empty() || sources.len() != offsets.len()
            || sources.len() != terminal_offsets.len()
            || offsets[0] != 0 || terminal_offsets[0] != 0
        {
            return Err("invalid effect layout".into());
        }
        let mut effects = Vec::new();
        let mut linked_effects = BTreeSet::new();
        let mut entries = BTreeMap::new();
        for (i, source) in sources.iter().enumerate() {
            source.validate()?;
            if offsets[i].checked_add(source.states)
                != Some(offsets.get(i + 1).copied().unwrap_or(states))
                || terminal_offsets[i].checked_add(source.terminals)
                    != Some(terminal_offsets.get(i + 1).copied().unwrap_or(terminals))
            {
                return Err("incomplete compiler effect coordinate".into());
            }
            let relocate = |e: &Effect| Effect {
                source: offsets[i] + e.source,
                pop: e.pop,
                pushes: e.pushes.iter().map(|q| offsets[i] + q).collect(),
            };
            if source.effects.windows(2).all(|pair| pair[0] <= pair[1]) {
                let mut previous = None;
                for effect in &source.effects {
                    if previous != Some(effect) {
                        effects.push(relocate(effect));
                    }
                    previous = Some(effect);
                }
            } else {
                let mut local = source.effects.iter().collect::<Vec<_>>();
                local.sort_unstable();
                local.dedup();
                effects.extend(local.into_iter().map(relocate));
            }
            for (&terminal, row) in &source.entries {
                entries.insert(
                    terminal_offsets[i] + terminal,
                    row.iter().map(relocate).collect(),
                );
            }
        }
        for &(parent, slot, child, start, pop, nullable) in links {
            let p = sources.get(parent as usize).ok_or("invalid effect parent")?;
            let c = sources.get(child as usize).ok_or("invalid effect child")?;
            if start >= c.states {
                return Err("invalid effect child start".into());
            }
            for e in p.entries.get(&slot).ok_or("uncertified compiler CALL effect")? {
                linked_effects.insert(Effect {
                    source: offsets[parent as usize] + e.source,
                    pop: e.pop,
                    pushes: vec![
                        offsets[parent as usize] + e.pushes[0],
                        offsets[child as usize] + start,
                    ],
                });
            }
            for &q in &c.accepting {
                linked_effects.insert(Effect {
                    source: offsets[child as usize] + q,
                    pop: pop as usize,
                    pushes: vec![],
                });
            }
            if nullable {
                linked_effects.insert(Effect {
                    source: offsets[child as usize] + start,
                    pop: 1,
                    pushes: vec![],
                });
            }
            entries.remove(&(terminal_offsets[parent as usize] + slot));
        }
        if !linked_effects.is_empty() {
            effects.extend(linked_effects);
            effects.sort_unstable();
            effects.dedup();
        }
        let summary = Self {
            states,
            terminals,
            effects,
            entries,
            accepting: sources[0].accepting.clone(),
        };
        summary.validate()?;
        Ok(summary)
    }
}

fn borrowed_action_effects<'a>(
    action: &'a Action,
    forwarded: bool,
    out: &mut Vec<(usize, &'a [u32])>,
) {
    match action {
        Action::Shift(target, replace)
        | Action::Split { shift: Some((target, replace)), .. } => {
            out.push((usize::from(*replace), std::slice::from_ref(target)));
            if *replace && forwarded {
                out.push((0, std::slice::from_ref(target)));
            }
        }
        Action::ReplaceShifts(targets) => {
            for target in targets.iter() {
                out.push((1, std::slice::from_ref(target)));
            }
        }
        Action::StackShifts(shifts) => {
            for shift in shifts {
                out.push((shift.pop as usize, &shift.pushes));
            }
        }
        Action::GuardedStackShifts(shifts) => {
            for shift in shifts {
                out.push((shift.pop as usize, &shift.pushes));
            }
        }
        Action::Reduce(..) | Action::Accept | Action::Skip
        | Action::Split { shift: None, .. } => {}
    }
}

fn project_table(
    table: &GLRTable, slots: &BTreeSet<u32>,
) -> Result<CompilerEffects, String> {
    if !table.control_terminals.is_empty() {
        return Err("unmapped in-table control terminals".into());
    }
    let mut entries = slots.iter().map(|&slot| (slot, Vec::new()))
        .collect::<BTreeMap<_, _>>();
    let mut effects = Vec::new();
    let mut accepting = Vec::new();
    let mut local = Vec::<(usize, &[u32])>::new();

    for source in 0..table.action.len().max(table.goto.len()) {
        local.clear();
        if let Some(row) = table.action.get(source) {
            for (terminal, action) in row.iter() {
                let forwarded = matches!(
                    action,
                    Action::Shift(_, true)
                        | Action::Split { shift: Some((_, true)), .. }
                ) && table.forwarded_shifts.contains(&(source as u32, terminal));
                borrowed_action_effects(action, forwarded, &mut local);
                if source < table.num_states as usize {
                    if terminal == EOF && matches!(
                        action, Action::Accept | Action::Split { accept: true, .. }
                    ) {
                        accepting.push(source as u32);
                    }
                    if let Some(entry) = entries.get_mut(&terminal)
                        && let Action::Shift(target, replace)
                            | Action::Split { shift: Some((target, replace)), .. } = action
                    {
                        entry.push(Effect {
                            source: source as u32,
                            pop: usize::from(*replace),
                            pushes: vec![*target],
                        });
                    }
                }
            }
        }
        if let Some(row) = table.goto.get(source) {
            for (_, (target, replace)) in row.iter() {
                local.push((usize::from(*replace), std::slice::from_ref(target)));
            }
        }
        local.sort_unstable();
        local.dedup();
        if local.iter().any(|&(pop, pushes)| {
            source >= table.num_states as usize || pop > 1024
                || pushes.len() > 1024
                || pushes.iter().any(|&target| target >= table.num_states)
        }) || effects.len().saturating_add(local.len()) > 200_000
        {
            return Err("stack effect coordinate or resource bound".into());
        }
        effects.extend(local.iter().map(|&(pop, pushes)| Effect {
            source: source as u32,
            pop,
            pushes: pushes.to_vec(),
        }));
    }
    Ok(CompilerEffects {
        states: table.num_states,
        terminals: table.num_terminals,
        effects,
        entries,
        accepting,
    })
}

fn action_effects(
    action: &Action,
    source: u32,
    offset: u32,
    extra_nonreplace: bool,
    out: &mut BTreeSet<Effect>,
) {
    let mut add = |pop: u32, pushes: &[u32]| {
        out.insert(Effect {
            source,
            pop: pop as usize,
            pushes: pushes.iter().map(|q| offset + q).collect(),
        });
    };
    match action {
        Action::Shift(q, replace) => {
            add(u32::from(*replace), &[*q]);
            if *replace && extra_nonreplace { add(0, &[*q]); }
        }
        Action::ReplaceShifts(targets) => {
            for q in targets.iter() { add(1, &[*q]); }
        }
        Action::StackShifts(shifts) => {
            for s in shifts { add(s.pop, &s.pushes); }
        }
        Action::GuardedStackShifts(shifts) => {
            for s in shifts { add(s.pop, &s.pushes); }
        }
        Action::Split { shift, .. } => {
            if let Some((q, replace)) = shift {
                add(u32::from(*replace), &[*q]);
                if *replace && extra_nonreplace { add(0, &[*q]); }
            }
        }
        Action::Reduce(..) | Action::Accept | Action::Skip => {}
    }
}

pub(super) fn read_effects(
    tables: &[&GLRTable], offsets: &[u32], links: &[Link], alphabet: u32,
) -> Result<Vec<Effect>, String> {
    if tables.is_empty() || offsets.len() != tables.len() || offsets[0] != 0 {
        return Err("invalid scoped table layout".into());
    }
    for (i, table) in tables.iter().enumerate() {
        let end = offsets[i].checked_add(table.num_states).ok_or("table state overflow")?;
        if end != offsets.get(i + 1).copied().unwrap_or(alphabet) {
            return Err("table offsets are not complete disjoint layout".into());
        }
        if !table.control_terminals.is_empty() {
            return Err("unmapped in-table control terminals".into());
        }
    }
    let mut effects = BTreeSet::new();
    for (owner, table) in tables.iter().enumerate() {
        let offset = offsets[owner];
        for (q, row) in table.action.iter().enumerate() {
            for (terminal, action) in row.iter() {
                action_effects(
                    action, offset + q as u32, offset,
                    table.forwarded_shifts.contains(&(q as u32, terminal)), &mut effects,
                );
            }
        }
        for (q, row) in table.goto.iter().enumerate() {
            for (_, &(target, replace)) in row.iter() {
                effects.insert(Effect {
                    source: offset + q as u32,
                    pop: usize::from(replace),
                    pushes: vec![offset + target],
                });
            }
        }
    }
    for &(parent, slot, child, child_start, return_pop, nullable) in links {
        let p = tables.get(parent as usize).ok_or("invalid parent")?;
        let c = tables.get(child as usize).ok_or("invalid child")?;
        if child_start >= c.num_states { return Err("invalid child start".into()); }
        for q in 0..p.num_states {
            match p.action(q, slot) {
                Some(Action::Shift(target, replace))
                | Some(Action::Split { shift: Some((target, replace)), .. }) => {
                    effects.insert(Effect {
                        source: offsets[parent as usize] + q,
                        pop: usize::from(*replace),
                        pushes: vec![
                            offsets[parent as usize] + target,
                            offsets[child as usize] + child_start,
                        ],
                    });
                }
                None | Some(Action::Reduce(..))
                | Some(Action::Split { shift: None, accept: false, .. }) => {}
                other => return Err(format!("unsupported CALL effect shape: {other:?}")),
            }
        }
        for q in 0..c.num_states {
            if matches!(
                c.action(q, EOF), Some(Action::Accept) | Some(Action::Split { accept: true, .. })
            ) {
                effects.insert(Effect {
                    source: offsets[child as usize] + q,
                    pop: return_pop as usize,
                    pushes: vec![],
                });
            }
        }
        if nullable {
            effects.insert(Effect {
                source: offsets[child as usize] + child_start,
                pop: 1,
                pushes: vec![],
            });
        }
    }
    if effects.len() > 200_000 || effects.iter().any(|e| {
        e.source >= alphabet || e.pushes.iter().any(|&q| q >= alphabet)
            || e.pop > 1024 || e.pushes.len() > 1024
    }) {
        return Err("stack effect coordinate or resource bound".into());
    }
    Ok(effects.into_iter().collect())
}

#[cfg(test)]
mod mapper_tests {
    use super::*;
    use glrmask_glr::__private::glr::table::action::{
        StackShift, GuardedStackShift, StackShiftGuard,
    };

    #[test]
    fn batched_entry_effect_rows_preserve_per_slot_source_order_and_default_cells() {
        use crate::compiler::glr::table::testing::build_test_table;
        for case in 0..32usize {
            let mut table = build_test_table(
                4, 19, &[&[], &[], &[], &[]], &[&[], &[], &[], &[]],
            );
            for state in 0..4usize {
                for terminal in 0..19u32 {
                    let action = match (state + terminal as usize + case) % 4 {
                        0 => Action::Shift(((state + 1) % 4) as u32, false),
                        1 => Action::Split {
                            shift: Some((((state + 2) % 4) as u32, true)),
                            reduces: vec![],
                            accept: false,
                        },
                        _ => Action::Reduce(0, 1),
                    };
                    table.action[state].insert(terminal, action);
                }
                table.action[state].compress_default(19);
            }
            let slots = (0..19u32).collect::<BTreeSet<_>>();
            let actual = CompilerEffects::from_table(&table, &slots).unwrap();
            let expected = slots.iter().map(|&slot| {
                let row = (0..table.num_states).filter_map(|source| {
                    match table.action(source, slot) {
                        Some(Action::Shift(target, replace)
                            | Action::Split { shift: Some((target, replace)), .. }) => {
                            Some(Effect {
                                source, pop: usize::from(*replace), pushes: vec![*target],
                            })
                        }
                        _ => None,
                    }
                }).collect::<Vec<_>>();
                (slot, row)
            }).collect::<BTreeMap<_, _>>();
            assert_eq!(actual.entries, expected, "case {case}");
            assert_eq!(
                actual.effects,
                read_effects(&[&table], &[0], &[], table.num_states).unwrap(),
            );
        }
    }

    #[test]
    fn maps_every_consuming_action_and_forwards_conservatively() {
        let cases = vec![
            Action::Shift(3, false),
            Action::Shift(3, true),
            Action::ReplaceShifts(vec![2, 3].into()),
            Action::StackShifts(vec![StackShift { pop: 2, pushes: vec![1, 3] }]),
            Action::GuardedStackShifts(vec![GuardedStackShift {
                guards: vec![StackShiftGuard { pop: 3, states: vec![2] }],
                pop: 1, pushes: vec![2, 3],
            }]),
            Action::Split {
                shift: Some((3, true)), reduces: vec![(0, 2)], accept: false,
            },
        ];
        for action in cases {
            let mut out = BTreeSet::new();
            action_effects(&action, 11, 10, true, &mut out);
            assert!(!out.is_empty());
            for e in out {
                assert_eq!(e.source, 11);
                assert!(e.pushes.iter().all(|q| (10..14).contains(q)));
            }
        }
    }

    #[test]
    fn retained_component_effects_match_the_existing_scoped_mapper_after_wire_roundtrip() {
        use crate::compiler::glr::table::testing::build_test_table;
        let mut parent = build_test_table(
            3, 2,
            &[
                &[(0, Action::Shift(2, true)), (1, Action::Shift(1, false))],
                &[(EOF, Action::Accept)],
                &[(0, Action::Reduce(0, 1))],
            ],
            &[&[(0, (2, false))], &[(0, (0, true))], &[]],
        );
        parent.forwarded_shifts.insert((0, 0));
        let child = build_test_table(
            2, 1,
            &[&[(0, Action::Shift(1, false))], &[(EOF, Action::Accept)]],
            &[&[], &[]],
        );
        let p = CompilerEffects::from_table(&parent, &BTreeSet::from([1])).unwrap();
        let c = CompilerEffects::from_table(&child, &BTreeSet::new()).unwrap();
        assert_eq!(p.effects, read_effects(&[&parent], &[0], &[], 3).unwrap());
        assert!(p.effects.contains(&Effect { source: 0, pop: 0, pushes: vec![2] }));
        assert_eq!(p.entries[&1], vec![Effect { source: 0, pop: 0, pushes: vec![1] }]);
        for nullable in [false, true] {
            let links = [(0, 1, 1, 0, 1, nullable)];
            let composed = CompilerEffects::compose(
                &[&p, &c], &[0, 3], &[0, 2], &links, 5, 3,
            ).unwrap();
            let mut unsorted = p.clone();
            unsorted.effects.reverse();
            unsorted.effects.push(p.effects[0].clone());
            let repeated = [links[0], links[0]];
            assert_eq!(
                CompilerEffects::compose(
                    &[&unsorted, &c], &[0, 3], &[0, 2], &repeated, 5, 3,
                ).unwrap(),
                composed,
            );
            let decoded: CompilerEffects =
                bincode::deserialize(&bincode::serialize(&composed).unwrap()).unwrap();
            assert_eq!(decoded, composed);
            assert_eq!(
                decoded.effects,
                read_effects(&[&parent, &child], &[0, 3], &links, 5).unwrap(),
            );
            assert_eq!(decoded.accepting, vec![1]);
            assert!(!decoded.entries.contains_key(&1));
            let certificate = super::super::from_compiler_effects(&decoded).unwrap();
            assert!(certificate.domain.accepts(&[3, 1, 0]));
            assert!(certificate.native_context().is_some());
        }
        let mut bad = p.clone();
        bad.effects.push(Effect { source: 3, pop: 0, pushes: vec![] });
        assert!(bad.validate().is_err());
        assert!(CompilerEffects::compose(
            &[&p, &c], &[0, 2], &[0, 2], &[], 5, 3,
        ).is_err());
    }

    #[test]
    fn native_index_certificate_and_borrowed_projection_match_reference_effects() {
        use crate::compiler::glr::table::testing::build_test_table;
        use glrmask_parser_dwa::__private::templates::native::NativeTableIndex;
        let table = build_test_table(
            3, 3,
            &[
                &[(0, Action::Shift(1, false)), (1, Action::Shift(1, false))],
                &[(2, Action::Reduce(0, 1))],
                &[(EOF, Action::Accept)],
            ],
            &[&[(0, (2, false))], &[], &[]],
        );
        let index = NativeTableIndex::new(&table, &[true; 3]).unwrap();
        let actual = CompilerEffects::from_index(&index).unwrap();
        let reference = CompilerEffects::from_table(
            &table, index.valid_slot_entry_terminals(),
        ).unwrap();
        assert_eq!(actual, reference);
        assert_eq!(
            actual.effects,
            read_effects(&[&table], &[0], &[], table.num_states).unwrap(),
        );
    }
}
