//! Standalone completion predicate compilation for the built-in LR frontend.
//!
//! The temporary LR projection is construction-only; callers can serialize the
//! returned acyclic template and drop the original table. Runtime completion
//! then queries `TemplateDomain`, exactly as an external data-only provider can.

use crate::compiler::glr::table::GLRTable;
use crate::runtime::CommitTemplateDfas;
use super::characterize::characterize_completion_domain;
use super::compile_dfa::{Templates, specialize_template_dfa_defaults_for_commit_split_input, try_split_commit_template_dfas};

/// Compile a standalone EOF acceptance relation. Its domain, not its output
/// stack, is the completion predicate. Linker controls/cyclic relations fail.
pub fn compile_completion_template(table: &GLRTable) -> Result<CommitTemplateDfas, String> {
    let characterization = characterize_completion_domain(table)?;
    let mut templates = Templates::from_characterizations(&std::collections::BTreeMap::from([(0, characterization)]));
    let raw = templates.by_terminal.remove(&0).ok_or("missing completion template")?;
    let specialized = specialize_template_dfa_defaults_for_commit_split_input(&raw);
    if !specialized.compute_is_acyclic() {
        return Err("completion template is cyclic".to_owned());
    }
    let split = try_split_commit_template_dfas(&specialized)
        .ok_or("completion relation violates the POP/READ/PUSH phase discipline")?;
    // Validate phase links/indices as well. This also independently checks the
    // domain can be represented without retaining any output machinery.
    super::admissibility::TemplateDomain::compile(&split)?;
    Ok(split)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::analysis::EOF;
    use crate::compiler::glr::accumulator::TerminalsDisallowed;
    use crate::compiler::glr::parser::{ParserGSS, stacks_finished};
    use crate::compiler::glr::table::{Action, AdmissionPolicy, StackShift, GuardedStackShift};
    use crate::compiler::glr::table::testing::build_test_table;
    use super::super::admissibility::{TemplateDomain, TopAdmission};

    fn matches(table: &GLRTable, stack: &[u32]) -> bool {
        let split = compile_completion_template(table).unwrap();
        TemplateDomain::compile(&split).unwrap().matches_top_first(stack.iter().rev().copied())
    }

    #[test]
    fn completion_follows_real_predecessor_reductions_and_rejects_bare_eof_progress() {
        let table = build_test_table(10, 1, &[
            &[], &[], &[(EOF, Action::Reduce(0, 1))], &[(EOF, Action::Accept)],
            &[(EOF, Action::Shift(3, false))], &[(EOF, Action::Skip)],
            &[(EOF, Action::ReplaceShifts(vec![3].into()))],
            &[(EOF, Action::StackShifts(vec![StackShift { pop: 0, pushes: vec![3] }]))],
            &[(EOF, Action::GuardedStackShifts(vec![GuardedStackShift { guards: vec![], pop: 0, pushes: vec![3] }]))],
            &[(EOF, Action::Split { shift: Some((3, false)), reduces: vec![], accept: false })],
        ], &[&[(0, (3, false))], &[], &[], &[], &[], &[], &[], &[], &[], &[]]);
        assert!(matches(&table, &[0, 2]));
        assert!(!matches(&table, &[1, 2]));
        assert!(!matches(&table, &[2]));
        assert!(matches(&table, &[3]));
        for top in 4..10 { assert!(!matches(&table, &[0, top]), "bare EOF progress top={top}"); }
        let domain = TemplateDomain::compile(&compile_completion_template(&table).unwrap()).unwrap();
        assert_eq!(domain.classify_top(2), TopAdmission::DependsOnSuffix);
        assert_eq!(domain.classify_top(3), TopAdmission::Always);
        assert_eq!(domain.classify_top(4), TopAdmission::Never);
    }

    #[test]
    fn completion_accepting_split_dominates_other_actions_but_nonaccepting_split_does_not() {
        let table = build_test_table(5, 1, &[
            &[], &[], &[(EOF, Action::Split { shift: Some((3, false)), reduces: vec![(0, 1)], accept: false })],
            &[(EOF, Action::Accept)],
            &[(EOF, Action::Split { shift: Some((3, false)), reduces: vec![(0, 1)], accept: true })],
        ], &[&[(0, (3, false))], &[], &[], &[], &[]]);
        assert!(matches(&table, &[0, 2]));
        assert!(!matches(&table, &[1, 2]));
        assert!(matches(&table, &[4]));
    }

    #[test]
    fn completion_handles_nullable_and_replacing_gotos_and_ignores_unrelated_forwarding() {
        for replace in [false, true] {
            let mut table = build_test_table(4, 1, &[
                &[], &[], &[(EOF, Action::Reduce(0, 0))], &[(EOF, Action::Accept)],
            ], &[&[], &[], &[(0, (3, replace))], &[]]);
            table.forwarded_shifts.insert((2, 0)); // Real terminal, not EOF.
            assert!(matches(&table, &[2]));
            assert!(!matches(&table, &[]));
            table.advance[3].clear(table.num_terminals as usize);
            assert!(!matches(&table, &[2])); // Respect authoritative admission support.
        }
    }

    #[test]
    fn completion_declines_linker_control_closure_instead_of_using_a_weaker_probe() {
        let mut table = build_test_table(1, 1, &[&[(EOF, Action::Accept)]], &[&[]]);
        table.control_terminals.insert(0);
        assert!(compile_completion_template(&table).unwrap_err().contains("linker-control"));
    }

    #[test]
    fn completion_matches_exact_lr_oracle_over_exhaustive_small_stack_population() {
        // Closure states are chosen acyclic; all top/source/suffix combinations
        // are tested, including unreachable and malformed-depth stacks. The
        // comparison must therefore not rely on a canonical-LR row-presence
        // claim about reachable parser stacks only.
        let rows = [
            vec![], vec![(EOF, Action::Reduce(0, 1))], vec![(EOF, Action::Reduce(1, 1))],
            vec![(EOF, Action::Accept)],
            vec![(EOF, Action::Split { shift: Some((3, false)), reduces: vec![(0, 1)], accept: false })],
        ];
        for replace in [false, true] {
            let gotos = [vec![(0, (3, replace)), (1, (1, false))], vec![(0, (3, false))], vec![], vec![], vec![]];
            let mut table = build_test_table(5, 1, &rows.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                &gotos.iter().map(Vec::as_slice).collect::<Vec<_>>());
            table.admission_policy = AdmissionPolicy::ExactSimulation;
            let split = compile_completion_template(&table).unwrap();
            let domain = TemplateDomain::compile(&split).unwrap();
            let mut all = vec![Vec::<u32>::new()]; let mut layer = vec![Vec::<u32>::new()];
            for _ in 0..4 {
                let mut next = Vec::new();
                for stack in layer { for top in 0..5 { let mut s = stack.clone(); s.push(top); next.push(s); } }
                all.extend(next.iter().cloned()); layer = next;
            }
            for stack in all {
                let gss = ParserGSS::from_single_stack(stack.clone(), TerminalsDisallowed::new());
                assert_eq!(domain.matches_top_first(stack.iter().rev().copied()), stacks_finished(&table, &gss),
                    "replace={replace} bottom_first={stack:?}");
            }
        }
    }
}
