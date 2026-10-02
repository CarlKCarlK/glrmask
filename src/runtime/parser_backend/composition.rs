//! Finite parser relations in the recursive composition stack coordinate.
//!
//! A consuming terminal and each zero-width CALL/RETURN are compiled separately.
//! The existing provider closure owns their sequencing, lexer-scope routing and
//! GSS annotation merges; this module introduces no second masking engine.

use std::sync::Arc;
use smallvec::SmallVec;

use crate::compiler::glr::analysis::EOF;
use crate::compiler::glr::parser::{ParserActionProvider, ParserGSS, ProvidedAction};
use crate::compiler::glr::table::GLRTable;
use crate::runtime::artifact::TemplateDfasByTerminal;
use super::scoped_program::ScopedProgram;
use super::{Constraint, ParserTableStorage, TemplateParser,
    TemplateDomain, TopAdmission};
#[cfg(test)]
use super::matches_gss;

/// The first `control_start` programs are exact leaf-local terminal aliases.
/// The remainder are zero-width controls. IDs are relative to the outer
/// terminal count, so outer aliases and exact leaf identities cannot collide.
#[derive(Debug)]
pub(crate) struct TemplateComposition {
    pub(crate) programs: TemplateDfasByTerminal,
    pub(crate) control_start: u32,
    pub(crate) views: Vec<ScopedProgram>,
    pub(crate) outer_views: Vec<ScopedProgram>,
    pub(crate) completion_view: Option<ScopedProgram>,
    domains: Vec<Arc<TemplateDomain>>,
    controls_by_top: Vec<SmallVec<[u32; 4]>>,
}

impl TemplateComposition {
    pub(crate) fn compile(
        state_count: u32,
        control_start: u32,
        outer_programs: &TemplateDfasByTerminal,
        programs: TemplateDfasByTerminal,
    ) -> crate::Result<Self> {
        if control_start as usize > programs.len() {
            return Err(crate::Error::Compilation("composition control offset exceeds its program inventory".into()));
        }
        if state_count as u64 * ((programs.len() - control_start as usize) as u64 * 4 + 24)
            > 256 * 1024 * 1024
        {
            return Err(crate::Error::Compilation("composition control certificates exceed the resource budget".into()));
        }
        let prepare = |programs: &TemplateDfasByTerminal| programs.iter().map(|program| {
            let mut view = ScopedProgram::prepare(Arc::clone(program.as_ref().ok_or_else(||
                crate::Error::Compilation("missing composition program".into()))?), state_count)?;
            view.guard_owner = false;
            Ok(view)
        }).collect::<crate::Result<Vec<_>>>();
        Self::from_views(state_count, control_start, prepare(outer_programs)?, prepare(&programs)?, None)
    }

    pub(crate) fn from_views(state_count: u32, control_start: u32,
        outer_views: Vec<ScopedProgram>, views: Vec<ScopedProgram>,
        completion_view: Option<ScopedProgram>) -> crate::Result<Self> {
        if control_start as usize > views.len() {
            return Err(crate::Error::Compilation("invalid composition control offset".into()));
        }
        if state_count as u64 * ((views.len() - control_start as usize) as u64 * 4 + 24)
            > 256 * 1024 * 1024 {
            return Err(crate::Error::Compilation("composition control certificates exceed resource budget".into()));
        }
        for view in views.iter().chain(&outer_views).chain(completion_view.iter()) {
            view.validate_coordinate(state_count)?;
        }
        let controls_by_top = (0..state_count).map(|top| {
            views.iter().enumerate().skip(control_start as usize)
                .filter_map(|(index, view)| (view.classify_top(top) != TopAdmission::Never)
                    .then_some(index as u32)).collect()
        }).collect();
        let programs = views.iter().map(|view| Some(Arc::clone(&view.source))).collect();
        let domains = views.iter().map(|view| Arc::clone(&view.domain)).collect();
        Ok(Self { programs, control_start, views, outer_views, completion_view, domains, controls_by_top })
    }

    pub(crate) fn has_controls(&self) -> bool {
        (self.control_start as usize) < self.programs.len()
    }

    pub(crate) fn admits(&self, stack: &ParserGSS, index: u32) -> bool {
        self.views.get(index as usize).is_some_and(|view| {
            if let Some(top) = stack.single_exclusive_top_value() {
                match view.classify_top(top) {
                    TopAdmission::Always => return true,
                    TopAdmission::Never => return false,
                    TopAdmission::DependsOnSuffix => {}
                }
            }
            view.admits(stack)
        })
    }

    pub(crate) fn admits_outer(&self, stack: &ParserGSS, terminal: u32) -> Option<bool> {
        if terminal == EOF { self.completion_view.as_ref().map(|view| view.admits(stack)) }
        else { self.outer_views.get(terminal as usize).map(|view| view.admits(stack)) }
    }

    fn advance(&self, stack: &ParserGSS, index: u32) -> ParserGSS {
        self.views.get(index as usize).map_or_else(ParserGSS::empty, |view| view.advance(stack))
    }

    pub(crate) fn report(&self) -> serde_json::Value {
        serde_json::json!({
            "scoped_terminal_relations": self.control_start,
            "control_relations": self.programs.len() - self.control_start as usize,
            "extra_domain_states": self.domains.iter().map(|domain| domain.state_count()).sum::<usize>(),
        })
    }
}

pub(crate) struct TemplateCompositionProvider<'a> {
    parser: &'a TemplateParser,
    composition: &'a TemplateComposition,
}

impl ParserActionProvider for TemplateCompositionProvider<'_> {
    type Symbol = u32;

    fn action(&self, _state: u32, _symbol: u32) -> Option<ProvidedAction<'_>> {
        unreachable!("template composition must not query an LR action")
    }
    fn scope_state(&self, _scope: u32, _local_state: u32) -> Option<u32> {
        unreachable!("template composition already uses the scoped stack coordinate")
    }
    fn goto_target(&self, _scope: u32, _goto_from: u32, _nonterminal: u32) -> Option<(u32, bool)> {
        unreachable!("template composition must not query an LR goto")
    }
    fn state_count_hint(&self) -> usize { self.parser.state_count as usize }

    fn control_symbols(&self, top: u32, out: &mut SmallVec<[u32; 4]>) {
        if let Some(controls) = self.composition.controls_by_top.get(top as usize) {
            out.extend(controls.iter().map(|index| self.parser.terminal_count + index));
        }
    }

    fn advance_relation(&self, stack: &ParserGSS, terminal: u32) -> Option<ParserGSS> {
        self.parser.record_advance();
        Some(if terminal < self.parser.terminal_count {
            self.composition.outer_views[terminal as usize].advance(stack)
        } else if terminal != EOF {
            self.composition.advance(stack, terminal - self.parser.terminal_count)
        } else {
            ParserGSS::empty()
        })
    }

    fn relation_admits(&self, stack: &ParserGSS, terminal: u32) -> Option<bool> {
        Some(terminal != EOF && self.parser.admits_control_closed(stack, terminal))
    }

    fn relation_finished(&self, stack: &ParserGSS, terminal: u32) -> Option<bool> {
        Some(terminal == EOF && self.parser.admits_control_closed(stack, EOF))
    }
}

#[cfg(test)]
mod prepared_relation_tests {
    use super::*;
    use crate::automata::unweighted_u32::dfa::DFA;
    use crate::compiler::glr::accumulator::TerminalsDisallowed;
    use crate::compiler::glr::labels::encode_negative_label;
    use crate::runtime::CommitTemplateDfas;
    use crate::runtime::commit::template_advance::{
        advance_with_prepared_template, PREPARED_RELATION_SHORTCUTS,
    };

    fn rejecting() -> CommitTemplateDfas {
        CommitTemplateDfas {
            pop: DFA::new(), read: DFA::new(), push: DFA::new(),
            pop_to_read: vec![None], pop_to_push: vec![None], read_to_push: vec![None],
        }
    }

    fn shift() -> CommitTemplateDfas {
        let mut t = rejecting();
        let read = t.read.add_state();
        t.read.add_transition(0, 5, read);
        let end = t.push.add_state();
        t.push.add_transition(0, encode_negative_label(7), end);
        t.push.set_accepting(end, true);
        t.pop_to_read[0] = Some(0);
        t.read_to_push = vec![None, Some(0)];
        t
    }

    fn suffix_dependent() -> CommitTemplateDfas {
        let mut t = rejecting();
        let first = t.pop.add_state();
        let second = t.pop.add_state();
        t.pop.add_transition(0, 5, first);
        t.pop.add_transition(first, 4, second);
        t.pop.set_accepting(second, true);
        t.pop_to_read.resize(3, None);
        t.pop_to_push.resize(3, None);
        t
    }

    fn inputs() -> Vec<ParserGSS> {
        vec![
            ParserGSS::empty(),
            ParserGSS::from_single_stack(vec![], TerminalsDisallowed::new()),
            ParserGSS::from_single_stack(vec![0, 5], TerminalsDisallowed::new()),
            ParserGSS::from_single_stack(vec![0, 4, 5], TerminalsDisallowed::new().with_insert(1, 3)),
            ParserGSS::from_single_stack(vec![0, 3, 5], TerminalsDisallowed::new()),
            ParserGSS::from_single_stack(vec![0, 6], TerminalsDisallowed::new()),
            ParserGSS::from_stacks(&[
                (vec![0, 4, 5], TerminalsDisallowed::new()),
                (vec![0, 3, 5], TerminalsDisallowed::new().with_insert(2, 4)),
                (vec![0, 6], TerminalsDisallowed::new().with_insert(0, 1)),
            ]),
        ]
    }

    #[test]
    fn large_scoped_alphabet_uses_exact_local_admission_and_completion() {
        // A scoped parser can have a large global state/terminal product
        // without requiring any global dense certificate. Exercise admission
        // through the actual provider, including its private completion callback.
        let symbols = 4_001;
        let terminals = 4_000;
        assert!(u64::from(symbols) * (u64::from(terminals) + 1) > 16_000_000);
        let view = ScopedProgram::prepare(Arc::new(shift()), symbols).unwrap();
        let composition = TemplateComposition::from_views(symbols, terminals,
            vec![view.clone(); terminals as usize], vec![view.clone(); terminals as usize],
            Some(view.clone())).unwrap();
        let parser = TemplateParser::from_composition(symbols, terminals, composition).unwrap();
        assert!(parser.possible.is_empty() && parser.unconditional.is_empty(),
            "scoped queries must not materialize the unused global Cartesian product");
        let mut candidates = crate::ds::bitset::BitSet::new(terminals as usize + 1);
        candidates.set(0); candidates.set(terminals as usize - 1);
        for stack in inputs() {
            // This program reads exactly top symbol 5 and pushes 7. Its input
            // language has no dependence on the lower concrete stack suffix.
            let expected = stack.peek_values().contains(&5);
            assert_eq!(parser.admits(&stack, 0), expected);
            assert_eq!(parser.admits(&stack, terminals - 1), expected);
            assert_eq!(parser.admits_any(&stack, &candidates), expected);
            let admitted = parser.admitted(&stack, &candidates);
            assert_eq!(admitted.contains(0), expected);
            assert_eq!(admitted.contains(terminals as usize - 1), expected);
            assert_eq!(parser.finished(&stack), expected);
        }
    }

    #[test]
    fn composition_scoped_and_outer_advance_use_the_shared_prepared_shift() {
        let outer = vec![Some(Arc::new(shift()))];
        let scoped = vec![Some(Arc::new(shift())), Some(Arc::new(suffix_dependent()))];
        let mut parser = TemplateParser::compile(8, 1, Default::default(), &outer, rejecting()).unwrap();
        parser.composition = Some(Arc::new(TemplateComposition::compile(8, 2, &outer, scoped).unwrap()));
        let provider = parser.composition_provider().unwrap();
        for input in inputs() {
            let before = input.to_stacks(128).unwrap();
            for terminal in [0, 1, 2] {
                let program = if terminal == 0 { outer[0].as_deref().unwrap() }
                    else { parser.composition.as_ref().unwrap().programs[(terminal-1) as usize].as_deref().unwrap() };
                let expected = advance_with_prepared_template(program, input.clone(), None);
                let actual = provider.advance_relation(&input, terminal).unwrap();
                assert_eq!(actual.semantically_eq(&expected, 65536), Some(true));
                assert_eq!(before, input.to_stacks(128).unwrap());
            }
        }
        let input = ParserGSS::from_single_stack(vec![0, 5], TerminalsDisallowed::new());
        for terminal in [0, 1] {
            PREPARED_RELATION_SHORTCUTS.with(|count| count.set(0));
            let actual = provider.advance_relation(&input, terminal).unwrap();
            assert_eq!(PREPARED_RELATION_SHORTCUTS.with(|count| count.get()), 1,
                "both outer and scoped composition must reach the common fast path");
            assert_eq!(actual.semantically_eq(&input.push(7), 65536), Some(true));
        }
    }

    #[test]
    fn composition_scoped_admission_preserves_suffix_dependent_and_empty_cases() {
        let outer = vec![Some(Arc::new(shift()))];
        let scoped = vec![Some(Arc::new(shift())), Some(Arc::new(suffix_dependent())),
                          Some(Arc::new(rejecting()))];
        let composition = TemplateComposition::compile(8, 2, &outer, scoped).unwrap();
        assert_eq!(composition.domains[0].classify_top(5), TopAdmission::Always);
        assert_eq!(composition.domains[0].classify_top(6), TopAdmission::Never);
        assert_eq!(composition.domains[1].classify_top(5), TopAdmission::DependsOnSuffix);
        for input in inputs() {
            for (index, domain) in composition.domains.iter().enumerate() {
                assert_eq!(composition.admits(&input, index as u32), matches_gss(domain, &input));
            }
            assert!(!composition.admits(&input, u32::MAX));
        }
    }
}

fn compile_table_relations(table: &GLRTable) -> crate::Result<TemplateDfasByTerminal> {
    use glrmask_parser_dwa::__private::templates::characterize::try_characterize_selected_terminals_for_terminal_count;
    use glrmask_parser_dwa::__private::templates::compile_dfa::{Templates,
        specialize_template_dfa_defaults_for_commit_split_input, try_split_commit_template_dfas};
    let characterizations = try_characterize_selected_terminals_for_terminal_count(
        table, table.num_terminals, &vec![true; table.num_terminals as usize])
        .map_err(crate::Error::Compilation)?;
    let mut programs = vec![None; table.num_terminals as usize];
    for (terminal, raw) in Templates::from_characterizations(&characterizations).by_terminal {
        let raw = specialize_template_dfa_defaults_for_commit_split_input(&raw);
        let split = try_split_commit_template_dfas(&raw).ok_or_else(|| crate::Error::Compilation(
            format!("composition relation {terminal} is not a finite POP/READ/PUSH program")))?;
        programs[terminal as usize] = Some(Arc::new(split));
    }
    Ok(programs)
}

impl TemplateParser {
    pub(crate) fn composition_provider(&self) -> Option<TemplateCompositionProvider<'_>> {
        let composition = self.composition.as_deref()?;
        Some(TemplateCompositionProvider { parser: self, composition })
    }
}

impl Constraint {
    /// Validate serialized namespace metadata using template dimensions alone.
    /// Every child is fully loaded before its containing core reaches this check.
    pub(crate) fn validate_template_composition_layout(&self) -> Result<(), String> {
        let parser = self.template_parser.as_ref().ok_or("missing template parser")?;
        let Some(composition) = &parser.composition else {
            if self.static_dynamic_overlay.is_some() {
                return Err("standalone template parser has unsupported composition machinery".into());
            }
            return Ok(());
        };
        let layout = self.recursive_parser_layout()?.ok_or("missing template composition layout")?;
        if layout.total_states != parser.state_count
            || layout.outer_terminal_count != parser.terminal_count
            || layout.total_leaf_terminals != composition.control_start
        {
            return Err("template composition programs use a different parser coordinate".into());
        }
        let overlay = self.static_dynamic_overlay.as_ref().ok_or("missing composition overlay")?;
        if overlay.recursive_compiler_table.get().is_some() {
            return Err("template composition must not retain a packed LR compiler table".into());
        }
        for component in &overlay.segmented_parser_components {
            if component.constraint.table.is_present() || !component.constraint.has_template_parser() {
                return Err("template composition contains an LR-backed component".into());
            }
            component.constraint.validate_template_composition_layout()?;
        }
        Ok(())
    }

    pub(crate) fn template_composition_provider(&self) -> Option<TemplateCompositionProvider<'_>> {
        self.template_parser.as_deref()?.composition_provider()
    }

    pub(crate) fn install_composition_template_parser(&mut self) -> crate::Result<()> {
        let layout = self.recursive_parser_layout().map_err(crate::Error::Compilation)?
            .ok_or_else(|| crate::Error::Compilation("missing recursive composition layout".into()))?;
        let outer_count = self.parser_terminal_count();
        let mut table = self.recursive_explicit_control_parser_table()
            .map_err(crate::Error::Compilation)?;
        let ordinary_count = outer_count.checked_add(layout.total_leaf_terminals)
            .ok_or_else(|| crate::Error::Compilation("composition terminal coordinate overflow".into()))?;
        if table.num_states != layout.total_states || table.num_terminals < ordinary_count {
            return Err(crate::Error::Compilation("composition projection changed the runtime coordinate".into()));
        }
        // Controls are ordinary labels for this one-step compilation. Runtime
        // closure, not control elimination, supplies their zero-width meaning.
        table.control_terminals.clear();
        let mut programs = compile_table_relations(&table)?;
        let extra = programs.split_off(outer_count as usize);
        let completion = glrmask_parser_dwa::__private::templates::completion::compile_completion_template(&table)
            .map_err(crate::Error::Compilation)?;
        let mut parser = TemplateParser::compile(layout.total_states, outer_count,
            table.skip_terminals.iter().copied().filter(|terminal| *terminal < outer_count).collect(),
            &programs, completion)?;
        parser.embedding = super::embedding::TemplateEmbedding::from_table(&table,
            self.composition_start_nullable().map_err(crate::Error::Compilation)?,
            self.composition_child_return_pop().map_err(crate::Error::Compilation)?,
            self.late_grammar_slots.iter().map(|slot| slot.terminal_id)).ok().map(Arc::new);
        parser.composition = Some(Arc::new(TemplateComposition::compile(
            layout.total_states, layout.total_leaf_terminals, &programs, extra)?));

        let overlay = self.static_dynamic_overlay.as_mut().expect("validated composition overlay");
        for component in &mut overlay.segmented_parser_components {
            let constraint = Arc::make_mut(&mut component.constraint);
            constraint.install_template_parser_in_coordinate(true)?;
        }
        // The packed flattened LR copy was only used for the old relinker. A
        // template artifact must not retain executable LR data in a side field.
        overlay.recursive_compiler_table = Default::default();
        self.template_dfas_by_terminal = programs;
        self.fast_template_dfas_by_terminal = self.compute_fast_template_dfas();
        self.template_parser = Some(Arc::new(parser));
        self.table = ParserTableStorage::absent();
        self.deferred_table_rules_blob = None;
        self.deferred_table_rules = Default::default();
        self.serialized_artifact_cache = None;
        self.validate_template_composition_layout().map_err(crate::Error::Compilation)
    }
}
