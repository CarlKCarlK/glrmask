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
use crate::runtime::commit::template_advance::advance_with_prepared_template;
use crate::runtime::FastCommitTemplateDfas;
use super::{Constraint, ParserTableStorage, TemplateParser,
    TemplateDomain, TopAdmission, matches_gss};

/// The first `control_start` programs are exact leaf-local terminal aliases.
/// The remainder are zero-width controls. IDs are relative to the outer
/// terminal count, so outer aliases and exact leaf identities cannot collide.
#[derive(Debug)]
pub(crate) struct TemplateComposition {
    pub(crate) programs: TemplateDfasByTerminal,
    pub(crate) control_start: u32,
    outer_programs: TemplateDfasByTerminal,
    outer_fast: Vec<Arc<FastCommitTemplateDfas>>,
    domains: Vec<TemplateDomain>,
    fast: Vec<Arc<FastCommitTemplateDfas>>,
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
        let mut outer_fast = Vec::with_capacity(outer_programs.len());
        for program in outer_programs {
            let program = program.as_deref().ok_or_else(|| crate::Error::Compilation(
                "missing outer composition program".into()))?;
            outer_fast.push(Arc::new(FastCommitTemplateDfas::from_template(program)));
        }
        let mut domains = Vec::with_capacity(programs.len());
        let mut fast = Vec::with_capacity(programs.len());
        for (index, program) in programs.iter().enumerate() {
            let program = program.as_deref().ok_or_else(|| crate::Error::Compilation(
                format!("missing scoped composition program {index}")))?;
            domains.push(super::compile_domain(program)?);
            fast.push(Arc::new(FastCommitTemplateDfas::from_template(program)));
        }
        let controls_by_top = (0..state_count).map(|top| {
            domains.iter().enumerate().skip(control_start as usize)
                .filter_map(|(index, domain)| (domain.classify_top(top) != TopAdmission::Never)
                    .then_some(index as u32)).collect()
        }).collect();
        Ok(Self { programs, control_start, outer_programs: outer_programs.clone(),
            outer_fast, domains, fast, controls_by_top })
    }

    pub(crate) fn has_controls(&self) -> bool {
        (self.control_start as usize) < self.programs.len()
    }

    pub(crate) fn admits(&self, stack: &ParserGSS, index: u32) -> bool {
        self.domains.get(index as usize).is_some_and(|domain| matches_gss(domain, stack))
    }

    fn advance(&self, stack: &ParserGSS, index: u32) -> ParserGSS {
        let Some(template) = self.programs.get(index as usize).and_then(Option::as_deref) else {
            return ParserGSS::empty();
        };
        advance_with_prepared_template(template, stack.clone(),
            self.fast.get(index as usize).map(Arc::as_ref))
    }

    pub(crate) fn report(&self) -> serde_json::Value {
        serde_json::json!({
            "scoped_terminal_relations": self.control_start,
            "control_relations": self.programs.len() - self.control_start as usize,
            "extra_domain_states": self.domains.iter().map(TemplateDomain::state_count).sum::<usize>(),
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
            let template = self.composition.outer_programs[terminal as usize].as_deref()
                .expect("validated outer composition template");
            advance_with_prepared_template(template, stack.clone(),
                self.composition.outer_fast.get(terminal as usize).map(Arc::as_ref))
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
