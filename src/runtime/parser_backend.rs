//! Parser-independent runtime seam for finite acyclic stack transducers.
//!
//! The built-in frontend may use LR construction to produce this data. Once
//! installed, however, the LR table is physically absent. Accidental accesses
//! through an unported helper panic instead of silently taking a table fallback.
//! Tokenizer execution, GSS ownership, delayed exclusions and mask generation
//! remain in their existing shared implementations.

pub(crate) mod wire;

use std::collections::{BTreeMap, BTreeSet};
use std::ops::{ControlFlow, Deref, DerefMut};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use glrmask_parser_dwa::__private::templates::admissibility::{DomainProbe, TemplateDomain, TopAdmission};
use glrmask_parser_dwa::__private::templates::characterize::characterize_selected_terminals_for_terminal_count;
use glrmask_parser_dwa::__private::templates::compile_dfa::{
    Templates, specialize_template_dfa_defaults_for_commit_split_input, try_split_commit_template_dfas,
};
use crate::automata::unweighted_u32::dfa::DFA;
use crate::compiler::glr::analysis::EOF;
use crate::compiler::glr::labels::encode_negative_label;
use crate::compiler::glr::parser::ParserGSS;
use crate::compiler::glr::table::{AdmissionPolicy, GLRTable};
use crate::ds::bitset::BitSet;
use crate::grammar::flat::{DirectRegularAutomaton, TerminalID};
use super::{CommitTemplateDfas, Constraint};

/// Compile-time / ordinary-LR storage. None really means no table remains.
/// Deref is a migration guard, not a template-mode fallback implementation.
#[derive(Debug, Clone)]
pub(crate) struct ParserTableStorage(Option<GLRTable>);

impl ParserTableStorage {
    pub(crate) fn absent() -> Self { Self(None) }
    pub(crate) fn is_present(&self) -> bool { self.0.is_some() }
    pub(crate) fn as_lr(&self) -> Option<&GLRTable> { self.0.as_ref() }
    pub(crate) fn clone_lr(&self) -> GLRTable { self.deref().clone() }
    pub(crate) fn into_lr(self) -> GLRTable {
        self.0.expect("LR table requested from a template-only parser; this operation is not supported")
    }
}
impl From<GLRTable> for ParserTableStorage {
    fn from(table: GLRTable) -> Self { Self(Some(table)) }
}
impl Deref for ParserTableStorage {
    type Target = GLRTable;
    #[track_caller]
    fn deref(&self) -> &GLRTable {
        self.0.as_ref().expect("LR TABLE ACCESS IN TEMPLATE-ONLY MODE: route this operation through the parser backend")
    }
}
impl DerefMut for ParserTableStorage {
    #[track_caller]
    fn deref_mut(&mut self) -> &mut GLRTable {
        self.0.as_mut().expect("LR TABLE MUTATION IN TEMPLATE-ONLY MODE")
    }
}

// Ordinary artifacts retain their existing wire layout. The template-only
// versioned section codec is implemented separately, never by inventing a
// dummy executable LR table in this adapter.
pub(crate) mod table_serde {
    use super::ParserTableStorage;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(table: &ParserTableStorage, serializer: S) -> Result<S::Ok, S::Error> {
        if crate::compiler::glr::table::artifact_serde::external_serde_enabled() {
            return 0u8.serialize(serializer);
        }
        match table.as_lr() {
            Some(table) => crate::compiler::glr::table::artifact_serde::serialize(table, serializer),
            None => Err(serde::ser::Error::custom("template-only table requires its dedicated artifact section")),
        }
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<ParserTableStorage, D::Error> {
        if crate::compiler::glr::table::artifact_serde::external_serde_enabled() {
            let marker = u8::deserialize(deserializer)?;
            if marker != 0 { return Err(serde::de::Error::custom("invalid external parser placeholder")); }
            // The real parser arrives in its dedicated section. An absent
            // placeholder cannot accidentally execute an empty LR table.
            return Ok(ParserTableStorage::absent());
        }
        crate::compiler::glr::table::artifact_serde::deserialize(deserializer).map(Into::into)
    }
}

#[derive(Debug)]
pub(crate) struct TemplateParser {
    pub(crate) state_count: u32,
    pub(crate) terminal_count: u32,
    pub(crate) skip_terminals: BTreeSet<TerminalID>,
    pub(crate) completion_template: Arc<CommitTemplateDfas>,
    domains: Vec<TemplateDomain>,
    completion: TemplateDomain,
    /// Possible is an upper bound over every lower stack suffix; unconditional
    /// is an exact certificate independent of the lower suffix. Both are
    /// derived from input-domain automata, not copied from LR rows.
    pub(crate) possible: Vec<BitSet>,
    pub(crate) unconditional: Vec<BitSet>,
    profile: bool,
    advances: AtomicU64,
    admissions: AtomicU64,
    completions: AtomicU64,
}

fn compile_domain(template: &CommitTemplateDfas) -> crate::Result<TemplateDomain> {
    TemplateDomain::compile(template).map_err(crate::Error::Compilation)
}

fn matches_gss(domain: &TemplateDomain, stack: &ParserGSS) -> bool {
    if stack.is_empty() { return false; }
    let cursor = match domain.start() {
        DomainProbe::Accept => return true,
        DomainProbe::Reject => return false,
        DomainProbe::NeedMore(cursor) => cursor,
    };
    stack.any_prefix_matching(cursor, |cursor, top| match domain.step(cursor, *top) {
        DomainProbe::Accept => ControlFlow::Break(true),
        DomainProbe::Reject => ControlFlow::Break(false),
        DomainProbe::NeedMore(cursor) => ControlFlow::Continue(cursor),
    })
}

impl TemplateParser {
    fn compile(
        state_count: u32,
        terminal_count: u32,
        skip_terminals: BTreeSet<TerminalID>,
        templates: &[Option<Arc<CommitTemplateDfas>>],
        completion_template: CommitTemplateDfas,
    ) -> crate::Result<Self> {
        if templates.len() != terminal_count as usize {
            return Err(crate::Error::Compilation("template parser must provide every terminal relation, including explicit empty relations".into()));
        }
        let mut domains = Vec::with_capacity(templates.len());
        for (terminal, template) in templates.iter().enumerate() {
            let template = template.as_deref().ok_or_else(|| crate::Error::Compilation(format!("missing template for terminal {terminal}")))?;
            domains.push(compile_domain(template)?);
        }
        let completion = compile_domain(&completion_template)?;
        let mut possible = Vec::with_capacity(state_count as usize);
        let mut unconditional = Vec::with_capacity(state_count as usize);
        for top in 0..state_count {
            let mut maybe = BitSet::new(terminal_count as usize + 1);
            let mut always = BitSet::new(terminal_count as usize + 1);
            for (terminal, domain) in domains.iter().chain(std::iter::once(&completion)).enumerate() {
                match domain.classify_top(top) {
                    TopAdmission::Never => {},
                    TopAdmission::Always => { maybe.set(terminal); always.set(terminal); },
                    TopAdmission::DependsOnSuffix => maybe.set(terminal),
                }
            }
            possible.push(maybe);
            unconditional.push(always);
        }
        let profile = std::env::var_os("GLRMASK_PROFILE_TEMPLATE_BACKEND").is_some();
        Ok(Self { state_count, terminal_count, skip_terminals,
            completion_template: Arc::new(completion_template), domains, completion,
            possible, unconditional, profile,
            advances: AtomicU64::new(0), admissions: AtomicU64::new(0), completions: AtomicU64::new(0),
        })
    }

    #[inline]
    pub(crate) fn record_advance(&self) {
        if self.profile { self.advances.fetch_add(1, Ordering::Relaxed); }
    }

    #[inline]
    pub(crate) fn admits(&self, stack: &ParserGSS, terminal: TerminalID) -> bool {
        if self.profile { self.admissions.fetch_add(1, Ordering::Relaxed); }
        let (bit, domain) = if terminal == EOF {
            (self.terminal_count as usize, &self.completion)
        } else {
            let Some(domain) = self.domains.get(terminal as usize) else { return false; };
            (terminal as usize, domain)
        };
        if let Some(top) = stack.single_exclusive_top_value() {
            if self.unconditional.get(top as usize).is_some_and(|row| row.contains(bit)) { return true; }
            if self.possible.get(top as usize).is_some_and(|row| !row.contains(bit)) { return false; }
        }
        matches_gss(domain, stack)
    }

    pub(crate) fn admits_any(&self, stack: &ParserGSS, candidates: &BitSet) -> bool {
        for bit in candidates.iter() {
            let terminal = if bit == self.terminal_count as usize { EOF } else { bit as u32 };
            if self.admits(stack, terminal) { return true; }
        }
        false
    }

    pub(crate) fn admitted(&self, stack: &ParserGSS, candidates: &BitSet) -> BitSet {
        let mut result = BitSet::new(candidates.len());
        for bit in candidates.iter() {
            let terminal = if bit == self.terminal_count as usize { EOF } else { bit as u32 };
            if self.admits(stack, terminal) { result.set(bit); }
        }
        result
    }

    pub(crate) fn finished(&self, stack: &ParserGSS) -> bool {
        if self.profile { self.completions.fetch_add(1, Ordering::Relaxed); }
        self.admits(stack, EOF)
    }

    pub(crate) fn report(&self) -> serde_json::Value {
        serde_json::json!({
            "backend":"acyclic-template-dfa", "lr_table_present":false,
            "terminal_count":self.terminal_count, "stack_symbol_count":self.state_count,
            "domain_states":self.domains.iter().map(TemplateDomain::state_count).sum::<usize>(),
            "domain_edges":self.domains.iter().map(TemplateDomain::edge_count).sum::<usize>(),
            "domain_heap_payload_bytes":self.domains.iter().map(TemplateDomain::heap_payload_bytes).sum::<usize>(),
            "completion_domain_states":self.completion.state_count(),
            "counters_enabled":self.profile,
            "template_advances":self.advances.load(Ordering::Relaxed),
            "template_admission_queries":self.admissions.load(Ordering::Relaxed),
            "template_completion_queries":self.completions.load(Ordering::Relaxed),
        })
    }
}

/// Convert the pre-existing sparse regular frontend directly to depth-one
/// stack relations. Logical state 0 denotes the initial epsilon frontier;
/// raw automaton state i uses logical stack label i+1. This is compilation,
/// not a runtime dependence on the original frontend automaton.
fn sparse_regular_templates(
    automaton: &DirectRegularAutomaton,
    terminal_count: u32,
) -> crate::Result<(Vec<Option<Arc<CommitTemplateDfas>>>, CommitTemplateDfas, u32)> {
    let state_count = u32::try_from(automaton.states.len() + 1).map_err(|_| crate::Error::Compilation("regular stack alphabet too large".into()))?;
    let mut by_terminal = vec![BTreeMap::<u32, Vec<u32>>::new(); terminal_count as usize];
    let mut finished_tops = Vec::new();
    for top in 0..state_count {
        let mut pending = if top == 0 { automaton.start_states.clone() } else { vec![top - 1] };
        let mut seen = BitSet::new(automaton.states.len());
        let mut targets = BTreeMap::<u32, BTreeSet<u32>>::new();
        let mut finished = false;
        while let Some(raw) = pending.pop() {
            if seen.contains(raw as usize) { continue; }
            let state = automaton.states.get(raw as usize).ok_or_else(|| crate::Error::Compilation("invalid sparse regular state".into()))?;
            seen.set(raw as usize);
            finished |= state.is_accepting;
            for (&terminal, destinations) in &state.transitions {
                if terminal >= terminal_count { continue; }
                targets.entry(terminal).or_default().extend(destinations.iter().map(|raw| raw + 1));
            }
            pending.extend(state.epsilons.iter().copied());
        }
        if finished { finished_tops.push(top); }
        for (terminal, targets) in targets { by_terminal[terminal as usize].insert(top, targets.into_iter().collect()); }
    }
    let mut templates = Vec::with_capacity(terminal_count as usize);
    for rows in by_terminal {
        let mut dfa = DFA::new();
        let accept = dfa.add_state(); dfa.set_accepting(accept, true);
        let mut middles = BTreeMap::<Vec<u32>, u32>::new();
        for (top, targets) in rows {
            let middle = if let Some(&id) = middles.get(&targets) { id } else {
                let id = dfa.add_state();
                for &target in &targets { dfa.add_transition(id, encode_negative_label(target), accept); }
                middles.insert(targets, id); id
            };
            dfa.add_transition(dfa.start_state, top as i32, middle);
        }
        let split = try_split_commit_template_dfas(&dfa).ok_or_else(|| crate::Error::Compilation("regular template did not satisfy the split phase contract".into()))?;
        templates.push(Some(Arc::new(split)));
    }
    let mut completion = DFA::new();
    let accept = completion.add_state(); completion.set_accepting(accept, true);
    for top in finished_tops { completion.add_transition(0, top as i32, accept); }
    let completion = try_split_commit_template_dfas(&completion).ok_or_else(|| crate::Error::Compilation("regular completion template failed phase contract".into()))?;
    Ok((templates, completion, state_count))
}

impl Constraint {
    #[inline]
    pub(crate) fn has_template_parser(&self) -> bool { self.template_parser.is_some() }
    #[inline]
    pub(crate) fn parser_symbol_count(&self) -> u32 {
        self.template_parser.as_ref().map_or_else(|| self.table.num_states, |parser| parser.state_count)
    }
    #[inline]
    pub(crate) fn parser_terminal_count(&self) -> u32 {
        self.template_parser.as_ref().map_or_else(|| self.table.num_terminals, |parser| parser.terminal_count)
    }
    #[inline]
    pub(crate) fn parser_has_controls(&self) -> bool {
        self.template_parser.is_none() && !self.table.control_terminals.is_empty()
    }
    #[inline]
    pub(crate) fn parser_skip_terminals(&self) -> &BTreeSet<TerminalID> {
        self.template_parser.as_ref().map_or_else(|| &self.table.skip_terminals, |parser| &parser.skip_terminals)
    }
    #[inline]
    pub(crate) fn parser_admission_policy(&self) -> AdmissionPolicy {
        if self.has_template_parser() { AdmissionPolicy::ExactSimulation } else { self.table.admission_policy }
    }
    #[inline]
    pub(crate) fn parser_advance_row(&self, top: u32) -> Option<&BitSet> {
        self.template_parser.as_ref().map_or_else(|| self.table.advance_row(top), |parser| parser.possible.get(top as usize))
    }
    #[inline]
    pub(crate) fn parser_advance_row_allows(&self, top: u32, terminal: u32) -> bool {
        if let Some(parser) = &self.template_parser {
            let bit = if terminal == EOF { parser.terminal_count as usize } else { terminal as usize };
            parser.possible.get(top as usize).is_some_and(|row| row.contains(bit))
        } else { self.table.advance_row_allows(top, terminal) }
    }
    #[inline]
    pub(crate) fn parser_advance_row_intersects(&self, top: u32, terminals: &BitSet) -> bool {
        if let Some(parser) = &self.template_parser {
            parser.possible.get(top as usize).is_some_and(|row| row.words().iter().zip(terminals.words()).any(|(a,b)| a & b != 0))
        } else { self.table.advance_row_intersects(top, terminals) }
    }
    #[inline]
    pub(crate) fn parser_unconditional_row(&self, top: u32) -> Option<&BitSet> {
        self.template_parser.as_ref().map_or_else(|| self.table.unconditional_advance_row(top), |parser| parser.unconditional.get(top as usize))
    }

    pub(crate) fn install_template_parser(&mut self) -> crate::Result<()> {
        if self.has_template_parser() { return Ok(()); }
        if self.parser_has_controls() || self.uses_compact_segmented_parser_runtime() || !self.late_grammar_slots.is_empty() {
            return Err(crate::Error::Compilation("template-only parser composition/control closure is not implemented; no LR fallback is permitted".into()));
        }
        let terminal_count = self.table.num_terminals;
        let (templates, completion, state_count) = if self.uses_sparse_direct_regular_runtime() {
            sparse_regular_templates(self.direct_regular_automaton.as_ref().unwrap(), terminal_count)?
        } else {
            let selected = vec![true; terminal_count as usize];
            let characterizations = characterize_selected_terminals_for_terminal_count(&self.table, terminal_count, &selected);
            let raw = Templates::from_characterizations(&characterizations);
            let mut templates = vec![None; terminal_count as usize];
            for (terminal, dfa) in raw.by_terminal {
                let dfa = specialize_template_dfa_defaults_for_commit_split_input(&dfa);
                let split = try_split_commit_template_dfas(&dfa).ok_or_else(|| crate::Error::Compilation(format!("terminal {terminal} template is not acyclic pop/read/push; refusing table fallback")))?;
                templates[terminal as usize] = Some(Arc::new(split));
            }
            let completion = glrmask_parser_dwa::__private::templates::completion::compile_completion_template(&self.table)
                .map_err(crate::Error::Compilation)?;
            (templates, completion, self.table.num_states)
        };
        let parser = TemplateParser::compile(state_count, terminal_count, self.table.skip_terminals.clone(), &templates, completion)?;
        self.template_dfas_by_terminal = templates;
        self.fast_template_dfas_by_terminal = self.compute_fast_template_dfas();
        self.serialized_artifact_cache = None;
        self.deferred_table_rules_blob = None;
        self.deferred_table_rules = Default::default();
        self.template_parser = Some(Arc::new(parser));
        // Drop, do not merely clear rows or change an environment flag.
        self.table = ParserTableStorage::absent();
        Ok(())
    }

    pub(crate) fn parser_backend_report(&self) -> serde_json::Value {
        match &self.template_parser {
            Some(parser) => {
                assert!(!self.table.is_present(), "template-only constraint retained an LR table");
                parser.report()
            }
            None => serde_json::json!({"backend":"lr-table", "lr_table_present":self.table.is_present()}),
        }
    }
}
