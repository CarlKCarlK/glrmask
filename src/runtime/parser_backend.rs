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
use glrmask_parser_dwa::__private::templates::characterize::try_characterize_selected_terminals_for_terminal_count;
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

/// Enumerate only possible members of an existential query. In particular,
/// do not iterate an entire wide lexer-future set and rediscover the same
/// exclusive GSS top for every impossible terminal.
fn intersecting_bits<'a>(left: &'a BitSet, right: &'a BitSet) -> impl Iterator<Item = usize> + 'a {
    left.words().iter().zip(right.words()).enumerate().flat_map(|(index, (&a, &b))| {
        let mut word = a & b;
        std::iter::from_fn(move || {
            if word == 0 { return None; }
            let bit = index * 64 + word.trailing_zeros() as usize;
            word &= word - 1;
            Some(bit)
        })
    })
}

impl TemplateParser {
    pub(crate) fn compile(
        state_count: u32, terminal_count: u32, skip_terminals: BTreeSet<TerminalID>,
        templates: &[Option<Arc<CommitTemplateDfas>>], completion_template: CommitTemplateDfas,
    ) -> crate::Result<Self> {
        Self::compile_inner(state_count, terminal_count, skip_terminals, templates,
            completion_template, false).map(|(parser, _)| parser)
    }

    pub(crate) fn compile_with_runtime(
        state_count: u32, terminal_count: u32, skip_terminals: BTreeSet<TerminalID>,
        templates: &[Option<Arc<CommitTemplateDfas>>], completion_template: CommitTemplateDfas,
    ) -> crate::Result<(Self, crate::runtime::artifact::FastTemplateDfasByTerminal)> {
        Self::compile_inner(state_count, terminal_count, skip_terminals, templates,
            completion_template, true)
    }

    fn compile_inner(
        state_count: u32, terminal_count: u32, skip_terminals: BTreeSet<TerminalID>,
        templates: &[Option<Arc<CommitTemplateDfas>>], completion_template: CommitTemplateDfas,
        prepare_runtime: bool,
    ) -> crate::Result<(Self, crate::runtime::artifact::FastTemplateDfasByTerminal)> {
        if templates.len() != terminal_count as usize {
            return Err(crate::Error::Compilation("template parser must provide every terminal relation, including explicit empty relations".into()));
        }
        let mut domains = Vec::with_capacity(templates.len());
        let mut runtime = Vec::with_capacity(if prepare_runtime { templates.len() } else { 0 });
        for (terminal, template) in templates.iter().enumerate() {
            let template = template.as_deref().ok_or_else(|| crate::Error::Compilation(format!("missing template for terminal {terminal}")))?;
            let validated = super::commit::template_prepare::TemplatePreparation::new(template)
                .map_err(crate::Error::Compilation)?;
            validated.validate_alphabet(state_count).map_err(crate::Error::Compilation)?;
            domains.push(TemplateDomain::from_validated(&validated).map_err(crate::Error::Compilation)?);
            if prepare_runtime {
                runtime.push(Some(Arc::new(crate::runtime::artifact::FastCommitTemplateDfas::from_prepared(&validated))));
            }
        }
        let validated = super::commit::template_prepare::TemplatePreparation::new(&completion_template)
            .map_err(crate::Error::Compilation)?;
        validated.validate_alphabet(state_count).map_err(crate::Error::Compilation)?;
        let completion = TemplateDomain::from_validated(&validated).map_err(crate::Error::Compilation)?;
        let (possible, unconditional) = top_certificate_rows(state_count, &domains, &completion);
        let profile = std::env::var_os("GLRMASK_PROFILE_TEMPLATE_BACKEND").is_some();
        Ok((Self { state_count, terminal_count, skip_terminals,
            completion_template: Arc::new(completion_template), domains, completion,
            possible, unconditional, profile,
            advances: AtomicU64::new(0), admissions: AtomicU64::new(0), completions: AtomicU64::new(0),
        }, runtime))
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
        } else {
            let tops = stack.peek_values();
            let mut possible = tops.is_empty();
            for top in tops {
                if self.unconditional.get(top as usize).is_some_and(|row| row.contains(bit)) { return true; }
                possible |= self.possible.get(top as usize).is_none_or(|row| row.contains(bit));
            }
            if !possible { return false; }
        }
        matches_gss(domain, stack)
    }

    pub(crate) fn admits_any(&self, stack: &ParserGSS, candidates: &BitSet) -> bool {
        if let Some(top) = stack.single_exclusive_top_value()
            && let Some(possible) = self.possible.get(top as usize)
        {
            if self.profile { self.admissions.fetch_add(1, Ordering::Relaxed); }
            if self.unconditional.get(top as usize).is_some_and(|row|
                row.words().iter().zip(candidates.words()).any(|(a,b)| a & b != 0))
            { return true; }
            return intersecting_bits(possible, candidates).any(|bit| {
                let domain = if bit == self.terminal_count as usize { Some(&self.completion) }
                    else { self.domains.get(bit) };
                domain.is_some_and(|domain| matches_gss(domain, stack))
            });
        }
        // A top certificate was derived from the complete relation with its
        // lower suffix universally quantified. One live certified top is
        // therefore sufficient for an existential GSS query, even when the
        // graph contains many branches and correlated annotations.
        for top in stack.peek_values() {
            if self.unconditional.get(top as usize).is_some_and(|row|
                row.words().iter().zip(candidates.words()).any(|(a,b)| a & b != 0))
            {
                if self.profile { self.admissions.fetch_add(1, Ordering::Relaxed); }
                return true;
            }
        }
        for bit in candidates.iter() {
            let terminal = if bit == self.terminal_count as usize { EOF } else { bit as u32 };
            if self.admits(stack, terminal) { return true; }
        }
        false
    }

    pub(crate) fn admitted(&self, stack: &ParserGSS, candidates: &BitSet) -> BitSet {
        if self.profile { self.admissions.fetch_add(1, Ordering::Relaxed); }
        let mut result = BitSet::new(candidates.len());
        let mut possible = BitSet::new(candidates.len());
        let tops = stack.peek_values();
        if tops.is_empty() {
            // Prefix-accepting domains may accept an empty concrete stack.
            // An empty language is still rejected by matches_gss below.
            possible = candidates.clone();
        }
        for top in tops {
            match self.possible.get(top as usize) {
                Some(row) => for ((dst, a), b) in possible.words_mut().iter_mut()
                    .zip(row.words()).zip(candidates.words()) { *dst |= a & b; },
                None => possible = candidates.clone(),
            }
            if let Some(row) = self.unconditional.get(top as usize) {
                for ((dst, a), b) in result.words_mut().iter_mut()
                    .zip(row.words()).zip(candidates.words()) { *dst |= a & b; }
            }
        }
        // U(tops) is a proven subset and P(tops) a proven superset of exact
        // admission. Only P minus U needs a deeper input-domain traversal.
        // Both sets come from template-domain automata, never LR rows.
        for (word_index, &word) in possible.words().iter().enumerate() {
            let mut remaining = word & !result.words()[word_index];
            while remaining != 0 {
                let bit = word_index * 64 + remaining.trailing_zeros() as usize;
                remaining &= remaining - 1;
                let domain = if bit == self.terminal_count as usize {
                    Some(&self.completion)
                } else { self.domains.get(bit) };
                if domain.is_some_and(|domain| matches_gss(domain, stack)) { result.set(bit); }
            }
        }
        result
    }

    /// Input-only query for the shared allocation-free flat commit frontier.
    /// No output stack, GSS node, or LR action is materialized.
    pub(crate) fn admits_flat_any(&self, stack: &[u32], candidates: &BitSet) -> bool {
        if self.profile { self.admissions.fetch_add(1, Ordering::Relaxed); }
        let top = stack.last().copied();
        if top.and_then(|top| self.unconditional.get(top as usize)).is_some_and(|row|
            row.words().iter().zip(candidates.words()).any(|(a,b)| a & b != 0))
        { return true; }
        if let Some(possible) = top.and_then(|top| self.possible.get(top as usize)) {
            return intersecting_bits(possible, candidates).any(|bit| {
                let domain = if bit == self.terminal_count as usize { Some(&self.completion) }
                    else { self.domains.get(bit) };
                domain.is_some_and(|domain| domain.matches_top_first(stack.iter().rev().copied()))
            });
        }
        for bit in candidates.iter() {
            if top.and_then(|top| self.possible.get(top as usize)).is_some_and(|row| !row.contains(bit)) { continue; }
            let domain = if bit == self.terminal_count as usize { Some(&self.completion) } else { self.domains.get(bit) };
            if domain.is_some_and(|domain| domain.matches_top_first(stack.iter().rev().copied())) { return true; }
        }
        false
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

/// Build the exact same dense runtime rows from sparse domain-root partitions.
/// The dense product remains the runtime representation; only preparation is
/// changed. An implicit root transition is shared by all missing symbols, so
/// begin with its bitsets and patch the explicit certificate exceptions.
fn top_certificate_rows(
    state_count: u32,
    domains: &[TemplateDomain],
    completion: &TemplateDomain,
) -> (Vec<BitSet>, Vec<BitSet>) {
    let mut default_possible = BitSet::new(domains.len() + 1);
    let mut default_unconditional = BitSet::new(domains.len() + 1);
    for (terminal, domain) in domains.iter().chain(std::iter::once(completion)).enumerate() {
        match domain.top_admission_partition().0 {
            TopAdmission::Never => {},
            TopAdmission::Always => {
                default_possible.set(terminal);
                default_unconditional.set(terminal);
            },
            TopAdmission::DependsOnSuffix => default_possible.set(terminal),
        }
    }
    let mut possible = Vec::with_capacity(state_count as usize);
    let mut unconditional = Vec::with_capacity(state_count as usize);
    for _ in 0..state_count {
        // Keep the prior interleaved row allocation order.
        possible.push(default_possible.clone());
        unconditional.push(default_unconditional.clone());
    }
    for (terminal, domain) in domains.iter().chain(std::iter::once(completion)).enumerate() {
        for (top, certificate) in domain.top_admission_partition().1 {
            let Some(maybe) = possible.get_mut(top as usize) else { continue; };
            let always = &mut unconditional[top as usize];
            match certificate {
                TopAdmission::Never => { maybe.clear(terminal); always.clear(terminal); },
                TopAdmission::Always => { maybe.set(terminal); always.set(terminal); },
                TopAdmission::DependsOnSuffix => { maybe.set(terminal); always.clear(terminal); },
            }
        }
    }
    (possible, unconditional)
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
        self.install_template_parser_impl(false)
    }

    /// Source compilation has just produced the retained terminal templates
    /// from this exact immutable LR table. Reuse those compiler-owned graphs;
    /// generic conversion of a loaded artifact deliberately does not opt in.
    pub(crate) fn install_template_parser_from_compile(&mut self) -> crate::Result<()> {
        self.install_template_parser_impl(true)
    }

    fn install_template_parser_impl(&mut self, reuse_compiler_templates: bool) -> crate::Result<()> {
        if self.has_template_parser() { return Ok(()); }
        if self.parser_has_controls() || self.uses_compact_segmented_parser_runtime() || !self.late_grammar_slots.is_empty() {
            return Err(crate::Error::Compilation("template-only parser composition/control closure is not implemented; no LR fallback is permitted".into()));
        }
        let terminal_count = self.table.num_terminals;
        let (templates, completion, state_count) = if self.uses_sparse_direct_regular_runtime() {
            sparse_regular_templates(self.direct_regular_automaton.as_ref().unwrap(), terminal_count)?
        } else {
            let retained = if reuse_compiler_templates && !self.uses_dynamic_runtime()
                && self.composition_parser_templates_by_terminal.len() == terminal_count as usize
            {
                self.composition_parser_templates_by_terminal.as_slice()
            } else {
                &[]
            };
            let selected = (0..terminal_count as usize)
                .map(|terminal| retained.get(terminal).is_none_or(Option::is_none))
                .collect::<Vec<_>>();
            let raw = if selected.iter().any(|&missing| missing) {
                let characterizations = try_characterize_selected_terminals_for_terminal_count(
                    &self.table, terminal_count, &selected,
                ).map_err(crate::Error::Compilation)?;
                Templates::dfas_from_characterizations(&characterizations)
            } else {
                std::collections::BTreeMap::new()
            };
            let mut rebuilt = raw.into_iter();
            let mut templates = vec![None; terminal_count as usize];
            for terminal in 0..terminal_count {
                let input = match retained.get(terminal as usize).and_then(Option::as_ref) {
                    Some(dfa) => std::borrow::Cow::Borrowed(dfa),
                    None => {
                        let (rebuilt_terminal, dfa) = rebuilt.next().ok_or_else(||
                            crate::Error::Compilation(format!(
                                "missing complete template for terminal {terminal}; refusing table fallback"
                            )))?;
                        if rebuilt_terminal != terminal {
                            return Err(crate::Error::Compilation(format!(
                                "template inventory mismatch for terminal {terminal}: found {rebuilt_terminal}"
                            )));
                        }
                        std::borrow::Cow::Owned(dfa)
                    }
                };
                let dfa = specialize_template_dfa_defaults_for_commit_split_input(input.as_ref());
                let split = try_split_commit_template_dfas(&dfa).ok_or_else(|| crate::Error::Compilation(format!("terminal {terminal} template is not acyclic pop/read/push; refusing table fallback")))?;
                templates[terminal as usize] = Some(Arc::new(split));
                // Reconstructed inputs are owned by this iteration, preserving
                // the original conversion path's prompt release of each raw
                // DFA rather than retaining the whole rebuilt inventory.
            }
            if rebuilt.next().is_some() {
                return Err(crate::Error::Compilation(
                    "unexpected extra reconstructed terminal template".into()
                ));
            }
            let completion = glrmask_parser_dwa::__private::templates::completion::compile_completion_template(&self.table)
                .map_err(crate::Error::Compilation)?;
            (templates, completion, self.table.num_states)
        };
        let (parser, runtime) = TemplateParser::compile_with_runtime(state_count, terminal_count,
            self.table.skip_terminals.clone(), &templates, completion)?;
        self.template_dfas_by_terminal = templates;
        self.fast_template_dfas_by_terminal = runtime;
        self.serialized_artifact_cache = None;
        self.deferred_table_rules_blob = None;
        self.deferred_table_rules = Default::default();
        self.template_parser = Some(Arc::new(parser));
        // Drop, do not merely clear rows or change an environment flag.
        self.table = ParserTableStorage::absent();
        if !self.uses_dynamic_runtime() {
            // Conversion occurs after ordinary compile finalization. Prime
            // the requested parser only after the table has been removed;
            // this cannot accidentally warm an LR fallback instead.
            self.prime_initial_commit_hot_path();
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::accumulator::TerminalsDisallowed;

    #[test]
    fn fresh_compiler_template_reuse_matches_reconstructed_programs() {
        let vocab = crate::Vocab::new(["a", "b", "(", ")", "ab", "aa", " ", "aab", "(()"]
            .into_iter().enumerate().map(|(id, value)| (id as u32, value.as_bytes().to_vec())).collect());
        let sources = [
            r#"start root; nt root ::= "a" | "(" root ")";"#,
            r#"start root; ignore WS; t WS ::= " "+; nt root ::= "a" root "b" | "";"#,
        ];
        let mut retained_total = 0;
        for source in sources {
            let ordinary = Constraint::compile(crate::Grammar::glrm(source), &vocab).unwrap();
            retained_total += ordinary.composition_parser_templates_by_terminal.iter()
                .filter(|template| template.is_some()).count();
            let mut reference = ordinary.clone();
            reference.install_template_parser().unwrap();
            for missing in 0..4 {
                let mut candidate = ordinary.clone();
                match missing {
                    1 => {
                        if let Some(slot) = candidate.composition_parser_templates_by_terminal
                            .iter_mut().find(|template| template.is_some())
                        {
                            *slot = None;
                        }
                    }
                    2 => candidate.composition_parser_templates_by_terminal.clear(),
                    3 => candidate.composition_parser_templates_by_terminal.push(None),
                    _ => {}
                }
                candidate.install_template_parser_from_compile().unwrap();
                assert!(candidate.table.as_lr().is_none());
                assert_eq!(candidate.template_dfas_by_terminal.len(), reference.template_dfas_by_terminal.len());
                for (a, b) in candidate.template_dfas_by_terminal.iter()
                    .zip(&reference.template_dfas_by_terminal)
                {
                    let (a, b) = (a.as_ref().unwrap(), b.as_ref().unwrap());
                    assert_eq!(a.pop, b.pop);
                    assert_eq!(a.read, b.read);
                    assert_eq!(a.push, b.push);
                    assert_eq!(a.pop_to_read, b.pop_to_read);
                    assert_eq!(a.pop_to_push, b.pop_to_push);
                    assert_eq!(a.read_to_push, b.read_to_push);
                }
                for bytes in [b"".as_slice(), b"a", b"(a)", b"aa", b"aab", b"aabb", b" "] {
                    let mut left = reference.start();
                    let mut right = candidate.start();
                    assert_eq!(left.commit_bytes(bytes).is_ok(), right.commit_bytes(bytes).is_ok());
                    assert_eq!(left.is_accepting(), right.is_accepting());
                    let mut a = vec![0; reference.mask_len()];
                    let mut b = vec![0; candidate.mask_len()];
                    left.fill_mask(&mut a);
                    right.fill_mask(&mut b);
                    assert_eq!(a, b, "prefix {bytes:?}, missing mode {missing}");
                }
            }
        }
        assert!(retained_total > 0, "fixture must exercise compiler template reuse");
    }

    fn pop_word(labels: &[i32], accepting: bool) -> CommitTemplateDfas {
        let mut pop = DFA::new();
        let mut cursor = pop.start_state;
        for &label in labels {
            let target = pop.add_state(); pop.add_transition(cursor, label, target); cursor = target;
        }
        pop.set_accepting(cursor, accepting);
        CommitTemplateDfas { pop, read: DFA::new(), push: DFA::new(),
            pop_to_read: Vec::new(), pop_to_push: Vec::new(), read_to_push: Vec::new() }
    }

    #[test]
    fn sparse_preparation_matches_every_dense_top_certificate() {
        use crate::compiler::glr::labels::DEFAULT_LABEL;
        for terminal_count in [0, 1, 5, 63, 64, 65, 129] {
            let mut domains = Vec::new();
            for terminal in 0..terminal_count {
                let label = (terminal % 70) as i32;
                let t = match terminal % 6 {
                    0 => pop_word(&[], true),
                    1 => pop_word(&[], false),
                    2 => pop_word(&[label], true),
                    3 => pop_word(&[label, DEFAULT_LABEL], true),
                    4 => {
                        let mut t = pop_word(&[DEFAULT_LABEL], true);
                        let dead = t.pop.add_state();
                        t.pop.add_transition(0, label, dead);
                        t
                    },
                    _ => {
                        let mut t = pop_word(&[DEFAULT_LABEL, 2], true);
                        let yes = t.pop.add_state();
                        t.pop.set_accepting(yes, true);
                        t.pop.add_transition(0, label, yes);
                        t
                    },
                };
                domains.push(compile_domain(&t).unwrap());
            }
            for completion in [pop_word(&[0], true), pop_word(&[], true), pop_word(&[], false)] {
                let completion = compile_domain(&completion).unwrap();
                for state_count in [0, 1, 4, 64, 71, 129] {
                    let (possible, unconditional) = top_certificate_rows(state_count, &domains, &completion);
                    assert_eq!(possible.len(), state_count as usize);
                    assert_eq!(unconditional.len(), state_count as usize);
                    for top in 0..state_count {
                        let mut expected_possible = BitSet::new(terminal_count + 1);
                        let mut expected_unconditional = BitSet::new(terminal_count + 1);
                        for (terminal, domain) in domains.iter().chain(std::iter::once(&completion)).enumerate() {
                            match domain.classify_top(top) {
                                TopAdmission::Never => {},
                                TopAdmission::Always => {
                                    expected_possible.set(terminal);
                                    expected_unconditional.set(terminal);
                                },
                                TopAdmission::DependsOnSuffix => expected_possible.set(terminal),
                            }
                        }
                        assert_eq!(possible[top as usize], expected_possible,
                            "possible: alphabet={state_count}, terminals={terminal_count}, top={top}");
                        assert_eq!(unconditional[top as usize], expected_unconditional,
                            "unconditional: alphabet={state_count}, terminals={terminal_count}, top={top}");
                    }
                }
            }
        }
    }

    #[test]
    fn bulk_top_certificates_match_literal_domains_on_shared_and_empty_languages() {
        let mut fallback = pop_word(&[crate::compiler::glr::labels::DEFAULT_LABEL, 2], true);
        let dead = fallback.pop.add_state();
        fallback.pop.add_transition(fallback.pop.start_state, 1, dead);
        let raw = vec![pop_word(&[1], true), fallback, pop_word(&[], true),
            pop_word(&[], false), pop_word(&[2, 3], true)];
        let raw: Vec<_> = raw.into_iter().map(|p| Some(Arc::new(p))).collect();
        let parser = TemplateParser::compile(4, 5, BTreeSet::new(), &raw, pop_word(&[0], true)).unwrap();
        let mut concrete = vec![Vec::<u32>::new()];
        for length in 1..=3u32 {
            for encoded in 0..6u32.pow(length) {
                let mut n=encoded; let mut stack=Vec::new();
                for _ in 0..length { stack.push(n%6); n/=6; }
                concrete.push(stack);
            }
        }
        let clean=TerminalsDisallowed::new(); let guarded=clean.with_insert(0, 3);
        let mut inputs=vec![ParserGSS::empty()];
        for stack in &concrete { inputs.push(ParserGSS::from_single_stack(stack.clone(), clean.clone())); }
        for group in concrete.chunks(7) {
            let paths:Vec<_>=group.iter().enumerate().map(|(i,s)|(s.clone(), if i%2==0 { clean.clone() } else { guarded.clone() })).collect();
            inputs.push(ParserGSS::from_stacks(&paths));
        }
        for input in inputs {
            let literal = input.to_stacks(4096).unwrap();
            for requested in 0..64u64 {
                // Include an unknown bit, and out-of-certificate stack tops
                // 4/5 above, to verify bounds fall back exactly, not reject.
                let mut candidates=BitSet::new(73); candidates.set(72);
                for bit in 0..6 { if requested & (1<<bit)!=0 { candidates.set(bit); } }
                let expected:Vec<_>=candidates.iter().filter(|&bit| {
                    let domain=if bit==5 { Some(&parser.completion) } else { parser.domains.get(bit) };
                    domain.is_some_and(|d| literal.iter().any(|(s,_)|d.matches_top_first(s.iter().rev().copied())))
                }).collect();
                let actual=parser.admitted(&input,&candidates);
                assert_eq!(actual.iter().collect::<Vec<_>>(),expected);
                assert_eq!(parser.admits_any(&input,&candidates),!expected.is_empty());
                for (stack,_) in &literal {
                    let expected=candidates.iter().any(|bit| {
                        let domain=if bit==5 { Some(&parser.completion) } else { parser.domains.get(bit) };
                        domain.is_some_and(|d|d.matches_top_first(stack.iter().rev().copied()))
                    });
                    assert_eq!(parser.admits_flat_any(stack,&candidates),expected);
                }
            }
        }
    }
}
