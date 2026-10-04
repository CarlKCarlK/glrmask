//! Parser-independent runtime seam for finite acyclic stack transducers.
//!
//! The built-in frontend may use temporary LR compiler analysis to produce
//! this data before any Constraint exists. Accidental Constraint table accesses
//! through an unported helper panic instead of silently taking a table fallback.
//! Tokenizer execution, GSS ownership, delayed exclusions and mask generation
//! remain in their existing shared implementations.

pub(crate) mod wire;
pub(crate) mod composition;
pub(crate) mod embedding;
#[cfg(test)]
mod sparse_composition_tests;
pub(crate) mod link;
pub(crate) mod link_program;
pub(crate) mod link_grammar;
pub(crate) mod scoped_program;

use std::collections::{BTreeMap, BTreeSet};
use std::ops::{ControlFlow, Deref, DerefMut};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use glrmask_parser_dwa::__private::templates::admissibility::{DomainProbe, TemplateDomain, TopAdmission};
use glrmask_parser_dwa::__private::templates::characterize::try_characterize_selected_terminals_for_terminal_count;
use glrmask_parser_dwa::__private::templates::compile_dfa::{
    TemplateDfaGroup, Templates, specialize_template_dfa_defaults_for_commit_split_input, try_split_commit_template_dfas,
};
use crate::automata::unweighted_u32::dfa::DFA;
use crate::compiler::glr::analysis::EOF;
use crate::compiler::glr::labels::encode_negative_label;
use crate::compiler::glr::parser::ParserGSS;
use crate::compiler::glr::table::{AdmissionPolicy, GLRTable};
use crate::ds::bitset::BitSet;
use crate::grammar::flat::{DirectRegularAutomaton, TerminalID};
use super::{CommitTemplateDfas, Constraint};

/// Each terminal's DEFAULT specialization and complete phase-equivalence
/// check depends only on that terminal's immutable raw DFA. Keep the output
/// terminal order (including which error is reported) independent of worker
/// scheduling. Owned inputs are released by their own transformation rather
/// than retained behind borrowed parallel iterators.
fn compile_terminal_template(
    terminal: TerminalID,
    raw: DFA,
) -> crate::Result<Arc<CommitTemplateDfas>> {
    let specialized = specialize_template_dfa_defaults_for_commit_split_input(&raw);
    let split = try_split_commit_template_dfas(&specialized).ok_or_else(|| {
        crate::Error::Compilation(format!(
            "terminal {terminal} template is not acyclic pop/read/push; refusing table fallback"
        ))
    })?;
    Ok(Arc::new(split))
}

#[cfg(test)]
fn compile_terminal_template_rows(
    raw: BTreeMap<TerminalID, DFA>,
    terminal_count: u32,
) -> crate::Result<crate::runtime::artifact::TemplateDfasByTerminal> {
    // Tiny inventories should not pay a worker handoff or a second collection.
    // This is a structural work estimate, independent of grammar identities.
    const PARALLEL_MIN_RAW_STATES: usize = 1_024;
    let parallel = raw.len() > 1
        && raw.values().map(|dfa| dfa.states.len()).sum::<usize>() >= PARALLEL_MIN_RAW_STATES
        && !crate::compiler::macro_parallelism_disabled();
    if parallel {
        crate::compiler::pipeline::run_with_compile_thread_pool(|| {
            compile_terminal_template_rows_with_parallelism(raw, terminal_count, true)
        })
    } else {
        compile_terminal_template_rows_with_parallelism(raw, terminal_count, false)
    }
}

#[cfg(test)]
fn compile_terminal_template_rows_with_parallelism(
    raw: BTreeMap<TerminalID, DFA>,
    terminal_count: u32,
    parallel: bool,
) -> crate::Result<crate::runtime::artifact::TemplateDfasByTerminal> {
    if raw.len() != terminal_count as usize
        || raw.keys().copied().ne(0..terminal_count)
    {
        return Err(crate::Error::Compilation(
            "terminal template inventory must cover the complete terminal domain".into()
        ));
    }
    if !parallel || rayon::current_num_threads() == 1 {
        return raw.into_iter()
            .map(|(terminal, dfa)| compile_terminal_template(terminal, dfa).map(Some))
            .collect();
    }
    use rayon::prelude::*;
    let inputs = raw.into_iter().collect::<Vec<_>>();
    // Collect ordered per-terminal Results first: parallel Result collection
    // can return whichever error a worker happens to encounter first.
    let results = inputs.into_par_iter()
        .map(|(terminal, dfa)| compile_terminal_template(terminal, dfa))
        .collect::<Vec<_>>();
    results.into_iter().map(|result| result.map(Some)).collect()
}

fn compile_terminal_template_groups(
    groups: Vec<TemplateDfaGroup>,
    terminal_count: u32,
) -> crate::Result<crate::runtime::artifact::TemplateDfasByTerminal> {
    // Count the same virtual per-terminal raw states as the previous fanout,
    // so avoiding clones does not change the existing worker-pool threshold.
    const PARALLEL_MIN_RAW_STATES: usize = 1_024;
    let states = groups.iter().map(|group| {
        group.dfa.states.len().saturating_mul(group.terminals.len())
    }).fold(0usize, usize::saturating_add);
    let parallel = terminal_count > 1 && states >= PARALLEL_MIN_RAW_STATES
        && !crate::compiler::macro_parallelism_disabled();
    if parallel {
        crate::compiler::pipeline::run_with_compile_thread_pool(|| {
            compile_terminal_template_groups_with_parallelism(groups, terminal_count, true)
        })
    } else {
        compile_terminal_template_groups_with_parallelism(groups, terminal_count, false)
    }
}

fn compile_terminal_template_groups_with_parallelism(
    mut groups: Vec<TemplateDfaGroup>,
    terminal_count: u32,
    parallel: bool,
) -> crate::Result<crate::runtime::artifact::TemplateDfasByTerminal> {
    let invalid_inventory = || crate::Error::Compilation(
        "terminal template inventory must cover the complete terminal domain".into()
    );
    let mut covered = vec![false; terminal_count as usize];
    for group in &groups {
        if group.terminals.is_empty() || group.terminals.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(invalid_inventory());
        }
        for &terminal in &group.terminals {
            let Some(slot) = covered.get_mut(terminal as usize) else {
                return Err(invalid_inventory());
            };
            if std::mem::replace(slot, true) { return Err(invalid_inventory()); }
        }
    }
    if covered.iter().any(|&present| !present) { return Err(invalid_inventory()); }
    // Characterization order differs from terminal order. Compare ordered
    // group results so the smallest failing terminal remains the first error.
    groups.sort_unstable_by_key(|group| group.terminals[0]);
    let mut output = vec![None; terminal_count as usize];
    let transform = |group: TemplateDfaGroup| {
        let result = compile_terminal_template(group.terminals[0], group.dfa);
        (group.terminals, result)
    };
    if parallel && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        let results = groups.into_par_iter().map(transform).collect::<Vec<_>>();
        for (terminals, result) in results {
            let template = result?;
            for terminal in terminals { output[terminal as usize] = Some(Arc::clone(&template)); }
        }
    } else {
        for group in groups {
            let (terminals, result) = transform(group);
            let template = result?;
            for terminal in terminals { output[terminal as usize] = Some(Arc::clone(&template)); }
        }
    }
    Ok(output)
}

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
    #[track_caller]
    fn from(_table: GLRTable) -> Self {
        panic!("LR-BACKED CONSTRAINT MATERIALIZATION IS FORBIDDEN: derive templates from compiler parts before constructing or loading a Constraint")
    }
}
impl Deref for ParserTableStorage {
    type Target = GLRTable;
    #[track_caller]
    fn deref(&self) -> &GLRTable {
        panic!("LR CONSTRAINT RUNTIME ACCESS IS FORBIDDEN: route this operation through native parser programs")
    }
}
impl DerefMut for ParserTableStorage {
    #[track_caller]
    fn deref_mut(&mut self) -> &mut GLRTable {
        panic!("LR CONSTRAINT TABLE MUTATION IS FORBIDDEN")
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
    pub(crate) composition: Option<Arc<composition::TemplateComposition>>,
    pub(crate) embedding: Option<Arc<embedding::TemplateEmbedding>>,
    pub(crate) link_grammar: Option<Arc<link_grammar::LinkGrammar>>,
    domains: Vec<Arc<TemplateDomain>>,
    completion: Arc<TemplateDomain>,
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

/// Complete immutable parser machinery prepared for the same constraint.
/// Runtime-only cache finalization may run before installation, but grammar,
/// parser actions, terminal identities, and skip terminals must not change.
pub(crate) struct PreparedTemplateParser {
    source_state_count: u32,
    source_terminal_count: u32,
    pub(crate) templates: Vec<Option<Arc<CommitTemplateDfas>>>,
    pub(crate) runtime: crate::runtime::artifact::FastTemplateDfasByTerminal,
    pub(crate) parser: TemplateParser,
}

impl PreparedTemplateParser {
    /// Derive executable templates from construction-only compiler parts.
    /// No Constraint exists while this temporary table is inspected.
    pub(crate) fn from_compiler_parts(
        table: &GLRTable,
        direct_regular: Option<&DirectRegularAutomaton>,
        retained_templates: &[Option<DFA>],
        ignore: Option<u32>,
        dynamic: bool,
        preserve_coordinate: bool,
    ) -> crate::Result<Self> {
        let terminal_count = table.num_terminals;
        let sparse_regular = direct_regular.is_some() && table.num_rules == 0 && table.action.is_empty();
        let (templates, completion, state_count) = if sparse_regular {
            sparse_regular_templates(direct_regular.unwrap(), terminal_count)?
        } else {
            let retained = if !dynamic
                && retained_templates.len() == terminal_count as usize
            {
                retained_templates
            } else {
                &[]
            };
            let selected = (0..terminal_count as usize)
                .map(|terminal| retained.get(terminal).is_none_or(Option::is_none))
                .collect::<Vec<_>>();
            let characterizations = if selected.iter().any(|&missing| missing) {
                try_characterize_selected_terminals_for_terminal_count(
                    &table, terminal_count, &selected,
                ).map_err(crate::Error::Compilation)?
            } else {
                std::collections::BTreeMap::new()
            };
            let templates = if dynamic {
                // The existing exact characterization quotient owns one raw
                // graph per group. Specialization and the complete split proof
                // run once on that graph; immutable results retain every slot.
                let (groups, _) = Templates::grouped_dfas_from_characterizations(&characterizations);
                compile_terminal_template_groups(groups, terminal_count)?
            } else {
                let raw = Templates::dfas_from_characterizations(&characterizations);
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
                templates
            };
            let completion = glrmask_parser_dwa::__private::templates::completion::compile_completion_template(&table)
                .map_err(crate::Error::Compilation)?;
            (templates, completion, table.num_states)
        };
        if preserve_coordinate && state_count != table.num_states {
            return Err(crate::Error::Compilation("template conversion would change a composed parser's stack coordinate".into()));
        }
        let (mut parser, runtime) = TemplateParser::compile_with_runtime(
            state_count, terminal_count, table.skip_terminals.clone(), &templates, completion,
        )?;
        // Not every standalone provider/legacy table has the finite canonical
        // embedding contract. Such a parser remains runnable, but linking it
        // returns an explicit error instead of reconstructing a table.
        if sparse_regular {
            // This built-in frontend keeps exactly one symbol per frame and
            // its generated completion relation POPs that symbol. Unlike an
            // arbitrary provider's predicate, this is the actual return action.
            parser.embedding = Some(Arc::new(embedding::TemplateEmbedding::from_sparse_regular(
                &parser.completion_template, parser.state_count, table.embedded_start_nullable(),
                0..terminal_count,
            ).map_err(crate::Error::Compilation)?));
        } else {
            parser.embedding = crate::compiler::glr::table::subgrammar_child_return_pop(table, &table.rules).ok().and_then(|return_pop| embedding::TemplateEmbedding::from_table(table, table.embedded_start_nullable(), return_pop, crate::compiler::boundary_transfer::valid_slot_entry_terminals(table)).ok()).map(Arc::new);
        }
        parser.link_grammar = link_grammar::LinkGrammar::from_compiler_table(table, ignore).map_err(crate::Error::Compilation)?;
        if let Some(grammar) = parser.link_grammar.as_mut() {
            use crate::compiler::boundary_stack_support::CompilerEffects;
            let summary = if sparse_regular {
                CompilerEffects::from_regular_programs(&templates,&parser.completion_template,state_count)
            } else {
                parser.embedding.as_ref().ok_or_else(|| "missing effect entry certificate".to_string())
                    .and_then(|embedding| CompilerEffects::from_table(table,&embedding.entries))
            };
            // A budget/unsupported-shape refusal leaves this proof absent;
            // later composition keeps its complete template fallback.
            Arc::make_mut(grammar).set_stack_effects(summary.ok());
        }
        Ok(PreparedTemplateParser {
            source_state_count: table.num_states,
            source_terminal_count: terminal_count,
            templates,
            runtime,
            parser,
        })
    }
}

fn compile_domain(template: &CommitTemplateDfas) -> crate::Result<TemplateDomain> {
    TemplateDomain::compile(template).map_err(crate::Error::Compilation)
}

type PreparedTerminalView = Option<Arc<crate::runtime::artifact::FastCommitTemplateDfas>>;

fn prepare_terminal_domain_and_view(
    terminal: usize,
    template: &Option<Arc<CommitTemplateDfas>>,
    state_count: u32,
    prepare_runtime: bool,
) -> crate::Result<(Arc<TemplateDomain>, PreparedTerminalView)> {
    let template = template.as_deref().ok_or_else(|| {
        crate::Error::Compilation(format!("missing template for terminal {terminal}"))
    })?;
    let validated = super::commit::template_prepare::TemplatePreparation::new(template)
        .map_err(crate::Error::Compilation)?;
    validated.validate_alphabet(state_count).map_err(crate::Error::Compilation)?;
    let domain = TemplateDomain::from_validated(&validated).map_err(crate::Error::Compilation)?;
    let view = prepare_runtime.then(|| Arc::new(
        crate::runtime::artifact::FastCommitTemplateDfas::from_prepared(&validated)
    ));
    Ok((Arc::new(domain), view))
}

/// Prepare independent immutable terminal programs without changing their
/// runtime representation. The ordered result collection preserves the same
/// first error and terminal coordinates as the serial constructor.
fn prepare_terminal_inventory(
    templates: &[Option<Arc<CommitTemplateDfas>>],
    state_count: u32,
    prepare_runtime: bool,
    parallel: bool,
) -> crate::Result<(Vec<Arc<TemplateDomain>>, crate::runtime::artifact::FastTemplateDfasByTerminal)> {
    let mut domains = Vec::with_capacity(templates.len());
    let mut runtime = Vec::with_capacity(if prepare_runtime { templates.len() } else { 0 });
    if parallel && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        let results = templates.par_iter().enumerate()
            .map(|(terminal, template)| {
                prepare_terminal_domain_and_view(terminal, template, state_count, prepare_runtime)
            })
            .collect::<Vec<_>>();
        for result in results {
            let (domain, view) = result?;
            domains.push(domain);
            if prepare_runtime { runtime.push(view); }
        }
    } else {
        for (terminal, template) in templates.iter().enumerate() {
            let (domain, view) = prepare_terminal_domain_and_view(
                terminal, template, state_count, prepare_runtime,
            )?;
            domains.push(domain);
            if prepare_runtime { runtime.push(view); }
        }
    }
    Ok((domains, runtime))
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
        const PARALLEL_PREPARATION_MIN_STATES: usize = 4_096;
        let large_inventory = templates.len() > 1 && templates.iter()
            .filter_map(Option::as_deref)
            .map(|template| template.pop.states.len()
                .saturating_add(template.read.states.len())
                .saturating_add(template.push.states.len()))
            .fold(0usize, usize::saturating_add) >= PARALLEL_PREPARATION_MIN_STATES;
        let (domains, runtime) = if large_inventory && !crate::compiler::macro_parallelism_disabled() {
            crate::compiler::pipeline::run_with_compile_thread_pool(|| {
                prepare_terminal_inventory(templates, state_count, prepare_runtime, true)
            })?
        } else {
            prepare_terminal_inventory(templates, state_count, prepare_runtime, false)?
        };
        let validated = super::commit::template_prepare::TemplatePreparation::new(&completion_template)
            .map_err(crate::Error::Compilation)?;
        validated.validate_alphabet(state_count).map_err(crate::Error::Compilation)?;
        let completion = Arc::new(TemplateDomain::from_validated(&validated).map_err(crate::Error::Compilation)?);
        let (possible, unconditional) = top_certificate_rows(state_count, &domains, &completion);
        let profile = std::env::var_os("GLRMASK_PROFILE_TEMPLATE_BACKEND").is_some();
        Ok((Self { state_count, terminal_count, skip_terminals,
            completion_template: Arc::new(completion_template), composition: None, embedding: None, link_grammar: None,
            domains, completion,
            possible, unconditional, profile,
            advances: AtomicU64::new(0), admissions: AtomicU64::new(0), completions: AtomicU64::new(0),
        }, runtime))
    }

    pub(crate) fn from_composition(state_count: u32, terminal_count: u32,
        composition: composition::TemplateComposition) -> crate::Result<Self> {
        if composition.outer_views.len() != terminal_count as usize {
            return Err(crate::Error::Compilation("outer scoped inventory count mismatch".into()));
        }
        let completion = composition.completion_view.as_ref().ok_or_else(||
            crate::Error::Compilation("missing scoped completion view".into()))?;
        // Scoped providers answer outer, control and completion queries through
        // their shared local domains. Every composed admission path dispatches
        // there before reading dense rows, and Constraint row helpers explicitly
        // decline composed rows. No global state/terminal product is needed.
        Ok(Self { state_count, terminal_count, skip_terminals: BTreeSet::new(),
            completion_template: Arc::clone(&completion.source), completion: Arc::clone(&completion.domain),
            domains: composition.outer_views.iter().map(|view| Arc::clone(&view.domain)).collect(),
            composition: Some(Arc::new(composition)), embedding: None, link_grammar: None,
            possible: Vec::new(), unconditional: Vec::new(),
            profile: std::env::var_os("GLRMASK_PROFILE_TEMPLATE_BACKEND").is_some(),
            advances: AtomicU64::new(0), admissions: AtomicU64::new(0), completions: AtomicU64::new(0) })
    }

    #[inline]
    pub(crate) fn record_advance(&self) {
        if self.profile { self.advances.fetch_add(1, Ordering::Relaxed); }
    }

    #[inline]
    pub(crate) fn admits(&self, stack: &ParserGSS, terminal: TerminalID) -> bool {
        if let Some(provider) = self.composition_provider() {
            return if terminal == EOF {
                crate::compiler::glr::parser::stacks_finished_with_provider(&provider, stack, EOF)
            } else {
                crate::compiler::glr::parser::stack_may_advance_on_with_provider(&provider, stack, terminal)
            };
        }
        self.admits_control_closed(stack, terminal)
    }

    fn admits_control_closed(&self, stack: &ParserGSS, terminal: TerminalID) -> bool {
        if self.profile { self.admissions.fetch_add(1, Ordering::Relaxed); }
        if let Some(admitted) = self.composition.as_ref().and_then(|composition|
            composition.admits_outer(stack, terminal)) { return admitted; }
        if terminal != EOF && terminal >= self.terminal_count {
            return self.composition.as_ref().is_some_and(|composition|
                composition.admits(stack, terminal - self.terminal_count));
        }
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
        if let Some(provider) = self.composition_provider() {
            return crate::compiler::glr::parser::stack_may_advance_on_any_with_provider(
                &provider, stack, candidates.iter().map(|terminal| terminal as u32));
        }
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
        if let Some(provider) = self.composition_provider() {
            let mut admitted = BitSet::new(candidates.len());
            crate::compiler::glr::parser::for_each_admitted_symbol_with_provider(
                &provider, stack, candidates.iter().map(|terminal| (terminal, terminal as u32)),
                |terminal| admitted.set(terminal));
            return admitted;
        }
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
        if self.composition.is_some() {
            let gss = ParserGSS::from_single_stack(stack.to_vec(),
                crate::compiler::glr::accumulator::TerminalsDisallowed::new());
            return self.admits_any(&gss, candidates);
        }
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

    /// Restore the nullable source start removed by compiler normalization in
    /// the executable EOF relation as well as its derived domain certificates.
    /// Stack symbol zero is the native parser's initial grammar frontier.
    pub(crate) fn preserve_start_nullable(&mut self) {
        if self.completion.matches_top_first([0]) { return; }
        let completion = link_program::compile(&[
            link_program::action_nfa(&self.completion_template)
                .expect("validated finite completion relation"),
            link_program::nullable_return(0),
        ]).expect("nullable completion must remain finite");
        self.completion = Arc::new(compile_domain(&completion)
            .expect("validated nullable completion domain"));
        self.completion_template = Arc::new(completion);
        (self.possible, self.unconditional) =
            top_certificate_rows(self.state_count, &self.domains, &self.completion);
    }

    pub(crate) fn finished(&self, stack: &ParserGSS) -> bool {
        if self.profile { self.completions.fetch_add(1, Ordering::Relaxed); }
        self.admits(stack, EOF)
    }

    pub(crate) fn report(&self) -> serde_json::Value {
        serde_json::json!({
            "backend":"acyclic-template-dfa", "lr_table_present":false,
            "terminal_count":self.terminal_count, "stack_symbol_count":self.state_count,
            "domain_states":self.domains.iter().map(|domain| domain.state_count()).sum::<usize>(),
            "domain_edges":self.domains.iter().map(|domain| domain.edge_count()).sum::<usize>(),
            "domain_heap_payload_bytes":self.domains.iter().map(|domain| domain.heap_payload_bytes()).sum::<usize>(),
            "completion_domain_states":self.completion.state_count(),
            "composition":self.composition.as_ref().map(|composition| composition.report()),
            "finite_embedding":self.embedding.is_some(),
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
    domains: &[Arc<TemplateDomain>],
    completion: &Arc<TemplateDomain>,
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
    let invalid = || crate::Error::Compilation("invalid sparse regular automaton coordinate".into());
    let state_count = automaton.states.len().checked_add(1).and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| crate::Error::Compilation("regular stack alphabet too large".into()))?;
    if automaton.start_states.iter().any(|&state| state as usize >= automaton.states.len())
        || automaton.states.iter().any(|row| {
            row.epsilons.iter().any(|&state| state as usize >= automaton.states.len())
                || row.transitions.iter().any(|(&terminal, targets)| terminal >= terminal_count
                    || targets.iter().any(|&state| state as usize >= automaton.states.len()))
        }) { return Err(invalid()); }
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
        self.template_parser.as_ref().map_or_else(
            || !self.table.control_terminals.is_empty(),
            |parser| parser.composition.as_ref().is_some_and(|composition| composition.has_controls()))
    }
    #[inline]
    pub(crate) fn parser_skip_terminals(&self) -> &BTreeSet<TerminalID> {
        self.template_parser.as_ref().map_or_else(|| &self.table.skip_terminals, |parser| &parser.skip_terminals)
    }
    pub(crate) fn parser_is_control_terminal(&self, terminal: TerminalID) -> bool {
        self.template_parser.as_ref().map_or_else(|| self.table.control_terminals.contains(&terminal),
            |parser| parser.link_grammar.as_ref().is_some_and(|grammar| grammar.control_terminals.contains(&terminal)))
    }
    pub(crate) fn parser_control_terminals(&self) -> BTreeSet<u32> {
        self.template_parser.as_ref().map_or_else(|| self.table.control_terminals.clone(),
            |parser| parser.link_grammar.as_ref().map_or_else(BTreeSet::new,
                |grammar| grammar.control_terminals.clone()))
    }
    #[inline]
    pub(crate) fn parser_admission_policy(&self) -> AdmissionPolicy {
        if self.has_template_parser() { AdmissionPolicy::ExactSimulation } else { self.table.admission_policy }
    }
    #[inline]
    pub(crate) fn parser_advance_row(&self, top: u32) -> Option<&BitSet> {
        if self.template_composition_provider().is_some() { return None; }
        self.template_parser.as_ref().map_or_else(|| self.table.advance_row(top), |parser| parser.possible.get(top as usize))
    }
    #[inline]
    pub(crate) fn parser_advance_row_allows(&self, top: u32, terminal: u32) -> bool {
        if self.template_composition_provider().is_some() {
            return top < self.parser_symbol_count();
        }
        if let Some(parser) = &self.template_parser {
            let bit = if terminal == EOF { parser.terminal_count as usize } else { terminal as usize };
            parser.possible.get(top as usize).is_some_and(|row| row.contains(bit))
        } else { self.table.advance_row_allows(top, terminal) }
    }
    #[inline]
    pub(crate) fn parser_advance_row_intersects(&self, top: u32, terminals: &BitSet) -> bool {
        if self.template_composition_provider().is_some() {
            return top < self.parser_symbol_count() && !terminals.is_empty();
        }
        if let Some(parser) = &self.template_parser {
            parser.possible.get(top as usize).is_some_and(|row| row.words().iter().zip(terminals.words()).any(|(a,b)| a & b != 0))
        } else { self.table.advance_row_intersects(top, terminals) }
    }
    #[inline]
    pub(crate) fn parser_unconditional_row(&self, top: u32) -> Option<&BitSet> {
        if self.template_composition_provider().is_some() { return None; }
        self.template_parser.as_ref().map_or_else(|| self.table.unconditional_advance_row(top), |parser| parser.unconditional.get(top as usize))
    }

    pub(crate) fn install_template_parser(&mut self) -> crate::Result<()> {
        self.install_template_parser_in_coordinate(false)
    }

    /// Source compilation has just produced the retained terminal templates
    /// from this exact immutable LR table. Reuse those compiler-owned graphs;
    /// generic conversion of a loaded artifact deliberately does not opt in.
    pub(crate) fn install_template_parser_from_compile(&mut self) -> crate::Result<()> {
        if self.has_template_parser() { return Ok(()); }
        if self.uses_compact_segmented_parser_runtime() {
            return self.install_composition_template_parser();
        }
        self.install_template_parser_impl(true, false)
    }

    pub(crate) fn install_template_parser_in_coordinate(&mut self, preserve_coordinate: bool) -> crate::Result<()> {
        if self.has_template_parser() { return Ok(()); }
        if self.uses_compact_segmented_parser_runtime() {
            return self.install_composition_template_parser();
        }
        self.install_template_parser_impl(false, preserve_coordinate)
    }

    fn install_template_parser_impl(
        &mut self,
        reuse_compiler_templates: bool,
        preserve_coordinate: bool,
    ) -> crate::Result<()> {
        if self.has_template_parser() { return Ok(()); }
        let prepared = self.prepare_template_parser_impl(reuse_compiler_templates, preserve_coordinate)?;
        self.install_prepared_template_parser(prepared)
    }

    /// Derive the complete table-free parser without changing the active
    /// backend. The compiler can overlap this work with vocabulary preparation
    /// while preserving the established runtime-cache finalization order.
    pub(crate) fn prepare_template_parser(&self) -> crate::Result<Option<PreparedTemplateParser>> {
        if self.has_template_parser() { return Ok(None); }
        if self.uses_compact_segmented_parser_runtime() {
            return Err(crate::Error::Compilation(
                "prepared standalone template parser is not valid for segmented composition".into()
            ));
        }
        self.prepare_template_parser_impl(false, false).map(Some)
    }

    fn prepare_template_parser_impl(
        &self,
        reuse_compiler_templates: bool,
        preserve_coordinate: bool,
    ) -> crate::Result<PreparedTemplateParser> {
        if self.parser_has_controls() || self.uses_compact_segmented_parser_runtime() {
            return Err(crate::Error::Compilation("template-only parser composition/control closure is not implemented; no LR fallback is permitted".into()));
        }
        PreparedTemplateParser::from_compiler_parts(&self.table,
            self.direct_regular_automaton.as_ref(),
            if reuse_compiler_templates { &self.composition_parser_templates_by_terminal } else { &[] },
            self.ignore_terminal, self.uses_dynamic_runtime(), preserve_coordinate)
    }

    pub(crate) fn install_prepared_template_parser(&mut self, prepared: PreparedTemplateParser) -> crate::Result<()> {
        if self.has_template_parser() || !self.table.is_present()
            || self.table.num_states != prepared.source_state_count
            || self.table.num_terminals != prepared.source_terminal_count
            || self.table.skip_terminals != prepared.parser.skip_terminals
        {
            return Err(crate::Error::Compilation(
                "prepared template parser no longer matches its source constraint".into()
            ));
        }
        self.template_dfas_by_terminal = prepared.templates;
        self.fast_template_dfas_by_terminal = prepared.runtime;
        self.serialized_artifact_cache = None;
        self.deferred_table_rules_blob = None;
        self.deferred_table_rules = Default::default();
        self.template_parser = Some(Arc::new(prepared.parser));
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

    /// Storage evidence only: drop optional compiler analysis in an isolated
    /// clone, preserving every executable graph and the original Constraint.
    #[cfg(feature="internal-api")]
    pub(crate) fn save_without_effect_metadata_for_diagnostic(&self) -> Vec<u8> {
        fn strip(constraint:&mut Constraint) {
            if let Some(parser)=constraint.template_parser.as_mut() {
                // TemplateParser contains atomics and intentionally has no general Clone.
                // Copy only this diagnostic view; executable graphs and domains stay shared.
                let source=parser.as_ref();
                let mut replacement=TemplateParser {
                    state_count:source.state_count,terminal_count:source.terminal_count,
                    skip_terminals:source.skip_terminals.clone(),completion_template:source.completion_template.clone(),
                    composition:source.composition.clone(),embedding:source.embedding.clone(),
                    link_grammar:source.link_grammar.clone(),domains:source.domains.clone(),completion:source.completion.clone(),
                    possible:source.possible.clone(),unconditional:source.unconditional.clone(),profile:source.profile,
                    advances:AtomicU64::new(source.advances.load(Ordering::Relaxed)),
                    admissions:AtomicU64::new(source.admissions.load(Ordering::Relaxed)),
                    completions:AtomicU64::new(source.completions.load(Ordering::Relaxed)),
                };
                if let Some(grammar)=replacement.link_grammar.as_mut() {Arc::make_mut(grammar).set_stack_effects(None);}
                *parser=Arc::new(replacement);
            }
            constraint.serialized_artifact_cache=None;
            if let Some(overlay)=constraint.static_dynamic_overlay.as_mut() {
                for component in &mut overlay.segmented_parser_components {
                    strip(Arc::make_mut(&mut component.constraint));
                }
            }
        }
        let mut isolated=self.clone();strip(&mut isolated);isolated.save()
    }

    pub(crate) fn parser_backend_report(&self) -> serde_json::Value {
        match &self.template_parser {
            Some(parser) => {
                assert!(!self.table.is_present(), "template-only constraint retained an LR table");
                let mut report = parser.report();
                if let Some(summary)=parser.link_grammar.as_ref().and_then(|grammar|grammar.stack_effects()) {
                    let entry_effects=summary.entries.values().map(Vec::len).sum::<usize>();
                    let pushes=summary.effects.iter().chain(summary.entries.values().flatten()).map(|effect|effect.pushes.len()).sum::<usize>();
                    let vector_bytes=std::mem::size_of_val(summary)
                        +summary.effects.capacity()*summary.effects.first().map_or(0,std::mem::size_of_val)
                        +summary.accepting.capacity()*std::mem::size_of::<u32>()
                        +summary.entries.values().map(|row|row.capacity()*summary.effects.first().map_or(0,std::mem::size_of_val)).sum::<usize>()
                        +summary.effects.iter().chain(summary.entries.values().flatten()).map(|effect|effect.pushes.capacity()*std::mem::size_of::<u32>()).sum::<usize>();
                    report["compiler_effect_metadata"] = serde_json::json!({"states":summary.states,"terminals":summary.terminals,
                        "effects":summary.effects.len(),"entry_terminals":summary.entries.len(),"entry_effects":entry_effects,
                        "accepting_states":summary.accepting.len(),"pushed_symbols":pushes,
                        "serialized_bytes":bincode::serialized_size(summary).ok(),
                        "owned_vector_bytes_excluding_btree_and_allocator":vector_bytes});
                }
                report["sparse_regular"] = self.direct_regular_automaton.is_some().into();
                report["virtual_lexer"] = self.tokenizer.has_any_virtual_runtime().into();
                if let Some(overlay) = &self.static_dynamic_overlay {
                    report["finite_observation_leaves"] = overlay.recursive_static_observation.as_ref()
                        .map_or(0, |observation| observation.leaf_offsets.len().saturating_sub(1)).into();
                    if let Some(observation) = &overlay.recursive_static_observation {
                        report["finite_observation_offsets"] = serde_json::json!(observation.leaf_offsets);
                    }
                    report["component_parsers"] = serde_json::Value::Array(overlay.segmented_parser_components.iter()
                        .map(|component| component.constraint.parser_backend_report()).collect());
                    report["packed_lr_compiler_table_present"] = overlay.recursive_compiler_table.get().is_some().into();
                    report["static_boundary_shards"] = overlay.segmented_parser_components.iter().filter(|component|
                        matches!(component.boundary.as_ref().map(|shard| &shard.backend),
                            Some(super::SegmentedBoundaryShardBackend::StaticParser(_)))).count().into();
                    report["dynamic_boundary_shards"] = overlay.segmented_parser_components.iter().filter(|component|
                        matches!(component.boundary.as_ref().map(|shard| &shard.backend),
                            Some(super::SegmentedBoundaryShardBackend::DynamicDirect))).count().into();
                    report["boundary_candidate_tokens"] = serde_json::Value::Array(
                        overlay.segmented_parser_components.iter().enumerate().map(|(index, component)|
                            serde_json::json!({"component": index, "count": component.boundary.as_ref()
                                .and_then(|shard| shard.candidate_tokens.as_ref()).map(|ids| ids.len())})).collect());
                    if let Some(composition) = &parser.composition {
                        let mut checked = 0usize; let mut sources = 0usize;
                        let mut domains = 0usize; let mut identities = 0usize;
                        let mut offset = 0usize;
                        for component in &overlay.segmented_parser_components {
                            let local = component.constraint.template_parser.as_ref().expect("native component");
                            for terminal in 0..local.terminal_count as usize {
                                let view = &composition.outer_views[offset + terminal];
                                if let Some(nested) = &local.composition {
                                    checked += 1;
                                    sources += usize::from(Arc::ptr_eq(&view.source, &nested.outer_views[terminal].source));
                                    domains += usize::from(Arc::ptr_eq(&view.domain, &nested.outer_views[terminal].domain));
                                } else if component.constraint.ignore_terminal == Some(terminal as u32)
                                    || local.skip_terminals.contains(&(terminal as u32)) {
                                    identities += 1;
                                } else {
                                    checked += 1;
                                    sources += usize::from(Arc::ptr_eq(&view.source,
                                        component.constraint.template_dfas_by_terminal[terminal].as_ref().unwrap()));
                                    domains += usize::from(Arc::ptr_eq(&view.domain, &local.domains[terminal]));
                                }
                            }
                            offset += local.terminal_count as usize;
                        }
                        report["component_sharing"] = serde_json::json!({
                            "checked_retained_terminal_relations": checked,
                            "shared_retained_sources": sources, "shared_retained_domains": domains,
                            "synthetic_ignore_or_skip_relations": identities,
                            "dense_global_certificate_rows": parser.possible.len() + parser.unconditional.len()});
                    }
                }
                report
            }
            None => serde_json::json!({"backend":"lr-table", "lr_table_present":self.table.is_present(),
                "sparse_regular": self.uses_sparse_direct_regular_runtime()}),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::accumulator::TerminalsDisallowed;

    #[test]
    fn concurrent_inventory_preparation_preserves_domains_and_every_runtime_view() {
        for count in [0u32, 1, 7, 32, 65] {
            let programs = (0..count).map(|terminal| {
                Some(compile_terminal_template(terminal, phase_parallel_fixture(terminal)).unwrap())
            }).collect::<Vec<_>>();
            for prepare_runtime in [false, true] {
                let mut reference_domains = Vec::new();
                let mut reference_views = Vec::new();
                for template in &programs {
                    let template = template.as_deref().unwrap();
                    let validated = super::super::commit::template_prepare::TemplatePreparation::new(template).unwrap();
                    validated.validate_alphabet(32).unwrap();
                    reference_domains.push(TemplateDomain::from_validated(&validated).unwrap());
                    if prepare_runtime {
                        reference_views.push(Some(Arc::new(
                            crate::runtime::artifact::FastCommitTemplateDfas::from_prepared(&validated)
                        )));
                    }
                }
                for threads in [1, 4] {
                    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                    for _ in 0..3 {
                        let (domains, views) = pool.install(|| {
                            prepare_terminal_inventory(&programs, 32, prepare_runtime, true)
                        }).unwrap();
                        assert_eq!(domains.len(), reference_domains.len());
                        assert_eq!(format!("{views:?}"), format!("{reference_views:?}"));
                        for (expected, actual) in reference_domains.iter().zip(&domains) {
                            assert_eq!(expected.to_bytes().unwrap(), actual.to_bytes().unwrap());
                            for top in 0..32 {
                                assert_eq!(expected.classify_top(top), actual.classify_top(top));
                                for suffix in [vec![top], vec![top, 3], vec![top, 7, 2]] {
                                    assert_eq!(expected.matches_top_first(suffix.iter().copied()),
                                               actual.matches_top_first(suffix.iter().copied()));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn concurrent_inventory_preparation_rejects_invalid_programs_in_terminal_order() {
        let mut programs = (0..8u32).map(|terminal| {
            Some(compile_terminal_template(terminal, phase_parallel_fixture(terminal)).unwrap())
        }).collect::<Vec<_>>();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        programs[2] = None;
        programs[6] = None;
        for _ in 0..4 {
            let serial = prepare_terminal_inventory(&programs, 32, true, false).unwrap_err().to_string();
            let parallel = pool.install(|| prepare_terminal_inventory(&programs, 32, true, true))
                .unwrap_err().to_string();
            assert_eq!(serial, parallel);
            assert!(serial.contains("terminal 2"));
        }
        programs[2] = Some(compile_terminal_template(2, phase_parallel_fixture(2)).unwrap());
        programs[6] = Some(compile_terminal_template(6, phase_parallel_fixture(6)).unwrap());
        let invalid = Arc::make_mut(programs[1].as_mut().unwrap());
        let unreachable = invalid.pop.add_state();
        invalid.pop.add_transition(unreachable, 0, unreachable);
        let serial = prepare_terminal_inventory(&programs, 32, true, false).unwrap_err().to_string();
        let parallel = pool.install(|| prepare_terminal_inventory(&programs, 32, true, true))
            .unwrap_err().to_string();
        assert_eq!(serial, parallel, "even unreachable malformed graph data must be rejected");
    }

    #[test]
    fn staged_parser_preparation_preserves_runtime_finalization_order() {
        let vocab = crate::Vocab::new(["a", "b", "(", ")", "ab", "aa", " "]
            .into_iter().enumerate().map(|(id, value)| (id as u32, value.as_bytes().to_vec())).collect());
        let grammar = crate::Grammar::glrm(r#"start root; nt root ::= "a" | "(" root ")";"#);
        let dynamic = crate::DynamicConstraint::compile_with_vocab_partition(grammar, &vocab).unwrap();
        let ordinary = dynamic.into_constraints().pop().unwrap();
        let mut reference = ordinary.clone();
        reference.rebuild_dynamic_runtime_caches();
        reference.install_template_parser().unwrap();
        let mut candidate = ordinary;
        // Compiler preparation completes before the Constraint exists.
        // Rebuilding runtime caches must preserve the shared native programs.
        assert!(candidate.has_template_parser() && !candidate.table.is_present());
        assert!(candidate.prepare_template_parser().unwrap().is_none());
        let programs = candidate.template_dfas_by_terminal.clone();
        candidate.rebuild_dynamic_runtime_caches();
        assert!(!candidate.table.is_present());
        for (before, after) in programs.iter().zip(&candidate.template_dfas_by_terminal) {
            assert!(Arc::ptr_eq(before.as_ref().unwrap(), after.as_ref().unwrap()));
        }
        let restored = Constraint::load(candidate.save()).unwrap();
        let external = Constraint::load_with_vocab(candidate.save_with_external_vocab().unwrap(), &vocab).unwrap();
        for constraint in [&candidate, &restored, &external] {
            assert!(constraint.has_template_parser() && !constraint.table.is_present());
            for bytes in [b"".as_slice(), b"a", b"(a)", b"((a))", b"b", b" "] {
                let mut a = reference.start();
                let mut b = constraint.start();
                assert_eq!(a.commit_bytes(bytes).is_ok(), b.commit_bytes(bytes).is_ok());
                assert_eq!(a.is_accepting(), b.is_accepting());
                let mut left = vec![0; reference.mask_len()];
                let mut right = vec![0; constraint.mask_len()];
                a.fill_mask(&mut left);
                b.fill_mask(&mut right);
                assert_eq!(left, right, "prefix {bytes:?}");
            }
        }
    }

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

    #[test]
    fn native_compiler_effect_metadata_survives_reload_and_nested_links_without_tables() {
        use crate::compiler::boundary_stack_support::CompilerEffects;
        let vocab = crate::Vocab::new(["a","b","(",")","ab"]
            .into_iter().enumerate().map(|(id,word)| (id as u32,word.as_bytes().to_vec())).collect());
        for source in [r#"start root; nt root ::= "a" | "ab";"#,
            r#"start root; nt root ::= "a" | "(" root ")";"#] {
            let component = Constraint::compile(crate::Grammar::glrm(source),&vocab).unwrap();
            let parser = component.template_parser.as_ref().unwrap();
            let summary = parser.link_grammar.as_ref().unwrap().stack_effects().unwrap();
            assert_eq!(summary.states,parser.state_count);
            assert!(!summary.effects.is_empty());
            let loaded = Constraint::load(&component.save()).unwrap();
            assert!(!loaded.table.is_present());
            assert_eq!(loaded.template_parser.as_ref().unwrap().link_grammar.as_ref().unwrap()
                .stack_effects().unwrap(),summary);
            // A later link needs only the retained scalar effects and validated
            // entry records. Its nested child offsets remain complete.
            let slot = *summary.entries.iter().find(|(_,row)| !row.is_empty()).unwrap().0;
            let end = summary.states.checked_mul(2).unwrap();
            let terminals = summary.terminals.checked_mul(2).unwrap();
            let linked = CompilerEffects::compose(&[summary,summary],&[0,summary.states],
                &[0,summary.terminals],&[(0,slot,1,0,1,false)],end,terminals).unwrap();
            let nested = CompilerEffects::compose(&[&linked,summary],&[0,end],
                &[0,terminals],&[],end+summary.states,terminals+summary.terminals).unwrap();
            assert!(nested.effects.iter().any(|e| e.source >= end));
            assert_eq!(nested.accepting,summary.accepting);
        }
    }

    fn phase_parallel_fixture(seed: u32) -> DFA {
        use crate::compiler::glr::labels::DEFAULT_LABEL;
        let mut dfa = DFA::new();
        let read_pop = dfa.add_state();
        let read_push = dfa.add_state();
        let final_state = dfa.add_state();
        let top = (seed % 7) as i32;
        dfa.add_transition(0, top, read_pop);
        dfa.add_transition(read_pop, encode_negative_label(top as u32), read_push);
        dfa.add_transition(read_push, encode_negative_label(17), final_state);
        dfa.set_accepting(final_state, true);
        let mut tail = dfa.add_state();
        dfa.add_transition(0, DEFAULT_LABEL, tail);
        for depth in 0..48 {
            let next = dfa.add_state();
            dfa.add_transition(tail, ((seed + depth) % 11) as i32, next);
            tail = next;
        }
        dfa.add_transition(tail, encode_negative_label(19), final_state);
        dfa
    }

    fn assert_identical_phase_rows(
        expected: &crate::runtime::artifact::TemplateDfasByTerminal,
        actual: &crate::runtime::artifact::TemplateDfasByTerminal,
    ) {
        assert_eq!(expected.len(), actual.len());
        for (a, b) in expected.iter().zip(actual) {
            let (a, b) = (a.as_ref().unwrap(), b.as_ref().unwrap());
            assert_eq!(a.pop, b.pop);
            assert_eq!(a.read, b.read);
            assert_eq!(a.push, b.push);
            assert_eq!(a.pop_to_read, b.pop_to_read);
            assert_eq!(a.pop_to_push, b.pop_to_push);
            assert_eq!(a.read_to_push, b.read_to_push);
        }
    }

    #[test]
    fn exact_characterization_groups_match_independent_templates_domains_and_views() {
        use glrmask_parser_dwa::__private::templates::characterize::{
            InitialEscape, StackMatcher, TerminalCharacterization,
        };
        let characterizations = (0..8u32).map(|terminal| {
            let top = terminal % 2;
            (terminal, TerminalCharacterization {
                escapes: vec![
                    InitialEscape { pop: vec![StackMatcher::State(top)], pushes: vec![top, 17] },
                    InitialEscape { pop: vec![StackMatcher::Any], pushes: vec![19] },
                    InitialEscape { pop: Vec::new(), pushes: vec![20] },
                ],
                reduces: Vec::new(), nt_escapes: Vec::new(), nt_rereduces: Vec::new(),
                all_nts: BTreeSet::new(),
            })
        }).collect::<BTreeMap<_, _>>();
        let reference = Templates::dfas_from_characterizations(&characterizations).into_iter()
            .map(|(terminal, raw)| Some(compile_terminal_template(terminal, raw).unwrap()))
            .collect::<Vec<_>>();
        let (groups, profile) = Templates::grouped_dfas_from_characterizations(&characterizations);
        assert_eq!(profile.unique_characterizations, 2);
        assert_eq!(profile.quotient_hits, 6);
        let (reference_domains, reference_views) = prepare_terminal_inventory(&reference, 32, true, false).unwrap();
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            for parallel in [false, true] {
                let actual = pool.install(|| {
                    compile_terminal_template_groups_with_parallelism(groups.clone(), 8, parallel)
                }).unwrap();
                assert_identical_phase_rows(&reference, &actual);
                for terminal in 2..8 {
                    assert!(Arc::ptr_eq(actual[terminal].as_ref().unwrap(), actual[terminal % 2].as_ref().unwrap()));
                }
                assert!(!Arc::ptr_eq(actual[0].as_ref().unwrap(), actual[1].as_ref().unwrap()));
                let (domains, views) = prepare_terminal_inventory(&actual, 32, true, false).unwrap();
                assert_eq!(format!("{views:?}"), format!("{reference_views:?}"));
                for (expected, domain) in reference_domains.iter().zip(&domains) {
                    assert_eq!(expected.to_bytes().unwrap(), domain.to_bytes().unwrap());
                    for top in 0..32 {
                        for suffix in [vec![top], vec![top, 3], vec![top, 7, 2]] {
                            assert_eq!(expected.matches_top_first(suffix.iter().copied()),
                                       domain.matches_top_first(suffix.iter().copied()));
                        }
                    }
                }
            }
        }
        let automatic = compile_terminal_template_groups(groups, 8).unwrap();
        assert_identical_phase_rows(&reference, &automatic);
    }

    #[test]
    fn grouped_phase_transform_preserves_default_shadow_and_empty_relations() {
        use crate::compiler::glr::labels::DEFAULT_LABEL;
        let mut identity = DFA::new();
        identity.set_accepting(0, true);
        let mut shadow = DFA::new();
        let pushed = shadow.add_state();
        let accepted = shadow.add_state();
        shadow.add_transition(0, 7, pushed);
        shadow.add_transition(0, DEFAULT_LABEL, pushed);
        shadow.add_transition(pushed, encode_negative_label(7), accepted);
        shadow.set_accepting(accepted, true);
        let groups = vec![
            TemplateDfaGroup { terminals: vec![1, 4], dfa: shadow },
            TemplateDfaGroup { terminals: vec![6, 7], dfa: identity },
            TemplateDfaGroup { terminals: vec![0, 3], dfa: phase_parallel_fixture(0) },
            TemplateDfaGroup { terminals: vec![2, 5], dfa: DFA::new() },
        ];
        let raw = groups.iter().flat_map(|group| group.terminals.iter()
            .map(move |&terminal| (terminal, group.dfa.clone()))).collect::<BTreeMap<_, _>>();
        let reference = compile_terminal_template_rows_with_parallelism(raw, 8, false).unwrap();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        for parallel in [false, true] {
            let actual = pool.install(|| {
                compile_terminal_template_groups_with_parallelism(groups.clone(), 8, parallel)
            }).unwrap();
            assert_identical_phase_rows(&reference, &actual);
            assert!(Arc::ptr_eq(actual[0].as_ref().unwrap(), actual[3].as_ref().unwrap()));
            let template = actual[1].as_ref().unwrap();
            let root = &template.pop.states[template.pop.start_state as usize];
            assert!(root.transitions.contains_key(&DEFAULT_LABEL));
            let dead = root.transitions[&7] as usize;
            assert!(!template.pop.states[dead].is_accepting);
            assert!(template.pop.states[dead].transitions.is_empty());
            assert!(template.pop_to_read[dead].is_none());
            assert!(template.pop_to_push[dead].is_none());
        }
    }

    #[test]
    fn grouped_phase_transform_preserves_smallest_error_and_rejects_invalid_inventory() {
        let mut unsupported = DFA::new();
        let pushed = unsupported.add_state();
        let after = unsupported.add_state();
        unsupported.add_transition(0, encode_negative_label(3), pushed);
        unsupported.add_transition(pushed, 9, after);
        unsupported.set_accepting(after, true);
        // Failed groups deliberately arrive in reverse terminal order.
        let groups = vec![
            TemplateDfaGroup { terminals: vec![6], dfa: unsupported.clone() },
            TemplateDfaGroup { terminals: vec![0, 1, 3, 4, 5, 7], dfa: phase_parallel_fixture(0) },
            TemplateDfaGroup { terminals: vec![2], dfa: unsupported },
        ];
        let raw = groups.iter().flat_map(|group| group.terminals.iter()
            .map(move |&terminal| (terminal, group.dfa.clone()))).collect::<BTreeMap<_, _>>();
        let reference = compile_terminal_template_rows_with_parallelism(raw, 8, false)
            .unwrap_err().to_string();
        assert!(reference.contains("terminal 2"));
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        for parallel in [false, true] {
            let actual = pool.install(|| {
                compile_terminal_template_groups_with_parallelism(groups.clone(), 8, parallel)
            }).unwrap_err().to_string();
            assert_eq!(actual, reference);
            for terminals in [vec![], vec![0], vec![0, 0], vec![0, 2], vec![1, 0]] {
                let invalid = vec![TemplateDfaGroup { terminals, dfa: phase_parallel_fixture(0) }];
                assert!(compile_terminal_template_groups_with_parallelism(invalid, 2, parallel).is_err());
            }
            let duplicate = vec![
                TemplateDfaGroup { terminals: vec![0, 1], dfa: phase_parallel_fixture(0) },
                TemplateDfaGroup { terminals: vec![1], dfa: phase_parallel_fixture(1) },
            ];
            assert!(compile_terminal_template_groups_with_parallelism(duplicate, 2, parallel).is_err());
            assert!(compile_terminal_template_groups_with_parallelism(Vec::new(), 1, parallel).is_err());
            assert!(compile_terminal_template_groups_with_parallelism(Vec::new(), 0, parallel).unwrap().is_empty());
        }
    }

    #[test]
    fn parallel_terminal_phases_match_literal_serial_transformations() {
        for count in [0u32, 1, 7, 32, 65] {
            let inputs = (0..count).map(|terminal| (terminal, phase_parallel_fixture(terminal)))
                .collect::<BTreeMap<_, _>>();
            let reference = inputs.iter().map(|(&terminal, dfa)| {
                let specialized = specialize_template_dfa_defaults_for_commit_split_input(dfa);
                Some(Arc::new(try_split_commit_template_dfas(&specialized)
                    .unwrap_or_else(|| panic!("fixture terminal {terminal} must split"))))
            }).collect::<Vec<_>>();
            for threads in [1, 4] {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                for _ in 0..3 {
                    let actual = pool.install(|| {
                        compile_terminal_template_rows_with_parallelism(inputs.clone(), count, true)
                    }).unwrap();
                    assert_identical_phase_rows(&reference, &actual);
                }
            }
            let automatic = compile_terminal_template_rows(inputs, count).unwrap();
            assert_identical_phase_rows(&reference, &automatic);
        }
    }

    #[test]
    fn parallel_terminal_phases_preserve_first_error_and_inventory_checks() {
        let mut unsupported = DFA::new();
        let pushed = unsupported.add_state();
        let popped_after_push = unsupported.add_state();
        unsupported.add_transition(0, encode_negative_label(3), pushed);
        unsupported.add_transition(pushed, 9, popped_after_push);
        unsupported.set_accepting(popped_after_push, true);
        let inputs = (0..8u32).map(|terminal| {
            let dfa = if terminal == 2 || terminal == 6 { unsupported.clone() }
                else { phase_parallel_fixture(terminal) };
            (terminal, dfa)
        }).collect::<BTreeMap<_, _>>();
        let reference = compile_terminal_template_rows_with_parallelism(inputs.clone(), 8, false)
            .unwrap_err().to_string();
        assert!(reference.contains("terminal 2"));
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        for _ in 0..8 {
            let actual = pool.install(|| {
                compile_terminal_template_rows_with_parallelism(inputs.clone(), 8, true)
            }).unwrap_err().to_string();
            assert_eq!(actual, reference);
        }
        for parallel in [false, true] {
            assert!(compile_terminal_template_rows_with_parallelism(BTreeMap::new(), 1, parallel).is_err());
            assert!(compile_terminal_template_rows_with_parallelism(
                BTreeMap::from([(1, phase_parallel_fixture(1))]), 1, parallel,
            ).is_err());
        }
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
                domains.push(Arc::new(compile_domain(&t).unwrap()));
            }
            for completion in [pop_word(&[0], true), pop_word(&[], true), pop_word(&[], false)] {
                let completion = Arc::new(compile_domain(&completion).unwrap());
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
