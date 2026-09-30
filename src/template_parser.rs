//! Data-only parsers over finite acyclic stack-action automata.
//!
//! A provider describes one POP/READ/PUSH relation per lexer terminal and a
//! completion relation. GLRMask derives input-only admission automata and bulk
//! top-of-stack certificates, then executes the same lexer, shared GSS, commit,
//! and dynamic masking engines used by built-in grammars. There are no parser
//! callbacks on the token hot path and no LR table, including during assembly.
//!
//! Stack symbols belong to the provider. A sequence starts at the one-symbol
//! stack `[0]`; source words are inspected top first and PUSH words are appended
//! bottom first. Each terminal relation is acyclic, but repeated terminal
//! advances can recognize recursive languages with unbounded stack depth.
//!
//! The direct constructor currently uses the shared dynamic mask engine. It
//! does not synthesize a static token-mask DWA or support component composition.
//! Built-in grammars can separately select [`crate::ParserBackend::TemplateDfa`]
//! with their ordinary compile options, including the static mask engine.

mod static_compile;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::automata::unweighted_u32::dfa::{DFA, DFAState};
use crate::compiler::glr::labels::{DEFAULT_LABEL, encode_negative_label};
use crate::runtime::CommitTemplateDfas;
use crate::runtime::parser_backend::TemplateParser;
use crate::{Constraint, Error, Result, Vocab};

const MAX_CERTIFICATE_BYTES: usize = 256 * 1024 * 1024;
const MAX_AUTOMATON_STATES: usize = 1_000_000;
const MAX_AUTOMATON_EDGES: usize = 4_000_000;

/// One transition label. DEFAULT is permitted only in a POP automaton.
/// An explicit edge always shadows DEFAULT, including an explicit dead edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum StackLabel {
    Symbol(u32),
    Default,
}

/// Deterministic labeled edge between states of the same phase automaton.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StackTransition {
    pub label: StackLabel,
    pub target: u32,
}

/// An accepting state emits the current stack; outgoing transitions can emit
/// additional stacks. Accepting is not an instruction to stop the traversal.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StackState {
    pub accepting: bool,
    pub transitions: Vec<StackTransition>,
}

/// A finite acyclic deterministic phase graph. Every state, including an
/// unreachable one, is validated for bounds, duplicate labels and cycles.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StackDfa {
    pub start: u32,
    pub states: Vec<StackState>,
}

impl StackDfa {
    /// An explicit empty relation, rather than a missing terminal definition.
    pub fn reject() -> Self {
        Self { start: 0, states: vec![StackState::default()] }
    }

    /// The empty action word; it accepts without inspecting the source stack.
    pub fn accept() -> Self {
        Self { start: 0, states: vec![StackState { accepting: true, transitions: Vec::new() }] }
    }

    /// A linear phase graph accepting exactly this action word.
    pub fn word(labels: impl IntoIterator<Item = StackLabel>) -> Self {
        let mut states = vec![StackState::default()];
        for label in labels {
            let target = states.len() as u32;
            states.last_mut().unwrap().transitions.push(StackTransition { label, target });
            states.push(StackState::default());
        }
        states.last_mut().unwrap().accepting = true;
        Self { start: 0, states }
    }
}

/// One terminal's complete relation, not a preferred path with a fallback.
///
/// POP consumes a matching source-stack symbol. READ tests the current top
/// without consuming it; consecutive READ edges test that same top. PUSH appends
/// symbols. Optional phase links are epsilon edges, independent of whether a
/// READ can inspect a symbol. Unspecified trailing links are absent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StackTemplate {
    pub pop: StackDfa,
    pub read: StackDfa,
    pub push: StackDfa,
    pub pop_to_read: Vec<Option<u32>>,
    pub pop_to_push: Vec<Option<u32>>,
    pub read_to_push: Vec<Option<u32>>,
}

impl StackTemplate {
    pub fn reject() -> Self {
        Self { pop: StackDfa::reject(), read: StackDfa::reject(), push: StackDfa::reject(),
            pop_to_read: Vec::new(), pop_to_push: Vec::new(), read_to_push: Vec::new() }
    }

    /// Identity relation, useful for an ignored whitespace terminal.
    pub fn identity() -> Self {
        Self { pop: StackDfa::accept(), ..Self::reject() }
    }

    /// Consume a top-first stack prefix and append a bottom-first suffix.
    pub fn rewrite(pop: impl IntoIterator<Item = StackLabel>, push: impl IntoIterator<Item = u32>) -> Self {
        let mut pop = StackDfa::word(pop);
        let last = pop.states.len() - 1;
        pop.states[last].accepting = false;
        let mut links = vec![None; pop.states.len()]; links[last] = Some(0);
        Self { pop, push: StackDfa::word(push.into_iter().map(StackLabel::Symbol)),
            pop_to_push: links, ..Self::reject() }
    }

    /// Test one of `tops` without popping it, then append `push` in order.
    /// An empty PUSH suffix implements an input-only top predicate.
    pub fn read_top_and_push(tops: impl IntoIterator<Item = u32>, push: impl IntoIterator<Item = u32>) -> Self {
        let mut read = StackDfa::reject(); read.states.push(StackState::default());
        read.states[0].transitions = tops.into_iter().collect::<BTreeSet<_>>().into_iter()
            .map(|top| StackTransition { label: StackLabel::Symbol(top), target: 1 }).collect();
        Self { read, push: StackDfa::word(push.into_iter().map(StackLabel::Symbol)),
            pop_to_read: vec![Some(0)], read_to_push: vec![None, Some(0)], ..Self::reject() }
    }
}

/// Provider-owned stack alphabet and complete per-terminal action relations.
/// Terminal indices are positional and match the terminal pattern vector.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParserDefinition {
    pub stack_symbol_count: u32,
    pub terminals: Vec<StackTemplate>,
    /// Only this relation's input domain is queried at end of generation.
    /// Completion never executes an output program or an implicit EOF shift.
    pub completion: StackTemplate,
}

/// A compile-time data provider. Implementations need not implement parsing,
/// GSS operations, masking, serialization, or any runtime callback.
pub trait ParserProvider {
    fn parser_definition(&self) -> Result<ParserDefinition>;
}

impl ParserProvider for ParserDefinition {
    fn parser_definition(&self) -> Result<ParserDefinition> { Ok(self.clone()) }
}

/// Independent lexer-terminal language. Empty/nullable terminals are rejected
/// by this constructor; use a nonempty whitespace pattern with identity action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerminalPattern {
    Literal(Vec<u8>),
    Regex { pattern: String, utf8: bool },
}

impl TerminalPattern {
    pub fn literal(bytes: impl Into<Vec<u8>>) -> Self { Self::Literal(bytes.into()) }
    pub fn regex(pattern: impl Into<String>) -> Self { Self::Regex { pattern: pattern.into(), utf8: true } }
}

/// Terminal patterns in the same order as [`ParserDefinition::terminals`].
/// Ignoring is opt-in and requires the canonical identity relation for that
/// terminal; silently discarding a nonidentity action is never permitted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexerDefinition {
    pub terminals: Vec<TerminalPattern>,
    pub ignore_terminal: Option<u32>,
}

impl LexerDefinition {
    pub fn new(terminals: Vec<TerminalPattern>) -> Self { Self { terminals, ignore_terminal: None } }
    pub fn ignoring(mut self, terminal: u32) -> Self { self.ignore_terminal = Some(terminal); self }
}

/// Generation controls for a data-only parser. Both choices use template
/// parser primitives with no LR backend, including during compilation.
#[derive(Debug, Clone)]
pub struct TemplateBuildOptions {
    end_tokens: Vec<u32>,
    optimization: crate::Optimization,
}

impl Default for TemplateBuildOptions {
    fn default() -> Self {
        Self { end_tokens: Vec::new(), optimization: crate::Optimization::FastBuild }
    }
}

impl TemplateBuildOptions {
    /// Select eager static masking with FastRuntime, or the shared dynamic
    /// mask engine with FastBuild/Auto. Default is FastBuild.
    pub fn optimization(mut self, optimization: crate::Optimization) -> Self {
        self.optimization = optimization;
        self
    }

    pub fn end_tokens(mut self, ids: impl IntoIterator<Item = u32>) -> Self {
        self.end_tokens = ids.into_iter().collect(); self
    }
}

/// Immutable validated parser program, reusable with multiple vocabularies.
/// Derived admission and PUSH caches are computed by GLRMask, not the provider.
#[derive(Debug, Clone)]
pub struct ParserProgram {
    parser: Arc<TemplateParser>,
    templates: Arc<[Option<Arc<CommitTemplateDfas>>]>,
    identity_terminals: BTreeSet<u32>,
}

fn fail(message: impl Into<String>) -> Error { Error::Compilation(message.into()) }

fn phase_graph(input: &StackDfa, symbols: u32, phase: &'static str,
    states_left: &mut usize, edges_left: &mut usize) -> Result<DFA> {
    if input.states.is_empty() || input.start as usize >= input.states.len() {
        return Err(fail(format!("{phase} automaton has an invalid start or no states")));
    }
    *states_left = states_left.checked_sub(input.states.len()).ok_or_else(||fail("template state budget exceeded"))?;
    let mut graph = DFA { states: Vec::with_capacity(input.states.len()), start_state: input.start };
    for (i, state) in input.states.iter().enumerate() {
        *edges_left = edges_left.checked_sub(state.transitions.len()).ok_or_else(||fail("template edge budget exceeded"))?;
        let mut transitions = BTreeMap::new();
        for edge in &state.transitions {
            if edge.target as usize >= input.states.len() { return Err(fail(format!("{phase} state {i}: target out of range"))); }
            let label = match edge.label {
                StackLabel::Default if phase == "POP" => DEFAULT_LABEL,
                StackLabel::Default => return Err(fail(format!("DEFAULT is only valid in POP, not {phase}"))),
                StackLabel::Symbol(symbol) if symbol < symbols => if phase == "PUSH" { encode_negative_label(symbol) } else { symbol as i32 },
                StackLabel::Symbol(_) => return Err(fail(format!("{phase} state {i}: stack symbol out of range"))),
            };
            if transitions.insert(label, edge.target).is_some() { return Err(fail(format!("{phase} state {i}: duplicate transition label"))); }
        }
        graph.states.push(DFAState { is_accepting: state.accepting, transitions });
    }
    if !graph.compute_is_acyclic() { return Err(fail(format!("{phase} automaton must be acyclic"))); }
    Ok(graph)
}

fn convert(input: &StackTemplate, symbols: u32, states_left: &mut usize, edges_left: &mut usize) -> Result<CommitTemplateDfas> {
    for (links, source, target, name) in [
        (&input.pop_to_read, input.pop.states.len(), input.read.states.len(), "POP-to-READ"),
        (&input.pop_to_push, input.pop.states.len(), input.push.states.len(), "POP-to-PUSH"),
        (&input.read_to_push, input.read.states.len(), input.push.states.len(), "READ-to-PUSH"),
    ] {
        if links.len() > source || links.iter().flatten().any(|&i| i as usize >= target) {
            return Err(fail(format!("{name} epsilon link is out of range")));
        }
    }
    Ok(CommitTemplateDfas {
        pop: phase_graph(&input.pop, symbols, "POP", states_left, edges_left)?,
        read: phase_graph(&input.read, symbols, "READ", states_left, edges_left)?,
        push: phase_graph(&input.push, symbols, "PUSH", states_left, edges_left)?,
        pop_to_read: input.pop_to_read.clone(), pop_to_push: input.pop_to_push.clone(),
        read_to_push: input.read_to_push.clone(),
    })
}

impl ParserProgram {
    pub fn from_provider(provider: &(impl ParserProvider + ?Sized)) -> Result<Self> {
        Self::new(provider.parser_definition()?)
    }

    pub fn new(definition: ParserDefinition) -> Result<Self> {
        let symbols = definition.stack_symbol_count;
        if symbols == 0 || symbols >= DEFAULT_LABEL as u32 { return Err(fail("stack alphabet must include initial symbol 0 and exclude reserved label values")); }
        let terminals = u32::try_from(definition.terminals.len()).map_err(|_|fail("too many terminals"))?;
        if terminals == 0 { return Err(fail("at least one explicit terminal relation is required")); }
        if u64::from(symbols) * (u64::from(terminals) + 1) > 16_000_000 {
            return Err(fail("template top-certificate construction exceeds 16 million symbol/terminal pairs"));
        }
        let row_bytes = (terminals as usize + 1).div_ceil(64).checked_mul(16)
            .and_then(|n|n.checked_add(2 * std::mem::size_of::<crate::ds::bitset::BitSet>()))
            .and_then(|n|n.checked_mul(symbols as usize)).ok_or_else(||fail("template certificate size overflow"))?;
        if row_bytes > MAX_CERTIFICATE_BYTES { return Err(fail("template admission certificate budget exceeds 256 MiB")); }
        let mut states_left = MAX_AUTOMATON_STATES;
        let mut edges_left = MAX_AUTOMATON_EDGES;
        let identity_terminals = definition.terminals.iter().enumerate()
            .filter_map(|(i, t)| (*t == StackTemplate::identity()).then_some(i as u32)).collect();
        let templates = definition.terminals.iter().map(|t|convert(t, symbols, &mut states_left, &mut edges_left)
            .map(|t|Some(Arc::new(t)))).collect::<Result<Vec<_>>>()?;
        let completion = convert(&definition.completion, symbols, &mut states_left, &mut edges_left)?;
        // TemplateDomain validates every epsilon link and phase shape while
        // deriving the exact input projection; no provider certificate is trusted.
        let parser = TemplateParser::compile(symbols, terminals, BTreeSet::new(), &templates, completion)?;
        Ok(Self { parser: Arc::new(parser), templates: templates.into(), identity_terminals })
    }

    pub fn stack_symbol_count(&self) -> u32 { self.parser.state_count }
    pub fn terminal_count(&self) -> u32 { self.parser.terminal_count }

    pub fn compile(&self, lexer: &LexerDefinition, vocab: &Vocab) -> Result<Constraint> {
        self.compile_with(lexer, vocab, TemplateBuildOptions::default())
    }

    pub fn compile_with(&self, lexer: &LexerDefinition, vocab: &Vocab,
        options: TemplateBuildOptions) -> Result<Constraint> {
        if lexer.terminals.len() != self.templates.len() { return Err(fail("lexer and parser terminal counts must agree exactly")); }
        if let Some(ignore) = lexer.ignore_terminal {
            if !self.identity_terminals.contains(&ignore) { return Err(fail("ignored terminal must exist and use StackTemplate::identity()")); }
        }
        let mut expressions = Vec::with_capacity(lexer.terminals.len());
        for (i, pattern) in lexer.terminals.iter().enumerate() {
            let expr = match pattern {
                TerminalPattern::Literal(bytes) => crate::automata::regex::Expr::U8Seq(bytes.clone()),
                TerminalPattern::Regex { pattern, utf8 } => {
                    crate::automata::lexer::regex::validate_regular_regex(pattern).map_err(fail)?;
                    crate::automata::lexer::regex::parse_regex(pattern, *utf8)
                }
            };
            if expr.is_nullable() { return Err(fail(format!("terminal {i} is nullable; data-only lexer terminals must consume bytes"))); }
            expressions.push(expr);
        }
        let names = (0..expressions.len()).map(|i|format!("terminal_{i}")).collect::<Vec<_>>();
        let tokenizer = crate::compiler::pipeline::build_tokenizer_from_exprs(&expressions, Some(&names));
        let constraint = match options.optimization {
            crate::Optimization::Auto | crate::Optimization::FastBuild => {
                crate::dynamic_constraint::DynamicConstraint::from_template_runtime_parts(
                    tokenizer, names, lexer.ignore_terminal, self.templates.to_vec(), self.parser.clone(), vocab,
                )
            }
            crate::Optimization::FastRuntime => {
                crate::error::catch_internal_invariant(|| {
                    static_compile::compile(self, tokenizer, lexer.ignore_terminal, vocab)
                })??
            }
        };
        constraint.with_end_tokens(&options.end_tokens)
    }
}
