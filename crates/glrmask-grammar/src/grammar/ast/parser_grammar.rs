//! Resolved parser expressions. Lexical languages stay on the host side of
//! this boundary; parser compilers receive terminal identities, not regexes.
use super::*;
mod cfg;
pub use cfg::{CfgRecursion, FlatParserGrammar, ParserAnalysis, ParserProduction, ParserSymbol};

/// A parser expression over resolved terminal and rule coordinates.
/// `Choice([])` is the empty language; `Epsilon` and `Sequence([])` are epsilon.
/// References retain sharing and recursion instead of expanding rule bodies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParserExpr {
    Terminal(u32),
    Nonterminal(u32),
    Epsilon,
    Sequence(Vec<ParserExpr>),
    Choice(Vec<ParserExpr>),
    Quantified(Box<ParserExpr>, Quantifier),
    /// Item quantifiers bind to the separator, not to an item's body. Omitting
    /// an optional item also omits its separator. `allow_empty` neither makes
    /// required groups optional nor permits trailing separators.
    SeparatedSequence {
        items: Vec<(ParserExpr, Option<Quantifier>)>,
        separator: Box<ParserExpr>,
        allow_empty: bool,
    },
    /// A complete rule's expression-labeled graph. Epsilon transitions and
    /// shared labels are preserved without determinization or path expansion.
    Automaton(ParserAutomaton),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParserAutomaton {
    pub start_states: Vec<u32>,
    pub states: Vec<ParserAutomatonState>,
    pub symbols: Vec<ParserExpr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParserAutomatonState {
    pub accepting: bool,
    /// `(symbol index, target state)` pairs, including nondeterministic edges.
    pub transitions: Vec<(u32, u32)>,
    pub epsilons: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParserRule {
    pub name: String,
    pub expr: ParserExpr,
}

/// Immutable parser-only grammar, shared cheaply across compiler experiments.
/// Rule and terminal IDs index the corresponding slices. There are no lexer
/// expressions, LR states, vocabulary token IDs, or mandated CFG normal forms.
#[derive(Debug, Clone)]
pub struct ParserGrammar {
    data: Arc<ParserGrammarData>,
}

#[derive(Debug)]
struct ParserGrammarData {
    rules: Vec<ParserRule>,
    start: u32,
    terminal_names: Vec<String>,
    ignore_terminal: Option<u32>,
    left_cfg: std::sync::OnceLock<Result<Arc<FlatParserGrammar>, String>>,
    right_cfg: std::sync::OnceLock<Result<Arc<FlatParserGrammar>, String>>,
    analysis: std::sync::OnceLock<Result<Arc<ParserAnalysis>, String>>,
}

impl ParserGrammar {
    pub fn rules(&self) -> &[ParserRule] { &self.data.rules }
    pub fn start(&self) -> u32 { self.data.start }
    pub fn terminal_names(&self) -> &[String] { &self.data.terminal_names }
    pub fn terminal_count(&self) -> u32 { self.data.terminal_names.len() as u32 }
    /// Global ignored terminal, requiring an identity action in the result.
    /// It does not consume a parser-language symbol between ordinary terminals.
    pub fn ignore_terminal(&self) -> Option<u32> { self.data.ignore_terminal }
}

fn error(message: impl Into<String>) -> GlrMaskError {
    GlrMaskError::GrammarParse(message.into())
}

struct Resolver<'a> {
    lexical: Lowerer<'a>,
    rule_ids: FxHashMap<String, u32>,
    named_terminals: FxHashMap<String, (u32, bool)>,
    remaining_nodes: usize,
}

impl Resolver<'_> {
    fn terminal(&mut self, name: &str, expression: Expr) -> (u32, bool) {
        let nullable = expression.is_nullable();
        let nonempty = if nullable {
            Expr::Exclude { expr: Box::new(expression), exclude: Box::new(Expr::Epsilon) }.optimize()
        } else { expression };
        (self.lexical.register_terminal_expr(name, nonempty), nullable)
    }

    fn atom(terminal: u32, nullable: bool) -> ParserExpr {
        if nullable { ParserExpr::Choice(vec![ParserExpr::Epsilon, ParserExpr::Terminal(terminal)]) }
        else { ParserExpr::Terminal(terminal) }
    }

    fn quantifier(quantifier: &Quantifier) -> Result<(), GlrMaskError> {
        if quantifier.max().is_some_and(|max| max < quantifier.min()) {
            return Err(error("parser expression repetition maximum is below its minimum"));
        }
        Ok(())
    }

    fn expression(&mut self, expression: &GrammarExpr, depth: usize) -> Result<ParserExpr, GlrMaskError> {
        if depth > 512 { return Err(error("parser expression nesting exceeds 512 levels")); }
        self.remaining_nodes = self.remaining_nodes.checked_sub(1)
            .ok_or_else(|| error("resolved parser grammar exceeds its expression-node budget"))?;
        let next = depth + 1;
        Ok(match expression {
            GrammarExpr::Ref(name) => {
                if let Some(&(terminal, nullable)) = self.named_terminals.get(name) {
                    Self::atom(terminal, nullable)
                } else if let Some(&rule) = self.rule_ids.get(name) {
                    ParserExpr::Nonterminal(rule)
                } else {
                    return Err(error(format!("unknown or internal-only parser rule {name:?}")));
                }
            }
            GrammarExpr::Grouped(inner) => return self.expression(inner, next),
            GrammarExpr::Sequence(parts) => ParserExpr::Sequence(parts.iter()
                .map(|part| self.expression(part, next)).collect::<Result<_, _>>()?),
            GrammarExpr::Choice(parts) => ParserExpr::Choice(parts.iter()
                .map(|part| self.expression(part, next)).collect::<Result<_, _>>()?),
            GrammarExpr::Epsilon => ParserExpr::Epsilon,
            GrammarExpr::Quantified(body, quantifier) => {
                Self::quantifier(quantifier)?;
                ParserExpr::Quantified(Box::new(self.expression(body, next)?), quantifier.clone())
            }
            GrammarExpr::SeparatedSequence { items, separator, allow_empty } => {
                let items = items.iter().map(|(body, quantifier)| {
                    if let Some(quantifier) = quantifier { Self::quantifier(quantifier)?; }
                    Ok((self.expression(body, next)?, quantifier.clone()))
                }).collect::<Result<_, GlrMaskError>>()?;
                ParserExpr::SeparatedSequence { items,
                    separator: Box::new(self.expression(separator, next)?), allow_empty: *allow_empty }
            }
            GrammarExpr::Exclude { .. } => {
                if let Some(filtered) = self.lexical.exact_nonterminal_subtraction_expr(expression)? {
                    return self.expression(&filtered, next);
                }
                let lexical = self.lexical.resolve_terminal_expr(None, expression)?;
                let (terminal, nullable) = self.terminal("<lexical subtraction>", lexical);
                Self::atom(terminal, nullable)
            }
            GrammarExpr::Literal(bytes) if bytes.is_empty() => ParserExpr::Epsilon,
            GrammarExpr::Literal(bytes) => ParserExpr::Terminal(self.lexical.literal_terminal_id(bytes)),
            GrammarExpr::SpecialToken(token) => ParserExpr::Terminal(
                self.lexical.special_terminal_id("<exact token>", *token)),
            GrammarExpr::ExprNFA(graph) => {
                let count = graph.nfa.states.len();
                self.remaining_nodes = self.remaining_nodes.checked_sub(count)
                    .ok_or_else(|| error("parser automaton exceeds its state budget"))?;
                if graph.nfa.start_states.iter().any(|&id| id as usize >= count) {
                    return Err(error("parser automaton start state is outside its graph"));
                }
                let mut states = Vec::with_capacity(count);
                for state in &graph.nfa.states {
                    let mut transitions = Vec::new();
                    for (&label, targets) in &state.transitions {
                        if label < 0 || label as usize >= graph.symbols.len() {
                            return Err(error("parser automaton transition has an invalid symbol index"));
                        }
                        for &target in targets {
                            if target as usize >= count { return Err(error("parser automaton edge leaves its graph")); }
                            transitions.push((label as u32, target));
                        }
                    }
                    if state.epsilons.iter().any(|&id| id as usize >= count) {
                        return Err(error("parser automaton epsilon leaves its graph"));
                    }
                    self.remaining_nodes = self.remaining_nodes.checked_sub(transitions.len() + state.epsilons.len())
                        .ok_or_else(|| error("parser automaton exceeds its edge budget"))?;
                    states.push(ParserAutomatonState { accepting: state.is_accepting,
                        transitions, epsilons: state.epsilons.clone() });
                }
                let symbols = graph.symbols.iter().map(|symbol| self.expression(symbol, next))
                    .collect::<Result<_, _>>()?;
                ParserExpr::Automaton(ParserAutomaton { start_states: graph.nfa.start_states.clone(), states, symbols })
            }
            GrammarExpr::CharClass { .. } | GrammarExpr::RawRegex(_) | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte | GrammarExpr::Intersect { .. } => {
                // The existing lexical resolver rejects nonterminal intersection
                // and recursive lexical references rather than approximating them.
                let lexical = self.lexical.resolve_terminal_expr(None, expression)?;
                let (terminal, nullable) = self.terminal("<lexical expression>", lexical);
                Self::atom(terminal, nullable)
            }
        })
    }
}

/// Workspace bridge: the second result is a lexical inventory only, with no
/// productions or parser-state analysis. Do not run the ordinary CFG lowerer
/// before an external constructor receives the first result.
#[doc(hidden)]
pub fn resolve_parser_grammar(grammar: &NamedGrammar) -> Result<(ParserGrammar, GrammarDef), GlrMaskError> {
    validate_expr_nfa_placement(grammar)?;
    let mut names = FxHashSet::default();
    for rule in &grammar.rules {
        if !names.insert(rule.name.as_str()) { return Err(error(format!("duplicate rule {:?}", rule.name))); }
    }
    let mut lexical = Lowerer::new();
    lexical.named_rule_exprs = grammar.rules.iter().map(|rule| (rule.name.clone(), &rule.expr)).collect();
    lexical.named_rule_is_terminal = grammar.rules.iter().map(|rule| (rule.name.clone(), rule.is_terminal)).collect();
    lexical.terminal_bodies = grammar.rules.iter().filter(|rule| rule.is_terminal)
        .map(|rule| (rule.name.clone(), &rule.expr)).collect();
    let rule_ids = grammar.rules.iter().filter(|rule| !rule.is_terminal).enumerate()
        .map(|(index, rule)| (rule.name.clone(), index as u32)).collect();
    let mut resolver = Resolver { lexical, rule_ids, named_terminals: FxHashMap::default(), remaining_nodes: 4_000_000 };
    for rule in grammar.rules.iter().filter(|rule| rule.is_terminal) {
        if rule.is_internal {
            if matches!(rule.expr, GrammarExpr::SpecialToken(_)) {
                return Err(error("an internal lexical helper cannot be an exact token"));
            }
            continue;
        }
        let terminal = if let GrammarExpr::SpecialToken(token) = &rule.expr {
            (resolver.lexical.special_terminal_id(&rule.name, *token), false)
        } else {
            let expression = resolver.lexical.resolve_terminal_expr(Some(&rule.name), &rule.expr)?;
            resolver.terminal(&rule.name, expression)
        };
        resolver.named_terminals.insert(rule.name.clone(), terminal);
    }
    let mut rules = Vec::with_capacity(resolver.rule_ids.len() + 1);
    for rule in grammar.rules.iter().filter(|rule| !rule.is_terminal) {
        rules.push(ParserRule { name: rule.name.clone(), expr: resolver.expression(&rule.expr, 0)? });
    }
    let start = if let Some(&start) = resolver.rule_ids.get(&grammar.start) { start }
    else if let Some(&(terminal, nullable)) = resolver.named_terminals.get(&grammar.start) {
        let start = rules.len() as u32;
        rules.push(ParserRule { name: grammar.start.clone(), expr: Resolver::atom(terminal, nullable) });
        start
    } else { return Err(error(format!("undefined or internal start rule {:?}", grammar.start))); };
    let ignore_terminal = grammar.ignore.as_ref().map(|name| {
        let &(terminal, _) = resolver.named_terminals.get(name)
            .ok_or_else(|| error(format!("ignore rule {name:?} is not an emitting terminal")))?;
        if matches!(resolver.lexical.terminals[terminal as usize], Terminal::SpecialToken { .. }) {
            return Err(error("an exact token cannot be an ignore terminal"));
        }
        Ok(terminal)
    }).transpose()?;
    let mut partitions = BTreeMap::<u32, String>::new();
    let mut assign = |terminal, partition: &String| -> Result<(), GlrMaskError> {
        if let Some(previous) = partitions.insert(terminal, partition.clone())
            && previous != *partition {
            return Err(error(format!("terminal {terminal} belongs to incompatible lexer partitions")));
        }
        Ok(())
    };
    for (name, partition) in &grammar.lexer_partitions {
        let &(terminal, _) = resolver.named_terminals.get(name)
            .ok_or_else(|| error(format!("lexer partition refers to non-emitting terminal {name:?}")))?;
        assign(terminal, partition)?;
    }
    for (bytes, partition) in &grammar.lexer_literal_partitions {
        if let Some(&(terminal, _)) = resolver.lexical.literal_terminal_ids.get(bytes) {
            assign(terminal, partition)?;
        }
    }
    if let Some(partition) = &grammar.default_lexer_partition {
        for terminal in 0..resolver.lexical.terminals.len() as u32 {
            partitions.entry(terminal).or_insert_with(|| partition.clone());
        }
    }
    let terminal_names = (0..resolver.lexical.terminals.len() as u32).map(|terminal|
        resolver.lexical.terminal_names.get(&terminal).cloned().unwrap_or_else(|| format!("terminal_{terminal}")))
        .collect();
    let parser = ParserGrammar { data: Arc::new(ParserGrammarData { rules, start, terminal_names, ignore_terminal,
        left_cfg: Default::default(), right_cfg: Default::default(), analysis: Default::default() }) };
    let lexical = GrammarDef { terminals: resolver.lexical.terminals,
        terminal_names: resolver.lexical.terminal_names, ignore_terminal, lexer_partitions: partitions,
        requires_global_terminal_observation: true, ..GrammarDef::default() };
    Ok((parser, lexical))
}
