//! Optional, shared normalization and analysis. The expression graph remains
//! authoritative; merely preparing it never runs this code.
use super::*;

/// Direction of recursion introduced for repetitions. User-written recursion
/// is preserved. Finite repetitions and separated sequences use the matching
/// linear shape rather than an environment-selected LR-oriented normal form.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CfgRecursion { #[default] Left, Right }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ParserSymbol { Terminal(u32), Nonterminal(u32) }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParserProduction { pub lhs: u32, pub rhs: Vec<ParserSymbol> }

/// A lexer-free CFG. Original rule IDs remain `0..source_rule_count`; generated
/// wrappers/helpers follow them. Terminal IDs are exactly those in ParserGrammar.
#[derive(Debug, Clone)]
pub struct FlatParserGrammar {
    pub rules: Vec<ParserProduction>,
    pub start: u32,
    pub nonterminal_count: u32,
    pub source_rule_count: u32,
    pub terminal_count: u32,
}

/// Conventional nullable/FIRST/FOLLOW data in the flat helper coordinate.
/// FIRST and FOLLOW contain terminal IDs only; EOF is represented separately.
#[derive(Debug, Clone)]
pub struct ParserAnalysis {
    pub nullable: Vec<bool>,
    pub first_terminals: Vec<BTreeSet<u32>>,
    pub follow_terminals: Vec<BTreeSet<u32>>,
    pub follows_eof: Vec<bool>,
}

impl ParserGrammar {
    pub fn lower_to_cfg(&self) -> Result<Arc<FlatParserGrammar>, GlrMaskError> {
        self.lower_to_cfg_with(CfgRecursion::Left)
    }

    /// Lazily normalize once per explicit shape. Clones of this ParserGrammar
    /// share both the expression graph and the memoized lowering result.
    pub fn lower_to_cfg_with(&self, shape: CfgRecursion) -> Result<Arc<FlatParserGrammar>, GlrMaskError> {
        let cache = match shape { CfgRecursion::Left => &self.data.left_cfg, CfgRecursion::Right => &self.data.right_cfg };
        cache.get_or_init(|| lower(self, shape).map(Arc::new).map_err(|err| err.to_string()))
            .as_ref().map(Arc::clone).map_err(|message| error(message.clone()))
    }

    /// Lazily derive generic grammar analyses once, independently of the
    /// parser compiler selected by the caller. This does not build an LR table.
    pub fn analysis(&self) -> Result<Arc<ParserAnalysis>, GlrMaskError> {
        self.data.analysis.get_or_init(|| {
            let flat = self.lower_to_cfg().map_err(|err| err.to_string())?;
            analyze(&flat).map(Arc::new)
        }).as_ref().map(Arc::clone).map_err(|message| error(message.clone()))
    }
}

fn reify(expr: &ParserExpr) -> GrammarExpr {
    match expr {
        ParserExpr::Terminal(id) => GrammarExpr::Ref(format!("T_{id}")),
        ParserExpr::Nonterminal(id) => GrammarExpr::Ref(format!("N_{id}")),
        ParserExpr::Epsilon => GrammarExpr::Epsilon,
        ParserExpr::Sequence(parts) => GrammarExpr::Sequence(parts.iter().map(reify).collect()),
        ParserExpr::Choice(parts) => GrammarExpr::Choice(parts.iter().map(reify).collect()),
        ParserExpr::Quantified(expr, quantifier) => GrammarExpr::Quantified(Box::new(reify(expr)), quantifier.clone()),
        ParserExpr::SeparatedSequence { items, separator, allow_empty } => GrammarExpr::SeparatedSequence {
            items: items.iter().map(|(expr, quantifier)| (reify(expr), quantifier.clone())).collect(),
            separator: Box::new(reify(separator)), allow_empty: *allow_empty,
        },
        ParserExpr::Automaton(graph) => {
            if graph.start_states.is_empty() { return GrammarExpr::Choice(Vec::new()); }
            let nfa = crate::automata::unweighted_u32::nfa::NFA {
                start_states: graph.start_states.clone(),
                states: graph.states.iter().map(|state| {
                    let mut transitions = BTreeMap::<i32, Vec<u32>>::new();
                    for &(label, target) in &state.transitions { transitions.entry(label as i32).or_default().push(target); }
                    crate::automata::unweighted_u32::nfa::NFAState {
                        is_accepting: state.accepting, transitions, epsilons: state.epsilons.clone(),
                    }
                }).collect(),
            };
            let mut expr = ExprNFA::new(nfa, graph.symbols.iter().map(reify).collect());
            expr.prefer_direct_nfa_emission = true;
            GrammarExpr::ExprNFA(Box::new(expr))
        }
    }
}

fn lower(grammar: &ParserGrammar, direction: CfgRecursion) -> Result<FlatParserGrammar, GlrMaskError> {
    let mut rules = grammar.rules().iter().enumerate().map(|(id, rule)| NamedRule {
        name: format!("N_{id}"), expr: reify(&rule.expr), is_terminal: false, is_internal: false,
    }).collect::<Vec<_>>();
    // Private opaque identities let the shared AST lowerer preserve terminals
    // without being given their lexical definitions or vocabulary token IDs.
    // This temporary AST is never compiled as a lexer or exposed to the user.
    rules.extend((0..grammar.terminal_count()).map(|id| NamedRule {
        name: format!("T_{id}"), expr: GrammarExpr::SpecialToken(id), is_terminal: true, is_internal: false,
    }));
    let named = NamedGrammar { rules, start: format!("N_{}", grammar.start()), ignore: None,
        lexer_partitions: BTreeMap::new(), lexer_literal_partitions: BTreeMap::new(), default_lexer_partition: None };
    let shape = match direction {
        CfgRecursion::Left => (RepeatTreeShape::Left, CommaSepShape::Left, false),
        CfgRecursion::Right => (RepeatTreeShape::Right, CommaSepShape::Right, true),
    };
    let flat = lower_with_resolved_terminal_exprs_impl(&named, None, Some(shape))?;
    let terminal_map = flat.terminals.iter().map(|terminal| match terminal {
        Terminal::SpecialToken { token_id, .. } => Ok(*token_id),
        _ => Err(error("parser-only CFG lowering introduced a lexical terminal")),
    }).collect::<Result<Vec<_>, _>>()?;
    let mut nonterminal_count = grammar.rules().len() as u32;
    let rules = flat.rules.into_iter().map(|rule| {
        nonterminal_count = nonterminal_count.max(rule.lhs + 1);
        let rhs = rule.rhs.into_iter().map(|symbol| match symbol {
            Symbol::Terminal(id) => ParserSymbol::Terminal(terminal_map[id as usize]),
            Symbol::Nonterminal(id) => {
                nonterminal_count = nonterminal_count.max(id + 1);
                ParserSymbol::Nonterminal(id)
            }
        }).collect();
        ParserProduction { lhs: rule.lhs, rhs }
    }).collect();
    Ok(FlatParserGrammar { rules, start: flat.start, nonterminal_count,
        source_rule_count: grammar.rules().len() as u32, terminal_count: grammar.terminal_count() })
}

fn analyze(grammar: &FlatParserGrammar) -> Result<ParserAnalysis, String> {
    let n = grammar.nonterminal_count as usize;
    let mut nullable = vec![false; n];
    let mut first = vec![BTreeSet::new(); n];
    let mut follow = vec![BTreeSet::new(); n];
    let mut eof = vec![false; n];
    let mut remaining = 32_000_000usize;
    let mut charge = |work: usize| -> Result<(), String> {
        remaining = remaining.checked_sub(work).ok_or("generic grammar analysis exceeds its work budget")?;
        Ok(())
    };
    loop {
        let mut changed = false;
        for rule in &grammar.rules {
            charge(rule.rhs.len() + 1)?;
            if !nullable[rule.lhs as usize] && rule.rhs.iter().all(|symbol| match symbol {
                ParserSymbol::Terminal(_) => false,
                ParserSymbol::Nonterminal(id) => nullable[*id as usize],
            }) { nullable[rule.lhs as usize] = true; changed = true; }
        }
        if !changed { break; }
    }
    loop {
        let mut changed = false;
        for rule in &grammar.rules {
            charge(1)?;
            for symbol in &rule.rhs {
                match symbol {
                    ParserSymbol::Terminal(id) => { changed |= first[rule.lhs as usize].insert(*id); break; }
                    ParserSymbol::Nonterminal(id) => {
                        charge(first[*id as usize].len() + 1)?;
                        let additions = first[*id as usize].iter().copied().collect::<Vec<_>>();
                        for terminal in additions { changed |= first[rule.lhs as usize].insert(terminal); }
                        if !nullable[*id as usize] { break; }
                    }
                }
            }
        }
        if !changed { break; }
    }
    eof[grammar.start as usize] = true;
    loop {
        let mut changed = false;
        for rule in &grammar.rules {
            charge(follow[rule.lhs as usize].len() + 1)?;
            let mut trailer = follow[rule.lhs as usize].clone();
            let mut trailer_eof = eof[rule.lhs as usize];
            for symbol in rule.rhs.iter().rev() {
                match symbol {
                    ParserSymbol::Terminal(id) => { trailer.clear(); trailer.insert(*id); trailer_eof = false; }
                    ParserSymbol::Nonterminal(id) => {
                        charge(trailer.len() + first[*id as usize].len() + 1)?;
                        for &terminal in &trailer { changed |= follow[*id as usize].insert(terminal); }
                        if trailer_eof && !eof[*id as usize] { eof[*id as usize] = true; changed = true; }
                        if !nullable[*id as usize] { trailer.clear(); trailer_eof = false; }
                        trailer.extend(first[*id as usize].iter().copied());
                    }
                }
            }
        }
        if !changed { break; }
    }
    Ok(ParserAnalysis { nullable, first_terminals: first, follow_terminals: follow, follows_eof: eof })
}
