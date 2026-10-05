use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::automata::regex::Expr;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GrammarDef {
    pub rules: Vec<Rule>,
    pub start: NonterminalID,
    pub terminals: Vec<Terminal>,
    #[serde(default)]
    pub nonterminal_names: BTreeMap<NonterminalID, String>,
    #[serde(default)]
    pub terminal_names: BTreeMap<TerminalID, String>,
    #[serde(default)]
    pub ignore_terminal: Option<TerminalID>,
    /// Explicit terminal-id -> named lexer partition assignments.
    #[serde(default)]
    pub lexer_partitions: BTreeMap<TerminalID, String>,
    /// Residual coordinates that must remain independently identifiable while
    /// the compile-time lexer is simplified.
    #[serde(default)]
    pub residual_isolation_classes: BTreeMap<TerminalID, u32>,
    /// Exact token-equivalence analysis must observe all terminal residuals.
    #[serde(default)]
    pub requires_global_terminal_observation: bool,
    /// Exact terminal-level automaton for a frontend-proven regular language.
    #[serde(default)]
    pub direct_regular_automaton: Option<DirectRegularAutomaton>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct DirectRegularAutomaton {
    pub states: Vec<DirectRegularState>,
    pub start_states: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct DirectRegularState {
    pub is_accepting: bool,
    pub transitions: BTreeMap<TerminalID, Vec<u32>>,
    pub epsilons: Vec<u32>,
}

pub type NonterminalID = u32;
pub type TerminalID = u32;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rule {
    pub lhs: NonterminalID,
    pub rhs: Vec<Symbol>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Symbol {
    Terminal(TerminalID),
    Nonterminal(NonterminalID),
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Symbol::Terminal(id) => write!(f, "T{}", id),
            Symbol::Nonterminal(id) => write!(f, "N{}", id),
        }
    }
}

impl Symbol {
    fn nonterminal_id(&self) -> Option<NonterminalID> {
        match self {
            Symbol::Nonterminal(nonterminal) => Some(*nonterminal),
            Symbol::Terminal(_) => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Terminal {
    Literal {
        id: TerminalID,
        bytes: Vec<u8>,
    },
    Pattern {
        id: TerminalID,
        pattern: String,
        utf8: bool,
    },
    Expr {
        id: TerminalID,
        expr: Expr,
    },
    /// An exact model-token event, not its byte spelling.
    SpecialToken {
        id: TerminalID,
        token_id: u32,
    },
}

impl Rule {
    fn nonterminal_ids(&self) -> impl Iterator<Item = NonterminalID> + '_ {
        std::iter::once(self.lhs).chain(self.rhs.iter().filter_map(Symbol::nonterminal_id))
    }
}

impl Terminal {
    /// Exact token identities are events, even with an empty byte spelling.
    pub fn is_nullable(&self) -> bool {
        match self {
            Self::Literal { bytes, .. } => bytes.is_empty(),
            Self::Pattern { pattern, utf8, .. } => {
                crate::automata::lexer::regex::parse_regex(pattern, *utf8).is_nullable()
            }
            Self::Expr { expr, .. } => expr.is_nullable(),
            Self::SpecialToken { .. } => false,
        }
    }

    pub fn id(&self) -> TerminalID {
        match self {
            Terminal::Literal { id, .. } => *id,
            Terminal::Pattern { id, .. } => *id,
            Terminal::Expr { id, .. } => *id,
            Terminal::SpecialToken { id, .. } => *id,
        }
    }

    pub fn name(&self) -> String {
        match self {
            Terminal::Literal { bytes, .. } => String::from_utf8_lossy(bytes).into_owned(),
            Terminal::Pattern { pattern, .. } => pattern.clone(),
            Terminal::Expr { expr, .. } => format!("{:?}", expr),
            Terminal::SpecialToken { token_id, .. } => format!("@token({token_id})"),
        }
    }
}

impl GrammarDef {
    /// Source-language epsilon membership.
    ///
    /// This fact must be captured before standalone epsilon elimination.
    /// Both terminal and nonterminal IDs may be sparse here.
    pub fn start_is_nullable(&self) -> bool {
        // Preserve source terminal evaluation order, including deterministic
        // regex-parse failures, before considering either parser representation.
        let nullable_terminals = self
            .terminals
            .iter()
            .filter_map(|terminal| terminal.is_nullable().then_some(terminal.id()))
            .collect::<BTreeSet<_>>();

        if let Some(automaton) = &self.direct_regular_automaton {
            let mut reachable = automaton.start_states.clone();
            let mut cursor = 0usize;
            let mut seen = BTreeSet::from_iter(reachable.iter().copied());
            while cursor < reachable.len() {
                let state = reachable[cursor] as usize;
                cursor += 1;
                let Some(node) = automaton.states.get(state) else {
                    continue;
                };
                if node.is_accepting {
                    return true;
                }
                for &target in node.epsilons.iter().chain(
                    node.transitions
                        .iter()
                        .filter(|(terminal, _)| nullable_terminals.contains(*terminal))
                        .flat_map(|(_, targets)| targets),
                ) {
                    if seen.insert(target) {
                        reachable.push(target);
                    }
                }
            }
        }

        // Rule counters avoid rescanning a long dependency chain once per
        // newly nullable nonterminal. Sparse maps avoid max-ID-sized storage.
        let mut remaining = vec![usize::MAX; self.rules.len()];
        let mut dependents = BTreeMap::<NonterminalID, Vec<usize>>::new();
        let mut nullable = BTreeSet::<NonterminalID>::new();
        let mut queue = Vec::<NonterminalID>::new();

        for (index, rule) in self.rules.iter().enumerate() {
            let mut count = 0usize;
            let mut possible = true;
            for symbol in &rule.rhs {
                match symbol {
                    Symbol::Terminal(terminal) => {
                        if !nullable_terminals.contains(terminal) {
                            possible = false;
                            break;
                        }
                    }
                    Symbol::Nonterminal(nonterminal) => {
                        dependents.entry(*nonterminal).or_default().push(index);
                        count += 1;
                    }
                }
            }
            if possible {
                remaining[index] = count;
                if count == 0 && nullable.insert(rule.lhs) {
                    queue.push(rule.lhs);
                }
            }
        }

        let mut cursor = 0usize;
        while cursor < queue.len() {
            let nonterminal = queue[cursor];
            cursor += 1;
            for &index in dependents.get(&nonterminal).into_iter().flatten() {
                let left = &mut remaining[index];
                if *left == usize::MAX || *left == 0 {
                    continue;
                }
                *left -= 1;
                if *left == 0 {
                    let lhs = self.rules[index].lhs;
                    if nullable.insert(lhs) {
                        queue.push(lhs);
                    }
                }
            }
        }

        nullable.contains(&self.start)
    }

    pub fn num_terminals(&self) -> u32 {
        self.terminals.len() as u32
    }

    pub fn num_nonterminals(&self) -> u32 {
        self.rules
            .iter()
            .flat_map(|rule| rule.nonterminal_ids())
            .max()
            .map(|id| id + 1)
            .unwrap_or(0)
    }

    pub fn terminal_display_name(&self, terminal: TerminalID) -> String {
        self.terminal_names
            .get(&terminal)
            .cloned()
            .or_else(|| self.terminal_by_id(terminal).map(Terminal::name))
            .unwrap_or_else(|| format!("T{terminal}"))
    }

    fn terminal_by_id(&self, terminal: TerminalID) -> Option<&Terminal> {
        self.terminals
            .iter()
            .find(|terminal_def| terminal_def.id() == terminal)
    }
}

#[cfg(test)]
#[path = "flat_nullability_tests.rs"]
mod nullability_tests;

#[cfg(test)]
mod worklist_nullability_tests {
    use super::*;

    fn reference(grammar: &GrammarDef) -> bool {
        let terminals = grammar
            .terminals
            .iter()
            .filter_map(|terminal| terminal.is_nullable().then_some(terminal.id()))
            .collect::<BTreeSet<_>>();
        let mut nullable = BTreeSet::new();
        loop {
            let before = nullable.len();
            for rule in &grammar.rules {
                if rule.rhs.iter().all(|symbol| match symbol {
                    Symbol::Terminal(terminal) => terminals.contains(terminal),
                    Symbol::Nonterminal(nonterminal) => nullable.contains(nonterminal),
                }) {
                    nullable.insert(rule.lhs);
                }
            }
            if before == nullable.len() {
                return nullable.contains(&grammar.start);
            }
        }
    }

    #[test]
    fn repeated_occurrences_and_ineligible_rule_prefixes_are_exact() {
        let grammar = GrammarDef {
            start: 999,
            rules: vec![
                Rule {
                    lhs: 999,
                    rhs: vec![Symbol::Nonterminal(7), Symbol::Nonterminal(7)],
                },
                Rule {
                    lhs: 8,
                    rhs: vec![Symbol::Nonterminal(7), Symbol::Terminal(2)],
                },
                Rule {
                    lhs: 7,
                    rhs: Vec::new(),
                },
            ],
            terminals: vec![Terminal::Literal {
                id: 2,
                bytes: b"x".to_vec(),
            }],
            ..Default::default()
        };
        assert!(grammar.start_is_nullable());
        assert_eq!(grammar.start_is_nullable(), reference(&grammar));
        let mut other = grammar;
        other.start = 8;
        assert!(!other.start_is_nullable());
    }

    #[test]
    fn sparse_long_chain_does_not_require_repeated_grammar_scans() {
        let mut grammar = GrammarDef {
            start: 1_000_000_000,
            ..Default::default()
        };
        for step in 0..4096u32 {
            grammar.rules.push(Rule {
                lhs: 1_000_000_000 + step,
                rhs: vec![Symbol::Nonterminal(1_000_000_001 + step)],
            });
        }
        grammar.rules.push(Rule {
            lhs: 1_000_004_096,
            rhs: Vec::new(),
        });
        assert!(grammar.start_is_nullable());
    }

    #[test]
    fn small_sparse_graphs_match_independent_fixed_point() {
        for mask in 0usize..512 {
            let ids = [7u32, 1000, 900_000];
            let mut grammar = GrammarDef {
                start: ids[0],
                terminals: vec![
                    Terminal::Literal {
                        id: 200,
                        bytes: Vec::new(),
                    },
                    Terminal::SpecialToken {
                        id: 201,
                        token_id: 0,
                    },
                ],
                ..Default::default()
            };
            for source in 0..3 {
                for target in 0..3 {
                    if mask & (1 << (source * 3 + target)) != 0 {
                        grammar.rules.push(Rule {
                            lhs: ids[source],
                            rhs: vec![Symbol::Nonterminal(ids[target])],
                        });
                    }
                }
            }
            grammar.rules.push(Rule {
                lhs: ids[2],
                rhs: vec![Symbol::Terminal(200)],
            });
            grammar.rules.push(Rule {
                lhs: ids[1],
                rhs: vec![Symbol::Terminal(201)],
            });
            assert_eq!(grammar.start_is_nullable(), reference(&grammar));
        }
    }
}
