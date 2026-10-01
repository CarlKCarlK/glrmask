use std::collections::BTreeMap;

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
    /// Explicit terminal-id → named lexer partition assignments.
    #[serde(default)]
    pub lexer_partitions: BTreeMap<TerminalID, String>,
    /// Residual coordinates that must remain independently identifiable while
    /// the compile-time lexer is simplified. Terminals with different class
    /// ids must not be deduplicated or jointly determinized before late
    /// full-lexer state-map expansion is complete.
    ///
    /// This is compiler metadata rather than language semantics. Ordinary
    /// grammars leave the map empty. Synthetic bounded-terminal lowering gives
    /// each independently simplified terminal its own class initially.
    #[serde(default)]
    pub residual_isolation_classes: BTreeMap<TerminalID, u32>,
    /// Exact token-equivalence analysis must observe all terminal residuals,
    /// rather than projecting to one L2P construction family.
    #[serde(default)]
    pub requires_global_terminal_observation: bool,
    /// Exact terminal-level automaton for the complete parser language when a
    /// front-end has proved that language regular. This allows table
    /// construction to bypass LR machinery without depending on transformed
    /// production shape.
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
    /// An exact byte sequence (e.g., a keyword or punctuation).
    Literal { id: TerminalID, bytes: Vec<u8> },
    /// A regex pattern string (e.g., `[a-z]+` or `\\d`).
    Pattern { id: TerminalID, pattern: String, utf8: bool },
    /// A pre-parsed regex expression.
    Expr { id: TerminalID, expr: Expr },
    /// A terminal matched only by committing one exact LLM token id.
    ///
    /// Special-token terminals are outside the byte tokenizer language. The
    /// token's vocabulary bytes, when present, do not match this terminal via
    /// `commit_bytes` and cannot partially match it.
    SpecialToken { id: TerminalID, token_id: u32 },
}

impl Rule {
    fn nonterminal_ids(&self) -> impl Iterator<Item = NonterminalID> + '_ {
        std::iter::once(self.lhs).chain(self.rhs.iter().filter_map(Symbol::nonterminal_id))
    }
}

impl Terminal {
    /// Whether this terminal's byte language contains epsilon. Exact token
    /// identities are events, not byte strings, even when their spelling is empty.
    pub fn is_nullable(&self) -> bool {
        match self {
            Self::Literal { bytes, .. } => bytes.is_empty(),
            Self::Pattern { pattern, utf8, .. } =>
                crate::automata::lexer::regex::parse_regex(pattern, *utf8).is_nullable(),
            Self::Expr { expr, .. } => expr.is_nullable(),
            Self::SpecialToken { .. } => false,
        }
    }

    /// Return the terminal's numeric ID.
    pub fn id(&self) -> TerminalID {
        match self {
            Terminal::Literal { id, .. } => *id,
            Terminal::Pattern { id, .. } => *id,
            Terminal::Expr { id, .. } => *id,
            Terminal::SpecialToken { id, .. } => *id,
        }
    }

    /// Return a display name for the terminal.
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
    /// Whether the source start nonterminal derives epsilon. The ordinary
    /// compiler later inlines nullable productions and deliberately removes a
    /// root-only empty generation path, but subgrammar composition must retain
    /// this source-language fact. Before lexical normalization, a terminal can
    /// itself derive epsilon; treat that exactly like its later optional wrapper.
    pub fn start_is_nullable(&self) -> bool {
        let nullable_terminals = self.terminals.iter()
            .filter_map(|terminal| terminal.is_nullable().then_some(terminal.id()))
            .collect::<std::collections::BTreeSet<_>>();
        if let Some(automaton) = &self.direct_regular_automaton {
            let mut reachable = automaton.start_states.clone();
            let mut cursor = 0usize;
            let mut seen = std::collections::BTreeSet::from_iter(reachable.iter().copied());
            while cursor < reachable.len() {
                let state = reachable[cursor] as usize;
                cursor += 1;
                let Some(node) = automaton.states.get(state) else {
                    continue;
                };
                if node.is_accepting {
                    return true;
                }
                for &target in node.epsilons.iter().chain(node.transitions.iter()
                    .filter(|(terminal, _)| nullable_terminals.contains(*terminal))
                    .flat_map(|(_, targets)| targets)) {
                    if seen.insert(target) {
                        reachable.push(target);
                    }
                }
            }
        }

        let mut nullable = std::collections::BTreeSet::<NonterminalID>::new();
        loop {
            let before = nullable.len();
            for rule in &self.rules {
                if rule.rhs.iter().all(|symbol| match symbol {
                    Symbol::Nonterminal(nonterminal) => nullable.contains(nonterminal),
                    Symbol::Terminal(terminal) => nullable_terminals.contains(terminal),
                }) {
                    nullable.insert(rule.lhs);
                }
            }
            if nullable.len() == before {
                break;
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
