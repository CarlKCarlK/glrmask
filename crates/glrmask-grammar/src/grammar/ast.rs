use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use rayon::prelude::*;

use crate::GlrMaskError;
use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::compile::compile_expression_labeled_nfa;
use crate::automata::lexer::DFA as LexerDFA;
use crate::automata::unweighted_u32::dfa::DFA;
use crate::automata::unweighted_u32::minimize_acyclic::reindex_minimized_acyclic_order;
use crate::automata::lexer::regex::parse_regex;
use crate::ds::u8set::U8Set;
use crate::grammar::flat::{
    DirectRegularAutomaton, DirectRegularState, GrammarDef, NonterminalID, Rule, Symbol, Terminal,
    TerminalID,
};
use crate::grammar::expr_nfa::ExprNFA;

pub mod parser_grammar;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Quantifier {
    Optional,
    ZeroPlus,
    OnePlus,
    Range(usize, Option<usize>),
}

fn quantifier_suffix(quantifier: &Quantifier) -> String {
    match quantifier {
        Quantifier::Optional => "?".to_string(),
        Quantifier::ZeroPlus => "*".to_string(),
        Quantifier::OnePlus => "+".to_string(),
        Quantifier::Range(min, Some(max)) if min == max => format!("{{{min}}}"),
        Quantifier::Range(min, Some(max)) => format!("{{{min},{max}}}"),
        Quantifier::Range(min, None) => format!("{{{min},}}"),
    }
}

impl Quantifier {
    pub fn min(&self) -> usize {
        match self {
            Quantifier::Optional | Quantifier::ZeroPlus => 0,
            Quantifier::OnePlus => 1,
            Quantifier::Range(min, _) => *min,
        }
    }

    pub fn max(&self) -> Option<usize> {
        match self {
            Quantifier::Optional => Some(1),
            Quantifier::ZeroPlus | Quantifier::OnePlus => None,
            Quantifier::Range(_, max) => *max,
        }
    }

    pub fn is_nullable(&self) -> bool {
        self.min() == 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GrammarExpr {
    Ref(String),
    Grouped(Box<GrammarExpr>),
    Sequence(Vec<GrammarExpr>),
    Choice(Vec<GrammarExpr>),
    /// Empty string / epsilon. Equivalent to `Sequence([])` for grammar purposes;
    /// maps to `Expr::Epsilon` in terminal-expression context.
    Epsilon,
    Exclude {
        expr: Box<GrammarExpr>,
        exclude: Box<GrammarExpr>,
    },
    Intersect {
        expr: Box<GrammarExpr>,
        intersect: Box<GrammarExpr>,
    },
    Quantified(Box<GrammarExpr>, Quantifier),
    Literal(Vec<u8>),
    /// Match one exact LLM token id. This is not a byte-language expression.
    SpecialToken(u32),
    CharClass { def: String, negate: bool, utf8: bool },
    RawRegex(String),
    LexerDfa(Arc<LexerDFA>),
    AnyByte,
    /// A separator-delimited sequence of quantified items.
    ///
    /// Each item is stored as `(item_expr, item_quantifier)`. The quantifier, if
    /// present, is the single postfix quantifier that binds to the separator. For
    /// example, `sep ~ (x*)` means zero or more `x` items separated by `sep`,
    /// while `sep ~ ((x*))` is a single grouped item whose body is `x*`.
    SeparatedSequence {
        items: Vec<(GrammarExpr, Option<Quantifier>)>,
        separator: Box<GrammarExpr>,
        allow_empty: bool,
    },
    /// An NFA whose transition labels are indices into a side table of grammar
    /// expressions.
    ///
    /// This is only valid as the complete expression of a named rule. In a
    /// nonterminal rule the labels denote parser terminals; in a terminal rule
    /// they denote byte-language expressions compiled into one lexer DFA.
    ExprNFA(Box<ExprNFA>),
}

/// Controls the tree shape used when lowering [`GrammarExpr::SeparatedSequence`].
///
/// The shape determines how the item list is recursively split into subtrees,
/// which affects parse-path counts and grammar size. Configure via the
/// `GLRMASK_ORDERED_OBJECT_SHAPE` environment variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommaSepShape {
    /// Split at the midpoint (balanced binary tree).
    Balanced,
    /// Always split one item from the left (left-linear tree). Default.
    Left,
    /// Always split one item from the right (right-linear / factored tree).
    Right,
    /// Split at the first optional item boundary; fall back to balanced.
    LeftBalanced,
}

/// Read the `CommaSepShape` from the `GLRMASK_ORDERED_OBJECT_SHAPE` environment variable.
pub fn comma_sep_shape() -> CommaSepShape {
    match std::env::var("GLRMASK_ORDERED_OBJECT_SHAPE")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("left") => CommaSepShape::Left,
        Some("balanced") => CommaSepShape::Balanced,
        Some("left-balanced") | Some("left_balanced") | Some("leftbalanced") => {
            CommaSepShape::LeftBalanced
        }
        Some("right") | Some("factored") => CommaSepShape::Right,
        None => CommaSepShape::Left,
        Some(_) => CommaSepShape::Left,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NamedRule {
    pub name: String,
    pub expr: GrammarExpr,
    pub is_terminal: bool,
    /// Internal-only terminals exist solely as sub-expressions of other
    /// terminal rules (resolved via `Expr::Shared`). They do not produce
    /// their own `TerminalID` or parser production.
    pub is_internal: bool,
}

#[derive(Debug, Clone)]
pub struct NamedGrammar {
    pub rules: Vec<NamedRule>,
    pub start: String,
    /// Name of the terminal rule whose body should be used as the ignore pattern.
    /// Set by Lark's `%ignore` directive.
    pub ignore: Option<String>,
    /// Optional terminal-name â†’ lexer-partition name assignments.
    ///
    /// Terminals in the same named partition are jointly determinized.
    /// Different partitions are combined by epsilon transitions, preventing
    /// cross-partition DFA blow-ups. Unassigned terminals follow the active
    /// tokenizer partition policy.
    pub lexer_partitions: BTreeMap<String, String>,
    /// Exact anonymous literal bytes â†’ lexer-partition name assignments.
    ///
    /// This applies to inline literals such as `"{"` used directly in
    /// nonterminal productions, not to named terminal rules whose bodies happen
    /// to be literal. Those continue to use `lexer_partitions`.
    pub lexer_literal_partitions: BTreeMap<Vec<u8>, String>,
    /// Optional catch-all partition for every emitted terminal not assigned by
    /// either an explicit named-terminal or anonymous-literal assignment.
    pub default_lexer_partition: Option<String>,
}

impl NamedGrammar {
    /// Exact byte strings that are emitted by anonymous inline literal atoms in
    /// nonterminal rules. Literals nested inside a terminal expression such as
    /// an intersection remain part of that one terminal and are not returned.
    pub fn emitted_anonymous_literals(&self) -> BTreeSet<Vec<u8>> {
        fn collect(expr: &GrammarExpr, literals: &mut BTreeSet<Vec<u8>>) {
            match expr {
                GrammarExpr::Literal(bytes) => {
                    if !bytes.is_empty() {
                        literals.insert(bytes.clone());
                    }
                }
                GrammarExpr::Grouped(inner) | GrammarExpr::Quantified(inner, _) => {
                    collect(inner, literals);
                }
                GrammarExpr::Sequence(parts) | GrammarExpr::Choice(parts) => {
                    for part in parts {
                        collect(part, literals);
                    }
                }
                GrammarExpr::SeparatedSequence {
                    items, separator, ..
                } => {
                    collect(separator, literals);
                    for (item, _) in items {
                        collect(item, literals);
                    }
                }
                GrammarExpr::ExprNFA(expr_nfa) => {
                    for symbol in &expr_nfa.symbols {
                        collect(symbol, literals);
                    }
                }
                GrammarExpr::Exclude { .. }
                | GrammarExpr::Intersect { .. }
                | GrammarExpr::Ref(_)
                | GrammarExpr::Epsilon
                | GrammarExpr::CharClass { .. }
                | GrammarExpr::RawRegex(_)
                | GrammarExpr::LexerDfa(_)
                | GrammarExpr::AnyByte
                | GrammarExpr::SpecialToken(_) => {}
            }
        }

        let mut literals = BTreeSet::new();
        for rule in &self.rules {
            if !rule.is_terminal {
                collect(&rule.expr, &mut literals);
            }
        }
        literals
    }

    /// Assign terminal rules to one jointly-determinized lexer partition.
    pub fn set_lexer_partition<I, S>(&mut self, partition: impl Into<String>, terminals: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let partition = partition.into();
        for terminal in terminals {
            self.lexer_partitions.insert(terminal.into(), partition.clone());
        }
    }

    /// Builder-style variant of [`Self::set_lexer_partition`].
    pub fn with_lexer_partition<I, S>(
        mut self,
        partition: impl Into<String>,
        terminals: I,
    ) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.set_lexer_partition(partition, terminals);
        self
    }

    /// Put one terminal in its own lexer partition.
    pub fn isolate_terminal(&mut self, terminal: impl Into<String>) {
        let terminal = terminal.into();
        let base = format!("__isolated_{terminal}");
        let mut partition = base.clone();
        let mut suffix = 2usize;
        while self
            .lexer_partitions
            .iter()
            .any(|(other_terminal, existing)| {
                other_terminal != &terminal && existing == &partition
            })
        {
            partition = format!("{base}_{suffix}");
            suffix += 1;
        }
        self.lexer_partitions.insert(terminal, partition);
    }

    /// Assign anonymous inline literals to one jointly-determinized partition.
    pub fn set_literal_lexer_partition<I, B>(
        &mut self,
        partition: impl Into<String>,
        literals: I,
    ) where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        let partition = partition.into();
        for literal in literals {
            self.lexer_literal_partitions
                .insert(literal.as_ref().to_vec(), partition.clone());
        }
    }

    /// Builder-style variant of [`Self::set_literal_lexer_partition`].
    pub fn with_literal_lexer_partition<I, B>(
        mut self,
        partition: impl Into<String>,
        literals: I,
    ) -> Self
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        self.set_literal_lexer_partition(partition, literals);
        self
    }

    /// Assign every otherwise-unassigned emitted terminal to `partition`.
    pub fn set_default_lexer_partition(&mut self, partition: impl Into<String>) {
        self.default_lexer_partition = Some(partition.into());
    }

    /// Returns the set of rule names marked as terminals.
    pub fn terminal_names_set(&self) -> HashSet<String> {
        self.rules
            .iter()
            .filter(|r| r.is_terminal)
            .map(|r| r.name.clone())
            .collect()
    }

    /// Remove rules that are not reachable from the start rule (or ignore rule).
    ///
    /// Traverses `GrammarExpr::Ref` edges to find all rules reachable from
    /// `self.start` (and `self.ignore` if set), then returns a new grammar
    /// containing only those rules in their original order.
    pub fn prune_unreachable(&self) -> Self {
        fn collect_refs(expr: &GrammarExpr, out: &mut HashSet<String>) {
            match expr {
                GrammarExpr::Ref(name) => { out.insert(name.clone()); }
                GrammarExpr::Grouped(inner) => collect_refs(inner, out),
                GrammarExpr::Sequence(items) => { for e in items { collect_refs(e, out); } }
                GrammarExpr::Choice(alts) => { for e in alts { collect_refs(e, out); } }
                GrammarExpr::Exclude { expr, exclude } => {
                    collect_refs(expr, out); collect_refs(exclude, out);
                }
                GrammarExpr::Intersect { expr, intersect } => {
                    collect_refs(expr, out); collect_refs(intersect, out);
                }
                GrammarExpr::Quantified(e, Quantifier::Optional) | GrammarExpr::Quantified(e, Quantifier::ZeroPlus) | GrammarExpr::Quantified(e, Quantifier::OnePlus) => {
                    collect_refs(e, out);
                }
                GrammarExpr::Quantified(expr, Quantifier::Range(_, _)) => collect_refs(expr, out),
                GrammarExpr::SeparatedSequence { items, separator, .. } => {
                    for (e, _) in items { collect_refs(e, out); }
                    collect_refs(separator, out);
                }
                GrammarExpr::ExprNFA(expr_nfa) => {
                    for symbol in &expr_nfa.symbols {
                        collect_refs(symbol, out);
                    }
                }
                GrammarExpr::Epsilon | GrammarExpr::Literal(_) | GrammarExpr::SpecialToken(_)
                | GrammarExpr::CharClass { .. } | GrammarExpr::RawRegex(_)
                | GrammarExpr::LexerDfa(_)
                | GrammarExpr::AnyByte => {}
            }
        }

        let rule_map: HashMap<String, &NamedRule> = self.rules.iter()
            .map(|r| (r.name.clone(), r))
            .collect();

        let mut reachable: HashSet<String> = HashSet::new();
        let mut worklist: Vec<String> = vec![self.start.clone()];
        if let Some(ref ign) = self.ignore {
            worklist.push(ign.clone());
        }

        while let Some(name) = worklist.pop() {
            if !reachable.insert(name.clone()) { continue; }
            if let Some(rule) = rule_map.get(&name) {
                let mut refs = HashSet::new();
                collect_refs(&rule.expr, &mut refs);
                for r in refs {
                    if !reachable.contains(&r) {
                        worklist.push(r);
                    }
                }
            }
        }

        let rules = self.rules.iter()
            .filter(|r| reachable.contains(&r.name))
            .cloned()
            .collect();

        let lexer_partitions = self
            .lexer_partitions
            .iter()
            .filter(|(terminal, _)| reachable.contains(*terminal))
            .map(|(terminal, partition)| (terminal.clone(), partition.clone()))
            .collect();
        NamedGrammar {
            rules,
            start: self.start.clone(),
            ignore: self.ignore.clone(),
            lexer_partitions,
            lexer_literal_partitions: self.lexer_literal_partitions.clone(),
            default_lexer_partition: self.default_lexer_partition.clone(),
        }
    }

    /// Dump the grammar in a Lark-like human-readable format.
    ///
    /// `GrammarExpr` variants with no direct Lark equivalent (e.g. `Exclude`,
    /// `TerminalExpr`) are preserved as inline comments so the dump still
    /// reflects the full original grammar structure.
    pub fn to_lark(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();

        // Start rule
        writeln!(out, "// start: {}", self.start).unwrap();
        if let Some(ref ign) = self.ignore {
            writeln!(out, "%ignore {}", ign).unwrap();
        }
        writeln!(out).unwrap();

        // Terminal rules first, then nonterminal rules
        let terminals: Vec<_> = self.rules.iter().filter(|r| r.is_terminal).collect();
        let nonterminals: Vec<_> = self.rules.iter().filter(|r| !r.is_terminal).collect();

        if !nonterminals.is_empty() {
            writeln!(out, "// === Nonterminal rules ===").unwrap();
            for rule in &nonterminals {
                let prefix = if rule.is_internal { "// [internal] " } else { "" };
                write!(out, "{}{}: ", prefix, rule.name).unwrap();
                grammar_expr_to_lark(&rule.expr, &mut out, false);
                writeln!(out).unwrap();
            }
            writeln!(out).unwrap();
        }

        if !terminals.is_empty() {
            writeln!(out, "// === Terminal rules ===").unwrap();
            for rule in &terminals {
                let prefix = if rule.is_internal { "// [internal] " } else { "" };
                write!(out, "{}{}: ", prefix, rule.name).unwrap();
                grammar_expr_to_lark(&rule.expr, &mut out, false);
                writeln!(out).unwrap();
            }
        }

        out
    }
}

/// Format a `GrammarExpr` in Lark-like syntax. `parens` controls whether
/// compound expressions get wrapped in parentheses for disambiguation.
fn grammar_expr_to_lark(expr: &GrammarExpr, out: &mut String, parens: bool) {
    grammar_expr_to_lark_with_indent(expr, out, parens, 0);
}

fn grammar_expr_to_lark_with_indent(
    expr: &GrammarExpr,
    out: &mut String,
    parens: bool,
    indent: usize,
) {
    use std::fmt::Write;
    match expr {
        GrammarExpr::Ref(name) => {
            out.push_str(name);
        }
        GrammarExpr::Grouped(inner) => {
            out.push('(');
            grammar_expr_to_lark_with_indent(inner, out, false, indent);
            out.push(')');
        }
        GrammarExpr::Sequence(items) => {
            if parens && items.len() > 1 {
                out.push('(');
            }
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(' ');
                }
                grammar_expr_to_lark_with_indent(item, out, true, indent);
            }
            if parens && items.len() > 1 {
                out.push(')');
            }
        }
        GrammarExpr::Choice(alts) => {
            let multiline = alts.len() > 6;
            if parens && alts.len() > 1 {
                out.push('(');
            }
            for (i, alt) in alts.iter().enumerate() {
                if i > 0 {
                    if multiline {
                        out.push('\n');
                        for _ in 0..(indent + 4) {
                            out.push(' ');
                        }
                        out.push_str("| ");
                    } else {
                        out.push_str(" | ");
                    }
                }
                let child_indent = if multiline { indent + 6 } else { indent };
                grammar_expr_to_lark_with_indent(alt, out, true, child_indent);
            }
            if parens && alts.len() > 1 {
                if multiline {
                    out.push('\n');
                    for _ in 0..indent {
                        out.push(' ');
                    }
                }
                out.push(')');
            }
        }
        GrammarExpr::Literal(bytes) => {
            // Try UTF-8 first; fall back to hex
            if let Ok(s) = std::str::from_utf8(bytes) {
                write!(out, "\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")).unwrap();
            } else {
                let hex_str: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
                write!(out, "/*hex:{}*/", hex_str).unwrap();
            }
        }
        GrammarExpr::SpecialToken(token_id) => {
            write!(out, "@token({token_id})").unwrap();
        }
        GrammarExpr::Quantified(inner, Quantifier::Optional) => {
            grammar_expr_to_lark_with_indent(inner, out, true, indent);
            out.push('?');
        }
        GrammarExpr::Quantified(inner, Quantifier::ZeroPlus) => {
            grammar_expr_to_lark_with_indent(inner, out, true, indent);
            out.push('*');
        }
        GrammarExpr::Quantified(inner, Quantifier::OnePlus) => {
            grammar_expr_to_lark_with_indent(inner, out, true, indent);
            out.push('+');
        }
        GrammarExpr::Quantified(inner, Quantifier::Range(min, max)) => {
            grammar_expr_to_lark_with_indent(inner, out, true, indent);
            match max {
                Some(max) if min == max => write!(out, "{{{}}}", min).unwrap(),
                Some(max) => write!(out, "{{{},{}}}", min, max).unwrap(),
                None => write!(out, "{{{},}}", min).unwrap(),
            }
        },
        GrammarExpr::Epsilon => {
            out.push_str("/*eps*/");
        }
        GrammarExpr::Exclude { expr: inner, exclude } => {
            write!(out, "/*Exclude(").unwrap();
            grammar_expr_to_lark_with_indent(inner, out, false, indent);
            write!(out, " \\ ").unwrap();
            grammar_expr_to_lark_with_indent(exclude, out, false, indent);
            write!(out, ")*/").unwrap();
        }
        GrammarExpr::Intersect { expr: inner, intersect } => {
            write!(out, "/*Intersect(").unwrap();
            grammar_expr_to_lark_with_indent(inner, out, false, indent);
            write!(out, " & ").unwrap();
            grammar_expr_to_lark_with_indent(intersect, out, false, indent);
            write!(out, ")*/").unwrap();
        }
        GrammarExpr::CharClass { def, negate, utf8 } => {
            if *negate {
                write!(out, "[^{}]", def).unwrap();
            } else {
                write!(out, "[{}]", def).unwrap();
            }
            if *utf8 {
                write!(out, "/*utf8*/").unwrap();
            }
        }
        GrammarExpr::RawRegex(pattern) => {
            write!(out, "/{}/", pattern).unwrap();
        }
        GrammarExpr::LexerDfa(_) => {
            write!(out, "/*LexerDfa*/").unwrap();
        }
        GrammarExpr::AnyByte => {
            out.push_str("/./ /*AnyByte*/");
        }
        GrammarExpr::SeparatedSequence { items, separator, allow_empty } => {
            write!(out, "/*SeparatedSequence(sep=").unwrap();
            grammar_expr_to_lark_with_indent(separator, out, false, indent);
            write!(out, ", allow_empty={}, items=[", allow_empty).unwrap();
            for (i, (item, quantifier)) in items.iter().enumerate() {
                if i > 0 { write!(out, ", ").unwrap(); }
                grammar_expr_to_lark_with_indent(item, out, true, indent);
                if let Some(quantifier) = quantifier {
                    write!(out, "{}", quantifier_suffix(quantifier)).unwrap();
                }
            }
            write!(out, "])*/").unwrap();
        }
        GrammarExpr::ExprNFA(expr_nfa) => {
            write!(
                out,
                "/*ExprNFA(states={}, symbols={})*/",
                expr_nfa.nfa.states.len(),
                expr_nfa.symbols.len()
            )
            .unwrap();
        }
    }
}

struct Lowerer<'a> {
    rules: Vec<Rule>,
    terminal_map: FxHashMap<String, TerminalID>,
    literal_terminal_ids: FxHashMap<Vec<u8>, (TerminalID, String)>,
    terminals: Vec<Terminal>,
    terminal_expr_hash_index: FxHashMap<u64, Vec<TerminalID>>,
    special_terminal_ids: FxHashMap<u32, TerminalID>,
    nonterminal_ids: FxHashMap<String, NonterminalID>,
    next_nonterminal_id: NonterminalID,
    generated_nonterminal_counter: u32,
    terminal_names: BTreeMap<TerminalID, String>,
    terminal_ids_by_name: FxHashMap<String, TerminalID>,
    internal_terminal_names: FxHashSet<String>,
    named_rule_exprs: FxHashMap<String, &'a GrammarExpr>,
    named_rule_is_terminal: FxHashMap<String, bool>,
    rule_nullable: FxHashMap<String, bool>,
    terminal_bodies: FxHashMap<String, &'a GrammarExpr>,
    terminal_expr_cache: FxHashMap<String, Arc<Expr>>,
    nonnullable_named_rule_cache: FxHashMap<String, NonterminalID>,
    /// Large literal subsets recur inside many ExprNFA edge labels (notably
    /// JSON-Schema anyOf object unions).  Share their parser productions
    /// globally instead of re-emitting hundreds of identical literal arms for
    /// every slightly-different surrounding Choice.
    literal_choice_nonterminal_cache: FxHashMap<Vec<Vec<u8>>, NonterminalID>,
    cache_compound_exprs: bool,
    /// Exact compound expressions recur heavily in generated JSON-Schema grammars.
    /// Hash borrowed expressions on lookup and clone only first-seen structures so
    /// repeated subtrees share one already-lowered parser symbol.
    lowered_expr_cache: FxHashMap<u64, Vec<(GrammarExpr, Symbol)>>,
    /// Shared cache for repeat-exact nonterminals, keyed by (symbol, count).
    repeat_exact_cache: BTreeMap<(Symbol, usize), NonterminalID>,
    /// Shared cache for repeat-range nonterminals, keyed by (symbol, min, max).
    /// Only used for Left/Right shapes (bucket-based decomposition).
    repeat_range_cache: BTreeMap<(Symbol, usize, usize), NonterminalID>,
    /// Shared cache for repeat-max nonterminals, keyed by (symbol, max).
    /// Used by LeftBalanced/Balanced shapes for O(log N) range decomposition.
    repeat_max_cache: BTreeMap<(Symbol, usize), NonterminalID>,
    /// Shared cache for repeat-min1-max nonterminals, keyed by (symbol, max).
    /// repeat_min1_max_N matches exactly 1..N elements (N >= 1).
    repeat_min1_max_cache: BTreeMap<(Symbol, usize), NonterminalID>,
    parser_cfg_shape: Option<(RepeatTreeShape, CommaSepShape, bool)>,
    direct_regular_automaton: Option<DirectRegularAutomaton>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RepeatTreeShape {
    Balanced,
    Left,
    Right,
    /// Deterministic bounded-range lowering that uses a countdown chain.
    Countdown,
    /// Balanced exact decomposition (O(log N) tree depth) with balanced
    /// range alternation (O(log N) close-bracket resolution).
    LeftBalanced,
}

fn repeat_tree_shape() -> RepeatTreeShape {
    match std::env::var("GLRMASK_REPEAT_TREE_SHAPE").ok().as_deref() {
        Some(v) => repeat_tree_shape_from_value(v),
        None => RepeatTreeShape::Balanced,
    }
}

fn repeat_tree_shape_from_value(value: &str) -> RepeatTreeShape {
    match value {
        "left" => RepeatTreeShape::Left,
        "balanced" => RepeatTreeShape::Balanced,
        "countdown" | "deterministic" => RepeatTreeShape::Countdown,
        "leftbalanced" | "left_balanced" => RepeatTreeShape::LeftBalanced,
        _ => RepeatTreeShape::Right,
    }
}

fn right_repeat_range_front_bucket() -> usize {
    std::env::var("GLRMASK_RIGHT_REPEAT_RANGE_FRONT_BUCKET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128)
}

fn left_repeat_range_back_bucket() -> usize {
    std::env::var("GLRMASK_LEFT_REPEAT_RANGE_BACK_BUCKET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128)
}

fn exact_repeat_split(count: usize, shape: RepeatTreeShape) -> (usize, usize) {
    debug_assert!(count > 1);
    match shape {
        RepeatTreeShape::Balanced | RepeatTreeShape::Countdown | RepeatTreeShape::LeftBalanced => {
            let left = count / 2;
            (left, count - left)
        }
        RepeatTreeShape::Left => (count - 1, 1),
        RepeatTreeShape::Right => (1, count - 1),
    }
}

fn range_repeat_split(min: usize, max: usize, shape: RepeatTreeShape) -> (usize, usize) {
    debug_assert!(min < max);
    let width = max - min + 1;
    match shape {
        RepeatTreeShape::Balanced | RepeatTreeShape::Countdown | RepeatTreeShape::LeftBalanced => {
            let left_width = width / 2;
            let split = min + left_width - 1;
            (split, width - left_width)
        }
        RepeatTreeShape::Left => (max - 1, 1),
        RepeatTreeShape::Right => (min, width - 1),
    }
}

fn highest_power_of_two_le(n: usize) -> usize {
    debug_assert!(n > 0);
    1usize << ((usize::BITS - 1 - n.leading_zeros()) as usize)
}

fn char_class_pattern(def: &str, negate: bool) -> String {
    if negate {
        format!("[^{def}]")
    } else {
        format!("[{def}]")
    }
}

impl<'a> Lowerer<'a> {
    fn new() -> Self {
        Self {
            rules: Vec::new(),
            terminal_map: FxHashMap::default(),
            literal_terminal_ids: FxHashMap::default(),
            terminals: Vec::new(),
            terminal_expr_hash_index: FxHashMap::default(),
            special_terminal_ids: FxHashMap::default(),
            nonterminal_ids: FxHashMap::default(),
            next_nonterminal_id: 0,
            generated_nonterminal_counter: 0,
            terminal_names: BTreeMap::new(),
            terminal_ids_by_name: FxHashMap::default(),
            internal_terminal_names: FxHashSet::default(),
            named_rule_exprs: FxHashMap::default(),
            named_rule_is_terminal: FxHashMap::default(),
            rule_nullable: FxHashMap::default(),
            terminal_bodies: FxHashMap::default(),
            terminal_expr_cache: FxHashMap::default(),
            nonnullable_named_rule_cache: FxHashMap::default(),
            literal_choice_nonterminal_cache: FxHashMap::default(),
            cache_compound_exprs: false,
            lowered_expr_cache: FxHashMap::default(),
            repeat_exact_cache: BTreeMap::new(),
            repeat_range_cache: BTreeMap::new(),
            repeat_max_cache: BTreeMap::new(),
            repeat_min1_max_cache: BTreeMap::new(),
            direct_regular_automaton: None,
            parser_cfg_shape: None,
        }
    }

    fn repeat_shape(&self) -> RepeatTreeShape {
        self.parser_cfg_shape.map_or_else(repeat_tree_shape, |shape| shape.0)
    }
    fn separator_shape(&self) -> CommaSepShape {
        self.parser_cfg_shape.map_or_else(comma_sep_shape, |shape| shape.1)
    }
    fn repeat_rhs(&self, lhs: NonterminalID, symbol: Symbol) -> Vec<Symbol> {
        if self.parser_cfg_shape.is_some_and(|shape| shape.2) {
            vec![symbol, Symbol::Nonterminal(lhs)]
        } else { vec![Symbol::Nonterminal(lhs), symbol] }
    }

    fn nonterminal_id(&mut self, name: &str) -> NonterminalID {
        if let Some(&id) = self.nonterminal_ids.get(name) {
            id
        } else {
            let id = self.next_nonterminal_id;
            self.next_nonterminal_id += 1;
            self.nonterminal_ids.insert(name.to_string(), id);
            id
        }
    }

    #[inline]
    fn fresh_anonymous_nonterminal(&mut self) -> NonterminalID {
        let id = self.next_nonterminal_id;
        self.next_nonterminal_id += 1;
        id
    }

    #[inline]
    fn fresh_nonterminal(&mut self) -> NonterminalID {
        // The old generated name was never consumed by any caller; only its
        // counter increment mattered for keeping separately named generated
        // terminals stable. Preserve that increment and allocate the parser
        // helper directly by numeric ID.
        self.generated_nonterminal_counter += 1;
        self.fresh_anonymous_nonterminal()
    }

    fn expr_is_nullable(&self, expr: &GrammarExpr) -> bool {
        grammar_expr_is_nullable(expr, &self.rule_nullable)
    }

    fn strip_grouping(expr: &GrammarExpr) -> &GrammarExpr {
        match expr {
            GrammarExpr::Grouped(inner) => Self::strip_grouping(inner),
            _ => expr,
        }
    }

    fn top_level_alternatives(expr: &GrammarExpr) -> Vec<GrammarExpr> {
        match Self::strip_grouping(expr) {
            GrammarExpr::Choice(options) => options
                .iter()
                .map(|option| Self::strip_grouping(option).clone())
                .collect(),
            other => vec![other.clone()],
        }
    }

    fn exact_subtraction_alternatives(
        &self,
        lhs_name: &str,
        exclude: &GrammarExpr,
    ) -> Result<Vec<GrammarExpr>, GlrMaskError> {
        match exclude {
            GrammarExpr::Choice(options) => {
                let mut out = Vec::new();
                for option in options {
                    out.extend(self.exact_subtraction_alternatives(lhs_name, option)?);
                }
                Ok(out)
            }
            GrammarExpr::Grouped(inner) => Ok(Self::top_level_alternatives(inner)),
            GrammarExpr::Ref(name) => {
                let Some(false) = self.named_rule_is_terminal.get(name).copied() else {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "{lhs_name} - {name} requires {name} to name a nonterminal rule"
                    )));
                };
                let referenced_expr = self.named_rule_exprs.get(name).ok_or_else(|| {
                    GlrMaskError::GrammarParse(format!(
                        "unknown rule referenced in exact alternative subtraction: {name}"
                    ))
                })?;
                Ok(Self::top_level_alternatives(referenced_expr))
            }
            other => Ok(Self::top_level_alternatives(other)),
        }
    }

    fn canonical_exact_expr(&self, expr: &GrammarExpr) -> GrammarExpr {
        let mut visiting = HashSet::new();
        let mut memo = HashMap::new();
        self.canonical_exact_expr_inner(expr, &mut visiting, &mut memo)
    }

    fn canonical_exact_expr_inner(
        &self,
        expr: &GrammarExpr,
        visiting: &mut HashSet<String>,
        memo: &mut HashMap<String, GrammarExpr>,
    ) -> GrammarExpr {
        match Self::strip_grouping(expr) {
            GrammarExpr::Ref(name) => {
                if self.named_rule_is_terminal.get(name).copied().unwrap_or(false) {
                    return GrammarExpr::Ref(name.clone());
                }
                let Some(referenced) = self.named_rule_exprs.get(name) else {
                    return GrammarExpr::Ref(name.clone());
                };
                if let Some(canonical) = memo.get(name) {
                    return canonical.clone();
                }
                if !visiting.insert(name.clone()) {
                    return GrammarExpr::Ref(name.clone());
                }
                let canonical = self.canonical_exact_expr_inner(referenced, visiting, memo);
                visiting.remove(name);
                memo.insert(name.clone(), canonical.clone());
                canonical
            }
            GrammarExpr::Grouped(inner) => self.canonical_exact_expr_inner(inner, visiting, memo),
            GrammarExpr::Sequence(items) => GrammarExpr::Sequence(
                items
                    .iter()
                    .map(|item| self.canonical_exact_expr_inner(item, visiting, memo))
                    .collect(),
            ),
            GrammarExpr::Choice(items) => GrammarExpr::Choice(
                items
                    .iter()
                    .map(|item| self.canonical_exact_expr_inner(item, visiting, memo))
                    .collect(),
            ),
            GrammarExpr::Exclude { expr, exclude } => GrammarExpr::Exclude {
                expr: Box::new(self.canonical_exact_expr_inner(expr, visiting, memo)),
                exclude: Box::new(self.canonical_exact_expr_inner(exclude, visiting, memo)),
            },
            GrammarExpr::Intersect { expr, intersect } => GrammarExpr::Intersect {
                expr: Box::new(self.canonical_exact_expr_inner(expr, visiting, memo)),
                intersect: Box::new(self.canonical_exact_expr_inner(intersect, visiting, memo)),
            },
            GrammarExpr::Quantified(inner, quantifier) => GrammarExpr::Quantified(
                Box::new(self.canonical_exact_expr_inner(inner, visiting, memo)),
                quantifier.clone(),
            ),
            GrammarExpr::SeparatedSequence {
                items,
                separator,
                allow_empty,
            } => GrammarExpr::SeparatedSequence {
                items: items
                    .iter()
                    .map(|(item, quantifier)| {
                        (
                            self.canonical_exact_expr_inner(item, visiting, memo),
                            quantifier.clone(),
                        )
                    })
                    .collect(),
                separator: Box::new(self.canonical_exact_expr_inner(separator, visiting, memo)),
                allow_empty: *allow_empty,
            },
            GrammarExpr::ExprNFA(expr_nfa) => GrammarExpr::ExprNFA(Box::new(ExprNFA {
                nfa: expr_nfa.nfa.clone(),
                symbols: expr_nfa
                    .symbols
                    .iter()
                    .map(|symbol| self.canonical_exact_expr_inner(symbol, visiting, memo))
                    .collect(),
                state_names: expr_nfa.state_names.clone(),
                // Symbol canonicalization can merge formerly distinct labels,
                // so conservatively re-run minimization before lowering.
                is_determinized_and_minimized: false,
                prefer_direct_nfa_emission: expr_nfa.prefer_direct_nfa_emission,
                complete_parser_language: expr_nfa.complete_parser_language,
                canonical_dfa: None,
            })),
            GrammarExpr::Epsilon
            | GrammarExpr::Literal(_)
            | GrammarExpr::SpecialToken(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte => Self::strip_grouping(expr).clone(),
        }
    }

    fn exact_nonterminal_subtraction_expr(
        &self,
        expr: &GrammarExpr,
    ) -> Result<Option<GrammarExpr>, GlrMaskError> {
        let GrammarExpr::Exclude { expr: lhs_expr, exclude } = expr else {
            return Ok(None);
        };
        let GrammarExpr::Ref(lhs_name) = Self::strip_grouping(lhs_expr) else {
            return Ok(None);
        };
        let Some(false) = self.named_rule_is_terminal.get(lhs_name).copied() else {
            return Ok(None);
        };

        let lhs_rule_expr = self.named_rule_exprs.get(lhs_name).ok_or_else(|| {
            GlrMaskError::GrammarParse(format!(
                "unknown nonterminal referenced in exact alternative subtraction: {lhs_name}"
            ))
        })?;
        let mut remaining = Self::top_level_alternatives(lhs_rule_expr);
        let mut remaining_keys = remaining
            .iter()
            .map(|candidate| self.canonical_exact_expr(candidate))
            .collect::<Vec<_>>();
        for remove_alt in self.exact_subtraction_alternatives(lhs_name, exclude)? {
            let remove_alt_key = self.canonical_exact_expr(&remove_alt);
            let Some(position) = remaining_keys
                .iter()
                .position(|candidate| candidate == &remove_alt_key)
            else {
                continue;
            };
            remaining.remove(position);
            remaining_keys.remove(position);
        }

        Ok(Some(match remaining.len() {
            0 => GrammarExpr::Choice(Vec::new()),
            1 => remaining.pop().unwrap(),
            _ => GrammarExpr::Choice(remaining),
        }))
    }

    fn resolve_terminal_expr(
        &mut self,
        owner_name: Option<&str>,
        expr: &GrammarExpr,
    ) -> Result<Expr, GlrMaskError> {
        let mut visiting = HashSet::new();
        if let Some(name) = owner_name {
            visiting.insert(name.to_string());
        }
        grammar_expr_to_expr(
            expr,
            &self.terminal_bodies,
            &mut self.terminal_expr_cache,
            &mut visiting,
        )
    }

    fn nonnullable_terminal_symbol(
        &mut self,
        expr: &GrammarExpr,
    ) -> Result<Option<Symbol>, GlrMaskError> {
        match expr {
            GrammarExpr::Literal(bytes) => {
                if bytes.is_empty() {
                    return Ok(None);
                }
                Ok(Some(Symbol::Terminal(self.literal_terminal_id(bytes))))
            }
            GrammarExpr::SpecialToken(token_id) => Ok(Some(Symbol::Terminal(
                self.special_terminal_id(&format!("@token({token_id})"), *token_id),
            ))),
            GrammarExpr::Grouped(inner) => self.nonnullable_terminal_symbol(inner),
            GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::AnyByte
            | GrammarExpr::Exclude { .. }
            | GrammarExpr::Intersect { .. } => {
                let expr = self.resolve_terminal_expr(None, expr)?;
                let expr = if expr.is_nullable() {
                    Expr::Exclude {
                        expr: Box::new(expr),
                        exclude: Box::new(Expr::Epsilon),
                    }
                    .optimize()
                } else {
                    expr
                };
                let name = format!("__nonnullable_terminal_{}", self.generated_nonterminal_counter);
                let tid = self.register_terminal_expr(&name, expr);
                Ok(Some(Symbol::Terminal(tid)))
            }
            _ => Ok(None),
        }
    }

    fn lower_nonnullable_named_rule(&mut self, name: &str) -> Result<Symbol, GlrMaskError> {
        if let Some(&nt) = self.nonnullable_named_rule_cache.get(name) {
            return Ok(Symbol::Nonterminal(nt));
        }

        let expr = self
            .named_rule_exprs
            .get(name)
            .cloned()
            .ok_or_else(|| GlrMaskError::GrammarParse(format!("unknown rule referenced from SeparatedSequence: {name}")))?;
        let is_terminal = *self.named_rule_is_terminal.get(name).unwrap_or(&false);

        // If the referenced named rule is already nonnullable, reuse its
        // ordinary lowered symbol instead of synthesizing a second alias.
        if !self.rule_nullable.get(name).copied().unwrap_or(false)
            && !(is_terminal && self.internal_terminal_names.contains(name))
        {
            return Ok(Symbol::Nonterminal(self.nonterminal_id(name)));
        }

        let nt = self.fresh_nonterminal();
        self.nonnullable_named_rule_cache.insert(name.to_string(), nt);

        if is_terminal {
            let terminal_expr = self.resolve_terminal_expr(Some(name), &expr)?;
            let terminal_expr = if terminal_expr.is_nullable() {
                Expr::Exclude {
                    expr: Box::new(terminal_expr),
                    exclude: Box::new(Expr::Epsilon),
                }
                .optimize()
            } else {
                terminal_expr
            };

            if !matches!(terminal_expr, Expr::Epsilon) {
                let terminal_name = format!("__nonnullable_ref_{name}");
                let tid = self.register_terminal_expr(&terminal_name, terminal_expr);
                self.rules.push(Rule {
                    lhs: nt,
                    rhs: vec![Symbol::Terminal(tid)],
                });
            }
        } else {
            self.emit_nonnullable_expr(nt, &expr)?;
        }

        Ok(Symbol::Nonterminal(nt))
    }

    fn lower_nonnullable_expr_symbol(
        &mut self,
        expr: &GrammarExpr,
    ) -> Result<Option<Symbol>, GlrMaskError> {
        match expr {
            GrammarExpr::Epsilon => Ok(None),
            GrammarExpr::Literal(bytes) if bytes.is_empty() => Ok(None),
            GrammarExpr::Grouped(inner) => self.lower_nonnullable_expr_symbol(inner),
            GrammarExpr::Ref(name) => Ok(Some(self.lower_nonnullable_named_rule(name)?)),
            GrammarExpr::Quantified(inner, Quantifier::Optional) => self.lower_nonnullable_expr_symbol(inner),
            GrammarExpr::Exclude { .. } => {
                if let Some(lowered) = self.exact_nonterminal_subtraction_expr(expr)? {
                    return self.lower_nonnullable_expr_symbol(&lowered);
                }
                self.nonnullable_terminal_symbol(expr)
            }
            GrammarExpr::Literal(_)
            | GrammarExpr::SpecialToken(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::AnyByte
            | GrammarExpr::Intersect { .. } => self.nonnullable_terminal_symbol(expr),
            _ => {
                let nt = self.fresh_nonterminal();
                self.emit_nonnullable_expr(nt, expr)?;
                Ok(Some(Symbol::Nonterminal(nt)))
            }
        }
    }

    fn emit_nonnullable_sequence(
        &mut self,
        lhs: NonterminalID,
        parts: &[GrammarExpr],
    ) -> Result<(), GlrMaskError> {
        if parts.iter().any(|part| !self.expr_is_nullable(part)) {
            let mut rhs = Vec::with_capacity(parts.len());
            for part in parts {
                rhs.push(self.lower_expr_terminalish(part)?);
            }
            self.rules.push(Rule { lhs, rhs });
            return Ok(());
        }

        for (nonempty_index, nonempty_part) in parts.iter().enumerate() {
            let Some(nonempty_symbol) = self.lower_nonnullable_expr_symbol(nonempty_part)? else {
                continue;
            };

            let mut rhs = Vec::with_capacity(parts.len());
            for (index, part) in parts.iter().enumerate() {
                if index == nonempty_index {
                    rhs.push(nonempty_symbol.clone());
                } else {
                    rhs.push(self.lower_expr_terminalish(part)?);
                }
            }
            self.rules.push(Rule { lhs, rhs });
        }
        Ok(())
    }

    fn emit_nonnullable_expr(
        &mut self,
        lhs: NonterminalID,
        expr: &GrammarExpr,
    ) -> Result<(), GlrMaskError> {
        match expr {
            GrammarExpr::Grouped(inner) => {
                self.emit_nonnullable_expr(lhs, inner)?;
            }
            GrammarExpr::Ref(name) => {
                let symbol = self.lower_nonnullable_named_rule(name)?;
                self.rules.push(Rule { lhs, rhs: vec![symbol] });
            }
            GrammarExpr::Exclude { .. } => {
                if let Some(lowered) = self.exact_nonterminal_subtraction_expr(expr)? {
                    self.emit_nonnullable_expr(lhs, &lowered)?;
                } else if let Some(symbol) = self.nonnullable_terminal_symbol(expr)? {
                    self.rules.push(Rule { lhs, rhs: vec![symbol] });
                }
            }
            GrammarExpr::Literal(_)
            | GrammarExpr::SpecialToken(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte
            | GrammarExpr::Intersect { .. } => {
                if let Some(symbol) = self.nonnullable_terminal_symbol(expr)? {
                    self.rules.push(Rule { lhs, rhs: vec![symbol] });
                }
            }
            GrammarExpr::Sequence(parts) => {
                self.emit_nonnullable_sequence(lhs, parts)?;
            }
            GrammarExpr::Choice(options) => {
                for option in options {
                    self.emit_nonnullable_expr(lhs, option)?;
                }
            }
            GrammarExpr::Quantified(inner, Quantifier::Optional) => {
                self.emit_nonnullable_expr(lhs, inner)?;
            }
            GrammarExpr::Quantified(inner, Quantifier::ZeroPlus) | GrammarExpr::Quantified(inner, Quantifier::OnePlus) => {
                if let Some(symbol) = self.lower_nonnullable_expr_symbol(inner)? {
                    self.rules.push(Rule {
                        lhs,
                        rhs: vec![symbol.clone()],
                    });
                    self.rules.push(Rule {
                        lhs,
                        rhs: self.repeat_rhs(lhs, symbol),
                    });
                }
            }
            GrammarExpr::Quantified(expr, Quantifier::Range(min, max)) => {
                let Some(symbol) = self.lower_nonnullable_expr_symbol(expr)? else {
                    return Ok(());
                };
                let adjusted_min = (*min).max(1);
                if let Some(max) = max {
                    if adjusted_min > *max {
                        return Ok(());
                    }
                    let shape = self.repeat_shape();
                    let range_nonterminal = self.repeat_range_nonterminal(
                        &symbol,
                        adjusted_min,
                        *max,
                        shape,
                    );
                    self.rules.push(Rule {
                        lhs,
                        rhs: vec![Symbol::Nonterminal(range_nonterminal)],
                    });
                } else {
                    self.emit_repeat_range(lhs, expr, adjusted_min, None)?;
                }
            }
            GrammarExpr::SeparatedSequence { items, separator, .. } => {
                let shape = self.separator_shape();
                let (symbol, _) = self.lower_separated_sequence_inner(items, separator, shape)?;
                self.rules.push(Rule {
                    lhs,
                    rhs: vec![symbol],
                });
            }
            GrammarExpr::ExprNFA(expr_nfa) => {
                self.emit_expr_nfa_nonnullable(lhs, expr_nfa)?;
            }
            GrammarExpr::Epsilon => {}
        }
        Ok(())
    }

    fn literal_terminal_id(&mut self, bytes: &[u8]) -> TerminalID {
        if let Some((id, name)) = self.literal_terminal_ids.get(bytes) {
            let id = *id;
            if self.terminal_ids_by_name.get(name).copied() != Some(id) {
                self.terminal_ids_by_name.insert(name.clone(), id);
            }
            return id;
        }
        let pattern = bytes
            .iter()
            .map(|&byte| regex_escape_byte(byte))
            .collect::<String>();
        let name = String::from_utf8_lossy(bytes).into_owned();
        let id = self.terminal_id(&name, &pattern, false);
        self.literal_terminal_ids.insert(bytes.to_vec(), (id, name));
        id
    }

    fn terminal_id(&mut self, name: &str, pattern: &str, utf8: bool) -> TerminalID {
        let pattern_key = format!("{pattern}:{utf8}");
        if let Some(&id) = self.terminal_map.get(&pattern_key) {
            self.terminal_ids_by_name.insert(name.to_string(), id);
            return id;
        }
        let id = self.terminals.len() as TerminalID;
        self.terminal_map.insert(pattern_key, id);
        self.terminal_names.insert(id, name.to_string());
        self.terminal_ids_by_name.insert(name.to_string(), id);
        let name_bytes = name.as_bytes();
        let literal_pattern: String = name_bytes.iter().map(|&byte| regex_escape_byte(byte)).collect();
        if literal_pattern == pattern && !utf8 {
            self.terminals.push(Terminal::Literal {
                id,
                bytes: name_bytes.to_vec(),
            });
        } else {
            self.terminals.push(Terminal::Pattern {
                id,
                pattern: pattern.to_string(),
                utf8,
            });
        }
        id
    }

    fn special_terminal_id(&mut self, name: &str, token_id: u32) -> TerminalID {
        if let Some(&id) = self.special_terminal_ids.get(&token_id) {
            self.terminal_ids_by_name.insert(name.to_string(), id);
            return id;
        }
        let id = self.terminals.len() as TerminalID;
        self.special_terminal_ids.insert(token_id, id);
        self.terminal_names.insert(id, name.to_string());
        self.terminal_ids_by_name.insert(name.to_string(), id);
        self.terminals.push(Terminal::SpecialToken { id, token_id });
        id
    }

    fn repeat_exact_nonterminal(
        &mut self,
        symbol: &Symbol,
        count: usize,
        shape: RepeatTreeShape,
    ) -> NonterminalID {
        let key = (symbol.clone(), count);
        if let Some(&nonterminal) = self.repeat_exact_cache.get(&key) {
            return nonterminal;
        }

        let nonterminal = self.fresh_nonterminal();
        self.repeat_exact_cache.insert(key, nonterminal);
        match count {
            0 => self.rules.push(Rule {
                lhs: nonterminal,
                rhs: Vec::new(),
            }),
            1 => self.rules.push(Rule {
                lhs: nonterminal,
                rhs: vec![symbol.clone()],
            }),
            _ => {
                let (left, right) = exact_repeat_split(count, shape);
                let left_nonterminal =
                    self.repeat_exact_nonterminal(symbol, left, shape);
                let right_nonterminal =
                    self.repeat_exact_nonterminal(symbol, right, shape);
                self.rules.push(Rule {
                    lhs: nonterminal,
                    rhs: vec![
                        Symbol::Nonterminal(left_nonterminal),
                        Symbol::Nonterminal(right_nonterminal),
                    ],
                });
            }
        }
        nonterminal
    }

    fn repeat_max_nonterminal(
        &mut self,
        symbol: &Symbol,
        max: usize,
    ) -> NonterminalID {
        let key = (symbol.clone(), max);
        if let Some(&nt) = self.repeat_max_cache.get(&key) {
            return nt;
        }

        if max == 0 {
            let nt = self.fresh_nonterminal();
            self.repeat_max_cache.insert(key, nt);
            self.rules.push(Rule {
                lhs: nt,
                rhs: Vec::new(),
            });
            return nt;
        }

        let nt = self.fresh_nonterminal();
        self.repeat_max_cache.insert(key, nt);

        if let Some(span) = max.checked_add(1).filter(|span| span.is_power_of_two()) {
            let high = span / 2;
            debug_assert!(high > 0);

            let low_nt = self.repeat_max_nonterminal(symbol, high - 1);
            let exact_high_nt =
                self.repeat_exact_nonterminal(symbol, high, RepeatTreeShape::LeftBalanced);

            self.rules.push(Rule {
                lhs: nt,
                rhs: vec![Symbol::Nonterminal(low_nt)],
            });

            let rhs = if high == 1 {
                vec![Symbol::Nonterminal(exact_high_nt)]
            } else {
                vec![
                    Symbol::Nonterminal(exact_high_nt),
                    Symbol::Nonterminal(low_nt),
                ]
            };
            self.rules.push(Rule { lhs: nt, rhs });
        } else {
            let high = highest_power_of_two_le(max);
            debug_assert!(high > 0);
            debug_assert!(high <= max);

            let below_nt = self.repeat_max_nonterminal(symbol, high - 1);
            let exact_high_nt =
                self.repeat_exact_nonterminal(symbol, high, RepeatTreeShape::LeftBalanced);

            self.rules.push(Rule {
                lhs: nt,
                rhs: vec![Symbol::Nonterminal(below_nt)],
            });

            if max == high {
                self.rules.push(Rule {
                    lhs: nt,
                    rhs: vec![Symbol::Nonterminal(exact_high_nt)],
                });
            } else {
                let tail_nt = self.repeat_max_nonterminal(symbol, max - high);
                self.rules.push(Rule {
                    lhs: nt,
                    rhs: vec![
                        Symbol::Nonterminal(exact_high_nt),
                        Symbol::Nonterminal(tail_nt),
                    ],
                });
            }
        }

        nt
    }

    /// Returns a nonterminal matching exactly 1..=max occurrences of `symbol`.
    /// Defined as: `symbol repeat_max_{max-1}` â€” the first element is mandatory,
    /// the rest (0..max-1) are handled by `repeat_max`.
    fn repeat_min1_max_nonterminal(&mut self, symbol: &Symbol, max: usize) -> NonterminalID {
        debug_assert!(max >= 1);
        let key = (symbol.clone(), max);
        if let Some(&nt) = self.repeat_min1_max_cache.get(&key) {
            return nt;
        }

        let nt = self.fresh_nonterminal();
        self.repeat_min1_max_cache.insert(key, nt);

        if max == 1 {
            self.rules.push(Rule {
                lhs: nt,
                rhs: vec![symbol.clone()],
            });
            return nt;
        }

        let tail_nt = self.repeat_max_nonterminal(symbol, max - 1);
        self.rules.push(Rule {
            lhs: nt,
            rhs: vec![symbol.clone(), Symbol::Nonterminal(tail_nt)],
        });
        nt
    }

    fn repeat_range_nonterminal(
        &mut self,
        symbol: &Symbol,
        min: usize,
        max: usize,
        shape: RepeatTreeShape,
    ) -> NonterminalID {
        debug_assert!(min <= max);
        if min == max {
            return self.repeat_exact_nonterminal(symbol, min, shape);
        }

        let key = (symbol.clone(), min, max);
        if let Some(&nonterminal) = self.repeat_range_cache.get(&key) {
            return nonterminal;
        }

        if shape == RepeatTreeShape::Countdown {
            return self.repeat_range_nonterminal_countdown(symbol, min, max, shape);
        }

        match shape {
            RepeatTreeShape::LeftBalanced | RepeatTreeShape::Balanced => {
                return self.repeat_range_nonterminal_balanced(symbol, min, max, shape);
            }
            _ => {}
        }

        let nonterminal = self.fresh_nonterminal();
        self.repeat_range_cache.insert(key, nonterminal);

        match shape {
            RepeatTreeShape::Right if (max - min + 1) > right_repeat_range_front_bucket() => {
                let cutoff = (min + right_repeat_range_front_bucket() - 1).min(max);
                for count in min..=cutoff {
                    let exact_nonterminal =
                        self.repeat_exact_nonterminal(symbol, count, shape);
                    self.rules.push(Rule {
                        lhs: nonterminal,
                        rhs: vec![Symbol::Nonterminal(exact_nonterminal)],
                    });
                }
                if cutoff < max {
                    let tail_nonterminal = self.repeat_range_nonterminal(
                        symbol,
                        cutoff + 1,
                        max,
                        shape,
                    );
                    self.rules.push(Rule {
                        lhs: nonterminal,
                        rhs: vec![Symbol::Nonterminal(tail_nonterminal)],
                    });
                }
                return nonterminal;
            }
            RepeatTreeShape::Left if (max - min + 1) > left_repeat_range_back_bucket() => {
                let cutoff = max.saturating_sub(left_repeat_range_back_bucket() - 1).max(min);
                if min < cutoff {
                    let head_nonterminal = self.repeat_range_nonterminal(
                        symbol,
                        min,
                        cutoff - 1,
                        shape,
                    );
                    self.rules.push(Rule {
                        lhs: nonterminal,
                        rhs: vec![Symbol::Nonterminal(head_nonterminal)],
                    });
                }
                for count in cutoff..=max {
                    let exact_nonterminal =
                        self.repeat_exact_nonterminal(symbol, count, shape);
                    self.rules.push(Rule {
                        lhs: nonterminal,
                        rhs: vec![Symbol::Nonterminal(exact_nonterminal)],
                    });
                }
                return nonterminal;
            }
            _ => {}
        }
        let (split, _) = range_repeat_split(min, max, shape);
        let left_nonterminal = self.repeat_range_nonterminal(
            symbol,
            min,
            split,
            shape,
        );
        let right_nonterminal = self.repeat_range_nonterminal(
            symbol,
            split + 1,
            max,
            shape,
        );
        self.rules.push(Rule {
            lhs: nonterminal,
            rhs: vec![Symbol::Nonterminal(left_nonterminal)],
        });
        self.rules.push(Rule {
            lhs: nonterminal,
            rhs: vec![Symbol::Nonterminal(right_nonterminal)],
        });
        nonterminal
    }

    fn repeat_range_nonterminal_countdown(
        &mut self,
        symbol: &Symbol,
        min: usize,
        max: usize,
        shape: RepeatTreeShape,
    ) -> NonterminalID {
        let key = (symbol.clone(), min, max);
        if let Some(&nonterminal) = self.repeat_range_cache.get(&key) {
            return nonterminal;
        }

        let nonterminal = self.fresh_nonterminal();
        self.repeat_range_cache.insert(key, nonterminal);

        if min == 0 {
            self.rules.push(Rule {
                lhs: nonterminal,
                rhs: Vec::new(),
            });
        }

        let tail_nonterminal = if min > 0 {
            self.repeat_range_nonterminal(symbol, min - 1, max - 1, shape)
        } else {
            self.repeat_range_nonterminal(symbol, 0, max - 1, shape)
        };
        self.rules.push(Rule {
            lhs: nonterminal,
            rhs: vec![symbol.clone(), Symbol::Nonterminal(tail_nonterminal)],
        });
        nonterminal
    }

    fn repeat_range_nonterminal_balanced(
        &mut self,
        symbol: &Symbol,
        min: usize,
        max: usize,
        shape: RepeatTreeShape,
    ) -> NonterminalID {
        debug_assert!(min <= max);
        debug_assert!(matches!(shape, RepeatTreeShape::LeftBalanced | RepeatTreeShape::Balanced));
        if min == max {
            return self.repeat_exact_nonterminal(symbol, min, shape);
        }
        let delta = max - min;
        let max_nt = self.repeat_max_nonterminal(symbol, delta);
        if min == 0 {
            max_nt
        } else {
            let exact_nt = self.repeat_exact_nonterminal(symbol, min, shape);
            let result_nt = self.fresh_nonterminal();
            self.rules.push(Rule {
                lhs: result_nt,
                rhs: vec![
                    Symbol::Nonterminal(exact_nt),
                    Symbol::Nonterminal(max_nt),
                ],
            });
            result_nt
        }
    }

    fn emit_repeat_range(
        &mut self,
        lhs: NonterminalID,
        inner: &GrammarExpr,
        min: usize,
        max: Option<usize>,
    ) -> Result<(), GlrMaskError> {
        let symbol = self.lower_expr_terminalish(inner)?;
        let shape = self.repeat_shape();
        if let Some(max) = max {
            debug_assert!(min <= max);
            let range_nonterminal = self.repeat_range_nonterminal(&symbol, min, max, shape);
            self.rules.push(Rule {
                lhs,
                rhs: vec![Symbol::Nonterminal(range_nonterminal)],
            });
            return Ok(());
        }

        let suffix_nt = self.repeat_range_nonterminal(&symbol, 0, 1, shape);
        // Replace the finite optional suffix with a proper Kleene tail.
        self.rules.push(Rule { lhs: suffix_nt, rhs: Vec::new() });
        self.rules.push(Rule {
            lhs: suffix_nt,
            rhs: self.repeat_rhs(suffix_nt, symbol.clone()),
        });

        if min == 0 {
            self.rules.push(Rule { lhs, rhs: vec![Symbol::Nonterminal(suffix_nt)] });
        } else {
            let prefix_nt = self.repeat_exact_nonterminal(&symbol, min, shape);
            self.rules.push(Rule {
                lhs,
                rhs: vec![Symbol::Nonterminal(prefix_nt), Symbol::Nonterminal(suffix_nt)],
            });
        }
        Ok(())
    }

    fn expr_nfa_state_nonterminals(
        &mut self,
        state_count: usize,
        start: usize,
        hint: &str,
        start_lhs: Option<NonterminalID>,
    ) -> Result<Vec<NonterminalID>, GlrMaskError> {
        if start >= state_count {
            return Err(GlrMaskError::GrammarParse(format!(
                "ExprNFA start state {} is out of range for {} states",
                start, state_count
            )));
        }

        let mut nts = Vec::with_capacity(state_count);
        for state_index in 0..state_count {
            if Some(state_index) == start_lhs.map(|_| start) {
                nts.push(start_lhs.unwrap());
            } else {
                let _ = hint;
                nts.push(self.fresh_anonymous_nonterminal());
            }
        }
        Ok(nts)
    }

    fn emit_expr_dfa_leftlinear(
        &mut self,
        lhs: NonterminalID,
        expr_nfa: &ExprNFA,
        dfa: &DFA,
    ) -> Result<(), GlrMaskError> {
        let profile_enabled = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
        let rules_before = self.rules.len();
        let state_count = dfa.states.len();
        // Preserve the historical one-symbol edge representation for ordinary
        // NFAs. Large closed discriminator unions are the only shape where
        // flattening saves thousands of anonymous wrapper productions.
        let flatten_sequence_transitions = state_count >= 512;
        let start = dfa.start_state as usize;
        let nts = self.expr_nfa_state_nonterminals(state_count, start, "expr_nfa_prefix", None)?;
        let start_nt = nts[start];
        self.rules.push(Rule {
            lhs: start_nt,
            rhs: Vec::new(),
        });

        // A transition label can occur on many DFA edges. Lower it exactly once
        // and reuse the resulting symbol sequence. Compound labels are immutable
        // grammar fragments just like literals and references; recreating their
        // wrapper nonterminals per edge can multiply a compact object automaton
        // into tens of thousands of equivalent productions.
        let mut reusable_symbols: Vec<Option<Vec<Symbol>>> = vec![None; expr_nfa.symbols.len()];
        let mut reusable_symbol_cache_hits = 0usize;
        let mut transition_count = 0usize;
        for (state_index, state) in dfa.states.iter().enumerate() {
            for (label, target) in &state.transitions {
                transition_count += 1;
                let target = *target as usize;
                if target >= state_count {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "ExprNFA transition from state {state_index} targets out-of-range state {target}"
                    )));
                }
                let symbol_expr = expr_nfa.symbol_for_label(*label).ok_or_else(|| {
                    GlrMaskError::GrammarParse(format!(
                        "ExprNFA transition label {label} is not a valid symbol index"
                    ))
                })?;
                let cache_slot = reusable_symbols.get_mut(*label as usize).ok_or_else(|| {
                    GlrMaskError::GrammarParse(format!(
                        "ExprNFA transition label {label} is not a valid symbol index"
                    ))
                })?;
                let symbols = if let Some(symbols) = cache_slot.as_ref() {
                    reusable_symbol_cache_hits += 1;
                    symbols.clone()
                } else {
                    let symbols = if flatten_sequence_transitions {
                        self.lower_expr_nfa_transition_symbols(symbol_expr)?
                    } else {
                        vec![self.lower_expr_terminalish(symbol_expr)?]
                    };
                    *cache_slot = Some(symbols.clone());
                    symbols
                };
                let mut rhs = Vec::with_capacity(1 + symbols.len());
                rhs.push(Symbol::Nonterminal(nts[state_index]));
                rhs.extend(symbols);
                self.rules.push(Rule { lhs: nts[target], rhs });
            }
        }

        for (state_index, state) in dfa.states.iter().enumerate() {
            if state.is_accepting {
                self.rules.push(Rule {
                    lhs,
                    rhs: vec![Symbol::Nonterminal(nts[state_index])],
                });
            }
        }

        if profile_enabled && transition_count >= 64 {
            eprintln!(
                "[glrmask/profile][expr_nfa_emit] dfa_states={} transitions={} symbols={} reusable_cache_hits={} emitted_rules={}",
                state_count,
                transition_count,
                expr_nfa.symbols.len(),
                reusable_symbol_cache_hits,
                self.rules.len() - rules_before,
            );
        }

        Ok(())
    }

    /// Emit an already-minimized acyclic DFA without materializing the
    /// intermediate DFA copy that only exists to apply the standard reindexing
    /// convention. The ordering and rule emission are intentionally identical
    /// to `emit_expr_dfa_leftlinear(reindex_minimized_acyclic_dfa(dfa))`.
    fn emit_canonical_expr_dfa_leftlinear(
        &mut self,
        lhs: NonterminalID,
        expr_nfa: &ExprNFA,
        dfa: &DFA,
    ) -> Result<(), GlrMaskError> {
        let profile_enabled = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
        let rules_before = self.rules.len();
        let reindex = reindex_minimized_acyclic_order(dfa);
        let state_count = reindex.old_states_in_new_order.len();
        // Preserve ordinary ExprNFA grammar structure; only this large-union
        // case benefits materially from flattening concatenated edge labels.
        let flatten_sequence_transitions = state_count >= 512;
        let start = reindex.start_state as usize;
        let nts = self.expr_nfa_state_nonterminals(state_count, start, "expr_nfa_prefix", None)?;
        let start_nt = nts[start];
        self.rules.push(Rule {
            lhs: start_nt,
            rhs: Vec::new(),
        });

        let mut reusable_symbols: Vec<Option<Vec<Symbol>>> = vec![None; expr_nfa.symbols.len()];
        let mut reusable_symbol_cache_hits = 0usize;
        let mut transition_count = 0usize;
        for (state_index, &old_state) in reindex.old_states_in_new_order.iter().enumerate() {
            let state = &dfa.states[old_state];
            for (label, target) in &state.transitions {
                let target = *target as usize;
                let Some(&target_state) = reindex.new_state_of_old.get(target) else {
                    continue;
                };
                if target_state == u32::MAX {
                    continue;
                }
                transition_count += 1;
                let symbol_expr = expr_nfa.symbol_for_label(*label).ok_or_else(|| {
                    GlrMaskError::GrammarParse(format!(
                        "ExprNFA transition label {label} is not a valid symbol index"
                    ))
                })?;
                let cache_slot = reusable_symbols.get_mut(*label as usize).ok_or_else(|| {
                    GlrMaskError::GrammarParse(format!(
                        "ExprNFA transition label {label} is not a valid symbol index"
                    ))
                })?;
                let symbols = if let Some(symbols) = cache_slot.as_ref() {
                    reusable_symbol_cache_hits += 1;
                    symbols.clone()
                } else {
                    let symbols = if flatten_sequence_transitions {
                        self.lower_expr_nfa_transition_symbols(symbol_expr)?
                    } else {
                        vec![self.lower_expr_terminalish(symbol_expr)?]
                    };
                    *cache_slot = Some(symbols.clone());
                    symbols
                };
                let mut rhs = Vec::with_capacity(1 + symbols.len());
                rhs.push(Symbol::Nonterminal(nts[state_index]));
                rhs.extend(symbols);
                self.rules.push(Rule {
                    lhs: nts[target_state as usize],
                    rhs,
                });
            }
        }

        for (state_index, &old_state) in reindex.old_states_in_new_order.iter().enumerate() {
            if dfa.states[old_state].is_accepting {
                self.rules.push(Rule {
                    lhs,
                    rhs: vec![Symbol::Nonterminal(nts[state_index])],
                });
            }
        }

        if profile_enabled && transition_count >= 64 {
            eprintln!(
                "[glrmask/profile][expr_nfa_emit] dfa_states={} transitions={} symbols={} reusable_cache_hits={} emitted_rules={}",
                state_count,
                transition_count,
                expr_nfa.symbols.len(),
                reusable_symbol_cache_hits,
                self.rules.len() - rules_before,
            );
        }

        Ok(())
    }

    fn emit_expr_dfa_leftlinear_nonnullable(
        &mut self,
        lhs: NonterminalID,
        expr_nfa: &ExprNFA,
        dfa: &DFA,
    ) -> Result<(), GlrMaskError> {
        let state_count = dfa.states.len();
        let start = dfa.start_state as usize;
        if start >= state_count {
            return Err(GlrMaskError::GrammarParse(format!(
                "ExprNFA start state {} is out of range for {} states",
                dfa.start_state, state_count
            )));
        }

        let nullable_nts = self.expr_nfa_state_nonterminals(
            state_count,
            start,
            "expr_nfa_nullable_prefix",
            None,
        )?;
        let nonnullable_nts = self.expr_nfa_state_nonterminals(
            state_count,
            start,
            "expr_nfa_nonnullable_prefix",
            None,
        )?;

        self.rules.push(Rule {
            lhs: nullable_nts[start],
            rhs: Vec::new(),
        });

        for (state_index, state) in dfa.states.iter().enumerate() {
            for (label, target) in &state.transitions {
                let target = *target as usize;
                if target >= state_count {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "ExprNFA transition from state {state_index} targets out-of-range state {target}"
                    )));
                }
                let symbol_expr = expr_nfa.symbol_for_label(*label).ok_or_else(|| {
                    GlrMaskError::GrammarParse(format!(
                        "ExprNFA transition label {label} is not a valid symbol index"
                    ))
                })?;

                if self.expr_is_nullable(symbol_expr) {
                    let symbol = self.lower_expr_terminalish(symbol_expr)?;
                    self.rules.push(Rule {
                        lhs: nullable_nts[target],
                        rhs: vec![Symbol::Nonterminal(nullable_nts[state_index]), symbol],
                    });
                }

                if let Some(symbol) = self.lower_nonnullable_expr_symbol(symbol_expr)? {
                    self.rules.push(Rule {
                        lhs: nonnullable_nts[target],
                        rhs: vec![Symbol::Nonterminal(nullable_nts[state_index]), symbol],
                    });
                }

                let symbol = self.lower_expr_terminalish(symbol_expr)?;
                self.rules.push(Rule {
                    lhs: nonnullable_nts[target],
                    rhs: vec![Symbol::Nonterminal(nonnullable_nts[state_index]), symbol],
                });
            }
        }

        for (state_index, state) in dfa.states.iter().enumerate() {
            if state.is_accepting {
                self.rules.push(Rule {
                    lhs,
                    rhs: vec![Symbol::Nonterminal(nonnullable_nts[state_index])],
                });
            }
        }

        Ok(())
    }

    fn direct_regular_add_state(automaton: &mut DirectRegularAutomaton) -> u32 {
        let state = automaton.states.len() as u32;
        automaton.states.push(DirectRegularState::default());
        state
    }

    fn direct_regular_terminal_id(
        &mut self,
        expr: &GrammarExpr,
    ) -> Result<TerminalID, GlrMaskError> {
        if let GrammarExpr::Ref(name) = expr {
            let Some(true) = self.named_rule_is_terminal.get(name).copied() else {
                return Err(GlrMaskError::GrammarParse(format!(
                    "direct regular edge references nonterminal rule {name}"
                )));
            };
            if let Some(&terminal) = self.terminal_ids_by_name.get(name) {
                return Ok(terminal);
            }
            let body = self.named_rule_exprs.get(name).ok_or_else(|| {
                GlrMaskError::GrammarParse(format!(
                    "direct regular edge references unknown terminal rule {name}"
                ))
            })?;
            let resolved = self.resolve_terminal_expr(Some(name), body)?;
            return Ok(self.register_terminal_expr(name, resolved));
        }

        match self.lower_expr_terminalish(expr)? {
            Symbol::Terminal(terminal) => Ok(terminal),
            Symbol::Nonterminal(nonterminal) => Err(GlrMaskError::GrammarParse(format!(
                "direct regular edge unexpectedly lowered to nonterminal {nonterminal}"
            ))),
        }
    }

    fn emit_direct_regular_expr(
        &mut self,
        automaton: &mut DirectRegularAutomaton,
        expr: &GrammarExpr,
        from: u32,
        to: u32,
    ) -> Result<(), GlrMaskError> {
        match expr {
            GrammarExpr::Grouped(inner) => {
                self.emit_direct_regular_expr(automaton, inner, from, to)
            }
            GrammarExpr::Sequence(parts) => {
                if parts.is_empty() {
                    automaton.states[from as usize].epsilons.push(to);
                    return Ok(());
                }
                let mut current = from;
                for (index, part) in parts.iter().enumerate() {
                    let next = if index + 1 == parts.len() {
                        to
                    } else {
                        Self::direct_regular_add_state(automaton)
                    };
                    self.emit_direct_regular_expr(automaton, part, current, next)?;
                    current = next;
                }
                Ok(())
            }
            GrammarExpr::Choice(options) => {
                for option in options {
                    self.emit_direct_regular_expr(automaton, option, from, to)?;
                }
                Ok(())
            }
            GrammarExpr::Epsilon => {
                automaton.states[from as usize].epsilons.push(to);
                Ok(())
            }
            GrammarExpr::Quantified(inner, Quantifier::Optional) => {
                automaton.states[from as usize].epsilons.push(to);
                self.emit_direct_regular_expr(automaton, inner, from, to)
            }
            GrammarExpr::Quantified(inner, Quantifier::ZeroPlus) => {
                automaton.states[from as usize].epsilons.push(to);
                let body_end = Self::direct_regular_add_state(automaton);
                self.emit_direct_regular_expr(automaton, inner, from, body_end)?;
                automaton.states[body_end as usize].epsilons.push(from);
                Ok(())
            }
            GrammarExpr::Quantified(inner, Quantifier::OnePlus) => {
                let body_end = Self::direct_regular_add_state(automaton);
                self.emit_direct_regular_expr(automaton, inner, from, body_end)?;
                automaton.states[body_end as usize].epsilons.push(from);
                automaton.states[body_end as usize].epsilons.push(to);
                Ok(())
            }
            GrammarExpr::Quantified(inner, Quantifier::Range(min, max)) => {
                let mut current = from;
                for index in 0..*min {
                    let next = if index + 1 == *min && max == &Some(*min) {
                        to
                    } else {
                        Self::direct_regular_add_state(automaton)
                    };
                    self.emit_direct_regular_expr(automaton, inner, current, next)?;
                    current = next;
                }
                if *min == 0 && max == &Some(0) {
                    automaton.states[from as usize].epsilons.push(to);
                    return Ok(());
                }
                match max {
                    Some(max) => {
                        for index in *min..*max {
                            automaton.states[current as usize].epsilons.push(to);
                            let next = if index + 1 == *max {
                                to
                            } else {
                                Self::direct_regular_add_state(automaton)
                            };
                            self.emit_direct_regular_expr(automaton, inner, current, next)?;
                            current = next;
                        }
                        if *min == *max && current != to {
                            automaton.states[current as usize].epsilons.push(to);
                        }
                    }
                    None => {
                        automaton.states[current as usize].epsilons.push(to);
                        let body_end = Self::direct_regular_add_state(automaton);
                        self.emit_direct_regular_expr(automaton, inner, current, body_end)?;
                        automaton.states[body_end as usize].epsilons.push(current);
                    }
                }
                Ok(())
            }
            GrammarExpr::SeparatedSequence { .. } | GrammarExpr::ExprNFA(_) => {
                Err(GlrMaskError::GrammarParse(
                    "unsupported nested expression in direct regular edge".into(),
                ))
            }
            GrammarExpr::Ref(_)
            | GrammarExpr::Literal(_)
            | GrammarExpr::SpecialToken(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte
            | GrammarExpr::Exclude { .. }
            | GrammarExpr::Intersect { .. } => {
                let terminal = self.direct_regular_terminal_id(expr)?;
                automaton.states[from as usize]
                    .transitions
                    .entry(terminal)
                    .or_default()
                    .push(to);
                Ok(())
            }
        }
    }

    fn lower_direct_regular_automaton(
        &mut self,
        expr_nfa: &ExprNFA,
    ) -> Result<DirectRegularAutomaton, GlrMaskError> {
        let mut automaton = DirectRegularAutomaton {
            states: vec![DirectRegularState::default(); expr_nfa.nfa.states.len()],
            start_states: expr_nfa.nfa.start_states.clone(),
        };
        for (state_index, state) in expr_nfa.nfa.states.iter().enumerate() {
            automaton.states[state_index].is_accepting = state.is_accepting;
            automaton.states[state_index]
                .epsilons
                .extend(state.epsilons.iter().copied());
            for (&label, targets) in &state.transitions {
                let label_index = usize::try_from(label).map_err(|_| {
                    GlrMaskError::GrammarParse(format!(
                        "ExprNFA transition label {label} is negative"
                    ))
                })?;
                let symbol_expr = expr_nfa.symbols.get(label_index).ok_or_else(|| {
                    GlrMaskError::GrammarParse(format!(
                        "ExprNFA transition label {label} is not a valid symbol index"
                    ))
                })?;
                for &target in targets {
                    self.emit_direct_regular_expr(
                        &mut automaton,
                        symbol_expr,
                        state_index as u32,
                        target,
                    )?;
                }
            }
        }
        for state in &mut automaton.states {
            state.epsilons.sort_unstable();
            state.epsilons.dedup();
            for targets in state.transitions.values_mut() {
                targets.sort_unstable();
                targets.dedup();
            }
        }
        Ok(automaton)
    }

    fn emit_expr_nfa_direct_leftlinear(
        &mut self,
        lhs: NonterminalID,
        expr_nfa: &ExprNFA,
    ) -> Result<(), GlrMaskError> {
        if expr_nfa.complete_parser_language {
            let direct_automaton = self.lower_direct_regular_automaton(expr_nfa)?;
            if self.direct_regular_automaton.replace(direct_automaton).is_some() {
                return Err(GlrMaskError::GrammarParse(
                    "multiple direct regular parser automata in one grammar".into(),
                ));
            }
            // The retained automaton is the complete parser language. Emitting
            // its states again as tens of thousands of CFG rules only creates
            // data that direct table/runtime paths immediately ignore.
            return Ok(());
        }
        let state_count = expr_nfa.nfa.states.len();
        if state_count == 0 || expr_nfa.nfa.start_states.is_empty() {
            return Err(GlrMaskError::GrammarParse(
                "ExprNFA direct emission requires at least one start state".into(),
            ));
        }
        let nts = self.expr_nfa_state_nonterminals(
            state_count,
            expr_nfa.nfa.start_states[0] as usize,
            "expr_nfa_direct",
            None,
        )?;
        let rules_before = self.rules.len();
        let mut reusable_symbols: Vec<Option<Vec<Symbol>>> = vec![None; expr_nfa.symbols.len()];
        let mut labeled_edges = 0usize;
        let mut epsilon_edges = 0usize;

        for &start in &expr_nfa.nfa.start_states {
            let start = start as usize;
            if start >= state_count {
                return Err(GlrMaskError::GrammarParse(format!(
                    "ExprNFA start state {start} is out of range for {state_count} states"
                )));
            }
            self.rules.push(Rule {
                lhs: nts[start],
                rhs: Vec::new(),
            });
        }

        for (state_index, state) in expr_nfa.nfa.states.iter().enumerate() {
            for &target in &state.epsilons {
                let target = target as usize;
                if target >= state_count {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "ExprNFA epsilon transition from state {state_index} targets out-of-range state {target}"
                    )));
                }
                epsilon_edges += 1;
                self.rules.push(Rule {
                    lhs: nts[target],
                    rhs: vec![Symbol::Nonterminal(nts[state_index])],
                });
            }
            for (&label, targets) in &state.transitions {
                let label_index = usize::try_from(label).map_err(|_| {
                    GlrMaskError::GrammarParse(format!(
                        "ExprNFA transition label {label} is negative"
                    ))
                })?;
                let symbol_expr = expr_nfa.symbols.get(label_index).ok_or_else(|| {
                    GlrMaskError::GrammarParse(format!(
                        "ExprNFA transition label {label} is not a valid symbol index"
                    ))
                })?;
                let symbols = if let Some(symbols) = reusable_symbols[label_index].as_ref() {
                    symbols.clone()
                } else {
                    let symbols = self.lower_expr_nfa_transition_symbols(symbol_expr)?;
                    reusable_symbols[label_index] = Some(symbols.clone());
                    symbols
                };
                for &target in targets {
                    let target = target as usize;
                    if target >= state_count {
                        return Err(GlrMaskError::GrammarParse(format!(
                            "ExprNFA transition from state {state_index} targets out-of-range state {target}"
                        )));
                    }
                    labeled_edges += 1;
                    let mut rhs = Vec::with_capacity(1 + symbols.len());
                    rhs.push(Symbol::Nonterminal(nts[state_index]));
                    rhs.extend(symbols.iter().cloned());
                    self.rules.push(Rule {
                        lhs: nts[target],
                        rhs,
                    });
                }
            }
        }

        for (state_index, state) in expr_nfa.nfa.states.iter().enumerate() {
            if state.is_accepting {
                self.rules.push(Rule {
                    lhs,
                    rhs: vec![Symbol::Nonterminal(nts[state_index])],
                });
            }
        }

        if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
        {
            eprintln!(
                "[glrmask/profile][expr_nfa_direct_emit] states={} labeled_edges={} epsilon_edges={} symbols={} emitted_rules={}",
                state_count,
                labeled_edges,
                epsilon_edges,
                expr_nfa.symbols.len(),
                self.rules.len() - rules_before,
            );
        }
        Ok(())
    }

    fn emit_expr_nfa(&mut self, lhs: NonterminalID, expr_nfa: &ExprNFA) -> Result<(), GlrMaskError> {
        if expr_nfa.prefer_direct_nfa_emission {
            return self.emit_expr_nfa_direct_leftlinear(lhs, expr_nfa);
        }
        if expr_nfa.is_determinized_and_minimized
            && let Some(dfa) = &expr_nfa.canonical_dfa
            && dfa.compute_is_acyclic()
        {
            return self.emit_canonical_expr_dfa_leftlinear(lhs, expr_nfa, dfa);
        }
        let dfa = expr_nfa.determinize_and_minimize();
        self.emit_expr_dfa_leftlinear(lhs, expr_nfa, &dfa)
    }

    fn emit_expr_nfa_nonnullable(
        &mut self,
        lhs: NonterminalID,
        expr_nfa: &ExprNFA,
    ) -> Result<(), GlrMaskError> {
        let dfa = expr_nfa.determinize_and_minimize();
        self.emit_expr_dfa_leftlinear_nonnullable(lhs, expr_nfa, &dfa)
    }

    fn try_emit_shared_literal_choice_subset(
        &mut self,
        lhs: NonterminalID,
        options: &[GrammarExpr],
    ) -> Result<bool, GlrMaskError> {
        const MIN_SHARED_LITERAL_ARMS: usize = 16;

        fn literal_bytes(expr: &GrammarExpr) -> Option<&[u8]> {
            match expr {
                GrammarExpr::Grouped(inner) => literal_bytes(inner),
                GrammarExpr::Literal(bytes) if !bytes.is_empty() => Some(bytes),
                _ => None,
            }
        }

        let literal_key = options
            .iter()
            .filter_map(literal_bytes)
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        if literal_key.len() < MIN_SHARED_LITERAL_ARMS {
            return Ok(false);
        }

        let shared_nt = if let Some(&cached) = self.literal_choice_nonterminal_cache.get(&literal_key) {
            cached
        } else {
            let nt = self.fresh_nonterminal();
            for bytes in &literal_key {
                let terminal = self.literal_terminal_id(bytes);
                self.rules.push(Rule {
                    lhs: nt,
                    rhs: vec![Symbol::Terminal(terminal)],
                });
            }
            self.literal_choice_nonterminal_cache
                .insert(literal_key, nt);
            nt
        };

        self.rules.push(Rule {
            lhs,
            rhs: vec![Symbol::Nonterminal(shared_nt)],
        });
        Ok(true)
    }

    fn lower_expr(&mut self, expr: &GrammarExpr) -> Symbol {
        let cacheable = self.cache_compound_exprs
            && matches!(
                expr,
                GrammarExpr::Grouped(_)
                    | GrammarExpr::Sequence(_)
                    | GrammarExpr::Choice(_)
                    | GrammarExpr::Exclude { .. }
                    | GrammarExpr::Intersect { .. }
                    | GrammarExpr::Quantified(_, _)
                    | GrammarExpr::SeparatedSequence { .. }
            );
        let cache_hash = cacheable.then(|| {
            let mut hasher = FxHasher::default();
            expr.hash(&mut hasher);
            hasher.finish()
        });
        if let Some(hash) = cache_hash
            && let Some(candidates) = self.lowered_expr_cache.get(&hash)
            && let Some((_, symbol)) = candidates.iter().find(|(candidate, _)| candidate == expr)
        {
            return symbol.clone();
        }

        fn emit(lowerer: &mut Lowerer, lhs: NonterminalID, expr: &GrammarExpr) -> Result<(), GlrMaskError> {
            match expr {
                GrammarExpr::Grouped(inner) => emit(lowerer, lhs, inner)?,
                GrammarExpr::Sequence(parts) => {
                    let mut rhs = Vec::new();
                    for part in parts {
                        rhs.push(lowerer.lower_expr(part));
                    }
                    lowerer.rules.push(Rule { lhs, rhs });
                }
                GrammarExpr::Choice(options) => {
                    let shared_literals =
                        lowerer.try_emit_shared_literal_choice_subset(lhs, options)?;
                    for option in options {
                        if shared_literals {
                            let is_nonempty_literal = match option {
                                GrammarExpr::Grouped(inner) => {
                                    matches!(inner.as_ref(), GrammarExpr::Literal(bytes) if !bytes.is_empty())
                                }
                                GrammarExpr::Literal(bytes) => !bytes.is_empty(),
                                _ => false,
                            };
                            if is_nonempty_literal {
                                continue;
                            }
                        }
                        emit(lowerer, lhs, option)?;
                    }
                }
                GrammarExpr::Quantified(inner, Quantifier::Optional) => {
                    lowerer.rules.push(Rule { lhs, rhs: Vec::new() });
                    emit(lowerer, lhs, inner)?;
                }
                GrammarExpr::Quantified(inner, Quantifier::ZeroPlus) => {
                    let symbol = lowerer.lower_expr(inner);
                    lowerer.rules.push(Rule { lhs, rhs: Vec::new() });
                    lowerer.rules.push(Rule {
                        lhs,
                        rhs: lowerer.repeat_rhs(lhs, symbol),
                    });
                }
                GrammarExpr::Quantified(inner, Quantifier::OnePlus) => {
                    let symbol = lowerer.lower_expr(inner);
                    lowerer.rules.push(Rule {
                        lhs,
                        rhs: vec![symbol.clone()],
                    });
                    lowerer.rules.push(Rule {
                        lhs,
                        rhs: lowerer.repeat_rhs(lhs, symbol),
                    });
                }
                GrammarExpr::Quantified(expr, Quantifier::Range(min, max)) => {
                    lowerer.emit_repeat_range(lhs, expr, *min, *max)?;
                }
                GrammarExpr::SeparatedSequence { items, separator, allow_empty } => {
                    let shape = lowerer.separator_shape();
                    let (sym, can_be_empty) =
                        lowerer.lower_separated_sequence_inner(items, separator, shape)?;
                    lowerer.rules.push(Rule { lhs, rhs: vec![sym] });
                    if *allow_empty && can_be_empty {
                        lowerer.rules.push(Rule { lhs, rhs: Vec::new() });
                    }
                }
                GrammarExpr::Epsilon => {
                    lowerer.rules.push(Rule { lhs, rhs: Vec::new() });
                }
                GrammarExpr::ExprNFA(expr_nfa) => {
                    lowerer.emit_expr_nfa(lhs, expr_nfa)?;
                }
                _ => {
                    if let Some(lowered) = lowerer.exact_nonterminal_subtraction_expr(expr)? {
                        emit(lowerer, lhs, &lowered)?;
                        return Ok(());
                    }
                    let symbol = lowerer.lower_expr_terminalish(expr)?;
                    lowerer.rules.push(Rule {
                        lhs,
                        rhs: vec![symbol],
                    });
                }
            }
            Ok(())
        }

        let nonterminal = self.fresh_nonterminal();
        emit(self, nonterminal, expr)
            .expect("grammar lowering should not fail for internal expression emission");
        let symbol = Symbol::Nonterminal(nonterminal);
        if let Some(hash) = cache_hash {
            self.lowered_expr_cache
                .entry(hash)
                .or_default()
                .push((expr.clone(), symbol.clone()));
        }
        symbol
    }

    /// Lowers an ExprNFA transition label to its grammar RHS.  Sequence labels
    /// occur heavily in JSON Schema object unions: e.g. `{` + body + `}`.
    /// Emitting those as a single anonymous nonterminal is language-equivalent
    /// but creates one extra production per edge.  Flattening is safe because
    /// an ExprNFA edge already denotes concatenation of its label expression.
    fn lower_expr_nfa_transition_symbols(
        &mut self,
        expr: &GrammarExpr,
    ) -> Result<Vec<Symbol>, GlrMaskError> {
        match expr {
            GrammarExpr::Grouped(inner) => self.lower_expr_nfa_transition_symbols(inner),
            GrammarExpr::Sequence(parts) => {
                let mut symbols = Vec::with_capacity(parts.len());
                for part in parts {
                    symbols.extend(self.lower_expr_nfa_transition_symbols(part)?);
                }
                Ok(symbols)
            }
            _ => Ok(vec![self.lower_expr_terminalish(expr)?]),
        }
    }

    fn lower_expr_terminalish(&mut self, expr: &GrammarExpr) -> Result<Symbol, GlrMaskError> {
        Ok(match expr {
            GrammarExpr::Grouped(inner) => return self.lower_expr_terminalish(inner),
            GrammarExpr::Ref(name) => {
                if !self.named_rule_exprs.contains_key(name) {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "unknown rule referenced from nonterminal context: {name}"
                    )));
                }
                if self.internal_terminal_names.contains(name) {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "internal-only terminal {name} referenced from nonterminal context"
                    )));
                }
                Symbol::Nonterminal(self.nonterminal_id(name))
            }
            GrammarExpr::Literal(bytes) => Symbol::Terminal(self.literal_terminal_id(bytes)),
            GrammarExpr::SpecialToken(token_id) => Symbol::Terminal(
                self.special_terminal_id(&format!("@token({token_id})"), *token_id),
            ),
            GrammarExpr::CharClass { def, negate, utf8 } => {
                let pattern = char_class_pattern(def, *negate);
                Symbol::Terminal(self.terminal_id(&pattern, &pattern, *utf8))
            }
            GrammarExpr::RawRegex(pattern) => {
                // assume utf8 true for raw regex from lark/ebnf
                Symbol::Terminal(self.terminal_id(pattern, pattern, true))
            }
            GrammarExpr::LexerDfa(_) => {
                let expr = self.resolve_terminal_expr(None, expr)?;
                let name = format!("__terminal_expr_{}", self.generated_nonterminal_counter);
                Symbol::Terminal(self.register_terminal_expr(&name, expr))
            }
            GrammarExpr::Epsilon => {
                // Epsilon as an inline NT atom: create a nonterminal with an empty production.
                let nt = self.fresh_nonterminal();
                self.rules.push(Rule { lhs: nt, rhs: Vec::new() });
                Symbol::Nonterminal(nt)
            }
            GrammarExpr::Exclude { .. } | GrammarExpr::Intersect { .. } => {
                if let Some(lowered) = self.exact_nonterminal_subtraction_expr(expr)? {
                    return Ok(self.lower_expr(&lowered));
                }
                let expr = self.resolve_terminal_expr(None, expr)?;
                let name = format!("__terminal_expr_{}", self.generated_nonterminal_counter);
                Symbol::Terminal(self.register_terminal_expr(&name, expr))
            }
            GrammarExpr::AnyByte => {
                Symbol::Terminal(self.terminal_id(".", ".", false))
            }
            GrammarExpr::Sequence(_)
            | GrammarExpr::Choice(_)
            | GrammarExpr::Quantified(_, Quantifier::Optional)
            | GrammarExpr::Quantified(_, Quantifier::ZeroPlus)
            | GrammarExpr::Quantified(_, Quantifier::OnePlus)
            | GrammarExpr::Quantified(_, Quantifier::Range(_, _))
            | GrammarExpr::SeparatedSequence { .. } => self.lower_expr(expr),
            GrammarExpr::ExprNFA(_) => {
                return Err(GlrMaskError::GrammarParse(
                    "GrammarExpr::ExprNFA must be the complete expression of a named rule"
                        .into(),
                ));
            }
        })
    }

    fn lower_sepseq_repetition_item_nonempty_symbol(
        &mut self,
        inner: &GrammarExpr,
        separator: &GrammarExpr,
        min: usize,
        max: Option<usize>,
    ) -> Result<Option<Symbol>, GlrMaskError> {
        let Some(item_sym) = self.lower_nonnullable_expr_symbol(inner)? else {
            return Ok(None);
        };

        let sep_sym = self.lower_expr_terminalish(separator)?;
        let pair_nt = self.fresh_nonterminal();
        self.rules.push(Rule {
            lhs: pair_nt,
            rhs: vec![sep_sym, item_sym.clone()],
        });
        let pair_symbol = Symbol::Nonterminal(pair_nt);
        let shape = self.repeat_shape();

        if max.is_none() {
            let min = min.max(1);
            let prefix_sym = if min == 1 {
                item_sym.clone()
            } else {
                let prefix_nt = self.fresh_nonterminal();
                let prefix_tail_nt = self.repeat_exact_nonterminal(&pair_symbol, min - 1, shape);
                self.rules.push(Rule {
                    lhs: prefix_nt,
                    rhs: vec![item_sym.clone(), Symbol::Nonterminal(prefix_tail_nt)],
                });
                Symbol::Nonterminal(prefix_nt)
            };
            let tail_nt = self.fresh_nonterminal();
            self.rules.push(Rule { lhs: tail_nt, rhs: Vec::new() });
            self.rules.push(Rule {
                lhs: tail_nt,
                rhs: self.repeat_rhs(tail_nt, pair_symbol),
            });
            let result_nt = self.fresh_nonterminal();
            self.rules.push(Rule {
                lhs: result_nt,
                rhs: vec![prefix_sym, Symbol::Nonterminal(tail_nt)],
            });
            return Ok(Some(Symbol::Nonterminal(result_nt)));
        }

        let max = max.expect("finite bound expected when max.is_none() is false");
        if min > max {
            return Ok(None);
        }
        if max == 0 {
            return Ok(None);
        }

        let min = min.max(1);

        let prefix_sym = if min == 1 {
            item_sym.clone()
        } else {
            let prefix_nt = self.fresh_nonterminal();
            let prefix_tail_nt = self.repeat_exact_nonterminal(&pair_symbol, min - 1, shape);
            self.rules.push(Rule {
                lhs: prefix_nt,
                rhs: vec![item_sym.clone(), Symbol::Nonterminal(prefix_tail_nt)],
            });
            Symbol::Nonterminal(prefix_nt)
        };

        if min == max {
            return Ok(Some(prefix_sym));
        }

        let extra_nt = self.repeat_range_nonterminal(&pair_symbol, 0, max - min, shape);
        let result_nt = self.fresh_nonterminal();
        self.rules.push(Rule {
            lhs: result_nt,
            rhs: vec![prefix_sym, Symbol::Nonterminal(extra_nt)],
        });
        Ok(Some(Symbol::Nonterminal(result_nt)))
    }

    fn lower_sepseq_item_nonempty_symbol(
        &mut self,
        item_expr: &GrammarExpr,
        item_quantifier: Option<&Quantifier>,
        separator: &GrammarExpr,
    ) -> Result<Option<Symbol>, GlrMaskError> {
        match item_quantifier {
            Some(Quantifier::Optional) | None => self.lower_nonnullable_expr_symbol(item_expr),
            Some(Quantifier::ZeroPlus) => {
                self.lower_sepseq_repetition_item_nonempty_symbol(item_expr, separator, 1, None)
            }
            Some(Quantifier::OnePlus) => {
                self.lower_sepseq_repetition_item_nonempty_symbol(item_expr, separator, 1, None)
            }
            Some(Quantifier::Range(min, max)) => {
                let required = (*min).max(1);
                self.lower_sepseq_repetition_item_nonempty_symbol(item_expr, separator, required, *max)
            }
        }
    }

    /// Lower a `SeparatedSequence` into a grammar symbol.
    ///
    /// Returns `(symbol, can_be_empty)` where `can_be_empty` is `true` if the
    /// symbol can derive the empty string (i.e., all items are optional).
    ///
    /// The tree is split according to `shape`, mirroring the same algorithm used
    /// for JSON Schema ordered objects.
    fn lower_separated_sequence_inner(
        &mut self,
        items: &[(GrammarExpr, Option<Quantifier>)],
        separator: &GrammarExpr,
        shape: CommaSepShape,
    ) -> Result<(Symbol, bool), GlrMaskError> {
        debug_assert!(!items.is_empty());

        if items.len() == 1 {
            let (item_expr, item_quantifier) = &items[0];
            // Always route through lower_sepseq_item_nonempty_symbol so that the
            // separator is correctly threaded through repetition items.
            // e.g. RepeatOne(item) must become `item (sep item)*`, not bare `item+`.
            // For non-repetition items the function falls through to
            // lower_nonnullable_expr_symbol which handles them correctly.
            let item_sym = self.lower_sepseq_item_nonempty_symbol(item_expr, item_quantifier.as_ref(), separator)?;
            // Return can_be_empty=true for optional items as a signal to the parent to add
            // a "without this item and its preceding separator" alternative.  We do NOT emit
            // an epsilon rule here â€” that would create dangling separators in the parent rule
            // (e.g. "key": , ).  The caller of lower_separated_sequence_inner handles the
            // all-optional empty case via an explicit separate alternative (e.g. "{}").
            let can_be_empty = item_quantifier.as_ref().is_some_and(Quantifier::is_nullable)
                || self.expr_is_nullable(item_expr);
            return Ok((item_sym.unwrap_or_else(|| self.lower_expr(&GrammarExpr::Epsilon)), can_be_empty));
        }

        let mid = match shape {
            CommaSepShape::Balanced => items.len() / 2,
            CommaSepShape::Left => items.len() - 1,
            CommaSepShape::Right => 1,
            CommaSepShape::LeftBalanced => {
                let first_optional = items
                    .iter()
                    .position(|(_, quantifier)| quantifier.as_ref().is_some_and(Quantifier::is_nullable));
                match first_optional {
                    None => items.len() - 1,
                    Some(0) => items.len() / 2,
                    Some(idx) => idx,
                }
            }
        };

        let sep_sym = self.lower_expr_terminalish(separator)?;
        let (left_sym, left_can_be_empty) =
            self.lower_separated_sequence_inner(&items[..mid], separator, shape)?;
        let (right_sym, right_can_be_empty) =
            self.lower_separated_sequence_inner(&items[mid..], separator, shape)?;

        let nt = self.fresh_nonterminal();

        // STICKY NOTE: DO NOT REMOVE THIS WARNING UNDER ANY CIRCUMSTANCES.
        // In generic SeparatedSequence lowering, "item derives empty" is NOT the
        // same thing as "item is absent": required nullable items can still be
        // structurally present and participate in separator placement/arity.
        // A naive right-linear lowering that treats nullable items as skippable
        // absence changes the accepted language by collapsing those cases.
        // Always: left sep right
        self.rules.push(Rule {
            lhs: nt,
            rhs: vec![left_sym.clone(), sep_sym, right_sym.clone()],
        });
        // If right side can be empty: left alone is valid
        if right_can_be_empty {
            self.rules.push(Rule { lhs: nt, rhs: vec![left_sym.clone()] });
        }
        // If left side can be empty: right alone is valid
        if left_can_be_empty {
            self.rules.push(Rule { lhs: nt, rhs: vec![right_sym.clone()] });
        }

        // Both sides can be empty: propagate the flag upward so the grandparent can add a
        // "without this subtree and its separator" alternative.  Do NOT emit nt -> Îµ here;
        // that would produce dangling separators in the enclosing rule.
        let can_be_empty = left_can_be_empty && right_can_be_empty;

        Ok((Symbol::Nonterminal(nt), can_be_empty))
    }

    /// Register a pre-resolved terminal Expr, deduplicating by value.
    fn register_terminal_expr(&mut self, name: &str, expr: Expr) -> TerminalID {
        let mut hasher = FxHasher::default();
        expr.hash(&mut hasher);
        let hash = hasher.finish();
        if let Some(candidates) = self.terminal_expr_hash_index.get(&hash) {
            for &id in candidates {
                if let Some(Terminal::Expr { expr: existing, .. }) = self.terminals.get(id as usize)
                    && *existing == expr
                {
                    self.terminal_ids_by_name.insert(name.to_string(), id);
                    return id;
                }
            }
        }

        let id = self.terminals.len() as TerminalID;
        self.terminal_names.insert(id, name.to_string());
        self.terminal_ids_by_name.insert(name.to_string(), id);
        self.terminals.push(Terminal::Expr { id, expr });
        self.terminal_expr_hash_index.entry(hash).or_default().push(id);
        id
    }
}

/// Resolve terminal-labelled subexpressions against the named terminal table.
///
/// The cache is shared across all inputs, so repeated helper references remain
/// `Arc`-shared while the caller compiles a larger explicit graph.
pub fn resolve_terminal_subexpressions(
    grammar: &NamedGrammar,
    exprs: &[GrammarExpr],
) -> Result<Vec<Expr>, GlrMaskError> {
    let terminal_bodies = grammar
        .rules
        .iter()
        .filter(|rule| rule.is_terminal)
        .map(|rule| (rule.name.clone(), &rule.expr))
        .collect::<FxHashMap<_, _>>();
    let mut terminal_expr_cache: FxHashMap<String, Arc<Expr>> = FxHashMap::default();

    exprs
        .iter()
        .map(|expr| {
            grammar_expr_to_expr(
                expr,
                &terminal_bodies,
                &mut terminal_expr_cache,
                &mut HashSet::new(),
            )
        })
        .collect()
}

/// Resolve every externally emitting named terminal to the exact lexer-level
/// expression used by [`lower`]. Equal values here are precisely the terminal
/// expressions that `register_terminal_expr` deduplicates to one `TerminalID`.
pub fn resolved_named_terminal_exprs(
    grammar: &NamedGrammar,
) -> Result<BTreeMap<String, Expr>, GlrMaskError> {
    let terminal_bodies = grammar
        .rules
        .iter()
        .filter(|rule| rule.is_terminal)
        .map(|rule| (rule.name.clone(), &rule.expr))
        .collect::<FxHashMap<_, _>>();
    let mut terminal_expr_cache: FxHashMap<String, Arc<Expr>> = FxHashMap::default();
    let mut resolved = BTreeMap::new();

    for rule in grammar
        .rules
        .iter()
        .filter(|rule| rule.is_terminal && !rule.is_internal)
    {
        if matches!(rule.expr, GrammarExpr::SpecialToken(_)) {
            continue;
        }
        let expr = if let Some(cached) = terminal_expr_cache.get(&rule.name) {
            cached.as_ref().clone()
        } else {
            let mut visiting = HashSet::from([rule.name.clone()]);
            let expr = grammar_expr_to_expr(
                &rule.expr,
                &terminal_bodies,
                &mut terminal_expr_cache,
                &mut visiting,
            )?;
            terminal_expr_cache.insert(rule.name.clone(), Arc::new(expr.clone()));
            expr
        };
        resolved.insert(rule.name.clone(), expr);
    }

    Ok(resolved)
}

/// Convert a GrammarExpr to an Expr tree, resolving terminal Ref nodes
/// via the `terminal_bodies` map and caching results in `terminal_expr_cache`.
fn grammar_expr_to_expr(
    expr: &GrammarExpr,
    terminal_bodies: &FxHashMap<String, &GrammarExpr>,
    terminal_expr_cache: &mut FxHashMap<String, Arc<Expr>>,
    visiting: &mut HashSet<String>,
) -> Result<Expr, GlrMaskError> {
    Ok(match expr {
        GrammarExpr::Grouped(inner) => {
            return grammar_expr_to_expr(inner, terminal_bodies, terminal_expr_cache, visiting);
        }
        GrammarExpr::Literal(bytes) => Expr::U8Seq(bytes.clone()),
        GrammarExpr::SpecialToken(token_id) => {
            return Err(GlrMaskError::GrammarParse(format!(
                "@token({token_id}) cannot be composed inside a byte terminal expression"
            )));
        }
        GrammarExpr::CharClass { def, negate, utf8 } => {
            let pattern = char_class_pattern(def, *negate);
            parse_regex(&pattern, *utf8)
        }
        GrammarExpr::RawRegex(pattern) => parse_regex(pattern, true),
        GrammarExpr::LexerDfa(dfa) => Expr::Dfa(dfa.clone()),
        GrammarExpr::AnyByte => Expr::U8Class(U8Set::from_range(0, 255)),
        GrammarExpr::Epsilon => Expr::Epsilon,
        GrammarExpr::Sequence(parts) => {
            let exprs: Vec<Expr> = parts.iter().map(|p| grammar_expr_to_expr(p, terminal_bodies, terminal_expr_cache, visiting)).collect::<Result<_, _>>()?;
            if exprs.len() == 1 {
                exprs.into_iter().next().unwrap()
            } else {
                Expr::Seq(exprs)
            }
        }
        GrammarExpr::Choice(options) => {
            let exprs: Vec<Expr> = options.iter().map(|o| grammar_expr_to_expr(o, terminal_bodies, terminal_expr_cache, visiting)).collect::<Result<_, _>>()?;
            if exprs.len() == 1 {
                exprs.into_iter().next().unwrap()
            } else {
                Expr::Choice(exprs)
            }
        }
        GrammarExpr::Exclude { expr, exclude } => Expr::Exclude {
            expr: Box::new(grammar_expr_to_expr(expr, terminal_bodies, terminal_expr_cache, visiting)?),
            exclude: Box::new(grammar_expr_to_expr(exclude, terminal_bodies, terminal_expr_cache, visiting)?),
        },
        GrammarExpr::Intersect { expr, intersect } => Expr::Intersect {
            expr: Box::new(grammar_expr_to_expr(expr, terminal_bodies, terminal_expr_cache, visiting)?),
            intersect: Box::new(grammar_expr_to_expr(intersect, terminal_bodies, terminal_expr_cache, visiting)?),
        },
        GrammarExpr::Quantified(inner, Quantifier::Optional) => Expr::Repeat {
            expr: Box::new(grammar_expr_to_expr(inner, terminal_bodies, terminal_expr_cache, visiting)?),
            min: 0,
            max: Some(1),
        },
        GrammarExpr::Quantified(inner, Quantifier::ZeroPlus) => Expr::Repeat {
            expr: Box::new(grammar_expr_to_expr(inner, terminal_bodies, terminal_expr_cache, visiting)?),
            min: 0,
            max: None,
        },
        GrammarExpr::Quantified(inner, Quantifier::OnePlus) => Expr::Repeat {
            expr: Box::new(grammar_expr_to_expr(inner, terminal_bodies, terminal_expr_cache, visiting)?),
            min: 1,
            max: None,
        },
        GrammarExpr::Quantified(expr, Quantifier::Range(min, max)) => Expr::Repeat {
            expr: Box::new(grammar_expr_to_expr(expr, terminal_bodies, terminal_expr_cache, visiting)?),
            min: *min,
            max: *max,
        },
        GrammarExpr::Ref(name) => {
            // Look up in cache first
            if let Some(cached) = terminal_expr_cache.get(name) {
                return Ok(Expr::Shared(cached.clone()));
            }
            // Must be a terminal rule â€” look up its body and resolve it
            if let Some(&body) = terminal_bodies.get(name) {
                if !visiting.insert(name.clone()) {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "cycle detected in terminal rule references: {name}"
                    )));
                }
                let expr = grammar_expr_to_expr(body, terminal_bodies, terminal_expr_cache, visiting)?;
                let arc = Arc::new(expr);
                terminal_expr_cache.insert(name.clone(), arc.clone());
                visiting.remove(name);
                Expr::Shared(arc)
            } else {
                return Err(GlrMaskError::GrammarParse(format!(
                    "unresolved Ref({name}) in terminal body â€” not found in terminal rules"
                )));
            }
        }
        GrammarExpr::SeparatedSequence { .. } => {
            return Err(GlrMaskError::GrammarParse(
                "GrammarExpr::SeparatedSequence cannot appear inside a terminal rule".into(),
            ));
        }
        GrammarExpr::ExprNFA(expr_nfa) => {
            let symbols = expr_nfa
                .symbols
                .iter()
                .map(|symbol| {
                    grammar_expr_to_expr(
                        symbol,
                        terminal_bodies,
                        terminal_expr_cache,
                        visiting,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let dfa = compile_expression_labeled_nfa(&expr_nfa.nfa, &symbols)
                .map_err(GlrMaskError::GrammarParse)?;
            Expr::Dfa(Arc::new(dfa))
        }
    })
}

fn collect_terminal_rule_refs<'a>(expr: &'a GrammarExpr, refs: &mut FxHashSet<&'a str>) {
    match expr {
        GrammarExpr::Ref(name) => {
            refs.insert(name.as_str());
        }
        GrammarExpr::Grouped(inner) | GrammarExpr::Quantified(inner, _) => {
            collect_terminal_rule_refs(inner, refs)
        }
        GrammarExpr::Sequence(parts) | GrammarExpr::Choice(parts) => {
            for part in parts {
                collect_terminal_rule_refs(part, refs);
            }
        }
        GrammarExpr::Exclude { expr, exclude } => {
            collect_terminal_rule_refs(expr, refs);
            collect_terminal_rule_refs(exclude, refs);
        }
        GrammarExpr::Intersect { expr, intersect } => {
            collect_terminal_rule_refs(expr, refs);
            collect_terminal_rule_refs(intersect, refs);
        }
        GrammarExpr::SeparatedSequence {
            items, separator, ..
        } => {
            for (item, _) in items {
                collect_terminal_rule_refs(item, refs);
            }
            collect_terminal_rule_refs(separator, refs);
        }
        GrammarExpr::ExprNFA(expr_nfa) => {
            for symbol in &expr_nfa.symbols {
                collect_terminal_rule_refs(symbol, refs);
            }
        }
        GrammarExpr::Epsilon
        | GrammarExpr::Literal(_)
        | GrammarExpr::SpecialToken(_)
        | GrammarExpr::CharClass { .. }
        | GrammarExpr::RawRegex(_)
        | GrammarExpr::LexerDfa(_)
        | GrammarExpr::AnyByte => {}
    }
}

fn grammar_expr_is_nullable(
    expr: &GrammarExpr,
    rule_nullable: &FxHashMap<String, bool>,
) -> bool {
    match expr {
        GrammarExpr::Ref(name) => rule_nullable.get(name).copied().unwrap_or(false),
        GrammarExpr::Grouped(inner) => grammar_expr_is_nullable(inner, rule_nullable),
        GrammarExpr::Sequence(parts) => parts.iter().all(|part| grammar_expr_is_nullable(part, rule_nullable)),
        GrammarExpr::Choice(options) => options.iter().any(|option| grammar_expr_is_nullable(option, rule_nullable)),
        GrammarExpr::Epsilon => true,
        GrammarExpr::Exclude { expr, exclude } => {
            grammar_expr_is_nullable(expr, rule_nullable)
                && !grammar_expr_is_nullable(exclude, rule_nullable)
        }
        GrammarExpr::Intersect { expr, intersect } => {
            grammar_expr_is_nullable(expr, rule_nullable)
                && grammar_expr_is_nullable(intersect, rule_nullable)
        }
        GrammarExpr::Quantified(_, Quantifier::Optional) | GrammarExpr::Quantified(_, Quantifier::ZeroPlus) => true,
        GrammarExpr::Quantified(inner, Quantifier::OnePlus) => grammar_expr_is_nullable(inner, rule_nullable),
        GrammarExpr::Quantified(expr, Quantifier::Range(min, _)) => {
            *min == 0 || grammar_expr_is_nullable(expr, rule_nullable)
        }
        GrammarExpr::Literal(bytes) => bytes.is_empty(),
        GrammarExpr::SpecialToken(_) => false,
        GrammarExpr::CharClass { def, negate, utf8 } => {
            parse_regex(&char_class_pattern(def, *negate), *utf8).is_nullable()
        }
        GrammarExpr::RawRegex(pattern) => parse_regex(pattern, true).is_nullable(),
        GrammarExpr::LexerDfa(dfa) => Expr::Dfa(dfa.clone()).is_nullable(),
        GrammarExpr::AnyByte => false,
        GrammarExpr::SeparatedSequence { items, allow_empty, .. } => {
            *allow_empty
                && items.iter().all(|(item, quantifier)| {
                    quantifier.as_ref().is_some_and(Quantifier::is_nullable)
                        || grammar_expr_is_nullable(item, rule_nullable)
                })
        }
        GrammarExpr::ExprNFA(expr_nfa) => {
            // An ExprNFA accepts epsilon iff an accepting NFA state is
            // reachable from any start state using only epsilon transitions and
            // labels that are nullable in the current rule fixed point. This is
            // exact for the original NFA; determinizing and minimizing first
            // only adds avoidable work to every fixed-point sweep.
            let mut seen = vec![false; expr_nfa.nfa.states.len()];
            let mut pending = VecDeque::new();
            let mut nullable_labels = vec![None; expr_nfa.symbols.len()];
            for &start in &expr_nfa.nfa.start_states {
                let start = start as usize;
                if start < seen.len() && !seen[start] {
                    seen[start] = true;
                    pending.push_back(start);
                }
            }
            while let Some(state_id) = pending.pop_front() {
                let state = &expr_nfa.nfa.states[state_id];
                if state.is_accepting {
                    return true;
                }
                for &target in &state.epsilons {
                    let target = target as usize;
                    if target < seen.len() && !seen[target] {
                        seen[target] = true;
                        pending.push_back(target);
                    }
                }
                for (&label, targets) in &state.transitions {
                    let Ok(label_index) = usize::try_from(label) else {
                        continue;
                    };
                    let Some(symbol) = expr_nfa.symbols.get(label_index) else {
                        continue;
                    };
                    let label_is_nullable = match nullable_labels[label_index] {
                        Some(nullable) => nullable,
                        None => {
                            let nullable = grammar_expr_is_nullable(symbol, rule_nullable);
                            nullable_labels[label_index] = Some(nullable);
                            nullable
                        }
                    };
                    if !label_is_nullable {
                        continue;
                    }
                    for &target in targets {
                        let target = target as usize;
                        if target < seen.len() && !seen[target] {
                            seen[target] = true;
                            pending.push_back(target);
                        }
                    }
                }
            }
            false
        }
    }
}

fn compute_rule_nullability(grammar: &NamedGrammar) -> FxHashMap<String, bool> {
    let mut nullable = grammar
        .rules
        .iter()
        .map(|rule| (rule.name.clone(), false))
        .collect::<FxHashMap<_, _>>();

    loop {
        let mut changed = false;
        for rule in &grammar.rules {
            let is_nullable = grammar_expr_is_nullable(&rule.expr, &nullable);
            if is_nullable && !nullable.get(&rule.name).copied().unwrap_or(false) {
                nullable.insert(rule.name.clone(), true);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    nullable
}

fn validate_expr_nfa_placement(grammar: &NamedGrammar) -> Result<(), GlrMaskError> {
    fn walk(expr: &GrammarExpr, top_level: bool, rule_name: &str) -> Result<(), GlrMaskError> {
        match expr {
            GrammarExpr::ExprNFA(_) => {
                if !top_level {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "GrammarExpr::ExprNFA must be the complete expression of a named rule; found nested in {rule_name}"
                    )));
                }
            }
            GrammarExpr::Sequence(parts) | GrammarExpr::Choice(parts) => {
                for part in parts {
                    walk(part, false, rule_name)?;
                }
            }
            GrammarExpr::Grouped(inner) => {
                walk(inner, false, rule_name)?;
            }
            GrammarExpr::Exclude { expr, exclude } => {
                walk(expr, false, rule_name)?;
                walk(exclude, false, rule_name)?;
            }
            GrammarExpr::Intersect { expr, intersect } => {
                walk(expr, false, rule_name)?;
                walk(intersect, false, rule_name)?;
            }
            GrammarExpr::Quantified(inner, Quantifier::Optional)
            | GrammarExpr::Quantified(inner, Quantifier::ZeroPlus)
            | GrammarExpr::Quantified(inner, Quantifier::OnePlus)
            | GrammarExpr::Quantified(inner, Quantifier::Range(_, _)) => {
                walk(inner, false, rule_name)?;
            }
            GrammarExpr::SeparatedSequence { items, separator, .. } => {
                for (item, _) in items {
                    walk(item, false, rule_name)?;
                }
                walk(separator, false, rule_name)?;
            }
            GrammarExpr::Ref(_)
            | GrammarExpr::Epsilon
            | GrammarExpr::Literal(_)
            | GrammarExpr::SpecialToken(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte => {}
        }
        Ok(())
    }

    for rule in &grammar.rules {
        walk(&rule.expr, true, &rule.name)?;
    }
    Ok(())
}

/// Convert a lexer-level [`Expr`] into an equivalent [`GrammarExpr`].
///
/// Every `Expr` variant has a `GrammarExpr` counterpart, so this is lossless.
/// `Expr::U8Class(U8Set)` is converted to `GrammarExpr::CharClass` using a
/// range-encoded string representation.
pub fn expr_to_grammar_expr(expr: &Expr) -> GrammarExpr {
    match expr {
        Expr::Shared(inner) => expr_to_grammar_expr(inner),
        Expr::U8Seq(bytes) => GrammarExpr::Literal(bytes.clone()),
        Expr::U8Class(set) => GrammarExpr::CharClass {
            def: u8set_to_class_def(set),
            negate: false,
            utf8: false,
        },
        Expr::Dfa(dfa) => GrammarExpr::LexerDfa(dfa.clone()),
        Expr::Epsilon => GrammarExpr::Epsilon,
        Expr::Seq(parts) => {
            let items: Vec<_> = parts.iter().map(expr_to_grammar_expr).collect();
            match items.len() {
                0 => GrammarExpr::Epsilon,
                1 => items.into_iter().next().unwrap(),
                _ => GrammarExpr::Sequence(items),
            }
        }
        Expr::Choice(alts) => {
            let items: Vec<_> = alts.iter().map(expr_to_grammar_expr).collect();
            match items.len() {
                0 => GrammarExpr::Epsilon,
                1 => items.into_iter().next().unwrap(),
                _ => GrammarExpr::Choice(items),
            }
        }
        Expr::Exclude { expr, exclude } => GrammarExpr::Exclude {
            expr: Box::new(expr_to_grammar_expr(expr)),
            exclude: Box::new(expr_to_grammar_expr(exclude)),
        },
        Expr::Intersect { expr, intersect } => GrammarExpr::Intersect {
            expr: Box::new(expr_to_grammar_expr(expr)),
            intersect: Box::new(expr_to_grammar_expr(intersect)),
        },
        Expr::Repeat { expr: inner, min, max } => {
            let g = expr_to_grammar_expr(inner);
            match (*min, *max) {
                (0, None) => GrammarExpr::Quantified(Box::new(g), Quantifier::ZeroPlus),
                (1, None) => GrammarExpr::Quantified(Box::new(g), Quantifier::OnePlus),
                (0, Some(1)) => GrammarExpr::Quantified(Box::new(g), Quantifier::Optional),
                (n, Some(m)) => GrammarExpr::Quantified(Box::new(g), Quantifier::Range(n, Some(m))),
                (n, None) => {
                    // n+ : express as exactly-n followed by zero-or-more
                    GrammarExpr::Sequence(vec![
                        GrammarExpr::Quantified(Box::new(g.clone()), Quantifier::Range(n, Some(n))),
                        GrammarExpr::Quantified(Box::new(g), Quantifier::ZeroPlus),
                    ])
                }
            }
        }
    }
}

/// Encode a [`U8Set`] as a character-class definition string (without the surrounding `[...]`).
///
/// Uses range notation where possible. Always produces a non-negated form.
pub fn u8set_to_class_def(set: &U8Set) -> String {
    let mut out = String::new();
    let bytes: Vec<u8> = set.iter().collect();
    let mut i = 0usize;
    while i < bytes.len() {
        let start = bytes[i];
        let mut end = start;
        i += 1;
        while i < bytes.len() && bytes[i] == end.wrapping_add(1) && end < 255 {
            end = bytes[i];
            i += 1;
        }
        push_class_char(&mut out, start);
        if end != start {
            if end == start + 1 {
                push_class_char(&mut out, end);
            } else {
                out.push('-');
                push_class_char(&mut out, end);
            }
        }
    }
    out
}

fn push_class_char(out: &mut String, b: u8) {
    use std::fmt::Write;
    match b {
        b'\\' => out.push_str("\\\\"),
        b']' => out.push_str("\\]"),
        b'-' => out.push_str("\\-"),
        b'^' => out.push_str("\\^"),
        0x20..=0x7E => out.push(b as char),
        _ => write!(out, "\\x{:02X}", b).unwrap(),
    }
}

fn dedup_rules_preserving_first_occurrence(rules: &mut Vec<Rule>) {
    let mut unique = Vec::with_capacity(rules.len());
    let mut hash_to_unique_indices: FxHashMap<u64, Vec<usize>> = FxHashMap::default();
    for rule in std::mem::take(rules) {
        let mut hasher = FxHasher::default();
        rule.hash(&mut hasher);
        let hash = hasher.finish();
        let duplicate = hash_to_unique_indices
            .get(&hash)
            .is_some_and(|candidates| candidates.iter().any(|&index| unique[index] == rule));
        if !duplicate {
            hash_to_unique_indices.entry(hash).or_default().push(unique.len());
            unique.push(rule);
        }
    }
    *rules = unique;
}

pub fn lower(grammar: &NamedGrammar) -> Result<GrammarDef, GlrMaskError> {
    lower_with_resolved_terminal_exprs_impl(grammar, None, None)
}

/// Lower a named grammar while reusing terminal expressions that were already
/// resolved by an immediately preceding preparation pass.
///
/// Entries are keyed by named terminal rule. Missing entries are resolved by
/// the ordinary lowering path, so callers may provide only the expressions
/// they already have without changing language semantics.
#[doc(hidden)]
pub fn lower_with_resolved_terminal_exprs(
    grammar: &NamedGrammar,
    resolved_terminal_exprs: BTreeMap<String, Expr>,
) -> Result<GrammarDef, GlrMaskError> {
    lower_with_resolved_terminal_exprs_impl(grammar, Some(resolved_terminal_exprs), None)
}

fn grammar_expr_node_count(expr: &GrammarExpr) -> usize {
    1 + match expr {
        GrammarExpr::Grouped(inner) | GrammarExpr::Quantified(inner, _) => {
            grammar_expr_node_count(inner)
        }
        GrammarExpr::Sequence(parts) | GrammarExpr::Choice(parts) => {
            parts.iter().map(grammar_expr_node_count).sum()
        }
        GrammarExpr::Exclude { expr, exclude } => {
            grammar_expr_node_count(expr) + grammar_expr_node_count(exclude)
        }
        GrammarExpr::Intersect { expr, intersect } => {
            grammar_expr_node_count(expr) + grammar_expr_node_count(intersect)
        }
        GrammarExpr::SeparatedSequence { items, separator, .. } => {
            items
                .iter()
                .map(|(item, _)| grammar_expr_node_count(item))
                .sum::<usize>()
                + grammar_expr_node_count(separator)
        }
        // ExprNFA transition labels are already compact shared structures and
        // are lowered by their dedicated emitter, not the generic subtree cache.
        GrammarExpr::ExprNFA(_) => 0,
        _ => 0,
    }
}

fn lower_with_resolved_terminal_exprs_impl(
    grammar: &NamedGrammar,
    resolved_terminal_exprs: Option<BTreeMap<String, Expr>>,
    parser_cfg_shape: Option<(RepeatTreeShape, CommaSepShape, bool)>,
) -> Result<GrammarDef, GlrMaskError> {
    let profile_enabled = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
    let validate_started_at = profile_enabled.then(std::time::Instant::now);
    validate_expr_nfa_placement(grammar)?;
    let validate_ms = validate_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0);

    let setup_started_at = profile_enabled.then(std::time::Instant::now);
    let mut lowerer = Lowerer::new();
    lowerer.parser_cfg_shape = parser_cfg_shape;
    const COMPOUND_EXPR_CACHE_NODE_THRESHOLD: usize = 2048;
    lowerer.cache_compound_exprs = grammar
        .rules
        .iter()
        .map(|rule| grammar_expr_node_count(&rule.expr))
        .sum::<usize>()
        >= COMPOUND_EXPR_CACHE_NODE_THRESHOLD;
    if let Some(resolved_terminal_exprs) = resolved_terminal_exprs {
        lowerer.terminal_expr_cache.extend(
            resolved_terminal_exprs
                .into_iter()
                .map(|(name, expr)| (name, Arc::new(expr))),
        );
    }
    let setup_detail_profile = std::env::var_os("GLRMASK_PROFILE_AST_LOWER_DETAIL").is_some();
    let named_exprs_started_at = setup_detail_profile.then(std::time::Instant::now);
    lowerer.named_rule_exprs = grammar
        .rules
        .iter()
        .map(|rule| (rule.name.clone(), &rule.expr))
        .collect();
    let named_exprs_ms = named_exprs_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let named_terminal_started_at = setup_detail_profile.then(std::time::Instant::now);
    lowerer.named_rule_is_terminal = grammar
        .rules
        .iter()
        .map(|rule| (rule.name.clone(), rule.is_terminal))
        .collect();
    let named_terminal_ms = named_terminal_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let nullable_started_at = setup_detail_profile.then(std::time::Instant::now);
    lowerer.rule_nullable = compute_rule_nullability(grammar);
    let nullable_ms = nullable_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    // Collect internal terminal names for validation.
    let internal_names_started_at = setup_detail_profile.then(std::time::Instant::now);
    lowerer.internal_terminal_names = grammar
        .rules
        .iter()
        .filter(|r| r.is_terminal && r.is_internal)
        .map(|r| r.name.clone())
        .collect();
    let internal_names_ms = internal_names_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let named_ids_started_at = setup_detail_profile.then(std::time::Instant::now);
    for rule in &grammar.rules {
        if rule.is_terminal && rule.is_internal {
            continue; // don't allocate nonterminal IDs for internal terminals
        }
        lowerer.nonterminal_id(&rule.name);
    }
    let named_ids_ms = named_ids_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    // Build a map of terminal rule bodies for resolving Ref nodes inside terminal exprs.
    let terminal_bodies_started_at = setup_detail_profile.then(std::time::Instant::now);
    lowerer.terminal_bodies = grammar
        .rules
        .iter()
        .filter(|r| r.is_terminal)
        .map(|r| (r.name.clone(), &r.expr))
        .collect();
    let terminal_bodies_ms = terminal_bodies_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let mut terminal_rule_refs = FxHashSet::default();
    for rule in grammar.rules.iter().filter(|rule| rule.is_terminal) {
        collect_terminal_rule_refs(&rule.expr, &mut terminal_rule_refs);
    }
    let setup_ms = setup_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0);

    let rule_lowering_started_at = profile_enabled.then(std::time::Instant::now);
    let detail_profile = std::env::var_os("GLRMASK_PROFILE_AST_LOWER_DETAIL").is_some();
    let mut terminal_rule_ms = 0.0f64;
    let mut expr_nfa_rule_ms = 0.0f64;
    let mut other_rule_ms = 0.0f64;
    let mut expr_nfa_rules = 0usize;
    let mut expr_nfa_states = 0usize;
    let mut expr_nfa_transitions = 0usize;
    // Determinize+minimize the expr-NFA rule bodies that miss the cached
    // canonical-DFA fast path. `determinize_and_minimize` is a pure method, so
    // precomputing across cores keeps the serial emit loop below (which assigns
    // nonterminal/terminal IDs in rule order) byte-for-byte identical.
    //
    // Most small JSON schemas have only one or two such rules. Spawning a Rayon
    // traversal over every named rule for that shape costs more than the useful
    // work, so first identify the actual candidates and keep tiny candidate sets
    // serial. Large grammars retain parallel determinize/minimize work.
    let expr_dfa_candidates = grammar
        .rules
        .iter()
        .enumerate()
        .filter_map(|(idx, rule)| {
            if rule.is_terminal {
                return None;
            }
            let GrammarExpr::ExprNFA(expr_nfa) = &rule.expr else {
                return None;
            };
            let hits_fast_path = expr_nfa.prefer_direct_nfa_emission
                || (expr_nfa.is_determinized_and_minimized
                    && expr_nfa
                        .canonical_dfa
                        .as_ref()
                        .is_some_and(|dfa| dfa.compute_is_acyclic()));
            (!hits_fast_path).then_some((idx, expr_nfa))
        })
        .collect::<Vec<_>>();
    let parallel_min_candidates = std::env::var("GLRMASK_AST_LOWER_EXPR_DFA_PARALLEL_MIN_CANDIDATES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(4);
    let parallel_min_work = std::env::var("GLRMASK_AST_LOWER_EXPR_DFA_PARALLEL_MIN_WORK")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(512);
    let expr_dfa_work = expr_dfa_candidates
        .iter()
        .map(|(_, expr_nfa)| {
            expr_nfa.nfa.states.len()
                + expr_nfa
                    .nfa
                    .states
                    .iter()
                    .map(|state| state.transitions.len())
                    .sum::<usize>()
        })
        .sum::<usize>();
    let use_parallel_expr_dfa = rayon::current_num_threads() > 1
        && (expr_dfa_candidates.len() >= parallel_min_candidates
            || expr_dfa_work >= parallel_min_work);
    let precomputed_expr_dfas: FxHashMap<usize, DFA> = if use_parallel_expr_dfa {
        expr_dfa_candidates
            .par_iter()
            .map(|(idx, expr_nfa)| (*idx, expr_nfa.determinize_and_minimize()))
            .collect()
    } else {
        expr_dfa_candidates
            .iter()
            .map(|(idx, expr_nfa)| (*idx, expr_nfa.determinize_and_minimize()))
            .collect()
    };
    for (rule_index, rule) in grammar.rules.iter().enumerate() {
        let rule_started_at = detail_profile.then(std::time::Instant::now);
        // Terminal rules: convert the entire body to a single Terminal::Expr.
        // Refs to other terminal rules are resolved via Expr::Shared.
        if rule.is_terminal {
            if let GrammarExpr::SpecialToken(token_id) = &rule.expr {
                if rule.is_internal {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "internal terminal {} cannot be a special LLM token",
                        rule.name
                    )));
                }
                let lhs = lowerer.nonterminal_id(&rule.name);
                let tid = lowerer.special_terminal_id(&rule.name, *token_id);
                lowerer.rules.push(Rule {
                    lhs,
                    rhs: vec![Symbol::Terminal(tid)],
                });
                if let Some(rule_started_at) = rule_started_at {
                    terminal_rule_ms += rule_started_at.elapsed().as_secs_f64() * 1000.0;
                }
                continue;
            }
            let cached_expr = lowerer.terminal_expr_cache.get(&rule.name).cloned();
            let expr = if let Some(cached_expr) = cached_expr {
                (*cached_expr).clone()
            } else {
                lowerer.resolve_terminal_expr(Some(&rule.name), &rule.expr)?
            };
            if terminal_rule_refs.contains(rule.name.as_str())
                && !lowerer.terminal_expr_cache.contains_key(&rule.name)
            {
                lowerer
                    .terminal_expr_cache
                    .insert(rule.name.clone(), Arc::new(expr.clone()));
            }

            if rule.is_internal {
                // Internal-only: cached for Shared resolution, no terminal or production.
                continue;
            }

            let lhs = lowerer.nonterminal_id(&rule.name);
            let tid = lowerer.register_terminal_expr(&rule.name, expr);
            lowerer.rules.push(Rule { lhs, rhs: vec![Symbol::Terminal(tid)] });
            if let Some(rule_started_at) = rule_started_at {
                terminal_rule_ms += rule_started_at.elapsed().as_secs_f64() * 1000.0;
            }
            continue;
        }

        let lhs = lowerer.nonterminal_id(&rule.name);

        match &rule.expr {
            GrammarExpr::Grouped(inner) => {
                let symbol = lowerer.lower_expr_terminalish(inner)?;
                lowerer.rules.push(Rule { lhs, rhs: vec![symbol] });
            }
            GrammarExpr::Sequence(parts) => {
                let rhs = parts.iter().map(|part| lowerer.lower_expr_terminalish(part)).collect::<Result<Vec<_>, _>>()?;
                lowerer.rules.push(Rule { lhs, rhs });
            }
            GrammarExpr::Choice(options) => {
                for option in options {
                    match option {
                        GrammarExpr::Sequence(parts) => {
                            let rhs = parts.iter().map(|part| lowerer.lower_expr_terminalish(part)).collect::<Result<Vec<_>, _>>()?;
                            lowerer.rules.push(Rule { lhs, rhs });
                        }
                        _ => {
                            let symbol = lowerer.lower_expr_terminalish(option)?;
                            lowerer.rules.push(Rule { lhs, rhs: vec![symbol] });
                        }
                    }
                }
            }
            GrammarExpr::Quantified(inner, Quantifier::Optional) => {
                lowerer.rules.push(Rule { lhs, rhs: Vec::new() });
                let symbol = lowerer.lower_expr_terminalish(inner)?;
                lowerer.rules.push(Rule { lhs, rhs: vec![symbol] });
            }
            GrammarExpr::Quantified(inner, Quantifier::ZeroPlus) => {
                let symbol = lowerer.lower_expr_terminalish(inner)?;
                lowerer.rules.push(Rule { lhs, rhs: Vec::new() });
                lowerer.rules.push(Rule { lhs, rhs: lowerer.repeat_rhs(lhs, symbol) });
            }
            GrammarExpr::Quantified(inner, Quantifier::OnePlus) => {
                let symbol = lowerer.lower_expr_terminalish(inner)?;
                lowerer.rules.push(Rule { lhs, rhs: vec![symbol.clone()] });
                lowerer.rules.push(Rule { lhs, rhs: lowerer.repeat_rhs(lhs, symbol) });
            }
            GrammarExpr::Quantified(expr, Quantifier::Range(min, max)) => {
                lowerer.emit_repeat_range(lhs, expr, *min, *max)?;
            }
            GrammarExpr::ExprNFA(expr_nfa) => {
                expr_nfa_rules += 1;
                expr_nfa_states += expr_nfa.nfa.states.len();
                expr_nfa_transitions += expr_nfa
                    .nfa
                    .states
                    .iter()
                    .map(|state| state.transitions.values().map(Vec::len).sum::<usize>())
                    .sum::<usize>();
                if let Some(dfa) = precomputed_expr_dfas.get(&rule_index) {
                    lowerer.emit_expr_dfa_leftlinear(lhs, expr_nfa, dfa)?;
                } else {
                    lowerer.emit_expr_nfa(lhs, expr_nfa)?;
                }
            }
            _ => {
                let symbol = lowerer.lower_expr_terminalish(&rule.expr)?;
                lowerer.rules.push(Rule { lhs, rhs: vec![symbol] });
            }
        }
        if let Some(rule_started_at) = rule_started_at {
            let elapsed_ms = rule_started_at.elapsed().as_secs_f64() * 1000.0;
            if matches!(rule.expr, GrammarExpr::ExprNFA(_)) {
                expr_nfa_rule_ms += elapsed_ms;
            } else {
                other_rule_ms += elapsed_ms;
            }
        }
    }
    let rule_lowering_ms = rule_lowering_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0);

    let finish_started_at = profile_enabled.then(std::time::Instant::now);
    let start = lowerer
        .nonterminal_ids
        .get(&grammar.start)
        .copied()
        .ok_or_else(|| {
            GlrMaskError::GrammarParse(format!("undefined start rule: {}", grammar.start))
        })?;
    let nonterminal_names = lowerer
        .nonterminal_ids
        .iter()
        .filter(|(name, _)| !name.starts_with("__"))
        .map(|(name, id)| (*id, name.clone()))
        .collect();

    let ignore_terminal = grammar.ignore.as_ref().and_then(|ignore_name| {
        lowerer.terminal_ids_by_name.get(ignore_name).copied()
    });
    if ignore_terminal.is_some_and(|ignore_id| {
        lowerer.terminals.iter().any(|terminal| {
            matches!(terminal, Terminal::SpecialToken { id, .. } if *id == ignore_id)
        })
    }) {
        return Err(GlrMaskError::GrammarParse(
            "a special LLM token terminal cannot be the ignore terminal".into(),
        ));
    }
    let mut lexer_partitions = BTreeMap::new();
    for (terminal_name, partition) in &grammar.lexer_partitions {
        let terminal_id = lowerer.terminal_ids_by_name.get(terminal_name).copied();
        let Some(terminal_id) = terminal_id else {
            return Err(GlrMaskError::GrammarParse(format!(
                "lexer group references unknown or non-emitting terminal '{terminal_name}'",
            )));
        };
        if let Some(previous) = lexer_partitions.insert(terminal_id, partition.clone())
            && previous != *partition
        {
            return Err(GlrMaskError::GrammarParse(format!(
                "terminal '{terminal_name}' is assigned to both lexer groups '{previous}' and '{partition}'",
            )));
        }
    }
    for (literal, partition) in &grammar.lexer_literal_partitions {
        let matching = lowerer
            .terminals
            .iter()
            .filter_map(|terminal| match terminal {
                Terminal::Literal { id, bytes } if bytes == literal => Some(*id),
                _ => None,
            })
            .collect::<Vec<_>>();
        for terminal_id in matching {
            if let Some(previous) = lexer_partitions.insert(terminal_id, partition.clone())
                && previous != *partition
            {
                return Err(GlrMaskError::GrammarParse(format!(
                    "anonymous literal {:?} is assigned to both lexer groups '{previous}' and '{partition}'",
                    String::from_utf8_lossy(literal),
                )));
            }
        }
    }
    if let Some(default_partition) = &grammar.default_lexer_partition {
        for terminal_id in 0..lowerer.terminals.len() as TerminalID {
            lexer_partitions
                .entry(terminal_id)
                .or_insert_with(|| default_partition.clone());
        }
    }

    let dedup_started_at = detail_profile.then(std::time::Instant::now);
    dedup_rules_preserving_first_occurrence(&mut lowerer.rules);
    let dedup_ms = dedup_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    if let (Some(validate_ms), Some(setup_ms), Some(rule_lowering_ms), Some(finish_started_at)) =
        (validate_ms, setup_ms, rule_lowering_ms, finish_started_at)
    {
        eprintln!(
            "[glrmask/profile][grammar_ast_lower] named_rules={} terminals={} generated_rules={} validate_ms={:.3} setup_ms={:.3} rule_lowering_ms={:.3} finish_ms={:.3}",
            grammar.rules.len(),
            grammar.rules.iter().filter(|rule| rule.is_terminal).count(),
            lowerer.rules.len(),
            validate_ms,
            setup_ms,
            rule_lowering_ms,
            finish_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    if detail_profile {
        eprintln!(
            "[glrmask/profile][grammar_ast_lower_detail] named_exprs_ms={:.3} named_terminal_ms={:.3} nullable_ms={:.3} internal_names_ms={:.3} named_ids_ms={:.3} terminal_bodies_ms={:.3} terminal_rule_ms={:.3} expr_nfa_rule_ms={:.3} other_rule_ms={:.3} dedup_ms={:.3} expr_nfa_rules={} expr_nfa_states={} expr_nfa_transitions={}",
            named_exprs_ms,
            named_terminal_ms,
            nullable_ms,
            internal_names_ms,
            named_ids_ms,
            terminal_bodies_ms,
            terminal_rule_ms,
            expr_nfa_rule_ms,
            other_rule_ms,
            dedup_ms,
            expr_nfa_rules,
            expr_nfa_states,
            expr_nfa_transitions,
        );
    }

    let requires_global_terminal_observation = grammar.rules.iter().any(|rule| {
        rule.is_terminal && rule.name.starts_with("json_string_complex_pattern_")
    });

    Ok(GrammarDef {
        rules: lowerer.rules,
        start,
        terminals: lowerer.terminals,
        nonterminal_names,
        terminal_names: lowerer.terminal_names,
        ignore_terminal,
        lexer_partitions,
        residual_isolation_classes: BTreeMap::new(),
        requires_global_terminal_observation,
        direct_regular_automaton: lowerer.direct_regular_automaton,
    })
}

fn escape_byte(b: u8) -> String {
    match b {
        b'\n' => "\\n".into(),
        b'\r' => "\\r".into(),
        b'\t' => "\\t".into(),
        b'\\' => "\\\\".into(),
        b'"' => "\\\"".into(),
        byte if byte.is_ascii_graphic() || byte == b' ' => (byte as char).to_string(),
        byte => format!("\\x{byte:02x}"),
    }
}

fn regex_escape_byte(b: u8) -> String {
    match b {
        b'.' | b'+' | b'*' | b'?' | b'(' | b')' | b'[' | b']' | b'{' | b'}' | b'|' | b'^' | b'$' | b'\\' => {
            format!("\\{}", b as char)
        }
        _ => escape_byte(b),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        collect_terminal_rule_refs, grammar_expr_is_nullable, lower, GrammarExpr, Lowerer,
        NamedGrammar, NamedRule, Quantifier,
    };
    use crate::grammar::expr_nfa::ExprNfaBuilder;
    use rustc_hash::{FxHashMap, FxHashSet};

    fn nonterminal(name: &str, expr: GrammarExpr) -> NamedRule {
        NamedRule {
            name: name.to_string(),
            expr,
            is_terminal: false,
            is_internal: false,
        }
    }

    fn terminal(name: &str, expr: GrammarExpr) -> NamedRule {
        NamedRule {
            name: name.to_string(),
            expr,
            is_terminal: true,
            is_internal: false,
        }
    }

    fn literal(text: &str) -> GrammarExpr {
        GrammarExpr::Literal(text.as_bytes().to_vec())
    }

    fn subtract(lhs: &str, exclude: GrammarExpr) -> GrammarExpr {
        GrammarExpr::Exclude {
            expr: Box::new(GrammarExpr::Ref(lhs.to_string())),
            exclude: Box::new(exclude),
        }
    }

    #[test]
    fn expr_nfa_nullability_uses_epsilon_and_nullable_label_reachability() {
        let mut builder = ExprNfaBuilder::new();
        let nullable_state = builder.add_state();
        let accept = builder.add_state();
        builder.add_epsilon(builder.start_state(), nullable_state);
        // Include a cycle to verify the reachability walk terminates.
        builder.add_epsilon(nullable_state, builder.start_state());
        builder.add_transition(
            nullable_state,
            GrammarExpr::Ref("EMPTY".to_string()),
            accept,
        );
        builder.add_transition(
            builder.start_state(),
            literal("x"),
            accept,
        );
        builder.set_accepting(accept);
        let expr = GrammarExpr::ExprNFA(Box::new(builder.build()));

        let mut nullable = FxHashMap::default();
        assert!(!grammar_expr_is_nullable(&expr, &nullable));
        nullable.insert("EMPTY".to_string(), true);
        assert!(grammar_expr_is_nullable(&expr, &nullable));
    }

    #[test]
    fn terminal_reference_collection_walks_nested_expression_forms() {
        let expr = GrammarExpr::Exclude {
            expr: Box::new(GrammarExpr::Choice(vec![
                GrammarExpr::Ref("first".to_string()),
                GrammarExpr::Grouped(Box::new(GrammarExpr::Ref("second".to_string()))),
            ])),
            exclude: Box::new(GrammarExpr::SeparatedSequence {
                items: vec![(GrammarExpr::Ref("third".to_string()), None)],
                separator: Box::new(GrammarExpr::Ref("separator".to_string())),
                allow_empty: false,
            }),
        };
        let mut refs = FxHashSet::default();
        collect_terminal_rule_refs(&expr, &mut refs);

        assert_eq!(refs.len(), 4);
        assert!(refs.contains("first"));
        assert!(refs.contains("second"));
        assert!(refs.contains("third"));
        assert!(refs.contains("separator"));
    }

    #[test]
    fn exact_subtraction_matches_nonterminal_alias_body() {
        let grammar = NamedGrammar {
            rules: vec![
                terminal("JSON_STRING_BODY", literal("body\"")),
                nonterminal(
                    "json_string",
                    GrammarExpr::Sequence(vec![
                        literal("\""),
                        GrammarExpr::Ref("JSON_STRING_BODY".to_string()),
                    ]),
                ),
                nonterminal(
                    "json_value",
                    GrammarExpr::Choice(vec![
                        GrammarExpr::Ref("json_string".to_string()),
                        literal("0"),
                    ]),
                ),
                nonterminal(
                    "start",
                    subtract("json_value", GrammarExpr::Ref("json_string".to_string())),
                ),
            ],
            start: "start".to_string(),
            ignore: None,
            lexer_partitions: Default::default(),
            lexer_literal_partitions: Default::default(),
            default_lexer_partition: None,
        };

        lower(&grammar).unwrap();
    }

    #[test]
    fn exact_subtraction_canonicalization_is_cycle_safe() {
        let grammar = NamedGrammar {
            rules: vec![
                nonterminal(
                    "loop",
                    GrammarExpr::Sequence(vec![
                        GrammarExpr::Ref("loop".to_string()),
                        literal("y"),
                    ]),
                ),
                nonterminal(
                    "A",
                    GrammarExpr::Choice(vec![
                        GrammarExpr::Ref("loop".to_string()),
                        literal("x"),
                    ]),
                ),
                nonterminal("start", subtract("A", literal("z"))),
            ],
            start: "start".to_string(),
            ignore: None,
            lexer_partitions: Default::default(),
            lexer_literal_partitions: Default::default(),
            default_lexer_partition: None,
        };

        lower(&grammar).unwrap();
    }

    fn repeated_compound_label_expr_nfa(canonical: bool) -> GrammarExpr {
        let mut builder = ExprNfaBuilder::new();
        let middle = builder.add_state();
        let accept = builder.add_state();
        let label = GrammarExpr::Sequence(vec![
            literal("a"),
            GrammarExpr::Ref("item".to_string()),
        ]);
        builder.add_transition(builder.start_state(), label.clone(), middle);
        builder.add_transition(middle, label, accept);
        builder.set_accepting(accept);
        let expr_nfa = builder.build();
        GrammarExpr::ExprNFA(Box::new(if canonical {
            expr_nfa.into_determinized_and_minimized()
        } else {
            expr_nfa
        }))
    }

    fn assert_repeated_compound_label_is_shared(canonical: bool) {
        let grammar = NamedGrammar {
            rules: vec![
                nonterminal("item", literal("b")),
                nonterminal("start", repeated_compound_label_expr_nfa(canonical)),
            ],
            start: "start".to_string(),
            ignore: None,
            lexer_partitions: Default::default(),
            lexer_literal_partitions: Default::default(),
            default_lexer_partition: None,
        };
        let gdef = lower(&grammar).unwrap();
        let a_terminal = gdef
            .terminals
            .iter()
            .position(|terminal| terminal.name() == "a")
            .expect("literal a terminal") as u32;
        let wrapper_rules = gdef
            .rules
            .iter()
            .filter(|rule| {
                matches!(
                    rule.rhs.first(),
                    Some(crate::grammar::flat::Symbol::Terminal(tid)) if *tid == a_terminal
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            wrapper_rules.len(),
            1,
            "one immutable wrapper should serve every edge carrying the same compound label"
        );
    }

    #[test]
    fn expr_nfa_dfa_edges_reuse_compound_labels() {
        assert_repeated_compound_label_is_shared(false);
    }

    #[test]
    fn canonical_expr_nfa_dfa_edges_reuse_compound_labels() {
        assert_repeated_compound_label_is_shared(true);
    }

    #[test]
    fn literal_terminal_cache_preserves_lossy_name_updates() {
        let mut lowerer = Lowerer::new();
        let first_bytes = [0xff];
        let second_bytes = [0xfe];
        let lossy_name = String::from_utf8_lossy(&first_bytes).into_owned();
        assert_eq!(lossy_name, String::from_utf8_lossy(&second_bytes));

        let first_id = lowerer.literal_terminal_id(&first_bytes);
        let second_id = lowerer.literal_terminal_id(&second_bytes);
        assert_ne!(first_id, second_id);
        assert_eq!(lowerer.terminal_ids_by_name.get(&lossy_name), Some(&second_id));

        let repeated_first_id = lowerer.literal_terminal_id(&first_bytes);
        assert_eq!(repeated_first_id, first_id);
        assert_eq!(lowerer.terminal_ids_by_name.get(&lossy_name), Some(&first_id));
        assert_eq!(lowerer.terminals.len(), 2);
    }

    #[test]
    fn lower_deduplicates_identical_rules() {
        let grammar = NamedGrammar {
            rules: vec![nonterminal(
                "start",
                GrammarExpr::Choice(vec![literal("a"), literal("a")]),
            )],
            start: "start".to_string(),
            ignore: None,
            lexer_partitions: Default::default(),
            lexer_literal_partitions: Default::default(),
            default_lexer_partition: None,
        };

        let gdef = lower(&grammar).unwrap();
        let start_rules = gdef
            .rules
            .iter()
            .filter(|rule| rule.lhs == gdef.start)
            .count();
        assert_eq!(start_rules, 1, "duplicate alternatives should not create duplicate rules");
    }

    #[test]
    fn nonnullable_sequence_with_nonnullable_part_reduces_rules() {
        let grammar = NamedGrammar {
            rules: vec![
                nonterminal(
                    "body",
                    GrammarExpr::Choice(vec![
                        literal("abc"),
                        GrammarExpr::Epsilon,
                    ]),
                ),
                nonterminal(
                    "item",
                    GrammarExpr::Quantified(Box::new(GrammarExpr::Sequence(vec![
                        literal("{"),
                        GrammarExpr::Ref("body".to_string()),
                        literal("}"),
                    ])), Quantifier::OnePlus),
                ),
                nonterminal("start", GrammarExpr::Ref("item".to_string())),
            ],
            start: "start".to_string(),
            ignore: None,
            lexer_partitions: Default::default(),
            lexer_literal_partitions: Default::default(),
            default_lexer_partition: None,
        };

        let gdef = lower(&grammar).unwrap();
        let brace_rules_count = gdef
            .rules
            .iter()
            .filter(|rule| {
                matches!(
                    rule.rhs.first(),
                    Some(crate::grammar::flat::Symbol::Terminal(tid))
                        if gdef.terminal_display_name(*tid) == "{"
                )
            })
            .count();
        assert_eq!(
            brace_rules_count,
            1,
            "nonnullable sequence should not synthesize duplicate brace-start alternatives"
        );
    }
}
