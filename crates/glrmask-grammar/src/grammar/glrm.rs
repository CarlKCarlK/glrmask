//! GLRM: GLRMask's native grammar format.
//!
//! New grammars use the versioned GLRM v1 surface. Unversioned input is parsed
//! as the legacy compatibility format, so existing `::=` grammars and numeric
//! `@token(id)` terminals continue to work unchanged.
//!
//! # GLRM v1
//!
//! ```text
//! glrm 1;
//! start value;
//! [ignore IGNORE;]
//!
//! internal t DIGIT = /[0-9]/;
//! t NUMBER = /-?(0|[1-9][0-9]*)/;
//! extern token END_TURN;
//! nt value = NUMBER | "null" | END_TURN;
//! ```
//!
//! V1 declarations use `=` and terminate with semicolons. Epsilon is written
//! explicitly as `eps`; empty alternatives and empty groups are errors. Single
//! and double quoted literals have the same byte-string semantics; the v1
//! formatter chooses the delimiter requiring fewer escapes and prefers double
//! quotes on a tie. Exact
//! model-token IDs do not appear in v1 source: `extern token NAME;` declares a
//! parser-visible exact-token terminal whose ID or IDs are supplied by the host.
//! External terminals may appear in nonterminal expressions and nonterminal FA
//! edges, but not in byte-level terminal expressions.
//!
//! Direct finite automata are declaration bodies in v1:
//!
//! ```text
//! nt member = fa {
//!     start begin;
//!     accept done;
//!     begin -> done: STRING;
//! };
//! ```
//!
//! States may be integers or identifiers. Epsilon edges use `: eps;`. Terminal
//! declarations may use the same `fa { ... }` form; those automata are compiled
//! directly into the lexer rather than expanded into regular-expression syntax.
//!
//! GLRMask-specific lexer partition hints are namespaced in v1:
//!
//! ```text
//! pragma glrmask {
//!     lexer group strings = STRING, KEY, @literals;
//! }
//! ```
//!
//! Inline subgrammars use `g name = { ... };`; `extern grammar name;` declares a
//! compiled subgrammar supplied by the host. Scope and ignore semantics remain
//! local to each grammar.
//!
//! ## Expressions
//!
//! | Syntax | Meaning |
//! |---|---|
//! | `name` | Reference to a rule or declared external |
//! | `"text"`, `'text'` | Literal bytes |
//! | `/regex/` | Raw regex pattern (terminal expressions only) |
//! | `[class]`, `[^class]` | Byte character class |
//! | `[class]/utf8` | UTF-8 character class |
//! | `.` | Any byte |
//! | `eps` | Epsilon |
//! | `a b c` | Sequence |
//! | `a \| b` | Choice |
//! | `e?`, `e*`, `e+`, `e{n}`, `e{n,m}` | Repetition |
//! | `a - b` | Exact-language subtraction |
//! | `a & b` | Intersection |
//! | `sep ~ (item_a? item_b)` | Separated sequence |
//!
//! # Legacy compatibility
//!
//! With no `glrm 1;` header the legacy parser is selected. It retains `::=`,
//! implicit epsilon alternatives, top-level `fa name ::= { ... };`, numeric
//! `@token(id)`, legacy lexer-group syntax, and the terminal-FA syntax added by
//! the earlier terminal-automata feature branch.
//!
//! `to_glrm` serializes the legacy form. `to_glrm_v1` serializes the versioned
//! form when the lowered grammar still contains enough source information to do
//! so. Bound external-token names cannot be reconstructed from raw token IDs.

use crate::GlrMaskError;
use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::regex::validate_regular_regex;
use crate::automata::unweighted_u32::dfa::Label;
use crate::automata::unweighted_u32::nfa::NFA;
use crate::grammar::ast::{
    GrammarExpr, NamedGrammar, NamedRule, Quantifier, resolved_named_terminal_exprs,
};
use crate::grammar::expr_nfa::ExprNFA;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

// ============================================================
// Dumper
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DumpFlavor {
    Legacy,
    V1,
}

fn lexer_groups_for_dump(grammar: &NamedGrammar, flavor: DumpFlavor) -> Vec<(String, Vec<String>)> {
    let anonymous_literals = grammar.emitted_anonymous_literals();
    let literal_selector_partition = if !anonymous_literals.is_empty()
        && anonymous_literals
            .iter()
            .all(|literal| grammar.lexer_literal_partitions.contains_key(literal))
    {
        let mut counts = BTreeMap::<&str, usize>::new();
        for literal in &anonymous_literals {
            let partition = grammar
                .lexer_literal_partitions
                .get(literal)
                .expect("all anonymous literals checked above");
            *counts.entry(partition.as_str()).or_default() += 1;
        }
        counts
            .into_iter()
            .max_by(|(left_name, left_count), (right_name, right_count)| {
                left_name
                    .contains("literal")
                    .cmp(&right_name.contains("literal"))
                    .then_with(|| left_count.cmp(right_count))
                    .then_with(|| right_name.cmp(left_name))
            })
            .map(|(partition, _)| partition)
    } else {
        None
    };

    let mut lexer_groups = BTreeMap::<&str, Vec<String>>::new();
    for (terminal, partition) in &grammar.lexer_partitions {
        lexer_groups
            .entry(partition.as_str())
            .or_default()
            .push(terminal.clone());
    }
    for (literal, partition) in &grammar.lexer_literal_partitions {
        if literal_selector_partition == Some(partition.as_str())
            && anonymous_literals.contains(literal)
        {
            continue;
        }
        lexer_groups
            .entry(partition.as_str())
            .or_default()
            .push(format_literal(literal, flavor));
    }
    if let Some(partition) = literal_selector_partition {
        lexer_groups
            .entry(partition)
            .or_default()
            .push("@literals".to_string());
    }
    if let Some(partition) = grammar.default_lexer_partition.as_deref() {
        lexer_groups
            .entry(partition)
            .or_default()
            .push("*".to_string());
    }

    lexer_groups
        .into_iter()
        .map(|(partition, mut members)| {
            members.sort_unstable();
            (partition.to_string(), members)
        })
        .collect()
}

/// Serialise `grammar` to the legacy, unversioned GLRM text format.
pub fn to_glrm(grammar: &NamedGrammar) -> String {
    let mut out = String::new();
    out.push_str(&format!("start {};\n", grammar.start));
    if let Some(ref ign) = grammar.ignore {
        out.push_str(&format!("ignore {};\n", ign));
    }
    for (partition, members) in lexer_groups_for_dump(grammar, DumpFlavor::Legacy) {
        out.push_str(&format!(
            "lexer group {} ::= {};\n",
            partition,
            members.join(", "),
        ));
    }
    out.push('\n');

    for rule in &grammar.rules {
        let prefix = match (rule.is_terminal, rule.is_internal) {
            (true, true) => "internal t",
            (true, false) => "t",
            (false, _) => "nt",
        };
        if let GrammarExpr::ExprNFA(expr_nfa) = &rule.expr {
            if rule.is_terminal {
                out.push_str(&format!("{prefix} {} ::= fa {{\n", rule.name));
            } else {
                // Preserve the established unversioned spelling for parser FAs.
                out.push_str(&format!("fa {} ::= {{\n", rule.name));
            }
            out.push_str(&dump_expr_nfa(expr_nfa));
            out.push_str("};\n");
            continue;
        }
        let body = dump_nt_expr(&rule.expr, false);
        out.push_str(&format!("{} {} ::= {};\n", prefix, rule.name, body));
    }

    out
}

/// Serialize a grammar using the versioned GLRM v1 surface syntax.
///
/// Bound exact-token terminals cannot be reconstructed as named `extern token`
/// declarations from a lowered [`NamedGrammar`], so grammars containing raw
/// [`GrammarExpr::SpecialToken`] nodes are rejected instead of leaking the
/// legacy numeric `@token(id)` syntax into v1 output.
pub fn to_glrm_v1(grammar: &NamedGrammar) -> Result<String, GlrMaskError> {
    fn contains_special(expr: &GrammarExpr) -> bool {
        match expr {
            GrammarExpr::SpecialToken(_) => true,
            GrammarExpr::Grouped(inner) | GrammarExpr::Quantified(inner, _) => contains_special(inner),
            GrammarExpr::Sequence(items) | GrammarExpr::Choice(items) => items.iter().any(contains_special),
            GrammarExpr::Exclude { expr, exclude } => contains_special(expr) || contains_special(exclude),
            GrammarExpr::Intersect { expr, intersect } => contains_special(expr) || contains_special(intersect),
            GrammarExpr::SeparatedSequence { items, separator, .. } => {
                items.iter().any(|(item, _)| contains_special(item)) || contains_special(separator)
            }
            GrammarExpr::ExprNFA(nfa) => nfa.symbols.iter().any(contains_special),
            _ => false,
        }
    }
    if grammar.rules.iter().any(|rule| contains_special(&rule.expr)) {
        return Err(err("cannot dump bound exact-token IDs as GLRM v1; source-level external terminal names are not retained in NamedGrammar"));
    }

    let lexer_groups = lexer_groups_for_dump(grammar, DumpFlavor::V1);
    let mut out = format!("glrm 1;\nstart {};\n", grammar.start);
    if let Some(ignore) = &grammar.ignore {
        out.push_str(&format!("ignore {ignore};\n"));
    }
    if !lexer_groups.is_empty() {
        out.push_str("pragma glrmask {\n");
        for (partition, members) in lexer_groups {
            out.push_str(&format!(
                "  lexer group {} = {};\n",
                partition,
                members.join(", "),
            ));
        }
        out.push_str("}\n");
    }
    out.push('\n');
    for rule in &grammar.rules {
        let prefix = match (rule.is_terminal, rule.is_internal) {
            (true, true) => "internal t",
            (true, false) => "t",
            (false, _) => "nt",
        };
        if let GrammarExpr::ExprNFA(expr_nfa) = &rule.expr {
            out.push_str(&format!("{prefix} {} = fa {{\n", rule.name));
            out.push_str(&dump_expr_nfa_v1(expr_nfa));
            out.push_str("};\n");
        } else {
            out.push_str(&format!(
                "{prefix} {} = {};\n",
                rule.name,
                dump_nt_expr_v1(&rule.expr, false)
            ));
        }
    }
    Ok(out)
}

/// Canonically format a versioned GLRM v1 source document without lowering
/// source-level declarations such as `extern token NAME;`.
///
/// This is intentionally a source-layer formatter: exact token IDs are not
/// accepted or emitted, and externally bound terminal names survive unchanged.
/// Unversioned legacy GLRM must first be migrated explicitly rather than being
/// silently reinterpreted as v1.
pub fn format_glrm_v1(input: &str) -> Result<String, GlrMaskError> {
    let tokens = Lexer::new(input).tokenize()?;
    let mut parser = GlrmParser::new(tokens)?;
    if parser.version != GlrmVersion::V1 {
        return Err(err(
            "format_glrm_v1 requires a leading 'glrm 1;' header",
        ));
    }
    let scope = parser.parse_root_scope()?;
    validate_scope_for_v1_format(&scope, "grammar")?;

    let mut out = String::from("glrm 1;\n");
    dump_parsed_scope_v1(&scope, 0, &mut out)?;
    Ok(out)
}

fn validate_scope_for_v1_format(
    scope: &ParsedGlrmScope,
    scope_label: &str,
) -> Result<(), GlrMaskError> {
    scope_symbol_kinds(scope, scope_label)?;
    for subgrammar in &scope.subgrammars {
        if let ParsedSubgrammarBody::Inline(child) = &subgrammar.body {
            validate_scope_for_v1_format(child, &format!("{scope_label}::{}", subgrammar.name))?;
        }
    }
    Ok(())
}

fn scope_grammar_for_v1_format(scope: &ParsedGlrmScope) -> NamedGrammar {
    let mut grammar = NamedGrammar {
        rules: scope.rules.clone(),
        start: scope.start.clone(),
        ignore: scope.ignore.clone(),
        lexer_partitions: scope.lexer_partitions.clone(),
        lexer_literal_partitions: scope.lexer_literal_partitions.clone(),
        default_lexer_partition: scope.default_lexer_partition.clone(),
    };
    if let Some(partition) = scope.all_literals_partition.as_deref() {
        for literal in grammar.emitted_anonymous_literals() {
            grammar
                .lexer_literal_partitions
                .entry(literal)
                .or_insert_with(|| partition.to_string());
        }
    }
    grammar
}

fn push_indented_line(out: &mut String, indent: usize, line: &str) {
    out.extend(std::iter::repeat_n(' ', indent));
    out.push_str(line);
    out.push('\n');
}

fn push_indented_block(out: &mut String, indent: usize, block: &str) {
    for line in block.lines() {
        push_indented_line(out, indent, line);
    }
}

fn dump_parsed_scope_v1(
    scope: &ParsedGlrmScope,
    indent: usize,
    out: &mut String,
) -> Result<(), GlrMaskError> {
    push_indented_line(out, indent, &format!("start {};", scope.start));
    if let Some(ignore) = &scope.ignore {
        push_indented_line(out, indent, &format!("ignore {ignore};"));
    }

    let grammar = scope_grammar_for_v1_format(scope);
    let lexer_groups = lexer_groups_for_dump(&grammar, DumpFlavor::V1);
    if !lexer_groups.is_empty() {
        push_indented_line(out, indent, "pragma glrmask {");
        for (partition, members) in lexer_groups {
            push_indented_line(
                out,
                indent + 2,
                &format!("lexer group {partition} = {};", members.join(", ")),
            );
        }
        push_indented_line(out, indent, "}");
    }

    for terminal in &scope.external_terminals {
        push_indented_line(out, indent, &format!("extern token {};", terminal.name));
    }

    for subgrammar in &scope.subgrammars {
        match &subgrammar.body {
            ParsedSubgrammarBody::External => {
                push_indented_line(out, indent, &format!("extern grammar {};", subgrammar.name));
            }
            ParsedSubgrammarBody::Inline(child) => {
                push_indented_line(out, indent, &format!("g {} = {{", subgrammar.name));
                dump_parsed_scope_v1(child, indent + 2, out)?;
                push_indented_line(out, indent, "};");
            }
        }
    }

    for rule in &scope.rules {
        let prefix = match (rule.is_terminal, rule.is_internal) {
            (true, true) => "internal t",
            (true, false) => "t",
            (false, _) => "nt",
        };
        if let GrammarExpr::ExprNFA(expr_nfa) = &rule.expr {
            push_indented_line(out, indent, &format!("{prefix} {} = fa {{", rule.name));
            push_indented_block(out, indent, &dump_expr_nfa_v1(expr_nfa));
            push_indented_line(out, indent, "};");
        } else {
            push_indented_line(
                out,
                indent,
                &format!(
                    "{prefix} {} = {};",
                    rule.name,
                    dump_nt_expr_v1(&rule.expr, false)
                ),
            );
        }
    }
    Ok(())
}

fn dump_expr_nfa_v1(expr_nfa: &ExprNFA) -> String {
    let mut out = String::new();
    let starts = expr_nfa
        .nfa
        .start_states
        .iter()
        .map(|&state| expr_nfa.state_name(state))
        .collect::<Vec<_>>()
        .join(", ");
    out.push_str(&format!("  start {starts};\n"));
    let accepts = expr_nfa
        .nfa
        .states
        .iter()
        .enumerate()
        .filter_map(|(state, value)| value.is_accepting.then(|| expr_nfa.state_name(state as u32)))
        .collect::<Vec<_>>()
        .join(", ");
    out.push_str(&format!("  accept {accepts};\n"));
    for (from, state) in expr_nfa.nfa.states.iter().enumerate() {
        let from = expr_nfa.state_name(from as u32);
        for &to in &state.epsilons {
            out.push_str(&format!("  {from} -> {}: eps;\n", expr_nfa.state_name(to)));
        }
        for (&label, targets) in &state.transitions {
            let symbol = expr_nfa
                .symbol_for_label(label)
                .map(|expr| dump_nt_expr_v1(expr, false))
                .unwrap_or_else(|| "eps".to_string());
            for &to in targets {
                out.push_str(&format!("  {from} -> {}: {symbol};\n", expr_nfa.state_name(to)));
            }
        }
    }
    out
}

fn dump_expr_nfa(expr_nfa: &ExprNFA) -> String {
    fn is_epsilon_only(expr: &GrammarExpr) -> bool {
        match expr {
            GrammarExpr::Epsilon => true,
            GrammarExpr::Sequence(parts) => parts.is_empty(),
            GrammarExpr::Grouped(inner) => is_epsilon_only(inner),
            _ => false,
        }
    }

    let mut out = String::new();
    let starts = expr_nfa
        .nfa
        .start_states
        .iter()
        .map(|&state| expr_nfa.state_name(state))
        .collect::<Vec<_>>()
        .join(", ");
    out.push_str(&format!("  start {};\n", starts));

    let accepts = expr_nfa
        .nfa
        .states
        .iter()
        .enumerate()
        .filter_map(|(state_id, state)| state.is_accepting.then(|| expr_nfa.state_name(state_id as u32)))
        .collect::<Vec<_>>()
        .join(", ");
    out.push_str(&format!("  accept {};\n\n", accepts));

    for (state_id, state) in expr_nfa.nfa.states.iter().enumerate() {
        let source = expr_nfa.state_name(state_id as u32);
        for &target in &state.epsilons {
            out.push_str(&format!("  {source} --> {};\n", expr_nfa.state_name(target)));
        }
        for (&label, targets) in &state.transitions {
            let symbol_expr = expr_nfa.symbol_for_label(label);
            for &target in targets {
                if symbol_expr.is_some_and(is_epsilon_only) {
                    out.push_str(&format!(
                        "  {source} --> {};\n",
                        expr_nfa.state_name(target)
                    ));
                } else {
                    let symbol = symbol_expr
                        .map(|expr| dump_nt_expr(expr, false))
                        .unwrap_or_else(|| format!("/*invalid-symbol-{label}*/ eps"));
                    out.push_str(&format!(
                        "  {source} -- {symbol} --> {};\n",
                        expr_nfa.state_name(target)
                    ));
                }
            }
        }
    }
    out
}

// ---- NT-expression dumper --------------------------------------------------

fn dump_nt_expr(expr: &GrammarExpr, needs_parens: bool) -> String {
    dump_nt_expr_with_flavor(expr, needs_parens, DumpFlavor::Legacy)
}

fn dump_nt_expr_v1(expr: &GrammarExpr, needs_parens: bool) -> String {
    dump_nt_expr_with_flavor(expr, needs_parens, DumpFlavor::V1)
}

fn dump_nt_expr_with_flavor(
    expr: &GrammarExpr,
    needs_parens: bool,
    flavor: DumpFlavor,
) -> String {
    match expr {
        GrammarExpr::Choice(alts) => {
            let inner = alts
                .iter()
                .map(|a| dump_nt_seq(a, flavor))
                .collect::<Vec<_>>()
                .join(" | ");
            if needs_parens && alts.len() > 1 {
                format!("({inner})")
            } else {
                inner
            }
        }
        GrammarExpr::Exclude { expr: inner, exclude } => {
            let lhs = dump_set_operand(inner, flavor);
            let rhs = match exclude.as_ref() {
                GrammarExpr::Choice(alts) if !alts.is_empty() => alts
                    .iter()
                    .map(|expr| dump_set_operand(expr, flavor))
                    .collect::<Vec<_>>()
                    .join(" - "),
                _ => dump_set_operand(exclude, flavor),
            };
            let infix = format!("{lhs} - {rhs}");
            if needs_parens { format!("({infix})") } else { infix }
        }
        GrammarExpr::Intersect { expr: inner, intersect } => {
            let infix = format!(
                "{} & {}",
                dump_set_operand(inner, flavor),
                dump_set_operand(intersect, flavor)
            );
            if needs_parens { format!("({infix})") } else { infix }
        }
        _ => dump_nt_seq(expr, flavor),
    }
}

/// Dump a sequence (or a single non-choice item).
fn dump_nt_seq(expr: &GrammarExpr, flavor: DumpFlavor) -> String {
    match expr {
        GrammarExpr::Sequence(items) if items.is_empty() && flavor == DumpFlavor::V1 => {
            "eps".to_string()
        }
        GrammarExpr::Sequence(items) => items
            .iter()
            .map(|expr| dump_nt_postfix(expr, flavor))
            .collect::<Vec<_>>()
            .join(" "),
        _ => dump_nt_postfix(expr, flavor),
    }
}

fn dump_quantifier(quantifier: &Quantifier) -> String {
    match quantifier {
        Quantifier::Optional => "?".to_string(),
        Quantifier::ZeroPlus => "*".to_string(),
        Quantifier::OnePlus => "+".to_string(),
        Quantifier::Range(min, Some(max)) if min == max => format!("{{{min}}}"),
        Quantifier::Range(min, Some(max)) => format!("{{{min},{max}}}"),
        Quantifier::Range(min, None) => format!("{{{min},}}"),
    }
}

fn dump_nt_postfix(expr: &GrammarExpr, flavor: DumpFlavor) -> String {
    match expr {
        GrammarExpr::Quantified(inner, Quantifier::Optional) => {
            format!("{}?", dump_nt_atom(inner, flavor))
        }
        GrammarExpr::Quantified(inner, Quantifier::ZeroPlus) => {
            format!("{}*", dump_nt_atom(inner, flavor))
        }
        GrammarExpr::Quantified(inner, Quantifier::OnePlus) => {
            format!("{}+", dump_nt_atom(inner, flavor))
        }
        GrammarExpr::Quantified(inner, Quantifier::Range(min, max)) => match max {
            Some(max) if min == max => format!("{}{{{min}}}", dump_nt_atom(inner, flavor)),
            Some(max) => format!("{}{{{min},{max}}}", dump_nt_atom(inner, flavor)),
            None => format!("{}{{{min},}}", dump_nt_atom(inner, flavor)),
        },
        _ => dump_nt_atom(expr, flavor),
    }
}

fn dump_nt_atom(expr: &GrammarExpr, flavor: DumpFlavor) -> String {
    match expr {
        GrammarExpr::Ref(name) => name.clone(),
        GrammarExpr::Grouped(inner) => {
            format!("({})", dump_nt_expr_with_flavor(inner, false, flavor))
        }
        GrammarExpr::Literal(bytes) => format_literal(bytes, flavor),
        GrammarExpr::SpecialToken(token_id) => format!("@token({token_id})"),
        GrammarExpr::RawRegex(pat) => format!("/{}/", escape_regex_for_slash(pat)),
        GrammarExpr::CharClass { def, negate, utf8 } => {
            let inner = if *negate { format!("^{def}") } else { def.clone() };
            let suffix = if *utf8 { "/utf8" } else { "" };
            format!("[{inner}]{suffix}")
        }
        GrammarExpr::LexerDfa(_) => "LexerDfa".to_string(),
        GrammarExpr::AnyByte => ".".to_string(),
        GrammarExpr::Epsilon => "eps".to_string(),
        GrammarExpr::Exclude { expr: inner, exclude } => {
            let lhs = dump_set_operand(inner, flavor);
            match exclude.as_ref() {
                GrammarExpr::Choice(alts) if !alts.is_empty() => {
                    let rhs = alts
                        .iter()
                        .map(|expr| dump_set_operand(expr, flavor))
                        .collect::<Vec<_>>()
                        .join(" - ");
                    format!("({lhs} - {rhs})")
                }
                _ => format!("({lhs} - {})", dump_set_operand(exclude, flavor)),
            }
        }
        GrammarExpr::Intersect { expr: inner, intersect } => format!(
            "({} & {})",
            dump_set_operand(inner, flavor),
            dump_set_operand(intersect, flavor)
        ),
        GrammarExpr::SeparatedSequence { items, separator, allow_empty } => {
            let sep_str = dump_nt_atom(separator, flavor);
            let items_str = items
                .iter()
                .map(|(expr, quantifier)| {
                    let mut text = dump_nt_atom(expr, flavor);
                    if let Some(quantifier) = quantifier {
                        text.push_str(&dump_quantifier(quantifier));
                    }
                    text
                })
                .collect::<Vec<_>>()
                .join(" ");
            if *allow_empty {
                format!("{sep_str} ~ ( {items_str} )")
            } else {
                format!("{sep_str} ~+ ( {items_str} )")
            }
        }
        GrammarExpr::ExprNFA(expr_nfa) => format!(
            "ExprNFA(states={}, symbols={})",
            expr_nfa.nfa.states.len(),
            expr_nfa.symbols.len()
        ),
        GrammarExpr::Sequence(_) | GrammarExpr::Choice(_) => {
            format!("({})", dump_nt_expr_with_flavor(expr, false, flavor))
        }
        GrammarExpr::Quantified(_, Quantifier::Optional)
        | GrammarExpr::Quantified(_, Quantifier::ZeroPlus)
        | GrammarExpr::Quantified(_, Quantifier::OnePlus)
        | GrammarExpr::Quantified(_, Quantifier::Range(_, _)) => {
            format!("({})", dump_nt_postfix(expr, flavor))
        }
    }
}

fn dump_set_operand(expr: &GrammarExpr, flavor: DumpFlavor) -> String {
    match expr {
        GrammarExpr::Choice(_) | GrammarExpr::Exclude { .. } | GrammarExpr::Intersect { .. } => {
            format!("({})", dump_nt_expr_with_flavor(expr, false, flavor))
        }
        _ => dump_nt_expr_with_flavor(expr, false, flavor),
    }
}

// ---- Helpers ---------------------------------------------------------------

fn escape_bytes_for_delimiter(bytes: &[u8], delimiter: u8) -> String {
    debug_assert!(delimiter == b'"' || delimiter == b'\'');
    let mut out = String::new();
    for &byte in bytes {
        match byte {
            b'\\' => out.push_str("\\\\"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            byte if byte == delimiter => {
                out.push('\\');
                out.push(byte as char);
            }
            0x20..=0x7E => out.push(byte as char),
            _ => out.push_str(&format!("\\x{byte:02X}")),
        }
    }
    out
}

fn format_literal(bytes: &[u8], flavor: DumpFlavor) -> String {
    let double_quoted = escape_bytes_for_delimiter(bytes, b'"');
    if flavor == DumpFlavor::Legacy {
        return format!("\"{double_quoted}\"");
    }

    let single_quoted = escape_bytes_for_delimiter(bytes, b'\'');
    if single_quoted.len() < double_quoted.len() {
        format!("'{single_quoted}'")
    } else {
        // GLRM v1 prefers double quotes when both spellings require the same
        // amount of escaping.
        format!("\"{double_quoted}\"")
    }
}

fn escape_regex_for_slash(pat: &str) -> String {
    pat.replace('/', "\\/")
}

// ============================================================
// Parser
// ============================================================

/// Parse a GLRM-format string into a [`NamedGrammar`].
pub fn from_glrm(input: &str) -> Result<NamedGrammar, GlrMaskError> {
    let tokens = Lexer::new(input).tokenize()?;
    let mut parser = GlrmParser::new(tokens)?;
    let mut scope = parser.parse_root_scope()?;
    if contains_external_subgrammar(&scope) {
        return Err(err(
            "external subgrammars require from_glrm_with_external_subgrammars and explicit bindings",
        ));
    }
    bind_external_terminals(&mut scope, &[])?;
    lower_parsed_grammar(scope, &BTreeMap::new()).map(|parsed| parsed.grammar)
}

/// Parse GLRM v1 and bind every `extern token NAME;` declaration to one or more
/// exact model token IDs. These terminals are parser-visible and have no byte
/// language; token IDs need not occur in the byte vocabulary.
pub fn from_glrm_with_external_terminals(
    input: &str,
    bindings: &[(&str, &[u32])],
) -> Result<NamedGrammar, GlrMaskError> {
    let tokens = Lexer::new(input).tokenize()?;
    let mut parser = GlrmParser::new(tokens)?;
    let mut scope = parser.parse_root_scope()?;
    if contains_external_subgrammar(&scope) {
        return Err(err(
            "external subgrammars require from_glrm_with_external_subgrammars and explicit bindings",
        ));
    }
    bind_external_terminals(&mut scope, bindings)?;
    lower_parsed_grammar(scope, &BTreeMap::new()).map(|parsed| parsed.grammar)
}

/// One external-subgrammar declaration lowered into a hidden special terminal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalSubgrammarPlaceholder {
    /// Source-level binding name. Nested externals use `outer::inner`.
    pub binding_name: String,
    /// Automatically allocated non-vocabulary token ID.
    pub token_id: u32,
}

/// Parsed GLRM parent shell plus its typed external-subgrammar binding manifest.
#[derive(Debug, Clone)]
pub struct GlrmWithExternalSubgrammars {
    pub grammar: NamedGrammar,
    pub placeholders: Vec<ExternalSubgrammarPlaceholder>,
}

/// The target-independent external declarations in a GLRM source descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlrmExternalDeclarations {
    pub token_names: Vec<String>,
    pub grammar_names: Vec<String>,
}

/// Parse and validate GLRM declarations without binding or lowering them.
///
/// Names nested inside inline grammar scopes are qualified with `::`.
pub fn external_declarations(
    input: &str,
) -> Result<GlrmExternalDeclarations, GlrMaskError> {
    let tokens = Lexer::new(input).tokenize()?;
    let mut parser = GlrmParser::new(tokens)?;
    let scope = parser.parse_root_scope()?;
    validate_scope_for_v1_format(&scope, "grammar")?;
    Ok(GlrmExternalDeclarations {
        token_names: collect_external_terminal_names(&scope),
        grammar_names: collect_external_subgrammar_names(&scope),
    })
}

/// Return explicit special-token IDs already present in a GLRM source.
///
/// This is used by compiled-module placeholder allocation so temporary
/// linker-only IDs never collide with a source-level @token(...) terminal.
#[doc(hidden)]
pub fn special_token_ids(input: &str) -> Result<Vec<u32>, GlrMaskError> {
    let tokens = Lexer::new(input).tokenize()?;
    let mut parser = GlrmParser::new(tokens)?;
    let scope = parser.parse_root_scope()?;
    let mut ids = BTreeSet::new();
    collect_scope_special_token_ids(&scope, &mut ids);
    Ok(ids.into_iter().collect())
}

/// Parse GLRM containing `extern grammar name;` declarations.
///
/// `first_placeholder_token_id` must lie outside the model vocabulary.
/// `reserved_token_ids` contains end-token and other caller-reserved IDs.
/// Existing `@token(...)` IDs in the source are discovered automatically.
/// Placeholder IDs are allocated upward from the supplied start, avoiding
/// dense-ID blowups and accidental collisions.
pub fn from_glrm_with_external_subgrammars(
    input: &str,
    first_placeholder_token_id: u32,
    reserved_token_ids: impl IntoIterator<Item = u32>,
) -> Result<GlrmWithExternalSubgrammars, GlrMaskError> {
    from_glrm_with_bindings_and_external_subgrammars(
        input,
        first_placeholder_token_id,
        reserved_token_ids,
        &[],
    )
}

/// Parse GLRM with both exact-token terminal and compiled-subgrammar manifests.
pub fn from_glrm_with_bindings_and_external_subgrammars(
    input: &str,
    first_placeholder_token_id: u32,
    reserved_token_ids: impl IntoIterator<Item = u32>,
    terminal_bindings: &[(&str, &[u32])],
) -> Result<GlrmWithExternalSubgrammars, GlrMaskError> {
    let tokens = Lexer::new(input).tokenize()?;
    let mut parser = GlrmParser::new(tokens)?;
    let mut scope = parser.parse_root_scope()?;
    bind_external_terminals(&mut scope, terminal_bindings)?;

    let mut used_token_ids = reserved_token_ids.into_iter().collect::<BTreeSet<_>>();
    collect_scope_special_token_ids(&scope, &mut used_token_ids);
    let external_names = collect_external_subgrammar_names(&scope);
    let mut next_candidate = first_placeholder_token_id;
    let mut external_tokens = BTreeMap::new();
    let external_count = external_names.len();
    for (index, binding_name) in external_names.into_iter().enumerate() {
        let token_id = next_free_token_id(&used_token_ids, next_candidate)?;
        used_token_ids.insert(token_id);
        external_tokens.insert(binding_name, token_id);
        if index + 1 < external_count {
            next_candidate = token_id.checked_add(1).ok_or_else(|| {
                err("no token ID is available for another external subgrammar placeholder")
            })?;
        }
    }
    lower_parsed_grammar(scope, &external_tokens)
}

/// Debug/internal grammar construction: parse a GLRM parent and structurally
/// replace selected `extern grammar NAME;` occurrences with inline subgrammars
/// taken from the supplied child GLRM sources.
///
/// `children` keys are fully qualified occurrence paths, not global names:
///
/// * `"child"` binds a direct `extern grammar child;` in the parent scope;
/// * `"outer::inner"` binds an `extern grammar inner;` nested inside an inline
///   `g outer ::= { ... };` subgrammar already present in the parent source;
/// * `"child::grandchild"` binds an `extern grammar grandchild;` inside the
///   child source bound to `"child"`.
///
/// Each occurrence is parsed once from its own source string and keeps its own
/// `start`, `ignore`, and lexer partitions; no global name substitution is
/// performed. Duplicate binding keys, external occurrences with no binding, and
/// supplied bindings that match no occurrence are all rejected explicitly.
///
/// This returns an ordinary [`NamedGrammar`]: it is grammar construction, not a
/// constraint/composition API, and the result compiles through the normal
/// non-composition compiler. It is GLRM-only and deliberately does not add a
/// `GrammarSource` variant. Bindings are a finite set of occurrence paths, so an
/// arbitrary cyclic `extern grammar` graph cannot be encoded; cycles are
/// therefore not detected as such (a path that is never supplied is reported as
/// an unbound external occurrence instead). Identical child source strings in
/// distinct occurrences are independent and are never treated as cycles.
#[cfg(feature = "internal-api")]
pub fn from_glrm_with_inline_subgrammars(
    parent: &str,
    children: &[(&str, &str)],
) -> Result<NamedGrammar, GlrMaskError> {
    const MAX_INLINE_DEPTH: usize = 64;

    let mut bindings = BTreeMap::<String, &str>::new();
    for &(key, source) in children {
        if bindings.insert(key.to_string(), source).is_some() {
            return Err(err(&format!(
                "inline subgrammar binding {key:?} was supplied more than once",
            )));
        }
    }
    let mut consumed = BTreeSet::<String>::new();

    let mut scope = parse_glrm_scope(parent)?;
    bind_inline_subgrammars(
        &mut scope,
        "",
        &bindings,
        &mut consumed,
        0,
        MAX_INLINE_DEPTH,
    )?;

    for key in bindings.keys() {
        if !consumed.contains(key) {
            return Err(err(&format!(
                "inline subgrammar binding {key:?} did not match any external grammar occurrence",
            )));
        }
    }

    let lowered = lower_parsed_grammar(scope, &BTreeMap::new())?;
    if !lowered.placeholders.is_empty() {
        return Err(err(
            "inline subgrammar replacement left an unbound external placeholder",
        ));
    }
    Ok(lowered.grammar)
}

#[cfg(feature = "internal-api")]
fn parse_glrm_scope(source: &str) -> Result<ParsedGlrmScope, GlrMaskError> {
    let tokens = Lexer::new(source).tokenize()?;
    let mut parser = GlrmParser::new(tokens)?;
    parser.parse_root_scope()
}

/// Structurally walk a parsed scope and replace every bound `External`
/// subgrammar with an `Inline` scope parsed from its child source. `prefix` is
/// the fully qualified occurrence path of `scope`; child paths extend it with
/// `::`. `consumed` records which supplied bindings were actually used so
/// unused bindings can be rejected after the walk.
#[cfg(feature = "internal-api")]
fn bind_inline_subgrammars(
    scope: &mut ParsedGlrmScope,
    prefix: &str,
    bindings: &BTreeMap<String, &str>,
    consumed: &mut BTreeSet<String>,
    depth: usize,
    max_depth: usize,
) -> Result<(), GlrMaskError> {
    if depth > max_depth {
        return Err(err(
            "inline subgrammar nesting exceeded the maximum supported depth",
        ));
    }
    for subgrammar in &mut scope.subgrammars {
        let path = if prefix.is_empty() {
            subgrammar.name.clone()
        } else {
            format!("{prefix}::{}", subgrammar.name)
        };
        if matches!(subgrammar.body, ParsedSubgrammarBody::External) {
            let source = bindings.get(&path).copied().ok_or_else(|| {
                err(&format!(
                    "external subgrammar {path:?} has no inline binding",
                ))
            })?;
            consumed.insert(path.clone());
            let mut child_scope = parse_glrm_scope(source)?;
            bind_inline_subgrammars(
                &mut child_scope,
                &path,
                bindings,
                consumed,
                depth + 1,
                max_depth,
            )?;
            subgrammar.body = ParsedSubgrammarBody::Inline(child_scope);
        } else if let ParsedSubgrammarBody::Inline(child_scope) = &mut subgrammar.body {
            bind_inline_subgrammars(
                child_scope,
                &path,
                bindings,
                consumed,
                depth + 1,
                max_depth,
            )?;
        }
    }
    Ok(())
}

// ---- Tokens ----------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    /// Identifier or keyword: `nt`, `t`, `internal`, `start`, `ignore`,
    /// `eps`
    Ident(String),
    /// String literal: `"..."` — bytes, after escape processing
    StringLit(Vec<u8>),
    /// Regex literal: `/.../` — raw pattern string (after unescape of `\/`)
    RegexLit(String),
    /// Character class: `[...]` or `[^...]` — (def_without_brackets, is_utf8)
    CharClass(String, bool),
    /// Integer
    Int(usize),
    /// `::=`
    DeclEq,
    /// `=`
    Equals,
    /// `:`
    Colon,
    /// `;`
    Semi,
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `|`
    Pipe,
    /// `&`
    Amp,
    /// `-`
    Minus,
    /// `--`
    Dashes,
    /// `-->`
    Arrow,
    /// `->`
    ThinArrow,
    /// `~`
    Tilde,
    /// `*`
    Star,
    /// `@`
    At,
    /// `+`
    Plus,
    /// `?`
    Quest,
    /// `,`
    Comma,
    /// `.`
    Dot,
    Eof,
}

// ---- Lexer -----------------------------------------------------------------

struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Self {
        Lexer { src: src.as_bytes(), pos: 0 }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn peek2(&self) -> Option<u8> {
        self.src.get(self.pos + 1).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let b = self.src.get(self.pos).copied()?;
        self.pos += 1;
        Some(b)
    }

    fn push_utf8_char_from(&mut self, start: usize, output: &mut String) {
        let remaining = std::str::from_utf8(&self.src[start..])
            .expect("GLRM source originated from valid UTF-8");
        let character = remaining
            .chars()
            .next()
            .expect("non-ASCII byte must begin a UTF-8 character");
        self.pos = start + character.len_utf8();
        output.push(character);
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            // Skip whitespace
            while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
                self.pos += 1;
            }
            // Skip line comment
            if self.peek() == Some(b'/') && self.peek2() == Some(b'/') {
                self.pos += 2;
                while !matches!(self.peek(), None | Some(b'\n')) {
                    self.pos += 1;
                }
                continue;
            }
            // Skip block comment
            if self.peek() == Some(b'/') && self.peek2() == Some(b'*') {
                self.pos += 2;
                loop {
                    if self.peek() == Some(b'*') && self.peek2() == Some(b'/') {
                        self.pos += 2;
                        break;
                    }
                    if self.advance().is_none() {
                        break; // unterminated block comment — treat as EOF
                    }
                }
                continue;
            }
            break;
        }
    }

    fn lex_string(&mut self, delim: u8) -> Result<Vec<u8>, GlrMaskError> {
        let mut bytes = Vec::new();
        loop {
            match self.advance() {
                Some(b) if b == delim => return Ok(bytes),
                Some(b'\\') => match self.advance() {
                    Some(b'n') => bytes.push(b'\n'),
                    Some(b't') => bytes.push(b'\t'),
                    Some(b'r') => bytes.push(b'\r'),
                    Some(b'\\') => bytes.push(b'\\'),
                    Some(b'"') if delim == b'"' => bytes.push(b'"'),
                    Some(b'\'') if delim == b'\'' => bytes.push(b'\''),
                    Some(b'x') => {
                        let hi = self.advance().ok_or_else(|| err("incomplete hex escape in string"))?;
                        let lo = self.advance().ok_or_else(|| err("incomplete hex escape in string"))?;
                        let val = (hex_digit(hi)? << 4) | hex_digit(lo)?;
                        bytes.push(val);
                    }
                    Some(c) => {
                        bytes.push(b'\\');
                        bytes.push(c);
                    }
                    None => return Err(err("unexpected EOF in string escape")),
                },
                Some(b) => bytes.push(b),
                None => return Err(err("unterminated string literal")),
            }
        }
    }

    fn lex_regex(&mut self) -> Result<String, GlrMaskError> {
        let mut pat = String::new();
        loop {
            match self.advance() {
                Some(b'/') => return Ok(pat),
                Some(b'\\') => {
                    match self.advance() {
                        Some(b'/') => pat.push('/'),  // escaped slash
                        Some(c) => {
                            pat.push('\\');
                            if c.is_ascii() {
                                pat.push(c as char);
                            } else {
                                self.push_utf8_char_from(self.pos - 1, &mut pat);
                            }
                        }
                        None => return Err(err("unexpected EOF in regex escape")),
                    }
                }
                Some(b) if b.is_ascii() => pat.push(b as char),
                Some(_) => self.push_utf8_char_from(self.pos - 1, &mut pat),
                None => return Err(err("unterminated regex literal")),
            }
        }
    }

    fn lex_char_class(&mut self) -> Result<(String, bool), GlrMaskError> {
        // We're past the opening `[`. Collect everything up to the closing `]`.
        let mut def = String::new();
        if self.peek() == Some(b'^') {
            def.push('^');
            self.pos += 1;
        }
        loop {
            match self.advance() {
                Some(b']') => break,
                Some(b'\\') => {
                    let c = self.advance().ok_or_else(|| err("unterminated char class escape"))?;
                    def.push('\\');
                    if c.is_ascii() {
                        def.push(c as char);
                    } else {
                        self.push_utf8_char_from(self.pos - 1, &mut def);
                    }
                }
                Some(b) if b.is_ascii() => def.push(b as char),
                Some(_) => self.push_utf8_char_from(self.pos - 1, &mut def),
                None => return Err(err("unterminated char class")),
            }
        }
        // Check for `/utf8` suffix
        let saved_pos = self.pos;
        self.skip_whitespace_and_comments();
        let is_utf8 = if self.peek() == Some(b'/') {
            // Try to read `utf8`
            self.pos += 1;
            let ident_start = self.pos;
            while matches!(self.peek(), Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')) {
                self.pos += 1;
            }
            let word = &self.src[ident_start..self.pos];
            if word == b"utf8" {
                true
            } else {
                // Not a utf8 annotation; backtrack
                self.pos = saved_pos;
                false
            }
        } else {
            self.pos = saved_pos;
            false
        };
        Ok((def, is_utf8))
    }

    fn lex_ident(&mut self, first: u8) -> String {
        let mut s = String::new();
        s.push(first as char);
        while matches!(self.peek(), Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')) {
            s.push(self.src[self.pos] as char);
            self.pos += 1;
        }
        s
    }

    fn lex_int(&mut self, first: u8) -> Result<usize, GlrMaskError> {
        let mut n = (first - b'0') as usize;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            n = n * 10 + (self.src[self.pos] - b'0') as usize;
            self.pos += 1;
        }
        Ok(n)
    }

    fn tokenize(mut self) -> Result<Vec<Tok>, GlrMaskError> {
        let mut tokens = Vec::new();
        loop {
            self.skip_whitespace_and_comments();
            match self.peek() {
                None => { tokens.push(Tok::Eof); break; }
                Some(b) => {
                    self.pos += 1;
                    match b {
                        b';' => tokens.push(Tok::Semi),
                        b'(' => tokens.push(Tok::LParen),
                        b')' => tokens.push(Tok::RParen),
                        b'{' => tokens.push(Tok::LBrace),
                        b'}' => tokens.push(Tok::RBrace),
                        b'|' => tokens.push(Tok::Pipe),
                        b'&' => tokens.push(Tok::Amp),
                        b'-' => {
                            if self.peek() == Some(b'-') && self.peek2() == Some(b'>') {
                                self.pos += 2;
                                tokens.push(Tok::Arrow);
                            } else if self.peek() == Some(b'-') {
                                self.pos += 1;
                                tokens.push(Tok::Dashes);
                            } else if self.peek() == Some(b'>') {
                                self.pos += 1;
                                tokens.push(Tok::ThinArrow);
                            } else {
                                tokens.push(Tok::Minus);
                            }
                        }
                        b'~' => tokens.push(Tok::Tilde),
                        b'*' => tokens.push(Tok::Star),
                        b'@' => tokens.push(Tok::At),
                        b'+' => tokens.push(Tok::Plus),
                        b'?' => tokens.push(Tok::Quest),
                        b',' => tokens.push(Tok::Comma),
                        b'.' => tokens.push(Tok::Dot),
                        b':' => {
                            if self.peek() == Some(b':') && self.peek2() == Some(b'=') {
                                self.pos += 2;
                                tokens.push(Tok::DeclEq);
                            } else {
                                tokens.push(Tok::Colon);
                            }
                        }
                        b'=' => tokens.push(Tok::Equals),
                        b'"' | b'\'' => {
                            let bytes = self.lex_string(b)?;
                            tokens.push(Tok::StringLit(bytes));
                        }
                        b'/' => {
                            let pat = self.lex_regex()?;
                            tokens.push(Tok::RegexLit(pat));
                        }
                        b'[' => {
                            let (def, is_utf8) = self.lex_char_class()?;
                            tokens.push(Tok::CharClass(def, is_utf8));
                        }
                        b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                            let s = self.lex_ident(b);
                            tokens.push(Tok::Ident(s));
                        }
                        b'0'..=b'9' => {
                            let n = self.lex_int(b)?;
                            tokens.push(Tok::Int(n));
                        }
                        other => {
                            return Err(err(&format!("unexpected character '{}' (0x{:02X})", other as char, other)));
                        }
                    }
                }
            }
        }
        Ok(tokens)
    }
}

// ---- Parser ----------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GlrmVersion {
    Legacy,
    V1,
}

struct GlrmParser {
    tokens: Vec<Tok>,
    pos: usize,
    version: GlrmVersion,
}

#[derive(Debug, Clone)]
struct ParsedSubgrammar {
    name: String,
    body: ParsedSubgrammarBody,
}

#[derive(Debug, Clone)]
enum ParsedSubgrammarBody {
    Inline(ParsedGlrmScope),
    External,
}

#[derive(Debug, Clone)]
struct ParsedGlrmScope {
    rules: Vec<NamedRule>,
    external_terminals: Vec<ParsedExternalTerminal>,
    subgrammars: Vec<ParsedSubgrammar>,
    start: String,
    ignore: Option<String>,
    lexer_partitions: BTreeMap<String, String>,
    lexer_literal_partitions: BTreeMap<Vec<u8>, String>,
    default_lexer_partition: Option<String>,
    all_literals_partition: Option<String>,
}

#[derive(Debug, Clone)]
struct ParsedExternalTerminal {
    name: String,
    token_ids: Vec<u32>,
}

fn collect_external_terminal_names(scope: &ParsedGlrmScope) -> Vec<String> {
    fn collect(scope: &ParsedGlrmScope, prefix: &str, names: &mut Vec<String>) {
        for terminal in &scope.external_terminals {
            names.push(if prefix.is_empty() {
                terminal.name.clone()
            } else {
                format!("{prefix}::{}", terminal.name)
            });
        }
        for subgrammar in &scope.subgrammars {
            if let ParsedSubgrammarBody::Inline(child) = &subgrammar.body {
                let child_prefix = if prefix.is_empty() {
                    subgrammar.name.clone()
                } else {
                    format!("{prefix}::{}", subgrammar.name)
                };
                collect(child, &child_prefix, names);
            }
        }
    }
    let mut names = Vec::new();
    collect(scope, "", &mut names);
    names
}

fn bind_external_terminals(
    scope: &mut ParsedGlrmScope,
    bindings: &[(&str, &[u32])],
) -> Result<(), GlrMaskError> {
    let declared_names = collect_external_terminal_names(scope);
    let declared = declared_names.iter().cloned().collect::<BTreeSet<_>>();
    if declared.len() != declared_names.len() {
        let mut seen = BTreeSet::new();
        let duplicate = declared_names
            .iter()
            .find(|name| !seen.insert((*name).clone()))
            .expect("length mismatch implies a duplicate");
        return Err(err(&format!(
            "external terminal {duplicate:?} is declared more than once"
        )));
    }
    let mut supplied = BTreeMap::<String, Vec<u32>>::new();
    for &(name, token_ids) in bindings {
        if supplied.contains_key(name) {
            return Err(err(&format!(
                "external terminal binding {name:?} was supplied more than once"
            )));
        }
        if token_ids.is_empty() {
            return Err(err(&format!(
                "external terminal binding {name:?} must contain at least one token ID"
            )));
        }
        let unique = token_ids.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() != token_ids.len() {
            return Err(err(&format!(
                "external terminal binding {name:?} contains a duplicate token ID"
            )));
        }
        supplied.insert(name.to_string(), unique.into_iter().collect());
    }
    if let Some(unknown) = supplied.keys().find(|name| !declared.contains(*name)) {
        return Err(err(&format!(
            "exact-token binding was supplied for unknown external terminal {unknown:?}"
        )));
    }
    for name in &declared {
        if !supplied.contains_key(name) {
            return Err(err(&format!(
                "GLRM declares external terminal {name:?}, but no exact-token binding was supplied"
            )));
        }
    }

    fn assign(
        scope: &mut ParsedGlrmScope,
        prefix: &str,
        supplied: &BTreeMap<String, Vec<u32>>,
    ) {
        for terminal in &mut scope.external_terminals {
            let binding_name = if prefix.is_empty() {
                terminal.name.clone()
            } else {
                format!("{prefix}::{}", terminal.name)
            };
            terminal.token_ids = supplied[&binding_name].clone();
        }
        for subgrammar in &mut scope.subgrammars {
            if let ParsedSubgrammarBody::Inline(child) = &mut subgrammar.body {
                let child_prefix = if prefix.is_empty() {
                    subgrammar.name.clone()
                } else {
                    format!("{prefix}::{}", subgrammar.name)
                };
                assign(child, &child_prefix, supplied);
            }
        }
    }
    assign(scope, "", &supplied);
    Ok(())
}

fn external_terminal_rules(scope: &ParsedGlrmScope) -> Vec<NamedRule> {
    scope
        .external_terminals
        .iter()
        .map(|terminal| NamedRule {
            name: terminal.name.clone(),
            expr: if terminal.token_ids.len() == 1 {
                GrammarExpr::SpecialToken(terminal.token_ids[0])
            } else {
                GrammarExpr::Choice(
                    terminal
                        .token_ids
                        .iter()
                        .copied()
                        .map(GrammarExpr::SpecialToken)
                        .collect(),
                )
            },
            // Exact-token terminals participate in parser tables, not the byte lexer.
            is_terminal: false,
            is_internal: false,
        })
        .collect()
}

fn contains_external_subgrammar(scope: &ParsedGlrmScope) -> bool {
    scope.subgrammars.iter().any(|subgrammar| match &subgrammar.body {
        ParsedSubgrammarBody::External => true,
        ParsedSubgrammarBody::Inline(child) => contains_external_subgrammar(child),
    })
}

fn collect_external_subgrammar_names(scope: &ParsedGlrmScope) -> Vec<String> {
    fn collect(scope: &ParsedGlrmScope, prefix: &str, names: &mut Vec<String>) {
        for subgrammar in &scope.subgrammars {
            let binding_name = if prefix.is_empty() {
                subgrammar.name.clone()
            } else {
                format!("{prefix}::{}", subgrammar.name)
            };
            match &subgrammar.body {
                ParsedSubgrammarBody::External => names.push(binding_name),
                ParsedSubgrammarBody::Inline(child) => collect(child, &binding_name, names),
            }
        }
    }

    let mut names = Vec::new();
    collect(scope, "", &mut names);
    names
}

fn collect_expr_special_token_ids(expr: &GrammarExpr, token_ids: &mut BTreeSet<u32>) {
    match expr {
        GrammarExpr::SpecialToken(token_id) => {
            token_ids.insert(*token_id);
        }
        GrammarExpr::Grouped(inner) | GrammarExpr::Quantified(inner, _) => {
            collect_expr_special_token_ids(inner, token_ids);
        }
        GrammarExpr::Sequence(parts) | GrammarExpr::Choice(parts) => {
            for part in parts {
                collect_expr_special_token_ids(part, token_ids);
            }
        }
        GrammarExpr::Exclude { expr, exclude } => {
            collect_expr_special_token_ids(expr, token_ids);
            collect_expr_special_token_ids(exclude, token_ids);
        }
        GrammarExpr::Intersect { expr, intersect } => {
            collect_expr_special_token_ids(expr, token_ids);
            collect_expr_special_token_ids(intersect, token_ids);
        }
        GrammarExpr::SeparatedSequence {
            items, separator, ..
        } => {
            for (item, _) in items {
                collect_expr_special_token_ids(item, token_ids);
            }
            collect_expr_special_token_ids(separator, token_ids);
        }
        GrammarExpr::ExprNFA(expr_nfa) => {
            for symbol in &expr_nfa.symbols {
                collect_expr_special_token_ids(symbol, token_ids);
            }
        }
        GrammarExpr::Ref(_)
        | GrammarExpr::Epsilon
        | GrammarExpr::Literal(_)
        | GrammarExpr::CharClass { .. }
        | GrammarExpr::RawRegex(_)
        | GrammarExpr::LexerDfa(_)
        | GrammarExpr::AnyByte => {}
    }
}

fn collect_scope_special_token_ids(scope: &ParsedGlrmScope, token_ids: &mut BTreeSet<u32>) {
    for rule in &scope.rules {
        collect_expr_special_token_ids(&rule.expr, token_ids);
    }
    for external in &scope.external_terminals {
        token_ids.extend(external.token_ids.iter().copied());
    }
    for subgrammar in &scope.subgrammars {
        if let ParsedSubgrammarBody::Inline(child) = &subgrammar.body {
            collect_scope_special_token_ids(child, token_ids);
        }
    }
}

fn next_free_token_id(
    used_token_ids: &BTreeSet<u32>,
    preferred_start: u32,
) -> Result<u32, GlrMaskError> {
    let mut candidate = preferred_start;
    for &used in used_token_ids.range(preferred_start..) {
        if used < candidate {
            continue;
        }
        if used > candidate {
            return Ok(candidate);
        }
        candidate = candidate.checked_add(1).ok_or_else(|| {
            err("no token ID is available for an external subgrammar placeholder")
        })?;
    }
    Ok(candidate)
}

impl GlrmParser {

    fn new(tokens: Vec<Tok>) -> Result<Self, GlrMaskError> {
        let version = match tokens.as_slice() {
            [Tok::Ident(keyword), Tok::Int(1), Tok::Semi, ..] if keyword == "glrm" => {
                GlrmVersion::V1
            }
            [Tok::Ident(keyword), Tok::Int(version), ..] if keyword == "glrm" => {
                return Err(err(&format!(
                    "unsupported or malformed GLRM version header: glrm {version}; only glrm 1; is supported"
                )));
            }
            _ => GlrmVersion::Legacy,
        };
        Ok(Self { tokens, pos: 0, version })
    }

    fn consume_decl_eq(&mut self) -> Result<(), GlrMaskError> {
        let expected = match self.version {
            GlrmVersion::Legacy => Tok::DeclEq,
            GlrmVersion::V1 => Tok::Equals,
        };
        self.consume(&expected).map_err(|_| {
            err(match self.version {
                GlrmVersion::Legacy => "legacy unversioned GLRM declarations require '::='",
                GlrmVersion::V1 => "GLRM v1 declarations require '='; '::=' is legacy syntax",
            })
        })
    }

    fn token_at(&self, offset: usize) -> &Tok {
        self.tokens.get(self.pos + offset).unwrap_or(&Tok::Eof)
    }

    fn starts_legacy_contextual_fa_body(&self) -> bool {
        if self.version != GlrmVersion::Legacy
            || !matches!(self.token_at(0), Tok::Ident(keyword) if keyword == "fa")
            || !matches!(self.token_at(1), Tok::LBrace)
        {
            return false;
        }
        match self.token_at(2) {
            Tok::RBrace => true,
            Tok::Ident(keyword) if keyword == "start" || keyword == "accept" => true,
            Tok::Int(_) | Tok::Ident(_) => matches!(self.token_at(3), Tok::Dashes | Tok::Arrow),
            _ => false,
        }
    }

    fn peek(&self) -> &Tok {
        self.tokens.get(self.pos).unwrap_or(&Tok::Eof)
    }

    fn advance(&mut self) -> &Tok {
        let tok = self.tokens.get(self.pos).unwrap_or(&Tok::Eof);
        self.pos += 1;
        tok
    }

    fn consume(&mut self, expected: &Tok) -> Result<(), GlrMaskError> {
        let tok = self.advance().clone();
        if &tok == expected {
            Ok(())
        } else {
            Err(err(&format!("expected {:?}, got {:?}", expected, tok)))
        }
    }

    fn expect_ident(&mut self) -> Result<String, GlrMaskError> {
        match self.advance().clone() {
            Tok::Ident(s) => Ok(s),
            other => Err(err(&format!("expected identifier, got {:?}", other))),
        }
    }

    fn expect_int(&mut self) -> Result<usize, GlrMaskError> {
        match self.advance().clone() {
            Tok::Int(n) => Ok(n),
            other => Err(err(&format!("expected integer, got {:?}", other))),
        }
    }

    fn parse_root_scope(&mut self) -> Result<ParsedGlrmScope, GlrMaskError> {
        if self.version == GlrmVersion::V1 {
            self.advance();
            self.advance();
            self.consume(&Tok::Semi)?;
        }
        self.parse_scope(false, "grammar")
    }

    fn parse_scope(
        &mut self,
        stop_at_rbrace: bool,
        scope_label: &str,
    ) -> Result<ParsedGlrmScope, GlrMaskError> {
        let mut start: Option<String> = None;
        let mut ignore: Option<String> = None;
        let mut rules: Vec<NamedRule> = Vec::new();
        let mut external_terminals = Vec::new();
        let mut subgrammars = Vec::new();
        let mut lexer_partitions = BTreeMap::<String, String>::new();
        let mut lexer_literal_partitions = BTreeMap::<Vec<u8>, String>::new();
        let mut default_lexer_partition = None::<String>;
        let mut all_literals_partition = None::<String>;

        loop {
            match self.peek().clone() {
                Tok::Eof if stop_at_rbrace => {
                    return Err(err(&format!("unterminated {scope_label}")));
                }
                Tok::Eof => break,
                Tok::RBrace if stop_at_rbrace => {
                    self.advance();
                    break;
                }
                Tok::RBrace => {
                    return Err(err("unexpected '}' at top level"));
                }
                Tok::Ident(ref kw) => match kw.as_str() {
                    "start" => {
                        self.advance();
                        let name = self.expect_ident()?;
                        self.consume(&Tok::Semi)?;
                        if start.replace(name).is_some() {
                            return Err(err(&format!(
                                "{scope_label} has more than one 'start' declaration",
                            )));
                        }
                    }
                    "ignore" => {
                        self.advance();
                        let name = self.expect_ident()?;
                        self.consume(&Tok::Semi)?;
                        if ignore.replace(name).is_some() {
                            return Err(err(&format!(
                                "{scope_label} has more than one 'ignore' declaration",
                            )));
                        }
                    }
                    "g" | "subgrammar" => {
                        self.advance();
                        let name = self.expect_ident()?;
                        self.consume_decl_eq()?;
                        self.consume(&Tok::LBrace)?;
                        let child_label = format!("subgrammar '{name}'");
                        let scope = self.parse_scope(true, &child_label)?;
                        self.consume(&Tok::Semi)?;
                        subgrammars.push(ParsedSubgrammar {
                            name,
                            body: ParsedSubgrammarBody::Inline(scope),
                        });
                    }
                    "extern" => {
                        self.advance();
                        match self.advance().clone() {
                            Tok::Ident(ref kind) if kind == "token" => {
                                if self.version != GlrmVersion::V1 {
                                    return Err(err("extern token declarations require a leading 'glrm 1;' header"));
                                }
                                let name = self.expect_ident()?;
                                self.consume(&Tok::Semi)?;
                                external_terminals.push(ParsedExternalTerminal {
                                    name,
                                    token_ids: Vec::new(),
                                });
                                continue;
                            }
                            Tok::Ident(ref kind) if kind == "grammar" => {}
                            other => {
                                return Err(err(&format!(
                                    "expected 'token' or 'grammar' after 'extern', got {:?}",
                                    other,
                                )));
                            }
                        }
                        let name = self.expect_ident()?;
                        self.consume(&Tok::Semi)?;
                        subgrammars.push(ParsedSubgrammar {
                            name,
                            body: ParsedSubgrammarBody::External,
                        });
                    }
                    "lexer" => {
                        if self.version == GlrmVersion::V1 {
                            return Err(err("GLRM v1 lexer groups must be inside 'pragma glrmask { ... }'"));
                        }
                        self.advance();
                        self.parse_lexer_group(
                            &mut lexer_partitions,
                            &mut lexer_literal_partitions,
                            &mut default_lexer_partition,
                            &mut all_literals_partition,
                        )?;
                    }
                    "nt" => {
                        self.advance();
                        rules.push(self.parse_rule(false, false)?);
                    }
                    "fa" | "nfa" => {
                        if self.version == GlrmVersion::V1 {
                            return Err(err("GLRM v1 FA syntax is a declaration body, for example 'nt name = fa { ... };'"));
                        }
                        self.advance();
                        rules.push(self.parse_expr_nfa_rule()?);
                    }
                    "t" => {
                        self.advance();
                        rules.push(self.parse_rule(true, false)?);
                    }
                    "internal" => {
                        self.advance();
                        // Expect `t`
                        match self.advance().clone() {
                            Tok::Ident(ref kw2) if kw2 == "t" => {}
                            other => return Err(err(&format!("expected 't' after 'internal', got {:?}", other))),
                        }
                        rules.push(self.parse_rule(true, true)?);
                    }
                    "pragma" if self.version == GlrmVersion::V1 => {
                        self.advance();
                        let namespace = self.expect_ident()?;
                        if namespace != "glrmask" {
                            return Err(err(&format!("unsupported GLRM v1 pragma namespace {namespace:?}")));
                        }
                        self.consume(&Tok::LBrace)?;
                        while !matches!(self.peek(), Tok::RBrace) {
                            match self.advance().clone() {
                                Tok::Ident(keyword) if keyword == "lexer" => self.parse_lexer_group(
                                    &mut lexer_partitions,
                                    &mut lexer_literal_partitions,
                                    &mut default_lexer_partition,
                                    &mut all_literals_partition,
                                )?,
                                other => return Err(err(&format!("unexpected token {other:?} in pragma glrmask"))),
                            }
                        }
                        self.consume(&Tok::RBrace)?;
                    }
                    other => {
                        return Err(err(&format!(
                            "unexpected keyword '{other}' in {scope_label}",
                        )));
                    }
                },
                other => {
                    return Err(err(&format!(
                        "unexpected token {:?} in {scope_label}",
                        other,
                    )));
                }
            }
        }

        let start = start.ok_or_else(|| err(&format!("{scope_label} has no 'start' declaration")))?;
        Ok(ParsedGlrmScope {
            rules,
            external_terminals,
            subgrammars,
            start,
            ignore,
            lexer_partitions,
            lexer_literal_partitions,
            default_lexer_partition,
            all_literals_partition,
        })
    }

    fn parse_lexer_group(
        &mut self,
        lexer_partitions: &mut BTreeMap<String, String>,
        lexer_literal_partitions: &mut BTreeMap<Vec<u8>, String>,
        default_lexer_partition: &mut Option<String>,
        all_literals_partition: &mut Option<String>,
    ) -> Result<(), GlrMaskError> {
        match self.advance().clone() {
            Tok::Ident(ref keyword) if keyword == "group" => {}
            other => {
                return Err(err(&format!(
                    "expected 'group' after 'lexer', got {:?}",
                    other,
                )));
            }
        }
        let partition = self.expect_ident()?;
        self.consume_decl_eq()?;
        loop {
            match self.advance().clone() {
                Tok::Ident(terminal) => {
                    if lexer_partitions
                        .insert(terminal.clone(), partition.clone())
                        .is_some()
                    {
                        return Err(err(&format!(
                            "terminal '{terminal}' is assigned to more than one lexer group",
                        )));
                    }
                }
                Tok::StringLit(literal) => {
                    if lexer_literal_partitions
                        .insert(literal.clone(), partition.clone())
                        .is_some()
                    {
                        return Err(err(&format!(
                            "literal {:?} is assigned to more than one lexer group",
                            String::from_utf8_lossy(&literal),
                        )));
                    }
                }
                Tok::Star => {
                    if default_lexer_partition
                        .replace(partition.clone())
                        .is_some()
                    {
                        return Err(err(
                            "the catch-all '*' is assigned to more than one lexer group",
                        ));
                    }
                }
                Tok::At => match self.advance().clone() {
                    Tok::Ident(selector) if selector == "literals" => {
                        if all_literals_partition
                            .replace(partition.clone())
                            .is_some()
                        {
                            return Err(err(
                                "'@literals' is assigned to more than one lexer group",
                            ));
                        }
                    }
                    other => {
                        return Err(err(&format!(
                            "expected 'literals' after '@', got {:?}",
                            other,
                        )));
                    }
                },
                other => {
                    return Err(err(&format!(
                        "expected terminal name, literal, or '*', got {:?}",
                        other,
                    )));
                }
            }
            if matches!(self.peek(), Tok::Comma) {
                self.advance();
                continue;
            }
            break;
        }
        self.consume(&Tok::Semi)
    }

    fn parse_rule(&mut self, is_terminal: bool, is_internal: bool) -> Result<NamedRule, GlrMaskError> {
        let name = self.expect_ident()?;
        self.consume_decl_eq()?;
        let expr = if self.version == GlrmVersion::V1
            && matches!(self.peek(), Tok::Ident(keyword) if keyword == "fa")
        {
            self.advance();
            GrammarExpr::ExprNFA(Box::new(self.parse_v1_expr_nfa_body(is_terminal)?))
        } else if self.starts_legacy_contextual_fa_body() {
            self.advance();
            GrammarExpr::ExprNFA(Box::new(self.parse_legacy_expr_nfa_body(is_terminal)?))
        } else {
            if self.version == GlrmVersion::V1 && !self.can_start_nt_atom() {
                return Err(err("GLRM v1 declaration bodies cannot be empty; use 'eps' explicitly"));
            }
            self.parse_nt_expr(is_terminal)?
        };
        self.consume(&Tok::Semi)?;
        Ok(NamedRule { name, expr, is_terminal, is_internal })
    }

    fn parse_v1_expr_nfa_body(&mut self, allow_raw_regex: bool) -> Result<ExprNFA, GlrMaskError> {
        self.consume(&Tok::LBrace)?;
        let mut nfa = NFA::new_empty();
        let mut state_ids = HashMap::<String, u32>::new();
        let mut state_names = Vec::<String>::new();
        let mut symbols = Vec::<GrammarExpr>::new();
        let mut symbol_labels = HashMap::<GrammarExpr, Label>::new();

        fn state(
            parser: &mut GlrmParser,
            nfa: &mut NFA,
            state_ids: &mut HashMap<String, u32>,
            state_names: &mut Vec<String>,
        ) -> Result<u32, GlrMaskError> {
            let (key, display) = match parser.advance().clone() {
                Tok::Int(value) => (format!("i:{value}"), value.to_string()),
                Tok::Ident(name) => (format!("n:{name}"), name),
                other => return Err(err(&format!("expected FA state name or integer, got {other:?}"))),
            };
            if let Some(&id) = state_ids.get(&key) {
                return Ok(id);
            }
            let id = state_names.len() as u32;
            nfa.add_state();
            state_ids.insert(key, id);
            state_names.push(display);
            Ok(id)
        }

        while !matches!(self.peek(), Tok::RBrace) {
            match self.peek().clone() {
                Tok::Eof => return Err(err("unterminated GLRM v1 FA body")),
                Tok::Ident(ref keyword)
                    if keyword == "start" && !matches!(self.token_at(1), Tok::ThinArrow) =>
                {
                    self.advance();
                    nfa.start_states.clear();
                    loop {
                        let id = state(self, &mut nfa, &mut state_ids, &mut state_names)?;
                        if !nfa.start_states.contains(&id) {
                            nfa.start_states.push(id);
                        }
                        if matches!(self.peek(), Tok::Comma) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    self.consume(&Tok::Semi)?;
                }
                Tok::Ident(ref keyword)
                    if keyword == "accept" && !matches!(self.token_at(1), Tok::ThinArrow) =>
                {
                    self.advance();
                    if matches!(self.peek(), Tok::Semi) {
                        return Err(err("GLRM v1 FA must declare at least one accept state"));
                    }
                    loop {
                        let id = state(self, &mut nfa, &mut state_ids, &mut state_names)?;
                        nfa.set_accepting(id);
                        if matches!(self.peek(), Tok::Comma) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    self.consume(&Tok::Semi)?;
                }
                Tok::Int(_) | Tok::Ident(_) => {
                    let from = state(self, &mut nfa, &mut state_ids, &mut state_names)?;
                    self.consume(&Tok::ThinArrow)?;
                    let to = state(self, &mut nfa, &mut state_ids, &mut state_names)?;
                    self.consume(&Tok::Colon)?;
                    if !self.can_start_nt_atom() {
                        return Err(err("GLRM v1 FA edge expression cannot be empty; use 'eps'"));
                    }
                    let symbol = self.parse_nt_expr(allow_raw_regex)?;
                    if matches!(symbol, GrammarExpr::Epsilon) {
                        nfa.add_epsilon(from, to);
                    } else {
                        let label = intern_expr_nfa_symbol(&mut symbols, &mut symbol_labels, symbol);
                        nfa.add_transition(from, label, to);
                    }
                    self.consume(&Tok::Semi)?;
                }
                other => return Err(err(&format!("unexpected token {other:?} in GLRM v1 FA body"))),
            }
        }
        self.consume(&Tok::RBrace)?;
        if nfa.start_states.is_empty() {
            return Err(err("FA definition has no start state"));
        }
        if !nfa.states.iter().any(|state| state.is_accepting) {
            return Err(err("FA definition has no accept state"));
        }
        Ok(ExprNFA::new(nfa, symbols).with_state_names(state_names))
    }

    fn parse_expr_nfa_rule(&mut self) -> Result<NamedRule, GlrMaskError> {
        let name = self.expect_ident()?;
        self.consume_decl_eq()?;
        let expr_nfa = self.parse_legacy_expr_nfa_body(false)?;
        self.consume(&Tok::Semi)?;
        Ok(NamedRule {
            name,
            expr: GrammarExpr::ExprNFA(Box::new(expr_nfa)),
            is_terminal: false,
            is_internal: false,
        })
    }

    fn parse_legacy_expr_nfa_body(
        &mut self,
        allow_raw_regex: bool,
    ) -> Result<ExprNFA, GlrMaskError> {
        self.consume(&Tok::LBrace)?;
        let mut nfa = NFA::new_empty();
        let mut state_ids = HashMap::<String, u32>::new();
        let mut state_names = Vec::<String>::new();
        let mut symbols = Vec::<GrammarExpr>::new();
        let mut symbol_labels = HashMap::<GrammarExpr, Label>::new();

        fn state(
            parser: &mut GlrmParser,
            nfa: &mut NFA,
            state_ids: &mut HashMap<String, u32>,
            state_names: &mut Vec<String>,
        ) -> Result<u32, GlrMaskError> {
            let (key, display) = match parser.advance().clone() {
                Tok::Int(value) => (format!("i:{value}"), value.to_string()),
                Tok::Ident(name) => (format!("n:{name}"), name),
                other => return Err(err(&format!("expected FA state name or integer, got {other:?}"))),
            };
            if let Some(&id) = state_ids.get(&key) {
                return Ok(id);
            }
            let id = nfa.add_state();
            state_ids.insert(key, id);
            state_names.push(display);
            Ok(id)
        }

        loop {
            match self.peek().clone() {
                Tok::RBrace => {
                    self.advance();
                    break;
                }
                Tok::Ident(ref keyword)
                    if keyword == "start" && !matches!(self.token_at(1), Tok::Dashes | Tok::Arrow) =>
                {
                    self.advance();
                    nfa.start_states.clear();
                    loop {
                        let id = state(self, &mut nfa, &mut state_ids, &mut state_names)?;
                        if !nfa.start_states.contains(&id) {
                            nfa.start_states.push(id);
                        }
                        if matches!(self.peek(), Tok::Comma) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    self.consume(&Tok::Semi)?;
                }
                Tok::Ident(ref keyword)
                    if keyword == "accept" && !matches!(self.token_at(1), Tok::Dashes | Tok::Arrow) =>
                {
                    self.advance();
                    if !matches!(self.peek(), Tok::Semi) {
                        loop {
                            let id = state(self, &mut nfa, &mut state_ids, &mut state_names)?;
                            nfa.set_accepting(id);
                            if matches!(self.peek(), Tok::Comma) {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                    }
                    self.consume(&Tok::Semi)?;
                }
                Tok::Int(_) | Tok::Ident(_) => {
                    let from = state(self, &mut nfa, &mut state_ids, &mut state_names)?;
                    match self.peek() {
                        Tok::Arrow => {
                            self.advance();
                            let to = state(self, &mut nfa, &mut state_ids, &mut state_names)?;
                            nfa.add_epsilon(from, to);
                        }
                        Tok::Dashes => {
                            self.advance();
                            let symbol = self.parse_expr_nfa_transition_expr(allow_raw_regex)?;
                            if matches!(symbol, GrammarExpr::Epsilon) {
                                return Err(err(
                                    "an epsilon FA transition is written `source --> target`, not `source -- eps --> target`",
                                ));
                            }
                            self.consume(&Tok::Arrow)?;
                            let to = state(self, &mut nfa, &mut state_ids, &mut state_names)?;
                            let label = intern_expr_nfa_symbol(
                                &mut symbols,
                                &mut symbol_labels,
                                symbol,
                            );
                            nfa.add_transition(from, label, to);
                        }
                        other => {
                            return Err(err(&format!(
                                "expected '--' or '-->' after FA transition source, got {other:?}"
                            )));
                        }
                    }
                    self.consume(&Tok::Semi)?;
                }
                other => return Err(err(&format!("unexpected token {other:?} in FA definition"))),
            }
        }
        if nfa.start_states.is_empty() {
            return Err(err("FA definition has no start state"));
        }
        Ok(ExprNFA::new(nfa, symbols).with_state_names(state_names))
    }

    fn parse_expr_nfa_transition_expr(
        &mut self,
        allow_raw_regex: bool,
    ) -> Result<GrammarExpr, GlrMaskError> {
        if matches!(self.peek(), Tok::Arrow) {
            return Err(err(
                "FA transition expression cannot be empty; use epsilon transition syntax",
            ));
        }
        if !self.can_start_nt_atom() {
            return Err(err(&format!(
                "expected FA transition expression item before '-->', got {:?}",
                self.peek()
            )));
        }
        self.parse_nt_expr(allow_raw_regex)
    }

    // ---- NT expression parsing ---------------------------------------------

    fn parse_nt_expr(&mut self, allow_raw_regex: bool) -> Result<GrammarExpr, GlrMaskError> {
        if self.version == GlrmVersion::V1 && !self.can_start_nt_atom() {
            return Err(err("GLRM v1 expressions cannot contain empty alternatives or groups; use 'eps' explicitly"));
        }
        // Legacy GLRM permits an implicit empty alternative for compatibility.
        let first = if self.can_start_nt_atom() {
            self.parse_nt_exclude(allow_raw_regex)?
        } else {
            GrammarExpr::Sequence(vec![]) // empty / ε
        };
        if !matches!(self.peek(), Tok::Pipe) {
            return Ok(first);
        }
        let mut alts = vec![first];
        while matches!(self.peek(), Tok::Pipe) {
            self.advance(); // consume `|`
            if self.version == GlrmVersion::V1 && !self.can_start_nt_atom() {
                return Err(err("GLRM v1 expressions cannot contain empty alternatives; use 'eps' explicitly"));
            }
            let alt = if self.can_start_nt_atom() {
                self.parse_nt_exclude(allow_raw_regex)?
            } else {
                GrammarExpr::Sequence(vec![]) // empty / ε
            };
            alts.push(alt);
        }
        Ok(GrammarExpr::Choice(alts))
    }

    fn parse_nt_exclude(&mut self, allow_raw_regex: bool) -> Result<GrammarExpr, GlrMaskError> {
        let expr = self.parse_nt_intersect(allow_raw_regex)?;
        let mut excludes = Vec::new();
        while matches!(self.peek(), Tok::Minus) {
            self.advance();
            excludes.push(self.parse_nt_exclude_rhs(allow_raw_regex)?);
        }
        if excludes.is_empty() {
            Ok(expr)
        } else {
            let exclude_expr = if excludes.len() == 1 {
                excludes.into_iter().next().unwrap()
            } else {
                GrammarExpr::Choice(excludes)
            };
            Ok(GrammarExpr::Exclude {
                expr: Box::new(expr),
                exclude: Box::new(exclude_expr),
            })
        }
    }

    fn parse_nt_intersect(&mut self, allow_raw_regex: bool) -> Result<GrammarExpr, GlrMaskError> {
        let mut expr = self.parse_nt_seq(allow_raw_regex)?;
        while matches!(self.peek(), Tok::Amp) {
            self.advance();
            let rhs = self.parse_nt_seq(allow_raw_regex)?;
            expr = GrammarExpr::Intersect {
                expr: Box::new(expr),
                intersect: Box::new(rhs),
            };
        }
        Ok(expr)
    }

    fn parse_nt_exclude_rhs(&mut self, allow_raw_regex: bool) -> Result<GrammarExpr, GlrMaskError> {
        if matches!(self.peek(), Tok::LParen) {
            self.advance();
            let inner = self.parse_nt_expr(allow_raw_regex)?;
            self.consume(&Tok::RParen)?;
            return Ok(GrammarExpr::Grouped(Box::new(inner)));
        }

        let mut expr = self.parse_nt_postfix(allow_raw_regex)?;
        while matches!(self.peek(), Tok::Amp) {
            self.advance();
            let rhs = self.parse_nt_postfix(allow_raw_regex)?;
            expr = GrammarExpr::Intersect {
                expr: Box::new(expr),
                intersect: Box::new(rhs),
            };
        }
        if self.can_start_nt_atom() {
            return Err(err("invalid syntax: RHS sequence subtraction must be parenthesized"));
        }
        Ok(expr)
    }

    fn parse_nt_seq(&mut self, allow_raw_regex: bool) -> Result<GrammarExpr, GlrMaskError> {
        let mut items = Vec::new();
        loop {
            // A new item can start with: Ident, StringLit, RegexLit, CharClass,
            // Dot, LParen. Otherwise stop.
            if !self.can_start_nt_atom() {
                break;
            }
            items.push(self.parse_nt_postfix(allow_raw_regex)?);
        }
        match items.len() {
            0 => Err(err("expected at least one expression item")),
            1 => Ok(items.pop().unwrap()),
            _ => Ok(GrammarExpr::Sequence(items)),
        }
    }

    fn can_start_nt_atom(&self) -> bool {
        match self.peek() {
            Tok::Ident(s) if s == "eps" => true,
            Tok::Ident(_) | Tok::StringLit(_) | Tok::RegexLit(_)
            | Tok::CharClass(_, _) | Tok::Dot | Tok::LParen | Tok::At => true,
            _ => false,
        }
    }

    fn parse_nt_postfix(&mut self, allow_raw_regex: bool) -> Result<GrammarExpr, GlrMaskError> {
        let atom = self.parse_nt_atom(allow_raw_regex)?;
        // `sep ~ ( items... )` / `sep ~+ ( items... )` — SeparatedSequence
        if matches!(self.peek(), Tok::Tilde) {
            self.advance(); // consume `~`
            let allow_empty = if matches!(self.peek(), Tok::Plus) {
                self.advance();
                false
            } else {
                true
            };
            self.consume(&Tok::LParen)?;
            let mut items = Vec::new();
            loop {
                let item = self.parse_sepseq_item(allow_raw_regex)?;
                items.push(item);
                if self.can_start_nt_atom() {
                    // more items
                } else {
                    break;
                }
            }
            self.consume(&Tok::RParen)?;
            if matches!(self.peek(), Tok::Quest | Tok::Star | Tok::Plus | Tok::LBrace) {
                return Err(err(
                    "quantifiers cannot be applied directly to SeparatedSequence; wrap it in a named rule instead",
                ));
            }
            return Ok(GrammarExpr::SeparatedSequence {
                items,
                separator: Box::new(atom),
                allow_empty,
            });
        }
        self.apply_nt_quantifier(atom)
    }

    fn parse_sepseq_item(
        &mut self,
        allow_raw_regex: bool,
    ) -> Result<(GrammarExpr, Option<Quantifier>), GlrMaskError> {
        let atom = match self.peek() {
            Tok::LParen => {
                self.advance();
                let inner = self.parse_nt_expr(allow_raw_regex)?;
                self.consume(&Tok::RParen)?;
                GrammarExpr::Grouped(Box::new(inner))
            }
            _ => self.parse_nt_atom(allow_raw_regex)?,
        };

        let quantifier = match self.peek() {
            Tok::Quest => {
                self.advance();
                Some(Quantifier::Optional)
            }
            Tok::Star => {
                self.advance();
                Some(Quantifier::ZeroPlus)
            }
            Tok::Plus => {
                self.advance();
                Some(Quantifier::OnePlus)
            }
            Tok::LBrace => {
                self.advance();
                let min = self.expect_int()?;
                let max = if matches!(self.peek(), Tok::Comma) {
                    self.advance();
                    if matches!(self.peek(), Tok::RBrace) {
                        None
                    } else {
                        Some(self.expect_int()?)
                    }
                } else {
                    Some(min)
                };
                self.consume(&Tok::RBrace)?;
                if let Some(max) = max
                    && max < min
                {
                    return Err(err(&format!(
                        "repetition upper bound {max} is smaller than lower bound {min}"
                    )));
                }
                Some(Quantifier::Range(min, max))
            }
            _ => None,
        };

        if quantifier.is_some() && matches!(self.peek(), Tok::Quest | Tok::Star | Tok::Plus | Tok::LBrace) {
            return Err(err(
                "only one postfix quantifier may bind to a separated-sequence item; use extra parentheses, e.g. `sep ~ ( (expr+)? )`, when the outer quantifier should bind to the separator",
            ));
        }

        Ok((atom, quantifier))
    }

    fn apply_nt_quantifier(&mut self, atom: GrammarExpr) -> Result<GrammarExpr, GlrMaskError> {
        match self.peek() {
            Tok::Quest => { self.advance(); Ok(GrammarExpr::Quantified(Box::new(atom), Quantifier::Optional)) }
            Tok::Star  => { self.advance(); Ok(GrammarExpr::Quantified(Box::new(atom), Quantifier::ZeroPlus)) }
            Tok::Plus  => { self.advance(); Ok(GrammarExpr::Quantified(Box::new(atom), Quantifier::OnePlus)) }
            Tok::LBrace => {
                self.advance(); // `{`
                let min = self.expect_int()?;
                let max = if matches!(self.peek(), Tok::Comma) {
                    self.advance(); // `,`
                    if matches!(self.peek(), Tok::RBrace) {
                        // `{n,}` — unbounded; not representable in RepeatRange so emit Repeat
                        self.consume(&Tok::RBrace)?;
                        // Minimum n, no max — we use RepeatRange with min=n, max=n for
                        // compatibility but actually there's no max. We'll use n as both.
                        // To keep it correct, use a large max sentinel via Optional+RepeatOne.
                        // Actually just use RepeatRange with min==max as fallback:
                        // Better: reject or use a very large max. For now: min.
                        return Ok(GrammarExpr::Quantified(Box::new(atom), Quantifier::Range(min, None)));
                    }
                    self.expect_int()?
                } else {
                    min
                };
                self.consume(&Tok::RBrace)?;
                if max < min {
                    return Err(err(&format!(
                        "repetition upper bound {max} is smaller than lower bound {min}"
                    )));
                }
                Ok(GrammarExpr::Quantified(Box::new(atom), Quantifier::Range(min, Some(max))))
            }
            _ => Ok(atom),
        }
    }

    fn parse_nt_atom(&mut self, allow_raw_regex: bool) -> Result<GrammarExpr, GlrMaskError> {
        match self.peek().clone() {
            Tok::StringLit(bytes) => {
                self.advance();
                Ok(GrammarExpr::Literal(bytes))
            }
            Tok::RegexLit(pat) => {
                if !allow_raw_regex {
                    return Err(err("raw regex literals are only allowed in terminal (`t`) rules"));
                }
                if self.version == GlrmVersion::V1 {
                    validate_regular_regex(&pat)
                        .map_err(|message| err(&format!("invalid GLRM v1 regex: {message}")))?;
                }
                self.advance();
                Ok(GrammarExpr::RawRegex(pat))
            }
            Tok::CharClass(def, is_utf8) => {
                self.advance();
                let negate = def.starts_with('^');
                let def_clean = if negate { def[1..].to_string() } else { def };
                Ok(GrammarExpr::CharClass { def: def_clean, negate, utf8: is_utf8 })
            }
            Tok::Dot => {
                self.advance();
                Ok(GrammarExpr::AnyByte)
            }
            Tok::At => {
                if self.version == GlrmVersion::V1 {
                    return Err(err("GLRM v1 does not expose numeric @token(id) syntax; declare 'extern token NAME;' and bind it by name"));
                }
                self.advance();
                match self.advance().clone() {
                    Tok::Ident(keyword) if keyword == "token" => {}
                    other => {
                        return Err(err(&format!(
                            "expected 'token' after '@' in expression, got {:?}",
                            other
                        )));
                    }
                }
                self.consume(&Tok::LParen)?;
                let token_id = self.expect_int()?;
                let token_id = u32::try_from(token_id)
                    .map_err(|_| err("special LLM token id does not fit in u32"))?;
                self.consume(&Tok::RParen)?;
                Ok(GrammarExpr::SpecialToken(token_id))
            }
            Tok::LParen => {
                self.advance();
                let inner = self.parse_nt_expr(allow_raw_regex)?;
                self.consume(&Tok::RParen)?;
                Ok(inner)
            }
            Tok::Ident(ref kw) => {
                match kw.as_str() {
                    "eps" => {
                        self.advance();
                        Ok(GrammarExpr::Epsilon)
                    }
                    _ => {
                        // Generic identifier = Ref
                        let name = kw.clone();
                        self.advance();
                        Ok(GrammarExpr::Ref(name))
                    }
                }
            }
            other => Err(err(&format!("unexpected token {:?} in NT expression", other))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScopedSymbolKind {
    Nonterminal,
    Terminal,
    InternalTerminal,
    ExternalTerminal,
    Subgrammar,
}

fn named_grammar_for_scope(scope: &ParsedGlrmScope) -> Result<NamedGrammar, GlrMaskError> {
    let mut rules = scope.rules.clone();
    rules.extend(external_terminal_rules(scope));
    let mut grammar = NamedGrammar {
        rules,
        start: scope.start.clone(),
        ignore: scope.ignore.clone(),
        lexer_partitions: scope.lexer_partitions.clone(),
        lexer_literal_partitions: scope.lexer_literal_partitions.clone(),
        default_lexer_partition: scope.default_lexer_partition.clone(),
    };
    if let Some(partition) = scope.all_literals_partition.as_deref() {
        for literal in grammar.emitted_anonymous_literals() {
            grammar
                .lexer_literal_partitions
                .entry(literal)
                .or_insert_with(|| partition.to_string());
        }
    }
    Ok(grammar)
}

fn scope_symbol_kinds(
    scope: &ParsedGlrmScope,
    scope_label: &str,
) -> Result<BTreeMap<String, ScopedSymbolKind>, GlrMaskError> {
    let mut symbols = BTreeMap::new();
    for rule in &scope.rules {
        let kind = match (rule.is_terminal, rule.is_internal) {
            (false, _) => ScopedSymbolKind::Nonterminal,
            (true, false) => ScopedSymbolKind::Terminal,
            (true, true) => ScopedSymbolKind::InternalTerminal,
        };
        if let Some(previous) = symbols.insert(rule.name.clone(), kind)
            && previous != kind
        {
            return Err(err(&format!(
                "definition '{}' has conflicting kinds in {scope_label}",
                rule.name,
            )));
        }
    }
    for terminal in &scope.external_terminals {
        if symbols
            .insert(terminal.name.clone(), ScopedSymbolKind::ExternalTerminal)
            .is_some()
        {
            return Err(err(&format!(
                "duplicate definition '{}' in {scope_label}",
                terminal.name
            )));
        }
    }
    for subgrammar in &scope.subgrammars {
        if symbols
            .insert(subgrammar.name.clone(), ScopedSymbolKind::Subgrammar)
            .is_some()
        {
            return Err(err(&format!(
                "subgrammar '{}' conflicts with another definition in {scope_label}",
                subgrammar.name,
            )));
        }
    }
    for rule in scope.rules.iter().filter(|rule| rule.is_terminal) {
        for external in &scope.external_terminals {
            if grammar_expr_contains_ref(&rule.expr, &external.name) {
                return Err(err(&format!(
                    "external terminal '{}' cannot be referenced from terminal body '{}' in {scope_label}",
                    external.name, rule.name
                )));
            }
        }
    }
    let Some(start_kind) = symbols.get(&scope.start).copied() else {
        return Err(err(&format!(
            "start '{}' is not defined in {scope_label}; definitions are scope-local",
            scope.start,
        )));
    };
    if start_kind == ScopedSymbolKind::InternalTerminal {
        return Err(err(&format!(
            "start '{}' in {scope_label} is an internal terminal",
            scope.start,
        )));
    }
    Ok(symbols)
}

fn validate_ignore_terminal(grammar: &NamedGrammar, scope_label: &str) -> Result<(), GlrMaskError> {
    let Some(ignore_name) = grammar.ignore.as_deref() else {
        return Ok(());
    };
    let Some(rule) = grammar
        .rules
        .iter()
        .find(|rule| rule.name == ignore_name)
    else {
        return Err(err(&format!(
            "ignore terminal '{ignore_name}' is not defined in {scope_label}; definitions are scope-local",
        )));
    };
    if !rule.is_terminal || rule.is_internal {
        return Err(err(&format!(
            "ignore '{ignore_name}' in {scope_label} must name a local emitting terminal",
        )));
    }
    if matches!(rule.expr, GrammarExpr::SpecialToken(_)) {
        return Err(err(
            "a special LLM token terminal cannot be the ignore terminal",
        ));
    }
    let resolved = resolved_named_terminal_exprs(grammar)?;
    let expr = resolved.get(ignore_name).ok_or_else(|| {
        err(&format!(
            "ignore terminal '{ignore_name}' could not be resolved in {scope_label}",
        ))
    })?;
    if expr.is_nullable() {
        return Err(err(&format!(
            "ignore terminal '{ignore_name}' in {scope_label} must consume at least one byte",
        )));
    }
    for rule in grammar.rules.iter().filter(|rule| !rule.is_terminal) {
        if grammar_expr_contains_ref(&rule.expr, ignore_name) {
            return Err(err(&format!(
                "ignore terminal '{ignore_name}' is referenced explicitly by parser rule '{}' in {scope_label}; ignored terminals are implicit at grammar boundaries and between lexical atoms",
                rule.name,
            )));
        }
    }
    Ok(())
}

fn grammar_expr_contains_ref(expr: &GrammarExpr, name: &str) -> bool {
    match expr {
        GrammarExpr::Ref(reference) => reference == name,
        GrammarExpr::Grouped(inner) | GrammarExpr::Quantified(inner, _) => {
            grammar_expr_contains_ref(inner, name)
        }
        GrammarExpr::Sequence(parts) | GrammarExpr::Choice(parts) => {
            parts.iter().any(|part| grammar_expr_contains_ref(part, name))
        }
        GrammarExpr::Exclude { expr, exclude } => {
            grammar_expr_contains_ref(expr, name) || grammar_expr_contains_ref(exclude, name)
        }
        GrammarExpr::Intersect { expr, intersect } => {
            grammar_expr_contains_ref(expr, name) || grammar_expr_contains_ref(intersect, name)
        }
        GrammarExpr::SeparatedSequence { items, separator, .. } => {
            items
                .iter()
                .any(|(item, _)| grammar_expr_contains_ref(item, name))
                || grammar_expr_contains_ref(separator, name)
        }
        GrammarExpr::ExprNFA(expr_nfa) => expr_nfa
            .symbols
            .iter()
            .any(|symbol| grammar_expr_contains_ref(symbol, name)),
        GrammarExpr::Epsilon
        | GrammarExpr::Literal(_)
        | GrammarExpr::SpecialToken(_)
        | GrammarExpr::CharClass { .. }
        | GrammarExpr::RawRegex(_)
        | GrammarExpr::LexerDfa(_)
        | GrammarExpr::AnyByte => false,
    }
}

struct FlattenContext {
    next_scope_id: usize,
    used_names: HashSet<String>,
    used_partition_names: HashSet<String>,
    rules: Vec<NamedRule>,
    lexer_partitions: BTreeMap<String, String>,
    lexer_literal_partition_constraints: Vec<(Vec<u8>, String)>,
    external_placeholders: Vec<ExternalSubgrammarPlaceholder>,
}

impl FlattenContext {
    fn fresh_global_name(&mut self, base: String) -> String {
        if self.used_names.insert(base.clone()) {
            return base;
        }
        for suffix in 1usize.. {
            let candidate = format!("{base}_{suffix}");
            if self.used_names.insert(candidate.clone()) {
                return candidate;
            }
        }
        unreachable!()
    }

    fn reserve_exact_name(&mut self, name: &str) -> Result<String, GlrMaskError> {
        if !self.used_names.insert(name.to_string()) {
            return Err(err(&format!("duplicate top-level definition '{name}'")));
        }
        Ok(name.to_string())
    }

    fn fresh_partition_name(&mut self, base: String) -> String {
        if self.used_partition_names.insert(base.clone()) {
            return base;
        }
        for suffix in 1usize.. {
            let candidate = format!("{base}_{suffix}");
            if self.used_partition_names.insert(candidate.clone()) {
                return candidate;
            }
        }
        unreachable!()
    }
}

/// Return true when every scope declares an ignore terminal with the same
/// fully resolved terminal expression.
///
/// In that case scope-local ignore semantics are exactly equivalent to one
/// global ignore terminal, so flattening can retain the compiler's original
/// terminal-DWA ignore path instead of materialising skip productions in the
/// GLR grammar. This is deliberately conservative: syntactically different
/// expressions that happen to denote the same byte language use the scoped
/// fallback until we have a language-equivalence check here.
fn scopes_share_one_ignore_language(
    scope: &ParsedGlrmScope,
    scope_label: &str,
) -> Result<bool, GlrMaskError> {
    let grammar = named_grammar_for_scope(scope)?;
    validate_ignore_terminal(&grammar, scope_label)?;
    let Some(ignore_name) = grammar.ignore.as_deref() else {
        return Ok(false);
    };
    let resolved = resolved_named_terminal_exprs(&grammar)?;
    let expected = resolved.get(ignore_name).ok_or_else(|| {
        err(&format!(
            "ignore terminal '{ignore_name}' could not be resolved in {scope_label}",
        ))
    })?;
    scopes_match_ignore_language(scope, expected, scope_label)
}

fn scopes_match_ignore_language(
    scope: &ParsedGlrmScope,
    expected: &Expr,
    scope_label: &str,
) -> Result<bool, GlrMaskError> {
    let grammar = named_grammar_for_scope(scope)?;
    validate_ignore_terminal(&grammar, scope_label)?;
    let Some(ignore_name) = grammar.ignore.as_deref() else {
        return Ok(false);
    };
    let resolved = resolved_named_terminal_exprs(&grammar)?;
    let Some(actual) = resolved.get(ignore_name) else {
        return Err(err(&format!(
            "ignore terminal '{ignore_name}' could not be resolved in {scope_label}",
        )));
    };
    if actual != expected {
        return Ok(false);
    }
    for subgrammar in &scope.subgrammars {
        let ParsedSubgrammarBody::Inline(child_scope) = &subgrammar.body else {
            continue;
        };
        let child_label = format!("{scope_label}::{}", subgrammar.name);
        if !scopes_match_ignore_language(child_scope, expected, &child_label)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn lower_parsed_grammar(
    scope: ParsedGlrmScope,
    external_tokens: &BTreeMap<String, u32>,
) -> Result<GlrmWithExternalSubgrammars, GlrMaskError> {
    scope_symbol_kinds(&scope, "grammar")?;
    if scope.subgrammars.is_empty() {
        let grammar = named_grammar_for_scope(&scope)?;
        validate_ignore_terminal(&grammar, "grammar")?;
        return Ok(GlrmWithExternalSubgrammars {
            grammar,
            placeholders: Vec::new(),
        });
    }
    let preserve_global_ignore = scopes_share_one_ignore_language(&scope, "grammar")?;
    flatten_scoped_grammar(scope, preserve_global_ignore, external_tokens)
}

fn flatten_scoped_grammar(
    scope: ParsedGlrmScope,
    preserve_global_ignore: bool,
    external_tokens: &BTreeMap<String, u32>,
) -> Result<GlrmWithExternalSubgrammars, GlrMaskError> {
    let global_ignore = preserve_global_ignore
        .then(|| scope.ignore.clone())
        .flatten();
    let mut context = FlattenContext {
        next_scope_id: 0,
        used_names: HashSet::new(),
        used_partition_names: HashSet::new(),
        rules: Vec::new(),
        lexer_partitions: BTreeMap::new(),
        lexer_literal_partition_constraints: Vec::new(),
        external_placeholders: Vec::new(),
    };
    let start = flatten_scope(
        &scope,
        true,
        "grammar",
        "",
        preserve_global_ignore,
        external_tokens,
        &mut context,
    )?;
    let (lexer_partitions, lexer_literal_partitions) = canonicalize_flattened_lexer_partitions(
        &context.rules,
        &start,
        context.lexer_partitions,
        context.lexer_literal_partition_constraints,
    )?;
    Ok(GlrmWithExternalSubgrammars {
        grammar: NamedGrammar {
            rules: context.rules,
            start,
            ignore: global_ignore,
            lexer_partitions,
            lexer_literal_partitions,
            default_lexer_partition: None,
        },
        placeholders: context.external_placeholders,
    })
}

fn canonicalize_flattened_lexer_partitions(
    rules: &[NamedRule],
    start: &str,
    lexer_partitions: BTreeMap<String, String>,
    literal_constraints: Vec<(Vec<u8>, String)>,
) -> Result<(BTreeMap<String, String>, BTreeMap<Vec<u8>, String>), GlrMaskError> {
    let grammar = NamedGrammar {
        rules: rules.to_vec(),
        start: start.to_string(),
        ignore: None,
        lexer_partitions: BTreeMap::new(),
        lexer_literal_partitions: BTreeMap::new(),
        default_lexer_partition: None,
    };
    let resolved_terminals = resolved_named_terminal_exprs(&grammar)?;
    let mut adjacency = BTreeMap::<String, BTreeSet<String>>::new();
    let mut connect = |left: &str, right: &str| {
        adjacency
            .entry(left.to_string())
            .or_default()
            .insert(right.to_string());
        adjacency
            .entry(right.to_string())
            .or_default()
            .insert(left.to_string());
    };

    let mut partition_by_expr = HashMap::new();
    for (terminal, partition) in &lexer_partitions {
        connect(partition, partition);
        let Some(expr) = resolved_terminals.get(terminal) else {
            continue;
        };
        if let Some(previous) = partition_by_expr.insert(expr.clone(), partition.clone()) {
            connect(&previous, partition);
        }
    }

    let mut partition_by_literal = BTreeMap::<Vec<u8>, String>::new();
    for (literal, partition) in &literal_constraints {
        connect(partition, partition);
        if let Some(previous) = partition_by_literal.insert(literal.clone(), partition.clone()) {
            connect(&previous, partition);
        }
    }

    let mut canonical = BTreeMap::<String, String>::new();
    let mut visited = BTreeSet::new();
    for partition in adjacency.keys() {
        if visited.contains(partition) {
            continue;
        }
        let mut stack = vec![partition.clone()];
        let mut component = Vec::new();
        while let Some(current) = stack.pop() {
            if !visited.insert(current.clone()) {
                continue;
            }
            component.push(current.clone());
            if let Some(neighbors) = adjacency.get(&current) {
                stack.extend(neighbors.iter().cloned());
            }
        }
        component.sort_unstable();
        let representative = component
            .first()
            .expect("partition component must be non-empty")
            .clone();
        for member in component {
            canonical.insert(member, representative.clone());
        }
    }

    let lexer_partitions = lexer_partitions
        .into_iter()
        .map(|(terminal, partition)| {
            let partition = canonical.get(&partition).cloned().unwrap_or(partition);
            (terminal, partition)
        })
        .collect();
    let mut lexer_literal_partitions = BTreeMap::new();
    for (literal, partition) in literal_constraints {
        let partition = canonical.get(&partition).cloned().unwrap_or(partition);
        if let Some(previous) = lexer_literal_partitions.insert(literal.clone(), partition.clone()) {
            debug_assert_eq!(previous, partition);
        }
    }
    Ok((lexer_partitions, lexer_literal_partitions))
}

fn flatten_scope(
    scope: &ParsedGlrmScope,
    top_level: bool,
    scope_label: &str,
    binding_prefix: &str,
    preserve_global_ignore: bool,
    external_tokens: &BTreeMap<String, u32>,
    context: &mut FlattenContext,
) -> Result<String, GlrMaskError> {
    let symbol_kinds = scope_symbol_kinds(scope, scope_label)?;
    let local_grammar = named_grammar_for_scope(scope)?;
    validate_ignore_terminal(&local_grammar, scope_label)?;

    let original_emitting_terminals = symbol_kinds
        .iter()
        .filter_map(|(name, kind)| (*kind == ScopedSymbolKind::Terminal).then_some(name.clone()))
        .collect::<HashSet<_>>();
    let subgrammar_names = symbol_kinds
        .iter()
        .filter_map(|(name, kind)| (*kind == ScopedSymbolKind::Subgrammar).then_some(name.clone()))
        .collect::<HashSet<_>>();

    let scope_id = context.next_scope_id;
    context.next_scope_id += 1;

    let mut local_existing_names = symbol_kinds.keys().cloned().collect::<HashSet<_>>();
    let mut working_rules = scope.rules.clone();
    working_rules.extend(external_terminal_rules(scope));
    let entry_local_name = fresh_local_name(&mut local_existing_names, "__glrm_scope_entry");

    let mut promoted_default_partitions = BTreeMap::<String, String>::new();
    if let Some(default_partition) = scope.default_lexer_partition.as_deref() {
        let mut promoter = DefaultPartitionAtomPromoter {
            default_partition,
            explicit_literal_partitions: &local_grammar.lexer_literal_partitions,
            emitting_terminals: &original_emitting_terminals,
            existing_names: &mut local_existing_names,
            promoted: HashMap::new(),
            generated_rules: Vec::new(),
            generated_partitions: BTreeMap::new(),
        };
        for rule in working_rules.iter_mut().filter(|rule| !rule.is_terminal) {
            rule.expr = promoter.rewrite_expr(&rule.expr)?;
        }
        working_rules.extend(promoter.generated_rules);
        promoted_default_partitions = promoter.generated_partitions;
    }

    let emitting_terminals = working_rules
        .iter()
        .filter(|rule| rule.is_terminal && !rule.is_internal)
        .map(|rule| rule.name.clone())
        .collect::<HashSet<_>>();
    let byte_emitting_terminals = working_rules
        .iter()
        .filter(|rule| {
            rule.is_terminal
                && !rule.is_internal
                && !matches!(rule.expr, GrammarExpr::SpecialToken(_))
        })
        .map(|rule| rule.name.clone())
        .collect::<HashSet<_>>();

    if !preserve_global_ignore
        && let Some(ignore_name) = scope.ignore.as_deref()
    {
        let skip_local_name = fresh_local_name(&mut local_existing_names, "__glrm_ignore_skip");
        working_rules.push(NamedRule {
            name: skip_local_name.clone(),
            expr: GrammarExpr::Choice(vec![
                GrammarExpr::Epsilon,
                GrammarExpr::Sequence(vec![
                    GrammarExpr::Ref(skip_local_name.clone()),
                    GrammarExpr::Ref(ignore_name.to_string()),
                ]),
            ]),
            is_terminal: false,
            is_internal: false,
        });

        let mut rewriter = IgnoreScopeRewriter {
            skip_name: &skip_local_name,
            emitting_terminals: &emitting_terminals,
            subgrammar_names: &subgrammar_names,
            existing_names: &mut local_existing_names,
            wrappers: HashMap::new(),
            generated_rules: Vec::new(),
        };
        for rule in working_rules.iter_mut().filter(|rule| !rule.is_terminal) {
            if rule.name == skip_local_name {
                continue;
            }
            rule.expr = rewriter.rewrite_expr(&rule.expr)?;
        }
        let entry_core = rewriter.rewrite_expr(&GrammarExpr::Ref(scope.start.clone()))?;
        let entry_expr = append_trailing_skip(entry_core, &skip_local_name);
        working_rules.extend(rewriter.generated_rules);
        working_rules.push(NamedRule {
            name: entry_local_name.clone(),
            expr: entry_expr,
            is_terminal: false,
            is_internal: false,
        });
    } else {
        working_rules.push(NamedRule {
            name: entry_local_name.clone(),
            expr: GrammarExpr::Ref(scope.start.clone()),
            is_terminal: false,
            is_internal: false,
        });
    }

    let original_names = symbol_kinds.keys().cloned().collect::<HashSet<_>>();
    let mut local_names = working_rules
        .iter()
        .map(|rule| rule.name.clone())
        .collect::<HashSet<_>>();
    local_names.extend(subgrammar_names.iter().cloned());
    let mut name_map = HashMap::<String, String>::new();
    let mut ordered_names = local_names.into_iter().collect::<Vec<_>>();
    ordered_names.sort_unstable();
    for local_name in ordered_names {
        let mapped = if top_level && original_names.contains(&local_name) {
            context.reserve_exact_name(&local_name)?
        } else {
            let base = if top_level {
                local_name.clone()
            } else {
                format!("__glrm_subgrammar_{scope_id}_{local_name}")
            };
            context.fresh_global_name(base)
        };
        name_map.insert(local_name, mapped);
    }

    merge_scope_lexer_config(
        scope,
        &local_grammar,
        &symbol_kinds,
        &byte_emitting_terminals,
        &promoted_default_partitions,
        &name_map,
        top_level,
        scope_id,
        scope_label,
        context,
    )?;

    for subgrammar in &scope.subgrammars {
        let child_label = format!("{scope_label}::{}", subgrammar.name);
        let binding_name = if binding_prefix.is_empty() {
            subgrammar.name.clone()
        } else {
            format!("{binding_prefix}::{}", subgrammar.name)
        };
        let alias_name = name_map
            .get(&subgrammar.name)
            .expect("subgrammar name must be allocated")
            .clone();
        match &subgrammar.body {
            ParsedSubgrammarBody::Inline(child_scope) => {
                let child_entry = flatten_scope(
                    child_scope,
                    false,
                    &child_label,
                    &binding_name,
                    preserve_global_ignore,
                    external_tokens,
                    context,
                )?;
                context.rules.push(NamedRule {
                    name: alias_name,
                    expr: GrammarExpr::Ref(child_entry),
                    is_terminal: false,
                    is_internal: false,
                });
            }
            ParsedSubgrammarBody::External => {
                let token_id = external_tokens.get(&binding_name).copied().ok_or_else(|| {
                    err(&format!(
                        "external subgrammar '{binding_name}' has no allocated placeholder token",
                    ))
                })?;
                context.rules.push(NamedRule {
                    name: alias_name.clone(),
                    expr: GrammarExpr::SpecialToken(token_id),
                    is_terminal: true,
                    is_internal: false,
                });
                context
                    .external_placeholders
                    .push(ExternalSubgrammarPlaceholder {
                        binding_name,
                        token_id,
                    });
            }
        }
    }

    for rule in working_rules {
        let mapped_name = name_map
            .get(&rule.name)
            .expect("working rule name must be allocated")
            .clone();
        let mapped_expr = rewrite_scope_refs(&rule.expr, &name_map, scope_label)?;
        context.rules.push(NamedRule {
            name: mapped_name,
            expr: mapped_expr,
            is_terminal: rule.is_terminal,
            is_internal: rule.is_internal,
        });
    }

    Ok(name_map
        .get(&entry_local_name)
        .expect("scope entry must be allocated")
        .clone())
}

fn merge_scope_lexer_config(
    scope: &ParsedGlrmScope,
    local_grammar: &NamedGrammar,
    symbol_kinds: &BTreeMap<String, ScopedSymbolKind>,
    emitting_terminals: &HashSet<String>,
    promoted_default_partitions: &BTreeMap<String, String>,
    name_map: &HashMap<String, String>,
    top_level: bool,
    scope_id: usize,
    scope_label: &str,
    context: &mut FlattenContext,
) -> Result<(), GlrMaskError> {
    let mut local_partition_names = local_grammar
        .lexer_partitions
        .values()
        .chain(local_grammar.lexer_literal_partitions.values())
        .chain(scope.default_lexer_partition.iter())
        .chain(promoted_default_partitions.values())
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut partition_name_map = BTreeMap::<String, String>::new();
    for partition in std::mem::take(&mut local_partition_names) {
        let base = if top_level {
            partition.clone()
        } else {
            format!("__glrm_subgrammar_{scope_id}_lexer_{partition}")
        };
        partition_name_map.insert(partition, context.fresh_partition_name(base));
    }
    let mapped_partition = |partition: &str| {
        partition_name_map
            .get(partition)
            .expect("scope-local partition must be allocated")
            .clone()
    };

    for terminal in emitting_terminals {
        let local_partition = local_grammar
            .lexer_partitions
            .get(terminal)
            .or_else(|| promoted_default_partitions.get(terminal))
            .or(scope.default_lexer_partition.as_ref());
        let Some(partition) = local_partition else {
            continue;
        };
        let mapped_terminal = name_map
            .get(terminal)
            .expect("local emitting terminal must be mapped")
            .clone();
        context
            .lexer_partitions
            .insert(mapped_terminal, mapped_partition(partition));
    }

    for (terminal, partition) in &local_grammar.lexer_partitions {
        if symbol_kinds.get(terminal) != Some(&ScopedSymbolKind::Terminal) {
            return Err(err(&format!(
                "lexer group in {scope_label} references non-local or non-emitting terminal '{terminal}'",
            )));
        }
        debug_assert_eq!(
            context
                .lexer_partitions
                .get(name_map.get(terminal).expect("local terminal must be mapped")),
            Some(&mapped_partition(partition)),
        );
    }

    if scope.default_lexer_partition.is_none() {
        for (literal, partition) in &local_grammar.lexer_literal_partitions {
            context
                .lexer_literal_partition_constraints
                .push((literal.clone(), mapped_partition(partition)));
        }
    }
    Ok(())
}

fn rewrite_scope_refs(
    expr: &GrammarExpr,
    name_map: &HashMap<String, String>,
    scope_label: &str,
) -> Result<GrammarExpr, GlrMaskError> {
    Ok(match expr {
        GrammarExpr::Ref(name) => GrammarExpr::Ref(
            name_map
                .get(name)
                .cloned()
                .ok_or_else(|| {
                    err(&format!(
                        "definition '{name}' is not visible in {scope_label}; definitions are scope-local",
                    ))
                })?,
        ),
        GrammarExpr::Grouped(inner) => {
            GrammarExpr::Grouped(Box::new(rewrite_scope_refs(inner, name_map, scope_label)?))
        }
        GrammarExpr::Sequence(parts) => GrammarExpr::Sequence(
            parts
                .iter()
                .map(|part| rewrite_scope_refs(part, name_map, scope_label))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        GrammarExpr::Choice(options) => GrammarExpr::Choice(
            options
                .iter()
                .map(|option| rewrite_scope_refs(option, name_map, scope_label))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        GrammarExpr::Exclude { expr, exclude } => GrammarExpr::Exclude {
            expr: Box::new(rewrite_scope_refs(expr, name_map, scope_label)?),
            exclude: Box::new(rewrite_scope_refs(exclude, name_map, scope_label)?),
        },
        GrammarExpr::Intersect { expr, intersect } => GrammarExpr::Intersect {
            expr: Box::new(rewrite_scope_refs(expr, name_map, scope_label)?),
            intersect: Box::new(rewrite_scope_refs(intersect, name_map, scope_label)?),
        },
        GrammarExpr::Quantified(inner, quantifier) => GrammarExpr::Quantified(
            Box::new(rewrite_scope_refs(inner, name_map, scope_label)?),
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
                    Ok((rewrite_scope_refs(item, name_map, scope_label)?, quantifier.clone()))
                })
                .collect::<Result<Vec<_>, GlrMaskError>>()?,
            separator: Box::new(rewrite_scope_refs(separator, name_map, scope_label)?),
            allow_empty: *allow_empty,
        },
        GrammarExpr::ExprNFA(expr_nfa) => GrammarExpr::ExprNFA(Box::new(ExprNFA::new(
            expr_nfa.nfa.clone(),
            expr_nfa
                .symbols
                .iter()
                .map(|symbol| rewrite_scope_refs(symbol, name_map, scope_label))
                .collect::<Result<Vec<_>, _>>()?,
        ))),
        GrammarExpr::Epsilon
        | GrammarExpr::Literal(_)
        | GrammarExpr::SpecialToken(_)
        | GrammarExpr::CharClass { .. }
        | GrammarExpr::RawRegex(_)
        | GrammarExpr::LexerDfa(_)
        | GrammarExpr::AnyByte => expr.clone(),
    })
}

struct DefaultPartitionAtomPromoter<'a> {
    default_partition: &'a str,
    explicit_literal_partitions: &'a BTreeMap<Vec<u8>, String>,
    emitting_terminals: &'a HashSet<String>,
    existing_names: &'a mut HashSet<String>,
    promoted: HashMap<GrammarExpr, String>,
    generated_rules: Vec<NamedRule>,
    generated_partitions: BTreeMap<String, String>,
}

impl DefaultPartitionAtomPromoter<'_> {
    fn rewrite_expr(&mut self, expr: &GrammarExpr) -> Result<GrammarExpr, GlrMaskError> {
        Ok(match expr {
            GrammarExpr::Ref(_) | GrammarExpr::Epsilon => expr.clone(),
            GrammarExpr::Literal(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte => self.promote_atom(expr.clone()),
            GrammarExpr::SpecialToken(_) => expr.clone(),
            GrammarExpr::Grouped(inner) => {
                GrammarExpr::Grouped(Box::new(self.rewrite_expr(inner)?))
            }
            GrammarExpr::Sequence(parts) => GrammarExpr::Sequence(
                parts
                    .iter()
                    .map(|part| self.rewrite_expr(part))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            GrammarExpr::Choice(options) => GrammarExpr::Choice(
                options
                    .iter()
                    .map(|option| self.rewrite_expr(option))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            GrammarExpr::Exclude { .. } | GrammarExpr::Intersect { .. }
                if !self.contains_nonterminal_ref(expr) =>
            {
                self.promote_atom(expr.clone())
            }
            GrammarExpr::Exclude { expr, exclude } => GrammarExpr::Exclude {
                expr: Box::new(self.rewrite_expr(expr)?),
                exclude: Box::new(self.rewrite_expr(exclude)?),
            },
            GrammarExpr::Intersect { expr, intersect } => GrammarExpr::Intersect {
                expr: Box::new(self.rewrite_expr(expr)?),
                intersect: Box::new(self.rewrite_expr(intersect)?),
            },
            GrammarExpr::Quantified(inner, quantifier) => GrammarExpr::Quantified(
                Box::new(self.rewrite_expr(inner)?),
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
                        Ok((self.rewrite_expr(item)?, quantifier.clone()))
                    })
                    .collect::<Result<Vec<_>, GlrMaskError>>()?,
                separator: Box::new(self.rewrite_expr(separator)?),
                allow_empty: *allow_empty,
            },
            GrammarExpr::ExprNFA(expr_nfa) => GrammarExpr::ExprNFA(Box::new(ExprNFA::new(
                expr_nfa.nfa.clone(),
                expr_nfa
                    .symbols
                    .iter()
                    .map(|symbol| self.rewrite_expr(symbol))
                    .collect::<Result<Vec<_>, _>>()?,
            ))),
        })
    }

    fn contains_nonterminal_ref(&self, expr: &GrammarExpr) -> bool {
        match expr {
            GrammarExpr::Ref(name) => !self.emitting_terminals.contains(name),
            GrammarExpr::Grouped(inner) | GrammarExpr::Quantified(inner, _) => {
                self.contains_nonterminal_ref(inner)
            }
            GrammarExpr::Sequence(parts) | GrammarExpr::Choice(parts) => {
                parts.iter().any(|part| self.contains_nonterminal_ref(part))
            }
            GrammarExpr::Exclude { expr, exclude } => {
                self.contains_nonterminal_ref(expr) || self.contains_nonterminal_ref(exclude)
            }
            GrammarExpr::Intersect { expr, intersect } => {
                self.contains_nonterminal_ref(expr) || self.contains_nonterminal_ref(intersect)
            }
            GrammarExpr::SeparatedSequence { items, separator, .. } => {
                items
                    .iter()
                    .any(|(item, _)| self.contains_nonterminal_ref(item))
                    || self.contains_nonterminal_ref(separator)
            }
            GrammarExpr::ExprNFA(expr_nfa) => expr_nfa
                .symbols
                .iter()
                .any(|symbol| self.contains_nonterminal_ref(symbol)),
            GrammarExpr::Epsilon
            | GrammarExpr::Literal(_)
            | GrammarExpr::SpecialToken(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte => false,
        }
    }

    fn promote_atom(&mut self, atom: GrammarExpr) -> GrammarExpr {
        if let Some(name) = self.promoted.get(&atom) {
            return GrammarExpr::Ref(name.clone());
        }
        let partition = match &atom {
            GrammarExpr::Literal(bytes) => self
                .explicit_literal_partitions
                .get(bytes)
                .map(String::as_str)
                .unwrap_or(self.default_partition),
            _ => self.default_partition,
        };
        let name = fresh_local_name(self.existing_names, "__glrm_lexer_atom");
        self.generated_rules.push(NamedRule {
            name: name.clone(),
            expr: atom.clone(),
            is_terminal: true,
            is_internal: false,
        });
        self.generated_partitions
            .insert(name.clone(), partition.to_string());
        self.promoted.insert(atom, name.clone());
        GrammarExpr::Ref(name)
    }
}

struct IgnoreScopeRewriter<'a> {
    skip_name: &'a str,
    emitting_terminals: &'a HashSet<String>,
    subgrammar_names: &'a HashSet<String>,
    existing_names: &'a mut HashSet<String>,
    wrappers: HashMap<GrammarExpr, String>,
    generated_rules: Vec<NamedRule>,
}

impl IgnoreScopeRewriter<'_> {
    fn rewrite_expr(&mut self, expr: &GrammarExpr) -> Result<GrammarExpr, GlrMaskError> {
        Ok(match expr {
            GrammarExpr::Ref(name)
                if self.emitting_terminals.contains(name) || self.subgrammar_names.contains(name) =>
            {
                self.wrap_atom(expr.clone())
            }
            GrammarExpr::Ref(_) | GrammarExpr::Epsilon => expr.clone(),
            GrammarExpr::Literal(_)
            | GrammarExpr::SpecialToken(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte => self.wrap_atom(expr.clone()),
            GrammarExpr::Grouped(inner) => {
                GrammarExpr::Grouped(Box::new(self.rewrite_expr(inner)?))
            }
            GrammarExpr::Sequence(parts) => GrammarExpr::Sequence(
                parts
                    .iter()
                    .map(|part| self.rewrite_expr(part))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            GrammarExpr::Choice(options) => GrammarExpr::Choice(
                options
                    .iter()
                    .map(|option| self.rewrite_expr(option))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            GrammarExpr::Exclude { .. } | GrammarExpr::Intersect { .. }
                if !self.contains_nonterminalish_ref(expr) =>
            {
                self.wrap_atom(expr.clone())
            }
            GrammarExpr::Exclude { expr, exclude } => GrammarExpr::Exclude {
                expr: Box::new(self.rewrite_expr(expr)?),
                exclude: Box::new(self.rewrite_expr(exclude)?),
            },
            GrammarExpr::Intersect { expr, intersect } => GrammarExpr::Intersect {
                expr: Box::new(self.rewrite_expr(expr)?),
                intersect: Box::new(self.rewrite_expr(intersect)?),
            },
            GrammarExpr::Quantified(inner, quantifier) => GrammarExpr::Quantified(
                Box::new(self.rewrite_expr(inner)?),
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
                        Ok((self.rewrite_expr(item)?, quantifier.clone()))
                    })
                    .collect::<Result<Vec<_>, GlrMaskError>>()?,
                separator: Box::new(self.rewrite_expr(separator)?),
                allow_empty: *allow_empty,
            },
            GrammarExpr::ExprNFA(expr_nfa) => GrammarExpr::ExprNFA(Box::new(ExprNFA::new(
                expr_nfa.nfa.clone(),
                expr_nfa
                    .symbols
                    .iter()
                    .map(|symbol| self.rewrite_expr(symbol))
                    .collect::<Result<Vec<_>, _>>()?,
            ))),
        })
    }

    fn contains_nonterminalish_ref(&self, expr: &GrammarExpr) -> bool {
        match expr {
            GrammarExpr::Ref(name) => !self.emitting_terminals.contains(name),
            GrammarExpr::Grouped(inner) | GrammarExpr::Quantified(inner, _) => {
                self.contains_nonterminalish_ref(inner)
            }
            GrammarExpr::Sequence(parts) | GrammarExpr::Choice(parts) => {
                parts.iter().any(|part| self.contains_nonterminalish_ref(part))
            }
            GrammarExpr::Exclude { expr, exclude } => {
                self.contains_nonterminalish_ref(expr)
                    || self.contains_nonterminalish_ref(exclude)
            }
            GrammarExpr::Intersect { expr, intersect } => {
                self.contains_nonterminalish_ref(expr)
                    || self.contains_nonterminalish_ref(intersect)
            }
            GrammarExpr::SeparatedSequence { items, separator, .. } => {
                items
                    .iter()
                    .any(|(item, _)| self.contains_nonterminalish_ref(item))
                    || self.contains_nonterminalish_ref(separator)
            }
            GrammarExpr::ExprNFA(expr_nfa) => expr_nfa
                .symbols
                .iter()
                .any(|symbol| self.contains_nonterminalish_ref(symbol)),
            GrammarExpr::Epsilon
            | GrammarExpr::Literal(_)
            | GrammarExpr::SpecialToken(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte => false,
        }
    }

    fn wrap_atom(&mut self, atom: GrammarExpr) -> GrammarExpr {
        if let Some(name) = self.wrappers.get(&atom) {
            return GrammarExpr::Ref(name.clone());
        }
        let name = fresh_local_name(self.existing_names, "__glrm_ignored_atom");
        self.generated_rules.push(NamedRule {
            name: name.clone(),
            expr: GrammarExpr::Sequence(vec![
                GrammarExpr::Ref(self.skip_name.to_string()),
                atom.clone(),
            ]),
            is_terminal: false,
            is_internal: false,
        });
        self.wrappers.insert(atom, name.clone());
        GrammarExpr::Ref(name)
    }
}

fn append_trailing_skip(expr: GrammarExpr, skip_name: &str) -> GrammarExpr {
    let skip = GrammarExpr::Ref(skip_name.to_string());
    match expr {
        GrammarExpr::Epsilon => skip,
        GrammarExpr::Sequence(mut parts) => {
            parts.push(skip);
            GrammarExpr::Sequence(parts)
        }
        other => GrammarExpr::Sequence(vec![other, skip]),
    }
}

fn fresh_local_name(existing_names: &mut HashSet<String>, base: &str) -> String {
    if existing_names.insert(base.to_string()) {
        return base.to_string();
    }
    for suffix in 1usize.. {
        let candidate = format!("{base}_{suffix}");
        if existing_names.insert(candidate.clone()) {
            return candidate;
        }
    }
    unreachable!()
}

// ---- Parse a byte character class string ----------------------------------

// ---- Small helpers ---------------------------------------------------------

fn hex_digit(b: u8) -> Result<u8, GlrMaskError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(err(&format!("invalid hex digit '{}'", b as char))),
    }
}

fn ensure_nfa_state(nfa: &mut NFA, state: u32) {
    while nfa.states.len() <= state as usize {
        nfa.add_state();
    }
}

fn intern_expr_nfa_symbol(
    symbols: &mut Vec<GrammarExpr>,
    symbol_labels: &mut HashMap<GrammarExpr, Label>,
    symbol: GrammarExpr,
) -> Label {
    if let Some(&label) = symbol_labels.get(&symbol) {
        return label;
    }
    let label = i32::try_from(symbols.len()).expect("ExprNFA symbol table exceeded i32 labels");
    symbols.push(symbol.clone());
    symbol_labels.insert(symbol, label);
    label
}

fn err(msg: &str) -> GlrMaskError {
    GlrMaskError::GrammarParse(msg.to_string())
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests ;

#[cfg(all(test, feature = "internal-api"))]
mod inline_subgrammar_tests {
    use super::*;

    fn assert_same_grammar(actual: &NamedGrammar, expected: &NamedGrammar) {
        assert_eq!(
            format!("{actual:#?}"),
            format!("{expected:#?}"),
            "inline subgrammar rewrite did not match the hand-inlined grammar",
        );
    }

    #[test]
    fn direct_external_binding_matches_hand_inlined_grammar() {
        let actual = from_glrm_with_inline_subgrammars(
            "start document; extern grammar SUB; nt document ::= \"X\" SUB \"!\";",
            &[("SUB", "start child; nt child ::= \"a\";")],
        )
        .unwrap();
        let expected = from_glrm(
            "start document; g SUB ::= { start child; nt child ::= \"a\"; }; nt document ::= \"X\" SUB \"!\";",
        )
        .unwrap();
        assert_same_grammar(&actual, &expected);
    }

    #[test]
    fn nested_external_child_and_grandchild_are_bound() {
        let actual = from_glrm_with_inline_subgrammars(
            "start document; extern grammar SUB; nt document ::= \"X\" SUB \"!\";",
            &[
                (
                    "SUB",
                    "start child; extern grammar GRAND; nt child ::= \"a\" GRAND;",
                ),
                ("SUB::GRAND", "start grand; nt grand ::= \"b\";"),
            ],
        )
        .unwrap();
        let expected = from_glrm(
            "start document; g SUB ::= { start child; g GRAND ::= { start grand; nt grand ::= \"b\"; }; nt child ::= \"a\" GRAND; }; nt document ::= \"X\" SUB \"!\";",
        )
        .unwrap();
        assert_same_grammar(&actual, &expected);
    }

    #[test]
    fn existing_inline_subgrammar_containing_external_is_bound() {
        let actual = from_glrm_with_inline_subgrammars(
            "start document; g OUTER ::= { start outer; extern grammar INNER; nt outer ::= \"a\" INNER; }; nt document ::= OUTER \"!\";",
            &[("OUTER::INNER", "start inner; nt inner ::= \"b\";")],
        )
        .unwrap();
        let expected = from_glrm(
            "start document; g OUTER ::= { start outer; g INNER ::= { start inner; nt inner ::= \"b\"; }; nt outer ::= \"a\" INNER; }; nt document ::= OUTER \"!\";",
        )
        .unwrap();
        assert_same_grammar(&actual, &expected);
    }

    #[test]
    fn identical_rule_and_terminal_names_stay_in_separate_scopes() {
        let actual = from_glrm_with_inline_subgrammars(
            "start document; t X ::= \"X\"; extern grammar SUB; nt document ::= X SUB;",
            &[("SUB", "start child; t X ::= \"Y\"; nt child ::= X;")],
        )
        .unwrap();
        let expected = from_glrm(
            "start document; t X ::= \"X\"; g SUB ::= { start child; t X ::= \"Y\"; nt child ::= X; }; nt document ::= X SUB;",
        )
        .unwrap();
        assert_same_grammar(&actual, &expected);
        // The child keeps its own `X`; no global name substitution collapsed
        // the two distinct `X` terminals into one.
        let dump = to_glrm(&actual);
        assert!(dump.contains("\"X\""), "parent X literal missing: {dump}");
        assert!(dump.contains("\"Y\""), "child X literal missing: {dump}");
    }

    #[test]
    fn repeated_identical_child_sources_are_independent_occurrences() {
        const CHILD: &str = "start child; nt child ::= \"a\";";
        let actual = from_glrm_with_inline_subgrammars(
            "start document; extern grammar LEFT; extern grammar RIGHT; nt document ::= LEFT RIGHT;",
            &[("LEFT", CHILD), ("RIGHT", CHILD)],
        )
        .unwrap();
        let expected = from_glrm(
            "start document; g LEFT ::= { start child; nt child ::= \"a\"; }; g RIGHT ::= { start child; nt child ::= \"a\"; }; nt document ::= LEFT RIGHT;",
        )
        .unwrap();
        assert_same_grammar(&actual, &expected);
    }

    #[test]
    fn missing_duplicate_and_unused_bindings_are_rejected() {
        let missing = from_glrm_with_inline_subgrammars(
            "start document; extern grammar SUB; nt document ::= SUB;",
            &[],
        )
        .unwrap_err();
        assert!(
            format!("{missing}").contains("has no inline binding"),
            "unexpected missing-binding error: {missing}",
        );

        let duplicate = from_glrm_with_inline_subgrammars(
            "start document; extern grammar SUB; nt document ::= SUB;",
            &[
                ("SUB", "start child; nt child ::= \"a\";"),
                ("SUB", "start other; nt other ::= \"b\";"),
            ],
        )
        .unwrap_err();
        assert!(
            format!("{duplicate}").contains("supplied more than once"),
            "unexpected duplicate-binding error: {duplicate}",
        );

        let unused = from_glrm_with_inline_subgrammars(
            "start document; nt document ::= \"a\";",
            &[("SUB", "start child; nt child ::= \"a\";")],
        )
        .unwrap_err();
        assert!(
            format!("{unused}").contains("did not match any external grammar occurrence"),
            "unexpected unused-binding error: {unused}",
        );
    }

    #[test]
    fn child_ignore_and_lexer_partitions_survive_inlining() {
        let actual = from_glrm_with_inline_subgrammars(
            "start document; extern grammar SUB; nt document ::= \"X\" SUB \"!\";",
            &[(
                "SUB",
                "start child; ignore WS; t WS ::= \" \"+; t WORD ::= /[a-z]+/; nt child ::= WORD;",
            )],
        )
        .unwrap();
        let expected = from_glrm(
            "start document; g SUB ::= { start child; ignore WS; t WS ::= \" \"+; t WORD ::= /[a-z]+/; nt child ::= WORD; }; nt document ::= \"X\" SUB \"!\";",
        )
        .unwrap();
        assert_same_grammar(&actual, &expected);
    }
}
