//! The grammar-only projection of the flat splice, without an unused LR table.
//!
//! The boundary parser uses the intact component tables in SignedLinkContext.
//! Its lexical analysis needs only these four fields of the legacy splice.
//! Keep this projection literal (including rule order and nullable restoration),
//! rather than replacing it by merely an equivalent grammar.
use crate::compiler::glr::table::{ComposedTable, GLRTable, SubgrammarTableInput};
use crate::grammar::flat::{NonterminalID, Rule, Symbol, TerminalID};

#[derive(Debug, PartialEq, Eq)]
pub(super) struct RuleLayout {
    pub terminal_offsets: Vec<TerminalID>,
    pub num_terminals: TerminalID,
    pub rules: Vec<Rule>,
    pub nonterminal_display_names: Vec<String>,
}

impl RuleLayout {
    /// Reference projection. This consumes and drops the unused ACTION/GOTO
    /// table instead of letting a partially populated GLRTable escape.
    pub fn from_table(composed: ComposedTable) -> Self {
        Self {
            terminal_offsets: composed.terminal_offsets,
            num_terminals: composed.table.num_terminals,
            rules: composed.table.rules,
            nonterminal_display_names: composed.table.nonterminal_display_names,
        }
    }
}

fn root(rules: &[Rule], nonterminals: usize) -> Result<NonterminalID, String> {
    match rules.first().map(|rule| rule.rhs.as_slice()) {
        Some([Symbol::Nonterminal(root)]) if (*root as usize) < nonterminals => Ok(*root),
        _ => Err("boundary rule layout requires a valid single-nonterminal augmented start".into()),
    }
}

fn validate_rules(rules: &[Rule], terminals: u32, nonterminals: usize) -> Result<(), String> {
    if rules.iter().any(|rule| (rule.lhs as usize) >= nonterminals
        || rule.rhs.iter().any(|symbol| match *symbol {
            Symbol::Terminal(t) => t >= terminals,
            Symbol::Nonterminal(n) => (n as usize) >= nonterminals,
        }))
    {
        return Err("boundary retained rule contains an out-of-domain symbol".into());
    }
    Ok(())
}

fn restore_nullable(rules: &mut Vec<Rule>, nonterminal: NonterminalID) {
    if !rules.iter().any(|rule| rule.lhs == nonterminal && rule.rhs.is_empty()) {
        rules.push(Rule { lhs: nonterminal, rhs: Vec::new() });
    }
}

fn start_nullable(rules: &[Rule], nonterminals: usize, start: u32) -> bool {
    let mut nullable = vec![false; nonterminals];
    loop {
        let mut changed = false;
        for rule in rules {
            if !nullable[rule.lhs as usize]
                && rule.rhs.iter().all(|symbol| match *symbol {
                    Symbol::Terminal(_) => false,
                    Symbol::Nonterminal(n) => nullable[n as usize],
                })
            {
                nullable[rule.lhs as usize] = true;
                changed = true;
            }
        }
        if nullable[start as usize] || !changed {
            return nullable[start as usize];
        }
    }
}

/// Mirror the existing table's serialized augmented-name marker. A literal
/// metadata comparison against the independent composer guards this private
/// encoding; child-name markers and other parent metadata must stay untouched.
fn set_nullable_name(names: &mut [String], augmented: u32, nullable: bool) {
    const SUFFIX: &str = "\0glrmask:embedded-nullable-start";
    let name = &mut names[augmented as usize];
    if name.ends_with(SUFFIX) {
        name.truncate(name.len() - SUFFIX.len());
    }
    if nullable {
        name.push_str(SUFFIX);
    }
}

/// For every successful control-free legacy splice, this is its literal
/// (rules, names, terminal layout) projection. It does NOT construct or certify
/// an executable spliced ACTION table. The caller must still validate its
/// actual SignedLinkContext and provider before building/publishing a parser.
pub(super) fn compose(
    parent: &GLRTable,
    parent_rules: &[Rule],
    children: &[SubgrammarTableInput<'_>],
    child_rules: &[&[Rule]],
) -> Result<RuleLayout, String> {
    if children.len() != child_rules.len() {
        return Err("child table/rule override count mismatch".into());
    }
    if !parent.control_terminals.is_empty()
        || children.iter().any(|child| !child.table.control_terminals.is_empty())
    {
        return Err("boundary rule projection requires control-free component tables".into());
    }
    validate_rules(parent_rules, parent.num_terminals, parent.nonterminal_display_names.len())?;
    let parent_root = root(parent_rules, parent.nonterminal_display_names.len())?;
    let mut terminal_offsets = Vec::with_capacity(children.len() + 1);
    terminal_offsets.push(0);
    let mut num_terminals = parent.num_terminals;
    let mut nonterminal_offsets = Vec::with_capacity(children.len());
    let mut num_nonterminals = u32::try_from(parent.nonterminal_display_names.len())
        .map_err(|_| "boundary nonterminal domain exceeds u32")?;
    for (child, rules) in children.iter().zip(child_rules) {
        validate_rules(rules, child.table.num_terminals, child.table.nonterminal_display_names.len())?;
        root(rules, child.table.nonterminal_display_names.len())?;
        if std::iter::once(child.placeholder_terminal)
            .chain(child.additional_placeholder_terminals.iter().copied())
            .any(|terminal| terminal >= parent.num_terminals)
        {
            return Err("boundary placeholder is outside the parent terminal domain".into());
        }
        terminal_offsets.push(num_terminals);
        num_terminals = num_terminals.checked_add(child.table.num_terminals)
            .ok_or("merged terminal ID overflow")?;
        nonterminal_offsets.push(num_nonterminals);
        num_nonterminals = num_nonterminals.checked_add(
            u32::try_from(child.table.nonterminal_display_names.len())
                .map_err(|_| "boundary child nonterminal domain exceeds u32")?,
        ).ok_or("merged nonterminal ID overflow")?;
    }

    let mut rules = parent_rules.to_vec();
    if parent.embedded_start_nullable() {
        restore_nullable(&mut rules, parent_root);
    }
    // Only the original parent prefix is substituted, never appended child
    // rules whose local terminal IDs might happen to equal a later slot ID.
    let parent_rule_count = rules.len();
    let mut nonterminal_display_names = parent.nonterminal_display_names.clone();
    for (index, (child, local_rules)) in children.iter().zip(child_rules).enumerate() {
        let terminal_offset = terminal_offsets[index + 1];
        let nonterminal_offset = nonterminal_offsets[index];
        let child_root = root(local_rules, child.table.nonterminal_display_names.len())?
            + nonterminal_offset;
        for rule in &mut rules[..parent_rule_count] {
            for symbol in &mut rule.rhs {
                if let Symbol::Terminal(terminal) = *symbol
                    && (terminal == child.placeholder_terminal
                        || child.additional_placeholder_terminals.contains(&terminal))
                {
                    *symbol = Symbol::Nonterminal(child_root);
                }
            }
        }
        rules.extend(local_rules.iter().map(|rule| Rule {
            lhs: rule.lhs + nonterminal_offset,
            rhs: rule.rhs.iter().map(|symbol| match *symbol {
                Symbol::Terminal(t) => Symbol::Terminal(t + terminal_offset),
                Symbol::Nonterminal(n) => Symbol::Nonterminal(n + nonterminal_offset),
            }).collect(),
        }));
        if child.start_nullable {
            restore_nullable(&mut rules, child_root);
        }
        nonterminal_display_names.extend(child.table.nonterminal_display_names.iter()
            .map(|name| format!("child{index}::{name}")));
    }
    let nullable = start_nullable(&rules, num_nonterminals as usize, parent_root);
    set_nullable_name(&mut nonterminal_display_names, rules[0].lhs, nullable);
    Ok(RuleLayout { terminal_offsets, num_terminals, rules, nonterminal_display_names })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::glr::table::compose_subgrammar_tables_with_rules;
    use crate::grammar::{ast::lower, glrm::from_glrm};

    fn table(source: &str) -> (GLRTable, AnalyzedGrammar) {
        let grammar = lower(&from_glrm(source).unwrap()).unwrap();
        let analysis = AnalyzedGrammar::from_grammar_def(&grammar);
        (GLRTable::build(&analysis), analysis)
    }

    fn terminal(analysis: &AnalyzedGrammar, name: &str) -> u32 {
        analysis.terminal_display_names.iter().position(|n| n == name).unwrap() as u32
    }

    #[test]
    fn boundary_rule_projection_matches_splice_for_compiled_families() {
        let mut comparisons = 0;
        for parent_body in ["LEFT", "\"<\" LEFT \">\" RIGHT \"!\"", "LEFT RIGHT", "LEFT | RIGHT", "(\"x\" LEFT)+ RIGHT?"] {
            let (parent, analysis) = table(&format!(
                "start document; t LEFT ::= @token(998); t RIGHT ::= @token(999); nt document ::= {parent_body};"));
            let left = terminal(&analysis, "LEFT");
            let right = terminal(&analysis, "RIGHT");
            for child_body in ["\"a\"", "\"a\" \"b\"", "\"a\" | \"b\"", "\"a\"+", "\"a\"?", "\"a\" (\"b\" \"c\")*"] {
                let (child, _) = table(&format!("start child; nt child ::= {child_body};"));
                let extra = [right];
                let inputs = [SubgrammarTableInput {
                    placeholder_terminal: left,
                    additional_placeholder_terminals: if parent_body == "LEFT" { &[] } else { &extra },
                    table: &child, ignore_terminal: None,
                    start_nullable: child.embedded_start_nullable(),
                }];
                let retained = [child.rules.as_slice()];
                let reference = compose_subgrammar_tables_with_rules(
                    &parent, &parent.rules, None, &inputs, &retained).unwrap();
                let actual = compose(&parent, &parent.rules, &inputs, &retained).unwrap();
                assert_eq!(actual, RuleLayout::from_table(reference), "{parent_body}; {child_body}");
                comparisons += 1;
            }
        }
        assert_eq!(comparisons, 30);
    }

    #[test]
    fn boundary_rule_projection_preserves_multiple_children_and_retained_overrides() {
        let (mut parent, analysis) = table(
            "start doc; t A ::= @token(998); t B ::= @token(999); t UNBOUND ::= @token(997); nt doc ::= A B | UNBOUND;");
        let (mut first, _) = table("start first; nt first ::= \"x\"?;");
        let (mut second, _) = table("start second; nt second ::= \"y\" \"z\";");
        let pr = std::mem::take(&mut parent.rules);
        let ar = std::mem::take(&mut first.rules);
        let br = std::mem::take(&mut second.rules);
        parent.set_embedded_start_nullable(true);
        parent.set_embedded_end_token_ids(&[701,702]);
        let inputs = [
            SubgrammarTableInput { placeholder_terminal: terminal(&analysis,"A"), additional_placeholder_terminals:&[],
                table:&first, ignore_terminal:None, start_nullable:true },
            SubgrammarTableInput { placeholder_terminal: terminal(&analysis,"B"), additional_placeholder_terminals:&[],
                table:&second, ignore_terminal:None, start_nullable:false },
        ];
        let retained = [ar.as_slice(),br.as_slice()];
        let expected = compose_subgrammar_tables_with_rules(&parent,&pr,None,&inputs,&retained).unwrap();
        let actual = compose(&parent,&pr,&inputs,&retained).unwrap();
        assert_eq!(actual,RuleLayout::from_table(expected));
        let unbound = Symbol::Terminal(terminal(&analysis,"UNBOUND"));
        assert!(actual.rules.iter().any(|r|r.rhs.contains(&unbound)));
        assert!(actual.nonterminal_display_names.iter().any(|n|n.starts_with("child1::")));
    }

    #[test]
    fn boundary_rule_projection_rejects_malformed_domains_and_overflow() {
        let (mut parent, analysis) = table("start doc; t A ::= @token(999); nt doc ::= A;");
        let (child,_) = table("start child; nt child ::= \"x\";");
        let input = [SubgrammarTableInput { placeholder_terminal:terminal(&analysis,"A"), additional_placeholder_terminals:&[],
            table:&child, ignore_terminal:None, start_nullable:false }];
        assert!(compose(&parent,&parent.rules,&input,&[]).is_err());
        assert!(compose(&parent,&[],&input,&[&child.rules]).is_err());
        let mut bad = parent.rules.clone();
        bad[0].rhs = vec![Symbol::Terminal(parent.num_terminals)];
        assert!(compose(&parent,&bad,&input,&[&child.rules]).is_err());
        parent.control_terminals.insert(0);
        assert!(compose(&parent,&parent.rules,&input,&[&child.rules]).is_err());
        parent.control_terminals.clear();
        parent.num_terminals = u32::MAX;
        assert!(compose(&parent,&parent.rules,&input,&[&child.rules]).is_err());
    }
}
