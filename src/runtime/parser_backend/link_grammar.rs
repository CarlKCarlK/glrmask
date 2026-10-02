//! Parser-independent source metadata for composition pruning.
//! No execution states, actions or gotos are retained here.
use std::{collections::BTreeSet, sync::Arc};
use crate::grammar::flat::{Rule, Symbol};
use crate::runtime::Constraint;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct LinkGrammar {
    pub(crate) rules: Arc<[Rule]>,
    pub(crate) nonterminal_names: Vec<String>,
    pub(crate) terminal_count: u32,
    pub(crate) control_terminals: BTreeSet<u32>,
    pub(crate) root_nullable: bool,
    pub(crate) embedded_end_token_ids: Vec<u32>,
    pub(crate) component_nonterminals: Vec<usize>,
    pub(crate) scoped_ignores: Vec<Vec<u32>>,
}

impl LinkGrammar {
    pub(crate) fn from_compiler_table(table: &crate::compiler::glr::table::GLRTable,
        ignore: Option<u32>) -> Result<Option<Arc<Self>>, String> {
        if !matches!(table.rules.first().map(|rule| rule.rhs.as_slice()),
            Some([Symbol::Nonterminal(_)])) { return Ok(None); }
        let mut ignores = table.skip_terminals.clone();
        ignores.extend(ignore);
        let names = table.nonterminal_display_names.clone();
        let grammar = Self { rules: Arc::from(table.rules.clone()),
            component_nonterminals: vec![names.len()], nonterminal_names: names,
            terminal_count: table.num_terminals,
            control_terminals: table.control_terminals.clone(),
            root_nullable: table.embedded_start_nullable(),
            embedded_end_token_ids: table.embedded_end_token_ids(),
            scoped_ignores: vec![ignores.into_iter().collect()] };
        grammar.validate()?;
        Ok(Some(Arc::new(grammar)))
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        let n = self.nonterminal_names.len();
        if self.rules.is_empty() || n == 0 || self.rules.len() > 1_000_000
            || self.component_nonterminals.len() != self.scoped_ignores.len()
            || self.component_nonterminals.iter().try_fold(0usize, |a, b| a.checked_add(*b)) != Some(n)
            || self.scoped_ignores.iter().flatten().chain(self.control_terminals.iter())
                .any(|&terminal| terminal >= self.terminal_count)
            || self.rules.iter().any(|rule| rule.lhs as usize >= n || rule.rhs.iter().any(|symbol|
                match symbol { Symbol::Terminal(t) => *t >= self.terminal_count,
                    Symbol::Nonterminal(id) => *id as usize >= n }))
        { return Err("invalid template composition grammar metadata".into()); }
        if !matches!(self.rules[0].rhs.as_slice(), [Symbol::Nonterminal(_)]) {
            return Err("template composition grammar lacks its augmented root".into());
        }
        Ok(())
    }

    fn root(&self) -> u32 {
        match self.rules[0].rhs[0] { Symbol::Nonterminal(id) => id, _ => unreachable!() }
    }

    pub(crate) fn analyze(&self, terminal_names: Vec<String>) -> crate::compiler::glr::analysis::AnalyzedGrammar {
        crate::compiler::glr::analysis::AnalyzedGrammar::from_composed_rules(
            self.rules.to_vec(), self.terminal_count, terminal_names,
            self.nonterminal_names.clone(), self.rules[0].lhs)
    }

    /// Substitute child roots into the parent's source rules, matching the
    /// ordinary grammar analysis without constructing any parser table.
    pub(crate) fn compose(components: &[Arc<Constraint>], terminal_offsets: &[u32],
        slots: &[Vec<u32>]) -> Result<Option<Arc<Self>>, String> {
        let sources = components.iter().map(|component| component.template_parser.as_ref()
            .and_then(|parser| parser.link_grammar.as_deref())).collect::<Option<Vec<_>>>();
        let Some(sources) = sources else { return Ok(None); };
        if sources.len() != terminal_offsets.len() || slots.len() + 1 != sources.len() {
            return Err("template composition grammar layout mismatch".into());
        }
        let parent = sources[0];
        let mut rules = parent.rules.to_vec();
        let parent_rules = rules.len();
        let mut names = parent.nonterminal_names.clone();
        let mut counts = parent.component_nonterminals.clone();
        let mut ignores = parent.scoped_ignores.clone();
        let mut controls = parent.control_terminals.clone();
        if parent.root_nullable && !rules.iter().any(|r| r.lhs == parent.root() && r.rhs.is_empty()) {
            rules.push(Rule { lhs: parent.root(), rhs: vec![] });
        }
        let mut terminals = parent.terminal_count;
        for (index, child) in sources.iter().enumerate().skip(1) {
            let nt_offset = u32::try_from(names.len()).map_err(|_| "composition nonterminal overflow")?;
            let terminal_offset = terminal_offsets[index];
            if terminal_offset != terminals { return Err("composition grammar terminal layout mismatch".into()); }
            let root = child.root().checked_add(nt_offset).ok_or("composition root overflow")?;
            for rule in &mut rules[..parent_rules] {
                for symbol in &mut rule.rhs {
                    if let Symbol::Terminal(terminal) = *symbol && slots[index-1].contains(&terminal) {
                        *symbol = Symbol::Nonterminal(root);
                    }
                }
            }
            for rule in child.rules.iter() {
                rules.push(Rule { lhs: rule.lhs + nt_offset, rhs: rule.rhs.iter().map(|symbol|
                    match *symbol { Symbol::Terminal(t) => Symbol::Terminal(t + terminal_offset),
                        Symbol::Nonterminal(id) => Symbol::Nonterminal(id + nt_offset) }).collect() });
            }
            if child.root_nullable && !rules.iter().any(|r| r.lhs == root && r.rhs.is_empty()) {
                rules.push(Rule { lhs: root, rhs: vec![] });
            }
            names.extend(child.nonterminal_names.iter().map(|name| format!("child{}::{name}", index-1)));
            counts.extend_from_slice(&child.component_nonterminals);
            ignores.extend(child.scoped_ignores.iter().map(|row|
                row.iter().map(|&t| t + terminal_offset).collect()));
            controls.extend(child.control_terminals.iter().map(|&t| t + terminal_offset));
            terminals = terminal_offset.checked_add(child.terminal_count).ok_or("composition terminal overflow")?;
        }
        let mut embedded_end_token_ids = sources.iter()
            .flat_map(|source| source.embedded_end_token_ids.iter().copied()).collect::<Vec<_>>();
        embedded_end_token_ids.sort_unstable();
        embedded_end_token_ids.dedup();
        let grammar = Self { rules: Arc::from(rules), nonterminal_names: names,
            terminal_count: terminals, control_terminals: controls, root_nullable: parent.root_nullable,
            embedded_end_token_ids,
            component_nonterminals: counts, scoped_ignores: ignores };
        grammar.validate()?;
        Ok(Some(Arc::new(grammar)))
    }
}
