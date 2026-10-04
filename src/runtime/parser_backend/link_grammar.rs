//! Parser-independent source metadata for composition pruning.
//! No execution states, actions or gotos are retained here.
use std::{collections::BTreeSet, sync::{Arc, OnceLock}};
use crate::grammar::flat::{Rule, Symbol};
use crate::runtime::Constraint;

/// Execution needs this scalar manifest. Compiler-only rules and effects are
/// shared through an immutable source forest and materialized at their consumer.
#[derive(Debug, Clone)]
pub(crate) struct LinkGrammar {
    pub(crate) terminal_count: u32,
    pub(crate) control_terminals: BTreeSet<u32>,
    pub(crate) root_nullable: bool,
    pub(crate) embedded_end_token_ids: Vec<u32>,
    nonterminal_count: usize,
    rule_count: usize,
    root_nonterminal: u32,
    root_has_empty_rule: bool,
    source: Option<Arc<CompositionRecipe>>,
    flat: OnceLock<Arc<FlatLinkGrammar>>,
    // Fresh dynamic links retain the exact outward-proof recipe. It is forced
    // by a compiler/query/save consumer, never by current runtime masks.
    pub(crate) deferred_boundary_summary: OnceLock<Arc<crate::compiler::boundary_candidates::DeferredCompositionSummary>>,
    // Transient flat proof only, validated against the complete component
    // identity at every use. It does not enter artifact bytes or runtime state.
    pub(crate) flat_boundary_tail_r1: OnceLock<crate::compiler::boundary_tail::FlatBoundaryTailR1>,
    // Original flat entry-prefix fact only; identity-validated on every use.
    pub(crate) entry_prefix_cover: OnceLock<crate::compiler::boundary_tail::EntryPrefixCoverProof>,
}

#[cfg(test)]
mod forest_tests {
    use super::*;
    use crate::compiler::boundary_stack_support::CompilerEffects;
    use serde_json::json;

    fn leaf(nullable: bool, empty_rule: bool, end: u32) -> Arc<LinkGrammar> {
        let mut rules = vec![Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] },
            Rule { lhs: 1, rhs: vec![Symbol::Terminal(0)] }];
        if empty_rule { rules.push(Rule { lhs: 1, rhs: vec![] }); }
        Arc::new(LinkGrammar::from_flat(Arc::new(FlatLinkGrammar {
            rules: Arc::from(rules), nonterminal_names: vec!["aug".into(), "body".into()],
            terminal_count: 2, control_terminals: BTreeSet::from([1]), root_nullable: nullable,
            embedded_end_token_ids: vec![end], component_nonterminals: vec![2], scoped_ignores: vec![vec![1]],
            stack_effects: Some(serde_json::from_value::<CompilerEffects>(json!({
                "states":2,"terminals":2,"effects":[{"source":0,"pop":0,"pushes":[1]}],
                "entries":{"0":[{"source":0,"pop":0,"pushes":[1]}]},"accepting":[1]
            })).unwrap()),
        })).unwrap())
    }

    #[test]
    fn descriptor_shares_sources_and_matches_existing_flat_wire_fields() {
        for parent_nullable in [false, true] {
            for child_nullable in [false, true] {
                for empty in [false, true] {
                    let parent = leaf(parent_nullable, empty, 91);
                    let child = leaf(child_nullable, empty, 92);
                    let linked = LinkGrammar::compose_sources(vec![parent.clone(), child.clone()],
                        vec![0, 2], vec![vec![0]], vec![0, 2], 4,
                        vec![(0, 0, 1, 0, 1, child_nullable)]).unwrap();
                    assert!(!linked.is_materialized());
                    assert!(Arc::ptr_eq(&linked.source.as_ref().unwrap().sources[0], &parent));
                    assert!(Arc::ptr_eq(&linked.source.as_ref().unwrap().sources[1], &child));
                    linked.validate().unwrap();
                    assert!(!linked.is_materialized());
                    let mut rules = vec![Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] },
                        Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(3)] }];
                    if empty || parent_nullable { rules.push(Rule { lhs: 1, rhs: vec![] }); }
                    rules.extend([Rule { lhs: 2, rhs: vec![Symbol::Nonterminal(3)] },
                        Rule { lhs: 3, rhs: vec![Symbol::Terminal(2)] }]);
                    if empty || child_nullable { rules.push(Rule { lhs: 3, rhs: vec![] }); }
                    let mut effects = vec![json!({"source":0,"pop":0,"pushes":[1]}),
                        json!({"source":0,"pop":0,"pushes":[1,2]}),
                        json!({"source":2,"pop":0,"pushes":[3]})];
                    if child_nullable { effects.push(json!({"source":2,"pop":1,"pushes":[]})); }
                    effects.push(json!({"source":3,"pop":1,"pushes":[]}));
                    let expected = FlatLinkGrammar { rules: Arc::from(rules),
                        nonterminal_names: vec!["aug".into(), "body".into(), "child0::aug".into(), "child0::body".into()],
                        terminal_count: 4, control_terminals: BTreeSet::from([1, 3]), root_nullable: parent_nullable,
                        embedded_end_token_ids: vec![91, 92], component_nonterminals: vec![2, 2],
                        scoped_ignores: vec![vec![1], vec![3]], stack_effects: Some(
                            serde_json::from_value::<CompilerEffects>(json!({"states":4,"terminals":4,"effects":effects,
                                "entries":{"2":[{"source":2,"pop":0,"pushes":[3]}]},"accepting":[1]})).unwrap()) };
                    let bytes = bincode::serialize(&linked).unwrap();
                    assert_eq!(bytes, bincode::serialize(&expected).unwrap());
                    assert!(linked.is_materialized());
                    let loaded: LinkGrammar = bincode::deserialize(&bytes).unwrap();
                    assert_eq!(loaded, linked);
                }
            }
        }
    }

    #[test]
    fn later_nullable_root_and_nested_sources_keep_exact_epsilon_rules() {
        let parent = leaf(false, false, 91);
        let child = leaf(false, false, 92);
        let mut first = LinkGrammar::compose_sources(vec![parent, child], vec![0, 2],
            vec![vec![0]], vec![0, 2], 4, vec![(0, 0, 1, 0, 1, false)]).unwrap();
        first.root_nullable = true;
        first.embedded_end_token_ids = vec![99];
        let first = Arc::new(first);
        let second = LinkGrammar::compose_sources(vec![first.clone(), leaf(true, false, 93)],
            vec![0, 4], vec![vec![2]], vec![0, 4], 6, vec![(0, 2, 1, 0, 1, true)]).unwrap();
        assert!(!first.is_materialized() && !second.is_materialized());
        assert_eq!(second.embedded_end_token_ids, vec![93, 99]);
        assert_eq!(second.rules().iter().filter(|r| r.lhs == 1 && r.rhs.is_empty()).count(), 1);
        assert_eq!(second.rules().iter().filter(|r| r.lhs == 5 && r.rhs.is_empty()).count(), 1);
        assert_eq!(second.rules().iter().find(|r| r.lhs == 3).unwrap().rhs, vec![Symbol::Nonterminal(5)]);
        assert_eq!(second.component_nonterminals(), &[2, 2, 2]);
        assert_eq!(second.scoped_ignores(), &[vec![1], vec![3], vec![5]]);
        let bytes = bincode::serialize(&second).unwrap();
        let loaded: LinkGrammar = bincode::deserialize(&bytes).unwrap();
        assert_eq!(loaded, second);
    }

    #[test]
    fn malformed_leaf_and_descriptor_layout_are_rejected_before_materialization() {
        let parent = leaf(false, false, 91);
        let mut flat = parent.flat().clone();
        flat.rules = Arc::from(vec![Rule { lhs: 0, rhs: vec![Symbol::Terminal(0)] }]);
        assert!(bincode::deserialize::<LinkGrammar>(&bincode::serialize(&flat).unwrap()).is_err());
        assert!(LinkGrammar::compose_sources(vec![parent.clone(), parent], vec![0, 3],
            vec![vec![0]], vec![0, 2], 4, vec![(0, 0, 1, 0, 1, false)]).is_err());
    }
}

#[derive(Debug)]
struct CompositionRecipe {
    sources: Vec<Arc<LinkGrammar>>,
    terminal_offsets: Vec<u32>,
    slots: Vec<Vec<u32>>,
    state_offsets: Vec<u32>,
    state_count: u32,
    effect_links: Vec<(u32, u32, u32, u32, u32, bool)>,
}

impl LinkGrammar {
    fn from_flat(flat: Arc<FlatLinkGrammar>) -> Result<Self, String> {
        flat.validate()?;
        let root = flat.root();
        Ok(Self {
            terminal_count: flat.terminal_count,
            control_terminals: flat.control_terminals.clone(),
            root_nullable: flat.root_nullable,
            embedded_end_token_ids: flat.embedded_end_token_ids.clone(),
            nonterminal_count: flat.nonterminal_names.len(),
            rule_count: flat.rules.len(), root_nonterminal: root,
            root_has_empty_rule: flat.rules.iter().any(|rule| rule.lhs == root && rule.rhs.is_empty()),
            source: None, flat: OnceLock::from(flat), deferred_boundary_summary: OnceLock::new(),
            flat_boundary_tail_r1: OnceLock::new(),
            entry_prefix_cover: OnceLock::new(),
        })
    }

    pub(crate) fn from_compiler_table(table: &crate::compiler::glr::table::GLRTable,
        ignore: Option<u32>) -> Result<Option<Arc<Self>>, String> {
        FlatLinkGrammar::from_compiler_table(table, ignore)?
            .map(|flat| Self::from_flat(flat).map(Arc::new)).transpose()
    }

    fn flat(&self) -> &FlatLinkGrammar {
        self.flat.get_or_init(|| FlatLinkGrammar::compose(
            self.source.as_ref().expect("unmaterialized grammar has a source recipe"))
            .expect("validated composition metadata must materialize exactly")).as_ref()
    }

    pub(crate) fn rules(&self) -> &[Rule] { &self.flat().rules }
    pub(crate) fn component_nonterminals(&self) -> &[usize] { &self.flat().component_nonterminals }
    pub(crate) fn scoped_ignores(&self) -> &[Vec<u32>] { &self.flat().scoped_ignores }
    pub(crate) fn stack_effects(&self) -> Option<&crate::compiler::boundary_stack_support::CompilerEffects> {
        self.flat().stack_effects.as_ref()
    }
    pub(crate) fn set_stack_effects(&mut self,
        effects: Option<crate::compiler::boundary_stack_support::CompilerEffects>) {
        let _ = self.flat();
        Arc::make_mut(self.flat.get_mut().unwrap()).stack_effects = effects;
    }
    pub(crate) fn is_materialized(&self) -> bool { self.flat.get().is_some() }

    fn validate_manifest(&self) -> Result<(), String> {
        if self.rule_count == 0 || self.rule_count > 1_000_000 || self.nonterminal_count == 0
            || self.nonterminal_count > u32::MAX as usize
            || self.root_nonterminal as usize >= self.nonterminal_count
            || self.control_terminals.iter().any(|&terminal| terminal >= self.terminal_count)
        { return Err("invalid template composition grammar metadata".into()); }
        Ok(())
    }
    pub(crate) fn validate(&self) -> Result<(), String> {
        self.validate_manifest()?;
        if let Some(flat) = self.flat.get() { flat.validate()?; }
        // Recipes are issued only from validated immutable sources with checked
        // coordinates; validating their manifest does not rebuild source rules.
        Ok(())
    }

    pub(crate) fn analyze(&self, names: Vec<String>) -> crate::compiler::glr::analysis::AnalyzedGrammar {
        self.flat().analyze(names)
    }

    pub(crate) fn compose(components: &[Arc<Constraint>], terminal_offsets: &[u32],
        slots: &[Vec<u32>]) -> Result<Option<Arc<Self>>, String> {
        let sources = components.iter().map(|component| component.template_parser.as_ref()
            .and_then(|parser| parser.link_grammar.clone())).collect::<Option<Vec<_>>>();
        let Some(sources) = sources else { return Ok(None); };
        if sources.len() != terminal_offsets.len() || slots.len() + 1 != sources.len() {
            return Err("template composition grammar layout mismatch".into());
        }
        let mut offsets = Vec::new(); let mut states = 0u32;
        for component in components {
            offsets.push(states);
            states = states.checked_add(component.template_parser.as_ref().unwrap().state_count)
                .ok_or("composition parser coordinate overflow")?;
        }
        let links = slots.iter().enumerate().flat_map(|(child, slots)| {
            let embedding = components[child + 1].template_parser.as_ref().unwrap().embedding.as_ref().unwrap();
            slots.iter().map(move |&slot| (0, slot, child as u32 + 1, 0, embedding.return_pop, embedding.nullable))
        }).collect();
        Self::compose_sources(sources, terminal_offsets.to_vec(), slots.to_vec(), offsets, states, links)
            .map(|grammar| Some(Arc::new(grammar)))
    }

    fn compose_sources(sources: Vec<Arc<Self>>, terminal_offsets: Vec<u32>, slots: Vec<Vec<u32>>,
        state_offsets: Vec<u32>, state_count: u32,
        effect_links: Vec<(u32, u32, u32, u32, u32, bool)>) -> Result<Self, String> {
        if sources.is_empty() || sources.len() != terminal_offsets.len()
            || sources.len() != state_offsets.len() || slots.len() + 1 != sources.len()
        { return Err("template composition grammar layout mismatch".into()); }
        let parent = &sources[0];
        parent.validate_manifest()?;
        let mut terminals = parent.terminal_count;
        let mut nonterminals = parent.nonterminal_count;
        let mut rules = parent.rule_count + usize::from(parent.root_nullable && !parent.root_has_empty_rule);
        let mut controls = parent.control_terminals.clone();
        for (index, child) in sources.iter().enumerate().skip(1) {
            child.validate_manifest()?;
            if terminal_offsets[index] != terminals {
                return Err("composition grammar terminal layout mismatch".into());
            }
            if slots[index - 1].iter().any(|&slot| slot >= parent.terminal_count) {
                return Err("composition slot outside parent terminal coordinate".into());
            }
            let nt_offset = u32::try_from(nonterminals).map_err(|_| "composition nonterminal overflow")?;
            child.root_nonterminal.checked_add(nt_offset).ok_or("composition root overflow")?;
            nonterminals = nonterminals.checked_add(child.nonterminal_count)
                .ok_or("composition nonterminal overflow")?;
            rules = rules.checked_add(child.rule_count)
                .and_then(|count| count.checked_add(usize::from(child.root_nullable && !child.root_has_empty_rule)))
                .ok_or("composition rule count overflow")?;
            terminals = terminals.checked_add(child.terminal_count).ok_or("composition terminal overflow")?;
            controls.extend(child.control_terminals.iter().map(|&t| t + terminal_offsets[index]));
        }
        let mut ends = sources.iter().flat_map(|source| source.embedded_end_token_ids.iter().copied()).collect::<Vec<_>>();
        ends.sort_unstable(); ends.dedup();
        let grammar = Self {
            terminal_count: terminals, control_terminals: controls,
            root_nullable: parent.root_nullable, embedded_end_token_ids: ends,
            nonterminal_count: nonterminals, rule_count: rules, root_nonterminal: parent.root_nonterminal,
            root_has_empty_rule: parent.root_has_empty_rule || parent.root_nullable,
            source: Some(Arc::new(CompositionRecipe { sources, terminal_offsets, slots,
                state_offsets, state_count, effect_links })), flat: OnceLock::new(),
            deferred_boundary_summary: OnceLock::new(),
            flat_boundary_tail_r1: OnceLock::new(),
            entry_prefix_cover: OnceLock::new(),
        };
        grammar.validate_manifest()?;
        Ok(grammar)
    }
}

// Preserve the exact existing flat serde field order. A save is an explicit
// compiler-metadata consumer; runtime linking and matching need no new schema.
impl serde::Serialize for LinkGrammar {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(serde::Serialize)]
        struct Wire<'a> {
            rules: &'a Arc<[Rule]>, nonterminal_names: &'a Vec<String>, terminal_count: u32,
            control_terminals: &'a BTreeSet<u32>, root_nullable: bool, embedded_end_token_ids: &'a Vec<u32>,
            component_nonterminals: &'a Vec<usize>, scoped_ignores: &'a Vec<Vec<u32>>,
            stack_effects: &'a Option<crate::compiler::boundary_stack_support::CompilerEffects>,
        }
        let flat = self.flat();
        Wire { rules: &flat.rules, nonterminal_names: &flat.nonterminal_names,
            terminal_count: self.terminal_count, control_terminals: &self.control_terminals,
            root_nullable: self.root_nullable, embedded_end_token_ids: &self.embedded_end_token_ids,
            component_nonterminals: &flat.component_nonterminals, scoped_ignores: &flat.scoped_ignores,
            stack_effects: &flat.stack_effects }.serialize(serializer)
    }
}
impl<'de> serde::Deserialize<'de> for LinkGrammar {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let flat = <FlatLinkGrammar as serde::Deserialize>::deserialize(deserializer)?;
        Self::from_flat(Arc::new(flat)).map_err(serde::de::Error::custom)
    }
}
impl PartialEq for LinkGrammar {
    fn eq(&self, other: &Self) -> bool {
        let a = self.flat(); let b = other.flat();
        self.terminal_count == other.terminal_count && self.control_terminals == other.control_terminals
            && self.root_nullable == other.root_nullable && self.embedded_end_token_ids == other.embedded_end_token_ids
            && a.rules == b.rules && a.nonterminal_names == b.nonterminal_names
            && a.component_nonterminals == b.component_nonterminals && a.scoped_ignores == b.scoped_ignores
            && a.stack_effects == b.stack_effects
    }
}
impl Eq for LinkGrammar {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct FlatLinkGrammar {
    pub(crate) rules: Arc<[Rule]>,
    pub(crate) nonterminal_names: Vec<String>,
    pub(crate) terminal_count: u32,
    pub(crate) control_terminals: BTreeSet<u32>,
    pub(crate) root_nullable: bool,
    pub(crate) embedded_end_token_ids: Vec<u32>,
    pub(crate) component_nonterminals: Vec<usize>,
    pub(crate) scoped_ignores: Vec<Vec<u32>>,
    pub(crate) stack_effects: Option<crate::compiler::boundary_stack_support::CompilerEffects>,
}

impl FlatLinkGrammar {
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
            scoped_ignores: vec![ignores.into_iter().collect()], stack_effects: None };
        grammar.validate()?;
        Ok(Some(Arc::new(grammar)))
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if let Some(effects) = &self.stack_effects {
            effects.validate()?;
            if effects.terminals != self.terminal_count {
                return Err("compiler stack effects disagree with grammar terminals".into());
            }
        }
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
    fn compose(recipe: &CompositionRecipe) -> Result<Arc<Self>, String> {
        let sources = recipe.sources.iter().map(|source| source.flat()).collect::<Vec<_>>();
        let terminal_offsets = &recipe.terminal_offsets;
        let slots = &recipe.slots;
        let parent = sources[0];
        let mut rules = parent.rules.to_vec();
        let parent_rules = rules.len();
        let mut names = parent.nonterminal_names.clone();
        let mut counts = parent.component_nonterminals.clone();
        let mut ignores = parent.scoped_ignores.clone();
        let mut controls = parent.control_terminals.clone();
        if recipe.sources[0].root_nullable && !rules.iter().any(|r| r.lhs == parent.root() && r.rhs.is_empty()) {
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
            if recipe.sources[index].root_nullable && !rules.iter().any(|r| r.lhs == root && r.rhs.is_empty()) {
                rules.push(Rule { lhs: root, rhs: vec![] });
            }
            names.extend(child.nonterminal_names.iter().map(|name| format!("child{}::{name}", index-1)));
            counts.extend_from_slice(&child.component_nonterminals);
            ignores.extend(child.scoped_ignores.iter().map(|row|
                row.iter().map(|&t| t + terminal_offset).collect()));
            controls.extend(child.control_terminals.iter().map(|&t| t + terminal_offset));
            terminals = terminal_offset.checked_add(child.terminal_count).ok_or("composition terminal overflow")?;
        }
        let mut embedded_end_token_ids = recipe.sources.iter()
            .flat_map(|source| source.embedded_end_token_ids.iter().copied()).collect::<Vec<_>>();
        embedded_end_token_ids.sort_unstable();
        embedded_end_token_ids.dedup();
        let summaries = sources.iter().map(|source| source.stack_effects.as_ref()).collect::<Option<Vec<_>>>();
        let stack_effects = summaries.and_then(|summaries| {
            let offsets = &recipe.state_offsets;
            let states = recipe.state_count;
            let links = &recipe.effect_links;
            crate::compiler::boundary_stack_support::CompilerEffects::compose(
                &summaries,offsets,terminal_offsets,links,states,terminals).ok()
        });
        let grammar = Self { rules: Arc::from(rules), nonterminal_names: names,
            terminal_count: terminals, control_terminals: controls, root_nullable: parent.root_nullable,
            embedded_end_token_ids,
            component_nonterminals: counts, scoped_ignores: ignores, stack_effects };
        grammar.validate()?;
        Ok(Arc::new(grammar))
    }
}
