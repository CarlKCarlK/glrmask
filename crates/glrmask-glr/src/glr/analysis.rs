use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::time::Instant;

use rustc_hash::FxHashSet;

use crate::ds::bitset::BitSet;
use crate::grammar::flat::{GrammarDef, NonterminalID, Rule, Symbol, Terminal, TerminalID};

pub const EOF: TerminalID = u32::MAX;

#[cfg(test)]
#[path = "analysis/native_normalization_tests.rs"]
mod native_normalization_tests;

fn compile_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
}


fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

fn emit_normalize_profile(
    stage: &str,
    iteration: Option<usize>,
    elapsed_ms: f64,
    rules_before: usize,
    rules_after: usize,
    extra: &str,
) {
    match iteration {
        Some(iteration) => eprintln!(
            "[glrmask-profile] normalize_grammar stage={} iteration={} ms={:.3} rules_before={} rules_after={}{}",
            stage,
            iteration,
            elapsed_ms,
            rules_before,
            rules_after,
            extra,
        ),
        None => eprintln!(
            "[glrmask-profile] normalize_grammar stage={} ms={:.3} rules_before={} rules_after={}{}",
            stage,
            elapsed_ms,
            rules_before,
            rules_after,
            extra,
        ),
    }
}

fn emit_inline_null_profile(
    stage: &str,
    elapsed_ms: f64,
    rules_before: usize,
    rules_after: usize,
    extra: &str,
) {
    eprintln!(
        "[glrmask-profile] inline_null_productions stage={} ms={:.3} rules_before={} rules_after={}{}",
        stage,
        elapsed_ms,
        rules_before,
        rules_after,
        extra,
    );
}

#[derive(Debug, Clone)]
pub struct AnalyzedGrammar {
    pub rules: Vec<Rule>,
    pub num_terminals: u32,
    pub terminal_display_names: Vec<String>,
    /// Terminal columns whose original LR shift/reduce form must remain intact.
    /// Special-token columns can be subgrammar call sentinels or EOS controls,
    /// so lowering them into origin-dependent stack effects would destroy the
    /// composable call-site boundary.
    pub protected_shift_terminals: BitSet,
    pub num_nonterminals: u32,
    pub nonterminal_display_names: Vec<String>,
    pub residual_isolation_classes: BTreeMap<TerminalID, u32>,
    pub requires_global_terminal_observation: bool,
    pub direct_regular_automaton: Option<crate::grammar::flat::DirectRegularAutomaton>,
    pub nullable: BTreeSet<NonterminalID>,
    pub first: Vec<BitSet>,
    pub follow: Vec<BitSet>,
    /// Index: nonterminal -> list of rule indices with that nonterminal as LHS.
    pub rules_by_lhs: Vec<Vec<u32>>,
}

impl AnalyzedGrammar {
    pub fn from_grammar_def(g: &GrammarDef) -> Self {
        if g.direct_regular_automaton.is_some() {
            return Self::from_direct_regular_grammar_def(g);
        }

        let num_terminals = g.num_terminals();
        let mut protected_shift_terminals = BitSet::new(num_terminals as usize);
        for terminal in &g.terminals {
            if let Terminal::SpecialToken { id, .. } = terminal {
                protected_shift_terminals.set(*id as usize);
            }
        }
        let mut rules = Vec::with_capacity(g.rules.len() + 1);
        let augmented_start = g.num_nonterminals();
        rules.push(Rule {
            lhs: augmented_start,
            rhs: vec![Symbol::Nonterminal(g.start)],
        });
        rules.extend(g.rules.iter().cloned());

        let num_nonterminals = augmented_start + 1;
        let nullable = compute_nullable(&rules, num_nonterminals);
        let first = compute_first(&rules, num_nonterminals, num_terminals, &nullable);
        let follow = compute_follow(
            &rules,
            num_nonterminals,
            num_terminals,
            augmented_start,
            &first,
            &nullable,
        );

        let mut rules_by_lhs = vec![Vec::new(); num_nonterminals as usize];
        for (i, r) in rules.iter().enumerate() {
            if (r.lhs as usize) < rules_by_lhs.len() {
                rules_by_lhs[r.lhs as usize].push(i as u32);
            }
        }

        Self {
            rules,
            num_terminals,
            terminal_display_names: (0..num_terminals)
                .map(|terminal| g.terminal_display_name(terminal))
                .collect(),
            protected_shift_terminals,
            num_nonterminals,
            nonterminal_display_names: (0..num_nonterminals)
                .map(|nonterminal| {
                    if nonterminal == augmented_start {
                        "<augmented-start>".to_string()
                    } else {
                        g.nonterminal_names
                            .get(&nonterminal)
                            .cloned()
                            .unwrap_or_else(|| format!("N{nonterminal}"))
                    }
                })
                .collect(),
            residual_isolation_classes: g.residual_isolation_classes.clone(),
            requires_global_terminal_observation: g.requires_global_terminal_observation,
            direct_regular_automaton: g.direct_regular_automaton.clone(),
            nullable,
            first,
            follow,
            rules_by_lhs,
        }
    }

    /// Reconstruct the grammar analysis needed by terminal/template/parser-DWA
    /// construction from an already-composed parse table. `rules` must include
    /// the parent's augmented-start rule and `augmented_start` must name its
    /// LHS. This deliberately omits direct-regular shortcuts: composition
    /// reuses the component parser artifacts and builds only the boundary
    /// repair against the ordinary merged table semantics.
    pub fn from_composed_rules(
        rules: Vec<Rule>,
        num_terminals: u32,
        terminal_display_names: Vec<String>,
        nonterminal_display_names: Vec<String>,
        augmented_start: NonterminalID,
    ) -> Self {
        let num_nonterminals = nonterminal_display_names.len() as u32;
        let nullable = compute_nullable(&rules, num_nonterminals);
        let first = compute_first(&rules, num_nonterminals, num_terminals, &nullable);
        let follow = compute_follow(
            &rules,
            num_nonterminals,
            num_terminals,
            augmented_start,
            &first,
            &nullable,
        );
        let mut rules_by_lhs = vec![Vec::new(); num_nonterminals as usize];
        for (index, rule) in rules.iter().enumerate() {
            if let Some(bucket) = rules_by_lhs.get_mut(rule.lhs as usize) {
                bucket.push(index as u32);
            }
        }
        Self {
            rules,
            num_terminals,
            terminal_display_names,
            protected_shift_terminals: BitSet::new(num_terminals as usize),
            num_nonterminals,
            nonterminal_display_names,
            residual_isolation_classes: BTreeMap::new(),
            // The boundary compiler must preserve every component residual. A
            // global observation request is conservative and exact.
            requires_global_terminal_observation: true,
            direct_regular_automaton: None,
            nullable,
            first,
            follow,
            rules_by_lhs,
        }
    }

    pub fn from_grammar_def_with_protected_shift_terminals(
        g: &GrammarDef,
        protected: impl IntoIterator<Item = TerminalID>,
    ) -> Self {
        let mut analyzed = Self::from_grammar_def(g);
        for terminal in protected {
            if terminal < analyzed.num_terminals {
                analyzed.protected_shift_terminals.set(terminal as usize);
            }
        }
        analyzed
    }

    fn from_direct_regular_grammar_def(g: &GrammarDef) -> Self {
        let num_terminals = g.num_terminals();
        let mut protected_shift_terminals = BitSet::new(num_terminals as usize);
        for terminal in &g.terminals {
            if let Terminal::SpecialToken { id, .. } = terminal {
                protected_shift_terminals.set(*id as usize);
            }
        }
        Self {
            rules: Vec::new(),
            num_terminals,
            terminal_display_names: (0..num_terminals)
                .map(|terminal| g.terminal_display_name(terminal))
                .collect(),
            protected_shift_terminals,
            num_nonterminals: 0,
            nonterminal_display_names: Vec::new(),
            residual_isolation_classes: g.residual_isolation_classes.clone(),
            requires_global_terminal_observation: g.requires_global_terminal_observation,
            direct_regular_automaton: g.direct_regular_automaton.clone(),
            nullable: BTreeSet::new(),
            first: Vec::new(),
            follow: Vec::new(),
            rules_by_lhs: Vec::new(),
        }
    }

    pub fn terminal_display_name(&self, terminal: TerminalID) -> &str {
        self.terminal_display_names
            .get(terminal as usize)
            .map(String::as_str)
            .unwrap_or("<unknown-terminal>")
    }

    /// Assert the pre-table-build grammar is in the normal form required by
    /// GLR table construction and downstream characterization.
    pub fn check_table_build_normal_form(&self) -> Result<(), String> {
        let mut violations: Vec<String> = Vec::new();
        if let Err(msg) = self.check_no_nullable_nonterminals() {
            violations.push(msg);
        }
        if let Err(msg) = self.check_no_reachable_zero_length_productions() {
            violations.push(msg);
        }
        if let Err(msg) = self.check_recursion_boundedness() {
            violations.push(msg);
        }
        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations.join("\n"))
        }
    }

    /// The dynamic GLR runtime does not run terminal characterization. Its
    /// table builder requires epsilon-free reachable productions, but ordinary
    /// CFG right/indirect-left recursion is parser language structure rather
    /// than a characterization-boundedness requirement. Keep this separate
    /// from the static normal form so experiments cannot silently weaken the
    /// characterization contract.
    pub fn check_dynamic_table_build_normal_form(&self) -> Result<(), String> {
        let mut violations = Vec::new();
        if let Err(msg) = self.check_no_nullable_nonterminals() {
            violations.push(msg);
        }
        if let Err(msg) = self.check_no_reachable_zero_length_productions() {
            violations.push(msg);
        }
        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations.join("\n"))
        }
    }

    pub fn debug_check_grammar_preconditions(&self) -> Result<(), String> {
        self.check_table_build_normal_form()
    }

    pub fn check_no_nullable_nonterminals(&self) -> Result<(), String> {
        let reachable = self.reachable_nonterminals();
        let synthetic_start = self.num_nonterminals.saturating_sub(1);
        if !self.nullable.is_empty() {
            let ids: Vec<u32> = self
                .nullable
                .iter()
                .filter(|&&nt| nt != synthetic_start && reachable.contains(&nt))
                .copied()
                .collect();
            if !ids.is_empty() {
                return Err(format!(
                    "nullable nonterminals reachable at the table-build boundary: {:?}. \
                     Rules with epsilon-productions or all-nullable RHS create \
                     reduce chains that the characterisation stage cannot \
                     handle when combined with recursion.",
                    ids,
                ));
            }
        }
        Ok(())
    }

    pub fn check_no_reachable_zero_length_productions(&self) -> Result<(), String> {
        let reachable = self.reachable_nonterminals();
        let zero_len_rules: Vec<String> = self
            .rules
            .iter()
            .enumerate()
            .filter(|(_, rule)| reachable.contains(&rule.lhs) && rule.rhs.is_empty())
            .map(|(index, rule)| format!("rule#{index}: lhs=N{}", rule.lhs))
            .collect();

        if zero_len_rules.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "zero-length productions reachable at the table-build boundary: {}",
                zero_len_rules.join(", ")
            ))
        }
    }

    pub fn check_recursion_boundedness(&self) -> Result<(), String> {
        let mut violations: Vec<String> = Vec::new();
        let reachable = self.reachable_nonterminals();

        let rr_graph = filter_graph_to_reachable(
            build_right_reachability_graph(&self.rules, &self.nullable),
            &reachable,
        );
        if let Some(cycle) = find_indirect_rr_cycle(&rr_graph) {
            violations.push(format!(
                "right-recursive cycle detected: {:?}. \
                 Right recursion causes unbounded reduce chains in \
                 terminal characterisation. Convert to left recursion \
                 or inline the cycle.",
                cycle,
            ));
        }

        let lr_graph = filter_graph_to_reachable(
            build_left_reachability_graph(&self.rules, &self.nullable),
            &reachable,
        );
        if let Some(cycle) = find_indirect_lr_cycle(&lr_graph) {
            if cycle.len() >= 2 {
                violations.push(format!(
                    "indirect left-recursive cycle detected: {:?}. \
                     Indirect left recursion may create unbounded GSS \
                     growth. Inline or rewrite the cycle.",
                    cycle,
                ));
            }
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations.join("\n"))
        }
    }

    fn reachable_nonterminals(&self) -> BTreeSet<NonterminalID> {
        let synthetic_start = self.num_nonterminals.saturating_sub(1);
        let mut reachable = BTreeSet::from([synthetic_start]);
        let mut queue = VecDeque::from([synthetic_start]);

        while let Some(nonterminal) = queue.pop_front() {
            for &rule_index in self.rules_by_lhs.get(nonterminal as usize).into_iter().flatten() {
                let rule = &self.rules[rule_index as usize];
                for next_nonterminal in rule.rhs.iter().filter_map(|symbol| match symbol {
                    Symbol::Nonterminal(nonterminal) => Some(*nonterminal),
                    Symbol::Terminal(_) => None,
                }) {
                    if reachable.insert(next_nonterminal) {
                        queue.push_back(next_nonterminal);
                    }
                }
            }
        }

        reachable
    }
}

/// Eliminate right recursion by first inlining indirect cycles and then
/// rewriting direct right recursion into left recursion.
pub fn eliminate_right_recursion(
    rules: &mut Vec<Rule>,
    fresh_nt: &mut impl FnMut() -> NonterminalID,
) -> bool {
    // Resolve indirect right recursion by inlining right ends.
    const MAX_INDIRECT_ROUNDS: usize = 200;
    let num_nt = max_nt_id(rules) + 1;
    let nullable = compute_nullable(rules, num_nt);
    let mut completed = !has_indirect_right_recursion_fast(rules, &nullable, num_nt);
    if !completed {
        for _ in 0..MAX_INDIRECT_ROUNDS {
            let num_nt = max_nt_id(rules) + 1;
            let nullable = compute_nullable(rules, num_nt);
            let graph = build_right_reachability_graph(rules, &nullable);
            match find_cycle_excluding_self_loops(&graph) {
                Some(cycle) => {
                    let from = cycle[0];
                    let to = cycle[1 % cycle.len()];
                    inline_right_end(rules, from, to, &nullable);
                }
                None => {
                    completed = true;
                    break;
                }
            }
        }
    }

    // Resolve direct right recursion for all nonterminals in a single pass.

    let rr_nts: BTreeMap<NonterminalID, NonterminalID> = rules
        .iter()
        .filter(|r| is_direct_right_recursive(r))
        .map(|r| r.lhs)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|nt| (nt, fresh_nt()))
        .collect();

    if !rr_nts.is_empty() {
        resolve_direct_rr_batched(rules, &rr_nts);
    }
    completed
}

/// Exact linear-time precheck for indirect right-recursion cycles.
///
/// The rewrite path needs an explicit cycle and retains the ordered graph
/// implementation below. Most normalization passes have no indirect cycle at
/// all, however. In that common case, building `BTreeMap<BTreeSet<_>>` rows is
/// pure overhead on large generated grammars. Kahn elimination over a compact
/// adjacency vector proves acyclicity exactly; direct self-loops are ignored
/// because the batched direct-right-recursion rewrite handles them separately.
fn has_indirect_right_recursion_fast(
    rules: &[Rule],
    nullable: &BTreeSet<NonterminalID>,
    num_nt: NonterminalID,
) -> bool {
    let num_nt = num_nt as usize;
    let mut edges = vec![Vec::<NonterminalID>::new(); num_nt];
    let mut indegree = vec![0u32; num_nt];

    for rule in rules {
        let lhs = rule.lhs as usize;
        for symbol in rule.rhs.iter().rev() {
            match symbol {
                Symbol::Nonterminal(target) => {
                    if *target != rule.lhs {
                        edges[lhs].push(*target);
                        indegree[*target as usize] += 1;
                    }
                    if !nullable.contains(target) {
                        break;
                    }
                }
                Symbol::Terminal(_) => break,
            }
        }
    }

    let mut queue = VecDeque::with_capacity(num_nt);
    for (node, &degree) in indegree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(node as NonterminalID);
        }
    }
    let mut removed = 0usize;
    while let Some(node) = queue.pop_front() {
        removed += 1;
        for &target in &edges[node as usize] {
            let degree = &mut indegree[target as usize];
            *degree -= 1;
            if *degree == 0 {
                queue.push_back(target);
            }
        }
    }
    removed != num_nt
}

fn max_nt_id(rules: &[Rule]) -> u32 {
    rules
        .iter()
        .flat_map(|rule| {
            std::iter::once(rule.lhs).chain(rule.rhs.iter().filter_map(|symbol| match symbol {
                Symbol::Nonterminal(nonterminal) => Some(*nonterminal),
                Symbol::Terminal(_) => None,
            }))
        })
        .max()
        .unwrap_or(0)
}

fn add_boundary_nonterminals<'a>(
    symbols: impl Iterator<Item = &'a Symbol>,
    nullable: &BTreeSet<NonterminalID>,
    targets: &mut BTreeSet<NonterminalID>,
) {
    for symbol in symbols {
        match symbol {
            Symbol::Nonterminal(nonterminal) => {
                targets.insert(*nonterminal);
                if !nullable.contains(nonterminal) {
                    break;
                }
            }
            Symbol::Terminal(_) => break,
        }
    }
}

fn build_right_reachability_graph(
    rules: &[Rule],
    nullable: &BTreeSet<NonterminalID>,
) -> BTreeMap<NonterminalID, BTreeSet<NonterminalID>> {
    let mut graph = BTreeMap::<NonterminalID, BTreeSet<NonterminalID>>::new();
    for rule in rules {
        add_boundary_nonterminals(
            rule.rhs.iter().rev(),
            nullable,
            graph.entry(rule.lhs).or_default(),
        );
    }
    graph
}

fn find_indirect_rr_cycle(
    graph: &BTreeMap<NonterminalID, BTreeSet<NonterminalID>>,
) -> Option<Vec<NonterminalID>> {
    find_cycle(graph, 1, false)
}

fn find_cycle(
    graph: &BTreeMap<NonterminalID, BTreeSet<NonterminalID>>,
    min_cycle_len: usize,
    skip_self_loops: bool,
) -> Option<Vec<NonterminalID>> {
    fn dfs(
        node: NonterminalID,
        graph: &BTreeMap<NonterminalID, BTreeSet<NonterminalID>>,
        colors: &mut BTreeMap<NonterminalID, u8>,
        stack: &mut Vec<NonterminalID>,
        min_cycle_len: usize,
        skip_self_loops: bool,
    ) -> Option<Vec<NonterminalID>> {
        colors.insert(node, 1);
        stack.push(node);
        for &next in graph.get(&node).into_iter().flatten() {
            if skip_self_loops && next == node {
                continue;
            }

            match colors.get(&next).copied().unwrap_or(0) {
                0 => {
                    if let Some(cycle) = dfs(
                        next,
                        graph,
                        colors,
                        stack,
                        min_cycle_len,
                        skip_self_loops,
                    ) {
                        return Some(cycle);
                    }
                }
                1 => {
                    if let Some(start) = stack.iter().position(|&entry| entry == next) {
                        let cycle = stack[start..].to_vec();
                        if cycle.len() >= min_cycle_len {
                            return Some(cycle);
                        }
                    }
                }
                _ => {}
            }
        }
        stack.pop();
        colors.insert(node, 2);
        None
    }

    let mut colors = BTreeMap::new();
    let mut stack = Vec::new();
    for &node in graph.keys() {
        if colors.get(&node).copied().unwrap_or(0) == 0 {
            if let Some(cycle) = dfs(
                node,
                graph,
                &mut colors,
                &mut stack,
                min_cycle_len,
                skip_self_loops,
            ) {
                return Some(cycle);
            }
        }
    }
    None
}

/// Build a graph where an edge A → B means B appears at the left edge of
/// a production for A (possibly after nullable symbols).
fn build_left_reachability_graph(
    rules: &[Rule],
    nullable: &BTreeSet<NonterminalID>,
) -> BTreeMap<NonterminalID, BTreeSet<NonterminalID>> {
    let mut graph = BTreeMap::<NonterminalID, BTreeSet<NonterminalID>>::new();
    for rule in rules {
        add_boundary_nonterminals(rule.rhs.iter(), nullable, graph.entry(rule.lhs).or_default());
    }
    graph
}

/// Find an indirect left-recursive cycle (length ≥ 2) in the left-reachability
/// graph.  Direct self-loops (A → A …) are excluded — they are fine for GLR.
fn find_indirect_lr_cycle(
    graph: &BTreeMap<NonterminalID, BTreeSet<NonterminalID>>,
) -> Option<Vec<NonterminalID>> {
    find_cycle(graph, 2, false)
}

pub fn has_indirect_left_recursion(rules: &[Rule]) -> bool {
    let nullable = compute_nullable(rules, max_nt_id(rules) + 1);
    has_indirect_left_recursion_with_nullable(rules, &nullable)
}

fn has_indirect_left_recursion_with_nullable(
    rules: &[Rule],
    nullable: &BTreeSet<NonterminalID>,
) -> bool {
    let num_nonterminals = (max_nt_id(rules) + 1) as usize;
    let mut nullable_mask = vec![false; num_nonterminals];
    for &nonterminal in nullable {
        nullable_mask[nonterminal as usize] = true;
    }

    let mut successors = vec![Vec::<NonterminalID>::new(); num_nonterminals];
    let mut indegree = vec![0usize; num_nonterminals];
    for rule in rules {
        for symbol in &rule.rhs {
            let Symbol::Nonterminal(target) = symbol else {
                break;
            };
            if *target != rule.lhs {
                successors[rule.lhs as usize].push(*target);
                indegree[*target as usize] += 1;
            }
            if !nullable_mask[*target as usize] {
                break;
            }
        }
    }

    let mut queue = VecDeque::new();
    for (node, &degree) in indegree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(node as NonterminalID);
        }
    }

    let mut removed = 0usize;
    while let Some(node) = queue.pop_front() {
        removed += 1;
        for &successor in &successors[node as usize] {
            let degree = &mut indegree[successor as usize];
            *degree -= 1;
            if *degree == 0 {
                queue.push_back(successor);
            }
        }
    }
    removed != num_nonterminals
}

fn find_nontrivial_sccs(
    graph: &BTreeMap<NonterminalID, BTreeSet<NonterminalID>>,
) -> Vec<BTreeSet<NonterminalID>> {
    let mut nodes: Vec<NonterminalID> = graph.keys().copied().collect();
    for neighbors in graph.values() {
        nodes.extend(neighbors.iter().copied());
    }
    nodes.sort_unstable();
    nodes.dedup();

    if nodes.is_empty() {
        return Vec::new();
    }

    let node_to_idx: HashMap<NonterminalID, usize> = nodes
        .iter()
        .enumerate()
        .map(|(index, &node)| (node, index))
        .collect();
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    let mut reverse_adjacency: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];

    for (&node, neighbors) in graph {
        let from_index = node_to_idx[&node];
        for &neighbor in neighbors {
            let to_index = node_to_idx[&neighbor];
            adjacency[from_index].push(to_index);
            reverse_adjacency[to_index].push(from_index);
        }
    }

    let mut visited = vec![false; nodes.len()];
    let mut finish_order = Vec::with_capacity(nodes.len());
    for start in 0..nodes.len() {
        if visited[start] {
            continue;
        }
        let mut stack = vec![(start, 0usize)];
        visited[start] = true;
        while let Some((node_index, neighbor_index)) = stack.last_mut() {
            if *neighbor_index < adjacency[*node_index].len() {
                let next = adjacency[*node_index][*neighbor_index];
                *neighbor_index += 1;
                if !visited[next] {
                    visited[next] = true;
                    stack.push((next, 0));
                }
            } else {
                finish_order.push(*node_index);
                stack.pop();
            }
        }
    }

    visited.fill(false);
    let mut sccs = Vec::new();
    for &start in finish_order.iter().rev() {
        if visited[start] {
            continue;
        }
        let mut component_indices = Vec::new();
        let mut stack = vec![start];
        visited[start] = true;
        while let Some(node_index) = stack.pop() {
            component_indices.push(node_index);
            for &next in &reverse_adjacency[node_index] {
                if !visited[next] {
                    visited[next] = true;
                    stack.push(next);
                }
            }
        }

        if component_indices.len() >= 2 {
            let component: BTreeSet<NonterminalID> = component_indices
                .into_iter()
                .map(|node_index| nodes[node_index])
                .collect();
            sccs.push(component);
        }
    }

    sccs
}

/// Find a cycle of length ≥ 2 in the graph (self-loops are skipped).
/// Used by `eliminate_right_recursion` to find indirect cycles.
fn find_cycle_excluding_self_loops(
    graph: &BTreeMap<NonterminalID, BTreeSet<NonterminalID>>,
) -> Option<Vec<NonterminalID>> {
    find_cycle(graph, 2, true)
}

/// Inline right-end: for rules `from_nt → α to_nt β` where β is all-nullable,
/// replace the `to_nt` occurrence with each of `to_nt`'s alternative RHSs.
///
/// This breaks indirect right-recursive cycles by removing the edge
/// `from_nt → to_nt` in the right-reachability graph.
fn inline_right_end(
    rules: &mut Vec<Rule>,
    from_nt: NonterminalID,
    to_nt: NonterminalID,
    nullable: &BTreeSet<NonterminalID>,
) {
    let to_rhss: Vec<Vec<Symbol>> = rules
        .iter()
        .filter(|r| r.lhs == to_nt)
        .map(|r| r.rhs.clone())
        .collect();
    if to_rhss.is_empty() {
        return;
    }

    let mut new_rules = Vec::new();
    for rule in rules.iter() {
        if rule.lhs != from_nt {
            new_rules.push(rule.clone());
            continue;
        }
        let pos = find_right_end_position(&rule.rhs, to_nt, nullable);
        if let Some(pos) = pos {
            for to_rhs in &to_rhss {
                let mut rhs = rule.rhs[..pos].to_vec();
                rhs.extend(to_rhs.iter().cloned());
                rhs.extend(rule.rhs[pos + 1..].iter().cloned());
                new_rules.push(Rule { lhs: from_nt, rhs });
            }
        } else {
            new_rules.push(rule.clone());
        }
    }
    *rules = new_rules;
}

/// Find the rightmost position of `target_nt` in `rhs` such that everything
/// after it is a nullable nonterminal.  Returns `None` if no such position.
fn find_right_end_position(
    rhs: &[Symbol],
    target_nt: NonterminalID,
    nullable: &BTreeSet<NonterminalID>,
) -> Option<usize> {
    for i in (0..rhs.len()).rev() {
        match &rhs[i] {
            Symbol::Nonterminal(nt) if *nt == target_nt => return Some(i),
            Symbol::Nonterminal(nt) if nullable.contains(nt) => continue,
            _ => return None,
        }
    }
    None
}

fn is_direct_right_recursive(rule: &Rule) -> bool {
    matches!(rule.rhs.last(), Some(Symbol::Nonterminal(nonterminal)) if *nonterminal == rule.lhs)
}

/// Resolve direct right recursion for a single nonterminal.
///
/// Given recursive rules `A → α A` and base rules `A → β`, transform to:
/// - Base rules (unchanged): `A → β`
/// - Composed rules: `A → new_nt β` (for each base rule)
/// - Tail rules: `new_nt → α` (body of each recursive rule, without trailing A)
/// - Left-recursive tails: `new_nt → new_nt α`
///
/// Note: if α is empty (rule `A → A`), this produces `new_nt → ε`.
/// The subsequent ε-elimination pass handles that.
fn resolve_direct_rr_single_nt(
    rules: &mut Vec<Rule>,
    nt: NonterminalID,
    new_nt: NonterminalID,
) {
    let (recursive, non_recursive): (Vec<Rule>, Vec<Rule>) = rules
        .iter()
        .filter(|r| r.lhs == nt)
        .cloned()
        .partition(|r| is_direct_right_recursive(r));

    if recursive.is_empty() {
        return;
    }

    // Keep all rules NOT for this NT
    let mut new_rules: Vec<Rule> = rules.iter().filter(|r| r.lhs != nt).cloned().collect();

    // Keep base rules: A → β
    new_rules.extend(non_recursive.iter().cloned());

    // Add A → new_nt β for each base rule
    for base in &non_recursive {
        let mut rhs = vec![Symbol::Nonterminal(new_nt)];
        rhs.extend(base.rhs.iter().cloned());
        new_rules.push(Rule { lhs: nt, rhs });
    }

    // Add new_nt → α (body without trailing A) for each recursive rule
    for rec in &recursive {
        let body = rec.rhs[..rec.rhs.len() - 1].to_vec();
        new_rules.push(Rule { lhs: new_nt, rhs: body });
    }

    // Add new_nt → new_nt α (left-recursive) for each recursive rule
    for rec in &recursive {
        let body = &rec.rhs[..rec.rhs.len() - 1];
        let mut rhs = vec![Symbol::Nonterminal(new_nt)];
        rhs.extend(body.iter().cloned());
        new_rules.push(Rule { lhs: new_nt, rhs });
    }

    *rules = new_rules;
}

/// Resolve direct right recursion for multiple nonterminals in a single pass.
///
/// Each entry maps a right-recursive NT to its fresh replacement NT.
/// Equivalent to calling `resolve_direct_rr_single_nt` for each NT independently,
/// but avoids the O(NTs × rules) cost of repeated full-vector rebuilds.
fn resolve_direct_rr_batched(
    rules: &mut Vec<Rule>,
    rr_map: &BTreeMap<NonterminalID, NonterminalID>,
) {
    // Partition rules by whether they belong to a right-recursive NT.
    let mut recursive_by_nt: BTreeMap<NonterminalID, Vec<Rule>> = BTreeMap::new();
    let mut non_recursive_by_nt: BTreeMap<NonterminalID, Vec<Rule>> = BTreeMap::new();
    let mut new_rules = Vec::with_capacity(rules.len() * 2);

    for rule in rules.iter() {
        if rr_map.contains_key(&rule.lhs) {
            if is_direct_right_recursive(rule) {
                recursive_by_nt.entry(rule.lhs).or_default().push(rule.clone());
            } else {
                non_recursive_by_nt.entry(rule.lhs).or_default().push(rule.clone());
            }
        } else {
            new_rules.push(rule.clone());
        }
    }

    for (&nt, &new_nt) in rr_map {
        let rec_rules = recursive_by_nt.remove(&nt).unwrap_or_default();
        let base_rules = non_recursive_by_nt.remove(&nt).unwrap_or_default();

        if rec_rules.is_empty() {
            new_rules.extend(base_rules);
            continue;
        }

        // Keep base rules: A → β
        new_rules.extend(base_rules.iter().cloned());

        // Add A → new_nt β for each base rule
        for base in &base_rules {
            let mut rhs = vec![Symbol::Nonterminal(new_nt)];
            rhs.extend(base.rhs.iter().cloned());
            new_rules.push(Rule { lhs: nt, rhs });
        }

        // Add new_nt → α (body without trailing A) for each recursive rule
        for rec in &rec_rules {
            let body = rec.rhs[..rec.rhs.len() - 1].to_vec();
            new_rules.push(Rule { lhs: new_nt, rhs: body });
        }

        // Add new_nt → new_nt α (left-recursive) for each recursive rule
        for rec in &rec_rules {
            let body = &rec.rhs[..rec.rhs.len() - 1];
            let mut rhs = vec![Symbol::Nonterminal(new_nt)];
            rhs.extend(body.iter().cloned());
            new_rules.push(Rule { lhs: new_nt, rhs });
        }
    }

    *rules = new_rules;
}

fn inline_null_productions_exhaustive(rules: &[Rule], num_nt: u32) -> Vec<Rule> {
    let profile_enabled = compile_profile_enabled();
    let total_started_at = profile_enabled.then(Instant::now);
    let rules_before = rules.len();

    let compute_nullable_started_at = profile_enabled.then(Instant::now);
    let nullable = compute_nullable(rules, num_nt);
    if let Some(started_at) = compute_nullable_started_at {
        let extra = format!(" nullable_count={}", nullable.len());
        emit_inline_null_profile(
            "exhaustive_compute_nullable",
            elapsed_ms(started_at),
            rules_before,
            rules_before,
            &extra,
        );
    }
    if nullable.is_empty() {
        let result = rules.to_vec();
        if let Some(started_at) = total_started_at {
            emit_inline_null_profile(
                "inline_null_productions_exhaustive",
                elapsed_ms(started_at),
                rules_before,
                result.len(),
                &format!(
                    " nullable_count=0 scanned_rules=0 total_nullable_positions=0 max_nullable_positions=0 output_rules={} shortcut=nullable_empty",
                    result.len(),
                ),
            );
        }
        return result;
    }

    if profile_enabled {
        let scan_started_at = Instant::now();
        let nullable_positions_by_rule: Vec<Vec<usize>> = rules
            .iter()
            .map(|rule| {
                rule.rhs
                    .iter()
                    .enumerate()
                    .filter_map(|(i, sym)| match sym {
                        Symbol::Nonterminal(nt) if nullable.contains(nt) => Some(i),
                        _ => None,
                    })
                    .collect()
            })
            .collect();
        let total_nullable_positions: usize = nullable_positions_by_rule
            .iter()
            .map(Vec::len)
            .sum();
        let max_nullable_positions = nullable_positions_by_rule
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0);
        emit_inline_null_profile(
            "exhaustive_nullable_position_scan",
            elapsed_ms(scan_started_at),
            rules_before,
            rules_before,
            &format!(
                " nullable_count={} scanned_rules={} total_nullable_positions={} max_nullable_positions={}",
                nullable.len(),
                rules.len(),
                total_nullable_positions,
                max_nullable_positions,
            ),
        );

        let emit_started_at = Instant::now();
        let mut seen = HashSet::<Rule>::new();
        let mut out = Vec::new();

        for (rule, nullable_positions) in rules.iter().zip(nullable_positions_by_rule.iter()) {
            let k = nullable_positions.len();
            assert!(
                k <= 20,
                "production for NT {} has {} nullable positions; refusing power-set",
                rule.lhs, k,
            );

            for mask in 0u64..(1u64 << k) {
                let new_rhs: Vec<Symbol> = rule
                    .rhs
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| match nullable_positions.binary_search(i) {
                        Ok(idx) => mask & (1u64 << idx) != 0,
                        Err(_) => true,
                    })
                    .map(|(_, sym)| sym.clone())
                    .collect();

                if new_rhs.is_empty() {
                    continue;
                }

                let candidate = Rule {
                    lhs: rule.lhs,
                    rhs: new_rhs,
                };
                if seen.insert(candidate.clone()) {
                    out.push(candidate);
                }
            }
        }

        emit_inline_null_profile(
            "exhaustive_powerset_emit_and_dedup",
            elapsed_ms(emit_started_at),
            rules_before,
            out.len(),
            &format!(
                " nullable_count={} scanned_rules={} total_nullable_positions={} max_nullable_positions={} output_rules={}",
                nullable.len(),
                rules.len(),
                total_nullable_positions,
                max_nullable_positions,
                out.len(),
            ),
        );

        if let Some(started_at) = total_started_at {
            emit_inline_null_profile(
                "inline_null_productions_exhaustive",
                elapsed_ms(started_at),
                rules_before,
                out.len(),
                &format!(
                    " nullable_count={} scanned_rules={} total_nullable_positions={} max_nullable_positions={} output_rules={}",
                    nullable.len(),
                    rules.len(),
                    total_nullable_positions,
                    max_nullable_positions,
                    out.len(),
                ),
            );
        }

        return out;
    }

    let mut seen = HashSet::<Rule>::new();
    let mut out = Vec::new();

    for rule in rules {
        let nullable_positions: Vec<usize> = rule
            .rhs
            .iter()
            .enumerate()
            .filter_map(|(i, sym)| match sym {
                Symbol::Nonterminal(nt) if nullable.contains(nt) => Some(i),
                _ => None,
            })
            .collect();

        let k = nullable_positions.len();
        // Safety guard: refuse power-set expansion beyond 20 nullable positions
        assert!(
            k <= 20,
            "production for NT {} has {} nullable positions; refusing power-set",
            rule.lhs, k,
        );

        for mask in 0u64..(1u64 << k) {
            let new_rhs: Vec<Symbol> = rule
                .rhs
                .iter()
                .enumerate()
                .filter(|(i, _)| {
                    match nullable_positions.binary_search(i) {
                        Ok(idx) => mask & (1u64 << idx) != 0, // bit set → keep
                        Err(_) => true,                        // non-nullable → always keep
                    }
                })
                .map(|(_, sym)| sym.clone())
                .collect();

            // Drop ε-rules
            if new_rhs.is_empty() {
                continue;
            }

            let candidate = Rule {
                lhs: rule.lhs,
                rhs: new_rhs,
            };
            if seen.insert(candidate.clone()) {
                out.push(candidate);
            }
        }
    }

    out
}

fn find_nullable_runs(
    rhs: &[Symbol],
    nullable: &BTreeSet<NonterminalID>,
    threshold: usize,
) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut run_start = None;

    for (idx, symbol) in rhs.iter().enumerate() {
        let is_nullable = matches!(symbol, Symbol::Nonterminal(nt) if nullable.contains(nt));
        if is_nullable {
            if run_start.is_none() {
                run_start = Some(idx);
            }
        } else if let Some(start) = run_start.take() {
            let len = idx - start;
            if len > threshold {
                runs.push((start, idx - 1));
            }
        }
    }

    if let Some(start) = run_start {
        let len = rhs.len() - start;
        if len > threshold {
            runs.push((start, rhs.len() - 1));
        }
    }

    runs
}

fn compute_nonempty_productive(rules: &[Rule], num_nt: u32) -> BTreeSet<NonterminalID> {
    let mut productive_any = BTreeSet::new();
    let mut nonempty_productive = BTreeSet::new();
    let mut changed = true;
    while changed {
        changed = false;
        for rule in rules {
            if rule.lhs >= num_nt {
                continue;
            }

            let mut rhs_productive = true;
            let mut rhs_nonempty = false;
            for symbol in &rule.rhs {
                match symbol {
                    Symbol::Terminal(_) => rhs_nonempty = true,
                    Symbol::Nonterminal(nonterminal) => {
                        if !productive_any.contains(nonterminal) {
                            rhs_productive = false;
                            break;
                        }
                        rhs_nonempty |= nonempty_productive.contains(nonterminal);
                    }
                }
            }

            if rhs_productive {
                changed |= productive_any.insert(rule.lhs);
                if rhs_nonempty {
                    changed |= nonempty_productive.insert(rule.lhs);
                }
            }
        }
    }
    nonempty_productive
}

fn compress_nullable_runs_with_optional_tree(rules: &[Rule], num_nt: u32) -> Vec<Rule> {
    let profile_enabled = compile_profile_enabled();
    let total_started_at = profile_enabled.then(Instant::now);
    let rules_before = rules.len();

    let compute_nullable_started_at = profile_enabled.then(Instant::now);
    let nullable = compute_nullable(rules, num_nt);
    if let Some(started_at) = compute_nullable_started_at {
        emit_inline_null_profile(
            "compress_compute_nullable",
            elapsed_ms(started_at),
            rules_before,
            rules_before,
            &format!(" nullable_count={}", nullable.len()),
        );
    }
    if nullable.is_empty() {
        let result = rules.to_vec();
        if let Some(started_at) = total_started_at {
            emit_inline_null_profile(
                "compress_nullable_runs_with_optional_tree",
                elapsed_ms(started_at),
                rules_before,
                result.len(),
                &format!(
                    " nullable_count=0 run_count=0 max_run_len=0 output_rules={} shortcut=nullable_empty",
                    result.len(),
                ),
            );
        }
        return result;
    }

    let mut run_count = 0usize;
    let mut max_run_len = 0usize;
    let run_scan_started_at = profile_enabled.then(Instant::now);
    for rule in rules {
        for (start, end) in find_nullable_runs(&rule.rhs, &nullable, 1) {
            run_count += 1;
            max_run_len = max_run_len.max(end - start + 1);
        }
    }
    if let Some(started_at) = run_scan_started_at {
        emit_inline_null_profile(
            "compress_run_scan",
            elapsed_ms(started_at),
            rules_before,
            rules_before,
            &format!(
                " nullable_count={} run_count={} max_run_len={}",
                nullable.len(),
                run_count,
                max_run_len,
            ),
        );
    }
    if run_count == 0 {
        let result = rules.to_vec();
        if let Some(started_at) = total_started_at {
            emit_inline_null_profile(
                "compress_nullable_runs_with_optional_tree",
                elapsed_ms(started_at),
                rules_before,
                result.len(),
                &format!(
                    " nullable_count={} run_count=0 max_run_len=0 output_rules={} shortcut=no_runs",
                    nullable.len(),
                    result.len(),
                ),
            );
        }
        return result;
    }

    let compute_nonempty_started_at = profile_enabled.then(Instant::now);
    let nonempty_productive = compute_nonempty_productive(rules, num_nt);
    if let Some(started_at) = compute_nonempty_started_at {
        emit_inline_null_profile(
            "compress_compute_nonempty_productive",
            elapsed_ms(started_at),
            rules_before,
            rules_before,
            &format!(
                " nullable_count={} run_count={} max_run_len={} nonempty_productive_count={}",
                nullable.len(),
                run_count,
                max_run_len,
                nonempty_productive.len(),
            ),
        );
    }

    let by_lhs_started_at = profile_enabled.then(Instant::now);
    let mut by_lhs = BTreeMap::<NonterminalID, Vec<Vec<Symbol>>>::new();
    for rule in rules {
        by_lhs.entry(rule.lhs).or_default().push(rule.rhs.clone());
    }
    if let Some(started_at) = by_lhs_started_at {
        emit_inline_null_profile(
            "compress_build_by_lhs",
            elapsed_ms(started_at),
            rules_before,
            rules_before,
            &format!(
                " nullable_count={} run_count={} max_run_len={} lhs_count={}",
                nullable.len(),
                run_count,
                max_run_len,
                by_lhs.len(),
            ),
        );
    }

    let mut next_nt = max_nt_id(rules) + 1;
    let mut fresh_nt = || {
        let id = next_nt;
        next_nt += 1;
        id
    };

    let mut nn_cache = BTreeMap::<NonterminalID, NonterminalID>::new();
    let mut result = Vec::<Rule>::new();
    let tree_build_started_at = profile_enabled.then(Instant::now);
    for rule in rules {
        let runs = find_nullable_runs(&rule.rhs, &nullable, 1);
        if runs.is_empty() {
            result.push(rule.clone());
            continue;
        }

        let mut new_rhs = rule.rhs.clone();
        for &(start, end) in runs.iter().rev() {
            let segment: Vec<Symbol> = new_rhs.drain(start..=end).collect();
            let Some(root_nn) = build_non_nullable_tree(
                &segment,
                2,
                &mut fresh_nt,
                &mut result,
                &nullable,
                &nonempty_productive,
                &by_lhs,
                &mut nn_cache,
            ) else {
                continue;
            };
            let root_opt = fresh_nt();
            result.push(Rule {
                lhs: root_opt,
                rhs: vec![Symbol::Nonterminal(root_nn)],
            });
            result.push(Rule {
                lhs: root_opt,
                rhs: vec![],
            });
            new_rhs.insert(start, Symbol::Nonterminal(root_opt));
        }

        result.push(Rule {
            lhs: rule.lhs,
            rhs: new_rhs,
        });
    }
    if let Some(started_at) = tree_build_started_at {
        emit_inline_null_profile(
            "compress_tree_build_and_emit",
            elapsed_ms(started_at),
            rules_before,
            result.len(),
            &format!(
                " nullable_count={} run_count={} max_run_len={} output_rules={} fresh_nt_upper_bound={}",
                nullable.len(),
                run_count,
                max_run_len,
                result.len(),
                next_nt,
            ),
        );
    }

    let dedup_rules_before = result.len();
    let dedup_started_at = profile_enabled.then(Instant::now);
    dedup_rules(&mut result);
    if let Some(started_at) = dedup_started_at {
        emit_inline_null_profile(
            "compress_final_dedup_rules",
            elapsed_ms(started_at),
            dedup_rules_before,
            result.len(),
            &format!(
                " nullable_count={} run_count={} max_run_len={} output_rules={}",
                nullable.len(),
                run_count,
                max_run_len,
                result.len(),
            ),
        );
    }
    if let Some(started_at) = total_started_at {
        emit_inline_null_profile(
            "compress_nullable_runs_with_optional_tree",
            elapsed_ms(started_at),
            rules_before,
            result.len(),
            &format!(
                " nullable_count={} run_count={} max_run_len={} output_rules={}",
                nullable.len(),
                run_count,
                max_run_len,
                result.len(),
            ),
        );
    }
    result
}

fn build_non_nullable_tree(
    segment: &[Symbol],
    k: usize,
    fresh_nt: &mut impl FnMut() -> NonterminalID,
    new_rules: &mut Vec<Rule>,
    nullable: &BTreeSet<NonterminalID>,
    nonempty_productive: &BTreeSet<NonterminalID>,
    by_lhs: &BTreeMap<NonterminalID, Vec<Vec<Symbol>>>,
    nn_cache: &mut BTreeMap<NonterminalID, NonterminalID>,
) -> Option<NonterminalID> {
    let k = k.max(2);
    let n = segment.len();
    if n == 0 {
        return None;
    }

    let nn_segment: Vec<Symbol> = segment
        .iter()
        .filter_map(|symbol| match symbol {
            Symbol::Terminal(terminal) => Some(Symbol::Terminal(*terminal)),
            Symbol::Nonterminal(nonterminal) if !nullable.contains(nonterminal) => {
                Some(Symbol::Nonterminal(*nonterminal))
            }
            Symbol::Nonterminal(nonterminal) if nonempty_productive.contains(nonterminal) => {
                get_or_create_non_nullable_nt(
                    *nonterminal,
                    fresh_nt,
                    new_rules,
                    nullable,
                    nonempty_productive,
                    by_lhs,
                    nn_cache,
                )
                .map(Symbol::Nonterminal)
            }
            Symbol::Nonterminal(_) => None,
        })
        .collect();
    let n = nn_segment.len();
    if n == 0 {
        return None;
    }

    if n <= k {
        let leaf_nt = fresh_nt();
        for mask in 1u64..(1u64 << n) {
            let rhs: Vec<Symbol> = nn_segment
                .iter()
                .enumerate()
                .filter(|(idx, _)| ((mask >> idx) & 1) == 1)
                .map(|(_, symbol)| symbol.clone())
                .collect();
            new_rules.push(Rule { lhs: leaf_nt, rhs });
        }
        return Some(leaf_nt);
    }

    // Keep the default right-heavy decomposition; alternate shapes were only
    // used for internal experiments.
    let (first, rest) = nn_segment.split_at(1);
    let chunks: Vec<&[Symbol]> = if rest.is_empty() {
        vec![first]
    } else {
        vec![first, rest]
    };
    let chunk_nts: Vec<NonterminalID> = chunks
        .into_iter()
        .map(|chunk| {
            build_non_nullable_tree(
                chunk,
                k,
                fresh_nt,
                new_rules,
                nullable,
                nonempty_productive,
                by_lhs,
                nn_cache,
            )
            .expect("nonempty chunk should have a nonnullable tree")
        })
        .collect();
    let chunk_symbols: Vec<Symbol> = chunk_nts
        .into_iter()
        .map(Symbol::Nonterminal)
        .collect();
    build_non_nullable_tree(
        &chunk_symbols,
        k,
        fresh_nt,
        new_rules,
        nullable,
        nonempty_productive,
        by_lhs,
        nn_cache,
    )
}

fn get_or_create_non_nullable_nt(
    nt: NonterminalID,
    fresh_nt: &mut impl FnMut() -> NonterminalID,
    new_rules: &mut Vec<Rule>,
    nullable: &BTreeSet<NonterminalID>,
    nonempty_productive: &BTreeSet<NonterminalID>,
    by_lhs: &BTreeMap<NonterminalID, Vec<Vec<Symbol>>>,
    nn_cache: &mut BTreeMap<NonterminalID, NonterminalID>,
) -> Option<NonterminalID> {
    if !nullable.contains(&nt) {
        return Some(nt);
    }
    if !nonempty_productive.contains(&nt) {
        return None;
    }
    if let Some(&cached) = nn_cache.get(&nt) {
        return Some(cached);
    }

    let Some(alts) = by_lhs.get(&nt) else {
        return None;
    };
    let nn_nt = fresh_nt();
    nn_cache.insert(nt, nn_nt);
    let mut emitted = false;
    for rhs in alts {
        if rhs.is_empty() {
            continue;
        }
        if rhs.iter().any(|symbol| match symbol {
            Symbol::Terminal(_) => true,
            Symbol::Nonterminal(inner) => !nullable.contains(inner),
        }) {
            new_rules.push(Rule { lhs: nn_nt, rhs: rhs.clone() });
            emitted = true;
            continue;
        }

        let Some(rhs_nn) = build_non_nullable_tree(
            rhs,
            2,
            fresh_nt,
            new_rules,
            nullable,
            nonempty_productive,
            by_lhs,
            nn_cache,
        ) else {
            continue;
        };
        new_rules.push(Rule {
            lhs: nn_nt,
            rhs: vec![Symbol::Nonterminal(rhs_nn)],
        });
        emitted = true;
    }

    if !emitted {
        nn_cache.remove(&nt);
        return None;
    }
    Some(nn_nt)
}

/// Inline null productions (ε-elimination).
///
/// Preprocess long nullable runs with a balanced binary tree before doing the
/// existing exhaustive elimination, to avoid the raw power-set blowups that
/// occur when many nullable nonterminals appear consecutively.
pub fn inline_null_productions(rules: &[Rule], num_nt: u32) -> Vec<Rule> {
    let profile_enabled = compile_profile_enabled();
    let total_started_at = profile_enabled.then(Instant::now);
    let rules_before = rules.len();
    let preprocessed = compress_nullable_runs_with_optional_tree(rules, num_nt);
    let result = inline_null_productions_exhaustive(&preprocessed, max_nt_id(&preprocessed) + 1);
    if let Some(started_at) = total_started_at {
        emit_inline_null_profile(
            "inline_null_productions",
            elapsed_ms(started_at),
            rules_before,
            result.len(),
            &format!(
                " preprocessed_rules={} output_rules={}",
                preprocessed.len(),
                result.len(),
            ),
        );
    }
    result
}

/// Eliminate hidden left recursion.
///
/// Hidden left recursion occurs when `A → β B …` where every symbol in
/// `β` is nullable and `B` is in an indirect left-recursive cycle with `A`.
/// We add shortened rules with the nullable prefix removed, exposing the
/// left recursion so it becomes direct (which GLR handles natively).
fn eliminate_hidden_left_recursion(
    rules: &mut Vec<Rule>,
    nullable: &BTreeSet<NonterminalID>,
    normalize_iteration: usize,
) -> bool {
    let profile_enabled = compile_profile_enabled();
    let mut changed = false;

    loop {
        let rules_len = rules.len();
        let lr_graph_started_at = profile_enabled.then(Instant::now);
        let lr_graph = build_left_reachability_graph(rules, nullable);
        let build_left_reachability_graph_ms = lr_graph_started_at
            .map(elapsed_ms)
            .unwrap_or(0.0);
        let left_reachability_node_count = lr_graph.len();
        let left_reachability_edge_count: usize =
            lr_graph.values().map(BTreeSet::len).sum();
        let cycle_sccs = find_nontrivial_sccs(&lr_graph);
        if cycle_sccs.is_empty() && cfg!(debug_assertions) {
            debug_assert!(find_indirect_lr_cycle(&lr_graph).is_none());
        }
        if cycle_sccs.is_empty() {
            if profile_enabled {
                emit_normalize_profile(
                    "hidden_left_recursion_helper_counters",
                    Some(normalize_iteration),
                    build_left_reachability_graph_ms,
                    rules_len,
                    rules_len,
                    &format!(
                        " rules_len={} build_rhs_by_lhs_ms=0.000 rhs_lhs_bucket_count=0 rhs_total_entries=0 build_left_reachability_graph_ms={:.3} left_reachability_node_count={} left_reachability_edge_count={}",
                        rules_len,
                        build_left_reachability_graph_ms,
                        left_reachability_node_count,
                        left_reachability_edge_count,
                    ),
                );
            }
            return changed;
        }
        let rhs_by_lhs_started_at = profile_enabled.then(Instant::now);
        let rhs_by_lhs = build_rhs_by_lhs(rules);
        let build_rhs_by_lhs_ms = rhs_by_lhs_started_at.map(elapsed_ms).unwrap_or(0.0);
        let rhs_lhs_bucket_count = rhs_by_lhs.len();
        let rhs_total_entries: usize = rhs_by_lhs.values().map(BTreeSet::len).sum();

        if profile_enabled {
            emit_normalize_profile(
                "hidden_left_recursion_helper_counters",
                Some(normalize_iteration),
                build_left_reachability_graph_ms + build_rhs_by_lhs_ms,
                rules_len,
                rules_len,
                &format!(
                    " rules_len={} build_rhs_by_lhs_ms={:.3} rhs_lhs_bucket_count={} rhs_total_entries={} build_left_reachability_graph_ms={:.3} left_reachability_node_count={} left_reachability_edge_count={}",
                    rules_len,
                    build_rhs_by_lhs_ms,
                    rhs_lhs_bucket_count,
                    rhs_total_entries,
                    build_left_reachability_graph_ms,
                    left_reachability_node_count,
                    left_reachability_edge_count,
                ),
            );
        }

        let mut additions = Vec::new();
        let mut replaced_cycle_rules = HashSet::new();
        let cycle_scc_by_nt: HashMap<NonterminalID, usize> = cycle_sccs
            .iter()
            .enumerate()
            .flat_map(|(scc_index, cycle_nodes)| {
                cycle_nodes
                    .iter()
                    .copied()
                    .map(move |nonterminal| (nonterminal, scc_index))
            })
            .collect();
        for rule in rules.iter() {
            let Some(&scc_index) = cycle_scc_by_nt.get(&rule.lhs) else {
                continue;
            };
            let cycle_nodes = &cycle_sccs[scc_index];

            let prefix_end = nullable_prefix_len(&rule.rhs, nullable);
            // For each skip length, if next symbol is a cycle member, add a shortened rule.
            for skip in 1..=prefix_end {
                let suffix = &rule.rhs[skip..];
                if let Some(Symbol::Nonterminal(nt)) = suffix.first() {
                    if cycle_nodes.contains(nt) {
                        additions.push(Rule {
                            lhs: rule.lhs,
                            rhs: suffix.to_vec(),
                        });
                    }
                }
            }

            let first_after_nullable = rule.rhs.get(prefix_end);
            if let Some(Symbol::Nonterminal(next_nt)) = first_after_nullable {
                if *next_nt != rule.lhs && cycle_nodes.contains(next_nt) {
                    let expansions =
                        expand_cycle_head_paths(&rhs_by_lhs, &cycle_nodes, *next_nt, rule.lhs);
                    let mut produced_replacement = false;
                    for expansion in expansions {
                        let mut rhs = expansion;
                        rhs.extend_from_slice(&rule.rhs[prefix_end + 1..]);
                        additions.push(Rule { lhs: rule.lhs, rhs });
                        produced_replacement = true;
                    }
                    if produced_replacement {
                        replaced_cycle_rules.insert(rule.clone());
                    }
                }
            }
        }

        let existing: HashSet<Rule> = rules.iter().cloned().collect();
        let mut unique_additions = HashSet::new();
        additions.retain(|rule| {
            !existing.contains(rule) && unique_additions.insert(rule.clone())
        });

        if !replaced_cycle_rules.is_empty() {
            rules.retain(|rule| !replaced_cycle_rules.contains(rule));
            changed = true;
        }

        if additions.is_empty() {
            return changed;
        }
        rules.extend(additions);
        changed = true;
    }
}

fn expand_cycle_head_paths(
    rhs_by_lhs: &BTreeMap<NonterminalID, BTreeSet<Vec<Symbol>>>,
    cycle_nodes: &BTreeSet<NonterminalID>,
    current: NonterminalID,
    goal: NonterminalID,
) -> Vec<Vec<Symbol>> {
    fn expand(
        rhs_by_lhs: &BTreeMap<NonterminalID, BTreeSet<Vec<Symbol>>>,
        cycle_nodes: &BTreeSet<NonterminalID>,
        current: NonterminalID,
        goal: NonterminalID,
        visited: &mut BTreeSet<NonterminalID>,
        out: &mut Vec<Vec<Symbol>>,
    ) {
        if !visited.insert(current) {
            return;
        }

        if let Some(expansions) = rhs_by_lhs.get(&current) {
            for expansion_rhs in expansions {
                match expansion_rhs.first() {
                    Some(Symbol::Nonterminal(head))
                        if cycle_nodes.contains(head) && *head != goal =>
                    {
                        let mut nested = Vec::new();
                        expand(
                            rhs_by_lhs,
                            cycle_nodes,
                            *head,
                            goal,
                            visited,
                            &mut nested,
                        );
                        for mut nested_rhs in nested {
                            nested_rhs.extend_from_slice(&expansion_rhs[1..]);
                            out.push(nested_rhs);
                        }
                    }
                    _ => out.push(expansion_rhs.clone()),
                }
            }
        }

        visited.remove(&current);
    }

    let mut out = Vec::new();
    expand(
        rhs_by_lhs,
        cycle_nodes,
        current,
        goal,
        &mut BTreeSet::new(),
        &mut out,
    );
    out
}

fn nullable_prefix_len(rhs: &[Symbol], nullable: &BTreeSet<NonterminalID>) -> usize {
    rhs.iter()
        .take_while(|symbol| matches!(symbol, Symbol::Nonterminal(nt) if nullable.contains(nt)))
        .count()
}

/// Remove rules for nonterminals not reachable from the start symbol.
fn remove_unreachable_rules(rules: &[Rule], start: NonterminalID) -> Vec<Rule> {
    let num_nonterminals = usize::max(max_nt_id(rules) as usize, start as usize) + 1;
    let mut rules_by_lhs = vec![Vec::<usize>::new(); num_nonterminals];
    for (index, rule) in rules.iter().enumerate() {
        rules_by_lhs[rule.lhs as usize].push(index);
    }

    let mut reachable = vec![false; num_nonterminals];
    let mut worklist = vec![start];
    while let Some(nonterminal) = worklist.pop() {
        let nonterminal_index = nonterminal as usize;
        if reachable[nonterminal_index] {
            continue;
        }
        reachable[nonterminal_index] = true;
        for &rule_index in &rules_by_lhs[nonterminal_index] {
            for symbol in &rules[rule_index].rhs {
                if let Symbol::Nonterminal(next_nonterminal) = symbol
                    && !reachable[*next_nonterminal as usize]
                {
                    worklist.push(*next_nonterminal);
                }
            }
        }
    }
    rules
        .iter()
        .filter(|rule| reachable[rule.lhs as usize])
        .cloned()
        .collect()
}

fn build_rhs_by_lhs(rules: &[Rule]) -> BTreeMap<NonterminalID, BTreeSet<Vec<Symbol>>> {
    let mut rhs_by_lhs = BTreeMap::<NonterminalID, BTreeSet<Vec<Symbol>>>::new();
    for rule in rules {
        rhs_by_lhs
            .entry(rule.lhs)
            .or_default()
            .insert(rule.rhs.clone());
    }
    rhs_by_lhs
}

fn compute_expandable_single_productions(
    rhs_by_lhs: &BTreeMap<NonterminalID, BTreeSet<Vec<Symbol>>>,
) -> (BTreeMap<NonterminalID, Vec<Symbol>>, BTreeSet<NonterminalID>) {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum VisitState {
        Visiting,
        Expandable,
        NotExpandable,
    }

    fn visit(
        nt: NonterminalID,
        unique_rhs_by_lhs: &BTreeMap<NonterminalID, Vec<Symbol>>,
        state: &mut BTreeMap<NonterminalID, VisitState>,
    ) -> bool {
        if let Some(existing) = state.get(&nt).copied() {
            return match existing {
                VisitState::Visiting => false,
                VisitState::Expandable => true,
                VisitState::NotExpandable => false,
            };
        }

        let Some(rhs) = unique_rhs_by_lhs.get(&nt) else {
            return false;
        };

        state.insert(nt, VisitState::Visiting);
        let expandable = rhs.iter().all(|symbol| match symbol {
            Symbol::Terminal(_) => true,
            Symbol::Nonterminal(child) => {
                if unique_rhs_by_lhs.contains_key(child) {
                    visit(*child, unique_rhs_by_lhs, state)
                } else {
                    true
                }
            }
        });
        state.insert(
            nt,
            if expandable {
                VisitState::Expandable
            } else {
                VisitState::NotExpandable
            },
        );
        expandable
    }

    let unique_rhs_by_lhs: BTreeMap<NonterminalID, Vec<Symbol>> = rhs_by_lhs
        .iter()
        .filter_map(|(&nt, rhss)| {
            if rhss.len() == 1 {
                rhss.iter().next().cloned().map(|rhs| (nt, rhs))
            } else {
                None
            }
        })
        .collect();

    let mut state = BTreeMap::<NonterminalID, VisitState>::new();
    let mut expandable = BTreeSet::new();
    for &nt in unique_rhs_by_lhs.keys() {
        if visit(nt, &unique_rhs_by_lhs, &mut state) {
            expandable.insert(nt);
        }
    }

    (unique_rhs_by_lhs, expandable)
}

fn flatten_rhs_symbols(
    rhs: &[Symbol],
    unique_rhs_by_lhs: &BTreeMap<NonterminalID, Vec<Symbol>>,
    expandable_single_productions: &BTreeSet<NonterminalID>,
    flatten_cache: &mut HashMap<NonterminalID, Option<Vec<Symbol>>>,
) -> Vec<Symbol> {
    const MAX_FLATTENED_RHS_LEN: usize = 4096;

    fn flatten_symbol(
        symbol: &Symbol,
        out: &mut Vec<Symbol>,
        unique_rhs_by_lhs: &BTreeMap<NonterminalID, Vec<Symbol>>,
        expandable_single_productions: &BTreeSet<NonterminalID>,
        flatten_cache: &mut HashMap<NonterminalID, Option<Vec<Symbol>>>,
    ) {
        match symbol {
            Symbol::Terminal(_) => out.push(symbol.clone()),
            Symbol::Nonterminal(nt)
                if expandable_single_productions.contains(nt) =>
            {
                if let Some(cached) = flatten_cache.get(nt) {
                    match cached {
                        Some(flattened) if out.len() + flattened.len() <= MAX_FLATTENED_RHS_LEN => {
                            out.extend(flattened.iter().cloned());
                        }
                        _ => out.push(symbol.clone()),
                    }
                    return;
                }

                if let Some(expanded_rhs) = unique_rhs_by_lhs.get(nt) {
                    let mut flattened_nt = Vec::new();
                    for expanded_symbol in expanded_rhs {
                        flatten_symbol(
                            expanded_symbol,
                            &mut flattened_nt,
                            unique_rhs_by_lhs,
                            expandable_single_productions,
                            flatten_cache,
                        );
                        if flattened_nt.len() > MAX_FLATTENED_RHS_LEN {
                            flatten_cache.insert(*nt, None);
                            out.push(symbol.clone());
                            return;
                        }
                    }

                    flatten_cache.insert(*nt, Some(flattened_nt.clone()));
                    out.extend(flattened_nt);
                } else {
                    out.push(symbol.clone());
                }
            }
            Symbol::Nonterminal(_) => out.push(symbol.clone()),
        }
    }

    let mut flattened = Vec::new();
    for symbol in rhs {
        flatten_symbol(
            symbol,
            &mut flattened,
            unique_rhs_by_lhs,
            expandable_single_productions,
            flatten_cache,
        );
        if flattened.len() > MAX_FLATTENED_RHS_LEN {
            return rhs.to_vec();
        }
    }
    flattened
}

/// Borrowed, dense indexing for rule deduplication.
///
/// This preserves the old distinction between a sole *physical* production
/// (which cannot need deduplication) and a sole *distinct* RHS (which can be
/// expanded while testing whether sibling productions deduplicate).  The index
/// stores positions into `rules`, rather than cloning every RHS into an ordered
/// map and set.
struct DedupSingleProductionIndex {
    first_rule_by_lhs: Vec<Option<usize>>,
    has_multiple_distinct_rhs: Vec<bool>,
    rule_count_by_lhs: Vec<usize>,
    expandable_by_lhs: Vec<bool>,
}

impl DedupSingleProductionIndex {
    fn build(rules: &[Rule]) -> Self {
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum VisitState {
            Unseen,
            Visiting,
            Expandable,
            NotExpandable,
        }

        fn visit(
            nonterminal: NonterminalID,
            rules: &[Rule],
            first_rule_by_lhs: &[Option<usize>],
            has_multiple_distinct_rhs: &[bool],
            states: &mut [VisitState],
        ) -> bool {
            let index = nonterminal as usize;
            if index >= first_rule_by_lhs.len()
                || has_multiple_distinct_rhs[index]
                || first_rule_by_lhs[index].is_none()
            {
                // As in the old map-based implementation, a child with no
                // unique production is an opaque symbol for this flattening
                // analysis and does not block its parent's expansion.
                return true;
            }

            match states[index] {
                VisitState::Visiting | VisitState::NotExpandable => return false,
                VisitState::Expandable => return true,
                VisitState::Unseen => {}
            }

            states[index] = VisitState::Visiting;
            let rule = &rules[first_rule_by_lhs[index].expect("checked above")];
            let expandable = rule.rhs.iter().all(|symbol| match symbol {
                Symbol::Terminal(_) => true,
                Symbol::Nonterminal(child) => visit(
                    *child,
                    rules,
                    first_rule_by_lhs,
                    has_multiple_distinct_rhs,
                    states,
                ),
            });
            states[index] = if expandable {
                VisitState::Expandable
            } else {
                VisitState::NotExpandable
            };
            expandable
        }

        let len = max_nt_id(rules) as usize + 1;
        let mut first_rule_by_lhs: Vec<Option<usize>> = vec![None; len];
        let mut has_multiple_distinct_rhs = vec![false; len];
        let mut rule_count_by_lhs = vec![0usize; len];

        for (rule_index, rule) in rules.iter().enumerate() {
            let lhs = rule.lhs as usize;
            rule_count_by_lhs[lhs] += 1;
            match first_rule_by_lhs[lhs] {
                Some(first_rule_index) => {
                    if rules[first_rule_index].rhs != rule.rhs {
                        has_multiple_distinct_rhs[lhs] = true;
                    }
                }
                None => first_rule_by_lhs[lhs] = Some(rule_index),
            }
        }

        let mut states = vec![VisitState::Unseen; len];
        let mut expandable_by_lhs = vec![false; len];
        for nonterminal in 0..len {
            if first_rule_by_lhs[nonterminal].is_some()
                && !has_multiple_distinct_rhs[nonterminal]
            {
                expandable_by_lhs[nonterminal] = visit(
                    nonterminal as NonterminalID,
                    rules,
                    &first_rule_by_lhs,
                    &has_multiple_distinct_rhs,
                    &mut states,
                );
            }
        }

        Self {
            first_rule_by_lhs,
            has_multiple_distinct_rhs,
            rule_count_by_lhs,
            expandable_by_lhs,
        }
    }

    #[inline]
    fn rule_count(&self, nonterminal: NonterminalID) -> usize {
        self.rule_count_by_lhs
            .get(nonterminal as usize)
            .copied()
            .unwrap_or(0)
    }

    #[inline]
    fn is_expandable(&self, nonterminal: NonterminalID) -> bool {
        self.expandable_by_lhs
            .get(nonterminal as usize)
            .copied()
            .unwrap_or(false)
    }

    #[inline]
    fn unique_rhs<'a>(&self, rules: &'a [Rule], nonterminal: NonterminalID) -> Option<&'a [Symbol]> {
        let index = nonterminal as usize;
        (!self.has_multiple_distinct_rhs.get(index).copied().unwrap_or(true))
            .then_some(())
            .and_then(|_| self.first_rule_by_lhs.get(index).copied().flatten())
            .map(|rule_index| rules[rule_index].rhs.as_slice())
    }

    fn len(&self) -> usize {
        self.first_rule_by_lhs.len()
    }
}

fn flatten_rhs_symbols_for_dedup(
    rhs: &[Symbol],
    rules: &[Rule],
    index: &DedupSingleProductionIndex,
    flatten_cache: &mut [Option<Option<Vec<Symbol>>>],
) -> Vec<Symbol> {
    const MAX_FLATTENED_RHS_LEN: usize = 4096;

    fn flatten_symbol(
        symbol: &Symbol,
        out: &mut Vec<Symbol>,
        rules: &[Rule],
        index: &DedupSingleProductionIndex,
        flatten_cache: &mut [Option<Option<Vec<Symbol>>>],
    ) {
        match symbol {
            Symbol::Terminal(_) => out.push(symbol.clone()),
            Symbol::Nonterminal(nonterminal) if index.is_expandable(*nonterminal) => {
                let nonterminal_index = *nonterminal as usize;
                if let Some(cached) = flatten_cache
                    .get(nonterminal_index)
                    .and_then(Option::as_ref)
                {
                    match cached {
                        Some(flattened)
                            if out.len() + flattened.len() <= MAX_FLATTENED_RHS_LEN =>
                        {
                            out.extend(flattened.iter().cloned());
                        }
                        _ => out.push(symbol.clone()),
                    }
                    return;
                }

                let Some(expanded_rhs) = index.unique_rhs(rules, *nonterminal) else {
                    out.push(symbol.clone());
                    return;
                };
                let mut flattened_nonterminal = Vec::new();
                for expanded_symbol in expanded_rhs {
                    flatten_symbol(
                        expanded_symbol,
                        &mut flattened_nonterminal,
                        rules,
                        index,
                        flatten_cache,
                    );
                    if flattened_nonterminal.len() > MAX_FLATTENED_RHS_LEN {
                        if let Some(slot) = flatten_cache.get_mut(nonterminal_index) {
                            *slot = Some(None);
                        }
                        out.push(symbol.clone());
                        return;
                    }
                }

                if let Some(slot) = flatten_cache.get_mut(nonterminal_index) {
                    *slot = Some(Some(flattened_nonterminal.clone()));
                }
                out.extend(flattened_nonterminal);
            }
            Symbol::Nonterminal(_) => out.push(symbol.clone()),
        }
    }

    let mut flattened = Vec::new();
    for symbol in rhs {
        flatten_symbol(symbol, &mut flattened, rules, index, flatten_cache);
        if flattened.len() > MAX_FLATTENED_RHS_LEN {
            return rhs.to_vec();
        }
    }
    flattened
}

/// Deduplicate rules, preserving order of first occurrence.
enum RuleDedupKey<'a> {
    Borrowed(NonterminalID, &'a [Symbol]),
    Owned(NonterminalID, Vec<Symbol>),
}

impl RuleDedupKey<'_> {
    fn lhs(&self) -> NonterminalID {
        match self {
            RuleDedupKey::Borrowed(lhs, _) | RuleDedupKey::Owned(lhs, _) => *lhs,
        }
    }

    fn rhs(&self) -> &[Symbol] {
        match self {
            RuleDedupKey::Borrowed(_, rhs) => rhs,
            RuleDedupKey::Owned(_, rhs) => rhs,
        }
    }
}

impl PartialEq for RuleDedupKey<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.lhs() == other.lhs() && self.rhs() == other.rhs()
    }
}

impl Eq for RuleDedupKey<'_> {}

impl Hash for RuleDedupKey<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.lhs().hash(state);
        self.rhs().hash(state);
    }
}

fn dedup_rules(rules: &mut Vec<Rule>) {
    let index = DedupSingleProductionIndex::build(rules);
    let mut keep = Vec::with_capacity(rules.len());
    {
        let mut seen = FxHashSet::with_capacity_and_hasher(rules.len(), Default::default());
        let mut flatten_cache = vec![None; index.len()];
        for rule in rules.iter() {
            // Rule equality includes the LHS. A sole physical production for an
            // LHS cannot duplicate any other rule, even after flattening chains.
            if index.rule_count(rule.lhs) == 1 {
                keep.push(true);
                continue;
            }
            let can_flatten = rule.rhs.iter().any(|symbol| {
                matches!(symbol, Symbol::Nonterminal(nonterminal) if index.is_expandable(*nonterminal))
            });
            let key = if can_flatten {
                RuleDedupKey::Owned(
                    rule.lhs,
                    flatten_rhs_symbols_for_dedup(
                        &rule.rhs,
                        rules,
                        &index,
                        &mut flatten_cache,
                    ),
                )
            } else {
                RuleDedupKey::Borrowed(rule.lhs, &rule.rhs)
            };
            keep.push(seen.insert(key));
        }
    }
    let mut keep_iter = keep.into_iter();
    rules.retain(|_| keep_iter.next().unwrap_or(false));
}

fn is_reflexive_unit_rule(rule: &Rule) -> bool {
    matches!(rule.rhs.as_slice(), [Symbol::Nonterminal(nonterminal)] if *nonterminal == rule.lhs)
}

pub fn merge_identical_nonterminals(
    rules: &[Rule],
    start: NonterminalID,
) -> Vec<Rule> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Build rhs_by_lhs: the set of productions for each nonterminal.
    let rhs_by_lhs = build_rhs_by_lhs(rules);

    if rhs_by_lhs.len() <= 1 {
        return rules.to_vec();
    }

    let nts: Vec<NonterminalID> = rhs_by_lhs.keys().copied().collect();

    // Fast O(1) lookup from NT ID → index, replacing BTreeMap (which is
    // O(log n) and dominates the hot loop when there are millions of
    // lookups across refinement iterations).
    let max_nt_id = *nts.last().unwrap() as usize;
    let mut nt_to_idx_fast = vec![u32::MAX; max_nt_id + 1];
    for (i, &nt) in nts.iter().enumerate() {
        nt_to_idx_fast[nt as usize] = i as u32;
    }

    // Pre-index production sets for O(1) access by NT index.
    let rhs_by_idx: Vec<&BTreeSet<Vec<Symbol>>> =
        nts.iter().map(|nt| &rhs_by_lhs[nt]).collect();
    let (unique_rhs_by_lhs, expandable_single_productions) =
        compute_expandable_single_productions(&rhs_by_lhs);
    let mut flatten_cache = HashMap::<NonterminalID, Option<Vec<Symbol>>>::new();
    let flattened_rhs_by_idx: Vec<Vec<Vec<Symbol>>> = nts
        .iter()
        .map(|nt| {
            rhs_by_lhs[nt]
                .iter()
                .map(|rhs| {
                    flatten_rhs_symbols(
                        rhs,
                        &unique_rhs_by_lhs,
                        &expandable_single_productions,
                        &mut flatten_cache,
                    )
                })
                .collect()
        })
        .collect();

    // ── Partition refinement (top-down) ──────────────────────────────────
    //
    // Instead of the classical bottom-up "find-one-merge, re-scan" loop
    // (which cascades for O(chain_length) iterations), we use partition
    // refinement: start with the *coarsest* partition consistent with
    // terminal structure, then iteratively refine by incorporating the
    // partition classes of referenced nonterminals.  This discovers all
    // transitively-isomorphic nonterminals in O(refinement_depth) passes,
    // with O(n) work per pass.
    //
    // Each refinement hashes each NT's normalised production set (with
    // self-refs → sentinel, other NT refs → current class ID), then
    // normalises the resulting class IDs by order of first appearance so
    // that the convergence check is stable.

    const SELF_SENTINEL: u64 = u64::MAX;
    const GENERIC_NT: u64 = u64::MAX - 1;

    // Hash a single NT's production set given current class assignments.
    // Uses commutative accumulation (wrapping_add of rotated hashes) to
    // avoid Vec allocation and sorting.
    let hash_nt = |nt_idx: usize, class_of: &[u64]| -> u64 {
        let nt = nts[nt_idx];
        let mut sig: u64 = 0;
        for rhs in &flattened_rhs_by_idx[nt_idx] {
            let mut h = DefaultHasher::new();
            for s in rhs {
                match s {
                    Symbol::Terminal(t) => {
                        0u8.hash(&mut h);
                        t.hash(&mut h);
                    }
                    Symbol::Nonterminal(n) if *n == nt => {
                        1u8.hash(&mut h);
                        SELF_SENTINEL.hash(&mut h);
                    }
                    Symbol::Nonterminal(n) => {
                        let ni = *n as usize;
                        if ni <= max_nt_id && nt_to_idx_fast[ni] != u32::MAX {
                            1u8.hash(&mut h);
                            class_of[nt_to_idx_fast[ni] as usize].hash(&mut h);
                        } else {
                            2u8.hash(&mut h);
                            n.hash(&mut h);
                        }
                    }
                }
            }
            // Commutative combine: wrapping_add of scrambled prod hash.
            let ph = h.finish();
            sig = sig.wrapping_add(ph.wrapping_mul(0x9E3779B97F4A7C15));
        }
        sig
    };

    // Normalise returns (normalised_vec, n_distinct_classes).
    let normalise_counted = |raw: &[u64]| -> (Vec<u64>, usize) {
        let mut map = HashMap::<u64, u64>::with_capacity(raw.len());
        let mut nc: u64 = 0;
        let v: Vec<u64> = raw
            .iter()
            .map(|&v| {
                *map.entry(v).or_insert_with(|| {
                    let c = nc;
                    nc += 1;
                    c
                })
            })
            .collect();
        (v, nc as usize)
    };

    // Quick pre-check: hash with REAL NT IDs (no GENERIC_NT). If every
    // NT already has a unique signature, there are no isomorphisms and
    // we can skip the expensive partition refinement entirely. This
    // handles the common "confirmatory call" case in O(n).
    {
        let real_classes: Vec<u64> = (0..nts.len())
            .map(|i| {
                // Use the NT's own index as its class (finest partition).
                let nt = nts[i];
                let mut sig: u64 = 0;
                for rhs in &flattened_rhs_by_idx[i] {
                    let mut h = DefaultHasher::new();
                    for s in rhs {
                        match s {
                            Symbol::Terminal(t) => {
                                0u8.hash(&mut h);
                                t.hash(&mut h);
                            }
                            Symbol::Nonterminal(n) if *n == nt => {
                                1u8.hash(&mut h);
                                SELF_SENTINEL.hash(&mut h);
                            }
                            Symbol::Nonterminal(n) => {
                                2u8.hash(&mut h);
                                n.hash(&mut h);
                            }
                        }
                    }
                    let ph = h.finish();
                    sig = sig.wrapping_add(ph.wrapping_mul(0x9E3779B97F4A7C15));
                }
                sig
            })
            .collect();
        let mut seen = HashSet::with_capacity(real_classes.len());
        if real_classes.iter().all(|h| seen.insert(*h)) {
            // Every NT has a unique raw-ID signature → no merges possible.
            return rules.to_vec();
        }
    }

    // Initial partition: all NT refs → GENERIC_NT.
    let initial_classes: Vec<u64> = {
        let init = vec![GENERIC_NT; nts.len()];
        (0..nts.len()).map(|i| hash_nt(i, &init)).collect()
    };
    let (mut class_of, n_classes) = normalise_counted(&initial_classes);

    // If every NT is already in its own class, no merges are possible.
    if n_classes == nts.len() {
        return rules.to_vec();
    }

    // ── Compute processing order: ascending "depth" in the reference DAG ──
    //
    // NTs at depth 0 have no references to other NTs (leaves). Depth d means
    // the longest path to a leaf is d hops. Processing in ascending depth
    // with in-place updates (Gauss-Seidel) propagates chain information in
    // ONE pass instead of one-per-chain-link. For purely acyclic chains of
    // depth D, this reduces D iterations to ~1.
    //
    // We compute exact depths via SCC condensation (Kosaraju's) + DAG depth,
    // which handles cycles correctly (NTs in the same SCC share a depth) and
    // works for arbitrarily deep chains.
    let processing_order: Vec<usize> = {
        let n = nts.len();

        // Build adjacency: for each NT index, which other NT indices does it reference?
        let mut refs_of: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, &nt) in nts.iter().enumerate() {
            for rhs in rhs_by_idx[i].iter() {
                for s in rhs {
                    if let Symbol::Nonterminal(r) = s {
                        if *r != nt {
                            let ri = *r as usize;
                            if ri <= max_nt_id && nt_to_idx_fast[ri] != u32::MAX {
                                refs_of[i].push(nt_to_idx_fast[ri] as usize);
                            }
                        }
                    }
                }
            }
        }

        // ── Kosaraju's SCC algorithm (O(V+E), iterative) ──

        // Step 1: iterative DFS on original graph → finish order.
        let mut visited = vec![false; n];
        let mut finish_order = Vec::with_capacity(n);
        for start in 0..n {
            if visited[start] { continue; }
            let mut stk: Vec<(usize, usize)> = vec![(start, 0)];
            visited[start] = true;
            while let Some((v, ni)) = stk.last_mut() {
                if *ni < refs_of[*v].len() {
                    let w = refs_of[*v][*ni];
                    *ni += 1;
                    if !visited[w] {
                        visited[w] = true;
                        stk.push((w, 0));
                    }
                } else {
                    finish_order.push(*v);
                    stk.pop();
                }
            }
        }

        // Step 2: reverse graph + DFS in reverse finish order → SCC IDs.
        // Kosaraju's numbers SCCs in topological order (sources first in the
        // original dependent→dependency edge direction).
        let mut rev_refs: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, neighbors) in refs_of.iter().enumerate() {
            for &j in neighbors {
                rev_refs[j].push(i);
            }
        }
        let mut scc_id = vec![0u32; n];
        let mut next_scc = 0u32;
        visited.fill(false);
        for &start in finish_order.iter().rev() {
            if visited[start] { continue; }
            let mut stk = vec![start];
            visited[start] = true;
            while let Some(v) = stk.pop() {
                scc_id[v] = next_scc;
                for &w in &rev_refs[v] {
                    if !visited[w] {
                        visited[w] = true;
                        stk.push(w);
                    }
                }
            }
            next_scc += 1;
        }
        let num_sccs = next_scc as usize;

        // Step 3: condensed DAG depth (longest path to a sink = leaf depth 0).
        // Build inter-SCC edges, deduplicate, then compute depth in reverse
        // topological order (SCCs numbered source-first, so iterate high→low).
        let mut scc_edges: Vec<Vec<u32>> = vec![Vec::new(); num_sccs];
        for (i, neighbors) in refs_of.iter().enumerate() {
            for &j in neighbors {
                if scc_id[i] != scc_id[j] {
                    scc_edges[scc_id[i] as usize].push(scc_id[j]);
                }
            }
        }
        for edges in &mut scc_edges {
            edges.sort_unstable();
            edges.dedup();
        }
        let mut scc_depth = vec![0u32; num_sccs];
        for s in (0..num_sccs).rev() {
            for &dst in &scc_edges[s] {
                scc_depth[s] = scc_depth[s].max(scc_depth[dst as usize] + 1);
            }
        }

        // Map SCC depth back to NTs, sort by ascending depth (leaves first).
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by_key(|&i| scc_depth[scc_id[i] as usize]);
        order
    };

    // Refine until stable, using depth-ordered in-place (Gauss-Seidel) updates.
    loop {
        let prev_class_of = class_of.clone();
        // In-place update: process NTs in depth order so that deeper NTs
        // (which depend on shallower ones) see already-updated classes.
        let mut raw = vec![0u64; nts.len()];
        for &i in &processing_order {
            raw[i] = hash_nt(i, &class_of);
            // Update class_of in-place for Gauss-Seidel propagation.
            class_of[i] = raw[i];
        }
        let (new_class_of, nc) = normalise_counted(&raw);
        class_of = new_class_of;
        if class_of == prev_class_of {
            break;
        }
        // Early termination: every NT is in its own class → no merges.
        if nc == nts.len() {
            return rules.to_vec();
        }
    }

    // ── Build merge map from final partition ─────────────────────────────
    // Within each equivalence class, pick a representative (prefer start,
    // otherwise lowest NT ID — which is naturally first since `nts` is
    // sorted from a BTreeMap).

    let mut class_to_rep: BTreeMap<u64, NonterminalID> = BTreeMap::new();
    for (idx, &nt) in nts.iter().enumerate() {
        let class = class_of[idx];
        let rep = class_to_rep.entry(class).or_insert(nt);
        if nt == start {
            *rep = start;
        }
    }

    let mut merge_map: BTreeMap<NonterminalID, NonterminalID> = BTreeMap::new();
    for (idx, &nt) in nts.iter().enumerate() {
        let rep = class_to_rep[&class_of[idx]];
        if nt != rep {
            merge_map.insert(nt, rep);
        }
    }

    if merge_map.is_empty() {
        return rules.to_vec();
    }

    // Apply merge map to produce the deduplicated rule set.
    let apply = |nt: NonterminalID| -> NonterminalID {
        *merge_map.get(&nt).unwrap_or(&nt)
    };

    let mut result = Vec::new();
    let mut seen = HashSet::with_capacity(rules.len());
    for rule in rules {
        let lhs = apply(rule.lhs);
        let rhs: Vec<Symbol> = rule
            .rhs
            .iter()
            .map(|symbol| match symbol {
                Symbol::Terminal(terminal) => Symbol::Terminal(*terminal),
                Symbol::Nonterminal(nonterminal) => Symbol::Nonterminal(apply(*nonterminal)),
            })
            .collect();
        let merged = Rule { lhs, rhs };
        if is_reflexive_unit_rule(&merged) {
            continue;
        }
        if seen.insert(merged.clone()) {
            result.push(merged);
        }
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Grammar Normalization Pipeline
// ─────────────────────────────────────────────────────────────────────────────
//
// Transforms a grammar so that it satisfies the preconditions required by the
// terminal-characterization stage:
//
//   1. No nullable nonterminals — every nonterminal derives at least one
//      terminal symbol.
//   2. No right recursion — neither direct (A → α A) nor indirect
//      (A →* α B, B →* β A).
//   3. No indirect left recursion — only direct left recursion (A → A α) is
//      permitted (safe for GLR).
//
// The normalization loop repeatedly inlines null productions, eliminates
// right recursion, and exposes hidden left recursion until the grammar stops
// changing, then runs the final epsilon-elimination and unreachable pruning.

/// Run the full grammar normalization pipeline (in place).
///
/// Mutates `rules` so that they satisfy the characterization
/// preconditions (no nullable NTs, no right recursion, no indirect LR).
/// `start` is used only for unreachable-production pruning and is never
/// changed.
pub fn normalize_grammar(rules: &mut Vec<Rule>, start: NonterminalID) {
    use std::cell::Cell;

    let profiling = compile_profile_enabled();
    let next_nt = Cell::new(max_nt_id(rules) + 1);
    let mut fresh_nt = || {
        let id = next_nt.get();
        next_nt.set(id + 1);
        id
    };

    let mut iteration = 0;
    let mut nullable_eliminated_before_exit = false;
    loop {
        let iteration_rules_before = rules.len();
        let iteration_started_at = profiling.then(Instant::now);

        let clone_started_at = profiling.then(Instant::now);
        let snap = rules.clone();
        if let Some(started_at) = clone_started_at {
            emit_normalize_profile(
                "clone_snapshot",
                Some(iteration + 1),
                elapsed_ms(started_at),
                iteration_rules_before,
                snap.len(),
                "",
            );
        }

        let inline_rules_before = rules.len();
        let inline_started_at = profiling.then(Instant::now);
        inline_null_productions_in_place_if_needed(rules, &next_nt);
        // A -> A contributes no terminal derivation. Epsilon elimination can
        // expose it, and rewriting it as right recursion would manufacture a
        // fresh nullable self-cycle on every subsequent iteration.
        rules.retain(|rule| !is_reflexive_unit_rule(rule));
        let inline_unchanged = *rules == snap;
        if let Some(started_at) = inline_started_at {
            emit_normalize_profile(
                "inline_null_productions",
                Some(iteration + 1),
                elapsed_ms(started_at),
                inline_rules_before,
                rules.len(),
                &format!(" phase=fixed_point changed={}", !inline_unchanged),
            );
        }

        let rr_rules_before = rules.len();
        let rr_started_at = profiling.then(Instant::now);
        let right_recursion_completed = with_resynced_next_nonterminal(rules, &next_nt, |rules| {
            eliminate_right_recursion(rules, &mut fresh_nt)
        });
        if let Some(started_at) = rr_started_at {
            emit_normalize_profile(
                "eliminate_right_recursion",
                Some(iteration + 1),
                elapsed_ms(started_at),
                rr_rules_before,
                rules.len(),
                "",
            );
        }

        let hlr_rules_before = rules.len();
        let hlr_started_at = profiling.then(Instant::now);
        let mut nullable_count = 0usize;
        let hidden_left_recursion_changed = with_resynced_next_nonterminal(rules, &next_nt, |rules| {
            let nullable = compute_nullable(rules, max_nt_id(rules) + 1);
            nullable_count = nullable.len();
            // With no nullable prefix, the pass still resolves ordinary
            // indirect left recursion. Avoid its more expensive SCC and rule
            // expansion work when a cheap reachability check proves there is
            // no such cycle.
            if nullable.is_empty()
                && !has_indirect_left_recursion_with_nullable(rules, &nullable)
            {
                false
            } else {
                eliminate_hidden_left_recursion(rules, &nullable, iteration + 1)
            }
        });

        if let Some(started_at) = hlr_started_at {
            emit_normalize_profile(
                "compute_nullable_and_eliminate_hidden_left_recursion",
                Some(iteration + 1),
                elapsed_ms(started_at),
                hlr_rules_before,
                rules.len(),
                &format!(" nullable_count={nullable_count}"),
            );
        }

        let dedup_rules_before = rules.len();
        let dedup_started_at = profiling.then(Instant::now);
        dedup_rules(rules);

        if let Some(started_at) = dedup_started_at {
            emit_normalize_profile(
                "dedup_rules",
                Some(iteration + 1),
                elapsed_ms(started_at),
                dedup_rules_before,
                rules.len(),
                " phase=fixed_point",
            );
        }

        let equality_started_at = profiling.then(Instant::now);
        let converged = *rules == snap;
        if let Some(started_at) = equality_started_at {
            emit_normalize_profile(
                "equality_convergence_check",
                Some(iteration + 1),
                elapsed_ms(started_at),
                rules.len(),
                snap.len(),
                &format!(" converged={converged}"),
            );
        }

        if let Some(started_at) = iteration_started_at {
            emit_normalize_profile(
                "fixed_point_iteration_total",
                Some(iteration + 1),
                elapsed_ms(started_at),
                iteration_rules_before,
                rules.len(),
                &format!(" converged={converged}"),
            );
        }

        if compute_nullable(rules, max_nt_id(rules) + 1).is_empty()
            && right_recursion_completed
            && !hidden_left_recursion_changed
        {
            // Check the rewritten grammar: right-recursion elimination can
            // introduce nullable helpers even when its input had none.
            // Post-merge transforms perform their own indirect-LR recovery.
            nullable_eliminated_before_exit = true;
            break;
        }
        iteration += 1;

        if converged {
            break;
        }
    }

    let post_inline_rules_before = rules.len();
    let post_inline_snapshot = (!nullable_eliminated_before_exit).then(|| rules.clone());
    let post_inline_started_at = profiling.then(Instant::now);
    if !nullable_eliminated_before_exit {
        inline_null_productions_in_place_if_needed(rules, &next_nt);
        rules.retain(|rule| !is_reflexive_unit_rule(rule));
    }
    let post_inline_changed = post_inline_snapshot
        .as_ref()
        .is_some_and(|snapshot| *rules != *snapshot);
    if let Some(started_at) = post_inline_started_at {
        let post_inline_extra = if nullable_eliminated_before_exit {
            " phase=post_loop skipped=nullable_eliminated".to_string()
        } else {
            format!(" phase=post_loop changed={post_inline_changed}")
        };
        emit_normalize_profile(
            "inline_null_productions",
            None,
            elapsed_ms(started_at),
            post_inline_rules_before,
            rules.len(),
            &post_inline_extra,
        );
    }

    let unreachable_rules_before = rules.len();
    let unreachable_started_at = profiling.then(Instant::now);
    *rules = remove_unreachable_rules(rules, start);
    if let Some(started_at) = unreachable_started_at {
        emit_normalize_profile(
            "remove_unreachable_rules",
            None,
            elapsed_ms(started_at),
            unreachable_rules_before,
            rules.len(),
            "",
        );
    }

    let final_dedup_rules_before = rules.len();
    let final_dedup_started_at = profiling.then(Instant::now);
    if post_inline_changed {
        dedup_rules(rules);
    }
    if let Some(started_at) = final_dedup_started_at {
        emit_normalize_profile(
            "dedup_rules",
            None,
            elapsed_ms(started_at),
            final_dedup_rules_before,
            rules.len(),
            if post_inline_changed {
                " phase=final"
            } else {
                " phase=final skipped=post_inline_unchanged"
            },
        );
    }
}

/// Minimal normalization required by the dynamic GLR runtime: remove epsilon
/// productions, prune unreachable rules, and deduplicate. Unlike the static
/// characterization normalizer this intentionally preserves right recursion
/// and indirect left recursion as CFG structure.
pub fn normalize_dynamic_glr_grammar(rules: &mut Vec<Rule>, start: NonterminalID) {
    let num_nt = max_nt_id(rules) + 1;
    *rules = inline_null_productions(rules, num_nt);
    *rules = remove_unreachable_rules(rules, start);
    dedup_rules(rules);
}

fn inline_null_productions_in_place_if_needed(
    rules: &mut Vec<Rule>,
    next_nt: &std::cell::Cell<u32>,
) {
    // In this CFG algebra terminals are nonnullable symbols. A nullable
    // derivation must therefore have an empty production at a leaf.
    // Without one, both existing epsilon helpers return exact input clones.
    // Retain the original allocation instead, without changing rule order.
    if rules.iter().any(|rule| rule.rhs.is_empty()) {
        replace_rules_with_resync(rules, next_nt, inline_null_productions);
    } else {
        resync_next_nonterminal(rules, next_nt);
    }
}

fn replace_rules_with_resync(
    rules: &mut Vec<Rule>,
    next_nt: &std::cell::Cell<u32>,
    update: impl FnOnce(&[Rule], u32) -> Vec<Rule>,
) {
    *rules = update(rules, next_nt.get());
    resync_next_nonterminal(rules, next_nt);
}

fn with_resynced_next_nonterminal<R>(
    rules: &mut Vec<Rule>,
    next_nt: &std::cell::Cell<u32>,
    update: impl FnOnce(&mut Vec<Rule>) -> R,
) -> R {
    let result = update(rules);
    resync_next_nonterminal(rules, next_nt);
    result
}

fn resync_next_nonterminal(rules: &[Rule], next_nt: &std::cell::Cell<u32>) {
    next_nt.set(max_nt_id(rules) + 1);
}

fn compute_nullable(rules: &[Rule], num_nt: u32) -> BTreeSet<NonterminalID> {
    use std::collections::VecDeque;

    // Exact fixed-point seed test. Cycles of nonempty productions alone
    // cannot derive epsilon, so avoid allocating the dependency graph.
    if !rules.iter().any(|rule| rule.lhs < num_nt && rule.rhs.is_empty()) {
        return BTreeSet::new();
    }

    let n = num_nt as usize;
    let mut remaining = vec![usize::MAX; rules.len()];
    let mut dependents = vec![Vec::<usize>::new(); n];
    let mut nullable = BTreeSet::new();
    let mut queue = VecDeque::new();

    for (rule_index, rule) in rules.iter().enumerate() {
        if rule.lhs as usize >= n {
            continue;
        }
        if rule.rhs.is_empty() {
            remaining[rule_index] = 0;
            if nullable.insert(rule.lhs) {
                queue.push_back(rule.lhs);
            }
            continue;
        }
        let mut count = 0usize;
        let mut possible = true;
        for symbol in &rule.rhs {
            match symbol {
                Symbol::Terminal(_) => {
                    possible = false;
                    break;
                }
                Symbol::Nonterminal(nonterminal) => {
                    let idx = *nonterminal as usize;
                    if idx >= n {
                        possible = false;
                        break;
                    }
                    dependents[idx].push(rule_index);
                    count += 1;
                }
            }
        }
        if possible {
            remaining[rule_index] = count;
        }
    }

    while let Some(nonterminal) = queue.pop_front() {
        for &rule_index in &dependents[nonterminal as usize] {
            let left = &mut remaining[rule_index];
            if *left == usize::MAX || *left == 0 {
                continue;
            }
            *left -= 1;
            if *left == 0 {
                let lhs = rules[rule_index].lhs;
                if nullable.insert(lhs) {
                    queue.push_back(lhs);
                }
            }
        }
    }
    nullable
}

fn compute_first(
    rules: &[Rule],
    num_nt: u32,
    num_terminals: u32,
    nullable: &BTreeSet<NonterminalID>,
) -> Vec<BitSet> {
    use std::collections::VecDeque;

    let n = num_nt as usize;
    let set_len = num_terminals as usize + 1;
    let mut first = vec![BitSet::new(set_len); n];
    // FIRST(source) changes must propagate to every lhs that can reach source
    // through a nullable prefix.
    let mut dependents = vec![Vec::<usize>::new(); n];

    for rule in rules {
        let lhs = rule.lhs as usize;
        if lhs >= n {
            continue;
        }
        for symbol in &rule.rhs {
            match symbol {
                Symbol::Terminal(terminal) => {
                    first[lhs].set(terminal_bit(*terminal, num_terminals));
                    break;
                }
                Symbol::Nonterminal(nonterminal) => {
                    let source = *nonterminal as usize;
                    if source >= n {
                        break;
                    }
                    dependents[source].push(lhs);
                    if !nullable.contains(nonterminal) {
                        break;
                    }
                }
            }
        }
    }
    for targets in &mut dependents {
        targets.sort_unstable();
        targets.dedup();
    }

    let mut queue = VecDeque::new();
    let mut queued = vec![false; n];
    for index in 0..n {
        if !first[index].is_empty() {
            queue.push_back(index);
            queued[index] = true;
        }
    }
    while let Some(source) = queue.pop_front() {
        queued[source] = false;
        let source_first = first[source].clone();
        for &target in &dependents[source] {
            let delta = source_first.difference(&first[target]);
            if delta.is_empty() {
                continue;
            }
            first[target].union_with(&delta);
            if !queued[target] {
                queued[target] = true;
                queue.push_back(target);
            }
        }
    }
    first
}

fn compute_follow(
    rules: &[Rule],
    num_nt: u32,
    num_terminals: u32,
    start: NonterminalID,
    first: &[BitSet],
    nullable: &BTreeSet<NonterminalID>,
) -> Vec<BitSet> {
    use std::collections::VecDeque;

    let n = num_nt as usize;
    let set_len = num_terminals as usize + 1;
    let mut follow = vec![BitSet::new(set_len); n];
    // FOLLOW(source) changes must propagate to targets whose occurrence is
    // followed only by nullable symbols in a source production.
    let mut dependents = vec![Vec::<usize>::new(); n];

    if let Some(start_follow) = follow.get_mut(start as usize) {
        start_follow.set(terminal_bit(EOF, num_terminals));
    }

    for rule in rules {
        let lhs = rule.lhs as usize;
        if lhs >= n {
            continue;
        }
        for (index, symbol) in rule.rhs.iter().enumerate() {
            let Symbol::Nonterminal(nonterminal) = symbol else {
                continue;
            };
            let target = *nonterminal as usize;
            if target >= n {
                continue;
            }
            let mut suffix_nullable = true;
            for suffix_symbol in &rule.rhs[index + 1..] {
                match suffix_symbol {
                    Symbol::Terminal(terminal) => {
                        follow[target].set(terminal_bit(*terminal, num_terminals));
                        suffix_nullable = false;
                        break;
                    }
                    Symbol::Nonterminal(next_nonterminal) => {
                        let next = *next_nonterminal as usize;
                        if next >= n {
                            suffix_nullable = false;
                            break;
                        }
                        follow[target].union_with(&first[next]);
                        if !nullable.contains(next_nonterminal) {
                            suffix_nullable = false;
                            break;
                        }
                    }
                }
            }
            if suffix_nullable {
                dependents[lhs].push(target);
            }
        }
    }
    for targets in &mut dependents {
        targets.sort_unstable();
        targets.dedup();
    }

    let mut queue = VecDeque::new();
    let mut queued = vec![false; n];
    for index in 0..n {
        if !follow[index].is_empty() {
            queue.push_back(index);
            queued[index] = true;
        }
    }
    while let Some(source) = queue.pop_front() {
        queued[source] = false;
        let source_follow = follow[source].clone();
        for &target in &dependents[source] {
            let delta = source_follow.difference(&follow[target]);
            if delta.is_empty() {
                continue;
            }
            follow[target].union_with(&delta);
            if !queued[target] {
                queued[target] = true;
                queue.push_back(target);
            }
        }
    }
    follow
}

#[inline]
fn terminal_bit(terminal: TerminalID, num_terminals: u32) -> usize {
    if terminal == EOF {
        num_terminals as usize
    } else {
        terminal as usize
    }
}

fn filter_graph_to_reachable(
    graph: BTreeMap<NonterminalID, BTreeSet<NonterminalID>>,
    reachable: &BTreeSet<NonterminalID>,
) -> BTreeMap<NonterminalID, BTreeSet<NonterminalID>> {
    graph
        .into_iter()
        .filter(|(nonterminal, _)| reachable.contains(nonterminal))
        .map(|(nonterminal, edges)| {
            (
                nonterminal,
                edges
                    .into_iter()
                    .filter(|edge| reachable.contains(edge))
                    .collect(),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::flat::{GrammarDef, Terminal};

    fn analyzed_grammar(rules: Vec<Rule>, start: NonterminalID) -> AnalyzedGrammar {
        AnalyzedGrammar::from_grammar_def(&GrammarDef {
            rules,
            start,
            terminals: vec![Terminal::Literal {
                id: 0,
                bytes: b"a".to_vec(),
            }],
            ..GrammarDef::default()
        })
    }

    fn bounded_language(
        rules: &[Rule],
        start: NonterminalID,
        num_nonterminals: u32,
        max_len: usize,
    ) -> BTreeSet<Vec<TerminalID>> {
        fn rhs_language(
            rhs: &[Symbol],
            languages: &[BTreeSet<Vec<TerminalID>>],
            max_len: usize,
        ) -> BTreeSet<Vec<TerminalID>> {
            let mut acc = BTreeSet::from([Vec::new()]);
            for symbol in rhs {
                let part = match symbol {
                    Symbol::Terminal(terminal) => {
                        BTreeSet::from([vec![*terminal]])
                    }
                    Symbol::Nonterminal(nonterminal) => {
                        languages[*nonterminal as usize].clone()
                    }
                };
                let mut next = BTreeSet::new();
                for prefix in &acc {
                    for suffix in &part {
                        if prefix.len() + suffix.len() <= max_len {
                            let mut combined = prefix.clone();
                            combined.extend(suffix);
                            next.insert(combined);
                        }
                    }
                }
                acc = next;
                if acc.is_empty() {
                    break;
                }
            }
            acc
        }

        let mut languages = vec![BTreeSet::new(); num_nonterminals as usize];
        loop {
            let mut changed = false;
            for rule in rules {
                let derived = rhs_language(&rule.rhs, &languages, max_len);
                let target = &mut languages[rule.lhs as usize];
                let old_len = target.len();
                target.extend(derived);
                changed |= target.len() != old_len;
            }
            if !changed {
                break;
            }
        }
        languages[start as usize].clone()
    }

    #[test]
    fn table_build_normal_form_rejects_nullable_zero_length_rules() {
        let grammar = analyzed_grammar(vec![Rule { lhs: 0, rhs: Vec::new() }], 0);

        let error = grammar.check_table_build_normal_form().unwrap_err();
        assert!(error.contains("nullable nonterminals reachable"));
        assert!(error.contains("zero-length productions reachable"));
    }

    #[test]
    fn indirect_left_recursion_precheck_observes_nullable_prefixes() {
        let rules = vec![
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Nonterminal(2), Symbol::Nonterminal(1)],
            },
            Rule {
                lhs: 1,
                rhs: vec![Symbol::Nonterminal(0), Symbol::Terminal(0)],
            },
            Rule { lhs: 2, rhs: Vec::new() },
        ];
        let nullable = compute_nullable(&rules, 3);

        assert!(has_indirect_left_recursion_with_nullable(&rules, &nullable));
    }

    #[test]
    fn indirect_left_recursion_precheck_ignores_direct_self_recursion() {
        let rules = vec![
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Nonterminal(0), Symbol::Terminal(0)],
            },
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            },
        ];
        let nullable = compute_nullable(&rules, 1);

        assert!(!has_indirect_left_recursion_with_nullable(&rules, &nullable));
    }

    #[test]
    fn indirect_right_recursion_fast_precheck_matches_ordered_reference() {
        const N: u32 = 3;
        let edge_count = (N * N) as usize;
        for mask in 0usize..(1usize << edge_count) {
            let mut rules = Vec::new();
            for lhs in 0..N {
                for target in 0..N {
                    let bit = (lhs * N + target) as usize;
                    if (mask >> bit) & 1 != 0 {
                        rules.push(Rule {
                            lhs,
                            rhs: vec![Symbol::Nonterminal(target)],
                        });
                    }
                }
            }
            let nullable = BTreeSet::new();
            let reference = find_cycle_excluding_self_loops(&build_right_reachability_graph(
                &rules, &nullable,
            ))
            .is_some();
            assert_eq!(
                has_indirect_right_recursion_fast(&rules, &nullable, N),
                reference,
                "edge mask {mask:#x}",
            );
        }
    }

    #[test]
    fn indirect_right_recursion_fast_precheck_observes_nullable_suffixes() {
        let rules = vec![
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Nonterminal(1), Symbol::Nonterminal(2)],
            },
            Rule {
                lhs: 1,
                rhs: vec![Symbol::Terminal(0), Symbol::Nonterminal(0)],
            },
            Rule {
                lhs: 2,
                rhs: Vec::new(),
            },
        ];
        let nullable = compute_nullable(&rules, 3);
        let reference = find_cycle_excluding_self_loops(&build_right_reachability_graph(
            &rules, &nullable,
        ))
        .is_some();

        assert!(reference);
        assert!(has_indirect_right_recursion_fast(&rules, &nullable, 3));
    }

    #[test]
    fn table_build_normal_form_rejects_direct_right_recursion() {
        let grammar = analyzed_grammar(
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0), Symbol::Nonterminal(0)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
            ],
            0,
        );

        let error = grammar.check_table_build_normal_form().unwrap_err();
        assert!(error.contains("right-recursive cycle detected"));
    }

    #[test]
    fn table_build_normal_form_rejects_indirect_left_recursion() {
        let grammar = analyzed_grammar(
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Nonterminal(0), Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(0)],
                },
            ],
            0,
        );

        let error = grammar.check_table_build_normal_form().unwrap_err();
        assert!(error.contains("indirect left-recursive cycle detected"));
    }

    #[test]
    fn table_build_normal_form_accepts_simple_nonnullable_grammar() {
        let grammar = analyzed_grammar(
            vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            0,
        );

        assert!(grammar.check_table_build_normal_form().is_ok());
    }

    #[test]
    fn nontrivial_sccs_include_multiple_disjoint_cycles_and_skip_self_loops() {
        let graph = BTreeMap::from([
            (0, BTreeSet::from([1])),
            (1, BTreeSet::from([0])),
            (2, BTreeSet::from([3])),
            (3, BTreeSet::from([4])),
            (4, BTreeSet::from([2])),
            (5, BTreeSet::from([5])),
            (6, BTreeSet::from([7])),
        ]);

        let mut sccs = find_nontrivial_sccs(&graph);
        sccs.sort_by_key(|component| component.iter().next().copied().unwrap_or(u32::MAX));

        assert_eq!(sccs, vec![BTreeSet::from([0, 1]), BTreeSet::from([2, 3, 4])]);
    }

    fn dedup_rules_reference(rules: &mut Vec<Rule>) {
        let mut rhs_by_lhs = BTreeMap::<NonterminalID, BTreeSet<Vec<Symbol>>>::new();
        let mut rule_counts_by_lhs = HashMap::<NonterminalID, usize>::new();
        for rule in rules.iter() {
            rhs_by_lhs
                .entry(rule.lhs)
                .or_default()
                .insert(rule.rhs.clone());
            *rule_counts_by_lhs.entry(rule.lhs).or_default() += 1;
        }
        let (unique_rhs_by_lhs, expandable_single_productions) =
            compute_expandable_single_productions(&rhs_by_lhs);
        let mut keep = Vec::with_capacity(rules.len());
        {
            let mut seen = HashSet::with_capacity(rules.len());
            let mut flatten_cache = HashMap::<NonterminalID, Option<Vec<Symbol>>>::new();
            for rule in rules.iter() {
                if rule_counts_by_lhs[&rule.lhs] == 1 {
                    keep.push(true);
                    continue;
                }
                let can_flatten = rule.rhs.iter().any(|symbol| {
                    matches!(symbol, Symbol::Nonterminal(nonterminal) if expandable_single_productions.contains(nonterminal))
                });
                let key = if can_flatten {
                    RuleDedupKey::Owned(
                        rule.lhs,
                        flatten_rhs_symbols(
                            &rule.rhs,
                            &unique_rhs_by_lhs,
                            &expandable_single_productions,
                            &mut flatten_cache,
                        ),
                    )
                } else {
                    RuleDedupKey::Borrowed(rule.lhs, &rule.rhs)
                };
                keep.push(seen.insert(key));
            }
        }
        let mut keep_iter = keep.into_iter();
        rules.retain(|_| keep_iter.next().unwrap_or(false));
    }

    #[test]
    fn dedup_rules_matches_reference_on_varied_small_grammars() {
        fn next(seed: &mut u64) -> u64 {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *seed
        }

        let mut seed = 0x17a1_ded0_0c42_u64;
        for case_index in 0..512 {
            let nonterminals = (next(&mut seed) % 7 + 1) as u32;
            let rules_len = (next(&mut seed) % 28 + 1) as usize;
            let mut rules = Vec::with_capacity(rules_len);
            for _ in 0..rules_len {
                let lhs = (next(&mut seed) % u64::from(nonterminals)) as u32;
                let rhs_len = (next(&mut seed) % 5) as usize;
                let mut rhs = Vec::with_capacity(rhs_len);
                for _ in 0..rhs_len {
                    if next(&mut seed) & 1 == 0 {
                        rhs.push(Symbol::Terminal((next(&mut seed) % 5) as u32));
                    } else {
                        rhs.push(Symbol::Nonterminal(
                            (next(&mut seed) % u64::from(nonterminals)) as u32,
                        ));
                    }
                }
                rules.push(Rule { lhs, rhs });
            }

            let mut expected = rules.clone();
            dedup_rules_reference(&mut expected);
            let mut actual = rules;
            dedup_rules(&mut actual);
            assert_eq!(actual, expected, "case_index={case_index}");
        }
    }

    #[test]
    fn dedup_rules_collapses_duplicate_physical_single_productions() {
        let mut rules = vec![
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Nonterminal(1)],
            },
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Nonterminal(1)],
            },
            Rule {
                lhs: 1,
                rhs: vec![Symbol::Terminal(0)],
            },
        ];

        dedup_rules(&mut rules);

        assert_eq!(
            rules,
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(0)],
                },
            ]
        );
    }

    #[test]
    fn dedup_rules_detects_duplicates_after_single_production_flattening() {
        let mut rules = vec![
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Nonterminal(1)],
            },
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            },
            Rule {
                lhs: 1,
                rhs: vec![Symbol::Terminal(0)],
            },
        ];

        dedup_rules(&mut rules);

        assert_eq!(
            rules,
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(0)],
                },
            ]
        );
    }

    #[test]
    fn dedup_rules_does_not_flatten_cyclic_single_productions() {
        let mut rules = vec![
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Nonterminal(1)],
            },
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Nonterminal(2)],
            },
            Rule {
                lhs: 1,
                rhs: vec![Symbol::Nonterminal(2)],
            },
            Rule {
                lhs: 2,
                rhs: vec![Symbol::Nonterminal(1)],
            },
        ];

        dedup_rules(&mut rules);

        assert_eq!(
            rules,
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(2)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Nonterminal(2)],
                },
                Rule {
                    lhs: 2,
                    rhs: vec![Symbol::Nonterminal(1)],
                },
            ]
        );
    }

    #[test]
    fn nullable_run_compression_preserves_nullable_only_nonempty_derivations() {
        let rules = vec![
            Rule {
                lhs: 3,
                rhs: vec![Symbol::Terminal(1)],
            },
            Rule { lhs: 2, rhs: vec![] },
            Rule {
                lhs: 2,
                rhs: vec![Symbol::Nonterminal(2), Symbol::Nonterminal(3)],
            },
            Rule {
                lhs: 0,
                rhs: vec![
                    Symbol::Terminal(0),
                    Symbol::Nonterminal(1),
                    Symbol::Nonterminal(2),
                    Symbol::Terminal(0),
                ],
            },
            Rule {
                lhs: 1,
                rhs: vec![Symbol::Terminal(2)],
            },
            Rule {
                lhs: 5,
                rhs: vec![Symbol::Terminal(3)],
            },
            Rule { lhs: 4, rhs: vec![] },
            Rule {
                lhs: 4,
                rhs: vec![Symbol::Nonterminal(4), Symbol::Nonterminal(5)],
            },
            Rule {
                lhs: 1,
                rhs: vec![Symbol::Nonterminal(4)],
            },
        ];

        let source = bounded_language(&rules, 0, 6, 3);
        let transformed_rules = inline_null_productions(&rules, 6);
        let transformed =
            bounded_language(&transformed_rules, 0, max_nt_id(&transformed_rules) + 1, 3);
        let mut normalized_rules = rules.clone();
        normalize_grammar(&mut normalized_rules, 0);
        let normalized =
            bounded_language(&normalized_rules, 0, max_nt_id(&normalized_rules) + 1, 3);

        assert!(source.contains(&vec![0, 3, 0]));
        assert_eq!(transformed, source);
        assert_eq!(normalized, source);
    }
}
