//! Ordered grammar normalization with iteration-local immutable rule payloads.
//!
//! The original implementation in `analysis` remains the ordered compiler
//! reference. This module changes storage and traversal work, not the rewrite
//! schedule, publication order, fresh-ID order, or convergence semantics.
//!
//! No arena handle escapes this module. Published rules retain the original
//! public Rule representation and coordinate system.

use std::borrow::Cow;
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use rustc_hash::FxHashSet;

use crate::grammar::flat::{NonterminalID, Rule, Symbol};

use super::normalization_reference as reference;

const NO_COMPONENT: usize = usize::MAX;
const MAX_INDIRECT_ROUNDS: usize = 200;
const MAX_FLATTENED_RHS_LEN: usize = 4096;

fn env_enabled(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
}

fn reference_validation_enabled() -> bool {
    cfg!(test)
        || env_enabled("GLRMASK_VALIDATE_COMPILER_TEMPLATES")
        || env_enabled("GLRMASK_VALIDATE_COMPILER_NORMALIZATION")
}

fn profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
}

fn profile(
    stage: &str,
    iteration: usize,
    started: Option<Instant>,
    before: usize,
    after: usize,
) {
    if let Some(started) = started {
        eprintln!(
            "[glrmask-profile] normalize_grammar stage={} iteration={} ms={:.3} rules_before={} rules_after={}",
            stage,
            iteration,
            started.elapsed().as_secs_f64() * 1000.0,
            before,
            after,
        );
    }
}

#[derive(Clone, Copy)]
struct View<'a> {
    arena: &'a [Rule],
    order: Option<&'a [usize]>,
}

impl<'a> View<'a> {
    fn plain(rules: &'a [Rule]) -> Self {
        Self {
            arena: rules,
            order: None,
        }
    }

    fn len(self) -> usize {
        self.order.map_or(self.arena.len(), <[usize]>::len)
    }

    fn indices(self) -> impl Iterator<Item = usize> + 'a {
        (0..self.len()).map(move |position| {
            self.order
                .map_or(position, |order| order[position])
        })
    }

    fn iter(self) -> impl Iterator<Item = &'a Rule> + 'a {
        self.indices().map(move |index| &self.arena[index])
    }

    fn at(self, position: usize) -> &'a Rule {
        &self.arena[self.order.map_or(position, |order| order[position])]
    }

    fn max_nt(self) -> NonterminalID {
        self.iter()
            .flat_map(|rule| {
                std::iter::once(rule.lhs).chain(rule.rhs.iter().filter_map(|symbol| {
                    match symbol {
                        Symbol::Nonterminal(nonterminal) => Some(*nonterminal),
                        Symbol::Terminal(_) => None,
                    }
                }))
            })
            .max()
            .unwrap_or(0)
    }

    fn nt_count(self) -> usize {
        self.max_nt() as usize + 1
    }
}

/// Payloads are immutable while an Edit exists. Only order handles change.
///
/// The original iteration snapshot is exactly arena[0..original_len].
/// Generated payloads are appended. Handles are never recycled.
struct Edit {
    arena: Vec<Rule>,
    order: Vec<usize>,
    original_len: usize,
}

impl Edit {
    fn new(rules: Vec<Rule>) -> Self {
        let original_len = rules.len();
        Self {
            arena: rules,
            order: (0..original_len).collect(),
            original_len,
        }
    }

    fn view(&self) -> View<'_> {
        View {
            arena: &self.arena,
            order: Some(&self.order),
        }
    }

    fn len(&self) -> usize {
        self.order.len()
    }

    fn push(&mut self, rule: Rule) -> usize {
        let index = self.arena.len();
        self.arena.push(rule);
        index
    }

    fn replace_with_owned(&mut self, rules: Vec<Rule>) {
        let first = self.arena.len();
        self.arena.extend(rules);
        self.order.clear();
        self.order.extend(first..self.arena.len());
    }

    fn retain_non_reflexive_units(&mut self) {
        let arena = &self.arena;
        self.order
            .retain(|&index| !is_reflexive_unit_rule(&arena[index]));
    }

    fn equals_original(&self) -> bool {
        self.order.len() == self.original_len
            && self
                .order
                .iter()
                .enumerate()
                .all(|(position, &index)| {
                    position == index || self.arena[position] == self.arena[index]
                })
    }

    /// Moves every surviving RHS allocation to the public result.
    ///
    /// All transformations preserve uniqueness of physical handles in order:
    /// retained handles appear once, and every emitted rule gets a fresh handle.
    fn publish(mut self) -> Vec<Rule> {
        if self
            .order
            .iter()
            .enumerate()
            .all(|(position, &index)| position == index)
        {
            self.arena.truncate(self.order.len());
            return self.arena;
        }

        let mut result = Vec::with_capacity(self.order.len());
        for index in self.order {
            let rule = &mut self.arena[index];
            result.push(Rule {
                lhs: rule.lhs,
                rhs: std::mem::take(&mut rule.rhs),
            });
        }
        // The remaining original/generated payloads are destroyed here,
        // synchronously, before publication returns.
        result
    }
}

fn is_reflexive_unit_rule(rule: &Rule) -> bool {
    matches!(
        rule.rhs.as_slice(),
        [Symbol::Nonterminal(nonterminal)] if *nonterminal == rule.lhs
    )
}

/// Ordinary CFG nullability. Lexically nullable terminals have already been
/// expanded by the frontend; every Terminal symbol here is consuming.
///
/// No empty production implies no finite epsilon derivation, including in a
/// pure unit cycle. That exact case requires no dependency index at all.
struct Nullable {
    flags: Vec<bool>,
    count: usize,
}

impl Nullable {
    fn compute(view: View<'_>) -> Self {
        if !view.iter().any(|rule| rule.rhs.is_empty()) {
            return Self {
                flags: Vec::new(),
                count: 0,
            };
        }

        let n = view.nt_count();
        let mut flags = vec![false; n];
        let mut remaining = vec![usize::MAX; view.len()];
        let mut dependents = vec![Vec::<usize>::new(); n];
        let mut queue = Vec::<usize>::new();
        let mut count = 0usize;

        for (position, rule) in view.iter().enumerate() {
            if rule.rhs.is_empty() {
                remaining[position] = 0;
                let lhs = rule.lhs as usize;
                if !flags[lhs] {
                    flags[lhs] = true;
                    count += 1;
                    queue.push(lhs);
                }
                continue;
            }

            let mut nonterminals = 0usize;
            let mut possible = true;
            for symbol in &rule.rhs {
                match symbol {
                    Symbol::Terminal(_) => {
                        possible = false;
                        break;
                    }
                    Symbol::Nonterminal(nonterminal) => {
                        dependents[*nonterminal as usize].push(position);
                        nonterminals += 1;
                    }
                }
            }
            if possible {
                remaining[position] = nonterminals;
            }
        }

        let mut cursor = 0usize;
        while cursor < queue.len() {
            let nonterminal = queue[cursor];
            cursor += 1;
            for &position in &dependents[nonterminal] {
                let left = &mut remaining[position];
                if *left == usize::MAX || *left == 0 {
                    continue;
                }
                *left -= 1;
                if *left == 0 {
                    let lhs = view.at(position).lhs as usize;
                    if !flags[lhs] {
                        flags[lhs] = true;
                        count += 1;
                        queue.push(lhs);
                    }
                }
            }
        }

        Self { flags, count }
    }

    fn contains(&self, nonterminal: NonterminalID) -> bool {
        self.flags
            .get(nonterminal as usize)
            .copied()
            .unwrap_or(false)
    }

    fn is_empty(&self) -> bool {
        self.count == 0
    }
}

#[derive(Clone, Copy)]
enum Boundary {
    Left,
    Right,
}

struct BoundaryGraph {
    edges: Vec<Vec<usize>>,
    ordered: bool,
}

impl BoundaryGraph {
    fn build(view: View<'_>, nullable: &Nullable, boundary: Boundary) -> Self {
        let mut edges = vec![Vec::<usize>::new(); view.nt_count()];
        for rule in view.iter() {
            let row = &mut edges[rule.lhs as usize];
            match boundary {
                Boundary::Left => {
                    for symbol in &rule.rhs {
                        match symbol {
                            Symbol::Terminal(_) => break,
                            Symbol::Nonterminal(nonterminal) => {
                                row.push(*nonterminal as usize);
                                if !nullable.contains(*nonterminal) {
                                    break;
                                }
                            }
                        }
                    }
                }
                Boundary::Right => {
                    for symbol in rule.rhs.iter().rev() {
                        match symbol {
                            Symbol::Terminal(_) => break,
                            Symbol::Nonterminal(nonterminal) => {
                                row.push(*nonterminal as usize);
                                if !nullable.contains(*nonterminal) {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        Self {
            edges,
            ordered: false,
        }
    }

    fn order_rows(&mut self) {
        if self.ordered {
            return;
        }
        for row in &mut self.edges {
            row.sort_unstable();
            row.dedup();
        }
        self.ordered = true;
    }

    /// Exact Kahn check. Self loops are intentionally ignored.
    ///
    /// Duplicate edges need not be removed for this Boolean query: every
    /// increment has its corresponding decrement.
    fn has_indirect_cycle(&self) -> bool {
        let n = self.edges.len();
        let mut indegree = vec![0usize; n];
        for (source, row) in self.edges.iter().enumerate() {
            for &target in row {
                if source != target {
                    indegree[target] += 1;
                }
            }
        }

        let mut queue = Vec::with_capacity(n);
        for (node, &degree) in indegree.iter().enumerate() {
            if degree == 0 {
                queue.push(node);
            }
        }

        let mut cursor = 0usize;
        while cursor < queue.len() {
            let source = queue[cursor];
            cursor += 1;
            for &target in &self.edges[source] {
                if source == target {
                    continue;
                }
                indegree[target] -= 1;
                if indegree[target] == 0 {
                    queue.push(target);
                }
            }
        }
        queue.len() != n
    }

    /// Same first cycle as the ordered BTreeMap/BTreeSet DFS reference.
    ///
    /// Absent dense coordinates are isolated nodes. Visiting them cannot alter
    /// the first cycle or the order within any non-isolated DFS.
    fn first_indirect_cycle(&mut self) -> Option<Vec<NonterminalID>> {
        self.order_rows();
        let n = self.edges.len();
        let mut colors = vec![0u8; n];
        let mut positions = vec![usize::MAX; n];
        let mut stack = Vec::<(usize, usize)>::new();

        for start in 0..n {
            if colors[start] != 0 {
                continue;
            }
            colors[start] = 1;
            positions[start] = 0;
            stack.push((start, 0));

            while let Some(&(node, next_index)) = stack.last() {
                if next_index == self.edges[node].len() {
                    colors[node] = 2;
                    positions[node] = usize::MAX;
                    stack.pop();
                    continue;
                }

                let next = self.edges[node][next_index];
                stack.last_mut().expect("nonempty DFS stack").1 += 1;
                if next == node {
                    continue;
                }

                match colors[next] {
                    0 => {
                        colors[next] = 1;
                        positions[next] = stack.len();
                        stack.push((next, 0));
                    }
                    1 => {
                        let first = positions[next];
                        return Some(
                            stack[first..]
                                .iter()
                                .map(|&(entry, _)| entry as NonterminalID)
                                .collect(),
                        );
                    }
                    _ => {}
                }
            }
        }
        None
    }

    /// Nontrivial SCC membership only. Component numbering is private.
    fn nontrivial_components(&mut self) -> Vec<usize> {
        self.order_rows();
        let n = self.edges.len();
        let mut reverse = vec![Vec::<usize>::new(); n];
        for (source, row) in self.edges.iter().enumerate() {
            for &target in row {
                reverse[target].push(source);
            }
        }

        let mut visited = vec![false; n];
        let mut finish = Vec::with_capacity(n);
        let mut dfs = Vec::<(usize, usize)>::new();

        for start in 0..n {
            if visited[start] {
                continue;
            }
            visited[start] = true;
            dfs.push((start, 0));
            while let Some(&(node, next_index)) = dfs.last() {
                if next_index < self.edges[node].len() {
                    let next = self.edges[node][next_index];
                    dfs.last_mut().expect("nonempty DFS stack").1 += 1;
                    if !visited[next] {
                        visited[next] = true;
                        dfs.push((next, 0));
                    }
                } else {
                    finish.push(node);
                    dfs.pop();
                }
            }
        }

        visited.fill(false);
        let mut membership = vec![NO_COMPONENT; n];
        let mut todo = Vec::new();
        let mut component = Vec::new();
        let mut next_component = 0usize;

        for &start in finish.iter().rev() {
            if visited[start] {
                continue;
            }
            todo.push(start);
            visited[start] = true;
            component.clear();

            while let Some(node) = todo.pop() {
                component.push(node);
                for &next in &reverse[node] {
                    if !visited[next] {
                        visited[next] = true;
                        todo.push(next);
                    }
                }
            }

            if component.len() >= 2 {
                for &node in &component {
                    membership[node] = next_component;
                }
                next_component += 1;
            }
        }

        membership
    }
}

fn find_right_end_position(
    rhs: &[Symbol],
    target: NonterminalID,
    nullable: &Nullable,
) -> Option<usize> {
    for position in (0..rhs.len()).rev() {
        match &rhs[position] {
            Symbol::Nonterminal(nonterminal) if *nonterminal == target => {
                return Some(position);
            }
            Symbol::Nonterminal(nonterminal) if nullable.contains(*nonterminal) => {}
            _ => return None,
        }
    }
    None
}

fn inline_right_end(
    edit: &mut Edit,
    from: NonterminalID,
    to: NonterminalID,
    nullable: &Nullable,
) {
    let old_order = std::mem::take(&mut edit.order);
    let alternatives = old_order
        .iter()
        .copied()
        .filter(|&index| edit.arena[index].lhs == to)
        .collect::<Vec<_>>();

    if alternatives.is_empty() {
        edit.order = old_order;
        return;
    }

    let mut order = Vec::with_capacity(old_order.len());
    for index in old_order {
        let position = {
            let rule = &edit.arena[index];
            (rule.lhs == from)
                .then(|| find_right_end_position(&rule.rhs, to, nullable))
                .flatten()
        };
        let Some(position) = position else {
            order.push(index);
            continue;
        };

        for &alternative in &alternatives {
            let candidate = {
                let rule = &edit.arena[index];
                let replacement = &edit.arena[alternative].rhs;
                let mut rhs = Vec::with_capacity(
                    rule.rhs.len() - 1 + replacement.len(),
                );
                rhs.extend_from_slice(&rule.rhs[..position]);
                rhs.extend_from_slice(replacement);
                rhs.extend_from_slice(&rule.rhs[position + 1..]);
                Rule { lhs: from, rhs }
            };
            order.push(edit.push(candidate));
        }
    }
    edit.order = order;
}

fn direct_right_recursive(rule: &Rule) -> bool {
    matches!(
        rule.rhs.last(),
        Some(Symbol::Nonterminal(nonterminal)) if *nonterminal == rule.lhs
    )
}

fn resolve_direct_right_recursion(
    edit: &mut Edit,
    fresh_nt: &mut impl FnMut() -> NonterminalID,
) {
    let recursive_nonterminals = edit
        .view()
        .iter()
        .filter(|rule| direct_right_recursive(rule))
        .map(|rule| rule.lhs)
        .collect::<BTreeSet<_>>();

    if recursive_nonterminals.is_empty() {
        return;
    }

    // Fresh IDs are allocated in exactly the reference's ascending LHS order.
    let replacements = recursive_nonterminals
        .into_iter()
        .map(|nonterminal| (nonterminal, fresh_nt()))
        .collect::<BTreeMap<_, _>>();

    let mut groups =
        BTreeMap::<NonterminalID, (Vec<usize>, Vec<usize>)>::new();
    let old_order = std::mem::take(&mut edit.order);
    let mut order = Vec::with_capacity(old_order.len().saturating_mul(2));

    for index in old_order {
        let rule = &edit.arena[index];
        if replacements.contains_key(&rule.lhs) {
            let group = groups.entry(rule.lhs).or_default();
            if direct_right_recursive(rule) {
                group.0.push(index);
            } else {
                group.1.push(index);
            }
        } else {
            order.push(index);
        }
    }

    for (&nonterminal, &helper) in &replacements {
        let (recursive, bases) = groups.remove(&nonterminal).unwrap_or_default();

        // Reference order: all bases, all composed bases, all tails, all
        // left-recursive tails. Do not interleave these four groups.
        order.extend(bases.iter().copied());

        for &base in &bases {
            let candidate = {
                let base = &edit.arena[base];
                let mut rhs = Vec::with_capacity(base.rhs.len() + 1);
                rhs.push(Symbol::Nonterminal(helper));
                rhs.extend_from_slice(&base.rhs);
                Rule {
                    lhs: nonterminal,
                    rhs,
                }
            };
            order.push(edit.push(candidate));
        }

        for &recursive_rule in &recursive {
            let candidate = {
                let rhs = &edit.arena[recursive_rule].rhs;
                Rule {
                    lhs: helper,
                    rhs: rhs[..rhs.len() - 1].to_vec(),
                }
            };
            order.push(edit.push(candidate));
        }

        for &recursive_rule in &recursive {
            let candidate = {
                let body = &edit.arena[recursive_rule].rhs;
                let mut rhs = Vec::with_capacity(body.len());
                rhs.push(Symbol::Nonterminal(helper));
                rhs.extend_from_slice(&body[..body.len() - 1]);
                Rule { lhs: helper, rhs }
            };
            order.push(edit.push(candidate));
        }
    }

    edit.order = order;
}

fn eliminate_right_recursion_edit(
    edit: &mut Edit,
    fresh_nt: &mut impl FnMut() -> NonterminalID,
) -> bool {
    let mut nullable = Nullable::compute(edit.view());
    let mut graph = BoundaryGraph::build(edit.view(), &nullable, Boundary::Right);
    let mut completed = !graph.has_indirect_cycle();

    if !completed {
        for round in 0..MAX_INDIRECT_ROUNDS {
            // Round zero can reuse the exact facts used by the precheck.
            if round != 0 {
                nullable = Nullable::compute(edit.view());
                graph = BoundaryGraph::build(edit.view(), &nullable, Boundary::Right);
            }
            match graph.first_indirect_cycle() {
                Some(cycle) => {
                    inline_right_end(edit, cycle[0], cycle[1], &nullable);
                }
                None => {
                    completed = true;
                    break;
                }
            }
        }
    }

    resolve_direct_right_recursion(edit, fresh_nt);
    completed
}

/// Compatible public entry point. The caller still owns fresh-ID allocation.
pub fn eliminate_right_recursion(
    rules: &mut Vec<Rule>,
    fresh_nt: &mut impl FnMut() -> NonterminalID,
) -> bool {
    let mut edit = Edit::new(std::mem::take(rules));
    let completed = eliminate_right_recursion_edit(&mut edit, fresh_nt);
    *rules = edit.publish();
    completed
}

pub fn has_indirect_left_recursion(rules: &[Rule]) -> bool {
    let view = View::plain(rules);
    let nullable = Nullable::compute(view);
    BoundaryGraph::build(view, &nullable, Boundary::Left).has_indirect_cycle()
}

struct ExpansionFrame<'a> {
    node: usize,
    next_alternative: usize,
    return_suffix: &'a [Symbol],
}

/// Enumerates exactly the reference DFS's simple head paths.
///
/// A frame's return_suffix is appended after the expansion returned by that
/// frame. Emitting one leaf followed by reverse frame suffixes avoids building
/// and repeatedly copying nested vectors at every recursion level.
fn expand_cycle_head_paths(
    rhs_by_nt: &[Vec<&[Symbol]>],
    membership: &[usize],
    component: usize,
    current: NonterminalID,
    goal: NonterminalID,
    final_suffix: &[Symbol],
    on_path: &mut [bool],
    additions: &mut Vec<Rule>,
) -> bool {
    let before = additions.len();
    let mut stack = vec![ExpansionFrame {
        node: current as usize,
        next_alternative: 0,
        return_suffix: &[],
    }];
    on_path[current as usize] = true;

    while let Some(frame) = stack.last() {
        let node = frame.node;
        let next_alternative = frame.next_alternative;
        if next_alternative == rhs_by_nt[node].len() {
            on_path[node] = false;
            stack.pop();
            continue;
        }

        let rhs = rhs_by_nt[node][next_alternative];
        stack
            .last_mut()
            .expect("nonempty expansion stack")
            .next_alternative += 1;

        if let Some(Symbol::Nonterminal(head)) = rhs.first() {
            let head_index = *head as usize;
            if membership[head_index] == component && *head != goal {
                if !on_path[head_index] {
                    on_path[head_index] = true;
                    stack.push(ExpansionFrame {
                        node: head_index,
                        next_alternative: 0,
                        return_suffix: &rhs[1..],
                    });
                }
                continue;
            }
        }

        let suffix_len = stack
            .iter()
            .map(|frame| frame.return_suffix.len())
            .sum::<usize>();
        let mut expanded =
            Vec::with_capacity(rhs.len() + suffix_len + final_suffix.len());
        expanded.extend_from_slice(rhs);
        for frame in stack.iter().rev() {
            expanded.extend_from_slice(frame.return_suffix);
        }
        expanded.extend_from_slice(final_suffix);
        additions.push(Rule {
            lhs: goal,
            rhs: expanded,
        });
    }

    additions.len() != before
}

fn eliminate_hidden_left_recursion(edit: &mut Edit, nullable: &Nullable) -> bool {
    let mut changed = false;

    loop {
        let mut graph = BoundaryGraph::build(edit.view(), nullable, Boundary::Left);
        if !graph.has_indirect_cycle() {
            return changed;
        }
        let membership = graph.nontrivial_components();

        let (additions, replaced) = {
            let view = edit.view();
            let mut rhs_by_nt = vec![Vec::<&[Symbol]>::new(); membership.len()];

            // Only SCC members can be recursively expanded. Exits remain
            // opaque RHS symbols, exactly as in the reference.
            for rule in view.iter() {
                let lhs = rule.lhs as usize;
                if membership[lhs] != NO_COMPONENT {
                    rhs_by_nt[lhs].push(rule.rhs.as_slice());
                }
            }
            for row in &mut rhs_by_nt {
                row.sort_unstable();
                row.dedup();
            }

            let mut additions = Vec::<Rule>::new();
            let mut replaced = vec![false; view.len()];
            let mut on_path = vec![false; membership.len()];

            for (position, rule) in view.iter().enumerate() {
                let component = membership[rule.lhs as usize];
                if component == NO_COMPONENT {
                    continue;
                }

                let prefix_end = rule
                    .rhs
                    .iter()
                    .take_while(|symbol| {
                        matches!(
                            symbol,
                            Symbol::Nonterminal(nonterminal)
                                if nullable.contains(*nonterminal)
                        )
                    })
                    .count();

                for skip in 1..=prefix_end {
                    let suffix = &rule.rhs[skip..];
                    if let Some(Symbol::Nonterminal(nonterminal)) = suffix.first() {
                        if membership[*nonterminal as usize] == component {
                            additions.push(Rule {
                                lhs: rule.lhs,
                                rhs: suffix.to_vec(),
                            });
                        }
                    }
                }

                if let Some(Symbol::Nonterminal(next)) = rule.rhs.get(prefix_end) {
                    if *next != rule.lhs
                        && membership[*next as usize] == component
                    {
                        replaced[position] = expand_cycle_head_paths(
                            &rhs_by_nt,
                            &membership,
                            component,
                            *next,
                            rule.lhs,
                            &rule.rhs[prefix_end + 1..],
                            &mut on_path,
                            &mut additions,
                        );
                    }
                }
            }

            // Compare against the complete old inventory BEFORE removal.
            // In particular, an addition equal to a soon-to-be-removed rule
            // is still filtered, matching d751 exactly.
            let keep = {
                let existing = view
                    .iter()
                    .map(|rule| (rule.lhs, rule.rhs.as_slice()))
                    .collect::<FxHashSet<_>>();
                let mut unique = FxHashSet::default();
                additions
                    .iter()
                    .map(|rule| {
                        let key = (rule.lhs, rule.rhs.as_slice());
                        !existing.contains(&key) && unique.insert(key)
                    })
                    .collect::<Vec<_>>()
            };
            let mut keep = keep.into_iter();
            additions.retain(|_| keep.next().expect("one addition decision per rule"));
            (additions, replaced)
        };

        if replaced.iter().any(|&replace| replace) {
            let mut replaced = replaced.into_iter();
            edit.order.retain(|_| {
                !replaced
                    .next()
                    .expect("one replacement decision per old rule")
            });
            changed = true;
        }

        if additions.is_empty() {
            return changed;
        }

        for addition in additions {
            let index = edit.push(addition);
            edit.order.push(index);
        }
        changed = true;
    }
}

struct DedupIndex {
    first: Vec<Option<usize>>,
    multiple_distinct: Vec<bool>,
    physical_count: Vec<usize>,
    expandable: Vec<bool>,
}

impl DedupIndex {
    fn build(view: View<'_>) -> Self {
        let n = view.nt_count();
        let mut first: Vec<Option<usize>> = vec![None; n];
        let mut multiple_distinct = vec![false; n];
        let mut physical_count = vec![0usize; n];

        for index in view.indices() {
            let rule = &view.arena[index];
            let lhs = rule.lhs as usize;
            physical_count[lhs] += 1;
            match first[lhs] {
                Some(previous) => {
                    if view.arena[previous].rhs != rule.rhs {
                        multiple_distinct[lhs] = true;
                    }
                }
                None => first[lhs] = Some(index),
            }
        }

        // A unique production is expandable iff its graph of unique-production
        // dependencies cannot reach a cycle. Missing/multiple productions are
        // opaque, not blocking, exactly as in the reference.
        let mut remaining = vec![0usize; n];
        let mut dependents = vec![Vec::<usize>::new(); n];
        for nonterminal in 0..n {
            let Some(index) = first[nonterminal] else {
                continue;
            };
            if multiple_distinct[nonterminal] {
                continue;
            }
            for symbol in &view.arena[index].rhs {
                if let Symbol::Nonterminal(child) = symbol {
                    let child = *child as usize;
                    if first[child].is_some() && !multiple_distinct[child] {
                        remaining[nonterminal] += 1;
                        dependents[child].push(nonterminal);
                    }
                }
            }
        }

        let mut ready = Vec::new();
        for nonterminal in 0..n {
            if first[nonterminal].is_some()
                && !multiple_distinct[nonterminal]
                && remaining[nonterminal] == 0
            {
                ready.push(nonterminal);
            }
        }

        let mut expandable = vec![false; n];
        let mut cursor = 0usize;
        while cursor < ready.len() {
            let nonterminal = ready[cursor];
            cursor += 1;
            expandable[nonterminal] = true;
            for &dependent in &dependents[nonterminal] {
                remaining[dependent] -= 1;
                if remaining[dependent] == 0 {
                    ready.push(dependent);
                }
            }
        }

        Self {
            first,
            multiple_distinct,
            physical_count,
            expandable,
        }
    }

    fn unique_rhs<'a>(
        &self,
        view: View<'a>,
        nonterminal: NonterminalID,
    ) -> Option<&'a [Symbol]> {
        let index = nonterminal as usize;
        if self.multiple_distinct[index] {
            return None;
        }
        self.first[index].map(|rule| view.arena[rule].rhs.as_slice())
    }
}

struct FlattenFrame<'a> {
    owner: Option<NonterminalID>,
    rhs: &'a [Symbol],
    position: usize,
    output: Vec<Symbol>,
}

/// Iterative simulation of d751's lazy flattening, including its cache-order
/// dependent 4096-symbol optimization guard.
///
/// The guard leaves an opaque nonterminal or the original RHS. It never
/// truncates a language or caps runtime derivation depth.
fn flatten_for_dedup<'a>(
    rhs: &'a [Symbol],
    view: View<'a>,
    index: &DedupIndex,
    cache: &mut [Option<Option<Vec<Symbol>>>],
) -> Vec<Symbol> {
    let mut stack = vec![FlattenFrame {
        owner: None,
        rhs,
        position: 0,
        output: Vec::new(),
    }];

    loop {
        let complete = {
            let frame = stack.last().expect("flattening has a root frame");
            frame.output.len() > MAX_FLATTENED_RHS_LEN
                || frame.position == frame.rhs.len()
        };

        if complete {
            let frame = stack.pop().expect("flattening has a root frame");
            let too_long = frame.output.len() > MAX_FLATTENED_RHS_LEN;
            match frame.owner {
                None => {
                    return if too_long {
                        rhs.to_vec()
                    } else {
                        frame.output
                    };
                }
                Some(nonterminal) => {
                    let parent = stack.last_mut().expect("owned frame has a parent");
                    if too_long {
                        cache[nonterminal as usize] = Some(None);
                        parent.output.push(Symbol::Nonterminal(nonterminal));
                    } else {
                        cache[nonterminal as usize] = Some(Some(frame.output.clone()));
                        // The first uncached expansion does not apply the
                        // cached-path combined-length check. Preserve that
                        // detail of the reference algorithm.
                        parent.output.extend(frame.output);
                    }
                    continue;
                }
            }
        }

        let symbol = {
            let frame = stack.last_mut().expect("flattening has a root frame");
            let symbol = frame.rhs[frame.position].clone();
            frame.position += 1;
            symbol
        };

        match symbol {
            Symbol::Nonterminal(nonterminal)
                if index.expandable[nonterminal as usize] =>
            {
                if let Some(cached) = &cache[nonterminal as usize] {
                    let parent = stack.last_mut().expect("flattening has a root frame");
                    match cached {
                        Some(flattened)
                            if parent.output.len() + flattened.len()
                                <= MAX_FLATTENED_RHS_LEN =>
                        {
                            parent.output.extend_from_slice(flattened);
                        }
                        _ => parent.output.push(Symbol::Nonterminal(nonterminal)),
                    }
                } else if let Some(expanded_rhs) = index.unique_rhs(view, nonterminal) {
                    stack.push(FlattenFrame {
                        owner: Some(nonterminal),
                        rhs: expanded_rhs,
                        position: 0,
                        output: Vec::new(),
                    });
                } else {
                    stack
                        .last_mut()
                        .expect("flattening has a root frame")
                        .output
                        .push(Symbol::Nonterminal(nonterminal));
                }
            }
            symbol => {
                stack
                    .last_mut()
                    .expect("flattening has a root frame")
                    .output
                    .push(symbol);
            }
        }
    }
}

fn dedup_edit(edit: &mut Edit) {
    let keep = {
        let view = edit.view();
        let index = DedupIndex::build(view);
        let mut cache = vec![None; index.first.len()];
        let mut seen =
            FxHashSet::<(NonterminalID, Cow<'_, [Symbol]>)>::default();
        seen.reserve(view.len());

        let mut keep = Vec::with_capacity(view.len());
        for rule in view.iter() {
            if index.physical_count[rule.lhs as usize] == 1 {
                keep.push(true);
                continue;
            }
            let can_flatten = rule.rhs.iter().any(|symbol| {
                matches!(
                    symbol,
                    Symbol::Nonterminal(nonterminal)
                        if index.expandable[*nonterminal as usize]
                )
            });
            let rhs = if can_flatten {
                Cow::Owned(flatten_for_dedup(
                    &rule.rhs,
                    view,
                    &index,
                    &mut cache,
                ))
            } else {
                Cow::Borrowed(rule.rhs.as_slice())
            };
            keep.push(seen.insert((rule.lhs, rhs)));
        }
        keep
    };

    let mut keep = keep.into_iter();
    edit.order
        .retain(|_| keep.next().expect("one dedup decision per rule"));
}

fn dedup_owned(rules: &mut Vec<Rule>) {
    let mut edit = Edit::new(std::mem::take(rules));
    dedup_edit(&mut edit);
    *rules = edit.publish();
}

fn remove_unreachable_owned(rules: &mut Vec<Rule>, start: NonterminalID) {
    let n = (View::plain(rules).max_nt().max(start) as usize) + 1;
    let mut by_lhs = vec![Vec::<usize>::new(); n];
    for (index, rule) in rules.iter().enumerate() {
        by_lhs[rule.lhs as usize].push(index);
    }

    let mut reachable = vec![false; n];
    let mut todo = vec![start as usize];
    while let Some(nonterminal) = todo.pop() {
        if reachable[nonterminal] {
            continue;
        }
        reachable[nonterminal] = true;
        for &rule_index in &by_lhs[nonterminal] {
            for symbol in &rules[rule_index].rhs {
                if let Symbol::Nonterminal(child) = symbol {
                    if !reachable[*child as usize] {
                        todo.push(*child as usize);
                    }
                }
            }
        }
    }

    rules.retain(|rule| reachable[rule.lhs as usize]);
}

fn normalize_impl(rules: &mut Vec<Rule>, start: NonterminalID) {
    let profiling = profile_enabled();
    let next_nt = Cell::new(View::plain(rules).max_nt() + 1);
    let mut fresh_nt = || {
        let result = next_nt.get();
        next_nt.set(result + 1);
        result
    };

    let mut iteration = 0usize;
    let mut nullable_eliminated_before_exit = false;

    loop {
        iteration += 1;
        let before = rules.len();
        let iteration_started = profiling.then(Instant::now);
        let mut edit = Edit::new(std::mem::take(rules));

        let inline_started = profiling.then(Instant::now);
        // Edit is newly created: its physical arena is exactly the input.
        // Preserve the existing nullable-run tree and epsilon publication.
        if edit.arena.iter().any(|rule| rule.rhs.is_empty()) {
            let emitted =
                reference::inline_null_productions(&edit.arena, next_nt.get());
            edit.replace_with_owned(emitted);
        }
        // Resynchronize BEFORE reflexive-unit removal, as d751 does.
        next_nt.set(edit.view().max_nt() + 1);
        edit.retain_non_reflexive_units();
        profile(
            "inline_null_productions",
            iteration,
            inline_started,
            before,
            edit.len(),
        );

        let rr_before = edit.len();
        let rr_started = profiling.then(Instant::now);
        let right_recursion_completed =
            eliminate_right_recursion_edit(&mut edit, &mut fresh_nt);
        next_nt.set(edit.view().max_nt() + 1);
        profile(
            "eliminate_right_recursion",
            iteration,
            rr_started,
            rr_before,
            edit.len(),
        );

        let hlr_before = edit.len();
        let hlr_started = profiling.then(Instant::now);
        let nullable = Nullable::compute(edit.view());
        let hidden_left_recursion_changed =
            eliminate_hidden_left_recursion(&mut edit, &nullable);
        next_nt.set(edit.view().max_nt() + 1);
        profile(
            "compute_nullable_and_eliminate_hidden_left_recursion",
            iteration,
            hlr_started,
            hlr_before,
            edit.len(),
        );

        let dedup_before = edit.len();
        let dedup_started = profiling.then(Instant::now);
        dedup_edit(&mut edit);
        profile(
            "dedup_rules",
            iteration,
            dedup_started,
            dedup_before,
            edit.len(),
        );

        let equality_started = profiling.then(Instant::now);
        let converged = edit.equals_original();
        profile(
            "ordered_handle_convergence_check",
            iteration,
            equality_started,
            edit.len(),
            before,
        );

        // This rescan is mandatory. RR may have manufactured nullable helpers.
        let can_exit_without_post_inline =
            Nullable::compute(edit.view()).is_empty()
                && right_recursion_completed
                && !hidden_left_recursion_changed;

        let after = edit.len();
        let publish_started = profiling.then(Instant::now);
        *rules = edit.publish();
        profile(
            "publish_and_destroy_iteration",
            iteration,
            publish_started,
            before,
            after,
        );
        profile(
            "fixed_point_iteration_total",
            iteration,
            iteration_started,
            before,
            after,
        );

        if can_exit_without_post_inline {
            nullable_eliminated_before_exit = true;
            break;
        }
        if converged {
            break;
        }
    }

    let mut post_inline_changed = false;
    if !nullable_eliminated_before_exit {
        let old = std::mem::take(rules);
        let mut replacement =
            reference::inline_null_productions(&old, next_nt.get());
        next_nt.set(View::plain(&replacement).max_nt() + 1);
        replacement.retain(|rule| !is_reflexive_unit_rule(rule));
        post_inline_changed = replacement != old;
        *rules = replacement;
        // `old` is destroyed here, inside normalization.
    }

    remove_unreachable_owned(rules, start);
    if post_inline_changed {
        dedup_owned(rules);
    }
}

/// Full, ordered d751-compatible normalization.
///
/// Expensive ordered-reference execution is an explicit compiler validation
/// operation. Structural transforms and nullable rescans are never gated.
pub fn normalize_grammar(rules: &mut Vec<Rule>, start: NonterminalID) {
    let expected = reference_validation_enabled().then(|| {
        let mut expected = rules.clone();
        reference::normalize_grammar(&mut expected, start);
        expected
    });

    normalize_impl(rules, start);

    if let Some(expected) = expected {
        assert_eq!(
            *rules, expected,
            "ordered normalization differs from the d751 compiler reference"
        );
    }
}

#[cfg(test)]
#[path = "normalization_tests.rs"]
mod tests;
