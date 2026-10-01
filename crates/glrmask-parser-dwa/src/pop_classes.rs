//! Compile-time consuming POP classes. These labels never enter an artifact.
//!
//! A phase-row DEFAULT means the finite alphabet minus that row's explicit
//! labels, including explicit dead edges. Naming the complement avoids copying
//! the entire stack alphabet at every POP. Unlike the historical DEFAULT code,
//! a class always consumes one symbol and cannot establish empty-stack finality.
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use crate::automata::weighted::nwa::{NWA, NWAState};
use crate::compiler::glr::labels::{DEFAULT_LABEL, negative_to_positive_label};
use crate::ds::weight::Weight;

#[derive(Clone, Debug)]
pub struct PopLabelClasses {
    symbols: u32,
    exclusions: Vec<Arc<[u32]>>,
    interned: BTreeMap<Vec<u32>, i32>,
}

impl PopLabelClasses {
    pub fn new(symbols: u32) -> Result<Self, String> {
        if symbols >= DEFAULT_LABEL as u32 { return Err("POP alphabet collides with reserved labels".into()); }
        Ok(Self { symbols, exclusions: Vec::new(), interned: BTreeMap::new() })
    }

    pub fn symbol_count(&self) -> u32 { self.symbols }
    pub fn len(&self) -> usize { self.exclusions.len() }

    /// `None` denotes the empty complement, never a wildcard or missing proof.
    pub fn intern_complement(&mut self, explicit: impl IntoIterator<Item = u32>)
        -> Result<Option<i32>, String>
    {
        let mut values = explicit.into_iter().collect::<Vec<_>>();
        values.sort_unstable(); values.dedup();
        if values.last().is_some_and(|&v| v >= self.symbols) {
            return Err("POP complement excludes an out-of-alphabet symbol".into());
        }
        if values.len() == self.symbols as usize { return Ok(None); }
        if let Some(&label) = self.interned.get(&values) { return Ok(Some(label)); }
        let index = i32::try_from(self.exclusions.len()).map_err(|_| "too many POP classes")?;
        let label = DEFAULT_LABEL.checked_sub(1).and_then(|v| v.checked_sub(index))
            .filter(|&label| label >= self.symbols as i32)
            .ok_or("POP classes collide with the concrete alphabet")?;
        self.exclusions.push(Arc::from(values.clone()));
        self.interned.insert(values, label);
        Ok(Some(label))
    }

    fn exclusion(&self, label: i32) -> Option<&[u32]> {
        let index = DEFAULT_LABEL.checked_sub(1)?.checked_sub(label)?;
        self.exclusions.get(usize::try_from(index).ok()?).map(AsRef::as_ref)
    }

    fn first_label(&self) -> i32 { DEFAULT_LABEL - self.exclusions.len() as i32 }

    pub fn matches(&self, label: i32, symbol: u32) -> bool {
        symbol < self.symbols && self.exclusion(label)
            .is_some_and(|excluded| excluded.binary_search(&symbol).is_err())
    }

    pub(crate) fn matching_targets<'a>(&'a self, row: &'a NWAState, symbol: u32)
        -> impl Iterator<Item = &'a (u32, Weight)> + 'a
    {
        row.transitions.range(self.first_label()..DEFAULT_LABEL)
            .filter(move |(label, _)| self.matches(**label, symbol))
            .flat_map(|(_, targets)| targets.iter())
    }

    pub(crate) fn validate(&self, graph: &NWA) -> Result<(), String> {
        let count = graph.states().len();
        if graph.start_states().iter().any(|&id| id as usize >= count) {
            return Err("classed stack program has an invalid start state".into());
        }
        for row in graph.states() {
            for (&label, targets) in &row.transitions {
                let valid = if label < 0 { (negative_to_positive_label(label) as u32) < self.symbols }
                    else { (label as u32) < self.symbols || self.exclusion(label).is_some() };
                if !valid { return Err("classed stack program contains an unknown label".into()); }
                if targets.iter().any(|(id, _)| *id as usize >= count) {
                    return Err("classed stack program transition leaves its graph".into());
                }
            }
            if row.epsilons.iter().any(|(id, _)| *id as usize >= count) {
                return Err("classed stack program epsilon leaves its graph".into());
            }
        }
        Ok(())
    }

    /// Expand only after PUSH cancellation has eliminated unreachable program
    /// fragments. Boolean trimming is a safe over-approximation of weighted
    /// productivity; it never discards a nonempty correlated token coefficient.
    pub fn expand_positive(&self, graph: NWA, edge_budget: usize) -> Result<NWA, String> {
        let graph = self.trim_positive(graph)?;
        self.expand_trimmed(graph, edge_budget)
    }

    /// Compress the positive symbolic language before substituting class
    /// symbols. Weighted-language determinization commutes with this
    /// length-preserving substitution, even where different classes overlap.
    /// Classes are ordinary, distinct symbols here, never DEFAULT fallbacks.
    /// Cyclic predicates retain the existing exact expansion path because the
    /// shared ordinary weighted determinizer currently requires an acyclic NWA.
    pub fn expand_positive_compressed(&self, graph: NWA, edge_budget: usize) -> Result<NWA, String> {
        let source_states = graph.states().len();
        let graph = self.trim_positive(graph)?;
        let trimmed_states = graph.states().len();
        let graph = if !graph.states().is_empty() && graph.is_acyclic() {
            let deterministic = crate::automata::weighted::determinize::determinize(&graph)
                .map_err(|error| format!("positive POP-class determinization: {error}"))?;
            let minimized = crate::automata::weighted::minimize_acyclic::minimize_acyclic_owned(deterministic);
            if minimized.num_transitions() > edge_budget {
                return Err("symbolic POP predicate exceeds its edge budget".into());
            }
            minimized.to_nwa()
        } else { graph };
        let symbolic_states = graph.states().len();
        let symbolic_edges = graph.num_transitions();
        let expanded = self.expand_trimmed(graph, edge_budget)?;
        if std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some() {
            eprintln!("[glrmask/profile][pop_classes_positive] source_states={source_states} trimmed_states={trimmed_states} symbolic_states={symbolic_states} symbolic_edges={symbolic_edges} expanded_edges={}", expanded.num_transitions());
        }
        Ok(expanded)
    }

    fn trim_positive(&self, graph: NWA) -> Result<NWA, String> {
        self.validate(&graph)?;
        if graph.states().iter().any(|row| row.transitions.keys().any(|&label| label < 0)) {
            return Err("POP expansion requires a negative-free program".into());
        }
        let count = graph.states().len();
        let mut predecessors = vec![Vec::new(); count];
        let mut live = vec![false; count];
        let mut todo = VecDeque::new();
        for (id, row) in graph.states().iter().enumerate() {
            if row.final_weight.as_ref().is_some_and(|w| !w.is_empty()) {
                live[id] = true; todo.push_back(id);
            }
            for (target, weight) in row.epsilons.iter().chain(row.transitions.values().flatten()) {
                if !weight.is_empty() { predecessors[*target as usize].push(id); }
            }
        }
        while let Some(id) = todo.pop_front() {
            for &previous in &predecessors[id] {
                if !live[previous] { live[previous] = true; todo.push_back(previous); }
            }
        }
        let mut reachable = vec![false; count];
        for &id in graph.start_states() {
            if live[id as usize] && !reachable[id as usize] {
                reachable[id as usize] = true; todo.push_back(id as usize);
            }
        }
        while let Some(id) = todo.pop_front() {
            let row = &graph.states()[id];
            for (target, weight) in row.epsilons.iter().chain(row.transitions.values().flatten()) {
                let target = *target as usize;
                if !weight.is_empty() && live[target] && !reachable[target] {
                    reachable[target] = true; todo.push_back(target);
                }
            }
        }
        let mut remap = vec![u32::MAX; count];
        let mut states = Vec::<NWAState>::new();
        for (id, retained) in reachable.iter().enumerate() {
            if *retained { remap[id] = states.len() as u32; states.push(NWAState::default()); }
        }
        for (id, source) in graph.states().iter().enumerate() {
            if !reachable[id] { continue; }
            let row = &mut states[remap[id] as usize];
            row.final_weight = source.final_weight.clone();
            for (target, weight) in &source.epsilons {
                if weight.is_empty() || !reachable[*target as usize] { continue; }
                row.epsilons.push((remap[*target as usize], weight.clone()));
            }
            for (&label, targets) in &source.transitions {
                let retained = targets.iter().filter(|(target, weight)|
                    !weight.is_empty() && reachable[*target as usize])
                    .map(|(target, weight)| (remap[*target as usize], weight.clone())).collect::<Vec<_>>();
                if retained.is_empty() { continue; }
                row.transitions.insert(label, retained);
            }
        }
        let starts = graph.start_states().iter().filter_map(|&id| reachable[id as usize].then_some(remap[id as usize])).collect();
        Ok(NWA::from_parts(states, starts))
    }

    fn expand_trimmed(&self, graph: NWA, edge_budget: usize) -> Result<NWA, String> {
        let mut states = vec![NWAState::default(); graph.states().len()];
        let mut edges = 0usize;
        for (source, row) in graph.states().iter().zip(&mut states) {
            row.final_weight = source.final_weight.clone();
            edges = edges.checked_add(source.epsilons.len()).ok_or("POP expansion edge overflow")?;
            if edges > edge_budget { return Err("surviving POP expansion exceeds its edge budget".into()); }
            row.epsilons = source.epsilons.clone();
            for (&label, retained) in &source.transitions {
                let labels: Box<dyn Iterator<Item = i32> + '_> = if let Some(excluded) = self.exclusion(label) {
                    Box::new((0..self.symbols).filter(|symbol| excluded.binary_search(symbol).is_err()).map(|s| s as i32))
                } else { Box::new(std::iter::once(label)) };
                for label in labels {
                    edges = edges.checked_add(retained.len()).ok_or("POP expansion edge overflow")?;
                    if edges > edge_budget { return Err("surviving POP expansion exceeds its edge budget".into()); }
                    row.transitions.entry(label).or_default().extend(retained.iter().cloned());
                }
            }
        }
        Ok(NWA::from_parts(states, graph.start_states().to_vec()))
    }
}

#[cfg(test)]
#[path = "pop_classes_tests.rs"]
mod tests;
