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
    domains: Vec<std::ops::Range<u32>>,
    interned: BTreeMap<(u32, u32, Vec<u32>), i32>,
}

impl PopLabelClasses {
    pub fn new(symbols: u32) -> Result<Self, String> {
        if symbols >= DEFAULT_LABEL as u32 { return Err("POP alphabet collides with reserved labels".into()); }
        Ok(Self { symbols, exclusions: Vec::new(), domains: Vec::new(), interned: BTreeMap::new() })
    }

    pub fn symbol_count(&self) -> u32 { self.symbols }
    pub fn len(&self) -> usize { self.exclusions.len() }

    /// `None` denotes the empty complement, never a wildcard or missing proof.
    pub fn intern_complement(&mut self, explicit: impl IntoIterator<Item = u32>)
        -> Result<Option<i32>, String>
    {
        self.intern_scoped_complement(0..self.symbols, explicit)
    }

    /// DEFAULT over one component range. Foreign symbols are outside this
    /// class rather than copied into a per-row exclusion list.
    pub fn intern_scoped_complement(&mut self, domain: std::ops::Range<u32>,
        explicit: impl IntoIterator<Item = u32>) -> Result<Option<i32>, String> {
        if domain.start > domain.end || domain.end > self.symbols {
            return Err("POP class domain outside parser alphabet".into());
        }
        let mut values = explicit.into_iter().collect::<Vec<_>>();
        values.sort_unstable(); values.dedup();
        if values.iter().any(|v| !domain.contains(v)) {
            return Err("POP complement excludes an out-of-domain symbol".into());
        }
        if values.len() == (domain.end - domain.start) as usize { return Ok(None); }
        let key = (domain.start, domain.end, values.clone());
        if let Some(&label) = self.interned.get(&key) { return Ok(Some(label)); }
        let index = i32::try_from(self.exclusions.len()).map_err(|_| "too many POP classes")?;
        let label = DEFAULT_LABEL.checked_sub(1).and_then(|v| v.checked_sub(index))
            .filter(|&label| label >= self.symbols as i32)
            .ok_or("POP classes collide with the concrete alphabet")?;
        self.exclusions.push(Arc::from(values.clone()));
        self.domains.push(domain);
        self.interned.insert(key, label);
        Ok(Some(label))
    }

    pub(crate) fn exclusion(&self, label: i32) -> Option<&[u32]> {
        let index = DEFAULT_LABEL.checked_sub(1)?.checked_sub(label)?;
        self.exclusions.get(usize::try_from(index).ok()?).map(AsRef::as_ref)
    }

    fn first_label(&self) -> i32 { DEFAULT_LABEL - self.exclusions.len() as i32 }

    pub fn matches(&self, label: i32, symbol: u32) -> bool {
        self.domain(label).is_some_and(|domain| domain.contains(&symbol)) && self.exclusion(label)
            .is_some_and(|excluded| excluded.binary_search(&symbol).is_err())
    }

    pub(crate) fn matching_symbol_count(&self, label: i32) -> usize {
        let domain = self.domain(label).expect("known POP class");
        (domain.end - domain.start) as usize - self.exclusion(label).unwrap().len()
    }

    pub(crate) fn matching_symbols(&self, label: i32) -> impl Iterator<Item = u32> + '_ {
        self.domain(label).expect("known POP class").clone()
            .filter(move |symbol| self.exclusion(label).unwrap().binary_search(symbol).is_err())
    }

    fn domain(&self, label: i32) -> Option<&std::ops::Range<u32>> {
        let index = DEFAULT_LABEL.checked_sub(1)?.checked_sub(label)?;
        self.domains.get(usize::try_from(index).ok()?)
    }

    /// Only final concrete parser-DWA normalization expands domain exceptions.
    /// Cancellation and query assembly retain the symbolic range above.
    pub(crate) fn excluded_symbols(&self, label: i32) -> impl Iterator<Item = u32> + '_ {
        let domain = self.domain(label).expect("known POP class");
        (0..domain.start).chain(self.exclusion(label).unwrap().iter().copied())
            .chain(domain.end..self.symbols)
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

    /// Publish an already deterministic positive class predicate directly.
    /// Its opaque labels and weights are already normalized by the shared
    /// template compiler; repeating weighted NWA subset construction changes
    /// storage only and can multiply equivalent conditional rows.
    pub fn compile_positive_dwa(
        &self, symbolic: crate::automata::weighted::dwa::DWA, edge_budget: usize,
    ) -> Result<crate::automata::weighted::dwa::DWA, String> {
        let started=std::time::Instant::now();
        let count=symbolic.states().len();
        if count==0 || count>1_000_000 || symbolic.start_state() as usize>=count
            || symbolic.num_transitions()>edge_budget {
            return Err("positive class DWA exceeds its coordinate or representation budget".into());
        }
        for row in symbolic.states() {for (label,target,_) in row.transitions.entries() {
            if label<0 || !((label as u32)<self.symbols || self.exclusion(label).is_some())
                || target as usize>=count {
                return Err("positive class DWA contains an invalid consuming transition".into());
            }
        }}
        let result=crate::parser_dwa::determinize_parser_dwa_with_pop_classes(&symbolic,self,edge_budget)?;
        if std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some() {
            eprintln!("[glrmask/profile][pop_class_direct_publication] input_states={count} input_edges={} result_states={} result_edges={} classes={} elapsed_ms={:.3}",
                symbolic.num_transitions(),result.num_states(),result.num_transitions(),self.len(),started.elapsed().as_secs_f64()*1000.0);
        }
        Ok(result)
    }

    /// Compile finite class substitution inside the existing weighted subset
    /// kernel, without expanding every NWA/DWA edge across the stack alphabet.
    /// The returned DEFAULT rows are exact complete derivatives, not the
    /// historical LR-domain shortcuts. Preserve explicit empty exceptions;
    /// ordinary symbol-language minimization must not remove their shadows.
    pub fn compile_positive(&self, graph: NWA, edge_budget: usize)
        -> Result<crate::automata::weighted::dwa::DWA, String>
    {
        self.compile_positive_with_minimizer(graph, edge_budget,
            crate::automata::weighted::minimize_acyclic::minimize_acyclic_owned)
    }

    /// Boundary masks observe final coefficients at every consumed prefix.
    /// A caller may use that established exact quotient while class labels
    /// are still opaque symbols, before their finite consuming substitution.
    pub fn compile_positive_with_minimizer(
        &self, graph: NWA, edge_budget: usize,
        minimize: impl FnOnce(crate::automata::weighted::dwa::DWA) -> crate::automata::weighted::dwa::DWA,
    ) -> Result<crate::automata::weighted::dwa::DWA, String>
    {
        let started = std::time::Instant::now();
        let graph = self.trim_positive(graph)?;
        let input_states = graph.states().len();
        let symbolic = if !graph.states().is_empty() && graph.is_acyclic() {
            let deterministic = crate::automata::weighted::determinize::determinize(&graph)
                .map_err(|error| format!("positive POP-class determinization: {error}"))?;
            minimize(deterministic)
        } else {
            crate::parser_dwa::determinize_opaque_stack_symbols(&graph, self.symbols)
        };
        if symbolic.num_states() > 1_000_000 || symbolic.num_transitions() > edge_budget {
            return Err("symbolic POP predicate exceeds its representation budget".into());
        }
        let result = crate::parser_dwa::determinize_parser_dwa_with_pop_classes(&symbolic, self, edge_budget)?;
        if std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some() {
            eprintln!("[glrmask/profile][pop_class_derivatives] input_states={input_states} symbolic_states={} symbolic_edges={} result_states={} result_edges={} classes={} elapsed_ms={:.3}",
                symbolic.num_states(), symbolic.num_transitions(), result.num_states(), result.num_transitions(), self.len(), started.elapsed().as_secs_f64()*1000.0);
        }
        Ok(result)
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
                    Box::new(self.domain(label).unwrap().clone().filter(|symbol| excluded.binary_search(symbol).is_err()).map(|s| s as i32))
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

#[cfg(test)]
mod group_reuse_tests {
    use super::*;
    use crate::automata::weighted::dwa::{DWA, DWAState};
    use range_set_blaze::RangeSetBlaze;

    fn test_weight(bits: u8) -> Weight {
        Weight::from_uniform(
            0..=0,
            (0..3u32)
                .filter(|&id| bits & (1 << id) != 0)
                .collect::<RangeSetBlaze<u32>>(),
        )
    }

    fn test_bits(weight: &Weight) -> u8 {
        (0..3).fold(0, |result, token| {
            result | if weight.tokens_for_tsid(0).contains(token) { 1 << token } else { 0 }
        })
    }

    fn literal_oracle(graph: &NWA, classes: &PopLabelClasses, stack: &[u32], prefix: bool) -> u8 {
        let mut reached = BTreeMap::<(u32, Vec<u32>), u8>::new();
        let mut queue = VecDeque::new();
        for &state in graph.start_states() {
            reached.insert((state, stack.to_vec()), 7);
            queue.push_back((state, stack.to_vec()));
        }
        let mut result = 0;
        while let Some(key) = queue.pop_front() {
            let live = reached[&key];
            let (state, cur_stack) = key;
            let row = &graph.states()[state as usize];
            if prefix || cur_stack.is_empty() {
                if let Some(final_weight) = &row.final_weight {
                    result |= live & test_bits(final_weight);
                }
            }
            let mut add = |target: u32, next_stack: Vec<u32>, coefficient: &Weight| {
                let value = live & test_bits(coefficient);
                if value == 0 {
                    return;
                }
                let key = (target, next_stack);
                let previous = reached.entry(key.clone()).or_default();
                if value & !*previous != 0 {
                    *previous |= value;
                    queue.push_back(key);
                }
            };
            for (target, coefficient) in &row.epsilons {
                add(*target, cur_stack.clone(), coefficient);
            }
            for (&label, targets) in &row.transitions {
                let mut next = cur_stack.clone();
                let Some(top) = next.pop() else { continue; };
                if top as i32 != label && !classes.matches(label, top) {
                    continue;
                }
                for (target, coefficient) in targets {
                    add(*target, next.clone(), coefficient);
                }
            }
            assert!(reached.len() < 100_000, "literal oracle finite graph check");
        }
        result
    }

    fn prefix_accepted(dwa: &DWA, stack: &[u32]) -> u8 {
        let mut state = dwa.start_state();
        let mut live = 7;
        let mut result = dwa.states()[state as usize]
            .final_weight
            .as_ref()
            .map_or(0, test_bits);
        for &top in stack.iter().rev() {
            let row = &dwa.states()[state as usize];
            let Some((target, coefficient)) = row
                .transitions
                .get(&(top as i32))
                .or_else(|| row.transitions.get(&DEFAULT_LABEL))
            else {
                break;
            };
            live &= test_bits(coefficient);
            state = *target;
            result |= live
                & dwa.states()[state as usize]
                    .final_weight
                    .as_ref()
                    .map_or(0, test_bits);
        }
        result
    }

    fn full_accepted(dwa: &DWA, stack: &[u32]) -> u8 {
        let mut state = dwa.start_state();
        let mut live = 7;
        for &top in stack.iter().rev() {
            let row = &dwa.states()[state as usize];
            let Some((target, coefficient)) = row
                .transitions
                .get(&(top as i32))
                .or_else(|| row.transitions.get(&DEFAULT_LABEL))
            else {
                return 0;
            };
            live &= test_bits(coefficient);
            if live == 0 {
                return 0;
            }
            state = *target;
        }
        live & dwa.states()[state as usize]
            .final_weight
            .as_ref()
            .map_or(0, test_bits)
    }

    fn expand_to_literal_nwa(graph: &NWA, classes: &PopLabelClasses) -> NWA {
        let mut states = vec![NWAState::default(); graph.states().len()];
        for (source, row) in graph.states().iter().zip(&mut states) {
            row.final_weight = source.final_weight.clone();
            row.epsilons = source.epsilons.clone();
            for (&label, targets) in &source.transitions {
                if let Some(excluded) = classes.exclusion(label) {
                    let domain = classes.domain(label).unwrap();
                    for sym in domain.clone().filter(|s| excluded.binary_search(s).is_err()) {
                        row.transitions.entry(sym as i32).or_default().extend(targets.iter().cloned());
                    }
                } else {
                    row.transitions.entry(label).or_default().extend(targets.iter().cloned());
                }
            }
        }
        NWA::from_parts(states, graph.start_states().to_vec())
    }

    fn nwa_to_dwa(graph: &NWA) -> DWA {
        let mut states = Vec::with_capacity(graph.states().len());
        for source in graph.states() {
            let mut state = DWAState::default();
            state.final_weight = source.final_weight.clone();
            for (&label, targets) in &source.transitions {
                assert!(targets.len() <= 1, "graph row must be deterministic for direct DWA");
                if let Some((target, weight)) = targets.first() {
                    state.transitions.insert(label, (*target, weight.clone()));
                }
            }
            states.push(state);
        }
        let start = *graph.start_states().first().expect("start state");
        DWA::from_parts(states, start)
    }

    fn all_words(depth: usize, alphabet: u32) -> Vec<Vec<u32>> {
        let mut all = vec![vec![]];
        let mut layer = vec![vec![]];
        for _ in 0..depth {
            let mut next = Vec::new();
            for word in layer {
                for sym in 0..alphabet {
                    let mut w = word.clone();
                    w.push(sym);
                    next.push(w);
                }
            }
            all.extend(next.iter().cloned());
            layer = next;
        }
        all
    }

    #[test]
    fn regression_pop_class_group_reuse_matches_literal_expansion() {
        let mut classes = PopLabelClasses::new(7).unwrap();
        // Class A: domain 0..5, excluding [4]. (matches 0, 1, 2, 3)
        let class_a = classes.intern_scoped_complement(0..5, [4]).unwrap().unwrap();
        // Class B: domain 2..6, excluding [3, 4]. (matches 2, 5)
        let class_b = classes.intern_scoped_complement(2..6, [3, 4]).unwrap().unwrap();

        assert!(classes.matches(class_a, 0));
        assert!(classes.matches(class_a, 1));
        assert!(classes.matches(class_a, 2));
        assert!(classes.matches(class_a, 3));
        assert!(!classes.matches(class_a, 4));
        assert!(!classes.matches(class_a, 5));
        assert!(!classes.matches(class_a, 6));

        assert!(!classes.matches(class_b, 0));
        assert!(!classes.matches(class_b, 1));
        assert!(classes.matches(class_b, 2));
        assert!(!classes.matches(class_b, 3));
        assert!(!classes.matches(class_b, 4));
        assert!(classes.matches(class_b, 5));
        assert!(!classes.matches(class_b, 6));

        let mut graph = NWA::new(1, 3);
        let start = graph.add_state();
        let s1 = graph.add_state();
        let s2 = graph.add_state();
        let s3 = graph.add_state();
        let s4 = graph.add_state();

        graph.set_start_states(vec![start]);
        graph.set_final_weight(s1, test_weight(1)); // bit 0
        graph.set_final_weight(s2, test_weight(2)); // bit 1
        graph.set_final_weight(s4, test_weight(6)); // bits 1, 2

        // Root row:
        // Classes A and B carry different correlated weights
        graph.add_transition(start, class_a, s1, test_weight(1));
        graph.add_transition(start, class_b, s2, test_weight(2));

        // Explicit 0 -> s3 weight 2, explicit 1 -> s3 weight 4 (same target, different observable bits)
        graph.add_transition(start, 0, s3, test_weight(2));
        graph.add_transition(start, 1, s3, test_weight(4));

        // Source explicit dead 4 -> s4 Weight::empty()
        graph.add_transition(start, 4, s4, Weight::empty());

        // Following row s3: SAME membership but DIFFERENT class coefficients (weight 6)
        graph.add_transition(s3, class_a, s4, test_weight(6));
        graph.add_transition(s3, class_b, s4, test_weight(6));

        // Live weighted loop on s4 on class_a (weight 6)
        graph.add_transition(s4, class_a, s4, test_weight(6));

        let literal_nwa = expand_to_literal_nwa(&graph, &classes);
        let empty_classes = PopLabelClasses::new(7).unwrap();

        // Exercise production direct compile_positive_dwa
        let dwa_direct = classes.compile_positive_dwa(nwa_to_dwa(&graph), 100_000).unwrap();
        let dwa = classes.compile_positive(graph.clone(), 100_000).unwrap();

        // Concrete assertions:
        // [] rejects
        assert_eq!(full_accepted(&dwa_direct, &[]), 0);
        assert_eq!(prefix_accepted(&dwa_direct, &[]), 0);
        assert_eq!(literal_oracle(&graph, &classes, &[], false), 0);
        assert_eq!(literal_oracle(&graph, &classes, &[], true), 0);

        // [0] root prefix 1, [1] root prefix 1
        assert_eq!(prefix_accepted(&dwa_direct, &[0]), 1);
        assert_eq!(prefix_accepted(&dwa_direct, &[1]), 1);
        assert_eq!(literal_oracle(&graph, &classes, &[0], true), 1);
        assert_eq!(literal_oracle(&graph, &classes, &[1], true), 1);

        // Consuming matching symbols after 0 allows bit 1 (value 2), after 1 allows bit 2 (value 4)
        assert_eq!(full_accepted(&dwa_direct, &[2, 0]), 2);
        assert_eq!(full_accepted(&dwa_direct, &[2, 1]), 4);
        assert_eq!(literal_oracle(&graph, &classes, &[2, 0], false), 2);
        assert_eq!(literal_oracle(&graph, &classes, &[2, 1], false), 4);

        // Continuing through live loop on s4 preserves bit 1 and bit 2
        assert_eq!(full_accepted(&dwa_direct, &[0, 2, 0]), 2);
        assert_eq!(full_accepted(&dwa_direct, &[0, 2, 1]), 4);
        assert_eq!(literal_oracle(&graph, &classes, &[0, 2, 0], false), 2);
        assert_eq!(literal_oracle(&graph, &classes, &[0, 2, 1], false), 4);

        // Symbol 4 is explicitly dead
        assert_eq!(full_accepted(&dwa_direct, &[4]), 0);
        assert_eq!(prefix_accepted(&dwa_direct, &[4]), 0);

        // Check exact prefix and full acceptance for all words through depth 4 over alphabet 0..7
        for word in all_words(4, 7) {
            let direct_prefix = prefix_accepted(&dwa_direct, &word);
            let direct_full = full_accepted(&dwa_direct, &word);
            let oracle_prefix = literal_oracle(&graph, &classes, &word, true);
            let oracle_full = literal_oracle(&graph, &classes, &word, false);
            let literal_prefix = literal_oracle(&literal_nwa, &empty_classes, &word, true);
            let literal_full = literal_oracle(&literal_nwa, &empty_classes, &word, false);

            assert_eq!(direct_prefix, oracle_prefix, "prefix mismatch on word {:?}", word);
            assert_eq!(direct_full, oracle_full, "full mismatch on word {:?}", word);
            assert_eq!(direct_prefix, literal_prefix, "literal prefix mismatch on word {:?}", word);
            assert_eq!(direct_full, literal_full, "literal full mismatch on word {:?}", word);

            let dwa_prefix = prefix_accepted(&dwa, &word);
            let dwa_full = full_accepted(&dwa, &word);
            assert_eq!(dwa_prefix, direct_prefix, "compile_positive prefix mismatch on word {:?}", word);
            assert_eq!(dwa_full, direct_full, "compile_positive full mismatch on word {:?}", word);
        }

        // Budget refusal
        assert!(classes.compile_positive(graph.clone(), 1).is_err(), "edge budget = 1 must be refused");
        assert!(classes.compile_positive_dwa(nwa_to_dwa(&graph), 1).is_err(), "direct edge budget = 1 must be refused");
    }

    #[test]
    fn regression_pop_class_arbitrary_word_membership_over_64_classes() {
        let mut classes = PopLabelClasses::new(9).unwrap();
        // FIRST intern distinguishing class domain 0..2 excludes [1] -> matches 0 only
        // (this is LAST BTree active-class bit 64)
        let dist_class = classes.intern_scoped_complement(0..2, [1]).unwrap().unwrap();
        assert!(classes.matches(dist_class, 0));
        assert!(!classes.matches(dist_class, 1));

        // Intern 64 distinct foreign classes domain 2..9 with exclusions derived from 6-bit masks 0..63
        // over symbols 2..7 (symbol 8 always included).
        let mut foreign_classes = Vec::with_capacity(64);
        for mask in 0..64u32 {
            let mut exclusions = Vec::new();
            for bit in 0..6 {
                if (mask & (1 << bit)) != 0 {
                    exclusions.push(2 + bit as u32);
                }
            }
            let c = classes
                .intern_scoped_complement(2..9, exclusions)
                .unwrap()
                .unwrap();
            foreign_classes.push(c);
        }
        assert_eq!(classes.len(), 65);

        let mut graph = NWA::new(1, 3);
        let start = graph.add_state();
        let target = graph.add_state();
        graph.set_start_states(vec![start]);
        graph.set_final_weight(target, test_weight(7));

        // root distinguishing class -> target weight 2
        graph.add_transition(start, dist_class, target, test_weight(2));
        // remaining 64 classes -> same target weight 1
        for &c in &foreign_classes {
            graph.add_transition(start, c, target, test_weight(1));
        }

        let literal_nwa = expand_to_literal_nwa(&graph, &classes);
        let empty_classes = PopLabelClasses::new(9).unwrap();

        // Direct compile_positive_dwa
        let dwa_direct = classes.compile_positive_dwa(nwa_to_dwa(&graph), 100_000).unwrap();
        let dwa = classes.compile_positive(graph.clone(), 100_000).unwrap();

        // Assert [0] exact 2 and [1] exact 0
        assert_eq!(full_accepted(&dwa_direct, &[0]), 2);
        assert_eq!(prefix_accepted(&dwa_direct, &[0]), 2);
        assert_eq!(full_accepted(&dwa_direct, &[1]), 0);
        assert_eq!(prefix_accepted(&dwa_direct, &[1]), 0);

        assert_eq!(full_accepted(&dwa, &[0]), 2);
        assert_eq!(prefix_accepted(&dwa, &[0]), 2);
        assert_eq!(full_accepted(&dwa, &[1]), 0);
        assert_eq!(prefix_accepted(&dwa, &[1]), 0);

        // Independent literal NWA compare all words depth 2 alphabet 9 prefix/full
        for word in all_words(2, 9) {
            let direct_full = full_accepted(&dwa_direct, &word);
            let direct_prefix = prefix_accepted(&dwa_direct, &word);
            let oracle_full = literal_oracle(&graph, &classes, &word, false);
            let oracle_prefix = literal_oracle(&graph, &classes, &word, true);
            let literal_full = literal_oracle(&literal_nwa, &empty_classes, &word, false);
            let literal_prefix = literal_oracle(&literal_nwa, &empty_classes, &word, true);

            assert_eq!(direct_full, oracle_full, "direct full mismatch on word {:?}", word);
            assert_eq!(direct_prefix, oracle_prefix, "direct prefix mismatch on word {:?}", word);
            assert_eq!(direct_full, literal_full, "literal full mismatch on word {:?}", word);
            assert_eq!(direct_prefix, literal_prefix, "literal prefix mismatch on word {:?}", word);
        }

        // Budget refusal
        assert!(classes.compile_positive(graph.clone(), 1).is_err(), "edge budget = 1 must be refused");
        assert!(classes.compile_positive_dwa(nwa_to_dwa(&graph), 1).is_err(), "direct edge budget = 1 must be refused");
    }
}
