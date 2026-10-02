//! Exact input-domain projection of an acyclic split stack transducer.
//!
//! This compiler requires only the automata: no LR action/goto table, grammar,
//! terminal analysis, or concrete output-stack enumeration. A productive PUSH
//! suffix is existentially eliminated. READ chains repeatedly inspect the same
//! current top, unlike POP edges which consume successive top-first symbols.
//!
//! The resulting deterministic prefix recognizer is a bottom-up hash-consed
//! DAG. Acceptance means that *some* template output exists, so an accepting
//! prefix accepts every remaining stack suffix. An explicit rejection edge is
//! retained when it overrides an otherwise productive DEFAULT transition.

use std::collections::{BTreeMap, VecDeque};
#[cfg(test)]
use std::collections::BTreeSet;
use rustc_hash::FxHashMap;

use crate::automata::unweighted_u32::dfa::DFA;
use crate::compiler::glr::labels::DEFAULT_LABEL;
use crate::runtime::CommitTemplateDfas;

const REJECT: u32 = u32::MAX;

/// Exact result after inspecting a possibly incomplete top-first stack prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainProbe {
    /// An output exists, independently of the unseen lower stack.
    Accept,
    /// No output can exist, independently of the unseen lower stack.
    Reject,
    /// More concrete stack symbols are required. The integer is a domain-DAG
    /// cursor, not an LR state; it is meaningful only for its owning domain.
    NeedMore(u32),
}

/// A template-derived certificate about one top value, with the lower stack
/// completely unknown (and allowed to be empty).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopAdmission {
    Never,
    Always,
    DependsOnSuffix,
}

#[derive(Debug, Clone)]
struct DomainState {
    accepts_prefix: bool,
    default_target: u32,
    first_edge: usize,
    edge_count: usize,
}

/// Compact exact admissibility program derived entirely from template data.
///
/// Construction validates all component graphs and phase links, including
/// unreachable nodes. Runtime queries allocate no output stacks or heap scratch.
#[derive(Debug, Clone)]
pub struct TemplateDomain {
    start: u32,
    states: Box<[DomainState]>,
    edges: Box<[(u32, u32)]>,
}

#[derive(Debug, Clone, Copy)]
enum Phase { Pop, Read, Push }

fn topological_order(dfa: &DFA, phase: Phase) -> Result<(Vec<usize>, Option<u32>), String> {
    if !dfa.states.is_empty() && dfa.start_state as usize >= dfa.states.len() {
        return Err(format!("{phase:?} start {} outside {} states", dfa.start_state, dfa.states.len()));
    }
    if dfa.states.is_empty() && dfa.start_state != 0 {
        return Err(format!("empty {phase:?} DFA has nonzero start {}", dfa.start_state));
    }
    if dfa.states.len() >= REJECT as usize {
        return Err(format!("{phase:?} state count exceeds the template domain coordinate"));
    }
    let mut indegree = vec![0usize; dfa.states.len()];
    let mut largest_symbol = None;
    for (source, state) in dfa.states.iter().enumerate() {
        for (&label, &target) in &state.transitions {
            let legal = match phase {
                Phase::Pop => label >= 0,
                Phase::Read => label >= 0 && label != DEFAULT_LABEL,
                Phase::Push => label < 0,
            };
            if !legal {
                return Err(format!("{phase:?} state {source} contains wrong-phase label {label}"));
            }
            if label != DEFAULT_LABEL {
                let symbol = if label < 0 { label.wrapping_sub(i32::MIN) as u32 } else { label as u32 };
                largest_symbol = Some(largest_symbol.map_or(symbol, |old: u32| old.max(symbol)));
            }
            let Some(degree) = indegree.get_mut(target as usize) else {
                return Err(format!("{phase:?} state {source} targets missing state {target}"));
            };
            *degree += 1;
        }
    }
    let mut pending: VecDeque<usize> = indegree.iter().enumerate()
        .filter_map(|(id, &degree)| (degree == 0).then_some(id)).collect();
    let mut order = Vec::with_capacity(dfa.states.len());
    while let Some(source) = pending.pop_front() {
        order.push(source);
        for &target in dfa.states[source as usize].transitions.values() {
            indegree[target as usize] -= 1;
            if indegree[target as usize] == 0 { pending.push_back(target as usize); }
        }
    }
    if order.len() != dfa.states.len() {
        return Err(format!("{phase:?} template is cyclic"));
    }
    Ok((order, largest_symbol))
}

fn validate_links(links: &[Option<u32>], source_len: usize, target_len: usize, name: &str) -> Result<(), String> {
    // Omitted trailing links have the same meaning as the existing evaluator's
    // `.get()`: no phase transition. Extra source entries are malformed.
    if links.len() > source_len {
        return Err(format!("{name} has {} links for {source_len} source states", links.len()));
    }
    for (source, target) in links.iter().enumerate() {
        if let Some(target) = target && *target as usize >= target_len {
            return Err(format!("{name} at state {source} targets missing state {target}"));
        }
    }
    Ok(())
}

fn linked_productive(links: &[Option<u32>], source: usize, productive: &[bool]) -> bool {
    links.get(source).copied().flatten().is_some_and(|target| productive[target as usize])
}

/// Merge two already-ordered sources of root transitions. READ acceptance
/// dominates a POP continuation on the same symbol; a POP's explicit dead
/// target remains an exception when DEFAULT is productive. Omit only edges
/// whose target is exactly the default target.
fn merged_domain_row(
    transitions: &BTreeMap<i32, u32>,
    canonical: &[u32],
    read_labels: Option<&[u32]>,
    default: u32,
) -> Vec<(u32, u32)> {
    let mut pop = transitions.iter().filter(|(label, _)| **label != DEFAULT_LABEL)
        .map(|(&label, &target)| (label as u32, canonical[target as usize])).peekable();
    let read_len = read_labels.map_or(0, <[u32]>::len);
    let mut row = Vec::with_capacity(transitions.len().saturating_add(read_len));
    for &label in read_labels.into_iter().flatten() {
        while pop.peek().is_some_and(|&(top, _)| top < label) {
            let edge = pop.next().unwrap();
            if edge.1 != default { row.push(edge); }
        }
        if pop.peek().is_some_and(|&(top, _)| top == label) { pop.next(); }
        if default != 0 { row.push((label, 0)); }
    }
    row.extend(pop.filter(|&(_, target)| target != default));
    row
}

/// A borrow of an immutable, completely validated split stack program.
///
/// The validation covers every node, including unreachable nodes, every phase
/// label and link, and acyclicity. The graph orders are preparation scratch:
/// consumers may share them but cannot construct this proof for another graph.
/// No parser table, output-stack enumeration, or persisted trusted flag is used.
#[derive(Debug)]
pub struct ValidatedTemplate<'a> {
    template: &'a CommitTemplateDfas,
    orders: [Vec<usize>; 3],
    largest_symbol: Option<u32>,
}

impl<'a> ValidatedTemplate<'a> {
    pub fn new(template: &'a CommitTemplateDfas) -> Result<Self, String> {
        let (pop, pop_symbol) = topological_order(&template.pop, Phase::Pop)?;
        let (read, read_symbol) = topological_order(&template.read, Phase::Read)?;
        let (push, push_symbol) = topological_order(&template.push, Phase::Push)?;
        validate_links(&template.pop_to_read, template.pop.states.len(), template.read.states.len(), "POP->READ")?;
        validate_links(&template.pop_to_push, template.pop.states.len(), template.push.states.len(), "POP->PUSH")?;
        validate_links(&template.read_to_push, template.read.states.len(), template.push.states.len(), "READ->PUSH")?;
        let largest_symbol = [pop_symbol, read_symbol, push_symbol].into_iter().flatten().max();
        Ok(Self { template, orders: [pop, read, push], largest_symbol })
    }

    pub fn template(&self) -> &'a CommitTemplateDfas { self.template }
    pub fn orders(&self) -> &[Vec<usize>; 3] { &self.orders }
    pub fn push_order(&self) -> &[usize] { &self.orders[2] }

    /// Alphabet checks reuse the maximum observed during graph validation,
    /// including labels in unreachable states. This is not a promise about an
    /// unchecked caller-supplied bound or a flag read from an artifact.
    pub fn validate_alphabet(&self, symbols: u32) -> Result<(), String> {
        if let Some(symbol) = self.largest_symbol && symbol >= symbols {
            return Err(format!("template stack symbol {symbol} outside alphabet {symbols}"));
        }
        Ok(())
    }
}

impl TemplateDomain {
    pub fn compile(template: &CommitTemplateDfas) -> Result<Self, String> {
        Self::from_validated(&ValidatedTemplate::new(template)?)
    }

    /// Derive the exact input domain while reusing a complete graph validation.
    pub fn from_validated(validated: &ValidatedTemplate<'_>) -> Result<Self, String> {
        let template = validated.template();
        let [pop_order, read_order, push_order] = validated.orders();

        let mut push_good = vec![false; template.push.states.len()];
        for &id in push_order.iter().rev() {
            let state = &template.push.states[id as usize];
            push_good[id as usize] = state.is_accepting
                || state.transitions.values().any(|&target| push_good[target as usize]);
        }
        let mut read_without_input = vec![false; template.read.states.len()];
        // Each input row already has strictly ordered, unique labels. Filtering
        // it preserves those properties; no ordered-tree insertion is needed
        // for the derived READ certificate set.
        let mut read_labels = vec![Vec::<u32>::new(); template.read.states.len()];
        for &id in read_order.iter().rev() {
            let i = id as usize;
            let state = &template.read.states[i];
            read_without_input[i] = state.is_accepting
                || linked_productive(&template.read_to_push, i, &push_good);
            if read_without_input[i] { continue; }
            let labels: Vec<u32> = state.transitions.iter().filter_map(|(&label, &target)| {
                let target = target as usize;
                (read_without_input[target] || read_labels[target].binary_search(&(label as u32)).is_ok())
                    .then_some(label as u32)
            }).collect();
            read_labels[i] = labels;
        }

        // ID zero is universal prefix acceptance. REJECT has no allocated
        // state. Every interned nonterminal row is productive; all children
        // precede parents, so the compiled graph itself cannot contain cycles.
        let mut states = vec![DomainState {
            accepts_prefix: true, default_target: REJECT, first_edge: 0, edge_count: 0,
        }];
        let mut edges = Vec::new();
        let mut canonical = vec![REJECT; template.pop.states.len()];
        // This map is used only for equality lookup. IDs are still assigned
        // by the fixed topological traversal, never by hash-map iteration.
        let mut row_ids: FxHashMap<(u32, Vec<(u32, u32)>), u32> = FxHashMap::default();
        for &id in pop_order.iter().rev() {
            let i = id as usize;
            let state = &template.pop.states[i];
            let read = template.pop_to_read.get(i).copied().flatten().map(|v| v as usize);
            if state.is_accepting || linked_productive(&template.pop_to_push, i, &push_good)
                || read.is_some_and(|r| read_without_input[r])
            {
                canonical[i] = 0;
                continue;
            }
            let default = state.transitions.get(&DEFAULT_LABEL)
                .map_or(REJECT, |&target| canonical[target as usize]);
            let row = merged_domain_row(&state.transitions, &canonical,
                read.map(|read| read_labels[read].as_slice()), default);
            if default == REJECT && row.is_empty() { continue; }
            // A wide row is expensive to hash. Entry performs the identical
            // exact-key lookup once, rather than hashing a new row again on
            // insertion. IDs still follow the fixed topological traversal;
            // hash-map iteration never determines the compiled representation.
            let next_id = match row_ids.entry((default, row)) {
                std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let id = u32::try_from(states.len()).map_err(|_| "template domain too large")?;
                    if id == REJECT { return Err("template domain too large".to_owned()); }
                    let (default, row) = entry.key();
                    states.push(DomainState { accepts_prefix: false, default_target: *default,
                        first_edge: edges.len(), edge_count: row.len() });
                    edges.extend_from_slice(row);
                    entry.insert(id);
                    id
                }
            };
            canonical[i] = next_id;
        }
        let start = canonical.get(template.pop.start_state as usize).copied().unwrap_or(REJECT);
        Ok(Self::retain_reachable(start, states, edges))
    }

    fn retain_reachable(start: u32, states: Vec<DomainState>, edges: Vec<(u32, u32)>) -> Self {
        if start == REJECT {
            return Self { start: REJECT, states: Box::new([]), edges: Box::new([]) };
        }
        let mut keep = vec![false; states.len()];
        keep[start as usize] = true;
        let mut reachable = 1usize;
        let mut pending = vec![start];
        while let Some(id) = pending.pop() {
            let state = &states[id as usize];
            for target in std::iter::once(state.default_target)
                .chain(edges[state.first_edge..state.first_edge + state.edge_count].iter().map(|edge| edge.1))
            {
                if target == REJECT || keep[target as usize] { continue; }
                // Mark at enqueue time so convergent rows do not repeatedly
                // retain the same target in scratch before it is visited.
                keep[target as usize] = true;
                reachable += 1;
                pending.push(target);
            }
        }
        if reachable == states.len() {
            // With every row reachable, the old stable-ID remap is exactly
            // the identity. Preserve those rows directly instead of copying
            // the full edge inventory into a second allocation.
            return Self { start, states: states.into_boxed_slice(), edges: edges.into_boxed_slice() };
        }
        let mut remap = vec![REJECT; states.len()];
        let mut count = 0u32;
        for (id, &kept) in keep.iter().enumerate() {
            if kept { remap[id] = count; count += 1; }
        }
        let mapped = |id: u32| if id == REJECT { REJECT } else { remap[id as usize] };
        let mut final_states = Vec::with_capacity(count as usize);
        let mut final_edges = Vec::new();
        for (id, state) in states.iter().enumerate() {
            if !keep[id] { continue; }
            let first_edge = final_edges.len();
            final_edges.extend(edges[state.first_edge..state.first_edge + state.edge_count].iter()
                .map(|&(label, target)| (label, mapped(target))));
            final_states.push(DomainState { accepts_prefix: state.accepts_prefix,
                default_target: mapped(state.default_target), first_edge, edge_count: state.edge_count });
        }
        Self { start: mapped(start), states: final_states.into_boxed_slice(), edges: final_edges.into_boxed_slice() }
    }

    /// The input cursor after zero symbols. Accept/Reject are final decisions
    /// even if the unseen stack is nonempty; NeedMore requires a lower symbol.
    #[inline]
    pub fn start(&self) -> DomainProbe { self.at(self.start) }

    #[inline]
    fn at(&self, state: u32) -> DomainProbe {
        match self.states.get(state as usize) {
            None => DomainProbe::Reject,
            Some(state) if state.accepts_prefix => DomainProbe::Accept,
            Some(_) => DomainProbe::NeedMore(state),
        }
    }

    /// Inspect one stack symbol without constructing, cloning, or popping a
    /// parser stack. Caller must keep each cursor paired with this domain.
    #[inline]
    pub fn step(&self, cursor: u32, top: u32) -> DomainProbe {
        let Some(state) = self.states.get(cursor as usize) else { return DomainProbe::Reject; };
        if state.accepts_prefix { return DomainProbe::Accept; }
        let row = &self.edges[state.first_edge..state.first_edge + state.edge_count];
        let target = if row.len() <= 6 {
            row.iter().find(|&&(label, _)| label == top).map(|e| e.1)
        } else {
            row.binary_search_by_key(&top, |e| e.0).ok().map(|i| row[i].1)
        }.unwrap_or(state.default_target);
        self.at(target)
    }

    /// Probe a visible prefix. NeedMore is not rejection until the caller
    /// knows the complete stack is exhausted (a hidden GSS floor is different).
    pub fn probe_top_first(&self, symbols: impl IntoIterator<Item = u32>) -> DomainProbe {
        let mut cursor = self.start();
        for symbol in symbols {
            cursor = match cursor {
                DomainProbe::NeedMore(state) => self.step(state, symbol),
                decision => return decision,
            };
        }
        cursor
    }

    /// Exact feasibility for one *complete* stack, supplied top first.
    pub fn matches_top_first(&self, symbols: impl IntoIterator<Item = u32>) -> bool {
        self.probe_top_first(symbols) == DomainProbe::Accept
    }

    pub fn classify_top(&self, top: u32) -> TopAdmission {
        let probe = match self.start() {
            DomainProbe::NeedMore(state) => self.step(state, top),
            decision => decision,
        };
        match probe {
            DomainProbe::Accept => TopAdmission::Always,
            DomainProbe::Reject => TopAdmission::Never,
            DomainProbe::NeedMore(_) => TopAdmission::DependsOnSuffix,
        }
    }

    /// Partition top-symbol certificates into a default and sparse exceptions.
    ///
    /// This is exactly `classify_top` for every possible top value, including
    /// symbols absent from the root row. In particular, an explicit rejection
    /// must still shadow a productive DEFAULT. The returned certificates say
    /// nothing about the rest of the relation: two different residual domains
    /// can both require a deeper suffix query.
    ///
    /// This lets a caller build all top rows by cloning one default bitset and
    /// visiting root edges, instead of searching this row once per alphabet
    /// symbol. It allocates no output stacks or additional graph data.
    pub fn top_admission_partition(
        &self,
    ) -> (TopAdmission, impl Iterator<Item = (u32, TopAdmission)> + '_) {
        let classify = |probe| match probe {
            DomainProbe::Accept => TopAdmission::Always,
            DomainProbe::Reject => TopAdmission::Never,
            DomainProbe::NeedMore(_) => TopAdmission::DependsOnSuffix,
        };
        let (default, edges): (_, &[(u32, u32)]) = match self.states.get(self.start as usize) {
            None => (TopAdmission::Never, &[]),
            Some(state) if state.accepts_prefix => (TopAdmission::Always, &[]),
            Some(state) => (
                classify(self.at(state.default_target)),
                &self.edges[state.first_edge..state.first_edge + state.edge_count],
            ),
        };
        let exceptions = edges.iter().filter_map(move |&(label, target)| {
            let certificate = classify(self.at(target));
            (certificate != default).then_some((label, certificate))
        });
        (default, exceptions)
    }

    /// Encode the already-compiled domain in a versioned internal cache wire.
    /// This representation contains only input-prefix transitions, never LR
    /// actions/gotos or a PUSH/output program. It is not a stable public format.
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let n = u32::try_from(self.states.len()).map_err(|_| "too many domain states")?;
        let m = u32::try_from(self.edges.len()).map_err(|_| "too many domain edges")?;
        let size = 16usize.checked_add(self.states.len().checked_mul(16).ok_or("domain size overflow")?)
            .and_then(|n| self.edges.len().checked_mul(8).and_then(|m| n.checked_add(m)))
            .ok_or("domain size overflow")?;
        let mut wire = Vec::with_capacity(size);
        wire.extend_from_slice(b"GTD1");
        for value in [self.start, n, m] { wire.extend_from_slice(&value.to_le_bytes()); }
        for state in &self.states {
            for value in [u32::from(state.accepts_prefix), state.default_target,
                u32::try_from(state.first_edge).map_err(|_| "domain edge offset overflow")?,
                u32::try_from(state.edge_count).map_err(|_| "domain edge count overflow")?]
            { wire.extend_from_slice(&value.to_le_bytes()); }
        }
        for &(label, target) in &self.edges {
            wire.extend_from_slice(&label.to_le_bytes()); wire.extend_from_slice(&target.to_le_bytes());
        }
        debug_assert_eq!(wire.len(), size);
        Ok(wire)
    }

    /// Decode and validate a compiled domain, without reconstructing templates
    /// or consulting a parser table. The backward-edge invariant proves the
    /// entire input graph acyclic in one pass; bad data cannot introduce a
    /// nonterminating query. Counts are checked against input before allocation.
    pub fn from_bytes(wire: &[u8]) -> Result<Self, String> {
        if wire.len() < 16 || !wire.starts_with(b"GTD1") {
            return Err("invalid template-domain header".to_owned());
        }
        let read = |offset: usize| u32::from_le_bytes(wire[offset..offset + 4].try_into().unwrap());
        let start = read(4); let n = read(8) as usize; let m = read(12) as usize;
        let edge_base = n.checked_mul(16).and_then(|n| n.checked_add(16)).ok_or("domain size overflow")?;
        let required = m.checked_mul(8).and_then(|m| m.checked_add(edge_base)).ok_or("domain size overflow")?;
        if required != wire.len() || n >= REJECT as usize {
            return Err("template-domain size/count mismatch".to_owned());
        }
        if (n == 0 && (start != REJECT || m != 0)) || (n != 0 && start as usize >= n) {
            return Err("invalid template-domain start".to_owned());
        }
        let mut states = Vec::with_capacity(n);
        let mut edges = Vec::with_capacity(m);
        let mut consumed = 0usize;
        for id in 0..n {
            let offset = 16 + id * 16;
            let flags = read(offset); let default = read(offset + 4);
            let first = read(offset + 8) as usize; let count = read(offset + 12) as usize;
            let end = first.checked_add(count).ok_or("domain edge interval overflow")?;
            if flags > 1 || first != consumed || end > m {
                return Err("invalid template-domain row".to_owned());
            }
            if default != REJECT && default as usize >= id {
                return Err("template-domain DEFAULT is not acyclic".to_owned());
            }
            if flags == 1 && (default != REJECT || count != 0) {
                return Err("accepting template-domain row has outgoing edges".to_owned());
            }
            let mut last = None;
            let mut productive = flags == 1 || default != REJECT;
            for edge in first..end {
                let label = read(edge_base + edge * 8); let target = read(edge_base + edge * 8 + 4);
                if label > i32::MAX as u32 || label == DEFAULT_LABEL as u32
                    || last.is_some_and(|last| last >= label)
                {
                    return Err("invalid or unordered template-domain symbol".to_owned());
                }
                if target != REJECT && target as usize >= id {
                    return Err("template-domain edge is not acyclic".to_owned());
                }
                productive |= target != REJECT;
                last = Some(label); edges.push((label, target));
            }
            if !productive { return Err("unproductive template-domain row".to_owned()); }
            states.push(DomainState { accepts_prefix: flags == 1, default_target: default,
                first_edge: first, edge_count: count });
            consumed = end;
        }
        if consumed != m { return Err("unreferenced template-domain edges".to_owned()); }
        Ok(Self { start, states: states.into_boxed_slice(), edges: edges.into_boxed_slice() })
    }

    pub fn state_count(&self) -> usize { self.states.len() }
    pub fn edge_count(&self) -> usize { self.edges.len() }

    /// Compile this existing read-only admission DAG into a scoped boundary
    /// predicate. DEFAULT stays local at every consumed symbol; explicit dead
    /// exceptions are interned before any subset union. Output phases are
    /// already existentially eliminated by domain preparation.
    pub fn scoped_prefix_program(&self,offset:u32,symbols:u32,guard_owner:bool,
        classes:&mut crate::pop_classes::PopLabelClasses,
    )->Result<crate::automata::weighted::nwa::NWA,String> {
        use crate::automata::weighted::nwa::{NWA,NWAState};
        use crate::ds::weight::Weight;
        let end=offset.checked_add(symbols).filter(|&end|end<=classes.symbol_count())
            .ok_or("admission program component range overflow")?;
        let dead=u32::try_from(self.states.len()).map_err(|_|"admission graph too large")?;
        let target=|q:u32| if q==REJECT {dead} else {q};
        let mut states=vec![NWAState::default();self.states.len()+1];
        for (q,row) in self.states.iter().enumerate() {
            if row.accepts_prefix {states[q].final_weight=Some(Weight::all());continue;}
            let edges=&self.edges[row.first_edge..row.first_edge+row.edge_count];
            for &(label,next) in edges {
                if label>=symbols {return Err("admission literal outside its component".into());}
                states[q].transitions.insert((offset+label) as i32,vec![(target(next),Weight::all())]);
            }
            if row.default_target!=REJECT {
                if let Some(label)=classes.intern_scoped_complement(offset..end,
                    edges.iter().map(|&(label,_)|offset+label))? {
                    states[q].transitions.insert(label,vec![(target(row.default_target),Weight::all())]);
                }
            }
        }
        let mut graph=NWA::from_parts(states,vec![target(self.start)]);
        if guard_owner && self.start()==DomainProbe::Accept {
            let start=graph.add_state();
            if let Some(label)=classes.intern_scoped_complement(offset..end,[])? {
                graph.add_transition(start,label,target(self.start),Weight::all());
            }
            graph.set_start_states(vec![start]);
        }
        Ok(graph)
    }
    /// Native heap payload, not serialized wire size or peak allocation.
    pub fn heap_payload_bytes(&self) -> usize {
        std::mem::size_of_val(self.states.as_ref()) + std::mem::size_of_val(self.edges.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::labels::encode_negative_label;

    fn template() -> CommitTemplateDfas {
        CommitTemplateDfas { pop: DFA::new(), read: DFA::new(), push: DFA::new(),
            ..CommitTemplateDfas::default() }
    }

    #[test]
    fn domain_reachability_identity_preserves_exact_rows_and_edges() {
        let states = vec![
            DomainState { accepts_prefix: true, default_target: REJECT, first_edge: 0, edge_count: 0 },
            DomainState { accepts_prefix: false, default_target: 0, first_edge: 0, edge_count: 1 },
            DomainState { accepts_prefix: false, default_target: 1, first_edge: 1, edge_count: 2 },
        ];
        // Explicit rejection at row1 must continue to shadow its productive
        // DEFAULT; row2 has two paths to a shared suffix.
        let edges = vec![(7, REJECT), (3, 0), (5, 1)];
        let reference = TemplateDomain { start: 2, states: states.clone().into_boxed_slice(),
            edges: edges.clone().into_boxed_slice() };
        let actual = TemplateDomain::retain_reachable(2, states, edges);
        assert_eq!(actual.to_bytes().unwrap(), reference.to_bytes().unwrap());
        for word in [vec![], vec![3], vec![5], vec![5, 7], vec![5, 8], vec![1, 8]] {
            assert_eq!(actual.matches_top_first(word.iter().copied()),
                reference.matches_top_first(word.iter().copied()));
        }
    }

    #[test]
    fn domain_reachability_prunes_without_reordering_kept_ids() {
        let states = vec![
            DomainState { accepts_prefix: true, default_target: REJECT, first_edge: 0, edge_count: 0 },
            DomainState { accepts_prefix: false, default_target: REJECT, first_edge: 0, edge_count: 1 },
            DomainState { accepts_prefix: false, default_target: REJECT, first_edge: 1, edge_count: 1 },
            DomainState { accepts_prefix: false, default_target: 2, first_edge: 2, edge_count: 1 },
        ];
        let edges = vec![(1, 0), (2, 0), (3, 2)];
        let actual = TemplateDomain::retain_reachable(3, states.clone(), edges.clone());
        let expected = TemplateDomain { start: 2, states: vec![
            states[0].clone(),
            DomainState { accepts_prefix: false, default_target: REJECT, first_edge: 0, edge_count: 1 },
            DomainState { accepts_prefix: false, default_target: 1, first_edge: 1, edge_count: 1 },
        ].into_boxed_slice(), edges: vec![(2, 0), (3, 1)].into_boxed_slice() };
        assert_eq!(actual.to_bytes().unwrap(), expected.to_bytes().unwrap());
        let dead = TemplateDomain::retain_reachable(REJECT, states, edges);
        assert_eq!(dead.state_count(), 0);
        assert_eq!(dead.edge_count(), 0);
        assert_eq!(dead.start(), DomainProbe::Reject);
        assert_eq!(dead.to_bytes().unwrap().len(), 16);
    }

    #[test]
    fn linear_root_merge_matches_ordered_map_reference() {
        let mut seed = 0x6088941134u64;
        let mut next = || {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            seed
        };
        for _ in 0..2048 {
            let canonical = (0..12).map(|_| match next() % 5 {
                0 => REJECT, value => value as u32 - 1,
            }).collect::<Vec<_>>();
            let mut transitions = BTreeMap::new();
            let mut reads = BTreeSet::new();
            for label in 0..24 {
                if next() % 3 == 0 { transitions.insert(label, (next() % 12) as u32); }
                if next() % 4 == 0 { reads.insert(label as u32); }
            }
            transitions.insert(DEFAULT_LABEL, (next() % 12) as u32);
            for default in [REJECT, 0, 1, 2, 3] {
                let read_values = reads.iter().copied().collect::<Vec<_>>();
                for read in [None, Some(read_values.as_slice())] {
                    let actual = merged_domain_row(&transitions, &canonical, read, default);
                    let mut reference: BTreeMap<u32, u32> = transitions.iter()
                        .filter(|(label, _)| **label != DEFAULT_LABEL)
                        .map(|(&label, &target)| (label as u32, canonical[target as usize])).collect();
                    for &label in read.into_iter().flatten() { reference.insert(label, 0); }
                    reference.retain(|_, target| *target != default);
                    assert_eq!(actual, reference.into_iter().collect::<Vec<_>>());
                    assert!(actual.windows(2).all(|pair| pair[0].0 < pair[1].0));
                }
            }
        }
    }

    // Independent literal split-transducer interpreter. It materializes output
    // stacks only in tests, and uses neither domain construction nor shortcuts.
    fn output_exists(t: &CommitTemplateDfas, top_first: &[u32]) -> bool {
        let mut work = vec![(0u8, t.pop.start_state, top_first.iter().rev().copied().collect::<Vec<_>>())];
        let mut seen = BTreeSet::new();
        while let Some((phase, id, stack)) = work.pop() {
            if !seen.insert((phase, id, stack.clone())) { continue; }
            let dfa = [&t.pop, &t.read, &t.push][phase as usize];
            let Some(state) = dfa.states.get(id as usize) else { continue; };
            if state.is_accepting { return true; }
            match phase {
                0 => {
                    if let Some(&top) = stack.last() {
                        if let Some(&target) = state.transitions.get(&(top as i32)).or_else(|| state.transitions.get(&DEFAULT_LABEL)) {
                            let mut next = stack.clone(); next.pop(); work.push((0, target, next));
                        }
                    }
                    if let Some(Some(target)) = t.pop_to_read.get(id as usize) { work.push((1, *target, stack.clone())); }
                    if let Some(Some(target)) = t.pop_to_push.get(id as usize) { work.push((2, *target, stack)); }
                }
                1 => {
                    if let Some(&top) = stack.last() && let Some(&target) = state.transitions.get(&(top as i32)) {
                        work.push((1, target, stack.clone()));
                    }
                    if let Some(Some(target)) = t.read_to_push.get(id as usize) { work.push((2, *target, stack)); }
                }
                2 => for (&label, &target) in &state.transitions {
                    let mut next = stack.clone(); next.push(label.wrapping_sub(i32::MIN) as u32);
                    work.push((2, target, next));
                },
                _ => unreachable!(),
            }
        }
        false
    }

    #[test]
    fn domain_retains_explicit_rejection_over_productive_default() {
        let mut t = template();
        let dead = t.pop.add_state();
        let yes = t.pop.add_state(); t.pop.set_accepting(yes, true);
        t.pop.add_transition(0, DEFAULT_LABEL, yes); t.pop.add_transition(0, 5, dead);
        let domain = TemplateDomain::compile(&t).unwrap();
        assert!(!domain.matches_top_first([]));
        assert!(!domain.matches_top_first([5]));
        assert!(!domain.matches_top_first([5, 6]));
        assert!(domain.matches_top_first([6]));
        assert_eq!(domain.classify_top(5), TopAdmission::Never);
        assert_eq!(domain.classify_top(6), TopAdmission::Always);
        assert_eq!(domain.edge_count(), 1); // The rejecting exception is essential.
        let (default, exceptions) = domain.top_admission_partition();
        assert_eq!(default, TopAdmission::Always);
        assert_eq!(exceptions.collect::<Vec<_>>(), vec![(5, TopAdmission::Never)]);
    }

    #[test]
    fn top_partition_retains_suffix_dependence_and_needs_no_alphabet_bound() {
        let mut t = template();
        let more = t.pop.add_state();
        let yes = t.pop.add_state();
        let no = t.pop.add_state();
        t.pop.set_accepting(yes, true);
        t.pop.add_transition(0, DEFAULT_LABEL, more);
        t.pop.add_transition(0, 2, yes);
        t.pop.add_transition(0, 5, no);
        t.pop.add_transition(more, 7, yes);
        let domain = TemplateDomain::compile(&t).unwrap();
        let (default, exceptions) = domain.top_admission_partition();
        assert_eq!(default, TopAdmission::DependsOnSuffix);
        let exceptions: BTreeMap<_, _> = exceptions.collect();
        assert_eq!(exceptions, BTreeMap::from([
            (2, TopAdmission::Always), (5, TopAdmission::Never),
        ]));
        for top in [0, 1, 2, 5, 7, 999, i32::MAX as u32, u32::MAX] {
            assert_eq!(exceptions.get(&top).copied().unwrap_or(default), domain.classify_top(top));
        }
        for accepted in [false, true] {
            let mut t = template();
            t.pop.set_accepting(0, accepted);
            let domain = TemplateDomain::compile(&t).unwrap();
            let (default, exceptions) = domain.top_admission_partition();
            assert_eq!(default, if accepted { TopAdmission::Always } else { TopAdmission::Never });
            assert_eq!(exceptions.count(), 0);
        }
    }

    #[test]
    fn domain_read_chains_inspect_same_top_not_successive_symbols() {
        let mut t = template();
        let r1 = t.read.add_state(); let r2 = t.read.add_state();
        t.read.set_accepting(r2, true); t.pop_to_read = vec![Some(0)];
        t.read.add_transition(0, 7, r1); t.read.add_transition(r1, 8, r2);
        let domain = TemplateDomain::compile(&t).unwrap();
        assert!(!domain.matches_top_first([7, 8]));
        assert_eq!(domain.start(), DomainProbe::Reject);
        t.read.states[r1 as usize].transitions.clear(); t.read.add_transition(r1, 7, r2);
        let domain = TemplateDomain::compile(&t).unwrap();
        assert!(!domain.matches_top_first([]));
        assert!(domain.matches_top_first([7]));
        assert!(domain.matches_top_first([7, 123]));
        assert!(!domain.matches_top_first([8, 7]));
    }

    #[test]
    fn domain_short_circuits_push_dag_and_exposes_top_certificates() {
        let mut t = template();
        let first = t.pop.add_state(); let second = t.pop.add_state();
        t.pop.add_transition(0, 1, first); t.pop.add_transition(first, 2, second);
        t.pop_to_push = vec![None, None, Some(0)];
        let mut last = 0;
        for _ in 0..30 { // More than a billion distinct accepted PUSH words.
            let next = t.push.add_state();
            for value in [10, 20] { t.push.add_transition(last, encode_negative_label(value), next); }
            last = next;
        }
        t.push.set_accepting(last, true);
        let domain = TemplateDomain::compile(&t).unwrap();
        assert_eq!(domain.state_count(), 3);
        assert_eq!(domain.edge_count(), 2);
        assert_eq!(domain.classify_top(1), TopAdmission::DependsOnSuffix);
        assert_eq!(domain.classify_top(2), TopAdmission::Never);
        assert!(!domain.matches_top_first([1]));
        assert!(domain.matches_top_first([1, 2]));
        assert!(domain.matches_top_first([1, 2, 123]));
    }

    #[test]
    fn domain_validation_rejects_cycles_bad_links_and_wrong_phase_labels() {
        let mut t = template(); t.pop.add_transition(0, DEFAULT_LABEL, 0);
        assert!(TemplateDomain::compile(&t).unwrap_err().contains("cyclic"));
        let mut t = template(); t.pop_to_push = vec![Some(100)];
        assert!(TemplateDomain::compile(&t).unwrap_err().contains("missing state"));
        let mut t = template(); t.pop.add_transition(0, 1, 100);
        assert!(TemplateDomain::compile(&t).unwrap_err().contains("missing state"));
        let mut t = template(); t.read.add_transition(0, DEFAULT_LABEL, 0);
        assert!(TemplateDomain::compile(&t).unwrap_err().contains("wrong-phase"));
        let mut t = template(); t.push.add_transition(0, 2, 0);
        assert!(TemplateDomain::compile(&t).unwrap_err().contains("wrong-phase"));
        let mut t = template(); t.pop.start_state = 1;
        assert!(TemplateDomain::compile(&t).unwrap_err().contains("start"));
    }

    fn stacks(max_depth: usize) -> Vec<Vec<u32>> {
        let mut all = vec![Vec::new()];
        let mut level = vec![Vec::new()];
        for _ in 0..max_depth {
            let mut next = Vec::new();
            for stack in level {
                for top in 0..4 { let mut s = stack.clone(); s.push(top); next.push(s); }
            }
            all.extend(next.iter().cloned()); level = next;
        }
        all
    }

    #[test]
    fn domain_matches_literal_executor_for_generated_transducers_and_stacks() {
        let stacks = stacks(4);
        for seed in 1..=128u64 {
            let mut random = seed;
            let mut next = || {
                random = random.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                random
            };
            let mut t = template();
            for dfa in [&mut t.pop, &mut t.read, &mut t.push] {
                for _ in 1..5 { dfa.add_state(); }
            }
            for (phase, dfa) in [&mut t.pop, &mut t.read, &mut t.push].into_iter().enumerate() {
                let alphabet: &[i32] = match phase {
                    0 => &[0, 1, 2, DEFAULT_LABEL], 1 => &[0, 1, 2],
                    _ => &[encode_negative_label(0), encode_negative_label(2)],
                };
                for source in 0..5u32 {
                    dfa.set_accepting(source, next() >> 62 == 0);
                    if source == 4 { continue; }
                    for &label in alphabet {
                        if next() >> 62 != 0 {
                            let target = source + 1 + (next() % (4 - source) as u64) as u32;
                            dfa.add_transition(source, label, target);
                        }
                    }
                }
            }
            for links in [&mut t.pop_to_read, &mut t.pop_to_push, &mut t.read_to_push] {
                *links = (0..5).map(|_| if next() >> 62 == 0 { None } else { Some((next() % 5) as u32) }).collect();
            }
            let compiled = TemplateDomain::compile(&t).unwrap();
            let wire = compiled.to_bytes().unwrap();
            let domain = TemplateDomain::from_bytes(&wire).unwrap();
            assert_eq!(domain.to_bytes().unwrap(), wire, "wire roundtrip seed={seed}");
            // Partition building and classify_top use different paths through
            // the root. Compare them independently for each generated graph,
            // both before and after the existing domain-wire roundtrip.
            for candidate in [&compiled, &domain] {
                let (default, exceptions) = candidate.top_admission_partition();
                let exceptions: BTreeMap<_, _> = exceptions.collect();
                for top in (0..9).chain([1234, i32::MAX as u32, u32::MAX]) {
                    assert_eq!(exceptions.get(&top).copied().unwrap_or(default),
                        candidate.classify_top(top), "seed={seed} top={top}");
                }
            }
            for stack in &stacks {
                assert_eq!(domain.matches_top_first(stack.iter().copied()), output_exists(&t, stack),
                    "seed={seed} top_first={stack:?}");
            }
        }
    }

    #[test]
    fn domain_handles_deep_acyclic_templates_iteratively() {
        let mut t = template(); let mut last = 0;
        for _ in 0..15_000 { let next = t.pop.add_state(); t.pop.add_transition(last, DEFAULT_LABEL, next); last = next; }
        t.pop.set_accepting(last, true);
        let domain = TemplateDomain::compile(&t).unwrap();
        assert!(!domain.matches_top_first(std::iter::repeat_n(0, 14_999)));
        assert!(domain.matches_top_first(std::iter::repeat_n(0, 15_000)));
        assert_eq!(TemplateDomain::compile(&CommitTemplateDfas::default()).unwrap().start(), DomainProbe::Reject);
    }
    #[test]
    fn domain_wire_rejects_truncation_invalid_indices_and_cycles() {
        let mut t = template(); let accepted = t.pop.add_state();
        t.pop.set_accepting(accepted, true); t.pop.add_transition(0, 7, accepted);
        let domain = TemplateDomain::compile(&t).unwrap(); let wire = domain.to_bytes().unwrap();
        assert_eq!(wire.len(), 16 + 2 * 16 + 8);
        for length in 0..wire.len() { assert!(TemplateDomain::from_bytes(&wire[..length]).is_err()); }
        let mut bad = wire.clone(); bad[0] ^= 1;
        assert!(TemplateDomain::from_bytes(&bad).is_err());
        let mut bad = wire.clone(); bad[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(TemplateDomain::from_bytes(&bad).is_err());
        let mut bad = wire.clone(); let root = domain.start;
        // The final explicit edge would cycle to its own root.
        let offset = bad.len() - 4; bad[offset..].copy_from_slice(&root.to_le_bytes());
        assert!(TemplateDomain::from_bytes(&bad).unwrap_err().contains("acyclic"));
        let mut bad = wire.clone(); bad[4..8].copy_from_slice(&999u32.to_le_bytes());
        assert!(TemplateDomain::from_bytes(&bad).is_err());
        let mut bad = wire.clone(); bad[16..20].copy_from_slice(&2u32.to_le_bytes());
        assert!(TemplateDomain::from_bytes(&bad).is_err());
        let empty = TemplateDomain::compile(&CommitTemplateDfas::default()).unwrap();
        let wire = empty.to_bytes().unwrap();
        assert_eq!(wire.len(), 16);
        assert_eq!(TemplateDomain::from_bytes(&wire).unwrap().start(), DomainProbe::Reject);
    }

}
