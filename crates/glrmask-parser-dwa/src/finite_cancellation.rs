//! Source-independent, exact cancellation summaries on a weighted DAG.
//!
//! R(q,a) is the weighted relation to the position immediately after one
//! matching read a (or DEFAULT), allowing balanced push/read excursions and
//! epsilon moves before it. Let E(q) contain the original epsilons and the
//! derived edges for pushes starting at q. Then
//!
//! E(q) = eps(q) + sum(push_b(q,r,w)) w * R(r,b)
//! R(q,a) = read_a_or_default(q) + sum((r,w) in E(q)) w * R(r,a).
//!
//! Every dependency goes forward in the original DAG. Reverse topological
//! induction proves both equations; union/intersection distribute, so the
//! relation can be reused independently of the originating push's weight.
//! No state identity, original edge, guard key, or normalization rule changes.
use super::*;
use super::finite_signed_graph::{SignedGraph,SignedRow};
#[path = "finite_cancellation_results.rs"]
mod result_storage;
use result_storage::{ResultRows, VectorRows, PackedRows};

const MAX_QUERY_PAIRS: usize = 400_000;
const MAX_RESULT_PAIRS: usize = 4_000_000;
const MAX_SUMMARY_WORK: usize = 20_000_000;

#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct SummaryStats {
    pub queries: usize,
    pub result_pairs: usize,
    pub work: usize,
    pub cache_hits: usize,
    pub max_stack: usize,
    pub rejected_queries: usize,
    pub rejected_edges: usize,
}

struct Frame {
    state: u32,
    edge: usize,
    result: FastBoundaryDerivedRow,
}

struct Solver<'a,G:SignedGraph+?Sized,R:ResultRows> {
    states: &'a G,
    interner: &'a mut FastBoundaryWeightInterner,
    effective: Vec<Vec<(u32, FastBoundaryWeightId)>>,
    known: FxHashMap<(u32, i32), R::Handle>,
    results: R,
    stack: Vec<Frame>,
    inflight_pairs: usize,
    stats: SummaryStats,
    may_read: Vec<u128>,
    filter: bool,
}

/// Two independent word positions form a conservative 128-bit read synopsis.
/// A collision can only fail to reject an impossible read, never invent a proof.
fn read_signature(label: i32) -> u128 {
    let x=(label as u32 as u64).wrapping_mul(0x9e3779b97f4a7c15);
    (1u128 << (x >> 58)) | (1u128 << (64 + ((x ^ (x >> 23)).wrapping_mul(0xd6e8feb86659fd93) >> 58)))
}

impl<G:SignedGraph+?Sized,R:ResultRows> Solver<'_,G,R> {
    fn account(&mut self, work: usize) -> Option<()> {
        self.stats.work = self.stats.work.checked_add(work)?;
        if self.stats.work > MAX_SUMMARY_WORK
            || !self.interner.allow_work(work, self.states.len(), 0) { return None; }
        Some(())
    }

    fn frame(&mut self, state: u32, label: i32) -> Option<Frame> {
        let graph=self.states;
        let row=graph.row(state as usize);
        self.account(2 * (row.transition_count() + 1).ilog2() as usize + 1)?;
        let mut result = FastBoundaryDerivedRow::default();
        // Ordinary callers keep additive literal+DEFAULT cancellation. Native
        // classes are already row-local complements, so test membership here;
        // no concrete alphabet expansion or wildcard reinterpretation occurs.
        self.account(graph.class_read_labels().len())?;
        let selected = [label,DEFAULT_LABEL].into_iter().chain(graph.class_read_labels().iter().copied()
            .filter(|&key|graph.class_read_matches(key,label)));
        for (position,key) in selected.into_iter().enumerate() {
            if position==1 && label==DEFAULT_LABEL{continue;}
            let Some(index)=row.find_label(key)
                else{continue;};
            self.account(row.branch_count(index))?;
            for branch in 0..row.branch_count(index) {
                let(target,weight)=row.branch(index,branch);
                if target as usize >= self.states.len() { return None; }
                result.merge(target, weight, self.interner);
            }
        }
        self.inflight_pairs = self.inflight_pairs.checked_add(result.len())?;
        if self.inflight_pairs.checked_add(self.stats.result_pairs)? > MAX_RESULT_PAIRS { return None; }
        Some(Frame { state, edge: 0, result })
    }

    fn query(&mut self, state: u32, label: i32) -> Option<R::Handle> {
        let wanted=read_signature(label);
        if self.filter {
            self.account(1)?;
            if self.may_read[state as usize] & wanted != wanted {
                self.stats.rejected_queries+=1;return Some(R::EMPTY);
            }
        }
        if let Some(&id) = self.known.get(&(state, label)) {
            self.stats.cache_hits += 1;
            return Some(id);
        }
        debug_assert!(self.stack.is_empty());
        let frame = self.frame(state, label)?;
        self.stack.push(frame);
        while !self.stack.is_empty() {
            let last = self.stack.len() - 1;
            let q = self.stack[last].state;
            let edge = self.stack[last].edge;
            if let Some(&(target, weight)) = self.effective[q as usize].get(edge) {
                if self.filter {
                    self.account(1)?;
                    if self.may_read[target as usize] & wanted != wanted {
                        self.stats.rejected_edges+=1;
                        self.stack[last].edge+=1;continue;
                    }
                }
                if let Some(&id) = self.known.get(&(target, label)) {
                    self.account(1 + self.results.row_len(id))?;
                    let previous = self.stack[last].result.len();
                    for i in 0..self.results.row_len(id) {
                        let (end, suffix) = self.results.pair(id, i);
                        let add = self.interner.intersection(weight, suffix);
                        self.stack[last].result.merge(end, add, self.interner);
                    }
                    self.inflight_pairs = self.inflight_pairs.checked_add(self.stack[last].result.len() - previous)?;
                    if self.inflight_pairs.checked_add(self.stats.result_pairs)? > MAX_RESULT_PAIRS { return None; }
                    self.stack[last].edge += 1;
                    self.stats.cache_hits += 1;
                } else {
                    if self.stack.len() >= self.states.len() { return None; }
                    let frame = self.frame(target, label)?;
                    self.stack.push(frame);
                    self.stats.max_stack = self.stats.max_stack.max(self.stack.len());
                }
            } else {
                let frame = self.stack.pop()?;
                let row_len = frame.result.len();
                self.inflight_pairs -= row_len;
                self.stats.result_pairs = self.stats.result_pairs.checked_add(row_len)?;
                if self.stats.result_pairs > MAX_RESULT_PAIRS || self.known.len() >= MAX_QUERY_PAIRS {
                    return None;
                }
                let id = self.results.push(frame.result)?;
                self.known.insert((q, label), id);
                self.stats.queries += 1;
            }
        }
        self.known.get(&(state, label)).copied()
    }
}

#[cfg(test)]
pub(super) fn compute(
    states: &[FastBoundaryNwaState], interner: &mut FastBoundaryWeightInterner,
) -> Option<(Vec<FastBoundaryDerivedRow>, SummaryStats)> {
    compute_with_topology(states, interner, None)
}

// The resolver has already checked every original edge against `topology`.
// This module consumes the unchanged graph until it returns derived rows.
pub(super) fn compute_with_topology(
    states: &[FastBoundaryNwaState], interner: &mut FastBoundaryWeightInterner,
    topology: Option<&CheckedNativeTopology>,
) -> Option<(Vec<FastBoundaryDerivedRow>, SummaryStats)> {
    compute_with_topology_mode(states,interner,topology,
        crate::optimized_env_flag("GLRMASK_BOUNDARY_CANCELLATION_READ_FILTER"))
}

fn compute_with_topology_mode(
    states:&[FastBoundaryNwaState],interner:&mut FastBoundaryWeightInterner,
    topology:Option<&CheckedNativeTopology>,filter:bool,
)->Option<(Vec<FastBoundaryDerivedRow>,SummaryStats)> {
    let owned_topo;
    let topo = if let Some(topology) = topology { topology.order.as_slice() } else {
        owned_topo = fast_boundary_topological_order(states)?;
        owned_topo.as_slice()
    };
    compute_on_graph(states,interner,topo,filter)
}

/// The graph constructor/resolver checks bounds and certifies this order.
/// Changing adjacency storage does not change the memo recurrence or IDs.
pub(super) fn compute_on_graph<G:SignedGraph+?Sized>(
    states:&G,interner:&mut FastBoundaryWeightInterner,topo:&[u32],filter:bool,
)->Option<(Vec<FastBoundaryDerivedRow>,SummaryStats)> {
    let policy = if crate::optimized_env_flag("GLRMASK_BOUNDARY_PACKED_CANCELLATION_RESULTS") {
        ResultPolicy::Packed
    } else { ResultPolicy::Vector };
    compute_on_graph_policy(states,interner,topo,filter,policy)
}

#[derive(Clone,Copy,Debug)]
enum ResultPolicy { Vector, Packed }

fn compute_on_graph_policy<G:SignedGraph+?Sized>(
    states:&G,interner:&mut FastBoundaryWeightInterner,topo:&[u32],filter:bool,
    policy:ResultPolicy,
)->Option<(Vec<FastBoundaryDerivedRow>,SummaryStats)> {
    match policy {
        ResultPolicy::Vector => compute_with_rows::<G,VectorRows>(states,interner,topo,filter),
        ResultPolicy::Packed => compute_with_rows::<G,PackedRows>(states,interner,topo,filter),
    }
}

fn compute_with_rows<G:SignedGraph+?Sized,R:ResultRows>(
    states:&G,interner:&mut FastBoundaryWeightInterner,topo:&[u32],filter:bool,
)->Option<(Vec<FastBoundaryDerivedRow>,SummaryStats)> {
    for q in 0..states.len() {
        let row=states.row(q);
        if (1..row.transition_count()).any(|i|row.label(i-1)>=row.label(i)){return None;}
    }
    let n = states.len();
    if topo.len() != n { return None; }
    let mut solver = Solver::<G,R> { states, interner, effective: vec![Vec::new(); n],
        known: FxHashMap::default(), results: R::new(), stack: Vec::new(), inflight_pairs: 0, stats: Default::default(),
        may_read:if filter{vec![u128::MAX;n]}else{Vec::new()},filter };
    let mut derived = vec![FastBoundaryDerivedRow::default(); n];
    let mut retained_edges = 0usize;
    for &q in topo.iter().rev() {
        let source=states.row(q as usize);
        for row in 0..source.transition_count() {
            let label = source.label(row);
            if !is_negative_label(label) { continue; }
            for branch in 0..source.branch_count(row) {
                let (target, weight) = source.branch(row,branch);
                if target as usize >= n { return None; }
                if weight == 0 { continue; }
                let result = solver.query(target, negative_to_positive_label(label))?;
                solver.account(solver.results.row_len(result))?;
                for i in 0..solver.results.row_len(result) {
                    let (end, suffix) = solver.results.pair(result, i);
                    let add = solver.interner.intersection(weight, suffix);
                    derived[q as usize].merge(end, add, solver.interner);
                }
            }
        }
        // Combining equal epsilon targets is safe inside this memo relation:
        // the original graph remains untouched and later gets only derived
        // epsilons, exactly as in the independent worklist implementation.
        let mut epsilon = FastBoundaryDerivedRow::default();
        for edge in 0..source.epsilon_count() {
            let(target,weight)=source.epsilon(edge);
            if target as usize >= n { return None; }
            epsilon.merge(target, weight, solver.interner);
        }
        derived[q as usize].for_each(|target, weight| { epsilon.merge(target, weight, solver.interner); });
        let mut row = epsilon.into_entries();
        row.sort_unstable_by_key(|&(target, _)| target);
        retained_edges = retained_edges.checked_add(row.len())?;
        if retained_edges > MAX_RESULT_PAIRS { return None; }
        if filter {
            solver.account(source.transition_count()+row.len()+1)?;
            let mut may=0u128;
            for index in 0..source.transition_count() {
                let label=source.label(index);
                if label==DEFAULT_LABEL || states.class_read_labels().binary_search(&label).is_ok() {
                    // A conservative synopsis cannot reject a PUSH merely
                    // because its matching POP uses an opaque class identity.
                    may=u128::MAX;break;
                }
                if !is_negative_label(label) {may|=read_signature(label);}
            }
            for &(target,_) in &row {may|=solver.may_read[target as usize];}
            solver.may_read[q as usize]=may;
        }
        solver.effective[q as usize] = row;
    }
    solver.account(0)?;
    if compile_profile_enabled() {
        let mut empty=0;
        let mut single=0;
        for &handle in solver.known.values() {
            match solver.results.row_len(handle) {0=>empty+=1,1=>single+=1,_=>{}}
        }
        eprintln!("[glrmask/profile][cancellation_packed_results] representation={} queries={} empty={} single={} storage_bytes={}",
            std::any::type_name::<R>(),solver.known.len(),empty,single,solver.results.storage_bytes());
    }
    Some((derived, solver.stats))
}

#[cfg(test)]
#[path = "finite_cancellation_results_tests.rs"]
mod result_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn canonical(row: &FastBoundaryDerivedRow) -> Vec<(u32, FastBoundaryWeightId)> {
        let mut result = Vec::new(); row.for_each(|q,w|result.push((q,w)));
        result.sort_unstable_by_key(|&(q,_)|q); result
    }

    /// A separate literal stack interpreter, not the worklist or summary
    /// recurrence: follow each signed path until its first return to empty.
    fn literal_paths(states: &[FastBoundaryNwaState], interner: &mut FastBoundaryWeightInterner)
        -> Vec<FastBoundaryDerivedRow>
    {
        let mut result = vec![FastBoundaryDerivedRow::default(); states.len()];
        for source in 0..states.len() {
            let mut todo = Vec::<(u32,Vec<i32>,u32)>::new();
            for (label, branches) in &states[source].transitions {
                if !is_negative_label(*label) { continue; }
                for &(target, weight) in branches {
                    if weight != 0 { todo.push((target,vec![negative_to_positive_label(*label)],weight)); }
                }
            }
            while let Some((q, stack, weight)) = todo.pop() {
                for &(target, edge) in &states[q as usize].epsilons {
                    let w=interner.intersection(weight,edge);
                    if w!=0 {todo.push((target,stack.clone(),w));}
                }
                for (label,branches) in &states[q as usize].transitions {
                    let mut next=stack.clone();
                    if is_negative_label(*label){next.push(negative_to_positive_label(*label));}
                    else if *label==DEFAULT_LABEL || next.last()==Some(label){next.pop();}
                    else{continue;}
                    for &(target,edge) in branches {
                        let w=interner.intersection(weight,edge);
                        if w==0{continue;}
                        if next.is_empty(){result[source].merge(target,w,interner);}
                        else{todo.push((target,next.clone(),w));}
                    }
                }
            }
        }
        result
    }

    #[test]
    fn cancellation_summary_matches_fifo_and_literal_signed_paths() {
        let mut seed=59119u64;
        let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32) as usize};
        let mut pairs=0usize;
        let mut rejected=0usize;
        for case in 0..1024 {
            let mut interner=FastBoundaryWeightInterner::new([1,2,17,64][case%4],64).unwrap();
            interner.limits=Some(Default::default());
            let mut source_weights=FxHashMap::default();
            let mut weights=vec![0,1];
            for pattern in [1u32,2,3,5,7] {
                let w=Weight::from_per_tsid_token_sets((0..interner.tsid_count as u32).map(|row|{
                    (row,(0..8u32).filter(|&bit|pattern.rotate_left(row%4)&(1<<bit)!=0).map(|bit|bit*8+7).collect())
                }));
                weights.push(interner.source_weight_id(&w,&mut source_weights,None).unwrap());
            }
            let n=3+next()%10;let mut states=Vec::with_capacity(n);
            for q in 0..n {
                let mut rows=BTreeMap::<i32,SmallVec<[(u32,u32);1]>>::new();
                let mut epsilons=Vec::new();
                for target in q+1..n {
                    let w=weights[next()%weights.len()];
                    match next()%6 {
                        0=>epsilons.push((target as u32,w)),
                        1|2=>rows.entry(crate::compiler::glr::labels::encode_negative_label((next()%3) as u32)).or_default().push((target as u32,w)),
                        3|4=>rows.entry((next()%3) as i32).or_default().push((target as u32,w)),
                        _=>rows.entry(DEFAULT_LABEL).or_default().push((target as u32,w)),
                    }
                }
                states.push(FastBoundaryNwaState{final_weight:weights[next()%weights.len()],epsilons,
                    transitions:rows.into_iter().collect()});
            }
            let expected=fast_boundary_cancellations_worklist(&states,&mut interner).unwrap();
            let (actual,stats)=compute_with_topology_mode(&states,&mut interner,None,false).unwrap();
            let (filtered,filter_stats) = compute_with_topology_mode(&states,&mut interner,None,true).unwrap();
            rejected+=filter_stats.rejected_queries+filter_stats.rejected_edges;
            let literal=literal_paths(&states,&mut interner);
            pairs+=stats.result_pairs;
            for q in 0..n {
                assert_eq!(canonical(&actual[q]),canonical(&expected.derived[q]),"FIFO case={case} q={q}");
                assert_eq!(canonical(&filtered[q]),canonical(&actual[q]),"read filter case={case} q={q}");
                assert_eq!(canonical(&actual[q]),canonical(&literal[q]),"literal case={case} q={q}");
            }
        }
        assert!(pairs>0,"fixtures must actually complete queries");
        assert!(rejected>0,"fixtures must exercise actual read rejection");
    }

    #[test]
    fn cancellation_summary_declines_cycles_and_work_exhaustion() {
        let mut interner=FastBoundaryWeightInterner::new(1,64).unwrap();
        let states=vec![FastBoundaryNwaState{epsilons:vec![(0,1)],transitions:vec![],final_weight:0}];
        assert!(compute(&states,&mut interner).is_none());
        let states=vec![
            FastBoundaryNwaState{epsilons:vec![],transitions:vec![(crate::compiler::glr::labels::encode_negative_label(0),smallvec::smallvec![(1,1)])],final_weight:0},
            FastBoundaryNwaState{epsilons:vec![],transitions:vec![(0,smallvec::smallvec![(2,1)])],final_weight:0},
            FastBoundaryNwaState{epsilons:vec![],transitions:vec![],final_weight:1},
        ];
        interner.limits=Some(FiniteCompileLimits{work:0,..Default::default()});
        assert!(compute(&states,&mut interner).is_none());
    }
}

#[cfg(test)]
#[test]
fn read_synopsis_collisions_cannot_create_or_remove_cancellations() {
    let mut signatures=FxHashMap::default();
    let mut collision=None;
    for label in 0..10_000i32 {
        if let Some(previous)=signatures.insert(read_signature(label),label){
            collision=Some((previous,label));break;
        }
    }
    let (a,b)=collision.expect("only4096two-bit combinations exist");
    assert_ne!(a,b);assert_eq!(read_signature(a),read_signature(b));
    let push=crate::compiler::glr::labels::encode_negative_label;
    let source=vec![
        FastBoundaryNwaState{final_weight:0,epsilons:vec![],transitions:vec![(push(a as u32),smallvec::smallvec![(1,1)])]},
        FastBoundaryNwaState{final_weight:0,epsilons:vec![],transitions:vec![(b,smallvec::smallvec![(2,1)])]},
        FastBoundaryNwaState{final_weight:1,epsilons:vec![],transitions:vec![]},
    ];
    let mut pool=FastBoundaryWeightInterner::new(1,64).unwrap();
    let old=compute_with_topology_mode(&source,&mut pool,None,false).unwrap().0;
    let new=compute_with_topology_mode(&source,&mut pool,None,true).unwrap().0;
    assert_eq!(old[0].len(),0);assert_eq!(new[0].len(),0);
    let mut source=source;
    source[1].transitions.push((DEFAULT_LABEL,smallvec::smallvec![(2,1)]));
    source[1].transitions.sort_unstable_by_key(|r|r.0);
    let new=compute_with_topology_mode(&source,&mut pool,None,true).unwrap().0;
    assert_eq!(new[0].len(),1,"DEFAULT must never be rejected");
}
