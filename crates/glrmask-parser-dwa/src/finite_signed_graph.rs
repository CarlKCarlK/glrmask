//! Exact read-only adjacency over an eager or factored signed template program.
//!
//! Logical IDs, row/key order, final presence and zero markers match the eager
//! builder. The virtual graph shares existing template TOPOLOGY only; it does
//! not merge parser states or summarize stack semantics.
use super::*;
use super::finite_template_program::FiniteTemplateProgram;
#[cfg(test)]
use super::finite_template_program::FiniteTemplateInstance;
use std::ops::Range;

pub(super) trait SignedRow: Copy {
    fn final_weight(&self) -> FastBoundaryWeightId;
    fn epsilon_count(&self) -> usize;
    fn epsilon(&self,index:usize) -> (u32,FastBoundaryWeightId);
    fn transition_count(&self) -> usize;
    fn label(&self,index:usize) -> i32;
    fn branch_count(&self,index:usize) -> usize;
    fn branch(&self,index:usize,branch:usize) -> (u32,FastBoundaryWeightId);
    fn find_label(&self,label:i32) -> Option<usize> {
        let(mut low,mut high)=(0,self.transition_count());
        while low<high {
            let mid=low+(high-low)/2;
            match self.label(mid).cmp(&label) {
                std::cmp::Ordering::Less=>low=mid+1,
                std::cmp::Ordering::Greater=>high=mid,
                std::cmp::Ordering::Equal=>return Some(mid),
            }
        }
        None
    }
}

pub(super) trait SignedGraph {
    type Row<'a>:SignedRow where Self:'a;
    fn len(&self) -> usize;
    fn row(&self,index:usize) -> Self::Row<'_>;
}

impl SignedRow for &FastBoundaryNwaState {
    #[inline] fn final_weight(&self)->u32 {self.final_weight}
    #[inline] fn epsilon_count(&self)->usize {self.epsilons.len()}
    #[inline] fn epsilon(&self,index:usize)->(u32,u32) {self.epsilons[index]}
    #[inline] fn transition_count(&self)->usize {self.transitions.len()}
    #[inline] fn label(&self,index:usize)->i32 {self.transitions[index].0}
    #[inline] fn branch_count(&self,index:usize)->usize {self.transitions[index].1.len()}
    #[inline] fn branch(&self,index:usize,branch:usize)->(u32,u32) {self.transitions[index].1[branch]}
    #[inline] fn find_label(&self,label:i32)->Option<usize> {
        self.transitions.binary_search_by_key(&label,|(label,_)|*label).ok()
    }
}

impl SignedGraph for [FastBoundaryNwaState] {
    type Row<'a>=&'a FastBoundaryNwaState;
    #[inline] fn len(&self)->usize {<[FastBoundaryNwaState]>::len(self)}
    #[inline] fn row(&self,index:usize)->Self::Row<'_> {&self[index]}
}

/// The reference terminal/default fixed point, over a read-only positive view.
/// Negative keys are ignored because the eager resolver removes them first.
/// Zero DEFAULT branches still create dependencies: they cannot be discarded
/// or ordered using the nonzero-edge DAG certificate.
pub(super) fn terminal_default_states<G:SignedGraph+?Sized>(
    graph:&G,pool:&FastBoundaryWeightInterner,
)->Vec<bool> {
    let n=graph.len();
    let mut terminal=(0..n).map(|q| {
        let row=graph.row(q);
        row.final_weight()!=0 && row.epsilon_count()==0
            && !(0..row.transition_count()).any(|e|
                !is_negative_label(row.label(e)) && row.branch_count(e)!=0)
    }).collect::<Vec<_>>();
    let mut dependents=vec![Vec::<usize>::new();n];
    let mut remaining=vec![usize::MAX;n];
    let mut queue=VecDeque::new();
    for(q,&done)in terminal.iter().enumerate(){if done{queue.push_back(q);}}
    for q in 0..n {
        if terminal[q]{continue;}
        let row=graph.row(q);
        if row.final_weight()==0 || row.epsilon_count()!=0
            || (0..row.transition_count()).any(|e| {
                let label=row.label(e);
                !is_negative_label(label) && label!=DEFAULT_LABEL && row.branch_count(e)!=0
            }) {continue;}
        let Some(edge)=row.find_label(DEFAULT_LABEL) else {
            terminal[q]=true;queue.push_back(q);continue;
        };
        if (0..row.branch_count(edge)).any(|b|!pool.is_subset(row.branch(edge,b).1,row.final_weight())) {
            continue;
        }
        let mut count=0usize;
        for b in 0..row.branch_count(edge) {
            let target=row.branch(edge,b).0 as usize;
            if target>=n {count+=1;}
            else if !terminal[target] {dependents[target].push(q);count+=1;}
        }
        remaining[q]=count;
        if count==0{terminal[q]=true;queue.push_back(q);}
    }
    while let Some(done)=queue.pop_front(){
        for &q in &dependents[done] {
            if terminal[q] || remaining[q]==usize::MAX || remaining[q]==0 {continue;}
            remaining[q]-=1;
            if remaining[q]==0{terminal[q]=true;queue.push_back(q);}
        }
    }
    terminal
}

struct FinalizedGraph<'a,G:SignedGraph+?Sized>{graph:&'a G,finals:&'a[u32]}
#[derive(Clone,Copy)]
struct FinalizedRow<R:SignedRow>{row:R,final_weight:u32}
impl<R:SignedRow> SignedRow for FinalizedRow<R>{
    #[inline] fn final_weight(&self)->u32{self.final_weight}
    #[inline] fn epsilon_count(&self)->usize{self.row.epsilon_count()}
    #[inline] fn epsilon(&self,index:usize)->(u32,u32){self.row.epsilon(index)}
    #[inline] fn transition_count(&self)->usize{self.row.transition_count()}
    #[inline] fn label(&self,index:usize)->i32{self.row.label(index)}
    #[inline] fn branch_count(&self,index:usize)->usize{self.row.branch_count(index)}
    #[inline] fn branch(&self,index:usize,branch:usize)->(u32,u32){self.row.branch(index,branch)}
    #[inline] fn find_label(&self,label:i32)->Option<usize>{self.row.find_label(label)}
}
impl<G:SignedGraph+?Sized> SignedGraph for FinalizedGraph<'_,G>{
    type Row<'a>=FinalizedRow<G::Row<'a>> where Self:'a;
    fn len(&self)->usize{self.graph.len()}
    fn row(&self,index:usize)->Self::Row<'_>{
        FinalizedRow{row:self.graph.row(index),final_weight:self.finals[index]}
    }
}

#[inline]
fn keep_positive_branch(label:i32,target:u32,weight:u32,final_weight:u32,
    terminal:&[bool],pool:&FastBoundaryWeightInterner,
)->bool {
    label!=DEFAULT_LABEL || final_weight==0 || target as usize>=terminal.len()
        || !terminal[target as usize] || !pool.is_subset(weight,final_weight)
}

/// Matches eager Kahn ordering, including transition-before-epsilon traversal.
pub(super) fn topological_order<G:SignedGraph+?Sized>(graph:&G)->Option<Vec<u32>> {
    let n=graph.len();
    let mut indegree=vec![0usize;n];
    for q in 0..n {
        let row=graph.row(q);
        for edge in 0..row.transition_count() {
            for branch in 0..row.branch_count(edge) {
                let(target,weight)=row.branch(edge,branch);
                if target as usize>=n{return None;}
                if weight!=0 {indegree[target as usize]=indegree[target as usize].checked_add(1)?;}
            }
        }
        for edge in 0..row.epsilon_count() {
            let(target,weight)=row.epsilon(edge);
            if target as usize>=n{return None;}
            if weight!=0 {indegree[target as usize]=indegree[target as usize].checked_add(1)?;}
        }
    }
    let mut queue=(0..n).filter(|&q|indegree[q]==0).map(|q|q as u32).collect::<VecDeque<_>>();
    let mut order=Vec::with_capacity(n);
    while let Some(q)=queue.pop_front() {
        order.push(q);
        let row=graph.row(q as usize);
        for edge in 0..row.transition_count() {
            for branch in 0..row.branch_count(edge) {
                let(target,weight)=row.branch(edge,branch);
                if weight==0 {continue;}
                indegree[target as usize]-=1;
                if indegree[target as usize]==0 {queue.push_back(target);}
            }
        }
        for edge in 0..row.epsilon_count() {
            let(target,weight)=row.epsilon(edge);
            if weight==0 {continue;}
            indegree[target as usize]-=1;
            if indegree[target as usize]==0 {queue.push_back(target);}
        }
    }
    (order.len()==n).then_some(order)
}

struct TemplateRow {epsilons:Range<usize>,transitions:Range<usize>,present_final:bool}
struct TemplateTransition {label:i32,targets:Range<usize>}
struct TemplateTopology {
    rows:Vec<TemplateRow>,epsilons:Vec<u32>,transitions:Vec<TemplateTransition>,targets:Vec<u32>,
    starts:Vec<u32>,logical_edges:usize,
}
struct Instance {template:usize,base:u32,coefficient:u32,continuation:u32}

#[derive(Debug,Default)]
pub(super) struct VirtualGraphProfile {
    pub logical_states:usize,
    pub logical_edges:usize,
    pub stored_template_states:usize,
    pub stored_template_edges:usize,
    pub instances:usize,
    pub port_edges:usize,
}

#[derive(Debug,Default)]
pub(super) struct VirtualResolveProfile {
    pub cancellation_ms:f64,
    pub finality_ms:f64,
    pub materialize_ms:f64,
    pub prune_ms:f64,
    pub reachability_ms:f64,
    pub read_context_ms:f64,
    pub context_input_states:usize,
    pub context_allocated_states:usize,
    pub materialized_states:usize,
    pub materialized_edges:usize,
}

pub(super) struct PositiveMaterialization {
    pub states:Vec<FastBoundaryNwaState>,
    pub starts:Vec<u32>,
    pub topology:Vec<u32>,
    pub profile:VirtualResolveProfile,
}

pub(super) struct VirtualSignedGraph {
    alphabet:u32,
    templates:Vec<TemplateTopology>,instances:Vec<Instance>,ports:Vec<FastBoundaryNwaState>,
    node_instance:Vec<u32>,
    pub profile:VirtualGraphProfile,
}

#[derive(Clone,Copy)]
enum RowKind<'a> {
    Port(&'a FastBoundaryNwaState),
    Template {
        epsilons:&'a[u32],transitions:&'a[TemplateTransition],targets:&'a[u32],
        base:u32,coefficient:u32,continuation:Option<u32>,
    },
}

#[derive(Clone,Copy)]
pub(super) struct VirtualRow<'a> {kind:RowKind<'a>}

impl SignedRow for VirtualRow<'_> {
    #[inline] fn final_weight(&self)->u32 {
        match self.kind {RowKind::Port(row)=>row.final_weight,RowKind::Template{..}=>0}
    }
    #[inline] fn epsilon_count(&self)->usize {
        match self.kind {RowKind::Port(row)=>row.epsilons.len(),
            RowKind::Template{epsilons,continuation,..}=>epsilons.len()+usize::from(continuation.is_some())}
    }
    #[inline] fn epsilon(&self,index:usize)->(u32,u32) {
        match self.kind {RowKind::Port(row)=>row.epsilons[index],
            RowKind::Template{epsilons,base,coefficient,continuation,..}=> {
                let target=if index<epsilons.len(){base+epsilons[index]}else{continuation.expect("present-final epsilon")};
                (target,coefficient)
            }}
    }
    #[inline] fn transition_count(&self)->usize {
        match self.kind {RowKind::Port(row)=>row.transitions.len(),RowKind::Template{transitions,..}=>transitions.len()}
    }
    #[inline] fn label(&self,index:usize)->i32 {
        match self.kind {RowKind::Port(row)=>row.transitions[index].0,RowKind::Template{transitions,..}=>transitions[index].label}
    }
    #[inline] fn branch_count(&self,index:usize)->usize {
        match self.kind {RowKind::Port(row)=>row.transitions[index].1.len(),
            RowKind::Template{transitions,coefficient,..}=>{
                let n=transitions[index].targets.len();
                if coefficient==0 {usize::from(n!=0)}else{n}
            }}
    }
    #[inline] fn branch(&self,index:usize,branch:usize)->(u32,u32) {
        match self.kind {RowKind::Port(row)=>row.transitions[index].1[branch],
            RowKind::Template{transitions,targets,base,coefficient,..}=>{
                if coefficient==0 {(0,0)}else{(base+targets[transitions[index].targets.start+branch],coefficient)}
            }}
    }
}

impl SignedGraph for VirtualSignedGraph {
    type Row<'a>=VirtualRow<'a>;
    #[inline] fn len(&self)->usize {self.node_instance.len()}
    #[inline] fn row(&self,q:usize)->VirtualRow<'_> {
        if q<self.ports.len(){return VirtualRow{kind:RowKind::Port(&self.ports[q])};}
        let instance=&self.instances[self.node_instance[q]as usize];
        let template=&self.templates[instance.template];
        let row=&template.rows[q-instance.base as usize];
        VirtualRow{kind:RowKind::Template{
            epsilons:&template.epsilons[row.epsilons.clone()],transitions:&template.transitions[row.transitions.clone()],
            targets:&template.targets,base:instance.base,coefficient:instance.coefficient,
            continuation:row.present_final.then_some(instance.continuation),
        }}
    }
}

impl VirtualSignedGraph {
    pub fn build(program:&FiniteTemplateProgram<'_>,alphabet:u32,
        pool:&mut FastBoundaryWeightInterner,limits:FiniteCompileLimits,
    )->Option<(Self,Vec<u32>)> {
        let ports=program.port_finals.len();
        if ports==0 || ports>limits.states || alphabet==0 || alphabet as usize>limits.states
            || alphabet>=DEFAULT_LABEL as u32 || program.starts.iter().any(|&q|q as usize>=ports)
            || program.coefficients.len()>limits.weights {return None;}
        let mut templates=Vec::with_capacity(program.templates.len());
        // Also bound topology retained for unused templates. The instantiated
        // graph budget alone does not bound this additional private storage.
        let mut stored_rows=0usize;
        let mut stored_entries=0usize;
        for source in program.templates {
            let n=source.states().len();
            if n==0 || n>limits.states || source.start_states().iter().any(|&q|q as usize>=n){return None;}
            stored_rows=stored_rows.checked_add(n)?;
            if stored_rows>limits.states{return None;}
            let mut template=TemplateTopology{rows:Vec::with_capacity(n),epsilons:Vec::new(),transitions:Vec::new(),
                targets:Vec::new(),starts:source.start_states().to_vec(),logical_edges:0};
            for row in source.states() {
                let row_entries=row.epsilons.len().checked_add(row.transitions.len())?
                    .checked_add(usize::from(row.final_weight.is_some()))?;
                stored_entries=stored_entries.checked_add(row_entries)?;
                for branches in row.transitions.values(){
                    stored_entries=stored_entries.checked_add(branches.len())?;
                }
                if stored_entries>limits.edges{return None;}
                let eps=template.epsilons.len();
                for &(target,_) in &row.epsilons {
                    if target as usize>=n{return None;}
                    template.epsilons.push(target);
                }
                let transitions=template.transitions.len();
                let mut prior=None;
                for (&label,branches) in &row.transitions {
                    let valid=label==DEFAULT_LABEL || (0..alphabet as i32).contains(&label)
                        || (is_negative_label(label) && (0..alphabet as i32).contains(&negative_to_positive_label(label)));
                    if !valid || prior.is_some_and(|p|p>=label){return None;}
                    prior=Some(label);
                    let start=template.targets.len();
                    for &(target,_) in branches {
                        if target as usize>=n{return None;}
                        template.targets.push(target);
                    }
                    template.transitions.push(TemplateTransition{label,targets:start..template.targets.len()});
                }
                template.logical_edges=template.logical_edges.checked_add(row.epsilons.len())?
                    .checked_add(usize::from(row.final_weight.is_some()))?
                    .checked_add(row.transitions.values().map(|branches|branches.len()).sum::<usize>())?;
                if template.logical_edges>limits.edges{return None;}
                template.rows.push(TemplateRow{epsilons:eps..template.epsilons.len(),
                    transitions:transitions..template.transitions.len(),present_final:row.final_weight.is_some()});
            }
            templates.push(template);
        }
        let mut total_states=ports;let mut total_edges=0usize;
        for instance in program.instances {
            let template=templates.get(instance.template)?;
            if instance.coefficient>=program.coefficients.len() || instance.continuation as usize>=ports
                || instance.entries.start>instance.entries.end || instance.entries.end as usize>ports {return None;}
            total_states=total_states.checked_add(template.rows.len())?;
            total_edges=total_edges.checked_add(template.logical_edges)?.checked_add(
                (instance.entries.end-instance.entries.start)as usize*template.starts.len())?;
            if total_states>limits.states || total_edges>limits.edges || total_states>u32::MAX as usize{return None;}
        }
        let mut source_ids=FxHashMap::default();
        let coefficients=program.coefficients.iter().map(|w|pool.source_weight_id(w,&mut source_ids,None))
            .collect::<Option<Vec<_>>>()?;
        let mut port_rows=Vec::with_capacity(ports);
        for final_weight in program.port_finals {
            let final_weight=match final_weight{Some(id)=>*coefficients.get(*id)?,None=>0};
            port_rows.push(FastBoundaryNwaState{epsilons:Vec::new(),transitions:Vec::new(),final_weight});
        }
        let mut instances=Vec::with_capacity(program.instances.len());
        let mut node_instance=vec![u32::MAX;ports];
        node_instance.reserve(total_states-ports);
        for instance in program.instances {
            let template=&templates[instance.template];
            let base=node_instance.len()as u32;
            node_instance.resize(node_instance.len()+template.rows.len(),instances.len()as u32);
            instances.push(Instance{template:instance.template,base,coefficient:coefficients[instance.coefficient],
                continuation:instance.continuation});
            for port in instance.entries.clone() {
                port_rows[port as usize].epsilons.extend(template.starts.iter().map(|&q|(base+q,pool.all_id())));
            }
        }
        if !pool.allow_work(0,total_states,total_edges){return None;}
        let profile=VirtualGraphProfile{logical_states:total_states,logical_edges:total_edges,
            stored_template_states:templates.iter().map(|t|t.rows.len()).sum(),
            stored_template_edges:templates.iter().map(|t|t.logical_edges).sum(),
            instances:instances.len(),port_edges:port_rows.iter().map(|r|r.epsilons.len()).sum()};
        let graph=Self{alphabet,templates,instances,ports:port_rows,node_instance,profile};
        let order=topological_order(&graph)?;
        Some((graph,order))
    }

    /// Read the original signed graph and add cancellation epsilons as an
    /// overlay. Original negative edges remain visible through finality, then
    /// only the positive graph is materialized for the unchanged next stages.
    #[cfg(test)]
    pub fn resolve_positive(&self,pool:&mut FastBoundaryWeightInterner,initial_order:&[u32],
        reuse_topology:bool,filter:bool,
    )->Option<(Vec<FastBoundaryNwaState>,VirtualResolveProfile)> {
        let result=self.resolve_positive_mode(pool,initial_order,reuse_topology,filter,None,None)?;
        Some((result.states,result.profile))
    }

    pub fn resolve_positive_reachable(&self,pool:&mut FastBoundaryWeightInterner,initial_order:&[u32],
        reuse_topology:bool,filter:bool,starts:&[u32],
    )->Option<PositiveMaterialization> {
        self.resolve_positive_mode(pool,initial_order,reuse_topology,filter,Some(starts),None)
    }

    pub fn resolve_positive_with_context(&self,pool:&mut FastBoundaryWeightInterner,
        initial_order:&[u32],reuse_topology:bool,filter:bool,starts:&[u32],
        context:&FiniteParserReadSupport,
    )->Option<PositiveMaterialization> {
        if context.alphabet()!=self.alphabet as usize{return None;}
        self.resolve_positive_mode(pool,initial_order,reuse_topology,filter,Some(starts),Some(context))
    }

    fn resolve_positive_mode(&self,pool:&mut FastBoundaryWeightInterner,initial_order:&[u32],
        reuse_topology:bool,filter:bool,reachable_starts:Option<&[u32]>,
        read_context:Option<&FiniteParserReadSupport>,
    )->Option<PositiveMaterialization> {
        let phase=Instant::now();
        let(derived,stats)=super::finite_cancellation::compute_on_graph(self,pool,initial_order,filter)?;
        let mut profile=VirtualResolveProfile{cancellation_ms:elapsed_ms(phase),..Default::default()};
        if compile_profile_enabled(){eprintln!("[glrmask/profile][virtual_cancellation_summary] {stats:?}");}
        let phase=Instant::now();
        let extra=derived.into_iter().map(|row|row.into_entries().into_iter()
            .filter(|&(_,weight)|weight!=0).collect::<Vec<_>>()).collect::<Vec<_>>();
        let overlay=EpsilonOverlay{base:self,extra:&extra};
        if reuse_topology {
            let certificate=CheckedNativeTopology::from_order(initial_order.to_vec())?;
            if certificate.order.len()!=self.len(){return None;}
            for(q,edges)in extra.iter().enumerate(){
                if edges.iter().any(|&(target,_)|!certificate.permits(q,target)){return None;}
            }
        }
        let owned_order;
        let order=if reuse_topology {initial_order}else{
            owned_order=topological_order(&overlay)?;&owned_order
        };
        let n=self.len();
        let mut finals=(0..n).map(|q|self.row(q).final_weight()).collect::<Vec<_>>();
        for &q in order.iter().rev(){
            let row=overlay.row(q as usize);
            let mut value=finals[q as usize];
            for index in 0..row.epsilon_count(){
                let(target,weight)=row.epsilon(index);
                let contribution=pool.intersection(weight,finals[target as usize]);
                value=pool.union(value,contribution);
            }
            for edge in 0..row.transition_count(){
                let label=row.label(edge);
                if label!=DEFAULT_LABEL && !is_negative_label(label){continue;}
                for branch in 0..row.branch_count(edge){
                    let(target,weight)=row.branch(edge,branch);
                    let contribution=pool.intersection(weight,finals[target as usize]);
                    value=pool.union(value,contribution);
                }
            }
            finals[q as usize]=value;
        }
        if !pool.allow_work(0,n,0){return None;}
        profile.finality_ms=elapsed_ms(phase);
        if let Some(starts)=reachable_starts {
            let finalized=FinalizedGraph{graph:&overlay,finals:&finals};
            if let Some(context)=read_context {
                // This order certifies the complete original/derived overlay;
                // positive pruning removes edges only. Context transfer can
                // therefore precede row allocation without another graph scan.
                return materialize_sparse_context_positive(&finalized,pool,order,starts,profile,context);
            }
            return materialize_reachable_positive(&finalized,pool,initial_order,starts,profile);
        }
        let phase=Instant::now();
        let mut states=Vec::with_capacity(n);
        for(q,final_weight)in finals.into_iter().enumerate(){
            let row=overlay.row(q);
            let epsilons=(0..row.epsilon_count()).map(|i|row.epsilon(i)).collect::<Vec<_>>();
            let mut transitions=Vec::new();
            for edge in 0..row.transition_count(){
                let label=row.label(edge);
                if is_negative_label(label){continue;}
                let branches=(0..row.branch_count(edge)).map(|branch|row.branch(edge,branch))
                    .collect::<SmallVec<[(u32,FastBoundaryWeightId);1]>>();
                transitions.push((label,branches));
            }
            profile.materialized_edges+=epsilons.len()+transitions.iter().map(|(_,b)|b.len()).sum::<usize>();
            states.push(FastBoundaryNwaState{epsilons,transitions,final_weight});
        }
        profile.materialize_ms=elapsed_ms(phase);
        let phase=Instant::now();
        // This helper is the old resolver's exact worklist, not a guessed
        // topological recurrence over possibly backward zero DEFAULT guards.
        fast_boundary_prune_terminal_defaults(&mut states,pool);
        profile.prune_ms=elapsed_ms(phase);
        profile.materialized_states=states.len();
        if !pool.allow_work(0,n,profile.materialized_edges){return None;}
        Some(PositiveMaterialization{states,starts:Vec::new(),topology:initial_order.to_vec(),profile})
    }
}

/// Same result as eager finality/pruning followed by the existing positive
/// trim, but never allocates an expanded row for an unreachable logical node.
fn materialize_reachable_positive<G:SignedGraph+?Sized>(graph:&G,pool:&mut FastBoundaryWeightInterner,
    original_order:&[u32],starts:&[u32],mut profile:VirtualResolveProfile,
)->Option<PositiveMaterialization> {
    let n=graph.len();
    if starts.iter().any(|&q|q as usize>=n){return None;}
    let phase=Instant::now();
    let terminal=terminal_default_states(graph,pool);
    profile.prune_ms=elapsed_ms(phase);
    let phase=Instant::now();
    let mut live=vec![false;n];
    let mut todo=starts.to_vec();
    while let Some(q)=todo.pop(){
        if std::mem::replace(&mut live[q as usize],true){continue;}
        let row=graph.row(q as usize);
        for e in 0..row.epsilon_count(){
            let(target,weight)=row.epsilon(e);
            if target as usize>=n{return None;}
            if weight!=0 && !live[target as usize]{todo.push(target);}
        }
        for e in 0..row.transition_count(){
            let label=row.label(e);
            if is_negative_label(label){continue;}
            for b in 0..row.branch_count(e){
                let(target,weight)=row.branch(e,b);
                if target as usize>=n{return None;}
                if weight!=0 && !live[target as usize]
                    && keep_positive_branch(label,target,weight,row.final_weight(),&terminal,pool){
                    todo.push(target);
                }
            }
        }
    }
    let mut mapping=vec![u32::MAX;n];
    let mut count=0u32;
    for(q,&keep)in live.iter().enumerate(){if keep{mapping[q]=count;count+=1;}}
    let starts=starts.iter().map(|&q|mapping[q as usize]).collect();
    let topology=original_order.iter().filter_map(|&q|{
        let mapped=mapping[q as usize];(mapped!=u32::MAX).then_some(mapped)
    }).collect();
    profile.reachability_ms=elapsed_ms(phase);
    let phase=Instant::now();
    let mut states=Vec::with_capacity(count as usize);
    for(q,&keep)in live.iter().enumerate(){
        if !keep{continue;}
        let row=graph.row(q);
        let mut epsilons=Vec::new();
        for e in 0..row.epsilon_count(){
            let(target,weight)=row.epsilon(e);
            if weight!=0{epsilons.push((mapping[target as usize],weight));}
        }
        let mut transitions=Vec::new();
        for e in 0..row.transition_count(){
            let label=row.label(e);
            if is_negative_label(label){continue;}
            let mut had_branches=false;
            let mut branches=SmallVec::<[(u32,FastBoundaryWeightId);1]>::new();
            for b in 0..row.branch_count(e){
                let(target,weight)=row.branch(e,b);
                if !keep_positive_branch(label,target,weight,row.final_weight(),&terminal,pool){continue;}
                had_branches=true;
                if weight!=0{branches.push((mapping[target as usize],weight));}
            }
            // Prune removes empty keys; trim retains zero-only guard keys as
            // exactly one dummy edge, without making an unreachable target live.
            if !had_branches{continue;}
            if branches.is_empty(){branches.push((0,0));}
            transitions.push((label,branches));
        }
        profile.materialized_edges+=epsilons.len()+transitions.iter().map(|(_,b)|b.len()).sum::<usize>();
        states.push(FastBoundaryNwaState{epsilons,transitions,final_weight:row.final_weight()});
    }
    profile.materialized_states=states.len();
    profile.materialize_ms=elapsed_ms(phase);
    if !pool.allow_work(0,n,profile.materialized_edges){return None;}
    Some(PositiveMaterialization{states,starts,topology,profile})
}

/// A private consumer of the complete, checked finalized positive view.
/// The only caller has validated all original and derived targets/topology,
/// plus equality of graph/domain alphabets. DEFAULT classification is still
/// evaluated over every logical row before any context-sensitive removal.
fn materialize_sparse_context_positive<G:SignedGraph+?Sized>(
    graph:&G,pool:&mut FastBoundaryWeightInterner,order:&[u32],starts:&[u32],
    mut profile:VirtualResolveProfile,domain:&FiniteParserReadSupport,
)->Option<PositiveMaterialization> {
    let n=graph.len();
    let phase=Instant::now();
    let terminal=terminal_default_states(graph,pool);
    profile.prune_ms=elapsed_ms(phase);
    let phase=Instant::now();
    let mut transfer=super::finite_read_support::SparseReadContextTransfer::new(domain,n,starts)?;
    for &q in order {
        if !transfer.load_source(q as usize)?{continue;}
        let row=graph.row(q as usize);
        for e in 0..row.epsilon_count(){
            let(target,weight)=row.epsilon(e);
            if weight!=0 {transfer.transfer_epsilon(target as usize)?;}
        }
        for e in 0..row.transition_count(){
            let label=row.label(e);
            if is_negative_label(label) || !transfer.read_allowed(label)? {continue;}
            for b in 0..row.branch_count(e){
                let(target,weight)=row.branch(e,b);
                if weight!=0 && keep_positive_branch(label,target,weight,row.final_weight(),&terminal,pool){
                    transfer.transfer_read(label,target as usize)?;
                }
            }
        }
    }
    profile.context_input_states=n;
    profile.context_allocated_states=transfer.allocated_nodes();
    profile.read_context_ms=elapsed_ms(phase);
    let phase=Instant::now();
    let mut mapping=vec![u32::MAX;n];
    let mut count=0u32;
    for(q,slot)in mapping.iter_mut().enumerate(){
        if transfer.state_nonempty(q)? {*slot=count;count+=1;}
    }
    let starts=starts.iter().map(|&q|mapping[q as usize]).collect();
    let topology=order.iter().filter_map(|&q|{
        let mapped=mapping[q as usize];(mapped!=u32::MAX).then_some(mapped)
    }).collect();
    profile.reachability_ms=elapsed_ms(phase);
    let phase=Instant::now();
    let mut states=Vec::with_capacity(count as usize);
    for(q,&mapped)in mapping.iter().enumerate(){
        if mapped==u32::MAX {continue;}
        if !transfer.load_source(q)? {return None;}
        let row=graph.row(q);
        let mut epsilons=Vec::new();
        for e in 0..row.epsilon_count(){let(target,weight)=row.epsilon(e);
            if weight!=0{
                let target=*mapping.get(target as usize)?;
                if target==u32::MAX{return None;}
                epsilons.push((target,weight));
            }
        }
        let mut transitions=Vec::new();
        for e in 0..row.transition_count(){
            let label=row.label(e);
            if is_negative_label(label){continue;}
            let allowed=transfer.read_allowed(label)?;
            let mut had_branches=false;
            let mut branches=SmallVec::<[(u32,FastBoundaryWeightId);1]>::new();
            for b in 0..row.branch_count(e){let(target,weight)=row.branch(e,b);
                if !keep_positive_branch(label,target,weight,row.final_weight(),&terminal,pool){continue;}
                had_branches=true;
                if weight!=0 && allowed{
                    let target=*mapping.get(target as usize)?;
                    if target==u32::MAX{return None;}
                    branches.push((target,weight));
                }
            }
            if !had_branches {continue;}
            if branches.is_empty(){branches.push((0,0));}
            transitions.push((label,branches));
        }
        profile.materialized_edges+=epsilons.len()+transitions.iter().map(|(_,b)|b.len()).sum::<usize>();
        states.push(FastBoundaryNwaState{epsilons,transitions,final_weight:row.final_weight()});
    }
    profile.materialized_states=states.len();
    profile.materialize_ms=elapsed_ms(phase);
    if !pool.allow_work(0,n,profile.materialized_edges){return None;}
    Some(PositiveMaterialization{states,starts,topology,profile})
}

/// Original epsilon sequence followed by derived cancellation edges.
pub(super) struct EpsilonOverlay<'a,G:SignedGraph+?Sized> {
    pub base:&'a G,pub extra:&'a[Vec<(u32,FastBoundaryWeightId)>],
}
#[derive(Clone,Copy)]
pub(super) struct OverlayRow<'a,R:SignedRow> {row:R,extra:&'a[(u32,FastBoundaryWeightId)]}
impl<R:SignedRow> SignedRow for OverlayRow<'_,R> {
    #[inline] fn final_weight(&self)->u32 {self.row.final_weight()}
    #[inline] fn epsilon_count(&self)->usize {self.row.epsilon_count()+self.extra.len()}
    #[inline] fn epsilon(&self,index:usize)->(u32,u32) {
        if index<self.row.epsilon_count(){self.row.epsilon(index)}else{self.extra[index-self.row.epsilon_count()]}
    }
    #[inline] fn transition_count(&self)->usize {self.row.transition_count()}
    #[inline] fn label(&self,index:usize)->i32 {self.row.label(index)}
    #[inline] fn branch_count(&self,index:usize)->usize {self.row.branch_count(index)}
    #[inline] fn branch(&self,index:usize,branch:usize)->(u32,u32) {self.row.branch(index,branch)}
    #[inline] fn find_label(&self,label:i32)->Option<usize> {self.row.find_label(label)}
}
impl<G:SignedGraph+?Sized> SignedGraph for EpsilonOverlay<'_,G> {
    type Row<'a>=OverlayRow<'a,G::Row<'a>> where Self:'a;
    fn len(&self)->usize {self.base.len()}
    fn row(&self,index:usize)->Self::Row<'_> {OverlayRow{row:self.base.row(index),extra:&self.extra[index]}}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_same_rows(left:&[FastBoundaryNwaState],right:&[FastBoundaryNwaState],case:usize){
        assert_eq!(left.len(),right.len(),"case={case}");
        for(q,(a,b))in left.iter().zip(right).enumerate(){
            assert_eq!(a.final_weight,b.final_weight,"final case={case} q={q}");
            assert_eq!(a.epsilons,b.epsilons,"eps case={case} q={q}");
            assert_eq!(a.transitions,b.transitions,"keys/branches case={case} q={q}");
        }
    }

    #[test]
    fn factored_program_matches_eager_rows_coefficients_and_kahn_order() {
        let mut seed=79331u64;
        let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32)as usize};
        for case in 0..512 {
            let mut templates=Vec::new();
            for _ in 0..3 {
                let n=3+next()%6;let mut t=NWA::new(0,0);
                for _ in 0..n{t.add_state();}
                t.set_start_states(if next()%2==0{vec![0,1]}else{vec![0]});
                for q in 0..n {
                    if next()%3==0 || q+1==n {t.set_final_weight(q as u32,Weight::empty());}
                    for target in q+1..n {
                        if next()%4==0{t.add_epsilon(q as u32,target as u32,Weight::empty());}
                        else {
                            let label=[0,1,DEFAULT_LABEL,crate::compiler::glr::labels::encode_negative_label(1)][next()%4];
                            t.add_transition(q as u32,label,target as u32,Weight::all());
                        }
                    }
                    if next()%3==0{t.states_mut()[q].transitions.entry(2).or_default();}
                }
                templates.push(t);
            }
            let refs=templates.iter().collect::<Vec<_>>();
            let coefficients=[Weight::all(),Weight::from_token_set_for_tsid(0,[1,3,7].into_iter().collect()),Weight::empty()];
            let finals=[None,None,Some(1)];
            let instances=(0..3).map(|i|FiniteTemplateInstance{template:i,coefficient:(case+i)%3,
                continuation:if i==0{1}else{2},entries:if i==0{0..1}else{0..2}}).collect::<Vec<_>>();
            let program=FiniteTemplateProgram{templates:&refs,coefficients:&coefficients,
                port_finals:&finals,starts:&[0],instances:&instances};
            let mut left=FastBoundaryWeightInterner::new(1,64).unwrap();
            let mut right=FastBoundaryWeightInterner::new(1,64).unwrap();
            let(eager,edges,topo)=super::super::finite_template_program::build(&program,3,&mut left,Default::default()).unwrap();
            let(virtual_graph,vt)=VirtualSignedGraph::build(&program,3,&mut right,Default::default()).unwrap();
            assert_eq!(topo,vt,"topo case={case}");assert_eq!(edges,virtual_graph.profile.logical_edges);
            assert_eq!(left.values,right.values);assert_eq!(eager.len(),virtual_graph.len());
            for(q,a)in eager.iter().enumerate(){
                let b=virtual_graph.row(q);assert_eq!(a.final_weight,b.final_weight());
                assert_eq!(a.epsilons,(0..b.epsilon_count()).map(|i|b.epsilon(i)).collect::<Vec<_>>());
                assert_eq!(a.transitions.len(),b.transition_count());
                for (i,(label,branches))in a.transitions.iter().enumerate(){
                    assert_eq!(*label,b.label(i));assert_eq!(b.find_label(*label),Some(i));
                    assert_eq!(branches.as_slice(),(0..b.branch_count(i)).map(|j|b.branch(i,j)).collect::<Vec<_>>());
                }
            }
            let(a,sa)=super::super::finite_cancellation::compute_with_topology(&eager,&mut left,None).unwrap();
            let(b,sb)=super::super::finite_cancellation::compute_on_graph(&virtual_graph,&mut right,&vt,true).unwrap();
            assert_eq!(sa.queries,sb.queries,"memo queries case={case}");
            assert_eq!(sa.result_pairs,sb.result_pairs);
            assert_eq!(left.values,right.values,"exact interner sequence case={case}");
            for (q,(a,b))in a.into_iter().zip(b).enumerate(){
                let mut a=a.into_entries();let mut b=b.into_entries();a.sort_unstable();b.sort_unstable();
                assert_eq!(a,b,"derived epsilon case={case} state={q}");
            }
            // Rebuild pools from the same coefficient import to compare all
            // finality/interner operations, not just the cancellation relation.
            let mut left=FastBoundaryWeightInterner::new(1,64).unwrap();
            let mut right=FastBoundaryWeightInterner::new(1,64).unwrap();
            let(mut eager,_,_)=super::super::finite_template_program::build(&program,3,&mut left,Default::default()).unwrap();
            let(graph,order)=VirtualSignedGraph::build(&program,3,&mut right,Default::default()).unwrap();
            fast_boundary_resolve_negative_codes_with_topology(&mut eager,&mut left,None).unwrap();
            let(positive,_)=graph.resolve_positive(&mut right,&order,false,true).unwrap();
            assert_eq!(left.values,right.values,"post-finality interner sequence case={case}");
            assert_eq!(eager.len(),positive.len());
            for(q,(a,b))in eager.iter().zip(&positive).enumerate(){
                assert_eq!(a.epsilons,b.epsilons,"positive eps case={case} q={q}");
                assert_eq!(a.transitions,b.transitions,"positive row case={case} q={q}");
                assert_eq!(a.final_weight,b.final_weight,"positive final case={case} q={q}");
            }
            let mut compact_pool=FastBoundaryWeightInterner::new(1,64).unwrap();
            let(graph,order)=VirtualSignedGraph::build(&program,3,&mut compact_pool,Default::default()).unwrap();
            let compact=graph.resolve_positive_reachable(&mut compact_pool,&order,false,true,&[0]).unwrap();
            assert_eq!(left.values,compact_pool.values,"compact interner sequence case={case}");
            let(expected,starts)=super::super::finite_template_program::trim(eager.clone(),&[0]).unwrap();
            assert_eq!(starts,compact.starts);
            assert_same_rows(&expected,&compact.states,case);
            let certificate=CheckedNativeTopology::from_order(compact.topology).unwrap();
            assert!(certificate.certifies(&compact.states));
            // D1 moves only reachability before the existing context pass.
            // Compare complete output after both operation orders, including
            // DEFAULT forgetting and a domain that blocks some ordinary reads.
            let domain=FiniteParserReadSupport::new_checked(3,0,
                &[vec![(0,1),(1,1),(2,2)],vec![(0,1)],vec![(2,2)]],&[true,true,true],true).unwrap();
            let mut expected=eager;
            let mut actual=compact.states;
            super::super::finite_read_support::restrict(&mut expected,&[0],&domain).unwrap();
            super::super::finite_read_support::restrict_with_topology(&mut actual,&compact.starts,&domain,Some(&certificate)).unwrap();
            let(expected,expected_starts)=super::super::finite_template_program::trim(expected,&[0]).unwrap();
            let(actual,actual_starts)=super::super::finite_template_program::trim(actual,&compact.starts).unwrap();
            assert_eq!(expected_starts,actual_starts);
            assert_same_rows(&expected,&actual,case);
            for reuse in [false,true] {
                let mut reference_pool=FastBoundaryWeightInterner::new(1,64).unwrap();
                let(mut reference,_,reference_order)=super::super::finite_template_program::build(
                    &program,3,&mut reference_pool,Default::default()).unwrap();
                let reference_topology=reuse.then(||CheckedNativeTopology::from_order(reference_order).unwrap());
                fast_boundary_resolve_negative_codes_with_topology(
                    &mut reference,&mut reference_pool,reference_topology.as_ref()).unwrap();
                super::super::finite_read_support::restrict_with_topology(
                    &mut reference,&[0],&domain,reference_topology.as_ref()).unwrap();
                let(reference,reference_starts)=super::super::finite_template_program::trim(reference,&[0]).unwrap();
                let mut sparse_pool=FastBoundaryWeightInterner::new(1,64).unwrap();
                let(graph,order)=VirtualSignedGraph::build(&program,3,&mut sparse_pool,Default::default()).unwrap();
                let sparse=graph.resolve_positive_with_context(
                    &mut sparse_pool,&order,reuse,true,&[0],&domain).unwrap();
                assert_eq!(sparse.starts,reference_starts,"sparse starts case={case}");
                assert_same_rows(&reference,&sparse.states,case);
                assert_eq!(reference_pool.values,sparse_pool.values,"sparse interner case={case} reuse={reuse}");
                assert_eq!(sparse.profile.context_allocated_states,sparse.states.len());
                let certificate=CheckedNativeTopology::from_order(sparse.topology).unwrap();
                assert!(certificate.certifies(&sparse.states));
            }
        }
    }

    #[test]
    fn terminal_classification_matches_literal_least_fixed_point_with_zero_cycles(){
        let mut pool=FastBoundaryWeightInterner::new(1,8).unwrap();
        let weights=(0..16u64).map(|w|pool.intern(smallvec::smallvec![w])).collect::<Vec<_>>();
        let mut seed=89435u64;
        let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32)as usize};
        for case in 0..512 {
            let n=3+next()%19;
            let mut states=Vec::new();
            for _ in 0..n {
                let mut row=FastBoundaryNwaState{final_weight:weights[next()%16],epsilons:Vec::new(),transitions:Vec::new()};
                if next()%7==0{row.epsilons.push((next()as u32%n as u32,0));}
                for label in [0,1,DEFAULT_LABEL] {
                    if next()%2==0 {
                        let branches=(0..next()%3).map(|_|(next()as u32%n as u32,weights[next()%16]))
                            .collect::<SmallVec<[(u32,u32);1]>>();
                        row.transitions.push((label,branches));
                    }
                }
                states.push(row);
            }
            let mut expected=vec![false;n];
            loop {
                let mut changed=false;
                for(q,row)in states.iter().enumerate(){
                    if expected[q] || row.final_weight==0 || !row.epsilons.is_empty()
                        || row.transitions.iter().any(|(l,b)|*l!=DEFAULT_LABEL && !b.is_empty()){continue;}
                    let valid=row.transitions.iter().filter(|(l,_)|*l==DEFAULT_LABEL)
                        .flat_map(|(_,b)|b).all(|&(t,w)|expected[t as usize] && pool.is_subset(w,row.final_weight));
                    if valid{expected[q]=true;changed=true;}
                }
                if !changed{break;}
            }
            assert_eq!(terminal_default_states(states.as_slice(),&pool),expected,"case={case}");
        }
    }

    #[test]
    fn virtual_graph_rejects_cycles_bad_ports_and_budgets() {
        let mut t=NWA::new(0,0);t.add_state();t.set_start_states(vec![0]);t.set_final_weight(0,Weight::all());
        let coefficients=[Weight::all()];let refs=[&t];let finals=[None];
        let instances=[FiniteTemplateInstance{template:0,coefficient:0,continuation:0,entries:0..1}];
        let p=FiniteTemplateProgram{templates:&refs,coefficients:&coefficients,port_finals:&finals,starts:&[0],instances:&instances};
        let mut pool=FastBoundaryWeightInterner::new(1,64).unwrap();
        assert!(VirtualSignedGraph::build(&p,4,&mut pool,Default::default()).is_none());
        assert!(VirtualSignedGraph::build(&p,4,&mut pool,FiniteCompileLimits{states:1,..Default::default()}).is_none());
        let mut bad_instances=instances.clone();
        bad_instances[0].continuation=8;
        let p=FiniteTemplateProgram{instances:&bad_instances,..p};
        assert!(VirtualSignedGraph::build(&p,4,&mut pool,Default::default()).is_none());
    }

    #[test]
    fn multiple_starts_zero_instances_and_reused_topology_match_eager() {
        let mut template=NWA::new(0,0);
        for _ in 0..4{template.add_state();}
        template.set_start_states(vec![0,1]);
        template.set_final_weight(3,Weight::empty());
        template.add_transition(0,crate::compiler::glr::labels::encode_negative_label(1),1,Weight::all());
        template.add_transition(1,1,2,Weight::all());
        template.add_transition(2,0,3,Weight::all());
        template.states_mut()[1].transitions.entry(2).or_default();
        let refs=[&template];
        let coefficients=[Weight::empty(),Weight::from_token_set_for_tsid(0,[2,5].into_iter().collect())];
        let finals=[None,None,Some(1)];
        let instances=(0..3).map(|i|FiniteTemplateInstance{template:0,coefficient:i%2,
            continuation:2,entries:0..2}).collect::<Vec<_>>();
        let starts=[0,1];
        let program=FiniteTemplateProgram{templates:&refs,coefficients:&coefficients,
            port_finals:&finals,starts:&starts,instances:&instances};
        for reuse in [false,true] {
            let mut left=FastBoundaryWeightInterner::new(1,64).unwrap();
            let mut right=FastBoundaryWeightInterner::new(1,64).unwrap();
            let(mut expected,_,order)=super::super::finite_template_program::build(&program,3,&mut left,Default::default()).unwrap();
            let certificate=CheckedNativeTopology::from_order(order).unwrap();
            fast_boundary_resolve_negative_codes_with_topology(&mut expected,&mut left,reuse.then_some(&certificate)).unwrap();
            let(expected,expected_starts)=super::super::finite_template_program::trim(expected,&starts).unwrap();
            let(graph,order)=VirtualSignedGraph::build(&program,3,&mut right,Default::default()).unwrap();
            let actual=graph.resolve_positive_reachable(&mut right,&order,reuse,true,&starts).unwrap();
            assert_eq!(left.values,right.values);
            assert_eq!(expected_starts,actual.starts);
            assert_same_rows(&expected,&actual.states,usize::from(reuse));
        }
        // Unused malformed or excessive topology must never bypass validation.
        let unused=[&template,&template];
        let no_instances=FiniteTemplateProgram{templates:&unused,instances:&[],..program};
        let mut pool=FastBoundaryWeightInterner::new(1,64).unwrap();
        assert!(VirtualSignedGraph::build(&no_instances,3,&mut pool,
            FiniteCompileLimits{states:6,..Default::default()}).is_none());
    }
}
