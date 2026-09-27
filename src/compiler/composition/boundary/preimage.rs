//! Backward bounded-control composition using the ordinary parser preimage DAG.
//!
//! A lane (P,w) denotes the input-stack predicate P with its exact lexical
//! TSID/token coefficient w. Preimage distributes over union, and multiplying
//! a path coefficient is intersection. No parser or vocabulary coordinates are
//! approximated. This compiler-only experiment declines unsupported templates.
use crate::compiler::stages::parser_dwa::LazyBooleanParserDomains;
use crate::compiler::glr::labels::{DEFAULT_LABEL,is_negative_label};
use glrmask_weight::__private::Weight;
use glrmask_weighted_automata::weighted_u32::{dwa::DWA,nwa::NWA};
use std::collections::{BTreeMap,BTreeSet,VecDeque};
use std::time::Instant;
use rustc_hash::FxHashMap;

type Lanes=BTreeMap<u32,Weight>;
#[derive(Debug,Default)]
pub struct Profile {
    pub template_states:usize,
    pub preimage_calls:usize,
    pub cache_hits:usize,
    pub expression_nodes:usize,
    pub max_lanes:usize,
    pub root_lanes:usize,
    pub prepare_ms:f64,
    pub dp_ms:f64,
    pub export_ms:f64,
}
pub struct Output { pub nwa:NWA,pub domain:Weight,pub profile:Profile }

fn unit_template(input:&NWA)->Option<NWA>{
    let n=input.states().len();
    if n==0||n>65_536{return None;}
    let mut degree=vec![0usize;n];
    for state in input.states(){for &(q,_) in state.epsilons.iter().chain(state.transitions.values().flatten()){
        if q as usize>=n{return None;}degree[q as usize]+=1;
    }}
    let mut queue=(0..n).filter(|&q|degree[q]==0).collect::<VecDeque<_>>();
    let mut count=0;let mut depth=vec![0usize;n];let mut pushed=vec![false;n];
    while let Some(q)=queue.pop_front(){
        count+=1;let state=&input.states()[q];
        for (&label,branches) in &state.transitions {
            if !is_negative_label(label)&&pushed[q]&&!branches.is_empty(){return None;}
            for &(next,_) in branches{
                pushed[next as usize]|=pushed[q]||is_negative_label(label);
            }
        }
        for &(next,_) in &state.epsilons{pushed[next as usize]|=pushed[q];}
        for &(next,_) in state.epsilons.iter().chain(state.transitions.values().flatten()){
            let next=next as usize;depth[next]=depth[next].max(depth[q]+1);
            if depth[next]>512{return None;}
            degree[next]-=1;if degree[next]==0{queue.push_back(next);}
        }
    }
    if count!=n{return None;}
    let mut output=input.clone();
    for state in output.states_mut(){
        if state.final_weight.is_some(){state.final_weight=Some(Weight::all());}
        for (_,weight) in state.epsilons.iter_mut().chain(state.transitions.values_mut().flatten()){
            // The ordinary template appender STAMPS all present branches,
            // including EMPTY annotations. Preserve that structural contract.
            *weight=Weight::all();
        }
    }
    Some(output)
}

fn merge(lanes:&mut Lanes,root:u32,weight:Weight){
    if root==LazyBooleanParserDomains::EMPTY||weight.is_empty(){return;}
    lanes.entry(root).and_modify(|prior|*prior=prior.union(&weight)).or_insert(weight);
}

struct Engine {
    arena:LazyBooleanParserDomains,
    templates:BTreeMap<u32,NWA>,
    cache:FxHashMap<(u32,u32),u32>,
    template_work:usize,
    profile:Profile,
}
impl Engine {
    fn preimage(&mut self,terminal:u32,target:u32)->Option<u32>{
        self.profile.preimage_calls+=1;
        if let Some(&root)=self.cache.get(&(terminal,target)){self.profile.cache_hits+=1;return Some(root);}
        let template=self.templates.get(&terminal)?;
        self.template_work=self.template_work.checked_add(template.states().len())?;
        if self.template_work>25_000_000||self.cache.len()>100_000{return None;}
        let root=self.arena.preimage_bundle(template,target)?;
        if self.arena.node_count()>1_000_000{return None;}
        self.cache.insert((terminal,target),root);Some(root)
    }
}

pub fn build(
    lexical:&DWA,templates:&BTreeMap<u32,NWA>,controls:&[u32],max_controls:usize,
)->Option<Output>{
    let started=Instant::now();let n=lexical.states().len();
    if n==0||n>4096||max_controls>16{return None;}
    let mut degree=vec![0usize;n];let mut used=controls.iter().copied().collect::<BTreeSet<_>>();
    for state in lexical.states(){for (label,target,_) in state.transitions.entries(){
        if label<0||label==DEFAULT_LABEL||target as usize>=n{return None;}
        used.insert(label as u32);degree[target as usize]+=1;
    }}
    let mut queue=(0..n).filter(|&q|degree[q]==0).collect::<VecDeque<_>>();
    let mut topo=Vec::with_capacity(n);
    while let Some(q)=queue.pop_front(){topo.push(q);for (_,target,_) in lexical.states()[q].transitions.entries(){
        degree[target as usize]-=1;if degree[target as usize]==0{queue.push_back(target as usize);}
    }}
    if topo.len()!=n{return None;}
    let mut engine=Engine{arena:LazyBooleanParserDomains::new(),templates:BTreeMap::new(),cache:FxHashMap::default(),template_work:0,profile:Profile::default()};
    for terminal in used{
        let unit=unit_template(templates.get(&terminal)?)?;
        engine.profile.template_states+=unit.states().len();
        if engine.profile.template_states>1_000_000{return None;}
        engine.templates.insert(terminal,unit);
    }
    engine.profile.prepare_ms=started.elapsed().as_secs_f64()*1000.0;
    let started=Instant::now();let mut domains=vec![Lanes::new();n];let mut lane_count=0usize;
    for source in topo.into_iter().rev(){
        let state=&lexical.states()[source];let mut ordinary=Lanes::new();
        for (terminal,target,weight) in state.transitions.entries(){
            if weight.is_empty(){continue;}
            for (&target_root,target_weight) in &domains[target as usize]{
                let support=weight.intersection(target_weight);if support.is_empty(){continue;}
                let root=engine.preimage(terminal as u32,target_root)?;merge(&mut ordinary,root,support);
            }
        }
        let mut current=ordinary.clone();
        // Every depth permits one lexical transfer back to depth zero. Only
        // the final depth-zero predicate below admits the lexical final.
        for _depth in (0..max_controls).rev(){
            let mut next=ordinary.clone();
            for &control in controls{for (&target_root,weight) in &current{
                let root=engine.preimage(control,target_root)?;merge(&mut next,root,weight.clone());
            }}
            if next.len()>16_384{return None;}
            engine.profile.max_lanes=engine.profile.max_lanes.max(next.len());current=next;
        }
        if let Some(weight)=&state.final_weight{merge(&mut current,LazyBooleanParserDomains::UNIVERSAL,weight.clone());}
        lane_count=lane_count.checked_add(current.len())?;if lane_count>200_000{return None;}
        domains[source]=current;
    }
    engine.profile.dp_ms=started.elapsed().as_secs_f64()*1000.0;
    let started=Instant::now();let roots=std::mem::take(domains.get_mut(lexical.start_state() as usize)?).into_iter().collect::<Vec<_>>();
    engine.profile.root_lanes=roots.len();engine.profile.expression_nodes=engine.arena.node_count();
    let domain=Weight::union_all(roots.iter().map(|(_,w)|w));if domain.is_full(){return None;}
    let mut nwa=engine.arena.to_weighted_nwa(&roots);
    // Each initial lane coefficient is contained in D. Replacing final ALL
    // by D preserves every weighted word while retaining a finite output
    // domain for the ordinary faithful native compiler/decoder.
    for state in nwa.states_mut(){if state.final_weight.is_some(){state.final_weight=Some(domain.clone());}}
    engine.profile.export_ms=started.elapsed().as_secs_f64()*1000.0;
    Some(Output{nwa,domain,profile:engine.profile})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::labels::{encode_negative_label,negative_to_positive_label};
    use glrmask_weighted_automata::weighted_u32::dwa::DWAState;
    fn run_transfer(template:&NWA,stack:&[u32])->BTreeSet<Vec<u32>>{
        let mut todo=template.start_states().iter().map(|&q|(q,stack.to_vec())).collect::<Vec<_>>();let mut seen=BTreeSet::new();let mut accepted=BTreeSet::new();
        while let Some((q,s))=todo.pop(){
            if !seen.insert((q,s.clone())){continue;}
            let state=&template.states()[q as usize];if state.final_weight.is_some(){accepted.insert(s.clone());}
            for &(target,_) in &state.epsilons{todo.push((target,s.clone()));}
            for (&label,targets) in &state.transitions{
                let mut next=s.clone();if is_negative_label(label){next.insert(0,negative_to_positive_label(label) as u32);}else{
                    if next.is_empty()||(label!=DEFAULT_LABEL&&next[0]!=label as u32){continue;}next.remove(0);
                }
                for &(target,_) in targets{todo.push((target,next.clone()));}
            }
        }accepted
    }
    fn original(lex:&DWA,templates:&BTreeMap<u32,NWA>,controls:&[u32],depth:usize,stack:&[u32])->Weight{
        let mut todo=vec![(lex.start_state(),0,stack.to_vec(),Weight::all())];let mut accepted=Weight::empty();
        while let Some((q,d,stack,weight))=todo.pop(){let state=&lex.states()[q as usize];
            if d==0{if let Some(final_weight)=&state.final_weight{accepted=accepted.union(&weight.intersection(final_weight));}}
            for (t,target,w) in state.transitions.entries(){let support=weight.intersection(w);if support.is_empty(){continue;}
                for output in run_transfer(&templates[&(t as u32)],&stack){todo.push((target,0,output,support.clone()));}
            }
            if d<depth{for t in controls{for output in run_transfer(&templates[t],&stack){todo.push((q,d+1,output,weight.clone()));}}}
        }accepted
    }
    fn evaluate(nwa:&NWA,stack:&[u32])->Weight{
        let mut todo=nwa.start_states().iter().map(|&q|(q,0,Weight::all())).collect::<Vec<_>>();let mut accepted=Weight::empty();
        while let Some((q,offset,weight))=todo.pop(){let state=&nwa.states()[q as usize];
            if let Some(final_weight)=&state.final_weight{accepted=accepted.union(&weight.intersection(final_weight));}
            for &(target,ref w) in &state.epsilons{let support=weight.intersection(w);if !support.is_empty(){todo.push((target,offset,support));}}
            if offset==stack.len(){continue;}
            for (&label,targets) in &state.transitions{if label!=DEFAULT_LABEL&&label!=stack[offset] as i32{continue;}
                for &(target,ref w) in targets{let support=weight.intersection(w);if !support.is_empty(){todo.push((target,offset+1,support));}}
            }
        }accepted
    }
    #[test]
    fn bounded_preimages_match_concrete_stack_execution_and_weight_correlation(){
        let make=|read:u32,push:u32|{let mut n=NWA::new(0,0);for _ in 0..3{n.add_state();}n.set_start_states(vec![0]);n.add_transition(0,read as i32,1,Weight::empty());n.add_transition(1,encode_negative_label(push),2,Weight::empty());n.set_final_weight(2,Weight::empty());n};
        let templates=BTreeMap::from([(0,make(0,1)),(1,make(1,2)),(2,make(2,0)),(3,make(0,2))]);
        let controls=[2u32];let mut seed=73u64;let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32) as usize};
        let w=|bits:usize|Weight::from_token_set_for_tsid(0,(0..4u32).filter(|&b|bits&(1<<b)!=0).collect());
        let mut words=vec![vec![]];let mut layer=vec![vec![]];for _ in 0..3{let mut following=Vec::new();for s in layer{for b in 0..3{let mut word=s.clone();word.push(b);following.push(word);}}words.extend(following.clone());layer=following;}
        for _ in 0..64{let mut states=vec![DWAState::default();4];
            for q in 0..4{states[q].final_weight=Some(w(next()%16));if q<3{for t in [0,1,3]{states[q].transitions.insert(t,(q as u32+1,w(next()%16)));}}}
            let lex=DWA::from_parts(states,0);let output=build(&lex,&templates,&controls,2).unwrap();
            for stack in &words{assert_eq!(evaluate(&output.nwa,stack),original(&lex,&templates,&controls,2,stack),"stack {stack:?}");}
        }
    }
}
