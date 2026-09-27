//! Parametric union of terminal-transfer word languages.
//!
//! Terminal identities live ONLY in final outputs. Every transition is unit
//! weighted, so one deterministic word path yields a set of matching terminals.
//! Instantiation replaces a final tag set I by union_{i in I} W_i, which is
//! valid even when different W_i overlap. Replacing tag weights on arbitrary
//! edges would not preserve intersection and is deliberately forbidden here.
use glrmask_weight::__private::Weight;
use glrmask_weighted_automata::weighted_u32::{
    dwa::DWA,nwa::{NWA,NWAState},determinize::determinize,minimize::reverse_hashcons_owned,
};
use std::collections::{BTreeMap,VecDeque};
use rustc_hash::FxHashMap;
use std::time::Instant;

#[derive(Debug,Default)]
pub struct Profile {
    pub input_states:usize,pub input_edges:usize,pub states:usize,pub edges:usize,
    pub tag_sets:usize,pub tag_members:usize,pub prepare_ms:f64,
}

pub struct TaggedTemplates {
    graph:DWA,
    terminals:Vec<u32>,
    final_sets:Vec<usize>,
    support_sets:Vec<usize>,
    tags:Vec<Vec<u32>>,
    pub profile:Profile,
}

impl TaggedTemplates {
    pub fn build(templates:&BTreeMap<u32,NWA>)->Option<Self>{
        if templates.is_empty()||templates.len()>4096{return None;}
        let started=Instant::now();let mut arena=NWA::new(0,0);arena.add_state();arena.set_start_states(vec![0]);
        let mut profile=Profile::default();
        for (&terminal,template) in templates{
            if !template.is_acyclic(){return None;}
            profile.input_states=profile.input_states.checked_add(template.states().len())?;
            profile.input_edges=profile.input_edges.checked_add(template.num_transitions())?;
            if profile.input_states>150_000||profile.input_edges>1_000_000{return None;}
            let offset=arena.states().len();let body=arena.append_with_body(template);
            let tag=Weight::from_token_set_for_tsid(0,[terminal].into_iter().collect());
            for state in &mut arena.states_mut()[offset..]{
                if state.final_weight.is_some(){state.final_weight=Some(tag.clone());}
                for (_,weight) in state.epsilons.iter_mut().chain(state.transitions.values_mut().flatten()){
                    *weight=Weight::all();
                }
            }
            for start in body.start_states{arena.add_epsilon(0,start,Weight::all());}
        }
        // Prototype uses the mature ordinary determinizer. The input and
        // published index are bounded here; a stricter in-flight output budget
        // is required before promoting this experimental preparation path.
        let graph=reverse_hashcons_owned(determinize(&arena).ok()?);
        if graph.states().len()>200_000||graph.num_transitions()>2_000_000{return None;}
        if graph.states().iter().any(|q|q.transitions.values().any(|(_,w)|!w.is_full())){return None;}
        let n=graph.states().len();let mut degree=vec![0usize;n];
        for state in graph.states(){for (_,target,_) in state.transitions.entries(){degree[target as usize]+=1;}}
        let mut queue=(0..n).filter(|&q|degree[q]==0).collect::<VecDeque<_>>();let mut topo=Vec::with_capacity(n);
        while let Some(q)=queue.pop_front(){topo.push(q);for (_,target,_) in graph.states()[q].transitions.entries(){degree[target as usize]-=1;if degree[target as usize]==0{queue.push_back(target as usize);}}}
        if topo.len()!=n{return None;}
        let mut supports=vec![Weight::empty();n];
        for q in topo.into_iter().rev(){
            let row=&graph.states()[q];
            supports[q]=Weight::union_all(row.final_weight.iter().chain(row.transitions.values().map(|(target,_)|&supports[*target as usize])));
        }
        let mut ids=FxHashMap::<usize,usize>::default();let mut tags=Vec::<Vec<u32>>::new();
        let mut intern=|weight:&Weight|->Option<usize>{
            if let Some(&id)=ids.get(&weight.ptr_key()){return Some(id);}
            if weight.is_full(){return None;}
            let mut values=Vec::new();for (lo,hi,terminals) in weight.range_entries(){
                if lo!=0||hi!=0{return None;}values.extend(terminals.iter());
            }
            profile.tag_members=profile.tag_members.checked_add(values.len())?;if profile.tag_members>8_000_000{return None;}
            let id=tags.len();tags.push(values);ids.insert(weight.ptr_key(),id);Some(id)
        };
        let empty=Weight::empty();let mut final_sets=Vec::with_capacity(n);let mut support_sets=Vec::with_capacity(n);
        for (q,row) in graph.states().iter().enumerate(){
            final_sets.push(intern(row.final_weight.as_ref().unwrap_or(&empty))?);support_sets.push(intern(&supports[q])?);
        }
        profile.states=n;profile.edges=graph.num_transitions();profile.tag_sets=tags.len();profile.prepare_ms=started.elapsed().as_secs_f64()*1000.0;
        Some(Self{graph,terminals:templates.keys().copied().collect(),final_sets,support_sets,tags,profile})
    }

    pub fn instantiate(&self,weights:&BTreeMap<u32,Weight>)->Option<NWA>{
        if weights.keys().any(|terminal|self.terminals.binary_search(terminal).is_err()){return None;}
        let mut mapped=Vec::with_capacity(self.tags.len());
        for tags in &self.tags {
            let contributions=if weights.len()<tags.len(){
                weights.iter().filter(|(terminal,_)|tags.binary_search(terminal).is_ok()).map(|(_,w)|w).collect::<Vec<_>>()
            }else{tags.iter().filter_map(|terminal|weights.get(terminal)).collect::<Vec<_>>()};
            mapped.push(Weight::union_all(contributions));
        }
        let n=self.graph.states().len();let root=self.graph.start_state() as usize;
        if mapped[self.support_sets[root]].is_empty(){let mut empty=NWA::new(0,0);empty.add_state();empty.set_start_states(vec![0]);return Some(empty);}
        let mut live=vec![false;n];let mut todo=vec![root];
        while let Some(q)=todo.pop(){if live[q]{continue;}live[q]=true;for (_,target,_) in self.graph.states()[q].transitions.entries(){
            if !mapped[self.support_sets[target as usize]].is_empty(){todo.push(target as usize);}
        }}
        let mut map=vec![u32::MAX;n];let mut states=Vec::new();for q in 0..n{if live[q]{map[q]=states.len() as u32;states.push(NWAState::default());}}
        for q in 0..n{if !live[q]{continue;}
            let state=&mut states[map[q] as usize];let final_weight=&mapped[self.final_sets[q]];
            if !final_weight.is_empty(){state.final_weight=Some(final_weight.clone());}
            for (label,target,_) in self.graph.states()[q].transitions.entries(){
                if live[target as usize]{
                    // Every accepting suffix coefficient is contained in this
                    // target support. Pushing it back loses no weighted word.
                    state.transitions.insert(label,vec![(map[target as usize],mapped[self.support_sets[target as usize]].clone())]);
                }
            }
        }
        Some(NWA::from_parts(states,vec![map[root]]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glrmask_weighted_automata::weighted_u32::equivalence::find_difference;
    fn reference(templates:&BTreeMap<u32,NWA>,weights:&BTreeMap<u32,Weight>)->NWA{
        let mut result=NWA::new(0,0);result.add_state();result.set_start_states(vec![0]);
        for (terminal,weight) in weights{let template=&templates[terminal];let offset=result.states().len();let body=result.append_with_body(template);
            for state in &mut result.states_mut()[offset..]{
                if state.final_weight.is_some(){state.final_weight=Some(weight.clone());}
                for (_,w) in state.epsilons.iter_mut().chain(state.transitions.values_mut().flatten()){*w=Weight::all();}
            }
            for start in body.start_states{result.add_epsilon(0,start,Weight::all());}
        }result
    }
    #[test]
    fn tagged_final_substitution_preserves_overlapping_weights_and_epsilon_words(){
        let mut seed=199u64;let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32) as usize};
        let w=|bits:usize|Weight::from_per_tsid_token_sets([(0,(0..8u32).filter(|&b|bits&(1<<b)!=0).collect()),(7,(0..8u32).filter(|&b|bits&(1<<(b+1))!=0).collect())]);
        for case in 0..128{
            let mut templates=BTreeMap::new();
            for terminal in 0..5{let n=3+next()%5;let mut graph=NWA::new(0,0);for _ in 0..n{graph.add_state();}graph.set_start_states(vec![0]);graph.set_final_weight(n as u32-1,Weight::empty());
                for q in 0..n{if next()%4==0{graph.set_final_weight(q as u32,Weight::empty());}
                    for target in q+1..n{match next()%6{0=>graph.add_epsilon(q as u32,target as u32,Weight::empty()),1=>graph.add_transition(q as u32,-((next()%3) as i32)-1,target as u32,Weight::empty()),2=>graph.add_transition(q as u32,(next()%3) as i32,target as u32,Weight::empty()),_=>{}}}
                }templates.insert(terminal,graph);
            }
            let index=TaggedTemplates::build(&templates).unwrap();
            for variant in 0..4{
                let weights=(0..5).filter_map(|terminal|(next()%3!=0).then(||(terminal,w(next()%256)))).collect::<BTreeMap<_,_>>();
                let instantiated=index.instantiate(&weights).unwrap();
                let a=determinize(&reference(&templates,&weights)).unwrap();let b=determinize(&instantiated).unwrap();
                assert_eq!(find_difference(&a,&b).unwrap(),None,"case {case} variant {variant}");
                assert_eq!(find_difference(&b,&a).unwrap(),None,"reverse case {case} variant {variant}");
            }
        }
    }
}
