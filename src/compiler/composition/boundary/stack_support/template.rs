//! Necessary input-domain pruning before a signed template is copied.
//!
//! The ordinary append operation replaces every PRESENT edge annotation by
//! the caller's weight. Consequently this analysis uses the template's unit
//! structural language, including edges annotated EMPTY. It never interprets
//! an old placeholder annotation as absence. Input reads are filtered by a
//! certified suffix-closed domain; after the first push, the untouched output
//! suffix is retained. Interleaved push/read programs conservatively decline.
use super::coarse::Certificate;
use crate::compiler::glr::labels::{DEFAULT_LABEL,is_negative_label,negative_to_positive_label};
use glrmask_weight::__private::Weight;
use glrmask_weighted_automata::weighted_u32::nwa::{NWA,NWAState};
use std::collections::VecDeque;

pub struct TemplateReadSupport {
    root:usize,
    words:usize,
    alphabet:usize,
    target:Vec<u32>,
    allowed:Vec<u64>,
}

#[derive(Debug,Default)]
pub struct Stats {
    pub states_before:usize,
    pub states_after:usize,
    pub edges_before:usize,
    pub edges_after:usize,
}

impl TemplateReadSupport {
    pub fn new(certificate:&Certificate)->Option<Self>{
        let domain=&certificate.domain;
        domain.validate().ok()?;
        let a=domain.alphabet as usize;
        let d=domain.nodes.len();
        if d==0||d>4096||a>50_000||domain.nodes.iter().any(|q|!q.epsilons.is_empty()){return None;}
        let root=domain.start as usize;
        let words=d.div_ceil(64);
        let live=domain.coaccessible();
        let mut target=vec![u32::MAX;a];
        let mut allowed=vec![0u64;a.checked_mul(words)?];
        for (source,node) in domain.nodes.iter().enumerate(){
            for (&label,destinations) in &node.transitions {
                if destinations.len()!=1{return None;}
                let q=*destinations.first()?;
                if target[label as usize]!=u32::MAX&&target[label as usize]!=q{return None;}
                target[label as usize]=q;
                if live[q as usize]{allowed[label as usize*words+source/64]|=1u64<<(source%64);}
            }
        }
        if !domain.nodes[root].accepting{return None;}
        for label in 0..a {
            if target[label]!=u32::MAX&&live[target[label] as usize]
                && allowed[label*words+root/64]&(1u64<<(root%64))==0{return None;}
        }
        Some(Self{root,words,alphabet:a,target,allowed})
    }

    /// Exact relative to stacks in the issuing certificate. No original input
    /// is mutated, and a declined analysis returns no partial candidate.
    pub fn restrict(&self,input:&NWA)->Option<(NWA,Stats)>{
        let n=input.states().len();
        if n==0||n>100_000||n.checked_mul(self.words)?>8_000_000{return None;}
        let mut degree=vec![0usize;n];let mut edges=0usize;
        for state in input.states(){
            for (&label,branches) in &state.transitions {
                let concrete=if is_negative_label(label){negative_to_positive_label(label)}else{label};
                if label!=DEFAULT_LABEL&&(concrete<0||concrete as usize>=self.alphabet){return None;}
                for &(q,_) in branches{if q as usize>=n{return None;}degree[q as usize]+=1;edges+=1;}
            }
            for &(q,_) in &state.epsilons{if q as usize>=n{return None;}degree[q as usize]+=1;edges+=1;}
        }
        if edges>2_000_000{return None;}
        let mut queue=(0..n).filter(|&q|degree[q]==0).collect::<VecDeque<_>>();
        let mut topo=Vec::with_capacity(n);
        while let Some(q)=queue.pop_front(){
            topo.push(q);
            for &(next,_) in input.states()[q].epsilons.iter().chain(input.states()[q].transitions.values().flatten()){
                degree[next as usize]-=1;if degree[next as usize]==0{queue.push_back(next as usize);}
            }
        }
        if topo.len()!=n{return None;}
        let mut contexts=vec![0u64;n*self.words];
        let mut output_phase=vec![false;n];
        let mut live=vec![false;n];
        let mut kept=vec![Vec::<i32>::new();n];
        for &q in input.start_states(){
            if q as usize>=n{return None;}
            contexts[q as usize*self.words+self.root/64]|=1u64<<(self.root%64);
        }
        let mut work=0usize;
        for q in topo {
            let source=contexts[q*self.words..(q+1)*self.words].to_vec();
            let reading=source.iter().any(|&b|b!=0);
            live[q]=reading||output_phase[q];
            if !live[q]{continue;}
            let state=&input.states()[q];
            for &(next,_) in &state.epsilons {
                let next=next as usize;
                output_phase[next]|=output_phase[q];
                for (a,b) in contexts[next*self.words..(next+1)*self.words].iter_mut().zip(&source){*a|=*b;}
                work=work.checked_add(self.words)?;if work>50_000_000{return None;}
            }
            for (&label,branches) in &state.transitions {
                if is_negative_label(label){
                    kept[q].push(label);
                    for &(next,_) in branches{output_phase[next as usize]=true;}
                    continue;
                }
                if output_phase[q]&&!branches.is_empty(){return None;}
                let possible=reading&&(label==DEFAULT_LABEL||source.iter()
                    .zip(&self.allowed[label as usize*self.words..(label as usize+1)*self.words]).any(|(a,b)|a&b!=0));
                if !possible{continue;}
                kept[q].push(label);
                let context=if label==DEFAULT_LABEL{self.root}else{self.target[label as usize] as usize};
                for &(next,_) in branches{contexts[next as usize*self.words+context/64]|=1u64<<(context%64);}
            }
        }
        let mut map=vec![u32::MAX;n];let mut states=Vec::new();
        for q in 0..n{if live[q]{map[q]=states.len() as u32;states.push(NWAState::default());}}
        let mut stats=Stats{states_before:n,states_after:states.len(),edges_before:edges,..Default::default()};
        for q in 0..n{if !live[q]{continue;}
            let original=&input.states()[q];let row=&mut states[map[q] as usize];
            row.final_weight=original.final_weight.as_ref().map(|_|Weight::all());
            for &(next,_) in &original.epsilons {
                row.epsilons.push((map[next as usize],Weight::all()));stats.edges_after+=1;
            }
            for (&label,branches) in &original.transitions {
                let output=row.transitions.entry(label).or_default();
                if kept[q].binary_search(&label).is_ok(){
                    for &(next,_) in branches{output.push((map[next as usize],Weight::all()));stats.edges_after+=1;}
                }
            }
        }
        Some((NWA::from_parts(states,input.start_states().iter().map(|&q|map[q as usize]).collect()),stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{coarse,effects::Effect};
    use crate::compiler::glr::labels::encode_negative_label;
    use std::collections::BTreeSet;
    fn execute(input:&NWA,stack:&[u32])->BTreeSet<Vec<u32>>{
        let mut todo=input.start_states().iter().map(|&q|(q,stack.to_vec())).collect::<Vec<_>>();
        let mut seen=BTreeSet::new();let mut result=BTreeSet::new();
        while let Some((q,stack))=todo.pop(){
            if !seen.insert((q,stack.clone())){continue;}
            let state=&input.states()[q as usize];
            if state.final_weight.is_some(){result.insert(stack.clone());}
            for &(next,_) in &state.epsilons{todo.push((next,stack.clone()));}
            for (&label,branches) in &state.transitions {
                let mut next_stack=stack.clone();
                if is_negative_label(label){next_stack.insert(0,negative_to_positive_label(label) as u32);}
                else {
                    if stack.is_empty()||(label!=DEFAULT_LABEL&&stack[0]!=label as u32){continue;}
                    next_stack.remove(0);
                }
                for &(next,_) in branches{todo.push((next,next_stack.clone()));}
            }
        }
        result
    }
    #[test]
    fn template_pruning_preserves_complete_stack_relation_on_certified_inputs(){
        let effects=vec![Effect{source:0,pop:0,pushes:vec![1]},Effect{source:1,pop:0,pushes:vec![2]},Effect{source:2,pop:1,pushes:vec![3]}];
        let certificate=coarse::certify(&[0],&effects,4).unwrap();
        let support=TemplateReadSupport::new(&certificate).unwrap();
        let mut words=vec![vec![]];let mut layer=vec![vec![]];
        for _ in 0..5{let mut next=Vec::new();for word in layer{for label in 0..4{let mut w=word.clone();w.push(label);next.push(w);}}words.extend(next.clone());layer=next;}
        words.retain(|word|certificate.domain.accepts(word));
        let mut seed=39u64;let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32) as usize};
        let mut removed=0;
        for _case in 0..128{
            let mut input=NWA::new(0,0);for _ in 0..9{input.add_state();}input.set_start_states(vec![0]);
            for q in 0..9 {
                if next()%3==0{input.states_mut()[q].final_weight=Some(Weight::empty());}
                for target in q+1..9 {
                    if next()%3!=0{continue;}
                    if next()%4==0{input.add_epsilon(q as u32,target as u32,Weight::empty());}
                    else {
                        let label=if q>=5{encode_negative_label((next()%4) as u32)}else if next()%5==0{DEFAULT_LABEL}else{(next()%4) as i32};
                        input.add_transition(q as u32,label,target as u32,Weight::empty());
                    }
                }
            }
            let (candidate,stats)=support.restrict(&input).unwrap();removed+=stats.edges_before-stats.edges_after;
            for word in &words{assert_eq!(execute(&input,word),execute(&candidate,word),"stack {word:?}");}
        }
        assert!(removed>0);
    }
    #[test]
    fn push_then_read_and_cycles_decline_without_mutating_input(){
        let cert=coarse::certify(&[0],&[],2).unwrap();let support=TemplateReadSupport::new(&cert).unwrap();
        let mut input=NWA::new(0,0);for _ in 0..3{input.add_state();}input.set_start_states(vec![0]);
        input.add_transition(0,encode_negative_label(1),1,Weight::empty());input.add_transition(1,1,2,Weight::empty());
        let before=input.clone();assert!(support.restrict(&input).is_none());assert_eq!(before.states(),input.states());
        input.add_epsilon(2,0,Weight::all());assert!(support.restrict(&input).is_none());
    }
}
