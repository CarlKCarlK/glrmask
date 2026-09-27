//! Necessary context support for a positive stack-reading NWA.
//!
//! A certified suffix-closed predecessor domain is an overapproximation of
//! real stack words. Contexts join by union. Concrete reads test domain
//! availability, epsilon preserves it, and DEFAULT forgets to the domain
//! root. Forgetting is sound because every suffix of an admitted word is
//! admitted from the root. This is a rejection proof, never mask admission.
//! Normalizer-specific contextual DEFAULT shortcuts require a separate gate.
use std::collections::VecDeque;
use glrmask_weight::__private::Weight;
use glrmask_weighted_automata::weighted_u32::nwa::NWA;
use super::coarse::Certificate;
const DEFAULT:i32=2147483646;

#[derive(Debug,Default)]
pub struct Stats {
    pub states:usize,
    pub domain_states:usize,
    pub live_states:usize,
    pub live_edges_before:usize,
    pub live_edges_after:usize,
    pub blocked_labels:usize,
    pub forgotten_defaults:usize,
    pub propagation_words:usize,
}

fn has(bits:&[u64],id:usize)->bool {bits[id/64]&(1u64<<(id%64))!=0}
fn insert(bits:&mut [u64],id:usize,root:usize){
    if has(bits,root){return;}
    if id==root{bits.fill(0);}
    bits[id/64]|=1u64<<(id%64);
}

/// `domain` must be an already-certified suffix-closed adjacency language.
/// We verify the graph shape, not the real-parser effect certificate here.
/// The caller retains that independent obligation. On failure no result is
/// published; call this on an owned candidate, not on the reference input.
pub fn restrict(nwa:&mut NWA,certificate:&Certificate)->Result<Stats,String>{
    let domain = &certificate.domain;
    domain.validate()?;
    let (n,d,a)=(nwa.states().len(),domain.nodes.len(),domain.alphabet as usize);
    if n==0||n>200_000||d==0||d>4096||a>50_000{return Err("support size budget".into());}
    if domain.nodes.iter().any(|q|!q.epsilons.is_empty()){return Err("domain must be deterministic without epsilon".into());}
    let root=domain.start as usize;let words=d.div_ceil(64);let live=domain.coaccessible();
    if n.checked_mul(words).ok_or("context overflow")?>8_000_000{return Err("context word budget".into());}
    let mut target=vec![u32::MAX;a];
    let mut allowed=vec![0u64;a*words];
    for (source,node) in domain.nodes.iter().enumerate(){
        for (&label,dests) in &node.transitions{
            if dests.len()!=1{return Err("nondeterministic domain".into());}
            let q=*dests.first().unwrap();let l=label as usize;
            if target[l]!=u32::MAX&&target[l]!=q{return Err("domain target is not label-local".into());}
            target[l]=q;
            if live[q as usize]{allowed[l*words+source/64]|=1u64<<(source%64);}
        }
    }
    // Root supports every nonempty residual: this verifies the particular
    // suffix-closed adjacency shape used for DEFAULT context forgetting.
    for label in 0..a{
        if target[label]!=u32::MAX && live[target[label] as usize]
            && !has(&allowed[label*words..(label+1)*words],root){
            return Err("domain root does not cover all suffix entries".into());
        }
    }
    if !domain.nodes[root].accepting{return Err("empty suffix not admitted".into());}
    let mut indegree=vec![0usize;n];let mut edges=0usize;
    for s in nwa.states(){
        for (&label,branches) in &s.transitions{
            if label!=DEFAULT && (label<0||label as usize>=a){return Err("non-positive/read label".into());}
            for &(q,ref w) in branches{
                if q as usize>=n{return Err("invalid target".into());}
                if !w.is_empty(){indegree[q as usize]+=1;edges+=1;}
            }
        }
        for &(q,ref w) in &s.epsilons{
            if q as usize>=n{return Err("invalid epsilon target".into());}
            if !w.is_empty(){indegree[q as usize]+=1;edges+=1;}
        }
    }
    if edges>4_000_000{return Err("edge budget".into());}
    let mut queue=(0..n).filter(|&q|indegree[q]==0).collect::<VecDeque<_>>();let mut topo=Vec::with_capacity(n);
    while let Some(q)=queue.pop_front(){
        topo.push(q);
        for (next,w) in nwa.states()[q].epsilons.iter().chain(nwa.states()[q].transitions.values().flatten()){
            if w.is_empty(){continue;}indegree[*next as usize]-=1;
            if indegree[*next as usize]==0{queue.push_back(*next as usize);}
        }
    }
    if topo.len()!=n{return Err("positive graph is cyclic".into());}
    let mut contexts=vec![0u64;n*words];
    for &q in nwa.start_states(){
        if q as usize>=n{return Err("invalid NWA start".into());}
        insert(&mut contexts[q as usize*words..(q as usize+1)*words],root,root);
    }
    let mut stats=Stats{states:n,domain_states:d,live_edges_before:edges,..Default::default()};
    for q in topo{
        let source=contexts[q*words..(q+1)*words].to_vec();
        let reachable=source.iter().any(|&word|word!=0);
        let state=&mut nwa.states_mut()[q];
        if !reachable{
            state.final_weight=None;
            for (_,w) in state.epsilons.iter_mut().chain(state.transitions.values_mut().flatten()){*w=Weight::empty();}
            continue;
        }
        stats.live_states+=1;
        for (next,w) in &mut state.epsilons{
            if w.is_empty(){continue;}
            let dest=&mut contexts[*next as usize*words..(*next as usize+1)*words];
            if has(&source,root){insert(dest,root,root);}else if !has(dest,root){
                for (x,y) in dest.iter_mut().zip(&source){*x|=*y;}stats.propagation_words+=words;
            }
            stats.live_edges_after+=1;
        }
        for (&label,branches) in &mut state.transitions{
            let possible=label==DEFAULT || source.iter().zip(&allowed[label as usize*words..(label as usize+1)*words]).any(|(x,y)|x&y!=0);
            if !possible{stats.blocked_labels+=1;for (_,w) in branches{*w=Weight::empty();}continue;}
            let context=if label==DEFAULT{root}else{target[label as usize] as usize};
            for (next,w) in branches{
                if w.is_empty(){continue;}
                insert(&mut contexts[*next as usize*words..(*next as usize+1)*words],context,root);
                stats.live_edges_after+=1;stats.forgotten_defaults+=usize::from(label==DEFAULT);
            }
        }
    }
    Ok(stats)
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::{coarse,effects::Effect};
    fn w(bits:u32)->Weight{Weight::from_token_set_for_tsid(0,(0..3).filter(|b|bits&(1<<b)!=0).collect())}
    fn eval(nwa:&NWA,word:&[u32])->Weight{
        let mut state=vec![Weight::empty();nwa.states().len()];
        for &q in nwa.start_states(){state[q as usize]=Weight::all();}
        let mut result=Weight::empty();
        for i in 0..=word.len(){
            // Tests generate all edges toward larger IDs, including epsilon.
            for q in 0..state.len(){for (target,weight) in &nwa.states()[q].epsilons{
                state[*target as usize]=state[*target as usize].union(&state[q].intersection(weight));
            }}
            for (q,s) in nwa.states().iter().enumerate(){if let Some(f)=&s.final_weight{result=result.union(&state[q].intersection(f));}}
            if i==word.len(){break;}
            let mut next=vec![Weight::empty();state.len()];
            for (q,s) in nwa.states().iter().enumerate(){for label in [word[i] as i32,DEFAULT]{
                for (target,w) in s.transitions.get(&label).into_iter().flatten(){
                    next[*target as usize]=next[*target as usize].union(&state[q].intersection(w));
                }
            }}state=next;
        }
        result
    }
    #[test]
    fn generated_necessary_support_preserves_all_admitted_small_words(){
        let mut seed=1021u64;let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32) as usize};
        let mut removed=0;
        for case in 0..128{
            let effects=(0..7).map(|_|{let len=next()%3;Effect{source:(next()%4) as u32,pop:next()%3,pushes:(0..len).map(|_|(next()%4) as u32).collect()}}).collect::<Vec<_>>();
            let certificate=coarse::certify(&[0],&effects,4).unwrap();
            let domain=&certificate.domain;
            let n=3+next()%6;let mut original=NWA::new(1,8);for _ in 0..n{original.add_state();}original.set_start_states(vec![0]);
            for q in 0..n{
                if next()%3==0{original.set_final_weight(q as u32,w((next()%8) as u32));}
                for target in q+1..n{
                    let weight=w((next()%8) as u32);let label=next()%7;
                    if label==5{original.add_epsilon(q as u32,target as u32,weight);}
                    else if label<5{original.add_transition(q as u32,if label==4{DEFAULT}else{label as i32},target as u32,weight);}
                }
            }
            let mut candidate=original.clone();let stats=restrict(&mut candidate,&certificate).unwrap();removed+=stats.live_edges_before-stats.live_edges_after;
            let mut words=vec![Vec::<u32>::new()];
            for depth in 0..=5{
                let mut following=Vec::new();for word in words{
                    if domain.accepts(&word){assert_eq!(eval(&original,&word),eval(&candidate,&word),"case {case} word {word:?}");}
                    if depth<5{for label in 0..4{let mut w=word.clone();w.push(label);following.push(w);}}
                }words=following;
            }
        }
        assert!(removed>0,"exercise real pruning");
    }
}
