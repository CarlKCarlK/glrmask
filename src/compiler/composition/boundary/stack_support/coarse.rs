//! Inductive predecessor abstraction with conservative widening for deep pops.
//! This is a proof domain only, never a token-admission approximation.
use std::collections::{BTreeMap,BTreeSet,VecDeque};
use super::{effects::Effect,domain::{Node,StackDomain}};
#[derive(Clone,Default)]
struct Pred {all:bool,finite:BTreeSet<u32>}
#[derive(Debug,Default)]
pub struct Stats {pub all_rows:usize,pub finite_edges:usize,pub work:usize,pub widened:usize,pub effects:usize}
/// Issued only after all effect inclusions and the initial stack are checked.
/// This module and its siblings are private to the static boundary compiler.
pub struct Certificate {pub(super) domain:StackDomain,pub(super) stats:Stats}

pub fn certify(initial:&[u32],effects:&[Effect],alphabet:u32)->Result<Certificate,String>{
    let n=alphabet as usize;if n==0||n>50_000{return Err("coarse alphabet budget".into());}
    if initial.iter().any(|&q|q>=alphabet){return Err("invalid initial word".into());}
    let mut pred=vec![Pred::default();n];let mut dependents=vec![BTreeSet::new();n];let mut stats=Stats::default();
    for pair in initial.windows(2){pred[pair[0] as usize].finite.insert(pair[1]);}
    if let Some(&q)=initial.last(){pred[q as usize].finite.insert(alphabet);}
    for e in effects{
        if e.source>=alphabet||e.pushes.iter().any(|&q|q>=alphabet){return Err("effect outside alphabet".into());}
        for pair in e.pushes.windows(2){pred[pair[1] as usize].finite.insert(pair[0]);}
        if let Some(&low)=e.pushes.first(){match e.pop{
            0=>{pred[low as usize].finite.insert(e.source);},
            1=>{dependents[e.source as usize].insert(low as usize);},
            _=>{pred[low as usize].all=true;},
        }}
    }
    for p in &mut pred{if p.all{p.finite.clear();}}
    let mut queue=(0..n).collect::<VecDeque<_>>();let mut queued=vec![true;n];
    while let Some(q)=queue.pop_front(){queued[q]=false;let source=pred[q].clone();
        for &target in &dependents[q]{
            stats.work+=source.finite.len().max(1);if stats.work>100_000_000{return Err("coarse propagation budget".into());}
            if pred[target].all{continue;}
            let before=pred[target].finite.len();
            if source.all{pred[target].all=true;pred[target].finite.clear();}
            else{
                pred[target].finite.extend(source.finite.iter().copied());
                if pred[target].finite.len()>1024{pred[target].all=true;pred[target].finite.clear();stats.widened+=1;}
            }
            if pred[target].all||before!=pred[target].finite.len(){if !queued[target]{queued[target]=true;queue.push_back(target);}}
        }
    }
    // Verify the full effect closure against the completed abstract relation.
    // Pop-only exposes a suffix, which is accepted by this language, including
    // the deliberately permitted empty word. No reachability assumption here.
    let contains=|q:u32,p:u32|pred[q as usize].all||pred[q as usize].finite.contains(&p);
    for e in effects{
        for pair in e.pushes.windows(2){if !contains(pair[1],pair[0]){return Err("internal push certificate failed".into());}}
        if let Some(&low)=e.pushes.first(){match e.pop{
            0=>if !contains(low,e.source){return Err("zero-pop certificate failed".into());},
            1=>if !pred[low as usize].all&&(pred[e.source as usize].all||!pred[e.source as usize].finite.is_subset(&pred[low as usize].finite)){
                return Err("one-pop certificate failed".into());
            },
            _=>if !pred[low as usize].all{return Err("deep-pop conservative certificate failed".into());},
        }}
    }
    // Equal predecessor rows have identical suffix languages. ALL rows
    // equal the root's free-first-state language; merge them directly instead
    // of epsilon-closing a separate alias for every parser symbol.
    let mut rows=BTreeMap::<BTreeSet<u32>,u32>::new();
    let mut classes=vec![0u32;n];
    for (q,row) in pred.iter().enumerate(){
        if row.all{stats.all_rows+=1;continue;}
        let next=rows.len() as u32+1;classes[q]=*rows.entry(row.finite.clone()).or_insert(next);
        stats.finite_edges+=row.finite.iter().filter(|&&p|p!=alphabet).count();
    }
    let mut nodes=vec![Node::default();rows.len()+1];nodes[0].accepting=true;
    for q in 0..n{nodes[0].transitions.insert(q as u32,BTreeSet::from([classes[q]]));}
    for (row,id) in rows{
        nodes[id as usize].accepting=row.contains(&alphabet);
        for p in row{if p!=alphabet{nodes[id as usize].transitions.insert(p,BTreeSet::from([classes[p as usize]]));}}
    }
    if stats.finite_edges>4_000_000{return Err("coarse domain edge budget".into());}
    stats.effects=effects.len();let domain=StackDomain{nodes,start:0,alphabet};domain.validate()?;
    if !domain.accepts(initial){return Err("initial stack omitted".into());}
    Ok(Certificate{domain,stats})
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn low_pop_relations_remain_precise_while_deep_pops_widen(){
        let effects=vec![Effect{source:0,pop:0,pushes:vec![1]},Effect{source:1,pop:1,pushes:vec![2]},Effect{source:2,pop:2,pushes:vec![3]}];
        let c=certify(&[0],&effects,4).unwrap();assert!(c.domain.accepts(&[2,0]));assert!(!c.domain.accepts(&[2,1,0]));
        assert!(c.domain.accepts(&[3]));assert!(!c.domain.accepts(&[0,0]));
    }
    #[test]
    fn exhaustive_small_effect_closure_checks_widening_soundness(){
        let mut seed=41u64;let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32) as usize};
        for case in 0..128{
            let effects=(0..10).map(|_|{let len=next()%4;Effect{source:(next()%4) as u32,pop:next()%4,pushes:(0..len).map(|_|(next()%4) as u32).collect()}}).collect::<Vec<_>>();
            let c=certify(&[0],&effects,4).unwrap();let mut words=vec![Vec::<u32>::new()];
            for depth in 0..=5{let mut following=Vec::new();for word in words{
                if c.domain.accepts(&word){for e in &effects{if word.first()==Some(&e.source)&&e.pop<=word.len(){
                    let mut out=e.pushes.iter().rev().copied().collect::<Vec<_>>();out.extend_from_slice(&word[e.pop..]);
                    assert!(c.domain.accepts(&out),"case={case} {word:?} {e:?} -> {out:?}");
                }}}
                if depth<5{for q in 0..4{let mut w=word.clone();w.push(q);following.push(w);}}
            }words=following;}
        }
    }
}
