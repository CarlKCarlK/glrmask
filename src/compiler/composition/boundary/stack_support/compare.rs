//! Exact prefix-mask comparison on complete words of a supplied stack domain.
//!
//! Already-accepted coordinates are represented by an absorbing Yes state,
//! not forgotten at a prefix that is not yet a complete domain word. A No
//! state is absorbing failure. The product tracks both query residuals and
//! the deterministic domain subset, and unions visited coordinate regions.
//! Equality is checked only when the domain accepts a COMPLETE stack.
//!
//! This helper does not certify that the supplied domain covers real parser
//! stacks. That obligation belongs to the separately audited domain builder.

use std::collections::{BTreeSet,VecDeque};
use rustc_hash::FxHashMap;
use glrmask_weight::__private::Weight;
use glrmask_weighted_automata::weighted_u32::dwa::DWA;
use super::domain::StackDomain;

const DEFAULT:i32=2147483646;

#[derive(Clone,Copy,Debug,PartialEq,Eq,Hash)]
enum Query { Run(u32), Yes, No }
#[derive(Clone,Copy,Debug,PartialEq,Eq,Hash)]
struct Product { left:Query,right:Query,domain:u32 }

#[derive(Clone,Debug)]
pub struct Difference {pub stack:Vec<u32>,pub left:Weight,pub right:Weight}
#[derive(Default,Debug)]
pub struct Comparison {pub difference:Option<Difference>,pub products:usize,pub branches:usize,pub region_visits:usize,pub domain_subsets:usize}
#[derive(Clone,Copy)]
pub struct Budget {pub products:usize,pub branches:usize,pub domain_subsets:usize,pub regions:usize}
impl Default for Budget {
    fn default()->Self{Self{products:500_000,branches:20_000_000,domain_subsets:100_000,regions:2_000_000}}
}

struct Domains<'a> {
    source:&'a StackDomain,
    live:Vec<bool>,
    ids:FxHashMap<Vec<u32>,u32>,
    states:Vec<Vec<u32>>,
    transitions:FxHashMap<(u32,u32),Option<u32>>,
    limit:usize,
}
impl<'a> Domains<'a> {
    fn intern(&mut self,mut states:Vec<u32>)->Result<Option<u32>,String> {
        states.retain(|&q|self.live[q as usize]);
        if states.is_empty(){return Ok(None);}
        if let Some(&id)=self.ids.get(&states){return Ok(Some(id));}
        if self.states.len()>=self.limit{return Err("domain subset budget exhausted".into());}
        let id=self.states.len() as u32;self.ids.insert(states.clone(),id);self.states.push(states);Ok(Some(id))
    }
    fn next(&mut self,state:u32,label:u32)->Result<Option<u32>,String> {
        if let Some(&next)=self.transitions.get(&(state,label)){return Ok(next);}
        let next=self.source.step(&self.states[state as usize],label);
        let id=self.intern(next)?;self.transitions.insert((state,label),id);Ok(id)
    }
    fn accepting(&self,state:u32)->bool{self.states[state as usize].iter().any(|&q|self.source.nodes[q as usize].accepting)}
    fn labels(&self,state:u32)->BTreeSet<u32>{
        self.states[state as usize].iter().flat_map(|&q|self.source.nodes[q as usize].transitions.keys().copied()).collect()
    }
}

fn final_parts(dwa:&DWA,query:Query,region:&Weight)->Vec<(Query,Weight)> {
    let Query::Run(state)=query else{return vec![(query,region.clone())];};
    let Some(final_weight)=dwa.states()[state as usize].final_weight.as_ref() else{return vec![(query,region.clone())];};
    let accepted=region.intersection(final_weight);let remaining=region.difference(&accepted);
    let mut parts=Vec::new();
    if !accepted.is_empty(){parts.push((Query::Yes,accepted));}
    if !remaining.is_empty(){parts.push((query,remaining));}
    parts
}

fn advance_parts(dwa:&DWA,query:Query,region:&Weight,label:u32)->Vec<(Query,Weight)> {
    let Query::Run(state)=query else{return vec![(query,region.clone())];};
    let row=&dwa.states()[state as usize];
    let Some((target,weight))=row.transitions.get(&(label as i32)).or_else(||row.transitions.get(&DEFAULT)) else {
        return vec![(Query::No,region.clone())];
    };
    let continuing=region.intersection(weight);let failed=region.difference(&continuing);
    let mut parts=Vec::new();
    if !continuing.is_empty(){parts.push((Query::Run(*target),continuing));}
    if !failed.is_empty(){parts.push((Query::No,failed));}
    parts
}

fn evaluate(dwa:&DWA,stack:&[u32])->Weight {
    let mut state=dwa.start_state();let mut path=Weight::all();let mut accepted=Weight::empty();
    for position in 0..=stack.len(){
        let row=&dwa.states()[state as usize];
        if let Some(weight)=row.final_weight.as_ref(){accepted=accepted.union(&path.intersection(weight));}
        if position==stack.len(){break;}
        let Some((target,weight))=row.transitions.get(&(stack[position] as i32)).or_else(||row.transitions.get(&DEFAULT)) else{break;};
        path=path.intersection(weight);if path.is_empty(){break;}state=*target;
    }
    accepted
}

fn validate_query(dwa:&DWA,alphabet:u32)->Result<(),String>{
    if dwa.states().is_empty()||dwa.start_state() as usize>=dwa.states().len(){return Err("invalid mask query start".into());}
    for row in dwa.states(){
        for weight in row.final_weight.iter().chain(row.transitions.values().map(|(_,weight)|weight)){
            if !weight.is_full()&&weight.range_entries().any(|(_,high,_)|high==u32::MAX){return Err("reserved finite TSID in domain query".into());}
        }
        for (&label,(target,_)) in &row.transitions{
        if label!=DEFAULT&&(label<0||label as u32>=alphabet){return Err("mask query label outside domain alphabet".into());}
        if *target as usize>=dwa.states().len(){return Err("mask query target outside state set".into());}
    }}
    Ok(())
}

pub fn compare(left:&DWA,right:&DWA,domain:&StackDomain,budget:Budget)->Result<Comparison,String>{
    domain.validate()?;validate_query(left,domain.alphabet)?;validate_query(right,domain.alphabet)?;
    if budget.products==0||budget.domain_subsets==0||budget.regions==0{return Err("empty equivalence budget".into());}
    let mut domains=Domains{source:domain,live:domain.coaccessible(),ids:FxHashMap::default(),states:Vec::new(),transitions:FxHashMap::default(),limit:budget.domain_subsets};
    let initial=domain.closure([domain.start]);
    let Some(initial)=domains.intern(initial)? else{return Ok(Comparison::default());};
    let start=Product{left:Query::Run(left.start_state()),right:Query::Run(right.start_state()),domain:initial};
    // Weight::all() is an algebraic sentinel whose difference operation is
    // not a finite complement representation. The ordinary exact oracle
    // instead uses this single compressed rectangle over every valid raw
    // coordinate. No coordinate enumeration is performed.
    let universe=Weight::from_uniform(0..=u32::MAX-1,[0..=u32::MAX].into_iter().collect());
    if universe.is_full(){return Err("comparison universe collided with identity sentinel".into());}
    let mut pending=VecDeque::from([(start,universe.clone(),Vec::<u32>::new())]);
    let mut seen=FxHashMap::<Product,Weight>::from_iter([(start,universe)]);
    let mut result=Comparison::default();
    while let Some((current,region,stack))=pending.pop_front(){
        result.region_visits+=1;if result.region_visits>budget.regions{return Err("domain comparison region budget exhausted".into());}
        for (l,region) in final_parts(left,current.left,&region){
            for (r,region) in final_parts(right,current.right,&region){
                if l==r&&matches!(l,Query::Yes|Query::No){continue;}
                if domains.accepting(current.domain)&&(l==Query::Yes)!=(r==Query::Yes){
                    let difference=Difference{stack:stack.clone(),left:evaluate(left,&stack),right:evaluate(right,&stack)};
                    if difference.left==difference.right{return Err(format!("internal domain witness validation failed: stack={stack:?} statuses={l:?}/{r:?} region={region:?}"));}
                    result.difference=Some(difference);result.products=seen.len();result.domain_subsets=domains.states.len();return Ok(result);
                }
                // If neither residual has a default/absorbing-Yes branch,
                // every label absent on both sides fails identically forever.
                // Inspect only the union of explicit labels in that case.
                let has_default=|dwa:&DWA,q:Query|match q{
                    Query::Yes=>true,Query::No=>false,
                    Query::Run(q)=>dwa.states()[q as usize].transitions.contains_key(&DEFAULT),
                };
                let labels=if !has_default(left,l)&&!has_default(right,r){
                    let mut labels=BTreeSet::new();
                    for (dwa,query) in [(left,l),(right,r)]{if let Query::Run(q)=query{
                        labels.extend(dwa.states()[q as usize].transitions.keys().filter_map(|&label|
                            (label>=0&&(label as u32)<domain.alphabet).then_some(label as u32)));
                    }}labels
                }else{domains.labels(current.domain)};
                for label in labels{
                    result.branches+=1;if result.branches>budget.branches{return Err("domain comparison branch budget exhausted".into());}
                    let Some(next_domain)=domains.next(current.domain,label)? else{continue;};
                    for (nl,next_region) in advance_parts(left,l,&region,label){
                        for (nr,next_region) in advance_parts(right,r,&next_region,label){
                            if nl==nr&&matches!(nl,Query::Yes|Query::No){continue;}
                            let key=Product{left:nl,right:nr,domain:next_domain};
                            let unseen=seen.get(&key).map_or_else(||next_region.clone(),|old|next_region.difference(old));
                            if unseen.is_empty(){continue;}
                            if !seen.contains_key(&key)&&seen.len()>=budget.products{return Err("domain comparison product budget exhausted".into());}
                            seen.entry(key).and_modify(|old|*old=old.union(&unseen)).or_insert_with(||unseen.clone());
                            let mut word=stack.clone();word.push(label);pending.push_back((key,unseen,word));
                        }
                    }
                }
            }
        }
    }
    result.products=seen.len();result.domain_subsets=domains.states.len();Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glrmask_weighted_automata::weighted_u32::dwa::DWAState;
    fn weight(mask:u32)->Weight{Weight::from_token_set_for_tsid(0,(0..3).filter(|bit|mask&(1<<bit)!=0).collect())}
    #[test]
    fn early_prefix_disagreement_can_disappear_on_complete_domain_words(){
        let mut l=vec![DWAState::default();1];l[0].final_weight=Some(weight(1));
        let mut r=vec![DWAState::default();2];r[0].transitions.insert(0,(1,weight(1)));r[1].final_weight=Some(weight(1));
        let (left,right)=(DWA::from_parts(l,0),DWA::from_parts(r,0));
        let allowed=StackDomain::from_words(&[vec![0]],2).unwrap();
        assert!(compare(&left,&right,&allowed,Budget::default()).unwrap().difference.is_none());
        let wrong=StackDomain::from_words(&[vec![1,0]],2).unwrap();
        let d=compare(&left,&right,&wrong,Budget::default()).unwrap().difference.unwrap();assert_eq!(d.stack,vec![1,0]);
        let empty=StackDomain::from_words(&[vec![]],2).unwrap();assert!(compare(&left,&right,&empty,Budget::default()).unwrap().difference.is_some());
    }
    #[test]
    fn finite_domain_comparison_agrees_with_exhaustive_words(){
        let mut random=951u64;let mut next=||{random=random.wrapping_mul(6364136223846793005).wrapping_add(1);(random>>32) as usize};
        for case in 0..512 {
            let mut make=||{let n=2+next()%5;let mut states=vec![DWAState::default();n];
                for source in 0..n{if next()%2==0{states[source].final_weight=Some(weight((next()%8) as u32));}
                    for label in [0,1,2,DEFAULT]{if next()%3==0{states[source].transitions.insert(label,((next()%n) as u32,weight((next()%8) as u32)));}}
                }DWA::from_parts(states,0)};
            let left=make();let right=make();
            let words=(0..12).map(|_|(0..next()%5).map(|_|(next()%3) as u32).collect::<Vec<_>>()).collect::<Vec<_>>();
            let domain=StackDomain::from_words(&words,3).unwrap();
            let expected=words.iter().any(|word|evaluate(&left,word)!=evaluate(&right,word));
            let actual=compare(&left,&right,&domain,Budget::default()).unwrap();
            assert_eq!(actual.difference.is_some(),expected,"case={case}");
            if let Some(d)=actual.difference{assert!(domain.accepts(&d.stack));assert_ne!(evaluate(&left,&d.stack),evaluate(&right,&d.stack));}
        }
    }
    #[test]
    fn universal_domain_matches_existing_all_prefix_oracle(){
        use glrmask_parser_dwa::__private::parser_equivalence::compare_parser_mask_prefix_languages;
        let mut domain=StackDomain::from_words(&[vec![]],3).unwrap();
        for label in 0..3 {domain.nodes[0].transitions.entry(label).or_default().insert(0);}
        let mut random=601u64;let mut next=||{random=random.wrapping_mul(6364136223846793005).wrapping_add(1);(random>>32) as usize};
        for case in 0..128 {
            let mut make=||{let n=3+next()%4;let mut states=vec![DWAState::default();n];
                for source in 0..n{if next()%2==0{states[source].final_weight=Some(weight((next()%8) as u32));}
                    for label in [0,1,2,DEFAULT]{if next()%3==0{states[source].transitions.insert(label,((next()%n) as u32,weight((next()%8) as u32)));}}
                }DWA::from_parts(states,0)};
            let left=make();let right=make();
            let reference=compare_parser_mask_prefix_languages(&left,&right,3,100_000).unwrap();
            let actual=compare(&left,&right,&domain,Budget::default()).unwrap();
            assert_eq!(actual.difference.is_some(),reference.difference.is_some(),"case={case}");
        }
    }
    #[test]
    fn empty_override_stays_blocked_and_unknown_is_not_equality(){
        let mut states=vec![DWAState::default();2];states[0].transitions.insert(DEFAULT,(1,weight(1)));states[0].transitions.insert(0,(1,Weight::empty()));states[1].final_weight=Some(weight(1));
        let left=DWA::from_parts(states.clone(),0);states[0].transitions.remove(&0);let right=DWA::from_parts(states,0);
        let domain=StackDomain::from_words(&[vec![0]],2).unwrap();
        assert!(compare(&left,&right,&domain,Budget::default()).unwrap().difference.is_some());
        assert!(compare(&left,&right,&domain,Budget{products:0,..Default::default()}).is_err());
        assert!(compare(&left,&right,&domain,Budget{branches:0,..Default::default()}).is_err());
    }
}
