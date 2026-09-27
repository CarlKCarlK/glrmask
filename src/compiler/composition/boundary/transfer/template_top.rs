//! Finite first-read observations of an already compiled signed template.
//! No grammar syntax or literal vocabulary is inspected here.
use super::NWA;
#[cfg(test)] use super::Weight;
use crate::compiler::glr::labels::{DEFAULT_LABEL,is_negative_label,negative_to_positive_label};
use std::collections::{BTreeMap,BTreeSet,VecDeque};

pub(super) struct TopSummary {
    pub identity: bool,
    pub no_push_after_read: bool,
    pub first: BTreeMap<u32,Vec<u32>>,
    pub wildcard: Vec<u32>,
    pub unread_pushes: Vec<u32>,
}

fn initial_closure(nwa:&NWA)->Option<Vec<usize>>{
    let n=nwa.states().len();let mut seen=vec![false;n];let mut queue=VecDeque::new();
    for &q in nwa.start_states(){if q as usize>=n{return None;}queue.push_back(q as usize);}
    while let Some(q)=queue.pop_front(){if seen[q]{continue;}seen[q]=true;
        for &(target,ref w) in &nwa.states()[q].epsilons{if target as usize>=n{return None;}if !w.is_empty(){queue.push_back(target as usize);}}
    }
    Some(seen.into_iter().enumerate().filter_map(|(q,yes)|yes.then_some(q)).collect())
}

/// Last pushed state on a complete template path is the resulting top when
/// the signed word has shape READ* PUSH*. Unsupported interleavings decline.
pub(super) fn summarize(nwa:&NWA,alphabet:u32)->Option<TopSummary>{
    let n=nwa.states().len();if n==0||n>100_000{return None;}
    let first=initial_closure(nwa)?;
    let mut indegree=vec![0usize;n];
    for row in nwa.states(){for &(target,_) in row.epsilons.iter().chain(row.transitions.values().flatten()){
        if target as usize>=n{return None;}indegree[target as usize]+=1;
    }}
    let mut queue=(0..n).filter(|&q|indegree[q]==0).collect::<VecDeque<_>>();let mut order=Vec::with_capacity(n);
    while let Some(q)=queue.pop_front(){order.push(q);for &(target,_) in nwa.states()[q].epsilons.iter().chain(nwa.states()[q].transitions.values().flatten()){
        indegree[target as usize]-=1;if indegree[target as usize]==0{queue.push_back(target as usize);}
    }}
    if order.len()!=n{return None;}
    let mut after_push=vec![false;n];
    for &q in &order{
        let row=&nwa.states()[q];
        for &(target,ref w) in &row.epsilons{if !w.is_empty(){after_push[target as usize]|=after_push[q];}}
        for (&label,branches) in &row.transitions{
            if label!=DEFAULT_LABEL && !is_negative_label(label) && (label<0||label as u32>=alphabet){return None;}
            for &(target,ref w) in branches{if w.is_empty(){continue;}
                if after_push[q]&&!is_negative_label(label){return None;}
                after_push[target as usize]|=after_push[q]||is_negative_label(label);
            }
        }
    }
    let mut no_push=vec![false;n];let mut last=vec![BTreeSet::<u32>::new();n];let mut work=0usize;
    for &q in order.iter().rev(){
        let row=&nwa.states()[q];no_push[q]=row.final_weight.as_ref().is_some_and(|w|!w.is_empty());
        let mut output=BTreeSet::new();
        for &(target,ref w) in &row.epsilons{if !w.is_empty(){no_push[q]|=no_push[target as usize];output.extend(last[target as usize].iter().copied());}}
        for (&label,branches) in &row.transitions{for &(target,ref w) in branches{if w.is_empty(){continue;}
            output.extend(last[target as usize].iter().copied());
            if is_negative_label(label){if no_push[target as usize]{let top=negative_to_positive_label(label) as u32;if top>=alphabet{return None;}output.insert(top);}}
            else {no_push[q]|=no_push[target as usize];}
        }}
        work+=output.len();if work>2_000_000{return None;}last[q]=output;
    }
    let mut result=TopSummary{identity:false,no_push_after_read:false,first:BTreeMap::new(),wildcard:Vec::new(),unread_pushes:Vec::new()};
    let mut wildcard=BTreeSet::new();let mut unread=BTreeSet::new();
    for q in first{
        let row=&nwa.states()[q];result.identity|=row.final_weight.as_ref().is_some_and(|w|!w.is_empty());
        for (&label,branches) in &row.transitions{for &(target,ref w) in branches{if w.is_empty(){continue;}
            if is_negative_label(label){
                unread.extend(last[target as usize].iter().copied());
                if no_push[target as usize]{unread.insert(negative_to_positive_label(label) as u32);}
            }else{
                result.no_push_after_read|=no_push[target as usize];
                if label==DEFAULT_LABEL{wildcard.extend(last[target as usize].iter().copied());}
                else {result.first.entry(label as u32).or_default().extend(last[target as usize].iter().copied());}
            }
        }}
    }
    for row in result.first.values_mut(){row.sort_unstable();row.dedup();}
    result.wildcard=wildcard.into_iter().collect();result.unread_pushes=unread.into_iter().collect();Some(result)
}

impl TopSummary {
    pub(super) fn output(&self,input:&BTreeSet<u32>)->Option<BTreeSet<u32>>{
        if self.no_push_after_read{return None;}
        if input.is_empty(){return Some(BTreeSet::new());}
        let mut result=self.wildcard.iter().chain(&self.unread_pushes).copied().collect::<BTreeSet<_>>();
        if self.identity{result.extend(input.iter().copied());}
        for q in input{if let Some(out)=self.first.get(q){result.extend(out.iter().copied());}}
        Some(result)
    }
}

/// Restrict only the first concrete read of the *input* stack. Prefix epsilon
/// states are copied into a private unread phase; after the read, continuation
/// nodes are separate originals. Thus a node revisited later is not filtered
/// again. Negative-before-read templates decline rather than filtering a push.
pub(super) fn restrict_first(nwa:&NWA,input:&BTreeSet<u32>,alphabet:u32)->Option<NWA>{
    let prefix=initial_closure(nwa)?;let n=nwa.states().len();
    if n>100_000{return None;}
    for &q in &prefix{for (&label,branches) in &nwa.states()[q].transitions{
        if branches.iter().any(|(_,w)|!w.is_empty())&&is_negative_label(label){return None;}
    }}
    let mut forward=vec![false;n];let mut queue=VecDeque::new();
    for &q in &prefix{for (&label,branches) in &nwa.states()[q].transitions{
        if label==DEFAULT_LABEL || (label>=0&&(label as u32)>=alphabet) || (label>=0&&input.contains(&(label as u32))){
            for &(target,ref w) in branches{if !w.is_empty(){queue.push_back(target as usize);}}
        }
    }}
    while let Some(q)=queue.pop_front(){if q>=n{return None;}if forward[q]{continue;}forward[q]=true;
        queue.extend(nwa.states()[q].epsilons.iter().chain(nwa.states()[q].transitions.values().flatten()).filter_map(|(t,w)|(!w.is_empty()).then_some(*t as usize)));
    }
    let mut out=NWA::new(0,0);let mut first_map=vec![u32::MAX;n];let mut after_map=vec![u32::MAX;n];
    for &q in &prefix{first_map[q]=out.add_state();}
    for (q,&live) in forward.iter().enumerate(){if live{after_map[q]=out.add_state();}}
    out.set_start_states(nwa.start_states().iter().map(|&q|first_map[q as usize]).collect());
    for &q in &prefix{
        let source=first_map[q];let row=&nwa.states()[q];
        out.states_mut()[source as usize].final_weight=row.final_weight.clone();
        for &(target,ref w) in &row.epsilons{if !w.is_empty(){out.add_epsilon(source,first_map[target as usize],w.clone());}}
        for (&label,branches) in &row.transitions{
            // Retain empty keys as guards without inventing weighted edges.
            out.states_mut()[source as usize].transitions.entry(label).or_default();
            if label==DEFAULT_LABEL || (label>=0&&(label as u32)>=alphabet) || (label>=0&&input.contains(&(label as u32))){
                for &(target,ref w) in branches{if !w.is_empty(){out.add_transition(source,label,after_map[target as usize],w.clone());}}
            }
        }
    }
    for (q,&live) in forward.iter().enumerate(){if !live{continue;}let source=after_map[q];let row=&nwa.states()[q];
        out.states_mut()[source as usize].final_weight=row.final_weight.clone();
        for &(target,ref w) in &row.epsilons{if !w.is_empty(){out.add_epsilon(source,after_map[target as usize],w.clone());}}
        for (&label,branches) in &row.transitions{out.states_mut()[source as usize].transitions.entry(label).or_default();
            for &(target,ref w) in branches{if !w.is_empty(){out.add_transition(source,label,after_map[target as usize],w.clone());}}
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::compiler::glr::labels::encode_negative_label as push;
    #[test]
    fn first_read_is_not_reapplied_to_shared_continuations(){
        let mut nwa=NWA::new(0,0);for _ in 0..4{nwa.add_state();}nwa.set_start_states(vec![0]);
        nwa.add_epsilon(0,1,Weight::all());nwa.add_transition(0,0,1,Weight::all());
        nwa.add_transition(1,1,2,Weight::all());nwa.add_transition(2,push(2),3,Weight::all());nwa.set_final_weight(3,Weight::all());
        let narrowed=restrict_first(&nwa,&BTreeSet::from([0]),3).unwrap();
        let mut first=narrowed.start_states().to_vec();
        // A concrete first 0 is allowed, and the subsequent 1 must survive.
        let q=first.pop().unwrap();let t=narrowed.states()[q as usize].transitions[&0][0].0;
        assert!(!narrowed.states()[t as usize].transitions[&1].is_empty());
        let summary=summarize(&nwa,3).unwrap();assert_eq!(summary.output(&BTreeSet::from([0])).unwrap(),BTreeSet::from([2]));
    }
    #[test]
    fn pop_only_and_push_before_read_are_not_false_certificates(){
        let mut nwa=NWA::new(0,0);for _ in 0..3{nwa.add_state();}nwa.set_start_states(vec![0]);
        nwa.add_transition(0,0,1,Weight::all());nwa.set_final_weight(1,Weight::all());
        assert!(summarize(&nwa,3).unwrap().output(&BTreeSet::from([0])).is_none());
        nwa.add_transition(0,push(1),2,Weight::all());nwa.add_transition(2,1,1,Weight::all());
        assert!(summarize(&nwa,3).is_none());assert!(restrict_first(&nwa,&BTreeSet::from([0]),3).is_none());
    }
}
