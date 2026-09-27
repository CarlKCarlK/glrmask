//! Input-domain projection of an already compiled read-then-push template.
//!
//! At a final lexical endpoint, only executability of the last transfer is
//! observed. Its trailing writes cannot fail and no later read observes them.
//! Reuse the ordinary minimized template DFA: a negative-only path to a final
//! makes its source an admission final; retain reads and drop writes. This
//! avoids determinizing a new NFA and does not depend on terminal syntax.
use super::UnweightedDfa;
use crate::compiler::glr::labels::{DEFAULT_LABEL,is_negative_label};
use crate::automata::unweighted_u32::minimize_acyclic::minimize_acyclic;
use std::collections::VecDeque;

pub(super) fn project(template:&UnweightedDfa,alphabet:u32)->Option<UnweightedDfa>{
    let n=template.states.len();
    if n==0||n>50_000||template.start_state as usize>=n{return None;}
    let mut degree=vec![0usize;n];let mut edge_count=0usize;
    for row in &template.states{for (&label,&target) in &row.transitions{
        if target as usize>=n{return None;}
        if !is_negative_label(label)&&label!=DEFAULT_LABEL&&(label<0||label as u32>=alphabet){return None;}
        degree[target as usize]+=1;edge_count+=1;if edge_count>500_000{return None;}
    }}
    let mut queue=(0..n).filter(|&q|degree[q]==0).collect::<VecDeque<_>>();let mut order=Vec::with_capacity(n);
    while let Some(q)=queue.pop_front(){order.push(q);for &target in template.states[q].transitions.values(){
        degree[target as usize]-=1;if degree[target as usize]==0{queue.push_back(target as usize);}
    }}
    if order.len()!=n{return None;}
    let mut reached=vec![false;n];let mut pushed=vec![false;n];reached[template.start_state as usize]=true;
    for &q in &order{
        if !reached[q]{continue;}
        for (&label,&target) in &template.states[q].transitions{
            let write=is_negative_label(label);
            if pushed[q]&&!write{return None;}
            reached[target as usize]=true;pushed[target as usize]|=pushed[q]||write;
        }
    }
    let mut accepting=template.states.iter().map(|row|row.is_accepting).collect::<Vec<_>>();
    for &q in order.iter().rev(){for (&label,&target) in &template.states[q].transitions{
        if is_negative_label(label){accepting[q]|=accepting[target as usize];}
    }}
    let mut result=template.clone();
    for (q,row) in result.states.iter_mut().enumerate(){
        row.is_accepting=accepting[q];row.transitions.retain(|&label,_|!is_negative_label(label));
    }
    Some(minimize_acyclic(&result))
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::compiler::glr::labels::encode_negative_label as push;
    use std::collections::BTreeSet;
    fn projected_words(dfa:&UnweightedDfa)->BTreeSet<Vec<i32>>{
        let mut output=BTreeSet::new();let mut pending=vec![(dfa.start_state,Vec::new())];
        while let Some((q,word))=pending.pop(){
            let row=&dfa.states[q as usize];if row.is_accepting{output.insert(word.iter().copied().filter(|l|!is_negative_label(*l)).collect());}
            for (&label,&next) in &row.transitions{let mut w=word.clone();w.push(label);pending.push((next,w));}
        }output
    }
    #[test]
    fn exact_projection_matches_all_words_on_generated_read_push_dags(){
        let mut seed=237u64;let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32) as usize};
        for case in 0..256{
            let n=5+next()%6;let split=2+next()%(n-2);let mut dfa=UnweightedDfa::new();for _ in 1..n{dfa.add_state();}
            for q in 0..n{
                dfa.set_accepting(q as u32,next()%4==0);
                for label in 0..4{
                    if q+1>=n||next()%3==0{continue;}
                    let target=q+1+next()%(n-q-1);
                    let code=if q>=split{push(label)}else if label==3{DEFAULT_LABEL}else{label as i32};
                    dfa.add_transition(q as u32,code,target as u32);
                }
            }
            let projected=project(&dfa,4).unwrap();
            assert_eq!(projected_words(&dfa),projected_words(&projected),"case={case}");
            assert!(projected.states.iter().all(|row|row.transitions.keys().all(|&l|!is_negative_label(l))));
        }
    }
    #[test]
    fn preserves_empty_admission_and_declines_push_then_read(){
        let mut dfa=UnweightedDfa::new();let q=dfa.add_state();dfa.add_transition(0,push(1),q);dfa.set_accepting(q,true);
        let projected=project(&dfa,3).unwrap();assert!(projected.states[projected.start_state as usize].is_accepting);
        let r=dfa.add_state();dfa.add_transition(q,2,r);dfa.set_accepting(r,true);assert!(project(&dfa,3).is_none());
    }
}
