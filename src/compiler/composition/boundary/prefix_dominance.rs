//! Remove terminal-word extensions already dominated by an accepting prefix.
//! This preserves existential weighted PREFIX acceptance, not complete-word
//! acceptance. A caller using parser transfers must separately prove that
//! executing an extension implies executable prefix and retain crossing policy.
use glrmask_weighted_automata::weighted_u32::dwa::{DWA,DWAState};
use glrmask_weight::__private::Weight;
use rustc_hash::FxHashMap;
#[derive(Debug,Default)]
pub struct Stats {pub before_states:usize,pub after_states:usize,pub before_edges:usize,pub after_edges:usize,pub changed_edges:usize,pub removed_edges:usize,pub unique_differences:usize}
pub fn reduce(input:&DWA)->Option<(DWA,Stats)> {
 let n=input.states().len();
 if n==0 || n>100_000 || input.start_state() as usize>=n{return None;}
 let mut stats=Stats{before_states:n,..Default::default()};
 let mut diff=FxHashMap::<(usize,usize),(Weight,Weight,Weight)>::default();
 let mut states=Vec::with_capacity(n);
 for source in input.states() {
  let mut out=DWAState::default();out.final_weight=source.final_weight.clone();
  for (label,target,weight) in source.transitions.entries() {
   if label<0 || label==i32::MAX-1 || target as usize>=n{return None;}
   stats.before_edges+=1;if stats.before_edges>2_000_000{return None;}
   let pruned=match source.final_weight.as_ref().filter(|f|!f.is_empty()) {
    Some(finals)=>{
     if weight.is_full()&&!finals.is_full(){return None;}
     let key=(weight.ptr_key(),finals.ptr_key());
     if !diff.contains_key(&key){
      if diff.len()>=250_000{return None;}
      let reduced=weight.difference(finals);
      diff.insert(key,(weight.clone(),finals.clone(),reduced));
     }
     diff[&key].2.clone()
    },None=>weight.clone()
   };
   if pruned!=*weight{stats.changed_edges+=1;}
   if pruned.is_empty(){stats.removed_edges+=1;continue;}
   out.transitions.insert(label,(target,pruned));
  }
  states.push(out);
 }
 stats.unique_differences=diff.len();
 let mut keep=vec![false;n];let mut todo=vec![input.start_state()];
 while let Some(q)=todo.pop(){if keep[q as usize]{continue;}keep[q as usize]=true;
  todo.extend(states[q as usize].transitions.values().map(|(target,_)|*target));}
 let mut map=vec![u32::MAX;n];let mut next=0;
 for (q,&live) in keep.iter().enumerate(){if live{map[q]=next;next+=1;}}
 let mut output=Vec::with_capacity(next as usize);
 for (q,mut state) in states.into_iter().enumerate(){if keep[q]{
  for (target,_) in state.transitions.values_mut(){*target=map[*target as usize];}
  stats.after_edges+=state.transitions.len();output.push(state);
 }}
 stats.after_states=output.len();
 Some((DWA::from_parts(output,map[input.start_state() as usize]),stats))
}
#[cfg(test)]
mod tests {
 use super::*;
 use glrmask_parser_dwa::__private::parser_equivalence::compare_parser_mask_prefix_languages;
 fn w(bits:usize)->Weight{Weight::from_token_set_for_tsid(0,(0..5u32).filter(|b|bits&(1usize<<b)!=0).collect())}
 #[test]
 fn generated_weighted_prefix_masks_match_exact_oracle(){
  let mut r=179u64;let mut next=||{r=r.wrapping_mul(6364136223846793005).wrapping_add(1);(r>>32) as usize};
  let mut changed=0;
  for case in 0..512 {
   let n=3+next()%10;let mut states=vec![DWAState::default();n];
   for q in 0..n {if next()%3!=0{states[q].final_weight=Some(w(next()%32));}
    for label in 0..4 {if q+1<n && next()%3!=0{states[q].transitions.insert(label,((q+1+next()%(n-q-1)) as u32,w(next()%32)));}}
   }
   let original=DWA::from_parts(states,0);let (candidate,stats)=reduce(&original).unwrap();changed+=usize::from(stats.changed_edges>0);
   let result=compare_parser_mask_prefix_languages(&original,&candidate,4,100_000).unwrap();
   assert!(result.difference.is_none(),"case={case} {:?}",result.difference);
  }
  assert!(changed>300,"fixtures must genuinely prune");
 }
 #[test]
 fn shared_coordinate_completion_dominates_only_its_own_extensions(){
  let mut states=vec![DWAState::default();3];
  states[0].final_weight=Some(w(1));states[0].transitions.insert(0,(1,w(3)));
  states[1].final_weight=Some(w(2));states[1].transitions.insert(1,(2,w(3)));states[2].final_weight=Some(w(3));
  let input=DWA::from_parts(states,0);let (out,stats)=reduce(&input).unwrap();
  assert_eq!(stats.changed_edges,2);assert_eq!(out.states()[0].transitions[&0].1,w(2));
  assert!(compare_parser_mask_prefix_languages(&input,&out,2,1000).unwrap().difference.is_none());
 }
 #[test]
 fn default_labels_and_symbolic_complements_decline(){
  let mut states=vec![DWAState::default();2];states[0].final_weight=Some(w(1));states[0].transitions.insert(0,(1,Weight::all()));
  assert!(reduce(&DWA::from_parts(states.clone(),0)).is_none());
  states[0].transitions.insert(i32::MAX-1,(1,w(1)));assert!(reduce(&DWA::from_parts(states,0)).is_none());
 }
}
