//! First-segment completion observations; never a quotient of reset states.
use glrmask_lexer::__private::automata::lexer::{Lexer,tokenizer::Tokenizer};
use glrmask_terminal_dwa::__private::terminal_dwa::scope::BoundaryAnalysisScope;
use glrmask_vocab::Vocab;
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;

#[derive(Default)]struct Node{edges:BTreeMap<u8,u32>}
#[derive(Clone)]enum Frontier{One(u32),Many(Vec<u32>)}
impl Frontier{fn states(&self)->&[u32]{match self{Self::One(q)=>std::slice::from_ref(q),Self::Many(v)=>v}}}
#[derive(Debug)]pub(crate) struct FirstRefinement{pub original_to_rep:Vec<u32>,pub representatives:Vec<bool>,pub before:usize,pub after:usize,pub steps:usize,pub prefixes:usize,pub signature_cells:usize}

pub(crate) fn prepare(tok:&Tokenizer,vocab:&Vocab,scope:&BoundaryAnalysisScope,flat:&[u32])->Option<FirstRefinement>{
 const MAX_STEPS:usize=16_000_000;const MAX_CELLS:usize=4_000_000;
 let n=tok.num_states()as usize;
 if !scope.require_crossing()||n==0||n>200_000||tok.has_virtual_residual_runtime()
  ||vocab.is_empty()||vocab.len()>4096||vocab.entries_map().values().any(Vec::is_empty)
  ||scope.initial_states().keep_raw().len()!=n||flat.len()!=n.checked_mul(256)? {return None;}
 let mut trie=vec![Node::default()];
 for word in vocab.entries_map().values(){let mut node=0;for &b in word{
  let next=if let Some(&q)=trie[node].edges.get(&b){q}else{if trie.len()>=65536{return None;}let q=trie.len()as u32;trie.push(Node::default());trie[node].edges.insert(b,q);q};node=next as usize;
 }}
 let mut outputs=FxHashMap::<Vec<u32>,u32>::default();outputs.insert(vec![],0);
 let mut raw_outputs=vec![u32::MAX;n];let mut classes=FxHashMap::<Vec<(u32,u32)>,u32>::default();
 let mut original_to_rep=(0..n as u32).collect::<Vec<_>>();let mut representatives=vec![false;n];let mut before=0;let mut after=0;let mut steps=0usize;let mut signature_cells=0usize;
 let observe=|states:&[u32],outputs:&mut FxHashMap<Vec<u32>,u32>,raw_outputs:&mut[u32]|->Option<u32>{
  if states.len()==1&&raw_outputs[states[0]as usize]!=u32::MAX{return Some(raw_outputs[states[0]as usize]);}
  let mut labels=states.iter().flat_map(|q|tok.matched_terminals_iter(*q)).collect::<Vec<_>>();labels.sort_unstable();labels.dedup();
  let id=if let Some(&id)=outputs.get(&labels){id}else{if outputs.len()>=65536{return None;}let id=outputs.len()as u32;outputs.insert(labels,id);id};
  if states.len()==1{raw_outputs[states[0]as usize]=id;}Some(id)
 };
 for(q,&keep)in scope.initial_states().keep_raw().iter().enumerate(){if !keep{continue;}before+=1;
  let closure=tok.singleton_epsilon_closure(q as u32).into_vec();
  // A first partial terminal owned elsewhere would itself constitute a
  // crossing. Preserve that source individually rather than erase its future.
  let pure=closure.iter().all(|q|tok.matched_terminals_iter(*q).chain(tok.possible_future_terminals_iter(*q)).all(|t|scope.ownership().owner_of_terminal(t)==Some(scope.start_component())));
  if !pure{representatives[q]=true;after+=1;continue;}
  let initial=if closure.len()==1{Frontier::One(closure[0])}else{Frontier::Many(closure)};
  let mut stack=vec![(0u32,initial)];let mut signature=Vec::new();
  while let Some((node,frontier))=stack.pop(){let states=frontier.states();let output=observe(states,&mut outputs,&mut raw_outputs)?;if output!=0{signature.push((node,output));}
   for (&b,&child)in trie[node as usize].edges.iter().rev(){
    steps=steps.checked_add(states.len())?;if steps>MAX_STEPS{return None;}
    let next=match frontier{
     Frontier::One(q) if !tok.state_has_epsilon_transitions(q)=>{
      let target=flat[q as usize*256+b as usize];if target==u32::MAX{continue;}
      if tok.state_has_epsilon_transitions(target){Frontier::Many(tok.singleton_epsilon_closure(target).into_vec())}else{Frontier::One(target)}
     },
     _=>{let v=tok.step_all(states,b).into_vec();if v.is_empty(){continue;}if v.len()==1{Frontier::One(v[0])}else{Frontier::Many(v)}}
    };stack.push((child,next));
   }
  }
  // Prefix-node IDs make the completed-terminal trace exact even when two
  // token spellings reach the same residual. Full keys are compared on hash hits.
  signature.sort_unstable();signature_cells=signature_cells.checked_add(signature.len())?;if signature_cells>MAX_CELLS{return None;}
  let rep=if let Some(&r)=classes.get(&signature){r}else{let r=q as u32;classes.insert(signature,r);representatives[q]=true;after+=1;r};original_to_rep[q]=rep;
 }
 Some(FirstRefinement{original_to_rep,representatives,before,after,steps,prefixes:trie.len(),signature_cells})
}

#[cfg(test)]mod tests{
 use super::*;use std::sync::Arc;
 use glrmask_terminal_dwa::__private::terminal_dwa::scope::ImmediateComponentId;
 use glrmask_lexer::__private::automata::lexer::{ast::{bytes,choice,plus},compile::{build_regex_monolithic,build_regex_partitioned}};
 use glrmask_terminal_dwa::__private::terminal_dwa::{scope::{InitialStateDomain,BoundaryOwnership},l1};
 #[test]fn every_equated_seed_has_identical_completion_trace(){
  let exprs=vec![choice(vec![bytes(b"abcd"),bytes(b"xycd"),bytes(b"abd")]),plus(bytes(b" ")),plus(choice(vec![bytes(b"a"),bytes(b"ab")]))];
  let mut random=22u64;let mut next=||{random=random.wrapping_mul(6364136223846793005).wrapping_add(1);(random>>32)as usize};let mut merged=0;
  for partitioned in [false,true]{let tok=if partitioned{build_regex_partitioned(&exprs,&[0,1,2])}else{build_regex_monolithic(&exprs)}.into_tokenizer(3,None);let flat=l1::build_flat_transition_table(&tok);
   for case in 0..128{let mut words=vec![b"cd!".to_vec(),b"d".to_vec(),b" !".to_vec()];for _ in 0..10{let len=1+next()%5;words.push((0..len).map(|_|b"abcdxy !"[next()%8]).collect());}let vocab=Vocab::new(words.iter().enumerate().map(|(i,w)|(i as u32,w.clone())).collect());
    let mut seeds=(0..tok.num_states()).map(|_|next()%3!=0).collect::<Vec<_>>();seeds[0]=true;
    let scope=BoundaryAnalysisScope::new(InitialStateDomain::from_mask(seeds.len(),seeds.clone()).unwrap(),tok.deterministic_reset_states().into_vec(),Arc::new(BoundaryOwnership::flat(&[0],3).unwrap()),ImmediateComponentId(0),true,None).unwrap();let p=prepare(&tok,&vocab,&scope,&flat).unwrap();merged+=p.before-p.after;
    for(q,&keep)in seeds.iter().enumerate(){if !keep{continue}let r=p.original_to_rep[q];assert!(p.representatives[r as usize]);for w in &words{for end in 0..=w.len(){let(a,ma)=tok.execute_summary_from_state(&w[..end],q as u32);let(b,mb)=tok.execute_summary_from_state(&w[..end],r);assert_eq!(ma,mb,"case{case} q{q} r{r} {w:?}");let _=(a,b);}}}
   }
  }assert!(merged>0);
 }
}
