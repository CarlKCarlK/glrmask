//! Query an already-certified component injected into the link's disjoint lexer.
//! Caller must own the same source-pinned injection proof as PreparedSourceSpan.
//! This helper is private: numeric offsets alone are not an injection certificate.
use super::completion_index::CompletionContext;
use glrmask_lexer::__private::automata::lexer::{Lexer,tokenizer::Tokenizer};
use glrmask_vocab::Vocab;
use glrmask_terminal_dwa::__private::terminal_dwa::scope::{BoundaryOwnership,ImmediateComponentId};

pub(crate) fn in_verified_span(context:&CompletionContext,offset:u32,merged:&Tokenizer,
    vocab:&Vocab,initial:&[bool],ownership:&BoundaryOwnership,start:ImmediateComponentId)->Option<Vec<bool>> {
    let n=merged.num_states()as usize;let local_n=context.raw_states();let offset=offset as usize;
    if vocab.is_empty()||vocab.entries_map().values().any(Vec::is_empty)
        ||initial.len()!=n||offset.checked_add(local_n)?>n
        ||ownership.num_terminals()!=merged.num_terminals()as usize
        ||start.0>=ownership.num_immediate_components()
        ||merged.has_virtual_residual_runtime(){return None;}
    let mut local=vec![false;local_n];let mut result=vec![false;n];
    let pure_at=|q:u32|merged.matched_terminals_iter(q).chain(merged.possible_future_terminals_iter(q))
        .all(|t|ownership.owner_of_terminal(t)==Some(start));
    for(q,&keep)in initial.iter().enumerate(){
        if !keep{continue;}
        if q<offset||q>=offset+local_n{return None;}
        // A foreign observation already at token entry is an event in the
        // original reference, even before a byte. Keep that source verbatim.
        let pure=if merged.state_has_epsilon_transitions(q as u32){
            merged.singleton_epsilon_closure(q as u32).iter().all(|q|pure_at(*q))
        }else{pure_at(q as u32)};
        if pure{local[q-offset]=true;}else{result[q]=true;}
    }
    let words=vocab.entries_map().values().cloned().collect::<Vec<_>>();
    let support=context.prefix_support(&local,&words).ok()?;
    for(q,keep)in support.into_iter().enumerate(){if keep{result[q+offset]=true;}}
    Some(result)
}

#[cfg(test)]mod tests{
 use super::*;use std::sync::Arc;
 use super::super::{prefix_observer::{PrefixObserver,Limits,Profile},completion_index::UntrustedCompletionIndex};
 use glrmask_lexer::__private::automata::lexer::{ast::{bytes,choice,plus,Expr},compile::{build_regex_monolithic,build_regex_partitioned}};
 use glrmask_terminal_dwa::__private::terminal_dwa::{scope,l1};
 fn prepared(tok:Arc<Tokenizer>,words:&[Vec<u8>])->CompletionContext{
  let p=PrefixObserver::prepare(tok.as_ref(),words,Limits::default(),&mut Profile::default()).unwrap().certify(tok.as_ref()).unwrap();
  let wire=UntrustedCompletionIndex::from_certified_prefix(&p).unwrap().to_bytes().unwrap();CompletionContext::load(tok,&wire).unwrap()
 }
 #[test]fn mixed_ownership_and_nested_epsilon_sources_match_original_support(){
  let exprs=vec![choice(vec![bytes(b"ab"),bytes(b"axc"),bytes(b"ca")]),plus(choice(vec![bytes(b"a"),bytes(b"bc")])),Expr::Epsilon];
  let a=build_regex_monolithic(&exprs).into_tokenizer(3,None);let b=build_regex_partitioned(&exprs,&[0,1,2]).into_tokenizer(3,None);
  let(c,_)=Tokenizer::disjoint_union_with_terminal_offsets(&[(&a,0),(&b,3)]);
  let mut seed=814u64;let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32)as usize};let mut cases=0;let mut impure=0;
  for tok in [a,b,c]{let tok=Arc::new(tok);
   let mut words=vec![b"a".to_vec(),b"bc".to_vec(),b"z".to_vec()];for _ in 0..64{words.push((0..1+next()%6).map(|_|b"abcxz"[next()%5]).collect());}words.sort();words.dedup();
   let ctx=prepared(Arc::clone(&tok),&words);let flat=l1::build_flat_transition_table(tok.as_ref());
   let offsets=(0..tok.num_terminals()).collect::<Vec<_>>();
   for shift in 0..2{
    let owners=offsets.iter().map(|q|ImmediateComponentId((q+shift)%2)).collect::<Vec<_>>();
    let ownership=BoundaryOwnership::from_leaf_layout(&offsets,tok.num_terminals(),&owners,2).unwrap();
    for owner in 0..2{for _ in 0..64{
      let mut entries=words.iter().enumerate().filter(|_|next()%3!=0).map(|(i,w)|(i as u32,w.clone())).collect::<Vec<_>>();if entries.is_empty(){entries.push((0,words[0].clone()));}
      let vocab=Vocab::new(entries);let initial=(0..tok.num_states()).map(|_|next()%3!=0).collect::<Vec<_>>();
      let result=in_verified_span(&ctx,0,&tok,&vocab,&initial,&ownership,ImmediateComponentId(owner)).unwrap();
      let(reference,_)=scope::crossing_prefix_seed_support(&tok,&vocab,&flat,&ownership,ImmediateComponentId(owner)).unwrap();
      let expected=reference.into_iter().zip(&initial).map(|(a,b)|a&&*b).collect::<Vec<_>>();assert_eq!(result,expected);
      impure+=result.iter().zip(&initial).filter(|(a,b)|**a&&**b).count();cases+=1;
    }}
   }
  }assert_eq!(cases,768);assert!(impure>0);
 }
 #[test]fn disjoint_component_injection_uses_offset_without_reset_or_foreign_confusion(){
  let a=Arc::new(build_regex_partitioned(&[bytes(b"aab"),plus(bytes(b"b"))],&[0,1]).into_tokenizer(2,None));
  let b=build_regex_monolithic(&[bytes(b"b!"),Expr::Epsilon]).into_tokenizer(2,None);
  let(merged,offsets)=Tokenizer::disjoint_union_with_terminal_offsets(&[(&b,0),(a.as_ref(),2)]);
  let words=vec![b"ab!".to_vec(),b"bb!".to_vec(),b"!".to_vec(),b"a".to_vec(),b"b".to_vec()];let ctx=prepared(Arc::clone(&a),&words);
  let vocab=Vocab::new(words.iter().enumerate().map(|(i,w)|(i as u32,w.clone())).collect());
  let ownership=BoundaryOwnership::flat(&[0,2],4).unwrap();let mut initial=vec![false;merged.num_states()as usize];
  for q in offsets[1]..offsets[1]+a.num_states(){initial[q as usize]=true;}
  let got=in_verified_span(&ctx,offsets[1],&merged,&vocab,&initial,&ownership,ImmediateComponentId(1)).unwrap();
  let(reference,_)=scope::crossing_prefix_seed_support(&merged,&vocab,&l1::build_flat_transition_table(&merged),&ownership,ImmediateComponentId(1)).unwrap();
  assert_eq!(got,reference.iter().zip(&initial).map(|(a,b)|*a&&*b).collect::<Vec<_>>());
  initial[0]=true;assert!(in_verified_span(&ctx,offsets[1],&merged,&vocab,&initial,&ownership,ImmediateComponentId(1)).is_none());initial[0]=false;
  assert!(in_verified_span(&ctx,u32::MAX,&merged,&vocab,&initial,&ownership,ImmediateComponentId(1)).is_none());
  let empty=Vocab::new(vec![(5,vec![])]);assert!(in_verified_span(&ctx,offsets[1],&merged,&empty,&initial,&ownership,ImmediateComponentId(1)).is_none());
  let unknown=Vocab::new(vec![(5,b"unknown".to_vec())]);assert!(in_verified_span(&ctx,offsets[1],&merged,&unknown,&initial,&ownership,ImmediateComponentId(1)).is_none());
 }
}
