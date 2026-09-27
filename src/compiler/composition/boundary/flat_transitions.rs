//! Same physical lexer transition rows, allocated directly in their final Arc.
use crate::automata::lexer::tokenizer::{Lexer,Tokenizer};
use std::{mem::MaybeUninit,sync::Arc};
use rayon::prelude::*;
const MAX_WORDS:usize=16*1024*1024;
pub fn build(tokenizer:&Tokenizer,parallel:bool)->Option<Arc<[u32]>>{build_bounded(tokenizer,parallel,MAX_WORDS)}
fn build_bounded(tokenizer:&Tokenizer,parallel:bool,word_limit:usize)->Option<Arc<[u32]>>{
 if tokenizer.has_any_virtual_runtime(){return None;}
 let states=tokenizer.num_states()as usize;let len=states.checked_mul(256)?;if len>word_limit{return None;}
 let mut output=Arc::<[u32]>::new_uninit_slice(len);
 let storage=Arc::get_mut(&mut output).expect("new unshared allocation");
 let fill=|state:usize,row:&mut[MaybeUninit<u32>]|{
  // Do not form an initialized reference into uninitialized storage. The
  // ordinary API operates on an actually initialized stack buffer first.
  let mut values=[u32::MAX;256];tokenizer.fill_transition_row(state as u32,&mut values);
  assert_eq!(row.len(),values.len());for (dst,value)in row.iter_mut().zip(values){dst.write(value);}
 };
 if parallel&&states>=2048&&rayon::current_num_threads()>1{
  storage.par_chunks_mut(256).with_min_len(256).enumerate().for_each(|(state,row)|fill(state,row));
 }else{for (state,row)in storage.chunks_exact_mut(256).enumerate(){fill(state,row);}}
 // SAFETY: len is a checked multiple of 256. Both branches initialize every
 // row/element before returning; Rayon joins all disjoint row tasks. On panic,
 // this line is never reached and MaybeUninit storage needs no element drops.
 Some(unsafe{output.assume_init()})
}
#[cfg(test)]
fn reference(tokenizer:&Tokenizer)->Arc<[u32]>{
 let mut values=vec![0;tokenizer.num_states()as usize*256];
 for (state,row)in values.chunks_exact_mut(256).enumerate(){tokenizer.fill_transition_row(state as u32,row.try_into().unwrap());}
 Arc::from(values)
}
#[cfg(test)]mod tests{
 use super::*;
 use crate::automata::lexer::{ast::{bytes,choice,plus},compile::{build_regex_monolithic,build_regex_partitioned}};
 #[test]fn every_cell_matches_ordinary_physical_rows(){
  let mut rng=71605u64;let mut next=||{rng=rng.wrapping_mul(6364136223846793005).wrapping_add(1);(rng>>32)as usize};
  let pool=rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
  for _case in 0..128{let mut exprs=Vec::new();for _ in 0..12{let word=(0..1+next()%12).map(|_|next()as u8).collect::<Vec<_>>();exprs.push(choice(vec![bytes(&word),plus(bytes(&word[..1]))]));}
   for regex in [build_regex_monolithic(&exprs),build_regex_partitioned(&exprs,&(0..12).collect::<Vec<u32>>())]{
    let tokenizer=regex.into_tokenizer(12,Some(Arc::from(exprs.clone())));let expected=reference(&tokenizer);
    for parallel in [false,true]{let got=pool.install(||build(&tokenizer,parallel)).unwrap();assert_eq!(&*got,&*expected);assert_eq!(Arc::strong_count(&got),1);}
    assert!(build_bounded(&tokenizer,true,tokenizer.num_states()as usize*256-1).is_none());
   }
  }
 }
 #[test]fn actual_parallel_branch_writes_all_rows(){
  let exprs=(0..80).map(|i|{let mut word=vec![b'a'+(i%20)as u8;80];word[0]=i as u8;bytes(&word)}).collect::<Vec<_>>();let tokenizer=build_regex_partitioned(&exprs,&(0..80).collect::<Vec<u32>>()).into_tokenizer(80,Some(Arc::from(exprs)));
  assert!(tokenizer.num_states()>=2048,"fixture must execute parallel branch");let expected=reference(&tokenizer);
  for threads in [1,2,4]{let p=rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();assert_eq!(&*p.install(||build(&tokenizer,true)).unwrap(),&*expected);}
 }
}
