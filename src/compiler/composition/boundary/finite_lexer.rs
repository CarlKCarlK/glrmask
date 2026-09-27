//! Vocabulary-relative observation synthesis, not a replacement runtime lexer.
//!
//! A context is an acyclic accept-all-prefixes byte automaton. Its root admits
//! either token prefixes (first segment) or token factors (reset segments).
//! The product with exact epsilon-closed lexer frontiers is minimized by full
//! observation/edge signatures. Context is part of the key: bounded trace
//! equality is never used as an unlayered raw-state congruence.

use glrmask_lexer::__private::automata::lexer::{Lexer, tokenizer::Tokenizer};
use glrmask_terminal_dwa::__private::terminal_dwa::scope::{BoundaryOwnership, ImmediateComponentId};
use glrmask_vocab::Vocab;
use rustc_hash::FxHashMap;
use std::{collections::BTreeMap, time::Instant};

const MAX_STEPS: usize = 16_000_000;
const MAX_MEMBERS: usize = 4_000_000;
const MAX_PRODUCTS: usize = 1_000_000;
const MAX_ROWS: usize = 100_000;

#[derive(Default)]
struct Trie { edges: BTreeMap<u8, usize> }

#[derive(Default)]
struct Contexts {
    rows: Vec<Vec<(u8, u32)>>,
    ids: FxHashMap<Vec<(u8, u32)>, u32>,
}
impl Contexts {
    fn build(&mut self, words: &[Vec<u8>], factors: bool) -> Option<u32> {
        let mut trie = vec![Trie::default()];
        let mut work = 0usize;
        for word in words {
            for begin in 0..if factors { word.len().max(1) } else { 1 } {
                let mut state = 0;
                for &byte in &word[begin..] {
                    work = work.checked_add(1)?;
                    if work > MAX_STEPS { return None; }
                    state = match trie[state].edges.get(&byte).copied() {
                        Some(next) => next,
                        None => {
                            if trie.len() >= MAX_ROWS { return None; }
                            let next = trie.len();
                            trie.push(Trie::default());
                            trie[state].edges.insert(byte, next);
                            next
                        }
                    };
                }
            }
        }
        let mut remap = vec![0u32; trie.len()];
        for old in (0..trie.len()).rev() {
            let row: Vec<_> = trie[old].edges.iter().map(|(&b,&q)| (b,remap[q])).collect();
            remap[old] = if let Some(&id) = self.ids.get(&row) { id } else {
                let id = self.rows.len() as u32;
                self.rows.push(row.clone()); self.ids.insert(row,id); id
            };
        }
        Some(remap[0])
    }
}

#[derive(Default)]
struct Labels { rows: Vec<Vec<usize>>, ids: FxHashMap<Vec<usize>,u32>, members: usize }
impl Labels {
    fn intern(&mut self, mut labels: Vec<usize>) -> Option<u32> {
        labels.sort_unstable(); labels.dedup();
        if let Some(&id)=self.ids.get(&labels) { return Some(id); }
        self.members=self.members.checked_add(labels.len())?;
        if self.members>MAX_MEMBERS || self.rows.len()>=MAX_ROWS {return None;}
        let id=self.rows.len()as u32;
        self.rows.push(labels.clone());self.ids.insert(labels,id);Some(id)
    }
}

struct Frontier { raw:Vec<u32>, matched:u32, future:u32 }
#[derive(Clone,Hash,Eq,PartialEq)]
struct Row { matched:u32, future:u32, edges:Vec<(u8,u32)> }
struct Builder<'a> {
    source:&'a Tokenizer,
    contexts:Contexts,
    labels:Labels,
    frontiers:Vec<Frontier>,
    frontier_ids:FxHashMap<Vec<u32>,u32>,
    steps:FxHashMap<(u32,u8),u32>,
    rows:Vec<Row>,
    row_ids:FxHashMap<Row,u32>,
    products:FxHashMap<(u32,u32,bool),u32>,
    raw_work:usize,
    raw_members:usize,
    edge_count:usize,
}
impl<'a> Builder<'a> {
    fn new(source:&'a Tokenizer,contexts:Contexts)->Option<Self> {
        let mut labels=Labels::default();assert_eq!(labels.intern(vec![])?,0);
        let dead=Row{matched:0,future:0,edges:vec![]};
        let mut row_ids=FxHashMap::default();row_ids.insert(dead.clone(),0);
        let mut frontier_ids=FxHashMap::default();frontier_ids.insert(vec![],0);
        Some(Self{source,contexts,labels,frontiers:vec![Frontier{raw:vec![],matched:0,future:0}],
            frontier_ids,steps:FxHashMap::default(),rows:vec![dead],row_ids,
            products:FxHashMap::default(),raw_work:0,raw_members:0,edge_count:0})
    }
    fn frontier(&mut self,mut raw:Vec<u32>)->Option<u32> {
        raw.sort_unstable();raw.dedup();
        if let Some(&id)=self.frontier_ids.get(&raw){return Some(id);}
        if self.frontiers.len()>=MAX_ROWS || raw.iter().any(|&q|q>=self.source.num_states()) {return None;}
        self.raw_members=self.raw_members.checked_add(raw.len())?;
        if self.raw_members>MAX_MEMBERS{return None;}
        let matched=self.labels.intern(raw.iter().flat_map(|&q|self.source.matched_terminals_iter(q)).map(|t|t as usize).collect())?;
        let future=self.labels.intern(raw.iter().flat_map(|&q|self.source.possible_future_terminals_iter(q)).map(|t|t as usize).collect())?;
        let id=self.frontiers.len()as u32;
        self.frontier_ids.insert(raw.clone(),id);self.frontiers.push(Frontier{raw,matched,future});Some(id)
    }
    fn step(&mut self,frontier:u32,byte:u8)->Option<u32>{
        if frontier==0{return Some(0);}
        if let Some(&id)=self.steps.get(&(frontier,byte)){return Some(id);}
        self.raw_work=self.raw_work.checked_add(self.frontiers[frontier as usize].raw.len())?;
        if self.raw_work>MAX_STEPS || self.steps.len()>=MAX_PRODUCTS{return None;}
        let raw=self.source.step_all(&self.frontiers[frontier as usize].raw,byte).into_vec();
        let id=self.frontier(raw)?;self.steps.insert((frontier,byte),id);Some(id)
    }
    fn product(&mut self,frontier:u32,context:u32,completion_only:bool)->Option<u32>{
        if frontier==0{return Some(0);}
        let key=(frontier,context,completion_only);
        if let Some(&q)=self.products.get(&key){return Some(q);}
        if self.products.len()>=MAX_PRODUCTS{return None;}
        let mut edges=Vec::new();
        let context_edges=self.contexts.rows[context as usize].clone();
        for (byte,next_context)in context_edges{
            let next=self.step(frontier,byte)?;
            if next!=0{
                let target=self.product(next,next_context,completion_only)?;
                if target!=0{edges.push((byte,target));}
            }
        }
        let matched=self.frontiers[frontier as usize].matched;
        let future=if completion_only{
            // This role can contribute to a crossing only after completing
            // a local terminal. Keep exactly the possible completions within
            // the permitted finite prefix language. Reset-role futures are
            // never truncated this way.
            let mut groups=Vec::new();
            for &(_,q)in &edges{
                let node=&self.rows[q as usize];
                groups.extend_from_slice(&self.labels.rows[node.matched as usize]);
                groups.extend_from_slice(&self.labels.rows[node.future as usize]);
                if groups.len()>MAX_MEMBERS{return None;}
            }
            self.labels.intern(groups)?
        }else{self.frontiers[frontier as usize].future};
        let row=Row{matched,future,edges};
        let id=if let Some(&id)=self.row_ids.get(&row){id}else{
            self.edge_count=self.edge_count.checked_add(row.edges.len())?;
            if self.edge_count>MAX_PRODUCTS || self.rows.len()>=MAX_ROWS{return None;}
            let id=self.rows.len()as u32;self.rows.push(row.clone());self.row_ids.insert(row,id);id
        };
        self.products.insert(key,id);Some(id)
    }
}

#[derive(Debug,Default)]
pub struct Profile {
    pub source_states:usize,pub source_seeds:usize,pub states:usize,pub edges:usize,
    pub contexts:usize,pub frontiers:usize,pub products:usize,pub raw_work:usize,
    pub pure_seeds:usize,pub build_ms:f64,pub materialize_ms:f64,
}
pub struct Synthesized {
    pub tokenizer:Tokenizer,
    pub original_to_synth:Vec<u32>,
    pub seeds:Vec<bool>,
    pub profile:Profile,
}

pub fn prepare(source:&Tokenizer,vocab:&Vocab,seeds:&[bool],owners:&BoundaryOwnership,
    start:ImmediateComponentId,completion_only:bool)->Option<Synthesized>{
    let clock=Instant::now();let n=source.num_states()as usize;
    if n==0||n>200_000||seeds.len()!=n||source.has_virtual_residual_runtime()
        ||vocab.is_empty()||vocab.len()>4096||vocab.entries_map().values().any(|w|w.is_empty()||w.len()>256){return None;}
    let words=vocab.entries_map().values().cloned().collect::<Vec<_>>();
    let mut contexts=Contexts::default();let first_context=contexts.build(&words,false)?;
    let reset_context=contexts.build(&words,true)?;
    let mut b=Builder::new(source,contexts)?;
    let mut reset_raw=Vec::new();
    for q in source.deterministic_reset_states(){reset_raw.extend_from_slice(&source.singleton_epsilon_closure(q));}
    let reset_frontier=b.frontier(reset_raw)?;
    let reset=b.product(reset_frontier,reset_context,false)?;
    let mut original_to_synth=vec![u32::MAX;n];let mut pure_seeds=0;
    for(q,&keep)in seeds.iter().enumerate(){if !keep{continue;}
        let raw=source.singleton_epsilon_closure(q as u32).into_vec();
        let pure=raw.iter().all(|&q|source.matched_terminals_iter(q).chain(source.possible_future_terminals_iter(q))
            .all(|t|owners.owner_of_terminal(t)==Some(start)));
        pure_seeds+=usize::from(pure);
        let frontier=b.frontier(raw)?;
        original_to_synth[q]=b.product(frontier,first_context,completion_only&&pure)?;
    }
    if b.rows.len().checked_mul((source.num_terminals()as usize).div_ceil(64))?.checked_mul(2)?>16_000_000{return None;}
    let mut finals=Vec::new();let mut futures=Vec::new();let mut offsets=vec![0];let mut edges=Vec::new();
    for row in &b.rows{
        finals.push(b.labels.rows[row.matched as usize].clone());
        futures.push(b.labels.rows[row.future as usize].clone());
        edges.extend_from_slice(&row.edges);offsets.push(edges.len()as u32);
    }
    let build_ms=clock.elapsed().as_secs_f64()*1000.0;
    let now=Instant::now();
    let(tokenizer,rebase)=source.materialize_deterministic_view(reset as usize,&finals,&futures,&offsets,&edges,&vec![true;source.num_terminals()as usize])?;
    let mut selected=vec![false;tokenizer.num_states()as usize];
    for q in &mut original_to_synth{if *q!=u32::MAX{*q=rebase[*q as usize];selected[*q as usize]=true;}}
    let profile=Profile{source_states:n,source_seeds:seeds.iter().filter(|&&s|s).count(),states:tokenizer.num_states()as usize,
        edges:edges.len(),contexts:b.contexts.rows.len(),frontiers:b.frontiers.len(),products:b.products.len(),raw_work:b.raw_work,
        pure_seeds,build_ms,materialize_ms:now.elapsed().as_secs_f64()*1000.0};
    Some(Synthesized{tokenizer,original_to_synth,seeds:selected,profile})
}

#[cfg(test)]mod tests{
    use super::*;
    use glrmask_lexer::__private::automata::lexer::{ast::{bytes,choice,plus,Expr},compile::{build_regex_monolithic,build_regex_partitioned}};
    fn observations(tok:&Tokenizer,raw:u32,word:&[u8])->(Vec<(u32,usize)>,Vec<u32>){
        let(end,mut matched)=tok.execute_summary_from_state(word,raw);matched.sort_unstable();matched.dedup();
        let mut future=end.iter().flat_map(|&q|tok.possible_future_terminals_iter(q)).collect::<Vec<_>>();future.sort_unstable();future.dedup();(matched.into_vec(),future)
    }
    #[test]fn generated_context_products_preserve_all_required_observations(){
        let exprs=vec![choice(vec![bytes(b"abcdef"),bytes(b"xydef"),bytes(b"ab")]),plus(choice(vec![bytes(b"a"),bytes(b"bc")])),Expr::Epsilon,bytes(b"!c")];
        let a=build_regex_monolithic(&exprs).into_tokenizer(4,None);let b=build_regex_partitioned(&exprs,&[0,1,2,3]).into_tokenizer(4,None);
        let(c,_)=Tokenizer::disjoint_union_with_terminal_offsets(&[(&a,0),(&b,4)]);
        let mut rng=18u64;let mut next=||{rng=rng.wrapping_mul(6364136223846793005).wrapping_add(1);(rng>>32)as usize};
        for source in [a,b,c]{let owners=BoundaryOwnership::flat(&[0,source.num_terminals()/2],source.num_terminals()).unwrap();
            for _ in 0..64{
                let mut words=vec![b"def!".to_vec(),b"ab!c".to_vec()];for _ in 0..8{let n=1+next()%6;words.push((0..n).map(|_|b"abcxydef!"[next()%9]).collect());}
                let vocab=Vocab::new(words.iter().enumerate().map(|(i,w)|((i*7)as u32,w.clone())).collect());
                let seeds=(0..source.num_states()).map(|_|next()%3!=0).collect::<Vec<_>>();
                for completion in [false,true]{let built=prepare(&source,&vocab,&seeds,&owners,ImmediateComponentId(0),completion).unwrap();
                    for(q,&keep)in seeds.iter().enumerate(){if !keep{continue;}
                        for word in &words{for end in 0..=word.len(){
                            let before=observations(&source,q as u32,&word[..end]);let after=observations(&built.tokenizer,built.original_to_synth[q],&word[..end]);
                            assert_eq!(before.0,after.0,"first completions q{q} word{word:?} end{end}");
                            if !completion{assert_eq!(before.1,after.1,"first future q{q} word{word:?} end{end}");}
                        }}
                    }
                    for word in &words{for begin in 0..word.len(){for end in begin..=word.len(){
                        assert_eq!(observations(&source,source.start_state(),&word[begin..end]),observations(&built.tokenizer,built.tokenizer.start_state(),&word[begin..end]),"reset factor {word:?}[{begin}..{end}]");
                    }}}
                }
            }
        }
    }
}
