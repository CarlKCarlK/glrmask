//! Component-only observations feeding the exact necessary cut-support DP.
use std::{collections::BTreeMap,time::Instant};
use rustc_hash::FxHashMap;
use glrmask_lexer::__private::{automata::lexer::Lexer,ds::bitset::BitSet};
use glrmask_terminal_dwa::__private::terminal_dwa::scope::{BoundaryOwnership,ImmediateComponentId};
use glrmask_vocab::Vocab;
use super::{completion_index::{CompletionContext,FirstUnionIndex},reset_index::ResetIndex};
#[derive(Clone,PartialEq,Eq,Hash)]
struct Observation{matched:Vec<u64>,future:Vec<u64>}
impl Observation{
 fn empty(width:usize)->Self{Self{matched:vec![0;width],future:vec![0;width]}}
 fn endpoint(&self,eligible:&[u64],foreign:&[u64],crossed:bool)->bool{
  self.matched.iter().zip(&self.future).zip(eligible).zip(foreign).any(|(((&m,&f),&e),&outside)|((m|f)&e&if crossed{u64::MAX}else{outside})!=0)
 }
}
struct Observations{rows:Vec<Observation>,ids:FxHashMap<Observation,u32>}
impl Observations{
 fn new(width:usize)->Self{let zero=Observation::empty(width);Self{rows:vec![zero.clone()],ids:FxHashMap::from_iter([(zero,0)])}}
 fn intern(&mut self,obs:Observation)->Option<u32>{
  if let Some(&q)=self.ids.get(&obs){return Some(q)}
  if self.rows.len()>=32768||self.rows.len().checked_add(1)?.checked_mul(obs.matched.len())?.checked_mul(2)?>4_000_000{return None}
  let id=self.rows.len()as u32;self.ids.insert(obs.clone(),id);self.rows.push(obs);Some(id)
 }
}
pub(crate) struct Part<'a>{pub prefix:&'a CompletionContext,pub unions:&'a FirstUnionIndex,pub reset:&'a ResetIndex,pub terminal_offset:u32}
struct Products{states:Vec<Vec<u32>>,ids:FxHashMap<Vec<u32>,u32>,edges:FxHashMap<(u32,u8),u32>,obs:Vec<u32>}
impl Products{
 fn new(parts:&[Part<'_>],pool:&mut Observations,width:usize)->Option<Self>{
  let mut obs=Observation::empty(width);for p in parts{let(m,f)=p.reset.observation(0)?;for &t in m{let t=t.checked_add(p.terminal_offset)?as usize;*obs.matched.get_mut(t/64)?|=1u64<<(t%64);}for &t in f{let t=t.checked_add(p.terminal_offset)?as usize;*obs.future.get_mut(t/64)?|=1u64<<(t%64);}}
  let root=pool.intern(obs)?;let n=parts.len();Some(Self{states:vec![vec![u32::MAX;n],vec![0;n]],ids:FxHashMap::from_iter([(vec![u32::MAX;n],0),(vec![0;n],1)]),edges:FxHashMap::default(),obs:vec![0,root]})
 }
 fn step(&mut self,q:u32,byte:u8,parts:&[Part<'_>],pool:&mut Observations,width:usize)->Option<u32>{
  if q==0{return Some(0)}
  if let Some(&n)=self.edges.get(&(q,byte)){return Some(n)}
  if self.edges.len()>=1_000_000{return None}
  let next=self.states[q as usize].iter().zip(parts).map(|(&s,p)|p.reset.step_id(s,byte)).collect::<Option<Vec<_>>>()?;
  let id=if let Some(&id)=self.ids.get(&next){id}else{
   if self.states.len()>=32768{return None}
   let mut obs=Observation::empty(width);
   for (&s,p)in next.iter().zip(parts){let(m,f)=p.reset.observation(s)?;
    for &t in m{let t=t.checked_add(p.terminal_offset)? as usize;*obs.matched.get_mut(t/64)?|=1u64<<(t%64);}
    for &t in f{let t=t.checked_add(p.terminal_offset)? as usize;*obs.future.get_mut(t/64)?|=1u64<<(t%64);}
   }
   let oid=pool.intern(obs)?;let id=self.states.len()as u32;self.ids.insert(next.clone(),id);self.states.push(next);self.obs.push(oid);id
  };
  self.edges.insert((q,byte),id);Some(id)
 }
}
struct Masks {
    rows: Vec<Vec<u64>>,
    ids: FxHashMap<Vec<u64>, u32>,
    joins: FxHashMap<(u32, u32), u32>,
}
impl Masks {
    fn new(all: Vec<u64>) -> Self {
        let rows = vec![vec![0; all.len()], all];
        let ids = rows
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, r)| (r, i as u32))
            .collect();
        Self {
            rows,
            ids,
            joins: FxHashMap::default(),
        }
    }
    fn intern(&mut self, row: Vec<u64>) -> Option<u32> {
        if let Some(&id) = self.ids.get(&row) {
            return Some(id);
        }
        if self.rows.len() >= 8192
            || self.rows.len().checked_add(1)?.checked_mul(row.len())? > 4_000_000
        {
            return None;
        }
        let id = self.rows.len() as u32;
        self.ids.insert(row.clone(), id);
        self.rows.push(row);
        Some(id)
    }
    fn join(&mut self, a: u32, b: u32) -> Option<u32> {
        if a == 0 || a == b {
            return Some(b);
        }
        if b == 0 || a == 1 {
            return Some(a);
        }
        if b == 1 {
            return Some(1);
        }
        let key = if a < b { (a, b) } else { (b, a) };
        if let Some(&id) = self.joins.get(&key) {
            return Some(id);
        }
        let row = self.rows[a as usize]
            .iter()
            .zip(&self.rows[b as usize])
            .map(|(a, b)| a | b)
            .collect();
        let id = self.intern(row)?;
        if self.joins.len() < 262144 {
            self.joins.insert(key, id);
        }
        Some(id)
    }
}
struct Follows<'a>{
 all:Vec<u64>,transparent_bits:Vec<u64>,transparent:Vec<bool>,disallowed:&'a BTreeMap<u32,BitSet>,adjacency:Option<&'a BTreeMap<u32,BitSet>>,rows:Vec<Option<u32>>,built:usize,
}
impl Follows<'_>{
 fn get(&mut self,t:usize,pool:&mut Masks)->Option<u32>{
  if let Some(q)=*self.rows.get(t)?{return Some(q)}
  let mut row=self.all.clone();
  if !self.transparent[t]{if let Some(blocked)=self.disallowed.get(&(t as u32)){for(a,&b)in row.iter_mut().zip(blocked.words()){*a&=!b;}}for(a,&b)in row.iter_mut().zip(&self.transparent_bits){*a|=b;}}
  if let Some(blocked)=self.adjacency.and_then(|a|a.get(&(t as u32))){for(a,&b)in row.iter_mut().zip(blocked.words()){*a&=!b;}}
  let id=pool.intern(row)?;self.rows[t]=Some(id);self.built+=1;Some(id)
 }
}
struct Transfers {
    rows: FxHashMap<(u32, u32), [u32; 2]>,
}
impl Transfers {
    fn get(
        &mut self,
        observation: u32,
        matched: &[u64],
        eligible: u32,
        follows: &mut Follows<'_>,
        foreign: &[bool],
        pool: &mut Masks,
        events: &mut usize,
    ) -> Option<[u32; 2]> {
        if eligible == 0 {
            return Some([0, 0]);
        }
        let key = (observation, eligible);
        if let Some(&row) = self.rows.get(&key) {
            return Some(row);
        }
        let mut out = [0u32; 2];
        for (i, &matched) in matched.iter().enumerate() {
            let mut bits = matched & pool.rows[eligible as usize][i];
            while bits != 0 {
                let t = i * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                *events = events.checked_add(1)?;
                if *events > 16_000_000 {
                    return None;
                }
                let lane = usize::from(foreign[t]);
                let next=follows.get(t,pool)?;
                out[lane] = pool.join(out[lane], next)?;
            }
        }
        if self.rows.len() < 262144 {
            self.rows.insert(key, out);
        }
        Some(out)
    }
}
fn propagate(
    transfer: [u32; 2],
    crossed: bool,
    target: &mut [u32; 2],
    pool: &mut Masks,
) -> Option<()> {
    if crossed {
        let all = pool.join(transfer[0], transfer[1])?;
        target[1] = pool.join(target[1], all)?;
    } else {
        target[0] = pool.join(target[0], transfer[0])?;
        target[1] = pool.join(target[1], transfer[1])?;
    }
    Some(())
}


#[derive(Debug,Default)]
pub(crate) struct Profile{pub first_query_ms:f64,pub setup_ms:f64,pub solve_ms:f64,pub total_ms:f64,pub trie_nodes:usize,pub frontier_cells:usize,pub products:usize,pub product_edges:usize,pub observations:usize,pub transfers:usize,pub events:usize,pub cut_pairs:usize,pub follow_rows:usize}
#[derive(Debug)]pub(crate) struct Result{pub tokens:Vec<u32>,pub profile:Profile}
/// Caller establishes the exact disjoint-union source/terminal layout once.
/// No terminal-language reshaping or original-state coordinate projection occurs.
fn query_impl(nt:usize,parts:&[Part<'_>],start:ImmediateComponentId,local_initial:&[bool],vocab:&Vocab,ownership:&BoundaryOwnership,disallowed:&BTreeMap<u32,BitSet>,ignore:Option<u32>,transparent:Option<&BitSet>,adjacency:Option<&BTreeMap<u32,BitSet>>,use_trie:bool)->Option<Result>{
 let clock=Instant::now();
 if nt==0||nt>65536||parts.is_empty()||parts.len()>64{return None}
 let start_idx=start.0 as usize;let first_part=parts.get(start_idx)?;
 if local_initial.len()!=first_part.prefix.raw_states(){return None}
 let width=nt.div_ceil(64);if nt.checked_mul(width)?>4_000_000{return None}
 for (owner,p)in parts.iter().enumerate(){
  if !std::ptr::eq(p.prefix.source(),p.reset.source())||!std::ptr::eq(p.unions.source(),p.prefix.source()){return None}
  if p.terminal_offset.checked_add(p.reset.source().num_terminals())? as usize>nt{return None}
  for t in 0..p.reset.source().num_terminals(){if ownership.owner_of_terminal(t+p.terminal_offset)!=Some(ImmediateComponentId(owner as u32)){return None}}
 }
 let entries=vocab.entries_map().iter().collect::<Vec<_>>();
 if entries.iter().try_fold(0usize,|n,(_,w)|n.checked_add(w.len()))?>262144{return None}
 let active=entries.iter().enumerate().filter(|(_,(_,w))|!w.is_empty()).map(|(i,(_,w))|(i,w.as_slice())).collect::<Vec<_>>();
 let words=active.iter().map(|(_,w)|*w).collect::<Vec<_>>();
 let begin=Instant::now();let first=first_part.unions.query_borrowed(local_initial,&words).ok()?;
 let mut profile=Profile{first_query_ms:begin.elapsed().as_secs_f64()*1000.0,..Default::default()};
 let mut observations=Observations::new(width);let mut columns=Vec::with_capacity(first.columns.len());
 for row in &first.columns{let mut obs=Observation::empty(width);for &t in row{let global=t.checked_add(first_part.terminal_offset)?as usize;*obs.matched.get_mut(global/64)?|=1u64<<(global%64);}columns.push(observations.intern(obs)?);}
 let mut paths=vec![Vec::new();entries.len()];for ((i,_),path)in active.iter().zip(&first.word_columns){paths[*i]=path.iter().map(|&c|columns[c]).collect();}
 let mut all=vec![u64::MAX;width];if nt%64!=0{*all.last_mut()?=(1u64<<(nt%64))-1;}
 let foreign=(0..nt).map(|t|ownership.owner_of_terminal(t as u32)!=Some(start)).collect::<Vec<_>>();
 let mut foreign_bits=vec![0u64;width];let mut transparent_bits=vec![0u64;width];
 let is_transparent=|t:usize|Some(t as u32)==ignore||transparent.is_some_and(|m|m.get(t));
 for t in 0..nt{if foreign[t]{foreign_bits[t/64]|=1u64<<(t%64)}if is_transparent(t){transparent_bits[t/64]|=1u64<<(t%64)}}
 let mut pool=Masks::new(all.clone());let mut transfers=Transfers{rows:FxHashMap::default()};
 let mut follows=Follows{all:all.clone(),transparent_bits,transparent:(0..nt).map(is_transparent).collect(),disallowed,adjacency,rows:vec![None;nt],built:0};
 let mut products=Products::new(parts,&mut observations,width)?;profile.setup_ms=clock.elapsed().as_secs_f64()*1000.0;let solve=Instant::now();let mut tokens=Vec::new();
 if use_trie{
  struct Trie{first:u32,children:BTreeMap<u8,usize>,leaves:Vec<u32>}
  let mut trie=vec![Trie{first:0,children:BTreeMap::new(),leaves:Vec::new()}];
  for(wi,(&id,word))in entries.iter().copied().enumerate(){
   if word.is_empty(){tokens.push(id);continue}
   if word.len()>4096||word.len().checked_mul(word.len())?>1_000_000||paths[wi].len()!=word.len()+1{return None}
   let mut q=0;
   for(depth,&byte)in word.iter().enumerate(){let next=if let Some(&n)=trie[q].children.get(&byte){if trie[n].first!=paths[wi][depth+1]{return None}n}else{if trie.len()>=262145{return None}let n=trie.len();trie.push(Trie{first:paths[wi][depth+1],children:BTreeMap::new(),leaves:Vec::new()});trie[q].children.insert(byte,n);n};q=next;}
   trie[q].leaves.push(id);
  }
  profile.trie_nodes=trie.len();
  let mut frontiers=vec![Vec::<(u32,[u32;2])>::new();trie.len()];
  for q in 0..trie.len(){
   let current=std::mem::take(&mut frontiers[q]);
   for(&byte,&child)in &trie[q].children{
    let mut next=FxHashMap::<u32,[u32;2]>::default();let mut reset=[0u32;2];let mut accepted=false;
    let first=trie[child].first;let obs=&observations.rows[first as usize];let tr=transfers.get(first,&obs.matched,1,&mut follows,&foreign,&mut pool,&mut profile.events)?;propagate(tr,false,&mut reset,&mut pool)?;
    for &(state,eligible)in &current{
     profile.cut_pairs+=1;if profile.cut_pairs>4_000_000{return None}
     let target=products.step(state,byte,parts,&mut observations,width)?;if target==0{continue}
     let oid=products.obs[target as usize];let obs=&observations.rows[oid as usize];let dst=next.entry(target).or_insert([0,0]);
     for lane in 0..2{
      dst[lane]=pool.join(dst[lane],eligible[lane])?;
      if !trie[child].leaves.is_empty(){accepted|=obs.endpoint(&pool.rows[eligible[lane]as usize],&foreign_bits,lane==1);}
      let tr=transfers.get(oid,&obs.matched,eligible[lane],&mut follows,&foreign,&mut pool,&mut profile.events)?;propagate(tr,lane==1,&mut reset,&mut pool)?;
     }
    }
    if reset!=[0,0]{let dst=next.entry(1).or_insert([0,0]);for lane in 0..2{dst[lane]=pool.join(dst[lane],reset[lane])?;}}
    profile.frontier_cells=profile.frontier_cells.checked_add(next.len())?;if profile.frontier_cells>4_000_000{return None}
    let mut entries=next.into_iter().collect::<Vec<_>>();entries.sort_unstable_by_key(|x|x.0);frontiers[child]=entries;
    if accepted{tokens.extend_from_slice(&trie[child].leaves);}
   }
  }
 }else{
 for (wi,(&id,word))in entries.iter().copied().enumerate(){
  if word.is_empty(){tokens.push(id);continue}
  if word.len()>4096||word.len().checked_mul(word.len())?>1_000_000||paths[wi].len()!=word.len()+1{return None}
  let mut cuts=vec![[0u32;2];word.len()];let mut accepted=false;
  // Uniform owner-local first seeds cannot cross at the token endpoint.
  // Preserve all proper-prefix completion observations (including prefixes
  // with no prior byte match); this is NOT longest-match token admission.
  for end in 1..word.len(){let oid=paths[wi][end];let obs=&observations.rows[oid as usize];let tr=transfers.get(oid,&obs.matched,1,&mut follows,&foreign,&mut pool,&mut profile.events)?;propagate(tr,false,&mut cuts[end],&mut pool)?;}
  for begin in 1..word.len(){if accepted{break}let eligible=cuts[begin];if eligible==[0,0]{continue}let mut state=1;
   for end in begin+1..=word.len(){profile.cut_pairs+=1;if profile.cut_pairs>4_000_000{return None}state=products.step(state,word[end-1],parts,&mut observations,width)?;let oid=products.obs[state as usize];let obs=&observations.rows[oid as usize];
    for lane in 0..2{if end==word.len(){accepted|=obs.endpoint(&pool.rows[eligible[lane]as usize],&foreign_bits,lane==1)}else{let tr=transfers.get(oid,&obs.matched,eligible[lane],&mut follows,&foreign,&mut pool,&mut profile.events)?;propagate(tr,lane==1,&mut cuts[end],&mut pool)?;}}
    if state==0{break}
   }
  }
  if accepted{tokens.push(id)}
 }
 }
 tokens.sort_unstable();profile.solve_ms=solve.elapsed().as_secs_f64()*1000.0;profile.total_ms=clock.elapsed().as_secs_f64()*1000.0;profile.products=products.states.len();profile.product_edges=products.edges.len();profile.observations=observations.rows.len();profile.transfers=transfers.rows.len();profile.follow_rows=follows.built;Some(Result{tokens,profile})
}

pub(crate) fn query(nt:usize,parts:&[Part<'_>],start:ImmediateComponentId,local_initial:&[bool],vocab:&Vocab,ownership:&BoundaryOwnership,disallowed:&BTreeMap<u32,BitSet>,ignore:Option<u32>,transparent:Option<&BitSet>,adjacency:Option<&BTreeMap<u32,BitSet>>)->Option<Result>{query_impl(nt,parts,start,local_initial,vocab,ownership,disallowed,ignore,transparent,adjacency,false)}
pub(crate) fn query_trie(nt:usize,parts:&[Part<'_>],start:ImmediateComponentId,local_initial:&[bool],vocab:&Vocab,ownership:&BoundaryOwnership,disallowed:&BTreeMap<u32,BitSet>,ignore:Option<u32>,transparent:Option<&BitSet>,adjacency:Option<&BTreeMap<u32,BitSet>>)->Option<Result>{query_impl(nt,parts,start,local_initial,vocab,ownership,disallowed,ignore,transparent,adjacency,true)}
