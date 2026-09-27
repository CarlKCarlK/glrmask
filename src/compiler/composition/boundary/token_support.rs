//! Conservative byte-level support for a scoped crossing vocabulary.
//!
//! Forget terminal-label history and grammar follow restrictions, but retain
//! whether a foreign-owned terminal has been emitted. Every ordinary lexical
//! path embeds in this two-frontier NFA: its byte/epsilon transitions agree,
//! and each positive-byte terminal completion permits the same reset. Extra
//! (non-longest or parser-infeasible) completions only add possible paths.
//! Therefore absence of any foreign completion/future at a token endpoint is
//! a proof that the token cannot occur in the boundary terminal language.
//! This is a support filter, NEVER an admission oracle.

use crate::automata::lexer::tokenizer::{Lexer, Tokenizer};
use crate::compiler::stages::id_map_and_terminal_dwa::scope::{
    BoundaryOwnership, ImmediateComponentId,
};
use crate::Vocab;
use crate::ds::bitset::BitSet;
use std::collections::BTreeMap;
use rustc_hash::FxHashMap;

#[derive(Debug)]
pub(crate) struct CrossingTokenSupport {
    pub tokens: Vec<u32>,
    pub byte_prefixes: usize,
    pub state_steps: usize,
    pub max_frontier: usize,
}

#[derive(Clone)]
struct Frontier {
    local: Vec<u32>,
    crossed: Vec<u32>,
    endpoint: bool,
}

fn add_resets(states: &mut Vec<u32>, resets: &[u32]) {
    states.extend_from_slice(resets);
    states.sort_unstable();
    states.dedup();
}

pub(crate) fn crossing_token_support(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    initial_mask: &[bool],
    ownership: &BoundaryOwnership,
    start: ImmediateComponentId,
) -> Option<CrossingTokenSupport> {
    const MAX_STATE_STEPS: usize = 12_000_000;
    const MAX_INPUT_BYTES: usize = 262_144;
    let n = tokenizer.num_states() as usize;
    if initial_mask.len() != n || vocab.is_empty()
        || tokenizer.has_virtual_residual_runtime()
        || vocab.entries_map().values().try_fold(0usize, |n, word| n.checked_add(word.len()))? > MAX_INPUT_BYTES
    { return None; }
    let mut observations = Vec::<u8>::with_capacity(n);
    for raw in 0..n as u32 {
        if tokenizer.state_is_virtual_runtime(raw) { return None; }
        let mut flags = 0;
        for terminal in tokenizer.matched_terminals_iter(raw) {
            flags |= if ownership.owner_of_terminal(terminal) == Some(start) { 1 } else { 2 };
        }
        for terminal in tokenizer.possible_future_terminals_iter(raw) {
            flags |= 4;
            if ownership.owner_of_terminal(terminal) != Some(start) { flags |= 8; }
        }
        observations.push(flags);
    }
    let resets = tokenizer.deterministic_reset_states().into_vec();
    let mut entries = vocab.entries_map().iter().collect::<Vec<_>>();
    entries.sort_unstable_by(|a, b| a.1.cmp(b.1).then(a.0.cmp(b.0)));
    let initial = initial_mask.iter().enumerate()
        .filter_map(|(q, &keep)| keep.then_some(q as u32)).collect::<Vec<_>>();
    let mut result = CrossingTokenSupport { tokens: Vec::new(), byte_prefixes: 0,
        state_steps: 0, max_frontier: initial.len() };
    let mut frames = vec![Frontier { local: initial, crossed: Vec::new(), endpoint: false }];
    let mut previous: &[u8] = &[];
    for (&token, bytes) in entries {
        // Zero-byte/special-token policy belongs to the established compiler.
        if bytes.is_empty() { result.tokens.push(token); continue; }
        let lcp = previous.iter().zip(bytes).take_while(|(a,b)| a == b).count();
        frames.truncate(lcp + 1);
        for &byte in &bytes[lcp..] {
            let old = frames.last()?;
            result.state_steps = result.state_steps.checked_add(old.local.len() + old.crossed.len())?;
            if result.state_steps > MAX_STATE_STEPS { return None; }
            let mut local = tokenizer.step_all(&old.local, byte).into_vec();
            let mut crossed = tokenizer.step_all(&old.crossed, byte).into_vec();
            if local.iter().chain(&crossed).any(|&q| q as usize >= n) { return None; }
            let mut local_match = false;
            let mut foreign_match = false;
            let mut crossed_match = false;
            let mut endpoint = false;
            for &q in &local {
                let flags = observations[q as usize];
                local_match |= flags & 1 != 0;
                foreign_match |= flags & 2 != 0;
                endpoint |= flags & (2 | 8) != 0;
            }
            for &q in &crossed {
                let flags = observations[q as usize];
                crossed_match |= flags & 3 != 0;
                endpoint |= flags & (1 | 2 | 4) != 0;
            }
            // Evaluate the endpoint BEFORE resets. A local completion at the
            // last byte must not invent an unconsumed foreign future label.
            if local_match { add_resets(&mut local, &resets); }
            if foreign_match || crossed_match { add_resets(&mut crossed, &resets); }
            result.max_frontier = result.max_frontier.max(local.len() + crossed.len());
            result.byte_prefixes += 1;
            frames.push(Frontier { local, crossed, endpoint });
        }
        if frames.last()?.endpoint { result.tokens.push(token); }
        previous = bytes;
    }
    result.tokens.sort_unstable();
    Some(result)
}

/// Eligibility sets form an exact join-semilattice for an existential
/// pair-follow observer. Multiple histories arriving at the same byte state
/// can be joined by union: tests for the next terminal distribute over that
/// union, and its subsequent follow set depends only on the emitted terminal.
struct FollowMasks {
    rows: Vec<Vec<u64>>,
    ids: FxHashMap<Vec<u64>, u32>,
    joins: FxHashMap<(u32,u32),u32>,
}

impl FollowMasks {
    fn new(terminals: usize) -> Self {
        let words=terminals.div_ceil(64);
        let mut all=vec![u64::MAX;words];
        if terminals%64!=0 { *all.last_mut().unwrap()=(1u64<<(terminals%64))-1; }
        let rows=vec![vec![0;words],all];
        let ids=rows.iter().cloned().enumerate().map(|(i,row)|(row,i as u32)).collect();
        Self { rows,ids,joins:FxHashMap::default() }
    }
    fn intern(&mut self,bits:Vec<u64>)->Option<u32> {
        if let Some(&id)=self.ids.get(&bits){return Some(id);}
        if self.rows.len()>=8192{return None;}
        let id=self.rows.len() as u32;self.rows.push(bits.clone());self.ids.insert(bits,id);Some(id)
    }
    fn join(&mut self,a:u32,b:u32)->Option<u32> {
        if a==b || b==0 || a==1{return Some(a);}
        if a==0 || b==1{return Some(b);}
        let key=if a<b{(a,b)}else{(b,a)};
        if let Some(&id)=self.joins.get(&key){return Some(id);}
        let bits=self.rows[a as usize].iter().zip(&self.rows[b as usize]).map(|(a,b)|a|b).collect();
        let id=self.intern(bits)?;
        if self.joins.len()<262_144{self.joins.insert(key,id);}
        Some(id)
    }
    fn contains(&self,mask:u32,terminal:u32)->bool {
        self.rows[mask as usize][terminal as usize/64]&(1u64<<(terminal%64))!=0
    }
    fn intersects(&self,mask:u32,observations:&BitSet)->bool {
        self.rows[mask as usize].iter().zip(observations.words()).any(|(a,b)|a&b!=0)
    }
}

#[derive(Clone)]
struct FollowFrontier { lanes:[Vec<(u32,u32)>;2], endpoint:bool }

pub(crate) fn crossing_token_support_with_follows(
    tokenizer:&Tokenizer, vocab:&Vocab, initial_mask:&[bool],
    ownership:&BoundaryOwnership, start:ImmediateComponentId,
    disallowed:&BTreeMap<u32,BitSet>, ignore:Option<u32>, transparent:Option<&BitSet>,
) -> Option<CrossingTokenSupport> {
    crossing_token_support_with_follows_and_adjacency(
        tokenizer,vocab,initial_mask,ownership,start,disallowed,ignore,transparent,None,
    )
}

/// Same necessary-support abstraction, with an additional explicit lexical
/// adjacency relation. Transparent labels still relax the original predecessor
/// constraints, but cannot erase this immediate-adjacency restriction. The
/// actual terminal pipeline applies both exact products after this filter.
/// Borrows the exact tokenizer and the ordinary compiler's prebuilt direct
/// byte table. Epsilon-source/target transitions retain the unchanged exact
/// NFA operation. This view stores no vocabulary, grammar or owner decisions.
pub(crate) struct PreparedSupportTransitions<'a> {
    tokenizer: &'a Tokenizer,
    flat: &'a [u32],
    has_epsilon: Vec<bool>,
}
impl<'a> PreparedSupportTransitions<'a> {
    /// Borrow the pinned direct table without accepting a lookalike context.
    pub(crate) fn flat_for(&self,tokenizer:&Tokenizer)->Option<&[u32]> {
        std::ptr::eq(self.tokenizer,tokenizer).then_some(self.flat)
    }

    /// `flat` must be the ordinary direct transition table built from this
    /// exact immutable tokenizer. The crate-private caller owns that link
    /// context; no vocabulary-dependent state map is accepted here.
    pub(crate) fn new(tokenizer:&'a Tokenizer, flat:&'a [u32])->Option<Self> {
        let n=tokenizer.num_states() as usize;
        if flat.len()!=n.checked_mul(256)? || tokenizer.has_virtual_residual_runtime(){return None;}
        Some(Self{tokenizer,flat,has_epsilon:(0..n as u32).map(|q|tokenizer.state_has_epsilon_transitions(q)).collect()})
    }
    #[inline]
    fn step(&self,source:u32,byte:u8)->crate::automata::lexer::tokenizer::TokenizerStateSet {
        if !self.has_epsilon[source as usize] {
            let target=self.flat[source as usize*256+byte as usize];
            if target==u32::MAX { return Default::default(); }
            if !self.has_epsilon[target as usize] { return crate::automata::lexer::tokenizer::TokenizerStateSet::from_buf([target]); }
        }
        self.tokenizer.step_all(&[source],byte)
    }
}

#[cfg(test)]
#[test]
fn prepared_transitions_equal_every_raw_byte_step_including_nested_epsilon_dispatch() {
    use crate::automata::lexer::ast::{bytes,choice,plus};
    use crate::automata::lexer::compile::{build_regex_monolithic,build_regex_partitioned};
    use crate::compiler::stages::id_map_and_terminal_dwa::l1::build_flat_transition_table;
    let expressions=vec![bytes(b"abc"),plus(choice(vec![bytes(b"a"),bytes(b"bc")])),bytes(b"!ab")];
    let first=build_regex_monolithic(&expressions).into_tokenizer(3,None);
    let second=build_regex_partitioned(&expressions,&[0,1,2]).into_tokenizer(3,None);
    let (nested,_)=Tokenizer::disjoint_union_with_terminal_offsets(&[(&first,0),(&second,3)]);
    for tokenizer in [first,second,nested] {
        let flat=build_flat_transition_table(&tokenizer);
        let prepared=PreparedSupportTransitions::new(&tokenizer,&flat).unwrap();
        for q in 0..tokenizer.num_states(){for byte in 0..=255u8{
            assert_eq!(prepared.step(q,byte),tokenizer.step_all(&[q],byte),"q={q} byte={byte}");
        }}
        assert!(PreparedSupportTransitions::new(&tokenizer,&flat[..flat.len()-1]).is_none());
    }
}

pub(crate) fn crossing_token_support_with_follows_and_adjacency(
    tokenizer:&Tokenizer, vocab:&Vocab, initial_mask:&[bool],
    ownership:&BoundaryOwnership, start:ImmediateComponentId,
    disallowed:&BTreeMap<u32,BitSet>, ignore:Option<u32>, transparent:Option<&BitSet>,
    adjacency:Option<&BTreeMap<u32,BitSet>>,
) -> Option<CrossingTokenSupport> {
    crossing_token_support_with_prepared_transitions(
        tokenizer,vocab,initial_mask,ownership,start,disallowed,ignore,transparent,adjacency,None,
    )
}

pub(crate) fn crossing_token_support_with_prepared_transitions(
    tokenizer:&Tokenizer, vocab:&Vocab, initial_mask:&[bool],
    ownership:&BoundaryOwnership, start:ImmediateComponentId,
    disallowed:&BTreeMap<u32,BitSet>, ignore:Option<u32>, transparent:Option<&BitSet>,
    adjacency:Option<&BTreeMap<u32,BitSet>>,
    transitions:Option<&PreparedSupportTransitions<'_>>,
) -> Option<CrossingTokenSupport> {
    let n=tokenizer.num_states() as usize;
    let terminals=tokenizer.num_terminals() as usize;
    if let Some(t)=transitions { assert!(std::ptr::eq(t.tokenizer, tokenizer)); }
    if initial_mask.len()!=n || terminals==0 || tokenizer.has_virtual_residual_runtime()
        || vocab.entries_map().values().try_fold(0usize,|n,w|n.checked_add(w.len()))?>262_144
    {return None;}
    let is_transparent=|t:u32|Some(t)==ignore || transparent.is_some_and(|m|m.get(t as usize));
    let mut masks=FollowMasks::new(terminals);
    let mut transparent_words=vec![0u64;terminals.div_ceil(64)];
    for t in 0..terminals as u32 {
        if is_transparent(t){transparent_words[t as usize/64]|=1u64<<(t%64);}
    }
    let mut follows=Vec::with_capacity(terminals);
    for t in 0..terminals as u32 {
        let mut bits=masks.rows[1].clone();
        if !is_transparent(t) {
            if let Some(blocked)=disallowed.get(&t) {
                for (word,blocked) in bits.iter_mut().zip(blocked.words()){*word&=!*blocked;}
            }
            // Transparent terminals do not add a new restriction. Widen their
            // reset relation to ALL below rather than accidentally forbidding
            // a path whose real predecessor survives across ignored text.
            for (word,&transparent) in bits.iter_mut().zip(&transparent_words){*word|=transparent;}
        }
        if let Some(blocked)=adjacency.and_then(|rows|rows.get(&t)) {
            for (word,blocked) in bits.iter_mut().zip(blocked.words()){*word&=!*blocked;}
        }
        follows.push(masks.intern(bits)?);
    }
    let foreign=(0..terminals as u32).map(|t|ownership.owner_of_terminal(t)!=Some(start)).collect::<Vec<_>>();
    // Keep metadata borrowed from the immutable tokenizer. A live witness
    // proves a positive intersection immediately; fallback checks both exact
    // matched and future sets. A missing witness proves the union is empty.
    let mut live_witness=Vec::with_capacity(n);
    let mut foreign_witness=Vec::with_capacity(n);
    let mut foreign_bits=vec![0u64;terminals.div_ceil(64)];
    for t in 0..terminals { if foreign[t] { foreign_bits[t/64]|=1u64<<(t%64); } }
    for q in 0..n as u32 {
        if tokenizer.state_is_virtual_runtime(q){return None;}
        live_witness.push(tokenizer.matched_terminals_slice(q).first().copied()
            .or_else(||tokenizer.possible_future_terminals_iter(q).next()).unwrap_or(u32::MAX));
        foreign_witness.push(tokenizer.possible_future_terminals_iter(q)
            .find(|&t|foreign[t as usize]).unwrap_or(u32::MAX));
    }
    let resets=tokenizer.deterministic_reset_states().into_vec();
    let initial=initial_mask.iter().enumerate().filter_map(|(q,&keep)|keep.then_some((q as u32,1))).collect::<Vec<_>>();
    let mut result=CrossingTokenSupport{tokens:Vec::new(),byte_prefixes:0,state_steps:0,max_frontier:initial.len()};
    let mut frames=vec![FollowFrontier{lanes:[initial,Vec::new()],endpoint:false}];
    let mut entries=vocab.entries_map().iter().collect::<Vec<_>>();
    entries.sort_unstable_by(|a,b|a.1.cmp(b.1).then(a.0.cmp(b.0)));
    let mut previous:&[u8]=&[];
    let mut target_ids=vec![0u32;n];
    let mut touched=Vec::<u32>::new();
    for (&token,bytes) in entries {
        if bytes.is_empty(){result.tokens.push(token);continue;}
        let lcp=previous.iter().zip(bytes).take_while(|(a,b)|a==b).count();frames.truncate(lcp+1);
        for &byte in &bytes[lcp..] {
            let old=frames.last()?;
            let mut next=FollowFrontier{lanes:[Vec::new(),Vec::new()],endpoint:false};
            let mut reset_masks=[0u32;2];
            for lane in 0..2 {
                for &(source,eligible) in &old.lanes[lane] {
                    result.state_steps+=1;if result.state_steps>12_000_000{return None;}
                    for target in transitions.map_or_else(|| tokenizer.step_all(&[source],byte), |t| t.step(source,byte)) {
                        let q=target as usize;if q>=n{return None;}
                        let witness=live_witness[q];
                        if witness==u32::MAX {continue;}
                        if !masks.contains(eligible,witness)
                            && !masks.intersects(eligible,tokenizer.possible_future_terminals(target))
                            && !tokenizer.matched_terminals_slice(target).iter().any(|&t|masks.contains(eligible,t))
                        {continue;}
                        if target_ids[q]==0{touched.push(target);}
                        target_ids[q]=masks.join(target_ids[q],eligible)?;
                    }
                }
                touched.sort_unstable();
                for q in touched.drain(..) {
                    let eligible=std::mem::take(&mut target_ids[q as usize]);
                    let foreign_terminal=foreign_witness[q as usize];
                    let observes_foreign=foreign_terminal!=u32::MAX
                        && (masks.contains(eligible,foreign_terminal)
                            || masks.rows[eligible as usize].iter().zip(tokenizer.possible_future_terminals(q).words())
                                .zip(&foreign_bits).any(|((&a,&b),&c)|a&b&c!=0));
                    if lane==1 || observes_foreign {
                        next.endpoint=true;
                    }
                    for &t in tokenizer.matched_terminals_slice(q) {
                        if masks.contains(eligible,t) {
                            let next_lane=usize::from(lane==1 || foreign[t as usize]);
                            if next_lane==1{next.endpoint=true;}
                            reset_masks[next_lane]=masks.join(reset_masks[next_lane],follows[t as usize])?;
                        }
                    }
                    next.lanes[lane].push((q,eligible));
                }
            }
            // As in the reference walker, new terminal starts cannot create
            // zero-byte endpoint observations. Add reset roots only afterward.
            for lane in 0..2 {
                if reset_masks[lane]==0{continue;}
                for &(q,eligible) in &next.lanes[lane]{target_ids[q as usize]=eligible;touched.push(q);}
                for &q in &resets {
                    if q as usize>=n{return None;}
                    if target_ids[q as usize]==0{touched.push(q);}
                    target_ids[q as usize]=masks.join(target_ids[q as usize],reset_masks[lane])?;
                }
                touched.sort_unstable();next.lanes[lane].clear();
                for q in touched.drain(..){next.lanes[lane].push((q,std::mem::take(&mut target_ids[q as usize])));}
            }
            result.max_frontier=result.max_frontier.max(next.lanes[0].len()+next.lanes[1].len());
            result.byte_prefixes+=1;frames.push(next);
        }
        if frames.last()?.endpoint{result.tokens.push(token);}
        previous=bytes;
    }
    result.tokens.sort_unstable();Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::automata::lexer::ast::{bytes, choice, plus};
    use crate::automata::lexer::compile::{build_regex_monolithic, build_regex_partitioned};

    #[test]
    fn prepared_raw_transitions_match_scalar_follow_queries() {
        let expressions=vec![plus(choice(vec![bytes(b"a"),bytes(b"ab")])),bytes(b"c"),
            choice(vec![bytes(b"!x"),bytes(b"!y")]),bytes(b" ")];
        let mut words=vec![Vec::<u8>::new()];
        let mut level=vec![Vec::<u8>::new()];
        for _ in 0..5 {
            let mut next=Vec::new();
            for word in level {for byte in b"ac! " {let mut w=word.clone();w.push(*byte);next.push(w);}}
            words.extend(next.iter().cloned());level=next;
        }
        let vocab=Vocab::new(words.into_iter().enumerate().map(|(i,w)|(11+i as u32*3,w)).collect());
        let owner=BoundaryOwnership::flat(&[0,2],4).unwrap();
        let mut blocked=BitSet::new(4);blocked.set(2);
        let follows=BTreeMap::from([(0,blocked)]);
        for tokenizer in [build_regex_monolithic(&expressions).into_tokenizer(4,None),
            build_regex_partitioned(&expressions,&[0,1,2,3]).into_tokenizer(4,None)] {
            let flat=crate::compiler::stages::id_map_and_terminal_dwa::l1::build_flat_transition_table(&tokenizer);
            let prepared=PreparedSupportTransitions::new(&tokenizer,&flat).unwrap();
            let initial=vec![true;tokenizer.num_states() as usize];
            for adjacency in [None,Some(&follows)] {
                let reference=crossing_token_support_with_prepared_transitions(&tokenizer,&vocab,&initial,&owner,
                    ImmediateComponentId(0),&follows,Some(3),None,adjacency,None).unwrap();
                let candidate=crossing_token_support_with_prepared_transitions(&tokenizer,&vocab,&initial,&owner,
                    ImmediateComponentId(0),&follows,Some(3),None,adjacency,Some(&prepared)).unwrap();
                assert_eq!(reference.tokens,candidate.tokens);
                assert_eq!(reference.state_steps,candidate.state_steps);
                assert_eq!(reference.max_frontier,candidate.max_frontier);
            }
            assert!(PreparedSupportTransitions::new(&tokenizer,&[0]).is_none());
        }
    }

    #[test]
    fn crossing_support_handles_partial_endpoints_resets_and_duplicate_tokens() {
        let expressions = vec![bytes(b"abcdefghijklmn"), plus(bytes(b"q")),
            choice(vec![bytes(b"!x"), bytes(b"$z")])];
        for tokenizer in [
            build_regex_monolithic(&expressions).into_tokenizer(3,Some(Arc::from(expressions.clone()))),
            build_regex_partitioned(&expressions,&[0,1,2]).into_tokenizer(3,Some(Arc::from(expressions.clone()))),
        ] {
            let ownership=BoundaryOwnership::flat(&[0,2],3).unwrap();
            let mask=(0..tokenizer.num_states()).map(|q| {
                tokenizer.possible_future_terminals_iter(q).any(|t| t<2)
                    && !tokenizer.possible_future_terminals_iter(q).any(|t| t==2)
            }).collect::<Vec<_>>();
            let vocab=Vocab::new(vec![(1,b"mn!".to_vec()),(99,b"mn!".to_vec()),
                (101,b"mn!xq".to_vec()),(102,b"q$".to_vec()),(103,b"q".to_vec()),
                (104,b"mn".to_vec()),(105,b"mn~".to_vec()),(999,vec![])]);
            let support=crossing_token_support(&tokenizer,&vocab,&mask,&ownership,ImmediateComponentId(0)).unwrap();
            for id in [1,99,101,102,999] { assert!(support.tokens.contains(&id),"missing {id}"); }
            for id in [103,104,105] { assert!(!support.tokens.contains(&id),"spurious simple token {id}"); }
            assert!(support.byte_prefixes < vocab.entries_map().values().map(Vec::len).sum());
        }
    }

    #[test]
    fn follow_support_eliminates_wrong_successor_but_keeps_transparent_paths() {
        let expressions=vec![bytes(b"a"),bytes(b"b"),bytes(b"!x"),bytes(b" ")];
        let tokenizer=build_regex_partitioned(&expressions,&[0,1,2,3]).into_tokenizer(4,Some(Arc::from(expressions)));
        let ownership=BoundaryOwnership::flat(&[0,2],4).unwrap();
        let initial=(0..tokenizer.num_states()).map(|q| tokenizer.possible_future_terminals_iter(q).any(|t|t==0)
            && !tokenizer.possible_future_terminals_iter(q).any(|t|t>=1)).collect::<Vec<_>>();
        let vocab=Vocab::new(vec![(1,b"a!".to_vec()),(2,b"ab!".to_vec()),(3,b"a !".to_vec()),(4,b"a".to_vec())]);
        let mut blocked=BitSet::new(4);blocked.set(2);
        let disallowed=BTreeMap::from([(0,blocked)]);
        let support=crossing_token_support_with_follows(&tokenizer,&vocab,&initial,&ownership,
            ImmediateComponentId(0),&disallowed,Some(3),None).unwrap();
        assert!(!support.tokens.contains(&1));assert!(!support.tokens.contains(&4));
        assert!(support.tokens.contains(&2));assert!(support.tokens.contains(&3));
    }
}

#[cfg(test)]
#[test]
fn explicit_adjacency_is_not_erased_by_transparent_follow_relaxation() {
    use std::sync::Arc;
    use crate::automata::lexer::ast::bytes;
    use crate::automata::lexer::compile::build_regex_partitioned;
    let expressions=vec![bytes(b"a"),bytes(b"b"),bytes(b"!x"),bytes(b" ")];
    let tokenizer=build_regex_partitioned(&expressions,&[0,1,2,3]).into_tokenizer(4,Some(Arc::from(expressions)));
    let ownership=BoundaryOwnership::flat(&[0,2],4).unwrap();
    let initial=(0..tokenizer.num_states()).map(|q| tokenizer.possible_future_terminals_iter(q).any(|t|t==0)
        && !tokenizer.possible_future_terminals_iter(q).any(|t|t>=1)).collect::<Vec<_>>();
    let vocab=Vocab::new(vec![(1,b"a!".to_vec()),(2,b"ab!".to_vec()),(3,b"a !".to_vec()),(4,b"a ".to_vec())]);
    let mut blocked=BitSet::new(4);blocked.set(3);
    let adjacency=BTreeMap::from([(0,blocked)]);
    let actual=crossing_token_support_with_follows_and_adjacency(&tokenizer,&vocab,&initial,
        &ownership,ImmediateComponentId(0),&BTreeMap::new(),Some(3),None,Some(&adjacency)).unwrap();
    assert!(actual.tokens.contains(&1));assert!(actual.tokens.contains(&2));
    assert!(!actual.tokens.contains(&3));assert!(!actual.tokens.contains(&4));
}
