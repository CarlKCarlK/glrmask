//! Exact one-model-token observation slice for the ordinary terminal compiler.
//!
//! Before the first terminal reset, every state lies on a prefix from a
//! checked initial state. After any reset, it lies on a factor of the SAME
//! model token. Computing these two supports and inducing their union keeps
//! every query path and cannot invent an original byte path. Full lexical
//! observations and raw coordinate lifting are preserved. No runtime lexer
//! or reusable incoming-component whitelist is replaced by this local view.
use crate::automata::lexer::tokenizer::{Lexer,Tokenizer,TokenizerObservationView};
use crate::compiler::stages::id_map_and_terminal_dwa::scope::BoundaryAnalysisScope;
use crate::Vocab;
use std::time::Instant;

mod fast;

pub(super) struct QueryView {
    pub view:TokenizerObservationView,
    pub scope:BoundaryAnalysisScope,
    pub footprint_ms:f64,
    pub materialize_ms:f64,
    pub first_states:usize,
    pub reset_states:usize,
    pub state_steps:usize,
}

pub(super) fn prepare(tokenizer:&Tokenizer,vocab:&Vocab,scope:&BoundaryAnalysisScope)->Option<QueryView>{
    let n=tokenizer.num_states() as usize;
    if n<256 || n>200_000 || tokenizer.has_virtual_residual_runtime() || !scope.require_crossing(){return None;}
    let started=Instant::now();
    let mut words=vocab.entries_map().values().map(Vec::as_slice).collect::<Vec<_>>();
    if words.is_empty() || words.iter().any(|w|w.is_empty() || w.len()>1024)
        || words.iter().try_fold(0usize,|a,w|a.checked_add(w.len()))?>262_144{return None;}
    words.sort_unstable();words.dedup();
    let mut first=scope.initial_states().keep_raw().to_vec();
    let mut continuation=vec![false;n];
    let closures=tokenizer.all_singleton_epsilon_closures();
    for q in 0..n{if first[q]{for &r in &closures[q]{if r as usize>=n{return None;}first[r as usize]=true;}}}
    let initial=first.iter().enumerate().filter_map(|(q,&keep)|keep.then_some(q as u32)).collect::<Vec<_>>();
    let reset=tokenizer.deterministic_reset_states().into_vec();
    for &q in &reset{for &r in &closures[q as usize]{if r as usize>=n{return None;}continuation[r as usize]=true;}}
    let mut frames=vec![(initial,Vec::<u32>::new())];
    let mut previous:&[u8]=&[];let mut state_steps=0usize;let mut stored=frames[0].0.len();
    for word in words {
        let common=previous.iter().zip(word).take_while(|(a,b)|a==b).count();
        for (a,b) in &frames[common+1..]{stored-=a.len()+b.len();}frames.truncate(common+1);
        for &byte in &word[common..]{
            let (old_first,old_reset)=frames.last()?;
            let mut resets=old_reset.clone();resets.extend_from_slice(&reset);resets.sort_unstable();resets.dedup();
            state_steps=state_steps.checked_add(old_first.len()+resets.len())?;
            if state_steps>30_000_000{return None;}
            let f=tokenizer.step_all(old_first,byte).into_vec();let c=tokenizer.step_all(&resets,byte).into_vec();
            for &q in &f{if q as usize>=n{return None;}first[q as usize]=true;}
            for &q in &c{if q as usize>=n{return None;}continuation[q as usize]=true;}
            stored=stored.checked_add(f.len()+c.len())?;if stored>8_000_000{return None;}
            frames.push((f,c));
        }
        previous=word;
    }
    let first_states=first.iter().filter(|&&x|x).count();
    let reset_states=continuation.iter().filter(|&&x|x).count();
    let keep=first.into_iter().zip(continuation).map(|(f,c)|f||c).collect::<Vec<_>>();
    if keep.iter().filter(|&&x|x).count()>=n{return None;}
    let footprint_ms=started.elapsed().as_secs_f64()*1000.0;
    let started=Instant::now();let view=tokenizer.induced_observation_view(&keep)?;
    let scope=scope.relocate_query_view(&view.original_to_view,view.tokenizer.num_states() as usize,
        view.tokenizer.deterministic_reset_states().into_vec()).ok()?;
    let materialize_ms=started.elapsed().as_secs_f64()*1000.0;
    Some(QueryView{view,scope,footprint_ms,materialize_ms,first_states,reset_states,state_steps})
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::automata::lexer::ast::{bytes,choice};
    use crate::automata::lexer::compile::{build_regex_monolithic,build_regex_partitioned};
    use crate::compiler::stages::id_map_and_terminal_dwa::scope::{BoundaryOwnership,InitialStateDomain,ImmediateComponentId};

    #[test]
    fn query_view_matches_independent_first_and_every_reset_factor(){
        let mut long=vec![b'q';600];long.extend_from_slice(b"ab");
        let expressions=vec![bytes(&long),choice(vec![bytes(b"!x"),bytes(b"!yz")]),bytes(b" ")];
        let words=vec![b"ab!".to_vec(),b"ab!x".to_vec(),b"a".to_vec(),b"ab".to_vec(),b"q!y".to_vec(),b"ab !".to_vec()];
        let vocab=Vocab::new(words.iter().enumerate().map(|(i,w)|(100+i as u32*7,w.clone())).collect());
        for tokenizer in [build_regex_monolithic(&expressions).into_tokenizer(3,Some(Arc::from(expressions.clone()))),
            build_regex_partitioned(&expressions,&[0,1,2]).into_tokenizer(3,Some(Arc::from(expressions.clone())))] {
            let resets=tokenizer.deterministic_reset_states().into_vec();
            let mut initial=resets.clone();
            for &b in &long[..600]{initial=tokenizer.step_all(&initial,b).into_vec();}
            assert!(!initial.is_empty());
            let mut keep=vec![false;tokenizer.num_states() as usize];for &q in &initial{keep[q as usize]=true;}
            let scope=BoundaryAnalysisScope::new(InitialStateDomain::from_mask(keep.len(),keep).unwrap(),
                resets.clone(),Arc::new(BoundaryOwnership::flat(&[0,1],3).unwrap()),ImmediateComponentId(0),true,None).unwrap();
            let q=prepare(&tokenizer,&vocab,&scope).expect("fixture must select a nontrivial query view");
            assert!(q.view.tokenizer.num_states()*10<tokenizer.num_states());
            let compare=|sources:&[u32],word:&[u8]| {
                let mut original=sources.to_vec();let mut view=sources.iter().map(|&raw|{
                    let v=q.view.original_to_view[raw as usize];assert_ne!(v,u32::MAX);v
                }).collect::<Vec<_>>();
                for &byte in word {
                    original=tokenizer.step_all(&original,byte).into_vec();
                    view=q.view.tokenizer.step_all(&view,byte).into_vec();
                    let mut decoded=view.iter().map(|&v|q.view.view_to_original[v as usize]).collect::<Vec<_>>();
                    decoded.sort_unstable();decoded.dedup();assert_eq!(decoded,original);
                }
            };
            for word in &words{
                compare(&initial,word);
                for start in 0..word.len(){for end in start+1..=word.len(){compare(&resets,&word[start..end]);}}
            }
            let empty=Vocab::new(vec![(99,vec![])]);assert!(prepare(&tokenizer,&empty,&scope).is_none());
        }
    }
}

/// Uses only the direct table of this exact immutable source, as built by the
/// caller at the link's disjoint-union boundary. A declined optimization uses
/// the original traversal and its original resource/coverage decisions.
pub(super) fn prepare_with_policy(
    tokenizer:&Tokenizer,vocab:&Vocab,scope:&BoundaryAnalysisScope,flat:Option<&[u32]>,
)->Option<QueryView>{
    if crate::compiler::composition::boundary::env::enabled("GLRMASK_BOUNDARY_FACTORED_QUERY_VIEW") {
        if let Some(candidate)=flat.and_then(|flat|fast::prepare(tokenizer,vocab,scope,flat)) {
            if std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_FACTORED_QUERY_VIEW").is_some(){
                let reference=prepare(tokenizer,vocab,scope).expect("factored query view changed reference eligibility");
                assert_eq!(candidate.view.original_to_view,reference.view.original_to_view);
                assert_eq!(candidate.scope.initial_states().keep_raw(),reference.scope.initial_states().keep_raw());
                assert_eq!(candidate.scope.reset_states(),reference.scope.reset_states());
                assert_eq!((candidate.first_states,candidate.reset_states,candidate.state_steps),
                    (reference.first_states,reference.reset_states,reference.state_steps));
                assert_eq!(crate::automata::lexer::tokenizer::artifact_serde::to_fast_bytes(&candidate.view.tokenizer),
                    crate::automata::lexer::tokenizer::artifact_serde::to_fast_bytes(&reference.view.tokenizer));
                eprintln!("[glrmask/validate][factored_query_view] exact=true states={}",candidate.view.tokenizer.num_states());
            }
            return Some(candidate);
        }
    }
    prepare(tokenizer,vocab,scope)
}

pub(super) struct BorrowedQuery {
    pub original_to_view:Vec<u32>,pub view_to_original:Vec<u32>,
    pub scope:BoundaryAnalysisScope,pub footprint_ms:f64,pub materialize_ms:f64,
    pub first_states:usize,pub reset_states:usize,pub state_steps:usize,
}
pub(super) fn prepare_borrowed(tokenizer:&Tokenizer,vocab:&Vocab,scope:&BoundaryAnalysisScope,flat:&[u32])->Option<BorrowedQuery>{
    let footprint=fast::footprint(tokenizer,vocab,scope,flat)?;
    let began=Instant::now();let n=tokenizer.num_states()as usize;let mut keep=footprint.keep;
    let closures=tokenizer.all_singleton_epsilon_closures();let start=tokenizer.initial_state_id();
    keep[start as usize]=true;let mut closure_work=0usize;
    for q in 0..n{if keep[q]{closure_work=closure_work.checked_add(closures[q].len())?;if closure_work>8_000_000{return None;}for &r in &closures[q]{*keep.get_mut(r as usize)?=true;}}}
    let mut inverse=vec![start];let mut mapping=vec![u32::MAX;n];mapping[start as usize]=0;
    for(q,&yes)in keep.iter().enumerate(){if yes&&q!=start as usize{mapping[q]=inverse.len()as u32;inverse.push(q as u32);}}
    if inverse.len()>=16384{return None;}
    let reset=scope.reset_states().iter().map(|&q|mapping.get(q as usize).copied().filter(|&q|q!=u32::MAX)).collect::<Option<Vec<_>>>()?;
    let scoped=scope.relocate_query_view(&mapping,inverse.len(),reset).ok()?;
    Some(BorrowedQuery{original_to_view:mapping,view_to_original:inverse,scope:scoped,
        footprint_ms:footprint.footprint_ms,materialize_ms:began.elapsed().as_secs_f64()*1000.,first_states:footprint.first_states,reset_states:footprint.reset_states,state_steps:footprint.state_steps})
}
