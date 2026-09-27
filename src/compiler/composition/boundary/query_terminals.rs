//! Conservative terminal observations for a bounded model-token query.
//! Permit reset at every byte cut, and ignore follows/longest-match pruning.
//! Every real terminal path is included; this helper only removes terminals
//! with no observation in that superset. Unsupported/budgeted cases decline.
use crate::automata::lexer::tokenizer::{Lexer,Tokenizer};
use crate::Vocab;
pub(super) fn query_support(tok:&Tokenizer,vocab:&Vocab,seeds:&[bool])->Option<(Vec<bool>,usize)>{
    let n=tok.num_states() as usize;let nt=tok.num_terminals() as usize;
    if seeds.len()!=n||n>200_000||nt==0||tok.has_virtual_residual_runtime(){return None;}
    let roots=tok.deterministic_reset_states().into_vec();
    for &q in &roots{for source in tok.singleton_epsilon_closure(q){if tok.matched_terminals_iter(source).next().is_some(){return None;}}}
    let mut words=vocab.entries_map().values().map(Vec::as_slice).collect::<Vec<_>>();
    if words.iter().any(|w|w.is_empty())||words.iter().try_fold(0usize,|a,w|a.checked_add(w.len()))?>262_144{return None;}
    words.sort_unstable();words.dedup();
    let mut initial=seeds.iter().enumerate().filter_map(|(q,&keep)|keep.then_some(q as u32)).collect::<Vec<_>>();
    initial.extend_from_slice(&roots);initial.sort_unstable();initial.dedup();
    let mut keep=vec![false;nt];
    // A pending terminal may finalize before the first consumed byte. Include
    // those observations conservatively; an endpoint after reset still needs
    // a byte (nullable reset terminals were rejected above).
    for &q in &initial { for source in tok.singleton_epsilon_closure(q) {
        for t in tok.matched_terminals_iter(source) { *keep.get_mut(t as usize)?=true; }
    }}
    let mut stored=initial.len();
    let mut frames=vec![initial];let mut previous:&[u8]=&[];let mut work=0usize;
    for word in words{
        let lcp=previous.iter().zip(word).take_while(|(a,b)|a==b).count();for frame in &frames[lcp+1..]{stored-=frame.len();}frames.truncate(lcp+1);
        for (position,&byte) in word.iter().enumerate().skip(lcp){
            work=work.checked_add(frames.last()?.len())?;if work>12_000_000{return None;}
            let mut next=tok.step_all(frames.last()?,byte).into_vec();
            for &q in &next{if q as usize>=n{return None;}for t in tok.matched_terminals_iter(q){*keep.get_mut(t as usize)?=true;}}
            // Every prefix can itself be a model token. Preserve its endpoint
            // observations now, before adding the optional reset frontier.
            if position+1==word.len(){for &q in &next{for t in tok.possible_future_terminals_iter(q){*keep.get_mut(t as usize)?=true;}}}
            next.extend_from_slice(&roots);next.sort_unstable();next.dedup();stored=stored.checked_add(next.len())?;if stored>8_000_000{return None;}frames.push(next);
        }
        // When an identical-prefix leaf was already processed for a longer
        // word, its post-byte frontier must still contribute future outputs.
        // Roots add only a conservative superset in this rare reused leaf.
        if lcp==word.len(){for &q in frames.last()?{for t in tok.possible_future_terminals_iter(q){*keep.get_mut(t as usize)?=true;}}}
        previous=word;
    }
    Some((keep,work))
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::automata::lexer::{ast::{bytes,choice,plus},compile::build_regex_partitioned};
    #[test]
    fn query_terminals_cover_direct_independent_reset_alignment_scans(){
        let expressions=vec![bytes(b"abc"),plus(bytes(b"q")),choice(vec![bytes(b"!z"),bytes(b"!x")]),bytes(b"unobservable")];
        let tok=build_regex_partitioned(&expressions,&[0,1,2,3]).into_tokenizer(4,None);
        let seeds=(0..tok.num_states()).map(|q|tok.possible_future_terminals_iter(q).any(|t|t==0)).collect::<Vec<_>>();
        let vocab=Vocab::new(vec![(0,b"bc!".to_vec()),(1,b"bc!zq".to_vec()),(2,b"bq".to_vec()),(99,b"bc!".to_vec())]);
        let (keep,_)=query_support(&tok,&vocab,&seeds).unwrap();let mut expected=vec![false;4];
        for word in vocab.entries_map().values(){
            for cut in 0..word.len(){
                let mut current=tok.deterministic_reset_states().into_vec();
                if cut==0{current.extend(seeds.iter().enumerate().filter_map(|(q,&yes)|yes.then_some(q as u32)));}
                for (index,&byte) in word.iter().enumerate().skip(cut){
                    current=tok.step_all(&current,byte).into_vec();for &q in &current{
                        for t in tok.matched_terminals_iter(q){expected[t as usize]=true;}
                        if index+1==word.len(){for t in tok.possible_future_terminals_iter(q){expected[t as usize]=true;}}
                    }
                }
            }
        }
        assert_eq!(keep,expected);assert!(!keep[3]);
    }
}
