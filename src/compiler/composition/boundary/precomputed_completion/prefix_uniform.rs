//! Cheap specialization of the existing, exact prepared prefix-seed query.
//!
//! PRECONDITION: the caller holds the same immutable disjoint-union injection
//! certificate as production PreparedSourceSpan. State and terminal offsets
//! MUST come from that union, not a guessed or merely hash-matched lexer.
//! This changes only how local ownership is proved. No token is admitted here.
use super::completion_index::CompletionContext;
use glrmask_lexer::__private::automata::lexer::{Lexer, tokenizer::Tokenizer};
use glrmask_vocab::Vocab;
use glrmask_terminal_dwa::__private::terminal_dwa::scope::{BoundaryOwnership, ImmediateComponentId};

pub(crate) fn in_verified_uniform_span(
    context: &CompletionContext, state_offset: u32, terminal_offset: u32,
    merged: &Tokenizer, vocab: &Vocab, initial: &[bool],
    ownership: &BoundaryOwnership, owner: ImmediateComponentId,
) -> Option<(Vec<bool>, bool)> {
    let n = merged.num_states() as usize;
    let off = state_offset as usize;
    let end = off.checked_add(context.raw_states())?;
    let terminal_end = terminal_offset.checked_add(context.source().num_terminals())?;
    if vocab.is_empty() || vocab.entries_map().values().any(Vec::is_empty)
        || initial.len() != n || end > n
        || ownership.num_terminals() != merged.num_terminals() as usize
        || owner.0 >= ownership.num_immediate_components()
        || terminal_end as usize > ownership.num_terminals()
        || merged.has_virtual_residual_runtime()
        || initial[..off].iter().any(|x| *x) || initial[end..].iter().any(|x| *x)
    { return None; }
    // Exact disjoint injection carries every observation from a local terminal
    // ID into this terminal interval. If the entire interval belongs to owner,
    // no state/epsilon-frontier can have the reference's foreign entry event.
    // All labels and all transitions remain unchanged; no graph is quotiented.
    let uniform = (terminal_offset..terminal_end)
        .all(|t| ownership.owner_of_terminal(t) == Some(owner));
    if !uniform {
        return super::prefix_span_support::in_verified_span(
            context, state_offset, merged, vocab, initial, ownership, owner,
        ).map(|mask| (mask, false));
    }
    let words = vocab.entries_map().values().cloned().collect::<Vec<_>>();
    let local = context.prefix_support(&initial[off..end], &words).ok()?;
    let mut result = vec![false; n];
    result[off..end].copy_from_slice(&local);
    Some((result, true))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use super::super::{prefix_observer::{PrefixObserver, Limits, Profile}, completion_index::UntrustedCompletionIndex};
    use glrmask_lexer::__private::automata::lexer::{
        ast::{bytes, choice, plus, Expr}, compile::{build_regex_monolithic, build_regex_partitioned},
    };
    use glrmask_terminal_dwa::__private::terminal_dwa::{scope, l1};
    fn prepare(source: Arc<Tokenizer>, words: &[Vec<u8>]) -> CompletionContext {
        let prefix = PrefixObserver::prepare(source.as_ref(), words, Limits::default(), &mut Profile::default())
            .unwrap().certify(source.as_ref()).unwrap();
        let wire = UntrustedCompletionIndex::from_certified_prefix(&prefix).unwrap().to_bytes().unwrap();
        CompletionContext::load(source, &wire).unwrap()
    }
    #[test]
    fn uniform_spans_match_original_byte_scanner_for_offsets_and_subsets() {
        let exprs = vec![choice(vec![bytes(b"ab"), bytes(b"axb"), bytes(b"ba")]), plus(bytes(b"a")), Expr::Epsilon];
        let mono = build_regex_monolithic(&exprs).into_tokenizer(3, None);
        let partitioned = build_regex_partitioned(&exprs, &[0,1,2]).into_tokenizer(3, None);
        let (nested, _) = Tokenizer::disjoint_union_with_terminal_offsets(&[(&mono, 0), (&partitioned, 3)]);
        let mut rng = 482301u64;
        let mut next = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); (rng >> 32) as usize };
        let mut cases = 0;
        for source in [mono, partitioned, nested] {
            let source = Arc::new(source);
            let mut words = vec![b"ab!".to_vec(), b"a".to_vec(), b"z".to_vec()];
            for _ in 0..96 { words.push((0..1+next()%7).map(|_| b"abxz!"[next()%5]).collect()); }
            words.sort(); words.dedup();
            let ctx = prepare(Arc::clone(&source), &words);
            let other = build_regex_monolithic(&[bytes(b"!"), Expr::Epsilon]).into_tokenizer(2, None);
            for first in [false, true] {
                let st = source.num_terminals();
                let (merged, offsets, owner, term) = if first {
                    let (m, o) = Tokenizer::disjoint_union_with_terminal_offsets(&[(source.as_ref(),0), (&other,st)]);
                    (m, o, 0u32, 0u32)
                } else {
                    let (m, o) = Tokenizer::disjoint_union_with_terminal_offsets(&[(&other,0),(source.as_ref(),2)]);
                    (m, o, 1u32, 2u32)
                };
                let ownership = BoundaryOwnership::flat(&[0, if first {st} else {2}], st+2).unwrap();
                let flat = l1::build_flat_transition_table(&merged);
                let off = offsets[owner as usize];
                for _ in 0..128 {
                    let mut entries = words.iter().enumerate().filter(|_|next()%3!=0)
                        .map(|(i,w)|(i as u32,w.clone())).collect::<Vec<_>>();
                    if entries.is_empty() { entries.push((900, words[0].clone())); }
                    let vocab = Vocab::new(entries);
                    let mut seeds = vec![false; merged.num_states() as usize];
                    for q in off..off+source.num_states() {seeds[q as usize]=next()%3!=0;}
                    let (actual, fast) = in_verified_uniform_span(&ctx,off,term,&merged,&vocab,&seeds,&ownership,ImmediateComponentId(owner)).unwrap();
                    assert!(fast);
                    let (raw, _) = scope::crossing_prefix_seed_support(&merged,&vocab,&flat,&ownership,ImmediateComponentId(owner)).unwrap();
                    let expected = raw.iter().zip(&seeds).map(|(a,b)| *a && *b).collect::<Vec<_>>();
                    assert_eq!(actual, expected);
                    assert_eq!(actual, super::super::prefix_span_support::in_verified_span(&ctx,off,&merged,&vocab,&seeds,&ownership,ImmediateComponentId(owner)).unwrap());
                    cases += 1;
                }
            }
        }
        assert_eq!(cases,768);
    }
    #[test]
    fn mixed_ownership_falls_back_and_invalid_domains_decline() {
        let tok = Arc::new(build_regex_partitioned(&[bytes(b"aa"),bytes(b"b!"),Expr::Epsilon], &[0,1,2]).into_tokenizer(3,None));
        let words = vec![b"a".to_vec(),b"b!".to_vec(),b"z".to_vec()];
        let ctx = prepare(Arc::clone(&tok),&words);
        let vocab = Vocab::new(words.iter().enumerate().map(|(i,w)|(i as u32,w.clone())).collect());
        let ownership = BoundaryOwnership::flat(&[0,1],3).unwrap();
        let seeds = vec![true; tok.num_states() as usize];
        for owner in 0..2 {
            let(actual,fast)=in_verified_uniform_span(&ctx,0,0,&tok,&vocab,&seeds,&ownership,ImmediateComponentId(owner)).unwrap();
            assert!(!fast);
            let(raw,_)=scope::crossing_prefix_seed_support(&tok,&vocab,&l1::build_flat_transition_table(&tok),&ownership,ImmediateComponentId(owner)).unwrap();
            assert_eq!(actual,raw);
        }
        let owner=ImmediateComponentId(0);
        assert!(in_verified_uniform_span(&ctx,u32::MAX,0,&tok,&vocab,&seeds,&ownership,owner).is_none());
        assert!(in_verified_uniform_span(&ctx,0,u32::MAX,&tok,&vocab,&seeds,&ownership,owner).is_none());
        assert!(in_verified_uniform_span(&ctx,0,0,&tok,&vocab,&[true],&ownership,owner).is_none());
        assert!(in_verified_uniform_span(&ctx,0,0,&tok,&Vocab::new(vec![(99,vec![])]),&seeds,&ownership,owner).is_none());
        assert!(in_verified_uniform_span(&ctx,0,0,&tok,&Vocab::new(vec![(99,b"uncovered".to_vec())]),&seeds,&ownership,owner).is_none());
    }
}
