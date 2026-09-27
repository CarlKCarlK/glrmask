//! Exact query-view reachability using an already source-certified direct
//! transition table. The selected raw states, induced tokenizer and its ID
//! layout must match `boundary_query_view::prepare`; only the traversal changes.
use super::QueryView;
use crate::automata::lexer::tokenizer::{Lexer, Tokenizer, SingletonEpsilonClosures};
use crate::compiler::stages::id_map_and_terminal_dwa::scope::BoundaryAnalysisScope;
use crate::Vocab;
use std::{sync::Arc, time::Instant};

struct Stepper<'a> {
    flat: &'a [u32],
    closures: Arc<SingletonEpsilonClosures>,
    marks: Vec<u32>,
    epoch: u32,
}
impl Stepper<'_> {
    fn step(&mut self, states: &[u32], byte: u8, extra: &[u32]) -> Option<Vec<u32>> {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 { self.marks.fill(0); self.epoch = 1; }
        let stamp = self.epoch;
        let mut out = Vec::with_capacity(states.len().min(4096) + extra.len());
        for &q in extra {
            let mark = self.marks.get_mut(q as usize)?;
            if *mark != stamp { *mark = stamp; out.push(q); }
        }
        // `states` and `extra` are epsilon-closed. Byte movement followed by
        // target closure is exactly the ordinary NFA step, with one dedup pass.
        for &q in states {
            let target = *self.flat.get(q as usize * 256 + byte as usize)?;
            if target == u32::MAX { continue; }
            for &next in self.closures.get(target as usize)?.iter() {
                let mark = self.marks.get_mut(next as usize)?;
                if *mark != stamp { *mark = stamp; out.push(next); }
            }
        }
        out.sort_unstable();
        Some(out)
    }
}

// Both inputs are sorted and unique. This counts the reference traversal's
// logical work without allocating its repeated reset-union vector.
fn union_len(a: &[u32], b: &[u32]) -> usize {
    let (mut i, mut j, mut overlap) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => { overlap += 1; i += 1; j += 1; }
        }
    }
    a.len() + b.len() - overlap
}

/// `flat` is the table constructed from this same immutable tokenizer by the
/// boundary linker. Never pass a table belonging to a different source/view.
pub(super) struct Footprint {pub keep:Vec<bool>,pub footprint_ms:f64,pub first_states:usize,pub reset_states:usize,pub state_steps:usize}
pub(super) fn prepare(
    tokenizer: &Tokenizer, vocab: &Vocab, scope: &BoundaryAnalysisScope, flat: &[u32],
) -> Option<QueryView> {
    let Footprint{keep,footprint_ms,first_states,reset_states,state_steps}=footprint(tokenizer,vocab,scope,flat)?;
    let began = Instant::now();
    let view = tokenizer.induced_observation_view(&keep)?;
    let scope = scope.relocate_query_view(&view.original_to_view, view.tokenizer.num_states() as usize,
        view.tokenizer.deterministic_reset_states().into_vec()).ok()?;
    Some(QueryView { view, scope, footprint_ms,
        materialize_ms: began.elapsed().as_secs_f64()*1000.0,
        first_states, reset_states, state_steps })
}
pub(super) fn footprint(
    tokenizer: &Tokenizer, vocab: &Vocab, scope: &BoundaryAnalysisScope, flat: &[u32],
) -> Option<Footprint> {
    let n = tokenizer.num_states() as usize;
    if n < 256 || n > 200_000 || tokenizer.has_virtual_residual_runtime()
        || !scope.require_crossing() || flat.len() != n.checked_mul(256)?
        || scope.initial_states().keep_raw().len() != n
    { return None; }
    let began = Instant::now();
    let mut words = vocab.entries_map().values().map(Vec::as_slice).collect::<Vec<_>>();
    if words.is_empty() || words.iter().any(|w| w.is_empty() || w.len() > 1024)
        || words.iter().try_fold(0usize, |a, w| a.checked_add(w.len()))? > 262_144
    { return None; }
    words.sort_unstable(); words.dedup();
    let closures = tokenizer.all_singleton_epsilon_closures();
    let mut first = scope.initial_states().keep_raw().to_vec();
    for q in 0..n {
        if first[q] {
            for &r in closures.get(q)?.iter() { *first.get_mut(r as usize)? = true; }
        }
    }
    let initial = first.iter().enumerate().filter_map(|(q, &on)| on.then_some(q as u32)).collect::<Vec<_>>();
    let mut reset = tokenizer.deterministic_reset_states().into_vec();
    reset.sort_unstable(); reset.dedup();
    let mut continuation = vec![false; n];
    for &q in &reset {
        for &r in closures.get(q as usize)?.iter() { *continuation.get_mut(r as usize)? = true; }
    }
    let closed_reset = continuation.iter().enumerate().filter_map(|(q, &on)| on.then_some(q as u32)).collect::<Vec<_>>();
    let mut stepper = Stepper { flat, closures, marks: vec![0; n], epoch: 0 };
    let mut reset_next = (0..256).map(|_| None).collect::<Vec<Option<Vec<u32>>>>();
    let mut frames = vec![(initial, Vec::<u32>::new())];
    let mut stored = frames[0].0.len();
    let mut previous: &[u8] = &[];
    let mut state_steps = 0usize;
    for word in words {
        let common = previous.iter().zip(word).take_while(|(a, b)| a == b).count();
        for (a,b) in &frames[common+1..] { stored -= a.len() + b.len(); }
        frames.truncate(common + 1);
        for &byte in &word[common..] {
            let (old_first, old_reset) = frames.last()?;
            // Preserve the old resource-admission decision and diagnostic work
            // definition, even though reset work is now performed only once.
            state_steps = state_steps.checked_add(old_first.len() + union_len(old_reset, &reset))?;
            if state_steps > 30_000_000 { return None; }
            if reset_next[byte as usize].is_none() {
                reset_next[byte as usize] = Some(stepper.step(&closed_reset, byte, &[])?);
            }
            let f = stepper.step(old_first, byte, &[])?;
            let c = stepper.step(old_reset, byte, reset_next[byte as usize].as_ref()?)?;
            for &q in &f { first[q as usize] = true; }
            for &q in &c { continuation[q as usize] = true; }
            stored = stored.checked_add(f.len() + c.len())?;
            if stored > 8_000_000 { return None; }
            frames.push((f,c));
        }
        previous = word;
    }
    let first_states = first.iter().filter(|&&x| x).count();
    let reset_states = continuation.iter().filter(|&&x| x).count();
    let keep = first.into_iter().zip(continuation).map(|(a,b)| a||b).collect::<Vec<_>>();
    if keep.iter().filter(|&&x| x).count() >= n { return None; }
    let footprint_ms = began.elapsed().as_secs_f64() * 1000.0;
    Some(Footprint{keep,footprint_ms,first_states,reset_states,state_steps})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::{ast::{bytes,choice,plus},compile::{build_regex_monolithic,build_regex_partitioned}};
    use glrmask_terminal_dwa::__private::terminal_dwa::{l1,scope::{BoundaryOwnership,InitialStateDomain,ImmediateComponentId}};
    #[test]
    fn factored_reset_steps_preserve_complete_query_views() {
        let exprs = vec![bytes(&vec![b'q';700]), choice(vec![bytes(b"ab"),bytes(b"a!"),bytes(b"ba")]),plus(bytes(b" ")),bytes(b"!")];
        let mut seed=792713u64;
        let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32) as usize};
        let mut checked=0;
        for partitioned in [false,true] {
            let tok=if partitioned{build_regex_partitioned(&exprs,&[0,1,2,3])}else{build_regex_monolithic(&exprs)}.into_tokenizer(4,None);
            let flat=l1::build_flat_transition_table(&tok);
            for case in 0..64 {
                let mut keep=(0..tok.num_states()).map(|_|next()%3==0).collect::<Vec<_>>();
                keep[tok.initial_state_id() as usize]=true;
                let scope=BoundaryAnalysisScope::new(InitialStateDomain::from_mask(keep.len(),keep).unwrap(),
                    tok.deterministic_reset_states().into_vec(),Arc::new(BoundaryOwnership::flat(&[0,2],4).unwrap()),ImmediateComponentId((case%2) as u32),true,None).unwrap();
                let mut entries=vec![(0,b"ab!".to_vec()),(13,b"ba".to_vec()),(47,b"ab!".to_vec())];
                for i in 0..24 {let n=1+next()%12;entries.push((100+i*11,(0..n).map(|_|b"ab! q"[next()%5]).collect()));}
                let vocab=Vocab::new(entries);
                let a=super::super::prepare(&tok,&vocab,&scope);
                let b=prepare(&tok,&vocab,&scope,&flat);
                assert_eq!(a.is_some(),b.is_some(),"case={case} partitioned={partitioned}");
                if let (Some(a),Some(b))=(a,b) {
                    assert_eq!(a.view.original_to_view,b.view.original_to_view);
                    assert_eq!(a.scope.initial_states().keep_raw(),b.scope.initial_states().keep_raw());
                    assert_eq!(a.scope.reset_states(),b.scope.reset_states());
                    assert_eq!((a.first_states,a.reset_states,a.state_steps),(b.first_states,b.reset_states,b.state_steps));
                    assert_eq!(crate::automata::lexer::tokenizer::artifact_serde::to_fast_bytes(&a.view.tokenizer),
                        crate::automata::lexer::tokenizer::artifact_serde::to_fast_bytes(&b.view.tokenizer));
                    checked+=1;
                }
                assert!(prepare(&tok,&vocab,&scope,&flat[..flat.len()-1]).is_none());
            }
        }
        assert!(checked>=64,"insufficient accepted-view coverage: {checked}");
    }
}
