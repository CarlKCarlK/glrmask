//! Physical compaction of a positive, already-supported NWA.
//!
//! Zero-weight-unreachable states cannot enter a weighted determinized support.
//! Retain every reachable state's complete explicit label-key set: empty keys
//! still guard DEFAULT normalization. The live state map is injective, unlike
//! suffix hash-consing, so distinct productive targets do not become identical.
use glrmask_weighted_automata::weighted_u32::nwa::{NWA,NWAState};

#[derive(Debug,Default)]
pub struct Stats {pub before_states:usize,pub after_states:usize,pub before_edges:usize,pub after_edges:usize,pub empty_label_guards:usize}

pub fn compact(input:&NWA)->Option<(NWA,Stats)>{
    let n=input.states().len();if n==0||n>200_000{return None;}
    let mut live=vec![false;n];let mut todo=Vec::new();let mut visited_edges=0usize;
    for &start in input.start_states(){if start as usize>=n{return None;}todo.push(start);}
    while let Some(q)=todo.pop(){
        if live[q as usize]{continue;}live[q as usize]=true;
        let state=&input.states()[q as usize];
        // Label validity is checked by the caller's positive normalizer.
        for &(target,ref weight) in state.epsilons.iter().chain(state.transitions.values().flatten()){
            visited_edges=visited_edges.checked_add(1)?;if visited_edges>4_000_000||target as usize>=n{return None;}
            if !weight.is_empty(){todo.push(target);}
        }
    }
    let mut map=vec![u32::MAX;n];let mut states=Vec::new();
    for q in 0..n{if live[q]{map[q]=states.len() as u32;states.push(NWAState::default());}}
    if states.is_empty(){return None;}
    let mut stats=Stats{before_states:n,after_states:states.len(),before_edges:input.num_transitions(),..Default::default()};
    for q in 0..n{if !live[q]{continue;}
        let row=&input.states()[q];let output=&mut states[map[q] as usize];
        output.final_weight=row.final_weight.clone();
        for &(target,ref weight) in &row.epsilons{
            if weight.is_empty(){continue;}
            debug_assert_ne!(map[target as usize],u32::MAX);
            output.epsilons.push((map[target as usize],weight.clone()));stats.after_edges+=1;
        }
        for (&label,branches) in &row.transitions{
            // A key with no nonzero branches is deliberately retained.
            let output_branches=output.transitions.entry(label).or_default();
            for &(target,ref weight) in branches{
                if weight.is_empty(){continue;}
                debug_assert_ne!(map[target as usize],u32::MAX);
                output_branches.push((map[target as usize],weight.clone()));stats.after_edges+=1;
            }
            stats.empty_label_guards+=usize::from(output_branches.is_empty());
        }
    }
    Some((NWA::from_parts(states,input.start_states().iter().map(|&q|map[q as usize]).collect()),stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use glrmask_weight::__private::Weight;
    use glrmask_parser_dwa::__private::{parser_dwa::normalize_weighted_parser_stack_nwa_for_parser_state_count,parser_equivalence::compare_parser_mask_prefix_languages};
    const DEFAULT:i32=2147483646;
    fn w(bits:usize)->Weight{Weight::from_token_set_for_tsid(0,(0..4u32).filter(|&b|bits&(1<<b)!=0).collect())}
    #[test]
    fn positive_trim_preserves_all_normalized_prefix_masks_including_default_guards(){
        let mut seed=83u64;let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32) as usize};
        let mut removed=0usize;let mut guards=0usize;
        for case in 0..512{
            let n=4+next()%8;let mut source=NWA::new(1,4);for _ in 0..n{source.add_state();}source.set_start_states(vec![0]);
            for q in 0..n{
                if next()%3==0{source.set_final_weight(q as u32,w(next()%16));}
                for target in q+1..n{
                    let choice=next()%7;let weight=if next()%2==0{Weight::empty()}else if next()%5==0{Weight::all()}else{w(next()%16)};
                    if choice==0{source.add_epsilon(q as u32,target as u32,weight);}
                    else if choice<=4{source.add_transition(q as u32,if choice==4{DEFAULT}else{choice as i32-1},target as u32,weight);}
                }
                if next()%4==0{source.states_mut()[q].transitions.entry(3).or_default();}
            }
            let before=source.clone();let (candidate,stats)=compact(&source).unwrap();
            assert_eq!(before.states(),source.states());removed+=stats.before_states-stats.after_states;guards+=stats.empty_label_guards;
            let left=normalize_weighted_parser_stack_nwa_for_parser_state_count(4,&source);
            let right=normalize_weighted_parser_stack_nwa_for_parser_state_count(4,&candidate);
            let check=compare_parser_mask_prefix_languages(&left,&right,4,100_000).unwrap();
            assert!(check.difference.is_none(),"case {case}: {:?}",check.difference);
        }
        assert!(removed>0&&guards>0);
    }
    #[test]
    fn positive_trim_keeps_nonzero_self_cycles_and_declines_invalid_reachable_edges(){
        let mut source=NWA::new(1,4);source.add_state();source.add_state();source.set_start_states(vec![0]);
        source.add_epsilon(0,0,Weight::all());source.add_transition(0,2,1,Weight::empty());
        let (candidate,stats)=compact(&source).unwrap();assert_eq!(stats.after_states,1);
        assert_eq!(candidate.states()[0].epsilons.len(),1);assert!(candidate.states()[0].transitions[&2].is_empty());
        source.add_epsilon(0,99,Weight::all());assert!(compact(&source).is_none());
    }
}
