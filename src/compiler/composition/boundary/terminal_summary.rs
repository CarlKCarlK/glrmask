//! Read-only exact support aggregation for boundary terminal automata.
//!
//! At each vertex, union incoming path weights only after all predecessors
//! have been visited. Distributivity of intersection over union gives exactly
//! the original forward recurrence. In particular, TSID/token correlations
//! survive each edge intersection; projecting tokens before that is unsound.
use crate::automata::weighted_u32::dwa::DWA;
use crate::compiler::stages::equiv_types::InternalIdMap;
use crate::ds::weight::{ScopedWeightOpCache, Weight};
use std::{collections::{BTreeSet, VecDeque}, time::Instant};

pub(super) fn accepted_weight(dwa: &DWA) -> Option<Weight> {
    const MAX_STATES: usize = 100_000;
    const MAX_EDGES: usize = 1_000_000;
    let n = dwa.states().len();
    if n == 0 || n > MAX_STATES || dwa.start_state() as usize >= n { return None; }
    let mut degree = vec![0usize; n];
    let mut edges = 0usize;
    for state in dwa.states() {
        edges = edges.checked_add(state.transitions.len())?;
        if edges > MAX_EDGES { return None; }
        for &(target, _) in state.transitions.values() {
            *degree.get_mut(target as usize)? += 1;
        }
    }
    let mut queue: VecDeque<_> = degree.iter().enumerate()
        .filter_map(|(i, &d)| (d == 0).then_some(i)).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(q) = queue.pop_front() {
        order.push(q);
        for &(target, _) in dwa.states()[q].transitions.values() {
            degree[target as usize] -= 1;
            if degree[target as usize] == 0 { queue.push_back(target as usize); }
        }
    }
    // Do not partially summarize a cyclic/malformed input. The original
    // implementation remains responsible for unsupported cases.
    if order.len() != n { return None; }
    let mut incoming = vec![Vec::<Weight>::new(); n];
    incoming[dwa.start_state() as usize].push(Weight::all());
    let mut accepting = Vec::<Weight>::new();
    let mut ops = ScopedWeightOpCache::default();
    for q in order {
        let current = ops.union_all(incoming[q].iter());
        incoming[q].clear();
        if current.is_empty() { continue; }
        let state = &dwa.states()[q];
        if let Some(weight) = &state.final_weight {
            let support = ops.intersection(&current, weight);
            if !support.is_empty() { accepting.push(support); }
        }
        for &(target, ref weight) in state.transitions.values() {
            let support = ops.intersection(&current, weight);
            if !support.is_empty() { incoming[target as usize].push(support); }
        }
    }
    Some(ops.union_all(accepting.iter()))
}

pub(super) fn accepted_tokens(dwa: &DWA, id_map: &InternalIdMap) -> Option<BTreeSet<u32>> {
    let profile = std::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some();
    let started = profile.then(Instant::now);
    let accepted = accepted_weight(dwa)?;
    let support_ms = started.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0);
    if std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_BATCHED_TOKEN_SUPPORT").is_some() {
        let reference = crate::compiler::composition::accepted_weight_support(dwa);
        assert_eq!(accepted, reference, "batched terminal summary changed correlated TSID/token support");
        eprintln!("[glrmask/validate][boundary_batched_token_support] exact_full_weight=true states={}", dwa.num_states());
    }
    let originals=project_accepted_tokens(&accepted,id_map);
    if let Some(t) = started {
        eprintln!("[glrmask/profile][boundary_batched_token_support] states={} edges={} tokens={} support_ms={support_ms:.3} total_ms={:.3}",
            dwa.num_states(), dwa.num_transitions(), originals.len(), t.elapsed().as_secs_f64()*1000.0);
    }
    Some(originals)
}

/// Project only AFTER correlated support has been proved by its producer.
/// This is the unchanged existing raw-TSID/token mapping loop.
pub(super) fn project_accepted_tokens(accepted:&Weight,id_map:&InternalIdMap)->BTreeSet<u32>{
    let groups = &id_map.vocab_tokens.internal_to_originals;
    let mut originals = BTreeSet::new();
    for (_, tokens) in accepted.raw_range_values() {
        for range in tokens.ranges() {
            let lo = *range.start() as usize;
            if lo >= groups.len() { continue; }
            let hi = (*range.end() as usize).min(groups.len() - 1);
            for ids in &groups[lo..=hi] { originals.extend(ids.iter().copied()); }
        }
    }
    originals
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::weighted_u32::dwa::DWAState;
    use crate::ds::weight::shared_rangeset;
    use range_set_blaze::RangeSetBlaze;
    fn finite(tsid: u32, token: u32) -> Weight {
        Weight::from_per_tsid_shared([(tsid, shared_rangeset(RangeSetBlaze::from_iter([token])))])
    }
    #[test]
    fn correlated_coordinates_must_survive_intersection() {
        let mut rows = vec![DWAState::default(); 3];
        rows[0].transitions.insert(7, (1, finite(0, 5)));
        rows[1].transitions.insert(8, (2, finite(1, 5)));
        rows[2].final_weight = Some(Weight::all());
        let graph = DWA::from_parts(rows, 0);
        assert_eq!(accepted_weight(&graph), Some(Weight::empty()));
        assert_eq!(accepted_weight(&graph), Some(crate::compiler::composition::accepted_weight_support(&graph)));
    }
    #[test]
    fn generated_dags_match_full_reference_support_and_do_not_mutate_graph() {
        let mut seed = 0x3471_6935_0519u64;
        fn random(seed: &mut u64) -> u64 { *seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1); *seed >> 32 }
        let mut palette = vec![Weight::empty(), Weight::all()];
        for q in [0,1,7,65,4096] { for t in [0,1,9,257,90001] { palette.push(finite(q,t)); } }
        palette.push(Weight::union_all(palette[2..].iter()));
        for case in 0..512 {
            let n = 1 + random(&mut seed) as usize % 20;
            let mut rows=vec![DWAState::default();n];
            for (q,row) in rows.iter_mut().enumerate() {
                if random(&mut seed)%3 != 0 { row.final_weight=Some(palette[random(&mut seed)as usize%palette.len()].clone()); }
                if q+1<n { for label in 0..(random(&mut seed)%8) as i32 {
                    let target = q+1+random(&mut seed)as usize%(n-q-1);
                    row.transitions.insert(label,(target as u32,palette[random(&mut seed)as usize%palette.len()].clone()));
                }}
            }
            let start=random(&mut seed)as usize%n;
            let graph=DWA::from_parts(rows,start as u32);
            let before=bincode::serialize(&graph).unwrap();
            let reference=crate::compiler::composition::accepted_weight_support(&graph);
            assert_eq!(accepted_weight(&graph),Some(reference),"case {case}");
            assert_eq!(bincode::serialize(&graph).unwrap(),before,"mutated case {case}");
        }
    }
    #[test]
    fn cyclic_and_invalid_graphs_decline() {
        let mut rows=vec![DWAState::default();2];
        rows[0].transitions.insert(1,(1,Weight::all()));
        rows[1].transitions.insert(2,(0,Weight::all()));
        assert!(accepted_weight(&DWA::from_parts(rows,0)).is_none());
        assert!(accepted_weight(&DWA::from_parts(Vec::new(),0)).is_none());
        assert!(accepted_weight(&DWA::from_parts(vec![DWAState::default()],1)).is_none());
    }
}
