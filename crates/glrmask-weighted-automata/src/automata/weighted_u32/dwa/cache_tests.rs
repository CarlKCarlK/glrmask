use super::*;
use crate::automata::weighted_u32::equivalence::find_difference;

fn token_weight(values: impl IntoIterator<Item = u32>) -> Weight {
    Weight::from_per_tsid_token_sets(std::iter::once((
        0,
        RangeSetBlaze::from_iter(values.into_iter().map(|value| value..=value)),
    )))
}

#[test]
fn packed_runtime_dwf5_roundtrip_preserves_direct_access() {
    let mut dwa = DWA::new(1, 1);
    let accept = dwa.add_state();
    let restricted = token_weight([3, 4, 9]);
    dwa.add_transition(0, 7, accept, restricted.clone());
    dwa.add_transition(0, i32::MAX - 1, accept, Weight::all());
    dwa.set_final_weight(accept, restricted);
    let dwa = dwa.share_exact_transition_rows_owned();

    let packed = PackedRuntimeDwa::from_dwa(&dwa).unwrap();
    assert!(packed.owned_narrow.is_none());
    let wire = packed.fast_wire_bytes();
    assert!(wire.starts_with(b"DWF5"));
    let loaded = PackedRuntimeDwa::from_fast_wire_bytes(&wire).unwrap();

    assert_eq!(loaded.start_state(), packed.start_state());
    assert_eq!(loaded.state_count(), packed.state_count());
    assert_eq!(loaded.token_set_count(), packed.token_set_count());
    assert_eq!(loaded.weight_count(), packed.weight_count());

    let (target, weight) = loaded.transition(0, 7).unwrap();
    assert_eq!(target, accept);
    assert!(!weight.is_full());
    let tokens = weight.token_set_for_tsid(0).unwrap();
    let mut ranges = Vec::new();
    tokens.for_each_range(|start, end| ranges.push((start, end)));
    assert_eq!(ranges, vec![(3, 4), (9, 9)]);

    let (default_target, default_weight) = loaded.transition(0, i32::MAX - 1).unwrap();
    assert_eq!(default_target, accept);
    assert!(default_weight.is_full());

    let final_weight = loaded.final_weight(accept).unwrap();
    let final_tokens = final_weight.token_set_for_tsid(0).unwrap();
    let mut final_ranges = Vec::new();
    final_tokens.for_each_range(|start, end| final_ranges.push((start, end)));
    assert_eq!(final_ranges, vec![(3, 4), (9, 9)]);
}

#[test]
fn packed_runtime_transition_token_set_ids_exclude_final_only_sets() {
    let mut dwa = DWA::new(1, 1);
    let accept = dwa.add_state();
    let transition_weight = token_weight([3, 4, 9]);
    let final_weight = token_weight([20, 21, 40]);
    dwa.add_transition(0, 7, accept, transition_weight);
    dwa.set_final_weight(accept, final_weight);
    let dwa = dwa.share_exact_transition_rows_owned();

    let packed = PackedRuntimeDwa::from_dwa(&dwa).unwrap();
    let wire = packed.fast_wire_bytes();
    let loaded = PackedRuntimeDwa::from_fast_wire_bytes(&wire).unwrap();

    let transition_id = loaded
        .transition(0, 7)
        .unwrap()
        .1
        .token_set_for_tsid(0)
        .unwrap()
        .id();
    let final_id = loaded
        .final_weight(accept)
        .unwrap()
        .token_set_for_tsid(0)
        .unwrap()
        .id();
    assert_ne!(transition_id, final_id);

    let transition_ids = loaded.transition_token_set_ids();
    assert!(transition_ids.contains(&transition_id));
    assert!(!transition_ids.contains(&final_id));
}

#[test]
fn packed_runtime_supports_more_than_65536_distinct_token_sets() {
    const TOKEN_SET_COUNT: u32 = 65_537;
    let weight = Weight::from_per_tsid_token_sets((0..TOKEN_SET_COUNT).map(|tsid| {
        (
            tsid,
            RangeSetBlaze::from_iter(std::iter::once(tsid..=tsid)),
        )
    }));
    let mut dwa = DWA::new(TOKEN_SET_COUNT, TOKEN_SET_COUNT);
    let accept = dwa.add_state();
    dwa.add_transition(0, 7, accept, weight);
    let dwa = dwa.share_exact_transition_rows_owned();

    let packed = PackedRuntimeDwa::from_dwa(&dwa).unwrap();
    assert_eq!(packed.token_set_count(), TOKEN_SET_COUNT as usize);
    assert!(packed.transition(0, 7).is_some());
}

#[test]
fn packed_runtime_dwf8_roundtrip_preserves_large_targets() {
    let mut dwa = DWA::new(1, 1);
    let large_target = 1u32 << 17;
    while dwa.num_states() <= large_target {
        dwa.add_state();
    }
    let restricted = Weight::from_per_tsid_token_sets(std::iter::once((
        0,
        RangeSetBlaze::from_iter([3..=3000, 4000..=4000]),
    )));
    dwa.add_transition(0, 7, large_target, restricted.clone());
    let dwa = dwa.share_exact_transition_rows_owned();

    let packed = PackedRuntimeDwa::from_dwa(&dwa).unwrap();
    assert!(packed.owned_narrow.is_some());
    let (owned_target, owned_weight) = packed.transition(0, 7).unwrap();
    assert_eq!(owned_target, large_target);
    assert!(!owned_weight.is_full());
    let owned_transition_token_id = owned_weight.token_set_for_tsid(0).unwrap().id();
    assert!(packed
        .transition_token_set_ids()
        .contains(&owned_transition_token_id));
    let narrow_wire = packed.fast_wire_bytes();
    assert!(narrow_wire.starts_with(b"DWF8"));
    let narrow_loaded = PackedRuntimeDwa::from_fast_wire_bytes(&narrow_wire).unwrap();
    assert_eq!(narrow_loaded.transition(0, 7).unwrap().0, large_target);
    let (target, weight) = narrow_loaded.transition(0, 7).unwrap();
    assert_eq!(target, large_target);
    assert!(!weight.is_full());
    let tokens = weight.token_set_for_tsid(0).unwrap();
    assert!(narrow_loaded
        .transition_token_set_ids()
        .contains(&tokens.id()));
    let mut ranges = Vec::new();
    tokens.for_each_range(|start, end| ranges.push((start, end)));
    assert_eq!(ranges, vec![(3, 3000), (4000, 4000)]);
}

#[test]
fn mutating_shared_transition_row_converts_it_to_owned() {
    let mut row = BTreeMap::new();
    row.insert(7, (1, Weight::all()));
    let shared = Arc::new(row);
    let mut left = DwaTransitionMap::from_arc(Arc::clone(&shared));
    let right = DwaTransitionMap::from_arc(shared);

    left.insert(8, (2, Weight::all()));

    assert!(matches!(left, DwaTransitionMap::Owned(_)));
    assert!(matches!(right, DwaTransitionMap::Shared(_)));
    assert!(left.contains_key(&8));
    assert!(!right.contains_key(&8));
}

#[test]
fn exact_duplicate_state_merge_preserves_weighted_language() {
    let mut dwa = DWA::new(1, 1);
    let left = dwa.add_state();
    let right = dwa.add_state();
    dwa.set_final_weight(left, Weight::all());
    dwa.set_final_weight(right, Weight::all());
    dwa.add_transition(0, 10, left, Weight::all());
    dwa.add_transition(0, 20, right, Weight::all());

    let reference = dwa.clone();
    let merged = dwa.merge_exact_duplicate_states_owned();
    assert_eq!(merged.num_states(), 2);
    assert_eq!(merged.num_transitions(), 2);
    assert_eq!(find_difference(&reference, &merged).unwrap(), None);
}

#[test]
fn graph_property_caches_invalidate_on_transition_mutation() {
    let mut dwa = DWA::new(1, 1);
    let next = dwa.add_state();
    dwa.add_transition(0, 7, next, Weight::all());

    assert_eq!(dwa.num_transitions(), 1);
    assert!(dwa.is_acyclic());
    assert_eq!(dwa.transition_count_cache.get(), Some(&1));
    assert_eq!(dwa.acyclic_cache.get(), Some(&true));

    dwa.add_transition(next, 8, next, Weight::all());
    assert!(dwa.transition_count_cache.get().is_none());
    assert!(dwa.acyclic_cache.get().is_none());
    assert_eq!(dwa.num_transitions(), 2);
    assert!(!dwa.is_acyclic());
}

#[test]
fn mutable_state_access_and_deserialization_reset_graph_caches() {
    let mut dwa = DWA::new(1, 1);
    let next = dwa.add_state();
    dwa.add_transition(0, 1, next, Weight::all());
    assert_eq!(dwa.num_transitions(), 1);
    assert!(dwa.is_acyclic());

    dwa.states_mut()[next as usize]
        .transitions
        .insert(2, (next, Weight::all()));
    assert!(dwa.transition_count_cache.get().is_none());
    assert!(dwa.acyclic_cache.get().is_none());
    assert_eq!(dwa.num_transitions(), 2);
    assert!(!dwa.is_acyclic());

    let decoded: DWA = bincode::deserialize(&bincode::serialize(&dwa).unwrap()).unwrap();
    assert!(decoded.transition_count_cache.get().is_none());
    assert!(decoded.acyclic_cache.get().is_none());
    assert_eq!(decoded.num_transitions(), 2);
    assert!(!decoded.is_acyclic());
}

#[test]
fn serde_pools_structural_weights_and_token_sets_not_arc_identity() {
    use std::sync::Arc;

    fn weight_with_tokens(
        tsid: u32,
        tokens: Arc<RangeSetBlaze<u32>>,
    ) -> Weight {
        let mut map = RangeMapBlaze::new();
        map.extend_simple(std::iter::once((tsid..=tsid, tokens)));
        finalize_weight_map(map)
    }

    // Deliberately bypass the token-set interner so the two equal token
    // languages have different Arc identities.  `finalize_weight_map`
    // consequently also sees distinct token-body pointers, giving us the
    // allocation-layout case that used to leak into artifact bytes.
    let token_a = Arc::new(RangeSetBlaze::from_iter([3..=7, 11..=13]));
    let token_b = Arc::new(RangeSetBlaze::from_iter([3..=7, 11..=13]));
    assert!(!Arc::ptr_eq(&token_a, &token_b));

    let equal_a = weight_with_tokens(5, Arc::clone(&token_a));
    let equal_b = weight_with_tokens(5, Arc::clone(&token_b));
    assert_eq!(equal_a, equal_b);
    assert_ne!(equal_a.ptr_key(), equal_b.ptr_key());

    let distinct_a = weight_with_tokens(6, token_a);
    let distinct_b = weight_with_tokens(7, token_b);
    assert_ne!(distinct_a, distinct_b);

    let mut dwa = DWA::new(8, 13);
    let target = dwa.add_state();
    dwa.add_transition(0, 1, target, equal_a);
    dwa.add_transition(0, 2, target, equal_b);
    dwa.add_transition(0, 3, target, distinct_a);
    dwa.add_transition(0, 4, target, distinct_b);

    let bytes = bincode::serialize(&dwa).unwrap();
    let encoded: DWASerde = bincode::deserialize(&bytes).unwrap();
    // equal_a/equal_b share one structural weight pool entry, while the
    // two TSID-distinct weights remain separate.
    assert_eq!(encoded.weight_pool.len(), 3);
    // All three structural weights refer to the same token language even
    // though the source Arcs were intentionally different.
    assert_eq!(encoded.token_set_pool.len(), 1);

    let decoded: DWA = bincode::deserialize(&bytes).unwrap();
    assert_eq!(decoded, dwa);
}
