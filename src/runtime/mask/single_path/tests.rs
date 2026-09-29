use super::*;
use crate::Vocab;

fn literal_constraint() -> Constraint {
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
    Constraint::from_glrm_grammar("start root; t A ::= 'a'; nt root ::= A;", &vocab).unwrap()
}

#[test]
fn singleton_direct_paths_stay_inline_and_leave_no_live_scratch() {
    let constraint = literal_constraint();
    let state = constraint.start();
    let mut output = vec![u32::MAX; constraint.body_mask_len()];
    for _ in 0..3 {
        assert!(state.try_fill_mask_single_path_direct(&mut output));
        assert_eq!(output[0] & 3, 1);
        let scratch = state.mask_scratch.lock().unwrap();
        assert!(scratch.single_path_paths.is_empty());
        assert!(!scratch.single_path_paths.spilled());
    }
}

#[test]
fn spilled_direct_path_capacity_survives_success_and_rejection() {
    let constraint = literal_constraint();
    let mut state = constraint.start();
    let (lexer, gss) = state.state.entries[0].clone();
    state.state.entries.clear();
    for _ in 0..65 {
        state.state.insert_flat_alternative(lexer, gss.clone());
    }
    let mut expected = vec![0; constraint.body_mask_len()];
    state.fill_mask_dynamic(&mut expected);
    let mut output = vec![u32::MAX; constraint.body_mask_len()];
    assert!(state.try_fill_mask_single_path_direct(&mut output));
    assert_eq!(output, expected);
    let (pointer, capacity) = {
        let scratch = state.mask_scratch.lock().unwrap();
        assert!(scratch.single_path_paths.is_empty());
        assert!(scratch.single_path_paths.spilled());
        (
            scratch.single_path_paths.as_ptr(),
            scratch.single_path_paths.capacity(),
        )
    };
    // Early admission failure must not discard scratch or turn into an approximate mask.
    for _ in 65..129 {
        state.state.insert_flat_alternative(lexer, gss.clone());
    }
    assert!(!state.try_fill_mask_single_path_direct(&mut output));
    state.state.entries.truncate(65);
    assert!(state.try_fill_mask_single_path_direct(&mut output));
    assert_eq!(output, expected);
    let scratch = state.mask_scratch.lock().unwrap();
    assert!(scratch.single_path_paths.is_empty());
    assert_eq!(scratch.single_path_paths.as_ptr(), pointer);
    assert_eq!(scratch.single_path_paths.capacity(), capacity);
}

#[test]
fn repeated_recursive_frontier_matches_dynamic_after_scratch_reuse() {
    let vocab = Vocab::new(vec![
        (0, b"(".to_vec()),
        (1, b")".to_vec()),
        (2, b"a".to_vec()),
    ]);
    let built =
        Constraint::from_glrm_grammar("start root; nt root ::= '(' root ')' | 'a';", &vocab)
            .unwrap();
    let loaded = Constraint::load(built.save()).unwrap();
    for constraint in [&built, &loaded] {
        let mut state = constraint.start();
        state.commit_bytes(b"((").unwrap();
        let (lexer, gss) = state.state.entries[0].clone();
        state.state.entries.clear();
        for _ in 0..64 {
            state.state.insert_flat_alternative(lexer, gss.clone());
        }
        let mut expected = vec![0; constraint.body_mask_len()];
        state.fill_mask_dynamic(&mut expected);
        let mut output = vec![u32::MAX; constraint.body_mask_len()];
        for _ in 0..3 {
            assert!(state.try_fill_mask_single_path_direct(&mut output));
            assert_eq!(output, expected);
            assert!(
                state
                    .mask_scratch
                    .lock()
                    .unwrap()
                    .single_path_paths
                    .is_empty()
            );
        }
    }
}

#[test]
fn partial_admission_failure_returns_empty_path_scratch() {
    let vocab = Vocab::new(vec![
        (0, b"(".to_vec()),
        (1, b")".to_vec()),
        (2, b"a".to_vec()),
    ]);
    let constraint =
        Constraint::from_glrm_grammar("start root; nt root ::= '(' root ')' | 'a';", &vocab)
            .unwrap();
    let mut state = constraint.start();
    let shallow = state.state.entries[0].clone();
    let mut deep = constraint.start();
    deep.commit_bytes(&vec![b'('; 80]).unwrap();
    let deep_path = deep
        .state
        .entries
        .iter()
        .find(|(_, gss)| gss.max_depth() > 64)
        .expect("recursive fixture must exceed the direct depth budget")
        .clone();
    state.state.entries.clear();
    state.state.entries.push(shallow.clone());
    state.state.entries.push(deep_path);
    let mut output = vec![u32::MAX; constraint.body_mask_len()];
    assert!(!state.try_fill_mask_single_path_direct(&mut output));
    assert!(
        state
            .mask_scratch
            .lock()
            .unwrap()
            .single_path_paths
            .is_empty()
    );
    state.state.entries.clear();
    state.state.entries.push(shallow);
    assert!(state.try_fill_mask_single_path_direct(&mut output));
    let mut expected = vec![0; constraint.body_mask_len()];
    state.fill_mask_dynamic(&mut expected);
    assert_eq!(output, expected);
}

// Isolate pure internal-to-original projection from parser construction. The
// source grammar provides a valid owner and output extent; the synthetic map
// is used only by these projection tests, never for parser execution or saving.
fn monotone_projection_fixture(backing_offset: Option<usize>) -> Constraint {
    use crate::runtime::artifact::{BackedInternalTokenBufMasks, PackedInternalTokenBufMask};
    use crate::runtime::constraint::TokenMaskCachePrebuild;
    use std::sync::Arc;

    const INTERNAL: usize = 193;
    const ALIASES: usize = 64;
    let count = INTERNAL * ALIASES;
    let vocab = Vocab::new(vec![
        (0, b"a".to_vec()),
        (1, b"b".to_vec()),
        ((count - 1) as u32, b"z".to_vec()),
    ]);
    let mut constraint = Constraint::from_glrm_grammar(
        "start root; t A ::= 'a'; nt root ::= A;", &vocab,
    ).unwrap();
    let originals: Vec<u32> = (0..count).map(|id| (id % INTERNAL) as u32).collect();
    constraint.internal_token_to_tokens = (0..INTERNAL)
        .map(|internal| (0..ALIASES).map(|alias| (alias * INTERNAL + internal) as u32).collect())
        .collect();
    constraint.original_token_to_internal = originals.clone();
    constraint.packed_original_token_to_internal = None;
    constraint.final_mask_mapping = Default::default();
    constraint.internal_token_dense_words = INTERNAL.div_ceil(64);
    let prebuilt = TokenMaskCachePrebuild::build(
        &originals, &constraint.internal_token_to_tokens, constraint.body_mask_len(),
    );
    prebuilt.install(&mut constraint);
    if let Some(offset) = backing_offset {
        let entries: &[PackedInternalTokenBufMask] = &constraint.internal_token_buf_flat;
        let mut bytes = vec![0xa5; offset];
        for entry in entries {
            bytes.extend_from_slice(&entry.word_idx.to_le_bytes());
            bytes.extend_from_slice(&entry.mask.to_le_bytes());
        }
        let backed = BackedInternalTokenBufMasks::new(Arc::new(bytes), offset, entries.len())
            .expect("the full synthetic map is in bounds");
        if offset == 1 {
            assert!(backed.slice(0, entries.len()).is_none());
        } else if cfg!(target_endian = "little") {
            assert!(backed.slice(0, entries.len()).is_some());
        }
        constraint.internal_token_buf_flat = Vec::new().into_boxed_slice();
        constraint.backed_internal_token_buf_flat = Some(backed);
    }
    assert_eq!(constraint.internal_token_count(), INTERNAL);
    assert_eq!(constraint.body_mask_len(), count.div_ceil(32));
    constraint
}

fn projection_oracle(constraint: &Constraint, dense: &[u64]) -> Vec<u32> {
    let mut result = vec![0; constraint.body_mask_len()];
    for (internal, originals) in constraint.internal_token_to_tokens.iter().enumerate() {
        if dense.get(internal / 64).is_some_and(|bits| bits & (1u64 << (internal % 64)) != 0) {
            for &original in originals {
                result[original as usize / 32] |= 1u32 << (original % 32);
            }
        }
    }
    result
}

fn alternating_internal_bitmap() -> Vec<u64> {
    vec![0x5555_5555_5555_5555; 3].into_iter().chain([1]).collect()
}

#[test]
fn single_path_monotone_cache_additions_match_owned_and_backed_oracles() {
    for storage in [None, Some(0), Some(1)] {
        let constraint = monotone_projection_fixture(storage);
        let state = constraint.start();
        let previous = alternating_internal_bitmap();
        let old_mask = projection_oracle(&constraint, &previous);
        for added in [vec![1usize], vec![63, 65], vec![127, 129, 191]] {
            let mut current = previous.clone();
            for id in added { current[id / 64] |= 1u64 << (id % 64); }
            let expected = projection_oracle(&constraint, &current);
            state.store_mask_cache(&old_mask, &previous);
            let pointers = {
                let cache = state.mask_cache.lock().unwrap();
                let cache = cache.as_ref().unwrap();
                (cache.mask.as_ptr(), cache.merged_dense.as_ptr(), cache.generation)
            };
            let mut guarded = vec![0xa5a5_a5a5; expected.len() + 4];
            assert!(state.try_replay_monotone_dense_cache(&current, &mut guarded[2..2 + expected.len()]));
            assert_eq!(&guarded[2..2 + expected.len()], expected);
            assert_eq!(&guarded[..2], &[0xa5a5_a5a5; 2]);
            assert_eq!(&guarded[2 + expected.len()..], &[0xa5a5_a5a5; 2]);
            let cache = state.mask_cache.lock().unwrap();
            let cache = cache.as_ref().unwrap();
            assert_eq!((cache.mask.as_ptr(), cache.merged_dense.as_ptr(), cache.generation), pointers);
            assert_eq!(cache.mask, old_mask, "the caller, not speculative replay, updates the cache");
            assert_eq!(cache.merged_dense, previous);
        }
    }
}

#[test]
fn single_path_monotone_cache_declines_incomplete_or_nonmonotone_history() {
    let constraint = monotone_projection_fixture(Some(1));
    let state = constraint.start();
    let previous = alternating_internal_bitmap();
    let old_mask = projection_oracle(&constraint, &previous);
    let mut current = previous.clone();
    current[0] |= 2;
    let mut output = vec![0x7654_3210; old_mask.len()];
    let untouched = output.clone();
    *state.mask_cache.lock().unwrap() = None;
    assert!(!state.try_replay_monotone_dense_cache(&current, &mut output));
    assert_eq!(output, untouched);
    for corrupt in 0..6 {
        state.store_mask_cache(&old_mask, &previous);
        let mut queried = current.clone();
        match corrupt {
            0 => state.mask_cache.lock().unwrap().as_mut().unwrap().merged_dense.clear(),
            1 => { state.mask_cache.lock().unwrap().as_mut().unwrap().mask.pop(); },
            2 => queried[0] &= !4,
            3 => { queried.pop(); },
            4 => queried[3] |= 2,
            5 => state.mask_cache.lock().unwrap().as_mut().unwrap().merged_dense[3] |= 2,
            _ => unreachable!(),
        }
        assert!(!state.try_replay_monotone_dense_cache(&queried, &mut output), "case {corrupt}");
        assert_eq!(output, untouched, "a declined proof must not mutate the output");
    }
    // A tiny addition to a tiny previous image does not justify copying a
    // full output buffer. Declining the work gate is independent of validity.
    let tiny = vec![1u64, 0, 0, 0];
    state.store_mask_cache(&projection_oracle(&constraint, &tiny), &tiny);
    assert!(!state.try_replay_monotone_dense_cache(&[3, 0, 0, 0], &mut output));
    assert_eq!(output, untouched);
}

#[test]
fn single_path_monotone_cache_equal_dense_preserves_all_output_words() {
    let constraint = monotone_projection_fixture(Some(0));
    let state = constraint.start();
    for dense in [vec![0; 4], vec![u64::MAX, u64::MAX, u64::MAX, 1], alternating_internal_bitmap()] {
        let expected = projection_oracle(&constraint, &dense);
        state.store_mask_cache(&expected, &dense);
        state.mask_cache.lock().unwrap().as_mut().unwrap().generation = u64::MAX;
        let mut output = vec![u32::MAX; expected.len()];
        assert!(state.try_replay_monotone_dense_cache(&dense, &mut output));
        assert_eq!(output, expected);
        assert_eq!(state.mask_cache.lock().unwrap().as_ref().unwrap().generation, u64::MAX);
    }
}

#[test]
fn single_path_monotone_cache_generated_replays_match_original_id_unions() {
    let mut rng = 0x12c4_88e5_d912_673bu64;
    let mut admitted = 0usize;
    let mut declined = 0usize;
    for storage in [None, Some(0), Some(1)] {
        let constraint = monotone_projection_fixture(storage);
        let state = constraint.start();
        for case in 0..96 {
            let mut previous = vec![0u64; 4];
            for internal in 0..193 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                if rng >> 63 != 0 { previous[internal / 64] |= 1u64 << (internal % 64); }
            }
            let mut current = previous.clone();
            for internal in [case % 193, (case * 7 + 63) % 193, (case * 13 + 128) % 193] {
                current[internal / 64] |= 1u64 << (internal % 64);
            }
            if case % 3 == 0 {
                let removed = (0..193).find(|&i| previous[i / 64] >> (i % 64) & 1 != 0).unwrap();
                current[removed / 64] &= !(1u64 << (removed % 64));
            }
            state.store_mask_cache(&projection_oracle(&constraint, &previous), &previous);
            let mut output = vec![0x1234_5678; constraint.body_mask_len()];
            let before = output.clone();
            if state.try_replay_monotone_dense_cache(&current, &mut output) {
                admitted += 1;
                assert_eq!(output, projection_oracle(&constraint, &current));
                assert!(previous.iter().zip(&current).all(|(&p, &c)| p & !c == 0));
            } else {
                declined += 1;
                assert_eq!(output, before);
            }
        }
    }
    assert!(admitted >= 96, "generated trials must actually exercise reuse");
    assert!(declined >= 96, "all deliberate removals must decline");
}

#[test]
fn bounded_monotone_growth_preserves_the_budget_and_late_removals() {
    let previous = [0u64, 0, 1];
    assert!(bounded_monotone_growth(&previous, &[u64::MAX, 0, 1]));
    assert!(!bounded_monotone_growth(&previous, &[u64::MAX, 1, 1]));
    assert!(!bounded_monotone_growth(&previous, &[1, 0, 0]));
    assert!(!bounded_monotone_growth(&previous, &[0, 0]));
    assert!(bounded_monotone_growth(&previous, &previous));
    for word in 0..3 {
        for added in 0..=64 {
            let mut current = previous;
            let bits = if added == 64 { u64::MAX } else { (1u64 << added) - 1 };
            current[word] |= bits;
            let reference = previous.iter().zip(current).all(|(&p, c)| p & !c == 0)
                && previous.iter().zip(current)
                    .map(|(&p, c)| (c & !p).count_ones()).sum::<u32>() <= 64;
            assert_eq!(bounded_monotone_growth(&previous, &current), reference);
        }
    }
}

#[test]
fn single_path_monotone_broad_growth_declines_without_writes() {
    for backing in [None, Some(0), Some(1)] {
        let constraint = monotone_projection_fixture(backing);
        let state = constraint.start();
        let previous = vec![1u64, 0, 0, 0];
        let current = vec![u64::MAX, u64::MAX, u64::MAX, 1];
        state.store_mask_cache(&projection_oracle(&constraint, &previous), &previous);
        let mut output = vec![0x7654_3210; constraint.body_mask_len()];
        let unchanged = output.clone();
        assert!(!state.try_replay_monotone_dense_cache(&current, &mut output));
        assert_eq!(output, unchanged);
        output.fill(0);
        constraint.or_internal_dense_to_buf_fast(&current, &mut output, true);
        assert_eq!(output, projection_oracle(&constraint, &current));
    }
}
