use super::*;
use crate::Vocab;

fn aliased_constraint() -> Constraint {
    let vocab = Vocab::new(vec![
        (0, b"a".to_vec()),
        (7, b"a".to_vec()),
        (1023, b"a".to_vec()),
        (511, b"b".to_vec()),
        (2047, b"+".to_vec()),
    ]);
    Constraint::from_glrm_grammar(
        "start start;\nt A ::= \"a\";\nt PLUS ::= \"+\";\nnt start ::= A PLUS;\n",
        &vocab,
    ).unwrap()
}

#[test]
fn uncached_final_masks_leave_output_untouched() {
    let mut constraint = aliased_constraint();
    // Supply an explicit range-cache entry: the simple grammar can compile
    // entirely to full weights, so cache existence is not a grammar invariant.
    let tokens = Arc::new(RangeSetBlaze::from_iter([0u32..=1]));
    let key = Arc::as_ptr(&tokens) as usize;
    constraint.range_final_token_sets.insert(key);
    constraint.weight_token_buf_masks.remove(&key);
    constraint.weight_token_sparse_buf_masks.remove(&key);
    let mut output = vec![0x1234_5678; constraint.mask_len()];
    let before = output.clone();
    assert!(!constraint.try_replay_cached_final_mask(&[0b11], &tokens, &mut output));
    assert_eq!(output, before, "uncached final sets must not expand individually");
}

#[test]
fn cached_final_masks_require_complete_internal_containment() {
    let mut constraint = aliased_constraint();
    // Test the cache contract directly, independently of compiler policy.
    // Internal tokens 0 and 2 map to the chosen output words in this fixture.
    let tokens = Arc::new(RangeSetBlaze::from_iter([0u32, 2]));
    let key = Arc::as_ptr(&tokens) as usize;
    constraint.weight_token_dense_masks.insert(key, Arc::from([0b101u64, 0]));
    let mut expected = vec![0; constraint.mask_len()];
    expected[0] = 0b101;
    let last = expected.len() - 1;
    expected[last] = 1u32 << 31;
    for sparse in [false, true] {
        constraint.weight_token_buf_masks.remove(&key);
        constraint.weight_token_sparse_buf_masks.remove(&key);
        if sparse {
            constraint.weight_token_sparse_buf_masks.insert(
                key, vec![(0, expected[0]), (last as u32, expected[last])].into_boxed_slice(),
            );
        } else {
            constraint.weight_token_buf_masks.insert(key, expected.clone().into_boxed_slice());
        }
        let mut output = vec![0; expected.len()];
        assert!(constraint.try_replay_cached_final_mask(&[0b101], &tokens, &mut output));
        assert_eq!(output, expected);
        for blocked in [vec![], vec![0b001], vec![0b100]] {
            let mut untouched = vec![0x1234_5678; expected.len()];
            let before = untouched.clone();
            assert!(!constraint.try_replay_cached_final_mask(&blocked, &tokens, &mut untouched));
            assert_eq!(untouched, before, "failed containment must not leak cached admissions");
        }
    }
}

#[test]
fn range_final_replay_preserves_aliases_and_loaded_masks() {
    let constraint = aliased_constraint();
    let loaded = Constraint::load(constraint.save()).unwrap();
    for value in [&constraint, &loaded] {
        let mut state = value.start();
        let mut actual = vec![0; value.mask_len()];
        let mut expected = vec![0; actual.len()];
        for token in [0usize, 7, 1023] {
            expected[token / 32] |= 1u32 << (token % 32);
        }
        state.fill_mask(&mut actual);
        assert_eq!(actual, expected);
        state.commit_token(7).unwrap();
        expected.fill(0);
        expected[2047 / 32] = 1u32 << (2047 % 32);
        state.fill_mask(&mut actual);
        assert_eq!(actual, expected);
        state.commit_token(2047).unwrap();
        assert!(state.is_accepting());
        state.fill_mask(&mut actual);
        assert!(actual.iter().all(|&word| word == 0));
    }
}

// Tiny unshared DWAs use the decoded DWF2 fallback. Build the same parser
// with its supported shared-row representation so saving emits a backed view.
fn artifact_backed_aliased_constraint() -> Constraint {
    use crate::automata::weighted::dwa::PackedRuntimeDwa;

    let mut fresh = aliased_constraint();
    let parser = fresh.parser_dwa.clone().share_exact_transition_rows_owned();
    assert!(parser.has_shared_transition_rows());
    let packed = PackedRuntimeDwa::from_dwa(&parser).unwrap();
    assert!(packed.backed_fast_wire_bytes().is_none());
    fresh.packed_parser_dwa = Some(Arc::new(packed));
    let expected = fresh.start().mask();
    let loaded = Constraint::load(fresh.save()).unwrap();
    assert!(loaded.packed_parser_dwa.as_ref()
        .unwrap().backed_fast_wire_bytes().is_some());
    assert_eq!(loaded.start().mask(), expected);
    loaded
}

#[test]
fn bounded_final_range_replay_intersects_before_expanding_aliases() {
    let mut fresh = aliased_constraint();
    let loaded = artifact_backed_aliased_constraint();
    let tokens = Arc::new(RangeSetBlaze::from_iter([0u32]));
    fresh.range_final_token_sets.insert(Arc::as_ptr(&tokens) as usize);
    assert!(fresh.packed_parser_dwa.is_none());
    let mut output = vec![0x1234_5678; fresh.mask_len()];
    let before = output.clone();
    assert!(!fresh.try_replay_range_final_mask(&[u64::MAX], &tokens, &mut output));
    assert_eq!(output, before);
    // Large fresh constraints also pack their DWA, but still own its storage.
    let packed = crate::automata::weighted::dwa::PackedRuntimeDwa::from_dwa(
        &fresh.parser_dwa,
    ).unwrap();
    assert!(packed.backed_fast_wire_bytes().is_none());
    fresh.packed_parser_dwa = Some(Arc::new(packed));
    assert!(!fresh.try_replay_range_final_mask(&[u64::MAX], &tokens, &mut output));
    assert_eq!(output, before, "owned packed storage must retain combined replay");
    for mut constraint in [loaded] {
        assert!(constraint.packed_parser_dwa.as_ref()
            .unwrap().backed_fast_wire_bytes().is_some());
        let count = constraint.internal_token_count();
        assert!(count > 0);
        assert_eq!(constraint.final_mask_mapping.internal_len(), 0);
        let last = (count - 1).min(63) as u32;
        let tokens = Arc::new(RangeSetBlaze::from_iter([0..=last]));
        let key = Arc::as_ptr(&tokens) as usize;
        let words = constraint.body_mask_len();
        let mut untouched = vec![0x1234_5678; words + 3];
        let before = untouched.clone();
        assert!(!constraint.try_replay_range_final_mask(&[u64::MAX], &tokens, &mut untouched));
        assert_eq!(untouched, before, "unplanned ranges must decline without writes");
        constraint.range_final_token_sets.insert(key);
        for selected in [0, 1, 0xaaaa_aaaa_aaaa_aaaa, u64::MAX] {
            let mut storage = vec![0x1234_5678; words + 7];
            let output = &mut storage[2..words + 5];
            let mut expected = output.to_vec();
            let mut ignored_stats = 0;
            for token in 0..=last as usize {
                if selected & (1u64 << token) != 0 {
                    constraint.or_internal_token_to_buf_fast::<false>(
                        token, &mut expected, &mut ignored_stats,
                    );
                }
            }
            assert!(constraint.try_replay_range_final_mask(&[selected], &tokens, output));
            assert_eq!(output, expected);
            assert_eq!(&storage[..2], &[0x1234_5678; 2]);
            assert_eq!(&storage[words + 5..], &[0x1234_5678; 2]);
        }
        assert!(constraint.try_replay_range_final_mask(&[], &tokens, &mut untouched));
        assert_eq!(untouched, before, "an empty intersection must not write output");
    }
}

#[test]
fn bounded_final_range_replay_declines_over_budget_or_short_output() {
    let mut constraint = artifact_backed_aliased_constraint();
    assert!(constraint.packed_parser_dwa.as_ref()
        .unwrap().backed_fast_wire_bytes().is_some());
    let tokens = Arc::new(RangeSetBlaze::from_iter([0u32..=2048]));
    constraint.range_final_token_sets.insert(Arc::as_ptr(&tokens) as usize);
    let mut output = vec![0x1234_5678; constraint.body_mask_len()];
    let before = output.clone();
    assert!(!constraint.try_replay_range_final_mask(&[u64::MAX], &tokens, &mut output));
    assert_eq!(output, before);
    let tiny = Arc::new(RangeSetBlaze::from_iter([0u32]));
    constraint.range_final_token_sets.insert(Arc::as_ptr(&tiny) as usize);
    let end = output.len() - 1;
    assert!(!constraint.try_replay_range_final_mask(&[u64::MAX], &tiny, &mut output[..end]));
    assert_eq!(output, before);
}
