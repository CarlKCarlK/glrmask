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
fn range_final_sets_leave_output_expansion_to_the_combined_mask() {
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
