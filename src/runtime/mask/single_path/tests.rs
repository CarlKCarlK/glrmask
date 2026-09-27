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
