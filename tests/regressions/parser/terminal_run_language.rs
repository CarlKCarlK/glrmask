use glrmask::{Constraint, Grammar, Vocab};

const SOURCE: &str = r#"start root; nt root ::= "a" root "b" | "z";"#;

fn language(prefix: &[u8]) -> (bool, bool) {
    let opening = prefix.iter().take_while(|&&byte| byte == b'a').count();
    let suffix = &prefix[opening..];
    if suffix.is_empty() { return (true, false); }
    if suffix[0] != b'z' { return (false, false); }
    let closing = &suffix[1..];
    let viable = closing.len() <= opening && closing.iter().all(|&byte| byte == b'b');
    (viable, viable && closing.len() == opening)
}

fn check(constraint: &Constraint, entries: &[(u32, Vec<u8>)], prefix: &[u8]) {
    let mut state = constraint.start();
    if !prefix.is_empty() { state.commit_bytes(prefix).unwrap(); }
    let mut expected = vec![0u32; constraint.mask_len() + 3];
    for (id, bytes) in entries {
        let mut extended = prefix.to_vec();
        extended.extend_from_slice(bytes);
        if language(&extended).0 { expected[*id as usize / 32] |= 1 << (*id % 32); }
    }
    for poison in [u32::MAX, 0x5555_5555] {
        let mut actual = vec![poison; expected.len()];
        state.fill_mask(&mut actual);
        assert_eq!(actual, expected, "prefix {prefix:?}");
    }
    assert_eq!(state.is_accepting(), language(prefix).1);
}

fn base_entries() -> Vec<(u32, Vec<u8>)> {
    vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"z".to_vec()),
         (3,b"x".to_vec()),(31,b"azb".to_vec()),(32,b"aazbb".to_vec()),
         (63,b"zz".to_vec()),(65535,b"a".to_vec()),(65536,b"b".to_vec())]
}

#[test]
fn long_vocabulary_token_preserves_existing_recursive_admissions() {
    let mut entries = base_entries();
    for long_token in [false, true] {
        if long_token { entries.push((11,vec![b'a';16])); }
        let vocab = Vocab::new(entries.clone());
        let built = Constraint::compile(Grammar::glrm(SOURCE), &vocab).unwrap();
        let loaded = Constraint::load(built.save()).unwrap();
        for constraint in [&built,&loaded] {
            for prefix in [b"".as_slice(),b"a",b"aa",b"az",b"aaz",b"aazb",b"aazbb"] {
                check(constraint,&entries,prefix);
            }
        }
    }
}

#[test]
fn nested_fused_tokens_retain_each_recursive_stack_obligation() {
    let mut entries = base_entries();
    entries.push((11,vec![b'a';16]));
    for n in 1..=8 {
        let mut bytes = vec![b'a';n]; bytes.push(b'z'); bytes.extend(vec![b'b';n]);
        entries.push((100+n as u32,bytes));
    }
    let vocab = Vocab::new(entries.clone());
    let built = Constraint::compile(Grammar::glrm(SOURCE), &vocab).unwrap();
    let loaded = Constraint::load(built.save()).unwrap();
    for constraint in [&built,&loaded] {
        for n in 0..=12 { check(constraint,&entries,&vec![b'a';n]); }
        for (id,bytes) in &entries {
            if !language(bytes).1 { continue; }
            let mut state = constraint.start();
            let mask = state.mask();
            assert_ne!(mask[*id as usize/32] & (1 << (*id%32)),0,"token {id}");
            state.commit_token(*id).unwrap();
            assert!(state.is_accepting());
            check(constraint,&entries,bytes);
        }
    }
}

#[test]
fn recursive_depth_boundaries_preserve_complete_masks() {
    let mut entries = base_entries(); entries.push((11,vec![b'a';16]));
    let vocab = Vocab::new(entries.clone());
    let constraint = Constraint::compile(Grammar::glrm(SOURCE), &vocab).unwrap();
    for depth in [0,1,3,31,62,63,64,65,66,80] {
        let mut prefix = vec![b'a';depth];
        check(&constraint,&entries,&prefix);
        prefix.push(b'z');
        for _ in 0..depth { check(&constraint,&entries,&prefix); prefix.push(b'b'); }
        check(&constraint,&entries,&prefix);
    }
}

#[test]
fn long_model_tokens_do_not_relax_finite_item_bounds() {
    let source = r#"start root; t A ::= "a"; nt root ::= A A? A?;"#;
    let entries = vec![(0,b"a".to_vec()),(1,b"aaa".to_vec()),
                       (2,b"aaaa".to_vec()),(3,vec![b'a';16])];
    let constraint = Constraint::compile(Grammar::glrm(source), &Vocab::new(entries)).unwrap();
    let mut state = constraint.start();
    assert_eq!(state.mask()[0] & 15, 3);
    state.commit_token(1).unwrap();
    assert!(state.is_accepting());
    assert_eq!(state.mask()[0] & 15,0);
}
