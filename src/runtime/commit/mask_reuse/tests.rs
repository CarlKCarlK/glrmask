use crate::{Constraint, DynamicConstraint, Grammar, Vocab};

fn check_accounting(constraint: &Constraint, detailed: bool) {
    let mut actual = constraint.start();
    let mut reference = constraint.start();
    for token in [0, 1, 0, 2] {
        // Populate the mask cache before each commit; its finalization is part
        // of the operation even on the exact self-loop and cache-hit paths.
        assert_eq!(actual.mask(), reference.mask());
        let before = actual.generation;
        let profile = if detailed {
            actual.commit_token_per_advance(token).unwrap().2
        } else {
            actual.commit_token_profiled(token).unwrap()
        };
        reference.commit_token(token).unwrap();
        assert_eq!(actual.generation, before + 1);
        assert!(profile.total_ns >= profile.mask_cache_reuse_ns);
        assert_eq!(actual.mask(), reference.mask());
        assert_eq!(actual.is_accepting(), reference.is_accepting());
    }
    assert!(actual.is_accepting());
}

fn fixture() -> (Grammar<'static>, Vocab) {
    (Grammar::glrm(r#"glrm 1; start root; t A = /a+/; nt root = A "b";"#),
     Vocab::new(vec![(0,b"a".to_vec()), (1,b"aa".to_vec()), (2,b"b".to_vec())]))
}

#[test]
fn profiled_commit_preserves_cached_masks_and_accounts_for_finalization() {
    let (grammar, vocab) = fixture();
    let built = Constraint::compile(grammar, &vocab).unwrap();
    let loaded = Constraint::load(built.save()).unwrap();
    for item in [&built, &loaded] { check_accounting(item, false); }
}

#[test]
fn per_advance_commit_preserves_cached_masks_and_accounts_for_finalization() {
    let (grammar, vocab) = fixture();
    let built = Constraint::compile(grammar, &vocab).unwrap();
    let loaded = Constraint::load(built.save()).unwrap();
    for item in [&built, &loaded] { check_accounting(item, true); }
}

#[test]
fn dynamic_profile_accounting_preserves_body_and_cache_generation() {
    let (grammar, vocab) = fixture();
    let built = DynamicConstraint::compile(grammar, &vocab).unwrap();
    let loaded = DynamicConstraint::load_with_vocab(&built.save(), &vocab).unwrap();
    for item in [&built.inner, &loaded.inner] {
        check_accounting(item, false);
        check_accounting(item, true);
    }
}
