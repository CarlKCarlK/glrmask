use super::*;

const DEFAULT: i32 = 2_147_483_646;

fn signature(masks: &mut Masks, final_bits: u64, a: u64, b: u64, guarded: bool) -> Signature {
    let mut edges = Vec::new();
    for (label, target, bits) in [(0, 2, a), (2, 3, b)] {
        if bits != 0 || guarded {
            edges.push(Edge { label, target: if bits == 0 { 0 } else { target },
                mask: masks.intern(vec![bits]).unwrap() });
        }
    }
    if guarded { edges.push(Edge { label: DEFAULT, target: 0, mask: 0 }); }
    Signature { final_mask: masks.intern(vec![final_bits]).unwrap(), edges }
}

#[test]
fn compatible_contained_domains_make_the_complete_merge_an_identity() {
    let mut masks = Masks::new(1);
    let mut checked = 0usize;
    let mut strict = 0usize;
    for final_bits in 0..8u64 {
        for a in 0..8u64 {
            for b in 0..8u64 {
                for selected in 0..8u64 {
                    for guarded in [false, true] {
                        let domain_bits = final_bits | a | b;
                        let domain = masks.intern(vec![domain_bits]).unwrap();
                        let mut group = Group { domain,
                            signature: signature(&mut masks, final_bits, a, b, guarded), guarded };
                        let state = signature(&mut masks, final_bits & selected, a & selected,
                            b & selected, guarded);
                        let state_domain = masks.intern(vec![domain_bits & selected]).unwrap();
                        assert!(compatible(&group, &state, state_domain, guarded, &masks));
                        assert!(merge_domain_is_contained(&group, state_domain, &masks));
                        let before = (group.domain, group.signature.clone(), group.guarded);
                        let values_before = masks.values.len();
                        merge(&mut group, &state, state_domain, &mut masks).unwrap();
                        assert_eq!((group.domain, group.signature, group.guarded), before);
                        assert_eq!(masks.values.len(), values_before, "identity must not add historical masks");
                        checked += 1;
                        strict += usize::from(domain != state_domain);
                    }
                }
            }
        }
    }
    assert_eq!(checked, 8192);
    assert!(strict > 1000);
}

#[test]
fn growing_domains_and_incompatible_guards_do_not_take_the_identity_shortcut() {
    let mut masks = Masks::new(1);
    let a = masks.intern(vec![1]).unwrap();
    let b = masks.intern(vec![2]).unwrap();
    let mut group = Group { domain: a, signature: signature(&mut masks, 1, 0, 0, false), guarded: false };
    let state = signature(&mut masks, 2, 0, 0, false);
    assert!(compatible(&group, &state, b, false, &masks));
    assert!(!merge_domain_is_contained(&group, b, &masks));
    merge(&mut group, &state, b, &mut masks).unwrap();
    assert_eq!(masks.values[group.domain as usize].as_ref(), &[3]);
    assert_eq!(masks.values[group.signature.final_mask as usize].as_ref(), &[3]);

    let group = Group { domain: a, signature: signature(&mut masks, 0, 1, 0, true), guarded: true };
    let mut state = group.signature.clone();
    state.edges[0].target = 9;
    assert!(!compatible(&group, &state, a, true, &masks), "target conflicts still reject before certificate");
    state = group.signature.clone();
    state.edges.retain(|edge| edge.label != 2);
    assert!(!compatible(&group, &state, a, true, &masks), "an explicit zero guard is not a missing edge");
}

#[test]
fn domain_containment_checks_every_word_and_preserves_sentinels() {
    for rows in [1, 2, 17, 44, 64] {
        let mut masks = Masks::new(rows);
        let parent = masks.intern(vec![0x5555; rows]).unwrap();
        let child = masks.intern(vec![0x1111; rows]).unwrap();
        let group = Group { domain: parent, signature: Signature { final_mask: parent, edges: vec![] }, guarded: false };
        assert!(merge_domain_is_contained(&group, child, &masks));
        assert!(merge_domain_is_contained(&group, 0, &masks));
        assert!(merge_domain_is_contained(&group, parent, &masks));
        for index in 0..rows {
            let mut different = vec![0x1111; rows];
            different[index] |= 2;
            let different = masks.intern(different).unwrap();
            assert!(!merge_domain_is_contained(&group, different, &masks));
        }
        let universal = Group { domain: 1, signature: Signature { final_mask: 1, edges: vec![] }, guarded: false };
        assert!(merge_domain_is_contained(&universal, parent, &masks));
    }
}
