//! Integration tests for the primary LeveledGSS::any_prefix_matching primitive
//! with independently compiled domains. Requires primary GSS patch + domain
//! primitive; keep in parser-dwa/tests/template_domain_gss.rs when integrating.
#![cfg(feature = "internal-api")]
use glrmask_artifact::CommitTemplateDfas;
use glrmask_finite_automata::unweighted_u32::dfa::DFA;
use glrmask_glr::__private::glr::{accumulator::TerminalsDisallowed, labels::DEFAULT_LABEL, parser::ParserGSS};
use glrmask_parser_dwa::__private::templates::admissibility::{TemplateDomain, DomainProbe};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::ControlFlow;

fn matches(domain: &TemplateDomain, gss: &ParserGSS) -> bool {
    if gss.is_empty() { return false; }
    let cursor = match domain.start() {
        DomainProbe::Accept => return true,
        DomainProbe::Reject => return false,
        DomainProbe::NeedMore(cursor) => cursor,
    };
    gss.any_prefix_matching(cursor, |cursor, top| match domain.step(cursor, *top) {
        DomainProbe::Accept => ControlFlow::Break(true),
        DomainProbe::Reject => ControlFlow::Break(false),
        DomainProbe::NeedMore(next) => ControlFlow::Continue(next),
    })
}

fn domain_for_prefix(prefix: &[u32]) -> TemplateDomain {
    let mut t = CommitTemplateDfas { pop: DFA::new(), ..Default::default() };
    let mut previous = 0;
    for &top in prefix {
        let next = t.pop.add_state(); t.pop.add_transition(previous, top as i32, next); previous = next;
    }
    t.pop.set_accepting(previous, true);
    let raw = TemplateDomain::compile(&t).unwrap().to_bytes().unwrap();
    TemplateDomain::from_bytes(&raw).unwrap()
}

fn acc(tag: u32) -> TerminalsDisallowed {
    TerminalsDisallowed::from_map(BTreeMap::from([(17, BTreeSet::from([tag]))]))
}

fn expected(domain: &TemplateDomain, gss: &ParserGSS) -> bool {
    gss.to_stacks(10_000).expect("bounded fixture exceeded literal limit").iter()
        .any(|(stack, _)| domain.matches_top_first(stack.iter().rev().copied()))
}

#[test]
fn empty_language_differs_from_a_language_containing_the_empty_stack() {
    let empty_language = ParserGSS::empty();
    let empty_stack = ParserGSS::from_single_stack(vec![], acc(1));
    let universal = domain_for_prefix(&[]);
    let one = domain_for_prefix(&[7]);
    assert!(!matches(&universal, &empty_language));
    assert!(matches(&universal, &empty_stack));
    assert!(!matches(&one, &empty_stack));
    assert!(!matches(&one, &empty_language));
    let mixed = empty_stack.merge(&ParserGSS::from_single_stack(vec![7], acc(2)));
    assert!(matches(&one, &mixed));
    assert!(matches(&universal, &mixed));
}

#[test]
fn mixed_accumulators_survive_queries_and_pruning_is_not_bypassed() {
    let original = ParserGSS::from_stacks(&[
        (vec![0, 2, 7], acc(1)), (vec![0, 3, 7], acc(2)),
        (vec![9, 4, 7], acc(1)), (vec![], acc(2)),
    ]);
    let snapshot = original.clone();
    let literal_before = original.to_stacks(100).unwrap();
    for prefix in [vec![], vec![7], vec![7,2], vec![7,3], vec![7,4,9], vec![7,2,9], vec![9]] {
        let domain = domain_for_prefix(&prefix);
        assert_eq!(matches(&domain, &original), expected(&domain, &original));
        for tag in [1,2] {
            let pruned = original.apply_and_prune_no_promote(|a|
                a.get(&17).is_some_and(|set| set.contains(&tag)).then(|| a.clone()));
            assert_eq!(matches(&domain, &pruned), expected(&domain, &pruned));
        }
    }
    assert!(original.ptr_eq(&snapshot));
    assert_eq!(original.to_stacks(100).unwrap(), literal_before);
    let only_tag2 = original.apply_and_prune_no_promote(|a|
        a.get(&17).is_some_and(|set| set.contains(&2)).then(|| a.clone()));
    assert!(matches(&domain_for_prefix(&[7,2]), &original));
    assert!(!matches(&domain_for_prefix(&[7,2]), &only_tag2));
}

#[test]
fn merge_fuse_pop_push_and_explicit_default_exception_match_literal_languages() {
    let left = ParserGSS::from_stacks(&[(vec![0,2,7],acc(1)),(vec![0,3,7],acc(2)),(vec![9],acc(1))]);
    let right = ParserGSS::from_stacks(&[(vec![1,4,7],acc(3)),(vec![0,2,7],acc(2)),(vec![],acc(3))]);
    let merged = left.merge(&right);
    let variants = vec![merged.clone(), merged.fuse(None), merged.fuse(Some(1)), merged.fuse(Some(2)),
        merged.push(8), merged.popn(1), merged.popn(2), merged.push(8).popn(1),
        merged.isolate(Some(7)), left.push(5).merge(&right.push(6)),
        merged.apply_and_prune_no_promote(|_| None), ParserGSS::empty()];
    let mut t = CommitTemplateDfas { pop: DFA::new(), ..Default::default() };
    let accepted = t.pop.add_state(); t.pop.set_accepting(accepted, true);
    let rejected = t.pop.add_state();
    t.pop.add_transition(0,DEFAULT_LABEL,accepted); t.pop.add_transition(0,7,rejected);
    let mut domains = vec![TemplateDomain::compile(&t).unwrap()];
    for prefix in [vec![],vec![7],vec![7,2],vec![7,2,0],vec![7,3,0],vec![7,4,1],
        vec![8,7,2,0],vec![5,7],vec![6,7],vec![99],vec![0,7]] {
        domains.push(domain_for_prefix(&prefix));
    }
    for (i, gss) in variants.iter().enumerate() {
        let before = gss.to_stacks(10_000).unwrap();
        for (j, domain) in domains.iter().enumerate() {
            assert_eq!(matches(domain,gss),expected(domain,gss),"variant={i} domain={j}");
        }
        assert_eq!(before,gss.to_stacks(10_000).unwrap(),"query mutated variant={i}");
    }
}

#[test]
fn generated_shared_gss_shapes_match_exhaustive_concrete_prefix_queries() {
    let mut random = 7u64;
    let mut next = || { random = random.wrapping_mul(6364136223846793005).wrapping_add(1); random };
    for case in 0..96 {
        let mut rows = Vec::new();
        for _ in 0..12 {
            let len = (next() % 7) as usize;
            let stack = (0..len).map(|_| (next() % 5) as u32).collect::<Vec<_>>();
            rows.push((stack,acc((next()%3) as u32)));
        }
        let gss = ParserGSS::from_stacks(&rows);
        let all = [gss.clone(),gss.fuse(None),gss.fuse(Some(2)),gss.push(7).popn(1),gss.popn(1)];
        let mut prefixes = vec![Vec::new()];
        for (stack,_) in &rows {
            let stack:Vec<u32> = stack.iter().rev().copied().collect();
            for n in 0..=stack.len() { prefixes.push(stack[..n].to_vec()); }
        }
        for _ in 0..24 { prefixes.push(vec![(next()%9) as u32,(next()%9) as u32,(next()%9) as u32]); }
        for prefix in prefixes {
            let domain = domain_for_prefix(&prefix);
            for (variant,gss) in all.iter().enumerate() {
                assert_eq!(matches(&domain,gss),expected(&domain,gss),"case={case} variant={variant} prefix={prefix:?}");
            }
        }
    }
}
