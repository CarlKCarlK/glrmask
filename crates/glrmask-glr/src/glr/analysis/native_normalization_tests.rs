use super::*;

fn reference_nullable(rules: &[Rule], count: u32) -> BTreeSet<u32> {
    let mut nullable = BTreeSet::new();
    loop {
        let before = nullable.len();
        for rule in rules {
            if rule.lhs < count && rule.rhs.iter().all(|symbol| {
                matches!(symbol, Symbol::Nonterminal(nt) if *nt < count && nullable.contains(nt))
            }) {
                nullable.insert(rule.lhs);
            }
        }
        if nullable.len() == before {
            return nullable;
        }
    }
}

/// The previous normalizer control flow, with unconditional epsilon helper
/// calls retained as an independent allocation/elision oracle.
fn reference_normalize(rules: &mut Vec<Rule>, start: u32) {
    use std::cell::Cell;
    let next_nt = Cell::new(max_nt_id(rules) + 1);
    let mut fresh_nt = || {
        let id = next_nt.get();
        next_nt.set(id + 1);
        id
    };
    let mut iteration = 0;
    let mut nullable_eliminated_before_exit = false;
    loop {
        let snapshot = rules.clone();
        replace_rules_with_resync(rules, &next_nt, inline_null_productions);
        rules.retain(|rule| !is_reflexive_unit_rule(rule));
        let right_recursion_completed =
            with_resynced_next_nonterminal(rules, &next_nt, |rules| {
                eliminate_right_recursion(rules, &mut fresh_nt)
            });
        let hidden_changed = with_resynced_next_nonterminal(rules, &next_nt, |rules| {
            let nullable = compute_nullable(rules, max_nt_id(rules) + 1);
            if nullable.is_empty()
                && !has_indirect_left_recursion_with_nullable(rules, &nullable)
            {
                false
            } else {
                eliminate_hidden_left_recursion(rules, &nullable, iteration + 1)
            }
        });
        dedup_rules(rules);
        let converged = *rules == snapshot;
        if compute_nullable(rules, max_nt_id(rules) + 1).is_empty()
            && right_recursion_completed && !hidden_changed
        {
            nullable_eliminated_before_exit = true;
            break;
        }
        iteration += 1;
        if converged { break; }
    }
    let snapshot = (!nullable_eliminated_before_exit).then(|| rules.clone());
    if !nullable_eliminated_before_exit {
        replace_rules_with_resync(rules, &next_nt, inline_null_productions);
        rules.retain(|rule| !is_reflexive_unit_rule(rule));
    }
    let changed = snapshot.as_ref().is_some_and(|snapshot| *rules != *snapshot);
    *rules = remove_unreachable_rules(rules, start);
    if changed {
        dedup_rules(rules);
    }
}

#[test]
fn nullable_seed_shortcut_matches_literal_fixed_point() {
    let mut seed = 0x61d28ab3u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        seed
    };
    for _ in 0..512 {
        let mut rules = Vec::new();
        for lhs in 0..12u32 {
            for _ in 0..3 {
                let length = (next() % 4) as usize;
                let rhs = (0..length).map(|_| {
                    if next() % 4 == 0 {
                        Symbol::Terminal((next() % 3) as u32)
                    } else {
                        Symbol::Nonterminal((next() % 14) as u32)
                    }
                }).collect();
                rules.push(Rule { lhs, rhs });
            }
        }
        assert_eq!(compute_nullable(&rules, 12), reference_nullable(&rules, 12));
        rules.retain(|rule| !rule.rhs.is_empty());
        assert!(reference_nullable(&rules, 12).is_empty());
        assert!(compute_nullable(&rules, 12).is_empty());
    }
}

#[test]
fn epsilon_free_in_place_path_reuses_rule_and_rhs_allocations() {
    let mut rules = vec![
        Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(0)] },
        Rule { lhs: 1, rhs: vec![Symbol::Terminal(1)] },
    ];
    let before = rules.clone();
    let vector = rules.as_ptr();
    let rhs = rules.iter().map(|rule| rule.rhs.as_ptr()).collect::<Vec<_>>();
    let next = std::cell::Cell::new(0);
    inline_null_productions_in_place_if_needed(&mut rules, &next);
    assert_eq!(rules, before);
    assert_eq!(rules.as_ptr(), vector);
    assert_eq!(
        rules.iter().map(|rule| rule.rhs.as_ptr()).collect::<Vec<_>>(),
        rhs,
    );
    assert_eq!(next.get(), 2);
}

#[test]
fn full_normalizer_keeps_exact_rule_order_and_recursion_removal() {
    for nullable in [false, true] {
        for duplicate in [false, true] {
            for reflexive in [false, true] {
                let mut rules = vec![
                    Rule {
                        lhs: 0,
                        rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(0)],
                    },
                    Rule {
                        lhs: 1,
                        rhs: vec![Symbol::Terminal(1), Symbol::Nonterminal(1)],
                    },
                    Rule { lhs: 1, rhs: vec![Symbol::Terminal(2)] },
                    Rule { lhs: 7, rhs: vec![Symbol::Terminal(7)] },
                ];
                if nullable {
                    rules.push(Rule { lhs: 1, rhs: vec![] });
                }
                if duplicate {
                    rules.push(rules[2].clone());
                }
                if reflexive {
                    rules.push(Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(1)] });
                }
                let mut expected = rules.clone();
                reference_normalize(&mut expected, 0);
                normalize_grammar(&mut rules, 0);
                assert_eq!(rules, expected);
                assert!(compute_nullable(&rules, max_nt_id(&rules) + 1).is_empty());
                let nullable = BTreeSet::new();
                assert!(find_indirect_rr_cycle(
                    &build_right_reachability_graph(&rules, &nullable)
                ).is_none());
                assert!(!has_indirect_left_recursion_with_nullable(&rules, &nullable));
            }
        }
    }
}
