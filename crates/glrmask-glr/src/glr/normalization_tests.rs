use super::*;

fn nt(id: u32) -> Symbol {
    Symbol::Nonterminal(id)
}

fn t(id: u32) -> Symbol {
    Symbol::Terminal(id)
}

fn rule(lhs: u32, rhs: Vec<Symbol>) -> Rule {
    Rule { lhs, rhs }
}

fn compare_ordered(input: &[Rule], start: u32) -> Vec<Rule> {
    let mut expected = input.to_vec();
    reference::normalize_grammar(&mut expected, start);
    let mut actual = input.to_vec();
    normalize_impl(&mut actual, start);
    assert_eq!(actual, expected);
    actual
}

/// Independent finite-word fixed point.
///
/// max_len bounds the tested word universe, not derivation depth. Unit and
/// epsilon cycles are solved to a true fixed point over that finite universe.
fn bounded_language(
    rules: &[Rule],
    start: u32,
    max_len: usize,
) -> BTreeSet<Vec<u32>> {
    let n = View::plain(rules).max_nt().max(start) as usize + 1;
    let mut languages = vec![BTreeSet::<Vec<u32>>::new(); n];

    loop {
        let mut changed = false;
        for rule in rules {
            let mut words = BTreeSet::from([Vec::<u32>::new()]);
            for symbol in &rule.rhs {
                let suffixes = match symbol {
                    Symbol::Terminal(terminal) => {
                        BTreeSet::from([vec![*terminal]])
                    }
                    Symbol::Nonterminal(nonterminal) => {
                        languages[*nonterminal as usize].clone()
                    }
                };
                let mut joined = BTreeSet::new();
                for prefix in &words {
                    for suffix in &suffixes {
                        if prefix.len() + suffix.len() <= max_len {
                            let mut word = prefix.clone();
                            word.extend_from_slice(suffix);
                            joined.insert(word);
                        }
                    }
                }
                words = joined;
            }
            let target = &mut languages[rule.lhs as usize];
            let old_len = target.len();
            target.extend(words);
            changed |= target.len() != old_len;
        }
        if !changed {
            return languages[start as usize].clone();
        }
    }
}

/// Independent Earley recognizer over literal terminal bytes/IDs.
///
/// Each chart is saturated to a fixed point. There is no derivation-depth cap;
/// the finite item universe ensures termination even with epsilon/unit cycles.
fn earley_accepts(rules: &[Rule], start: u32, word: &[u32]) -> bool {
    type Item = (usize, usize, usize); // rule, dot, origin
    let mut chart = vec![BTreeSet::<Item>::new(); word.len() + 1];

    for (index, rule) in rules.iter().enumerate() {
        if rule.lhs == start {
            chart[0].insert((index, 0, 0));
        }
    }

    for position in 0..=word.len() {
        loop {
            let old_len = chart[position].len();
            let items = chart[position].iter().copied().collect::<Vec<_>>();

            for (rule_index, dot, origin) in items {
                let current = &rules[rule_index];
                if let Some(Symbol::Nonterminal(next)) = current.rhs.get(dot) {
                    for (index, candidate) in rules.iter().enumerate() {
                        if candidate.lhs == *next {
                            chart[position].insert((index, 0, position));
                        }
                    }
                } else if dot == current.rhs.len() {
                    let parents = chart[origin].iter().copied().collect::<Vec<_>>();
                    for (parent_index, parent_dot, parent_origin) in parents {
                        if matches!(
                            rules[parent_index].rhs.get(parent_dot),
                            Some(Symbol::Nonterminal(next)) if *next == current.lhs
                        ) {
                            chart[position].insert((
                                parent_index,
                                parent_dot + 1,
                                parent_origin,
                            ));
                        }
                    }
                }
            }

            if chart[position].len() == old_len {
                break;
            }
        }

        if position < word.len() {
            let items = chart[position].iter().copied().collect::<Vec<_>>();
            for (rule_index, dot, origin) in items {
                if matches!(
                    rules[rule_index].rhs.get(dot),
                    Some(Symbol::Terminal(terminal)) if *terminal == word[position]
                ) {
                    chart[position + 1].insert((rule_index, dot + 1, origin));
                }
            }
        }
    }

    chart[word.len()].iter().any(|&(index, dot, origin)| {
        origin == 0 && rules[index].lhs == start && dot == rules[index].rhs.len()
    })
}

fn compare_language(input: &[Rule], start: u32, max_len: usize) {
    let normalized = compare_ordered(input, start);
    let mut source = bounded_language(input, start, max_len);
    // Standalone normalization removes epsilon. Source nullability is retained
    // separately by the compiler and is tested at the runtime/persistence seam.
    source.remove(&Vec::new());
    assert_eq!(
        bounded_language(&normalized, start, max_len),
        source,
    );
    for word in &source {
        assert!(earley_accepts(input, start, word));
        assert!(earley_accepts(&normalized, start, word));
    }
}

#[test]
fn reference_validation_is_forced_for_unit_tests() {
    assert!(reference_validation_enabled());
}

#[test]
fn empty_reflexive_and_pure_unit_cycles_match_reference() {
    for rules in [
        Vec::new(),
        vec![rule(0, vec![])],
        vec![rule(0, vec![nt(0)])],
        vec![rule(0, vec![nt(0)]), rule(0, vec![t(0)])],
        vec![rule(0, vec![nt(1)]), rule(1, vec![nt(0)])],
        vec![
            rule(0, vec![nt(1)]),
            rule(1, vec![nt(0)]),
            rule(1, vec![]),
            rule(1, vec![t(0)]),
        ],
    ] {
        compare_language(&rules, 0, 5);
    }
}

#[test]
fn direct_indirect_and_hidden_recursion_preserve_order_and_language() {
    let fixtures = [
        vec![
            rule(0, vec![t(0), nt(0)]),
            rule(0, vec![t(1)]),
        ],
        vec![
            rule(0, vec![t(0), nt(1)]),
            rule(1, vec![t(1), nt(0)]),
            rule(1, vec![t(2)]),
        ],
        vec![
            rule(0, vec![nt(1), t(0)]),
            rule(0, vec![t(2)]),
            rule(1, vec![nt(0), t(1)]),
            rule(1, vec![t(3)]),
        ],
        vec![
            rule(0, vec![nt(2), nt(1), t(0)]),
            rule(0, vec![t(2)]),
            rule(1, vec![nt(0), t(1)]),
            rule(1, vec![t(3)]),
            rule(2, vec![]),
            rule(2, vec![t(4)]),
        ],
        vec![
            rule(0, vec![nt(0), nt(0)]),
            rule(0, vec![t(0)]),
            rule(0, vec![]),
        ],
    ];

    for rules in fixtures {
        compare_language(&rules, 0, 5);
    }
}

#[test]
fn nullable_run_tree_and_fresh_id_resynchronization_are_preserved() {
    let rules = vec![
        rule(0, vec![nt(1), nt(2), nt(3), t(0), nt(0)]),
        rule(0, vec![t(1)]),
        rule(1, vec![]),
        rule(1, vec![t(2)]),
        rule(2, vec![]),
        rule(2, vec![t(3)]),
        rule(3, vec![]),
        rule(3, vec![t(4)]),
    ];
    compare_language(&rules, 0, 4);
}

#[test]
fn all_three_node_unit_graphs_have_exact_ordered_outputs() {
    const N: u32 = 3;
    for mask in 0usize..(1usize << (N * N)) {
        let mut rules = Vec::new();
        for lhs in 0..N {
            for target in 0..N {
                if mask & (1usize << (lhs * N + target)) != 0 {
                    rules.push(rule(lhs, vec![nt(target)]));
                }
            }
            rules.push(rule(lhs, vec![t(lhs)]));
        }
        let actual = compare_ordered(&rules, 0);
        assert_eq!(
            bounded_language(&actual, 0, 2),
            bounded_language(&rules, 0, 2),
            "unit graph {mask:#x}",
        );
    }
}

#[test]
fn right_recursion_public_adapter_preserves_fresh_id_calls_and_completion() {
    let fixtures = [
        vec![
            rule(4, vec![t(1), nt(4)]),
            rule(4, vec![t(2)]),
            rule(1, vec![t(3), nt(1)]),
            rule(1, vec![t(4)]),
        ],
        vec![
            rule(0, vec![t(0), nt(1)]),
            rule(1, vec![t(1), nt(0)]),
            rule(1, vec![t(2)]),
        ],
        vec![
            rule(0, vec![nt(1), nt(2)]),
            rule(1, vec![t(0), nt(0)]),
            rule(1, vec![t(1)]),
            rule(2, vec![]),
        ],
    ];

    for rules in fixtures {
        let mut expected = rules.clone();
        let mut expected_next = View::plain(&rules).max_nt() + 1;
        let expected_completed = reference::eliminate_right_recursion(
            &mut expected,
            &mut || {
                let next = expected_next;
                expected_next += 1;
                next
            },
        );

        let mut actual = rules;
        let mut actual_next = View::plain(&actual).max_nt() + 1;
        let actual_completed = eliminate_right_recursion(
            &mut actual,
            &mut || {
                let next = actual_next;
                actual_next += 1;
                next
            },
        );

        assert_eq!(actual, expected);
        assert_eq!(actual_next, expected_next);
        assert_eq!(actual_completed, expected_completed);
    }
}

#[test]
fn ordered_cycle_search_is_iterative_at_large_graph_depth() {
    let n = 50_000usize;
    let mut graph = BoundaryGraph {
        edges: (0..n)
            .map(|index| {
                if index + 1 < n {
                    vec![index + 1]
                } else {
                    vec![n - 2]
                }
            })
            .collect(),
        ordered: false,
    };
    assert_eq!(
        graph.first_indirect_cycle(),
        Some(vec![(n - 2) as u32, (n - 1) as u32]),
    );
}

#[test]
fn dedup_flattening_is_iterative_at_large_dependency_depth() {
    let n = 20_000u32;
    let mut rules = vec![
        rule(0, vec![nt(1)]),
        rule(0, vec![t(7)]),
    ];
    for id in 1..n {
        rules.push(rule(id, vec![nt(id + 1)]));
    }
    rules.push(rule(n, vec![t(7)]));

    // Deliberately exercise the production implementation directly: the frozen
    // reference uses recursive Rust calls and is not the deep-stack oracle.
    dedup_owned(&mut rules);

    assert_eq!(rules[0], rule(0, vec![nt(1)]));
    assert_eq!(rules.len(), n as usize + 1);
    assert!(earley_accepts(&rules, 0, &[7]));
}

#[test]
fn long_valid_recursive_continuations_have_no_derivation_cap() {
    let source = vec![
        rule(0, vec![t(b'!' as u32), nt(0)]),
        rule(0, vec![t(b'x' as u32)]),
    ];
    let normalized = compare_ordered(&source, 0);

    for depth in [0usize, 1, 17, 257, 1024] {
        let mut word = vec![b'!' as u32; depth];
        word.push(b'x' as u32);
        assert!(earley_accepts(&source, 0, &word));
        assert!(earley_accepts(&normalized, 0, &word));

        word.push(b'!' as u32);
        assert!(!earley_accepts(&source, 0, &word));
        assert!(!earley_accepts(&normalized, 0, &word));
    }
}

#[test]
fn varied_acyclic_and_recursive_grammars_match_ordered_reference() {
    fn next(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed
    }

    let mut seed = 0x4e6f_726d_5f72_656d_u64;
    for case in 0..128 {
        let n = 2 + (next(&mut seed) % 4) as u32;
        let mut rules = Vec::new();
        for lhs in 0..n {
            rules.push(rule(lhs, vec![t(lhs % 3)]));
            if lhs + 1 < n {
                rules.push(rule(lhs, vec![nt(lhs + 1)]));
            }
            if next(&mut seed) & 1 != 0 {
                rules.push(rule(lhs, vec![t(3), nt(lhs)]));
            }
            if next(&mut seed) % 4 == 0 {
                rules.push(rule(lhs, vec![]));
            }
            if next(&mut seed) % 3 == 0 {
                rules.push(rule(lhs, vec![t(lhs % 3)]));
            }
        }
        let normalized = compare_ordered(&rules, 0);
        let mut expected = bounded_language(&rules, 0, 4);
        expected.remove(&Vec::new());
        assert_eq!(
            bounded_language(&normalized, 0, 4),
            expected,
            "case {case}",
        );
    }
}

#[test]
fn arena_publication_moves_unchanged_rhs_allocations() {
    let rules = vec![
        rule(0, vec![t(0), t(1), t(2)]),
        rule(1, vec![t(3)]),
    ];
    let first_pointer = rules[0].rhs.as_ptr();
    let second_pointer = rules[1].rhs.as_ptr();
    let mut edit = Edit::new(rules);
    edit.order.swap(0, 1);
    let published = edit.publish();
    assert_eq!(published[0].rhs.as_ptr(), second_pointer);
    assert_eq!(published[1].rhs.as_ptr(), first_pointer);
}
