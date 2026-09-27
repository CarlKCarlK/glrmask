use std::collections::VecDeque;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;

use super::{
    PossibleOutgoingIds, build_parser_nwa_from_terminal_dwa,
    collapse_final_leaf_targets, determinize_parser_dwa_with_fallbacks,
    determinize_with_supports, immediate_acceptance_certificates,
    local_epsilon_closure, local_epsilon_closure_canonical,
    try_build_direct_regular_parser_top_accept_parts,
    try_build_direct_regular_parser_top_accept_parts_table_product_reference,
    try_build_immediate_parser_top_accept_parts,
    subtract_final_weights_from_outgoing_dwa_impl,
};
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::nwa::NWA;
use crate::automata::weighted::terminal_automaton::TerminalAutomaton;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::labels::DEFAULT_LABEL;
use crate::compiler::glr::table::testing::build_test_table;
use crate::compiler::glr::table::Action;
use crate::compiler::stages::resolve_negatives::resolve_negative_codes_in_nwa;
use crate::compiler::stages::templates::Templates;
use crate::ds::weight::Weight;
use crate::grammar::flat::{
    DirectRegularAutomaton, GrammarDef, Rule, Symbol, Terminal,
};

#[test]
fn extended_boundary_weight_rows_preserve_exact_finite_algebra() {
    for rows in [1usize, 16, 17, 64] {
        let mut interner = super::FastBoundaryWeightInterner::new(rows, 64).unwrap();
        let domain = Weight::from_uniform(0..=rows as u32 - 1, RangeSetBlaze::from_iter([0..=63]));
        let mut sources = vec![Weight::empty(), Weight::all()];
        for i in 0..24u32 {
            sources.push(Weight::from_per_tsid_token_sets((0..rows as u32).filter_map(|row| {
                let tokens = (0..64).filter(|&token| (token + 3 * row + i) % 11 < (i % 7) + 1).collect::<RangeSetBlaze<u32>>();
                (!tokens.is_empty()).then_some((row, tokens))
            })));
        }
        let mut by_ptr = rustc_hash::FxHashMap::default();
        let ids = sources.iter().map(|weight| interner.source_weight_id(weight, &mut by_ptr, None).unwrap()).collect::<Vec<_>>();
        for (a, &left) in sources.iter().zip(&ids) {
            assert_eq!(domain.intersection(&interner.to_weight(left)), domain.intersection(a));
            for (b, &right) in sources.iter().zip(&ids) {
                let union = interner.union(left, right);
                let intersection = interner.intersection(left, right);
                let difference = interner.difference(left, right);
                assert_eq!(domain.intersection(&interner.to_weight(union)), domain.intersection(&a.union(b)));
                assert_eq!(domain.intersection(&interner.to_weight(intersection)), domain.intersection(&a.intersection(b)));
                assert_eq!(domain.intersection(&interner.to_weight(difference)), domain.intersection(a).difference(&domain.intersection(b)));
            }
        }
        assert_eq!(interner.compact_runtime_weights().is_some(), rows <= 16,
            "large compile domains must not be truncated into the 16-row runtime format");
    }
    assert!(super::FastBoundaryWeightInterner::new(65, 64).is_none());
}

fn weight(tokens: std::ops::RangeInclusive<u32>) -> Weight {
    Weight::from_token_set_for_tsid(0, RangeSetBlaze::from_iter([tokens]))
}

fn eval_with_default(dwa: &DWA, word: &[i32]) -> Weight {
    let mut state_id = dwa.start_state();
    let mut accumulated = Weight::all();
    for &label in word {
        let Some((target, edge_weight)) = dwa.states()[state_id as usize]
            .transitions
            .get(&label)
            .or_else(|| dwa.states()[state_id as usize].transitions.get(&DEFAULT_LABEL))
        else {
            return Weight::empty();
        };
        accumulated = accumulated.intersection(edge_weight);
        if accumulated.is_empty() {
            return accumulated;
        }
        state_id = *target;
    }
    dwa.states()[state_id as usize]
        .final_weight
        .as_ref()
        .map_or_else(Weight::empty, |final_weight| {
            accumulated.intersection(final_weight)
        })
}

#[test]
fn flat_canonical_epsilon_closure_matches_map_reference() {
    fn next_u32(state: &mut u64) -> u32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*state >> 32) as u32
    }

    const STATE_COUNT: usize = 64;
    let mut random = 0x8f4d_3a2b_1907_65ceu64;
    let mut nwa = NWA::new(0, 0);
    for _ in 0..STATE_COUNT {
        nwa.add_state();
    }
    for source in 0..STATE_COUNT - 1 {
        let remaining = STATE_COUNT - source - 1;
        let edge_count = 1 + next_u32(&mut random) as usize % remaining.min(4);
        for _ in 0..edge_count {
            let target = source + 1 + next_u32(&mut random) as usize % remaining;
            let start = next_u32(&mut random) % 24;
            let end = (start + next_u32(&mut random) % 8).min(31);
            nwa.add_epsilon(source as u32, target as u32, weight(start..=end));
        }
    }

    for case in 0..256 {
        let mut seeds = FxHashMap::<u32, Weight>::default();
        let seed_count = 1 + next_u32(&mut random) as usize % 8;
        for _ in 0..seed_count {
            let state = next_u32(&mut random) as usize % STATE_COUNT;
            let start = next_u32(&mut random) % 24;
            let end = (start + next_u32(&mut random) % 8).min(31);
            let add = weight(start..=end);
            seeds
                .entry(state as u32)
                .and_modify(|existing| *existing = existing.union(&add))
                .or_insert(add);
        }

        let mut reference = seeds.clone();
        let mut reference_weights = vec![None; STATE_COUNT];
        let mut reference_queue = VecDeque::new();
        local_epsilon_closure(
            &nwa,
            &mut reference_weights,
            &mut reference_queue,
            &mut reference,
        );
        let mut reference_canonical = reference.into_iter().collect::<Vec<_>>();
        reference_canonical.sort_unstable_by_key(|(state, _)| *state);

        let mut seed_canonical = seeds.into_iter().collect::<Vec<_>>();
        seed_canonical.sort_unstable_by_key(|(state, _)| *state);
        let mut flat_weights = vec![None; STATE_COUNT];
        let mut flat_queue = VecDeque::new();
        let mut touched = Vec::new();
        let mut flat_canonical = Vec::new();
        local_epsilon_closure_canonical(
            &nwa,
            &mut flat_weights,
            &mut flat_queue,
            &seed_canonical,
            &mut touched,
            &mut flat_canonical,
            &mut super::ScopedWeightOpCache::default(),
            true,
        );

        assert_eq!(
            flat_canonical, reference_canonical,
            "epsilon-closure mismatch in generated case {case}",
        );
        assert!(flat_weights.iter().all(Option::is_none));
        assert!(flat_queue.is_empty());
    }
}

#[test]
fn immediate_acceptance_parts_union_matches_combined_certificates() {
    let mut terminal_dwa = DWA::new(1, 31);
    let accept_a = terminal_dwa.add_state();
    let accept_b = terminal_dwa.add_state();
    let accept_c = terminal_dwa.add_state();
    terminal_dwa.set_final_weight(accept_a, Weight::all());
    terminal_dwa.set_final_weight(accept_b, Weight::all());
    terminal_dwa.set_final_weight(accept_c, Weight::all());
    terminal_dwa.add_transition(
        terminal_dwa.start_state(),
        0,
        accept_a,
        weight(0..=7),
    );
    terminal_dwa.add_transition(
        terminal_dwa.start_state(),
        1,
        accept_b,
        weight(4..=15),
    );
    terminal_dwa.add_transition(
        terminal_dwa.start_state(),
        2,
        accept_c,
        weight(12..=23),
    );
    let terminal_automaton = TerminalAutomaton::Dwa(terminal_dwa);

    let grammar = AnalyzedGrammar::from_grammar_def(&GrammarDef {
        rules: vec![Rule {
            lhs: 0,
            rhs: vec![Symbol::Terminal(0)],
        }],
        start: 0,
        terminals: (0..3)
            .map(|id| Terminal::Literal {
                id,
                bytes: vec![b'a' + id as u8],
            })
            .collect(),
        ..GrammarDef::default()
    });
    let table = build_test_table(
        3,
        3,
        &[
            &[
                (0, Action::Shift(0, true)),
                (1, Action::Shift(1, true)),
            ],
            &[
                (1, Action::Shift(1, true)),
                (2, Action::Shift(2, true)),
            ],
            &[],
        ],
        &[&[], &[], &[]],
    );

    let parts = try_build_immediate_parser_top_accept_parts(
        &terminal_automaton,
        &grammar,
        &table,
    )
    .expect("test terminal automaton is an immediate-completion family");
    let combined = immediate_acceptance_certificates(&terminal_automaton, &grammar, &table);

    assert_eq!(parts.get(&0).map(Vec::len), Some(2));
    assert_eq!(parts.get(&1).map(Vec::len), Some(2));
    assert!(!parts.contains_key(&2));
    for parser_top in 0..table.num_states {
        let parts_union = parts
            .get(&(parser_top as i32))
            .map(|weights| Weight::union_all(weights.iter()))
            .unwrap_or_else(Weight::empty);
        assert_eq!(parts_union, combined[parser_top as usize]);
    }
}

#[test]
fn direct_regular_parser_product_matches_generic_parser_dwa() {
    let mut terminal_dwa = DWA::new(1, 31);
    let after_zero = terminal_dwa.add_state();
    let accept = terminal_dwa.add_state();
    terminal_dwa.set_final_weight(after_zero, weight(0..=7));
    terminal_dwa.set_final_weight(accept, Weight::all());
    terminal_dwa.add_transition(
        terminal_dwa.start_state(),
        0,
        after_zero,
        weight(0..=15),
    );
    terminal_dwa.add_transition(after_zero, 0, after_zero, weight(8..=12));
    terminal_dwa.add_transition(after_zero, 1, accept, weight(4..=20));
    terminal_dwa.add_transition(
        terminal_dwa.start_state(),
        2,
        accept,
        weight(16..=23),
    );
    let terminal_automaton = TerminalAutomaton::Dwa(terminal_dwa);

    let grammar = AnalyzedGrammar::from_grammar_def(&GrammarDef {
        rules: vec![Rule {
            lhs: 0,
            rhs: vec![Symbol::Terminal(0)],
        }],
        start: 0,
        terminals: (0..3)
            .map(|id| Terminal::Literal {
                id,
                bytes: vec![b'a' + id as u8],
            })
            .collect(),
        direct_regular_automaton: Some(DirectRegularAutomaton {
            states: vec![
                crate::grammar::flat::DirectRegularState {
                    is_accepting: false,
                    transitions: [
                        (0, vec![0]),
                        (1, vec![1]),
                        (2, vec![1]),
                    ]
                    .into_iter()
                    .collect(),
                    epsilons: Vec::new(),
                },
                crate::grammar::flat::DirectRegularState {
                    is_accepting: true,
                    transitions: Default::default(),
                    epsilons: Vec::new(),
                },
            ],
            start_states: vec![0],
        }),
        ..GrammarDef::default()
    });
    let table = build_test_table(
        3,
        3,
        &[
            &[
                (0, Action::Shift(1, true)),
                (1, Action::Shift(2, true)),
                (2, Action::Shift(2, true)),
            ],
            &[
                (0, Action::Shift(1, true)),
                (1, Action::Shift(2, true)),
                (2, Action::Shift(2, true)),
            ],
            &[],
        ],
        &[&[], &[], &[]],
    );
    let templates = Templates::from_direct_regular_table(&table, grammar.num_terminals)
        .expect("test table has direct-regular actions");
    let (mut generic_nwa, _) = build_parser_nwa_from_terminal_dwa(
        &terminal_automaton,
        &grammar,
        &templates,
        &table,
    )
    .expect("generic parser NWA should build for direct templates");
    resolve_negative_codes_in_nwa(&mut generic_nwa, false);
    let generic = determinize_with_supports(&generic_nwa, Some(table.num_states)).dwa;
    let direct = try_build_direct_regular_parser_top_accept_parts(
        &terminal_automaton,
        &grammar,
        &table,
    )
    .expect("sparse direct product should accept the direct parser metadata");
    let table_reference =
        try_build_direct_regular_parser_top_accept_parts_table_product_reference(
            &terminal_automaton,
            &grammar,
            &table,
        )
        .expect("table-product reference should accept the direct parser table");

    for parser_top in 0..table.num_states {
        let direct_weight = direct
            .get(&(parser_top as i32))
            .map(|weights| Weight::union_all(weights.iter()))
            .unwrap_or_else(Weight::empty);
        let reference_weight = table_reference
            .get(&(parser_top as i32))
            .map(|weights| Weight::union_all(weights.iter()))
            .unwrap_or_else(Weight::empty);
        assert_eq!(
            direct_weight,
            reference_weight,
            "sparse and table products differ at parser top {parser_top}",
        );
        let mut generic_prefix_weight = generic
            .states()
            .get(generic.start_state() as usize)
            .and_then(|state| state.final_weight.clone())
            .unwrap_or_else(Weight::empty);
        generic_prefix_weight = generic_prefix_weight.union(
            &generic.eval_word(&[parser_top as i32]),
        );
        assert_eq!(
            direct_weight,
            generic_prefix_weight,
            "direct product mismatch at parser top {parser_top}",
        );
    }
}

#[test]
fn parallel_final_subtraction_matches_serial_rows() {
    let mut source = DWA::new(1, 31);
    let left = source.add_state();
    let right = source.add_state();
    source.set_final_weight(source.start_state(), weight(4..=11));
    source.add_transition(source.start_state(), 1, left, weight(0..=15));
    source.add_transition(source.start_state(), 2, right, weight(8..=20));
    source.set_final_weight(left, weight(0..=3));
    source.add_transition(left, 3, right, weight(0..=9));
    source.set_final_weight(right, weight(16..=23));
    source.add_transition(right, 4, left, weight(12..=27));

    let mut serial = source.clone();
    let mut parallel = source;
    subtract_final_weights_from_outgoing_dwa_impl(&mut serial, false);
    subtract_final_weights_from_outgoing_dwa_impl(&mut parallel, true);

    assert_eq!(serial.start_state(), parallel.start_state());
    assert_eq!(serial.states().len(), parallel.states().len());
    for (serial_state, parallel_state) in serial.states().iter().zip(parallel.states()) {
        assert_eq!(serial_state.final_weight, parallel_state.final_weight);
        assert_eq!(serial_state.transitions, parallel_state.transitions);
    }
}

#[test]
fn fallback_determinization_reuses_singleton_source_states() {
    let mut source = DWA::new(1, 31);
    let middle = source.add_state();
    let leaf = source.add_state();
    source.add_transition(source.start_state(), 10, middle, weight(0..=15));
    source.add_transition(source.start_state(), 11, middle, weight(8..=23));
    source.set_final_weight(middle, weight(4..=19));
    source.add_transition(middle, 12, leaf, weight(2..=27));
    source.set_final_weight(leaf, weight(6..=25));

    let possible = (0..source.states().len())
        .map(|_| PossibleOutgoingIds::Empty)
        .collect::<Vec<_>>();
    let determinized = determinize_parser_dwa_with_fallbacks(&source, &possible, 32);

    for word in [
        vec![],
        vec![10],
        vec![11],
        vec![10, 12],
        vec![11, 12],
        vec![12],
    ] {
        assert_eq!(
            determinized.eval_word(&word),
            source.eval_word(&word),
            "word={word:?}",
        );
    }
    assert_eq!(determinized.states().len(), source.states().len());
}

#[test]
fn fallback_determinization_combines_explicit_and_default_branches() {
    let mut source = DWA::new(1, 31);
    let explicit = source.add_state();
    let fallback = source.add_state();
    source.add_transition(source.start_state(), 7, explicit, weight(0..=15));
    source.add_transition(
        source.start_state(),
        DEFAULT_LABEL,
        fallback,
        weight(8..=23),
    );
    source.set_final_weight(explicit, weight(0..=5));
    source.set_final_weight(fallback, weight(12..=27));

    let possible = vec![
        PossibleOutgoingIds::All,
        PossibleOutgoingIds::Empty,
        PossibleOutgoingIds::Empty,
    ];
    let determinized = determinize_parser_dwa_with_fallbacks(&source, &possible, 32);

    let explicit_result = weight(0..=5);
    let fallback_result = weight(12..=23);
    assert_eq!(
        eval_with_default(&determinized, &[7]),
        explicit_result.union(&fallback_result),
    );
    assert_eq!(eval_with_default(&determinized, &[8]), fallback_result);
    assert_eq!(determinized.eval_word(&[DEFAULT_LABEL]), weight(12..=23));
}

#[test]
fn final_leaf_weights_are_pushed_into_shared_sink_edges() {
    let mut dwa = DWA::new(1, 5);
    let left = dwa.add_state();
    let right = dwa.add_state();
    dwa.add_transition(0, 10, left, weight(0..=5));
    dwa.add_transition(0, 11, right, weight(0..=5));
    dwa.set_final_weight(left, weight(0..=2));
    dwa.set_final_weight(right, weight(3..=4));

    let collapsed = collapse_final_leaf_targets(dwa);

    assert_eq!(collapsed.states().len(), 2);
    assert_eq!(collapsed.eval_word(&[10]), weight(0..=2));
    assert_eq!(collapsed.eval_word(&[11]), weight(3..=4));
    let targets: Vec<u32> = collapsed.states()[collapsed.start_state() as usize]
        .transitions
        .values()
        .map(|(target, _)| *target)
        .collect();
    assert_eq!(targets.len(), 2);
    assert!(targets.iter().all(|target| *target == targets[0]));
    assert!(collapsed.states()[targets[0] as usize]
        .final_weight
        .as_ref()
        .is_some_and(Weight::is_full));
}

#[test]
fn nonleaf_continuations_are_not_shortened() {
    let mut dwa = DWA::new(1, 5);
    let middle = dwa.add_state();
    let leaf = dwa.add_state();
    dwa.add_transition(0, 10, middle, weight(0..=5));
    dwa.add_transition(middle, 11, leaf, weight(0..=5));
    dwa.set_final_weight(leaf, weight(1..=3));

    let collapsed = collapse_final_leaf_targets(dwa);

    assert_eq!(collapsed.states().len(), 3);
    assert!(collapsed.eval_word(&[10]).is_empty());
    assert_eq!(collapsed.eval_word(&[10, 11]), weight(1..=3));
}
