use super::*;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::labels::encode_negative_label;
use crate::parser_dwa::normalize_weighted_stack_predicate_for_symbol_count;
use crate::resolve_negatives::{resolve_negative_codes_in_nwa, resolve_negative_codes_in_nwa_with_pop_classes};
use range_set_blaze::RangeSetBlaze;

fn weight(bits: u8) -> Weight {
    Weight::from_uniform(0..=0, (0..2u32).filter(|&id| bits & (1 << id) != 0).collect::<RangeSetBlaze<u32>>())
}
fn bits(weight: &Weight) -> u8 {
    (0..2).fold(0, |result, token| result | if weight.tokens_for_tsid(0).contains(token) { 1 << token } else { 0 })
}

fn literal(graph: &NWA, classes: &PopLabelClasses, stack: &[u32]) -> u8 {
    let mut reached = BTreeMap::<(u32, Vec<u32>), u8>::new();
    let mut queue = VecDeque::new();
    for &state in graph.start_states() {
        reached.insert((state, stack.to_vec()), 3); queue.push_back((state, stack.to_vec()));
    }
    let mut result = 0;
    while let Some(key) = queue.pop_front() {
        let live = reached[&key];
        let (state, stack) = key;
        let row = &graph.states()[state as usize];
        if let Some(final_weight) = &row.final_weight { result |= live & bits(final_weight); }
        let mut add = |target: u32, stack: Vec<u32>, coefficient: &Weight| {
            let value = live & bits(coefficient);
            let key = (target, stack);
            let previous = reached.entry(key.clone()).or_default();
            if value & !*previous != 0 { *previous |= value; queue.push_back(key); }
        };
        for (target, coefficient) in &row.epsilons { add(*target, stack.clone(), coefficient); }
        for (&label, targets) in &row.transitions {
            let mut next = stack.clone();
            if label < 0 { next.push(negative_to_positive_label(label) as u32); }
            else {
                let Some(top) = next.pop() else { continue; };
                if top as i32 != label && !classes.matches(label, top) { continue; }
            }
            for (target, coefficient) in targets { add(*target, next.clone(), coefficient); }
        }
        assert!(reached.len() < 100_000, "literal oracle is only used on finite configuration graphs");
    }
    result
}

fn accepted(dwa: &DWA, stack: &[u32]) -> u8 {
    let mut state = dwa.start_state(); let mut live = 3;
    let mut result = dwa.states()[state as usize].final_weight.as_ref().map_or(0, bits);
    for &top in stack.iter().rev() {
        let row = &dwa.states()[state as usize];
        let Some((target, coefficient)) = row.transitions.get(&(top as i32)).or_else(|| row.transitions.get(&DEFAULT_LABEL)) else { break; };
        live &= bits(coefficient); state = *target;
        result |= live & dwa.states()[state as usize].final_weight.as_ref().map_or(0, bits);
    }
    result
}

fn compiled(graph: &NWA, classes: &PopLabelClasses) -> DWA {
    let mut graph = graph.clone();
    resolve_negative_codes_in_nwa_with_pop_classes(&mut graph, classes).unwrap();
    let result = classes.compile_positive(graph, 1_000_000).unwrap();
    assert!(result.states().iter().all(|row| row.transitions.keys().all(|&label|
        label == DEFAULT_LABEL || (label >= 0 && label < classes.symbol_count() as i32))));
    result
}

#[test]
fn scoped_defaults_reject_foreign_cancellation_and_preserve_dead_exclusions() {
    let mut classes = PopLabelClasses::new(108).unwrap();
    let local = classes.intern_scoped_complement(100..104, [103]).unwrap().unwrap();
    for symbol in [5, 100, 103, 105] {
        let mut graph = NWA::new(1, 2);
        let start = graph.add_state(); let pop = graph.add_state(); let end = graph.add_state();
        graph.set_start_states(vec![start]); graph.set_final_weight(end, weight(3));
        graph.add_transition(start, encode_negative_label(symbol), pop, weight(3));
        graph.add_transition(pop, local, end, weight(3));
        let predicate = compiled(&graph, &classes);
        assert_eq!(accepted(&predicate, &[]), if symbol == 100 { 3 } else { 0 });
        for word in [vec![], vec![101], vec![5, 101], vec![103, 101]] {
            assert_eq!(accepted(&predicate, &word), literal(&graph, &classes, &word));
        }
    }
    assert!(!classes.matches(local, 5));
    assert!(!classes.matches(local, 103));
    assert!(classes.matches(local, 102));
    assert_eq!(classes.exclusions[0].as_ref(), &[103], "foreign ranges remain symbolic");
}

fn all_stacks(depth: usize) -> Vec<Vec<u32>> {
    let mut all = vec![vec![]]; let mut layer = vec![vec![]];
    for _ in 0..depth {
        let mut next = Vec::new();
        for stack in layer { for top in 0..3 {
            let mut word = stack.clone(); word.push(top); next.push(word);
        }}
        all.extend(next.iter().cloned()); layer = next;
    }
    all
}

#[test]
fn pop_classes_consume_input_and_preserve_dead_default_exceptions() {
    let mut classes = PopLabelClasses::new(3).unwrap();
    let complement = classes.intern_complement([0]).unwrap().unwrap();
    assert_eq!(classes.intern_complement([0, 0]).unwrap(), Some(complement));
    let mut graph = NWA::new(1, 1);
    let start = graph.add_state(); let end = graph.add_state();
    graph.set_start_states(vec![start]); graph.set_final_weight(end, weight(3));
    graph.add_transition(start, complement, end, weight(3));
    let dwa = compiled(&graph, &classes);
    for stack in all_stacks(4) { assert_eq!(accepted(&dwa, &stack), literal(&graph, &classes, &stack)); }
    assert_eq!(accepted(&dwa, &[]), 0);
    assert_eq!(accepted(&dwa, &[0]), 0);
    assert_eq!(accepted(&dwa, &[1]), 3);
    // A distinct nondeterministic branch may admit the excluded symbol without
    // changing which branch's coefficient contributes to that symbol.
    graph.states_mut()[end as usize].final_weight = Some(weight(3));
    graph.add_transition(start, 0, end, weight(1));
    let dwa = compiled(&graph, &classes);
    assert_eq!(accepted(&dwa, &[0]), 1);
    assert_eq!(accepted(&dwa, &[1]), 3);
}

#[test]
fn classed_cancellation_matches_literal_weighted_acyclic_programs() {
    let mut classes = PopLabelClasses::new(3).unwrap();
    let labels = [0, 1, 2, encode_negative_label(0), encode_negative_label(1), encode_negative_label(2),
        classes.intern_complement([]).unwrap().unwrap(),
        classes.intern_complement([0]).unwrap().unwrap(),
        classes.intern_complement([1, 2]).unwrap().unwrap()];
    let mut seed = 271828_u64;
    let mut random = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); (seed >> 32) as usize };
    let stacks = all_stacks(5);
    for case in 0..256 {
        let count = 3 + random() % 7;
        let mut states = vec![NWAState::default(); count];
        for from in 0..count {
            if random() % 3 == 0 { states[from].final_weight = Some(weight((random() % 3 + 1) as u8)); }
            for to in from+1..count {
                if random() % 3 != 0 { continue; }
                let coefficient = weight((random() % 3 + 1) as u8);
                if random() % 5 == 0 { states[from].epsilons.push((to as u32, coefficient)); }
                else { states[from].transitions.entry(labels[random() % labels.len()]).or_default().push((to as u32, coefficient)); }
            }
        }
        let graph = NWA::from_parts(states, vec![0]);
        let dwa = compiled(&graph, &classes);
        for stack in &stacks { assert_eq!(accepted(&dwa, stack), literal(&graph, &classes, stack), "case={case} stack={stack:?}"); }
    }
}

#[test]
fn classed_control_cycles_preserve_exact_finite_and_unbounded_pop_closures() {
    let mut classes = PopLabelClasses::new(3).unwrap();
    let pop_nonzero = classes.intern_complement([0]).unwrap().unwrap();
    let mut graph = NWA::new(1, 1);
    let start = graph.add_state(); let end = graph.add_state();
    graph.set_start_states(vec![start]); graph.set_final_weight(end, weight(3));
    graph.add_transition(start, pop_nonzero, start, weight(3));
    graph.add_transition(start, 0, end, weight(3));
    let dwa = compiled(&graph, &classes);
    for depth in 0..256 {
        let mut stack = vec![0]; stack.extend((0..depth).map(|index| 1 + index % 2));
        assert_eq!(accepted(&dwa, &stack), 3);
    }
    for stack in all_stacks(5) { assert_eq!(accepted(&dwa, &stack), literal(&graph, &classes, &stack)); }
    // Depth-preserving control cycles exercise transitive PUSH/class-POP
    // cancellation in the same weighted least fixed point as ordinary labels.
    let mut states = vec![NWAState::default(); 4];
    states[0].transitions.insert(pop_nonzero, vec![(1, weight(3))]);
    states[1].transitions.insert(encode_negative_label(1), vec![(0, weight(1))]);
    states[1].transitions.insert(encode_negative_label(2), vec![(0, weight(2))]);
    states[0].transitions.insert(1, vec![(2, weight(1))]);
    states[0].transitions.insert(2, vec![(3, weight(2))]);
    states[2].final_weight = Some(weight(3)); states[3].final_weight = Some(weight(3));
    let graph = NWA::from_parts(states, vec![0]); let dwa = compiled(&graph, &classes);
    for stack in all_stacks(4) { assert_eq!(accepted(&dwa, &stack), literal(&graph, &classes, &stack)); }
}

#[test]
fn pop_class_validation_and_expansion_budgets_do_not_publish_partial_programs() {
    let mut classes = PopLabelClasses::new(3).unwrap();
    assert_eq!(classes.intern_complement([0, 1, 2]).unwrap(), None);
    assert!(classes.intern_complement([3]).is_err());
    let wildcard = classes.intern_complement([]).unwrap().unwrap();
    let mut graph = NWA::from_parts(vec![NWAState::default(); 2], vec![0]);
    graph.set_final_weight(1, weight(3));
    graph.add_transition(0, wildcard, 1, weight(3));
    assert!(classes.expand_positive(graph.clone(), 2).is_err());
    assert!(classes.expand_positive(graph.clone(), 3).is_ok());
    for label in [DEFAULT_LABEL, 3, encode_negative_label(3)] {
        let mut bad = graph.clone(); bad.add_transition(0, label, 1, weight(3));
        assert!(resolve_negative_codes_in_nwa_with_pop_classes(&mut bad, &classes).is_err());
    }
    // An ordinary all-concrete program still uses exactly the same solver.
    let mut ordinary = NWA::from_parts(vec![NWAState::default(); 3], vec![0]);
    ordinary.add_transition(0, encode_negative_label(2), 1, weight(3));
    ordinary.add_transition(1, 2, 2, weight(3)); ordinary.set_final_weight(2, weight(3));
    let mut reference = ordinary.clone(); resolve_negative_codes_in_nwa(&mut reference, false);
    let expected = normalize_weighted_stack_predicate_for_symbol_count(3, &reference);
    let candidate = compiled(&ordinary, &classes);
    for stack in all_stacks(3) { assert_eq!(accepted(&candidate, &stack), accepted(&expected, &stack)); }
}

#[test]
fn symbolic_compression_before_substitution_avoids_repeated_alphabet_expansion() {
    let mut classes = PopLabelClasses::new(4096).unwrap();
    let all = classes.intern_complement([]).unwrap().unwrap();
    let mut graph = NWA::from_parts(vec![NWAState::default(); 257], vec![0]);
    for state in 1..=256 {
        graph.add_transition(0, all, state, weight(if state % 2 == 0 { 1 } else { 2 }));
        graph.set_final_weight(state, weight(3));
    }
    assert!(classes.expand_positive(graph.clone(), 4096).is_err());
    let compact = classes.expand_positive_compressed(graph, 4096).unwrap();
    assert_eq!(compact.states().len(), 2);
    assert_eq!(compact.num_transitions(), 4096);
    let predicate = normalize_weighted_stack_predicate_for_symbol_count(4096, &compact);
    assert_eq!(accepted(&predicate, &[]), 0);
    for top in 0..4096 { assert_eq!(accepted(&predicate, &[top]), 3); }
}

#[test]
fn symbolic_substitution_preserves_overlapping_class_and_literal_coefficients() {
    let mut classes = PopLabelClasses::new(3).unwrap();
    let not_zero = classes.intern_complement([0]).unwrap().unwrap();
    let not_one = classes.intern_complement([1]).unwrap().unwrap();
    let mut graph = NWA::from_parts(vec![NWAState::default(); 4], vec![0]);
    graph.add_transition(0, not_zero, 1, weight(1));
    graph.add_transition(0, not_one, 2, weight(2));
    graph.add_transition(0, 0, 3, weight(1));
    graph.add_transition(1, not_one, 3, weight(1));
    graph.add_transition(2, not_zero, 3, weight(2));
    for state in 1..4 { graph.set_final_weight(state, weight(3)); }
    let direct = classes.expand_positive(graph.clone(), 100).unwrap();
    let compact = classes.expand_positive_compressed(graph.clone(), 100).unwrap();
    let direct = normalize_weighted_stack_predicate_for_symbol_count(3, &direct);
    let compact = normalize_weighted_stack_predicate_for_symbol_count(3, &compact);
    for stack in all_stacks(4) {
        let expected = literal(&graph, &classes, &stack);
        assert_eq!(accepted(&direct, &stack), expected);
        assert_eq!(accepted(&compact, &stack), expected);
    }
}

#[test]
fn class_derivatives_keep_empty_exceptions_and_never_lift_consuming_finals() {
    let mut classes = PopLabelClasses::new(4096).unwrap();
    let except_zero = classes.intern_complement([0]).unwrap().unwrap();
    let mut graph = NWA::from_parts(vec![NWAState::default(); 2], vec![0]);
    graph.add_transition(0, except_zero, 1, weight(3));
    graph.set_final_weight(1, weight(3));
    // Two output rows suffice for this 4096-symbol language: one DEFAULT and
    // a real rejecting exception. There must be no 4095-edge expansion.
    let result = classes.compile_positive(graph.clone(), 2).unwrap();
    assert_eq!(result.num_transitions(), 2);
    assert!(result.states()[result.start_state() as usize].transitions.contains_key(&0));
    assert!(result.states()[result.start_state() as usize].transitions.contains_key(&DEFAULT_LABEL));
    assert_eq!(accepted(&result, &[]), 0);
    assert_eq!(accepted(&result, &[0]), 0);
    for top in 1..4096 { assert_eq!(accepted(&result, &[top]), 3); }
    assert!(classes.compile_positive(graph, 1).unwrap_err().contains("budget"));
}

#[test]
fn small_scoped_class_does_not_materialize_foreign_rejecting_edges() {
    let mut classes = PopLabelClasses::new(4096).unwrap();
    let local = classes.intern_scoped_complement(100..104, [103]).unwrap().unwrap();
    let mut graph = NWA::from_parts(vec![NWAState::default(); 3], vec![0]);
    graph.add_transition(0, local, 1, weight(3));
    graph.add_transition(1, local, 2, weight(3));
    graph.set_final_weight(2, weight(3));
    // Two local rows each need only three consuming edges. Foreign symbols
    // must reject at every depth without thousands of explicit dead shadows.
    let result = classes.compile_positive(graph.clone(), 6).unwrap();
    assert_eq!(result.num_transitions(), 6);
    assert_eq!(accepted(&result, &[]), 0);
    for symbol in 0..4096 {
        let expected = if (100..103).contains(&symbol) { 3 } else { 0 };
        assert_eq!(accepted(&result, &[100, symbol]), expected);
        assert_eq!(accepted(&result, &[symbol, 100]), expected);
        assert_eq!(accepted(&result, &[symbol]), 0);
    }
    assert!(classes.compile_positive(graph, 5).unwrap_err().contains("budget"));
}

#[test]
fn class_derivatives_match_concrete_expansion_over_complete_alphabet_exceptions() {
    let mut classes = PopLabelClasses::new(3).unwrap();
    let not_zero = classes.intern_complement([0]).unwrap().unwrap();
    let not_one = classes.intern_complement([1]).unwrap().unwrap();
    let not_two = classes.intern_complement([2]).unwrap().unwrap();
    let mut graph = NWA::from_parts(vec![NWAState::default(); 5], vec![0]);
    graph.add_transition(0, not_zero, 1, weight(1));
    graph.add_transition(0, not_one, 2, weight(2));
    graph.add_transition(0, not_two, 3, weight(3));
    graph.add_transition(1, 2, 4, weight(1));
    graph.add_transition(2, 0, 4, weight(2));
    graph.add_transition(3, 1, 4, weight(3));
    graph.set_final_weight(4, weight(3));
    let result = classes.compile_positive(graph.clone(), 64).unwrap();
    let expanded = classes.expand_positive(graph.clone(), 64).unwrap();
    let reference = normalize_weighted_stack_predicate_for_symbol_count(3, &expanded);
    for stack in all_stacks(5) {
        assert_eq!(accepted(&result, &stack), literal(&graph, &classes, &stack));
        assert_eq!(accepted(&result, &stack), accepted(&reference, &stack));
    }
}

#[test]
fn direct_dwa_publication_matches_literal_class_paths_with_overlap_dead_guards_and_cycles() {
    // Symbol 3 is foreign to every local class, but valid in the global stack alphabet.
    let mut classes=PopLabelClasses::new(4).unwrap();
    let first=classes.intern_scoped_complement(1..3,[2]).unwrap().unwrap();
    let overlap=classes.intern_scoped_complement(0..3,[0]).unwrap().unwrap();
    let deep=classes.intern_scoped_complement(1..3,[1]).unwrap().unwrap();
    let mut source=DWA::from_parts(vec![Default::default();4],0);
    source.add_transition(0,first,1,weight(3));
    source.add_transition(0,overlap,2,weight(2));
    source.add_transition(0,2,3,Weight::empty());
    source.set_final_weight(1,weight(1));
    source.add_transition(1,2,0,weight(3));
    source.add_transition(2,deep,3,weight(2));source.set_final_weight(3,weight(2));
    let raw=source.to_nwa();
    let result=classes.compile_positive_dwa(source,10000).unwrap();
    for stack in all_stacks(5).into_iter().chain([vec![3],vec![3,1],vec![3,2],vec![0,2]]) {
        assert_eq!(accepted(&result,&stack),literal(&raw,&classes,&stack),"stack={stack:?}");
    }
    let reference=classes.compile_positive(raw,10000).unwrap();
    let comparison=crate::parser_equivalence::compare_parser_mask_prefix_languages(&reference,&result,4,10000).unwrap();
    assert!(comparison.difference.is_none(),"{:?}",comparison.difference);
}

#[test]
fn direct_dwa_publication_rejects_negative_push_unknown_labels_and_budget_refusal() {
    let classes=PopLabelClasses::new(3).unwrap();
    for label in [encode_negative_label(0),DEFAULT_LABEL,7] {
        let mut source=DWA::from_parts(vec![Default::default();2],0);
        source.add_transition(0,label,1,weight(3));source.set_final_weight(1,weight(3));
        assert!(classes.compile_positive_dwa(source,10000).is_err());
    }
    let mut source=DWA::from_parts(vec![Default::default();2],0);
    source.add_transition(0,1,1,weight(3));source.set_final_weight(1,weight(3));
    assert!(classes.compile_positive_dwa(source,0).is_err());
}
