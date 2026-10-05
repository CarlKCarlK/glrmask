use super::*;
use super::prepare::prepare_terminal_inventory;

use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::labels::DEFAULT_LABEL;
use crate::runtime::artifact::FastCommitTemplateDfas;

use glrmask_parser_dwa::__private::templates::compile_dfa::{
    specialize_template_dfa_defaults_for_commit_split_input,
    try_split_commit_template_dfas,
};

fn phase_fixture(seed: u32) -> DFA {
    let mut dfa = DFA::new();
    let read_pop = dfa.add_state();
    let read_push = dfa.add_state();
    let accepted = dfa.add_state();
    let top = (seed % 7) as i32;
    dfa.add_transition(0, top, read_pop);
    dfa.add_transition(read_pop, encode_negative_label(top as u32), read_push);
    dfa.add_transition(read_push, encode_negative_label(17), accepted);
    dfa.set_accepting(accepted, true);
    let mut tail = dfa.add_state();
    dfa.add_transition(0, DEFAULT_LABEL, tail);
    for depth in 0..48 {
        let next = dfa.add_state();
        dfa.add_transition(tail, ((seed + depth) % 11) as i32, next);
        tail = next;
    }
    dfa.add_transition(tail, encode_negative_label(19), accepted);
    dfa
}

fn split(raw: &DFA) -> Arc<CommitTemplateDfas> {
    let specialized = specialize_template_dfa_defaults_for_commit_split_input(raw);
    Arc::new(try_split_commit_template_dfas(&specialized).unwrap())
}

fn pop_word(labels: &[i32], accepting: bool) -> CommitTemplateDfas {
    let mut pop = DFA::new();
    let mut cursor = pop.start_state;
    for &label in labels {
        let target = pop.add_state();
        pop.add_transition(cursor, label, target);
        cursor = target;
    }
    pop.set_accepting(cursor, accepting);
    CommitTemplateDfas {
        pop,
        read: DFA::new(),
        push: DFA::new(),
        pop_to_read: Vec::new(),
        pop_to_push: Vec::new(),
        read_to_push: Vec::new(),
    }
}

#[test]
fn concurrent_inventory_preparation_preserves_domains_and_every_runtime_view() {
    for count in [0u32, 1, 7, 32, 65] {
        let programs = (0..count)
            .map(|terminal| Some(split(&phase_fixture(terminal))))
            .collect::<Vec<_>>();
        for runtime in [false, true] {
            let mut reference_domains = Vec::new();
            let mut reference_views = Vec::new();
            for source in &programs {
                let source = source.as_deref().unwrap();
                let validated =
                    crate::runtime::commit::template_prepare::TemplatePreparation::new(source)
                        .unwrap();
                validated.validate_alphabet(32).unwrap();
                reference_domains.push(TemplateDomain::from_validated(&validated).unwrap());
                if runtime {
                    reference_views.push(Some(Arc::new(
                        FastCommitTemplateDfas::from_prepared(&validated)
                    )));
                }
            }
            for threads in [1, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads).build().unwrap();
                for _ in 0..3 {
                    let (domains, views) = pool.install(|| {
                        prepare_terminal_inventory(&programs, 32, runtime, true)
                    }).unwrap();
                    assert_eq!(format!("{views:?}"), format!("{reference_views:?}"));
                    for (expected, actual) in reference_domains.iter().zip(&domains) {
                        assert_eq!(expected.to_bytes().unwrap(), actual.to_bytes().unwrap());
                        for top in 0..32 {
                            assert_eq!(expected.classify_top(top), actual.classify_top(top));
                            for suffix in [vec![top], vec![top, 3], vec![top, 7, 2]] {
                                assert_eq!(
                                    expected.matches_top_first(suffix.iter().copied()),
                                    actual.matches_top_first(suffix.iter().copied()),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn alias_preparation_shares_domains_and_all_fast_views() {
    let left = split(&phase_fixture(0));
    let right = split(&phase_fixture(1));
    let programs = (0..65).map(|terminal| {
        Some(Arc::clone(if terminal % 2 == 0 { &left } else { &right }))
    }).collect::<Vec<_>>();
    let (domains, views) =
        prepare_terminal_inventory(&programs, 32, true, true).unwrap();
    for terminal in 2..65 {
        assert!(Arc::ptr_eq(&domains[terminal], &domains[terminal % 2]));
        assert!(Arc::ptr_eq(
            views[terminal].as_ref().unwrap(),
            views[terminal % 2].as_ref().unwrap(),
        ));
    }
    assert!(!Arc::ptr_eq(&domains[0], &domains[1]));
}

#[test]
fn concurrent_inventory_preparation_rejects_invalid_programs_in_terminal_order() {
    let mut programs = (0..8u32)
        .map(|terminal| Some(split(&phase_fixture(terminal))))
        .collect::<Vec<_>>();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    programs[2] = None;
    programs[6] = None;
    for _ in 0..4 {
        let serial = prepare_terminal_inventory(&programs, 32, true, false)
            .unwrap_err().to_string();
        let parallel = pool.install(|| {
            prepare_terminal_inventory(&programs, 32, true, true)
        }).unwrap_err().to_string();
        assert_eq!(serial, parallel);
        assert!(serial.contains("terminal 2"));
    }
    programs[2] = Some(split(&phase_fixture(2)));
    programs[6] = Some(split(&phase_fixture(6)));
    let invalid = Arc::make_mut(programs[1].as_mut().unwrap());
    let unreachable = invalid.pop.add_state();
    invalid.pop.add_transition(unreachable, 0, unreachable);
    let serial = prepare_terminal_inventory(&programs, 32, true, false)
        .unwrap_err().to_string();
    let parallel = pool.install(|| {
        prepare_terminal_inventory(&programs, 32, true, true)
    }).unwrap_err().to_string();
    assert_eq!(serial, parallel);
    assert!(serial.contains("cyclic"));
}

#[test]
fn staged_parser_preparation_preserves_runtime_finalization_order_and_view_identity() {
    let vocab = crate::Vocab::new(
        ["a", "b", "(", ")", "ab", "aa", " "].into_iter().enumerate()
            .map(|(id, value)| (id as u32, value.as_bytes().to_vec())).collect()
    );
    let grammar = crate::Grammar::glrm(
        r#"start root; nt root ::= "a" | "(" root ")";"#
    );
    let dynamic =
        crate::DynamicConstraint::compile_with_vocab_partition(grammar, &vocab).unwrap();
    let mut candidate = dynamic.into_constraints().pop().unwrap();
    assert!(candidate.has_template_parser() && !candidate.table.is_present());
    assert!(candidate.prepare_template_parser().unwrap().is_none());
    let programs = candidate.template_dfas_by_terminal.clone();
    let views = candidate.fast_template_dfas_by_terminal.clone();

    candidate.rebuild_dynamic_runtime_caches();
    assert!(!candidate.table.is_present());
    for ((source, after), (view, after_view)) in programs.iter()
        .zip(&candidate.template_dfas_by_terminal)
        .zip(views.iter().zip(&candidate.fast_template_dfas_by_terminal))
    {
        assert!(Arc::ptr_eq(source.as_ref().unwrap(), after.as_ref().unwrap()));
        assert!(Arc::ptr_eq(view.as_ref().unwrap(), after_view.as_ref().unwrap()),
            "runtime finalization must reuse the already-prepared immutable view");
    }

    let restored = Constraint::load(candidate.save()).unwrap();
    let external = Constraint::load_with_vocab(
        candidate.save_with_external_vocab().unwrap(), &vocab,
    ).unwrap();
    for constraint in [&restored, &external] {
        assert!(constraint.has_template_parser() && !constraint.table.is_present());
        for bytes in [b"".as_slice(), b"a", b"(a)", b"((a))", b"b", b" "] {
            let mut a = candidate.start();
            let mut b = constraint.start();
            assert_eq!(a.commit_bytes(bytes).is_ok(), b.commit_bytes(bytes).is_ok());
            assert_eq!(a.is_accepting(), b.is_accepting());
            let mut left = vec![0; candidate.mask_len()];
            let mut right = vec![0; constraint.mask_len()];
            a.fill_mask(&mut left);
            b.fill_mask(&mut right);
            assert_eq!(left, right, "prefix {bytes:?}");
        }
    }
}

#[test]
fn immutable_source_witness_is_invalidated_by_arc_make_mut() {
    let source = split(&phase_fixture(0));
    let validated =
        crate::runtime::commit::template_prepare::TemplatePreparation::new(&source).unwrap();
    let view = FastCommitTemplateDfas::from_shared_preparation(
        Arc::clone(&source), &validated,
    );
    assert!(view.is_for_source(&source));
    let mut changed = Arc::clone(&source);
    Arc::make_mut(&mut changed).pop.states[0].is_accepting ^= true;
    assert!(!view.is_for_source(&changed));
    assert!(view.is_for_source(&source));
}

#[test]
fn constructor_installs_prepared_views_without_rebuilding() {
    let vocab = crate::Vocab::new(vec![(0, b"a".to_vec())]);
    let expression = crate::automata::regex::Expr::U8Seq(b"a".to_vec());
    let names = vec!["a".to_owned()];
    let tokenizer = crate::compiler::pipeline::build_tokenizer_from_exprs(
        &[expression], Some(&names),
    );
    let templates = vec![Some(Arc::new(pop_word(&[0], true)))];
    let (parser, runtime) = TemplateParser::compile_with_runtime(
        1, 1, BTreeSet::new(), &templates, pop_word(&[0], true),
    ).unwrap();
    let expected = Arc::clone(runtime[0].as_ref().unwrap());
    let dynamic_vocab =
        crate::compiler::constraint_possible_matches::runtime_dynamic_vocab_for_vocab(&vocab);
    let constraint =
        crate::dynamic_constraint::DynamicConstraint::from_template_runtime_parts_unfinalized_with_views(
            tokenizer, names, None, templates, Arc::new(parser), &vocab,
            dynamic_vocab, Some(runtime),
        );
    assert!(Arc::ptr_eq(
        &expected, constraint.fast_template_dfas_by_terminal[0].as_ref().unwrap(),
    ));
    assert!(!constraint.table.is_present());
}

#[test]
fn native_compiler_effect_metadata_survives_reload_and_nested_links_without_tables() {
    use crate::compiler::boundary_stack_support::CompilerEffects;
    let vocab = crate::Vocab::new(
        ["a", "b", "(", ")", "ab"].into_iter().enumerate()
            .map(|(id, word)| (id as u32, word.as_bytes().to_vec())).collect()
    );
    for source in [
        r#"start root; nt root ::= "a" | "ab";"#,
        r#"start root; nt root ::= "a" | "(" root ")";"#,
    ] {
        let component = Constraint::compile(crate::Grammar::glrm(source), &vocab).unwrap();
        let parser = component.template_parser.as_ref().unwrap();
        let summary = parser.link_grammar.as_ref().unwrap().stack_effects().unwrap();
        assert_eq!(summary.states, parser.state_count);
        assert!(!summary.effects.is_empty());
        let loaded = Constraint::load(&component.save()).unwrap();
        assert!(!loaded.table.is_present());
        assert_eq!(
            loaded.template_parser.as_ref().unwrap().link_grammar.as_ref().unwrap()
                .stack_effects().unwrap(),
            summary,
        );
        let slot = *summary.entries.iter().find(|(_, row)| !row.is_empty()).unwrap().0;
        let end = summary.states.checked_mul(2).unwrap();
        let terminals = summary.terminals.checked_mul(2).unwrap();
        let linked = CompilerEffects::compose(
            &[summary, summary], &[0, summary.states],
            &[0, summary.terminals], &[(0, slot, 1, 0, 1, false)],
            end, terminals,
        ).unwrap();
        let nested = CompilerEffects::compose(
            &[&linked, summary], &[0, end], &[0, terminals], &[],
            end + summary.states, terminals + summary.terminals,
        ).unwrap();
        assert!(nested.effects.iter().any(|effect| effect.source >= end));
        assert_eq!(nested.accepting, summary.accepting);
    }
}

#[test]
fn sparse_preparation_matches_every_dense_top_certificate() {
    for terminal_count in [0, 1, 5, 63, 64, 65, 129] {
        let mut domains = Vec::new();
        for terminal in 0..terminal_count {
            let label = (terminal % 70) as i32;
            let program = match terminal % 6 {
                0 => pop_word(&[], true),
                1 => pop_word(&[], false),
                2 => pop_word(&[label], true),
                3 => pop_word(&[label, DEFAULT_LABEL], true),
                4 => {
                    let mut program = pop_word(&[DEFAULT_LABEL], true);
                    let dead = program.pop.add_state();
                    program.pop.add_transition(0, label, dead);
                    program
                }
                _ => {
                    let mut program = pop_word(&[DEFAULT_LABEL, 2], true);
                    let yes = program.pop.add_state();
                    program.pop.set_accepting(yes, true);
                    program.pop.add_transition(0, label, yes);
                    program
                }
            };
            domains.push(Arc::new(compile_domain(&program).unwrap()));
        }
        for completion in [
            pop_word(&[0], true), pop_word(&[], true), pop_word(&[], false),
        ] {
            let completion = Arc::new(compile_domain(&completion).unwrap());
            for state_count in [0, 1, 4, 64, 71, 129] {
                let (possible, unconditional) =
                    top_certificate_rows(state_count, &domains, &completion);
                for top in 0..state_count {
                    let mut expected_possible = BitSet::new(terminal_count + 1);
                    let mut expected_unconditional = BitSet::new(terminal_count + 1);
                    for (terminal, domain) in domains.iter()
                        .chain(std::iter::once(&completion)).enumerate()
                    {
                        match domain.classify_top(top) {
                            TopAdmission::Never => {}
                            TopAdmission::Always => {
                                expected_possible.set(terminal);
                                expected_unconditional.set(terminal);
                            }
                            TopAdmission::DependsOnSuffix => expected_possible.set(terminal),
                        }
                    }
                    assert_eq!(possible[top as usize], expected_possible);
                    assert_eq!(unconditional[top as usize], expected_unconditional);
                }
            }
        }
    }
}

#[test]
fn grouped_certificate_masks_cover_aliases_across_word_boundaries() {
    let yes = Arc::new(compile_domain(&pop_word(&[1], true)).unwrap());
    let no = Arc::new(compile_domain(&pop_word(&[], false)).unwrap());
    let domains = (0..257).map(|terminal| {
        Arc::clone(if terminal % 3 == 0 { &yes } else { &no })
    }).collect::<Vec<_>>();
    let completion = Arc::new(compile_domain(&pop_word(&[0], true)).unwrap());
    let (possible, unconditional) = top_certificate_rows(3, &domains, &completion);
    for terminal in 0..257 {
        assert_eq!(possible[1].contains(terminal), terminal % 3 == 0);
        assert_eq!(unconditional[1].contains(terminal), terminal % 3 == 0);
        assert!(!possible[0].contains(terminal));
    }
    assert!(possible[0].contains(257));
    assert!(unconditional[0].contains(257));
}

#[test]
fn bulk_top_certificates_match_literal_domains_on_shared_and_empty_languages() {
    let mut fallback = pop_word(&[DEFAULT_LABEL, 2], true);
    let dead = fallback.pop.add_state();
    fallback.pop.add_transition(fallback.pop.start_state, 1, dead);
    let raw = vec![
        pop_word(&[1], true), fallback, pop_word(&[], true),
        pop_word(&[], false), pop_word(&[2, 3], true),
    ];
    let raw = raw.into_iter().map(|program| Some(Arc::new(program))).collect::<Vec<_>>();
    let parser = TemplateParser::compile(
        4, 5, BTreeSet::new(), &raw, pop_word(&[0], true),
    ).unwrap();

    let mut concrete = vec![Vec::<u32>::new()];
    for length in 1..=3u32 {
        for encoded in 0..6u32.pow(length) {
            let mut encoded = encoded;
            let mut stack = Vec::new();
            for _ in 0..length {
                stack.push(encoded % 6);
                encoded /= 6;
            }
            concrete.push(stack);
        }
    }
    let clean = TerminalsDisallowed::new();
    let guarded = clean.with_insert(0, 3);
    let mut inputs = vec![ParserGSS::empty()];
    for stack in &concrete {
        inputs.push(ParserGSS::from_single_stack(stack.clone(), clean.clone()));
    }
    for group in concrete.chunks(7) {
        let paths = group.iter().enumerate().map(|(index, stack)| {
            (stack.clone(), if index % 2 == 0 { clean.clone() } else { guarded.clone() })
        }).collect::<Vec<_>>();
        inputs.push(ParserGSS::from_stacks(&paths));
    }
    for input in inputs {
        let literal = input.to_stacks(4096).unwrap();
        for requested in 0..64u64 {
            let mut candidates = BitSet::new(73);
            candidates.set(72);
            for bit in 0..6 {
                if requested & (1 << bit) != 0 { candidates.set(bit); }
            }
            let expected = candidates.iter().filter(|&bit| {
                let domain = if bit == 5 { Some(&parser.completion) }
                    else { parser.domains.get(bit) };
                domain.is_some_and(|domain| {
                    literal.iter().any(|(stack, _)| {
                        domain.matches_top_first(stack.iter().rev().copied())
                    })
                })
            }).collect::<Vec<_>>();
            let actual = parser.admitted(&input, &candidates);
            assert_eq!(actual.iter().collect::<Vec<_>>(), expected);
            assert_eq!(parser.admits_any(&input, &candidates), !expected.is_empty());
            for (stack, _) in &literal {
                let expected = candidates.iter().any(|bit| {
                    let domain = if bit == 5 { Some(&parser.completion) }
                        else { parser.domains.get(bit) };
                    domain.is_some_and(|domain| {
                        domain.matches_top_first(stack.iter().rev().copied())
                    })
                });
                assert_eq!(parser.admits_flat_any(stack, &candidates), expected);
            }
        }
    }
}

#[test]
fn native_rewrite_preserves_nullable_recursive_masks_commits_and_persistence() {
    let pieces = [
        (0, b"a".to_vec()), (1, b"b".to_vec()),
        (2, b"aa".to_vec()), (3, b"ab".to_vec()),
        (7, b"aabb".to_vec()), (11, b"ab".to_vec()),
        (31, b" ".to_vec()), (64, Vec::new()),
    ];
    let vocab = crate::Vocab::new(pieces.to_vec());
    let source = crate::Grammar::glrm(
        r#"start root; ignore WS; t WS ::= " "+; nt root ::= "a" root "b" | "";"#
    );
    for optimization in [crate::Optimization::FastBuild, crate::Optimization::FastRuntime] {
        let fresh = source.compile_with(
            &vocab,
            crate::BuildOptions::default().optimization(optimization),
        ).unwrap().with_end_tokens(&[64]).unwrap();
        let self_bytes = fresh.save();
        let loaded = Constraint::load(&self_bytes).unwrap();
        assert_eq!(loaded.save(), self_bytes);
        let external_bytes = fresh.save_with_external_vocab().unwrap();
        let external = Constraint::load_with_vocab(&external_bytes, &vocab).unwrap();
        for constraint in [&fresh, &loaded, &external] {
            assert!(constraint.has_template_parser());
            assert!(!constraint.table.is_present());
            for depth in [0usize, 1, 2, 8, 64] {
                let prefix = vec![b'a'; depth];
                let mut state = constraint.start();
                state.commit_bytes(&prefix).unwrap();
                assert_eq!(state.is_accepting(), depth == 0);
                let mut mask = vec![0; constraint.mask_len()];
                state.fill_mask(&mut mask);
                assert_eq!(mask[0] & 1 != 0, true);
                assert_eq!(mask[0] & 2 != 0, depth != 0);
                assert_eq!(mask[2] & 1 != 0, depth == 0);
                state.commit_bytes(&vec![b'b'; depth]).unwrap();
                assert!(state.is_accepting());
                state.commit_token(64).unwrap();
                state.fill_mask(&mut mask);
                assert!(mask.iter().all(|word| *word == 0));
            }
            let mut rejected = constraint.start();
            assert!(rejected.commit_bytes(b"b").is_err());
            assert!(rejected.is_rejected());
        }
    }
}

#[test]
fn dynamic_envelope_contains_exactly_the_native_constraint_bodies() {
    let vocab = crate::Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
    let a = Constraint::compile(
        crate::Grammar::glrm(r#"start root; nt root ::= "a";"#), &vocab,
    ).unwrap();
    let b = Constraint::compile(
        crate::Grammar::glrm(r#"start root; nt root ::= "b";"#), &vocab,
    ).unwrap();
    let dynamic = crate::dynamic_constraint::DynamicConstraint::from_constraints(
        vec![a.clone(), b.clone()]
    );
    for external in [false, true] {
        let actual = if external {
            dynamic.save_with_external_vocab()
        } else {
            dynamic.save()
        };
        let mut payload = 2u32.to_le_bytes().to_vec();
        for constraint in [&a, &b] {
            let body = if external {
                constraint.save_with_external_vocab().unwrap()
            } else {
                constraint.save()
            };
            payload.extend_from_slice(&(body.len() as u64).to_le_bytes());
            payload.extend_from_slice(&body);
        }
        assert_eq!(
            u64::from_le_bytes(actual[10..18].try_into().unwrap()) as usize,
            payload.len(),
        );
        assert_eq!(&actual[18..], payload.as_slice());
    }
}
