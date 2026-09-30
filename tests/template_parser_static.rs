//! Independent language oracles for data-only static compilation. No grammar,
//! LR table or parser callback constructs any candidate in this file.
use glrmask::template_parser::{
    LexerDefinition, ParserDefinition, ParserProgram, StackDfa, StackLabel, StackState,
    StackTemplate, StackTransition, TemplateBuildOptions, TerminalPattern,
};
use glrmask::{Constraint, Optimization, ParserBackend, Vocab};

fn compile_static(p: &ParserProgram, l: &LexerDefinition, v: &Vocab) -> Constraint {
    let c = p
        .compile_with(
            l,
            v,
            TemplateBuildOptions::default().optimization(Optimization::FastRuntime),
        )
        .unwrap();
    assert_eq!(c.parser_backend(), ParserBackend::TemplateDfa);
    c
}

fn representations(c: &Constraint, v: &Vocab) -> Vec<Constraint> {
    let bytes = c.save();
    let loaded = Constraint::load(&bytes).unwrap();
    assert_eq!(loaded.save(), bytes);
    vec![
        c.clone(),
        loaded,
        Constraint::load_with_vocab(c.save_with_external_vocab().unwrap(), v).unwrap(),
    ]
}

fn vocabulary(words: &[&[u8]]) -> Vocab {
    Vocab::new(
        words
            .iter()
            .enumerate()
            .map(|(id, bytes)| (id as u32, bytes.to_vec()))
            .collect(),
    )
}

fn paren_depth(bytes: &[u8], mut depth: usize) -> Option<usize> {
    for b in bytes {
        match b {
            b'(' => depth += 1,
            b')' => depth = depth.checked_sub(1)?,
            b' ' | b'\t' | b'\n' => (),
            _ => return None,
        }
    }
    Some(depth)
}

#[test]
fn recursive_static_parser_matches_literal_prefix_oracle_in_all_artifact_forms() {
    let parser = ParserProgram::new(ParserDefinition {
        stack_symbol_count: 2,
        terminals: vec![
            StackTemplate::read_top_and_push([0, 1], [1]),
            StackTemplate::rewrite([StackLabel::Symbol(1)], []),
            StackTemplate::identity(),
        ],
        completion: StackTemplate::read_top_and_push([0], []),
    })
    .unwrap();
    let lexer = LexerDefinition::new(vec![
        TerminalPattern::literal(b"(".to_vec()),
        TerminalPattern::literal(b")".to_vec()),
        TerminalPattern::regex(r"[ \t\n]+"),
    ])
    .ignoring(2);
    let words: &[&[u8]] = &[
        b"(", b")", b"()", b"((", b"))", b"(()", b"())", b"()()", b" ", b" \t\n", b"a", b"(a)",
        b"()", b"( )", b")(",
    ];
    let mut entries: Vec<_> = words
        .iter()
        .enumerate()
        .map(|(i, w)| (i as u32, w.to_vec()))
        .collect();
    entries.push((64, vec![]));
    let v = Vocab::new(entries);
    let c = parser
        .compile_with(
            &lexer,
            &v,
            TemplateBuildOptions::default()
                .optimization(Optimization::FastRuntime)
                .end_tokens([64]),
        )
        .unwrap();
    for c in representations(&c, &v) {
        assert_eq!(c.parser_backend(), ParserBackend::TemplateDfa);
        for length in 0..=8 {
            for bits in 0..1usize << length {
                let prefix: Vec<_> = (0..length)
                    .map(|i| if bits & (1 << i) == 0 { b'(' } else { b')' })
                    .collect();
                let Some(depth) = paren_depth(&prefix, 0) else {
                    continue;
                };
                let mut state = c.start();
                state.commit_bytes(&prefix).unwrap();
                assert_eq!(state.is_accepting(), depth == 0);
                let mask = state.mask();
                for (id, word) in words.iter().enumerate() {
                    assert_eq!(
                        mask[id / 32] & (1 << (id % 32)) != 0,
                        paren_depth(word, depth).is_some(),
                        "prefix={prefix:?} word={word:?}"
                    );
                }
                assert_eq!(mask[2] & 1 != 0, depth == 0);
            }
        }
        let mut rejected = c.start();
        assert!(rejected.commit_bytes(b")").is_err());
        assert!(rejected.is_rejected());
    }
}

#[test]
fn static_default_keeps_explicit_dead_shadowing_across_substitution() {
    let mut a = StackTemplate::rewrite([StackLabel::Default], [1]);
    let dead = a.pop.states.len() as u32;
    a.pop.states.push(StackState::default());
    a.pop.states[0].transitions.push(StackTransition {
        label: StackLabel::Symbol(0),
        target: dead,
    });
    let p = ParserProgram::new(ParserDefinition {
        stack_symbol_count: 2,
        terminals: vec![a, StackTemplate::rewrite([StackLabel::Symbol(0)], [1])],
        completion: StackTemplate::identity(),
    })
    .unwrap();
    let l = LexerDefinition::new(vec![
        TerminalPattern::literal(b"a".to_vec()),
        TerminalPattern::literal(b"b".to_vec()),
    ]);
    let v = vocabulary(&[b"a", b"b", b"ba", b"ab"]);
    for c in representations(&compile_static(&p, &l, &v), &v) {
        let mut s = c.start();
        assert_eq!(s.mask()[0] & 15, 6);
        s.commit_token(1).unwrap();
        assert_eq!(s.mask()[0] & 15, 1);
    }
}

#[test]
fn static_read_epsilon_links_work_on_empty_stack() {
    let mut a = StackTemplate::rewrite([StackLabel::Symbol(0)], []);
    a.pop_to_push.clear();
    a.pop_to_read = vec![None, Some(0)];
    a.read = StackDfa::accept();
    let mut b = StackTemplate::rewrite([], [0]);
    b.pop_to_push.clear();
    b.pop_to_read = vec![Some(0)];
    b.read_to_push = vec![Some(0)];
    let p = ParserProgram::new(ParserDefinition {
        stack_symbol_count: 1,
        terminals: vec![a, b],
        completion: StackTemplate::identity(),
    })
    .unwrap();
    let l = LexerDefinition::new(vec![
        TerminalPattern::literal(b"a".to_vec()),
        TerminalPattern::literal(b"b".to_vec()),
    ]);
    let words: &[&[u8]] = &[b"a", b"b", b"ab", b"ba", b"aa", b"bb"];
    let v = vocabulary(words);
    fn advance(bytes: &[u8], mut count: usize) -> Option<usize> {
        for b in bytes {
            count = if *b == b'a' {
                count.checked_sub(1)?
            } else {
                count + 1
            };
        }
        Some(count)
    }
    for c in representations(&compile_static(&p, &l, &v), &v) {
        for length in 0..=7 {
            for bits in 0..1usize << length {
                let prefix: Vec<_> = (0..length)
                    .map(|i| if bits & (1 << i) == 0 { b'a' } else { b'b' })
                    .collect();
                let Some(count) = advance(&prefix, 1) else {
                    continue;
                };
                let mut s = c.start();
                s.commit_bytes(&prefix).unwrap();
                assert!(s.is_accepting());
                let mask = s.mask();
                for (id, word) in words.iter().enumerate() {
                    assert_eq!(
                        mask[id / 32] & (1 << (id % 32)) != 0,
                        advance(word, count).is_some(),
                        "prefix={prefix:?} word={word:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn repeated_read_labels_preserve_the_same_concrete_top() {
    let read = StackDfa {
        start: 0,
        states: vec![
            StackState {
                accepting: false,
                transitions: vec![StackTransition {
                    label: StackLabel::Symbol(0),
                    target: 1,
                }],
            },
            StackState {
                accepting: false,
                transitions: vec![StackTransition {
                    label: StackLabel::Symbol(0),
                    target: 2,
                }],
            },
            StackState {
                accepting: true,
                transitions: vec![],
            },
        ],
    };
    let t = StackTemplate {
        read,
        pop_to_read: vec![Some(0)],
        ..StackTemplate::reject()
    };
    let p = ParserProgram::new(ParserDefinition {
        stack_symbol_count: 1,
        terminals: vec![t],
        completion: StackTemplate::identity(),
    })
    .unwrap();
    let l = LexerDefinition::new(vec![TerminalPattern::literal(b"a".to_vec())]);
    let v = vocabulary(&[b"a", b"aa", b"b", b"aaa"]);
    for c in representations(&compile_static(&p, &l, &v), &v) {
        let mut s = c.start();
        for _ in 0..10 {
            assert_eq!(s.mask()[0] & 15, 11);
            s.commit_token(1).unwrap();
            assert!(s.is_accepting());
        }
    }
}

#[test]
fn static_compilation_keeps_exponential_push_languages_as_graphs() {
    let depth = 24;
    let mut push = StackDfa {
        start: 0,
        states: vec![StackState::default(); depth + 1],
    };
    for i in 0..depth {
        for symbol in [1, 2] {
            push.states[i].transitions.push(StackTransition {
                label: StackLabel::Symbol(symbol),
                target: i as u32 + 1,
            });
        }
    }
    push.states[depth].accepting = true;
    let p = ParserProgram::new(ParserDefinition {
        stack_symbol_count: 3,
        terminals: vec![StackTemplate {
            push,
            pop_to_push: vec![Some(0)],
            ..StackTemplate::reject()
        }],
        completion: StackTemplate::identity(),
    })
    .unwrap();
    let l = LexerDefinition::new(vec![TerminalPattern::literal(b"a".to_vec())]);
    let v = vocabulary(&[b"a", b"aa", b"b"]);
    let c = compile_static(&p, &l, &v);
    assert!(
        c.save().len() < 100_000,
        "a linear relation must not serialize millions of paths"
    );
    for c in representations(&c, &v) {
        let mut s = c.start();
        for _ in 0..3 {
            assert_eq!(s.mask()[0] & 7, 3);
            s.commit_token(0).unwrap();
            assert!(s.is_accepting());
        }
    }
}

#[test]
fn identity_after_final_pop_does_not_recreate_bottom_symbol_in_static_masks() {
    let p = ParserProgram::new(ParserDefinition {
        stack_symbol_count: 3,
        terminals: vec![
            StackTemplate::identity(),
            StackTemplate::rewrite([StackLabel::Default], []),
        ],
        completion: StackTemplate::identity(),
    })
    .unwrap();
    let l = LexerDefinition::new(vec![
        TerminalPattern::literal(b"a".to_vec()),
        TerminalPattern::literal(b"b".to_vec()),
    ]);
    let v = vocabulary(&[b"a", b"b", b"ba", b"bb", b"ab", b"aaa"]);
    for c in representations(&compile_static(&p, &l, &v), &v) {
        for prefix in [vec![1], vec![2], vec![1, 0], vec![5, 2]] {
            let mut s = c.start();
            for token in &prefix {
                s.commit_token(*token).unwrap();
            }
            assert_eq!(s.mask()[0] & 63, 33, "prefix={prefix:?}");
        }
    }
}

#[test]
fn partial_utf8_and_duplicate_model_tokens_use_the_same_static_lexer() {
    let p = ParserProgram::new(ParserDefinition {
        stack_symbol_count: 2,
        terminals: vec![
            StackTemplate::read_top_and_push([0], [0]),
            StackTemplate::read_top_and_push([0], [1]),
            StackTemplate::identity(),
        ],
        completion: StackTemplate::read_top_and_push([1], []),
    }).unwrap();
    let l = LexerDefinition::new(vec![
        TerminalPattern::literal(b"a".to_vec()),
        TerminalPattern::literal("é".as_bytes().to_vec()),
        TerminalPattern::regex("[ ]+"),
    ]).ignoring(2);
    let pieces: &[&[u8]] = &[b"a", "é".as_bytes(), &[0xc3], &[0xa9], b" a", b" ",
        &[0xc3, b'a'], &[0xc3, 0xa9, b'a'], &[0xc3, 0xa9, b' '], b"aa",
        &[b'a', 0xc3, 0xa9], &[0], "é".as_bytes()];
    let v = vocabulary(pieces);
    fn language(bytes: &[u8]) -> Option<bool> {
        let mut phase = 0;
        for &byte in bytes {
            phase = match (phase, byte) {
                (0, b'a' | b' ') => 0,
                (0, 0xc3) => 1,
                (1, 0xa9) => 2,
                (2, b' ') => 2,
                _ => return None,
            };
        }
        Some(phase == 2)
    }
    for c in representations(&compile_static(&p, &l, &v), &v) {
        let mut todo = vec![(Vec::<u32>::new(), Vec::<u8>::new())];
        while let Some((prefix, bytes)) = todo.pop() {
            let mut s = c.start();
            for token in &prefix { s.commit_token(*token).unwrap(); }
            assert_eq!(s.is_accepting(), language(&bytes).unwrap());
            let mask = s.mask();
            for (id, piece) in pieces.iter().enumerate() {
                let mut extended = bytes.clone(); extended.extend_from_slice(piece);
                let viable = language(&extended).is_some();
                assert_eq!(mask[id / 32] & (1 << (id % 32)) != 0, viable,
                    "prefix={bytes:?} piece={piece:?}");
                if viable && prefix.len() < 2 {
                    let mut next = prefix.clone(); next.push(id as u32); todo.push((next, extended));
                }
            }
        }
    }
}

#[test]
fn static_expansion_budget_is_a_build_error_not_a_table_or_dynamic_fallback() {
    // A tiny symbolic template denotes over a million concrete default edges.
    // The exact static compiler must reject excessive expansion without
    // silently truncating it or changing its selected runtime backend.
    let p = ParserProgram::new(ParserDefinition {
        stack_symbol_count: 60_000,
        terminals: vec![StackTemplate::rewrite((0..18).map(|_| StackLabel::Default), [])],
        completion: StackTemplate::identity(),
    }).unwrap();
    let l = LexerDefinition::new(vec![TerminalPattern::literal(b"a".to_vec())]);
    let v = vocabulary(&[b"a", b"aa"]);
    let error = p.compile_with(&l, &v, TemplateBuildOptions::default()
        .optimization(Optimization::FastRuntime)).unwrap_err();
    assert!(error.to_string().contains("budget"), "{error}");
    // The immutable program remains usable with the explicitly requested
    // dynamic compiler; an 18-symbol pop is not valid on the initial [0].
    let dynamic = p.compile(&l, &v).unwrap();
    assert_eq!(dynamic.parser_backend(), ParserBackend::TemplateDfa);
    assert_eq!(dynamic.start().mask()[0] & 3, 0);
}

#[test]
fn an_empty_model_vocabulary_has_an_empty_mask_without_an_lr_table() {
    let p = ParserProgram::new(ParserDefinition {
        stack_symbol_count: 1,
        terminals: vec![StackTemplate::identity()],
        completion: StackTemplate::identity(),
    }).unwrap();
    let l = LexerDefinition::new(vec![TerminalPattern::literal(b"a".to_vec())]);
    let v = Vocab::new(vec![]);
    let c = compile_static(&p, &l, &v);
    for c in representations(&c, &v) {
        assert!(c.start().is_accepting());
        assert!(c.start().mask().iter().all(|word| *word == 0));
    }
}
