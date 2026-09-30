//! Static-only variant of the independent literal phase interpreter corpus.
//! The oracle follows public stack semantics and does not share compiler,
//! domain, weighted-automata, or GSS code with the implementation.
use glrmask::template_parser::{
    LexerDefinition, ParserDefinition, ParserProgram, StackDfa, StackLabel, StackState,
    StackTemplate, StackTransition, TerminalPattern,
};
use glrmask::{Constraint, ParserBackend, Vocab};
use std::collections::BTreeSet;

fn lexer(n: usize) -> LexerDefinition {
    LexerDefinition::new(
        (0..n)
            .map(|i| TerminalPattern::literal(vec![b'a' + i as u8]))
            .collect(),
    )
}
fn vocab() -> Vocab {
    Vocab::new(
        ["a", "b", "ab", "ba", "aa", "bb", "aaa", "bbb"]
            .into_iter()
            .enumerate()
            .map(|(i, s)| (i as u32, s.as_bytes().to_vec()))
            .collect(),
    )
}
fn program(a: StackTemplate, b: StackTemplate, completion: StackTemplate) -> ParserProgram {
    ParserProgram::new(ParserDefinition {
        stack_symbol_count: 3,
        terminals: vec![a, b],
        completion,
    })
    .unwrap()
}
fn mask(c: &Constraint, prefix: &[u32]) -> (Vec<u32>, bool) {
    let mut state = c.start();
    for &token in prefix {
        state.commit_token(token).unwrap();
    }
    let mut words = vec![0; c.mask_len()];
    state.fill_mask(&mut words);
    (words, state.is_accepting())
}
fn has(words: &[u32], token: u32) -> bool {
    words
        .get(token as usize / 32)
        .is_some_and(|w| w & (1 << (token % 32)) != 0)
}

#[test]
fn can_pop_the_bottom_marker_and_continue_from_an_empty_concrete_stack() {
    let p = program(
        StackTemplate::rewrite([StackLabel::Symbol(0)], []),
        StackTemplate::rewrite([], [0]),
        StackTemplate::identity(),
    );
    let c = p.compile_static(&lexer(2), &vocab()).unwrap();
    assert_eq!(c.parser_backend(), ParserBackend::TemplateDfa);
    let (initial, accepting) = mask(&c, &[]);
    assert!(accepting);
    assert!(has(&initial, 0));
    let (after, accepting) = mask(&c, &[0]);
    assert!(accepting);
    assert!(!has(&after, 0));
    assert!(has(&after, 1));
    let (restored, accepting) = mask(&c, &[0, 1]);
    assert!(accepting);
    assert!(has(&restored, 0));
    let d = Constraint::load(&c.save()).unwrap();
    assert_eq!(mask(&d, &[0]), mask(&c, &[0]));
}

#[test]
fn epsilon_read_link_is_available_without_a_top_symbol() {
    let mut b = StackTemplate::reject();
    b.pop_to_read = vec![Some(0)];
    b.read_to_push = vec![Some(0)];
    b.push = StackDfa::word([StackLabel::Symbol(0)]);
    // An unrelated READ transition must not suppress the phase epsilon.
    b.read.states.push(StackState::default());
    b.read.states[0].transitions.push(StackTransition {
        label: StackLabel::Symbol(2),
        target: 1,
    });
    let p = program(
        StackTemplate::rewrite([StackLabel::Symbol(0)], []),
        b,
        StackTemplate::identity(),
    );
    let c = p.compile_static(&lexer(2), &vocab()).unwrap();
    let (m, _) = mask(&c, &[0]);
    assert!(has(&m, 1));
    assert!(has(&m, 2) == false);
    assert!(has(&mask(&c, &[0, 1]).0, 0));
}

#[test]
fn unreachable_very_deep_cycle_is_rejected_without_recursive_validation() {
    let mut a = StackTemplate::identity();
    a.pop
        .states
        .extend((0..50_000).map(|_| StackState::default()));
    for i in 1..50_000 {
        a.pop.states[i].transitions.push(StackTransition {
            label: StackLabel::Symbol(0),
            target: i as u32 + 1,
        });
    }
    a.pop.states[50_000].transitions.push(StackTransition {
        label: StackLabel::Symbol(0),
        target: 25_000,
    });
    let e = ParserProgram::new(ParserDefinition {
        stack_symbol_count: 3,
        terminals: vec![a],
        completion: StackTemplate::identity(),
    })
    .unwrap_err();
    assert!(e.to_string().contains("acyclic"), "{e}");
}

#[test]
fn unreachable_deep_valid_chain_is_accepted_and_does_not_change_identity_language() {
    let mut a = StackTemplate::identity();
    a.pop
        .states
        .extend((0..50_000).map(|_| StackState::default()));
    for i in 1..50_000 {
        a.pop.states[i].transitions.push(StackTransition {
            label: StackLabel::Symbol(0),
            target: i as u32 + 1,
        });
    }
    let p = ParserProgram::new(ParserDefinition {
        stack_symbol_count: 3,
        terminals: vec![a, StackTemplate::identity()],
        completion: StackTemplate::identity(),
    })
    .unwrap();
    let c = p.compile_static(&lexer(2), &vocab()).unwrap();
    assert!((0..8).all(|id| has(&mask(&c, &[2, 3]).0, id)));
}

// Literal finite-stack interpreter. This shares no GLRMask DAG, GSS or domain
// code: it follows the public phase semantics and enumerates only tiny fixtures.
fn apply(t: &StackTemplate, stack: &[u32]) -> BTreeSet<Vec<u32>> {
    let mut todo = vec![(0u8, t.pop.start, stack.to_vec())];
    let mut result = BTreeSet::new();
    let mut budget = 100_000;
    while let Some((phase, id, word)) = todo.pop() {
        budget -= 1;
        assert!(
            budget > 0,
            "test generator created an oversized oracle fixture"
        );
        let graph = match phase {
            0 => &t.pop,
            1 => &t.read,
            _ => &t.push,
        };
        let state = &graph.states[id as usize];
        if state.accepting {
            result.insert(word.clone());
        }
        if phase == 0 {
            for (links, next_phase) in [(&t.pop_to_read, 1), (&t.pop_to_push, 2)] {
                if let Some(Some(target)) = links.get(id as usize) {
                    todo.push((next_phase, *target, word.clone()));
                }
            }
        } else if phase == 1 {
            if let Some(Some(target)) = t.read_to_push.get(id as usize) {
                todo.push((2, *target, word.clone()));
            }
        }
        if phase == 2 {
            for edge in &state.transitions {
                let StackLabel::Symbol(symbol) = edge.label else {
                    panic!("bad generator")
                };
                let mut after = word.clone();
                after.push(symbol);
                todo.push((2, edge.target, after));
            }
        } else if let Some(&top) = word.last() {
            let edge = state
                .transitions
                .iter()
                .find(|e| e.label == StackLabel::Symbol(top))
                .or_else(|| {
                    if phase == 0 {
                        state
                            .transitions
                            .iter()
                            .find(|e| e.label == StackLabel::Default)
                    } else {
                        None
                    }
                });
            if let Some(edge) = edge {
                let mut after = word.clone();
                if phase == 0 {
                    after.pop();
                }
                todo.push((phase, edge.target, after));
            }
        }
    }
    result
}
fn closure(t: &StackTemplate, words: &BTreeSet<Vec<u32>>) -> BTreeSet<Vec<u32>> {
    words.iter().flat_map(|word| apply(t, word)).collect()
}
fn next(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed
}
fn random_template(seed: &mut u64) -> StackTemplate {
    let mut t = StackTemplate::reject();
    for phase in 0..3 {
        let g = match phase {
            0 => &mut t.pop,
            1 => &mut t.read,
            _ => &mut t.push,
        };
        g.states = vec![StackState::default(); 4];
        for i in 0..4 {
            g.states[i].accepting = next(seed) % 5 == 0;
            if i == 3 {
                continue;
            }
            for symbol in 0..3u32 {
                if next(seed) % 3 == 0 {
                    let target = i + 1 + (next(seed) as usize % (3 - i));
                    g.states[i].transitions.push(StackTransition {
                        label: StackLabel::Symbol(symbol),
                        target: target as u32,
                    });
                }
            }
            if phase == 0 && next(seed) % 2 == 0 {
                let target = i + 1 + (next(seed) as usize % (3 - i));
                g.states[i].transitions.push(StackTransition {
                    label: StackLabel::Default,
                    target: target as u32,
                });
            }
        }
    }
    for links in [&mut t.pop_to_read, &mut t.pop_to_push, &mut t.read_to_push] {
        *links = (0..4)
            .map(|_| {
                if next(seed) % 3 == 0 {
                    Some((next(seed) % 4) as u32)
                } else {
                    None
                }
            })
            .collect();
    }
    t
}

#[test]
fn arbitrary_phase_programs_match_an_independent_full_token_mask_oracle() {
    let pieces = ["a", "b", "ab", "ba", "aa", "bb", "aaa", "bbb"];
    let mut seed = 608894u64;
    for case in 0..128 {
        let templates = [random_template(&mut seed), random_template(&mut seed)];

        let p = program(
            templates[0].clone(),
            templates[1].clone(),
            StackTemplate::identity(),
        );
        let c = p.compile_static(&lexer(2), &vocab()).unwrap();
        let loaded = Constraint::load(&c.save()).unwrap();
        let external = c.save_with_external_vocab().unwrap();
        let external_loaded = Constraint::load_with_vocab(&external, &vocab()).unwrap();
        let mut pending = vec![(Vec::<u32>::new(), BTreeSet::from([vec![0]]))];
        while let Some((prefix, words)) = pending.pop() {
            let (m, complete) = mask(&c, &prefix);
            assert_eq!(
                mask(&loaded, &prefix),
                (m.clone(), complete),
                "self reload case{case} {prefix:?}"
            );
            assert_eq!(
                mask(&external_loaded, &prefix),
                (m.clone(), complete),
                "external reload case{case} {prefix:?}"
            );
            for (id, piece) in pieces.iter().enumerate() {
                let mut after = words.clone();
                for byte in piece.bytes() {
                    after = closure(&templates[(byte - b'a') as usize], &after);
                }
                assert_eq!(
                    has(&m, id as u32),
                    !after.is_empty(),
                    "case={case} prefix={prefix:?} token={piece} words={words:?}"
                );
                if prefix.len() < 2 && !after.is_empty() && after.len() < 128 {
                    let mut new = prefix.clone();
                    new.push(id as u32);
                    pending.push((new, after));
                }
            }
        }
    }
}

// Independent byte recognizer for: spaces* ("ab" | "a" spaces* "c") spaces*.
// Returns whether the bytes are a viable prefix, and whether they are complete.
fn ambiguous_language(bytes: &[u8]) -> (bool, bool) {
    let bytes = bytes
        .iter()
        .copied()
        .skip_while(|b| *b == b' ')
        .collect::<Vec<_>>();
    if bytes.is_empty() {
        return (true, false);
    }
    if bytes[0] != b'a' {
        return (false, false);
    }
    let mut tail = &bytes[1..];
    if tail.is_empty() {
        return (true, false);
    }
    if tail[0] == b'b' {
        let ok = tail[1..].iter().all(|b| *b == b' ');
        return (ok, ok);
    }
    while tail.first() == Some(&b' ') {
        tail = &tail[1..];
    }
    if tail.is_empty() {
        return (true, false);
    }
    if tail[0] != b'c' {
        return (false, false);
    }
    let ok = tail[1..].iter().all(|b| *b == b' ');
    (ok, ok)
}

#[test]
fn overlapping_terminal_lexer_and_ignored_whitespace_match_byte_language() {
    let pieces = [
        "a", "b", "c", "ab", "ac", "a c", "b c", " a", " c", "  ", " ",
    ];
    let vocab = Vocab::new(
        pieces
            .iter()
            .enumerate()
            .map(|(i, p)| (i as u32, p.as_bytes().to_vec()))
            .collect(),
    );
    let definition = ParserDefinition {
        stack_symbol_count: 4,
        terminals: vec![
            StackTemplate::read_top_and_push([0], [1]),           // a
            StackTemplate::read_top_and_push([0], [2]),           // ab
            StackTemplate::rewrite([StackLabel::Symbol(1)], [3]), // c following a
            StackTemplate::identity(),                            // ignored spaces
        ],
        completion: StackTemplate::read_top_and_push([2, 3], []),
    };
    let lex = LexerDefinition::new(vec![
        TerminalPattern::literal(b"a".to_vec()),
        TerminalPattern::literal(b"ab".to_vec()),
        TerminalPattern::literal(b"c".to_vec()),
        TerminalPattern::regex("[ ]+"),
    ])
    .ignoring(3);
    let fresh = ParserProgram::new(definition)
        .unwrap()
        .compile_static(&lex, &vocab)
        .unwrap();
    let loaded = Constraint::load(&fresh.save()).unwrap();
    let external =
        Constraint::load_with_vocab(&fresh.save_with_external_vocab().unwrap(), &vocab).unwrap();
    let mut prefixes = vec![(Vec::<u32>::new(), Vec::<u8>::new())];
    while let Some((prefix, bytes)) = prefixes.pop() {
        for c in [&fresh, &loaded, &external] {
            let (mask, complete) = mask(c, &prefix);
            assert_eq!(
                complete,
                ambiguous_language(&bytes).1,
                "completion {bytes:?}"
            );
            for (id, piece) in pieces.iter().enumerate() {
                let mut candidate = bytes.clone();
                candidate.extend_from_slice(piece.as_bytes());
                assert_eq!(
                    has(&mask, id as u32),
                    ambiguous_language(&candidate).0,
                    "prefix={bytes:?} next={piece}"
                );
            }
        }
        if prefix.len() < 3 {
            for (id, piece) in pieces.iter().enumerate() {
                let mut candidate = bytes.clone();
                candidate.extend_from_slice(piece.as_bytes());
                if ambiguous_language(&candidate).0 {
                    let mut next = prefix.clone();
                    next.push(id as u32);
                    prefixes.push((next, candidate));
                }
            }
        }
    }
}

trait CompileStatic {
    fn compile_static(&self, lexer: &LexerDefinition, vocab: &Vocab)
    -> glrmask::Result<Constraint>;
}
impl CompileStatic for ParserProgram {
    fn compile_static(
        &self,
        lexer: &LexerDefinition,
        vocab: &Vocab,
    ) -> glrmask::Result<Constraint> {
        self.compile_with(
            lexer,
            vocab,
            glrmask::template_parser::TemplateBuildOptions::default()
                .optimization(glrmask::Optimization::FastRuntime),
        )
    }
}
