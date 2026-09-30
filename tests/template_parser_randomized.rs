//! Cross-check user-defined parser programs independently of the production
//! DFA, admission, GSS and weighted-mask implementations. These bounded fixtures
//! intentionally permute state numbers, vary completion relations, pop the
//! bottom marker, use sparse/duplicate vocabulary entries, and cross lexical
//! boundaries within a single model token.
//!
//! Bounds below restrict the generated test corpus, never the runtime language.
//! Larger seed batches can reuse the same literal oracles outside the CI gate.
use glrmask::template_parser::{
    LexerDefinition, ParserDefinition, ParserProgram, StackLabel, StackState, StackTemplate,
    StackTransition, TemplateBuildOptions, TerminalPattern,
};
use glrmask::{Constraint, Optimization, ParserBackend, Vocab};
use std::collections::{BTreeSet, VecDeque};

fn next(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
fn shuffle<T>(v: &mut [T], s: &mut u64) {
    for i in (1..v.len()).rev() {
        let j = next(s) as usize % (i + 1);
        v.swap(i, j);
    }
}
fn random_template(s: &mut u64) -> StackTemplate {
    let mut t = StackTemplate::reject();
    let n = 2 + (next(s) % 3) as usize;
    for phase in 0..3 {
        let g = match phase {
            0 => &mut t.pop,
            1 => &mut t.read,
            _ => &mut t.push,
        };
        g.states = vec![StackState::default(); n];
        let mut rank = (0..n).collect::<Vec<_>>();
        shuffle(&mut rank, s);
        g.start = rank[0] as u32;
        for at in 0..n {
            let id = rank[at];
            g.states[id].accepting = next(s) % 4 == 0;
            if at + 1 == n {
                continue;
            }
            for symbol in 0..3 {
                if next(s) % 4 == 0 {
                    let target = rank[at + 1 + next(s) as usize % (n - at - 1)];
                    g.states[id].transitions.push(StackTransition {
                        label: StackLabel::Symbol(symbol),
                        target: target as u32,
                    });
                }
            }
            if phase == 0 && next(s) % 3 == 0 {
                let target = rank[at + 1 + next(s) as usize % (n - at - 1)];
                g.states[id].transitions.push(StackTransition {
                    label: StackLabel::Default,
                    target: target as u32,
                });
            }
            shuffle(&mut g.states[id].transitions, s);
        }
    }
    for link in [&mut t.pop_to_read, &mut t.pop_to_push, &mut t.read_to_push] {
        *link = (0..n)
            .map(|_| {
                if next(s) % 3 == 0 {
                    Some((next(s) % n as u64) as u32)
                } else {
                    None
                }
            })
            .collect();
    }
    t
}
fn apply(t: &StackTemplate, stack: &[u32]) -> BTreeSet<Vec<u32>> {
    let mut todo = vec![(0u8, t.pop.start, stack.to_vec())];
    let mut seen = BTreeSet::new();
    let mut result = BTreeSet::new();
    while let Some((phase, id, word)) = todo.pop() {
        if !seen.insert((phase, id, word.clone())) {
            continue;
        }
        assert!(seen.len() <= 100_000, "oracle fixture too large");
        let g = match phase {
            0 => &t.pop,
            1 => &t.read,
            _ => &t.push,
        };
        let q = &g.states[id as usize];
        if q.accepting {
            result.insert(word.clone());
        }
        if phase == 0 {
            for (links, target_phase) in [(&t.pop_to_read, 1), (&t.pop_to_push, 2)] {
                if let Some(Some(to)) = links.get(id as usize) {
                    todo.push((target_phase, *to, word.clone()));
                }
            }
        } else if phase == 1 {
            if let Some(Some(to)) = t.read_to_push.get(id as usize) {
                todo.push((2, *to, word.clone()));
            }
        }
        if phase == 2 {
            for e in &q.transitions {
                let StackLabel::Symbol(label) = e.label else {
                    panic!("invalid generator")
                };
                let mut w = word.clone();
                w.push(label);
                todo.push((phase, e.target, w));
            }
        } else if let Some(top) = word.last() {
            let edge = q
                .transitions
                .iter()
                .find(|e| e.label == StackLabel::Symbol(*top))
                .or_else(|| {
                    if phase == 0 {
                        q.transitions
                            .iter()
                            .find(|e| e.label == StackLabel::Default)
                    } else {
                        None
                    }
                });
            if let Some(e) = edge {
                let mut w = word.clone();
                if phase == 0 {
                    w.pop();
                }
                todo.push((phase, e.target, w));
            }
        }
    }
    result
}
fn advance(ts: &[StackTemplate], words: &BTreeSet<Vec<u32>>, bytes: &[u8]) -> BTreeSet<Vec<u32>> {
    let mut out = words.clone();
    for b in bytes {
        let idx = match *b {
            b'a' => 0,
            b'b' => 1,
            b'0' => 2,
            b'1' => 3,
            b'2' => 4,
            b'x' => 5,
            _ => return BTreeSet::new(),
        };
        out = out.iter().flat_map(|w| apply(&ts[idx], w)).collect();
        assert!(
            out.len() < 100_000,
            "generated relation exceeds oracle limit"
        );
    }
    out
}
fn has(m: &[u32], id: u32) -> bool {
    m.get(id as usize / 32)
        .is_some_and(|x| x & (1 << (id % 32)) != 0)
}
fn check(
    c: &Constraint,
    prefix: &[u8],
    expected: &BTreeSet<Vec<u32>>,
    ts: &[StackTemplate],
    done: &StackTemplate,
    pieces: &[(u32, Vec<u8>)],
    case: u64,
    representation: &str,
    checks: &mut u64,
) {
    let mut state = c.start();
    for b in prefix {
        state
            .commit_bytes(&[*b])
            .unwrap_or_else(|e| panic!("case{case} {representation} prefix{prefix:?}: {e}"));
    }
    let complete = expected.iter().any(|w| !apply(done, w).is_empty());
    assert_eq!(
        state.is_accepting(),
        complete,
        "completion case{case} {representation} prefix{prefix:?} words{expected:?} completion={done:?}"
    );
    let mut mask = vec![0; c.mask_len()];
    state.fill_mask(&mut mask);
    for (id, bytes) in pieces {
        let yes = if *id == 255 {
            complete
        } else {
            !advance(ts, expected, bytes).is_empty()
        };
        assert_eq!(
            has(&mask, *id),
            yes,
            "mask case{case} {representation} prefix{prefix:?} token{id}={bytes:?} words{expected:?} templates={ts:?}"
        );
        *checks += 1;
        if yes && *id != 255 && case % 4 == 0 {
            let next_words = advance(ts, expected, bytes);
            let next_complete = next_words.iter().any(|w| !apply(done, w).is_empty());
            let mut next_state = state.clone();
            next_state.commit_token(*id).unwrap();
            assert_eq!(
                next_state.is_accepting(),
                next_complete,
                "token-commit completion case{case} {representation} prefix{prefix:?} token{id}"
            );
            let next_mask = next_state.mask();
            for (next_id, next_bytes) in pieces {
                let expect = if *next_id == 255 {
                    next_complete
                } else {
                    !advance(ts, &next_words, next_bytes).is_empty()
                };
                assert_eq!(
                    has(&next_mask, *next_id),
                    expect,
                    "token-commit mask case{case} {representation} prefix{prefix:?} token{id} next{next_id}"
                );
                *checks += 1;
            }
        }
    }
    for id in 0..256 {
        if !pieces.iter().any(|p| p.0 == id) {
            assert!(!has(&mask, id), "gap {id},case{case} {representation}");
        }
    }
    if complete {
        state.commit_token(255).unwrap();
        state.fill_mask(&mut mask);
        assert!(mask.iter().all(|x| *x == 0), "terminated mask");
    }
}
fn verify_phase_programs(offset: u64, count: u64, prefixes_limit: usize) {
    let mut pieces: Vec<(u32, Vec<u8>)> = [
        "a", "b", "aa", "ab", "ba", "bb", "0a", "1b", "2a", "xaa", "x", "0", "1", "2", "00", "102",
        "x01a", "?",
    ]
    .iter()
    .enumerate()
    .map(|(i, s)| ((i as u32) * 3, s.as_bytes().to_vec()))
    .collect();
    pieces.push((63, b"ab".to_vec()));
    pieces.push((127, b"a".to_vec()));
    pieces.push((255, b"<END>".to_vec()));
    let vocab = Vocab::new(pieces.clone());
    let lexer = LexerDefinition::new(
        [b'a', b'b', b'0', b'1', b'2', b'x']
            .iter()
            .map(|b| TerminalPattern::literal(vec![*b]))
            .collect(),
    );
    let mut checks = 0;
    let mut prefixes = 0;
    for case in offset..offset + count {
        let mut seed = 0x6088940720u64.wrapping_add(case.wrapping_mul(0x8173221));
        let ts = vec![
            random_template(&mut seed),
            random_template(&mut seed),
            StackTemplate::rewrite([], [0]),
            StackTemplate::rewrite([], [1]),
            StackTemplate::rewrite([], [2]),
            StackTemplate::rewrite([StackLabel::Default], []),
        ];
        let done = if case % 7 == 0 {
            StackTemplate::identity()
        } else {
            random_template(&mut seed)
        };
        let definition = ParserDefinition {
            stack_symbol_count: 3,
            terminals: ts.clone(),
            completion: done.clone(),
        };
        let program = ParserProgram::new(definition).unwrap();
        let mut compiled = Vec::new();
        for optimization in [Optimization::FastBuild, Optimization::FastRuntime] {
            let c = program
                .compile_with(
                    &lexer,
                    &vocab,
                    TemplateBuildOptions::default()
                        .optimization(optimization)
                        .end_tokens([255]),
                )
                .unwrap();
            let name = if optimization == Optimization::FastBuild {
                "dynamic"
            } else {
                "static"
            };
            assert_eq!(c.parser_backend(), ParserBackend::TemplateDfa);
            let loaded = Constraint::load(&c.save()).unwrap();
            let external =
                Constraint::load_with_vocab(&c.save_with_external_vocab().unwrap(), &vocab)
                    .unwrap();
            compiled.extend([
                (format!("{name}-fresh"), c),
                (format!("{name}-self"), loaded),
                (format!("{name}-external"), external),
            ]);
        }
        let mut queue = VecDeque::from([(Vec::<u8>::new(), BTreeSet::from([vec![0]]))]);
        let mut visited = BTreeSet::new();
        let mut visited_count = 0;
        while let Some((prefix, words)) = queue.pop_front() {
            if !visited.insert(words.clone()) {
                continue;
            }
            for (name, c) in &compiled {
                check(
                    c,
                    &prefix,
                    &words,
                    &ts,
                    &done,
                    &pieces,
                    case,
                    name,
                    &mut checks,
                );
            }
            prefixes += 1;
            visited_count += 1;
            if visited_count >= prefixes_limit {
                break;
            }
            if prefix.len() < 4 && words.len() < 80 {
                for b in [b'a', b'b', b'0', b'1', b'2', b'x'] {
                    let after = advance(&ts, &words, &[b]);
                    if after.is_empty() || after.len() > 160 || after.iter().any(|s| s.len() > 12) {
                        continue;
                    }
                    let mut next = prefix.clone();
                    next.push(b);
                    queue.push_back((next, after));
                }
            }
        }
        println!("case={case} prefixes={visited_count} checks={checks}");
    }
    println!(
        "PASS cases={count} prefixes={prefixes} checks={checks} representations=6 sparse_end_token=255"
    );
}

// Enumerate complete segmentations at each byte offset. With literal terminal
// patterns each terminal has one fixed width; no within-terminal maximal-munch
// alternative is ignored by this recognizer.
fn language(
    patterns: &[Vec<u8>],
    ts: &[StackTemplate],
    done: &StackTemplate,
    bytes: &[u8],
) -> (bool, bool) {
    let mut boundaries = vec![BTreeSet::<Vec<u32>>::new(); bytes.len() + 1];
    boundaries[0].insert(vec![0]);
    let mut pending = false;
    for pos in 0..=bytes.len() {
        if boundaries[pos].is_empty() {
            continue;
        }
        for (pat, t) in patterns.iter().zip(ts) {
            let tail = &bytes[pos..];
            if tail.starts_with(pat) {
                let outputs = boundaries[pos]
                    .iter()
                    .flat_map(|w| apply(t, w))
                    .collect::<BTreeSet<_>>();
                boundaries[pos + pat.len()].extend(outputs);
            } else if tail.len() < pat.len() && pat.starts_with(tail) && !tail.is_empty() {
                pending |= boundaries[pos].iter().any(|w| !apply(t, w).is_empty());
            }
        }
    }
    let words = &boundaries[bytes.len()];
    (
        pending || !words.is_empty(),
        words.iter().any(|w| !apply(done, w).is_empty()),
    )
}
// Independent recognizer restricted to the exact regex family in this test:
// a+, b+, and spaces+. Each matching terminal takes its longest known run;
// live partial lexemes remain viable. Other alternatives retain literal widths.
// No production lexer, regex engine, DFA or GSS code is called here.
fn greedy_language(
    patterns: &[Vec<u8>],
    ts: &[StackTemplate],
    done: &StackTemplate,
    bytes: &[u8],
) -> (bool, bool) {
    let mut boundaries = vec![BTreeSet::<Vec<u32>>::new(); bytes.len() + 1];
    boundaries[0].insert(vec![0]);
    let mut pending = false;
    for pos in 0..=bytes.len() {
        if boundaries[pos].is_empty() {
            continue;
        }
        for (terminal, (pat, t)) in patterns.iter().zip(ts).enumerate() {
            let tail = &bytes[pos..];
            let plus = matches!(terminal, 0 | 2 | 7);
            let matched = if plus {
                let n = tail.iter().take_while(|&&b| b == pat[0]).count();
                (n > 0).then_some(n)
            } else {
                tail.starts_with(pat).then_some(pat.len())
            };
            if let Some(width) = matched {
                let outputs = boundaries[pos]
                    .iter()
                    .flat_map(|w| apply(t, w))
                    .collect::<BTreeSet<_>>();
                boundaries[pos + width].extend(outputs);
            }
            let partial = if plus {
                !tail.is_empty() && tail.iter().all(|&b| b == pat[0])
            } else {
                !tail.is_empty() && tail.len() < pat.len() && pat.starts_with(tail)
            };
            if partial {
                pending |= boundaries[pos].iter().any(|w| !apply(t, w).is_empty());
            }
        }
    }
    let words = &boundaries[bytes.len()];
    (
        pending || !words.is_empty(),
        words.iter().any(|w| !apply(done, w).is_empty()),
    )
}
fn verify_overlapping_programs(offset: u64, count: u64, repeats: bool) {
    let pieces: Vec<(u32, Vec<u8>)> = [
        "a", "b", "ab", "ba", "aba", "abb", "aa", "bb", "a0", "a b", " ab", "0a", "x", "0", "1",
        "2", " ", "  ", "a ", " a", "axab", "?", "<END>",
    ]
    .iter()
    .enumerate()
    .map(|(i, s)| {
        (
            if i == 22 { 255 } else { i as u32 * 3 },
            s.as_bytes().to_vec(),
        )
    })
    .collect();
    let vocab = Vocab::new(pieces.clone());
    let patterns = ["a", "ab", "b", "0", "1", "2", "x", " "]
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect::<Vec<_>>();
    let lex = LexerDefinition::new(
        patterns
            .iter()
            .enumerate()
            .map(|(i, p)| match (repeats, i) {
                (true, 0) => TerminalPattern::regex("a+"),
                (true, 2) => TerminalPattern::regex("b+"),
                (true, 7) => TerminalPattern::regex(" +"),
                _ => TerminalPattern::literal(p.clone()),
            })
            .collect(),
    )
    .ignoring(7);
    type Oracle = fn(&[Vec<u8>], &[StackTemplate], &StackTemplate, &[u8]) -> (bool, bool);
    let recognize: Oracle = if repeats { greedy_language } else { language };
    let mut total = 0;
    let mut checks = 0;
    for case in offset..offset + count {
        let mut seed = 0x1236088940720u64.wrapping_add(case.wrapping_mul(0x831277));
        let ts = vec![
            random_template(&mut seed),
            random_template(&mut seed),
            random_template(&mut seed),
            StackTemplate::rewrite([], [0]),
            StackTemplate::rewrite([], [1]),
            StackTemplate::rewrite([], [2]),
            StackTemplate::rewrite([StackLabel::Default], []),
            StackTemplate::identity(),
        ];
        let done = if case % 7 == 0 {
            StackTemplate::identity()
        } else {
            random_template(&mut seed)
        };
        let p = ParserProgram::new(ParserDefinition {
            stack_symbol_count: 3,
            terminals: ts.clone(),
            completion: done.clone(),
        })
        .unwrap();
        let mut parsers = Vec::new();
        for optimization in [Optimization::FastBuild, Optimization::FastRuntime] {
            let c = p
                .compile_with(
                    &lex,
                    &vocab,
                    TemplateBuildOptions::default()
                        .optimization(optimization)
                        .end_tokens([255]),
                )
                .unwrap();
            let name = if optimization == Optimization::FastBuild {
                "dynamic"
            } else {
                "static"
            };
            let loaded = Constraint::load(c.save()).unwrap();
            let external =
                Constraint::load_with_vocab(c.save_with_external_vocab().unwrap(), &vocab).unwrap();
            parsers.extend([
                (format!("{name}-fresh"), c),
                (format!("{name}-self"), loaded),
                (format!("{name}-external"), external),
            ]);
        }
        let mut todo = VecDeque::from([vec![]]);
        let mut n = 0;
        while let Some(prefix) = todo.pop_front() {
            let (viable, complete) = recognize(&patterns, &ts, &done, &prefix);
            assert!(viable);
            let want = pieces
                .iter()
                .map(|(id, word)| {
                    if *id == 255 {
                        return complete;
                    }
                    let mut b = prefix.clone();
                    b.extend(word);
                    recognize(&patterns, &ts, &done, &b).0
                })
                .collect::<Vec<_>>();
            for (name, c) in &parsers {
                assert_eq!(c.parser_backend(), ParserBackend::TemplateDfa);
                let mut st = c.start();
                for b in &prefix {
                    st.commit_bytes(&[*b]).unwrap();
                }
                assert_eq!(
                    st.is_accepting(),
                    complete,
                    "completion case{case} {name} prefix{prefix:?} done{done:?}"
                );
                let mask = st.mask();
                for ((id, word), yes) in pieces.iter().zip(&want) {
                    assert_eq!(
                        has(&mask, *id),
                        *yes,
                        "case{case} {name} prefix{prefix:?} word{word:?} templates={ts:?}"
                    );
                    checks += 1;
                    if *yes && *id != 255 && case % 4 == 0 {
                        let mut next_prefix = prefix.clone();
                        next_prefix.extend(word);
                        let expected_complete = recognize(&patterns, &ts, &done, &next_prefix).1;
                        let mut next_state = st.clone();
                        next_state.commit_token(*id).unwrap();
                        assert_eq!(
                            next_state.is_accepting(),
                            expected_complete,
                            "model-token completion case{case} {name} prefix{prefix:?} token{id}"
                        );
                        let next_mask = next_state.mask();
                        for (nid, nword) in &pieces {
                            let expect = if *nid == 255 {
                                expected_complete
                            } else {
                                let mut extended = next_prefix.clone();
                                extended.extend(nword);
                                recognize(&patterns, &ts, &done, &extended).0
                            };
                            assert_eq!(
                                has(&next_mask, *nid),
                                expect,
                                "model-token mask case{case} {name} prefix{prefix:?} token{id} next{nid}"
                            );
                            checks += 1;
                        }
                    }
                }
            }
            total += 1;
            n += 1;
            if n >= 100 {
                break;
            }
            if prefix.len() < 4 {
                for b in [b'a', b'b', b'0', b'1', b'2', b'x', b' '] {
                    let mut next = prefix.clone();
                    next.push(b);
                    if recognize(&patterns, &ts, &done, &next).0 {
                        todo.push_back(next);
                    }
                }
            }
        }
        println!("case={case} prefixes={n} cumulative_checks={checks}");
    }
    println!("PASS overlap_cases={count} prefixes={total} token_checks={checks} representations=6");
}

#[test]
fn permuted_phase_graphs_and_completion_match_literal_oracle_in_all_formats() {
    verify_phase_programs(0, 32, 80);
}

#[test]
fn ambiguous_literal_segmentation_matches_oracle_after_model_token_commits() {
    verify_overlapping_programs(0, 32, false);
}

#[test]
fn greedy_regex_lexemes_match_oracle_across_model_token_boundaries() {
    verify_overlapping_programs(0, 16, true);
}
