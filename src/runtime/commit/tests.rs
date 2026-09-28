use super::advance::{apply_single_top_action_fast, materialize_reduce_chain_outputs};
use super::small_queue::try_advance_unique_actionable_top_fast;
use super::*;
use crate::{Constraint as Constraint, DynamicConstraint, Grammar, Vocab};
use std::collections::BTreeSet;

type CanonicalCommitState =
    Vec<(u32, Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>)>;

#[test]
fn lexer_only_actionable_cache_keeps_distinct_lexer_lanes() {
    let vocab = Vocab::new(vec![
        (0, b"ab".to_vec()), (1, b"cd".to_vec()),
        (2, b"abcd".to_vec()), (3, b"abx".to_vec()),
        (4, b"zabc".to_vec()),
    ]);
    let compiled = Constraint::compile(Grammar::glrm(r#"
            start document;
            lexer group first ::= A, C;
            lexer group second ::= B;
            t A ::= "abx";
            t C ::= "abc";
            t B ::= "abcd";
            nt document ::= A | B | "z" C;
        "#), &vocab).unwrap();
    for constraint in [&compiled, &Constraint::load(compiled.save()).unwrap()] {
        let mut state = constraint.start();
        state.commit_bytes(b"ab").unwrap();
        let mut reference = state.state.clone();
        assert!(commit_token_no_fast_path_reference(constraint, &mut reference, 1).is_ok());
        let mut actual = state.state.clone();
        let mut scratch = CommitBuffers::default();
        assert!(commit_token_impl(constraint, &mut actual, &mut scratch, 1).is_ok(),
            "lexer lanes sharing parser stacks must not share actionable-match answers");
        assert_commit_fast_path_equivalence(constraint, state.state.clone(), 1, &actual, true);
    }
}

#[test]
fn unfinished_ignore_prefixes_survive_all_commit_paths() {
    let fragments: &[&[u8]] = &[
        b"a", b"+", b"b", b"//", b"/////", b"line", b"\n",
        b"/*", b"/***", b"body", b"*/", b" /*unfinished", b"//line\n+b",
    ];
    let vocab = Vocab::new(fragments.iter().enumerate()
        .map(|(id, bytes)| (id as u32, bytes.to_vec())).collect());
    for partitions in ["", "lexer group words ::= WORD; lexer group trivia ::= WS;"] {
        let source = format!(r#"
                start document;
                {partitions}
                ignore WS;
                t WS ::= " "+ | "//" [^\n]* "\n" | "/*" ([^*] | "*" [^/])* "*/";
                t WORD ::= /[a-z]+/;
                nt document ::= WORD ("+" WORD)*;
            "#);
        let built = Constraint::compile(Grammar::glrm(&source), &vocab).unwrap();
        let loaded = Constraint::load(built.save()).unwrap();
        for constraint in [&built, &loaded] {
            for ids in [
                vec![0,4,5,6,1,2],
                vec![0,7,9,10,1,2],
                vec![0,8,9,10,1,2],
                vec![0,11,10,1,2],
                vec![0,12],
            ] {
                let mut actual = constraint.start();
                let mut bytewise = constraint.start();
                for token in ids {
                    let before = actual.state.clone();
                    assert!(token_in_mask(&actual.mask(), token),
                        "ignore prefix absent from mask partitions={partitions} token={token}");
                    let mut reference = before.clone();
                    assert!(commit_token_no_fast_path_reference(constraint, &mut reference, token).is_ok(),
                        "general queue must retain unfinished ignore token={token}");
                    actual.commit_token(token).unwrap();
                    assert_commit_fast_path_equivalence(constraint, before, token, &actual.state, true);
                    for byte in fragments[token as usize] { bytewise.commit_bytes(&[*byte]).unwrap(); }
                    assert_eq!(actual.mask(), bytewise.mask(), "token-vs-byte prefix token={token}");
                }
                assert!(actual.is_accepting());
                assert!(bytewise.is_accepting());
            }
        }
    }
}

#[test]
fn recursive_batched_admission_matches_scalar_provider_reference() {
    use crate::ds::bitset::BitSet;
    let vocab = Vocab::new(b"[ab]! ".iter().enumerate()
        .map(|(id,byte)|(id as u32,vec![*byte])).collect());
    for source in [
        r#"start child; t WORD ::= /[ab]+/; nt child ::= WORD;"#,
        r#"start child; ignore WS; t WS ::= " "+; t WORD ::= /[ab]+/; nt child ::= WORD?;"#,
        r#"start child; t A ::= "a"; t AB ::= "ab"; nt child ::= (A | AB)*;"#,
    ] {
        let child = Constraint::compile(Grammar::glrm(source), &vocab).unwrap();
        let middle = Constraint::compile(Grammar::glrm(
            r#"start middle; extern grammar child; nt middle ::= "[" child "]";"#,
        ), &vocab).unwrap().bind_grammar_dynamic_boundary("child", child).unwrap();
        let outer = Constraint::compile(Grammar::glrm(
            r#"start outer; extern grammar middle; nt outer ::= middle "!";"#,
        ), &vocab).unwrap().bind_grammar_dynamic_boundary("middle", middle).unwrap();
        let loaded = Constraint::load(outer.save()).unwrap();
        for constraint in [&outer, &loaded] {
            for prefix in ["", "[", "[a", "[ab", "[ab]", "[ab]!", "[ ", "[]"] {
                let mut state = constraint.start();
                if state.commit_bytes(prefix.as_bytes()).is_err() {continue;}
                let runtime = state.constraint;
                let count = runtime_terminal_count(runtime);
                for (_, gss) in state.state.iter() {
                    for stride in [1,2,3] {
                        let mut candidates=BitSet::new(count);
                        for terminal in (0..count).step_by(stride) {candidates.set(terminal);}
                        let actual=runtime.compact_segmented_parser_admitted_terminals(gss,&candidates).unwrap();
                        let mut expected=BitSet::new(count);
                        for terminal in candidates.iter_ones() {
                            if runtime.compact_segmented_parser_may_advance_on(gss,terminal as u32).unwrap() {
                                expected.set(terminal);
                            }
                        }
                        assert_eq!(actual,expected,"prefix={prefix} stride={stride} source={source}");
                        assert_eq!(runtime.compact_segmented_parser_may_advance_on_any(gss,&candidates),Some(!expected.is_empty()));
                    }
                }
            }
        }
    }
}

#[test]
fn unchanged_runtime_state_preserves_fill_mask_cache() {
    let vocab = Vocab::new(vec![
        (0, b"a".to_vec()),
        (1, b"aa".to_vec()),
        (2, b"b".to_vec()),
    ]);
    let constraint = Constraint::compile(
        Grammar::glrm(
            r#"glrm 1;
                start start;
                t A = /a+/;
                t B = "b";
                nt start = A B;
                "#,
        ),
        &vocab,
    )
    .unwrap();

    let mut state = constraint.start();
    state.commit_token(0).unwrap();

    // Populate the ordinary per-sequence mask cache while inside A. A
    // further `a` is an exact lexer self-loop: no parser terminal is
    // consumed and the complete runtime state remains identical.
    let cached_mask = state.mask();
    let before_generation = state.generation;
    let before_state = state.state.clone();
    {
        let cache = state.mask_cache.lock().unwrap();
        assert_eq!(cache.as_ref().unwrap().generation, before_generation);
    }

    state.commit_token(0).unwrap();
    assert_eq!(state.generation, before_generation + 1);
    assert_eq!(before_state, state.state);
    {
        let cache = state.mask_cache.lock().unwrap();
        assert_eq!(cache.as_ref().unwrap().generation, state.generation);
        assert_eq!(cache.as_ref().unwrap().mask, cached_mask);
    }
    assert_eq!(state.mask(), cached_mask);

    // Completing A and advancing the parser must invalidate the cached
    // generation. The following mask is therefore recomputed for B.
    state.commit_token(2).unwrap();
    {
        let cache = state.mask_cache.lock().unwrap();
        assert_ne!(cache.as_ref().unwrap().generation, state.generation);
    }
}

#[test]
fn unchanged_dynamic_mask_projection_preserves_fill_mask_cache() {
    let vocab = Vocab::new(vec![
        (0, b"\"".to_vec()),
        (1, b"a".to_vec()),
        (2, b"aa".to_vec()),
        (3, b"b".to_vec()),
    ]);
    let dynamic = DynamicConstraint::from_json_schema(
        r#"{"type":"string","maxLength":1000000000}"#,
        &vocab,
    )
    .unwrap();
    let mut state = dynamic.inner.start();

    // Enter the quoted lazy bounded-string residual, consume one interior
    // byte, then cache its mask. Far from the upper bound, another interior
    // byte changes the exact symbolic residual coordinate while the parser
    // has not seen a terminal boundary and the finite vocabulary-horizon
    // mask coordinate remains unchanged.
    state.commit_token(0).unwrap(); // opening quote
    state.commit_token(1).unwrap(); // first interior 'a'
    let cached_mask = state.mask();
    let before_generation = state.generation;
    let before_state = state.state.clone();

    state.commit_token(1).unwrap(); // second interior 'a'
    assert_ne!(before_state, state.state, "test must exercise a changed exact lexer state");
    assert!(
        state.dynamic_mask_projection_state_eq(&before_state, &state.state),
        "changed exact state should remain equal in the mask-only coordinate",
    );
    {
        let cache = state.mask_cache.lock().unwrap();
        assert_eq!(cache.as_ref().unwrap().generation, before_generation + 1);
        assert_eq!(cache.as_ref().unwrap().mask, cached_mask);
    }
    assert_eq!(state.mask(), cached_mask);
}

#[test]
fn recursive_radix_candidate_admission_matches_pointwise_exact_commit() {
    let vocab = Vocab::new(vec![
        (0, b"X".to_vec()),
        (1, b"X[".to_vec()),
        (2, b"X[a".to_vec()),
        (3, b"X[a]".to_vec()),
        (4, b"X[a]!".to_vec()),
        (5, b"X[a]!".to_vec()),
        (6, b"[".to_vec()),
        (7, b"[a".to_vec()),
        (8, b"[a]".to_vec()),
        (9, b"[a]!".to_vec()),
        (10, b"a".to_vec()),
        (11, b"a]".to_vec()),
        (12, b"a]!".to_vec()),
        (13, b"]".to_vec()),
        (14, b"]!".to_vec()),
        (15, b"!".to_vec()),
    ]);
    let child = Constraint::compile(
        Grammar::glrm("glrm 1; start child; nt child = \"[\" \"a\" \"]\";"),
        &vocab,
    )
    .unwrap();
    let parent = Constraint::compile(
        Grammar::glrm(
            "glrm 1; start document; extern grammar child; nt document = \"X\" child \"!\";",
        ),
        &vocab,
    )
    .unwrap();
    let bound = parent
        .bind_grammar_dynamic_boundary("child", child)
        .unwrap();
    assert!(bound.uses_compact_segmented_parser_runtime());

    for path in [
        &[][..],
        &[0][..],
        &[0, 6][..],
        &[0, 6, 10][..],
        &[0, 6, 10, 13][..],
    ] {
        let mut state = bound.start();
        for &token in path {
            state.commit_token(token).unwrap();
        }

        let mut batch_candidates = bound.token_bytes_iter().collect::<Vec<_>>();
        let mut batch_buffers = CommitBuffers::default();
        let mut batched = Vec::new();
        admissible_byte_token_candidates_from_state_exact(
            &bound,
            &state.state,
            &mut batch_buffers,
            &mut batch_candidates,
            &mut batched,
        );
        batched.sort_unstable();

        let mut pointwise_buffers = CommitBuffers::default();
        let mut pointwise = bound
            .token_bytes_iter()
            .filter_map(|(token_id, _)| {
                token_admissible_from_state_exact(
                    &bound,
                    &state.state,
                    &mut pointwise_buffers,
                    token_id,
                )
                .then_some(token_id)
            })
            .collect::<Vec<_>>();
        pointwise.sort_unstable();
        assert_eq!(batched, pointwise, "candidate mismatch after path {path:?}");
    }
}

#[test]
fn recursive_scoped_tokenizer_exec_uses_only_the_active_leaf() {
    let vocab = Vocab::new(vec![
        (0, b"X".to_vec()),
        (1, b"a".to_vec()),
        (2, b"!".to_vec()),
        (3, b"Xa!".to_vec()),
    ]);
    let child = Constraint::compile(
        Grammar::glrm("glrm 1; start child; t A = \"a\"; nt child = A;"),
        &vocab,
    )
    .unwrap();
    let parent = Constraint::compile(
        Grammar::glrm(
            "glrm 1; start document; extern grammar child; t X = \"X\"; t BANG = \"!\"; nt document = X child BANG;",
        ),
        &vocab,
    )
    .unwrap();
    let local_x = parent
        .terminal_display_names
        .iter()
        .position(|name| name == "X")
        .unwrap() as u32;
    let local_a = child
        .terminal_display_names
        .iter()
        .position(|name| name == "A")
        .unwrap() as u32;
    let bound = parent.bind_grammar("child", child).unwrap();
    let loaded = Constraint::load(bound.save()).unwrap();
    for constraint in [&bound, &loaded] {
        let layout = constraint.recursive_parser_layout().unwrap().unwrap();
        assert_eq!(layout.leaves.len(), 2);
        let scoped_x = constraint.recursive_terminal_scoped_id(0, local_x).unwrap();
        let scoped_a = constraint.recursive_terminal_scoped_id(1, local_a).unwrap();
        assert!(scoped_x >= constraint.table.num_terminals);
        assert!(scoped_a >= constraint.table.num_terminals);

        let root_reset = constraint.recursive_tokenizer_reset_state(0).unwrap();
        let root_x = tokenizer_scan::execute_recursive_tokenizer_from_state_small(
            constraint,
            b"X",
            root_reset,
        )
        .unwrap();
        assert!(root_x.matches.iter().any(|matched| matched.id == scoped_x));
        assert!(root_x.end_state.iter().all(|&state| {
            constraint.recursive_tokenizer_leaf_state(state).unwrap().0 == 0
        }));
        let root_a = tokenizer_scan::execute_recursive_tokenizer_from_state_small(
            constraint,
            b"a",
            root_reset,
        )
        .unwrap();
        assert!(
            root_a.matches.iter().all(|matched| matched.id != scoped_a),
            "child terminal leaked from inactive leaf: {:?}",
            root_a.matches,
        );

        let child_reset = constraint.recursive_tokenizer_reset_state(1).unwrap();
        let child_a = tokenizer_scan::execute_recursive_tokenizer_from_state_small(
            constraint,
            b"a",
            child_reset,
        )
        .unwrap();
        assert!(child_a.matches.iter().any(|matched| matched.id == scoped_a));
        assert!(child_a.end_state.iter().all(|&state| {
            constraint.recursive_tokenizer_leaf_state(state).unwrap().0 == 1
        }));
    }
}

fn canonical_commit_state(
    state: &ParserStateMap,
) -> CanonicalCommitState {
    canonical_commit_state_for_equivalence_assert(state)
}

#[test]
fn exact_prefilter_rejects_multiple_conditional_candidates_independent_of_row_order() {
    let mut terminals = crate::ds::bitset::BitSet::new(8);
    terminals.set(1);
    terminals.set(2);
    let unconditional = crate::ds::bitset::BitSet::new(8);

    let row_a = ActionRow::from_iter([
        (1, Action::Reduce(0, 1)),
        (2, Action::Reduce(0, 1)),
    ]);
    let row_b = ActionRow::from_iter([
        (2, Action::Reduce(0, 1)),
        (1, Action::Reduce(0, 1)),
    ]);

    assert_eq!(
        single_conditional_candidate(&row_a, &unconditional, &terminals),
        Err(()),
    );
    assert_eq!(
        single_conditional_candidate(&row_b, &unconditional, &terminals),
        Err(()),
    );

    let mut one_terminal = crate::ds::bitset::BitSet::new(8);
    one_terminal.set(2);
    assert_eq!(
        single_conditional_candidate(&row_a, &unconditional, &one_terminal),
        Ok(Some(2)),
    );
}

#[test]
fn batched_and_cached_end_state_admission_match_pointwise_simulation() {
    let constraint = Constraint::from_glrm_grammar(
        r#"
                start start;
                t A ::= "a";
                t AB ::= "ab";
                t B ::= "b";
                t C ::= "c";
                nt item ::= A | AB | B;
                nt start ::= item C;
            "#,
        &Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"b".to_vec()),
            (3, b"c".to_vec()),
            (4, b"ac".to_vec()),
            (5, b"abc".to_vec()),
        ]),
    )
    .unwrap();
    let state = constraint.start();
    let gss = state.state.values().next().unwrap();
    let initial = constraint.runtime_commit_initial_state();
    let non_initial = (0..constraint.tokenizer.num_states())
        .find(|&end_state| {
            end_state != initial
                && !constraint
                    .tokenizer
                    .possible_future_terminals(end_state)
                    .is_empty()
        })
        .expect("test tokenizer should have a live non-initial state");
    // Repeating one exact continuation is enough to exercise batching and
    // cache reuse; production callers usually provide many distinct states,
    // but the theorem does not depend on their distinctness.
    let end_states = vec![initial, non_initial, non_initial];

    let admitted = batched_end_state_admitted_terminals(
        &constraint,
        gss,
        &end_states,
    )
    .expect("multiple non-initial tokenizer states should batch");
    for &end_state in &end_states {
        assert_eq!(
            end_state_may_advance_with_batch(
                &constraint,
                gss,
                end_state,
                Some(&admitted),
            ),
            end_state_may_advance(&constraint, gss, end_state),
            "batched admission differs for tokenizer state {end_state}",
        );
    }

    let mut cache = SmallVec::<[ParserAdmissionCacheEntry; 8]>::new();
    let index = cached_batched_end_state_admission(
        &constraint,
        gss,
        &end_states,
        &mut cache,
    )
    .expect("multiple non-initial tokenizer states should populate cache");
    for &end_state in &end_states {
        assert_eq!(
            end_state_may_advance_from_cache_entry(
                &constraint,
                end_state,
                &cache[index],
            ),
            end_state_may_advance(&constraint, gss, end_state),
            "cached admission differs for tokenizer state {end_state}",
        );
    }

    // Repeating the identical query must be a pure cache hit and retain the
    // exact pointwise facts.
    let tested = cache[index].tested.clone();
    let admitted_before = cache[index].admitted.clone();
    let repeat = cached_batched_end_state_admission(
        &constraint,
        gss,
        &end_states,
        &mut cache,
    )
    .unwrap();
    assert_eq!(repeat, index);
    assert_eq!(cache[index].tested, tested);
    assert_eq!(cache[index].admitted, admitted_before);
}

#[test]
fn admission_cache_reset_drops_persistent_gss_references() {
    let gss = ParserGSS::from_single_stack(
        vec![0_u32, 1],
        TerminalsDisallowed::new(),
    );
    let mut buffers = CommitBuffers::default();
    buffers.admission_cache.push(ParserAdmissionCacheEntry {
        gss,
        tested: crate::ds::bitset::BitSet::new(2),
        admitted: crate::ds::bitset::BitSet::new(2),
        boolean_queries: SmallVec::new(),
    });
    buffers.clear_all();
    assert_eq!(
        buffers.admission_cache.len(),
        1,
        "ordinary scratch reuse should preserve admission facts",
    );
    buffers.reset_all();
    assert!(buffers.admission_cache.is_empty());
}


#[test]
#[ignore]
fn debug_selected10_completed_child_lookahead_actions() {
    use crate::compiler::glr::analysis::EOF;
    use crate::compiler::glr::parser::stack_admissible_terminals;
    use crate::compiler::glr::table::Action;
    use crate::ds::bitset::BitSet;
    use std::collections::BTreeMap;

    let path = std::env::var("GLRMASK_DEBUG_ARTIFACT").expect("GLRMASK_DEBUG_ARTIFACT");
    let bytes = std::fs::read(path).unwrap();
    let constraint = Constraint::load(&bytes).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(b"const x = tools.tool_0({})").unwrap();
    eprintln!("STACKS {:?}", state.debug_parser_stacks());
    for (_, gss) in state.state.iter() {
        let all = BitSet::all(constraint.table.num_terminals as usize + 1);
        let admitted = stack_admissible_terminals(&constraint.table, gss, &all);
        let mut factor = gss.clone();
        let mut blockers = Vec::<u32>::new();
        for depth in 0..16 {
            let top = factor.single_exclusive_top_value();
            let admitted_now =
                stack_admissible_terminals(&constraint.table, &factor, &all);
            eprintln!(
                "FACTOR_DEPTH {depth} top={top:?} stack={:?} admitted={:?} blockers={:?}",
                factor.to_stacks(64),
                admitted_now.iter_ones().collect::<Vec<_>>(),
                blockers,
            );
            let Some((next, blocked)) =
                crate::compiler::glr::parser::lookahead_reduction_factor(
                    &constraint.table,
                    &factor,
                )
            else {
                break;
            };
            for terminal in blocked {
                if !blockers.contains(&terminal) {
                    blockers.push(terminal);
                }
            }
            factor = next;
        }
        for top in gss.peek_values() {
            let mut kinds = BTreeMap::<String, Vec<u32>>::new();
            for bit in admitted.iter_ones() {
                let terminal = if bit == constraint.table.num_terminals as usize {
                    EOF
                } else {
                    bit as u32
                };
                let desc = match constraint.table.action(top, terminal) {
                    Some(Action::Reduce(nt, len)) => format!("reduce:{nt}:{len}"),
                    Some(Action::Shift(target, replace)) => format!("shift:{target}:{replace}"),
                    Some(Action::StackShifts(shifts)) => format!("stackshifts:{}", shifts.len()),
                    Some(Action::GuardedStackShifts(shifts)) => format!("guarded:{}", shifts.len()),
                    Some(Action::ReplaceShifts(targets)) => format!("replace:{}", targets.len()),
                    Some(Action::Split { shift, reduces, accept }) => format!("split:{shift:?}:{reduces:?}:{accept}"),
                    Some(Action::Accept) => "accept".into(),
                    Some(Action::Skip) => "skip".into(),
                    None => "none".into(),
                };
                kinds.entry(desc).or_default().push(terminal);
            }
            eprintln!("TOP {top} ADMITTED {} KINDS {}", admitted.count_ones(), kinds.len());
            for (kind, terminals) in kinds {
                eprintln!("  {kind} count={} sample={:?}", terminals.len(), &terminals[..terminals.len().min(24)]);
            }
        }
    }
}


#[test]
#[ignore]
fn debug_selected10_core_fast_weight_transport() {
    use crate::compiler::glr::labels::DEFAULT_LABEL;
    use crate::ds::weight::Weight;

    fn accepted_weight(constraint: &Constraint, labels: &[u32]) -> Weight {
        let mut state = constraint.parser_dwa.start_state();
        let mut weight = Weight::all();
        for &label in labels {
            let row = &constraint.parser_dwa.states()[state as usize];
            let Some((target, edge_weight)) = row
                .transitions
                .get(&(label as i32))
                .or_else(|| row.transitions.get(&DEFAULT_LABEL))
            else {
                return Weight::empty();
            };
            weight = weight.intersection(edge_weight);
            state = *target;
        }
        let Some(final_weight) = constraint.parser_dwa.states()[state as usize]
            .final_weight
            .as_ref()
        else {
            return Weight::empty();
        };
        weight.intersection(final_weight)
    }

    fn dump(name: &str, constraint: &Constraint, tokenizer_state: u32, token: u32) {
        let internal_token = constraint
            .original_token_to_internal
            .get(token as usize)
            .copied()
            .unwrap_or(u32::MAX);
        let tsids = constraint.internal_tsids_for_state(tokenizer_state).to_vec();
        eprintln!("WEIGHT_TRANSPORT {name} tokenizer_state={tokenizer_state} tsids={tsids:?} token={token} internal_token={internal_token}");
        for labels in [&[22_u32][..], &[22_u32, 0][..]] {
            let weight = accepted_weight(constraint, labels);
            let memberships = tsids
                .iter()
                .map(|&tsid| (tsid, internal_token != u32::MAX && weight.tokens_for_tsid(tsid).contains(internal_token), weight.tokens_for_tsid(tsid).len()))
                .collect::<Vec<_>>();
            eprintln!("WEIGHT_TRANSPORT {name} labels={labels:?} empty={} memberships={memberships:?}", weight.is_empty());
        }
        for key in [22_i32, DEFAULT_LABEL] {
            if let Some(weight) = constraint.parser_top_accept.get(&key) {
                let memberships = tsids.iter().map(|&tsid| (tsid, weight.tokens_for_tsid(tsid).contains(internal_token), weight.tokens_for_tsid(tsid).len())).collect::<Vec<_>>();
                eprintln!("WEIGHT_TRANSPORT {name} top_accept key={key} memberships={memberships:?}");
            }
            if let Some(parts) = constraint.parser_top_accept_parts.get(&key) {
                for (part, weight) in parts.iter().enumerate() {
                    let memberships = tsids.iter().map(|&tsid| (tsid, weight.tokens_for_tsid(tsid).contains(internal_token), weight.tokens_for_tsid(tsid).len())).collect::<Vec<_>>();
                    eprintln!("WEIGHT_TRANSPORT {name} top_part key={key} part={part} memberships={memberships:?}");
                }
            }
        }
        let mut l1 = Vec::new();
        constraint.for_each_direct_regular_l1_acceptance(22, |weight| {
            l1.push(tsids.iter().map(|&tsid| { let tokens = weight.token_set_for_tsid(tsid).map(|tokens| tokens.to_range_set()).unwrap_or_default(); (tsid, tokens.contains(internal_token), tokens.len()) }).collect::<Vec<_>>());
        });
        eprintln!("WEIGHT_TRANSPORT {name} l1={l1:?}");
    }

    let core_path = std::env::var("GLRMASK_DEBUG_CORE_ARTIFACT").unwrap();
    let fast_path = std::env::var("GLRMASK_DEBUG_ARTIFACT").unwrap();
    let core = Constraint::load(&std::fs::read(core_path).unwrap()).unwrap();
    let fast = Constraint::load(&std::fs::read(fast_path).unwrap()).unwrap();
    dump("core", &core, 38, 443);
    dump("fast", &fast, 39, 443);
}

#[test]
#[ignore]
fn debug_selected10_boundary_shallow_acceptance() {
    use crate::compiler::glr::labels::DEFAULT_LABEL;
    use crate::ds::weight::Weight;

    fn accepted_weight(constraint: &Constraint, labels: &[u32]) -> Weight {
        let mut state = constraint.parser_dwa.start_state();
        let mut weight = Weight::all();
        for &label in labels {
            let row = &constraint.parser_dwa.states()[state as usize];
            let Some((target, edge_weight)) = row
                .transitions
                .get(&(label as i32))
                .or_else(|| row.transitions.get(&DEFAULT_LABEL))
            else { return Weight::empty(); };
            weight = weight.intersection(edge_weight);
            state = *target;
        }
        let Some(final_weight) = constraint.parser_dwa.states()[state as usize].final_weight.as_ref()
        else { return Weight::empty(); };
        weight.intersection(final_weight)
    }

    fn contains(constraint: &Constraint, tokenizer_state: u32, token: u32, weight: &Weight) -> bool {
        let internal = constraint.original_token_to_internal[token as usize];
        constraint.internal_tsids_for_state(tokenizer_state)
            .iter()
            .any(|&tsid| weight.tokens_for_tsid(tsid).contains(internal))
    }

    fn dump_case(constraint: &Constraint, name: &str, tokenizer_state: u32, stack: &[u32], tokens: &[u32]) {
        let reversed = stack.iter().rev().copied().collect::<Vec<_>>();
        eprintln!("SHALLOW {name} tokenizer_state={tokenizer_state} stack={stack:?} tsids={:?}", constraint.internal_tsids_for_state(tokenizer_state));
        let start_final = constraint.parser_dwa.states()[constraint.parser_dwa.start_state() as usize]
            .final_weight.clone().unwrap_or_else(Weight::empty);
        for &token in tokens {
            let mut first = contains(constraint, tokenizer_state, token, &start_final).then_some(0usize);
            for depth in 1..=reversed.len() {
                let w = accepted_weight(constraint, &reversed[..depth]);
                if contains(constraint, tokenizer_state, token, &w) {
                    first = Some(depth);
                    break;
                }
            }
            eprintln!("SHALLOW {name} token={token} first_depth={first:?}");
        }
    }

    let full_path = std::env::var("GLRMASK_DEBUG_FULL_ARTIFACT").unwrap();
    let full = Constraint::load(&std::fs::read(full_path).unwrap()).unwrap();
    dump_case(&full, "tools_reset_a", 0, &[0,22,198,18], &[739,2446,7255,21966]);
    dump_case(&full, "tools_reset_b", 0, &[0,22,198,167], &[739,2446,7255,21966]);
    dump_case(&full, "tools_residual", 444, &[0,22,198], &[739,2446,7255,21966]);
    dump_case(&full, "tool0", 0, &[0,22,198,193,862], &[49209,69906]);
    dump_case(&full, "inside", 0, &[0,22,198,193,873,903], &[3033,3602,4649,9000,14419,16638,17041,29448,31893,32988,35183,36199,39942,44160,53511,71741,79237,82274,95445]);
    dump_case(&full, "await_tools", 0, &[0,22,198,109,193], &[3324,32809]);
}

#[test]
#[ignore]
fn debug_selected10_iterated_mask_factor() {
    use std::time::Instant;

    let fast_path = std::env::var("GLRMASK_DEBUG_ARTIFACT").unwrap();
    let full_path = std::env::var("GLRMASK_DEBUG_FULL_ARTIFACT").unwrap();
    let fast = Constraint::load(&std::fs::read(fast_path).unwrap()).unwrap();
    let full = Constraint::load(&std::fs::read(full_path).unwrap()).unwrap();
    unsafe {
        std::env::set_var("GLRMASK_EXPERIMENT_MASK_LOOKAHEAD_FACTOR", "1");
        std::env::set_var("GLRMASK_EXPERIMENT_MASK_LOOKAHEAD_FACTOR_MAX_DEPTH", "2");
        std::env::set_var("GLRMASK_EXPERIMENT_SCOPED_IGNORE_EXACT_OVERLAY", "1");
        std::env::remove_var("GLRMASK_EXPERIMENT_STATIC_DYNAMIC_OVERLAY");
    }
    let prefixes: &[&[u8]] = &[
        b"",
        b"const x",
        b"const x =",
        b"const x = tools",
        b"const x = tools.tool_0",
        b"const x = tools.tool_0({",
        b"const x = tools.tool_0({})",
        b"const x = tools.tool_0({});",
        b"const r = await tools.",
    ];
    for &prefix in prefixes {
        let mut fs = full.start();
        let mut hs = fast.start();
        if !prefix.is_empty() {
            fs.commit_bytes(prefix).unwrap();
            hs.commit_bytes(prefix).unwrap();
        }
        let fm = fs.mask();
        let started = Instant::now();
        let hm = hs.mask();
        let elapsed = started.elapsed().as_micros();
        let mut extra = 0usize;
        let mut missing = 0usize;
        for (&left, &right) in hm.iter().zip(&fm) {
            extra += (left & !right).count_ones() as usize;
            missing += (right & !left).count_ones() as usize;
        }
        eprintln!(
            "IGNORE_PM prefix={:?} us={} extra={} missing={}",
            String::from_utf8_lossy(prefix), elapsed, extra, missing
        );
    }
    std::mem::forget(fast);
    std::mem::forget(full);
}

#[test]
fn language_queue_structural_gate_rejects_tiny_and_accepts_wide_complex_gss() {
    let mut tiny = ParserStateMap::default();
    tiny.insert(
        0,
        ParserGSS::from_single_stack(
            vec![0_u32, 1, 2],
            TerminalsDisallowed::new(),
        ),
    );
    assert!(
        language_queue_top_value_count_at_most(&tiny, LANGUAGE_QUEUE_MIN_TOP_VALUES)
            < LANGUAGE_QUEUE_MIN_TOP_VALUES
    );

    let stacks = (0_u32..64)
        .map(|index| {
            (
                vec![0, 1_000 + index, 2_000 + index, 10 + index % 4],
                TerminalsDisallowed::new(),
            )
        })
        .collect::<Vec<_>>();
    let mut complex = ParserStateMap::default();
    complex.insert(0, ParserGSS::from_stacks(&stacks));
    assert_eq!(
        language_queue_top_value_count_at_most(
            &complex,
            LANGUAGE_QUEUE_MIN_TOP_VALUES,
        ),
        LANGUAGE_QUEUE_MIN_TOP_VALUES,
    );
    assert_eq!(
        language_queue_path_count_at_most(&complex, LANGUAGE_QUEUE_MIN_PATHS),
        LANGUAGE_QUEUE_MIN_PATHS,
    );
    assert_eq!(
        language_queue_node_count_at_most(&complex, LANGUAGE_QUEUE_MIN_NODES),
        LANGUAGE_QUEUE_MIN_NODES,
    );
}

#[test]
fn actionable_terminal_boundary_trigger_requires_distinct_offsets() {
    let constraint = Constraint::from_glrm_grammar(
        r#"
                start start;
                t A ::= "a";
                t AB ::= "ab";
                nt start ::= A | AB;
            "#,
        &Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"b".to_vec()),
        ]),
    )
    .expect("boundary-trigger grammar should compile");
    let state = constraint.start();
    let mut scratch = tokenizer_scan::ReusableTokenizerExecScratch::default();

    assert!(
        !has_multiple_actionable_terminal_boundaries(
            &constraint,
            &state.state,
            b"a",
            &mut scratch,
        ),
        "one actionable completion boundary must not select the language queue",
    );
    assert!(
        has_multiple_actionable_terminal_boundaries(
            &constraint,
            &state.state,
            b"ab",
            &mut scratch,
        ),
        "actionable terminal completions at byte offsets one and two must select",
    );
}

#[test]
fn actionable_terminal_boundary_trigger_ignores_same_offset_ambiguity() {
    let constraint = Constraint::from_glrm_grammar(
        r#"
                start start;
                t A ::= "a";
                t ALSO_A ::= "a";
                nt start ::= A | ALSO_A;
            "#,
        &Vocab::new(vec![(0, b"a".to_vec())]),
    )
    .expect("same-offset trigger grammar should compile");
    let state = constraint.start();
    let mut scratch = tokenizer_scan::ReusableTokenizerExecScratch::default();

    assert!(
        !has_multiple_actionable_terminal_boundaries(
            &constraint,
            &state.state,
            b"a",
            &mut scratch,
        ),
        "multiple actionable terminals ending at one offset must remain on the ordinary queue",
    );
}

#[test]
fn flat_stack_effect_can_replace_the_only_state() {
    let mut stack = FlatInlineStack::new();
    stack.push(36);

    assert_eq!(apply_flat_stack_effect(&mut stack, 1, &[37]), Some(true));
    assert_eq!(stack.as_slice(), &[37]);
}

#[test]
fn compile_time_initial_commit_priming_preserves_runtime_semantics() {
    let constraint = Constraint::from_glrm_grammar(
        r#"
                start start;
                t A ::= "a";
                t B ::= "b";
                nt start ::= A B;
            "#,
        &Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"b".to_vec()),
            (3, b"x".to_vec()),
        ]),
    )
    .unwrap();

    let mut state = constraint.start();
    assert_eq!(
        state
            .mask()
            .iter()
            .enumerate()
            .flat_map(|(word, &bits)| {
                (0..32).filter_map(move |bit| {
                    ((bits >> bit) & 1 == 1).then_some((word * 32 + bit) as u32)
                })
            })
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([0, 1]),
    );
    state.commit_token(0).unwrap();
    state.commit_token(2).unwrap();
    assert!(state.is_accepting());

    let loaded = Constraint::load(&constraint.save()).unwrap();
    let mut loaded_state = loaded.start();
    loaded_state.commit_token(1).unwrap();
    assert!(loaded_state.is_accepting());
}

#[test]
fn duplicate_key_flat_state_reports_completion_without_normalization() {
    let constraint = Constraint::from_glrm_grammar(
        r#"
                start start;
                t A ::= "a";
                t B ::= "a" | "ab";
                nt item ::= A | B;
                nt start ::= item;
            "#,
        &Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]),
    )
    .unwrap();

    let mut state = constraint.start();
    state.commit_token(0).unwrap();
    assert!(
        state.state.has_duplicate_keys(),
        "the bounded runtime should retain the completed alternatives separately"
    );
    assert!(state.is_accepting());

    let mut normalized = state.clone();
    normalized.state.normalize_duplicate_keys();
    assert!(normalized.is_accepting());
    assert_eq!(
        canonical_commit_state(&state.state),
        canonical_commit_state(&normalized.state),
    );
}

#[test]
fn special_token_advance_matches_normalized_duplicate_key_state() {
    let constraint = Constraint::from_glrm_grammar(
        r#"
                start start;
                t A ::= "a";
                t B ::= "a" | "ab";
                nt item ::= A | B;
                nt start ::= item @token(100);
            "#,
        &Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]),
    )
    .unwrap();

    let mut flat = constraint.start();
    flat.commit_token(0).unwrap();
    assert!(flat.state.has_duplicate_keys());

    let mut normalized = flat.clone();
    normalized.state.normalize_duplicate_keys();
    assert_eq!(flat.mask(), normalized.mask());

    flat.commit_token(100).unwrap();
    normalized.commit_token(100).unwrap();
    assert!(flat.is_accepting());
    assert!(normalized.is_accepting());
    assert_eq!(
        canonical_commit_state(&flat.state),
        canonical_commit_state(&normalized.state),
    );
}

#[test]
fn carried_virtual_stack_stops_when_stack_effect_exposes_branched_floor() {
    let acc = TerminalsDisallowed::new();
    let left = ParserGSS::from_single_stack(vec![0, 1, 10], acc.clone());
    let right = ParserGSS::from_single_stack(vec![0, 2, 10], acc);
    let merged = left.merge(&right);

    let mut exhausted = merged
        .try_virtual_stack()
        .expect("merged common top should form a virtual stack");
    assert!(!try_apply_action_to_carried_virtual_stack(
        &mut exhausted,
        &Action::StackShifts(vec![crate::compiler::glr::table::StackShift {
            pop: 1,
            pushes: Vec::new(),
        }]),
    ));
    assert_eq!(exhausted.top(), Some(&10));

    let mut restored_common_top = merged
        .try_virtual_stack()
        .expect("merged common top should form a virtual stack");
    assert!(try_apply_action_to_carried_virtual_stack(
        &mut restored_common_top,
        &Action::StackShifts(vec![crate::compiler::glr::table::StackShift {
            pop: 1,
            pushes: vec![40],
        }]),
    ));
    assert_eq!(restored_common_top.top(), Some(&40));
    let mut stacks = restored_common_top.into_gss().to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
    stacks.sort_by(|left, right| left.0.cmp(&right.0));
    assert_eq!(
        stacks.into_iter().map(|(stack, _)| stack).collect::<Vec<_>>(),
        vec![vec![0, 1, 40], vec![0, 2, 40]],
    );

    let mut emptied = ParserGSS::from_single_stack(
        vec![10],
        TerminalsDisallowed::new(),
    )
    .try_virtual_stack()
    .expect("single stack should form a virtual stack");
    assert!(!try_apply_action_to_carried_virtual_stack(
        &mut emptied,
        &Action::StackShifts(vec![crate::compiler::glr::table::StackShift {
            pop: 1,
            pushes: Vec::new(),
        }]),
    ));
    assert_eq!(emptied.top(), Some(&10));
}

fn canonical_gss(gss: &ParserGSS) -> Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)> {
    canonical_commit_state(&ParserStateMap::singleton(0, gss.clone()))
        .pop()
        .unwrap()
        .1
}

#[test]
fn fused_exact_admission_matches_legacy_two_pass_reference() {
    let vocab = Vocab::new(vec![
        (0, b"a".to_vec()),
        (1, b"b".to_vec()),
        (2, b"ab".to_vec()),
    ]);
    let constraint = Constraint::from_glrm_grammar(
        r#"
                start start;
                t A ::= "a";
                t AB ::= "ab";
                t B ::= "b";
                nt item ::= A | AB | A B;
                nt start ::= item item?;
            "#,
        &vocab,
    )
    .unwrap();
    assert_eq!(
        constraint.table.admission_policy,
        AdmissionPolicy::ExactSimulation,
    );

    let mut states = vec![constraint.start()];
    let mut after_a = constraint.start();
    after_a.commit_token(0).unwrap();
    states.push(after_a);

    for state in states {
        for gss in state.state.values() {
            for terminal in 0..constraint.table.num_terminals {
                let legacy = if stack_may_advance_on(&constraint.table, gss, terminal) {
                    let advanced = advance_parser_stacks(&constraint, gss, terminal);
                    (!advanced.is_empty()).then_some(advanced)
                } else {
                    None
                };
                let fused = advance_parser_stacks_if_possible(&constraint, gss, terminal);
                assert_eq!(
                    fused.as_ref().map(canonical_gss),
                    legacy.as_ref().map(canonical_gss),
                    "terminal={terminal} source={:#?}",
                    canonical_gss(gss),
                );

                let profiled =
                    advance_parser_stacks_profiled_if_possible(&constraint, gss, terminal);
                assert_eq!(
                    (!profiled.advanced.is_empty())
                        .then(|| canonical_gss(&profiled.advanced)),
                    legacy.as_ref().map(canonical_gss),
                    "profiled terminal={terminal} source={:#?}",
                    canonical_gss(gss),
                );
                assert_eq!(profiled.may_ns, 0);
            }
        }
    }
}

fn top_local_prune_reference(
    constraint: &Constraint,
    gss: &ParserGSS,
    bytes: &[u8],
) -> ParserGSS {
    let prune_partition = |partition: ParserGSS| {
        partition.apply_and_prune_no_promote(
            |terminals_disallowed: &TerminalsDisallowed| {
                if terminals_disallowed.is_empty() {
                    return Some(TerminalsDisallowed::new());
                }

                let mut remapped = BTreeMap::new();
                for (&continuation_tokenizer_state, disallowed) in
                    terminals_disallowed.iter()
                {
                    let execution = execute_tokenizer_from_state_small(
                        constraint,
                        bytes,
                        continuation_tokenizer_state,
                    );
                    if execution
                        .matches
                        .iter()
                        .any(|matched| disallowed.contains(&matched.id))
                    {
                        return None;
                    }
                    for end_state in execution.end_state {
                        let future = constraint
                            .tokenizer
                            .possible_future_terminals(end_state);
                        for &terminal in disallowed.iter() {
                            if future.contains(terminal as usize) {
                                remapped
                                    .entry(end_state)
                                    .or_insert_with(BTreeSet::new)
                                    .insert(terminal);
                            }
                        }
                    }
                }
                Some(TerminalsDisallowed::from_map(remapped))
            },
        )
    };

    let mut partitions = Vec::new();
    let root = gss.isolate(None);
    if !root.is_empty() {
        let root = prune_partition(root);
        if !root.is_empty() {
            partitions.push(root);
        }
    }
    for parser_state in gss.peek_values() {
        let partition = prune_partition(gss.isolate(Some(parser_state)));
        if !partition.is_empty() {
            partitions.push(partition);
        }
    }

    let mut iter = partitions.into_iter();
    let Some(mut merged) = iter.next() else {
        return ParserGSS::empty();
    };
    for partition in iter {
        merged = merged.merge(&partition);
    }
    merged
}

#[test]
fn initial_prune_advances_each_continuation_tokenizer_state() {
    let vocab = Vocab::new(
        vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
        ]);
    let grammar = r#"
start start;
t A ::= "a";
t B ::= "a" | "ab";
nt item ::= A | B;
nt start ::= item item? item?;
"#;
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_token(0).unwrap();

    let tokenizer_state = constraint.runtime_commit_initial_state();
    // This test exercises the GSS pruning primitive itself, so cross the
    // bounded flat-state boundary explicitly before inspecting one map value.
    state.state.normalize_duplicate_keys();
    let gss = state
        .state
        .get(&tokenizer_state)
        .expect("ambiguous a must retain a reset lexer branch");
    assert!(
        !gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()),
        "MRE requires a stale residual exclusion on the reset branch"
    );

    let exec_result =
        execute_tokenizer_from_state_small(&constraint, b"b", tokenizer_state);
    assert!(exec_result.matches.is_empty());
    assert!(exec_result.end_state.is_empty());

    let pruned = prune_single_initial_state_for_exec(
        &constraint,
        gss.clone(),
        tokenizer_state,
        &exec_result,
        b"b",
    );
    assert!(
        !pruned.is_empty()
            && pruned.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()),
        "the valid A=a branch must survive while the provisional B=a branch is invalidated by B=ab"
    );
}

#[test]
fn delayed_exclusion_survives_across_model_token_boundaries() {
    let vocab = Vocab::new(vec![
        (0, b"@".to_vec()),
        (1, b" double".to_vec()),
        (2, b"Quote".to_vec()),
        (3, b"x".to_vec()),
        (4, b"=".to_vec()),
        (5, b"%".to_vec()),
        (6, b"0".to_vec()),
    ]);
    let constraint = Constraint::from_glrm_grammar(
        r#"
                start start;
                ignore WS;
                nt start ::= declaration expression;
                nt declaration ::= "@" ID;
                nt expression ::= ID OP "0";
                t WS ::= [ \t\r\n]+;
                t OP ::= "=" | "%=";
                t ID ::= [A-Za-z_$] [A-Za-z0-9_$]*;
            "#,
        &vocab,
    )
    .unwrap();

    let mut state = constraint.start();
    state.commit_token(0).unwrap();
    state.commit_token(1).unwrap();
    state.commit_token(2).unwrap();

    let mask = state.mask();
    let allowed = |token: u32| {
        ((mask[token as usize / 32] >> (token % 32)) & 1) != 0
    };
    assert!(allowed(3), "continuing the current ID remains viable");
    assert!(!allowed(4), "the consumed ID cannot also satisfy the next ID");
    assert!(!allowed(5), "a prefix of %= cannot follow until a new ID is consumed");
    let mut probe = state.clone();
    assert!(probe.commit_token(4).is_err());
    let mut probe = state.clone();
    assert!(probe.commit_token(5).is_err());
}

#[test]
fn recursive_delayed_exclusion_stays_in_leaf_tokenizer_terminal_coordinate() {
    let vocab = Vocab::new(vec![
        (0, b"@".to_vec()),
        (1, b" double".to_vec()),
        (2, b"Quote".to_vec()),
        (3, b"x".to_vec()),
        (4, b"=".to_vec()),
        (5, b"%".to_vec()),
        (6, b"0".to_vec()),
    ]);
    let body = Constraint::from_glrm_grammar(
        r#"
                start start;
                ignore WS;
                nt start ::= declaration expression;
                nt declaration ::= "@" ID;
                nt expression ::= ID OP "0";
                t WS ::= [ \t\r\n]+;
                t OP ::= "=" | "%=";
                t ID ::= [A-Za-z_$] [A-Za-z0-9_$]*;
            "#,
        &vocab,
    )
    .unwrap();
    let parent = Constraint::compile(
        Grammar::glrm("glrm 1; start document; extern grammar body; nt document = body;"),
        &vocab,
    )
    .unwrap();
    let bound = parent
        .bind_grammar_dynamic_boundary("body", body.clone())
        .unwrap();
    assert!(bound.uses_compact_segmented_parser_runtime());
    let loaded = Constraint::load(bound.save()).unwrap();

    let mut expected = body.start();
    for token in [0, 1, 2] {
        expected.commit_token(token).unwrap();
    }
    let expected_mask = expected.mask();
    for constraint in [&bound, &loaded] {
        let mut state = constraint.start();
        for token in [0, 1, 2] {
            state.commit_token(token).unwrap();
        }
        assert_eq!(state.mask(), expected_mask);
        assert!(state.state.iter().any(|(_, gss)| {
            !gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty())
        }));
        let mut probe = state.clone();
        assert!(probe.commit_token(4).is_err());
        let mut probe = state.clone();
        assert!(probe.commit_token(5).is_err());
    }
}

#[test]
fn initial_prune_matches_top_local_reference_on_generated_small_languages() {
    const WORDS: [&str; 4] = ["a", "b", "ab", "ba"];
    let vocab = Vocab::new(
        [
            "a", "b", "ab", "ba", " ", " a", "a ", " b", "b ", " a ", " b ",
        ]
        .into_iter()
        .enumerate()
        .map(|(id, word)| (id as u32, word.as_bytes().to_vec()))
        .collect());
    let languages = (1u32..1u32 << WORDS.len())
        .filter(|mask| mask.count_ones() <= 2)
        .collect::<Vec<_>>();
    let rule = |name: &str, mask: u32| {
        let rhs = WORDS
            .iter()
            .enumerate()
            .filter_map(|(index, word)| {
                (mask & (1 << index) != 0).then(|| format!("\"{word}\""))
            })
            .collect::<Vec<_>>()
            .join(" | ");
        format!("t {name} ::= {rhs};\n")
    };

    for &a in &languages {
        for &b in &languages {
            for start_rule in [
                "nt item ::= A | B;\nnt start ::= item item? item?;",
                "nt start ::= A A | B B;",
                "nt start ::= A B | B A;",
            ] {
                let grammar = format!(
                    "start start;\nignore WS;\nt WS ::= \" \"+;\n{}{}{start_rule}\n",
                    rule("A", a),
                    rule("B", b),
                );
                let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
                let mut frontier = vec![(constraint.start(), Vec::<u32>::new())];
                let mut seen = BTreeSet::new();

                for depth in 0..=4 {
                    let mut next = Vec::new();
                    for (state, path) in frontier {
                        let state_key = canonical_commit_state(&state.state);
                        if !seen.insert(state_key) {
                            continue;
                        }
                        for (token_id, bytes) in constraint.token_bytes_iter() {
                            for (&tokenizer_state, gss) in &state.state {
                                if gss
                                    .all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty())
                                {
                                    continue;
                                }
                                let exec_result = execute_tokenizer_from_state_small(
                                    &constraint,
                                    bytes,
                                    tokenizer_state,
                                );
                                let actual = prune_single_initial_state_for_exec(
                                    &constraint,
                                    gss.clone(),
                                    tokenizer_state,
                                    &exec_result,
                                    bytes,
                                );
                                let expected = top_local_prune_reference(
                                    &constraint,
                                    gss,
                                    bytes,
                                );
                                assert_eq!(
                                    canonical_gss(&actual),
                                    canonical_gss(&expected),
                                    "initial prune crossed parser-top correlations: A={a:#06b} B={b:#06b} depth={depth} path={path:?} token={token_id} bytes={bytes:?} tokenizer_state={tokenizer_state}\ngrammar:\n{grammar}\nsource={:#?}\nactual={:#?}\nexpected={:#?}",
                                    canonical_gss(gss),
                                    canonical_gss(&actual),
                                    canonical_gss(&expected),
                                );
                            }

                            if depth < 4 {
                                let mut advanced = state.clone();
                                if advanced.commit_bytes(bytes).is_ok() {
                                    let mut next_path = path.clone();
                                    next_path.push(token_id);
                                    next.push((advanced, next_path));
                                }
                            }
                        }
                    }
                    frontier = next;
                }
            }
        }
    }
}

#[test]
fn rejected_public_commits_enter_fail_state() {
    let vocab = Vocab::new(
        vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
    let constraint = Constraint::from_glrm_grammar(
        r#"
                start start;
                t A ::= "a";
                nt start ::= A;
            "#,
        &vocab,
    )
    .unwrap();

    let assert_failed = |state: &ConstraintState<'_>| {
        assert!(state.state.is_empty());
        assert!(state.mask().iter().all(|&word| word == 0));
    };

    let mut state = constraint.start();
    assert!(state.commit_token(1).is_err());
    assert_failed(&state);

    let mut state = constraint.start();
    assert!(state.commit_token_timed_ns(1).is_err());
    assert_failed(&state);

    let mut state = constraint.start();
    assert!(state.commit_token_profiled(1).is_err());
    assert_failed(&state);

    let mut state = constraint.start();
    assert!(state.commit_token_per_advance(1).is_err());
    assert_failed(&state);

    let mut state = constraint.start();
    assert!(state.commit_bytes(b"b").is_err());
    assert_failed(&state);
}

fn assert_fast_and_general_queue_match<'a>(
    constraint: &'a Constraint,
    fast_state: &ConstraintState<'a>,
    token_id: u32,
    bytes: &[u8],
    context: &str,
) -> Option<ConstraintState<'a>> {
    let mut fast = fast_state.clone();
    let mut profiled = fast_state.clone();
    let mut general = fast_state.clone();

    let fast_result = commit_bytes_impl(
        constraint,
        &mut fast.state,
        bytes,
        &mut fast.buffers,
    );
    let profiled_result = commit_bytes_impl_profiled(
        constraint,
        &mut profiled.state,
        bytes,
        &mut profiled.buffers,
        None,
        true,
    );
    let general_result = commit_bytes_impl_profiled(
        constraint,
        &mut general.state,
        bytes,
        &mut general.buffers,
        None,
        false,
    );

    assert_eq!(
        fast_result.is_ok(),
        general_result.is_ok(),
        "commit result mismatch: {context} token_id={token_id} bytes={bytes:?}\nfast={:?}\ngeneral={:?}",
        fast.state,
        general.state,
    );
    assert_eq!(
        profiled_result.is_ok(),
        general_result.is_ok(),
        "profiled commit result mismatch: {context} token_id={token_id} bytes={bytes:?}\nprofiled={:?}\ngeneral={:?}",
        profiled.state,
        general.state,
    );
    if fast_result.is_err() {
        return None;
    }
    assert_eq!(
        canonical_commit_state(&fast.state),
        canonical_commit_state(&general.state),
        "successful commit state mismatch: {context} token_id={token_id} bytes={bytes:?}\nfast_stacks={:#?}\ngeneral_stacks={:#?}",
        fast.state
            .iter()
            .map(|(&ts, gss)| (ts, gss.to_stacks(4_096).expect("stack enumeration exceeded explicit limit")))
            .collect::<Vec<_>>(),
        general
            .state
            .iter()
            .map(|(&ts, gss)| (ts, gss.to_stacks(4_096).expect("stack enumeration exceeded explicit limit")))
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        canonical_commit_state(&profiled.state),
        canonical_commit_state(&general.state),
        "successful profiled commit state mismatch: {context} token_id={token_id} bytes={bytes:?}\nprofiled_stacks={:#?}\ngeneral_stacks={:#?}",
        profiled
            .state
            .iter()
            .map(|(&ts, gss)| (ts, gss.to_stacks(4_096).expect("stack enumeration exceeded explicit limit")))
            .collect::<Vec<_>>(),
        general
            .state
            .iter()
            .map(|(&ts, gss)| (ts, gss.to_stacks(4_096).expect("stack enumeration exceeded explicit limit")))
            .collect::<Vec<_>>(),
    );

    Some(fast)
}

#[test]
fn flat_frontier_accepts_input_past_the_inline_state_capacity() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec())]);
    let constraint = Constraint::from_glrm_grammar(
        r#"
                start start;
                t A ::= "a";
                nt start ::= A;
            "#,
        &vocab,
    )
    .expect("single-terminal grammar should compile");
    let start = constraint.start();
    let (tokenizer_state, parser_gss) = start.state.entries[0].clone();
    let mut state = ParserStateMap::default();
    for _ in 0..=INLINE_PARSER_STATE_CAPACITY {
        state.insert_flat_alternative(tokenizer_state, parser_gss.clone());
    }
    assert_eq!(state.len(), INLINE_PARSER_STATE_CAPACITY + 1);

    let mut original = Vec::with_capacity(LINEAR_STACK_RESERVE);
    let mut work = Vec::with_capacity(LINEAR_STACK_RESERVE);
    let mut tokenizer_scratch = tokenizer_scan::ReusableTokenizerExecScratch::default();
    let mut frontier = FlatFrontierScratch::default();
    let result = try_commit_flat_frontier_in_place(
        &constraint,
        &mut state,
        b"a",
        &mut original,
        &mut work,
        &mut tokenizer_scratch,
        &mut frontier,
    );
    assert!(matches!(result, Some(Ok(()))));
    assert!(!state.is_empty());
}

#[test]
fn flat_frontier_preserves_all_stacks_in_a_live_lexer_continuation_group() {
    let vocab = Vocab::new(vec![
        (0, b"a".to_vec()),
        (1, b"b".to_vec()),
        (2, b"ab".to_vec()),
        (3, b"ba".to_vec()),
    ]);
    let grammar = r#"
            start start;
            t A ::= "a";
            t B ::= "b" | "ab";
            nt item ::= A | B;
            nt start ::= item item? item?;
        "#;
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
    let start = constraint.start();
    let after_ab = assert_fast_and_general_queue_match(
        &constraint,
        &start,
        2,
        b"ab",
        "grouped lexer continuation regression after ab",
    )
    .unwrap();
    let after_ba = assert_fast_and_general_queue_match(
        &constraint,
        &after_ab,
        3,
        b"ba",
        "grouped lexer continuation regression after ab, ba",
    )
    .unwrap();

    let stacks = canonical_commit_state(&after_ba.state);
    let tokenizer_one = stacks
        .iter()
        .find(|(tokenizer_state, _)| *tokenizer_state == 1)
        .expect("lexer continuation state must remain live");
    assert_eq!(
        tokenizer_one.1,
        vec![
            (vec![0, 3, 2], Vec::new()),
            (vec![0, 3, 5, 2], Vec::new()),
        ],
    );
}

#[test]
fn monolithic_commit_fast_paths_match_general_queue_on_small_language_space() {
    const WORDS: [&str; 4] = ["a", "b", "ab", "ba"];
    let vocab = Vocab::new(
        WORDS
            .iter()
            .enumerate()
            .map(|(id, word)| (id as u32, word.as_bytes().to_vec()))
            .collect());
    let languages = (1u32..1u32 << WORDS.len())
        .filter(|mask| mask.count_ones() <= 2)
        .collect::<Vec<_>>();

    let rule = |name: &str, mask: u32| {
        let rhs = WORDS
            .iter()
            .enumerate()
            .filter_map(|(index, word)| {
                (mask & (1 << index) != 0).then(|| format!("\"{word}\""))
            })
            .collect::<Vec<_>>()
            .join(" | ");
        format!("t {name} ::= {rhs};\n")
    };

    for &a in &languages {
        for &b in &languages {
            let grammar = format!(
                "start start;\n{}{}nt item ::= A | B;\nnt start ::= item item? item?;\n",
                rule("A", a),
                rule("B", b),
            );
            let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
            let mut frontier = vec![(constraint.start(), Vec::<u32>::new())];

            for depth in 0..3 {
                let mut next = Vec::new();
                for (state, path) in frontier {
                    let mask = state.mask();
                    for (&token_id, bytes) in vocab.entries_map().iter() {
                        let context = format!(
                            "A_mask={a:#06b} B_mask={b:#06b} depth={depth} path={path:?}\ngrammar:\n{grammar}"
                        );
                        let next_state = assert_fast_and_general_queue_match(
                            &constraint,
                            &state,
                            token_id,
                            bytes,
                            &context,
                        );
                        let token_in_mask = mask
                            .get(token_id as usize / 32)
                            .is_some_and(|word| {
                                word & (1u32 << (token_id % 32)) != 0
                            });
                        assert_eq!(
                            token_in_mask,
                            next_state.is_some(),
                            "mask/commit mismatch: {context} token_id={token_id} bytes={bytes:?}"
                        );
                        if let Some(next_state) = next_state {
                            let mut next_path = path.clone();
                            next_path.push(token_id);
                            next.push((next_state, next_path));
                        }
                    }
                }
                frontier = next;
            }
        }
    }
}

#[test]
fn residual_bc_fast_path_matches_general_queue() {
    let vocab = Vocab::new(
        vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"c".to_vec()),
            (3, b"ab".to_vec()),
            (4, b"ba".to_vec()),
            (5, b"bc".to_vec()),
            (6, b"abc".to_vec()),
        ]);
    let constraint = Constraint::from_glrm_grammar(
        r#"
                start start;
                t A ::= "a" | "ab";
                t B ::= "bc";
                nt item ::= A | B;
                nt start ::= item item? item?;
            "#,
        &vocab,
    )
    .unwrap();

    let mut fast = constraint.start();
    let mut slow = constraint.start();
    let fast_result = commit_bytes_impl(
        &constraint,
        &mut fast.state,
        vocab.entries_map().get(&0).unwrap(),
        &mut fast.buffers,
    );
    fast.generation += 1;
    assert!(fast_result.is_ok());
    let slow_result = commit_bytes_impl_profiled(
        &constraint,
        &mut slow.state,
        vocab.entries_map().get(&0).unwrap(),
        &mut slow.buffers,
        None,
        false,
    );
    slow.generation += 1;
    assert!(slow_result.is_ok());
    assert_eq!(fast.state, slow.state, "state mismatch after token a");

    let mut next_fast = fast.clone();
    let mut next_slow = slow.clone();
    let fast_result = commit_bytes_impl(
        &constraint,
        &mut next_fast.state,
        vocab.entries_map().get(&5).unwrap(),
        &mut next_fast.buffers,
    );
    let slow_result = commit_bytes_impl_profiled(
        &constraint,
        &mut next_slow.state,
        vocab.entries_map().get(&5).unwrap(),
        &mut next_slow.buffers,
        None,
        false,
    );
    assert_eq!(
        fast_result.is_ok(),
        slow_result.is_ok(),
        "fast={:?}\nslow={:?}",
        next_fast.state,
        next_slow.state,
    );
    if fast_result.is_ok() {
        assert_eq!(next_fast.state, next_slow.state);
    }
}

#[test]
fn epsilon_commit_fast_paths_match_no_fast_path_reference() {
    let vocab = Vocab::new(
        vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"c".to_vec()),
            (3, b"aa".to_vec()),
            (4, b"ab".to_vec()),
            (5, b" ".to_vec()),
            (6, b" a".to_vec()),
            (7, b"a ".to_vec()),
            (8, b" a ".to_vec()),
            (9, b"abc".to_vec()),
            (10, b"aab".to_vec()),
        ]);
    let grammar = crate::grammar::glrm::from_glrm(
        r#"
                start start;
                ignore WS;
                lexer group ws ::= WS;
                lexer group a ::= A;
                lexer group b ::= B;
                lexer group c ::= C;
                t WS ::= " "+;
                t A ::= "a"+;
                t B ::= "b";
                t C ::= "c";
                nt item ::= A | B | C;
                nt start ::= item item? item?;
            "#,
    )
    .unwrap();
    let grammar = crate::grammar::ast::lower(&grammar).unwrap();
    let constraint = crate::compiler::pipeline::compile_owned_with_lexer_adaptive(
        grammar,
        &vocab,
        false,
    );
    assert!(constraint.tokenizer_has_epsilon_transitions);

    let mut frontier = vec![(
        constraint.start(),
        constraint.start(),
        constraint.start(),
        Vec::<u32>::new(),
    )];
    for depth in 0..=4 {
        let mut next = Vec::new();
        for (fast, profiled, general, path) in frontier {
            assert_eq!(
                fast.mask(),
                general.mask(),
                "epsilon mask mismatch after path {path:?}\nfast={:#?}\ngeneral={:#?}",
                canonical_commit_state(&fast.state),
                canonical_commit_state(&general.state),
            );
            assert_eq!(
                profiled.mask(),
                general.mask(),
                "epsilon profiled-mask mismatch after path {path:?}\nprofiled={:#?}\ngeneral={:#?}",
                canonical_commit_state(&profiled.state),
                canonical_commit_state(&general.state),
            );
            assert_eq!(
                fast.is_accepting(),
                general.is_accepting(),
                "epsilon completion mismatch after path {path:?}",
            );
            assert_eq!(
                profiled.is_accepting(),
                general.is_accepting(),
                "epsilon profiled completion mismatch after path {path:?}",
            );
            if depth == 4 {
                continue;
            }

            for (&token_id, bytes) in vocab.entries_map().iter() {
                let mut next_fast = fast.clone();
                let mut next_profiled = profiled.clone();
                let mut next_general = general.clone();
                let fast_result = commit_bytes_impl(
                    &constraint,
                    &mut next_fast.state,
                    bytes,
                    &mut next_fast.buffers,
                );
                let profiled_result = commit_bytes_impl_profiled(
                    &constraint,
                    &mut next_profiled.state,
                    bytes,
                    &mut next_profiled.buffers,
                    None,
                    true,
                );
                let general_result = commit_bytes_impl_profiled(
                    &constraint,
                    &mut next_general.state,
                    bytes,
                    &mut next_general.buffers,
                    None,
                    false,
                );
                assert_eq!(
                    fast_result.is_ok(),
                    general_result.is_ok(),
                    "epsilon commit result mismatch after path {path:?} token_id={token_id} bytes={bytes:?}\nfast={:#?}\ngeneral={:#?}",
                    canonical_commit_state(&next_fast.state),
                    canonical_commit_state(&next_general.state),
                );
                assert_eq!(
                    profiled_result.is_ok(),
                    general_result.is_ok(),
                    "epsilon profiled commit result mismatch after path {path:?} token_id={token_id} bytes={bytes:?}\nprofiled={:#?}\ngeneral={:#?}",
                    canonical_commit_state(&next_profiled.state),
                    canonical_commit_state(&next_general.state),
                );
                if fast_result.is_ok() {
                    let mut next_path = path.clone();
                    next_path.push(token_id);
                    next.push((next_fast, next_profiled, next_general, next_path));
                }
            }
        }
        frontier = next;
    }
}

#[test]
fn epsilon_full_width_terminal_with_empty_accumulators_uses_fast_path() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
    let grammar = crate::grammar::glrm::from_glrm(
        r#"
                start start;
                lexer group left ::= A;
                lexer group right ::= B;
                t A ::= "a";
                t B ::= "b";
                nt start ::= A | B;
            "#,
    )
    .unwrap();
    let grammar = crate::grammar::ast::lower(&grammar).unwrap();
    let constraint = crate::compiler::pipeline::compile_owned_with_lexer_adaptive(
        grammar,
        &vocab,
        false,
    );

    let mut state = constraint.start();
    let profile = state.commit_token_profiled(0).unwrap();
    assert!(profile.fast_path_total_ns > 0, "profile={profile:?}");
    assert_eq!(profile.n_queue_entries, 0, "profile={profile:?}");
    assert_eq!(profile.n_advances, 1, "profile={profile:?}");
}

#[test]
fn reduce_chain_output_factoring_preserves_stacks_and_exclusions() {
    let plain = TerminalsDisallowed::new();
    let guarded = plain.clone().with_insert(7, 3);
    for depth in [1usize, 2, 14, 64, 256] {
        let prefix: Vec<u32> = (0..depth as u32).collect();
        let extend = |suffix: &[u32]| {
            let mut stack = prefix.clone();
            stack.extend_from_slice(suffix);
            stack
        };
        let cases = vec![
            vec![],
            vec![(prefix.clone(), plain.clone())],
            vec![(prefix.clone(), plain.clone()), (prefix.clone(), plain.clone())],
            vec![(extend(&[500]), plain.clone()), (extend(&[501]), plain.clone())],
            vec![(extend(&[500, 502]), plain.clone()), (extend(&[501, 502]), plain.clone())],
            vec![(prefix.clone(), plain.clone()), (extend(&[501]), plain.clone())],
            vec![(vec![], plain.clone()), (prefix.clone(), plain.clone())],
            vec![(prefix.clone(), plain.clone()), (prefix.clone(), guarded.clone())],
            vec![(extend(&[500]), guarded.clone()), (extend(&[501]), plain.clone())],
        ];
        for outputs in cases {
            let reference = ParserGSS::from_stacks(&outputs);
            let actual = materialize_reduce_chain_outputs(outputs);
            assert!(actual.semantically_eq(&reference, 32).unwrap(), "depth={depth}");
        }
    }
}

#[test]
fn unique_actionable_top_matches_isolation_reference_including_empty_paths() {
    fn reference(constraint: &Constraint, gss: &ParserGSS, terminal: u32) -> Option<ParserGSS> {
        if !constraint.table.control_terminals.is_empty() || template_advance_enabled() {
            return None;
        }
        let mut selected = None;
        for top in gss.peek_values() {
            let Some(action) = constraint.table.action(top, terminal) else { continue; };
            if selected.is_some() { return None; }
            selected = Some((top, action));
        }
        let (top, action) = selected?;
        let isolated = gss.isolate(Some(top));
        (!isolated.is_empty())
            .then(|| apply_single_top_action_fast(constraint, &isolated, top, terminal, action))
            .flatten()
    }
    let vocab = Vocab::new(["a", "b", "ab", "ba", ",", " ", "(", ")"]
        .into_iter().enumerate().map(|(i, token)| (i as u32, token.as_bytes().to_vec())).collect());
    let grammars = [
        r#"start start; t A ::= "a" | "ab"; t B ::= "a" | "ba";
                nt start ::= A B? | B A?;"#,
        r#"start start; ignore WS; t WS ::= " "+; t A ::= "a"+; t B ::= "a"+ "b"?;
                nt item ::= A | B; nt start ::= item ("," item)*;"#,
        r#"start start; t A ::= "a"; nt start ::= A? | "(" start ")";"#,
    ];
    let empty = ParserGSS::from_single_stack(Vec::new(), TerminalsDisallowed::new());
    let mut exclusive = 0usize;
    let mut with_empty = 0usize;
    let mut comparisons = 0usize;
    for source in grammars {
        let built = Constraint::compile(Grammar::glrm(source), &vocab).unwrap();
        let loaded = Constraint::load(built.save()).unwrap();
        for constraint in [&built, &loaded] {
            let mut frontier = vec![constraint.start()];
            let mut seen = BTreeSet::new();
            for depth in 0..=3 {
                let mut next = Vec::new();
                for state in frontier {
                    if !seen.insert(canonical_commit_state(&state.state)) { continue; }
                    for gss in state.state.values() {
                        for input in [gss.clone(), gss.merge(&empty)] {
                            if let Some(top) = input.single_exclusive_top_value() {
                                exclusive += 1;
                                assert!(input.ptr_eq(&input.isolate(Some(top))));
                            } else {
                                with_empty += 1;
                            }
                            for terminal in 0..constraint.table.num_terminals {
                                let expected = reference(constraint, &input, terminal);
                                let actual = try_advance_unique_actionable_top_fast(constraint, &input, terminal);
                                assert_eq!(actual.as_ref().map(canonical_gss), expected.as_ref().map(canonical_gss),
                                    "terminal={terminal} depth={depth} source={source}");
                                comparisons += 1;
                            }
                        }
                    }
                    if depth < 3 {
                        let mask = state.mask();
                        for (token, _) in constraint.token_bytes_iter() {
                            if !token_in_mask(&mask, token) { continue; }
                            let mut advanced = state.clone();
                            advanced.commit_token(token).unwrap();
                            next.push(advanced);
                        }
                    }
                }
                frontier = next;
            }
        }
    }
    assert!(exclusive > 0 && with_empty > 0 && comparisons > 20);
}
