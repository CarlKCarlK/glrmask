#[test]
fn retained_template_relocation_matches_fresh_characterization() {
    let mut mixed = identity_transfer(5);
    mixed.escapes.push(InitialEscape {
        pop: vec![StackMatcher::States(vec![1, 3]), StackMatcher::Any],
        pushes: vec![4, 2],
    });
    mixed.escapes.push(InitialEscape {
        pop: vec![StackMatcher::Any], pushes: vec![0],
    });
    let mut chained = empty_transfer();
    chained.reduces.push(InitialReduce {
        pop: vec![StackMatcher::State(2)], nonterminal: 0,
    });
    chained.nt_escapes.push(NtEscape {
        source_nonterminal: 0,
        pop: vec![StackMatcher::Any], pushes: vec![3],
    });
    chained.all_nts.insert(0);
    for local in [empty_transfer(), identity_transfer(5), mixed, chained] {
        let original = Templates::from_characterizations(&BTreeMap::from([(0, local.clone())]));
        for offset in [0, 7, 1031] {
            let injection = StateInjection { offset };
            let scoped = scope_characterization(&local, &injection).unwrap();
            let fresh = Templates::from_characterizations(&BTreeMap::from([(0, scoped)]));
            let relation = (0..5).map(|state| vec![state + offset]).collect::<Vec<_>>();
            let (relocated, skeleton) =
                crate::compiler::composition::transport_composition_template_dfa_with_skeleton(
                    original.by_terminal[&0].clone(), &relation,
                ).unwrap();
            assert_same_template_language(&relocated, &fresh.by_terminal[&0]).unwrap();
            let rebuilt = Templates::from_terminal_dfas(BTreeMap::from([(0, relocated)]));
            assert_eq!(skeleton.start_states(), rebuilt.by_terminal_nwa[&0].start_states());
            assert_eq!(skeleton.states(), rebuilt.by_terminal_nwa[&0].states());
        }
    }
}

#[test]
fn retained_template_comparator_does_not_compare_empty_skeletons() {
    let mut left = UnweightedDfa::new();
    let end = left.add_state();
    left.add_transition(left.start_state, 3, end);
    left.set_accepting(end, true);
    let right = UnweightedDfa::new();
    assert!(assert_same_template_language(&left, &right).is_err());
}

use super::*;

#[test]
fn positive_boundary_nwa_hashcons_preserves_exact_weighted_language() {
    use crate::automata::weighted_u32::{determinize::determinize, equivalence::find_difference};
    use crate::compiler::stages::id_map_and_terminal_dwa::l2p::canonicalize_acyclic_nwa;

    let weights = vec![
        Weight::all(), Weight::empty(),
        Weight::from_per_tsid_token_sets([(0, RangeSetBlaze::from_iter([0..=2]))]),
        Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([1..=3])),
            (2, RangeSetBlaze::from_iter([7..=7])),
        ]),
    ];
    let mut random = 19u64;
    let mut next = || {
        random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
        (random >> 32) as usize
    };
    let mut reductions = 0usize;
    for case in 0..256 {
        let n = 5 + next() % 7;
        let mut original = NWA::new(3, 8);
        for _ in 0..n { original.add_state(); }
        original.set_start_states(vec![0]);
        // Two distinct but exactly equal leaves ensure suffix sharing is
        // exercised, including nondeterministic edges into both copies.
        original.set_final_weight((n - 2) as u32, weights[2].clone());
        original.set_final_weight((n - 1) as u32, weights[2].clone());
        original.add_transition(0, 7, (n - 2) as u32, weights[0].clone());
        original.add_transition(0, 7, (n - 1) as u32, weights[3].clone());
        for source in 0..n - 2 {
            if next() % 4 == 0 {
                original.set_final_weight(source as u32, weights[next() % weights.len()].clone());
            }
            for target in source + 1..n {
                let kind = next() % 7;
                let weight = weights[next() % weights.len()].clone();
                if kind == 0 {
                    original.add_epsilon(source as u32, target as u32, weight);
                } else if kind <= 3 {
                    original.add_transition(source as u32, kind as i32, target as u32, weight);
                }
            }
        }
        let mut reduced = original.clone();
        canonicalize_acyclic_nwa(&mut reduced);
        reductions += usize::from(reduced.num_states() < original.num_states());
        let baseline = determinize(&original).expect("generated DAG determinizes");
        let candidate = determinize(&reduced).expect("quotient DAG determinizes");
        assert_eq!(find_difference(&baseline, &candidate).unwrap(), None, "case {case}");
    }
    assert_eq!(reductions, 256, "all fixtures must exercise an actual quotient");
}

/// Memo must reproduce the exact reference on every input class.
#[test]
fn projection_memo_matches_reference() {
    use range_set_blaze::RangeSetBlaze;
    fn set(ranges: &[(u32, u32)]) -> RangeSetBlaze<u32> {
        ranges.iter().copied().map(|(s, e)| s..=e).collect()
    }
    fn weight(entries: &[(u32, &[(u32, u32)])]) -> Weight {
        Weight::from_per_tsid_token_sets(
            entries.iter().copied().map(|(tsid, rs)| (tsid, set(rs))),
        )
    }
    fn ranges_of(w: &Weight) -> Vec<(u32, u32, Vec<(u32, u32)>)> {
        w.raw_range_values()
            .map(|(r, t)| {
                (
                    *r.start(),
                    *r.end(),
                    t.ranges().map(|x| (*x.start(), *x.end())).collect(),
                )
            })
            .collect()
    }
    // kept pattern: alternating keep/drop over 0..32.
    let kept: Vec<bool> = (0..32).map(|i| i % 2 == 0).collect();
    let cases: Vec<Vec<(u32, &[(u32, u32)])>> = vec![
        vec![(0, &[(0, 31)])],                        // mixed set
        vec![(0, &[(0, 7)]), (1, &[(0, 7)])],          // repeated shape
        vec![(0, &[(0, 31)]), (2, &[(0, 31)])],        // structurally equal Arcs
        vec![(3, &[(100, 200)])],                      // out-of-domain retained
        vec![(0, &[(u32::MAX - 4, u32::MAX)])],        // u32::MAX sparse
        vec![(0, &[])],                                // empty inner
    ];
    for (ci, entries) in cases.iter().enumerate() {
        let input = weight(entries);
        let want = project_weight_to_kept(&input, &kept);
        let mut memo = ProjectionMemo::new(&kept);
        let got = project_weight_to_kept_memo(&input, &mut memo);
        assert_eq!(ranges_of(&got), ranges_of(&want), "case {ci}");
    }
    // Repeated same Arc: second call must hit.
    {
        let input = weight(&[(0, &[(0, 31)])]);
        let mut memo = ProjectionMemo::new(&kept);
        let first = project_weight_to_kept_memo(&input, &mut memo);
        let second = project_weight_to_kept_memo(&input, &mut memo);
        assert_eq!(ranges_of(&first), ranges_of(&second));
        assert!(memo.hits >= 1, "same-Arc repeat must hit");
    }
    // All-kept / all-dropped universes.
    {
        let all_kept = vec![true; 16];
        let input = weight(&[(0, &[(0, 15)])]);
        let mut memo = ProjectionMemo::new(&all_kept);
        let got = project_weight_to_kept_memo(&input, &mut memo);
        assert_eq!(ranges_of(&got), ranges_of(&input));
        let none_kept = vec![false; 16];
        let mut memo = ProjectionMemo::new(&none_kept);
        let got = project_weight_to_kept_memo(&input, &mut memo);
        assert!(got.raw_range_values().next().is_none());
    }
    // Multi-TSID preservation: outer ranges untouched.
    {
        let input = weight(&[(0, &[(0, 31)]), (2, &[(0, 31)]), (u32::MAX, &[(0, 5)])]);
        let mut memo = ProjectionMemo::new(&kept);
        let got = project_weight_to_kept_memo(&input, &mut memo);
        let want = project_weight_to_kept(&input, &kept);
        assert_eq!(ranges_of(&got), ranges_of(&want));
        let outers: Vec<(u32, u32)> =
            got.raw_range_values().map(|(r, _)| (*r.start(), *r.end())).collect();
        assert!(outers.contains(&(u32::MAX, u32::MAX)));
    }
    // Two separate cache contexts, different kept: no cross-context reuse.
    // (Each memo owns its own kept borrow, so reuse is unrepresentable.)
    {
        let input = weight(&[(0, &[(0, 15)])]);
        let kept_a = vec![true; 16];
        let kept_b = vec![false; 16];
        let mut memo_a = ProjectionMemo::new(&kept_a);
        let mut memo_b = ProjectionMemo::new(&kept_b);
        let got_a = project_weight_to_kept_memo(&input, &mut memo_a);
        let got_b = project_weight_to_kept_memo(&input, &mut memo_b);
        assert_eq!(ranges_of(&got_a), ranges_of(&input));
        assert!(got_b.raw_range_values().next().is_none());
        assert_eq!(ranges_of(&got_a), ranges_of(&project_weight_to_kept(&input, &kept_a)));
        assert_eq!(ranges_of(&got_b), ranges_of(&project_weight_to_kept(&input, &kept_b)));
    }
    // Saturation: small-capacity constructor stops inserting at cap.
    // Distinct inner sets under ONE kept mask (the production pattern):
    // first `cap` unique sets insert, later ones reject but stay exact.
    {
        let kept_a: Vec<bool> = (0..64).map(|i| i % 2 == 0).collect();
        let mut memo = ProjectionMemo::with_cap(&kept_a, 2);
        for i in 0..8u32 {
            let w = weight(&[(0, &[(i * 8, i * 8 + 7)])]);
            let got = project_weight_to_kept_memo(&w, &mut memo);
            let want = project_weight_to_kept(&w, &kept_a);
            assert_eq!(ranges_of(&got), ranges_of(&want));
        }
        assert!(memo.cap_rejections > 0, "cap must reject inserts");
        assert!(memo.map.len() <= 2);
    }
    // In-place wrapper: no-op weights untouched, changed weights match.
    {
        // All-kept: every inner set ptr-equals through the memo → true.
        let all_kept = vec![true; 200];
        let mut memo = ProjectionMemo::new(&all_kept);
        let mut untouched = weight(&[(0, &[(0, 15)]), (2, &[(100, 110)])]);
        let before: Vec<usize> = untouched
            .raw_range_values()
            .map(|(_, t)| std::sync::Arc::as_ptr(t) as usize)
            .collect();
        assert!(
            project_weight_to_kept_in_place(&mut untouched, &mut memo),
            "all-kept weight must be a no-op",
        );
        let after: Vec<usize> = untouched
            .raw_range_values()
            .map(|(_, t)| std::sync::Arc::as_ptr(t) as usize)
            .collect();
        assert_eq!(before, after, "no-op must not reassign inner Arcs");
        assert_eq!(memo.unchanged_weights, 1);
        // Mixed multi-TSID: one dropped token forces the remap path.
        let kept_mixed: Vec<bool> =
            (0..16).map(|i| i != 5).collect();
        let mut memo = ProjectionMemo::new(&kept_mixed);
        let mut changed = weight(&[(0, &[(0, 15)]), (2, &[(0, 15)])]);
        assert!(
            !project_weight_to_kept_in_place(&mut changed, &mut memo),
            "changed weight must take the remap path",
        );
        assert_eq!(memo.remapped_weights, 1);
        assert_eq!(
            ranges_of(&changed),
            ranges_of(&project_weight_to_kept(
                &weight(&[(0, &[(0, 15)]), (2, &[(0, 15)])]),
                &kept_mixed,
            )),
        );
    }
}

fn matcher_top_is(pop: &[StackMatcher], state: u32) -> bool {
    matches!(pop.first(), Some(StackMatcher::State(top)) if *top == state)
}

/// Literal concrete-stack interpreter for escapes (bottom-to-top stacks).
/// Pop matchers apply top-down (`pop[0]` is the stack top); `Any` matches
/// anything; pushes append bottom-to-top. Returns `None` when the escape
/// does not apply.
fn apply_escape(pop: &[StackMatcher], pushes: &[u32], stack: &[u32]) -> Option<Vec<u32>> {
    if stack.len() < pop.len() {
        return None;
    }
    for (offset, matcher) in pop.iter().enumerate() {
        let actual = stack[stack.len() - 1 - offset];
        match matcher {
            StackMatcher::Any => {}
            StackMatcher::State(want) => {
                if actual != *want {
                    return None;
                }
            }
            StackMatcher::States(wants) => {
                if !wants.contains(&actual) {
                    return None;
                }
            }
        }
    }
    let mut next = stack[..stack.len() - pop.len()].to_vec();
    next.extend_from_slice(pushes);
    Some(next)
}

fn apply_characterization(
    characterization: &TerminalCharacterization,
    stack: &[u32],
) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    for escape in &characterization.escapes {
        if let Some(next) = apply_escape(&escape.pop, &escape.pushes, stack) {
            out.push(next);
        }
    }
    out
}

/// Advisor §6 algebra, scoped single-link coordinates: parent target
/// states `I_L/I_R`, continuations `Qx/Qy`, child start `S`, child
/// interior `J`. `T_a; F; T_x` must admit the matching caller and reject
/// the sibling caller; dropping the child-start push must break the chain.
#[test]
#[ignore]
fn entry_finish_cancellation_distinguishes_call_sites() {
    const I_L: u32 = 10;
    const QX: u32 = 11;
    const QY: u32 = 12;
    const S: u32 = 100;
    const J: u32 = 101;
    const X: u32 = 13;
    // Slot escape R(I) P(I) P(Q) as characterized, then Entry appends S.
    let slot = TerminalCharacterization {
        escapes: vec![InitialEscape {
            pop: vec![StackMatcher::State(I_L)],
            pushes: vec![I_L, QX],
        }],
        reduces: vec![],
        nt_escapes: vec![],
        nt_rereduces: vec![],
        all_nts: BTreeSet::new(),
    };
    let entry = instantiate_entry(slot, S, 0);
    assert!(matcher_top_is(&entry.characterization.escapes[0].pop, I_L));
    assert_eq!(entry.characterization.escapes[0].pushes, vec![I_L, QX, S]);
    // Child terminal a: R(S) P(S) P(J).
    let child_ta = TerminalCharacterization {
        escapes: vec![InitialEscape {
            pop: vec![StackMatcher::State(S)],
            pushes: vec![S, J],
        }],
        reduces: vec![],
        nt_escapes: vec![],
        nt_rereduces: vec![],
        all_nts: BTreeSet::new(),
    };
    // Finish: R(J) R(Any) with return_pop=2.
    let finish = TerminalCharacterization {
        escapes: vec![InitialEscape {
            pop: vec![StackMatcher::State(J), StackMatcher::Any],
            pushes: vec![],
        }],
        reduces: vec![],
        nt_escapes: vec![],
        nt_rereduces: vec![],
        all_nts: BTreeSet::new(),
    };
    // Parent continuation terminal x: R(Qx) P(Qx) P(X).
    let parent_tx = TerminalCharacterization {
        escapes: vec![InitialEscape {
            pop: vec![StackMatcher::State(QX)],
            pushes: vec![QX, X],
        }],
        reduces: vec![],
        nt_escapes: vec![],
        nt_rereduces: vec![],
        all_nts: BTreeSet::new(),
    };
    // Matching caller: [I_L] -> E -> T_a -> F -> T_x.
    let stack = vec![I_L];
    let stack = apply_characterization(&entry.characterization, &stack)
        .into_iter()
        .next()
        .expect("entry applies at I_L");
    assert_eq!(stack, vec![I_L, QX, S]);
    let stack = apply_characterization(&child_ta, &stack)
        .into_iter()
        .next()
        .expect("child terminal applies on child start");
    assert_eq!(stack, vec![I_L, QX, S, J]);
    let stack = apply_characterization(&finish, &stack)
        .into_iter()
        .next()
        .expect("finish pops the child frame");
    assert_eq!(stack, vec![I_L, QX]);
    let stack = apply_characterization(&parent_tx, &stack)
        .into_iter()
        .next()
        .expect("matching continuation applies after return");
    assert_eq!(stack, vec![I_L, QX, X]);
    // Sibling caller carrying Qy: an entry from the sibling call site would
    // expose Qy, which T_x rejects.
    let sibling_stack = vec![I_L, QY];
    assert!(
        apply_characterization(&parent_tx, &sibling_stack).is_empty(),
        "sibling continuation must not match Qx transfer",
    );
    // Ablation: entry without the child-start push breaks the chain at T_a.
    let broken_entry = TerminalCharacterization {
        escapes: vec![InitialEscape {
            pop: vec![StackMatcher::State(I_L)],
            pushes: vec![I_L, QX],
        }],
        reduces: vec![],
        nt_escapes: vec![],
        nt_rereduces: vec![],
        all_nts: BTreeSet::new(),
    };
    let stack = apply_characterization(&broken_entry, &[I_L])
        .into_iter()
        .next()
        .unwrap();
    assert!(
        apply_characterization(&child_ta, &stack).is_empty(),
        "removing the child-start push must break the called-frame chain",
    );
}

#[test]
#[ignore]
fn shared_child_link_disagreement_declines_loudly() {
    use crate::compiler::glr::parser::ScopedSubgrammarLink;
    let links = vec![
        ScopedSubgrammarLink {
            parent_component: 0,
            slot_terminal: 3,
            child_component: 1,
            child_start: 0,
            return_pop: 1,
            child_start_nullable: false,
        },
        ScopedSubgrammarLink {
            parent_component: 0,
            slot_terminal: 4,
            child_component: 1,
            child_start: 0,
            return_pop: 2,
            child_start_nullable: false,
        },
    ];
    assert!(validate_shared_child_links(&links).is_err());
    let agreeing = vec![links[0]];
    assert!(validate_shared_child_links(&agreeing).is_ok());
}

#[test]
fn strict_static_trap_fires_on_every_dynamic_fallback_entry() {
    if crate::isolate_environment_test(false) { return; }
    // The trap itself is env-gated so genuine dynamic compositions are
    // unaffected; strict-static tests set the var and every dynamic mask
    // fallback panics loudly instead of contributing hidden admissions.
    // Serialized with all other process-env mutation in the test build.
    let _env_lock = crate::TEST_ENV_LOCK.lock().unwrap();
    unsafe { std::env::set_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", "1") };
    assert!(strict_static_dynamic_trap_enabled());
    let fired = std::panic::catch_unwind(|| strict_static_trap_dynamic("test_caller"));
    unsafe { std::env::remove_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC") };
    assert!(
        fired.is_err(),
        "strict-static trap must panic on a dynamic fallback entry point"
    );
    assert!(!strict_static_dynamic_trap_enabled());
    strict_static_trap_dynamic("test_caller");
}

#[test]
fn nested_nonnullable_links_certify_with_doubled_depth() {
    use crate::compiler::glr::parser::ScopedSubgrammarLink;
    // An acyclic nonnullable nested chain certifies with the 2H bound:
    // depth-2 links allow at most 4 control events per gap (R^a E^b with
    // a <= 2, b <= 2). Flat links keep the exact depth-2 certificate.
    let nested = vec![
        ScopedSubgrammarLink {
            parent_component: 0,
            slot_terminal: 3,
            child_component: 1,
            child_start: 0,
            return_pop: 1,
            child_start_nullable: false,
        },
        ScopedSubgrammarLink {
            parent_component: 1,
            slot_terminal: 4,
            child_component: 2,
            child_start: 0,
            return_pop: 1,
            child_start_nullable: false,
        },
    ];
    let certificate =
        certify_bounded_closure(&nested).expect("nonnullable nested chain must certify");
    assert_eq!(certificate.max_nesting_depth, 2);
    assert_eq!(certificate.max_controls_per_gap, 4);
    let flat = vec![nested[0]];
    let flat_certificate =
        certify_bounded_closure(&flat).expect("flat link must certify");
    assert_eq!(flat_certificate.max_nesting_depth, 1);
    assert_eq!(flat_certificate.max_controls_per_gap, 2);
    // Link-level control cycles keep their loud decline: unbounded stack
    // growth and re-entry ping-pong need general C* support.
    let cyclic = vec![
        nested[0],
        nested[1],
        ScopedSubgrammarLink {
            parent_component: 2,
            slot_terminal: 5,
            child_component: 1,
            child_start: 0,
            return_pop: 1,
            child_start_nullable: false,
        },
    ];
    let error =
        certify_bounded_closure(&cyclic).expect_err("cyclic links must decline loudly");
    assert!(
        error.contains("cycle"),
        "decline must name the cycle, got: {error}",
    );
    // Nullable links keep their loud decline.
    let mut nullable = nested;
    nullable[1].child_start_nullable = true;
    let error =
        certify_bounded_closure(&nullable).expect_err("nullable links must decline loudly");
    assert!(
        error.contains("nullable"),
        "decline must name nullability, got: {error}",
    );
}

#[test]
#[ignore]
fn composer_scaffold_declines_loudly_without_hidden_fallback() {
    assert!(assemble_boundary_transfer_query().is_err());
}

#[test]
fn nested_ready_depth_four_is_load_bearing() {
    use crate::compiler::glr::parser::ScopedSubgrammarLink;
    use crate::runtime::Constraint;
    use crate::Vocab;

    fn terminal_id(constraint: &Constraint, name: &str) -> TerminalID {
        constraint
            .terminal_display_names
            .iter()
            .position(|candidate| candidate == name)
            .unwrap() as u32
    }

    fn find_local_containing(constraint: &Constraint, needle: &str) -> TerminalID {
        constraint
            .terminal_display_names
            .iter()
            .position(|candidate| candidate.contains(needle))
            .unwrap() as u32
    }

    // Walk a deterministic parser DWA with a bottom-to-top stack-state
    // word using the same transition API the runtime uses. Admission is
    // exact reachability of a non-empty final weight over non-empty
    // edge weights.
    fn dwa_admits_stack_word(dwa: &DWA, word: &[u32]) -> bool {
        let mut state = dwa.start_state();
        for &symbol in word {
            let label = symbol as i32;
            let Some((target, weight)) = dwa.states()[state as usize]
                .transitions
                .get_entry(&label)
            else {
                return false;
            };
            if weight.is_empty() {
                return false;
            }
            state = target;
        }
        dwa.states()[state as usize]
            .final_weight
            .as_ref()
            .is_some_and(|weight| !weight.is_empty())
    }

    // Enumerate accepted stack words (bottom-to-top) over the
    // deterministic parser DWA via DFS. The bounded program is acyclic,
    // so the language is finite.
    fn collect_accepted_words(dwa: &DWA, max_depth: usize) -> Vec<Vec<u32>> {
        let mut out = Vec::new();
        let mut stack: Vec<(u32, Vec<u32>)> = vec![(dwa.start_state(), Vec::new())];
        while let Some((state, word)) = stack.pop() {
            if dwa.states()[state as usize]
                .final_weight
                .as_ref()
                .is_some_and(|weight| !weight.is_empty())
            {
                out.push(word.clone());
            }
            if word.len() >= max_depth {
                continue;
            }
            for (label, target, weight) in dwa.states()[state as usize].transitions.entries() {
                if weight.is_empty() {
                    continue;
                }
                if label < 0 {
                    continue;
                }
                let mut next_word = word.clone();
                next_word.push(label as u32);
                stack.push((target, next_word));
            }
        }
        out
    }

    // Real nested fixture through real grammars: P ::= "L" SUB SUB "x",
    // M ::= "m" SUB2, G ::= "g". Depth-2 chain certifies 2H = 4.
    let vocab = Vocab::new(vec![
        (0, b"L".to_vec()),
        (1, b"R".to_vec()),
        (2, b"x".to_vec()),
        (3, b"y".to_vec()),
        (4, b"m".to_vec()),
        (5, b"g".to_vec()),
    ]);
    let parent = Constraint::from_glrm_grammar(
        r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "L" SUB SUB "x" | "R" SUB SUB "y";
            "#,
        &vocab,
    )
    .unwrap();
    let mid = Constraint::from_glrm_grammar(
        r#"
                start m;
                t SUB2 ::= @token(998);
                nt m ::= "m" SUB2;
            "#,
        &vocab,
    )
    .unwrap();
    let grandchild = Constraint::from_glrm_grammar(
        r#"
                start g;
                nt g ::= "g";
            "#,
        &vocab,
    )
    .unwrap();
    let sub_p = terminal_id(&parent, "SUB");
    let sub2_m = terminal_id(&mid, "SUB2");
    let n_p = parent.table.num_terminals;
    let n_m = mid.table.num_terminals;
    let n_g = grandchild.table.num_terminals;
    let leaf_offsets = vec![0, n_p, n_p + n_m];
    let num_terminals = n_p + n_m + n_g;
    // Local id of grandchild "g" (display names quote the literal).
    let g_local = find_local_containing(&grandchild, "g");
    let g_global = leaf_offsets[2] + g_local;
    let links = vec![
        ScopedSubgrammarLink {
            parent_component: 0,
            slot_terminal: sub_p,
            child_component: 1,
            child_start: 0,
            return_pop: 1,
            child_start_nullable: false,
        },
        ScopedSubgrammarLink {
            parent_component: 1,
            slot_terminal: sub2_m,
            child_component: 2,
            child_start: 0,
            return_pop: 1,
            child_start_nullable: false,
        },
    ];
    let mut context =
        build_signed_link_context_from_parts(
            vec![&parent.table, &mid.table, &grandchild.table],
            vec![None, None, None],
            links.clone(),
            &leaf_offsets,
            num_terminals,
            false,
            BTreeSet::new(),
        )
        .expect("nested signed context");
    assert_eq!(context.closure.max_nesting_depth, 2);
    assert_eq!(context.closure.max_controls_per_gap, 4);

    // Synthetic *compiler-fixture* shard: g -> g across two lexical
    // vertices. The middle gap must chain R(G1->M1) R(M1->P) E(P->M2)
    // E(M2->G2): four control advancements before the second terminal.
    let mut shard_dwa = DWA::new(0, 1);
    let s1 = shard_dwa.add_state();
    let s2 = shard_dwa.add_state();
    shard_dwa.add_transition(0, g_global as i32, s1, Weight::all());
    shard_dwa.add_transition(s1, g_global as i32, s2, Weight::all());
    shard_dwa.set_final_weight(s2, Weight::all());
    assert!(shard_dwa.is_acyclic());

    let singleton = crate::compiler::stages::equiv_types::ManyToOneIdMap {
        original_to_internal: vec![0],
        internal_to_originals: vec![vec![0]],
        representative_original_ids: vec![0],
    };
    let id_map = InternalIdMap {
        tokenizer_states: singleton.clone(),
        vocab_tokens: singleton,
        deferred_vocab_singleton_original_ids: None,
    };

    let mut emitted = vec![false; num_terminals as usize];
    emitted[g_global as usize] = true;
    let library = build_fragment_library(&context, &emitted, 0).expect("fragment library");
    let full = compile_signed_shard_parser(&context, &library, &shard_dwa, &id_map, 0)
        .expect("depth-4 compilation must succeed");

    context.closure.max_controls_per_gap = 2;
    let library2 = build_fragment_library(&context, &emitted, 0).expect("fragment library");
    let truncated = compile_signed_shard_parser(&context, &library2, &shard_dwa, &id_map, 0)
        .expect("depth-2 compilation must succeed");

    // Exact DWA-level load-bearing proof: some stack-state word reaches
    // a final in the depth-4 program but not in the depth-2 program.
    let full_words = collect_accepted_words(&full.parser_dwa, 12);
    assert!(!full_words.is_empty(), "depth-4 program must admit something");
    let mut witness: Option<Vec<u32>> = None;
    for word in &full_words {
        if !dwa_admits_stack_word(&truncated.parser_dwa, word) {
            witness = Some(word.clone());
            break;
        }
    }
    let witness =
        witness.expect("depth-4 must admit a stack word that depth-2 truncates (R,R,E,E)");
    assert!(dwa_admits_stack_word(&full.parser_dwa, &witness));
    assert!(!dwa_admits_stack_word(&truncated.parser_dwa, &witness));
    eprintln!(
        "[nested_ready_depth_four] witness_len={} witness={:?} full_states={} trunc_states={}",
        witness.len(),
        witness,
        full.parser_dwa.num_states(),
        truncated.parser_dwa.num_states(),
    );
}

/// Build a two-component (parent + child) signed context where both
/// components carry their own ignore terminal. `global` selects whether the
/// context certifies the ignores as uniformly globally erasable.
fn two_component_ignore_context<'a>(
    parent: &'a Constraint,
    child: &'a Constraint,
    global: bool,
) -> (SignedLinkContext<'a>, TerminalID) {
    let sub_p = parent
        .terminal_display_names
        .iter()
        .position(|candidate| candidate == "SUB")
        .expect("parent SUB terminal") as TerminalID;
    let parent_terminals = parent.table.num_terminals;
    let terminal_offsets = vec![0, parent_terminals];
    let num_terminals = parent_terminals + child.table.num_terminals;
    let links = vec![ScopedSubgrammarLink {
        parent_component: 0,
        slot_terminal: sub_p,
        child_component: 1,
        child_start: 0,
        return_pop: 1,
        child_start_nullable: false,
    }];
    let context = build_signed_link_context_from_parts(
        vec![&parent.table, &child.table],
        vec![parent.ignore_terminal, child.ignore_terminal],
        links,
        &terminal_offsets,
        num_terminals,
        global,
        BTreeSet::new(),
    )
    .expect("two-component signed context");
    let child_ignore_local = child.ignore_terminal.expect("child ignore terminal");
    (context, parent_terminals + child_ignore_local)
}

fn ignore_test_constraints() -> (Constraint, Constraint) {
    use crate::Vocab;
    let vocab = Vocab::new(vec![
        (0, b"X".to_vec()),
        (1, b" ".to_vec()),
        (2, b"\t".to_vec()),
        (3, b"a".to_vec()),
        (4, b"!".to_vec()),
    ]);
    let parent = Constraint::from_glrm_grammar(
        r#"
                start document;
                ignore WS;
                t WS ::= " "+;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
        &vocab,
    )
    .unwrap();
    let child = Constraint::from_glrm_grammar(
        r#"
                start child;
                ignore WS;
                t WS ::= "\t"+;
                nt child ::= "a";
            "#,
        &vocab,
    )
    .unwrap();
    (parent, child)
}

/// A globally erasable ignore identity must apply while the parser is
/// inside the parent *and* inside the child.
#[test]
fn global_ignore_identity_spans_parent_and_child_scopes() {
    let (parent, child) = ignore_test_constraints();
    let (context, child_ignore) = two_component_ignore_context(&parent, &child, true);
    assert!(context.global_ignores);
    let mut cache = None;
    let transfer =
        scoped_transfer_for_terminal(&context, child_ignore, &mut cache).expect("transfer");
    assert!(cache.is_some(), "global identity is cached once per library");
    // Child scope: the child's own start state.
    let child_start = context.state_offsets[1];
    assert!(
        !apply_characterization(&transfer, &[child_start]).is_empty(),
        "global ignore identity must apply inside the child scope",
    );
    // Parent scope: parent state zero and the last parent state.
    assert!(
        !apply_characterization(&transfer, &[0]).is_empty(),
        "global ignore identity must apply inside the parent scope",
    );
    assert!(
        !apply_characterization(&transfer, &[parent.table.num_states - 1]).is_empty(),
        "global ignore identity must apply across the whole parent interval",
    );
}

/// A scope-dependent (non-global) child ignore identity must stay inside
/// the child scope and must not apply in the parent.
#[test]
fn scoped_child_ignore_identity_does_not_apply_in_parent() {
    let (parent, child) = ignore_test_constraints();
    let (context, child_ignore) = two_component_ignore_context(&parent, &child, false);
    assert!(!context.global_ignores);
    let mut cache = None;
    let transfer =
        scoped_transfer_for_terminal(&context, child_ignore, &mut cache).expect("transfer");
    assert!(cache.is_none(), "scoped ignore must not build a global identity");
    let child_start = context.state_offsets[1];
    assert!(
        !apply_characterization(&transfer, &[child_start]).is_empty(),
        "scoped child ignore identity must apply inside the child scope",
    );
    assert!(
        apply_characterization(&transfer, &[0]).is_empty(),
        "scoped child ignore identity must not apply in the parent scope",
    );
}
