use super::*;
use crate::automata::weighted_u32::terminal_automaton::TerminalAutomaton;
use crate::compiler::composition::{
    CompiledSubgrammarInput, SegmentedBoundaryBackend,
    component_ignores_are_globally_erasable, compose_constraints_owned_parent_segmented,
    eliminate_composed_runtime_controls, install_published_static_boundary_shards,
    load_vocab, merged_ignore_terminals, merged_retained_terminal_exprs,
    merged_terminal_display_names, publish_walk_boundary_shard_work, WalkBoundaryShardWork,
};
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::table::{
    GLRTable, SubgrammarTableInput, compose_subgrammar_tables, control_elimination_budget_exhausted,
    control_elimination_run_count, empty_terminals_in_composed_table,
};
use crate::compiler::pipeline::compute_disallowed_follows;
use crate::grammar::flat::TerminalID;
use crate::runtime::Constraint;

/// Process-env guard for strict-reference tests. Saves the prior value and
/// restores it on drop (callers must hold `crate::TEST_ENV_LOCK`).
struct EnvVarGuard {
    key: &'static str,
    original: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let original = std::env::var_os(key);
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, original }
    }

    fn unset(key: &'static str) -> Self {
        let original = std::env::var_os(key);
        unsafe {
            std::env::remove_var(key);
        }
        Self { key, original }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.original {
            Some(value) => unsafe {
                std::env::set_var(self.key, value);
            },
            None => unsafe {
                std::env::remove_var(self.key);
            },
        }
    }
}

struct LowLevelComposed {
    table: ComposedTable,
    tokenizer: Tokenizer,
    tokenizer_offsets: Vec<u32>,
    terminal_names: Vec<String>,
    ignore_canonical: Option<u32>,
}

fn terminal_id(constraint: &Constraint, name: &str) -> TerminalID {
    constraint
        .terminal_display_names
        .iter()
        .position(|candidate| candidate == name)
        .unwrap() as u32
}

// Existing composition code only: table splice + control elimination +
// disjoint-union tokenizer. Mirrors the phase1 probe helper.
fn low_level_compose(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
) -> LowLevelComposed {
    let global_ignores = component_ignores_are_globally_erasable(parent, children);
    let table_inputs: Vec<SubgrammarTableInput> = children
        .iter()
        .map(|child| SubgrammarTableInput {
            placeholder_terminal: child.placeholder_terminal,
            additional_placeholder_terminals: &[],
            table: &child.constraint.table,
            ignore_terminal: (!global_ignores)
                .then_some(child.constraint.ignore_terminal)
                .flatten(),
            start_nullable: child.constraint.table.embedded_start_nullable(),
        })
        .collect();
    let mut composed = compose_subgrammar_tables(
        &parent.table,
        (!global_ignores).then_some(parent.ignore_terminal).flatten(),
        &table_inputs,
    )
    .expect("compose tables");
    eliminate_composed_runtime_controls(&mut composed).expect("eliminate controls");
    let terminal_names = merged_terminal_display_names(parent, children);
    let mut tokenizer_inputs: Vec<(&Tokenizer, u32)> =
        Vec::with_capacity(children.len() + 1);
    tokenizer_inputs.push((&parent.tokenizer, composed.terminal_offsets[0]));
    for (index, child) in children.iter().enumerate() {
        tokenizer_inputs
            .push((&child.constraint.tokenizer, composed.terminal_offsets[index + 1]));
    }
    let (mut merged, tokenizer_offsets) =
        Tokenizer::disjoint_union_with_terminal_offsets(&tokenizer_inputs);
    if merged.terminal_exprs().is_none() {
        let all: Vec<&Constraint> = std::iter::once(parent)
            .chain(children.iter().map(|child| child.constraint))
            .collect();
        if let Some(exprs) = merged_retained_terminal_exprs(
            &all,
            &composed.terminal_offsets,
            composed.table.num_terminals,
        ) {
            merged.restore_terminal_exprs(Some(exprs)).expect("restore merged exprs");
        }
    }
    let ignores = merged_ignore_terminals(
        parent,
        children,
        &composed.terminal_offsets,
        global_ignores,
    );
    assert_eq!(
        tokenizer_offsets[0], 1,
        "merged state 0 must be the fresh reset fan-out"
    );
    LowLevelComposed {
        table: composed,
        tokenizer: merged,
        tokenizer_offsets,
        terminal_names,
        ignore_canonical: ignores.canonical,
    }
}

fn analyzed_grammar(table: &GLRTable, names: &[String]) -> AnalyzedGrammar {
    let augmented_start =
        table.rules.first().expect("composed table has augmented start").lhs;
    AnalyzedGrammar::from_composed_rules(
        table.rules.clone(),
        table.num_terminals,
        names.to_vec(),
        table.nonterminal_display_names.clone(),
        augmented_start,
    )
}

#[test]
fn scoped_witnessed_ti_candidates_are_owner_and_policy_local() {
    let _env_lock = crate::TEST_ENV_LOCK.lock().expect("test env lock poisoned");
    let _strict = EnvVarGuard::set(
        "GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_STRICT_REFERENCE",
        "1",
    );
    let _no_small_skip = EnvVarGuard::set("GLRMASK_P0_TI_SKIP_MAX_TOKENIZER_STATES", "0");

    let vocab = Vocab::new(vec![
        (0, b"a".to_vec()),
        (1, b"b".to_vec()),
        (2, b"c".to_vec()),
        (3, b"x".to_vec()),
        (4, b"ax".to_vec()),
        (5, b"bx".to_vec()),
        (6, b"xa".to_vec()),
        (7, b"xc".to_vec()),
    ]);
    let constraint = Constraint::from_glrm_grammar(
        r#"
                start doc;
                t A ::= "a";
                t B ::= "b";
                t C ::= "c";
                t X ::= "x";
                nt doc ::= A X | B X | X C;
            "#,
        &vocab,
    )
    .unwrap();
    let grammar = analyzed_grammar(&constraint.table, &constraint.terminal_display_names);
    let disallowed = compute_disallowed_follows(&grammar);
    let a = terminal_id(&constraint, "A");
    let b = terminal_id(&constraint, "B");
    let c = terminal_id(&constraint, "C");
    assert_eq!([a, b, c], [0, 1, 2], "fixture relies on declared terminal order");

    // Deliberately split the otherwise identical C into a second immediate
    // owner.  The TI pre-certificate must never cross that ownership
    // boundary, even though the lexical expression is byte-identical.
    let ownership = Arc::new(
        tdwa::scope::BoundaryOwnership::flat(&[0, 2], grammar.num_terminals)
            .expect("two-owner TI fixture"),
    );
    let initial = tdwa::scope::InitialStateDomain::from_mask(
        constraint.tokenizer.num_states() as usize,
        vec![true; constraint.tokenizer.num_states() as usize],
    )
    .expect("full test initial domain");
    let mut resets = constraint
        .tokenizer
        .deterministic_reset_states()
        .into_iter()
        .collect::<Vec<_>>();
    resets.sort_unstable();
    resets.dedup();
    let scope = tdwa::scope::BoundaryAnalysisScope::new(
        initial,
        resets,
        ownership,
        tdwa::scope::ImmediateComponentId(0),
        true,
        None,
    )
    .expect("TI fixture scope");
    let active = vec![true; grammar.num_terminals as usize];
    let groups = tdwa::l2p::scoped_terminal_interchangeability_candidate_groups(
        &constraint.tokenizer,
        &active,
        &grammar,
        &disallowed,
        constraint.ignore_terminal,
        &scope,
    );
    assert_eq!(groups, vec![vec![a, b]]);

    // Exercise the actual scoped family pipeline.  If the witnessed A/B
    // merge activates, strict-reference mode rebuilds this exact scope with
    // TI suppressed and compares the completed weighted language in
    // original tokenizer-state/token coordinates.
    let flat: Arc<[u32]> =
        Arc::from(tdwa::l1::build_flat_transition_table(&constraint.tokenizer));
    let coloring = crate::compiler::stages::id_map_and_terminal_dwa::types::TerminalColoring::identity(
        grammar.num_terminals as usize,
    );
    let (_artifact, _profile) = tdwa::build_scoped_boundary_id_map_and_terminal_dwa(
        &constraint.tokenizer,
        &vocab,
        &coloring,
        constraint.ignore_terminal,
        &grammar,
        &disallowed,
        flat,
        &scope,
    );
}

/// Step-4c boundary parser table: the control-free spliced composed table
/// (the Phase-1 object — never the dynamic path's control-bearing
/// recursive table) with every unbound slot terminal emptied. Controls
/// are a runtime device of the dynamic path; they must not appear in the
/// static boundary parser's table at all, so a nonzero control count is
/// a hard error, not a fallback. `unbound_slots` holds
/// (component_index, component-local terminal) pairs; they are resolved
/// to composed IDs through the composed terminal offsets.
fn prepare_spliced_boundary_table(
    name: &str,
    composed: &LowLevelComposed,
    unbound_slots: &[(usize, TerminalID)],
) -> (Arc<GLRTable>, f64) {
    assert!(
        composed.table.table.control_terminals.is_empty(),
        "{name}: spliced composed table must be control-free, has {:?}",
        composed.table.table.control_terminals,
    );
    assert!(
        composed.table.control_terminals.is_empty(),
        "{name}: composed control set must be empty",
    );
    let started = Instant::now();
    let mut table = composed.table.table.clone();
    let emptied: Vec<TerminalID> = unbound_slots
        .iter()
        .map(|(component, local)| {
            composed.table.terminal_offsets[*component]
                .checked_add(*local)
                .expect("slot terminal offset overflow")
        })
        .collect();
    for &terminal in &emptied {
        assert!(
            (terminal as usize) < table.num_terminals as usize,
            "{name}: unbound slot terminal {terminal} out of range",
        );
    }
    empty_terminals_in_composed_table(&mut table, &emptied);
    let ms = started.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "BOUNDARY_TABLE name={name} states={} terms={} controls={} emptied={} prep_ms={ms:.3}",
        table.num_states,
        table.num_terminals,
        table.control_terminals.len(),
        emptied.len(),
    );
    (Arc::new(table), ms)
}

/// Pin the installing runtime's scoped tokenizer packing: leaves packed
/// back-to-back from 0 in link-component order (parent first). The walk
/// shard's private TSID map is indexed by exactly this coordinate; a
/// layout change here must update `publish_walk_boundary_shard_work`.
fn assert_scoped_tokenizer_packing(
    name: &str,
    installed: &Constraint,
    component_state_counts: &[u32],
) {
    let layout = installed
        .recursive_parser_layout()
        .expect("recursive layout")
        .expect("recursive layout present");
    let mut expected = Vec::with_capacity(component_state_counts.len());
    let mut next = 0u32;
    for &count in component_state_counts {
        expected.push(next);
        next = next.checked_add(count).expect("scoped total overflow");
    }
    assert_eq!(
        layout.leaf_tokenizer_state_offsets, expected,
        "{name}: runtime leaf packing must match link-component order",
    );
    assert_eq!(
        layout.total_tokenizer_states, next,
        "{name}: runtime scoped total must match link components",
    );
}

/// Component-local IDs of `TOOL_ARGS_SLOT_*` terminals in `constraint`
/// except the bound ones. The sweep parents/children follow the
/// `TOOL_ARGS_SLOT_{index}` slot convention; the shared `slot`-named
/// JSON-string terminals (`subgrammar2::"..."`) never match the prefix.
fn unbound_slot_terminals(
    constraint: &Constraint,
    bound_names: &[&str],
) -> Vec<TerminalID> {
    constraint
        .terminal_display_names
        .iter()
        .enumerate()
        .filter(|(_, name)| {
            name.starts_with("TOOL_ARGS_SLOT_") && !bound_names.contains(&name.as_str())
        })
        .map(|(id, _)| id as TerminalID)
        .collect()
}

#[test]
fn toy_two_component_crossing_sets() {
    // Parent `doc ::= PA SUB PB` with child `item ::= CC CD` bound at SUB.
    // Hand-verified crossing sets (trace the merged lexer from each
    // component's states; the composed table's pairwise follows prune the
    // rest):
    // - "ac": PA[parent] then CC[child], pair (PA,CC) allowed -> X_parent.
    // - "db": CD[child] then PB[parent], pair (CD,PB) allowed -> X_child.
    // - "ca": pair (CC,PA) never adjacent -> pruned everywhere.
    // - "bd": pair (PB,CD) never adjacent (PB is last) -> pruned.
    // - "ab", "cd": single-component paths -> internal, not crossing.
    // - single bytes: no reset strictly inside -> never crossing.
    let vocab = Vocab::new(vec![
        (0, b"a".to_vec()),
        (1, b"b".to_vec()),
        (2, b"c".to_vec()),
        (3, b"d".to_vec()),
        (4, b"ac".to_vec()),
        (5, b"db".to_vec()),
        (6, b"ca".to_vec()),
        (7, b"bd".to_vec()),
        (8, b"ab".to_vec()),
        (9, b"cd".to_vec()),
    ]);
    let parent = Constraint::from_glrm_grammar(
        r#"
                start doc;
                t PA ::= "a";
                t PB ::= "b";
                t SUB ::= @token(999);
                nt doc ::= PA SUB PB;
            "#,
        &vocab,
    )
    .unwrap();
    let child = Constraint::from_glrm_grammar(
        r#"
                start item;
                t CC ::= "c";
                t CD ::= "d";
                nt item ::= CC CD;
            "#,
        &vocab,
    )
    .unwrap();
    let children = [CompiledSubgrammarInput {
        placeholder_terminal: terminal_id(&parent, "SUB"),
        additional_placeholder_terminals: &[],
        constraint: &child,
    }];
    let composed = low_level_compose(&parent, &children);
    let grammar = analyzed_grammar(&composed.table.table, &composed.terminal_names);
    let disallowed = compute_disallowed_follows(&grammar);
    let flat: Arc<[u32]> =
        Arc::from(tdwa::l1::build_flat_transition_table(&composed.tokenizer));
    let ownership = Arc::new(
        tdwa::scope::BoundaryOwnership::flat(
            &composed.table.terminal_offsets,
            grammar.num_terminals,
        )
        .expect("toy terminal ownership"),
    );
    if std::env::var_os("GLRMASK_DEBUG_TOY_SCOPE").is_some() {
        for raw in composed.tokenizer_offsets[0]
            ..composed.tokenizer_offsets[0] + parent.tokenizer.num_states()
        {
            eprintln!(
                "TOY_SCOPE raw={} matched={:?} future={:?}",
                raw,
                composed
                    .tokenizer
                    .matched_terminals_iter(raw)
                    .collect::<Vec<_>>(),
                composed
                    .tokenizer
                    .possible_future_terminals_iter(raw)
                    .collect::<Vec<_>>(),
            );
            for token_id in [2u32, 3, 4, 5, 9] {
                let bytes = vocab.get(token_id).unwrap();
                let exec = composed.tokenizer.execute_from_state(bytes, raw);
                eprintln!(
                    "TOY_SCOPE raw={} token={} bytes={:?} matches={:?} end={:?}",
                    raw,
                    token_id,
                    bytes,
                    exec.matches,
                    exec.end_state,
                );
            }
        }
    }
    for (index, (num_states, expected)) in
        [(parent.tokenizer.num_states(), vec![4u32]), (child.tokenizer.num_states(), vec![5u32])]
            .into_iter()
            .enumerate()
    {
        let commit = commit_states_for_component(
            &composed.tokenizer_offsets,
            num_states,
            index,
            composed.tokenizer.num_states() as usize,
        );
        let scope = tdwa::scope::BoundaryAnalysisScope::new(
            tdwa::scope::InitialStateDomain::from_mask(
                composed.tokenizer.num_states() as usize,
                commit.clone(),
            )
            .expect("toy initial-state domain"),
            composed
                .tokenizer
                .deterministic_reset_states()
                .into_iter()
                .collect(),
            Arc::clone(&ownership),
            tdwa::scope::ImmediateComponentId(index as u32),
            true,
            None,
        )
        .expect("toy boundary scope");
        let output = build_boundary_terminal_dwa(&BoundaryWalkInputs {
            merged_tokenizer: &composed.tokenizer,
            vocab: &vocab,
            grammar: &grammar,
            disallowed_follows: &disallowed,
            ignore_terminal: composed.ignore_canonical,
            follow_transparent_ignores: None,
            scope: &scope,
            flat_trans: Some(&flat),
            retain_non_crossing_paths: false,
        })
        .expect("toy shard walk must produce a DWA");
        if std::env::var_os("GLRMASK_DEBUG_TOY_SCOPE").is_some() {
            eprintln!(
                "TOY_SCOPE shard={} idmap_o2i={:?} idmap_classes={:?} reps={:?}",
                index,
                output.id_map.tokenizer_states.original_to_internal,
                output.id_map.tokenizer_states.internal_to_originals,
                output.id_map.tokenizer_states.representative_original_ids,
            );
        }
        assert!(output.dwa.is_acyclic(), "toy shard {index} DWA must be acyclic");
        let tokens = boundary_accepted_tokens(&output.dwa, &output.id_map);
        assert_eq!(
            tokens.into_iter().collect::<Vec<_>>(),
            expected,
            "toy shard {index} crossing set",
        );
    }
}

fn restore_component(component: &mut Constraint, label: &str) {
    if component.tokenizer.terminal_exprs().is_none()
        && let Some(exprs) = component.retained_terminal_exprs().map(|exprs| exprs.to_vec())
    {
        Arc::make_mut(&mut component.tokenizer).restore_terminal_exprs(Some(exprs))
            .expect("restore component terminal exprs");
    }
    let inline_rules = component.table.rules.len();
    let retained = component.retained_table_rules().expect("decode retained rules").len();
    if inline_rules != retained {
        component.table.rules =
            component.retained_table_rules().expect("decode retained rules").to_vec();
    }
    component
        .materialize_composition_metadata_for_compilation()
        .expect("materialize composition metadata");
    eprintln!("BOUNDARY_WALK restore {label} inline_rules={inline_rules} retained_rules={retained}");
}

struct Selected10Outer {
    vocab: Vocab,
    core: Constraint,
    dispatch: Constraint,
    composed: LowLevelComposed,
    grammar: AnalyzedGrammar,
    disallowed: BTreeMap<u32, BitSet>,
}

fn load_selected10_outer() -> Selected10Outer {
    let root = phase1_root();
    let vocab_path = phase1_vocab_path();
    let vocab = load_vocab(&vocab_path);
    assert_eq!(
        vocab.entries_map().len(),
        128_256,
        "selected10 expects the CFA Llama-3.1 vocabulary",
    );
    let mut core =
        Constraint::load(&std::fs::read(root.join("core.bin")).expect("read core.bin"))
            .expect("load core");
    let dispatch_name = std::env::var("PHASE1_DISPATCH")
        .unwrap_or_else(|_| "dispatch-literal.bin".to_string());
    let mut dispatch =
        Constraint::load(&std::fs::read(root.join(&dispatch_name)).expect("read dispatch"))
            .expect("load dispatch");
    restore_component(&mut core, "core");
    restore_component(&mut dispatch, "dispatch");
    let children = [CompiledSubgrammarInput {
        placeholder_terminal: terminal_id(&core, "PROGRAMMATIC_TOOL_SUFFIX"),
        additional_placeholder_terminals: &[],
        constraint: &dispatch,
    }];
    let composed = low_level_compose(&core, &children);
    let grammar = analyzed_grammar(&composed.table.table, &composed.terminal_names);
    let disallowed = compute_disallowed_follows(&grammar);
    Selected10Outer { vocab, core, dispatch, composed, grammar, disallowed }
}

fn require_release_selected10_gate(name: &str) {
    assert!(
        !cfg!(debug_assertions),
        "{name} is a selected10 integration/performance gate and must be run with an optimized test binary; rerun with `cargo test --release ...`"
    );
}

/// Production-path selected10 crossing flow shared by the plain gate and the
/// two-stage strict gate below: one shared equivalence, one standard walk
/// per component, NWA-level crossing filter. Asserts the 143-token dispatch
/// crossing set (MINBOUND oracle dump) and the 26-state true-minimal
/// crossing DWA, plus the empty core shard.
fn run_selected10_boundary_walk_crossing() {
    require_release_selected10_gate("selected10_boundary_walk_crossing");
    let fixture = load_selected10_outer();
    let vocab = &fixture.vocab;
    let core = &fixture.core;
    let dispatch = &fixture.dispatch;
    let composed = &fixture.composed;
    let grammar = &fixture.grammar;
    let disallowed = &fixture.disallowed;
    let flat: Arc<[u32]> =
        Arc::from(tdwa::l1::build_flat_transition_table(&composed.tokenizer));
    let ownership = Arc::new(
        tdwa::scope::BoundaryOwnership::flat(
            &composed.table.terminal_offsets,
            grammar.num_terminals,
        )
        .expect("selected10 terminal ownership"),
    );
    // Frozen output of the independent Phase-1 MINBOUND B-A oracle for
    // this exact selected10 fixture. The original artifact is preserved
    // in the 2026-09-18 linker compaction with SHA-256
    // 689820493b69f3a603c1c38235a74b4b91aa2a6449f22cddbdbc5a5e3d60bf6d.
    // Keep the oracle in-repo so this regression never depends on
    // historical /tmp state or a machine-local PHASE1_DUMP_DIR.
    let oracle_text = include_str!("../../../testdata/selected10_minbound_tokens.txt");
    let oracle: BTreeSet<u32> =
        oracle_text.split_whitespace().map(|value| value.parse::<u32>().unwrap()).collect();
    assert_eq!(oracle.len(), 143, "oracle dump must hold the 143 MINBOUND tokens");

    let component_states = [core.tokenizer.num_states(), dispatch.tokenizer.num_states()];
    for (index, num_states) in component_states.into_iter().enumerate() {
        let commit = commit_states_for_component(
            &composed.tokenizer_offsets,
            num_states,
            index,
            composed.tokenizer.num_states() as usize,
        );
        let scope = tdwa::scope::BoundaryAnalysisScope::new(
            tdwa::scope::InitialStateDomain::from_mask(
                composed.tokenizer.num_states() as usize,
                commit.clone(),
            )
            .expect("selected10 initial-state domain"),
            composed
                .tokenizer
                .deterministic_reset_states()
                .into_iter()
                .collect(),
            Arc::clone(&ownership),
            tdwa::scope::ImmediateComponentId(index as u32),
            true,
            None,
        )
        .expect("selected10 boundary scope");
        let output = build_boundary_terminal_dwa(&BoundaryWalkInputs {
            merged_tokenizer: &composed.tokenizer,
            vocab,
            grammar,
            disallowed_follows: disallowed,
            ignore_terminal: composed.ignore_canonical,
            follow_transparent_ignores: None,
            scope: &scope,
            flat_trans: Some(&flat),
            retain_non_crossing_paths: false,
        })
        .expect("selected10 shard walk must produce a DWA");
        assert!(output.dwa.is_acyclic(), "shard {index} DWA must be acyclic");
        let tokens = boundary_accepted_tokens(&output.dwa, &output.id_map);
        let emitted =
            boundary_emitted_terminals(&output.dwa, grammar.num_terminals as usize);
        let emitted_count = emitted.iter().filter(|slot| **slot).count();
        let profile = &output.profile;
        eprintln!(
            "BOUNDARY_WALK shard={index} states={} trans={} tokens={} emitted_terms={} \
                 setup_ms={:.3} walk_ms={:.3} id_map_ms={:.3} dwa_ms={:.3} det_ms={:.3} min_ms={:.3} compact_ms={:.3}",
            output.dwa.num_states(),
            output.dwa.num_transitions(),
            tokens.len(),
            emitted_count,
            profile.setup_ms,
            profile.walk_ms,
            profile.id_map_ms,
            profile.terminal_dwa_ms,
            profile.determinize_ms,
            profile.minimize_ms,
            profile.compact_ms,
        );
        if index == 0 {
            assert!(tokens.is_empty(), "core shard must be empty, got {}", tokens.len());
            assert_eq!(output.dwa.num_states(), 1, "empty core shard DWA shape");
        } else {
            if tokens != oracle {
                let missing = oracle.difference(&tokens).copied().collect::<Vec<_>>();
                let extra = tokens.difference(&oracle).copied().collect::<Vec<_>>();
                eprintln!(
                    "BOUNDARY_WALK oracle_diff missing={missing:?} extra={extra:?}"
                );
            }
            assert_eq!(tokens.len(), 143, "dispatch shard token count");
            assert_eq!(tokens, oracle, "dispatch shard must match the oracle set");
            assert_eq!(
                output.dwa.num_states(),
                26,
                "dispatch crossing DWA must be the true-minimal 26-state form"
            );
        }
    }
}

/// Production-path selected10 crossing gate (no strict reference).
#[test]
#[ignore]
fn selected10_boundary_walk_crossing() {
    run_selected10_boundary_walk_crossing();
}

/// Exact weighted-language gate for scoped vocabulary partitioning. The
/// full-vocabulary all-L2P build is the conservative reference. Reconcile
/// both private TSID/token quotients into one common coordinate before
/// asking the acyclic weighted-DWA equivalence checker for a distinguishing
/// terminal word.
#[test]
#[ignore]
fn selected10_partitioned_scoped_matches_full_weighted_relation() {
    require_release_selected10_gate(
        "selected10_partitioned_scoped_matches_full_weighted_relation",
    );
    let _env_lock = crate::TEST_ENV_LOCK.lock().unwrap();
    let fixture = load_selected10_outer();
    let component_state_counts = [
        fixture.core.tokenizer.num_states(),
        fixture.dispatch.tokenizer.num_states(),
    ];
    let build = || {
        build_boundary_shard_walks(&BoundaryShardLinkInputs {
            merged_tokenizer: &fixture.composed.tokenizer,
            vocab: &fixture.vocab,
            grammar: &fixture.grammar,
            disallowed_follows: &fixture.disallowed,
            ignore_terminal: fixture.composed.ignore_canonical,
            follow_transparent_ignores: None,
            terminal_offsets: &fixture.composed.table.terminal_offsets,
            leaf_to_immediate: None,
            tokenizer_offsets: &fixture.composed.tokenizer_offsets,
            component_state_counts: &component_state_counts,
            candidate_tokens_by_component: None,
            retain_parent_non_crossing_paths: false,
            walk_plans: None,
        })
        .expect("selected10 scoped boundary build")
        .0
    };

    let full = {
        let _full = EnvVarGuard::set("GLRMASK_DEBUG_SCOPED_FORCE_FULL_VOCAB", "1");
        build()
    };
    let partitioned = {
        let _partitioned = EnvVarGuard::unset("GLRMASK_DEBUG_SCOPED_FORCE_FULL_VOCAB");
        build()
    };
    assert_eq!(full.len(), partitioned.len(), "nonempty shard set differs");
    for (full, partitioned) in full.into_iter().zip(partitioned) {
        let start_component = full.start_component;
        assert_eq!(
            start_component, partitioned.start_component,
            "shard publication order differs",
        );
        let full = crate::compiler::stages::mapped_artifact::MappedArtifact::new(
            full.output.dwa,
            full.output.id_map,
        );
        let partitioned = crate::compiler::stages::mapped_artifact::MappedArtifact::new(
            partitioned.output.dwa,
            partitioned.output.id_map,
        );
        let reconciled: crate::compiler::stages::mapped_artifact::MappedArtifact<(DWA, DWA)> =
            (full, partitioned).into();
        let ((full, partitioned), _) = reconciled.into_parts();
        match crate::automata::weighted_u32::equivalence::find_difference(
            &full,
            &partitioned,
        ) {
            Ok(None) => {}
            Ok(Some(word)) => panic!(
                "partitioned scoped shard {start_component} changed weighted language at terminal word {word:?}",
            ),
            Err(error) => panic!("weighted equivalence checker failed: {error:?}"),
        }
    }
}

/// Durable two-stage strict acceptance gate for boundary shards. Arms the
/// factor strict reference for BOTH the shard label (`boundary_shard`,
/// stage-2 completed-artifact compare in each walk) and the
/// shared-equivalence label (`boundary_shard_test`, stage-1
/// factored-vs-exact class assert), then runs the production crossing
/// flow. The selector assertions below prove both stages are ENABLED, not
/// that they execute (they share the test's hardcoded label names, so a
/// construction-side label rename would still pass them while disabling a
/// stage). Execution is proven by the stage-1 and stage-2 `differs=false`
/// markers, which the gate runner script checks in the captured output.
#[test]
#[ignore]
fn selected10_boundary_shard_two_stage_strict_reference() {
    let _env_lock = crate::TEST_ENV_LOCK.lock().unwrap();
    let _strict = EnvVarGuard::set(
        "GLRMASK_L2P_FIRST_BYTE_VOCAB_FACTOR_STRICT_REFERENCE",
        "boundary_shard,boundary_shard_test",
    );
    // The stage-2 completed-artifact marker only prints when L2P timing
    // profiling is on (`l2p_timing_profile_enabled`); arm it in-process
    // so the gate markers appear without relying on ambient env.
    let _timing = EnvVarGuard::set("GLRMASK_PROFILE_L2P_TIMING", "1");
    assert!(
        tdwa::l2p::l2p_first_byte_vocab_factor_strict_reference_enabled_for_partition(
            "boundary_shard",
        ),
        "stage-2 shard strict reference must be armed for boundary_shard",
    );
    assert!(
        tdwa::l2p::l2p_first_byte_vocab_factor_strict_reference_enabled_for_partition(
            "boundary_shard_test",
        ),
        "stage-1 shared-equivalence strict reference must be armed for boundary_shard_test",
    );
    run_selected10_boundary_walk_crossing();
}

/// Convergence-bound regression gate (Phase 2 step 4): the selected10
/// outer recursive table's control elimination provably diverges (cyclic
/// reduce graph defeats the DFS visiting set), so it must decline via
/// the convergence budget quickly instead of stalling the link.
#[test]
#[ignore]
fn recursive_table_convergence_bound_fails_fast() {
    let fixture = load_selected10_outer();
    let dyn_children = [CompiledSubgrammarInput {
        placeholder_terminal: terminal_id(&fixture.core, "PROGRAMMATIC_TOOL_SUFFIX"),
        additional_placeholder_terminals: &[],
        constraint: &fixture.dispatch,
    }];
    let dynamic = compose_constraints_owned_parent_segmented(
        fixture.core.clone(),
        &dyn_children,
        &fixture.vocab,
        SegmentedBoundaryBackend::Dynamic,
    )
    .expect("dynamic compose")
    .constraint;
    let started = Instant::now();
    let error = dynamic
        .recursive_control_eliminated_parser_table()
        .expect_err("outer recursive table must decline via the convergence budget");
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    eprintln!("BOUNDARY_INSTALL convergence_decline_ms={elapsed_ms:.3} error={error}");
    assert!(
        control_elimination_budget_exhausted(&error),
        "decline must be budget exhaustion, got: {error}"
    );
    assert!(elapsed_ms < 15_000.0, "decline must fail fast, took {elapsed_ms:.0} ms");
}

fn phase1_root() -> std::path::PathBuf {
    use std::path::Path;

    let root = std::env::var("PHASE1_DIR").unwrap_or_else(|_| {
        "/Users/isaacbreen/Projects2/temp/2026-09/glrmask-selected10-cache-v29".to_string()
    });
    Path::new(&root).to_path_buf()
}

/// Phase-1 cache vocab path, reusing the existing `load_selected10_outer`
/// pattern (PHASE1_VOCAB override, else PHASE1_DIR/vocab_dump.bin).
/// No new hardcoded fallbacks beyond the established default root.
fn phase1_vocab_path() -> String {
    std::env::var("PHASE1_VOCAB")
        .unwrap_or_else(|_| phase1_root().join("vocab_dump.bin").to_string_lossy().into_owned())
}

fn phase1_vocab() -> Vocab {
    load_vocab(&phase1_vocab_path())
}

/// Setup manifest for a prepared install pair: strict structured JSON
/// emitted AFTER all pins pass. Never written on failure (no vacuous
/// pass). Schema `install-prepare/2`: schema/test/shards (structured
/// start_component + candidate_tokens arrays), component_state_counts,
/// scenario byte strings, scenario_lengths. Parsed + validated by the
/// window test (never treated as an opaque string).
fn write_install_prepare_manifest(
    dir: &std::path::Path,
    test_name: &str,
    shards: &[(u32, usize)],
    component_state_counts: &[u32],
    vocab: &Vocab,
) {
    assert!(!shards.is_empty(), "{test_name}: refusing empty-shard manifest");
    let manifest = serde_json::json!({
        "schema": "install-prepare/2",
        "test": test_name,
        "shards": shards.iter().map(|(component, candidates)| {
            serde_json::json!({"start_component": component, "candidate_tokens": candidates})
        }).collect::<Vec<_>>(),
        "component_state_counts": component_state_counts,
        "scenarios": INSTALL_DIFFERENTIAL_PREFIXES.iter().map(|prefix| {
            String::from_utf8_lossy(prefix).into_owned()
        }).collect::<Vec<_>>(),
        "scenario_lengths": INSTALL_DIFFERENTIAL_PREFIXES.iter().map(|prefix| {
            prefix.len() + 1
        }).collect::<Vec<_>>(),
        "vocab_entries": vocab.entries_map().len(),
    });
    std::fs::write(
        dir.join("MANIFEST.json"),
        serde_json::to_string_pretty(&manifest).expect("serialize manifest"),
    )
    .expect("write manifest");
    std::fs::write(dir.join("PREPARE_OK"), b"ok").expect("write marker");
}

/// Save + reload round-trip for an installed pair: persists static and
/// dynamic with the caller's vocab, reloads both, reasserts the
/// backend/packing pins on the RELOADED static, compares fresh-vs-reload
/// initial full masks per backend, and returns the reloaded pair.
/// Stops (no vacuous pass) if serialization loses any pin.
fn save_reload_install_pair(
    test_name: &str,
    dir: &std::path::Path,
    dynamic: &Constraint,
    static_comp: &Constraint,
    component_state_counts: &[u32],
    packing_name: &str,
    vocab: &Vocab,
) -> (Constraint, Constraint) {
    let raw_static = static_comp.save();
    let raw_dynamic = dynamic.save();
    assert!(!raw_static.is_empty(), "{test_name}: refusing empty static.bin");
    assert!(!raw_dynamic.is_empty(), "{test_name}: refusing empty dynamic.bin");
    std::fs::write(dir.join("static.bin"), &raw_static).expect("write static.bin");
    std::fs::write(dir.join("dynamic.bin"), &raw_dynamic).expect("write dynamic.bin");
    // Same-vocabulary reload with the caller's fixture vocab.
    let re_static = Constraint::load_with_vocab(&raw_static, vocab).expect("reload static");
    let re_dynamic = Constraint::load_with_vocab(&raw_dynamic, vocab).expect("reload dynamic");
    // Backend pins on the RELOADED static (serialization must not lose them).
    assert!(
        re_static.uses_compact_segmented_parser_runtime(),
        "{test_name}: reloaded static must stay on the compact runtime",
    );
    let mut re_shards = 0usize;
    for (index, component) in re_static
        .static_dynamic_overlay
        .as_ref()
        .expect("reloaded overlay")
        .segmented_parser_components
        .iter()
        .enumerate()
    {
        if let Some(shard) = component.boundary.as_ref() {
            re_shards += 1;
            assert!(
                matches!(
                    shard.backend,
                    crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                ),
                "{test_name}: reloaded component {index} must be a StaticParser shard",
            );
        }
    }
    assert!(re_shards > 0, "{test_name}: reloaded static must carry shards");
    assert_scoped_tokenizer_packing(packing_name, &re_static, component_state_counts);
    // Fresh-vs-reload initial full masks per backend (exact vectors).
    assert_eq!(
        static_comp.start().mask(),
        re_static.start().mask(),
        "{test_name}: static initial mask must survive reload",
    );
    assert_eq!(
        dynamic.start().mask(),
        re_dynamic.start().mask(),
        "{test_name}: dynamic initial mask must survive reload",
    );
    (re_dynamic, re_static)
}

/// Optional prepare-output dir for install-gate tests: when the named env
/// var is set, its value is the output directory (created fresh; refuses
/// non-empty dirs so stale bins can never pass).
fn install_prepare_dir(env_key: &str) -> Option<std::path::PathBuf> {
    std::env::var_os(env_key).map(|dir| {
        let dir = std::path::PathBuf::from(dir);
        if dir.exists() {
            let count = std::fs::read_dir(&dir).expect("read prepare dir").count();
            assert_eq!(count, 0, "{env_key}: refusing non-empty prepare dir");
        } else {
            std::fs::create_dir_all(&dir).expect("create prepare dir");
        }
        dir
    })
}

/// Step-4 install gate: build walk shards through the link orchestration,
/// publish against the spliced control-free boundary table (unbound
/// slots emptied), install by replacing the dynamic composition's
/// dynamic shards, and require mask-for-mask equality with the dynamic
/// backend over a byte-prefix corpus. No fallback: a nonzero control
/// count or a publish failure is a hard gate failure.
#[test]
#[ignore]
fn selected10_walk_shard_install_matches_dynamic() {
    require_release_selected10_gate("selected10_walk_shard_install_matches_dynamic");
    let fixture = load_selected10_outer();
    let dyn_children = [CompiledSubgrammarInput {
        placeholder_terminal: terminal_id(&fixture.core, "PROGRAMMATIC_TOOL_SUFFIX"),
        additional_placeholder_terminals: &[],
        constraint: &fixture.dispatch,
    }];
    let compose_started = Instant::now();
    let dynamic = compose_constraints_owned_parent_segmented(
        fixture.core.clone(),
        &dyn_children,
        &fixture.vocab,
        SegmentedBoundaryBackend::Dynamic,
    )
    .expect("dynamic compose")
    .constraint;
    eprintln!(
        "BOUNDARY_INSTALL dynamic_compose_ms={:.3}",
        compose_started.elapsed().as_secs_f64() * 1000.0,
    );

    let component_state_counts =
        [fixture.core.tokenizer.num_states(), fixture.dispatch.tokenizer.num_states()];
    let link_started = Instant::now();
    let (built, link_profile) = build_boundary_shard_walks(&BoundaryShardLinkInputs {
        merged_tokenizer: &fixture.composed.tokenizer,
        vocab: &fixture.vocab,
        grammar: &fixture.grammar,
        disallowed_follows: &fixture.disallowed,
        ignore_terminal: fixture.composed.ignore_canonical,
        follow_transparent_ignores: None,
        terminal_offsets: &fixture.composed.table.terminal_offsets,
        leaf_to_immediate: None,
        tokenizer_offsets: &fixture.composed.tokenizer_offsets,
        component_state_counts: &component_state_counts,
        candidate_tokens_by_component: None,
        // Core and dispatch schemas are never start-nullable.
        retain_parent_non_crossing_paths: false,
        walk_plans: None,
    })
    .expect("walk shards");
    eprintln!(
        "BOUNDARY_INSTALL walks shared_ms={:.3} flat_ms={:.3} shards={:?} link_ms={:.3}",
        link_profile.shared_wall_ms,
        link_profile.flat_ms,
        built
            .iter()
            .map(|shard| (
                shard.start_component,
                shard.candidate_tokens.len(),
                shard.output.profile.walk_ms,
            ))
            .collect::<Vec<_>>(),
        link_started.elapsed().as_secs_f64() * 1000.0,
    );
    assert_eq!(built.len(), 1, "only the dispatch shard is nonempty");
    assert_eq!(built[0].start_component, 1);
    assert_eq!(built[0].candidate_tokens.len(), 143);

    // Outer unbound slots: dispatch's TOOL_ARGS_SLOT_* (component 1, none
    // bound — the 10-way segmented dispatch link is unavailable) plus any
    // unbound parent slots (core binds its only slot; expect none).
    let mut unbound: Vec<(usize, TerminalID)> = unbound_slot_terminals(&fixture.core, &[])
        .into_iter()
        .map(|local| (0usize, local))
        .collect();
    unbound.extend(
        unbound_slot_terminals(&fixture.dispatch, &[]).into_iter().map(|local| (1usize, local)),
    );
    let (boundary_table, table_ms) =
        prepare_spliced_boundary_table("outer", &fixture.composed, &unbound);
    eprintln!("BOUNDARY_INSTALL table_ms={table_ms:.3}");
    let mut published = Vec::with_capacity(built.len());
    for shard in built {
        let work = WalkBoundaryShardWork {
            start_component: shard.start_component as u32,
            terminal_automaton: TerminalAutomaton::Dwa(shard.output.dwa),
            id_map: shard.output.id_map,
            candidate_tokens: shard.candidate_tokens.into_iter().collect::<Vec<_>>().into(),
        };
        let (shard, profile) = publish_walk_boundary_shard_work(
            work,
            &boundary_table,
            &fixture.composed.tokenizer_offsets,
            &component_state_counts,
        )
        .expect("publish walk shard");
        eprintln!(
            "BOUNDARY_INSTALL shard={} parser_states={} parser_trans={} candidates={} uses_composed={} terms={} templates_ms={:.3} materialize_ms={:.3} normalize_ms={:.3}",
            shard.start_component,
            profile.parser_states,
            profile.parser_trans,
            shard.candidate_tokens.len(),
            shard.boundary.uses_composed_tsid_coordinate,
            profile.terms,
            profile.templates_ms,
            profile.materialize_ms,
            profile.normalize_ms,
        );
        published.push(shard);
    }

    let mut static_comp = dynamic.clone();
    install_published_static_boundary_shards(
        static_comp.static_dynamic_overlay.as_mut().expect("overlay"),
        published,
    )
    .expect("install");
    assert!(
        static_comp.uses_compact_segmented_parser_runtime(),
        "installed composition must stay on the compact runtime",
    );
    assert_scoped_tokenizer_packing("outer", &static_comp, &component_state_counts);
    // Optional prepare: save/reload round-trip + pins, then return BEFORE
    // the in-process differential (the prepared-pair differential test
    // owns windowed mask comparison). Default path unchanged.
    if let Some(dir) = install_prepare_dir("BOUNDARY_INSTALL_PREPARE_OUT_DIR") {
        let shard_ids: Vec<(u32, usize)> = static_comp
            .static_dynamic_overlay
            .as_ref()
            .expect("overlay")
            .segmented_parser_components
            .iter()
            .filter_map(|component| {
                component.boundary.as_ref().map(|shard| {
                    (
                        shard.start_component,
                        shard.candidate_tokens.as_ref().map(|tokens| tokens.len()).unwrap_or(0),
                    )
                })
            })
            .collect();
        save_reload_install_pair(
            "selected10_walk_shard_install_matches_dynamic",
            &dir,
            &dynamic,
            &static_comp,
            &component_state_counts,
            "outer",
            &fixture.vocab,
        );
        write_install_prepare_manifest(
            &dir,
            "selected10_walk_shard_install_matches_dynamic",
            &shard_ids,
            &component_state_counts,
            &fixture.vocab,
        );
        return;
    }
    run_install_differential(&dynamic, &mut static_comp, false);
}

/// Byte-prefix corpus for the install differential. The expected coverage
/// is 248 mask positions (initial + one per committed byte across all 12
/// scenarios); the windowed prepared-pair test derives its actual count
/// from this list and reports any deviation instead of forcing 248.
const INSTALL_DIFFERENTIAL_PREFIXES: [&[u8]; 12] = [
    b"",
    b"const x = tools",
    b"const x = tools.tool_0(",
    b"const x = tools.tool_3({",
    b"const x = tools.tool_3({\"p\": ",
    b"const x = tools.tool_9({\"outer\": {\"inner\": [1, ",
    b"function f(a, b) { return ",
    b"for (let i = 0; i < ",
    b"const s = \"hello",
    b"const x = { a: ",
    b"const x = tools.",
    b"tools",
];

/// Byte-prefix mask differential between a dynamic composition and its
/// walk-shard-installed twin. `fallback` labels runs where convergence
/// decline left the dynamic shards in place (vacuous by construction).
fn run_install_differential(
    dynamic: &Constraint,
    static_comp: &mut Constraint,
    fallback: bool,
) {

    let prefixes: Vec<&[u8]> = INSTALL_DIFFERENTIAL_PREFIXES.to_vec();
    let mut positions = 0usize;
    let mut checksum: u64 = 0xcbf29ce484222325;
    for (scenario, prefix) in prefixes.iter().enumerate() {
        let scenario_started = Instant::now();
        let mut st_dyn = dynamic.start();
        let mut st_static = static_comp.start();
        let mut check = |st_dyn: &mut crate::runtime::ConstraintState<'_>,
                         st_static: &mut crate::runtime::ConstraintState<'_>,
                         scenario: usize,
                         consumed: usize| {
            let mask_dyn = st_dyn.mask();
            let mask_static = st_static.mask();
            assert_eq!(
                mask_static,
                mask_dyn,
                "mask mismatch scenario={scenario} consumed={consumed}",
            );
            for (index, &word) in mask_dyn.iter().enumerate() {
                checksum ^= (word as u64).wrapping_add(index as u64);
                checksum = checksum.wrapping_mul(0x100000001b3);
            }
            positions += 1;
        };
        check(&mut st_dyn, &mut st_static, scenario, 0);
        for (consumed, &byte) in prefix.iter().enumerate() {
            let r_dyn = st_dyn.commit_bytes(&[byte]);
            let r_static = st_static.commit_bytes(&[byte]);
            assert_eq!(
                r_dyn.is_ok(),
                r_static.is_ok(),
                "commit divergence scenario={scenario} consumed={consumed}",
            );
            if r_dyn.is_err() {
                break;
            }
            check(&mut st_dyn, &mut st_static, scenario, consumed + 1);
        }
        eprintln!(
            "BOUNDARY_INSTALL scenario={scenario} len={} ms={:.3}",
            prefix.len(),
            scenario_started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    eprintln!(
        "BOUNDARY_INSTALL fallback={fallback} positions={positions} checksum={checksum:016x}"
    );
    assert!(positions > 100, "corpus must cover real positions");
}

/// Windowed differential over a PREPARED pair: reloads both backends from
/// `GLRMASK_INSTALL_DIFF_PREPARE_DIR`, validates the structured prepare
/// manifest (schema `install-prepare/2`, known test name, PREPARE_OK,
/// 12 scenario identities, nonempty shards exactly matching the reloaded
/// component IDs/candidate counts, packing against manifest counts),
/// reasserts pins, then in ONE pass commits through `start+count-1`,
/// comparing complete masks and collecting hashes at wanted positions
/// (initial position 0 plus each committed byte — exactly the positions
/// `run_install_differential` checks). Commit behavior is STRICTER than
/// the original: the old `fallback` arg labels convergence-decline runs
/// (where both-error breaks the byte loop); here every commit in the
/// validated fixture must succeed. One strict JSON object plus the
/// `INSTALL_WINDOW_OK` marker are emitted only AFTER full-vector
/// equality. Selected by env: prepare dir (required), scenario index
/// (required), window start/count (required, 1..=8). No new corpus.
/// The global 248 total is asserted from the shared prefix list.
#[test]
#[ignore]
fn prepared_install_pair_windowed_differential() {
    let _env_lock = crate::TEST_ENV_LOCK.lock().unwrap();
    let dir = std::env::var("GLRMASK_INSTALL_DIFF_PREPARE_DIR")
        .expect("GLRMASK_INSTALL_DIFF_PREPARE_DIR must name a prepared pair dir");
    let dir = std::path::PathBuf::from(dir);
    let scenario: usize = std::env::var("GLRMASK_INSTALL_DIFF_SCENARIO")
        .expect("GLRMASK_INSTALL_DIFF_SCENARIO required")
        .parse()
        .expect("scenario integer");
    let start: usize = std::env::var("GLRMASK_INSTALL_DIFF_START")
        .expect("GLRMASK_INSTALL_DIFF_START required")
        .parse()
        .expect("start integer");
    let count: usize = std::env::var("GLRMASK_INSTALL_DIFF_COUNT")
        .expect("GLRMASK_INSTALL_DIFF_COUNT required")
        .parse()
        .expect("count integer");
    assert!((1..=8).contains(&count), "window admits 1..=8 comparisons");
    let prefix = INSTALL_DIFFERENTIAL_PREFIXES
        .get(scenario)
        .expect("scenario indexes the original 12-prefix corpus");
    assert_eq!(
        INSTALL_DIFFERENTIAL_PREFIXES
            .iter()
            .map(|prefix| prefix.len() + 1)
            .sum::<usize>(),
        248,
        "global corpus must total 248 positions",
    );
    let end = start.checked_add(count).expect("window end overflow");
    assert!(
        end <= prefix.len() + 1,
        "window [{start}..{end}) out of bounds (scenario len {})",
        prefix.len() + 1,
    );
    let want: Vec<usize> = (start..end).collect();
    // Structured manifest validation (parsed, never opaque).
    let manifest_raw =
        std::fs::read_to_string(dir.join("MANIFEST.json")).expect("prepare manifest present");
    let manifest: serde_json::Value =
        serde_json::from_str(&manifest_raw).expect("manifest must be valid JSON");
    assert_eq!(
        manifest.get("schema").and_then(|value| value.as_str()),
        Some("install-prepare/2"),
        "manifest schema must be install-prepare/2",
    );
    let pair_test = manifest
        .get("test")
        .and_then(|value| value.as_str())
        .expect("manifest test name");
    assert!(
        [
            "selected10_walk_shard_install_matches_dynamic",
            "prepared_transfer_row918_no_global_elimination"
        ]
        .contains(&pair_test),
        "manifest test name must be a known install gate, got {pair_test:?}",
    );
    assert_eq!(
        std::fs::read(dir.join("PREPARE_OK")).expect("PREPARE_OK present"),
        b"ok",
        "PREPARE_OK marker must read ok",
    );
    let manifest_scenarios = manifest
        .get("scenarios")
        .and_then(|value| value.as_array())
        .expect("manifest scenarios array");
    assert_eq!(
        manifest_scenarios.len(),
        12,
        "manifest must carry the 12 scenario identities",
    );
    for (index, expected) in INSTALL_DIFFERENTIAL_PREFIXES.iter().enumerate() {
        let actual = manifest_scenarios[index]
            .as_str()
            .expect("scenario identity must be a string");
        assert_eq!(
            actual,
            String::from_utf8_lossy(expected),
            "manifest scenario {index} must match the corpus",
        );
    }
    let manifest_counts = manifest
        .get("component_state_counts")
        .and_then(|value| value.as_array())
        .expect("manifest component_state_counts");
    // Reload both backends with the shared phase-1 vocab; reassert pins.
    let vocab = phase1_vocab();
    let raw_static = std::fs::read(dir.join("static.bin")).expect("read static.bin");
    let raw_dynamic = std::fs::read(dir.join("dynamic.bin")).expect("read dynamic.bin");
    assert!(!raw_static.is_empty() && !raw_dynamic.is_empty(), "refusing empty bins");
    let static_comp =
        Constraint::load_with_vocab(&raw_static, &vocab).expect("reload static");
    let dynamic =
        Constraint::load_with_vocab(&raw_dynamic, &vocab).expect("reload dynamic");
    assert!(
        static_comp.uses_compact_segmented_parser_runtime(),
        "reloaded static must stay on the compact runtime",
    );
    let mut actual_shards: Vec<(u32, usize)> = Vec::new();
    for (index, component) in static_comp
        .static_dynamic_overlay
        .as_ref()
        .expect("overlay")
        .segmented_parser_components
        .iter()
        .enumerate()
    {
        if let Some(shard) = component.boundary.as_ref() {
            assert!(
                matches!(
                    shard.backend,
                    crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                ),
                "reloaded component {index} must be a StaticParser shard",
            );
            actual_shards.push((
                shard.start_component,
                shard.candidate_tokens.as_ref().map(|tokens| tokens.len()).unwrap_or(0),
            ));
        }
    }
    assert!(!actual_shards.is_empty(), "reloaded static must carry shards");
    let manifest_shards = manifest
        .get("shards")
        .and_then(|value| value.as_array())
        .expect("manifest shards array");
    let mut manifest_ids: Vec<(u32, usize)> = manifest_shards
        .iter()
        .map(|entry| {
            (
                entry
                    .get("start_component")
                    .and_then(|value| value.as_u64())
                    .expect("shard start_component u64") as u32,
                entry
                    .get("candidate_tokens")
                    .and_then(|value| value.as_u64())
                    .expect("shard candidate_tokens u64") as usize,
            )
        })
        .collect();
    manifest_ids.sort();
    actual_shards.sort();
    assert_eq!(
        manifest_ids, actual_shards,
        "manifest shards must exactly match reloaded component IDs/candidate counts",
    );
    let manifest_counts: Vec<u32> = manifest_counts
        .iter()
        .map(|value| value.as_u64().expect("component count u64") as u32)
        .collect();
    assert_scoped_tokenizer_packing("windowed", &static_comp, &manifest_counts);
    // Single pass: commit through start+count-1, compare full vectors
    // and collect hashes at wanted positions only.
    let mut st_dyn = dynamic.start();
    let mut st_static = static_comp.start();
    let mut positions: Vec<serde_json::Value> = Vec::with_capacity(count);
    let hash_one = |mask: &[u32]| {
        let mut hash: u64 = 0xcbf29ce484222325;
        for (index, &word) in mask.iter().enumerate() {
            hash ^= (word as u64).wrapping_add(index as u64);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        format!("{hash:016x}")
    };
    let record = |consumed: usize,
                      st_dyn: &mut crate::runtime::ConstraintState<'_>,
                      st_static: &mut crate::runtime::ConstraintState<'_>,
                      positions: &mut Vec<serde_json::Value>| {
        let mask_dyn = st_dyn.mask();
        let mask_static = st_static.mask();
        assert_eq!(
            mask_static, mask_dyn,
            "mask mismatch scenario={scenario} consumed={consumed}",
        );
        assert!(!mask_dyn.is_empty(), "refusing empty-mask hash");
        let static_hash = hash_one(&mask_static);
        let dynamic_hash = hash_one(&mask_dyn);
        assert_eq!(static_hash.len(), 16);
        assert_eq!(dynamic_hash.len(), 16);
        assert!(
            static_hash.chars().all(|char| char.is_ascii_hexdigit()),
            "static hash must be 16 hex digits",
        );
        assert!(
            dynamic_hash.chars().all(|char| char.is_ascii_hexdigit()),
            "dynamic hash must be 16 hex digits",
        );
        assert_eq!(static_hash, dynamic_hash);
        positions.push(serde_json::json!({
            "index": consumed,
            "static_hash": static_hash,
            "dynamic_hash": dynamic_hash,
        }));
    };
    if want.contains(&0) {
        record(0, &mut st_dyn, &mut st_static, &mut positions);
    }
    for (consumed, &byte) in prefix.iter().enumerate() {
        let r_dyn = st_dyn.commit_bytes(&[byte]);
        let r_static = st_static.commit_bytes(&[byte]);
        assert_eq!(
            r_dyn.is_ok(),
            r_static.is_ok(),
            "commit divergence scenario={scenario} consumed={consumed}",
        );
        // Strict: every commit in the validated fixture must succeed
        // (stronger than the original both-error break).
        assert!(
            r_dyn.is_ok(),
            "strict window: commit must succeed scenario={scenario} consumed={consumed}",
        );
        if want.contains(&(consumed + 1)) {
            record(consumed + 1, &mut st_dyn, &mut st_static, &mut positions);
        }
        if consumed + 1 >= end {
            break;
        }
    }
    assert_eq!(positions.len(), count, "every requested position compared");
    // One strict JSON object + marker, only AFTER full-vector equality.
    let doc = serde_json::json!({
        "schema": "install-window/1",
        "pair_test": pair_test,
        "scenario": scenario,
        "prefix": String::from_utf8_lossy(prefix).into_owned(),
        "start": start,
        "count": count,
        "total_positions": prefix.len() + 1,
        "positions": positions,
    });
    println!("{}", serde_json::to_string(&doc).expect("serialize window"));
    eprintln!(
        "INSTALL_WINDOW_OK scenario={scenario} start={start} count={count} total={}",
        prefix.len() + 1,
    );
}

/// Dispatch-parent grammar source for the inner-link sweep. Mirrors
/// `dispatcher_literal_names_parent_source` in
/// `examples/static_link_measure.rs` (10 `.tool_i(` slots over the same
/// placeholder token IDs).
fn dispatch_parent_source() -> String {
    let mut source = String::from("start suffix;\n");
    for index in 0..10 {
        source.push_str(&format!(
            "t TOOL_ARGS_SLOT_{index} ::= @token({});\n",
            128_320 + 10 + index
        ));
    }
    source.push_str("nt suffix ::=\n    ");
    for index in 0..10 {
        if index != 0 {
            source.push_str("\n  | ");
        }
        source.push_str(&format!(r#"".tool_{index}(" TOOL_ARGS_SLOT_{index} ")""#));
    }
    source.push_str(";\n");
    source
}

/// Per-link stage timing record for the sweep table.
struct LinkTiming {
    name: String,
    dyn_compose_ms: f64,
    low_level_ms: f64,
    shared_ms: f64,
    flat_ms: f64,
    walk_ms: f64,
    det_ms: f64,
    min_ms: f64,
    shards_built: usize,
    table_ms: f64,
    templates_ms: f64,
    parser_ms: f64,
    install_ms: f64,
    installed: usize,
    diff_positions: usize,
    diff_mismatches: usize,
    diff_checksum: u64,
    diff_ms: f64,
}

impl LinkTiming {
    fn total_link_ms(&self) -> f64 {
        self.shared_ms
            + self.walk_ms
            + self.table_ms
            + self.templates_ms
            + self.parser_ms
            + self.install_ms
    }
}

/// Seeded RNG token-walk differential between two compositions (dynamic
/// reference vs installed/fallback twin). Language-agnostic: drives on
/// the dynamic masks. Returns (positions, mismatches, checksum).
fn rng_choose_token(mask: &[u32], rng: &mut u64) -> Option<u32> {
    let allowed: usize = mask.iter().map(|word| word.count_ones() as usize).sum();
    if allowed == 0 {
        return None;
    }
    *rng ^= *rng >> 12;
    *rng ^= *rng << 25;
    *rng ^= *rng >> 27;
    let draw = rng.wrapping_mul(0x2545_f491_4f6c_dd1d);
    let mut rank = draw as usize % allowed;
    for (word_index, &word) in mask.iter().enumerate() {
        let count = word.count_ones() as usize;
        if rank >= count {
            rank -= count;
            continue;
        }
        let mut live = word;
        for _ in 0..rank {
            live &= live - 1;
        }
        let bit = live.trailing_zeros() as usize;
        return Some((word_index * 32 + bit) as u32);
    }
    None
}

fn rng_differential(
    dynamic: &Constraint,
    other: &Constraint,
    steps: usize,
    seed: u64,
) -> (usize, usize, u64, Option<String>) {
    fn mask_bits(mask: &[u32]) -> Vec<u32> {
        let mut out = Vec::new();
        for (word_index, &word) in mask.iter().enumerate() {
            let mut live = word;
            while live != 0 {
                let bit = live.trailing_zeros() as usize;
                out.push((word_index * 32 + bit) as u32);
                live &= live - 1;
            }
        }
        out
    }
    let mut rng = seed;
    let mut checksum: u64 = 0xcbf29ce484222325;
    let mut positions = 0usize;
    let mut mismatches = 0usize;
    let mut first: Option<String> = None;
    let mut st_dyn = dynamic.start();
    let mut st_other = other.start();
    let mut check = |st_dyn: &mut crate::runtime::ConstraintState<'_>,
                     st_other: &mut crate::runtime::ConstraintState<'_>,
                     step_index: usize,
                     first: &mut Option<String>| {
        let mask_dyn = st_dyn.mask();
        let mask_other = st_other.mask();
        for (index, &word) in mask_dyn.iter().enumerate() {
            checksum ^= (word as u64).wrapping_add(index as u64);
            checksum = checksum.wrapping_mul(0x100000001b3);
        }
        positions += 1;
        if mask_dyn != mask_other {
            mismatches += 1;
            if first.is_none() {
                let dyn_bits = mask_bits(&mask_dyn);
                let other_bits = mask_bits(&mask_other);
                let dyn_set: BTreeSet<u32> = dyn_bits.into_iter().collect();
                let other_set: BTreeSet<u32> = other_bits.into_iter().collect();
                let dyn_only: Vec<u32> =
                    dyn_set.difference(&other_set).copied().take(12).collect();
                let other_only: Vec<u32> =
                    other_set.difference(&dyn_set).copied().take(12).collect();
                *first = Some(format!(
                    "mask step={step_index} dyn_admits={} other_admits={} dyn_only={dyn_only:?} other_only={other_only:?}",
                    dyn_set.len(),
                    other_set.len(),
                ));
            }
        }
    };
    check(&mut st_dyn, &mut st_other, 0, &mut first);
    let mut path: Vec<u32> = Vec::new();
    for step in 0..steps {
        let mask_dyn = st_dyn.mask();
        let Some(token) = rng_choose_token(&mask_dyn, &mut rng) else {
            break;
        };
        path.push(token);
        let r_dyn = st_dyn.commit_token(token);
        let r_other = st_other.commit_token(token);
        if r_dyn.is_ok() != r_other.is_ok() {
            mismatches += 1;
            if first.is_none() {
                first = Some(format!(
                    "commit step={} token={token} dyn_ok={} other_ok={}",
                    step + 1,
                    r_dyn.is_ok(),
                    r_other.is_ok(),
                ));
            }
            break;
        }
        if r_dyn.is_err() {
            break;
        }
        check(&mut st_dyn, &mut st_other, step + 1, &mut first);
    }
    if mismatches > 0 {
        eprintln!("BOUNDARY_LINK_PATH positions={positions} path={path:?}");
    }
    (positions, mismatches, checksum, first)
}

/// Run the full walk-shard link for one composition and time every stage.
/// Publication uses the production signed-transfer route (same fragment
/// library, composer, and publisher as `build_walk_static_boundary_link`),
/// so this sweep validates what production installs. There is no table
/// stage on this route (`table_ms` is exactly 0).
fn link_and_time(
    name: &str,
    parent: &Constraint,
    slots: &[(String, &Constraint)],
    vocab: &Vocab,
    diff_steps: usize,
) -> LinkTiming {
    let inputs: Vec<CompiledSubgrammarInput> = slots
        .iter()
        .map(|(slot, child)| CompiledSubgrammarInput {
            placeholder_terminal: terminal_id(parent, slot),
            additional_placeholder_terminals: &[],
            constraint: child,
        })
        .collect();
    let compose_started = Instant::now();
    let dynamic = compose_constraints_owned_parent_segmented(
        parent.clone(),
        &inputs,
        vocab,
        SegmentedBoundaryBackend::Dynamic,
    )
    .expect("dynamic compose")
    .constraint;
    let dyn_compose_ms = compose_started.elapsed().as_secs_f64() * 1000.0;
    let low_level_started = Instant::now();
    let composed = low_level_compose(parent, &inputs);
    let grammar = analyzed_grammar(&composed.table.table, &composed.terminal_names);
    let disallowed = compute_disallowed_follows(&grammar);
    let low_level_ms = low_level_started.elapsed().as_secs_f64() * 1000.0;
    let counts: Vec<u32> = std::iter::once(parent.tokenizer.num_states())
        .chain(inputs.iter().map(|child| child.constraint.tokenizer.num_states()))
        .collect();
    let retain_parent_non_crossing_paths = inputs
        .iter()
        .any(|child| child.constraint.table.embedded_start_nullable());
    let (built, profile) = build_boundary_shard_walks(&BoundaryShardLinkInputs {
        merged_tokenizer: &composed.tokenizer,
        vocab,
        grammar: &grammar,
        disallowed_follows: &disallowed,
        ignore_terminal: composed.ignore_canonical,
        follow_transparent_ignores: None,
        terminal_offsets: &composed.table.terminal_offsets,
        leaf_to_immediate: None,
        tokenizer_offsets: &composed.tokenizer_offsets,
        component_state_counts: &counts,
        candidate_tokens_by_component: None,
        retain_parent_non_crossing_paths,
        walk_plans: None,
    })
    .expect("walk shards");
    let shards_built = built.len();
    let (mut walk_ms, mut det_ms, mut min_ms) = (0.0, 0.0, 0.0);
    for (_, shard_profile) in &profile.per_shard {
        walk_ms += shard_profile.walk_ms;
        det_ms += shard_profile.determinize_ms;
        min_ms += shard_profile.minimize_ms;
    }
    // Signed-transfer shard publication (production route): scoped ordinary
    // transfers + Entry/Finish exports, exact resolution, table-free
    // normalization. No composed/provider table exists here; unbound slots
    // use the same empty-language definition as production
    // (`unbound_link_slot_terminals`). There is no table stage anymore, so
    // `table_ms` below is exactly 0.
    use crate::compiler::composition::boundary::transfer::{
        build_fragment_library, build_signed_link_context, compile_signed_shard_parser,
        publish_signed_shard,
    };
    let global_ignores = component_ignores_are_globally_erasable(parent, &inputs);
    let unbound_set: std::collections::BTreeSet<TerminalID> =
        unbound_link_slot_terminals(parent, &inputs, &composed.table.terminal_offsets)
            .expect("unbound slots")
            .into_iter()
            .collect();
    let signed_context = build_signed_link_context(
        parent,
        &inputs,
        &composed.table.terminal_offsets,
        composed.table.table.num_terminals,
        global_ignores,
        unbound_set,
    )
    .expect("signed link context");
    let table_ms = 0.0;
    let (mut templates_ms, mut parser_ms, mut install_ms) = (0.0, 0.0, 0.0);
    let mut static_comp = dynamic.clone();
    let mut published = Vec::with_capacity(built.len());
    for shard in built {
        let emitted = boundary_emitted_terminals(
            &shard.output.dwa,
            composed.table.table.num_terminals as usize,
        );
        let library = build_fragment_library(
            &signed_context,
            &emitted,
            shard.start_component as u32,
        )
        .expect("fragment library");
        templates_ms += library.templates_ms;
        let compiled = compile_signed_shard_parser(
            &signed_context,
            &library,
            &shard.output.dwa,
            &shard.output.id_map,
            shard.start_component as u32,
        )
        .expect("signed shard compile");
        parser_ms += compiled.compose_ms + compiled.resolve_ms + compiled.normalize_ms;
        let work = WalkBoundaryShardWork {
            start_component: shard.start_component as u32,
            terminal_automaton: TerminalAutomaton::Dwa(shard.output.dwa),
            id_map: shard.output.id_map,
            candidate_tokens: shard
                .candidate_tokens
                .into_iter()
                .collect::<Vec<_>>()
                .into(),
        };
        let (one, _publish_profile) = publish_signed_shard(
            work,
            compiled,
            &composed.tokenizer_offsets,
            &counts,
        )
        .expect("publish signed shard");
        published.push(one);
    }
    let installed = published.len();
    if !published.is_empty() {
        let install_started = Instant::now();
        install_published_static_boundary_shards(
            static_comp.static_dynamic_overlay.as_mut().expect("overlay"),
            published,
        )
        .expect("install");
        install_ms = install_started.elapsed().as_secs_f64() * 1000.0;
    }
    assert!(
        static_comp.uses_compact_segmented_parser_runtime(),
        "{name}: installed composition must stay on the compact runtime",
    );
    assert_scoped_tokenizer_packing(name, &static_comp, &counts);
    let diff_started = Instant::now();
    let (diff_positions, diff_mismatches, diff_checksum, diff_first) =
        rng_differential(&dynamic, &static_comp, diff_steps, 0x9e37_79b9_7f4a_7c15);
    let diff_ms = diff_started.elapsed().as_secs_f64() * 1000.0;
    let timing = LinkTiming {
        name: name.to_string(),
        dyn_compose_ms,
        low_level_ms,
        shared_ms: profile.shared_wall_ms,
        flat_ms: profile.flat_ms,
        walk_ms,
        det_ms,
        min_ms,
        shards_built,
        table_ms,
        templates_ms,
        parser_ms,
        install_ms,
        installed,
        diff_positions,
        diff_mismatches,
        diff_checksum,
        diff_ms,
    };
    eprintln!(
        "BOUNDARY_LINK name={} dyn={:.1} low={:.1} shared={:.1} flat={:.1} walk={:.1} det={:.1} min={:.1} shards={} table={:.1} templates={:.1} parser={:.1} install={:.1} installed={} diff_pos={} diff_mm={} diffck={:016x} diff_ms={:.0} total_link={:.1}",
        timing.name,
        timing.dyn_compose_ms,
        timing.low_level_ms,
        timing.shared_ms,
        timing.flat_ms,
        timing.walk_ms,
        timing.det_ms,
        timing.min_ms,
        timing.shards_built,
        timing.table_ms,
        timing.templates_ms,
        timing.parser_ms,
        timing.install_ms,
        timing.installed,
        timing.diff_positions,
        timing.diff_mismatches,
        timing.diff_checksum,
        timing.diff_ms,
        timing.total_link_ms(),
    );
    if let Some(detail) = &diff_first {
        eprintln!("BOUNDARY_LINK_FIRST_MISMATCH name={name} {detail}");
    }
    timing
}

/// Step-4 link timing sweep: full walk-shard link over every available
/// composition (10× dispatch-parent+schema + 1× core+dispatch outer)
/// with per-stage max/median and >3×-median outlier flags.
#[test]
#[ignore]
fn link_timing_all_compositions() {
    let root = phase1_root();
    let vocab = phase1_vocab();
    let parent =
        Constraint::from_glrm_grammar(&dispatch_parent_source(), &vocab).expect("parent");
    let mut schema_paths: Vec<_> = std::fs::read_dir(&root)
        .expect("read cache dir")
        .filter_map(|entry| {
            let name = entry.ok()?.file_name().to_string_lossy().into_owned();
            name.starts_with("schema-").then_some(name)
        })
        .collect();
    schema_paths.sort();
    assert_eq!(schema_paths.len(), 10, "cache must hold the 10 schema constraints");
    let mut schemas = Vec::with_capacity(10);
    for name in &schema_paths {
        let mut schema =
            Constraint::load(&std::fs::read(root.join(name)).expect("read schema"))
                .expect("load schema");
        restore_component(&mut schema, name);
        schemas.push(schema);
    }
    let mut core =
        Constraint::load(&std::fs::read(root.join("core.bin")).expect("read core.bin"))
            .expect("load core");
    let dispatch_name = std::env::var("PHASE1_DISPATCH")
        .unwrap_or_else(|_| "dispatch-literal.bin".to_string());
    let mut dispatch =
        Constraint::load(&std::fs::read(root.join(&dispatch_name)).expect("read dispatch"))
            .expect("load dispatch");
    restore_component(&mut core, "core");
    restore_component(&mut dispatch, "dispatch");

    let mut timings = Vec::new();
    for (index, schema) in schemas.iter().enumerate() {
        timings.push(link_and_time(
            &format!("inner-{index}"),
            &parent,
            &[(format!("TOOL_ARGS_SLOT_{index}"), schema)],
            &vocab,
            64,
        ));
    }
    // NOTE: no 10-way segmented dispatch link — N-way segmented compose
    // rejects it ("component 2 has a non-functional LR-state relation").
    // The architecture composes dispatch flat and segments only the outer
    // link, so the sweep covers the 10 synthetic inner links + outer.
    timings.push(link_and_time(
        "outer",
        &core,
        &[("PROGRAMMATIC_TOOL_SUFFIX".to_string(), &dispatch)],
        &vocab,
        64,
    ));

    let stages: &[(&str, fn(&LinkTiming) -> f64)] = &[
        ("shared", |t| t.shared_ms),
        ("walk", |t| t.walk_ms),
        ("det", |t| t.det_ms),
        ("min", |t| t.min_ms),
        ("templates", |t| t.templates_ms),
        ("parser", |t| t.parser_ms),
        ("table", |t| t.table_ms),
        ("install", |t| t.install_ms),
        ("total_link", |t| t.total_link_ms()),
    ];
    for (stage, get) in stages {
        let mut values: Vec<f64> = timings.iter().map(get).collect();
        values.sort_by(f64::total_cmp);
        let median = values[values.len() / 2];
        let max = values[values.len() - 1];
        let argmax = timings.iter().map(get).enumerate().max_by(|a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        let flagged = max > 3.0 * median;
        eprintln!(
            "BOUNDARY_STAGE stage={stage} n={} median_ms={median:.1} max_ms={max:.1} max_link={} flagged_3x={flagged}",
            values.len(),
            argmax.map(|(i, _)| timings[i].name.as_str()).unwrap_or("?"),
        );
    }
    let total_mismatches: usize = timings.iter().map(|timing| timing.diff_mismatches).sum();
    let failing: Vec<&str> = timings
        .iter()
        .filter(|timing| timing.diff_mismatches > 0)
        .map(|timing| timing.name.as_str())
        .collect();
    eprintln!("BOUNDARY_SWEEP links={} total_mismatches={total_mismatches} failing={failing:?}", timings.len());
    for timing in &timings {
        assert_eq!(
            timing.installed,
            timing.shards_built,
            "link {} must install every built shard",
            timing.name,
        );
    }
    assert_eq!(total_mismatches, 0, "walk-shard sweep must match DynamicDirect on every link");
}

/// Called-frame non-equivalence pin (Phase 2b reconciliation audit).
///
/// The compact segmented runtime evaluates boundary shards against live
/// GSS stacks in the leaf/provider coordinate, where an in-progress call
/// appears as Call-pushed frames `[.., parent_target, child_start, ..]`.
/// The spliced composed table speaks a different calling convention (child
/// start overlaid on callers, returns via reduce+goto through caller
/// states, fresh appended child-state identities), so its template labels
/// never match called frames: a splice-published shard systematically
/// under-admits wherever the live stack holds an active call. The
/// provider-materialized table speaks the live coordinate with call-site
/// guards and is exact here. The Phase 2 sweep stayed green only because
/// no corpus position needed a child-shard admission (the full sweep
/// passes with every child shard removed, outer with zero shards).
///
/// Fixture: one placeholder called from two states with divergent
/// continuations (`L SUB x | R SUB y`, child `a`), fused exit tokens
/// `ax`/`ay` discriminated by call-site guards. Asserts DynamicDirect
/// admits each fused token at its own call site, the provider-published
/// shard matches DynamicDirect exactly, and the splice-published shard
/// misses both fused tokens. If the splice side ever starts admitting
/// them, the coordinate story changed: revisit this pin and the audit.
#[test]
fn splice_boundary_shard_underadmits_called_frames_divergent() {
    fn admits(mask: &[u32], token: u32) -> bool {
        mask.get(token as usize / 32)
            .is_some_and(|word| word & (1u32 << (token % 32)) != 0)
    }

    let vocab = Vocab::new(vec![
        (0, b"ax".to_vec()),
        (1, b"ay".to_vec()),
        (2, b"L".to_vec()),
        (3, b"R".to_vec()),
        (4, b"a".to_vec()),
        (5, b"x".to_vec()),
        (6, b"y".to_vec()),
    ]);
    let parent = Constraint::from_glrm_grammar(
        r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "L" SUB "x" | "R" SUB "y";
            "#,
        &vocab,
    )
    .unwrap();
    let child = Constraint::from_glrm_grammar(
        r#"
                start child;
                nt child ::= "a";
            "#,
        &vocab,
    )
    .unwrap();
    let inputs = [CompiledSubgrammarInput {
        placeholder_terminal: terminal_id(&parent, "SUB"),
        additional_placeholder_terminals: &[],
        constraint: &child,
    }];
    let dynamic = compose_constraints_owned_parent_segmented(
        parent.clone(),
        &inputs,
        &vocab,
        SegmentedBoundaryBackend::Dynamic,
    )
    .expect("dynamic compose")
    .constraint;

    // Publish the alternative shards into a real static coordinator. A
    // DynamicDirect coordinator deliberately has no outer static seed-mask
    // universe; swapping only its shard enum is not a backend conversion.
    let static_shell = compose_constraints_owned_parent_segmented(
        parent.clone(), &inputs, &vocab, SegmentedBoundaryBackend::StaticParserDwa,
    ).expect("static coordinator for publication fixture").constraint;

    let composed = low_level_compose(&parent, &inputs);
    let grammar = analyzed_grammar(&composed.table.table, &composed.terminal_names);
    let disallowed = compute_disallowed_follows(&grammar);
    let counts: Vec<u32> = vec![
        parent.tokenizer.num_states(),
        child.tokenizer.num_states(),
    ];
    let (built, _) = build_boundary_shard_walks(&BoundaryShardLinkInputs {
        merged_tokenizer: &composed.tokenizer,
        vocab: &vocab,
        grammar: &grammar,
        disallowed_follows: &disallowed,
        ignore_terminal: composed.ignore_canonical,
        follow_transparent_ignores: None,
        terminal_offsets: &composed.table.terminal_offsets,
        leaf_to_immediate: None,
        tokenizer_offsets: &composed.tokenizer_offsets,
        component_state_counts: &counts,
        candidate_tokens_by_component: None,
        retain_parent_non_crossing_paths: inputs
            .iter()
            .any(|child| child.constraint.table.embedded_start_nullable()),
        walk_plans: None,
    })
    .expect("walk shards");
    assert_eq!(built.len(), 1, "only the child shard crosses here");
    assert_eq!(built[0].start_component, 1);
    assert!(
        unbound_slot_terminals(&parent, &["SUB"]).is_empty()
            && unbound_slot_terminals(&child, &[]).is_empty(),
        "fixture has no unbound slots; emptying must not be involved",
    );
    let (splice_table, _) =
        prepare_spliced_boundary_table("divergent", &composed, &[]);
    let global_ignores = component_ignores_are_globally_erasable(&parent, &inputs);
    let provider_table = Arc::new(
        link_provider_boundary_table(
            &parent,
            &inputs,
            &composed.table.terminal_offsets,
            composed.table.table.num_terminals,
            global_ignores,
        )
        .expect("provider boundary table"),
    );

    let publish_against = |table: &Arc<GLRTable>| {
        let shard = &built[0];
        let work = WalkBoundaryShardWork {
            start_component: shard.start_component as u32,
            terminal_automaton: TerminalAutomaton::Dwa(shard.output.dwa.clone()),
            id_map: shard.output.id_map.clone(),
            candidate_tokens: shard
                .candidate_tokens
                .iter()
                .copied()
                .collect::<Vec<_>>()
                .into(),
        };
        let (one, _) =
            publish_walk_boundary_shard_work(work, table, &composed.tokenizer_offsets, &counts)
                .expect("publish walk shard");
        let mut installed = static_shell.clone();
        install_published_static_boundary_shards(
            installed.static_dynamic_overlay.as_mut().expect("overlay"),
            vec![one],
        )
        .expect("install");
        assert!(
            installed.uses_compact_segmented_parser_runtime(),
            "installed composition must stay on the compact runtime",
        );
        installed
    };
    let splice_static = publish_against(&splice_table);
    let provider_static = publish_against(&provider_table);

    // After L: fused `ax` (token 0) must be admitted; after R: `ay` (1).
    for (commit, fused, sibling, site) in [(2u32, 0u32, 1u32, "L"), (3u32, 1u32, 0u32, "R")] {
        let mut st_dyn = dynamic.start();
        let mut st_splice = splice_static.start();
        let mut st_provider = provider_static.start();
        st_dyn.commit_token(commit).unwrap();
        st_splice.commit_token(commit).unwrap();
        st_provider.commit_token(commit).unwrap();
        let mask_dyn = st_dyn.mask();
        let mask_splice = st_splice.mask();
        let mask_provider = st_provider.mask();
        assert!(
            admits(&mask_dyn, fused),
            "dynamic must admit fused token {fused} after {site}",
        );
        assert!(
            !admits(&mask_dyn, sibling),
            "dynamic must reject sibling fused token {sibling} after {site}",
        );
        assert_eq!(
            mask_provider, mask_dyn,
            "provider shard must match DynamicDirect after {site}",
        );
        assert!(
            !admits(&mask_splice, fused),
            "splice shard under-admits fused token {fused} after {site} \
                 (called-frame labels unmatched); if this now admits, revisit the audit pin",
        );
    }
}

/// Production-outer decline pin (Phase 2b reconciliation audit).
///
/// The provider-materialized boundary table's `exact_control_elimination`
/// diverges on the selected10 outer (core+dispatch) link at the known
/// row-918 / terminal-3 cell (`state=918 terminal=3`, guard-differentiated
/// cyclic fan-out over a ~10-state reduce cycle — the identical cell the
/// old segmented-static path hung on), so current production static
/// composition DECLINES the outer link loudly via the convergence cap
/// instead of linking it. Phase 2 linked outer exactly via the splice in
/// ~7.6 s; that object is inexact on called frames (see the divergent pin
/// above), so neither table object currently serves outer statically.
///
/// If this test observes convergence (Ok), the elimination pathology is
/// fixed and the production contract can be revisited; the panic message
/// says so. Fast decline (well under the 10 s wall cap) is part of the
/// pin: the budget must fail fast, never hang.
#[test]
#[ignore]
fn provider_boundary_table_diverges_on_selected10_outer() {
    let fixture = load_selected10_outer();
    let children = [CompiledSubgrammarInput {
        placeholder_terminal: terminal_id(&fixture.core, "PROGRAMMATIC_TOOL_SUFFIX"),
        additional_placeholder_terminals: &[],
        constraint: &fixture.dispatch,
    }];
    let global_ignores = component_ignores_are_globally_erasable(&fixture.core, &children);
    let started = Instant::now();
    let result = link_provider_boundary_table(
        &fixture.core,
        &children,
        &fixture.composed.table.terminal_offsets,
        fixture.composed.table.table.num_terminals,
        global_ignores,
    );
    let ms = started.elapsed().as_secs_f64() * 1000.0;
    match result {
        Ok(table) => panic!(
            "provider boundary table unexpectedly converged on outer: states={} terms={} \
                 controls={} ms={ms:.1}; the elimination pathology is fixed — update this pin \
                 and revisit the production static-link contract",
            table.num_states,
            table.num_terminals,
            table.control_terminals.len(),
        ),
        Err(error) => {
            eprintln!("BOUNDARY_PROVIDER_DIVERGENCE ms={ms:.1} error={error}");
            assert!(
                error.contains("state=918")
                    && error.contains("terminal=3")
                    && error.contains("stack_effect_visits"),
                "outer decline must be the row-918 convergence cap, got: {error}",
            );
            assert!(
                ms < 60_000.0,
                "convergence cap must fail fast, took {ms:.1} ms",
            );
        }
    }
}

/// Prepared-transfer outer link through the NEW signed-transfer production
/// Row-918 local transfer/entry/finish characterization (no link, no
/// install): links agree, core terminal-3 transfer finite, slot Entry
/// shape valid + finite + child-start-pushed, dispatch Finish finite +
/// no local EOF effects. Extracted so the cheap asserts run as their own
/// <=20s diagnostic; the original row918 test calls this helper.
fn assert_row918_local_transfers(
    fixture: &Selected10Outer,
    links: &[crate::runtime::SegmentedParserLink],
) {
    use crate::compiler::composition::boundary::transfer::{
        StateInjection, diagnose_transfer, instantiate_entry, instantiate_finish,
        scope_characterization, validate_slot_entry_shape,
    };
    use crate::compiler::stages::templates::characterize::characterize_selected_terminals_for_terminal_count;

    assert_eq!(links.len(), 1, "outer link has one child link");
    let link = &links[0];
    let parent_injection = StateInjection { offset: 0 };
    let child_injection = StateInjection {
        offset: fixture.core.table.num_states,
    };
    // Ordinary local transfer for core terminal 3 (the row-918 terminal).
    let mut selected = vec![false; fixture.core.table.num_terminals as usize];
    selected[3] = true;
    let characterized = characterize_selected_terminals_for_terminal_count(
        &fixture.core.table,
        fixture.core.table.num_terminals,
        &selected,
    );
    let local_t3 = characterized.get(&3).expect("core terminal 3 characterizes");
    let scoped_t3 =
        scope_characterization(local_t3, &parent_injection).expect("scope core terminal 3");
    let diag_t3 = diagnose_transfer(&scoped_t3);
    eprintln!(
        "BOUNDARY_TRANSFER t3 escapes={} reduces={} nt_escapes={} nt_rereduces={} read={} push={} cycle={:?} kinds={:?}",
        diag_t3.num_escapes,
        diag_t3.num_reduces,
        diag_t3.num_nt_escapes,
        diag_t3.num_nt_rereduces,
        diag_t3.max_input_read,
        diag_t3.max_push_len,
        diag_t3.cycle_status,
        diag_t3.endpoint_kinds,
    );
    assert!(
        diag_t3.cycle_status.is_none(),
        "core terminal-3 local transfer must be finite",
    );
    // Slot Entry: validate the provider-supported shape, then instantiate.
    validate_slot_entry_shape(&fixture.core.table, link.slot_terminal)
        .expect("outer slot supports static Entry");
    let mut slot_selected = vec![false; fixture.core.table.num_terminals as usize];
    slot_selected[link.slot_terminal as usize] = true;
    let slot_characterized = characterize_selected_terminals_for_terminal_count(
        &fixture.core.table,
        fixture.core.table.num_terminals,
        &slot_selected,
    );
    let local_slot = slot_characterized
        .get(&link.slot_terminal)
        .expect("slot terminal characterizes");
    let scoped_slot =
        scope_characterization(local_slot, &parent_injection).expect("scope slot");
    let scoped_child_start = child_injection
        .scope_state(link.child_start)
        .expect("scope child start");
    let entry = instantiate_entry(scoped_slot, scoped_child_start, 0);
    eprintln!(
        "BOUNDARY_TRANSFER entry escapes={} reduces={} read={} push={} cycle={:?} kinds={:?}",
        entry.diagnostics.num_escapes,
        entry.diagnostics.num_reduces,
        entry.diagnostics.max_input_read,
        entry.diagnostics.max_push_len,
        entry.diagnostics.cycle_status,
        entry.diagnostics.endpoint_kinds,
    );
    assert!(
        entry.diagnostics.cycle_status.is_none(),
        "slot Entry transfer must be finite",
    );
    assert!(
        entry
            .characterization
            .escapes
            .iter()
            .all(|escape| escape.pushes.last() == Some(&scoped_child_start)),
        "every Entry escape must carry the scoped child-start push",
    );
    // Finish export for the dispatch child under this link's policy.
    let (finish, has_local_eof_effects) =
        instantiate_finish(&fixture.dispatch.table, link, &child_injection)
            .expect("dispatch finish transfer");
    eprintln!(
        "BOUNDARY_TRANSFER finish return_pop={} nullable={} escapes={} reduces={} nt_escapes={} nt_rereduces={} read={} push={} cycle={:?} kinds={:?} local_eof={has_local_eof_effects}",
        link.return_pop,
        link.child_start_nullable,
        finish.diagnostics.num_escapes,
        finish.diagnostics.num_reduces,
        finish.diagnostics.num_nt_escapes,
        finish.diagnostics.num_nt_rereduces,
        finish.diagnostics.max_input_read,
        finish.diagnostics.max_push_len,
        finish.diagnostics.cycle_status,
        finish.diagnostics.endpoint_kinds,
    );
    assert!(
        finish.diagnostics.cycle_status.is_none(),
        "dispatch finish transfer must be finite",
    );
    assert!(
        !has_local_eof_effects,
        "bounded flat prototype requires canonical EOF completion (reductions + pure Accept); local EOF work needs outer control-choice points",
    );
}

/// Row-918 transfer/entry/finish characterization WITHOUT the expensive
/// link: runs the extracted helper only (<=20s diagnostic). The counter
/// proof stays in the fresh setup case, not here.
#[test]
#[ignore]
fn row918_local_transfers_without_link() {
    use crate::compiler::composition::boundary::transfer::validate_shared_child_links;
    use crate::compiler::composition::build_segmented_parser_links;

    let _env_lock = crate::TEST_ENV_LOCK.lock().unwrap();
    let fixture = load_selected10_outer();
    let children = [CompiledSubgrammarInput {
        placeholder_terminal: terminal_id(&fixture.core, "PROGRAMMATIC_TOOL_SUFFIX"),
        additional_placeholder_terminals: &[],
        constraint: &fixture.dispatch,
    }];
    let links = build_segmented_parser_links(&children).expect("segmented links");
    validate_shared_child_links(&links).expect("incoming links agree");
    assert_row918_local_transfers(&fixture, &links);
}

/// route (advisor v3 Prototype 1, milestones F + 9).
///
/// Links only LOCAL tables: the ordinary core terminal-3 transfer plus
/// the dispatch Finish export under this link's child-start/return-pop
/// policy, with the slot Entry shape validated. Records local
/// characterization cycle status / read bound / push bound / sizes for
/// both transfers, then compiles, installs, and differentially validates
/// the full outer link. This path never calls `exact_control_elimination`
/// by construction (proven by the run counter); there is no global table
/// object here at all.
#[test]
#[ignore]
fn prepared_transfer_row918_no_global_elimination() {
    use crate::compiler::composition::boundary::transfer::validate_shared_child_links;
    use crate::compiler::composition::build_segmented_parser_links;

    let fixture = load_selected10_outer();
    let children = [CompiledSubgrammarInput {
        placeholder_terminal: terminal_id(&fixture.core, "PROGRAMMATIC_TOOL_SUFFIX"),
        additional_placeholder_terminals: &[],
        constraint: &fixture.dispatch,
    }];
    let links = build_segmented_parser_links(&children).expect("segmented links");
    validate_shared_child_links(&links).expect("incoming links agree");
    assert_row918_local_transfers(&fixture, &links);
    // Full outer link through the NEW signed-transfer production route
    // (milestone 9): local transfers, Ready-port composition, exact
    // resolution, table-free normalization — and no control elimination.
    let elim_before = control_elimination_run_count();
    let link_output = build_walk_static_boundary_link(&WalkStaticLinkInputs {
        parent: &fixture.core,
        children: &children,
        vocab: &fixture.vocab,
        static_components: None,
        expected_terminal_offsets: &fixture.composed.table.terminal_offsets,
    })
    .expect("outer link compiles through the signed-transfer route");
    assert_eq!(
        control_elimination_run_count(),
        elim_before,
        "signed-transfer outer link must not invoke control elimination",
    );
    assert!(
        !link_output.all_dynamic,
        "outer link must take the static route",
    );
    for shard in &link_output.published_shards {
        assert!(
            link_output
                .effective_static_components
                .contains(shard.start_component as usize),
            "published shard {} must be effectively static",
            shard.start_component,
        );
    }
    eprintln!(
        "BOUNDARY_TRANSFER outer published={:?} tokens={:?}",
        link_output
            .published_shards
            .iter()
            .map(|shard| shard.start_component)
            .collect::<Vec<_>>(),
        link_output.boundary_tokens_by_start_component,
    );
    // Install the new-route shards over a dynamic composition and require
    // mask-for-mask equality: the outer shard must WORK, not just compile.
    let dyn_children = [CompiledSubgrammarInput {
        placeholder_terminal: terminal_id(&fixture.core, "PROGRAMMATIC_TOOL_SUFFIX"),
        additional_placeholder_terminals: &[],
        constraint: &fixture.dispatch,
    }];
    let dynamic = compose_constraints_owned_parent_segmented(
        fixture.core.clone(),
        &dyn_children,
        &fixture.vocab,
        SegmentedBoundaryBackend::Dynamic,
    )
    .expect("dynamic compose")
    .constraint;
    let mut static_comp = dynamic.clone();
    install_published_static_boundary_shards(
        static_comp.static_dynamic_overlay.as_mut().expect("overlay"),
        link_output.published_shards,
    )
    .expect("install");
    assert!(
        static_comp.uses_compact_segmented_parser_runtime(),
        "installed composition must stay on the compact runtime",
    );
    for (index, component) in static_comp
        .static_dynamic_overlay
        .as_ref()
        .expect("overlay")
        .segmented_parser_components
        .iter()
        .enumerate()
    {
        if let Some(shard) = component.boundary.as_ref() {
            assert!(
                matches!(
                    shard.backend,
                    crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                ),
                "installed outer component {index} must be a StaticParser shard",
            );
        }
    }
    let component_state_counts =
        [fixture.core.tokenizer.num_states(), fixture.dispatch.tokenizer.num_states()];
    assert_scoped_tokenizer_packing("outer-signed", &static_comp, &component_state_counts);
    // Optional prepare: save/reload round-trip + pins, then return BEFORE
    // the in-process differential. Default path unchanged.
    if let Some(dir) = install_prepare_dir("ROW918_PREPARE_OUT_DIR") {
        let shard_ids: Vec<(u32, usize)> = static_comp
            .static_dynamic_overlay
            .as_ref()
            .expect("overlay")
            .segmented_parser_components
            .iter()
            .filter_map(|component| {
                component.boundary.as_ref().map(|shard| {
                    (
                        shard.start_component,
                        shard.candidate_tokens.as_ref().map(|tokens| tokens.len()).unwrap_or(0),
                    )
                })
            })
            .collect();
        save_reload_install_pair(
            "prepared_transfer_row918_no_global_elimination",
            &dir,
            &dynamic,
            &static_comp,
            &component_state_counts,
            "outer-signed",
            &fixture.vocab,
        );
        write_install_prepare_manifest(
            &dir,
            "prepared_transfer_row918_no_global_elimination",
            &shard_ids,
            &component_state_counts,
            &fixture.vocab,
        );
        return;
    }
    run_install_differential(&dynamic, &mut static_comp, false);
}

/// Divergent called-frame fixture through the PRODUCTION static link
/// (milestones E/G/H at fixture scale).
///
/// Nullable bound children are OUTSIDE the signed-transfer static
/// linker's supported class (bounded flat closure certificate): silent
/// Entry/Return episodes have no uniform bound, so the link declines
/// loudly here instead of under-admitting. This pins that decline.
#[test]
fn nullable_static_link_declines_loudly() {
    let vocab = Vocab::new(vec![
        (0, b"L".to_vec()),
        (1, b"a".to_vec()),
        (2, b"x".to_vec()),
    ]);
    let parent = Constraint::from_glrm_grammar(
        r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "L" SUB "x";
            "#,
        &vocab,
    )
    .unwrap();
    let child = Constraint::from_glrm_grammar(
        r#"
                start child;
                nt item ::= "a";
                nt child ::= item?;
            "#,
        &vocab,
    )
    .unwrap();
    assert!(
        child.table.embedded_start_nullable(),
        "pin fixture must stay effectively nullable",
    );
    let inputs = [CompiledSubgrammarInput {
        placeholder_terminal: terminal_id(&parent, "SUB"),
        additional_placeholder_terminals: &[],
        constraint: &child,
    }];
    let composed = low_level_compose(&parent, &inputs);
    let error = match build_walk_static_boundary_link(&WalkStaticLinkInputs {
        parent: &parent,
        children: &inputs,
        vocab: &vocab,
        static_components: None,
        expected_terminal_offsets: &composed.table.terminal_offsets,
    }) {
        Err(error) => error,
        Ok(_) => panic!("nullable static links must decline loudly (general C* is future work)"),
    };
    assert!(
        error.contains("nullable"),
        "decline must name nullability, got: {error}",
    );
}

/// Divergent called-frame fixture through the PRODUCTION static link
/// (milestones E/G/H at fixture scale).
///
/// `build_walk_static_boundary_link` must install a real static shard that
/// matches DynamicDirect at both call sites (caller-sensitive fused
/// tokens), with no `DynamicDirect` backend anywhere in the installed
/// overlay. Removing the child shard (ablation) must lose the fused
/// tokens — a nonempty shard that never fires is not evidence. With
/// `GLRMASK_STRICT_STATIC_TRAP_DYNAMIC=1`, masking the installed static
/// composition must not trip the DynamicDirect trap.
#[test]
#[ignore]
fn walk_static_link_divergent_fixture_matches_dynamic_and_ablates() {
    if crate::isolate_environment_test(true) { return; }
    fn admits(mask: &[u32], token: u32) -> bool {
        mask.get(token as usize / 32)
            .is_some_and(|word| word & (1u32 << (token % 32)) != 0)
    }

    // Panic-safe env restore: Drop removes the trap var even if an assert
    // below fails, so the trap cannot leak into other tests. The unsafe
    // blocks are sound because the caller holds TEST_ENV_LOCK, serializing
    // all process-env mutation in the test build.
    struct TrapGuard;
    impl TrapGuard {
        fn set() -> Self {
            unsafe {
                std::env::set_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", "1");
            }
            Self
        }
    }
    impl Drop for TrapGuard {
        fn drop(&mut self) {
            unsafe {
                std::env::remove_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC");
            }
        }
    }

    let vocab = Vocab::new(vec![
        (0, b"ax".to_vec()),
        (1, b"ay".to_vec()),
        (2, b"L".to_vec()),
        (3, b"R".to_vec()),
        (4, b"a".to_vec()),
        (5, b"x".to_vec()),
        (6, b"y".to_vec()),
    ]);
    let parent = Constraint::from_glrm_grammar(
        r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "L" SUB "x" | "R" SUB "y";
            "#,
        &vocab,
    )
    .unwrap();
    let child = Constraint::from_glrm_grammar(
        r#"
                start child;
                nt child ::= "a";
            "#,
        &vocab,
    )
    .unwrap();
    let inputs = [CompiledSubgrammarInput {
        placeholder_terminal: terminal_id(&parent, "SUB"),
        additional_placeholder_terminals: &[],
        constraint: &child,
    }];
    let composed = low_level_compose(&parent, &inputs);
    // No control elimination anywhere on the new parser-side path: the
    // signed-transfer compiler links local tables only.
    let elim_before = control_elimination_run_count();
    let link_output = build_walk_static_boundary_link(&WalkStaticLinkInputs {
        parent: &parent,
        children: &inputs,
        vocab: &vocab,
        static_components: None,
        expected_terminal_offsets: &composed.table.terminal_offsets,
    })
    .expect("production static link on the divergent fixture");
    assert_eq!(
        control_elimination_run_count(),
        elim_before,
        "signed-transfer link must not invoke control elimination",
    );
    assert!(
        !link_output.all_dynamic,
        "fixture link must be static, not all-dynamic",
    );
    assert_eq!(
        link_output.published_shards.len(),
        1,
        "only the child shard crosses here",
    );
    assert_eq!(link_output.published_shards[0].start_component, 1);

    let dynamic = compose_constraints_owned_parent_segmented(
        parent.clone(),
        &inputs,
        &vocab,
        SegmentedBoundaryBackend::Dynamic,
    )
    .expect("dynamic compose")
    .constraint;
    let mut installed = dynamic.clone();
    install_published_static_boundary_shards(
        installed.static_dynamic_overlay.as_mut().expect("overlay"),
        link_output
            .published_shards
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
    )
    .expect("install");
    // No DynamicDirect backend anywhere in the installed overlay.
    for (index, component) in installed
        .static_dynamic_overlay
        .as_ref()
        .expect("overlay")
        .segmented_parser_components
        .iter()
        .enumerate()
    {
        if let Some(shard) = component.boundary.as_ref() {
            assert!(
                matches!(
                    shard.backend,
                    crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                ),
                "installed component {index} must be a StaticParser shard",
            );
        }
    }
    // Call-site-exact admission at both sites.
    for (commit, fused, sibling, site) in [(2u32, 0u32, 1u32, "L"), (3u32, 1u32, 0u32, "R")] {
        let mut st_dyn = dynamic.start();
        let mut st_static = installed.start();
        st_dyn.commit_token(commit).unwrap();
        st_static.commit_token(commit).unwrap();
        let mask_dyn = st_dyn.mask();
        let mask_static = st_static.mask();
        assert!(admits(&mask_dyn, fused), "dynamic admits fused {fused} after {site}");
        assert!(!admits(&mask_dyn, sibling), "dynamic rejects sibling {sibling} after {site}");
        assert_eq!(mask_static, mask_dyn, "static matches dynamic after {site}");
    }
    // Strict-static trap: no DynamicDirect evaluation on this path.
    // Masks are collected under the env lock, then the guard is dropped
    // (removing the var) before asserting, so a failure cannot leak the
    // trap into other tests.
    let trapped: Vec<(u32, Vec<u32>)> = {
        let _env_lock = crate::TEST_ENV_LOCK.lock().unwrap();
        let _trap = TrapGuard::set();
        let mut trapped = Vec::new();
        for (commit, fused, _) in [(2u32, 0u32, "L"), (3u32, 1u32, "R")] {
            let mut st = installed.start();
            st.commit_token(commit).unwrap();
            trapped.push((fused, st.mask()));
        }
        trapped
    };
    for (fused, mask) in &trapped {
        assert!(admits(mask, *fused), "static admits fused {fused} under the trap");
    }
    // Ablation: with the child shard removed, the fused tokens must go
    // missing (the shard is load-bearing, not redundant).
    let mut ablated = dynamic.clone();
    install_published_static_boundary_shards(
        ablated.static_dynamic_overlay.as_mut().expect("overlay"),
        Vec::new(),
    )
    .expect("install empty");
    for (commit, fused, site) in [(2u32, 0u32, "L"), (3u32, 1u32, "R")] {
        let mut st = ablated.start();
        st.commit_token(commit).unwrap();
        assert!(
            !admits(&st.mask(), fused),
            "ablated composition must lose fused token {fused} after {site}",
        );
    }
}

/// Strict-harness scope regression: the L2P strict baseline must inherit the
/// caller's shard scope (seed filter, shared equivalence, crossing filter and
/// TI/compaction options). A baseline rebuilt without them compares a
/// shard-scoped candidate against a full-scope artifact and spuriously
/// mismatches (selected10: state=0 token=0 word=[81], the first coordinate
/// in deterministic enumeration order). Runs the divergent fixture's
/// production link with the factor strict reference armed: without the
/// scope inheritance this panics inside the shard-build strict compare;
/// with it the link succeeds.
#[test]
fn strict_baseline_inherits_shard_scope_on_divergent_fixture() {
    let vocab = Vocab::new(vec![
        (0, b"ax".to_vec()),
        (1, b"ay".to_vec()),
        (2, b"L".to_vec()),
        (3, b"R".to_vec()),
        (4, b"a".to_vec()),
        (5, b"x".to_vec()),
        (6, b"y".to_vec()),
    ]);
    let parent = Constraint::from_glrm_grammar(
        r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "L" SUB "x" | "R" SUB "y";
            "#,
        &vocab,
    )
    .unwrap();
    let child = Constraint::from_glrm_grammar(
        r#"
                start child;
                nt child ::= "a";
            "#,
        &vocab,
    )
    .unwrap();
    let inputs = [CompiledSubgrammarInput {
        placeholder_terminal: terminal_id(&parent, "SUB"),
        additional_placeholder_terminals: &[],
        constraint: &child,
    }];
    let composed = low_level_compose(&parent, &inputs);
    let link_output = {
        let _env_lock = crate::TEST_ENV_LOCK.lock().unwrap();
        let _strict = EnvVarGuard::set(
            "GLRMASK_L2P_FIRST_BYTE_VOCAB_FACTOR_STRICT_REFERENCE",
            "boundary_shard",
        );
        build_walk_static_boundary_link(&WalkStaticLinkInputs {
            parent: &parent,
            children: &inputs,
            vocab: &vocab,
            static_components: None,
            expected_terminal_offsets: &composed.table.terminal_offsets,
        })
        .expect("shard-scoped strict link on the divergent fixture")
    };
    assert!(
        !link_output.all_dynamic,
        "fixture link must be static, not all-dynamic",
    );
    assert_eq!(
        link_output.published_shards.len(),
        1,
        "only the child shard crosses here",
    );
}

/// Nested depth-2 end-to-end fixture through the nested signed-transfer
/// link (production walk/compile/publish/install, strict trap, curated
/// differential, light ablation).
///
/// Shape: P ::= "L" SUB SUB "x" | "R" SUB SUB "y" (two adjacent call
/// sites into the same mid child M); M ::= "m" SUB2 (leading visible
/// terminal, trailing slot); G ::= "g". All links effectively
/// nonnullable, acyclic, canonical. Fused vocabulary tokens span every
/// adjacent cross-leaf terminal pair ("Lm", "Rm", "mg", "gm", "gx",
/// "gy"); without them no walk path can cross components (single bytes
/// never cross) and the shards would be vacuous.
///
/// This fixture is end-to-end integration only: the 2H certificate
/// (depth 2, bound 4) is asserted, but the R,R,E,E load-bearing proof
/// lives in `boundary_transfer::nested_ready_depth_four_is_load_bearing`.
/// Here the arbiter is a curated reachable-prefix differential of the
/// installed StaticParser composition against DynamicDirect, run under
/// the strict-static trap; a light empty-shard ablation confirms the
/// installed shards are genuinely used.
///
/// Link architecture under test: two top-level walks (parent commit with
/// the parent-leaf crossing filter; whole-block commit with the mid-leaf
/// crossing filter — a path survives iff it touches a terminal outside
/// the filter leaf, so the block walk keeps every leaf-crossing gap
/// including g→m), each compiled with the full 3-leaf signed context and
/// installed on a nested composition whose mid+grandchild block is already
/// static. The retained inner shard covers genuinely block-internal repair;
/// the outer block shard covers outward completion/re-entry candidates and
/// must coexist with that inner static coverage under the strict trap.
#[test]
fn nested_static_link_depth_two_chain_matches_dynamic_and_ablates() {
    if crate::isolate_environment_test(false) { return; }
    use crate::compiler::glr::parser::ScopedSubgrammarLink;

    fn admits(mask: &[u32], token: u32) -> bool {
        mask.get(token as usize / 32)
            .is_some_and(|word| word & (1u32 << (token % 32)) != 0)
    }

    // Panic-safe trap guard (mirrors the flat divergent fixture): Drop
    // removes the var even on failure. The whole test holds TEST_ENV_LOCK
    // so no foreign trap can leak into the dynamic reference phase.
    struct TrapGuard(bool);
    impl TrapGuard {
        fn set() -> Self {
            unsafe {
                std::env::set_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", "1");
            }
            Self(true)
        }
        fn clear(&mut self) {
            if self.0 {
                unsafe {
                    std::env::remove_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC");
                }
                self.0 = false;
            }
        }
    }
    impl Drop for TrapGuard {
        fn drop(&mut self) {
            self.clear();
        }
    }

    fn compile_publish_shard(
        ctx: &crate::compiler::composition::boundary::transfer::SignedLinkContext<'_>,
        dwa: &DWA,
        id_map: &InternalIdMap,
        start: u32,
        num_terminals: u32,
        tok_offsets: &[u32],
        leaf_counts: &[u32],
    ) -> crate::compiler::composition::PublishedStaticBoundaryShard {
        let emitted = boundary_emitted_terminals(dwa, num_terminals as usize);
        let library =
            crate::compiler::composition::boundary::transfer::build_fragment_library(ctx, &emitted, start)
                .expect("nested fragment library");
        let compiled = crate::compiler::composition::boundary::transfer::compile_signed_shard_parser(
            ctx, &library, dwa, id_map, start,
        )
        .expect("nested shard compile");
        let candidates: Arc<[u32]> = boundary_accepted_tokens(dwa, id_map)
            .into_iter()
            .collect::<Vec<_>>()
            .into();
        let work = WalkBoundaryShardWork {
            start_component: start,
            terminal_automaton: TerminalAutomaton::Dwa(dwa.clone()),
            id_map: id_map.clone(),
            candidate_tokens: candidates,
        };
        crate::compiler::composition::boundary::transfer::publish_signed_shard(
            work,
            compiled,
            tok_offsets,
            leaf_counts,
        )
        .expect("nested shard publish")
        .0
    }

    fn install_nested(
        base: &Constraint,
        top_shards: Vec<crate::compiler::composition::PublishedStaticBoundaryShard>,
    ) -> Constraint {
        let mut installed = base.clone();
        assert_eq!(
            installed
                .static_dynamic_overlay
                .as_ref()
                .expect("overlay")
                .segmented_parser_components
                .len(),
            2,
            "nested install needs the 2-component top overlay",
        );
        install_published_static_boundary_shards(
            installed.static_dynamic_overlay.as_mut().expect("overlay"),
            top_shards,
        )
        .expect("install nested shards");
        installed
    }

    // Token ids: 0 L, 1 R, 2 x, 3 y, 4 m, 5 g, 6 Lm, 7 Rm, 8 mg, 9 gm,
    // 10 gx, 11 gy.
    const L: u32 = 0;
    const R: u32 = 1;
    const X: u32 = 2;
    const Y: u32 = 3;
    const M: u32 = 4;
    const G: u32 = 5;
    const LM: u32 = 6;
    const RM: u32 = 7;
    const MG: u32 = 8;
    const GM: u32 = 9;
    const GX: u32 = 10;
    const GY: u32 = 11;

    let _env_lock = crate::TEST_ENV_LOCK.lock().unwrap();
    let vocab = Vocab::new(vec![
        (0, b"L".to_vec()),
        (1, b"R".to_vec()),
        (2, b"x".to_vec()),
        (3, b"y".to_vec()),
        (4, b"m".to_vec()),
        (5, b"g".to_vec()),
        (6, b"Lm".to_vec()),
        (7, b"Rm".to_vec()),
        (8, b"mg".to_vec()),
        (9, b"gm".to_vec()),
        (10, b"gx".to_vec()),
        (11, b"gy".to_vec()),
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
    assert!(
        !mid.table.embedded_start_nullable(),
        "nested fixture mid must stay effectively nonnullable",
    );
    assert!(
        !grandchild.table.embedded_start_nullable(),
        "nested fixture grandchild must stay effectively nonnullable",
    );
    assert!(parent.ignore_terminal.is_none());
    assert!(mid.ignore_terminal.is_none());
    assert!(grandchild.ignore_terminal.is_none());
    let sub_p = terminal_id(&parent, "SUB");
    let sub2_m = terminal_id(&mid, "SUB2");

    // Reference nesting through the production dynamic composer (nested
    // dynamic masking is proven by existing tests; it is the arbiter).
    let mid_inputs = [CompiledSubgrammarInput {
        placeholder_terminal: sub2_m,
        additional_placeholder_terminals: &[],
        constraint: &grandchild,
    }];
    let mid_dyn = compose_constraints_owned_parent_segmented(
        mid.clone(),
        &mid_inputs,
        &vocab,
        SegmentedBoundaryBackend::Dynamic,
    )
    .expect("inner dynamic compose")
    .constraint;
    let mid_static = compose_constraints_owned_parent_segmented(
        mid.clone(),
        &mid_inputs,
        &vocab,
        SegmentedBoundaryBackend::StaticParserDwa,
    )
    .expect("inner static compose")
    .constraint;
    assert!(
        mid_static
            .static_dynamic_overlay
            .as_ref()
            .is_some_and(|overlay| !overlay.segmented_boundary_shards.is_empty()),
        "inner static compose must install its boundary repair before nesting",
    );
    let outer_inputs = [CompiledSubgrammarInput {
        placeholder_terminal: sub_p,
        additional_placeholder_terminals: &[],
        constraint: &mid_dyn,
    }];
    let outer_dyn = compose_constraints_owned_parent_segmented(
        parent.clone(),
        &outer_inputs,
        &vocab,
        SegmentedBoundaryBackend::Dynamic,
    )
    .expect("outer dynamic compose")
    .constraint;
    let outer_mixed_inputs = [CompiledSubgrammarInput {
        placeholder_terminal: sub_p,
        additional_placeholder_terminals: &[],
        constraint: &mid_static,
    }];
    let outer_with_static_inner = compose_constraints_owned_parent_segmented(
        parent.clone(),
        &outer_mixed_inputs,
        &vocab,
        SegmentedBoundaryBackend::Dynamic,
    )
    .expect("outer dynamic compose over static inner block")
    .constraint;
    assert!(
        outer_with_static_inner
            .static_dynamic_overlay
            .as_ref()
            .and_then(|overlay| overlay.segmented_parser_components.get(1))
            .and_then(|component| component.constraint.static_dynamic_overlay.as_ref())
            .is_some_and(|overlay| !overlay.segmented_boundary_shards.is_empty()),
        "outer dynamic wrapper must preserve the child's static internal boundary repair",
    );
    assert!(
        outer_dyn.uses_compact_segmented_parser_runtime(),
        "nested dynamic composition must stay on the compact runtime",
    );
    assert!(
        mid_dyn.uses_compact_segmented_parser_runtime(),
        "nested inner composition must stay on the compact runtime",
    );

    // Leaf layout pins from the real compose machinery: leaves
    // [[0],[1,0],[1,1]] with back-to-back terminal ranges.
    let layout = outer_dyn
        .recursive_parser_layout()
        .expect("layout")
        .expect("compact layout");
    let paths: Vec<Vec<u32>> = layout
        .leaves
        .iter()
        .map(|leaf| leaf.component_path.clone())
        .collect();
    assert_eq!(
        paths,
        vec![vec![0], vec![1, 0], vec![1, 1]],
        "nested runtime leaf order must be parent, mid, grandchild",
    );
    let n_p = parent.table.num_terminals;
    let n_m = mid.table.num_terminals;
    let n_g = grandchild.table.num_terminals;
    let outer_overlay = outer_dyn.static_dynamic_overlay.as_ref().expect("overlay");
    assert_eq!(outer_overlay.segmented_parser_components.len(), 2);
    let mid_overlay = mid_dyn.static_dynamic_overlay.as_ref().expect("inner overlay");
    assert_eq!(mid_overlay.segmented_parser_components.len(), 2);
    let t0 = outer_overlay.segmented_parser_components[0].terminal_offset;
    let t1 = outer_overlay.segmented_parser_components[1].terminal_offset;
    assert_eq!((t0, t1), (0, n_p), "outer terminal layout must pack parent first");
    let leaf_offsets = vec![0, n_p, n_p + n_m];
    let num_terminals = n_p + n_m + n_g;

    // Expanded links with live return pops and nullability.
    let rp_mid = mid_dyn.composition_child_return_pop().expect("mid return pop");
    let rp_g = grandchild
        .composition_child_return_pop()
        .expect("grandchild return pop");
    assert!(!mid_dyn.composition_start_nullable().expect("mid null"));
    assert!(!grandchild.composition_start_nullable().expect("g null"));
    let links = vec![
        ScopedSubgrammarLink {
            parent_component: 0,
            slot_terminal: sub_p,
            child_component: 1,
            child_start: 0,
            return_pop: rp_mid,
            child_start_nullable: false,
        },
        ScopedSubgrammarLink {
            parent_component: 1,
            slot_terminal: sub2_m,
            child_component: 2,
            child_start: 0,
            return_pop: rp_g,
            child_start_nullable: false,
        },
    ];

    // Recursive machine splice (layout-only, never parser behavior):
    // inner splice first, then the inner composed table as one child.
    let mut splice1 = crate::compiler::glr::table::compose_subgrammar_tables_with_rules(
        &mid.table,
        mid.retained_table_rules().expect("mid rules"),
        None,
        &[SubgrammarTableInput {
            placeholder_terminal: sub2_m,
            additional_placeholder_terminals: &[],
            table: &grandchild.table,
            ignore_terminal: None,
            start_nullable: false,
        }],
        &[grandchild.retained_table_rules().expect("g rules")],
    )
    .expect("inner splice");
    eliminate_composed_runtime_controls(&mut splice1).expect("inner eliminate");
    let mut splice2 = crate::compiler::glr::table::compose_subgrammar_tables_with_rules(
        &parent.table,
        parent.retained_table_rules().expect("parent rules"),
        None,
        &[SubgrammarTableInput {
            placeholder_terminal: sub_p,
            additional_placeholder_terminals: &[],
            table: &splice1.table,
            ignore_terminal: None,
            start_nullable: false,
        }],
        &[splice1.table.rules.as_slice()],
    )
    .expect("outer splice");
    eliminate_composed_runtime_controls(&mut splice2).expect("outer eliminate");
    assert!(
        splice2.table.control_terminals.is_empty(),
        "nested splice must be control-free",
    );
    assert_eq!(
        splice1.terminal_offsets.as_slice(),
        &[0, n_m],
        "inner splice layout must pack mid then grandchild",
    );
    assert_eq!(
        splice2.terminal_offsets.as_slice(),
        &[0, n_p],
        "outer splice layout must pack parent then the mid block",
    );
    assert_eq!(
        splice2.table.num_terminals, num_terminals,
        "nested terminal domain must be exactly the three leaf ranges",
    );

    // Merged tokenizer over intact leaf tokenizers (production union).
    let tokenizer_inputs = vec![
        (parent.composition_tokenizer(), leaf_offsets[0]),
        (mid.composition_tokenizer(), leaf_offsets[1]),
        (grandchild.composition_tokenizer(), leaf_offsets[2]),
    ];
    let (mut merged, tok_offsets) =
        Tokenizer::disjoint_union_with_terminal_offsets(&tokenizer_inputs);
    assert_eq!(
        tok_offsets.first(),
        Some(&1),
        "merged tokenizer must place the fresh reset fan-out at state 0",
    );
    let c_p = parent.composition_tokenizer().num_states();
    let c_m = mid.composition_tokenizer().num_states();
    let c_g = grandchild.composition_tokenizer().num_states();
    let leaf_counts = vec![c_p, c_m, c_g];
    if merged.terminal_exprs().is_none() {
        let all = vec![&parent, &mid, &grandchild];
        if let Some(exprs) = merged_retained_terminal_exprs(
            &all,
            &leaf_offsets,
            num_terminals,
        ) {
            merged.restore_terminal_exprs(Some(exprs)).expect("restore exprs");
        }
    }

    // Grammar + follows over the spliced rules (production analysis).
    let mut terminal_names = parent.terminal_display_names.clone();
    terminal_names.extend(mid.terminal_display_names.iter().cloned());
    terminal_names.extend(grandchild.terminal_display_names.iter().cloned());
    assert_eq!(terminal_names.len(), num_terminals as usize);
    let augmented_start = splice2
        .table
        .rules
        .first()
        .map(|rule| rule.lhs)
        .expect("spliced rules");
    let grammar = AnalyzedGrammar::from_composed_rules(
        splice2.table.rules.clone(),
        num_terminals,
        terminal_names,
        splice2.table.nonterminal_display_names.clone(),
        augmented_start,
    );
    let disallowed = compute_disallowed_follows(&grammar);

    // Two top-level walks under immediate-component ownership: the parent
    // owns leaf 0 and the already-composed block owns leaves 1+2. Pure
    // m↔g crossings are internal to the block and must not become outer
    // boundary candidates; g→x/g→y still crosses the block boundary.
    let flat: Arc<[u32]> = Arc::from(tdwa::l1::build_flat_transition_table(&merged));
    let ownership = Arc::new(
        tdwa::scope::BoundaryOwnership::from_leaf_layout(
            &leaf_offsets,
            num_terminals,
            &[
                tdwa::scope::ImmediateComponentId(0),
                tdwa::scope::ImmediateComponentId(1),
                tdwa::scope::ImmediateComponentId(1),
            ],
            2,
        )
        .expect("nested block ownership"),
    );
    let total_states = merged.num_states() as usize;
    let run_walk = |component_index: usize, commit: &[bool], retain_non_crossing: bool| {
        let scope = tdwa::scope::BoundaryAnalysisScope::new(
            tdwa::scope::InitialStateDomain::from_mask(
                total_states,
                commit.to_vec(),
            )
            .expect("nested initial-state domain"),
            merged
                .deterministic_reset_states()
                .into_iter()
                .collect(),
            Arc::clone(&ownership),
            tdwa::scope::ImmediateComponentId(component_index as u32),
            !retain_non_crossing,
            None,
        )
        .expect("nested boundary scope");
        build_boundary_terminal_dwa(&BoundaryWalkInputs {
            merged_tokenizer: &merged,
            vocab: &vocab,
            grammar: &grammar,
            disallowed_follows: &disallowed,
            ignore_terminal: None,
            follow_transparent_ignores: None,
            scope: &scope,
            retain_non_crossing_paths: retain_non_crossing,
            flat_trans: Some(&flat),
        })
        .expect("nonempty-vocab nested walks must produce a DWA")
    };
    let commit_p = commit_states_for_component(&tok_offsets, c_p, 0, total_states);
    let mut commit_block = commit_states_for_component(&tok_offsets, c_m, 1, total_states);
    let commit_g = commit_states_for_component(&tok_offsets, c_g, 2, total_states);
    for (slot, extra) in commit_block.iter_mut().zip(commit_g.iter()) {
        *slot |= *extra;
    }
    let walk_p = run_walk(0, &commit_p, false);
    // The block can complete after `g` and immediately re-enter at the
    // parent's second SUB call before `m`. Both visible terminals are
    // block-owned, so the lexical ownership filter alone would miss GM.
    // Production narrows this conservative lane to the block's summary
    // candidates; this fixture uses the full tiny vocabulary deliberately.
    let walk_b = run_walk(1, &commit_block, true);
    let cand_p = boundary_accepted_tokens(&walk_p.dwa, &walk_p.id_map);
    let cand_b = boundary_accepted_tokens(&walk_b.dwa, &walk_b.id_map);
    assert!(
        !cand_p.is_empty(),
        "parent walk must cross (fused Lm/Rm expected)",
    );
    assert!(
        !cand_b.is_empty(),
        "block walk must cross (fused mg/gm/gx expected)",
    );
    assert!(
        cand_b.contains(&GM),
        "block walk must retain zero-width completion/re-entry token GM, got {cand_b:?}",
    );
    assert!(
        cand_b.contains(&GX) || cand_b.contains(&GY),
        "block walk must cover an outer-boundary fused token, got {cand_b:?}",
    );

    // Signed-transfer compilation over the full 3-leaf context. The 2H
    // certificate is asserted: depth-2 nonnullable links allow exactly 4
    // control events per gap.
    // Nested unbound-slot scan over intact leaves: a slot is bound iff
    // some link fills it, located by (parent leaf offset + local slot).
    // The flat helper assumes children are leaves (no bound check for
    // child specials), so the nested link generalizes it here.
    let bound_global: std::collections::BTreeSet<u32> = links
        .iter()
        .map(|link| {
            leaf_offsets[link.parent_component as usize]
                .checked_add(link.slot_terminal)
                .expect("bound slot offset overflow")
        })
        .collect();
    let mut unbound = std::collections::BTreeSet::new();
    for (constraint, base) in
        [&parent, &mid, &grandchild].into_iter().zip(leaf_offsets.iter())
    {
        for special in &constraint.special_token_terminals {
            let global = base
                .checked_add(special.terminal_id)
                .expect("slot terminal offset overflow");
            if bound_global.contains(&global) {
                continue;
            }
            if constraint.is_late_grammar_placeholder_terminal(special.terminal_id)
                || constraint.token_bytes_for_id(special.token_id).is_none()
            {
                unbound.insert(global);
            }
        }
    }
    assert!(
        unbound.is_empty(),
        "nested fixture must bind every slot, got {unbound:?}",
    );
    let example_children = [CompiledSubgrammarInput {
        placeholder_terminal: sub_p,
        additional_placeholder_terminals: &[],
        constraint: &mid_dyn,
    }];
    let global_ignores = component_ignores_are_globally_erasable(&parent, &example_children);
    let signed_context =
        crate::compiler::composition::boundary::transfer::build_signed_link_context_from_parts(
            vec![&parent.table, &mid.table, &grandchild.table],
            vec![None, None, None],
            links,
            &leaf_offsets,
            num_terminals,
            global_ignores,
            unbound,
        )
        .expect("nested signed context");
    assert_eq!(signed_context.closure.max_nesting_depth, 2);
    assert_eq!(signed_context.closure.max_controls_per_gap, 4);
    let shard_p = compile_publish_shard(
        &signed_context,
        &walk_p.dwa,
        &walk_p.id_map,
        0,
        num_terminals,
        &tok_offsets,
        &leaf_counts,
    );
    let shard_b = compile_publish_shard(
        &signed_context,
        &walk_b.dwa,
        &walk_b.id_map,
        1,
        num_terminals,
        &tok_offsets,
        &leaf_counts,
    );
    assert_eq!(
        (shard_p.start_component, shard_b.start_component),
        (0, 1),
        "nested link must publish exactly the parent and block shards",
    );

    // The manually published/ablated static shards need a genuine static
    // coordinator, including its seed-mask universe. The dynamic reference
    // above must stay provider-only rather than paying for unused static data.
    let outer_static_shell = compose_constraints_owned_parent_segmented(
        parent.clone(), &outer_mixed_inputs, &vocab,
        SegmentedBoundaryBackend::StaticParserDwa,
    ).expect("static coordinator for nested publication fixture").constraint;
    let installed = install_nested(&outer_static_shell, vec![shard_p, shard_b]);
    for (index, component) in installed
        .static_dynamic_overlay
        .as_ref()
        .expect("overlay")
        .segmented_parser_components
        .iter()
        .enumerate()
    {
        let shard = component.boundary.as_ref().unwrap_or_else(|| {
            panic!("installed top component {index} must carry a static shard")
        });
        assert!(
            matches!(
                shard.backend,
                crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
            ),
            "installed component {index} must be a StaticParser shard",
        );
    }
    let mid_installed = &installed
        .static_dynamic_overlay
        .as_ref()
        .expect("overlay")
        .segmented_parser_components[1]
        .constraint
        .static_dynamic_overlay
        .as_ref()
        .expect("mid overlay");
    assert!(
        mid_installed
            .segmented_parser_components
            .iter()
            .all(|component| component.boundary.as_ref().is_none_or(|shard| matches!(
                shard.backend,
                crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
            ))),
        "mid overlay must not acquire a dynamic internal boundary shard",
    );
    assert!(
        !mid_installed.segmented_boundary_shards.is_empty(),
        "mid overlay static shard list must survive the outer install",
    );
    assert!(
        mid_installed.segmented_boundary_shards.iter().all(|shard| matches!(
            shard.backend,
            crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
        )),
        "every retained inner shard must stay static",
    );

    // Curated reachable-prefix corpus (fast): prefixes of the two
    // complete L/R strings plus fused-spelling entry points. Only fully
    // mask-admitted paths are kept. This covers root entry, mid entry,
    // grandchild entry, both returns, and both call sites without an
    // exhaustive state search.
    struct Node {
        path: Vec<u32>,
        mask: Vec<u32>,
    }
    let candidates: Vec<Vec<u32>> = vec![
        vec![],
        vec![L],
        vec![R],
        vec![L, M],
        vec![R, M],
        vec![LM],
        vec![RM],
        vec![L, M, G],
        vec![R, M, G],
        vec![LM, G],
        vec![RM, G],
        vec![L, M, G, M],
        vec![R, M, G, M],
        vec![L, M, G, M, G],
        vec![R, M, G, M, G],
        vec![L, M, G, M, G, X],
        vec![R, M, G, M, G, Y],
    ];
    let mut nodes = Vec::new();
    for path in &candidates {
        let mut st = outer_dyn.start();
        let mut ok = true;
        for &token in path {
            // Mask-gated stepping: the runtime treats committing a
            // non-admitted token whose bytes still advance the lexer as
            // a loud mismatch, so only mask-admitted steps extend a
            // corpus prefix (mirrors the production differential walk).
            if !admits(&st.mask(), token) {
                ok = false;
                break;
            }
            if st.commit_token(token).is_err() {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        let mask = st.mask();
        nodes.push(Node { path: path.clone(), mask });
    }
    assert!(
        nodes.len() >= 10,
        "nested corpus must cover real positions, got {}",
        nodes.len()
    );
    // The corpus must contain the deep gap sources on both sites.
    for key_path in [
        vec![L, M, G],
        vec![R, M, G],
        vec![L, M, G, M, G],
        vec![R, M, G, M, G],
    ] {
        assert!(
            nodes.iter().any(|node| node.path == key_path),
            "nested corpus must visit gap source {key_path:?}",
        );
    }

    // Phase 2 (trap armed): every recorded path replays identically on
    // the installed static composition. Any hidden dynamic fallback on
    // the static side panics here instead of contributing exact masks.
    let mut trap = TrapGuard::set();
    let mut mismatches = 0usize;
    for node in &nodes {
        let mut st = installed.start();
        for &token in &node.path {
            st.commit_token(token).expect("static replay");
        }
        let actual = st.mask();
        if actual != node.mask {
            eprintln!(
                "[glrmask/test][nested_static_mismatch] path={:?} dynamic={:?} static={:?}",
                node.path, node.mask, actual,
            );
            mismatches += 1;
        }
    }
    trap.clear();
    assert_eq!(mismatches, 0, "nested static must match dynamic on every path");

    // Readable call-site discrimination through two nesting levels. Note
    // the fused spellings overlap single-byte prefixes: gx (bytes g,x)
    // is admissible where g is still unconsumed, i.e. after Lmgm, not
    // after Lmgmg; gm is admissible after Lm; mg is admissible after L.
    for (prefix, fused, sibling, site) in [
        (vec![L, M, G, M], GX, GY, "L"),
        (vec![R, M, G, M], GY, GX, "R"),
    ] {
        let mut st = installed.start();
        for &token in &prefix {
            st.commit_token(token).unwrap();
        }
        let mask = st.mask();
        assert!(admits(&mask, fused), "nested admits {fused} after {site}mgm");
        assert!(!admits(&mask, sibling), "nested rejects {sibling} after {site}mgm");
    }
    {
        let mut st = installed.start();
        st.commit_token(L).unwrap();
        assert!(admits(&st.mask(), M), "nested admits m after L");
        assert!(admits(&st.mask(), MG), "nested admits mg after L");
        assert!(!admits(&st.mask(), RM), "nested rejects Rm after L");
    }
    {
        let mut st = installed.start();
        st.commit_token(LM).unwrap();
        assert!(admits(&st.mask(), GM), "nested admits gm after Lm");
        assert!(!admits(&st.mask(), GX), "nested rejects gx after Lm");
    }

    // Light ablation: empty shards must lose admissions (confirms the
    // installed shards are genuinely used, not redundant). Removing the
    // boundary can only remove admissions (component masks are
    // unchanged), so every ablated mask is a subset; at least one must
    // be strict or the fixture is vacuous. (The R,R,E,E load-bearing
    // control proof lives in
    // `boundary_transfer::nested_ready_depth_four_is_load_bearing`.)
    let ablated = install_nested(&outer_static_shell, Vec::new());
    // Lockstep walk: at every shared prefix the shard-less mask must be
    // a subset of the dynamic mask; a prefix where the reference admits
    // a token the ablated mask lacks (or a final-mask difference) proves
    // the installed shards are genuinely used. Mask-gated stepping keeps
    // the runtime commit oracle satisfied on the shard-less side.
    fn is_subset(a: &[u32], b: &[u32]) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| x & !y == 0) && a.len() <= b.len()
    }
    let mut strict = 0usize;
    for node in &nodes {
        let mut st_dyn = outer_dyn.start();
        let mut st_abl = ablated.start();
        assert!(
            is_subset(&st_abl.mask(), &st_dyn.mask()),
            "ablated masks must be subsets at path {:?}",
            Vec::<u32>::new(),
        );
        let mut diverged = false;
        for &token in &node.path {
            assert!(
                admits(&st_dyn.mask(), token),
                "corpus path must stay dynamic-admitted at {:?}",
                node.path,
            );
            if !admits(&st_abl.mask(), token) {
                diverged = true;
                break;
            }
            st_dyn.commit_token(token).expect("dynamic replay");
            st_abl.commit_token(token).expect("ablated replay");
            assert!(
                is_subset(&st_abl.mask(), &st_dyn.mask()),
                "ablated masks must be subsets at path {:?}",
                node.path,
            );
        }
        if !diverged {
            assert_eq!(
                st_dyn.mask(),
                node.mask,
                "dynamic replay must reproduce the recorded mask",
            );
            if st_abl.mask() != node.mask {
                diverged = true;
            }
        }
        if diverged {
            strict += 1;
        }
    }
    assert!(
        strict > 0,
        "nested ablation must lose admissions somewhere (fixture vacuous otherwise)",
    );
}

#[test]
fn test_compute_conservative_follow_disallowed() {
    let mut disallowed = BTreeMap::new();
    let mut set0 = BitSet::new(5);
    set0.set(1);
    set0.set(2);
    disallowed.insert(0, set0);

    let mut set1 = BitSet::new(5);
    set1.set(2);
    disallowed.insert(1, set1);

    let mut transparent = BitSet::new(5);
    transparent.set(1);

    let conservative = compute_conservative_follow_disallowed(&disallowed, Some(&transparent)).unwrap();
    // terminal 1 was transparent, so key 1 is removed, and bit 1 is cleared from key 0
    assert!(!conservative.contains_key(&1));
    assert!(conservative.contains_key(&0));
    let set0_after = &conservative[&0];
    assert!(!set0_after.contains(1));
    assert!(set0_after.contains(2));
}

/// TEMPORARY diagnostic (untracked-candidate, pending cleanup/integration):
/// count alive ORIGINAL model tokens on the final installed static
/// boundary parser DWAs stored in an existing composed artifact.
///
/// Reads the artifact path from `PARSER_ALIVE_BIN` (no recomposition),
/// walks `static_dynamic_overlay.segmented_boundary_shards` (plus each
/// retained component's own `boundary` shard, printed distinctly), and for
/// every `StaticParser` backend with a `recursive_parser_dwa` runs the
/// exact forward-support traversal of `accepted_original_tokens` over the
/// FINAL parser DWA with that shard's OWN `internal_token_to_originals`
/// map (never the host constraint map). Prints one JSON line per shard to
/// stdout: `{bin_sha, start_component, scope, backend, parser_states,
/// parser_trans, acyclic, unique_internals, unique_originals,
/// candidate_tokens_len}`.
#[test]
#[ignore = "diagnostic: requires existing parser artifacts"]
fn parser_alive_installed_shards() {
    use crate::ds::weight::{ScopedWeightOpCache, Weight};
    use std::collections::{BTreeSet, VecDeque};

    fn alive_support(
        dwa: &DWA,
        internal_to_originals: &[Vec<u32>],
    ) -> (BTreeSet<u32>, BTreeSet<u32>) {
        assert!(
            dwa.is_acyclic(),
            "parser-alive diagnostic expects an acyclic parser DWA"
        );
        let n = dwa.num_states() as usize;
        let mut indegree = vec![0usize; n];
        for state in dwa.states() {
            for &(target, _) in state.transitions.values() {
                indegree[target as usize] += 1;
            }
        }
        let mut queue = VecDeque::new();
        for (state, &degree) in indegree.iter().enumerate() {
            if degree == 0 {
                queue.push_back(state as u32);
            }
        }
        let mut topo = Vec::with_capacity(n);
        while let Some(source) = queue.pop_front() {
            topo.push(source);
            for &(target, _) in dwa.states()[source as usize].transitions.values() {
                indegree[target as usize] -= 1;
                if indegree[target as usize] == 0 {
                    queue.push_back(target);
                }
            }
        }
        assert_eq!(topo.len(), n, "parser DWA topo walk must visit every state");
        // Full (tsid, token) weights propagate through every edge/final
        // intersection before any original expansion: start/final/empty
        // edges and unreachable states handled by construction.
        let mut reach = vec![Weight::empty(); n];
        if n > 0 {
            reach[dwa.start_state() as usize] = Weight::all();
        }
        let mut accepted = Weight::empty();
        let mut ops = ScopedWeightOpCache::default();
        for source in topo {
            let source_support = reach[source as usize].clone();
            if source_support.is_empty() {
                continue;
            }
            let state = &dwa.states()[source as usize];
            if let Some(final_weight) = state.final_weight.as_ref() {
                let support = ops.intersection(&source_support, final_weight);
                accepted = ops.union(&accepted, &support);
            }
            for &(target, ref edge_weight) in state.transitions.values() {
                let support = ops.intersection(&source_support, edge_weight);
                if support.is_empty() {
                    continue;
                }
                reach[target as usize] = ops.union(&reach[target as usize], &support);
            }
        }
        let mut internals = BTreeSet::new();
        let mut originals = BTreeSet::new();
        for (_, internal_tokens) in accepted.raw_range_values() {
            for range in internal_tokens.ranges() {
                for internal_token in range {
                    if let Some(ids) =
                        internal_to_originals.get(internal_token as usize)
                    {
                        if ids.is_empty() {
                            continue;
                        }
                        internals.insert(internal_token);
                        originals.extend(ids.iter().copied());
                    }
                }
            }
        }
        (internals, originals)
    }

    let path = std::env::var("PARSER_ALIVE_BIN")
        .expect("PARSER_ALIVE_BIN must name an existing composed .static.bin artifact");
    let bytes = std::fs::read(&path).expect("read artifact bytes");
    let digest = blake3::hash(&bytes);
    let bin_sha = digest.to_hex().to_string();
    let constraint = Constraint::load(bytes).expect("load composed artifact");
    let overlay = constraint
        .static_dynamic_overlay
        .as_ref()
        .expect("composed artifact must carry a static/dynamic overlay");
    let mut rows: Vec<String> = Vec::new();
    let mut report = |scope: String,
                      start_component: u32,
                      backend: &str,
                      parser: Option<&DWA>,
                      ito: Option<&[Vec<u32>]>,
                      candidates: Option<&[u32]>| {
        let (states, trans, acyclic, internals, originals) = match (parser, ito) {
            (Some(dwa), Some(map)) => {
                let (i, o) = alive_support(dwa, map);
                (
                    dwa.num_states(),
                    dwa.num_transitions(),
                    dwa.is_acyclic(),
                    i.len(),
                    o.len(),
                )
            }
            _ => (0, 0, false, 0, 0),
        };
        rows.push(format!(
            "{{\"bin_sha\":\"{bin_sha}\",\"scope\":\"{scope}\",\"start_component\":{start_component},\"backend\":\"{backend}\",\"parser_states\":{states},\"parser_trans\":{trans},\"acyclic\":{acyclic},\"unique_internals\":{internals},\"unique_originals\":{originals},\"candidate_tokens_len\":{}}}",
            candidates.map_or(-1i64, |c| c.len() as i64),
        ));
    };
    for shard in &overlay.segmented_boundary_shards {
        let (backend, parser, ito) = match &shard.backend {
            crate::runtime::SegmentedBoundaryShardBackend::StaticParser(b) => (
                "StaticParser",
                b.recursive_parser_dwa.as_ref().or(Some(&b.parser_dwa)),
                Some(b.internal_token_to_originals.as_slice()),
            ),
            crate::runtime::SegmentedBoundaryShardBackend::DynamicTerminalTrie(_) => {
                ("DynamicTerminalTrie", None, None)
            }
            crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect => {
                ("DynamicDirect", None, None)
            }
        };
        report(
            "overlay".to_string(),
            shard.start_component,
            backend,
            parser,
            ito,
            shard.candidate_tokens.as_deref(),
        );
    }
    for (index, component) in overlay.segmented_parser_components.iter().enumerate() {
        if let Some(shard) = component.boundary.as_ref() {
            let (backend, parser, ito) = match &shard.backend {
                crate::runtime::SegmentedBoundaryShardBackend::StaticParser(b) => (
                    "StaticParser",
                    b.recursive_parser_dwa.as_ref().or(Some(&b.parser_dwa)),
                    Some(b.internal_token_to_originals.as_slice()),
                ),
                crate::runtime::SegmentedBoundaryShardBackend::DynamicTerminalTrie(_) => {
                    ("DynamicTerminalTrie", None, None)
                }
                crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect => {
                    ("DynamicDirect", None, None)
                }
            };
            report(
                format!("component[{index}]"),
                shard.start_component,
                backend,
                parser,
                ito,
                shard.candidate_tokens.as_deref(),
            );
        }
    }
    for row in &rows {
        println!("{row}");
    }
    assert!(!rows.is_empty(), "artifact must carry at least one boundary shard");
}
