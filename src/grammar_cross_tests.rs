use crate::grammar::ast::lower;

#[test]
fn hash_collision_merge_keeps_exact_masks_commits_and_loaded_language() {
    use std::collections::{BTreeSet, HashSet};
    use crate::compiler::glr::analysis::merge_identical_nonterminals;
    use crate::compiler::glr::table::GlrTableConstruction;
    use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};
    use crate::runtime::parser_backend::DynamicParserBackend;

    let left = [[10, 122], [163, 97], [271, 586], [443, 404]];
    let right = [[549, 362], [683, 617], [791, 48], [900, 749]];
    let mut rules = vec![
        Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] },
        Rule { lhs: 0, rhs: vec![Symbol::Terminal(1024), Symbol::Nonterminal(2)] },
    ];
    let literal = |id: u32| format!("{id:04x}").into_bytes();
    let mut words: HashSet<Vec<u8>> = (0..=1024).map(literal).collect();
    for id in 0..=1024 {
        rules.push(Rule { lhs: 0, rhs: vec![Symbol::Terminal(id)] });
    }
    for (lhs, alternatives) in [(1, left), (2, right)] {
        for pair in alternatives {
            rules.push(Rule { lhs, rhs: pair.into_iter().map(Symbol::Terminal).collect() });
            let mut word = if lhs == 2 { literal(1024) } else { Vec::new() };
            word.extend(literal(pair[0])); word.extend(literal(pair[1]));
            words.insert(word);
        }
    }
    assert_eq!(words.len(), 1033);
    let invalid = right.into_iter().map(|pair| {
        let mut word = literal(pair[0]); word.extend(literal(pair[1])); word
    }).chain(left.into_iter().map(|pair| {
        let mut word = literal(1024); word.extend(literal(pair[0]));
        word.extend(literal(pair[1])); word
    })).collect::<Vec<_>>();
    assert!(invalid.iter().all(|word| !words.contains(word)));

    // Keep exact witness IDs at the merge boundary. Frontend terminal
    // renumbering is deliberately not assumed to preserve a hash collision.
    // The downstream compiler and matcher paths are the real implementations.
    let grammar = GrammarDef {
        rules: merge_identical_nonterminals(&rules, 0), start: 0,
        terminals: (0..=1024).map(|id| Terminal::Literal { id, bytes: literal(id) }).collect(),
        ..GrammarDef::default()
    };
    let mut token_bytes = b"0123456789abcdef".iter().map(|&byte| vec![byte]).collect::<Vec<_>>();
    token_bytes.extend((0..=1024).map(literal));
    token_bytes.extend(invalid.iter().cloned());
    let vocab = crate::Vocab::new(token_bytes.iter().enumerate()
        .map(|(id, bytes)| (id as u32, bytes.clone())).collect());
    let prefixes: BTreeSet<Vec<u8>> = words.iter().flat_map(|word|
        (0..=word.len()).map(|end| word[..end].to_vec())).collect();
    let prefix_lookup: HashSet<_> = prefixes.iter().cloned().collect();
    // Independently precompute complete model-token masks from the finite
    // language, including byte tokens, whole terminals and invalid cross-token
    // words. Never use another engine's masks as this regression's oracle.
    let expected_masks = prefixes.iter().map(|prefix| {
        let mut mask = vec![0u32; token_bytes.len().div_ceil(32)];
        for (token, bytes) in token_bytes.iter().enumerate() {
            let mut extended = prefix.clone(); extended.extend(bytes);
            if prefix_lookup.contains(&extended) { mask[token / 32] |= 1 << (token % 32); }
        }
        mask
    }).collect::<Vec<_>>();

    for mode in ["dynamic", "o2", "static"] {
        let constraint = match mode {
            "dynamic" => crate::compiler::pipeline::compile_dynamic_owned_with_backend(
                grammar.clone(), &vocab, GlrTableConstruction::Lalr,
                DynamicParserBackend::LrTable).unwrap().inner,
            "o2" => crate::compiler::pipeline::compile_dynamic_owned_with_vocab_partition_for_parser_replacement(
                grammar.clone(), &vocab, GlrTableConstruction::Lalr).unwrap().inner,
            _ => crate::compiler::pipeline::compile_owned(grammar.clone(), &vocab),
        };
        if mode == "o2" { assert!(constraint.has_template_parser() && !constraint.table.is_present()); }
        let loaded = crate::Constraint::load(constraint.save()).unwrap();
        for (variant, c) in [("built", &constraint), ("loaded", &loaded)] {
            for (prefix, expected) in prefixes.iter().zip(&expected_masks) {
                let mut state = c.start(); state.commit_bytes(prefix).unwrap();
                assert_eq!(state.is_accepting(), words.contains(prefix), "{mode}/{variant} EOF {prefix:?}");
                assert_eq!(&state.mask(), expected, "{mode}/{variant} mask {prefix:?}");
                for (token, bytes) in token_bytes.iter().enumerate() {
                    if expected[token / 32] & (1 << (token % 32)) == 0 { continue; }
                    let mut branch = state.clone(); branch.commit_token(token as u32).unwrap();
                    let mut extended = prefix.clone(); extended.extend(bytes);
                    assert_eq!(branch.is_accepting(), words.contains(&extended),
                        "{mode}/{variant} token commit {prefix:?}+{bytes:?}");
                }
            }
            for word in &invalid {
                let mut state = c.start();
                assert!(state.commit_bytes(word).is_err() || !state.is_accepting(),
                    "{mode}/{variant} false acceptance {word:?}");
            }
        }
    }
}

#[test]
fn direct_regular_metadata_survives_compile_preparation() {
    let mut source = String::from("start: s0\n");
    for index in 0..40 {
        source.push_str(&format!("s{index}: A s{} | B\n", index + 1));
    }
    source.push_str("s40: B\nA: /a/\nB: /b/\n");

    let mut named = crate::import::lark::parse_lark_to_named_uncompressed(&source).unwrap();
    assert!(crate::grammar::right_linear::compress_large_right_linear_grammar(&mut named));
    let factored = crate::grammar::factoring::factor_named_grammar(named);
    let lowered = crate::grammar::ast::lower(&factored).unwrap();
    assert!(
        lowered.direct_regular_automaton.is_some(),
        "AST lower lost direct regular metadata"
    );
    let prepared =
        crate::compiler::grammar::transforms::prepare_grammar_transforms_only(lowered);
    assert!(
        prepared.direct_regular_automaton.is_some(),
        "grammar transforms lost direct regular metadata"
    );
    let analyzed = crate::compiler::glr::analysis::AnalyzedGrammar::from_grammar_def(&prepared);
    assert!(
        analyzed.direct_regular_automaton.is_some(),
        "analysis lost direct regular metadata"
    );
}

#[test]
fn lexer_groups_round_trip_and_control_tokenizer_partitions() {
    let grammar = crate::grammar::glrm::from_glrm(
        r#"
start s;
lexer group words ::= A, B;
t A ::= "a";
t B ::= "ab";
t C ::= "z";
nt s ::= A | B | C;
"#,
    )
    .unwrap();
    assert_eq!(
        grammar.lexer_partitions.get("A").map(String::as_str),
        Some("words")
    );
    assert_eq!(
        grammar.lexer_partitions.get("B").map(String::as_str),
        Some("words")
    );
    assert!(!grammar.lexer_partitions.contains_key("C"));

    let dumped = crate::grammar::glrm::to_glrm(&grammar);
    assert!(dumped.contains("lexer group words ::= A, B;"), "{dumped}");
    let reparsed = crate::grammar::glrm::from_glrm(&dumped).unwrap();
    assert_eq!(reparsed.lexer_partitions, grammar.lexer_partitions);

    let lowered = lower(&grammar).unwrap();
    assert_eq!(lowered.lexer_partitions.len(), 2);
    let tokenizer = crate::compiler::pipeline::build_tokenizer_with_partition_options(
        &lowered,
        false,
        false,
    );
    assert!(tokenizer.has_epsilon_transitions());
    assert_eq!(
        tokenizer.initial_epsilon_branch_count(),
        2,
        "A/B should share one component while unspecified C is isolated in stress mode",
    );

}
