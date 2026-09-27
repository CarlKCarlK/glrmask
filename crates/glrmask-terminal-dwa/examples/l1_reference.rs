//! Tiny driver and equivalence test for the swappable L1 implementations.

use std::sync::Arc;

use glrmask_glr::__private::glr::analysis::AnalyzedGrammar;
use glrmask_grammar::__private::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};
use glrmask_lexer::__private::automata::lexer::tokenizer::Tokenizer;
#[cfg(test)]
use glrmask_lexer::__private::automata::lexer::{ast::bytes, compile::build_regex};
use glrmask_terminal_dwa::__private::terminal_dwa::{
    l1::{
        build_flat_transition_table,
        implementations::{BuildInput, Implementation, Plan, build_with_plan},
    },
    types::TerminalColoring,
};
use glrmask_vocab::Vocab;

fn grammar() -> AnalyzedGrammar {
    AnalyzedGrammar::from_grammar_def(&GrammarDef {
        rules: vec![Rule { lhs: 0, rhs: vec![Symbol::Terminal(0)] }],
        start: 0,
        terminals: (0..2)
            .map(|id| Terminal::Literal { id, bytes: vec![b'a' + id as u8] })
            .collect(),
        ..GrammarDef::default()
    })
}

fn run(tokenizer: &Tokenizer, vocab: &Vocab, active: &[bool], plan: Plan) {
    let grammar = grammar();
    let coloring = TerminalColoring::identity(2);
    let flat = Arc::from(build_flat_transition_table(tokenizer));
    let result = build_with_plan(
        BuildInput {
            partition_label: "reference",
            tokenizer,
            vocab,
            terminal_coloring: &coloring,
            use_terminal_coloring: false,
            ignore_terminal: None,
            grammar: &grammar,
            active_terminals: active,
            flat_trans: &flat,
            transitions_by_byte: None,
            initial_state_map: None,
            initial_state_domain_is_exact: false,
            shared_generic_nfa_topology: None,
            shared_generic_nfa_trie: None,
            subset_parent_order: None,
            id_map_only: false,
        },
        plan,
    )
    .expect("L1 result");
    println!("states={} transitions={} tsids={} tokens={}", result.dwa.num_states(), result.dwa.num_transitions(), result.id_map.num_tsids(), result.id_map.num_internal_tokens());
}

fn main() {
    let tokenizer = glrmask_lexer::__private::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer();
    let vocab = Vocab::new(vec![(0, vec![]), (1, b"a".to_vec()), (2, b"a".to_vec()), (3, b"aa".to_vec()), (4, b"b".to_vec())]);
    run(&tokenizer, &vocab, &[true, true], Plan { use_implementation: Implementation::Scalar, check_against: Some(Implementation::Quotient) });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_both_directions(tokenizer: Tokenizer, vocab: Vocab) {
        for active in [[true, true], [true, false], [false, true]] {
            for implementation in [
                Implementation::Projected,
                Implementation::Quotient,
                Implementation::Auto,
                Implementation::Scalar,
                Implementation::Trie,
                Implementation::Bulk,
                Implementation::Dense,
                Implementation::Frontier,
            ] {
                let checker = if implementation == Implementation::Quotient {
                    Implementation::Projected
                } else {
                    Implementation::Quotient
                };
                run(&tokenizer, &vocab, &active, Plan { use_implementation: implementation, check_against: Some(checker) });
            }
        }
    }

    #[test]
    fn epsilon_nfa_implementations_are_equivalent() {
        check_both_directions(
            glrmask_lexer::__private::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer(),
            Vocab::new(vec![(0, vec![]), (1, b"a".to_vec()), (2, b"a".to_vec()), (3, b"aa".to_vec()), (4, b"b".to_vec()), (5, b"x".to_vec())]),
        );
    }

    #[test]
    fn deterministic_implementations_are_equivalent() {
        let exprs = vec![bytes(b"a"), bytes(b"ab")];
        let tokenizer = build_regex(&exprs).into_tokenizer(2, Some(Arc::from(exprs.into_boxed_slice())));
        check_both_directions(tokenizer, Vocab::new(vec![(0, vec![]), (1, b"a".to_vec()), (2, b"ab".to_vec()), (3, b"aba".to_vec()), (4, b"x".to_vec())]));
    }
}
