//! A compile-time top-down parser using only GLRMask's public API.
//!
//! Each alternative must start with a terminal; its remaining items can be
//! terminals or nonterminals. Recursion after that first terminal is allowed.
//! This deliberately small compiler rejects other forms instead of quietly
//! changing their language. No LR table or per-token user callback is involved.
use std::collections::{BTreeMap, BTreeSet};
use glrmask::{Constraint, Error, Grammar, Optimization, Result, Vocab};
use glrmask::template_parser::{ParserCompiler, ParserDefinition, ParserExpr,
    ParserGrammar, StackState, StackTemplate, StackTransition, StackLabel, TemplateBuildOptions};

struct TerminalLeading;

fn unsupported() -> Error {
    Error::Compilation("example compiler requires terminal-leading alternatives with atomic tails; normalize other forms in your own compiler".into())
}

/// One POP symbol can have several output words. Encode their shared prefix
/// trie rather than enumerating runtime paths or introducing nondeterminism
/// into an individual DFA row. The shared GSS retains the resulting branches.
fn relation(alternatives: BTreeMap<u32, BTreeSet<Vec<u32>>>) -> StackTemplate {
    let mut result = StackTemplate::reject(); result.pop_to_push.push(None);
    for (top, words) in alternatives {
        let popped = result.pop.states.len() as u32;
        result.pop.states.push(StackState::default());
        result.pop.states[0].transitions.push(StackTransition { label: StackLabel::Symbol(top), target: popped });
        let root = result.push.states.len() as u32;
        result.push.states.push(StackState::default()); result.pop_to_push.push(Some(root));
        for word in words {
            let mut state = root;
            for symbol in word {
                let next = result.push.states[state as usize].transitions.iter()
                    .find(|edge| edge.label == StackLabel::Symbol(symbol)).map(|edge| edge.target);
                state = if let Some(next) = next { next } else {
                    let target = result.push.states.len() as u32;
                    result.push.states.push(StackState::default());
                    result.push.states[state as usize].transitions.push(StackTransition { label: StackLabel::Symbol(symbol), target });
                    target
                };
            }
            result.push.states[state as usize].accepting = true;
        }
    }
    result
}

impl ParserCompiler for TerminalLeading {
    fn compile(&self, grammar: &ParserGrammar) -> Result<ParserDefinition> {
        // Reject cycles without finite derivations: their apparent prefixes
        // can never complete and must not be advertised by a token mask.
        let mut productive = vec![false; grammar.rules().len()];
        loop {
            let mut changed = false;
            for (id, rule) in grammar.rules().iter().enumerate() {
                let alternatives = match &rule.expr { ParserExpr::Choice(items) => items.as_slice(), other => std::slice::from_ref(other) };
                let derives_word = alternatives.iter().any(|alternative| {
                    let sequence = match alternative { ParserExpr::Sequence(items) => items.as_slice(), other => std::slice::from_ref(other) };
                    matches!(sequence.first(), Some(ParserExpr::Terminal(_))) && sequence.iter().all(|item| match item {
                        ParserExpr::Terminal(id) => Some(*id) != grammar.ignore_terminal(),
                        ParserExpr::Nonterminal(id) => productive[*id as usize],
                        _ => false,
                    })
                });
                if derives_word && !productive[id] { productive[id] = true; changed = true; }
            }
            if !changed { break; }
        }
        if productive.iter().any(|value| !value) {
            return Err(Error::Compilation("example compiler requires every rule to derive a finite terminal word".into()));
        }
        // 0 = initial source state; 1 = completed bottom marker; subsequent
        // symbols describe pending nonterminals and literal terminal identities.
        let rule_count = u32::try_from(grammar.rules().len()).map_err(|_| unsupported())?;
        let terminal_offset = rule_count.checked_add(2).ok_or_else(unsupported)?;
        let alphabet = terminal_offset.checked_add(grammar.terminal_count()).ok_or_else(unsupported)?;
        let mut actions = vec![BTreeMap::<u32, BTreeSet<Vec<u32>>>::new(); grammar.terminal_count() as usize];
        for terminal in 0..grammar.terminal_count() {
            actions[terminal as usize].entry(terminal_offset + terminal).or_default().insert(vec![]);
        }
        for (id, rule) in grammar.rules().iter().enumerate() {
            let alternatives = match &rule.expr { ParserExpr::Choice(items) => items.as_slice(), other => std::slice::from_ref(other) };
            for alternative in alternatives {
                let sequence = match alternative { ParserExpr::Sequence(items) => items.as_slice(), other => std::slice::from_ref(other) };
                let Some(ParserExpr::Terminal(terminal)) = sequence.first() else { return Err(unsupported()); };
                if Some(*terminal) == grammar.ignore_terminal() { return Err(unsupported()); }
                let pending = sequence[1..].iter().rev().map(|item| match item {
                    ParserExpr::Terminal(id) if Some(*id) != grammar.ignore_terminal() => Ok(terminal_offset + id),
                    ParserExpr::Nonterminal(id) => Ok(2 + id),
                    _ => Err(unsupported()),
                }).collect::<Result<Vec<_>>>()?;
                actions[*terminal as usize].entry(2 + id as u32).or_default().insert(pending.clone());
                if id as u32 == grammar.start() {
                    let mut initial = vec![1]; initial.extend(pending);
                    actions[*terminal as usize].entry(0).or_default().insert(initial);
                }
            }
        }
        let mut terminals = actions.into_iter().map(relation).collect::<Vec<_>>();
        if let Some(ignore) = grammar.ignore_terminal() { terminals[ignore as usize] = StackTemplate::identity(); }
        Ok(ParserDefinition { stack_symbol_count: alphabet, terminals,
            completion: StackTemplate::read_top_and_push([1], []) })
    }
}

fn main() -> Result<()> {
    let grammar = Grammar::from_ebnf(r#"root ::= "a" | "(" root ")""#).prepare_parser_grammar()?;
    let parser = grammar.compile_parser(&TerminalLeading)?;
    let vocab = Vocab::new(vec![(0, b"(".to_vec()), (1, b")".to_vec()), (2, b"a".to_vec()),
        (3, b"((a))".to_vec()), (4, b"()".to_vec())]);
    for optimization in [Optimization::FastBuild, Optimization::FastRuntime] {
        let constraint = parser.compile_with(&vocab, TemplateBuildOptions::default().optimization(optimization))?;
        let mut state = constraint.start();
        assert_eq!(state.mask()[0] & (1 << 4), 0, "empty parentheses are not in this grammar");
        state.commit_token(3)?; assert!(state.is_accepting());
        let loaded = Constraint::load(constraint.save())?;
        let mut state = loaded.start(); state.commit_bytes(b"((((a))))")?; assert!(state.is_accepting());
        println!("{optimization:?}: recursive public constructor and saved template program passed");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dead_recursion_and_unsupported_shapes_are_errors() {
        for source in ["root ::= \"a\" root", "root ::= \"a\" | \"b\" dead\ndead ::= \"c\" dead",
            "root ::= \"a\"?", "root ::= child\nchild ::= \"a\""] {
            let prepared = Grammar::from_ebnf(source).prepare_parser_grammar().unwrap();
            assert!(prepared.compile_parser(&TerminalLeading).is_err(), "{source}");
        }
    }

    #[test]
    fn ambiguous_terminal_leading_alternatives_preserve_both_suffixes() {
        let prepared = Grammar::from_ebnf(r#"root ::= "a" "b" | "a" "c""#).prepare_parser_grammar().unwrap();
        let parser = prepared.compile_parser(&TerminalLeading).unwrap();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"c".to_vec()),
            (3, b"ab".to_vec()), (4, b"ac".to_vec()), (5, b"aa".to_vec())]);
        for mode in [Optimization::FastBuild, Optimization::FastRuntime] {
            let c = parser.compile_with(&vocab, TemplateBuildOptions::default().optimization(mode)).unwrap();
            for c in [&c, &Constraint::load(c.save()).unwrap()] {
                let mut state = c.start(); assert_eq!(state.mask()[0], 0b011001);
                state.commit_token(0).unwrap(); assert_eq!(state.mask()[0], 0b000110);
                for end in [1, 2] { let mut branch = state.clone(); branch.commit_token(end).unwrap(); assert!(branch.is_accepting()); }
            }
        }
    }
}
