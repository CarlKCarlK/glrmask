use super::{PartitionOptions, build_regex_partitioned_with_options};
use super::partition::compile_terminal_partitions;
use super::settings::adaptive_lexer_enabled;
use crate::automata::lexer::ast::Expr;

#[test]
fn named_partition_options_preserve_all_optional_input_combinations() {
    let exprs = [
        Expr::U8Seq(b"a".to_vec()),
        Expr::Choice(vec![Expr::U8Seq(b"ab".to_vec()), Expr::U8Seq(b"ac".to_vec())]),
        Expr::Repeat { expr: Box::new(Expr::U8Seq(b"a".to_vec())), min: 0, max: Some(4) },
    ];
    let partitions = [0, 1, 2];
    let labels = ["first".to_owned(), "choice".to_owned(), "repeat".to_owned()];
    let isolation = [Some(0), None, Some(1)];
    for adaptive in [None, Some(false), Some(true)] {
        for profile_labels in [None, Some(labels.as_slice())] {
            for residual_isolation_classes in [None, Some(isolation.as_slice())] {
                let expected = compile_terminal_partitions(
                    &exprs, profile_labels, &partitions, residual_isolation_classes,
                    adaptive.unwrap_or_else(adaptive_lexer_enabled),
                );
                let actual = build_regex_partitioned_with_options(
                    &exprs, &partitions,
                    PartitionOptions { profile_labels, residual_isolation_classes, adaptive },
                );
                assert_eq!(bincode::serialize(&actual.dfa).unwrap(), bincode::serialize(&expected).unwrap(), "option forwarding changed the compiled automaton");
            }
        }
    }
}
