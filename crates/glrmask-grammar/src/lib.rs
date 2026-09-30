#![deny(warnings)]
#![allow(dead_code)]
#![allow(unused_variables)]

use thiserror::Error as ThisError;

#[derive(ThisError, Debug)]
pub enum Error {
    #[error("Grammar parse error: {0}")]
    GrammarParse(String),
}

pub type Result<T> = std::result::Result<T, Error>;
pub(crate) type GlrMaskError = Error;

pub(crate) mod automata {
    pub(crate) use glrmask_finite_automata::unweighted_u32;
    pub(crate) use glrmask_lexer::__private::automata::lexer;
    pub(crate) use glrmask_lexer::__private::automata::regex;
}

pub(crate) mod ds {
    pub(crate) use glrmask_lexer::__private::ds::u8set;
}

mod grammar;

pub(crate) mod import {
    pub(crate) use crate::grammar::ast;
    pub(crate) mod ebnf;
    pub(crate) mod lark;

    pub(crate) fn choice_or_single(mut options: Vec<ast::GrammarExpr>) -> ast::GrammarExpr {
        if options.len() == 1 {
            options.pop().unwrap()
        } else {
            ast::GrammarExpr::Choice(options)
        }
    }

    pub(crate) fn sequence_or_single(mut items: Vec<ast::GrammarExpr>) -> ast::GrammarExpr {
        match items.len() {
            0 => ast::GrammarExpr::Sequence(Vec::new()),
            1 => items.pop().unwrap(),
            _ => ast::GrammarExpr::Sequence(items),
        }
    }
}

pub use grammar::ast::{GrammarExpr, NamedGrammar, NamedRule, Quantifier};
pub use grammar::ast::parser_grammar::{ParserAutomaton, ParserAutomatonState,
    ParserExpr, ParserGrammar, ParserRule, CfgRecursion, FlatParserGrammar,
    ParserAnalysis, ParserProduction, ParserSymbol};
pub use import::ebnf::parse_ebnf_to_named;
pub use import::lark::{parse_lark_to_named, parse_lark_to_named_uncompressed};

/// Implementation details shared by the GLRMask workspace.
///
/// This module is deliberately feature-gated and is not a stable API.
#[cfg(feature = "internal-api")]
#[doc(hidden)]
pub mod __private {
    pub mod grammar {
        pub mod ast {
            pub use crate::grammar::ast::*;
        }
        pub mod exact_subtraction_lowering {
            pub use crate::grammar::exact_subtraction_lowering::*;
        }
        pub mod expr_nfa {
            pub use crate::grammar::expr_nfa::*;
        }
        pub mod factoring {
            pub use crate::grammar::factoring::*;
        }
        pub mod flat {
            pub use crate::grammar::flat::*;
        }
        pub mod glrm {
            pub use crate::grammar::glrm::*;
        }
        pub mod named_simplify {
            pub use crate::grammar::named_simplify::*;
        }
        pub mod right_linear {
            pub use crate::grammar::right_linear::*;
        }
        pub mod terminal_choice_promotion {
            pub use crate::grammar::terminal_choice_promotion::*;
        }

    }

    pub mod import {
        pub mod ebnf {
            pub use crate::import::ebnf::*;
        }
        pub mod lark {
            pub use crate::import::lark::*;
        }
    }
}
