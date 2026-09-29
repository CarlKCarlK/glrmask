#![deny(warnings)]
#![allow(dead_code)]
#![allow(unused_variables)]

pub(crate) use glrmask_vocab::Vocab;

fn optimized_env_value_enabled(value: &str) -> bool {
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "" | "0" | "false" | "no" | "off"
    )
}

pub(crate) fn optimized_env_flag(name: &str) -> bool {
    std::env::var_os(name)
        .map(|value| optimized_env_value_enabled(&value.to_string_lossy()))
        .unwrap_or(true)
}

#[cfg(test)]
mod optimized_env_policy_tests {
    use super::*;

    #[test]
    fn accepted_boolean_override_values_are_unambiguous() {
        for value in ["", "0", "false", "FALSE", " no ", "Off"] {
            assert!(!optimized_env_value_enabled(value), "{value:?}");
        }
        for value in ["1", "true", "yes", "on", "garbage"] {
            assert!(optimized_env_value_enabled(value), "{value:?}");
        }
    }
}

pub(crate) mod automata {
    pub(crate) use glrmask_finite_automata::unweighted_u32;
    pub(crate) use glrmask_weighted_automata::weighted_u32;
    pub(crate) use weighted_u32 as weighted;
}

pub(crate) mod ds {
    pub(crate) use glrmask_lexer::__private::ds::bitset;
    pub(crate) use glrmask_weight::__private as weight;
}

pub(crate) mod grammar {
    pub(crate) use glrmask_grammar::__private::grammar::*;
}

pub(crate) mod runtime {
    pub(crate) use glrmask_artifact::__private::CommitTemplateDfas;
}

pub(crate) mod compiler {
    pub(crate) use glrmask_glr::__private::glr;

    pub(crate) mod stages {
        pub(crate) use crate::resolve_negatives;
        pub(crate) use crate::templates;
        pub(crate) use glrmask_artifact::__private::equiv_types;
    }
}

#[cfg(feature = "internal-api")]
pub(crate) mod merge;
pub(crate) mod parser_dwa;
pub(crate) mod parser_equivalence;
pub(crate) mod resolve_negatives;
pub(crate) mod templates;

/// Implementation details shared by the GLRMask workspace.
///
/// This module is deliberately feature-gated and is not a stable API.
#[cfg(feature = "internal-api")]
#[doc(hidden)]
pub mod __private {
    pub mod merge {
        pub use crate::merge::*;
    }
    pub mod parser_dwa {
        pub use crate::parser_dwa::*;
    }
    pub mod parser_equivalence {
        pub use crate::parser_equivalence::*;
    }
    pub mod resolve_negatives {
        pub use crate::resolve_negatives::*;
    }
    pub mod templates {
        pub use crate::templates::*;
        pub mod admissibility {
            pub use crate::templates::admissibility::*;
        }
        pub mod characterize {
            pub use crate::templates::characterize::*;
        }
        pub mod compile_bundle {
            pub use crate::templates::compile_bundle::*;
        }
        pub mod compile_dfa {
            pub use crate::templates::compile_dfa::*;
        }
    }
}
