#![deny(warnings)]
#![allow(dead_code)]
#![allow(unused_variables)]

pub(crate) mod grammar {
    pub(crate) use glrmask_grammar::__private::grammar::*;
}

pub(crate) mod ds {
    pub(crate) use glrmask_lexer::__private::ds::bitset;
    pub(crate) mod leveled_gss;
    pub(crate) mod stack_vecs;
}

mod glr;

#[cfg(test)]
pub(crate) mod compiler {
    pub(crate) use crate::glr;
}

/// Implementation details shared by the GLRMask workspace.
///
/// This module is deliberately feature-gated and is not a stable API.
#[cfg(feature = "internal-api")]
#[doc(hidden)]
pub mod __private {
    pub mod ds {
        pub mod leveled_gss {
            pub use crate::ds::leveled_gss::*;
        }
        pub mod stack_vecs {
            pub use crate::ds::stack_vecs::*;
        }
    }

    pub mod glr {
        pub mod accumulator {
            pub use crate::glr::accumulator::*;
        }
        pub mod analysis {
            pub use crate::glr::analysis::*;
            // Explicit imports take precedence over the reference module's
            // glob imports. All other analysis APIs remain unchanged.
            pub use crate::glr::normalization::{
                eliminate_right_recursion, has_indirect_left_recursion, normalize_grammar,
            };
        }
        pub mod labels {
            pub use crate::glr::labels::*;
        }
        pub mod parser {
            pub use crate::glr::parser::*;
        }
        pub mod table {
            pub use crate::glr::table::*;
            pub mod action {
                pub use crate::glr::table::action::*;
            }
            pub mod row {
                pub use crate::glr::table::row::*;
            }
            pub mod testing {
                pub use crate::glr::table::testing::*;
            }
        }
    }
}
