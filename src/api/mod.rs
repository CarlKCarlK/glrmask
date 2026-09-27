//! Public lifecycle and internal compilation facade.
//!
//! Source binding, compilation, and compiled linking are separate stages.
//! Only the exports in `lib.rs` form the supported external API.

mod options;
mod unlinked;
mod bindings;
mod grammar;
mod partition;
mod compile;
pub use options::{BuildOptions, Optimization};
pub use unlinked::{UnlinkedConstraint};
use unlinked::{
    ModuleBinding, constraint_vocab, ensure_runnable_constraint, select_supported_boundary,
};
pub(crate) use bindings::{
    GrammarBinding, GrammarValue, IntoGrammarBinding, IntoGrammarValue, IntoUnlinkedValue,
    UnlinkedValue,
};
use bindings::{ExternKind};
pub use grammar::{Grammar};
use grammar::{GrammarSource};
pub use partition::VocabPartitionStrategy;
#[cfg(any(test, feature = "internal-api"))]
pub use partition::VocabPartition;
pub use compile::ConstraintSpec;
#[cfg(any(test, feature = "internal-api"))]
pub use compile::{BoundarySummaryPolicy, ConstraintSpecBuilder};
use compile::{
    ChildCompileMode, CompiledChild, MODULE_HEADER_LEN, MODULE_MAGIC, compose_named_children,
    prepare_compiled_children, static_constraint_targets,
};
#[cfg(test)]
mod tests;
#[cfg(test)]
mod final_unlinked_optimization_tests;
#[cfg(test)]
mod cached_parent_main_tests;


#[cfg(test)]
use std::sync::Arc;



#[cfg(test)]
use crate::runtime::Constraint as RuntimeConstraint;
#[cfg(test)]
use crate::runtime::dynamic::DynamicConstraint;

#[cfg(test)]
use crate::{Vocab};


#[cfg(test)]
use compile::{FIRST_SAVE_TEMPLATE_STATE_PRIME_THRESHOLD, should_prime_first_save_artifact};
