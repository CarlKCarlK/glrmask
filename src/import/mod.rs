//! Grammar source import and private compilation adapters.
//!
//! Lowering owns source semantics. Static and dynamic adapters retain their
//! distinct table policies; linked grammars and direct artifact production are
//! separated from ordinary parsing.

pub use crate::grammar::ast;
pub(crate) use glrmask_grammar::__private::import::{ebnf, lark};
pub mod json_schema;

mod bindings;
mod dynamic_compile;
mod lowering;
mod programmatic;
mod serialized;
mod static_compile;

#[cfg(test)]
use crate::grammar::factoring::factor_named_grammar;
#[cfg(test)]
use crate::runtime::Constraint;
#[cfg(test)]
use crate::runtime::dynamic::DynamicConstraint;
pub(crate) use bindings::external_placeholder_token_id_avoiding;
#[cfg(test)]
use bindings::first_external_placeholder_token_id;
#[doc(hidden)]
#[allow(unused_imports)]
pub use lowering::__profile_json_schema_import;
#[cfg(all(test, windows))]
use lowering::LARGE_IMPORT_SOURCE_BYTES;
pub(crate) use lowering::choice_or_single;
pub(crate) use lowering::lower_source_for_vocab_partition;
#[cfg(test)]
use lowering::parse_json_schema_to_named_dynamic;
#[cfg(test)]
use lowering::prepare_json_schema_named;
// Retain this existing crate-internal diagnostic entry point.
#[allow(unused_imports)]
pub(crate) use lowering::sequence_or_single;
#[cfg(all(test, windows))]
use lowering::with_large_import_stack;
pub(crate) use static_compile::compile_from_named_grammar;
// Retain this existing crate-internal diagnostic entry point.
#[allow(unused_imports)]
pub(crate) use static_compile::compile_glrm_with_protected_shift_terminals;

#[cfg(test)]
mod tests;
