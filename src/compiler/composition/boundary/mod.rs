//! Crossing-token summaries, support certificates, and boundary construction.

pub(crate) mod env;
pub(crate) mod transfer;
pub(crate) mod bit_minimize;
pub(crate) mod weight_codec;
pub(crate) mod stack_support;
pub(crate) mod preimage;
pub(crate) mod tagged_templates;
pub(crate) mod query_view;
pub(crate) mod query_terminals;
pub(crate) mod token_support;
pub(crate) mod cut_support;
pub(crate) mod terminal_summary;
pub(crate) mod scoped_follow;
pub(crate) mod scoped_follow_delta;
pub(crate) mod flat_transitions;
pub(crate) mod candidates;
pub(crate) mod tail;
pub(crate) mod walk;
pub(crate) mod prefix_dominance;
pub(crate) mod first_completion;
pub(crate) mod finite_lexer;
pub(crate) mod precomputed_completion;
