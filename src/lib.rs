#![recursion_limit = "256"]

//! Extremely fast grammar-constrained decoding for language models.
//!
//! GLRMask has three core immutable/reusable layers:
//!
//! - [`Grammar`] describes source grammar semantics and bindings.
//! - [`UnlinkedConstraint`] stores reusable compiled machinery for one exact
//!   [`Vocab`] in pre-link form; it may retain unresolved request-specific slots.
//! - [`Constraint`] is closed and immediately runnable. Call [`Constraint::start`]
//!   to create one mutable [`ConstraintState`] per generated sequence.
//!
//! `Grammar::bind` and `UnlinkedConstraint::bind` are immutable: they return a
//! new value and leave the receiver reusable.
//!
//! # Quick start
//!
//! ```
//! use glrmask::{Grammar, Vocab};
//!
//! let vocab = Vocab::new(vec![
//!     (0, b"hello".to_vec()),
//!     (1, b" ".to_vec()),
//!     (2, b"world".to_vec()),
//! ]);
//! let constraint = Grammar::from_ebnf(r#"start ::= "hello" " " "world""#)
//!     .compile(&vocab)
//!     .unwrap();
//!
//! let mut state = constraint.start();
//! assert_ne!(state.mask()[0] & (1 << 0), 0);
//! state.commit_token(0).unwrap();
//! assert_ne!(state.mask()[0] & (1 << 1), 0);
//! state.commit_token(1).unwrap();
//! state.commit_token(2).unwrap();
//! assert!(state.is_accepting());
//! ```
//!
//! Rust masks are packed `u32` bitsets. Bit `token_id % 32` of word
//! `token_id / 32` indicates whether that token is allowed.
//!
//! # Typed bindings
//!
//! Native GLRM can declare `extern grammar child;` and `extern token MARK;`.
//! Both use [`Grammar::bind`]. Exact tokens come from [`Vocab::token`] or
//! [`Vocab::tokens`], so a binding carries the vocabulary identity it belongs to
//! rather than accepting an unqualified integer.
//!
//! Source grammars compose with source grammars. Compiled children are attached
//! only to [`UnlinkedConstraint`] values, keeping source compilation and compiled
//! linking as separate lifecycle stages.
//!
//! # Cached parents
//!
//! Use [`Grammar::compile_unlinked`] when a compiled parent will be reused. Bind
//! compiled request-specific [`Constraint`] children or exact-token values with
//! [`UnlinkedConstraint::bind`], then call [`UnlinkedConstraint::link`] or
//! [`UnlinkedConstraint::link_with`]. Binding itself does not compile
//! cross-component boundaries; final link options select the build/runtime trade-off.
//!
//! # Final options and termination
//!
//! [`BuildOptions`] belongs only to the final compile/link operation. In
//! particular, `end_tokens` is generation-root policy: an end token is admitted
//! only once the grammar is accepting and is not inherited if that constraint is
//! later embedded as a child. [`Optimization`] expresses build/runtime intent
//! without exposing GLRMask's internal engines.
//!
//! # Persistence
//!
//! [`UnlinkedConstraint::save`] / [`UnlinkedConstraint::load`] retain open
//! compiled bindings. A
//! [`Constraint`] has its own [`Constraint::save`] / [`Constraint::load`] artifact
//! and remains embeddable after loading. Artifacts are pre-release formats and
//! are not yet promised to remain compatible across GLRMask releases.

#![deny(warnings)]
#![allow(dead_code)]
#![allow(unused_variables)]

pub(crate) mod automata;
pub(crate) mod compiler;
pub(crate) mod ds;
mod error;
mod api;
pub(crate) use glrmask_grammar::__private::grammar;
pub(crate) mod import;
pub(crate) mod programmatic_js;
pub(crate) mod runtime;
pub(crate) use glrmask_vocab::__private as vocab;

pub use runtime::{Constraint, ConstraintState};
pub use glrmask_vocab::{ExactToken, ExactTokens, Vocab};
pub use error::{Error, Result};
pub use api::{BuildOptions, Grammar, Optimization, UnlinkedConstraint};

/// Model token identifier.
pub type TokenId = u32;

/// Unstable compatibility surface for repository tooling and benchmark
/// harnesses. It is deliberately absent from normal dependency builds.
#[cfg(any(test, feature = "internal-api"))]
#[doc(hidden)]
pub use runtime::dynamic::{DynamicConstraint, DynamicConstraintState};
#[cfg(any(test, feature = "internal-api"))]
#[doc(hidden)]
pub use runtime::BoundaryTriggerDetail;
#[cfg(any(test, feature = "internal-api"))]
#[doc(hidden)]
pub use api::{
    BoundarySummaryPolicy, ConstraintSpec, ConstraintSpecBuilder, VocabPartition,
    VocabPartitionStrategy,
};

#[cfg(test)]
pub(crate) static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod grammar_cross_tests;
#[cfg(test)]
mod terminal_dwa_cross_tests;

pub(crate) use error::GlrMaskError;

/// Compile a Constraint from a serialized GrammarDef JSON + vocab.
/// This runs the full compile pipeline (equivalence analysis, terminal DWA, parser DWA).
pub(crate) fn compile_grammar_def_json(
    grammar_def_json: &str,
    vocab: &Vocab,
) -> Result<Constraint> {
    let gdef: grammar::flat::GrammarDef = serde_json::from_str(grammar_def_json)
        .map_err(|e| GlrMaskError::GrammarParse(format!("invalid GrammarDef JSON: {e}")))?;
    error::catch_internal_invariant(|| {
        compiler::stages::id_map_and_terminal_dwa::l2p::with_ti_pool(|| {
            compiler::compile_owned(gdef, vocab)
        })
    })
}

/// Populate compile-time artifacts that are pure functions of the vocabulary.
///
/// This intentionally does not compile any grammar/schema-dependent artifact.
pub(crate) fn prepare_vocab_for_compile(vocab: &Vocab) {
    compiler::compile::prepare_vocab_for_compile(vocab);
}

/// Populate compile-time vocabulary artifacts needed specifically by dynamic mode.
pub(crate) fn prepare_vocab_for_dynamic_compile(vocab: &Vocab) {
    compiler::compile::prepare_vocab_for_dynamic_compile(vocab);
}

/// Build (and, if configured, start the keepalive for) the terminal
/// interchangeability certification thread pool ahead of first use.
///
/// Calling this at Python module import warms the pool so discovery does not
/// pay the first-use worker-wake handoff (a large latency on macOS).
pub(crate) fn warm_ti_pool() {
    compiler::stages::id_map_and_terminal_dwa::l2p::warm_ti_pool();
}

/// Dump the imported JSON Schema grammar in GLRM format.
///
/// This intentionally preserves exact subtraction syntax so dumps reflect the
/// source-level structure. The compile/import pipeline may still apply exact
/// subtraction lowering.
pub(crate) fn dump_json_schema_grammar_glrm(schema_json: &str) -> Result<String> {
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| GlrMaskError::GrammarParse(format!("invalid JSON: {e}")))?;
    let named = import::json_schema::schema_to_named_grammar(&schema)?;
    let mut factored = grammar::factoring::factor_named_grammar(named);
    import::json_schema::prepare_named_grammar_for_dump(&mut factored)?;
    Ok(grammar::glrm::to_glrm(&factored))
}

pub(crate) fn set_test_compat_mode(enabled: bool) {
    glrmask_json_schema::__private::set_test_compat_mode(enabled);
}

#[cfg(feature = "internal-api")]
#[doc(hidden)]
pub mod __private;
