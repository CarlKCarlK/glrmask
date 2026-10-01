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
mod public_api;
pub mod template_parser;
pub(crate) use glrmask_grammar::__private::grammar;
pub(crate) mod import;
pub(crate) mod programmatic_js;
pub(crate) mod runtime;
#[path = "runtime/dynamic_constraint.rs"]
mod dynamic_constraint;
pub(crate) use glrmask_vocab::__private as vocab;

pub use runtime::{Constraint, ConstraintState};
pub use glrmask_vocab::{ExactToken, ExactTokens, Vocab};
pub use error::{Error, Result};
pub use public_api::{BuildOptions, Grammar, Optimization, ParserBackend, UnlinkedConstraint};

/// Model token identifier.
pub type TokenId = u32;

/// Unstable compatibility surface for repository tooling and benchmark
/// harnesses. It is deliberately absent from normal dependency builds.
#[cfg(any(test, feature = "internal-api"))]
#[doc(hidden)]
pub use dynamic_constraint::{DynamicConstraint, DynamicConstraintState};
#[cfg(any(test, feature = "internal-api"))]
#[doc(hidden)]
pub use runtime::BoundaryTriggerDetail;
#[cfg(any(test, feature = "internal-api"))]
#[doc(hidden)]
pub use public_api::{
    BoundarySummaryPolicy, ConstraintSpec, ConstraintSpecBuilder, VocabPartition,
    VocabPartitionStrategy,
};

#[cfg(test)]
pub(crate) static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Run an environment-mutating test in its own process. A writers-only mutex
/// cannot protect unrelated tests that legitimately read production flags.
/// Call this from the libtest entry, before spawning any custom-stack worker.
/// Returns true in the parent after the exact child test has passed.
#[cfg(test)]
pub(crate) fn isolate_environment_test(ignored: bool) -> bool {
    const CHILD: &str = "GLRMASK_ISOLATED_ENVIRONMENT_TEST";
    let current = std::thread::current();
    let name = current.name().expect("libtest names each test thread");
    if std::env::var(CHILD).as_deref() == Ok(name) {
        return false;
    }
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command
        .arg(name)
        .arg("--exact")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(CHILD, name)
        .env_remove("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC");
    if ignored {
        command.arg("--ignored");
    }
    let output = command.output().expect("run isolated environment test");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && stdout.contains("test result: ok. 1 passed; 0 failed"),
        "isolated test {name} failed or did not run exactly one test: {}\n{stdout}\n{stderr}",
        output.status,
    );
    true
}

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
pub mod __private {
    pub use crate::runtime::boundary_cpu_profile::{
        begin as begin_boundary_cpu_profile, take as take_boundary_cpu_profile, BoundaryCpuReport,
    };
    #[derive(Debug, Clone, Copy, Default)]
    pub struct CompilerCacheStats {
        pub token_set_entries: usize,
        pub live_token_set_entries: usize,
        pub weight_buckets: usize,
        pub weight_entries: usize,
        pub live_weight_entries: usize,
        pub current_thread_weight_ops: usize,
        pub current_thread_token_set_ops: usize,
        pub current_thread_public_intersections: usize,
        pub current_thread_weight_hashes: usize,
        pub weight_op_generation: u64,
        pub weight_hash_generation: u64,
        pub vocab_artifacts: usize,
    }

    pub fn compiler_cache_stats(vocab: Option<&crate::Vocab>) -> CompilerCacheStats {
        let stats = crate::ds::weight::weight_cache_stats();
        CompilerCacheStats {
            token_set_entries: stats.token_set_entries,
            live_token_set_entries: stats.live_token_set_entries,
            weight_buckets: stats.weight_buckets,
            weight_entries: stats.weight_entries,
            live_weight_entries: stats.live_weight_entries,
            current_thread_weight_ops: stats.current_thread_weight_ops,
            current_thread_token_set_ops: stats.current_thread_token_set_ops,
            current_thread_public_intersections: stats.current_thread_public_intersections,
            current_thread_weight_hashes: stats.current_thread_weight_hashes,
            weight_op_generation: stats.weight_op_generation,
            weight_hash_generation: stats.weight_hash_generation,
            vocab_artifacts: vocab.map_or(0, crate::Vocab::compiler_cache_entry_count),
        }
    }

    pub use crate::compiler::glr::table::TableAmbiguity;
    pub use crate::error::Error;
    pub use crate::runtime::{
        AdvanceTrace,
        AdvanceTraceStep,
        CommitProfile,
        GssProfileSummary,
        MaskProfile,
        PerAdvanceEntry,
    };

    use crate::{Constraint, ConstraintState, DynamicConstraint, Vocab};

    pub type Result<T> = std::result::Result<T, Error>;

    /// Diagnostic upper bound for outgoing model tokens of ONE compiled
    /// component. No caller, sibling, placeholder binding or link graph is
    /// accepted here. An unavailable certificate is an error, never empty.
    pub fn debug_component_boundary_candidate_ids(component:&Constraint,vocab:&Vocab)->Result<Vec<u32>>{
        crate::compiler::boundary_candidates::boundary_candidate_ids(component,vocab).0
            .ok_or_else(||Error::Compilation("component-only outgoing boundary summary unavailable".into()))
    }

    /// Internal debug bridge: compile an ordinary
    /// [`crate::grammar::ast::NamedGrammar`] through the normal
    /// (non-composition) compiler path, using the same default table
    /// construction as the ordinary GLRM route
    /// (`Constraint::from_glrm_grammar` -> `ExperimentalCoreMerged`).
    pub fn compile_named_grammar(
        named: crate::grammar::ast::NamedGrammar,
        vocab: &Vocab,
    ) -> Result<Constraint> {
        crate::import::compile_from_named_grammar(
            named,
            vocab,
            "named_grammar_internal",
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
            &[],
        )
    }

    /// Internal release-only benchmark for exact o21137 subgrammar decomposition.
    pub fn run_o21137_subgrammar_benchmark(mode: &str) {
        crate::compiler::o21137_subgrammar_bench::run(mode);
    }

    /// Experimental mandatory template backend. The LR table is dropped and
    /// any accidental runtime table access fails loudly. Composition is not
    /// supported by this experimental internal bridge.
    pub fn into_template_parser(mut constraint: Constraint) -> Result<Constraint> {
        constraint.install_template_parser()?;
        Ok(constraint)
    }

    pub fn into_dynamic_template_parser(mut constraint: DynamicConstraint) -> Result<DynamicConstraint> {
        for alternative in constraint.constraints_mut() { alternative.install_template_parser()?; }
        Ok(constraint)
    }

    /// Check derived packed final-weight indices and projections against the
    /// original wire decoder and original-token fragments. Internal regression
    /// support only; never called by compilation, loading, or timed execution.
    pub fn assert_packed_final_mask_cache(constraint: &Constraint, rounds: usize) -> usize {
        constraint.check_packed_final_cache_against_uncached(rounds)
    }

    /// Check every cached tokenizer state/byte against its canonical decoder.
    /// Internal regression support only; do not call during a timing pass.
    pub fn assert_tokenizer_transition_cache(constraint: &Constraint) -> usize {
        constraint.check_tokenizer_fast_transitions_against_uncached()
    }

    pub fn parser_backend_report(constraint: &Constraint) -> serde_json::Value {
        constraint.parser_backend_report()
    }

    /// Native research gate for the bounded O2 frontend; compare against the
    /// ordinary O2 parser independently before adopting as a public policy.
    pub fn compile_bounded_template_o2_glrm(source: &str, vocab: &Vocab) -> Result<DynamicConstraint> {
        DynamicConstraint::from_glrm_with_bounded_template_parser(source, vocab)
    }

    pub fn dynamic_parser_backend_report(constraint: &DynamicConstraint) -> serde_json::Value {
        serde_json::Value::Array(constraint.clone_constraints().iter()
            .map(Constraint::parser_backend_report).collect())
    }

    /// Bounded diagnostic only. This compares canonical stack languages and
    /// accumulators, not pointer identity or incidental GSS sharing shape.
    /// The reference is a separate oracle constraint, never retained inside
    /// the table-free candidate. Do not call this inside performance intervals.
    pub fn compare_parser_states_debug(
        reference: &ConstraintState<'_>, candidate: &ConstraintState<'_>,
    ) -> Option<serde_json::Value> {
        use crate::compiler::glr::parser::ParserGSS;
        use std::collections::BTreeMap;
        fn canonical(state: &ConstraintState<'_>) -> BTreeMap<u32, ParserGSS> {
            let mut map = BTreeMap::<u32, ParserGSS>::new();
            for (&key, gss) in state.state.iter() {
                map.entry(key).and_modify(|prior| *prior = prior.merge(gss)).or_insert_with(|| gss.clone());
            }
            map
        }
        let left = canonical(reference); let right = canonical(candidate);
        let keys: std::collections::BTreeSet<_> = left.keys().chain(right.keys()).copied().collect();
        for key in keys {
            let a = left.get(&key).cloned().unwrap_or_else(ParserGSS::empty);
            let b = right.get(&key).cloned().unwrap_or_else(ParserGSS::empty);
            if a.semantically_eq(&b, 65_536) != Some(true) {
                return Some(serde_json::json!({
                    "tokenizer_state":key, "reference_keys":left.keys().collect::<Vec<_>>(),
                    "candidate_keys":right.keys().collect::<Vec<_>>(),
                    "reference_stacks":format!("{:?}",a.to_stacks(32)),
                    "candidate_stacks":format!("{:?}",b.to_stacks(32)),
                    "reference_stack_vectors":a.to_stacks(4096).map(|stacks|stacks.into_iter().map(|(stack,_)|stack).collect::<Vec<_>>()),
                    "candidate_stack_vectors":b.to_stacks(4096).map(|stacks|stacks.into_iter().map(|(stack,_)|stack).collect::<Vec<_>>()),
                    "reference_table":serde_json::to_value(&*reference.constraint.table).ok(),
                    "candidate_templates":serde_json::to_value(&candidate.constraint.template_dfas_by_terminal).ok(),
                    "ignore_terminal":reference.constraint.ignore_terminal,
                }));
            }
        }
        None
    }

    pub trait ConstraintExt: Sized {
        fn compile_grammar_def_json(grammar_def_json: &str, vocab: &Vocab) -> Result<Self>;
        fn dump_json_schema_grammar_glrm(schema_json: &str) -> Result<String>;
        fn profile_json_schema_import(schema_json: &str) -> Result<()>;
        fn warm_ti_pool();
        fn clear_stale_weights();
        fn clear_weight_interners();
        fn clear_weight_op_caches();
        fn set_test_compat_mode(enabled: bool);
        fn prepare_composition_grammar_summary(&mut self) -> Result<()>;

        fn bind_vocab_exact(&mut self, vocab: &Vocab) -> std::result::Result<(), String>;
        fn prepare_for_composition(&mut self, vocab: &Vocab) -> Result<()>;
        fn prepare_for_dynamic_composition(&mut self, vocab: &Vocab) -> Result<()>;

        fn num_parser_states(&self) -> u32;
        fn num_terminals(&self) -> u32;
        fn num_tokenizer_states(&self) -> usize;
        fn internal_tsid_count(&self) -> usize;
        fn parser_dwa_num_states(&self) -> u32;
        fn parser_dwa_num_transitions(&self) -> usize;
        fn compute_forced_minimized_tokenizer_state_count(&self) -> usize;
        fn max_original_token_id(&self) -> Option<u32>;
        fn final_internal_token_count(&self) -> usize;
        fn final_original_token_map(&self) -> Vec<u32>;
        fn table_ambiguous_actions(&self) -> Vec<TableAmbiguity>;
        fn table_has_ambiguity(&self) -> bool;
        fn terminal_display_names(&self) -> &[String];
        fn terminal_display_name(&self, terminal_id: u32) -> Option<&str>;
        /// Internal benchmark/debug bridge for linking already-compiled
        /// subgrammar artifacts after the legacy public composition API was
        /// removed.  Production callers should express subgrammars in GLRM.
        fn compose_compiled_subgrammars(
            self,
            children: &[(&str, &Constraint)],
            vocab: &Vocab,
        ) -> Result<Self>;
        fn compose_compiled_subgrammars_shared(
            self,
            children: &[(&str, std::sync::Arc<Constraint>)],
            vocab: &Vocab,
        ) -> Result<Self>;
        /// Measurement-only bridge: same placeholder-substitution composition
        /// but with the DynamicDirect segmented boundary backend.
        fn compose_compiled_subgrammars_dynamic(
            self,
            children: &[(&str, &Constraint)],
            vocab: &Vocab,
        ) -> Result<Self>;
        fn compose_compiled_subgrammars_dynamic_shared(
            self,
            children: &[(&str, std::sync::Arc<Constraint>)],
            vocab: &Vocab,
        ) -> Result<Self>;
        /// Measurement-only bridge: compile static boundary shards only for
        /// the listed starting-component indices. Other components retain an
        /// exact DynamicDirect shard with the static walk's candidate-token
        /// domain, allowing direct static-B vs candidate-dynamic-B timing.
        fn compose_compiled_subgrammars_hybrid(
            self,
            children: &[(&str, &Constraint)],
            vocab: &Vocab,
            static_components: &[usize],
        ) -> Result<Self>;
    }

    pub trait DynamicConstraintExt: Sized {
        fn compile_ebnf_serialized_profiled_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
            vocab_partition: bool,
        ) -> Result<(Vec<u8>, u64, u64)>;
        fn compile_lark_serialized_profiled_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
            vocab_partition: bool,
        ) -> Result<(Vec<u8>, u64, u64)>;
        fn compile_json_schema_serialized_profiled_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
            vocab_partition: bool,
        ) -> Result<(Vec<u8>, u64, u64)>;
        fn compile_glrm_serialized_profiled_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
            vocab_partition: bool,
        ) -> Result<(Vec<u8>, u64, u64)>;
        fn compile_ebnf_serialized_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
        ) -> Result<Vec<u8>>;
        fn compile_lark_serialized_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
        ) -> Result<Vec<u8>>;
        fn compile_json_schema_serialized_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
        ) -> Result<Vec<u8>>;
        fn compile_glrm_serialized_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
        ) -> Result<Vec<u8>>;
        fn load_with_vocab(bytes: &[u8], vocab: &Vocab) -> Result<Self>;
        fn max_original_token_id(&self) -> Option<u32>;
    }

    impl ConstraintExt for Constraint {
        fn compile_grammar_def_json(grammar_def_json: &str, vocab: &Vocab) -> Result<Self> {
            crate::compile_grammar_def_json(grammar_def_json, vocab)
        }

        fn dump_json_schema_grammar_glrm(schema_json: &str) -> Result<String> {
            crate::dump_json_schema_grammar_glrm(schema_json)
        }

        fn profile_json_schema_import(schema_json: &str) -> Result<()> {
            crate::import::__profile_json_schema_import(schema_json)
        }

        fn warm_ti_pool() {
            crate::warm_ti_pool();
        }

        fn bind_vocab_exact(&mut self, vocab: &Vocab) -> std::result::Result<(), String> {
            Constraint::bind_vocab_exact(self, vocab)
        }

        fn prepare_for_composition(&mut self, vocab: &Vocab) -> Result<()> {
            self.prepare_for_composition_internal(vocab)
        }

        fn prepare_for_dynamic_composition(&mut self, vocab: &Vocab) -> Result<()> {
            Constraint::bind_vocab_exact(self, vocab).map_err(Error::Compilation)?;
            self.materialize_composition_link_metadata_for_compilation()
                .map_err(Error::Compilation)
        }

        fn compose_compiled_subgrammars(
            self,
            children: &[(&str, &Constraint)],
            vocab: &Vocab,
        ) -> Result<Self> {
            use crate::compiler::constraint_compose::{
                CompiledSubgrammarInput, SegmentedBoundaryBackend,
                compose_constraints_owned_parent_segmented,
            };
            use std::collections::BTreeSet;

            let mut inputs = Vec::with_capacity(children.len());
            let mut seen = BTreeSet::new();
            for &(name, child) in children {
                let placeholder_terminal = self
                    .terminal_display_names
                    .iter()
                    .position(|candidate| candidate == name)
                    .ok_or_else(|| {
                        Error::Compilation(format!(
                            "parent has no subgrammar placeholder terminal {name:?}",
                        ))
                    })? as u32;
                if !seen.insert(placeholder_terminal) {
                    return Err(Error::Compilation(format!(
                        "parent placeholder terminal {name:?} was supplied more than once",
                    )));
                }
                inputs.push(CompiledSubgrammarInput {
                    placeholder_terminal,
                    additional_placeholder_terminals: &[],
                    constraint: child,
                });
            }
            compose_constraints_owned_parent_segmented(
                self,
                &inputs,
                vocab,
                SegmentedBoundaryBackend::StaticParserDwa,
            )
            .map(|composition| composition.constraint)
            .map_err(Error::Compilation)
        }

        fn compose_compiled_subgrammars_shared(
            self,
            children: &[(&str, std::sync::Arc<Constraint>)],
            vocab: &Vocab,
        ) -> Result<Self> {
            use crate::compiler::constraint_compose::{
                CompiledSubgrammarInput, SegmentedBoundaryBackend,
                compose_constraints_owned_parent_segmented_shared,
            };
            use std::collections::BTreeSet;
            use std::sync::Arc;

            let mut inputs = Vec::with_capacity(children.len());
            let mut shared = Vec::with_capacity(children.len());
            let mut seen = BTreeSet::new();
            for (name, child) in children {
                let placeholder_terminal = self
                    .terminal_display_names
                    .iter()
                    .position(|candidate| candidate == name)
                    .ok_or_else(|| {
                        Error::Compilation(format!(
                            "parent has no subgrammar placeholder terminal {name:?}",
                        ))
                    })? as u32;
                if !seen.insert(placeholder_terminal) {
                    return Err(Error::Compilation(format!(
                        "parent placeholder terminal {name:?} was supplied more than once",
                    )));
                }
                inputs.push(CompiledSubgrammarInput {
                    placeholder_terminal,
                    additional_placeholder_terminals: &[],
                    constraint: child.as_ref(),
                });
                shared.push(Arc::clone(child));
            }
            compose_constraints_owned_parent_segmented_shared(
                self,
                &inputs,
                &shared,
                vocab,
                SegmentedBoundaryBackend::StaticParserDwa,
            )
            .map(|composition| composition.constraint)
            .map_err(Error::Compilation)
        }

        fn compose_compiled_subgrammars_dynamic(
            self,
            children: &[(&str, &Constraint)],
            vocab: &Vocab,
        ) -> Result<Self> {
            use crate::compiler::constraint_compose::{
                CompiledSubgrammarInput, SegmentedBoundaryBackend,
                compose_constraints_owned_parent_segmented,
            };
            use std::collections::BTreeSet;

            let mut inputs = Vec::with_capacity(children.len());
            let mut seen = BTreeSet::new();
            for &(name, child) in children {
                let placeholder_terminal = self
                    .terminal_display_names
                    .iter()
                    .position(|candidate| candidate == name)
                    .ok_or_else(|| {
                        Error::Compilation(format!(
                            "parent has no subgrammar placeholder terminal {name:?}",
                        ))
                    })? as u32;
                if !seen.insert(placeholder_terminal) {
                    return Err(Error::Compilation(format!(
                        "parent placeholder terminal {name:?} was supplied more than once",
                    )));
                }
                inputs.push(CompiledSubgrammarInput {
                    placeholder_terminal,
                    additional_placeholder_terminals: &[],
                    constraint: child,
                });
            }
            compose_constraints_owned_parent_segmented(
                self,
                &inputs,
                vocab,
                SegmentedBoundaryBackend::Dynamic,
            )
            .map(|composition| composition.constraint)
            .map_err(Error::Compilation)
        }


        fn compose_compiled_subgrammars_dynamic_shared(
            self,
            children: &[(&str, std::sync::Arc<Constraint>)],
            vocab: &Vocab,
        ) -> Result<Self> {
            use crate::compiler::constraint_compose::{
                CompiledSubgrammarInput, SegmentedBoundaryBackend,
                compose_constraints_owned_parent_segmented_shared,
            };
            use std::collections::BTreeSet;
            use std::sync::Arc;

            let mut inputs = Vec::with_capacity(children.len());
            let mut shared = Vec::with_capacity(children.len());
            let mut seen = BTreeSet::new();
            for (name, child) in children {
                let placeholder_terminal = self
                    .terminal_display_names
                    .iter()
                    .position(|candidate| candidate == name)
                    .ok_or_else(|| {
                        Error::Compilation(format!(
                            "parent has no subgrammar placeholder terminal {name:?}",
                        ))
                    })? as u32;
                if !seen.insert(placeholder_terminal) {
                    return Err(Error::Compilation(format!(
                        "parent placeholder terminal {name:?} was supplied more than once",
                    )));
                }
                inputs.push(CompiledSubgrammarInput {
                    placeholder_terminal,
                    additional_placeholder_terminals: &[],
                    constraint: child.as_ref(),
                });
                shared.push(Arc::clone(child));
            }
            compose_constraints_owned_parent_segmented_shared(
                self,
                &inputs,
                &shared,
                vocab,
                SegmentedBoundaryBackend::Dynamic,
            )
            .map(|composition| composition.constraint)
            .map_err(Error::Compilation)
        }

        fn compose_compiled_subgrammars_hybrid(
            self,
            children: &[(&str, &Constraint)],
            vocab: &Vocab,
            static_components: &[usize],
        ) -> Result<Self> {
            use crate::compiler::constraint_compose::{
                CompiledSubgrammarInput, compose_constraints_owned_parent_segmented_hybrid,
            };
            use crate::ds::bitset::BitSet;
            use std::collections::BTreeSet;

            let mut inputs = Vec::with_capacity(children.len());
            let mut seen = BTreeSet::new();
            for &(name, child) in children {
                let placeholder_terminal = self
                    .terminal_display_names
                    .iter()
                    .position(|candidate| candidate == name)
                    .ok_or_else(|| {
                        Error::Compilation(format!(
                            "parent has no subgrammar placeholder terminal {name:?}",
                        ))
                    })? as u32;
                if !seen.insert(placeholder_terminal) {
                    return Err(Error::Compilation(format!(
                        "parent placeholder terminal {name:?} was supplied more than once",
                    )));
                }
                inputs.push(CompiledSubgrammarInput {
                    placeholder_terminal,
                    additional_placeholder_terminals: &[],
                    constraint: child,
                });
            }
            let mut selected = BitSet::new(children.len() + 1);
            for &component in static_components {
                if component >= children.len() + 1 {
                    return Err(Error::Compilation(format!(
                        "hybrid static component {component} out of range 0..{}",
                        children.len() + 1,
                    )));
                }
                selected.set(component);
            }
            compose_constraints_owned_parent_segmented_hybrid(
                self,
                &inputs,
                vocab,
                &selected,
            )
            .map(|composition| composition.constraint)
            .map_err(Error::Compilation)
        }

        fn clear_stale_weights() {
            crate::ds::weight::clear_stale_weights();
        }

        fn clear_weight_interners() {
            crate::ds::weight::clear_weight_interners();
        }

        fn clear_weight_op_caches() {
            crate::ds::weight::clear_weight_op_caches();
        }

        fn set_test_compat_mode(enabled: bool) {
            crate::set_test_compat_mode(enabled);
        }

        fn prepare_composition_grammar_summary(&mut self) -> Result<()> {
            if self.composition_grammar_summary.is_some() {
                return Ok(());
            }
            let augmented_start = self
                .table
                .rules
                .first()
                .map(|rule| rule.lhs)
                .ok_or_else(|| Error::Compilation("constraint table has no augmented-start rule".to_string()))?;
            let analyzed = crate::compiler::glr::analysis::AnalyzedGrammar::from_composed_rules(
                self.table.rules.clone(),
                self.table.num_terminals,
                self.terminal_display_names.clone(),
                self.table.nonterminal_display_names.clone(),
                augmented_start,
            );
            self.composition_grammar_summary = Some(
                crate::compiler::pipeline::composition_grammar_summary_from_analysis(&analyzed),
            );
            Ok(())
        }


        fn num_parser_states(&self) -> u32 {
            Constraint::num_parser_states(self)
        }

        fn num_terminals(&self) -> u32 {
            self.parser_terminal_count()
        }

        fn num_tokenizer_states(&self) -> usize {
            Constraint::num_tokenizer_states(self)
        }

        fn internal_tsid_count(&self) -> usize {
            Constraint::internal_tsid_count(self)
        }

        fn parser_dwa_num_states(&self) -> u32 {
            Constraint::parser_dwa(self).num_states()
        }

        fn parser_dwa_num_transitions(&self) -> usize {
            Constraint::parser_dwa(self).num_transitions()
        }

        fn compute_forced_minimized_tokenizer_state_count(&self) -> usize {
            Constraint::compute_forced_minimized_tokenizer_state_count(self)
        }

        fn max_original_token_id(&self) -> Option<u32> {
            Constraint::max_original_token_id(self)
        }

        fn final_internal_token_count(&self) -> usize {
            self.internal_token_count()
        }

        fn final_original_token_map(&self) -> Vec<u32> {
            self.original_token_map().to_vec()
        }

        fn table_ambiguous_actions(&self) -> Vec<TableAmbiguity> {
            Constraint::table_ambiguous_actions(self)
        }

        fn table_has_ambiguity(&self) -> bool {
            Constraint::table_has_ambiguity(self)
        }

        fn terminal_display_names(&self) -> &[String] {
            Constraint::terminal_display_names(self)
        }

        fn terminal_display_name(&self, terminal_id: u32) -> Option<&str> {
            Constraint::terminal_display_name(self, terminal_id)
        }

    }

    impl DynamicConstraintExt for DynamicConstraint {
        fn compile_ebnf_serialized_profiled_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
            vocab_partition: bool,
        ) -> Result<(Vec<u8>, u64, u64)> {
            DynamicConstraint::compile_ebnf_serialized_profiled_with_end_tokens(
                source,
                vocab,
                end_token_ids,
                vocab_partition,
            )
        }

        fn compile_lark_serialized_profiled_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
            vocab_partition: bool,
        ) -> Result<(Vec<u8>, u64, u64)> {
            DynamicConstraint::compile_lark_serialized_profiled_with_end_tokens(
                source,
                vocab,
                end_token_ids,
                vocab_partition,
            )
        }

        fn compile_json_schema_serialized_profiled_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
            vocab_partition: bool,
        ) -> Result<(Vec<u8>, u64, u64)> {
            DynamicConstraint::compile_json_schema_serialized_profiled_with_end_tokens(
                source,
                vocab,
                end_token_ids,
                vocab_partition,
            )
        }

        fn compile_glrm_serialized_profiled_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
            vocab_partition: bool,
        ) -> Result<(Vec<u8>, u64, u64)> {
            DynamicConstraint::compile_glrm_serialized_profiled_with_end_tokens(
                source,
                vocab,
                end_token_ids,
                vocab_partition,
            )
        }

        fn compile_ebnf_serialized_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
        ) -> Result<Vec<u8>> {
            DynamicConstraint::compile_ebnf_serialized_with_end_tokens(source, vocab, end_token_ids)
        }

        fn compile_lark_serialized_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
        ) -> Result<Vec<u8>> {
            DynamicConstraint::compile_lark_serialized_with_end_tokens(source, vocab, end_token_ids)
        }

        fn compile_json_schema_serialized_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
        ) -> Result<Vec<u8>> {
            DynamicConstraint::compile_json_schema_serialized_with_end_tokens(
                source,
                vocab,
                end_token_ids,
            )
        }

        fn compile_glrm_serialized_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
        ) -> Result<Vec<u8>> {
            DynamicConstraint::compile_glrm_serialized_with_end_tokens(source, vocab, end_token_ids)
        }

        fn load_with_vocab(bytes: &[u8], vocab: &Vocab) -> Result<Self> {
            DynamicConstraint::load_with_vocab(bytes, vocab)
        }

        fn max_original_token_id(&self) -> Option<u32> {
            DynamicConstraint::max_original_token_id(self)
        }
    }

    pub trait ConstraintStateExt {
        fn commit_token_timed_ns(&mut self, token_id: u32) -> std::result::Result<u64, String>;
        fn commit_token_profiled(
            &mut self,
            token_id: u32,
        ) -> std::result::Result<CommitProfile, String>;
        fn commit_token_per_advance(
            &mut self,
            token_id: u32,
        ) -> std::result::Result<
            (Vec<PerAdvanceEntry>, Vec<(u32, Vec<Vec<u32>>)>, CommitProfile),
            String,
        >;
        fn debug_parser_stacks(&self) -> Vec<(u32, Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>)>;
        fn fill_mask_profiled(&self, buf: &mut [u32]) -> MaskProfile;
        fn fill_mask_timed_ns(&self, buf: &mut [u32]) -> u64;
        /// Measurement-only: exact dynamic-reference mask (no parser DWA).
        fn fill_mask_dynamic_vec(&self) -> Vec<u32>;
        fn has_parser_ambiguity(&self) -> bool;
        fn parser_path_count(&self, limit: usize) -> usize;
        fn parser_root_count(&self) -> usize;
    }

    impl ConstraintStateExt for ConstraintState<'_> {
        fn commit_token_timed_ns(&mut self, token_id: u32) -> std::result::Result<u64, String> {
            ConstraintState::commit_token_timed_ns(self, token_id)
        }

        fn commit_token_profiled(
            &mut self,
            token_id: u32,
        ) -> std::result::Result<CommitProfile, String> {
            ConstraintState::commit_token_profiled(self, token_id)
        }

        fn commit_token_per_advance(
            &mut self,
            token_id: u32,
        ) -> std::result::Result<
            (Vec<PerAdvanceEntry>, Vec<(u32, Vec<Vec<u32>>)>, CommitProfile),
            String,
        > {
            ConstraintState::commit_token_per_advance(self, token_id)
        }

        fn debug_parser_stacks(&self) -> Vec<(u32, Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>)> {
            ConstraintState::debug_parser_stacks(self)
        }

        fn fill_mask_profiled(&self, buf: &mut [u32]) -> MaskProfile {
            ConstraintState::fill_mask_profiled(self, buf)
        }

        fn fill_mask_timed_ns(&self, buf: &mut [u32]) -> u64 {
            ConstraintState::fill_mask_timed_ns(self, buf)
        }

        fn fill_mask_dynamic_vec(&self) -> Vec<u32> {
            let mut buf = vec![0u32; self.constraint.mask_len()];
            ConstraintState::fill_mask_dynamic(self, &mut buf);
            buf
        }

        fn has_parser_ambiguity(&self) -> bool {
            ConstraintState::has_parser_ambiguity(self)
        }

        fn parser_path_count(&self, limit: usize) -> usize {
            ConstraintState::parser_path_count(self, limit)
        }

        fn parser_root_count(&self) -> usize {
            ConstraintState::parser_root_count(self)
        }

    }

    pub trait VocabExt {
        fn prepare_for_compile(&self);
        fn prepare_for_dynamic_compile(&self);
    }

    impl VocabExt for Vocab {
        fn prepare_for_compile(&self) {
            crate::prepare_vocab_for_compile(self);
        }

        fn prepare_for_dynamic_compile(&self) {
            crate::prepare_vocab_for_dynamic_compile(self);
        }
    }
}
