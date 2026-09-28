//! Unstable repository tooling and benchmark bridge.
//! Not part of the normal public dependency surface.

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
        crate::compiler::composition::boundary::candidates::boundary_candidate_ids(component,vocab).0
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
            use crate::compiler::composition::{
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
            use crate::compiler::composition::{
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
            use crate::compiler::composition::{
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
            use crate::compiler::composition::{
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
            use crate::compiler::composition::{
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
            self.table.num_terminals
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
