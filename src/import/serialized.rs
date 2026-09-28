//! Direct dynamic-artifact compilation; preparation and timing boundaries stay explicit.

use super::ast;
use super::dynamic_compile::dynamic_named_alternatives;
use super::lowering::NamedGrammarParser;
use super::lowering::NamedGrammarTransform;
use super::lowering::parse_ebnf_to_named;
use super::lowering::parse_glrm_to_named;
use super::lowering::parse_json_schema_to_named_dynamic;
use super::lowering::parse_lark_to_named;
use super::lowering::prepare_json_schema_named;
use super::lowering::with_large_import_stack;
use crate::automata::lexer::Lexer;
use crate::compiler::compile::compile_profile_enabled;
use crate::compiler::compile::compile_top_profile_enabled;
use crate::compiler::glr::table::GlrTableConstruction;
use crate::compiler::pipeline::compile_dynamic_owned_unfinalized_with_table_construction;
use crate::compiler::pipeline::compile_dynamic_owned_with_vocab_partition_unfinalized_with_table_construction;
use crate::runtime::dynamic::DynamicConstraint;
pub(super) fn compile_dynamic_serialized_from_source_profiled(
    source: &str,
    vocab: &crate::Vocab,
    default_table_construction: GlrTableConstruction,
    parse: NamedGrammarParser,
    transform: Option<NamedGrammarTransform>,
    end_token_ids: &[u32],
    vocab_partition: bool,
) -> crate::Result<(Vec<u8>, u64, u64)> {
    let wall_started = std::time::Instant::now();
    let profile = compile_profile_enabled() || compile_top_profile_enabled();
    let total_started = profile.then(std::time::Instant::now);
    let import_started = profile.then(std::time::Instant::now);
    let alternatives = dynamic_named_alternatives(source, parse, transform, end_token_ids)?;
    let import_ms = import_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let compile_started = profile.then(std::time::Instant::now);
    let mut compiled = Vec::with_capacity(alternatives.len());
    let mut lower_ms = 0.0;
    for alternative in alternatives {
        let lower_started = profile.then(std::time::Instant::now);
        let grammar = ast::lower(&alternative)?;
        lower_ms += lower_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let mut constraint = if vocab_partition {
            compile_dynamic_owned_with_vocab_partition_unfinalized_with_table_construction(
                grammar,
                vocab,
                default_table_construction,
            )?
        } else {
            compile_dynamic_owned_unfinalized_with_table_construction(
                grammar,
                vocab,
                default_table_construction,
            )?
        };
        let direct_residual_master_provers_enabled =
            std::env::var("GLRMASK_DYNAMIC_DIRECT_RESIDUAL_MASTER_PROVERS")
                .ok()
                .map(|value| {
                    !matches!(
                        value.trim().to_ascii_lowercase().as_str(),
                        "" | "0" | "false" | "no" | "off"
                    )
                })
                .unwrap_or(true)
                || std::env::var("GLRMASK_EXPERIMENT_DIRECT_RESIDUAL_MASTER_PROVERS")
                    .ok()
                    .is_some_and(|value| {
                        !matches!(
                            value.trim().to_ascii_lowercase().as_str(),
                            "" | "0" | "false" | "no" | "off"
                        )
                    });
        let mut terminal_observation_prepared_in_parallel = false;
        if direct_residual_master_provers_enabled
            && constraint.inner.tokenizer.terminal_residual_coordinates().is_some()
        {
            use crate::compiler::stages::id_map_and_terminal_dwa::classify::VocabPartitionDfa;
            let safe_plus =
                VocabPartitionDfa::compile_utf8_regex("llg-safe+", r#"[^"\\\x00-\x1F\x7F]+"#)
                    .expect("safe-string slice regex must compile");
            let whitespace =
                VocabPartitionDfa::compile_utf8_regex("llg-whitespace", r"[\x20\x0A\x0D\x09]+")
                    .expect("whitespace slice regex must compile");
            let safe_slice_bytes =
                crate::compiler::constraint_possible_matches::llg_safe_slice_token_bytes_for_vocab(
                    vocab,
                );
            let max_safe_chars =
                crate::compiler::constraint_possible_matches::llg_master_max_safe_chars_for_vocab(
                    vocab,
                );
            let prepare_observation_classes =
                !constraint.inner.dynamic_mask_vocab.has_terminal_observation_classes()
                    && std::env::var("GLRMASK_DYNAMIC_TERMINAL_OBSERVATION_CLASSES")
                        .ok()
                        .is_none_or(|value| !matches!(value.trim(), "0" | "false" | "no" | "off"));
            let source_state_count = constraint.inner.tokenizer.num_states() as usize;
            let mut dynamic_mask_vocab = std::mem::take(&mut constraint.inner.dynamic_mask_vocab);
            let ((result, proof_elapsed_ms), observation_classes) = rayon::join(
                || {
                    let started = std::time::Instant::now();
                    let result = dynamic_mask_vocab
                        .prepare_master_provers_from_residual_coordinates(
                            &constraint.inner.tokenizer,
                            source_state_count,
                            &safe_plus,
                            &whitespace,
                            safe_slice_bytes,
                            max_safe_chars,
                        );
                    (result, started.elapsed().as_secs_f64() * 1e3)
                },
                || {
                    prepare_observation_classes
                        .then(|| constraint.inner.build_dynamic_terminal_observation_classes())
                },
            );
            if let Some(classes) = observation_classes {
                dynamic_mask_vocab.set_terminal_observation_classes(classes);
                terminal_observation_prepared_in_parallel = true;
            }
            constraint.inner.dynamic_mask_vocab = dynamic_mask_vocab;
            if profile
                || std::env::var_os("GLRMASK_PROFILE_DIRECT_RESIDUAL_MASTER_PROVERS").is_some()
            {
                if let Some((entries, product_pairs, edges)) = result {
                    eprintln!(
                        "[glrmask/profile][direct_residual_master_provers] entries={} product_pairs={} edges={} max_safe_chars={} elapsed_ms={:.3}",
                        entries, product_pairs, edges, max_safe_chars, proof_elapsed_ms,
                    );
                } else {
                    eprintln!(
                        "[glrmask/profile][direct_residual_master_provers] unavailable elapsed_ms={:.3}",
                        proof_elapsed_ms,
                    );
                }
            }
        }
        if !terminal_observation_prepared_in_parallel {
            constraint.inner.prepare_dynamic_terminal_observation_classes_for_artifact();
        }
        constraint.inner.prepare_dynamic_virtual_residual_mask_projections_for_artifact();
        compiled.push(constraint);
    }
    let compile_ms =
        compile_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let compile_ns = wall_started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    let serialize_wall_started = std::time::Instant::now();
    let serialize_started = profile.then(std::time::Instant::now);
    let bytes = DynamicConstraint::from_alternatives(compiled).into_saved();
    let serialize_ns = serialize_wall_started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    let serialize_ms =
        serialize_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    if let Some(total_started) = total_started {
        eprintln!(
            "[glrmask/profile][dynamic_serialized_source] vocab_partition={} import_ms={:.3} lower_ms={:.3} compile_with_lower_ms={:.3} serialize_ms={:.3} bytes={} total_ms={:.3}",
            vocab_partition,
            import_ms,
            lower_ms,
            compile_ms,
            serialize_ms,
            bytes.len(),
            total_started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok((bytes, compile_ns, serialize_ns))
}

pub(super) fn compile_dynamic_serialized_from_source(
    source: &str,
    vocab: &crate::Vocab,
    default_table_construction: GlrTableConstruction,
    parse: NamedGrammarParser,
    transform: Option<NamedGrammarTransform>,
    end_token_ids: &[u32],
) -> crate::Result<Vec<u8>> {
    compile_dynamic_serialized_from_source_profiled(
        source,
        vocab,
        default_table_construction,
        parse,
        transform,
        end_token_ids,
        false,
    )
    .map(|(bytes, _, _)| bytes)
}

impl DynamicConstraint {
    #[doc(hidden)]
    pub(crate) fn compile_ebnf_serialized_profiled_with_end_tokens(
        ebnf: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
        vocab_partition: bool,
    ) -> crate::Result<(Vec<u8>, u64, u64)> {
        with_large_import_stack(ebnf.len(), || {
            compile_dynamic_serialized_from_source_profiled(
                ebnf,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_ebnf_to_named,
                None,
                end_token_ids,
                vocab_partition,
            )
        })
    }

    #[doc(hidden)]
    pub(crate) fn compile_lark_serialized_profiled_with_end_tokens(
        lark: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
        vocab_partition: bool,
    ) -> crate::Result<(Vec<u8>, u64, u64)> {
        with_large_import_stack(lark.len(), || {
            compile_dynamic_serialized_from_source_profiled(
                lark,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_lark_to_named,
                None,
                end_token_ids,
                vocab_partition,
            )
        })
    }

    #[doc(hidden)]
    pub(crate) fn compile_json_schema_serialized_profiled_with_end_tokens(
        schema: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
        vocab_partition: bool,
    ) -> crate::Result<(Vec<u8>, u64, u64)> {
        with_large_import_stack(schema.len(), || {
            compile_dynamic_serialized_from_source_profiled(
                schema,
                vocab,
                GlrTableConstruction::Lalr,
                parse_json_schema_to_named_dynamic,
                Some(prepare_json_schema_named),
                end_token_ids,
                vocab_partition,
            )
        })
    }

    #[doc(hidden)]
    pub(crate) fn compile_glrm_serialized_profiled_with_end_tokens(
        glrm: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
        vocab_partition: bool,
    ) -> crate::Result<(Vec<u8>, u64, u64)> {
        with_large_import_stack(glrm.len(), || {
            compile_dynamic_serialized_from_source_profiled(
                glrm,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_glrm_to_named,
                None,
                end_token_ids,
                vocab_partition,
            )
        })
    }

    /// Compile EBNF directly to a serialized dynamic artifact without building
    /// runtime-only caches in the producing process.
    #[doc(hidden)]
    pub(crate) fn compile_ebnf_serialized_with_end_tokens(
        ebnf: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Vec<u8>> {
        with_large_import_stack(ebnf.len(), || {
            compile_dynamic_serialized_from_source(
                ebnf,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_ebnf_to_named,
                None,
                end_token_ids,
            )
        })
    }

    /// Compile Lark directly to a serialized dynamic artifact.
    #[doc(hidden)]
    pub(crate) fn compile_lark_serialized_with_end_tokens(
        lark: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Vec<u8>> {
        with_large_import_stack(lark.len(), || {
            compile_dynamic_serialized_from_source(
                lark,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_lark_to_named,
                None,
                end_token_ids,
            )
        })
    }

    /// Compile JSON Schema directly to a serialized dynamic artifact.
    #[doc(hidden)]
    pub(crate) fn compile_json_schema_serialized_with_end_tokens(
        schema: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Vec<u8>> {
        with_large_import_stack(schema.len(), || {
            compile_dynamic_serialized_from_source(
                schema,
                vocab,
                GlrTableConstruction::Lalr,
                parse_json_schema_to_named_dynamic,
                Some(prepare_json_schema_named),
                end_token_ids,
            )
        })
    }

    /// Compile GLRM directly to a serialized dynamic artifact.
    #[doc(hidden)]
    pub(crate) fn compile_glrm_serialized_with_end_tokens(
        glrm: &str,
        vocab: &crate::Vocab,
        end_token_ids: &[u32],
    ) -> crate::Result<Vec<u8>> {
        with_large_import_stack(glrm.len(), || {
            compile_dynamic_serialized_from_source(
                glrm,
                vocab,
                GlrTableConstruction::ExperimentalCoreMerged,
                parse_glrm_to_named,
                None,
                end_token_ids,
            )
        })
    }
}
