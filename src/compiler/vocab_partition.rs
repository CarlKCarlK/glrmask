use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::api::VocabPartitionStrategy;
use crate::Vocab;
use crate::automata::lexer::Lexer;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::grammar::transforms::prepare_grammar_for_vocab_partition;
use crate::compiler::pipeline::{
    build_vocab_partition_compile_context, compute_disallowed_follows, run_with_compile_thread_pool,
};
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::compiler::stages::id_map_and_terminal_dwa::{
    build_global_max_length_state_map_with_initial,
    build_vocab_partition_from_static_id_maps,
    build_vocab_equivalence_partition_with_precomputed_global_max_length,
    l1,
    types::TerminalColoring,
};
use crate::grammar::flat::{GrammarDef, Symbol, Terminal};

const SINGLETON_LITERAL_PARTITION_MAX_BYTES: usize = 64;

/// Prove that the prepared CFG denotes exactly one literal byte string.
///
/// This intentionally accepts only the simplest proof shape: every reachable
/// nonterminal has exactly one production and every terminal on that expansion
/// is literal. Cycles, alternatives, special tokens, patterns, and expressions
/// all fall back to the general vocabulary-partition machinery.
fn singleton_literal_language_bytes(grammar: &GrammarDef) -> Option<Vec<u8>> {
    if grammar.ignore_terminal.is_some() || grammar.requires_global_terminal_observation {
        return None;
    }

    let max_nonterminal = grammar
        .rules
        .iter()
        .flat_map(|rule| {
            std::iter::once(rule.lhs).chain(rule.rhs.iter().filter_map(|symbol| match symbol {
                Symbol::Nonterminal(nonterminal) => Some(*nonterminal),
                Symbol::Terminal(_) => None,
            }))
        })
        .max()
        .unwrap_or(grammar.start)
        .max(grammar.start) as usize;
    let mut rules_by_lhs = vec![Vec::<usize>::new(); max_nonterminal + 1];
    for (index, rule) in grammar.rules.iter().enumerate() {
        rules_by_lhs[rule.lhs as usize].push(index);
    }

    fn expand(
        nonterminal: usize,
        grammar: &GrammarDef,
        rules_by_lhs: &[Vec<usize>],
        state: &mut [u8],
        memo: &mut [Option<Vec<u8>>],
    ) -> Option<Vec<u8>> {
        match state.get(nonterminal).copied()? {
            1 => return None, // recursion: singleton proof intentionally declines
            2 => return memo[nonterminal].clone(),
            _ => {}
        }
        let rules = rules_by_lhs.get(nonterminal)?;
        if rules.len() != 1 {
            return None;
        }
        state[nonterminal] = 1;
        let rule = &grammar.rules[rules[0]];
        let mut bytes = Vec::new();
        for symbol in &rule.rhs {
            match *symbol {
                Symbol::Terminal(terminal) => match grammar.terminals.get(terminal as usize)? {
                    Terminal::Literal { bytes: literal, .. } => {
                        if bytes.len().saturating_add(literal.len())
                            > SINGLETON_LITERAL_PARTITION_MAX_BYTES
                        {
                            return None;
                        }
                        bytes.extend_from_slice(literal);
                    }
                    Terminal::Pattern { .. }
                    | Terminal::Expr { .. }
                    | Terminal::SpecialToken { .. } => return None,
                },
                Symbol::Nonterminal(child) => {
                    let child = expand(child as usize, grammar, rules_by_lhs, state, memo)?;
                    if bytes.len().saturating_add(child.len())
                        > SINGLETON_LITERAL_PARTITION_MAX_BYTES
                    {
                        return None;
                    }
                    bytes.extend_from_slice(&child);
                }
            }
        }
        state[nonterminal] = 2;
        memo[nonterminal] = Some(bytes.clone());
        Some(bytes)
    }

    let mut state = vec![0u8; rules_by_lhs.len()];
    let mut memo = vec![None; rules_by_lhs.len()];
    expand(
        grammar.start as usize,
        grammar,
        &rules_by_lhs,
        &mut state,
        &mut memo,
    )
}

/// Exact whole-token equivalence for a singleton byte language.
///
/// At any byte position in the one accepted string, a model token either equals
/// a substring beginning there and advances by its byte length, or it is dead.
/// Therefore all non-substring token byte strings have the same transition
/// vector, while each distinct substring byte string has one exact vector. This
/// is the complete token effect needed by O2; it does not construct or use a
/// terminal DWA at runtime.
fn singleton_literal_vocab_partition(vocab: &Vocab, language: &[u8]) -> ManyToOneIdMap {
    let mut live_bytes = Vec::<Vec<u8>>::with_capacity(
        language.len().saturating_mul(language.len().saturating_add(1)) / 2 + 1,
    );
    live_bytes.push(Vec::new());
    for start in 0..language.len() {
        for end in start + 1..=language.len() {
            live_bytes.push(language[start..end].to_vec());
        }
    }
    live_bytes.sort_unstable();
    live_bytes.dedup();

    // Normal O2 construction prepares this byte-sorted identity order once per
    // model vocabulary. Reuse it to resolve only the O(n²) possible live byte
    // strings instead of touching every token's bytes for every singleton
    // grammar. Standalone/cold VocabPartition callers fall back to the direct
    // scan below rather than forcing this comparatively expensive index.
    if let Some(live_groups) = l1::cached_l1_exact_token_groups(vocab, &live_bytes) {
        let live_count = live_groups.iter().map(Vec::len).sum::<usize>();
        let mut live_ids = live_groups
            .iter()
            .flat_map(|group| group.iter().copied())
            .collect::<Vec<_>>();
        live_ids.sort_unstable();
        live_ids.dedup();
        debug_assert_eq!(live_ids.len(), live_count);

        let mut internal_to_originals = Vec::<Vec<u32>>::with_capacity(live_groups.len() + 1);
        let mut representative_original_ids = Vec::<u32>::with_capacity(live_groups.len() + 1);
        if live_count != vocab.len() {
            // Reserve the complete vocabulary length: the runtime quotient can
            // then reuse this overwhelmingly-large first class as its flattened
            // alias buffer and append the few live IDs without reallocating it.
            let mut dead = Vec::<u32>::with_capacity(vocab.len());
            let mut live_index = 0usize;
            let dense_ids = vocab.len() == vocab.max_token_id() as usize + 1;
            if dense_ids {
                for token_id in 0..=vocab.max_token_id() {
                    if live_ids.get(live_index).copied() == Some(token_id) {
                        live_index += 1;
                    } else {
                        dead.push(token_id);
                    }
                }
            } else {
                for &token_id in vocab.entries_map().keys() {
                    if live_ids.get(live_index).copied() == Some(token_id) {
                        live_index += 1;
                    } else {
                        dead.push(token_id);
                    }
                }
            }
            debug_assert_eq!(live_index, live_ids.len());
            if let Some(&representative) = dead.first() {
                representative_original_ids.push(representative);
                internal_to_originals.push(dead);
            }
        }
        for group in live_groups {
            if let Some(&representative) = group.first() {
                representative_original_ids.push(representative);
                internal_to_originals.push(group);
            }
        }
        return ManyToOneIdMap {
            original_to_internal: Vec::new(),
            internal_to_originals,
            representative_original_ids,
        };
    }

    // Cold fallback: avoid hashing long tokens at all. Only byte strings no
    // longer than the singleton language can possibly be live.
    let mut live_classes = live_bytes
        .into_iter()
        .map(|bytes| (bytes, None::<u32>))
        .collect::<FxHashMap<_, _>>();
    let mut internal_to_originals = Vec::<Vec<u32>>::new();
    let mut representative_original_ids = Vec::<u32>::new();
    let mut dead_class = None::<u32>;
    for (&token_id, token_bytes) in vocab.entries_map() {
        let live_slot = (token_bytes.len() <= language.len())
            .then(|| live_classes.get_mut(token_bytes.as_slice()))
            .flatten();
        let class = if let Some(slot) = live_slot {
            *slot.get_or_insert_with(|| {
                let class = internal_to_originals.len() as u32;
                internal_to_originals.push(Vec::new());
                representative_original_ids.push(token_id);
                class
            })
        } else {
            *dead_class.get_or_insert_with(|| {
                let class = internal_to_originals.len() as u32;
                internal_to_originals.push(Vec::new());
                representative_original_ids.push(token_id);
                class
            })
        };
        internal_to_originals[class as usize].push(token_id);
    }

    ManyToOneIdMap {
        original_to_internal: Vec::new(),
        internal_to_originals,
        representative_original_ids,
    }
}

/// Compile only the grammar-dependent vocabulary equivalence partition.
///
/// This deliberately stops before terminal-DWA, possible-matches, parser-DWA,
/// and runtime-artifact construction. The returned map may be finer than the
/// final Static token quotient, but never coarser than the exact relations used
/// by this fast path.
pub(crate) fn compile_vocab_partition_owned(
    grammar: GrammarDef,
    vocab: &Vocab,
    strategy: VocabPartitionStrategy,
) -> ManyToOneIdMap {
    let profile = crate::compiler::compile::compile_profile_enabled();
    let prepare_started = std::time::Instant::now();
    let prepared_grammar = prepare_grammar_for_vocab_partition(grammar);
    let grammar_prepare_ms = prepare_started.elapsed().as_secs_f64() * 1000.0;

    // A proved singleton literal language has a much smaller exact token-effect
    // problem than the generic tokenizer/L1/L2P pipeline. Classify whole model
    // tokens directly by their transitions over positions in that one byte
    // string. Keep a same-binary kill switch for validation.
    if std::env::var_os("GLRMASK_DISABLE_SINGLETON_LITERAL_VOCAB_PARTITION").is_none()
        && let Some(language) = singleton_literal_language_bytes(&prepared_grammar)
    {
        let started = std::time::Instant::now();
        let result = singleton_literal_vocab_partition(vocab, &language);
        if profile {
            eprintln!(
                "[glrmask/profile][vocab_partition_singleton_literal] bytes={} classes={} grammar_prepare_ms={:.3} partition_ms={:.3}",
                language.len(),
                result.num_internal_ids(),
                grammar_prepare_ms,
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return result;
    }

    // Pure vocabulary artifacts are reusable across every grammar using this
    // Vocab. Populate them once so grammar-dependent latency does not repeatedly
    // pay radix/reverse-trie construction. The singleton-literal proof above
    // does not consume these artifacts, so do not warm them unnecessarily.
    crate::compiler::stages::id_map_and_terminal_dwa::prepare_vocab_for_terminal_dwa(vocab);
    run_with_compile_thread_pool(|| {
        // Match Static's compile DAG: grammar analysis is independent of the
        // entire tokenizer -> flat-transition -> max-length chain. Running
        // those lanes concurrently avoids serializing grammar work in front of
        // an otherwise lexer-only API.
        let (
            (
                tokenizer,
                initial_state_map,
                partition_local_synthesis_plan,
                direct_mask_tokenizer,
                flat_trans,
                global_max_length_state_map,
                context_ms,
                flat_ms,
                max_length_ms,
            ),
            (analyzed_grammar, disallowed_follows, analysis_ms),
        ) = crate::compiler::macro_join(
            "vocab_partition_frontend",
            || {
                let context_started = std::time::Instant::now();
                let (
                    tokenizer,
                    initial_state_map,
                    partition_local_synthesis_plan,
                    direct_mask_tokenizer,
                ) =
                    build_vocab_partition_compile_context(&prepared_grammar, vocab);
                if std::env::var_os("GLRMASK_PROFILE_VOCAB_PROJECTED_QUOTIENTS").is_some() {
                    let projected_started = std::time::Instant::now();
                    let quotients = tokenizer
                        .build_shared_component_terminal_projected_quotients(256);
                    eprintln!(
                        "[glrmask/profile][vocab_projected_quotients] retained={} elapsed_ms={:.3}",
                        quotients.len(),
                        projected_started.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                let context_ms = context_started.elapsed().as_secs_f64() * 1000.0;

                let flat_started = std::time::Instant::now();
                let flat_trans: Arc<[u32]> =
                    Arc::from(l1::build_flat_transition_table(&tokenizer));
                let flat_ms = flat_started.elapsed().as_secs_f64() * 1000.0;

                let max_length_started = std::time::Instant::now();
                let global_max_length_state_map = build_global_max_length_state_map_with_initial(
                    &tokenizer,
                    vocab,
                    &flat_trans,
                    initial_state_map.as_ref(),
                );
                let max_length_ms = max_length_started.elapsed().as_secs_f64() * 1000.0;
                (
                    tokenizer,
                    initial_state_map,
                    partition_local_synthesis_plan,
                    direct_mask_tokenizer,
                    flat_trans,
                    global_max_length_state_map,
                    context_ms,
                    flat_ms,
                    max_length_ms,
                )
            },
            || {
                let analysis_started = std::time::Instant::now();
                let analyzed_grammar = AnalyzedGrammar::from_grammar_def(&prepared_grammar);
                let disallowed_follows = compute_disallowed_follows(&analyzed_grammar);
                let analysis_ms = analysis_started.elapsed().as_secs_f64() * 1000.0;
                (analyzed_grammar, disallowed_follows, analysis_ms)
            },
        );

        if std::env::var_os("GLRMASK_PROFILE_TERMINAL_OBSERVATION_PARTITIONS").is_some() {
            let started = std::time::Instant::now();
            let mut total_configs = 0usize;
            let mut total_classes = 0usize;
            for terminal in 0..tokenizer.num_terminals() {
                let terminal_started = std::time::Instant::now();
                match tokenizer.exact_terminal_observation_partition(
                    terminal,
                    1_000_000,
                    100_000_000,
                ) {
                    Some((classes, configs, rounds)) => {
                        let class_count = classes.iter().copied().max().unwrap_or(0) as usize;
                        total_configs += configs;
                        total_classes += class_count;
                        eprintln!(
                            "[glrmask/profile][terminal_observation_partition] terminal={} configs={} classes={} rounds={} ms={:.3}",
                            terminal,
                            configs,
                            class_count,
                            rounds,
                            terminal_started.elapsed().as_secs_f64() * 1000.0,
                        );
                    }
                    None => eprintln!(
                        "[glrmask/profile][terminal_observation_partition] terminal={} declined=true ms={:.3}",
                        terminal,
                        terminal_started.elapsed().as_secs_f64() * 1000.0,
                    ),
                }
            }
            eprintln!(
                "[glrmask/profile][terminal_observation_partitions] terminals={} total_configs={} summed_classes={} total_ms={:.3}",
                tokenizer.num_terminals(),
                total_configs,
                total_classes,
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }

        let partition_started = std::time::Instant::now();
        // The dedicated vocab-only relation avoids expensive L2 proofs and
        // terminal-DWA artifacts, which is a substantial win once the lexer
        // has enough state/terminal interaction. On small lexers its own
        // classification/setup cost dominates, and the existing id-map-only
        // Static path is both faster and materially closer to Static's final
        // quotient. Select by the size of the observable lexer topology rather
        // than by schema/source kind.
        const DEDICATED_TOPOLOGY_MIN: usize = 100_000;
        let topology = (tokenizer.num_states() as usize)
            .saturating_mul(analyzed_grammar.num_terminals as usize);
        let use_dedicated = match strategy {
            VocabPartitionStrategy::Automatic => std::env::var("GLRMASK_VOCAB_PARTITION_DEDICATED")
                .ok()
                .map(|value| {
                    let value = value.trim();
                    !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
                })
                .unwrap_or(direct_mask_tokenizer || topology >= DEDICATED_TOPOLOGY_MIN),
            VocabPartitionStrategy::Compact => false,
            VocabPartitionStrategy::Dedicated => true,
        };
        let result = if use_dedicated {
            build_vocab_equivalence_partition_with_precomputed_global_max_length(
                &tokenizer,
                vocab,
                prepared_grammar.ignore_terminal,
                &analyzed_grammar,
                &disallowed_follows,
                Arc::clone(&flat_trans),
                &global_max_length_state_map,
                partition_local_synthesis_plan.as_deref(),
            )
        } else {
            let terminal_coloring =
                TerminalColoring::identity(analyzed_grammar.num_terminals as usize);
            build_vocab_partition_from_static_id_maps(
                &tokenizer,
                vocab,
                &terminal_coloring,
                false,
                prepared_grammar.ignore_terminal,
                &analyzed_grammar,
                &disallowed_follows,
                None,
                Arc::clone(&flat_trans),
                &global_max_length_state_map,
                None,
                None,
                partition_local_synthesis_plan.as_deref(),
                None,
            )
        };
        let partition_ms = partition_started.elapsed().as_secs_f64() * 1000.0;
        if profile {
            eprintln!(
                "[glrmask/profile][vocab_partition_stages] grammar_prepare_ms={grammar_prepare_ms:.3} context_ms={context_ms:.3} analysis_ms={analysis_ms:.3} flat_ms={flat_ms:.3} max_length_ms={max_length_ms:.3} partition_ms={partition_ms:.3} tokenizer_states={} terminals={} topology={} direct_mask={} dedicated={}",
                tokenizer.num_states(),
                prepared_grammar.terminals.len(),
                topology,
                direct_mask_tokenizer,
                use_dedicated,
            );
        }
        result
    })
}
