//! Internal vocabulary equivalence analysis; not a default public API.

use std::time::Instant;
use crate::{Error, Result, Vocab};
use super::{Grammar, GrammarSource};

/// Strategy used to build a [`VocabPartition`].
///
/// All strategies are conservative: they may keep tokens separate when a more
/// expensive proof could merge them, but they never intentionally merge tokens
/// whose grammar-visible behavior differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VocabPartitionStrategy {
    /// Choose automatically from the observable lexer topology.
    #[default]
    Automatic,
    /// Prefer the Static-derived quotient, which is usually more compact but may
    /// spend more time proving multi-token equivalences on small lexers.
    Compact,
    /// Use the dedicated vocabulary-only relation. This often compiles faster on
    /// small lexers but may return a finer partition with more classes.
    Dedicated,
}

/// A conservative grammar-specific equivalence partition of model vocabulary tokens.
///
/// Tokens in the same class have been proved interchangeable by the fast
/// vocabulary analysis. The partition may be finer than the token partition of
/// a fully compiled [`Constraint`](crate::Constraint): expensive proof steps may
/// deliberately be skipped, in which case affected tokens remain separated.
#[derive(Debug, Clone)]
pub struct VocabPartition {
    pub(super) original_to_class: Vec<u32>,
    pub(super) classes: Vec<Vec<u32>>,
    pub(super) representatives: Vec<u32>,
    pub(super) class_output_masks: Vec<Vec<(u32, u32)>>,
}

impl VocabPartition {
    /// Analyze `grammar` for `vocab` without constructing terminal/parser DWAs.
    pub fn compile(grammar: Grammar<'_>, vocab: &Vocab) -> Result<Self> {
        Self::compile_with_strategy(grammar, vocab, VocabPartitionStrategy::Automatic)
    }

    /// Analyze `grammar` for `vocab` using an explicit partition strategy.
    pub fn compile_with_strategy(
        grammar: Grammar<'_>,
        vocab: &Vocab,
        strategy: VocabPartitionStrategy,
    ) -> Result<Self> {
        let profile = crate::compiler::pipeline::compile_top_profile_enabled();
        let total_started = profile.then(Instant::now);
        if !grammar.bindings.is_empty() {
            return Err(Error::Compilation(
                "vocabulary partition analysis does not yet support bound grammar values".to_owned(),
            ));
        }
        let source_kind = match grammar.source {
            GrammarSource::Ebnf(_) => "ebnf",
            GrammarSource::Lark(_) => "lark",
            GrammarSource::JsonSchema(_) => "json_schema",
            GrammarSource::Glrm(_) => "glrm",
        };
        let source = match grammar.source {
            GrammarSource::Ebnf(source)
            | GrammarSource::Lark(source)
            | GrammarSource::JsonSchema(source)
            | GrammarSource::Glrm(source) => source,
        };
        let lower_started = profile.then(Instant::now);
        let grammar_def = crate::import::lower_source_for_vocab_partition(source_kind, source)?;
        let lower_ms = lower_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let compile_started = profile.then(Instant::now);
        let map = crate::error::catch_internal_invariant(|| {
            crate::compiler::vocab_partition::compile_vocab_partition_owned(grammar_def, vocab, strategy)
        })?;
        if profile {
            eprintln!(
                "[glrmask/profile][vocab_partition_compile] source={} lower_ms={:.3} compile_ms={:.3} total_ms={:.3}",
                source_kind,
                lower_ms,
                compile_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                total_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            );
        }
        let classes = map.internal_to_originals;
        // Internal O2/vocab-only compilation is allowed to carry only grouped
        // class membership and deliberately omit the model-vocab-sized dense
        // original-token -> class vector. `VocabPartition` is a public API,
        // however, and explicitly exposes that dense coordinate through
        // `class_of()` / `original_to_class()`. Materialize it once at this
        // boundary rather than forcing every internal fast path to carry it.
        let original_to_class = if map.original_to_internal.is_empty() && !classes.is_empty() {
            let mut dense = vec![u32::MAX; vocab.max_token_id() as usize + 1];
            for (class_id, class) in classes.iter().enumerate() {
                for &token_id in class {
                    dense[token_id as usize] = class_id as u32;
                }
            }
            dense
        } else {
            map.original_to_internal
        };
        let class_output_masks = classes
            .iter()
            .map(|class| {
                let mut words = Vec::<(u32, u32)>::new();
                for &token_id in class {
                    let word = token_id / 32;
                    let bit = 1u32 << (token_id % 32);
                    if let Some((last_word, last_bits)) = words.last_mut()
                        && *last_word == word
                    {
                        *last_bits |= bit;
                    } else {
                        words.push((word, bit));
                    }
                }
                words
            })
            .collect();
        Ok(Self {
            original_to_class,
            classes,
            representatives: map.representative_original_ids,
            class_output_masks,
        })
    }

    /// Number of equivalence classes.
    pub fn num_classes(&self) -> usize { self.classes.len() }

    /// Class containing `token_id`, or `None` if that token ID was not in the vocabulary.
    pub fn class_of(&self, token_id: u32) -> Option<u32> {
        self.original_to_class
            .get(token_id as usize)
            .copied()
            .filter(|&class| class != u32::MAX)
    }

    /// Original model token IDs grouped by equivalence class.
    pub fn classes(&self) -> &[Vec<u32>] { &self.classes }

    /// A stable original-token representative for `class_id`.
    pub fn representative(&self, class_id: u32) -> Option<u32> {
        self.representatives
            .get(class_id as usize)
            .copied()
            .filter(|&token| token != u32::MAX)
    }

    /// Dense original-token-ID to class-ID map. Missing token IDs contain `u32::MAX`.
    pub fn original_to_class(&self) -> &[u32] { &self.original_to_class }

    /// Number of packed `u64` words needed for a class-space mask.
    pub fn internal_mask_len(&self) -> usize { self.num_classes().div_ceil(64) }

    /// Number of packed `u32` words needed for an original-token-space mask.
    pub fn original_mask_len(&self) -> usize { self.original_to_class.len().div_ceil(32) }

    /// Expand a packed class-space mask into the original model-token ID space.
    ///
    /// Bit `i` in `internal_mask` selects equivalence class `i`. Every original
    /// token in each selected class is set in the returned packed `u32` mask.
    /// Bits beyond [`Self::num_classes`] are ignored.
    pub fn expand_mask(&self, internal_mask: &[u64]) -> Vec<u32> {
        let mut out = vec![0u32; self.original_mask_len()];
        self.fill_expanded_mask(internal_mask, &mut out);
        out
    }

    /// Expand a packed class-space mask into an existing original-token mask buffer.
    ///
    /// `out` must contain at least [`Self::original_mask_len`] `u32` words. The
    /// entire supplied buffer is cleared before expansion, matching the overwrite
    /// semantics of [`ConstraintState::fill_mask`](crate::ConstraintState::fill_mask).
    pub fn fill_expanded_mask(&self, internal_mask: &[u64], out: &mut [u32]) {
        let required = self.original_mask_len();
        assert!(
            out.len() >= required,
            "expanded mask buffer is smaller than original vocabulary mask"
        );
        out.fill(0);

        for (word_index, &word) in internal_mask.iter().enumerate() {
            let class_base = word_index * 64;
            if class_base >= self.num_classes() {
                break;
            }
            let mut selected = word;
            while selected != 0 {
                let bit = selected.trailing_zeros() as usize;
                let class_id = class_base + bit;
                if class_id >= self.num_classes() {
                    break;
                }
                for &(output_word, output_bits) in &self.class_output_masks[class_id] {
                    out[output_word as usize] |= output_bits;
                }
                selected &= selected - 1;
            }
        }
    }
}
