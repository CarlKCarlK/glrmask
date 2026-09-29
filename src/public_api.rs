use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use crate::compiler::constraint_compose::{
    CompiledSubgrammarInput, SegmentedBoundaryBackend,
    compose_constraints_owned_parent_segmented_shared,
};
use crate::runtime::Constraint as RuntimeConstraint;
use crate::dynamic_constraint::DynamicConstraint;
use crate::runtime::BoundaryTriggerDetail;
use crate::{Error, ExactToken, ExactTokens, Result, Vocab};

/// Preferred build/runtime trade-off.
///
/// These variants express caller intent rather than exposing GLRMask's
/// internal static/dynamic implementation choices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Optimization {
    /// Let GLRMask choose automatically.
    #[default]
    Auto,
    /// Prefer low compile/link latency.
    FastBuild,
    /// Prefer lower per-token masking latency, accepting more build work.
    FastRuntime,
}

/// Parser implementation used by a final runnable constraint. TemplateDfa
/// physically omits LR action/goto storage; unsupported composition fails
/// explicitly rather than retaining a hidden table fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ParserBackend {
    #[default]
    LrTable,
    TemplateDfa,
}

/// Options that apply only when producing a final runnable constraint.
#[derive(Debug, Clone, Default)]
pub struct BuildOptions {
    end_tokens: Vec<u32>,
    optimization: Optimization,
    parser_backend: ParserBackend,
}

impl BuildOptions {
    /// Configure tokens that may terminate generation once the final
    /// constraint is accepting.
    pub fn end_tokens(mut self, ids: impl IntoIterator<Item = u32>) -> Self {
        self.end_tokens = ids.into_iter().collect();
        self
    }

    /// Select the preferred build/runtime trade-off.
    pub fn optimization(mut self, optimization: Optimization) -> Self {
        self.optimization = optimization;
        self
    }

    pub fn parser_backend(mut self, backend: ParserBackend) -> Self {
        self.parser_backend = backend; self
    }

    pub(crate) fn end_token_ids(&self) -> &[u32] {
        &self.end_tokens
    }

    pub(crate) fn optimization_value(&self) -> Optimization {
        self.optimization
    }
}

/// Reusable, vocabulary-specific compiled constraint in pre-link form.
///
/// An unlinked constraint may retain unresolved external slots. It is
/// deliberately not runnable; bind already-compiled child [`crate::Constraint`]s and
/// call [`UnlinkedConstraint::link`] once all required slots are satisfied.
#[derive(Debug, Clone)]
pub struct UnlinkedConstraint {
    inner: Arc<RuntimeConstraint>,
    /// Direct token slots in this component. The runtime linker slots keep
    /// both token and grammar terminals; this manifest preserves their public
    /// kind while bindings stay deferred until the final link operation.
    token_slots: BTreeSet<String>,
    /// Compiled-only immutable attachments. Delaying composition is what lets
    /// the terminal link call choose the boundary/runtime trade-off.
    bindings: BTreeMap<String, ModuleBinding>,
}

#[derive(Debug, Clone)]
enum ModuleBinding {
    Module(Box<UnlinkedConstraint>),
    Constraint(Arc<RuntimeConstraint>),
    ExactTokens(Arc<[u32]>),
}

#[derive(Debug, Clone)]
pub(crate) enum GrammarValue<'a> {
    Source(Grammar<'a>),
    ExactToken(ExactToken),
    ExactTokens(ExactTokens),
}

impl GrammarValue<'_> {
    fn extern_kind(&self) -> ExternKind {
        match self {
            Self::ExactToken(_) | Self::ExactTokens(_) => ExternKind::Token,
            Self::Source(_) => ExternKind::Grammar,
        }
    }
}

#[doc(hidden)]
pub(crate) trait IntoGrammarValue<'a> {
    fn into_grammar_value(self) -> GrammarValue<'a>;
}

#[derive(Debug, Clone)]
pub(crate) enum UnlinkedValue<'a> {
    StaticBorrowed(&'a RuntimeConstraint),
    StaticOwned(Arc<RuntimeConstraint>),
    ExactToken(ExactToken),
    ExactTokens(ExactTokens),
}

#[doc(hidden)]
pub(crate) trait IntoUnlinkedValue<'a> {
    fn into_unlinked_value(self) -> UnlinkedValue<'a>;
}

/// Grammar description with immutable source-grammar or exact-token bindings.
#[derive(Debug, Clone)]
pub struct Grammar<'a> {
    source: GrammarSource<'a>,
    bindings: BTreeMap<String, GrammarValue<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GrammarSource<'a> {
    Ebnf(&'a str),
    Lark(&'a str),
    JsonSchema(&'a str),
    Glrm(&'a str),
}

impl<'a> Grammar<'a> {
    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn ebnf(source: &'a str) -> Self { Self::new(GrammarSource::Ebnf(source)) }
    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn lark(source: &'a str) -> Self { Self::new(GrammarSource::Lark(source)) }
    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn json_schema(source: &'a str) -> Self { Self::new(GrammarSource::JsonSchema(source)) }
    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn glrm(source: &'a str) -> Self { Self::new(GrammarSource::Glrm(source)) }

    /// Create a grammar from EBNF source.
    pub fn from_ebnf(source: &'a str) -> Self { Self::new(GrammarSource::Ebnf(source)) }

    /// Create a grammar from Lark source.
    pub fn from_lark(source: &'a str) -> Self { Self::new(GrammarSource::Lark(source)) }

    /// Create a grammar from JSON Schema text.
    pub fn from_json_schema(source: &'a str) -> Self { Self::new(GrammarSource::JsonSchema(source)) }

    /// Create a grammar from native GLRM source.
    pub fn from_glrm(source: &'a str) -> Self { Self::new(GrammarSource::Glrm(source)) }

    fn new(source: GrammarSource<'a>) -> Self {
        Self { source, bindings: BTreeMap::new() }
    }

    /// Bind a declared slot to a source grammar or vocabulary-qualified
    /// exact-token value.
    ///
    /// Binding is immutable: the original grammar remains usable.
    #[allow(private_bounds)]
    pub fn bind<T>(&self, name: impl AsRef<str>, value: T) -> Result<Self>
    where
        T: IntoGrammarValue<'a>,
    {
        let name = name.as_ref();
        let GrammarSource::Glrm(source) = self.source else {
            return Err(Error::Compilation(
                "external bindings require a GLRM parent grammar".to_owned(),
            ));
        };
        let declarations = crate::grammar::glrm::external_declarations(source)?;
        let value = value.into_grammar_value();
        let expected = value.extern_kind();
        let declared_as_expected = match expected {
            ExternKind::Token => declarations.token_names.iter().any(|declared| declared == name),
            ExternKind::Grammar => declarations
                .grammar_names
                .iter()
                .any(|declared| declared == name),
        };
        let declared_as_other = match expected {
            ExternKind::Token => declarations
                .grammar_names
                .iter()
                .any(|declared| declared == name),
            ExternKind::Grammar => declarations.token_names.iter().any(|declared| declared == name),
        };
        if !declared_as_expected {
            if declared_as_other {
                return Err(Error::Compilation(format!(
                    "external {name:?} has kind {}, not {}",
                    expected.opposite_name(),
                    expected.name(),
                )));
            }
            return Err(Error::Compilation(format!(
                "no external {} named {name:?} is declared",
                expected.name(),
            )));
        }
        if self.bindings.contains_key(name) {
            return Err(Error::Compilation(format!(
                "external {name:?} was bound more than once",
            )));
        }
        let mut next = self.clone();
        next.bindings.insert(name.to_owned(), value);
        Ok(next)
    }

    /// Compatibility spelling for source-only grammar binding.
    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn bind_grammar(self, name: impl AsRef<str>, grammar: Grammar<'a>) -> Result<Self> {
        self.bind(name, grammar)
    }

    /// Compile this complete grammar description into a runnable constraint.
    pub fn compile(&self, vocab: &Vocab) -> Result<RuntimeConstraint> {
        self.compile_with(vocab, BuildOptions::default())
    }

    /// Compile this complete grammar description with final build options.
    pub fn compile_with(
        &self,
        vocab: &Vocab,
        options: BuildOptions,
    ) -> Result<RuntimeConstraint> {
        if options.parser_backend == ParserBackend::TemplateDfa {
            if !self.bindings.is_empty() {
                return Err(Error::Compilation("template-parser component composition is not implemented; no LR fallback is permitted".into()));
            }
            if options.optimization == Optimization::FastBuild {
                // Use the existing O2 runtime with the bounded grammar normal
                // form required by finite templates. Ordinary dynamic grammar
                // preparation intentionally allows recursive action closures.
                let dynamic = match self.source {
                    GrammarSource::Glrm(source) => DynamicConstraint::from_glrm_with_bounded_template_parser(source, vocab),
                    GrammarSource::Ebnf(source) => DynamicConstraint::from_ebnf_with_bounded_template_parser(source, vocab),
                    GrammarSource::Lark(source) => DynamicConstraint::from_lark_with_bounded_template_parser(source, vocab),
                    GrammarSource::JsonSchema(source) => DynamicConstraint::from_json_schema_with_bounded_template_parser(source, vocab),
                }?;
                let mut components = dynamic.into_constraints();
                if components.len() != 1 {
                    return Err(Error::Compilation("a template-parser Constraint currently requires one compiled component; union composition is not implemented".into()));
                }
                let constraint = components.pop().unwrap();
                ensure_runnable_constraint(&constraint)?;
                return constraint.with_end_tokens(options.end_token_ids());
            }
        }
        let mut spec = ConstraintSpec::builder(self.clone(), vocab)?.build()?;
        spec.automatic_boundary_selection = true;
        if let Some(name) = spec.unbound_grammar_names.first() {
            return Err(Error::Compilation(format!(
                "external grammar {name:?} is unbound; compile_unlinked() if a reusable pre-link artifact is intended",
            )));
        }
        let mut constraint = spec.compile_final(options.optimization_value())?;
        ensure_runnable_constraint(&constraint)?;
        if options.parser_backend == ParserBackend::TemplateDfa {
            constraint.install_template_parser()?;
        }
        constraint.with_end_tokens(options.end_token_ids())
    }

    /// Compile reusable local machinery while allowing unresolved grammar
    /// or exact-token slots to remain open.
    pub fn compile_unlinked(&self, vocab: &Vocab) -> Result<UnlinkedConstraint> {
        // Compile this component locally. Grammar-child attachments remain
        // compiled values in the unlinked graph and are linked only by link_with,
        // where the caller's Optimization choice is finally known.
        let (local_grammar, source_bindings) = self.clone().into_source_only_and_bindings();
        let mut builder = ConstraintSpec::builder(local_grammar, vocab)?;
        let mut bindings = BTreeMap::<String, ModuleBinding>::new();
        let mut source_children = Vec::<(String, Grammar<'a>)>::new();
        for (name, value) in source_bindings {
            match value {
                GrammarValue::ExactToken(token) => {
                    if !token.targets(vocab) {
                        return Err(Error::Compilation(format!(
                            "external token {name:?} was built for an incompatible vocabulary",
                        )));
                    }
                    builder.token_bindings.insert(name, vec![token.id()]);
                }
                GrammarValue::ExactTokens(tokens) => {
                    if !tokens.targets(vocab) {
                        return Err(Error::Compilation(format!(
                            "external token {name:?} was built for an incompatible vocabulary",
                        )));
                    }
                    builder.token_bindings.insert(name, tokens.ids().to_vec());
                }
                GrammarValue::Source(child) => {
                    source_children.push((name, child));
                }
            }
        }
        let unbound_tokens = builder
            .declared_tokens
            .iter()
            .filter(|name| !builder.token_bindings.contains_key(*name))
            .cloned()
            .collect::<Vec<_>>();
        let mut placeholder_ids = BTreeMap::<String, u32>::new();
        if !unbound_tokens.is_empty() {
            let mut reserved = BTreeSet::<u32>::new();
            if let GrammarSource::Glrm(source) = builder.grammar.source {
                reserved.extend(crate::grammar::glrm::special_token_ids(source)?);
            }
            reserved.extend(
                builder
                    .token_bindings
                    .values()
                    .flat_map(|ids| ids.iter().copied()),
            );
            for name in unbound_tokens {
                let placeholder = crate::import::external_placeholder_token_id_avoiding(
                    vocab,
                    reserved.iter().copied(),
                )?;
                reserved.insert(placeholder);
                builder
                    .token_bindings
                    .insert(name.clone(), vec![placeholder]);
                placeholder_ids.insert(name, placeholder);
            }
        }
        let mut spec = builder.build()?;
        spec.allow_open_source_tokens = true;
        spec.automatic_boundary_selection = true;
        spec.open_token_placeholders = placeholder_ids.clone();
        let (constraint, source_modules) = rayon::join(
            || spec.compile(),
            || {
                source_children
                    .into_par_iter()
                    .map(|(name, child)| {
                        Ok((name, ModuleBinding::Module(Box::new(child.compile_unlinked(vocab)?))))
                    })
                    .collect::<Result<Vec<_>>>()
            },
        );
        let constraint = constraint?;
        for (name, binding) in source_modules? {
            bindings.insert(name, binding);
        }
        let module = UnlinkedConstraint {
            inner: Arc::new(constraint),
            token_slots: placeholder_ids.keys().cloned().collect(),
            bindings,
        };
        module.validate_slot_manifest()?;
        Ok(module)
    }

    fn unresolved_token_names(&self) -> Result<BTreeSet<String>> {
        let mut names = match self.source {
            GrammarSource::Glrm(source) => crate::grammar::glrm::external_declarations(source)?
                .token_names.into_iter()
                .filter(|name| !self.bindings.contains_key(name)).collect(),
            _ => BTreeSet::new(),
        };
        for (name, value) in &self.bindings {
            let nested = match value {
                GrammarValue::Source(grammar) => grammar.unresolved_token_names()?,
                _ => continue,
            };
            names.extend(nested.into_iter().map(|slot| format!("{name}.{slot}")));
        }
        Ok(names)
    }

    fn glrm_source(&self) -> Option<&'a str> {
        match self.source {
            GrammarSource::Glrm(source) => Some(source),
            _ => None,
        }
    }

    fn into_source_only_and_bindings(self) -> (Self, BTreeMap<String, GrammarValue<'a>>) {
        let Self { source, bindings } = self;
        (Self::new(source), bindings)
    }
}

impl<'a> IntoGrammarValue<'a> for Grammar<'a> {
    fn into_grammar_value(self) -> GrammarValue<'a> {
        GrammarValue::Source(self)
    }
}

impl<'a> IntoGrammarValue<'a> for &Grammar<'a> {
    fn into_grammar_value(self) -> GrammarValue<'a> {
        GrammarValue::Source(self.clone())
    }
}

impl<'a> IntoGrammarValue<'a> for ExactToken {
    fn into_grammar_value(self) -> GrammarValue<'a> {
        GrammarValue::ExactToken(self)
    }
}

impl<'a> IntoGrammarValue<'a> for ExactTokens {
    fn into_grammar_value(self) -> GrammarValue<'a> {
        GrammarValue::ExactTokens(self)
    }
}

impl<'a> IntoUnlinkedValue<'a> for &'a RuntimeConstraint {
    fn into_unlinked_value(self) -> UnlinkedValue<'a> {
        UnlinkedValue::StaticBorrowed(self)
    }
}

impl<'a> IntoUnlinkedValue<'a> for RuntimeConstraint {
    fn into_unlinked_value(self) -> UnlinkedValue<'a> {
        UnlinkedValue::StaticOwned(Arc::new(self))
    }
}

impl<'a> IntoUnlinkedValue<'a> for Arc<RuntimeConstraint> {
    fn into_unlinked_value(self) -> UnlinkedValue<'a> {
        UnlinkedValue::StaticOwned(self)
    }
}

impl<'a> IntoUnlinkedValue<'a> for ExactToken {
    fn into_unlinked_value(self) -> UnlinkedValue<'a> {
        UnlinkedValue::ExactToken(self)
    }
}

impl<'a> IntoUnlinkedValue<'a> for ExactTokens {
    fn into_unlinked_value(self) -> UnlinkedValue<'a> {
        UnlinkedValue::ExactTokens(self)
    }
}


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
    original_to_class: Vec<u32>,
    classes: Vec<Vec<u32>>,
    representatives: Vec<u32>,
    class_output_masks: Vec<Vec<(u32, u32)>>,
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

/// A grammar, vocabulary, and complete set of extern bindings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BoundarySummaryPolicy {
    /// Do not prepare or persist the static-composition candidate summary.
    Disabled,
    /// Prepare the summary only when a later static composition actually needs
    /// it. This is the default so ordinary standalone compilation pays no tax.
    #[default]
    LazyOnCompose,
    /// Prepare the bounded conservative summary during component compilation so
    /// later composition can consume it without first-touch analysis.
    BoundedPrepare,
}

/// A grammar, vocabulary, and complete set of extern bindings.
#[derive(Debug, Clone)]
pub struct ConstraintSpec<'a> {
    grammar: Grammar<'a>,
    vocab: &'a Vocab,
    token_bindings: BTreeMap<String, Vec<u32>>,
    grammar_bindings: BTreeMap<String, GrammarBinding<'a>>,
    unbound_grammar_names: Vec<String>,
    // Only explicit module compilation may preserve unbound source token
    // slots. Legacy specification compilation keeps its validation contract.
    allow_open_source_tokens: bool,
    automatic_boundary_selection: bool,
    // These belong to this source component. Register their terminal identity
    // before linking: another component may independently use the same hidden
    // token ID, which is not a globally unique linker coordinate.
    open_token_placeholders: BTreeMap<String, u32>,
    boundary_trigger_detail: BoundaryTriggerDetail,
    boundary_summary_policy: BoundarySummaryPolicy,
}

/// Builder for [`ConstraintSpec`].
#[derive(Debug)]
pub struct ConstraintSpecBuilder<'a> {
    grammar: Grammar<'a>,
    vocab: &'a Vocab,
    declared_tokens: BTreeSet<String>,
    declared_grammars: BTreeSet<String>,
    token_bindings: BTreeMap<String, Vec<u32>>,
    grammar_bindings: BTreeMap<String, GrammarBinding<'a>>,
    boundary_trigger_detail: BoundaryTriggerDetail,
    boundary_summary_policy: BoundarySummaryPolicy,
}

/// Internal input accepted by [`ConstraintSpecBuilder::bind_grammar`].
#[doc(hidden)]
#[non_exhaustive]
#[derive(Debug, Clone)]
pub(crate) enum GrammarBinding<'a> {
    Source(Grammar<'a>),
    Spec(Box<ConstraintSpec<'a>>),
    #[doc(hidden)]
    StaticBorrowed(&'a RuntimeConstraint),
    #[doc(hidden)]
    StaticOwned(Arc<RuntimeConstraint>),
    #[doc(hidden)]
    DynamicBorrowed(&'a DynamicConstraint),
    #[doc(hidden)]
    DynamicOwned(Arc<DynamicConstraint>),
}

/// Converts a supported child grammar into an internal binding.
#[doc(hidden)]
pub(crate) trait IntoGrammarBinding<'a> {
    #[doc(hidden)]
    fn into_grammar_binding(self) -> GrammarBinding<'a>;
}

impl<'a> IntoGrammarBinding<'a> for Grammar<'a> {
    fn into_grammar_binding(self) -> GrammarBinding<'a> { GrammarBinding::Source(self) }
}

impl<'a> IntoGrammarBinding<'a> for ConstraintSpec<'a> {
    fn into_grammar_binding(self) -> GrammarBinding<'a> {
        GrammarBinding::Spec(Box::new(self))
    }
}

impl<'a> IntoGrammarBinding<'a> for &ConstraintSpec<'a> {
    fn into_grammar_binding(self) -> GrammarBinding<'a> {
        GrammarBinding::Spec(Box::new(self.clone()))
    }
}

impl<'a> IntoGrammarBinding<'a> for RuntimeConstraint {
    fn into_grammar_binding(self) -> GrammarBinding<'a> {
        GrammarBinding::StaticOwned(Arc::new(self))
    }
}

impl<'a> IntoGrammarBinding<'a> for &'a RuntimeConstraint {
    fn into_grammar_binding(self) -> GrammarBinding<'a> {
        GrammarBinding::StaticBorrowed(self)
    }
}

impl<'a> IntoGrammarBinding<'a> for Arc<RuntimeConstraint> {
    fn into_grammar_binding(self) -> GrammarBinding<'a> {
        GrammarBinding::StaticOwned(self)
    }
}

impl<'a> IntoGrammarBinding<'a> for DynamicConstraint {
    fn into_grammar_binding(self) -> GrammarBinding<'a> {
        GrammarBinding::DynamicOwned(Arc::new(self))
    }
}

impl<'a> IntoGrammarBinding<'a> for &'a DynamicConstraint {
    fn into_grammar_binding(self) -> GrammarBinding<'a> {
        GrammarBinding::DynamicBorrowed(self)
    }
}

impl<'a> IntoGrammarBinding<'a> for Arc<DynamicConstraint> {
    fn into_grammar_binding(self) -> GrammarBinding<'a> {
        GrammarBinding::DynamicOwned(self)
    }
}

impl<'a> ConstraintSpec<'a> {
    /// Start a specification for `grammar` and `vocab`.
    pub fn builder(
        grammar: Grammar<'a>,
        vocab: &'a Vocab,
    ) -> Result<ConstraintSpecBuilder<'a>> {
        ConstraintSpecBuilder::new(grammar, vocab)
    }

    /// Compile this specification into a [`Constraint`](crate::Constraint).
    pub fn compile(&self) -> Result<RuntimeConstraint> {
        let mut constraint = self.compile_static_with_trigger_uncached()?;
        // A constraint with unresolved external grammars is explicitly a
        // reusable late-bind parent.  Cache its exact tokenizer-reset
        // terminal -> model-token relation now, while compilation already owns
        // the vocabulary, instead of rescanning the vocabulary on every later
        // bind.  The cache is part of composition metadata and therefore
        // survives ordinary save/load.
        if !constraint.late_grammar_slots.is_empty() {
            constraint.ensure_composition_reset_tokens_by_terminal();
        }
        match self.boundary_summary_policy {
            BoundarySummaryPolicy::Disabled => {
                let _ = constraint.boundary_candidate_summary.set(
                    crate::runtime::BoundaryCandidateSummary::Unknown {
                        reason: crate::runtime::SummaryUnavailable::Disabled,
                    },
                );
            }
            BoundarySummaryPolicy::LazyOnCompose => {}
            BoundarySummaryPolicy::BoundedPrepare => {
                crate::compiler::boundary_candidates::prepare_boundary_candidate_summary(
                    &constraint,
                    self.vocab,
                );
            }
        }
        // Keep first-save latency bounded without charging serialization to
        // every ordinary compile. Static virtual-residual constraints can carry
        // very large runtime sections, while exceptionally large parser-template
        // caches are expensive to encode on demand. Prime only those structural
        // tails; ordinary constraints serialize when save() is actually called.
        if should_prime_first_save_artifact(&constraint) {
            constraint.cache_serialized_artifact_for_save();
        }
        Ok(constraint)
    }

    fn compile_final(&self, optimization: Optimization) -> Result<RuntimeConstraint> {
        match optimization {
            Optimization::FastBuild => {
                let dynamic = self.compile_dynamic()?;
                collapse_dynamic_alternatives(
                    dynamic.into_constraints(),
                    self.vocab,
                    SegmentedBoundaryBackend::Dynamic,
                )
            }
            Optimization::Auto | Optimization::FastRuntime => self.compile(),
        }
    }

    fn compile_static_with_trigger_uncached(&self) -> Result<RuntimeConstraint> {
        let mut constraint = self.compile_static_uncached()?;
        constraint
            .build_boundary_trigger(self.boundary_trigger_detail)
            .map_err(Error::Compilation)?;
        Ok(constraint)
    }

    fn register_open_token_placeholders(&self, parent: &mut RuntimeConstraint) -> Result<()> {
        for (name, &placeholder_id) in &self.open_token_placeholders {
            let terminals = parent.special_token_terminals.iter()
                .filter(|special| special.token_id == placeholder_id)
                .map(|special| special.terminal_id).collect::<BTreeSet<_>>();
            if terminals.len() != 1 {
                return Err(Error::Compilation(format!(
                    "compiled external token {name:?} did not resolve to exactly one component-local terminal",
                )));
            }
            parent.late_grammar_slots.push(crate::runtime::LateGrammarSlot {
                name: name.clone(),
                terminal_id: *terminals.iter().next().expect("length checked"),
            });
        }
        if !self.open_token_placeholders.is_empty() {
            parent.serialized_artifact_cache = None;
            if parent.sanitize_late_grammar_placeholder_token_domain() {
                parent.rebuild_runtime_caches();
            }
        }
        Ok(())
    }

    fn compile_static_uncached(&self) -> Result<RuntimeConstraint> {
        let token_bindings = self.token_binding_refs();
        let compile_parent = || {
            if let Some(source) = self.grammar.glrm_source() {
                RuntimeConstraint::from_glrm_grammar_with_subgrammars_bindings_and_end_tokens(
                    source, &[], self.vocab, &token_bindings, &[],
                )
            } else {
                compile_static_source(&self.grammar, self.vocab, &token_bindings)
            }
        };
        if self.grammar_bindings.is_empty() {
            let mut parent = compile_parent()?;
            self.register_open_token_placeholders(&mut parent)?;
            return Ok(parent);
        }

        // Parent-local compilation and child-local compilation have no
        // semantic dependency. Run them in the same Rayon pool and join only
        // before boundary construction, which genuinely needs both artifacts.
        let (parent, children) = rayon::join(
            compile_parent,
            || self.compile_children(ChildCompileMode::Static),
        );
        let mut parent = parent?;
        self.register_open_token_placeholders(&mut parent)?;
        let children = children?;
        let children = prepare_compiled_children(
            children, self.vocab, SegmentedBoundaryBackend::StaticParserDwa,
        )?;
        let boundary_backend = if self.automatic_boundary_selection {
            select_supported_boundary(&parent, &children)?
        } else {
            SegmentedBoundaryBackend::StaticParserDwa
        };
        compose_named_children(parent, &children, self.vocab, boundary_backend)
    }

    /// Compile this specification into a [`DynamicConstraint`].
    pub fn compile_dynamic(&self) -> Result<DynamicConstraint> {
        let mut constraint = self.compile_dynamic_uncached()?;
        for component in constraint.constraints_mut() {
            // Retain the supplied shared vocabulary, just as static compilation
            // does. Reconstructing it from every token's bytes on the first bind
            // costs more than compiling many small dynamic components.
            let _ = component.late_bind_vocab.set(self.vocab.clone());
            component
                .build_boundary_trigger(self.boundary_trigger_detail)
                .map_err(Error::Compilation)?;
            match self.boundary_summary_policy {
                BoundarySummaryPolicy::Disabled => {
                    let _ = component.boundary_candidate_summary.set(
                        crate::runtime::BoundaryCandidateSummary::Unknown {
                            reason: crate::runtime::SummaryUnavailable::Disabled,
                        },
                    );
                }
                BoundarySummaryPolicy::LazyOnCompose => {}
                BoundarySummaryPolicy::BoundedPrepare => {
                    crate::compiler::boundary_candidates::prepare_boundary_candidate_summary(
                        component,
                        self.vocab,
                    );
                }
            }
        }
        Ok(constraint)
    }

    fn compile_dynamic_uncached(&self) -> Result<DynamicConstraint> {
        let token_bindings = self.token_binding_refs();
        if self.grammar_bindings.is_empty() {
            if let Some(source) = self.grammar.glrm_source() {
                return DynamicConstraint::from_glrm_grammar_with_subgrammars_and_bindings(
                    source,
                    &[],
                    self.vocab,
                    &token_bindings,
                );
            }
            return compile_dynamic_source(&self.grammar, self.vocab, &token_bindings);
        }

        let source = self.grammar.glrm_source().ok_or_else(|| {
            Error::Compilation("external grammar bindings require a GLRM grammar".to_owned())
        })?;
        let (parents, children) = rayon::join(
            || {
                DynamicConstraint::from_glrm_grammar_with_subgrammars_and_bindings(
                    source,
                    &[],
                    self.vocab,
                    &token_bindings,
                )
            },
            || self.compile_children(ChildCompileMode::Dynamic),
        );
        let parents = parents?;
        let children = children?;
        let children = prepare_compiled_children(
            children,
            self.vocab,
            SegmentedBoundaryBackend::Dynamic,
        )?;
        let alternatives = parents
            .clone_constraints()
            .into_iter()
            .map(|parent| {
                compose_named_children(
                    parent,
                    &children,
                    self.vocab,
                    SegmentedBoundaryBackend::Dynamic,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(DynamicConstraint::from_constraints(alternatives))
    }

    fn token_binding_refs(&self) -> Vec<(&str, &[u32])> {
        self.token_bindings
            .iter()
            .map(|(name, ids)| (name.as_str(), ids.as_slice()))
            .collect()
    }

    fn compile_children(
        &self,
        mode: ChildCompileMode,
    ) -> Result<Vec<(String, CompiledChild<'_>)>> {
        let bindings = self.grammar_bindings.iter().collect::<Vec<_>>();
        bindings
            .par_iter()
            .map(|(name, binding)| {
                Ok((
                    (*name).clone(),
                    binding.compile(self.vocab, mode, self.allow_open_source_tokens)?,
                ))
            })
            .collect()
    }

    fn targets(&self, vocab: &Vocab) -> bool {
        self.vocab.same_model_vocab(vocab)
    }
}

impl<'a> ConstraintSpecBuilder<'a> {
    fn new(grammar: Grammar<'a>, vocab: &'a Vocab) -> Result<Self> {
        let (grammar, source_bindings) = grammar.into_source_only_and_bindings();
        let declarations = match grammar.source {
            GrammarSource::Glrm(source) => crate::grammar::glrm::external_declarations(source)?,
            _ => crate::grammar::glrm::GlrmExternalDeclarations {
                token_names: Vec::new(),
                grammar_names: Vec::new(),
            },
        };
        let declared_grammars = declarations
            .grammar_names
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let declared_tokens = declarations
            .token_names
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let mut token_bindings = BTreeMap::new();
        let mut grammar_bindings = BTreeMap::new();
        for (name, value) in source_bindings {
            match value {
                GrammarValue::ExactToken(token) => {
                    if !declared_tokens.contains(&name) {
                        return Err(Error::Compilation(format!(
                            "exact-token binding was supplied for unknown external token {name:?}",
                        )));
                    }
                    if !token.targets(vocab) {
                        return Err(Error::Compilation(format!(
                            "external token {name:?} was built for an incompatible vocabulary",
                        )));
                    }
                    token_bindings.insert(name, vec![token.id()]);
                }
                GrammarValue::ExactTokens(tokens) => {
                    if !declared_tokens.contains(&name) {
                        return Err(Error::Compilation(format!(
                            "exact-token binding was supplied for unknown external token {name:?}",
                        )));
                    }
                    if !tokens.targets(vocab) {
                        return Err(Error::Compilation(format!(
                            "external token {name:?} was built for an incompatible vocabulary",
                        )));
                    }
                    token_bindings.insert(name, tokens.ids().to_vec());
                }
                GrammarValue::Source(child) => {
                    if !declared_grammars.contains(&name) {
                        return Err(Error::Compilation(format!(
                            "source binding was supplied for unknown external grammar {name:?}",
                        )));
                    }
                    grammar_bindings.insert(name, GrammarBinding::Source(child));
                }
            }
        }
        for (name, binding) in &mut grammar_bindings {
            binding.bind_target(vocab, name)?;
        }
        Ok(Self {
            grammar,
            vocab,
            declared_tokens,
            declared_grammars,
            token_bindings,
            grammar_bindings,
            boundary_trigger_detail: BoundaryTriggerDetail::None,
            boundary_summary_policy: BoundarySummaryPolicy::LazyOnCompose,
        })
    }

    /// Request reusable dynamic-boundary trigger metadata for this component.
    ///
    /// The default is [`BoundaryTriggerDetail::None`], which adds no trigger
    /// construction cost. This setting is independent of static vs dynamic
    /// ordinary masking.
    pub fn boundary_trigger_detail(mut self, detail: BoundaryTriggerDetail) -> Self {
        self.boundary_trigger_detail = detail;
        self
    }

    /// Configure preparation of the grammar-aware static composition boundary
    /// candidate summary. This is independent of [`BoundaryTriggerDetail`],
    /// which controls dynamic-runtime trigger metadata.
    pub fn boundary_summary_policy(mut self, policy: BoundarySummaryPolicy) -> Self {
        self.boundary_summary_policy = policy;
        self
    }

    /// Bind an `extern token NAME;` declaration to exact token IDs.
    pub fn bind_token(
        mut self,
        name: impl AsRef<str>,
        token_ids: impl IntoIterator<Item = u32>,
    ) -> Result<Self> {
        let name = name.as_ref();
        self.require_kind(name, ExternKind::Token)?;
        if self.token_bindings.contains_key(name) {
            return Err(Error::Compilation(format!(
                "external token {name:?} was bound more than once",
            )));
        }
        let token_ids = token_ids.into_iter().collect::<Vec<_>>();
        if token_ids.is_empty() {
            return Err(Error::Compilation(format!(
                "external token {name:?} must bind at least one exact token ID",
            )));
        }
        let unique = token_ids.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() != token_ids.len() {
            return Err(Error::Compilation(format!(
                "external token {name:?} contains a duplicate token ID",
            )));
        }
        self.token_bindings
            .insert(name.to_owned(), unique.into_iter().collect());
        Ok(self)
    }

    /// Bind an `extern grammar NAME;` to a source, spec, or compiled constraint.
    #[allow(private_bounds)]
    pub fn bind_grammar<T>(mut self, name: impl AsRef<str>, realization: T) -> Result<Self>
    where
        T: IntoGrammarBinding<'a>,
    {
        let name = name.as_ref();
        self.require_kind(name, ExternKind::Grammar)?;
        if self.grammar_bindings.contains_key(name) {
            return Err(Error::Compilation(format!(
                "external grammar {name:?} was bound more than once",
            )));
        }
        let mut realization = realization.into_grammar_binding();
        realization.bind_target(self.vocab, name)?;
        self.grammar_bindings.insert(name.to_owned(), realization);
        Ok(self)
    }

    /// Check that every exact-token extern is bound and finish the specification.
    ///
    /// External grammars may remain unresolved. Their names are retained in
    /// the compiled constraint and can be supplied later with
    /// [`RuntimeConstraint::bind_grammar`] or [`DynamicConstraint::bind_grammar`].
    pub fn build(self) -> Result<ConstraintSpec<'a>> {
        if let Some(name) = self
            .declared_tokens
            .iter()
            .find(|name| !self.token_bindings.contains_key(*name))
        {
            return Err(Error::Compilation(format!(
                "GLRM declares external token {name:?}, but no exact-token binding was supplied",
            )));
        }
        let unbound_grammar_names = self
            .declared_grammars
            .iter()
            .filter(|name| !self.grammar_bindings.contains_key(*name))
            .cloned()
            .collect::<Vec<_>>();
        Ok(ConstraintSpec {
            grammar: self.grammar,
            vocab: self.vocab,
            token_bindings: self.token_bindings,
            grammar_bindings: self.grammar_bindings,
            unbound_grammar_names,
            allow_open_source_tokens: false,
            automatic_boundary_selection: false,
            open_token_placeholders: BTreeMap::new(),
            boundary_trigger_detail: self.boundary_trigger_detail,
            boundary_summary_policy: self.boundary_summary_policy,
        })
    }

    fn require_kind(&self, name: &str, expected: ExternKind) -> Result<()> {
        let correct = match expected {
            ExternKind::Token => self.declared_tokens.contains(name),
            ExternKind::Grammar => self.declared_grammars.contains(name),
        };
        if correct {
            return Ok(());
        }
        let wrong_kind = match expected {
            ExternKind::Token => self.declared_grammars.contains(name),
            ExternKind::Grammar => self.declared_tokens.contains(name),
        };
        if wrong_kind {
            return Err(Error::Compilation(format!(
                "external {name:?} has kind {}, not {}",
                expected.opposite_name(),
                expected.name(),
            )));
        }
        Err(Error::Compilation(format!(
            "no external {} named {name:?} is declared",
            expected.name(),
        )))
    }
}

#[derive(Clone, Copy)]
enum ExternKind {
    Token,
    Grammar,
}

impl ExternKind {
    fn name(self) -> &'static str {
        match self {
            Self::Token => "token",
            Self::Grammar => "grammar",
        }
    }

    fn opposite_name(self) -> &'static str {
        match self {
            Self::Token => "grammar",
            Self::Grammar => "token",
        }
    }
}

#[derive(Clone, Copy)]
enum ChildCompileMode {
    Static,
    Dynamic,
}

enum CompiledChild<'a> {
    StaticBorrowed(&'a RuntimeConstraint),
    StaticOwned(RuntimeConstraint),
    DynamicBorrowed(&'a DynamicConstraint),
    DynamicOwned(DynamicConstraint),
}

impl CompiledChild<'_> {
    fn into_constraints(self) -> Vec<RuntimeConstraint> {
        match self {
            Self::StaticBorrowed(constraint) => vec![constraint.clone()],
            Self::StaticOwned(constraint) => vec![constraint],
            Self::DynamicBorrowed(constraint) => constraint.clone_constraints(),
            Self::DynamicOwned(constraint) => constraint.clone_constraints(),
        }
    }
}

fn static_constraint_targets(constraint: &RuntimeConstraint, vocab: &Vocab) -> bool {
    constraint
        .late_bind_vocab
        .get()
        .map_or_else(
            || constraint.token_bytes_match_vocab(vocab),
            |compiled_vocab| compiled_vocab.same_model_vocab(vocab),
        )
}

impl GrammarBinding<'_> {
    fn bind_target(&mut self, vocab: &Vocab, name: &str) -> Result<()> {
        let compatible = match self {
            Self::Source(_) => return Ok(()),
            Self::Spec(spec) => spec.targets(vocab),
            Self::StaticBorrowed(constraint) => static_constraint_targets(constraint, vocab),
            Self::StaticOwned(constraint) => static_constraint_targets(constraint, vocab),
            Self::DynamicBorrowed(constraint) => constraint.targets_vocab(vocab),
            Self::DynamicOwned(constraint) => constraint.targets_vocab(vocab),
        };
        if compatible {
            Ok(())
        } else {
            Err(Error::Compilation(format!(
                "external grammar {name:?} was built for an incompatible vocabulary",
            )))
        }
    }

    fn compile<'a>(
        &'a self,
        vocab: &Vocab,
        mode: ChildCompileMode,
        allow_open_source_tokens: bool,
    ) -> Result<CompiledChild<'a>> {
        match self {
            Self::Source(grammar) => match mode {
                ChildCompileMode::Static => {
                    if !allow_open_source_tokens {
                        if let Some(name) = grammar.unresolved_token_names()?.first() {
                            return Err(Error::Compilation(format!(
                                "external token {name:?} is unbound in source child; use compile_unlinked() to retain open token slots",
                            )));
                        }
                    }
                    let module = grammar.compile_unlinked(vocab)?;
                    Ok(CompiledChild::StaticOwned(module.materialize(Optimization::Auto)?))
                }
                ChildCompileMode::Dynamic => {
                    let spec = ConstraintSpec::builder(grammar.clone(), vocab)?.build()?;
                    Ok(CompiledChild::DynamicOwned(spec.compile_dynamic()?))
                }
            },
            Self::Spec(spec) => match mode {
                ChildCompileMode::Static => {
                    Ok(CompiledChild::StaticOwned(spec.compile_static_with_trigger_uncached()?))
                }
                ChildCompileMode::Dynamic => {
                    Ok(CompiledChild::DynamicOwned(spec.compile_dynamic()?))
                }
            },
            Self::StaticBorrowed(constraint) => Ok(CompiledChild::StaticBorrowed(constraint)),
            Self::StaticOwned(constraint) => Ok(CompiledChild::StaticBorrowed(constraint)),
            Self::DynamicBorrowed(constraint) => Ok(CompiledChild::DynamicBorrowed(constraint)),
            Self::DynamicOwned(constraint) => Ok(CompiledChild::DynamicBorrowed(constraint)),
        }
    }

    /// One-shot late binding can consume the user's binding. Preserve that
    /// ownership instead of immediately downgrading an owned/Arc constraint to
    /// a borrowed child and deep-cloning it again during preparation.
    fn into_compiled<'a>(
        self,
        vocab: &Vocab,
        mode: ChildCompileMode,
    ) -> Result<CompiledChild<'a>>
    where
        Self: 'a,
    {
        match self {
            Self::Source(grammar) => match mode {
                ChildCompileMode::Static => {
                    if let Some(name) = grammar.unresolved_token_names()?.first() {
                        return Err(Error::Compilation(format!(
                            "external token {name:?} is unbound in source child; use compile_unlinked() to retain open token slots",
                        )));
                    }
                    let module = grammar.compile_unlinked(vocab)?;
                    Ok(CompiledChild::StaticOwned(module.materialize(Optimization::Auto)?))
                }
                ChildCompileMode::Dynamic => {
                    let spec = ConstraintSpec::builder(grammar, vocab)?.build()?;
                    Ok(CompiledChild::DynamicOwned(spec.compile_dynamic()?))
                }
            },
            Self::Spec(spec) => match mode {
                ChildCompileMode::Static => {
                    Ok(CompiledChild::StaticOwned(spec.compile_static_with_trigger_uncached()?))
                }
                ChildCompileMode::Dynamic => {
                    Ok(CompiledChild::DynamicOwned(spec.compile_dynamic()?))
                }
            },
            Self::StaticBorrowed(constraint) => Ok(CompiledChild::StaticBorrowed(constraint)),
            Self::StaticOwned(constraint) => Ok(CompiledChild::StaticOwned(
                Arc::try_unwrap(constraint).unwrap_or_else(|shared| (*shared).clone()),
            )),
            Self::DynamicBorrowed(constraint) => Ok(CompiledChild::DynamicBorrowed(constraint)),
            Self::DynamicOwned(constraint) => Ok(CompiledChild::DynamicOwned(
                Arc::try_unwrap(constraint).unwrap_or_else(|shared| (*shared).clone()),
            )),
        }
    }
}

fn prepare_compiled_children(
    children: Vec<(String, CompiledChild<'_>)>,
    vocab: &Vocab,
    boundary_backend: SegmentedBoundaryBackend,
) -> Result<Vec<(String, Arc<RuntimeConstraint>)>> {
    children
        .into_iter()
        .map(|(name, child)| {
            let mut constraint = collapse_dynamic_alternatives(
                child.into_constraints(),
                vocab,
                boundary_backend,
            )?;
            // Root termination is policy, not an embedded grammar terminal.
            // The native artifact cache already contains body-only bytes.
            constraint.end_tokens = Arc::from([]);
            // `compose_named_children` always uses the segmented linker. Decode
            // composition-only metadata on this first owned child copy so the
            // linker does not clone the whole compiled child a second time
            // merely to materialize the same cold metadata.
            match boundary_backend {
                SegmentedBoundaryBackend::StaticParserDwa => constraint
                    .materialize_composition_metadata_for_compilation()
                    .map_err(Error::Compilation)?,
                SegmentedBoundaryBackend::Dynamic => constraint
                    .materialize_composition_link_metadata_for_compilation()
                    .map_err(Error::Compilation)?,
            }
            Ok((name, Arc::new(constraint)))
        })
        .collect()
}

fn collapse_dynamic_alternatives(
    mut alternatives: Vec<RuntimeConstraint>,
    vocab: &Vocab,
    boundary_backend: SegmentedBoundaryBackend,
) -> Result<RuntimeConstraint> {
    if alternatives.len() == 1 {
        return Ok(alternatives.pop().expect("length checked"));
    }
    if alternatives.is_empty() {
        return Err(Error::Compilation(
            "external grammar realization has no alternatives".to_owned(),
        ));
    }

    let mut source = String::from("glrm 1;\nstart start;\n");
    for index in 0..alternatives.len() {
        source.push_str(&format!("extern grammar alternative_{index};\n"));
    }
    source.push_str("nt start = ");
    for index in 0..alternatives.len() {
        if index != 0 {
            source.push_str(" | ");
        }
        source.push_str(&format!("alternative_{index}"));
    }
    source.push_str(";\n");

    let names = (0..alternatives.len())
        .map(|index| format!("alternative_{index}"))
        .collect::<Vec<_>>();
    let children = names
        .iter()
        .cloned()
        .zip(alternatives.into_iter().map(Arc::new))
        .collect::<Vec<_>>();
    let parent = RuntimeConstraint::from_glrm_grammar_with_subgrammars(&source, &[], vocab)?;
    let mut union = compose_named_children(parent, &children, vocab, boundary_backend)?;
    for slot in &mut union.late_grammar_slots {
        if let Some((alternative, nested)) = slot.name.split_once('.')
            && alternative.starts_with("alternative_")
        {
            slot.name = nested.to_owned();
        }
    }
    Ok(union)
}

fn compose_named_children(
    parent: RuntimeConstraint,
    children: &[(String, Arc<RuntimeConstraint>)],
    vocab: &Vocab,
    boundary_backend: SegmentedBoundaryBackend,
) -> Result<RuntimeConstraint> {
    let bound_names = children
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<BTreeSet<_>>();
    let mut present = Vec::new();
    let mut matching_terminals = Vec::new();
    for (child_index, (name, _)) in children.iter().enumerate() {
        let terminals = parent
            .late_grammar_slots
            .iter()
            .filter(|slot| slot.name == *name)
            .map(|slot| slot.terminal_id)
            .collect::<Vec<_>>();
        if !terminals.is_empty() {
            present.push(child_index);
            matching_terminals.push(terminals);
        }
    }
    if present.is_empty() {
        return Ok(parent);
    }

    let remaining_parent_slots = parent
        .late_grammar_slots
        .iter()
        .filter(|slot| !bound_names.contains(slot.name.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let inputs = present
        .iter()
        .zip(&matching_terminals)
        .map(|(&child_index, terminals)| CompiledSubgrammarInput {
            placeholder_terminal: terminals[0],
            additional_placeholder_terminals: &terminals[1..],
            constraint: children[child_index].1.as_ref(),
        })
        .collect::<Vec<_>>();
    let shared_children = present
        .iter()
        .map(|&child_index| Arc::clone(&children[child_index].1))
        .collect::<Vec<_>>();
    let mut composition = compose_constraints_owned_parent_segmented_shared(
        parent,
        &inputs,
        &shared_children,
        vocab,
        boundary_backend,
    )
    .map_err(Error::Compilation)?;
    composition.constraint.late_grammar_slots = remaining_parent_slots;
    for (component_index, &child_index) in present.iter().enumerate() {
        let terminal_offset = composition.terminal_offsets[component_index + 1];
        let (binding_name, child) = &children[child_index];
        composition.constraint.late_grammar_slots.extend(
            child.late_grammar_slots.iter().map(|slot| crate::runtime::LateGrammarSlot {
                name: format!("{binding_name}.{}", slot.name),
                terminal_id: terminal_offset + slot.terminal_id,
            }),
        );
    }
    if composition
        .constraint
        .sanitize_late_grammar_placeholder_token_domain()
    {
        composition.constraint.rebuild_runtime_caches();
    }
    Ok(composition.constraint)
}

fn compile_static_source(
    grammar: &Grammar<'_>,
    vocab: &Vocab,
    token_bindings: &[(&str, &[u32])],
) -> Result<RuntimeConstraint> {
    match grammar.source {
        GrammarSource::Ebnf(source) => RuntimeConstraint::from_ebnf(source, vocab),
        GrammarSource::Lark(source) => RuntimeConstraint::from_lark(source, vocab),
        GrammarSource::JsonSchema(source) => RuntimeConstraint::from_json_schema(source, vocab),
        GrammarSource::Glrm(source) => RuntimeConstraint::from_glrm_grammar_with_unbound_subgrammars_bindings_and_end_tokens(
            source,
            vocab,
            token_bindings,
            &[],
        ),
    }
}

fn compile_dynamic_source(
    grammar: &Grammar<'_>,
    vocab: &Vocab,
    token_bindings: &[(&str, &[u32])],
) -> Result<DynamicConstraint> {
    match grammar.source {
        GrammarSource::Ebnf(source) => DynamicConstraint::from_ebnf(source, vocab),
        GrammarSource::Lark(source) => DynamicConstraint::from_lark(source, vocab),
        GrammarSource::JsonSchema(source) => DynamicConstraint::from_json_schema(source, vocab),
        GrammarSource::Glrm(source) => DynamicConstraint::from_glrm_grammar_with_bindings_and_end_tokens(
            source,
            vocab,
            token_bindings,
            &[],
        ),
    }
}

#[cfg(any(test, feature = "internal-api"))]
impl RuntimeConstraint {
    /// Compile `grammar` into a [`Constraint`](crate::Constraint) for `vocab`.
    pub fn compile(grammar: Grammar<'_>, vocab: &Vocab) -> Result<Self> {
        ConstraintSpec::builder(grammar, vocab)?.build()?.compile()
    }

    /// Bind one retained `extern grammar` slot using a fully compiled boundary.
    ///
    /// Other named slots remain unresolved and may be bound by later calls.
    /// The compiled parent and the supplied child's masking backends are reused;
    /// only their cross-boundary behavior is compiled here.
    #[allow(private_bounds)]
    pub fn bind_grammar<'a, T>(&self, name: impl AsRef<str>, child: T) -> Result<Self>
    where
        T: IntoGrammarBinding<'a>,
    {
        bind_static_parent_grammar(
            self,
            name.as_ref(),
            child,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
    }

    /// Bind one retained `extern grammar` slot using the dynamic boundary walker.
    ///
    /// Component-local masking remains independently static or dynamic according
    /// to the backend with which each component was supplied.
    #[allow(private_bounds)]
    pub fn bind_grammar_dynamic_boundary<'a, T>(
        &self,
        name: impl AsRef<str>,
        child: T,
    ) -> Result<Self>
    where
        T: IntoGrammarBinding<'a>,
    {
        bind_static_parent_grammar(
            self,
            name.as_ref(),
            child,
            SegmentedBoundaryBackend::Dynamic,
        )
    }

    /// Private compatibility hook for internal benchmark/composition callers.
    /// Public late binding no longer requires the caller to resupply `Vocab`,
    /// but old internal cached-parent probes still use this to eagerly prepare
    /// only the small compiler-facing composition metadata while leaving the
    /// packed runtime automata untouched.
    pub(crate) fn prepare_for_composition_internal(&mut self, vocab: &Vocab) -> Result<()> {
        self.bind_vocab_exact(vocab).map_err(Error::Compilation)?;
        self.materialize_composition_metadata_for_compilation()
            .map_err(Error::Compilation)?;
        crate::compiler::boundary_candidates::persist_boundary_candidate_summary(self, vocab);
        if std::env::var_os("GLRMASK_PREPARE_BOUNDARY_COMPLETION").is_some() {
            crate::compiler::boundary_precomputed_completion::prepare_component(self, vocab)
                .map_err(crate::GlrMaskError::Compilation)?;
        }
        Ok(())
    }
}

#[cfg(any(test, feature = "internal-api"))]
impl DynamicConstraint {
    /// Compile `grammar` into a [`DynamicConstraint`] for `vocab`.
    pub fn compile(grammar: Grammar<'_>, vocab: &Vocab) -> Result<Self> {
        ConstraintSpec::builder(grammar, vocab)?.build()?.compile_dynamic()
    }


    /// Compile `grammar` into a dynamic constraint whose local mask walk uses
    /// one representative per proven vocabulary-equivalence class.
    ///
    /// The complete original vocabulary is still retained for commits,
    /// composition compatibility, and boundary triggers. Consequently a
    /// partition-optimized constraint can be composed with ordinary static or
    /// dynamic constraints without changing the public token coordinate.
    pub fn compile_with_vocab_partition(grammar: Grammar<'_>, vocab: &Vocab) -> Result<Self> {
        if !grammar.bindings.is_empty() {
            return Err(Error::Compilation(
                "compile_with_vocab_partition does not yet compile source-level bound subgrammars; compile the components first and bind the compiled constraints"
                    .to_owned(),
            ));
        }
        match grammar.source {
            GrammarSource::Ebnf(source) => Self::from_ebnf_with_vocab_partition(source, vocab),
            GrammarSource::Lark(source) => Self::from_lark_with_vocab_partition(source, vocab),
            GrammarSource::JsonSchema(source) => {
                Self::from_json_schema_with_vocab_partition(source, vocab)
            }
            GrammarSource::Glrm(source) => {
                Self::from_glrm_grammar_with_vocab_partition(source, vocab)
            }
        }
    }

    /// Bind one retained `extern grammar` slot using a fully compiled boundary.
    ///
    /// Dynamic parent alternatives and dynamic component-local masking remain
    /// dynamic; the boundary choice is independent of component backends.
    #[allow(private_bounds)]
    pub fn bind_grammar<'a, T>(&self, name: impl AsRef<str>, child: T) -> Result<Self>
    where
        T: IntoGrammarBinding<'a>,
    {
        bind_dynamic_parent_grammar(
            self,
            name.as_ref(),
            child,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
    }

    /// Bind one retained `extern grammar` slot using the dynamic boundary walker.
    #[allow(private_bounds)]
    pub fn bind_grammar_dynamic_boundary<'a, T>(
        &self,
        name: impl AsRef<str>,
        child: T,
    ) -> Result<Self>
    where
        T: IntoGrammarBinding<'a>,
    {
        bind_dynamic_parent_grammar(
            self,
            name.as_ref(),
            child,
            SegmentedBoundaryBackend::Dynamic,
        )
    }
}

fn constraint_vocab(constraint: &RuntimeConstraint) -> Vocab {
    constraint
        .late_bind_vocab
        .get_or_init(|| {
            Vocab::new(
                constraint
                    .token_bytes_iter()
                    .map(|(token_id, bytes)| (token_id, bytes.to_vec()))
                    .collect(),
            )
        })
        .clone()
}

fn ensure_runnable_constraint(constraint: &RuntimeConstraint) -> Result<()> {
    if let Some(slot) = constraint.late_grammar_slots.first() {
        return Err(Error::Compilation(format!(
            "external grammar {:?} is still unbound",
            slot.name,
        )));
    }
    Ok(())
}

/// The current bounded static linker requires nonnullable child links. Decide
/// from retained semantic metadata before construction, never by catching a
/// compiler error or silently falling back during mask generation.
fn select_supported_boundary(
    parent: &RuntimeConstraint,
    children: &[(String, Arc<RuntimeConstraint>)],
) -> Result<SegmentedBoundaryBackend> {
    fn requires_dynamic_boundary(constraint: &RuntimeConstraint) -> bool {
        // A FastBuild child can carry exact virtual lexer residuals. Static
        // boundary shards cannot execute those coordinates, even when the
        // child is non-nullable. Select the supported backend from metadata
        // before linking rather than asking callers to know internal engines.
        constraint.tokenizer.has_virtual_residual_runtime()
            || constraint.static_dynamic_overlay.as_ref().is_some_and(|overlay| {
                overlay.segmented_parser_links.iter().any(|link| link.child_start_nullable)
                    || overlay.segmented_parser_components.iter()
                        .any(|component| requires_dynamic_boundary(&component.constraint))
            })
    }
    if requires_dynamic_boundary(parent) {
        return Ok(SegmentedBoundaryBackend::Dynamic);
    }
    for (_, child) in children {
        if child.composition_start_nullable().map_err(Error::Compilation)?
            || requires_dynamic_boundary(child)
        {
            return Ok(SegmentedBoundaryBackend::Dynamic);
        }
    }
    Ok(SegmentedBoundaryBackend::StaticParserDwa)
}

fn compile_exact_token_adapter(vocab: &Vocab, token_ids: &[u32]) -> Result<RuntimeConstraint> {
    if token_ids.is_empty() {
        return Err(Error::Compilation(
            "exact-token binding must contain at least one token ID".to_owned(),
        ));
    }
    use crate::grammar::ast::{GrammarExpr, NamedGrammar, NamedRule};
    let named = NamedGrammar {
        start: "__exact_tokens".to_owned(),
        rules: vec![NamedRule {
            name: "__exact_tokens".to_owned(),
            expr: crate::import::choice_or_single(
                token_ids.iter().copied().map(GrammarExpr::SpecialToken).collect(),
            ),
            is_terminal: false,
            is_internal: false,
        }],
        ignore: None,
        lexer_partitions: BTreeMap::new(),
        lexer_literal_partitions: BTreeMap::new(),
        default_lexer_partition: None,
    };
    crate::import::compile_from_named_grammar(
        named,
        vocab,
        "exact_token_binding",
        crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        &[],
    )
}

fn boundary_for_optimization(
    parent: &RuntimeConstraint,
    children: &[(String, Arc<RuntimeConstraint>)],
    optimization: Optimization,
) -> Result<SegmentedBoundaryBackend> {
    match optimization {
        Optimization::FastBuild => Ok(SegmentedBoundaryBackend::Dynamic),
        Optimization::Auto | Optimization::FastRuntime => {
            select_supported_boundary(parent, children)
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ModuleArtifactManifest {
    exact_only_token_ids: Vec<u32>,
    token_slots: Vec<String>,
    bindings: Vec<(String, ModuleBindingArtifact)>,
}

#[derive(serde::Serialize, serde::Deserialize)]
enum ModuleBindingArtifact {
    Module(Vec<u8>),
    Constraint(Vec<u8>),
    ExactTokens(Vec<u32>),
}

const MODULE_MAGIC: &[u8; 8] = b"GLRMOD03";
const MODULE_HEADER_LEN: usize = 16;

impl UnlinkedConstraint {
    fn direct_slot_names(&self) -> BTreeSet<&str> {
        self.inner
            .late_grammar_slots
            .iter()
            .map(|slot| slot.name.as_str())
            .collect()
    }

    fn open_token_names(&self) -> Result<BTreeSet<String>> {
        let mut names = self
            .token_slots
            .iter()
            .filter(|name| !self.bindings.contains_key(*name))
            .cloned()
            .collect::<BTreeSet<_>>();
        for (name, binding) in &self.bindings {
            if let ModuleBinding::Module(child) = binding {
                names.extend(
                    child
                        .open_token_names()?
                        .into_iter()
                        .map(|nested| format!("{name}.{nested}")),
                );
            }
        }
        Ok(names)
    }

    fn first_open_slot(&self) -> Result<Option<(String, ExternKind)>> {
        let direct_slots = self.direct_slot_names();
        for name in direct_slots {
            if !self.bindings.contains_key(name) {
                let kind = if self.token_slots.contains(name) {
                    ExternKind::Token
                } else {
                    ExternKind::Grammar
                };
                return Ok(Some((name.to_owned(), kind)));
            }
        }
        for (name, binding) in &self.bindings {
            if let ModuleBinding::Module(child) = binding
                && let Some((nested, kind)) = child.first_open_slot()?
            {
                return Ok(Some((format!("{name}.{nested}"), kind)));
            }
        }
        Ok(None)
    }

    fn targets_vocab(&self, vocab: &Vocab) -> bool {
        static_constraint_targets(self.inner.as_ref(), vocab)
            && self.bindings.values().all(|binding| match binding {
                ModuleBinding::Module(child) => child.targets_vocab(vocab),
                ModuleBinding::Constraint(child) => {
                    static_constraint_targets(child.as_ref(), vocab)
                }
                ModuleBinding::ExactTokens(ids) => {
                    ids.iter().all(|&id| vocab.contains_exact_token_id(id))
                }
            })
    }

    fn validate_slot_manifest(&self) -> Result<()> {
        if !self.inner.end_tokens.is_empty() {
            return Err(Error::Serialization(
                "unlinked-constraint artifact contains final-root policy".to_owned(),
            ));
        }
        let direct_slots = self.direct_slot_names();
        for name in &self.token_slots {
            if name.is_empty() || !direct_slots.contains(name.as_str()) {
                return Err(Error::Serialization(format!(
                    "unlinked-constraint token slot {name:?} has no compiled linker slot",
                )));
            }
        }
        let vocab = constraint_vocab(self.inner.as_ref());
        for (name, binding) in &self.bindings {
            if name.is_empty() || !direct_slots.contains(name.as_str()) {
                return Err(Error::Serialization(format!(
                    "unlinked-constraint binding {name:?} has no compiled linker slot",
                )));
            }
            let is_token = self.token_slots.contains(name);
            match binding {
                ModuleBinding::ExactTokens(ids) => {
                    if !is_token || ids.is_empty() {
                        return Err(Error::Serialization(format!(
                            "unlinked-constraint binding {name:?} has the wrong slot kind or no token IDs",
                        )));
                    }
                    let mut previous = None;
                    for &id in ids.iter() {
                        if !vocab.contains_exact_token_id(id)
                            || previous.is_some_and(|value| value >= id)
                        {
                            return Err(Error::Serialization(format!(
                                "unlinked-constraint exact-token binding {name:?} is invalid for its vocabulary",
                            )));
                        }
                        previous = Some(id);
                    }
                }
                ModuleBinding::Module(child) => {
                    if is_token || !child.targets_vocab(&vocab) {
                        return Err(Error::Serialization(format!(
                            "unlinked-constraint grammar binding {name:?} has incompatible kind or vocabulary",
                        )));
                    }
                    child.validate_slot_manifest()?;
                }
                ModuleBinding::Constraint(child) => {
                    if is_token || !static_constraint_targets(child.as_ref(), &vocab) {
                        return Err(Error::Serialization(format!(
                            "unlinked-constraint grammar binding {name:?} has incompatible kind or vocabulary",
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    fn require_direct_slot_kind(&self, name: &str, expected: ExternKind) -> Result<()> {
        if !self
            .inner
            .late_grammar_slots
            .iter()
            .any(|slot| slot.name == name)
        {
            return Err(Error::Compilation(format!(
                "no unresolved external {} named {name:?} is present in this unlinked constraint",
                expected.name(),
            )));
        }
        if self.bindings.contains_key(name) {
            return Err(Error::Compilation(format!(
                "external {name:?} was bound more than once",
            )));
        }
        let is_token = self.token_slots.contains(name);
        if is_token != matches!(expected, ExternKind::Token) {
            return Err(Error::Compilation(format!(
                "external {name:?} has kind {}, not {}",
                expected.opposite_name(),
                expected.name(),
            )));
        }
        Ok(())
    }

    fn bind_value(&self, name: &str, value: UnlinkedValue<'_>) -> Result<Self> {
        if let Some((head, tail)) = name.split_once('.') {
            let Some(ModuleBinding::Module(child)) = self.bindings.get(head) else {
                return Err(Error::Compilation(format!(
                    "external grammar {head:?} is not bound to an open compiled child",
                )));
            };
            let mut next = self.clone();
            let child = child.bind_value(tail, value)?;
            next.bindings
                .insert(head.to_owned(), ModuleBinding::Module(Box::new(child)));
            next.validate_slot_manifest()?;
            return Ok(next);
        }

        let vocab = constraint_vocab(self.inner.as_ref());
        let binding = match value {
            UnlinkedValue::ExactToken(token) => {
                self.require_direct_slot_kind(name, ExternKind::Token)?;
                if !token.targets(&vocab) {
                    return Err(Error::Compilation(format!(
                        "external token {name:?} was built for an incompatible vocabulary",
                    )));
                }
                ModuleBinding::ExactTokens(Arc::from([token.id()]))
            }
            UnlinkedValue::ExactTokens(tokens) => {
                self.require_direct_slot_kind(name, ExternKind::Token)?;
                if !tokens.targets(&vocab) {
                    return Err(Error::Compilation(format!(
                        "external token {name:?} was built for an incompatible vocabulary",
                    )));
                }
                ModuleBinding::ExactTokens(Arc::from(tokens.ids()))
            }
            UnlinkedValue::StaticBorrowed(child) => {
                self.require_direct_slot_kind(name, ExternKind::Grammar)?;
                if !static_constraint_targets(child, &vocab) {
                    return Err(Error::Compilation(format!(
                        "external grammar {name:?} was built for an incompatible vocabulary",
                    )));
                }
                ModuleBinding::Constraint(Arc::new(child.clone()))
            }
            UnlinkedValue::StaticOwned(child) => {
                self.require_direct_slot_kind(name, ExternKind::Grammar)?;
                if !static_constraint_targets(child.as_ref(), &vocab) {
                    return Err(Error::Compilation(format!(
                        "external grammar {name:?} was built for an incompatible vocabulary",
                    )));
                }
                ModuleBinding::Constraint(child)
            }
        };
        let mut next = self.clone();
        next.bindings.insert(name.to_owned(), binding);
        next.validate_slot_manifest()?;
        Ok(next)
    }

    /// Bind one compiled [`crate::Constraint`] child or vocabulary-qualified
    /// exact-token value.
    ///
    /// Binding is intentionally cheap and immutable: no boundary is compiled
    /// here. The final link operation chooses and materializes composition.
    #[allow(private_bounds)]
    pub fn bind<'a, T>(&self, name: impl AsRef<str>, value: T) -> Result<Self>
    where
        T: IntoUnlinkedValue<'a>,
    {
        self.bind_value(name.as_ref(), value.into_unlinked_value())
    }

    fn materialize(&self, optimization: Optimization) -> Result<RuntimeConstraint> {
        self.validate_slot_manifest()?;
        if self.bindings.is_empty() {
            return Ok(self.inner.as_ref().clone());
        }
        let vocab = constraint_vocab(self.inner.as_ref());
        let binding_entries = self.bindings.iter().collect::<Vec<_>>();
        let raw_children = binding_entries
            .par_iter()
            .map(|(name, binding)| {
                let mut child = match binding {
                    ModuleBinding::Module(module) => module.materialize(optimization)?,
                    ModuleBinding::Constraint(constraint) => constraint.as_ref().clone(),
                    ModuleBinding::ExactTokens(ids) => {
                        compile_exact_token_adapter(&vocab, ids.as_ref())?
                    }
                };
                // Standalone termination is root policy, never an embedded body.
                child.end_tokens = Arc::from([]);
                Ok(((*name).clone(), child))
            })
            .collect::<Result<Vec<_>>>()?;
        let probe_children = raw_children
            .iter()
            .map(|(name, child)| (name.clone(), Arc::new(child.clone())))
            .collect::<Vec<_>>();
        let backend = boundary_for_optimization(
            self.inner.as_ref(),
            &probe_children,
            optimization,
        )?;
        let prepared = prepare_compiled_children(
            raw_children
                .into_iter()
                .map(|(name, child)| (name, CompiledChild::StaticOwned(child)))
                .collect(),
            &vocab,
            backend,
        )?;
        compose_named_children(
            self.inner.as_ref().clone(),
            &prepared,
            &vocab,
            backend,
        )
    }

    /// Link a fully bound artifact into a runnable constraint.
    pub fn link(&self) -> Result<RuntimeConstraint> {
        self.link_with(BuildOptions::default())
    }

    /// Link a fully bound artifact with final build options.
    pub fn link_with(&self, options: BuildOptions) -> Result<RuntimeConstraint> {
        if options.parser_backend == ParserBackend::TemplateDfa && !self.bindings.is_empty() {
            return Err(Error::Compilation("template-parser component composition is not implemented; no LR fallback is permitted".into()));
        }
        if let Some((name, kind)) = self.first_open_slot()? {
            return Err(Error::Compilation(format!(
                "external {} {name:?} is still unbound",
                kind.name(),
            )));
        }
        let mut constraint = self.materialize(options.optimization_value())?;
        ensure_runnable_constraint(&constraint)?;
        if options.parser_backend == ParserBackend::TemplateDfa {
            constraint.install_template_parser()?;
        }
        constraint.with_end_tokens(options.end_token_ids())
    }

    fn artifact_manifest(&self) -> ModuleArtifactManifest {
        let bindings = self
            .bindings
            .iter()
            .map(|(name, binding)| {
                let artifact = match binding {
                    ModuleBinding::Module(module) => ModuleBindingArtifact::Module(module.save()),
                    ModuleBinding::Constraint(constraint) => {
                        ModuleBindingArtifact::Constraint(constraint.save())
                    }
                    ModuleBinding::ExactTokens(ids) => {
                        ModuleBindingArtifact::ExactTokens(ids.to_vec())
                    }
                };
                (name.clone(), artifact)
            })
            .collect();
        ModuleArtifactManifest {
            exact_only_token_ids: constraint_vocab(self.inner.as_ref())
                .exact_only_token_ids()
                .collect(),
            token_slots: self.token_slots.iter().cloned().collect(),
            bindings,
        }
    }

    /// Serialize compiled local machinery and deferred bindings as bytes.
    pub fn save(&self) -> Vec<u8> {
        let manifest = bincode::serialize(&self.artifact_manifest())
            .expect("in-memory unlinked-constraint manifest is serializable");
        // Open-module vocabulary metadata lives in this manifest. Keep the
        // embedded constraint bytes as the raw compiled body rather than
        // wrapping them in the closed-root vocabulary/policy envelope.
        let body = self.inner.save_body();
        let mut bytes = Vec::with_capacity(MODULE_HEADER_LEN + manifest.len() + body.len());
        bytes.extend_from_slice(MODULE_MAGIC);
        bytes.extend_from_slice(&(manifest.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&manifest);
        bytes.extend_from_slice(&body);
        bytes
    }

    fn from_artifact_parts(
        mut inner: RuntimeConstraint,
        manifest: ModuleArtifactManifest,
    ) -> Result<Self> {
        let mut previous = None;
        for &id in &manifest.exact_only_token_ids {
            if inner.token_bytes_for_id(id).is_some()
                || previous.is_some_and(|value| value >= id)
            {
                return Err(Error::Serialization(
                    "unlinked-constraint exact-only token domain is invalid".to_owned(),
                ));
            }
            previous = Some(id);
        }
        let vocab = Vocab::new_with_exact_token_ids(
            inner
                .token_bytes_iter()
                .map(|(token_id, bytes)| (token_id, bytes.to_vec()))
                .collect(),
            manifest.exact_only_token_ids.iter().copied(),
        );
        inner
            .bind_vocab_exact(&vocab)
            .map_err(Error::Serialization)?;
        let mut token_slots = BTreeSet::new();
        for name in manifest.token_slots {
            if name.is_empty() || !token_slots.insert(name) {
                return Err(Error::Serialization(
                    "empty or duplicate unlinked-constraint token slot".to_owned(),
                ));
            }
        }
        let mut bindings = BTreeMap::new();
        for (name, artifact) in manifest.bindings {
            if name.is_empty() || bindings.contains_key(&name) {
                return Err(Error::Serialization(
                    "empty or duplicate unlinked-constraint binding".to_owned(),
                ));
            }
            let binding = match artifact {
                ModuleBindingArtifact::Module(bytes) => {
                    ModuleBinding::Module(Box::new(Self::load(bytes)?))
                }
                ModuleBindingArtifact::Constraint(bytes) => {
                    ModuleBinding::Constraint(Arc::new(RuntimeConstraint::load(bytes)?))
                }
                ModuleBindingArtifact::ExactTokens(mut ids) => {
                    ids.sort_unstable();
                    if ids.windows(2).any(|pair| pair[0] == pair[1]) {
                        return Err(Error::Serialization(
                            "unlinked-constraint exact-token binding contains duplicate IDs".to_owned(),
                        ));
                    }
                    ModuleBinding::ExactTokens(Arc::from(ids))
                }
            };
            bindings.insert(name, binding);
        }
        let module = Self {
            inner: Arc::new(inner),
            token_slots,
            bindings,
        };
        module.validate_slot_manifest()?;
        Ok(module)
    }

    /// Load an unlinked constraint, retaining deferred bindings and open slots.
    pub fn load<'b>(bytes: impl Into<std::borrow::Cow<'b, [u8]>>) -> Result<Self> {
        use bincode::Options;
        let bytes = bytes.into();
        let data = bytes.as_ref();
        if data.len() < MODULE_HEADER_LEN || !data.starts_with(MODULE_MAGIC) {
            return Err(Error::Serialization("invalid unlinked-constraint artifact header".to_owned()));
        }
        let manifest_len = u64::from_le_bytes(data[8..16].try_into().expect("header length checked"));
        let body_start = usize::try_from(manifest_len)
            .ok()
            .and_then(|len| MODULE_HEADER_LEN.checked_add(len))
            .filter(|&end| end < data.len())
            .ok_or_else(|| Error::Serialization("invalid unlinked-constraint manifest length".to_owned()))?;
        let manifest: ModuleArtifactManifest = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_limit(manifest_len)
            .reject_trailing_bytes()
            .deserialize(&data[MODULE_HEADER_LEN..body_start])
            .map_err(|error| Error::Serialization(format!("invalid unlinked-constraint manifest: {error}")))?;
        let inner = match bytes {
            std::borrow::Cow::Owned(mut bytes) => {
                RuntimeConstraint::load_body_artifact(bytes.split_off(body_start))?
            }
            std::borrow::Cow::Borrowed(bytes) => {
                RuntimeConstraint::load_body_artifact(&bytes[body_start..])?
            }
        };
        Self::from_artifact_parts(inner, manifest)
    }

    fn bind_vocab_recursive(&mut self, vocab: &Vocab) -> Result<()> {
        Arc::make_mut(&mut self.inner)
            .bind_vocab_exact(vocab)
            .map_err(Error::Serialization)?;
        for binding in self.bindings.values_mut() {
            match binding {
                ModuleBinding::Module(module) => module.bind_vocab_recursive(vocab)?,
                ModuleBinding::Constraint(constraint) => Arc::make_mut(constraint)
                    .bind_vocab_exact(vocab)
                    .map_err(Error::Serialization)?,
                ModuleBinding::ExactTokens(ids) => {
                    if ids.iter().any(|&id| !vocab.contains_exact_token_id(id)) {
                        return Err(Error::Serialization(
                            "unlinked-constraint exact-token binding is incompatible with supplied vocabulary"
                                .to_owned(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Load a module while sharing an already-existing exact vocabulary.
    pub fn load_with_vocab<'b>(
        bytes: impl Into<std::borrow::Cow<'b, [u8]>>,
        vocab: &Vocab,
    ) -> Result<Self> {
        let mut module = Self::load(bytes)?;
        module.bind_vocab_recursive(vocab)?;
        module.validate_slot_manifest()?;
        Ok(module)
    }
}

const FIRST_SAVE_TEMPLATE_STATE_PRIME_THRESHOLD: usize = 100_000;

fn should_prime_first_save_artifact(constraint: &RuntimeConstraint) -> bool {
    if retains_dynamic_component(constraint) {
        return false;
    }
    if constraint.tokenizer.has_virtual_residual_runtime() {
        return true;
    }
    let mut template_states = 0usize;
    for template in constraint
        .composition_parser_templates_by_terminal
        .iter()
        .flatten()
    {
        template_states = template_states.saturating_add(template.states.len());
        if template_states >= FIRST_SAVE_TEMPLATE_STATE_PRIME_THRESHOLD {
            return true;
        }
    }
    false
}

fn retains_dynamic_component(constraint: &RuntimeConstraint) -> bool {
    if constraint.uses_dynamic_runtime() {
        return true;
    }
    constraint
        .static_dynamic_overlay
        .as_ref()
        .is_some_and(|overlay| {
            overlay
                .segmented_parser_components
                .iter()
                .any(|component| retains_dynamic_component(component.constraint.as_ref()))
        })
}

fn require_late_grammar_slot(
    parents: &[RuntimeConstraint],
    name: &str,
) -> Result<()> {
    if parents.iter().any(|parent| {
        parent
            .late_grammar_slots
            .iter()
            .any(|slot| slot.name == name)
    }) {
        return Ok(());
    }
    Err(Error::Compilation(format!(
        "compiled constraint has no unresolved external grammar named {name:?}",
    )))
}

fn prepare_late_child<'a, T>(
    name: &str,
    child: T,
    vocab: &Vocab,
    boundary_backend: SegmentedBoundaryBackend,
) -> Result<Vec<(String, Arc<RuntimeConstraint>)>>
where
    T: IntoGrammarBinding<'a>,
{
    let mut binding = child.into_grammar_binding();
    binding.bind_target(vocab, name)?;
    let mode = match boundary_backend {
        SegmentedBoundaryBackend::StaticParserDwa => ChildCompileMode::Static,
        SegmentedBoundaryBackend::Dynamic => ChildCompileMode::Dynamic,
    };
    let compiled = binding.into_compiled(vocab, mode)?;
    prepare_compiled_children(
        vec![(name.to_owned(), compiled)],
        vocab,
        boundary_backend,
    )
}

fn bind_static_parent_grammar<'a, T>(
    parent: &RuntimeConstraint,
    name: &str,
    child: T,
    boundary_backend: SegmentedBoundaryBackend,
) -> Result<RuntimeConstraint>
where
    T: IntoGrammarBinding<'a>,
{
    let profile = std::env::var_os("GLRMASK_PROFILE_PUBLIC_BIND").is_some();
    let total_started = Instant::now();
    let phase = Instant::now();
    require_late_grammar_slot(std::slice::from_ref(parent), name)?;
    let require_ms = phase.elapsed().as_secs_f64() * 1000.0;

    let phase = Instant::now();
    let vocab = constraint_vocab(parent);
    let vocab_ms = phase.elapsed().as_secs_f64() * 1000.0;

    let phase = Instant::now();
    let children = prepare_late_child(name, child, &vocab, boundary_backend)?;
    let child_ms = phase.elapsed().as_secs_f64() * 1000.0;

    let phase = Instant::now();
    let parent = parent.clone();
    let parent_clone_ms = phase.elapsed().as_secs_f64() * 1000.0;

    let phase = Instant::now();
    let result = compose_named_children(parent, &children, &vocab, boundary_backend)?;
    let compose_ms = phase.elapsed().as_secs_f64() * 1000.0;

    // Late binding is a construction operation, not an implicit persistence
    // request. Serializing an all-static result here made bind latency include
    // the full first-save cost even when the caller never saves the result.
    // `Constraint::save()` still installs/reuses the serialized artifact cache
    // when persistence is actually requested; ordinary `Constraint::compile()`
    // keeps its existing eager first-save priming policy.
    let cache_save_ms = 0.0;
    if profile {
        eprintln!(
            "[glrmask/profile][public_bind_static_parent] require_ms={require_ms:.3} vocab_ms={vocab_ms:.3} child_ms={child_ms:.3} parent_clone_ms={parent_clone_ms:.3} compose_ms={compose_ms:.3} cache_save_ms={cache_save_ms:.3} total_ms={:.3}",
            total_started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(result)
}

fn bind_dynamic_parent_grammar<'a, T>(
    parent: &DynamicConstraint,
    name: &str,
    child: T,
    boundary_backend: SegmentedBoundaryBackend,
) -> Result<DynamicConstraint>
where
    T: IntoGrammarBinding<'a>,
{
    let parents = parent.clone_constraints();
    require_late_grammar_slot(&parents, name)?;
    parents.first().ok_or_else(|| {
        Error::Compilation("dynamic parent has no alternatives".to_owned())
    })?;
    // Cache on the retained parent, not a temporary clone discarded after this
    // bind. Loaded constraints must reconstruct the vocabulary at most once.
    let vocab = constraint_vocab(&parent.inner);
    if !parents
        .iter()
        .all(|alternative| static_constraint_targets(alternative, &vocab))
    {
        return Err(Error::Compilation(
            "dynamic parent alternatives target incompatible vocabularies".to_owned(),
        ));
    }
    let children = prepare_late_child(name, child, &vocab, boundary_backend)?;
    let alternatives = parents
        .into_iter()
        .map(|alternative| {
            compose_named_children(alternative, &children, &vocab, boundary_backend)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(DynamicConstraint::from_constraints(alternatives))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::tokenizer::Lexer;

    fn token_allowed(mask: &[u32], token_id: u32) -> bool {
        let word = token_id as usize / 32;
        let bit = token_id % 32;
        mask.get(word)
            .is_some_and(|bits| bits & (1u32 << bit) != 0)
    }

    #[test]
    fn vocab_partition_covers_sparse_vocab_and_merges_identical_tokens() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (3, b"a".to_vec()),
            (7, b"b".to_vec()),
            (11, b"ab".to_vec()),
            (67, b"a".to_vec()),
        ]);
        let partition = VocabPartition::compile(Grammar::ebnf(r#"start ::= "a"+"#), &vocab)
            .unwrap();

        assert_eq!(partition.class_of(1), None);
        for token in [0, 3, 7, 11, 67] {
            assert!(partition.class_of(token).is_some(), "token {token} is unmapped");
        }
        assert_eq!(partition.class_of(0), partition.class_of(3));
        assert_eq!(partition.class_of(0), partition.class_of(67));
        let mut covered = partition.classes().iter().flatten().copied().collect::<Vec<_>>();
        covered.sort_unstable();
        assert_eq!(covered, vec![0, 3, 7, 11, 67]);
        for class in 0..partition.num_classes() as u32 {
            let representative = partition.representative(class).unwrap();
            assert_eq!(partition.class_of(representative), Some(class));
        }

        assert_eq!(partition.internal_mask_len(), partition.num_classes().div_ceil(64));
        assert_eq!(partition.original_mask_len(), 3);

        let class = partition.class_of(0).unwrap() as usize;
        let mut internal = vec![0u64; partition.internal_mask_len()];
        internal[class / 64] |= 1u64 << (class % 64);
        let expanded = partition.expand_mask(&internal);
        assert_ne!(expanded[0] & (1u32 << 0), 0);
        assert_ne!(expanded[0] & (1u32 << 3), 0);
        assert_eq!(expanded[0] & (1u32 << 1), 0);
        assert_ne!(expanded[2] & (1u32 << 3), 0);

        let mut reused = vec![u32::MAX; partition.original_mask_len() + 2];
        partition.fill_expanded_mask(&internal, &mut reused);
        assert_eq!(&reused[..partition.original_mask_len()], expanded.as_slice());
        assert_eq!(&reused[partition.original_mask_len()..], &[0, 0]);
    }

    #[test]
    fn vocab_partition_expands_multiple_classes_and_ignores_high_internal_bits() {
        let partition = VocabPartition {
            original_to_class: vec![0, u32::MAX, 1, 0, u32::MAX, 2, 2, u32::MAX, 1],
            classes: vec![vec![0, 3], vec![2, 8], vec![5, 6]],
            representatives: vec![0, 2, 5],
            class_output_masks: vec![
                vec![(0, (1u32 << 0) | (1u32 << 3))],
                vec![(0, (1u32 << 2) | (1u32 << 8))],
                vec![(0, (1u32 << 5) | (1u32 << 6))],
            ],
        };

        let expanded = partition.expand_mask(&[(1u64 << 0) | (1u64 << 2) | (1u64 << 63)]);
        assert_eq!(expanded, vec![(1u32 << 0) | (1u32 << 3) | (1u32 << 5) | (1u32 << 6)]);
    }

    #[test]
    fn vocab_partition_supports_json_schema_frontend() {
        let vocab = Vocab::new(vec![
            (0, b"null".to_vec()),
            (1, b"true".to_vec()),
            (2, b"false".to_vec()),
            (3, b"0".to_vec()),
            (4, b"x".to_vec()),
        ]);
        let partition = VocabPartition::compile(
            Grammar::json_schema(r#"{"type":["null","boolean"]}"#),
            &vocab,
        )
        .unwrap();
        assert!(partition.num_classes() > 0);
        assert!(partition.num_classes() <= vocab.len());
    }

    #[test]
    fn explicit_vocab_partition_strategies_are_conservative() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"ba".to_vec()),
            (4, b"aa".to_vec()),
            (5, b"bb".to_vec()),
        ]);
        let source = r#"start ::= ("a" | "b")+"#;
        let compact = VocabPartition::compile_with_strategy(
            Grammar::ebnf(source),
            &vocab,
            VocabPartitionStrategy::Compact,
        )
        .unwrap();
        let dedicated = VocabPartition::compile_with_strategy(
            Grammar::ebnf(source),
            &vocab,
            VocabPartitionStrategy::Dedicated,
        )
        .unwrap();

        for class in dedicated.classes() {
            let mut compact_class = None;
            for &token_id in class {
                let current = compact.class_of(token_id).unwrap();
                assert!(compact_class.is_none_or(|expected| expected == current));
                compact_class = Some(current);
            }
        }
    }
    #[test]
    fn vocab_partition_rejects_bound_subgrammar_instead_of_ignoring_it() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec())]);
        let grammar = Grammar::glrm(
            "glrm 1; extern grammar child; start root; nt root = child;",
        )
        .bind_grammar("child", Grammar::ebnf(r#"start ::= "a""#))
        .unwrap();
        let error = VocabPartition::compile(grammar, &vocab).unwrap_err();
        assert!(error.to_string().contains("bound grammar values"));
    }

    #[test]
    fn partition_optimized_dynamic_mask_matches_ordinary_dynamic() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"bb".to_vec()),
            (4, b"ab".to_vec()),
            (5, b"ba".to_vec()),
            (6, b"x".to_vec()),
        ]);
        let grammar_source = "start ::= [ab]+".to_owned();
        let grammar = Grammar::ebnf(&grammar_source);
        let partition = VocabPartition::compile(grammar.clone(), &vocab).unwrap();
        assert_eq!(partition.class_of(0), partition.class_of(1));
        assert!(partition.num_classes() < vocab.len());

        let ordinary = DynamicConstraint::compile(grammar.clone(), &vocab).unwrap();
        let optimized =
            DynamicConstraint::compile_with_vocab_partition(grammar, &vocab).unwrap();

        assert_eq!(optimized.inner.token_bytes_count(), vocab.len());
        assert!(
            optimized
                .inner
                .dynamic_mask_vocab_for_runtime()
                .is_grammar_quotiented(),
            "O2 must not silently fall back to ordinary dynamic for tiny grammars",
        );
        assert!(
            optimized
                .inner
                .dynamic_mask_vocab_for_runtime()
                .canonical_token_count()
                < ordinary
                    .inner
                    .dynamic_mask_vocab_for_runtime()
                    .canonical_token_count(),
            "optimized dynamic runtime did not reduce the mask vocabulary",
        );

        let ordinary_start = ordinary.start();
        let optimized_start = optimized.start();
        assert_eq!(ordinary_start.mask(), optimized_start.mask());

        for token in [0u32, 1, 2, 3, 4, 5] {
            let mut ordinary_state = ordinary.start();
            let mut optimized_state = optimized.start();
            ordinary_state.commit_token(token).unwrap();
            optimized_state.commit_token(token).unwrap();
            assert_eq!(ordinary_state.is_accepting(), optimized_state.is_accepting());
            assert_eq!(ordinary_state.mask(), optimized_state.mask(), "token {token}");
        }
    }

    #[test]
    fn singleton_literal_partition_matches_dynamic_masks_exactly() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"{".to_vec()),
            (2, b"}".to_vec()),
            (3, b"{}".to_vec()),
            (4, b"{}x".to_vec()),
            (5, b"{".to_vec()),
            (6, b"x{}".to_vec()),
        ]);
        let grammar_source = r#"start ::= "{" "}""#.to_owned();
        let grammar = Grammar::ebnf(&grammar_source);
        let partition = VocabPartition::compile(grammar.clone(), &vocab).unwrap();

        assert_eq!(partition.num_classes(), 4);
        assert_eq!(partition.class_of(0), partition.class_of(4));
        assert_eq!(partition.class_of(0), partition.class_of(6));
        assert_eq!(partition.class_of(1), partition.class_of(5));
        assert_ne!(partition.class_of(1), partition.class_of(2));
        assert_ne!(partition.class_of(1), partition.class_of(3));
        assert_ne!(partition.class_of(2), partition.class_of(3));

        let ordinary = DynamicConstraint::compile(grammar.clone(), &vocab).unwrap();
        let optimized =
            DynamicConstraint::compile_with_vocab_partition(grammar, &vocab).unwrap();

        let ordinary_start = ordinary.start();
        let optimized_start = optimized.start();
        assert_eq!(ordinary_start.mask(), optimized_start.mask());
        for token in [0u32, 1, 2, 3, 4, 5, 6] {
            assert_eq!(
                token_allowed(&ordinary_start.mask(), token),
                token_allowed(&optimized_start.mask(), token),
                "start mask differs for token {token}",
            );
        }

        for open in [1u32, 5] {
            let mut ordinary_state = ordinary.start();
            let mut optimized_state = optimized.start();
            ordinary_state.commit_token(open).unwrap();
            optimized_state.commit_token(open).unwrap();
            assert_eq!(ordinary_state.is_accepting(), optimized_state.is_accepting());
            assert_eq!(ordinary_state.mask(), optimized_state.mask());
        }

        let mut ordinary_done = ordinary.start();
        let mut optimized_done = optimized.start();
        ordinary_done.commit_token(3).unwrap();
        optimized_done.commit_token(3).unwrap();
        assert!(ordinary_done.is_accepting());
        assert!(optimized_done.is_accepting());
        assert_eq!(ordinary_done.mask(), optimized_done.mask());

        // The production singleton path uses the flat alias representation and
        // deliberately defers its first transfer-artifact serialization. Verify
        // that both survive the real external-vocab save/load boundary.
        let saved = optimized.save_with_external_vocab();
        let loaded = DynamicConstraint::load_with_vocab(&saved, &vocab).unwrap();
        assert_eq!(loaded.start().mask(), optimized.start().mask());
        let mut loaded_done = loaded.start();
        loaded_done.commit_token(3).unwrap();
        assert!(loaded_done.is_accepting());
    }

    #[test]
    fn singleton_literal_partition_matches_all_reachable_cross_terminal_prefixes() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (2, b"b".to_vec()),
            (5, b"c".to_vec()),
            (9, b"d".to_vec()),
            (12, b"ab".to_vec()),
            (18, b"bc".to_vec()),
            (24, b"cd".to_vec()),
            (31, b"abc".to_vec()),
            (37, b"bcd".to_vec()),
            (44, b"abcd".to_vec()),
            (51, b"x".to_vec()),
            (58, b"bd".to_vec()),
            (63, b"abcx".to_vec()),
            (71, b"bc".to_vec()),
        ]);
        let token_ids = vocab.iter().map(|(token_id, _)| token_id).collect::<Vec<_>>();
        let grammar_source = r#"start ::= "ab" "cd""#.to_owned();
        let grammar = Grammar::ebnf(&grammar_source);

        // Compile ordinary first so the optimized compile also exercises the
        // cached byte-order lookup used by prepared production vocabularies.
        let ordinary = DynamicConstraint::compile(grammar.clone(), &vocab).unwrap();
        let optimized =
            DynamicConstraint::compile_with_vocab_partition(grammar, &vocab).unwrap();

        let mut pending = vec![Vec::<u32>::new()];
        let mut seen = std::collections::BTreeSet::<Vec<u32>>::new();
        seen.insert(Vec::new());
        while let Some(prefix) = pending.pop() {
            let mut ordinary_state = ordinary.start();
            let mut optimized_state = optimized.start();
            for &token in &prefix {
                ordinary_state.commit_token(token).unwrap();
                optimized_state.commit_token(token).unwrap();
            }

            assert_eq!(
                ordinary_state.is_accepting(),
                optimized_state.is_accepting(),
                "acceptance differs after token prefix {prefix:?}",
            );
            let ordinary_mask = ordinary_state.mask();
            let optimized_mask = optimized_state.mask();
            assert_eq!(
                ordinary_mask, optimized_mask,
                "mask differs after token prefix {prefix:?}",
            );

            for &token in &token_ids {
                let ordinary_allowed = token_allowed(&ordinary_mask, token);
                let optimized_allowed = token_allowed(&optimized_mask, token);
                assert_eq!(
                    ordinary_allowed, optimized_allowed,
                    "token {token} differs after token prefix {prefix:?}",
                );
                if ordinary_allowed {
                    let mut next = prefix.clone();
                    next.push(token);
                    // Four one-byte tokens can consume the complete language;
                    // no valid path can require a deeper token prefix.
                    if next.len() <= 4 && seen.insert(next.clone()) {
                        pending.push(next);
                    }
                }
            }
        }
    }

    #[test]
    fn partition_optimized_dynamic_save_load_preserves_quotient_mask() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"bb".to_vec()),
            (4, b"x".to_vec()),
        ]);
        let grammar_source = format!(
            "start ::= [ab]+{}",
            (0..17)
                .map(|index| format!(r#" | "unused_{index}""#))
                .collect::<String>()
        );
        let optimized =
            DynamicConstraint::compile_with_vocab_partition(Grammar::ebnf(&grammar_source), &vocab)
                .unwrap();
        let original_canonical = optimized
            .inner
            .dynamic_mask_vocab_for_runtime()
            .canonical_token_count();
        assert!(original_canonical < vocab.len());
        let bytes = optimized.save_with_external_vocab();
        let loaded = DynamicConstraint::load_with_vocab(&bytes, &vocab).unwrap();
        assert_eq!(loaded.start().mask(), optimized.start().mask());
        assert_eq!(
            loaded
                .inner
                .dynamic_mask_vocab_for_runtime()
                .canonical_token_count(),
            original_canonical,
            "save/load must retain the representative vocabulary rather than rebuilding the full vocabulary",
        );
    }

    #[test]
    fn partition_optimized_bounded_string_schema_does_not_fall_back_to_ordinary_dynamic() {
        let vocab = Vocab::new(vec![
            (0, b"\"abcdefgh\"".to_vec()),
            (1, b"\"ijklmnop\"".to_vec()),
            (2, b"true".to_vec()),
            (3, b"false".to_vec()),
            (4, b"{".to_vec()),
            (5, b"}".to_vec()),
            (6, b":".to_vec()),
            (7, b",".to_vec()),
            (8, b" ".to_vec()),
        ]);
        let mut properties = serde_json::Map::new();
        properties.insert(
            "bounded".to_owned(),
            serde_json::json!({"type":"string", "minLength":8, "maxLength":8}),
        );
        for index in 0..20 {
            properties.insert(format!("flag_{index}"), serde_json::json!({"type":"boolean"}));
        }
        let schema = serde_json::json!({
            "type":"object",
            "properties": properties,
            "additionalProperties": false
        })
        .to_string();

        let optimized = DynamicConstraint::compile_with_vocab_partition(
            Grammar::json_schema(&schema),
            &vocab,
        )
        .unwrap();
        assert!(
            optimized
                .inner
                .dynamic_mask_vocab_for_runtime()
                .is_grammar_quotiented(),
            "O2 must retain its grammar quotient for bounded-string residual schemas",
        );

        let bytes = optimized.save_with_external_vocab();
        let loaded = DynamicConstraint::load_with_vocab(&bytes, &vocab).unwrap();
        assert!(
            loaded
                .inner
                .dynamic_mask_vocab_for_runtime()
                .is_grammar_quotiented(),
            "external-vocab transfer must preserve the O2 grammar quotient",
        );
        assert_eq!(loaded.start().mask(), optimized.start().mask());
    }

    #[test]
    fn partition_optimized_plain_bounded_string_builds_a_real_quotient() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"b".to_vec()),
            (3, b"aa".to_vec()),
            (4, b"bb".to_vec()),
            (5, b"ab".to_vec()),
            (6, b"ba".to_vec()),
            (7, b"\"a".to_vec()),
            (8, b"\"b".to_vec()),
            (9, b"x".to_vec()),
        ]);
        let schema = r#"{"type":"string","maxLength":100}"#;

        let ordinary =
            DynamicConstraint::compile(Grammar::json_schema(schema), &vocab).unwrap();
        let optimized = DynamicConstraint::compile_with_vocab_partition(
            Grammar::json_schema(schema),
            &vocab,
        )
        .unwrap();

        let optimized_vocab = optimized.inner.dynamic_mask_vocab_for_runtime();
        assert!(optimized_vocab.is_grammar_quotiented());
        assert!(
            optimized_vocab.canonical_token_count() < vocab.len(),
            "bounded-string O2 must not degrade to the identity vocabulary quotient",
        );
        assert_eq!(ordinary.start().mask(), optimized.start().mask());

        let mut ordinary_state = ordinary.start();
        let mut optimized_state = optimized.start();
        ordinary_state.commit_token(0).unwrap();
        optimized_state.commit_token(0).unwrap();
        assert_eq!(ordinary_state.mask(), optimized_state.mask());
    }

    #[test]
    fn partition_optimized_multi_alternative_roundtrips_without_sharing_one_quotient() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"bb".to_vec()),
            (4, b"ab".to_vec()),
            (5, b"ba".to_vec()),
            (6, b"x".to_vec()),
        ]);
        let first = DynamicConstraint::compile_with_vocab_partition(
            Grammar::ebnf(r#"start ::= [ab]+"#),
            &vocab,
        )
        .unwrap();
        let second = DynamicConstraint::compile_with_vocab_partition(
            Grammar::ebnf(r#"start ::= "a"+"#),
            &vocab,
        )
        .unwrap();
        let optimized = DynamicConstraint::from_alternatives(vec![first, second]);
        assert_eq!(optimized.clone_constraints().len(), 2);
        let expected_mask = optimized.start().mask();

        // The self-contained wire has one shared vocab accelerator, so this
        // case intentionally drops the per-alternative quotients and rebuilds
        // the full vocab lazily after load rather than applying one quotient to
        // every alternative.
        let loaded = DynamicConstraint::load(&optimized.save()).unwrap();
        assert_eq!(loaded.start().mask(), expected_mask);

        // The external-vocab transfer has per-alternative metadata and can keep
        // each quotient independently.
        let transferred = DynamicConstraint::load_with_vocab(
            &optimized.save_with_external_vocab(),
            &vocab,
        )
        .unwrap();
        assert_eq!(transferred.start().mask(), expected_mask);
    }

    #[test]
    fn partition_optimized_component_keeps_boundary_token_distinctions() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ax".to_vec()),
            (3, b"ay".to_vec()),
            (4, b"bx".to_vec()),
            (5, b"by".to_vec()),
            (6, b"x".to_vec()),
            (7, b"y".to_vec()),
        ]);
        let child_grammar = Grammar::ebnf(r#"start ::= [ab]"#);
        let child_partition = VocabPartition::compile(child_grammar.clone(), &vocab).unwrap();
        assert_eq!(child_partition.class_of(2), child_partition.class_of(3));
        assert_eq!(child_partition.class_of(4), child_partition.class_of(5));

        let ordinary_child = DynamicConstraint::compile(child_grammar.clone(), &vocab).unwrap();
        let optimized_child =
            DynamicConstraint::compile_with_vocab_partition(child_grammar, &vocab).unwrap();
        assert_eq!(optimized_child.inner.token_bytes_count(), vocab.len());

        let parent = Grammar::glrm(
            "glrm 1; start start; extern grammar child; nt start = child \"x\";",
        );
        let ordinary = ConstraintSpec::builder(parent.clone(), &vocab)
            .unwrap()
            .bind_grammar("child", &ordinary_child)
            .unwrap()
            .build()
            .unwrap()
            .compile_dynamic()
            .unwrap();
        let optimized = ConstraintSpec::builder(parent, &vocab)
            .unwrap()
            .bind_grammar("child", &optimized_child)
            .unwrap()
            .build()
            .unwrap()
            .compile_dynamic()
            .unwrap();

        let ordinary_mask = ordinary.start().mask();
        let optimized_mask = optimized.start().mask();
        assert_eq!(ordinary_mask, optimized_mask);
        assert!(token_allowed(&optimized_mask, 2), "ax must cross child -> parent boundary");
        assert!(!token_allowed(&optimized_mask, 3), "ay must remain rejected by the parent boundary");
        assert!(token_allowed(&optimized_mask, 4), "bx must cross child -> parent boundary");
        assert!(!token_allowed(&optimized_mask, 5), "by must remain rejected by the parent boundary");
    }

    #[test]
    fn first_save_priming_threshold_tracks_large_composition_cache() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec())]);
        let mut constraint = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1;\nstart start;\nt A = \"a\";\nnt start = A;\n"),
            &vocab,
        )
        .unwrap();
        assert!(!should_prime_first_save_artifact(&constraint));

        let mut large = crate::automata::unweighted_u32::dfa::DFA::new();
        large
            .states
            .resize_with(FIRST_SAVE_TEMPLATE_STATE_PRIME_THRESHOLD, Default::default);
        constraint.composition_parser_templates_by_terminal = vec![Some(large)];
        assert!(should_prime_first_save_artifact(&constraint));
    }

    fn component_backend_flags(constraint: &RuntimeConstraint) -> Vec<bool> {
        constraint
            .static_dynamic_overlay
            .as_ref()
            .expect("bound constraint must use the explicit segmented runtime")
            .segmented_parser_components
            .iter()
            .map(|component| component.constraint.uses_dynamic_runtime())
            .collect()
    }

    fn leaf_backend_flags(constraint: &RuntimeConstraint) -> Vec<bool> {
        match constraint.static_dynamic_overlay.as_ref() {
            Some(overlay) if !overlay.segmented_parser_components.is_empty() => overlay
                .segmented_parser_components
                .iter()
                .flat_map(|component| leaf_backend_flags(&component.constraint))
                .collect(),
            _ => vec![constraint.uses_dynamic_runtime()],
        }
    }

    fn assert_recursive_compiler_views_detached(constraint: &RuntimeConstraint) {
        if !constraint.uses_compact_segmented_parser_runtime() {
            return;
        }
        assert_eq!(
            constraint.table.num_states, 0,
            "recursive coordinator retained a flattened LR state machine",
        );
        assert!(constraint.table.action.is_empty() && constraint.table.goto.is_empty());
        let overlay = constraint.static_dynamic_overlay.as_ref().unwrap();
        assert!(
            overlay.recursive_compiler_table.get().is_some(),
            "recursive coordinator lost its lazy compiler table",
        );
        for component in &overlay.segmented_parser_components {
            assert_recursive_compiler_views_detached(&component.constraint);
        }
    }

    fn poison_materialized_outer_table(constraint: &mut RuntimeConstraint) {
        constraint.recursive_parser_layout().unwrap().unwrap();
        constraint.table.action.clear();
        constraint.table.goto.clear();
        constraint.table.advance.clear();
        constraint.table.unconditional_advance.clear();
        constraint.table.rules.clear();
        constraint.table.forwarded_shifts.clear();
        constraint.table.control_terminals.clear();
        constraint.table.skip_terminals.clear();
        constraint.table.guarded_shift_index.clear();
        constraint.table.direct_regular_wide_frontiers.clear();
        constraint.table.num_states = 0;
        constraint.table.num_terminals = 0;
        constraint.table.num_rules = 0;
    }

    fn assert_static_boundary(constraint: &RuntimeConstraint) {
        let overlay = constraint
            .static_dynamic_overlay
            .as_ref()
            .expect("bound constraint must use the explicit segmented runtime");
        assert!(overlay.segmented_boundary_parser.is_none());
        assert!(overlay.segmented_boundary_terminal_trie.is_none());
        assert!(
            !overlay.segmented_boundary_shards.is_empty(),
            "static boundary policy must publish component-owned shards",
        );
        assert!(overlay.segmented_boundary_shards.iter().all(|shard| matches!(
            shard.backend,
            crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
        )));
    }

    fn assert_dynamic_boundary(constraint: &RuntimeConstraint) {
        let overlay = constraint
            .static_dynamic_overlay
            .as_ref()
            .expect("bound constraint must use the explicit segmented runtime");
        assert!(overlay.segmented_boundary_parser.is_none());
        assert!(overlay.segmented_boundary_terminal_trie.is_none());
        assert!(overlay.segmented_boundary_shards.iter().all(|shard| matches!(
            shard.backend,
            crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
        )));
    }

    #[test]
    fn segmented_composition_preserves_supplied_component_backends() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            // Forces B to carry a strictly internal parent -> child crossing.
            (2, b"xy".to_vec()),
        ]);
        let parent = Grammar::glrm(
            "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
        );
        let static_child = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap();
        let dynamic_child =
            DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab).unwrap();

        let static_with_dynamic = ConstraintSpec::builder(parent.clone(), &vocab)
            .unwrap()
            .bind_grammar("child", &dynamic_child)
            .unwrap()
            .build()
            .unwrap()
            .compile()
            .unwrap();
        assert_eq!(component_backend_flags(&static_with_dynamic), vec![false, true]);
        assert!(
            static_with_dynamic.serialized_artifact_cache.is_none(),
            "compiling a hybrid must not eagerly staticify its dynamic leaf for serialization",
        );
        assert_static_boundary(&static_with_dynamic);

        let dynamic_with_static = ConstraintSpec::builder(parent.clone(), &vocab)
            .unwrap()
            .bind_grammar("child", &static_child)
            .unwrap()
            .build()
            .unwrap()
            .compile_dynamic()
            .unwrap();
        for alternative in dynamic_with_static.clone_constraints() {
            assert_eq!(component_backend_flags(&alternative), vec![true, false]);
            let overlay = alternative.static_dynamic_overlay.as_ref().unwrap();
            assert!(overlay.segmented_boundary_parser.is_none());
            assert!(overlay.segmented_boundary_terminal_trie.is_none());
            assert!(overlay.segmented_boundary_shards.iter().all(|shard| matches!(
                shard.backend,
                crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
            )));
        }

        let dynamic_with_dynamic = ConstraintSpec::builder(parent, &vocab)
            .unwrap()
            .bind_grammar("child", &dynamic_child)
            .unwrap()
            .build()
            .unwrap()
            .compile_dynamic()
            .unwrap();
        for alternative in dynamic_with_dynamic.clone_constraints() {
            assert_eq!(component_backend_flags(&alternative), vec![true, true]);
        }
    }

    #[test]
    fn compiled_parent_late_binding_preserves_leaf_backends_and_boundary_choice() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
            (3, b"z".to_vec()),
            (4, b"yz".to_vec()),
            (5, b"xyz".to_vec()),
        ]);
        let one_slot = Grammar::glrm(
            "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
        );
        let static_parent = RuntimeConstraint::compile(one_slot.clone(), &vocab).unwrap();
        let dynamic_parent = DynamicConstraint::compile(one_slot, &vocab).unwrap();
        let static_child = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap();
        let dynamic_child =
            DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab).unwrap();

        let bound = static_parent.bind_grammar("child", &static_child).unwrap();
        assert_eq!(leaf_backend_flags(&bound), vec![false, false]);
        assert_static_boundary(&bound);
        assert!(
            bound.serialized_artifact_cache.is_none(),
            "late binding must not pay first-save serialization eagerly",
        );

        let bound = static_parent
            .bind_grammar_dynamic_boundary("child", &static_child)
            .unwrap();
        assert_eq!(leaf_backend_flags(&bound), vec![false, false]);
        assert_dynamic_boundary(&bound);

        let bound = static_parent.bind_grammar("child", &dynamic_child).unwrap();
        assert_eq!(leaf_backend_flags(&bound), vec![false, true]);
        assert_static_boundary(&bound);
        assert!(
            bound.serialized_artifact_cache.is_none(),
            "late binding a dynamic child must not trigger serialization-time staticification",
        );

        let bound = static_parent
            .bind_grammar_dynamic_boundary("child", &dynamic_child)
            .unwrap();
        assert_eq!(leaf_backend_flags(&bound), vec![false, true]);
        assert_dynamic_boundary(&bound);

        let bound = dynamic_parent.bind_grammar("child", &static_child).unwrap();
        for alternative in bound.clone_constraints() {
            assert_eq!(leaf_backend_flags(&alternative), vec![true, false]);
            assert_static_boundary(&alternative);
        }

        let bound = dynamic_parent
            .bind_grammar_dynamic_boundary("child", &static_child)
            .unwrap();
        for alternative in bound.clone_constraints() {
            assert_eq!(leaf_backend_flags(&alternative), vec![true, false]);
            assert_dynamic_boundary(&alternative);
        }

        let bound = dynamic_parent.bind_grammar("child", &dynamic_child).unwrap();
        for alternative in bound.clone_constraints() {
            assert_eq!(leaf_backend_flags(&alternative), vec![true, true]);
            assert_static_boundary(&alternative);
        }

        let bound = dynamic_parent
            .bind_grammar_dynamic_boundary("child", &dynamic_child)
            .unwrap();
        for alternative in bound.clone_constraints() {
            assert_eq!(leaf_backend_flags(&alternative), vec![true, true]);
            assert_dynamic_boundary(&alternative);
        }

        let two_slots = Grammar::glrm(
            "glrm 1; start start; extern grammar left; extern grammar right; \
             nt start = \"x\" left right;",
        );
        let open = RuntimeConstraint::compile(two_slots, &vocab).unwrap();
        let partially_bound = open.bind_grammar("left", &static_child).unwrap();
        assert!(
            partially_bound.late_bind_vocab.get().is_some(),
            "partially bound constraints should carry the shared vocabulary cache into the next bind",
        );
        assert!(partially_bound
            .late_grammar_slots
            .iter()
            .any(|slot| slot.name == "right"));
        let fully_bound = partially_bound
            .bind_grammar(
                "right",
                DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "z""#), &vocab).unwrap(),
            )
            .unwrap();
        assert!(fully_bound.late_grammar_slots.is_empty());
        assert_eq!(leaf_backend_flags(&fully_bound), vec![false, false, true]);
        assert_static_boundary(&fully_bound);

        let fully_bound = partially_bound
            .bind_grammar_dynamic_boundary(
                "right",
                DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "z""#), &vocab).unwrap(),
            )
            .unwrap();
        assert!(fully_bound.late_grammar_slots.is_empty());
        assert_eq!(leaf_backend_flags(&fully_bound), vec![false, false, true]);
        assert_dynamic_boundary(&fully_bound);

        let open = DynamicConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar left; extern grammar right; \
                 nt start = \"x\" left right;",
            ),
            &vocab,
        )
        .unwrap();
        let partially_bound = open.bind_grammar("left", &static_child).unwrap();
        let dynamic_right =
            DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "z""#), &vocab).unwrap();
        let fully_bound = partially_bound.bind_grammar("right", &dynamic_right).unwrap();
        for alternative in fully_bound.clone_constraints() {
            assert_eq!(leaf_backend_flags(&alternative), vec![true, false, true]);
            assert_static_boundary(&alternative);
        }
        let fully_bound = partially_bound
            .bind_grammar_dynamic_boundary("right", &dynamic_right)
            .unwrap();
        for alternative in fully_bound.clone_constraints() {
            assert_eq!(leaf_backend_flags(&alternative), vec![true, false, true]);
            assert_dynamic_boundary(&alternative);
        }
    }

    #[test]
    fn all_static_late_bound_segmented_roundtrip_preserves_fused_boundary_tokens() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
        ]);
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
            ),
            &vocab,
        )
        .unwrap();
        let child = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap();

        for dynamic_boundary in [false, true] {
            let bound = if dynamic_boundary {
                parent
                    .bind_grammar_dynamic_boundary("child", child.clone())
                    .unwrap()
            } else {
                parent.bind_grammar("child", child.clone()).unwrap()
            };
            let live_mask = bound.start().mask();
            assert_ne!(
                live_mask[0] & (1 << 2),
                0,
                "fused parent/child token must be live before serialization",
            );

            let loaded = RuntimeConstraint::load(bound.save()).unwrap();
            let overlay = loaded
                .static_dynamic_overlay
                .as_ref()
                .expect("round-tripped segmented runtime must retain A/B metadata");
            assert!(
                !overlay.segmented_static_baseline,
                "new segmented saves must not synthesize a flattened static parser baseline",
            );
            assert!(
                overlay.segmented_parser_components.len() == 2,
                "new segmented saves must retain the parent and child A components exactly",
            );
            assert!(overlay
                .segmented_parser_components
                .iter()
                .all(|component| component.root_disallowed_terminal.is_none()));
            assert_eq!(
                loaded.start().mask(),
                live_mask,
                "all-static segmented A+B changed across save/load (dynamic_boundary={dynamic_boundary})",
            );

            let mut state = loaded.start();
            state.commit_token(2).unwrap();
            assert!(state.is_accepting());

            let mut live_state = bound.start();
            let mut loaded_state = loaded.start();
            for token in [0, 1] {
                assert_eq!(
                    loaded_state.mask(),
                    live_state.mask(),
                    "all-static segmented state changed after round-trip before token {token} (dynamic_boundary={dynamic_boundary})",
                );
                live_state.commit_token(token).unwrap();
                loaded_state.commit_token(token).unwrap();
            }
            assert_eq!(loaded_state.is_accepting(), live_state.is_accepting());
            assert!(loaded_state.is_accepting());
        }
    }

    #[test]
    fn composed_dynamic_constraint_transfer_roundtrip_preserves_recursive_runtime() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
            (3, b"xyz".to_vec()),
        ]);
        let reference = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "x" "y""#),
            &vocab,
        )
        .unwrap();
        let parent = DynamicConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
            ),
            &vocab,
        )
        .unwrap();
        let child = DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap();

        for dynamic_boundary in [false, true] {
            let bound = if dynamic_boundary {
                parent
                    .bind_grammar_dynamic_boundary("child", &child)
                    .unwrap()
            } else {
                parent.bind_grammar("child", &child).unwrap()
            };
            assert_eq!(bound.start().mask(), reference.start().mask());

            let transfer = bound.clone().into_saved();
            let loaded = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
            assert_eq!(
                loaded.start().mask(),
                bound.start().mask(),
                "composed dynamic transfer changed initial mask (dynamic_boundary={dynamic_boundary})",
            );
            let mut state = loaded.start();
            state.commit_token(2).unwrap();
            assert!(state.is_accepting());
        }
    }

    #[test]
    fn dynamic_boundary_token_triggers_survive_component_roundtrip() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
            (3, b"z".to_vec()),
        ]);
        let reference = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "x" "y""#),
            &vocab,
        )
        .unwrap();
        let mut parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
            ),
            &vocab,
        )
        .unwrap();
        let mut child = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "y""#),
            &vocab,
        )
        .unwrap();

        parent.build_boundary_token_trigger().unwrap();
        child.build_boundary_token_trigger().unwrap();
        assert!(parent
            .boundary_trigger
            .token_summary()
            .expect("parent token trigger must be built")
            .contains(&2));

        let loaded_parent = RuntimeConstraint::load_body_artifact(parent.save()).unwrap();
        let loaded_child = RuntimeConstraint::load(child.save()).unwrap();
        let bound = loaded_parent
            .bind_grammar_dynamic_boundary("child", loaded_child)
            .unwrap();
        assert_eq!(bound.start().mask(), reference.start().mask());

        let overlay = bound
            .static_dynamic_overlay
            .as_ref()
            .expect("dynamic composition must retain component runtime metadata");
        let parent_tokens = overlay.segmented_parser_components[0]
            .constraint
            .boundary_trigger
            .token_summary()
            .expect("parent Tokens trigger must survive save/load");
        let child_tokens = overlay.segmented_parser_components[1]
            .constraint
            .boundary_trigger
            .token_summary()
            .expect("child Tokens trigger must survive save/load");
        assert!(parent_tokens.contains(&2));
        assert!(child_tokens.iter().all(|&token| token < 4));
    }

    #[test]
    fn exact_boundary_triggers_roundtrip_and_gate_entry_and_finish() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"z".to_vec()),
            (3, b"xy".to_vec()),
            (4, b"yz".to_vec()),
            (5, b"xyz".to_vec()),
        ]);
        let reference = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "x" "y" "z""#),
            &vocab,
        )
        .unwrap();
        let mut parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" child \"z\";",
            ),
            &vocab,
        )
        .unwrap();
        let mut child = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "y""#),
            &vocab,
        )
        .unwrap();

        parent.build_exact_boundary_trigger().unwrap();
        child.build_exact_boundary_trigger().unwrap();
        assert!(matches!(
            parent.boundary_trigger,
            crate::runtime::BoundaryTrigger::Exact(_)
        ));
        assert!(matches!(
            child.boundary_trigger,
            crate::runtime::BoundaryTrigger::Exact(_)
        ));

        let loaded_parent = RuntimeConstraint::load_body_artifact(parent.save()).unwrap();
        let loaded_child = RuntimeConstraint::load(child.save()).unwrap();
        let bound = loaded_parent
            .bind_grammar_dynamic_boundary("child", loaded_child)
            .unwrap();
        let overlay = bound
            .static_dynamic_overlay
            .as_ref()
            .expect("dynamic composition must retain component runtime metadata");
        assert!(overlay.segmented_parser_components.iter().all(|component| {
            matches!(
                component.constraint.boundary_trigger,
                crate::runtime::BoundaryTrigger::Exact(_)
            )
        }));

        for tokens in [&[5][..], &[0, 4][..], &[3, 2][..], &[0, 1, 2][..]] {
            let mut actual = bound.start();
            let mut expected = reference.start();
            for &token in tokens {
                assert_eq!(
                    actual.mask(),
                    expected.mask(),
                    "Exact-trigger dynamic boundary mismatch before {tokens:?} token {token}",
                );
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert_eq!(actual.is_accepting(), expected.is_accepting(), "{tokens:?}");
            assert!(actual.is_accepting(), "{tokens:?}");
        }
    }

    #[test]
    fn exact_trigger_supports_control_bearing_composed_component() {
        let vocab = Vocab::new(vec![
            (0, b"<".to_vec()),
            (1, b"a".to_vec()),
            (2, b">".to_vec()),
            (3, b"!".to_vec()),
            (4, b"<a>!".to_vec()),
            (5, b"a>!".to_vec()),
            (6, b">!".to_vec()),
        ]);

        let inner_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"glrm 1; start inner; extern grammar leaf; nt inner = "<" leaf ">";"#,
            ),
            &vocab,
        )
        .unwrap();
        let leaf = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "a""#), &vocab)
            .unwrap();
        let mut inner = inner_parent
            .bind_grammar_dynamic_boundary("leaf", leaf)
            .unwrap();
        assert_recursive_compiler_views_detached(&inner);
        let mut compiler_view = inner.clone();
        compiler_view
            .prepare_recursive_compiler_table_for_composition()
            .unwrap();
        assert!(
            !compiler_view.table.control_terminals.is_empty(),
            "fixture must exercise Exact construction over compiler-materialized linker controls",
        );
        inner.build_exact_boundary_trigger().unwrap();
        assert_recursive_compiler_views_detached(&inner);
        assert!(matches!(
            inner.boundary_trigger,
            crate::runtime::BoundaryTrigger::Exact(_)
        ));

        let outer = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"glrm 1; start document; extern grammar inner; nt document = inner "!";"#,
            ),
            &vocab,
        )
        .unwrap();
        let composed = outer
            .bind_grammar_dynamic_boundary("inner", RuntimeConstraint::load(inner.save()).unwrap())
            .unwrap();
        let reference = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "<" "a" ">" "!""#),
            &vocab,
        )
        .unwrap();

        let overlay = composed.static_dynamic_overlay.as_ref().unwrap();
        assert!(matches!(
            overlay.segmented_parser_components[1]
                .constraint
                .boundary_trigger,
            crate::runtime::BoundaryTrigger::Exact(_)
        ));

        for tokens in [&[4][..], &[0, 5][..], &[0, 1, 6][..], &[0, 1, 2, 3][..]] {
            let mut actual = composed.start();
            let mut expected = reference.start();
            for &token in tokens {
                assert_eq!(
                    actual.mask(),
                    expected.mask(),
                    "control-bearing Exact trigger mismatch before {tokens:?} token {token}",
                );
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert_eq!(actual.is_accepting(), expected.is_accepting(), "{tokens:?}");
            assert!(actual.is_accepting(), "{tokens:?}");
        }
    }

    #[test]
    fn loaded_constraint_trigger_upgrade_resaves_updated_link_metadata() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
        ]);
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
            ),
            &vocab,
        )
        .unwrap();
        let child = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap();

        let mut loaded = RuntimeConstraint::load_body_artifact(parent.save()).unwrap();
        assert!(loaded.deferred_composition_metadata_blob.is_some());
        loaded.build_exact_boundary_trigger().unwrap();
        assert!(matches!(
            loaded.boundary_trigger,
            crate::runtime::BoundaryTrigger::Exact(_)
        ));
        assert!(
            loaded.deferred_composition_metadata_blob.is_some(),
            "trigger upgrade should not force materialization of the heavy parser-cache section",
        );

        let reloaded = RuntimeConstraint::load_body_artifact(loaded.save()).unwrap();
        let bound = reloaded
            .bind_grammar_dynamic_boundary("child", child)
            .unwrap();
        let overlay = bound.static_dynamic_overlay.as_ref().unwrap();
        assert!(matches!(
            overlay.segmented_parser_components[0]
                .constraint
                .boundary_trigger,
            crate::runtime::BoundaryTrigger::Exact(_)
        ));
        assert_ne!(bound.start().mask()[0] & (1 << 2), 0);
    }

    #[test]
    fn exact_triggers_survive_outer_runtime_lexer_product_multi_source_states() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"c".to_vec()),
            (3, b"!".to_vec()),
            (4, b"?".to_vec()),
            (5, b"c!".to_vec()),
            (6, b"c?".to_vec()),
        ]);
        let child_spec = ConstraintSpec::builder(
            Grammar::ebnf(r#"start ::= "a" "b" "c""#),
            &vocab,
        )
        .unwrap()
        .boundary_trigger_detail(crate::BoundaryTriggerDetail::Exact)
        .build()
        .unwrap();
        // Keep one child component so its local-LR -> composed-LR relation is
        // functional, while the parent also has an equivalent local lexical
        // lane. The outer lexer product can then coalesce parent + child lanes
        // after `a`/`b` without relying on duplicate child call sites.
        let parent = Grammar::glrm(
            "glrm 1; start document; extern grammar child; \
             nt document = child \"!\" | \"a\" \"b\" \"c\" \"?\";",
        );
        let composed = ConstraintSpec::builder(parent, &vocab)
            .unwrap()
            .bind_grammar("child", child_spec)
            .unwrap()
            .build()
            .unwrap()
            .compile_dynamic()
            .unwrap();
        let alternatives = composed.clone_constraints();
        assert_eq!(alternatives.len(), 1);
        let composed = &alternatives[0];
        let reference = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "a" "b" "c" "!" | "a" "b" "c" "?""#),
            &vocab,
        )
        .unwrap();

        let explicitly_disabled = std::env::var("GLRMASK_COMPOSE_RUNTIME_LEXER_PRODUCT")
            .ok()
            .is_some_and(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "" | "0" | "false" | "no" | "off"
                )
            });

        let mut actual = composed.start();
        let mut expected = reference.start();
        for token in [0, 1] {
            assert_eq!(actual.mask(), expected.mask());
            actual.commit_token(token).unwrap();
            expected.commit_token(token).unwrap();
        }

        if !explicitly_disabled
            && let Some(source_offset) = composed.runtime_source_state_offset()
        {
            let product_state = *actual.state.keys().next().unwrap();
            assert!(product_state < source_offset);
            assert!(
                composed
                    .runtime_product_source_states(product_state)
                    .is_some_and(|sources| sources.len() >= 2),
                "a selected outer product state must preserve every represented lexer lane",
            );
        }

        let mask = actual.mask();
        let reference_mask = expected.mask();
        assert_eq!(mask, reference_mask);
        assert_ne!(mask[0] & (1 << 5), 0, "c! must cross child -> parent internally");
        assert_ne!(mask[0] & (1 << 6), 0, "c? must remain valid on the parent-local branch");
    }

    #[test]
    fn exact_trigger_detail_is_independent_of_dynamic_component_compilation() {
        let vocab = Vocab::new(vec![
            (0, b"y".to_vec()),
            (1, b"z".to_vec()),
            (2, b"yz".to_vec()),
        ]);
        let spec = ConstraintSpec::builder(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap()
            .boundary_trigger_detail(crate::BoundaryTriggerDetail::Exact)
            .build()
            .unwrap();

        let static_constraint = spec.compile().unwrap();
        assert!(matches!(
            static_constraint.boundary_trigger,
            crate::runtime::BoundaryTrigger::Exact(_)
        ));

        let dynamic_constraint = spec.compile_dynamic().unwrap();
        assert!(dynamic_constraint.clone_constraints().iter().all(|constraint| {
            matches!(constraint.boundary_trigger, crate::runtime::BoundaryTrigger::Exact(_))
        }));

        let loaded_dynamic = DynamicConstraint::load(&dynamic_constraint.save()).unwrap();
        assert!(loaded_dynamic.clone_constraints().iter().all(|constraint| {
            matches!(constraint.boundary_trigger, crate::runtime::BoundaryTrigger::Exact(_))
        }));

        let transfer = dynamic_constraint.clone().into_saved();
        let loaded_transfer = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert!(loaded_transfer.clone_constraints().iter().all(|constraint| {
            matches!(constraint.boundary_trigger, crate::runtime::BoundaryTrigger::Exact(_))
        }));
    }

    #[test]
    fn boundary_token_trigger_follows_lexeme_resets_inside_model_token() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"q".to_vec()),
            (2, b"y".to_vec()),
            (3, b"xqy".to_vec()),
        ]);
        let mut parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" \"q\" child;",
            ),
            &vocab,
        )
        .unwrap();
        parent.build_boundary_token_trigger().unwrap();
        assert!(
            parent
                .boundary_trigger
                .token_summary()
                .expect("Tokens trigger must be built")
                .contains(&3),
            "a token that reaches the child only after multiple local lexemes must remain boundary-relevant",
        );

        let child = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap();
        let bound = parent
            .bind_grammar_dynamic_boundary("child", child)
            .unwrap();
        assert_ne!(bound.start().mask()[0] & (1 << 3), 0);
    }

    #[test]
    fn nested_dynamic_boundaries_preserve_token_crossing_leaf_middle_and_outer() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"[".to_vec()),
            (2, b"a".to_vec()),
            (3, b"]".to_vec()),
            (4, b"!".to_vec()),
            (5, b"a]!".to_vec()),
            (6, b"[a]!".to_vec()),
            (7, b"X[a]!".to_vec()),
        ]);
        let leaf = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start leaf; nt leaf = \"a\";"),
            &vocab,
        )
        .unwrap();
        let middle_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start middle; extern grammar leaf; nt middle = \"[\" leaf \"]\";",
            ),
            &vocab,
        )
        .unwrap();
        let middle = middle_parent
            .bind_grammar_dynamic_boundary("leaf", leaf)
            .unwrap();
        let outer_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar middle; nt document = \"X\" middle \"!\";",
            ),
            &vocab,
        )
        .unwrap();
        let bound = outer_parent
            .bind_grammar_dynamic_boundary("middle", middle)
            .unwrap();
        let monolithic = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start document; nt document = \"X\" \"[\" \"a\" \"]\" \"!\";"),
            &vocab,
        )
        .unwrap();

        let loaded = RuntimeConstraint::load(&bound.save()).unwrap();
        assert_recursive_compiler_views_detached(&bound);
        assert_recursive_compiler_views_detached(&loaded);
        let mut no_trigger = bound.clone();
        {
            let overlay = no_trigger.static_dynamic_overlay.as_mut().unwrap();
            for component in &mut overlay.segmented_parser_components {
                let Some(shard) = component.boundary.as_mut() else {
                    continue;
                };
                if !matches!(
                    shard.backend,
                    crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
                ) {
                    continue;
                }
                shard.candidate_tokens = None;
                std::sync::Arc::make_mut(&mut component.constraint).boundary_trigger =
                    crate::runtime::BoundaryTrigger::None;
            }
        }
        let mut no_outer_tokenizer = loaded.clone();
        {
            let root = no_outer_tokenizer
                .static_dynamic_overlay
                .as_ref()
                .unwrap()
                .segmented_parser_components[0]
                .constraint
                .clone();
            no_outer_tokenizer.tokenizer = root.tokenizer.clone();
            no_outer_tokenizer.tokenizer_fast_transitions =
                root.tokenizer_fast_transitions.clone();
            no_outer_tokenizer.tokenizer_has_epsilon_transitions =
                root.tokenizer_has_epsilon_transitions;
        }
        let mut no_outer_table = loaded.clone();
        poison_materialized_outer_table(&mut no_outer_table);
        for constraint in [
            &bound,
            &loaded,
            &no_trigger,
            &no_outer_tokenizer,
            &no_outer_table,
        ] {
            let mut pending = vec![Vec::<u32>::new()];
            while let Some(path) = pending.pop() {
                let mut actual = constraint.start();
                let mut expected = monolithic.start();
                for &token in &path {
                    actual.commit_token(token).unwrap();
                    expected.commit_token(token).unwrap();
                }
                let actual_mask = actual.mask();
                let expected_mask = expected.mask();
                assert_eq!(actual_mask, expected_mask, "mask mismatch at {path:?}");
                assert_eq!(
                    actual.is_accepting(),
                    expected.is_accepting(),
                    "acceptance mismatch at {path:?}",
                );
                if path.len() == 5 {
                    continue;
                }
                for token in 0..8u32 {
                    if actual_mask
                        .get(token as usize / 32)
                        .is_none_or(|word| *word & (1u32 << (token % 32)) == 0)
                    {
                        continue;
                    }
                    let mut next = path.clone();
                    next.push(token);
                    pending.push(next);
                }
            }
        }

        for tokens in [
            &[7][..],
            &[0, 6][..],
            &[0, 1, 5][..],
            &[0, 1, 2, 3, 4][..],
        ] {
            let mut actual = bound.start();
            let mut expected = monolithic.start();
            for &token in tokens {
                assert_eq!(actual.mask(), expected.mask(), "before {tokens:?} token {token}");
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert_eq!(actual.is_accepting(), expected.is_accepting(), "{tokens:?}");
            assert!(actual.is_accepting(), "{tokens:?}");
        }
    }

    #[test]
    fn recursive_live_parser_expands_static_nested_component_with_native_boundary() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"[".to_vec()),
            (2, b"a".to_vec()),
            (3, b"]".to_vec()),
            (4, b"!".to_vec()),
            (5, b"a]!".to_vec()),
            (6, b"[a]!".to_vec()),
            (7, b"X[a]!".to_vec()),
        ]);
        let leaf = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start leaf; nt leaf = \"a\";"),
            &vocab,
        )
        .unwrap();
        let middle_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start middle; extern grammar leaf; nt middle = \"[\" leaf \"]\";",
            ),
            &vocab,
        )
        .unwrap();
        // Static B is transported exactly onto the recursive parser coordinate,
        // so the middle wrapper can expose its intact parent/child leaves.
        let middle = middle_parent.bind_grammar("leaf", leaf).unwrap();
        assert!(middle.uses_compact_segmented_parser_runtime());
        let outer_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar middle; nt document = \"X\" middle \"!\";",
            ),
            &vocab,
        )
        .unwrap();
        let bound = outer_parent
            .bind_grammar_dynamic_boundary("middle", middle)
            .unwrap();
        let layout = bound.recursive_parser_layout().unwrap().unwrap();
        assert_eq!(layout.leaves.len(), 3);
        let monolithic = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start document; nt document = \"X\" \"[\" \"a\" \"]\" \"!\";"),
            &vocab,
        )
        .unwrap();

        for constraint in [&bound, &RuntimeConstraint::load(&bound.save()).unwrap()] {
            for tokens in [
                &[7][..],
                &[0, 6][..],
                &[0, 1, 5][..],
                &[0, 1, 2, 3, 4][..],
            ] {
                let mut actual = constraint.start();
                let mut expected = monolithic.start();
                for &token in tokens {
                    assert_eq!(actual.mask(), expected.mask(), "before {tokens:?} token {token}");
                    actual.commit_token(token).unwrap();
                    expected.commit_token(token).unwrap();
                }
                assert_eq!(actual.is_accepting(), expected.is_accepting(), "{tokens:?}");
                assert!(actual.is_accepting(), "{tokens:?}");
            }
        }
    }

    #[test]
    fn recursive_special_token_routes_through_active_child_leaf_and_returns_to_parent() {
        let vocab = Vocab::new(vec![(0, b"X".to_vec()), (1, b"!".to_vec())]);
        let child = RuntimeConstraint::compile(
            Grammar::glrm("start child; nt child ::= @token(100);"),
            &vocab,
        )
        .unwrap();
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar child; nt document = \"X\" child \"!\";",
            ),
            &vocab,
        )
        .unwrap();
        let bound = parent.bind_grammar("child", child).unwrap();
        assert!(bound.uses_compact_segmented_parser_runtime());

        let loaded = RuntimeConstraint::load(&bound.save()).unwrap();
        let mut no_outer_table = loaded.clone();
        poison_materialized_outer_table(&mut no_outer_table);
        for constraint in [&bound, &loaded, &no_outer_table] {
            let mut state = constraint.start();
            state.commit_token(0).unwrap();
            let mask = state.mask();
            assert_ne!(mask[100 / 32] & (1u32 << (100 % 32)), 0);

            state.commit_token(100).unwrap();
            let mask = state.mask();
            assert_ne!(mask[1 / 32] & (1u32 << (1 % 32)), 0);

            state.commit_token(1).unwrap();
            assert!(state.is_accepting());
        }
    }

    #[test]
    fn recursive_static_virtual_residual_child_handles_fused_boundaries_and_roundtrip() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"!".to_vec()),
            (2, b"\"".to_vec()),
            (3, b"x:".to_vec()),
            (4, b"a".to_vec()),
            (5, b"X\"".to_vec()),
            (6, b"\"!".to_vec()),
            (7, b"xa".to_vec()),
            (8, b":".to_vec()),
            (9, b"X\"x:".to_vec()),
            (10, b"a\"!".to_vec()),
            (11, b"aa".to_vec()),
            (12, b"a\"".to_vec()),
        ]);
        let child = RuntimeConstraint::from_json_schema(
            r#"{"type":"string","format":"uri","minLength":1,"maxLength":5000}"#,
            &vocab,
        )
        .unwrap();
        assert!(child.tokenizer.has_virtual_residual_runtime());

        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar payload; nt document = \"X\" payload \"!\";",
            ),
            &vocab,
        )
        .unwrap();
        // A virtual-residual child cannot take a static shard: requesting
        // static must decline loudly, never silently succeed as dynamic.
        assert!(
            parent.bind_grammar("payload", &child).is_err(),
            "static bind of a virtual-residual child must decline loudly",
        );
        let bound = parent
            .bind_grammar_dynamic_boundary("payload", child)
            .unwrap();
        let loaded = RuntimeConstraint::load(&bound.save()).unwrap();

        let token_allowed = |mask: &[u32], token: u32| {
            mask[token as usize / 32] & (1u32 << (token % 32)) != 0
        };
        for constraint in [&bound, &loaded] {
            let mut state = constraint.start();
            assert!(token_allowed(&state.mask(), 9));
            state.commit_token(9).unwrap();
            assert!(token_allowed(&state.mask(), 10));
            state.commit_token(10).unwrap();
            assert!(state.is_accepting());

            let mut exact = constraint.start();
            let mut exact_bytes = b"X\"x:".to_vec();
            exact_bytes.extend(std::iter::repeat_n(b'a', 4998));
            exact_bytes.extend_from_slice(b"\"!");
            exact.commit_bytes(&exact_bytes).unwrap();
            assert!(exact.is_accepting());

            let mut over = constraint.start();
            let mut over_bytes = b"X\"x:".to_vec();
            over_bytes.extend(std::iter::repeat_n(b'a', 4999));
            over_bytes.extend_from_slice(b"\"!");
            assert!(over.commit_bytes(&over_bytes).is_err());
        }
    }

    #[test]
    fn nested_boundaries_use_recursive_parser_coordinate_live_and_loaded() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"[".to_vec()),
            (2, b"a".to_vec()),
            (3, b"]".to_vec()),
            (4, b"!".to_vec()),
            (5, b"a]!".to_vec()),
            (6, b"[a]!".to_vec()),
            (7, b"X[a]!".to_vec()),
        ]);
        let leaf = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start leaf; nt leaf = \"a\";"),
            &vocab,
        )
        .unwrap();
        let middle_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start middle; extern grammar leaf; nt middle = \"[\" leaf \"]\";",
            ),
            &vocab,
        )
        .unwrap();
        let middle = middle_parent.bind_grammar("leaf", leaf).unwrap();
        assert!(middle.uses_compact_segmented_parser_runtime());

        let outer_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar middle; nt document = \"X\" middle \"!\";",
            ),
            &vocab,
        )
        .unwrap();
        let bound = outer_parent
            .bind_grammar_dynamic_boundary("middle", middle)
            .unwrap();
        assert!(bound.uses_compact_segmented_parser_runtime());
        assert!(
            bound
                .static_dynamic_overlay
                .as_ref()
                .unwrap()
                .segmented_component_union_root_dispatch
                .is_empty(),
            "recursive runtime must not retain materialized-state component dispatch",
        );
        assert!(
            bound
                .static_dynamic_overlay
                .as_ref()
                .unwrap()
                .segmented_parser_state_offsets
                .is_empty(),
            "recursive runtime must derive state intervals from its component tree",
        );
        let bound_overlay = bound.static_dynamic_overlay.as_ref().unwrap();
        assert!(bound_overlay.segmented_boundary_shards.iter().all(|shard| {
            shard.start_parser_states.len() == 0
        }));
        assert!(bound_overlay.segmented_parser_components.iter().all(|component| {
            component
                .boundary
                .as_ref()
                .is_none_or(|shard| shard.start_parser_states.len() == 0)
        }));
        let layout = bound.recursive_parser_layout().unwrap().unwrap();
        assert_eq!(layout.leaves.len(), 3);

        let loaded = RuntimeConstraint::load(&bound.save()).unwrap();
        assert!(loaded.uses_compact_segmented_parser_runtime());
        assert!(
            loaded
                .static_dynamic_overlay
                .as_ref()
                .unwrap()
                .segmented_component_union_root_dispatch
                .is_empty(),
        );
        assert!(
            loaded
                .static_dynamic_overlay
                .as_ref()
                .unwrap()
                .segmented_parser_state_offsets
                .is_empty(),
        );
        let loaded_overlay = loaded.static_dynamic_overlay.as_ref().unwrap();
        assert!(loaded_overlay.segmented_boundary_shards.iter().all(|shard| {
            shard.start_parser_states.len() == 0
        }));
        assert!(loaded_overlay.segmented_parser_components.iter().all(|component| {
            component
                .boundary
                .as_ref()
                .is_none_or(|shard| shard.start_parser_states.len() == 0)
        }));
        assert_eq!(loaded.recursive_parser_layout().unwrap().unwrap().leaves.len(), 3);

        let monolithic = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start document; nt document = \"X\" \"[\" \"a\" \"]\" \"!\";"),
            &vocab,
        )
        .unwrap();
        let mut no_outer_table = loaded.clone();
        poison_materialized_outer_table(&mut no_outer_table);
        fn compare_reachable_prefix_tree(
            actual_constraint: &RuntimeConstraint,
            expected_constraint: &RuntimeConstraint,
            max_depth: usize,
        ) {
            let mut pending = vec![Vec::<u32>::new()];
            while let Some(path) = pending.pop() {
                let mut actual = actual_constraint.start();
                let mut expected = expected_constraint.start();
                for &token in &path {
                    actual.commit_token(token).unwrap();
                    expected.commit_token(token).unwrap();
                }
                let actual_mask = actual.mask();
                let expected_mask = expected.mask();
                assert_eq!(actual_mask, expected_mask, "mask mismatch at {path:?}");
                assert_eq!(
                    actual.is_accepting(),
                    expected.is_accepting(),
                    "acceptance mismatch at {path:?}",
                );
                if path.len() == max_depth {
                    continue;
                }
                for token in 0..8u32 {
                    let word = token as usize / 32;
                    let bit = token % 32;
                    if actual_mask
                        .get(word)
                        .is_none_or(|word| *word & (1u32 << bit) == 0)
                    {
                        continue;
                    }
                    let mut next = path.clone();
                    next.push(token);
                    pending.push(next);
                }
            }
        }
        for constraint in [&bound, &loaded, &no_outer_table] {
            compare_reachable_prefix_tree(constraint, &monolithic, 5);
            for tokens in [
                &[7][..],
                &[0, 6][..],
                &[0, 1, 5][..],
                &[0, 1, 2, 3, 4][..],
            ] {
                let mut actual = constraint.start();
                let mut expected = monolithic.start();
                for &token in tokens {
                    assert_eq!(actual.mask(), expected.mask(), "before {tokens:?} token {token}");
                    actual.commit_token(token).unwrap();
                    expected.commit_token(token).unwrap();
                }
                assert_eq!(actual.is_accepting(), expected.is_accepting(), "{tokens:?}");
                assert!(actual.is_accepting(), "{tokens:?}");
            }
        }
    }

    #[test]
    fn nested_static_nullable_wrapper_returns_in_recursive_live_runtime() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
            (3, b"X!".to_vec()),
            (4, b"Xa!".to_vec()),
            (5, b"a!".to_vec()),
        ]);
        let leaf = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start leaf; nt leaf = \"a\";"),
            &vocab,
        )
        .unwrap();
        let middle_parent = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start middle; extern grammar leaf; nt middle = leaf?;"),
            &vocab,
        )
        .unwrap();
        assert!(middle_parent.table.embedded_start_nullable());
        let middle = middle_parent.bind_grammar("leaf", leaf).unwrap();
        assert!(middle.table.embedded_start_nullable());
        assert!(middle.uses_compact_segmented_parser_runtime());

        let outer_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar middle; nt document = \"X\" middle \"!\";",
            ),
            &vocab,
        )
        .unwrap();
        let bound = outer_parent
            .bind_grammar_dynamic_boundary("middle", middle)
            .unwrap();
        assert!(bound.uses_compact_segmented_parser_runtime());
        let loaded = RuntimeConstraint::load(&bound.save()).unwrap();
        assert!(loaded.uses_compact_segmented_parser_runtime());
        let monolithic = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start document; nt document = \"X\" \"a\"? \"!\";"),
            &vocab,
        )
        .unwrap();
        let mut no_outer_table = loaded.clone();
        poison_materialized_outer_table(&mut no_outer_table);

        for constraint in [&bound, &loaded, &no_outer_table] {
            for tokens in [&[3][..], &[4][..], &[0, 2][..], &[0, 5][..], &[0, 1, 2][..]] {
                let mut actual = constraint.start();
                let mut expected = monolithic.start();
                for &token in tokens {
                    assert_eq!(actual.mask(), expected.mask(), "before {tokens:?} token {token}");
                    actual.commit_token(token).unwrap();
                    expected.commit_token(token).unwrap();
                }
                assert_eq!(actual.is_accepting(), expected.is_accepting(), "{tokens:?}");
                assert!(actual.is_accepting(), "{tokens:?}");
            }
        }
    }

    #[test]
    fn recursive_parser_layout_flattens_state_coordinate_but_preserves_outer_wrapper_owner() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"[".to_vec()),
            (2, b"a".to_vec()),
            (3, b"]".to_vec()),
            (4, b"!".to_vec()),
        ]);
        let leaf = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start leaf; nt leaf = \"a\";"),
            &vocab,
        )
        .unwrap();
        let leaf_states = leaf.table.num_states;
        let leaf_tokenizer_states = leaf.tokenizer.num_states();
        let leaf_tokenizer_reset = leaf.runtime_commit_initial_state();
        let middle_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start middle; extern grammar leaf; nt middle = \"[\" leaf \"]\";",
            ),
            &vocab,
        )
        .unwrap();
        let middle_parent_states = middle_parent.table.num_states;
        let middle_parent_tokenizer_states = middle_parent.tokenizer.num_states();
        let middle_parent_tokenizer_reset = middle_parent.runtime_commit_initial_state();
        let middle = middle_parent
            .bind_grammar_dynamic_boundary("leaf", leaf)
            .unwrap();
        let outer_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar middle; nt document = \"X\" middle \"!\";",
            ),
            &vocab,
        )
        .unwrap();
        let outer_parent_states = outer_parent.table.num_states;
        let outer_parent_tokenizer_states = outer_parent.tokenizer.num_states();
        let outer_parent_tokenizer_reset = outer_parent.runtime_commit_initial_state();
        let bound = outer_parent
            .bind_grammar_dynamic_boundary("middle", middle)
            .unwrap();

        for constraint in [&bound, &RuntimeConstraint::load(&bound.save()).unwrap()] {
            let layout = constraint
                .recursive_parser_layout()
                .unwrap()
                .expect("nested composition must expose a recursive parser layout");
            assert_eq!(layout.component_offsets, vec![0, outer_parent_states]);
            assert_eq!(layout.leaves.len(), 3);
            assert_eq!(
                layout
                    .leaves
                    .iter()
                    .map(|leaf| leaf.component_path.clone())
                    .collect::<Vec<_>>(),
                vec![vec![0], vec![1, 0], vec![1, 1]],
            );
            assert_eq!(
                layout
                    .leaves
                    .iter()
                    .map(|leaf| (leaf.state_offset, leaf.state_count, leaf.top_component))
                    .collect::<Vec<_>>(),
                vec![
                    (0, outer_parent_states, 0),
                    (outer_parent_states, middle_parent_states, 1),
                    (
                        outer_parent_states + middle_parent_states,
                        leaf_states,
                        1,
                    ),
                ],
            );
            assert_eq!(
                layout.total_states,
                outer_parent_states + middle_parent_states + leaf_states,
            );
            assert_eq!(
                layout.leaf_tokenizer_state_offsets,
                vec![
                    0,
                    outer_parent_tokenizer_states,
                    outer_parent_tokenizer_states + middle_parent_tokenizer_states,
                ],
            );
            assert_eq!(
                layout.total_tokenizer_states,
                outer_parent_tokenizer_states
                    + middle_parent_tokenizer_states
                    + leaf_tokenizer_states,
            );
            for (global_terminal, targets) in layout.terminal_targets.iter().enumerate() {
                for &(leaf_index, local_terminal) in targets {
                    let scoped = constraint
                        .recursive_terminal_scoped_id(leaf_index as usize, local_terminal)
                        .unwrap();
                    assert_eq!(
                        constraint.recursive_terminal_leaf_local(scoped),
                        Some((leaf_index as usize, local_terminal)),
                        "terminal routing lost global={global_terminal} leaf={leaf_index} local={local_terminal}",
                    );
                }
            }
            let tokenizer_counts = [
                outer_parent_tokenizer_states,
                middle_parent_tokenizer_states,
                leaf_tokenizer_states,
            ];
            let tokenizer_resets = [
                outer_parent_tokenizer_reset,
                middle_parent_tokenizer_reset,
                leaf_tokenizer_reset,
            ];
            for leaf_index in 0..layout.leaves.len() {
                for local_state in 0..tokenizer_counts[leaf_index] {
                    let scoped = constraint
                        .recursive_tokenizer_scoped_state(leaf_index, local_state)
                        .unwrap();
                    assert_eq!(
                        constraint.recursive_tokenizer_leaf_state(scoped),
                        Some((leaf_index, local_state)),
                    );
                }
                assert_eq!(
                    constraint.recursive_tokenizer_reset_state(leaf_index),
                    constraint.recursive_tokenizer_scoped_state(
                        leaf_index,
                        tokenizer_resets[leaf_index],
                    ),
                );
            }
            assert!(layout.leaves[1].state_count >= 2);
            let root_top = layout.leaves[0].state_offset;
            let middle_top_a = layout.leaves[1].state_offset;
            let middle_top_b = middle_top_a + 1;
            let leaf_top = layout.leaves[2].state_offset;
            let acc0 = crate::compiler::glr::accumulator::TerminalsDisallowed::new()
                .with_insert(10, 20);
            let acc1 = crate::compiler::glr::accumulator::TerminalsDisallowed::new()
                .with_insert(11, 21);
            let acc2 = crate::compiler::glr::accumulator::TerminalsDisallowed::new()
                .with_insert(12, 22);
            let acc3 = crate::compiler::glr::accumulator::TerminalsDisallowed::new()
                .with_insert(13, 23);
            let mixed = crate::compiler::glr::parser::ParserGSS::from_stacks(&[
                (vec![root_top], acc0.clone()),
                (vec![root_top, middle_top_a], acc1.clone()),
                (vec![root_top, middle_top_b], acc2.clone()),
                (vec![root_top, leaf_top], acc3.clone()),
            ]);
            let partitions = constraint
                .partition_recursive_parser_gss_by_active_leaf(&mixed)
                .expect("recursive GSS must partition by active tokenizer leaf");
            assert_eq!(
                partitions.iter().map(|(leaf, _)| *leaf).collect::<Vec<_>>(),
                vec![0, 1, 2],
            );
            let mut middle_stacks = partitions[1].1.to_stacks(16).unwrap();
            middle_stacks.sort_by_key(|(stack, _)| *stack.last().unwrap());
            assert_eq!(
                middle_stacks,
                vec![
                    (vec![root_top, middle_top_a], acc1),
                    (vec![root_top, middle_top_b], acc2),
                ],
            );
            assert_eq!(
                partitions[0].1.to_stacks(16).unwrap(),
                vec![(vec![root_top], acc0)],
            );
            assert_eq!(
                partitions[2].1.to_stacks(16).unwrap(),
                vec![(vec![root_top, leaf_top], acc3)],
            );
            assert_eq!(constraint.recursive_parser_state_span().unwrap(), layout.total_states);
            assert_eq!(layout.links.len(), 2);
            assert!(layout
                .links
                .iter()
                .any(|link| link.parent_component == 1 && link.child_component == 2));
            assert!(layout
                .links
                .iter()
                .any(|link| link.parent_component == 0 && link.child_component == 1));
        }
    }

    #[test]
    fn recursive_parser_reference_executes_nested_calls_without_materialized_child_table() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"[".to_vec()),
            (2, b"a".to_vec()),
            (3, b"]".to_vec()),
            (4, b"!".to_vec()),
        ]);
        let terminal = |constraint: &RuntimeConstraint, name: &str| {
            constraint
                .terminal_display_names
                .iter()
                .position(|candidate| candidate == name)
                .unwrap() as u32
        };
        let leaf = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start leaf; t A = \"a\"; nt leaf = A;"),
            &vocab,
        )
        .unwrap();
        let middle_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start middle; extern grammar leaf; t L = \"[\"; t R = \"]\"; nt middle = L leaf R;",
            ),
            &vocab,
        )
        .unwrap();
        let middle = middle_parent
            .bind_grammar_dynamic_boundary("leaf", &leaf)
            .unwrap();
        let outer_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar middle; t X = \"X\"; t BANG = \"!\"; nt document = X middle BANG;",
            ),
            &vocab,
        )
        .unwrap();
        let bound = outer_parent
            .bind_grammar_dynamic_boundary("middle", &middle)
            .unwrap();

        let outer_overlay = bound.static_dynamic_overlay.as_ref().unwrap();
        let middle_outer_offset = outer_overlay.segmented_parser_components[1].terminal_offset;
        let middle_overlay = middle.static_dynamic_overlay.as_ref().unwrap();
        let middle_parent_offset = middle_overlay.segmented_parser_components[0].terminal_offset;
        let leaf_offset = middle_overlay.segmented_parser_components[1].terminal_offset;
        let terminals = [
            outer_overlay.segmented_parser_components[0].terminal_offset
                + terminal(&outer_parent, "X"),
            middle_outer_offset + middle_parent_offset + terminal(&middle_parent, "L"),
            middle_outer_offset + leaf_offset + terminal(&leaf, "A"),
            middle_outer_offset + middle_parent_offset + terminal(&middle_parent, "R"),
            outer_overlay.segmented_parser_components[0].terminal_offset
                + terminal(&outer_parent, "BANG"),
        ];

        for constraint in [&bound, &RuntimeConstraint::load(&bound.save()).unwrap()] {
            let start = crate::compiler::glr::parser::ParserGSS::from_single_stack(
                vec![0],
                crate::compiler::glr::accumulator::TerminalsDisallowed::new(),
            );
            let mut parser = constraint
                .close_recursive_segmented_parser_reference(&start)
                .unwrap()
                .unwrap();
            for &terminal in &terminals {
                parser = constraint
                    .advance_recursive_segmented_parser_reference(&parser, terminal)
                    .unwrap()
                    .unwrap();
                assert!(
                    !parser.is_empty(),
                    "recursive parser rejected terminal {terminal}",
                );
            }
            assert_eq!(
                constraint
                    .recursive_segmented_parser_is_finished_reference(&parser)
                    .unwrap(),
                Some(true),
            );

            let mut invalid = constraint
                .close_recursive_segmented_parser_reference(&start)
                .unwrap()
                .unwrap();
            for &terminal in &[terminals[0], terminals[1], terminals[3]] {
                invalid = constraint
                    .advance_recursive_segmented_parser_reference(&invalid, terminal)
                    .unwrap()
                    .unwrap();
            }
            assert!(invalid.is_empty(), "recursive parser accepted a missing leaf token");
        }
    }

    #[test]
    fn recursive_parser_reference_resolves_new_link_through_precomposed_parent() {
        let vocab = Vocab::new(vec![
            (0, b"<".to_vec()),
            (1, b"a".to_vec()),
            (2, b"b".to_vec()),
            (3, b">".to_vec()),
        ]);
        let terminal = |constraint: &RuntimeConstraint, name: &str| {
            constraint
                .terminal_display_names
                .iter()
                .position(|candidate| candidate == name)
                .unwrap() as u32
        };
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar left; extern grammar right; t LT = \"<\"; t GT = \">\"; nt document = LT left right GT;",
            ),
            &vocab,
        )
        .unwrap();
        let left = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start left; t A = \"a\"; nt left = A;"),
            &vocab,
        )
        .unwrap();
        let right = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start right; t B = \"b\"; nt right = B;"),
            &vocab,
        )
        .unwrap();
        let half = parent
            .bind_grammar_dynamic_boundary("left", &left)
            .unwrap();
        let full = half
            .bind_grammar_dynamic_boundary("right", &right)
            .unwrap();

        let full_overlay = full.static_dynamic_overlay.as_ref().unwrap();
        let half_offset = full_overlay.segmented_parser_components[0].terminal_offset;
        let right_offset = full_overlay.segmented_parser_components[1].terminal_offset;
        let half_overlay = half.static_dynamic_overlay.as_ref().unwrap();
        let original_parent_offset = half_overlay.segmented_parser_components[0].terminal_offset;
        let left_offset = half_overlay.segmented_parser_components[1].terminal_offset;
        let terminals = [
            half_offset + original_parent_offset + terminal(&parent, "LT"),
            half_offset + left_offset + terminal(&left, "A"),
            right_offset + terminal(&right, "B"),
            half_offset + original_parent_offset + terminal(&parent, "GT"),
        ];

        for constraint in [&full, &RuntimeConstraint::load(&full.save()).unwrap()] {
            let layout = constraint.recursive_parser_layout().unwrap().unwrap();
            assert_eq!(
                layout
                    .leaves
                    .iter()
                    .map(|leaf| leaf.component_path.clone())
                    .collect::<Vec<_>>(),
                vec![vec![0, 0], vec![0, 1], vec![1]],
            );
            assert_eq!(layout.links.len(), 2);
            assert!(layout
                .links
                .iter()
                .any(|link| link.parent_component == 0 && link.child_component == 1));
            assert!(layout
                .links
                .iter()
                .any(|link| link.parent_component == 0 && link.child_component == 2));

            let start = crate::compiler::glr::parser::ParserGSS::from_single_stack(
                vec![0],
                crate::compiler::glr::accumulator::TerminalsDisallowed::new(),
            );
            let mut parser = constraint
                .close_recursive_segmented_parser_reference(&start)
                .unwrap()
                .unwrap();
            for &terminal in &terminals {
                parser = constraint
                    .advance_recursive_segmented_parser_reference(&parser, terminal)
                    .unwrap()
                    .unwrap();
                assert!(!parser.is_empty(), "recursive parser rejected terminal {terminal}");
            }
            assert_eq!(
                constraint
                    .recursive_segmented_parser_is_finished_reference(&parser)
                    .unwrap(),
                Some(true),
            );
        }
    }

    #[test]
    fn loaded_recursive_composition_rebinds_from_lazy_compiler_views() {
        let vocab = Vocab::new(vec![
            (0, b"<".to_vec()),
            (1, b"a".to_vec()),
            (2, b"b".to_vec()),
            (3, b">".to_vec()),
            (4, b"<ab>".to_vec()),
        ]);
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar left; extern grammar right; nt document = \"<\" left right \">\";",
            ),
            &vocab,
        )
        .unwrap();
        let left = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start left; nt left = \"a\";"),
            &vocab,
        )
        .unwrap();
        let right = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start right; nt right = \"b\";"),
            &vocab,
        )
        .unwrap();

        let half = parent
            .bind_grammar_dynamic_boundary("left", &left)
            .unwrap();
        let half_overlay = half.static_dynamic_overlay.as_ref().unwrap();
        let half_layout = half.recursive_parser_layout().unwrap().unwrap();
        assert_eq!(
            half.tokenizer.num_states(),
            half_overlay.segmented_parser_components[0]
                .constraint
                .tokenizer
                .num_states(),
            "recursive coordinator must retain only the root-leaf tokenizer",
        );
        assert!(
            half.tokenizer.num_states() < half_layout.total_tokenizer_states,
            "test fixture must actually contain more than one tokenizer leaf",
        );
        assert_eq!(
            half.table.num_states, 0,
            "recursive coordinator must not retain a flattened LR state machine",
        );
        assert!(half.table.action.is_empty() && half.table.goto.is_empty());
        assert!(half_overlay
            .segmented_parser_components
            .iter()
            .all(|component| component.global_to_local_parser_state.is_empty()));

        let loaded_half = RuntimeConstraint::load_body_artifact(half.save()).unwrap();
        let loaded_half_overlay = loaded_half.static_dynamic_overlay.as_ref().unwrap();
        let loaded_half_layout = loaded_half.recursive_parser_layout().unwrap().unwrap();
        assert_eq!(
            loaded_half.tokenizer.num_states(),
            loaded_half_overlay.segmented_parser_components[0]
                .constraint
                .tokenizer
                .num_states(),
            "loaded recursive coordinator must not reconstruct the outer union tokenizer eagerly",
        );
        assert!(loaded_half.tokenizer.num_states() < loaded_half_layout.total_tokenizer_states);
        assert_eq!(
            loaded_half.table.num_states, 0,
            "loaded recursive coordinator must keep the flattened parser table lazy",
        );
        assert!(loaded_half.table.action.is_empty() && loaded_half.table.goto.is_empty());
        assert!(loaded_half_overlay
            .segmented_parser_components
            .iter()
            .all(|component| component.global_to_local_parser_state.is_empty()));

        let fresh_full = half
            .bind_grammar_dynamic_boundary("right", &right)
            .unwrap();
        let loaded_full = loaded_half
            .bind_grammar_dynamic_boundary("right", &right)
            .unwrap();
        for constraint in [&fresh_full, &loaded_full] {
            assert_recursive_compiler_views_detached(constraint);
        }
        let fresh_start = crate::compiler::glr::parser::ParserGSS::from_single_stack(
            vec![0],
            crate::compiler::glr::accumulator::TerminalsDisallowed::new(),
        );
        let loaded_start = fresh_start.clone();
        let fresh_closed = fresh_full
            .close_compact_segmented_parser(&fresh_start)
            .unwrap();
        let loaded_closed = loaded_full
            .close_compact_segmented_parser(&loaded_start)
            .unwrap();
        assert!(fresh_closed.semantically_eq(&loaded_closed, 4096).unwrap());
        for terminal in 0..fresh_full.table.num_terminals {
            let fresh_advanced = fresh_full
                .advance_compact_segmented_parser(&fresh_closed, terminal)
                .unwrap();
            let loaded_advanced = loaded_full
                .advance_compact_segmented_parser(&loaded_closed, terminal)
                .unwrap();
            assert!(
                fresh_advanced
                    .semantically_eq(&loaded_advanced, 4096)
                    .unwrap(),
                "recursive parser advance differs for terminal {terminal}",
            );
        }
        for constraint in [&fresh_full, &loaded_full] {
            let overlay = constraint.static_dynamic_overlay.as_ref().unwrap();
            assert!(overlay
                .segmented_parser_components
                .iter()
                .all(|component| component.global_to_local_parser_state.is_empty()));
            let mut split = constraint.start();
            for token in [0, 1, 2, 3] {
                assert_ne!(split.mask()[0] & (1 << token), 0);
                split.commit_token(token).unwrap();
            }
            assert!(split.is_accepting());

            let mut fused = constraint.start();
            assert_ne!(fused.mask()[0] & (1 << 4), 0);
            fused.commit_token(4).unwrap();
            assert!(fused.is_accepting());
        }
    }

    #[test]
    fn recursive_parser_reference_preserves_nested_nullable_wrapper_return() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
        ]);
        let terminal = |constraint: &RuntimeConstraint, name: &str| {
            constraint
                .terminal_display_names
                .iter()
                .position(|candidate| candidate == name)
                .unwrap() as u32
        };
        let leaf = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start leaf; t A = \"a\"; nt leaf = A?;"),
            &vocab,
        )
        .unwrap();
        let middle_parent = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start middle; extern grammar leaf; nt middle = leaf;"),
            &vocab,
        )
        .unwrap();
        assert!(!middle_parent.table.embedded_start_nullable());
        let middle = middle_parent
            .bind_grammar_dynamic_boundary("leaf", &leaf)
            .unwrap();
        assert!(middle.table.embedded_start_nullable());
        let outer_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar middle; t X = \"X\"; t BANG = \"!\"; nt document = X middle BANG;",
            ),
            &vocab,
        )
        .unwrap();
        let bound = outer_parent
            .bind_grammar_dynamic_boundary("middle", &middle)
            .unwrap();

        let outer_overlay = bound.static_dynamic_overlay.as_ref().unwrap();
        let middle_outer_offset = outer_overlay.segmented_parser_components[1].terminal_offset;
        let middle_overlay = middle.static_dynamic_overlay.as_ref().unwrap();
        let leaf_offset = middle_overlay.segmented_parser_components[1].terminal_offset;
        let x = outer_overlay.segmented_parser_components[0].terminal_offset
            + terminal(&outer_parent, "X");
        let a = middle_outer_offset + leaf_offset + terminal(&leaf, "A");
        let bang = outer_overlay.segmented_parser_components[0].terminal_offset
            + terminal(&outer_parent, "BANG");

        for constraint in [&bound, &RuntimeConstraint::load(&bound.save()).unwrap()] {
            let layout = constraint.recursive_parser_layout().unwrap().unwrap();
            let outer_link = layout
                .links
                .iter()
                .find(|link| link.parent_component == 0 && link.child_component == 1)
                .expect("outer link must target the middle wrapper root leaf");
            assert!(outer_link.child_start_nullable);

            for terminals in [&[x, bang][..], &[x, a, bang][..]] {
                let start = crate::compiler::glr::parser::ParserGSS::from_single_stack(
                    vec![0],
                    crate::compiler::glr::accumulator::TerminalsDisallowed::new(),
                );
                let mut parser = constraint
                    .close_recursive_segmented_parser_reference(&start)
                    .unwrap()
                    .unwrap();
                for &terminal in terminals {
                    parser = constraint
                        .advance_recursive_segmented_parser_reference(&parser, terminal)
                        .unwrap()
                        .unwrap();
                    assert!(!parser.is_empty(), "recursive nullable parser rejected {terminals:?}");
                }
                assert_eq!(
                    constraint
                        .recursive_segmented_parser_is_finished_reference(&parser)
                        .unwrap(),
                    Some(true),
                    "{terminals:?}",
                );
            }
        }
    }

    #[test]
    fn static_boundary_shards_are_authoritative_across_multiple_internal_crossings() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"z".to_vec()),
            (3, b"xy".to_vec()),
            (4, b"yz".to_vec()),
            (5, b"xyz".to_vec()),
        ]);
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar left; extern grammar right; \
                 nt start = \"x\" left right;",
            ),
            &vocab,
        )
        .unwrap();
        let left = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap();
        let right = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "z""#), &vocab)
            .unwrap();
        let bound = parent
            .bind_grammar("left", left)
            .unwrap()
            .bind_grammar("right", right)
            .unwrap();

        let overlay = bound
            .static_dynamic_overlay
            .as_ref()
            .expect("static composition must retain segmented A+B metadata");
        assert!(
            !overlay.segmented_boundary_shards.is_empty(),
            "static composition must publish component-scoped B shards",
        );
        assert!(
            overlay.segmented_boundary_parser.is_none(),
            "partitioned static B must not retain a redundant global boundary parser",
        );
        assert!(overlay.segmented_boundary_terminal_trie.is_none());

        let loaded = RuntimeConstraint::load(bound.save()).unwrap();
        let loaded_overlay = loaded.static_dynamic_overlay.as_ref().unwrap();
        assert!(loaded_overlay.segmented_boundary_parser.is_none());
        assert!(loaded_overlay.segmented_boundary_terminal_trie.is_none());
        assert!(!loaded_overlay.segmented_boundary_shards.is_empty());
        let reference = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "x" "y" "z""#),
            &vocab,
        )
        .unwrap();

        for tokens in [&[5][..], &[0, 4][..], &[0, 1, 2][..]] {
            let mut sharded = bound.start();
            let mut restored = loaded.start();
            let mut expected = reference.start();
            for &token in tokens {
                assert_eq!(
                    sharded.mask(),
                    expected.mask(),
                    "partitioned static B mismatch before {tokens:?} token {token}",
                );
                assert_eq!(
                    restored.mask(),
                    expected.mask(),
                    "restored static shards differ before {tokens:?} token {token}",
                );
                sharded.commit_token(token).unwrap();
                restored.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert_eq!(sharded.is_accepting(), expected.is_accepting(), "{tokens:?}");
            assert_eq!(restored.is_accepting(), expected.is_accepting(), "{tokens:?}");
            assert!(sharded.is_accepting(), "{tokens:?}");
        }
    }

    #[test]
    fn authoritative_ab_keeps_component_parser_dwas_unchanged_across_scoped_ignores() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b" ".to_vec()),
            (2, b"\t".to_vec()),
            (3, b"a".to_vec()),
            (4, b"!".to_vec()),
            (5, b"X \ta!".to_vec()),
            (6, b"X\t a!".to_vec()),
            (7, b"Xa\t !".to_vec()),
            (8, b"Xa \t!".to_vec()),
        ]);
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"
                    glrm 1;
                    start document;
                    ignore PARENT_WS;
                    t PARENT_WS = " "+;
                    extern grammar child;
                    nt document = "X" child "!";
                "#,
            ),
            &vocab,
        )
        .unwrap();
        let child = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"
                    glrm 1;
                    start child;
                    ignore CHILD_WS;
                    t CHILD_WS = "\t"+;
                    nt child = "a";
                "#,
            ),
            &vocab,
        )
        .unwrap();
        assert!(
            !parent.composition_reset_tokens_by_terminal.is_empty(),
            "late-bind parent compilation should precompute reset-token composition metadata",
        );
        let mut cached_parent = RuntimeConstraint::load_body_artifact(parent.save()).unwrap();
        cached_parent
            .materialize_composition_metadata_for_compilation()
            .unwrap();
        assert_eq!(
            cached_parent.composition_reset_tokens_by_terminal,
            parent.composition_reset_tokens_by_terminal,
            "late-bind reset-token cache must survive save/load",
        );
        let monolithic = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"
                    glrm 1;
                    start document;
                    ignore PARENT_WS;
                    t PARENT_WS = " "+;
                    g child = {
                        start child;
                        ignore CHILD_WS;
                        t CHILD_WS = "\t"+;
                        nt child = "a";
                    };
                    nt document = "X" child "!";
                "#,
            ),
            &vocab,
        )
        .unwrap();

        let parent_dwa = parent.parser_dwa.clone();
        let child_dwa = child.parser_dwa.clone();
        let bound = parent
            .bind_grammar_dynamic_boundary("child", child.clone())
            .unwrap();
        let overlay = bound
            .static_dynamic_overlay
            .as_ref()
            .expect("authoritative A+B metadata");
        assert!(overlay.segmented_mask_authoritative);
        assert!(!overlay.segmented_static_baseline);
        assert_eq!(overlay.segmented_parser_components.len(), 2);
        assert_eq!(overlay.segmented_parser_components[0].constraint.parser_dwa, parent_dwa);
        assert_eq!(overlay.segmented_parser_components[1].constraint.parser_dwa, child_dwa);
        assert!(overlay
            .segmented_parser_components
            .iter()
            .all(|component| component.root_disallowed_terminal.is_none()));

        let loaded = RuntimeConstraint::load(bound.save()).unwrap();
        assert!(bound.uses_compact_segmented_parser_runtime());
        assert!(loaded.uses_compact_segmented_parser_runtime());
        let loaded_overlay = loaded
            .static_dynamic_overlay
            .as_ref()
            .expect("round-tripped authoritative A+B metadata");
        assert_eq!(loaded_overlay.segmented_parser_components.len(), 2);
        let mut loaded_parent = loaded_overlay.segmented_parser_components[0]
            .constraint
            .as_ref()
            .clone();
        let mut loaded_child = loaded_overlay.segmented_parser_components[1]
            .constraint
            .as_ref()
            .clone();
        loaded_parent.materialize_parser_dwa_for_compilation().unwrap();
        loaded_child.materialize_parser_dwa_for_compilation().unwrap();
        assert_eq!(loaded_parent.parser_dwa, parent_dwa);
        assert_eq!(loaded_child.parser_dwa, child_dwa);
        for (kind, constraint) in [("source", &bound), ("loaded", &loaded)] {
            let mut actual = constraint.start();
            let mut expected = monolithic.start();
            assert_eq!(actual.mask(), expected.mask(), "{kind} initial mask");
            for token in [1, 0, 2, 3, 1, 4] {
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
                assert_eq!(actual.mask(), expected.mask(), "after token {token}");
            }
            assert!(actual.is_accepting());
            assert!(expected.is_accepting());

            for token in [5, 6, 7, 8] {
                let mut actual = constraint.start();
                let mut expected = monolithic.start();
                assert_eq!(
                    actual.commit_token(token).is_ok(),
                    expected.commit_token(token).is_ok(),
                    "fused token {token}",
                );
                assert_eq!(actual.is_accepting(), expected.is_accepting());
            }
        }
    }

    #[test]
    fn authoritative_ab_handles_multi_terminal_parent_token_after_zero_width_return() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
            (3, b"?".to_vec()),
            (4, b"!?".to_vec()),
        ]);
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"
                    glrm 1;
                    start document;
                    extern grammar child;
                    nt document = "X" child "!" "?";
                "#,
            ),
            &vocab,
        )
        .unwrap();
        let child = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start child; nt child = \"a\";"),
            &vocab,
        )
        .unwrap();
        let monolithic = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"
                    glrm 1;
                    start document;
                    g child = { start child; nt child = "a"; };
                    nt document = "X" child "!" "?";
                "#,
            ),
            &vocab,
        )
        .unwrap();

        let bound = parent
            .bind_grammar_dynamic_boundary("child", child)
            .unwrap();
        for constraint in [&bound, &RuntimeConstraint::load(bound.save()).unwrap()] {
            let mut actual = constraint.start();
            let mut expected = monolithic.start();
            for token in [0, 1] {
                assert_eq!(actual.mask(), expected.mask(), "before token {token}");
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            let actual_mask = actual.mask();
            let expected_mask = expected.mask();
            assert_eq!(actual_mask, expected_mask, "after child completion");
            assert_ne!(actual_mask[0] & (1 << 4), 0, "fused parent token !? must be allowed");
            actual.commit_token(4).unwrap();
            expected.commit_token(4).unwrap();
            assert!(actual.is_accepting());
            assert!(expected.is_accepting());
        }
    }

    #[test]
    fn dynamic_late_bind_vocab_cache_is_shared_reused_and_not_serialized() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()), (1, b"y".to_vec()), (2, b"xy".to_vec()),
        ]);
        let parent = DynamicConstraint::compile(
            Grammar::glrm("glrm 1; start start; extern grammar child; nt start = \"x\" child;"),
            &vocab,
        ).unwrap();
        let child = DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab).unwrap();
        let retained = parent.inner.late_bind_vocab.get().expect("retain supplied vocabulary");
        assert!(Arc::ptr_eq(&retained.entries_arc(), &vocab.entries_arc()));

        let loaded = DynamicConstraint::load(&parent.save()).unwrap();
        assert!(loaded.inner.late_bind_vocab.get().is_none());
        let first = loaded.bind_grammar_dynamic_boundary("child", &child).unwrap();
        let first_backing = loaded.inner.late_bind_vocab.get()
            .expect("prime retained loaded parent, not a throwaway clone").entries_arc();
        let second = loaded.bind_grammar_dynamic_boundary("child", &child).unwrap();
        assert!(Arc::ptr_eq(
            &first_backing, &loaded.inner.late_bind_vocab.get().unwrap().entries_arc(),
        ));
        assert_eq!(first.start().mask(), second.start().mask());
        let reloaded = DynamicConstraint::load(&loaded.save()).unwrap();
        assert!(reloaded.inner.late_bind_vocab.get().is_none(), "cache stays off wire");
        let rebound = reloaded.bind_grammar_dynamic_boundary("child", &child).unwrap();
        assert_eq!(first.start().mask(), rebound.start().mask());
    }

    #[test]
    fn late_bind_vocab_cache_is_reused_and_not_serialized() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
        ]);
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
            ),
            &vocab,
        )
        .unwrap();
        let child =
            DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab).unwrap();

        assert!(
            parent.late_bind_vocab.get().is_some(),
            "fresh compilation should retain the supplied vocabulary for later binds",
        );

        let loaded_parent = RuntimeConstraint::load_body_artifact(parent.save()).unwrap();
        assert!(
            loaded_parent.late_bind_vocab.get().is_none(),
            "late-bind vocabulary memoization is runtime-only and must not enter the wire format",
        );
        let first = loaded_parent
            .bind_grammar_dynamic_boundary("child", &child)
            .unwrap();
        assert!(loaded_parent.late_bind_vocab.get().is_some());
        let second = loaded_parent
            .bind_grammar_dynamic_boundary("child", &child)
            .unwrap();
        assert_eq!(first.start().mask(), second.start().mask());

        let loaded = RuntimeConstraint::load_body_artifact(loaded_parent.save()).unwrap();
        assert!(
            loaded.late_bind_vocab.get().is_none(),
            "late-bind vocabulary memoization is runtime-only and must not enter the wire format",
        );
        let rebound = loaded
            .bind_grammar_dynamic_boundary("child", &child)
            .unwrap();
        assert_eq!(first.start().mask(), rebound.start().mask());
    }
}


#[cfg(test)]
mod final_unlinked_optimization_tests {
    use super::*;

    fn assert_static_boundary(constraint: &RuntimeConstraint) {
        let overlay = constraint.static_dynamic_overlay.as_ref().unwrap();
        assert!(!overlay.segmented_boundary_shards.is_empty());
        assert!(overlay.segmented_boundary_shards.iter().all(|shard| matches!(
            shard.backend,
            crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
        )));
    }

    fn assert_dynamic_boundary(constraint: &RuntimeConstraint) {
        let overlay = constraint.static_dynamic_overlay.as_ref().unwrap();
        assert!(!overlay.segmented_boundary_shards.is_empty());
        assert!(overlay.segmented_boundary_shards.iter().all(|shard| matches!(
            shard.backend,
            crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
        )));
    }

    #[test]
    fn unlinked_bind_defers_composition_until_link_and_link_optimization_selects_boundary() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
        ]);
        let host = Grammar::glrm(
            r#"glrm 1; start start; extern grammar child; nt start = "x" child;"#,
        )
        .compile_unlinked(&vocab)
        .unwrap();
        let child = Grammar::ebnf(r#"start ::= "y""#)
            .compile(&vocab)
            .unwrap();

        let bound = host.bind("child", &child).unwrap();
        assert!(bound.inner.late_grammar_slots.iter().any(|slot| slot.name == "child"));
        assert!(bound.inner.static_dynamic_overlay.is_none(), "bind must not compile a boundary eagerly");

        let fast_build = bound.link_with(
            BuildOptions::default().optimization(Optimization::FastBuild),
        ).unwrap();
        let fast_runtime = bound.link_with(
            BuildOptions::default().optimization(Optimization::FastRuntime),
        ).unwrap();
        assert_dynamic_boundary(&fast_build);
        assert_static_boundary(&fast_runtime);
        assert_eq!(fast_build.start().mask(), fast_runtime.start().mask());

        let loaded = UnlinkedConstraint::load(bound.save()).unwrap();
        assert!(loaded.inner.late_grammar_slots.iter().any(|slot| slot.name == "child"));
        assert_dynamic_boundary(&loaded.link_with(
            BuildOptions::default().optimization(Optimization::FastBuild),
        ).unwrap());
        assert_static_boundary(&loaded.link_with(
            BuildOptions::default().optimization(Optimization::FastRuntime),
        ).unwrap());
    }
}

#[cfg(test)]
mod cached_parent_main_tests {
    use super::*;

    fn vocab() -> Vocab {
        Vocab::new(vec![
            (0, b"<".to_vec()),
            (1, b">".to_vec()),
            (2, b"[".to_vec()),
            (3, b"]".to_vec()),
            (4, b"a".to_vec()),
            (5, b"b".to_vec()),
            (6, b"x".to_vec()),
            (7, b"<a>".to_vec()),
            (8, b"<b>".to_vec()),
            (9, b"<[a]>".to_vec()),
        ])
    }

    fn accepts(constraint: &RuntimeConstraint, bytes: &[u8]) -> bool {
        let mut state = constraint.start();
        state.commit_bytes(bytes).is_ok() && state.is_accepting()
    }

    #[test]
    fn compiled_parent_can_bind_external_grammar_after_compile_and_load() {
        let vocab = vocab();
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"
                    glrm 1;
                    extern grammar payload;
                    start document;
                    nt document = "x" | "<" payload ">";
                "#,
            ),
            &vocab,
        )
        .unwrap();
        assert_eq!(
            parent
                .late_grammar_slots
                .iter()
                .map(|slot| slot.name.as_str())
                .collect::<Vec<_>>(),
            vec!["payload"],
        );
        assert!(accepts(&parent, b"x"));
        assert!(!accepts(&parent, b"<a>"));

        let child_a = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start value; nt value = \"a\";"),
            &vocab,
        )
        .unwrap();
        let child_b = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start value; nt value = \"b\";"),
            &vocab,
        )
        .unwrap();

        let with_a = parent.bind_grammar("payload", &child_a).unwrap();
        let with_b = parent.bind_grammar("payload", &child_b).unwrap();
        assert!(accepts(&with_a, b"x"));
        assert!(accepts(&with_a, b"<a>"));
        assert!(!accepts(&with_a, b"<b>"));
        assert!(accepts(&with_b, b"<b>"));
        assert!(!accepts(&with_b, b"<a>"));
        assert!(parent
            .late_grammar_slots
            .iter()
            .any(|slot| slot.name == "payload"));

        let saved = parent.save();
        let loaded = RuntimeConstraint::load_body_artifact(&saved).unwrap();
        let loaded_with_a = loaded.bind_grammar("payload", &child_a).unwrap();
        assert!(accepts(&loaded_with_a, b"<a>"));
        assert!(!accepts(&loaded_with_a, b"<b>"));
    }

    #[test]
    fn compiled_parent_can_fill_multiple_slots_incrementally() {
        let vocab = vocab();
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"
                    glrm 1;
                    extern grammar left;
                    extern grammar right;
                    start document;
                    nt document = "<" left right ">";
                "#,
            ),
            &vocab,
        )
        .unwrap();
        let a = RuntimeConstraint::compile(
            Grammar::glrm(r#"glrm 1; start value; nt value = "a";"#),
            &vocab,
        )
        .unwrap();
        let b = RuntimeConstraint::compile(
            Grammar::glrm(r#"glrm 1; start value; nt value = "b";"#),
            &vocab,
        )
        .unwrap();
        let half = parent.bind_grammar_dynamic_boundary("left", &a).unwrap();
        assert!(half
            .late_grammar_slots
            .iter()
            .any(|slot| slot.name == "right"));
        let full = half.bind_grammar_dynamic_boundary("right", &b).unwrap();
        assert!(full.late_grammar_slots.is_empty());
        assert!(accepts(&full, b"<ab>"));

        let unresolved_child = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; extern grammar leaf; start value; nt value = leaf;",
            ),
            &vocab,
        )
        .unwrap();
        // Main used to reject this because independently compiled unresolved
        // children could collide in the private placeholder-token namespace.
        // The successor architecture transports those slots by qualified name
        // and sanitizes the private linker token from the public token domain.
        let open_left = parent
            .bind_grammar_dynamic_boundary("left", &unresolved_child)
            .unwrap();
        assert!(open_left
            .late_grammar_slots
            .iter()
            .any(|slot| slot.name == "left.leaf"));
        let left = open_left.bind_grammar_dynamic_boundary("left.leaf", &a).unwrap();
        let nested_full = left.bind_grammar_dynamic_boundary("right", &b).unwrap();
        assert!(nested_full.late_grammar_slots.is_empty());
        assert!(accepts(&nested_full, b"<ab>"));
    }

    #[test]
    fn dynamic_compile_preserves_unbound_external_grammar() {
        let vocab = vocab();
        let open = DynamicConstraint::compile(
            Grammar::glrm(
                "glrm 1; extern grammar payload; start document; nt document = payload;",
            ),
            &vocab,
        )
        .unwrap();
        assert!(open
            .clone_constraints()
            .iter()
            .all(|alternative| alternative
                .late_grammar_slots
                .iter()
                .any(|slot| slot.name == "payload")));
        let child = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start value; nt value = \"a\";"),
            &vocab,
        )
        .unwrap();
        let bound = open.bind_grammar("payload", &child).unwrap();
        let mut state = bound.start();
        state.commit_bytes(b"a").unwrap();
        assert!(state.is_accepting());
    }

    #[test]
    fn final_optimization_dispatch_selects_dynamic_build_and_static_runtime_paths() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
        ]);
        let grammar = Grammar::ebnf(r#"start ::= "a" "b""#);
        let fast_build = grammar
            .compile_with(
                &vocab,
                BuildOptions::default().optimization(Optimization::FastBuild),
            )
            .unwrap();
        let fast_runtime = grammar
            .compile_with(
                &vocab,
                BuildOptions::default().optimization(Optimization::FastRuntime),
            )
            .unwrap();
        assert!(fast_build.uses_dynamic_runtime());
        assert!(!fast_runtime.uses_dynamic_runtime());

        let mut build_state = fast_build.start();
        let mut runtime_state = fast_runtime.start();
        assert_eq!(build_state.mask(), runtime_state.mask());
        build_state.commit_token(2).unwrap();
        runtime_state.commit_token(2).unwrap();
        assert_eq!(build_state.is_accepting(), runtime_state.is_accepting());

        let loaded = RuntimeConstraint::load(fast_build.save()).unwrap();
        assert!(loaded.uses_dynamic_runtime());
        assert_eq!(loaded.start().mask(), fast_runtime.start().mask());
    }
}

impl RuntimeConstraint {
    /// Inspect the actual stored parser backend, not a process-wide switch.
    pub fn parser_backend(&self) -> ParserBackend {
        if self.has_template_parser() {
            assert!(!self.table.is_present(), "template parser retained LR storage");
            ParserBackend::TemplateDfa
        } else { ParserBackend::LrTable }
    }
}
