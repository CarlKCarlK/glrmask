//! Resolve source/bindings into static or dynamic compiled components.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;
use rayon::prelude::*;
use crate::compiler::composition::{
    CompiledSubgrammarInput, SegmentedBoundaryBackend,
    compose_constraints_owned_parent_segmented_shared,
};
use crate::runtime::Constraint as RuntimeConstraint;
use crate::runtime::dynamic::DynamicConstraint;
use crate::runtime::BoundaryTriggerDetail;
use crate::{Error, Result, Vocab};
use super::{
    ExternKind, Grammar, GrammarBinding, GrammarSource, GrammarValue, IntoGrammarBinding,
    Optimization, constraint_vocab, select_supported_boundary,
};

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
    pub(super) grammar: Grammar<'a>,
    pub(super) vocab: &'a Vocab,
    pub(super) token_bindings: BTreeMap<String, Vec<u32>>,
    pub(super) grammar_bindings: BTreeMap<String, GrammarBinding<'a>>,
    pub(super) unbound_grammar_names: Vec<String>,
    // Only explicit module compilation may preserve unbound source token
    // slots. Legacy specification compilation keeps its validation contract.
    pub(super) allow_open_source_tokens: bool,
    pub(super) automatic_boundary_selection: bool,
    // These belong to this source component. Register their terminal identity
    // before linking: another component may independently use the same hidden
    // token ID, which is not a globally unique linker coordinate.
    pub(super) open_token_placeholders: BTreeMap<String, u32>,
    pub(super) boundary_trigger_detail: BoundaryTriggerDetail,
    pub(super) boundary_summary_policy: BoundarySummaryPolicy,
}

/// Builder for [`ConstraintSpec`].
#[derive(Debug)]
pub struct ConstraintSpecBuilder<'a> {
    pub(super) grammar: Grammar<'a>,
    pub(super) vocab: &'a Vocab,
    pub(super) declared_tokens: BTreeSet<String>,
    pub(super) declared_grammars: BTreeSet<String>,
    pub(super) token_bindings: BTreeMap<String, Vec<u32>>,
    pub(super) grammar_bindings: BTreeMap<String, GrammarBinding<'a>>,
    pub(super) boundary_trigger_detail: BoundaryTriggerDetail,
    pub(super) boundary_summary_policy: BoundarySummaryPolicy,
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
                crate::compiler::composition::boundary::candidates::prepare_boundary_candidate_summary(
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

    pub(super) fn compile_final(&self, optimization: Optimization) -> Result<RuntimeConstraint> {
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

    pub(super) fn compile_static_with_trigger_uncached(&self) -> Result<RuntimeConstraint> {
        let mut constraint = self.compile_static_uncached()?;
        constraint
            .build_boundary_trigger(self.boundary_trigger_detail)
            .map_err(Error::Compilation)?;
        Ok(constraint)
    }

    pub(super) fn register_open_token_placeholders(&self, parent: &mut RuntimeConstraint) -> Result<()> {
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

    pub(super) fn compile_static_uncached(&self) -> Result<RuntimeConstraint> {
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
                    crate::compiler::composition::boundary::candidates::prepare_boundary_candidate_summary(
                        component,
                        self.vocab,
                    );
                }
            }
        }
        Ok(constraint)
    }

    pub(super) fn compile_dynamic_uncached(&self) -> Result<DynamicConstraint> {
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

    pub(super) fn token_binding_refs(&self) -> Vec<(&str, &[u32])> {
        self.token_bindings
            .iter()
            .map(|(name, ids)| (name.as_str(), ids.as_slice()))
            .collect()
    }

    pub(super) fn compile_children(
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

    pub(super) fn targets(&self, vocab: &Vocab) -> bool {
        self.vocab.same_model_vocab(vocab)
    }
}

impl<'a> ConstraintSpecBuilder<'a> {
    pub(super) fn new(grammar: Grammar<'a>, vocab: &'a Vocab) -> Result<Self> {
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

    pub(super) fn require_kind(&self, name: &str, expected: ExternKind) -> Result<()> {
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
pub(super) enum ChildCompileMode {
    Static,
    Dynamic,
}

pub(super) enum CompiledChild<'a> {
    StaticBorrowed(&'a RuntimeConstraint),
    StaticOwned(RuntimeConstraint),
    DynamicBorrowed(&'a DynamicConstraint),
    DynamicOwned(DynamicConstraint),
}

impl CompiledChild<'_> {
    pub(super) fn into_constraints(self) -> Vec<RuntimeConstraint> {
        match self {
            Self::StaticBorrowed(constraint) => vec![constraint.clone()],
            Self::StaticOwned(constraint) => vec![constraint],
            Self::DynamicBorrowed(constraint) => constraint.clone_constraints(),
            Self::DynamicOwned(constraint) => constraint.clone_constraints(),
        }
    }
}

pub(super) fn static_constraint_targets(constraint: &RuntimeConstraint, vocab: &Vocab) -> bool {
    constraint
        .late_bind_vocab
        .get()
        .map_or_else(
            || constraint.token_bytes_match_vocab(vocab),
            |compiled_vocab| compiled_vocab.same_model_vocab(vocab),
        )
}

pub(super) fn prepare_compiled_children(
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

pub(super) fn collapse_dynamic_alternatives(
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

pub(super) fn compose_named_children(
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

pub(super) fn compile_static_source(
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

pub(super) fn compile_dynamic_source(
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
        crate::compiler::composition::boundary::candidates::persist_boundary_candidate_summary(self, vocab);
        if std::env::var_os("GLRMASK_PREPARE_BOUNDARY_COMPLETION").is_some() {
            crate::compiler::composition::boundary::precomputed_completion::prepare_component(self, vocab)
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

pub(super) const MODULE_MAGIC: &[u8; 8] = b"GLRMOD03";

pub(super) const MODULE_HEADER_LEN: usize = 16;

pub(super) const FIRST_SAVE_TEMPLATE_STATE_PRIME_THRESHOLD: usize = 100_000;

pub(super) fn should_prime_first_save_artifact(constraint: &RuntimeConstraint) -> bool {
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

pub(super) fn retains_dynamic_component(constraint: &RuntimeConstraint) -> bool {
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

pub(super) fn require_late_grammar_slot(
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

pub(super) fn prepare_late_child<'a, T>(
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

pub(super) fn bind_static_parent_grammar<'a, T>(
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

pub(super) fn bind_dynamic_parent_grammar<'a, T>(
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
