//! Reusable compiled graphs, immutable attachments, linking, and persistence.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use rayon::prelude::*;
use crate::compiler::composition::{SegmentedBoundaryBackend};
use crate::runtime::Constraint as RuntimeConstraint;
use crate::{Error, Result, Vocab};
use super::{
    BuildOptions, CompiledChild, ExternKind, IntoUnlinkedValue, MODULE_HEADER_LEN, MODULE_MAGIC,
    Optimization, UnlinkedValue, compose_named_children, prepare_compiled_children,
    static_constraint_targets,
};

/// Reusable, vocabulary-specific compiled constraint in pre-link form.
///
/// An unlinked constraint may retain unresolved external slots. It is
/// deliberately not runnable; bind already-compiled child [`crate::Constraint`]s and
/// call [`UnlinkedConstraint::link`] once all required slots are satisfied.
#[derive(Debug, Clone)]
pub struct UnlinkedConstraint {
    pub(super) inner: Arc<RuntimeConstraint>,
    /// Direct token slots in this component. The runtime linker slots keep
    /// both token and grammar terminals; this manifest preserves their public
    /// kind while bindings stay deferred until the final link operation.
    pub(super) token_slots: BTreeSet<String>,
    /// Compiled-only immutable attachments. Delaying composition is what lets
    /// the terminal link call choose the boundary/runtime trade-off.
    pub(super) bindings: BTreeMap<String, ModuleBinding>,
}

#[derive(Debug, Clone)]
pub(super) enum ModuleBinding {
    Module(Box<UnlinkedConstraint>),
    Constraint(Arc<RuntimeConstraint>),
    ExactTokens(Arc<[u32]>),
}

pub(super) fn constraint_vocab(constraint: &RuntimeConstraint) -> Vocab {
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

pub(super) fn ensure_runnable_constraint(constraint: &RuntimeConstraint) -> Result<()> {
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
pub(super) fn select_supported_boundary(
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

pub(super) fn compile_exact_token_adapter(vocab: &Vocab, token_ids: &[u32]) -> Result<RuntimeConstraint> {
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

pub(super) fn boundary_for_optimization(
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
pub(super) struct ModuleArtifactManifest {
    pub(super) exact_only_token_ids: Vec<u32>,
    pub(super) token_slots: Vec<String>,
    pub(super) bindings: Vec<(String, ModuleBindingArtifact)>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub(super) enum ModuleBindingArtifact {
    Module(Vec<u8>),
    Constraint(Vec<u8>),
    ExactTokens(Vec<u32>),
}

impl UnlinkedConstraint {
    pub(super) fn direct_slot_names(&self) -> BTreeSet<&str> {
        self.inner
            .late_grammar_slots
            .iter()
            .map(|slot| slot.name.as_str())
            .collect()
    }

    pub(super) fn open_token_names(&self) -> Result<BTreeSet<String>> {
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

    pub(super) fn first_open_slot(&self) -> Result<Option<(String, ExternKind)>> {
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

    pub(super) fn targets_vocab(&self, vocab: &Vocab) -> bool {
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

    pub(super) fn validate_slot_manifest(&self) -> Result<()> {
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

    pub(super) fn require_direct_slot_kind(&self, name: &str, expected: ExternKind) -> Result<()> {
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

    pub(super) fn bind_value(&self, name: &str, value: UnlinkedValue<'_>) -> Result<Self> {
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

    pub(super) fn materialize(&self, optimization: Optimization) -> Result<RuntimeConstraint> {
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
        if let Some((name, kind)) = self.first_open_slot()? {
            return Err(Error::Compilation(format!(
                "external {} {name:?} is still unbound",
                kind.name(),
            )));
        }
        let constraint = self.materialize(options.optimization_value())?;
        ensure_runnable_constraint(&constraint)?;
        constraint.with_end_tokens(options.end_token_ids())
    }

    pub(super) fn artifact_manifest(&self) -> ModuleArtifactManifest {
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

    pub(super) fn from_artifact_parts(
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

    pub(super) fn bind_vocab_recursive(&mut self, vocab: &Vocab) -> Result<()> {
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
