//! Immutable source grammar descriptions and source-only bindings.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use rayon::prelude::*;
use crate::runtime::Constraint as RuntimeConstraint;
use crate::{Error, Result, Vocab};
use super::{
    BuildOptions, ConstraintSpec, ExternKind, GrammarValue, IntoGrammarValue, ModuleBinding,
    UnlinkedConstraint, ensure_runnable_constraint,
};

/// Grammar description with immutable source-grammar or exact-token bindings.
#[derive(Debug, Clone)]
pub struct Grammar<'a> {
    pub(super) source: GrammarSource<'a>,
    pub(super) bindings: BTreeMap<String, GrammarValue<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GrammarSource<'a> {
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

    pub(super) fn new(source: GrammarSource<'a>) -> Self {
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
        let mut spec = ConstraintSpec::builder(self.clone(), vocab)?.build()?;
        spec.automatic_boundary_selection = true;
        if let Some(name) = spec.unbound_grammar_names.first() {
            return Err(Error::Compilation(format!(
                "external grammar {name:?} is unbound; compile_unlinked() if a reusable pre-link artifact is intended",
            )));
        }
        let constraint = spec.compile_final(options.optimization_value())?;
        ensure_runnable_constraint(&constraint)?;
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

    pub(super) fn unresolved_token_names(&self) -> Result<BTreeSet<String>> {
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

    pub(super) fn glrm_source(&self) -> Option<&'a str> {
        match self.source {
            GrammarSource::Glrm(source) => Some(source),
            _ => None,
        }
    }

    pub(super) fn into_source_only_and_bindings(self) -> (Self, BTreeMap<String, GrammarValue<'a>>) {
        let Self { source, bindings } = self;
        (Self::new(source), bindings)
    }
}
