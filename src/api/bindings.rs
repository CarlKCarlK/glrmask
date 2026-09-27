//! Typed binding conversions and slot-kind validation.

use std::sync::Arc;
use crate::runtime::Constraint as RuntimeConstraint;
use crate::runtime::dynamic::DynamicConstraint;
use crate::{Error, ExactToken, ExactTokens, Result, Vocab};
use super::{
    ChildCompileMode, CompiledChild, ConstraintSpec, Grammar, Optimization,
    static_constraint_targets,
};

#[derive(Debug, Clone)]
pub(crate) enum GrammarValue<'a> {
    Source(Grammar<'a>),
    ExactToken(ExactToken),
    ExactTokens(ExactTokens),
}

impl GrammarValue<'_> {
    pub(super) fn extern_kind(&self) -> ExternKind {
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

#[derive(Clone, Copy)]
pub(super) enum ExternKind {
    Token,
    Grammar,
}

impl ExternKind {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Token => "token",
            Self::Grammar => "grammar",
        }
    }

    pub(super) fn opposite_name(self) -> &'static str {
        match self {
            Self::Token => "grammar",
            Self::Grammar => "token",
        }
    }
}

impl GrammarBinding<'_> {
    pub(super) fn bind_target(&mut self, vocab: &Vocab, name: &str) -> Result<()> {
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

    pub(super) fn compile<'a>(
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
    pub(super) fn into_compiled<'a>(
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
