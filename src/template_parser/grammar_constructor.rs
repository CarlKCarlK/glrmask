//! Grammar resolution and parser construction happen once, before vocabulary
//! assembly. Nothing supplied by the compiler is called during generation.
use super::*;
use crate::grammar::flat::GrammarDef;

/// Compile-time parser constructor. The input retains expression structure and
/// resolved rule references; the output is one complete stack relation for each
/// terminal plus completion. GLRMask owns lexing, GSS execution, admission
/// projections, masks, commits, and persistence. A closure can implement this
/// trait directly; it need not be `Send`, `Sync`, or `'static`.
pub trait ParserCompiler {
    fn compile(&self, grammar: &ParserGrammar) -> Result<ParserDefinition>;
}

impl<F> ParserCompiler for F where F: Fn(&ParserGrammar) -> Result<ParserDefinition> {
    fn compile(&self, grammar: &ParserGrammar) -> Result<ParserDefinition> { self(grammar) }
}

/// One resolved source grammar, reusable with different parser paradigms. The
/// parser-facing view has no regex or vocabulary dependencies. Lexical patterns
/// and partition metadata remain private and preserve the same terminal order.
#[derive(Debug, Clone)]
pub struct PreparedParserGrammar {
    grammar: ParserGrammar,
    lexical: Arc<GrammarDef>,
}

impl PreparedParserGrammar {
    /// Resolve a programmatic source grammar without building a parser table,
    /// flattening parser expressions, or computing LR-specific analyses.
    pub fn from_named(grammar: &glrmask_grammar::NamedGrammar) -> Result<Self> {
        let (grammar, lexical) = crate::error::catch_internal_invariant(|| {
            crate::grammar::ast::parser_grammar::resolve_parser_grammar(grammar)
        })??;
        Ok(Self { grammar, lexical: Arc::new(lexical) })
    }

    pub fn grammar(&self) -> &ParserGrammar { &self.grammar }

    /// Invoke the compiler exactly once and validate its complete result. No
    /// compiler object or callback is retained by the returned program.
    pub fn compile_parser(&self, compiler: &(impl ParserCompiler + ?Sized)) -> Result<GrammarParserProgram> {
        let definition = compiler.compile(&self.grammar)?;
        if definition.terminals.len() != self.grammar.terminal_count() as usize {
            return Err(fail("parser compiler must return one relation for every resolved terminal"));
        }
        let program = ParserProgram::new(definition)?;
        if let Some(ignore) = self.grammar.ignore_terminal()
            && !program.identity_terminals.contains(&ignore) {
            return Err(fail("the resolved ignore terminal requires StackTemplate::identity()"));
        }
        Ok(GrammarParserProgram { grammar: self.grammar.clone(), lexical: Arc::clone(&self.lexical), program })
    }
}

/// A validated parser and its matching lexical inventory. Reuse this object
/// with multiple vocabularies and static/dynamic build policies; parser
/// construction and derived input-domain compilation are not repeated.
#[derive(Debug, Clone)]
pub struct GrammarParserProgram {
    grammar: ParserGrammar,
    lexical: Arc<GrammarDef>,
    program: ParserProgram,
}

impl GrammarParserProgram {
    pub fn grammar(&self) -> &ParserGrammar { &self.grammar }
    pub fn parser_program(&self) -> &ParserProgram { &self.program }

    pub fn compile(&self, vocab: &Vocab) -> Result<Constraint> {
        self.compile_with(vocab, TemplateBuildOptions::default())
    }

    pub fn compile_with(&self, vocab: &Vocab, options: TemplateBuildOptions) -> Result<Constraint> {
        let tokenizer = crate::error::catch_internal_invariant(|| {
            crate::compiler::pipeline::build_tokenizer(&self.lexical)
        })?;
        let mut specials = self.lexical.terminals.iter().filter_map(|terminal| match terminal {
            crate::grammar::flat::Terminal::SpecialToken { id, token_id } => Some(
                crate::runtime::SpecialTokenTerminal { terminal_id: *id, token_id: *token_id }),
            _ => None,
        }).collect::<Vec<_>>();
        specials.sort_unstable_by_key(|special| (special.token_id, special.terminal_id));
        self.program.compile_tokenizer(tokenizer, self.grammar.terminal_names().to_vec(),
            self.grammar.ignore_terminal(), specials, vocab, options)
    }
}
