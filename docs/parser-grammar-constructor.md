# Compile a resolved grammar with your own parser

`ParserCompiler` is a compile-time interface. GLRMask resolves the source and
lexical definitions, passes an immutable `ParserGrammar` to the compiler, and
validates the returned `ParserDefinition`. The compiler is not retained by a
running constraint and is never called for masks or commits.

```rust
use glrmask::{Grammar, Optimization, Vocab};
use glrmask::template_parser::{ParserCompiler, TemplateBuildOptions};

fn build(vocab: &Vocab, compiler: &impl ParserCompiler)
    -> glrmask::Result<glrmask::Constraint>
{
    let prepared = Grammar::from_ebnf(r#"root ::= "a" | "(" root ")""#)
        .prepare_parser_grammar()?;
    let parser = prepared.compile_parser(compiler)?;
    parser.compile_with(vocab,
        TemplateBuildOptions::default().optimization(Optimization::FastRuntime))
}
```

For a one-off build, `Grammar::compile_with_parser(vocab, compiler, options)`
combines these steps. EBNF, Lark, GLRM and JSON Schema use their existing source
importers. Programmatic sources can use `PreparedParserGrammar::from_named`.
The current source constructor requires a closed grammar; unresolved external
bindings and component linking are not implicit compiler responsibilities.

## Input: rules and expressions, not a prescribed parser

`ParserGrammar::rules()` preserves named rules and references. Rule IDs index
that slice; terminal IDs index `terminal_names()`. `start()` names the start
rule. The compiler owns its stack alphabet independently of these IDs.

The expression variants are terminal and nonterminal references, sequence,
choice, epsilon, quantified expressions, separator-delimited sequences, and
expression-labeled automata. Recursion remains in references rather than being
expanded. Repetition bounds and separator-group quantifiers remain explicit.
Automata retain epsilon edges, nondeterministic edges, and their shared symbol
table instead of being forced through determinization.

`Choice([])` is the empty language. `Epsilon` and `Sequence([])` denote the empty
string. These distinctions also hold for grammars with no emitting terminals.
In a separated sequence, omitting an optional item omits its separator;
`allow_empty` does not make a required nonempty item optional or permit trailing
separators. A required nullable item is not equivalent to an omitted item.

Lexical expressions, partitions, and exact vocabulary-token bindings remain on
the GLRMask side. A nullable lexical expression is represented as epsilon plus
its nonempty terminal language. Internal lexical helpers do not become parser
rules. Supported structural nonterminal subtraction is resolved before the
callback; arbitrary context-free language subtraction or intersection is not
silently approximated.

## Optional shared lowering and analyses

Compilers can consume the expression graph directly. Nothing is flattened and
no LR analysis is performed merely because the grammar was prepared.

`lower_to_cfg()` requests a conventional CFG. `lower_to_cfg_with` explicitly
selects `CfgRecursion::Left` or `Right` for generated repetition helpers; it does
not rewrite user-authored recursion. Original rule IDs are retained and generated
helper IDs follow them. Terminal IDs still index the original lexical inventory.

`analysis()` requests nullable, FIRST, FOLLOW and EOF-follow information in the
default flat coordinate. FIRST and FOLLOW contain terminal IDs only, with EOF
reported separately. Lowered grammars and analyses are cached on demand and
shared across clones. Grammar analysis has a work budget and returns an error
instead of performing unbounded analysis. It does not construct an LR table.

## Output: complete stack-action relations

Return `ParserDefinition` with a stack-symbol count, one complete `StackTemplate`
for every resolved terminal, and a completion relation. Initial stack `[0]`,
top-first POP, bottom-first PUSH, and READ semantics are documented in
[`template-parser.md`](template-parser.md). An ignored terminal requires the
canonical `StackTemplate::identity()` relation.

GLRMask checks dimensions, alphabet use, phase-graph structure and resource
budgets. It derives admission predicates from those same relations; a compiler
does not provide separate, trusted admission certificates. Validation does not
prove that the returned parser recognizes the input grammar. That semantic
obligation remains with the compiler and its tests.

The validated `GrammarParserProgram` can be reused for multiple vocabularies and
build policies without rerunning the parser compiler. `FastRuntime` uses the
ordinary static mask compiler; `FastBuild` and `Auto` use the ordinary dynamic
engine. Both retain the common lexer, shared GSS and token commit machinery.
An oversized static expansion returns an error; it does not silently fall back
to a different parser or masking mode. Saved constraints contain the validated
template program, not a compiler callback or an LR table.

## Runnable non-LR example

```sh
cargo run --release --example parser_grammar_constructor
```

The example implements a small terminal-leading top-down compiler. It reads the
structured grammar directly and recognizes recursive languages such as
`a`, `(a)`, and `((a))`, using pending grammar symbols as its parser stack. It
demonstrates static and dynamic assembly and save/reload with the default public
API: no `internal-api` feature is needed.

This example intentionally accepts only terminal-leading alternatives with
atomic tails. It rejects epsilon productions, leading nonterminal expansion,
unproductive rules, explicit ignored terminals in productions, and unhandled
expression forms. Rejecting unproductive rules matters for exact prefix masks:
a terminal-leading cycle with no finite derivation must not admit dead prefixes.
It is an executable starting point for a parser
experiment, not a general grammar compiler or a claim that every parser paradigm
has finite acyclic single-terminal relations.

The separate dependency smoke test builds the same example source without
workspace dev-dependency feature unification:

```sh
cargo run --release --manifest-path tests/fixtures/parser_compiler_external/Cargo.toml
```
