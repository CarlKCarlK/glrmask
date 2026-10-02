# Acyclic template parser backend

## Select the backend for a built-in grammar

```rust
use glrmask::{BuildOptions, Grammar, Optimization, ParserBackend, Vocab};

let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"ab".to_vec())]);
let grammar = Grammar::from_glrm(r#"start root; nt root ::= "a" "b"?;"#);
let constraint = grammar.compile_with(
    &vocab,
    BuildOptions::default()
        .optimization(Optimization::FastRuntime)
        .parser_backend(ParserBackend::TemplateDfa),
)?;
assert_eq!(constraint.parser_backend(), ParserBackend::TemplateDfa);
# Ok::<(), glrmask::Error>(())
```

`FastRuntime` retains the existing static mask engine. With `TemplateDfa`,
`FastBuild` uses the existing vocabulary-partitioned dynamic engine (O2), after
normalizing the grammar so each terminal advance has a finite acyclic action
relation. The normalization is exact; it does not truncate recursive action
paths.

The built-in grammar compiler may derive action relations using temporary LR
analysis. It discards those tables before constructing a `Constraint`. Native
`TemplateDfa` is the default and the only permitted runtime backend; explicit
LR-backed construction, loading or runtime access panics. An unsupported native
composition request returns an error.

The older experimental `GLRMASK_ENABLE_TEMPLATE_DFA_ADVANCE` environment switch
is not this storage guarantee. Use `BuildOptions::parser_backend` and inspect
`Constraint::parser_backend()`.

## Supply a parser as data, without an LR grammar

The following example recognizes balanced parentheses with whitespace. Symbol
`0` is the initial bottom marker; symbol `1` records an unmatched opening
parenthesis. The stack may grow without bound across tokens, although every
single-terminal automaton is acyclic.

```rust
use glrmask::{Optimization, ParserBackend, Vocab};
use glrmask::template_parser::{
    LexerDefinition, ParserDefinition, ParserProgram, StackLabel, StackTemplate,
    TemplateBuildOptions, TerminalPattern,
};

let definition = ParserDefinition {
    stack_symbol_count: 2,
    terminals: vec![
        StackTemplate::read_top_and_push([0, 1], [1]), // (
        StackTemplate::rewrite([StackLabel::Symbol(1)], []), // )
        StackTemplate::identity(), // whitespace
    ],
    completion: StackTemplate::read_top_and_push([0], []),
};
let program = ParserProgram::new(definition)?;
let lexer = LexerDefinition::new(vec![
    TerminalPattern::literal(b"(".to_vec()),
    TerminalPattern::literal(b")".to_vec()),
    TerminalPattern::regex(r"[ \t\n]+"),
]).ignoring(2);
let vocab = Vocab::new(vec![
    (0, b"(".to_vec()), (1, b")".to_vec()),
    (2, b"()".to_vec()), (3, b"((".to_vec()),
]);
let constraint = program.compile_with(
    &lexer, &vocab,
    TemplateBuildOptions::default().optimization(Optimization::FastRuntime),
)?;
assert_eq!(constraint.parser_backend(), ParserBackend::TemplateDfa);
let mut state = constraint.start();
state.commit_token(2)?;
assert!(state.is_accepting());
# Ok::<(), Box<dyn std::error::Error>>(())
```

The data-only constructor assembles an **absent table from the beginning**.
It does not construct an LR table, reconstruct one from the program, or call a
user parser at each token. `ParserProgram` is immutable and reusable with multiple
vocabularies. `ParserProvider` is a compile-time trait returning
`ParserDefinition`; it is called when validating the program, not during token
generation.

`TemplateBuildOptions::optimization(Optimization::FastRuntime)` compiles the
program through the existing static token-mask DWA pipeline.
`FastBuild`, `Auto`, and the convenience `program.compile(...)` use the existing
dynamic mask engine. This is a build-time choice; an oversized static expansion
returns an error rather than silently changing modes. Both choices retain the
same token-level commit code and template parser primitives. Built-in compiled
components with finite embedding transfers support
[template composition](template-parser-composition.md). An arbitrary data-only
program supplies no return-frame contract and is not implicitly composable.

### Static compilation without a grammar or table

The static route builds the ordinary terminal automaton from the lexer and
vocabulary, substitutes the supplied terminal relations, cancels signed stack
actions, and compiles a stack-prefix predicate. Possible-match data is compiled
independently. Its equivalence mapping is then reconciled with the terminal
compiler's mapping before constructing the ordinary static runtime. No mask or
commit algorithm is duplicated.

The lexical compiler currently receives a conservative metadata adapter:
terminal names/count, identity terminal colouring, and global observation. It
contains **no productions, nonterminals, FIRST/FOLLOW certificates, or parser
states**. It cannot introduce an LR parser or apply grammar-specific lexical
shortcuts.

Static normalization is exact over all finite concrete stack words. In
particular, it does not inherit the old LR-only assumptions that a stack suffix
is nonempty or that symbols missing from a transition row are unreachable.
A DEFAULT edge consumes a real top symbol even when every symbol has that edge;
it is not epsilon acceptance. Missing symbols reject unless their own DEFAULT
branch accepts them. The static compiler shares determinization, row compression,
final-weight subtraction and minimization with the ordinary compiler, with these
reachability assumptions disabled explicitly.

## Relation semantics

A definition contains one complete `StackTemplate` for each lexer terminal, plus
one completion relation. Terminal indices and lexer-pattern indices must agree.
Sequences start with stack `[0]`; all stack symbols must belong to the declared
alphabet. A program may pop the initial `0`, leaving an empty concrete stack;
that is not the same as a rejected state with no possible stacks. Epsilon and
PUSH actions may subsequently accept or extend the empty stack.

A template consists of three acyclic deterministic phase graphs and optional
forward epsilon links: POP-to-READ, POP-to-PUSH, and READ-to-PUSH. POP consumes a
matching symbol from the top of the input stack. READ checks the current top
without consuming it; multiple READ transitions inspect the same top. PUSH
appends symbols in traversal order. Reaching an accepting state emits the
current stack **and still permits further transitions**. The result is the union
of all emitted stacks, represented with the existing shared GSS.

An explicit POP edge always shadows DEFAULT, even when that explicit edge leads
to a rejecting state. DEFAULT is not legal in READ or PUSH. Epsilon phase links
remain available even when READ has no matching transition or the input stack is
empty. A terminal with no accepted action words rejects; it never requests a
fallback to another parser.

Completion is the input domain of its supplied relation. It is queried without
executing output stack changes or implicitly shifting EOF. Providers remain
responsible for describing their intended language and viable parser
configurations. Structural validation is not a proof of equivalence to an
unspecified reference language.

## Exact admission and shared execution

GLRMask derives an input-only automaton from each complete action relation.
PUSH reachability is summarized during compilation, so admission need not
construct output stacks. For a top symbol, an exact domain yields either a
rejection certificate, an unconditional-admission certificate, or a deeper
suffix query.

For a GSS frontier, the union of unconditional certificates is a lower bound on
admission and the union of possible certificates is an upper bound. Bulk queries
therefore evaluate deeper domains only for possible-but-not-unconditional
terminals. These certificates are derived from the supplied automata, not copied
from LR action rows.

The token-level lexer, masking traversal, commitment queue, and shared GSS remain
common. Parser-specific primitives use template transitions and admission
projections. Runtime optimizations include live-frontier lookup and bounded
PUSH-suffix attachment through the existing shared-prefix GSS constructors.
Large output languages retain their exact DAG evaluator; the bounded cache is
not permission to truncate an action relation or fall back to LR.

## Validation and resource bounds

Construction rejects cyclic graphs, including unreachable cycles; invalid
starts and targets; duplicate labels; out-of-alphabet symbols; wrong-phase
DEFAULT edges; and invalid epsilon links. It bounds the total graph input and
top-certificate allocation/work before constructing runtime indices.

Static compilation also bounds concrete DEFAULT expansion and unweighted
subset construction. Across terminal relations, current ceilings are 131,072
constructed states, 1,048,576 edges, 2,097,152 retained subset members and
33,554,432 accounted work items. These are implementation resource ceilings,
not language restrictions that truncate a result. Exceeding one returns a build
error; the caller may explicitly choose `FastBuild` instead. The later shared
weighted compiler retains its own resource characteristics; these counters are
not a whole-process memory or wall-clock guarantee.

DEFAULT specialization happens before epsilon/subset union, preserving an
explicit rejecting edge's shadow. READ is lowered to a matching pop/push pair
before signed cancellation. Compact DAGs remain graphs: a linear PUSH graph
representing millions of output words is not enumerated by this conversion.

The direct lexer rejects empty or nullable terminals and malformed regular
expressions. Ignoring a terminal requires the canonical
`StackTemplate::identity()` relation, preventing an ignore optimization from
silently discarding a meaningful parser action. Nonignored identity terminals
also work.

## Serialization

`Constraint::save()` produces a self-contained artifact. In template mode it
contains the lexer, required masking data, complete finite template relations,
completion, and minimal metadata, but no LR actions, gotos, or production table.
Derived runtime lookup structures are rebuilt from validated template data.

`Constraint::save_with_external_vocab()` omits the model vocabulary. Load such
an artifact with `Constraint::load_with_vocab(bytes, &vocab)`. Loading without a
vocabulary, or with a different mapping, fails. End-token policy uses
`TemplateBuildOptions::end_tokens` for data-only parsers and ordinary
`BuildOptions::end_tokens` for built-in grammars. Both forms retain the policy
through save/load.

Current native artifacts use self-contained envelope33 or external-vocabulary
envelope34 and parser section `TPR6` (wrapped by `TPX1` for an external binding).
Obsolete pre-release formats are rejected explicitly; known LR artifacts panic
rather than materializing a table. Compiled components link directly with
shared template graphs and scoped views; see [composition](template-parser-composition.md).


## Large finite output languages

Acyclicity bounds the length of a single parser step, but it does **not** bound
its number of possible output stacks. A layered PUSH graph with two labels per
layer and a shared successor represents `2^d` outputs using only `d + 1` states.
The executor must not turn that compact representation into a list of words.

Small output languages retain the bounded prepared-suffix fast path. Larger
ones use a **PUSH-only acceleration index**, not a different commit or
mask engine: an incoming stack language is accumulated once per reachable PUSH
state in topological order. For input language `B`, the invariant is

```text
L[q] = union { append(B, w) : entry --w--> q }
L[entry] = B
L[target] |= push(L[source], label)
result = union { L[q] : q is accepting }
```

Appending distributes over language union. Processing every predecessor before
its target therefore preserves every output exactly while merging convergent
paths before traversing their common continuation. The existing GSS push/merge
operations preserve shared lower prefixes; neither a new stack representation
nor an LR table is introduced. Scratch is proportional to reachable pending
states, not the full declared stack alphabet or the represented word count.
The layered binary example retains a linear number of GSS nodes. This is **not**
a universal linear-time guarantee for every finite relation: union of unrelated
input languages can still require a larger GSS, and other parser phases have
their own product costs.

This reordering currently requires a uniform path annotation. Correlated
annotations use the established exact traversal order. The optimization is a
derived runtime index rebuilt after loading, so it adds no serialized program
section and does not relax acyclicity or input validation.
