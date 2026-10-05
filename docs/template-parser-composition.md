# Compose template-DFA parsers

Composition keeps the ordinary lexer, graph-structured stack, token commitment,
and mask engines. Parser advances and zero-width calls and returns are template
relations. A linked template artifact contains no executable LR table, including
in its retained children or a packed compiler cache.

## Link an already-compiled child

```rust
use glrmask::{BuildOptions, Constraint, Grammar, Optimization, ParserBackend, Vocab};

let vocab = Vocab::new(vec![
    (0, b"x".to_vec()), (1, b"a".to_vec()), (2, b"y".to_vec()),
    (3, b"xay".to_vec()), (4, b"xy".to_vec()),
]);
let child = Grammar::from_ebnf(r#"start ::= "a"?"#).compile_with(
    &vocab,
    BuildOptions::default()
        .optimization(Optimization::FastRuntime)
        .parser_backend(ParserBackend::TemplateDfa),
)?;
let parent = Grammar::from_glrm(
    r#"glrm 1; start root; extern grammar child; nt root = "x" child "y";"#,
).compile_unlinked(&vocab)?;

for optimization in [Optimization::FastBuild, Optimization::FastRuntime] {
    let linked = parent.bind("child", &child)?.link_with(
        BuildOptions::default()
            .optimization(optimization)
            .parser_backend(ParserBackend::TemplateDfa),
    )?;
    assert_eq!(linked.parser_backend(), ParserBackend::TemplateDfa);
    let mut state = linked.start();
    state.commit_token(4)?; // The child body can be empty.
    assert!(state.is_accepting());

    let loaded = Constraint::load(linked.save())?;
    let mut state = loaded.start();
    state.commit_token(3)?;
    assert!(state.is_accepting());
}
# Ok::<(), glrmask::Error>(())
```

Binding is immutable. Selecting a backend for the result does not mutate a
shared child or change the process-wide default. End-token policy belongs to the
final root, not an embedded child's standalone termination policy. Exact grammar
tokens remain grammar tokens, and parent and child ignore rules retain their
own scopes.

Body nullability comes from the original source, including the empty string in
a nullable lexical rule. For example, a child whose only rule references the
regular expression `a?` may have an empty body when embedded; an extra optional
grammar wrapper is not required. An exact special-token identity is still an
event, even when its vocabulary spelling is empty, and never becomes epsilon.

## Build-time choices

Source-bound grammars use the common lexical/query compiler. Temporary LR
analysis may derive their templates, but all tables are discarded before
constructing a `Constraint`. Compiled children link through immutable prepared
graphs and scoped views without reconstructing or retaining execution tables.

An already table-free child takes the direct template linker. `FastBuild` and
`Auto` use dynamic boundary traversal. `FastRuntime` compiles the boundary token
queries to static parser automata. The local masking engine of an independently
compiled child is retained: a dynamic child does not become a fully static
constraint merely because its parent requests static boundaries. To obtain
static local masks as well, build those children with `FastRuntime`.

Nested and repeated compiled children, including nullable bodies, are supported
within the compiler's representation limits. The final constraint can itself
be saved, reloaded, and embedded as a child. Existing static boundary automata
are reused when their parser and lexer coordinates are unchanged.

Sparse regular frontends use their finite, single-symbol stack representation
directly. Virtual lexers use a persisted finite observation projection for the
current vocabulary. Their exact runtime lexer states remain separate from that
projection, so length limits and other residual conditions are still checked
when tokens are committed. This works for virtual parents as well as nested or
repeated virtual children.
Static linking builds a separate boundary projection covering the full linked
vocabulary. A child's own projection may cover shorter tokens and is not reused
as a boundary proof. Neither its local masking coordinate nor the caller's
reusable child is changed.

Static nullable calls are not expanded to an arbitrary number of iterations.
Their zero-width closure is compiled by the shared weighted fixed-point solver.
Runtime control closure similarly stops at its fixed point, not after a number
of passes guessed from the parser-state count. Exceeding an explicit resource
budget produces an error rather than a successful partial closure.

## Current limits

Direct static construction requires a finite lexical observation coordinate.
Virtual-lexer components without a supported finite projection are rejected.
Older template artifacts and arbitrary data-only parser programs may lack the
finite embedding contract required by the linker. These requests return errors;
the linker does not reconstruct an LR table or guess a child-return convention.

Static template and weighted-automaton construction have representation and
work limits. A successful dynamic build does not imply that a static build will
fit those limits. A static build error never silently changes the request to
dynamic masking. These limits are not a whole-process memory or elapsed-time
guarantee. Native component/provider composition stays table-free. Ordinary Dynamic
uses retained LR by default; its development template override does not change
O2/Static or permit a native component to fall back to LR.

## Persistence and embedding

The parser section retains a finite child-return relation, source-body
nullability, and certified call slots in addition to ordinary terminal and
completion relations. Completion is an input predicate; it does not specify
how an arbitrary parser should remove a child frame. The built-in compiler
derives that embedding contract before discarding its LR construction data.

Embedding-capable programs use the `TPR7` parser section in current envelope35
or36. Obsolete pre-release program sections are rejected. An artifact without an embedding
contract cannot acquire one merely by being loaded. Invalid indices, cycles,
alphabet labels, control inventories, and embedding fields are rejected.

Projected virtual compositions save a versioned finite-observation descriptor
with the full token horizon and fingerprints of both the finite graph and the
exact-state mapping. Loading reconstructs this vocabulary-dependent lexer
metadata and checks the fingerprints, leaf offsets, dimensions, and physical
images before accepting the saved boundary. It does not rebuild the parser.
Only current native artifact variants are accepted. Obsolete projected and
non-projected variants must be rebuilt.

`save_with_external_vocab()` omits the separately supplied model vocabulary and
requires `Constraint::load_with_vocab()` with the exact original binding. Both
artifact forms retain parser, component, and final-root termination semantics.

## Verification

`tests/template_parser_composition.rs` compares complete masks, token and byte
commits, acceptance, scopes, and fresh/self-contained/external-vocabulary forms.
Strict-static tests fail on a hidden dynamic boundary fallback. Independent
oracles cover stack-relation scoping, weighted control closure, and safe lexical
follow exclusions.
`tests/template_virtual_composition.rs` additionally exercises projected virtual
parents and children, near-limit residual states, nested reuse, and malformed
saved observation metadata.

The selected10 regression example is a separate full-vocabulary replay:

```sh
cargo run --release --features internal-api --example template_precompiled_composition -- \
  FIXTURE_DIR NEW_OUTPUT_DIR static PATH_TO_JS_GRAMMAR
```

The fixture contains `schema-00.json` through `schema-09.json` in canonical
selected10 order, `vocab_dump.bin`, and `traces.json`. The example prepares a
native child and unlinked parent, and compares direct linking with a separately
compiled grammar-inlined native reference. It reports child preparation, parent
compilation, reference compilation and precompiled linking separately.
The example checks vocabulary and trace dimensions, all 11,767 recorded mask
positions, token commits, and three explicit grammar-EOF completions. It refuses
to overwrite an output directory. Its `summary.json` is written only after the
whole replay succeeds; compile failure or a partial CSV is not a passing result.
Timings recorded during this correctness run are not an isolated benchmark.

The compile-time structured grammar interface for independent parser compilers
is documented in [parser-grammar-constructor.md](parser-grammar-constructor.md).
