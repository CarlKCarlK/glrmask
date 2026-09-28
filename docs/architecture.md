# Architecture

GLRMask has one public lifecycle and several internal execution strategies. Keep
those concerns separate: callers choose a build/runtime trade-off, not an engine.

## Public lifecycle

`Grammar` describes source semantics. `UnlinkedConstraint` retains compiled,
reusable components and unresolved bindings. `Constraint` is closed and runnable;
`ConstraintState` owns one sequence's mutable execution state.

The facade in `src/api/mod.rs` is divided by responsibility:

| Module | Responsibility |
| --- | --- |
| `options` | Final compile/link policy and optimization intent. |
| `grammar` | Source grammars and immutable source bindings. |
| `bindings` | Typed binding conversions and slot-kind checks. |
| `compile` | Resolve source and bindings into compiled components. |
| `unlinked` | Attach compiled children, link, and persist reusable graphs. |
| `partition` | Internal vocabulary-equivalence analysis. |

`src/lib.rs` is the authority for supported public exports. Vocabulary partitioning
and benchmark/compiler hooks remain behind the existing `internal-api` bridge;
organizing their implementation does not promote them into the public API.
The Rust tooling bridge lives in `src/__private/mod.rs`. Python equivalence
analysis lives in `python/src/partition.rs` and is registered only under
`glrmask._internal`, not the supported top-level facade.

## Compiler and composition

Ordinary compilation stays in `src/compiler/`. Composition has its own subsystem
in `src/compiler/composition/`:

- `coordinates` reconciles component-local parser, tokenizer, and token domains.
- `discovery` finds crossing-token paths; `publication` turns validated results
  into runnable boundary shards.
- `parser_union` combines weighted parser automata; `templates` retains and
  transports reusable parser structure.
- `boundary_repair` builds the cross-component parser overlay; `assembly`
  finalizes its runtime artifact; `link` owns the linking entry points.

The linker requires an explicit `StaticParserDwa` or `Dynamic` boundary backend.
There is no implicit third backend selected through retired experiment flags.
Unsupported flattened-artifact adapters fail explicitly; their unreachable
implementations are gone. This keeps the supported pipeline free of branches
whose only premise was an absent backend.

The `boundary/` namespace contains support certificates, transfer programs,
lexer views, and prepared-completion machinery. These are not a second public
compiler API.

Coordinate conversion is a semantic operation, not a renaming convenience. A
component state can have several images after composition. DEFAULT transitions
must preserve their positive-pop domain and exclusions; independently mapping
stack symbols does not generally preserve recursive stack-language correlations.
See [the state-domain proof](subgrammar-parser-default-domains-proof.md). A
materialized parser and a recursive component provider remain distinct exact
runtime representations.

## Runtime ownership

`src/runtime/` owns immutable compiled data, mutable state, mask generation,
commit operations, and dynamic execution. `runtime/dynamic/` contains dynamic
constraint construction, transport, and union state; it is no longer mounted as
an unrelated crate-root module.

The private artifact facade now separates layouts, vocabulary ownership, proof
construction, cache operations, and transfer validation into focused modules.
See [artifact module boundaries](runtime-artifact-modules.md).

See [runtime implementation](runtime-implementation.md) for constraint cache,
parser, observation, and mask-replay ownership. See [lexer compilation](lexer-architecture.md)
for expression lowering, product construction, repeat horizons, and state-map
proofs. The exact commit language must remain distinct from a certified smaller
mask-analysis representation.

The existing workspace crates remain the compilation/dependency boundaries for
finite and weighted automata, grammar import, lexer, GLR, vocabulary, terminal
DWA construction, and parser DWA construction. Moving an existing inline module
into a file does not create a crate boundary or change its Rust namespace.
Large tests and serialization implementations have separate files so runtime
algorithms can be read without traversing thousands of unrelated lines.

## Persistence

`src/runtime/persistence/` separates the current artifact's responsibilities:

| Module | Responsibility |
| --- | --- |
| `envelope` | Checked version/section framing and current runtime wire. |
| `core` | Small core metadata and retained lexer expressions. |
| `composition` | Lazy link metadata and retained compiler caches. |
| `segmented` | Exact materialized and recursive component artifacts. |
| `token_masks` | Packed mask caches and token-coordinate maps. |
| `residual` | Validated residual-lexer mask projections. |
| `weights` | Packed weight pools and non-DWA weight attachment. |
| `save`, `load` | Orchestration and restoration of mutable caches. |

Only current envelope versions are accepted. Historical compatibility readers
are not part of the pre-release contract. Current version numbers are retained;
removing old readers does not itself require a new current wire format.

Dynamic persistence uses `DynamicCore`, `DynamicAlternative`, `DynamicArtifact`,
and `TransferMetadata`, rather than a chain of historical-version wrappers.
Current component restoration likewise operates directly on the materialized
representation instead of converting through previous layouts.

Three invariants are more important than reducing line count:

1. Validate vocabulary identity, field lengths, component domains, and lexer
   certificates before accepting a loaded runtime.
2. Retain immutable byte backing and deferred compiler views; do not introduce
   copies or eager reconstruction merely to simplify a type signature.
3. Recreate mutable caches per runtime instance. An empty but prepared proof set
   is different from a proof set that has not been computed.

## Working on performance-sensitive code

Refactor a coherent subsystem, then run its combined correctness and performance
gates. Preserve a separately built baseline; rebuilding the baseline after edits
can obscure provenance or silently compare the candidate with itself.

For timing comparisons, use the same host, vocabulary, compiler configuration,
and inputs, with no competing builds. Compare static and dynamic compile/link,
mask, commit, save, and load separately. A mask optimization must not hide its
cost in commit or first-use work. Use full token-mask traces for semantic
comparison, not only the one generated token's membership.

Repository tooling under `scripts/refactor_acceptance.py` supports isolated
baseline/candidate Python extensions, native runtime timing, full-vocabulary mask
hashes, and alternating ABBA measurements. Its selected cases are a regression
screen, not a substitute for the canonical CFA distribution or tail acceptance.

## Commitment responsibilities

The commitment lifecycle and execution kernels are described in [Commitment runtime](commit-runtime.md).
