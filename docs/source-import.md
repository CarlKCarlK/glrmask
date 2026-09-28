# Source-import modules

`src/import` adapts source grammars into the existing compiler and private
artifact-producing entry points. The public lifecycle remains `Grammar`,
`UnlinkedConstraint`, `Constraint`, and per-sequence state; this module is not a
second public API.

## Responsibilities

| Module | Responsibility |
| --- | --- |
| `lowering` | Parse EBNF, Lark, GLRM and JSON Schema, factor named grammars, prepare schema terminals and lower vocabulary-partition inputs. |
| `static_compile` | Static compilation, ordinary private constructors and their existing parser-table policies. |
| `dynamic_compile` | Prepare dynamic alternatives and invoke ordinary or vocabulary-partitioned dynamic compilation. |
| `serialized` | Compile dynamic source directly into an artifact, retaining explicit preparation and timing boundaries. |
| `bindings` | Allocate non-vocabulary linker IDs, validate named children and assemble static/dynamic compiled-child bindings. |
| `programmatic` | Keep unsupported legacy errors explicit and retain the separate test-only composition prototype. |
| `tests` | Existing source-import regression tests, moved without changing their assertions. |

`mod.rs` retains the internal import paths used by the rest of the crate. The
existing JSON Schema facade and its tests remain in `json_schema`.

## Behavior retained

Static JSON Schema uses its existing legacy row-bisimulation table default;
dynamic JSON Schema uses its existing LALR default. The other source adapters
retain their experimental core-merged table policy. No default is unified merely
because the call sites look similar.

The direct-artifact path deliberately uses unfinalized dynamic compilation and
then prepares the data required by the artifact. Its compile and serialization
timers retain their original boundaries. The ordinary dynamic constructor still
performs its own runtime preparation. A structural extraction is not a reason to
make these distinct paths call one another or to exclude work from a timer.

Embedded end-token rules in the legacy adapters remain separate from the public
generation-root termination policy. Linker IDs still avoid vocabulary, explicit
end tokens and child special tokens. Unknown or duplicated child bindings retain
the same errors; unresolved children remain explicit late-binding slots.

The dedicated large-stack wrapper for large source imports on Windows is
unchanged. Linux checks do not establish that the Windows-only test has run.

## Removed unreachable code

Two unsupported private programmatic-JSON constructors already returned the same
compilation error before reaching their historical implementations. Those
post-return blocks were removed, while their signatures, first error statement,
error string and caller-visible behavior were retained. The independent
`cfg(test)` backend prototype remains available. An unused private parser alias
and an obsolete facade reexport were removed; the programmatic schema adapter
used only by the prototype is imported only in test builds.

No masking kernel, commit kernel, cache policy, vocabulary-partition algorithm,
serialized layout, dependency version or public method is changed by this batch.
Source audit and exact-output tests establish the structural scope. Performance
still requires comparisons of the built binaries; unchanged source alone does
not prove identical code generation or latency.
