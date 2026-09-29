# Structural overhaul: architecture and validation

This is a responsibility and ownership cleanup of GLRMask, not a ground-up
rewrite of every automata algorithm. The public Rust and Python facade remains
the integration boundary. Vocabulary equivalence analysis is retained as
private tooling, rather than promoted into the supported API.

For the current candidate, validation receipts, and remaining performance
qualifications, see [overhaul PR #4](https://github.com/IsaacBreen/glrmask/pull/4).
This guide describes the code organization; it is not a benchmark certificate.

## Where responsibilities live

| Area | Responsibility |
| --- | --- |
| [`src/api`](../src/api) | Public grammar, compilation, bindings, options, and unlinked-constraint facade. |
| [`src/import`](../src/import) | Source lowering and dispatch into static, dynamic, or serialized compilation. |
| [`src/compiler/composition`](../src/compiler/composition) | Discovery, boundary construction, parser composition, assembly, and linking. |
| [`src/runtime/artifact`](../src/runtime/artifact) | Compiled payloads and owned, packed, or retained-byte-backed representations. |
| [`src/runtime/constraint`](../src/runtime/constraint) | Immutable constraint views, derived caches, vocabulary coordinates, and output-mask projection. |
| [`src/runtime/commit`](../src/runtime/commit) | Token admission, lexical progress, parser-frontier execution, controls, and generation-scoped mask retention. |
| [`src/runtime/mask`](../src/runtime/mask) and [`dynamic_mask`](../src/runtime/dynamic_mask) | Exact static and dynamic evaluation, specialized paths, and conservative fallbacks. |
| [`src/runtime/persistence`](../src/runtime/persistence) | Artifact envelopes, sections, loading, saving, and retained backing. |
| [`python/src`](../python/src) | Binding registration, ownership, checked buffers, conversion, profiling, and private tooling. |

Workspace crates retain the specialized grammar, lexer, GLR, vocabulary,
terminal-DWA, parser-DWA, and automata implementations. Separating a concern
into a module does not imply that its underlying algorithm has been rewritten
or that its compilation cost has improved. Large algorithmic cores may still
warrant a later focused pass.

## Invariants across the boundaries

**Compiled body and root termination are different domains.** Internal walkers
use body-mask coordinates. The public dispatcher applies generation end-token
policy afterwards. Nullable embedded starts and grammar-level controls have
explicit metadata; display names do not encode their semantics.

**Ownership is explicit.** A sequence owns its mutable parser state and scratch
capacity. The compiled constraint owns immutable grammar data and shared
preparation. Artifact-backed views retain the byte allocation they borrow.
Python state wrappers retain their owner while native methods run.

**A mask is exact, including on a fallback.** A budget or specialized-path
eligibility check may decline an optimization, not relax the language. A
partially established proof must never become a partial admission mask. Mask
and commit disagreement must be repaired in the responsible construction or
execution logic, not hidden by filtering candidate tokens through commit.

**Cache provenance matters.** A cached output with a retained internal bitmap
is a pure projection through the same immutable vocabulary mapping. Writers
with independent output contributions clear that bitmap certificate. Direct
mask replay can reuse equality or a proved addition-only change; any removed
bit, incompatible extent, invalid padding, or alternate mapping declines
before writing. Cost estimates decide whether to reuse, never which tokens are
valid. Speculative generation-to-generation equality proofs are bounded.

**Storage size is not an admission limit.** Small inline scratch arrays may
spill and retain capacity. Their size must not silently narrow the existing
parser-path or stack-work limits. Early exits return scratch without retaining
live parser state unnecessarily.

**The binding validates writable buffers before borrowing native slices.**
Packed NumPy masks require the expected dtype, contiguity, write access,
alignment, and sufficient length. Rejected buffers return Python errors rather
than Rust panics. The write guard remains alive for the native operation.

## Artifact compatibility

The structural-overhaul prerelease uses static envelope version 33. Older
version-30 artifacts require recompilation. Round-trip and older-input checks
must name the producer and supported format explicitly; they are not a promise
that every historical artifact or older reader remains compatible. Wide model
token IDs must retain their full output-word coordinates.

## Validation discipline

Build and test the actual combined source, not an assumed sum of separately
validated branches. Pin the imported Python facade and native-library hash in
the process executing the tests. Keep platform packaging smoke tests distinct
from the complete Rust suite, and exclude nested subprocess summaries from
reported test totals.

Semantic checks cover the public surface, full-vocabulary masks, private
partitions, artifact loads, state lifecycle, recursive languages, and the
specific regressions fixed during extraction. Newly merged upstream tests
remain reachable even when their containing module has been split.

Compare native binaries built with matching settings. Record fresh and loaded
constraints, grammar build/save/load costs, masking and commitment, and the
Python-call boundary separately. Keep adverse cases and identical-binary
controls visible; a favorable aggregate does not erase a local slowdown, and
control noise is not subtracted to manufacture a win. Development screens are
not the canonical 1,000-case CFA benchmark or a universal tail guarantee.

## Historical design and performance records

The following documents retain the reasoning and measurements of their named
checkpoints. Their in-progress status and warnings should not be mistaken for
the status of a later combined candidate:

- [Coherent runtime and artifact checkpoint](coherent-overhaul.md).
- [Direct static-mask workspace checkpoint](direct-mask-workspace.md).
- [Final-mask replay checkpoint](runtime-mask-replay-cleanup.md).

The [boundary runtime report](performance/boundary-runtime-final-2026-09-29.md)
belongs to its stated upstream revision and preserves its unresolved scaling
and raw-tail limits. Its test totals and timings are not automatically totals
for the combined overhaul. Diagnostic fallback settings are described in
[environment variables](env-vars.md).
