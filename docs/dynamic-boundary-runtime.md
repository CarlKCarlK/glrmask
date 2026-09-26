# Retained dynamic boundary runtime

This change makes certified boundary candidate vocabularies usable by the existing
shared dynamic masker. It does not introduce another byte walker or replace the
GLR parser. Ordinary component masks remain the baseline; boundary execution adds
only admissions not already present.

## Correctness contract

A shard's token domain is a certified conservative set of original model token
IDs. A missing domain remains unknown and uses the existing exact fallback; it is
never interpreted as empty. Initial frontiers retain the full lower parser stack
and its correlated exclusions while selecting the shard's active component.

For baseline output A and exact boundary output B, an additive evaluation may
omit any output already in A: A union B equals A union (B minus A). The canonical
trie implementation skips a subtree only when every original token ID beneath it,
including every alias for identical bytes, is already in A. It neither identifies
lexer-equivalent tokens nor infers parser equivalence. A baseline-dependent partial
result is never published to the complete-mask result cache. The former filtered
vocabulary implementation remains an explicit differential reference.

The two-byte preflight is a necessary-condition rejection only. It may skip
vocabulary construction when all candidate prefixes are byte-dead. It declines
across terminal finalization, one-byte tokens, explicit special-token semantics,
or unavailable summaries; the same exact shared walker then handles the query.
Parser-conditioned root projection likewise declines when ownership, controls,
nullable returns, virtual coordinates, or empty-stack alternatives prevent a
sound restriction. Initial exclusions remain attached to the original roots.

Empty-byte model IDs cannot create a zero-progress ordinary route. Explicit
special-token routes for those IDs remain independently admissible. The empty-ID
inventory is derived once at compile/load, without building a full vocabulary
trie in the mask path.

## Runtime and storage

Prepared vocabularies belong to an immutable binding, not a global token-ID cache.
Clones share immutable storage; serialization rebuilds runtime-only preparation.
The vocabulary object and empty-ID inventory do not change the serialized grammar
language. Read-only mask shadows do not reserve the ordinary commit branch stacks.
Actual commit buffers retain their original reservations.

The direct flat vocabulary constructor produces the same node, edge, byte,
subtree, and traversal metadata as the retained reference constructor. It removes
intermediate prefix-tree construction, not runtime semantics.

The retained optimizations are default behavior. Reference overrides are for
validation, not prerequisites for obtaining the improvements. The experimental
LR/compiler oracles, alternate cache layouts, parser-carrier experiments, template
execution, checked-control proofs, and end-only-reset prototype are excluded.

## Measurement

`boundary_runtime_replay` checks complete masks, acceptance, and chosen-token
commits against the static artifact. The internal-only observer reports whole-mask
CPU, DynamicDirect CPU, and entry-inclusive complete boundary-dispatch CPU. Dispatch
contains the dynamic interval; these values must not be added. Non-boundary work
is whole-mask CPU minus dispatch CPU from the same invocation. Do not subtract
unrelated percentile maxima or independently selected repetitions.

The release comparison uses fresh binding instances, complete authentic prefix
history, shuffled 16-position measurement blocks, identical A/A controls, and two
independent min-of-two-by-position cohorts. Raw maxima remain available. The shared
CX33 can vary despite thread-CPU timing. Performance claims require the actual
recorded release validation; correctness passes alone do not establish a latency
improvement or achievement of the boundary/non-boundary parity goal.
