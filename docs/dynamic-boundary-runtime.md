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

## Validation checkpoint: 27 September 2026

The retained runtime was validated before merging the latest upstream boundary
compiler improvements. The combined code commit is
`e815c23ab0b911b99748869d32861a32a08588ee`.

The combined workspace passed 1,931 library tests (54 ignored) and 170 focused
public API tests. An independent eight-fixture finite-language oracle passed
437,760 checks covering native/static children, nullable alternatives, aliases,
clone/reload behavior, and concurrent cold binding initialization. Two further
11,767-position replays checked the merged executable against the static
reference, including a newly composed dynamic artifact.

The preceding retained-runtime comparison also passed 472,248 independent word
checks, 44 complete 11,767-position replays with cache enabled and disabled,
four ordinary full-vocabulary comparison rounds, and a fresh-artifact replay.
Raw runs include an identical release A/A control and preserve host variability.
The invalid first attempt using a stale copied control binary was quarantined;
none of its numbers are included in the validation record.

The [machine-readable validation record](performance/dynamic-boundary-runtime-20260927.json)
contains source/binary hashes, separate timing scopes, all primary cohorts,
and matching non-boundary budgets. The old all-dynamic control bypasses the
instrumented per-shard dispatcher, so its boundary-only times are **unavailable**,
not zero; its whole-mask comparison remains valid.

These results establish a large improvement over the old unmerged trunk path,
not achievement of boundary/non-boundary latency parity. The remaining boundary
p100 is still too high. Rejected parser, cache-layout, and template experiments
are not part of this release, and the next experiments must be evaluated
separately without weakening the correctness contract above.
