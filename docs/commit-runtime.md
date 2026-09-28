# Commitment runtime

Commitment advances one sequence through model-token or byte input. The public
`ConstraintState` lifecycle is separate from private execution kernels: callers
should not need to know which parser representation or shortcut is active.

## Responsibility boundaries

`runtime/commit/mod.rs` is the lifecycle facade. It handles root termination,
known-token validation, mask-state snapshots, generation finalization, and the
ordinary, timed, profiled, and per-advance entry points. Diagnostic wrappers stay
explicit because their observation and reset policies are not interchangeable.

The private implementation modules are organized by the state they operate on:

| Module | Responsibility |
| --- | --- |
| `engine` | Ordinary token/byte dispatch, authoritative queue, exact admission probes. |
| `controls` | Exact-token parser alternatives and their union with byte alternatives. |
| `frontier` | Parser-state map operations, runtime coordinate expansion and coalescing. |
| `admission` | Exact parser admission and bounded continuation-admission caches. |
| `advance` | Parser advancement, template choice, and concrete stack actions. |
| `lexical` | Relevant lexer matches, longest-match exclusions, and future-terminal filtering. |
| `linear` | Single-match and linear-stack execution with its own admission rules. |
| `small_queue` | Bounded concrete and stack-language queues and their reusable scratch. |
| `flat_frontier` | Bounded flat-stack execution, continuation decisions, and recycled storage. |
| `lexer_only` | Shortcuts whose parser frontier is unchanged. |
| `profiled` | Instrumented queue execution and per-advance observations. |
| `assertions` | Optional equivalence oracles and diagnostic runtime configuration. |

The existing `profile`, `mask_reuse`, `template_advance`, and `tokenizer_scan`
modules retain their responsibilities. Scratch capacities, struct field order,
cache ownership, and existing test definitions are not changed by the extraction.
No additional crate or dependency boundary is introduced.

## Input semantics

A model token may have both a nonempty byte spelling and an exact-token grammar
binding. These are alternatives: commitment unions their surviving parser paths.
A byte spelling that fails does not erase a valid exact-token alternative.

An empty model-token spelling does **not** add an identity alternative. An exact
marker with an empty spelling still consumes its bound parser terminal; it must
not also retain the state before that marker. An unbound empty model token cannot
advance the language merely by doing nothing. In contrast, the explicitly
byte-oriented `commit_bytes(b"")` operation remains a no-op on a live sequence.
Root end-token policy is handled before these alternatives and remains local to
the generation root, not an embedded child.

This distinction is applied at the four token-alternative entry points: ordinary
execution, the no-fast-path reference, profiling, and per-advance observation.
Known-token lookup is unchanged, so an empty-but-known token is not confused with
an unknown ID.

## Linked components and profiling

Compiled-child grammars can require zero-width CALL/RETURN control closure
between lexer transitions. The monolithic fast paths do not implement that
closure. Ordinary dispatch already excludes them when linker controls or compact
recursive runtime are present; profiled dispatch now applies the same restriction.

The diagnostic call still executes the instrumented authoritative queue. It does
not replace a requested profile with an ordinary uninstrumented call, fabricate
per-advance records, or run a different recognizer. A caller explicitly enabling
per-advance fast paths remains subject to this correctness restriction.

## Cache and timing invariants

The lifecycle snapshots a reusable mask state before commitment, then finalizes
the generation and any proved mask reuse. The ordinary native commit timer still
includes execution **and** mask-retention proof work. Moving that proof after the
timer would change the measurement, not make commitment faster.

`reset_all()` and `clear_all()` retain their existing distinct uses. The extraction
also preserves the per-advance fast-path preference, diagnostic fields, and
normalization order. The source audit treats changes to those operations as real
changes, not formatting.

## Validation

The structural extraction has a complete item inventory. The only intended
executable differences are the profiled control-runtime admission restriction
and the four empty-byte-alternative filters. Existing masks, partitioning,
serialization layouts, public signatures, and native mask kernels are unchanged
in source.

The additional regression tests use finite languages rather than another parser
as the expected result. They cover compiled children, exact-token/byte unions,
empty exact markers, empty ordinary model tokens, root termination, saved/loaded
constraints, masks observed at each step or only at completion, and diagnostic
wrappers. The independent lifecycle guard retains rejected-path behavior rather
than assuming that every diagnostic reset policy is identical.

A source audit alone is not a runtime or performance certificate. Native package
identity, executable tests, complete-mask comparisons, and paired runtime/build
measurements are recorded separately in the checkpoint validation report.

## Private dynamic rejection state

The single-alternative profiled dynamic wrapper removes its alternative only
when the inner state reports rejection, matching the outer `is_rejected()`
contract. An error alone is not sufficient: an unknown token ID must leave a
live state available for a subsequent valid commitment.
