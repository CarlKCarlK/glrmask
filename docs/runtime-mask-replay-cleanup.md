# Final-mask replay cleanup

## Scope

This change sits on a frozen copy of the integrated overhaul, commit
`ec9b0a613e86cb1a8ab20485bbfeb7c86f1383cb`. It does not change the independent
integration checkout or publish an accepted release. Normal Rust/Python APIs,
vocabulary partitioning, and current artifact encoding are unchanged.

## Problem

Small final-weight token sets previously expanded their internal token IDs
straight into the original vocabulary mask, independently for each parser
acceptance. Small internal sets can still cause repeated sparse writes into a
large output mask. In two schema cases, native sampling attributed 62–77% of
fresh execution samples to this final-weight merge. Reloaded packed weights
used a different route and largely avoided that cost.

This is a bottleneck in both the original and refactored implementations. The
investigation does not establish that source-file movement alone caused the
measured difference between the binaries.

## Change

Final token sets without a precomputed output mask now intersect and union in
the existing internal-coordinate accumulator. The resulting union is expanded
once by the existing grouped output mapper. A cached output mask remains usable
when its complete token set is contained in the active internal mask.

The single-path and general mask evaluators share one final-token merge helper.
The repeated direct-expansion helper and an unused containment predicate are
removed. Internal cache names now describe range-replayed final token sets,
rather than a direct-output shortcut which no longer exists.

Cache planning and its conservative work bounds are retained: this is not the
workaround of materializing more caches at construction time. An independent
source comparison verifies that the seven non-kernel production files differ
only by identifier changes and comments. Constraint field types/order and the
serialized format are not changed.

## Why the result is equivalent

Let `A_p` be the admissible internal tokens for parser path `p`, `F_p` its final
weight's token set, and `M(S)` the original-vocabulary mask obtained by unioning
the output memberships of internal tokens in `S`. The old route contributes
`M(A_p ∩ F_p)` independently for each path. The new route contributes:

```
M(union_p (A_p ∩ F_p)) = union_p M(A_p ∩ F_p)
```

The equality follows from the union-preserving definition of the vocabulary
mapping; it does not require one-to-one tokens or disjoint output memberships.
Full weights contribute the entire active internal mask. A missing token set
contributes nothing. Cached output masks are replayed only after complete
containment succeeds. Control tokens, late bindings, and boundary masks retain
their existing processing after the body mask is formed.

## Evidence and limitations

Reproducible assets are in `/root/runtime-cleanup-789322/acceptance-0325`, with
raw traces, binary hashes, source manifests, and the shared measurement lock.
No compilation or benchmarks ran on Mac.

Before the fix, four focused schemas, eight independent build pairs, and nine
replays reproduced fresh static p99 TBM regressions of 26.1% (`o82377`) and
15.0% (`o50683`) versus the original binary. Identical-binary controls were
retained. Ordinary, instrumented, and reloaded masks matched exactly.

A diagnostic using the same combined binary with its direct-sparse cache switch
disabled reduced the focused fresh static p99 TBM ratio to 0.210, with identical
masks. That diagnostic also changes cache construction, so its timings are not
claimed as the result of this implementation and the switch is not made the
new default. The implementation's own correctness/performance results must be
recorded separately below.

The evidence below supersedes the initial build-in-progress status; full
correctness and performance acceptance remain explicit gates.

## Built-implementation checkpoint

The default-configuration extension has SHA256
`fdee65114fe87da0edc745372c80276e54f9fc77d5c7422925aaa12607e6bc17`.

All 74 Python tests pass. Reflection of the eight public Python types and
their signatures matches the original. The independent 128,256-token guard
matches 38 complete constraint traces and 24 vocabulary partitions; current
artifact sizes also match. The corrected Rust cache-contract tests still
await workspace validation.

Focused runs use eight independent build pairs and nine replays, excluding
repeat zero from warm position-median quantiles. Every full mask and outcome
matches. These are selected schemas, not a whole-corpus guarantee.

| Reference | Case index | Reference fresh p99 TBM (us) | Fixed fresh p99 TBM (us) | Fresh ratio | Loaded ratio | Build p50 ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| combined | 1 | 18.204 | 15.173 | 0.834 | 1.011 | 1.132 |
| combined | 17 | 77.746 | 30.443 | 0.392 | 1.017 | 0.961 |
| combined | 26 | 96.698 | 34.406 | 0.356 | 1.036 | 1.029 |
| combined | 32 | 58.447 | 21.431 | 0.367 | 0.977 | 0.998 |
| original | 1 | 16.657 | 15.972 | 0.959 | 1.036 | 1.014 |
| original | 17 | 64.099 | 28.111 | 0.439 | 1.073 | 1.012 |
| original | 26 | 88.319 | 36.862 | 0.417 | 1.085 | 0.997 |
| original | 32 | 54.469 | 21.554 | 0.396 | 1.046 | 0.984 |

Aggregate fresh p99 ratios are 0.298 versus combined and 0.334 versus original.
Corresponding aggregate build p99 ratios are 0.998 and 0.985; the per-case build
ratios above must not be hidden by the aggregate. Loaded p99 ratios are 1.050
and 1.065, so the loaded-tail gate is not declared passed from this small set.
The complete 100-schema/four-mode screen is running separately.
