# Exact dynamic cache-key traversal order — 27 September 2026

Code: `90aad8081a257219db13681e9eac3d489388f02c`. Baseline: `bde309073`.

## Change and invariant

Dynamic mask result-cache queries already enumerate parser stacks top-first.
The old key builder repeatedly compared them in reverse order, allocated
reversed copies to hash deep stacks, and reversed them again for owned keys.
The new code sorts, hashes, compares and stores the existing top-first slices.

Reversal is a bijection. Applying one consistent orientation to every path,
then sorting with its correlated maximal-munch exclusions, preserves exactly
the same key equivalence classes. Lexer quotient/scoped coordinates, all stack
values and all exclusions remain in exact equality. Hashes only select buckets.
No cache admission, budget, payload, parser/grammar, compiler or serialization
policy changed. Result-cache keys are private, in-memory values and are not
serialized. This optimization requires no environment variable.

## Paired production-cache measurement

Same Mac, independent frozen baseline/candidate packages, ABBA order, three
fresh loaded-state traversals in each of two processes per side. The metric is
CFA's mask-at-position plus commit-at-position, taking the minimum **complete
same-run pair** across six equal repetitions before calculating percentiles.
All 4,099 JavaScript positions have llguidance measurements. The original
JavaScript grammar is unchanged. Raw observations are retained separately.

| JS dynamic metric | Baseline | Candidate |
| --- | ---: | ---: |
| P50, ms | 0.760125 | 0.748334 |
| P90, ms | 2.630550 | 2.627100 |
| P99, ms | 3.742010 | 3.585919 |
| P99.9, ms | 4.787089 | 4.066492 |
| P100, ms | 4.991458 | 4.205042 |
| Raw maximum, ms | 5.954208 | 5.254167 |

Median JS dynamic construction was 95.777 versus 96.165 ms. No build-time
improvement is attributed to this runtime-only change.

## Regression guards and limits

All 4,099 JS full masks match in both static and dynamic modes. All 9,045
full-mask keys in the 32-schema JSB guard match; performance is scored only at
the 9,021 positions with matching llguidance measurements. Four independent
126-schema streams, each covering 77,582 full masks plus outcome markers,
match their pinned reference hashes in compiled and loaded static/dynamic
forms. The source passed 2,270 Rust tests (zero failures, 55 ignored), the
documentation test and all 42 Python tests.

The first JSB guard showed a small P100 increase, 2,265.833 to 2,321.333 us.
A separate reverse-order guard instead measured 2,275.917 to 2,260.958 us;
P90 was 115.334 to 115.209 us. The small initial JSB difference did not repeat.
This guard is not a full-population JSB performance result.

Static JS retains a small timing shift: first guard P90 29.250 to 30.059 us,
P99 46.126 to 47.251 us and P100 81.000 to 81.916 us. The reverse guard gave
P90 29.308 to 29.875 us, P99 45.460 to 48.381 us and P100 77.375 to 78.833 us.
Do not describe static execution as proven regression-free. The absolute
shift is approximately 1–3 us; no static algorithm was changed. The initial
static-build increase (3.570 to 3.678 s) reversed in the second guard
(3.597 to 3.533 s), so no consistent static-build regression was established.
Cold raw maxima remain substantially higher than stabilized tails.

## Fresh llguidance comparison

A separate `make example-js` run used production cache defaults, thread CPU
clock, zero warmup, six retained measured passes and six builds for each
engine. Both engines produced measurements at all 4,099 positions. Their
grammars differ, so masks are **not** asserted equal across engines; the
optimization's semantic oracle is the unchanged GLRMask reference above.
The harness's cross-grammar discrepancy/adjudication output is retained, not
used as evidence that the optimization changed the accepted language.

| Metric | GLRMask dynamic | llguidance |
| --- | ---: | ---: |
| P50, ms | 0.786001 | 1.073459 |
| P90, ms | 2.710417 | 2.739934 |
| P99, ms | 3.739525 | 3.066152 |
| P100, ms | 4.355500 | 3.504375 |
| Raw maximum, ms | 8.053459 | 4.223458 |
| Median build, ms | 96.299125 | 3.911250 |

The P100 target is still unmet: 4.356 ms versus 3.504 ms. The whole upper tail
is not yet below llguidance, and build time remains far higher. A sampled
native loop exposed the cache-key cost and justified this change, but its
warm cache-hit throughput is not substituted for these fresh trace results.

The companion JSON retains summaries, package/source identities, equal-run
counts, raw maxima, worst positions, capture hashes and an evidence manifest.
The original raw timing sidecar is retained in the isolated worktree.
