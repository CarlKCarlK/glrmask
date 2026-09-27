# Pending-range ownership in weight construction

Validated on the Mac on 27 September 2026. Source control for the paired test was
`e07744d3a3b4358bc5c8f1f6cc72e93415d56487`. The accompanying JSON retains measurements,
per-case raw values, binary fingerprints and evidence hashes.

## Change and invariants

`CompactRangeBuilder` stores an optional owned pending range rather than separate
fields with a shared empty-token-set placeholder. Flushing moves the real value
into exactly the same `RangeMapBlaze::extend_simple` call. Previously every flush
cloned the shared empty `Arc`, and the next push dropped it again.

This eliminates unnecessary reference-count writes. It does not alter the
adjacency test, output ranges, insertion order, handling of overlaps or out-of-order
input, token-set allocation representatives, public API, grammar, cache policy,
or serialized format. There is no environment switch to enable the improvement.

A test compares the former implementation with 1,024 deterministic streams,
including explicit flushes, overlaps, out-of-order ranges, empty token sets,
zero and `u32::MAX`, and equal token sets in different allocations. It checks
both raw map values and exact token-set allocation identities before interning.

## Static JavaScript

The unchanged original grammar and all 31 examples were used. Each of four
interleaved baseline/candidate/candidate/baseline processes performed three
builds and three fresh loaded-state traversals. Every one of the 4,099 full
vocabulary masks matched, and all scored positions had llguidance measurements.

Median build time across six builds per side: **3619.845 →
3426.742 ms** (5.33% lower).
There is visible run-to-run variation, retained in the JSON; this is not a
constant speedup on every individual run. Stabilized same-position TBM P90 was
30.166 → 29.133 µs and P100
80.458 → 80.042 µs. Raw maxima were
271.958 → 273.375 µs, so the cold tail is not fixed.

## Native static JSB guard

The fixed first 1,000 retained llguidance-success schemas all built in both
versions with identical outcomes. Runtime scoring matched all
367,640 llguidance-measured `(problem, example, position)` intervals.
The same Rust runner, dependency lock, compiler flags and allocator were used.
Four 250-schema chunks used interleaved ABBA/BAAB order and fresh native processes.
The table is the distribution of the minimum of two builds per schema.

| Build time (ms) | Control | Candidate |
|---|---:|---:|
| P50 | 9.524 | 9.419 |
| P90 | 74.622 | 72.279 |
| P95 | 128.381 | 129.446 |
| P99 | 227.215 | 223.232 |
| P100 | 655.531 | 587.971 |

The sum of these minimum build times was 27.043 →
26.887 seconds, and the median paired build ratio was
0.99311. The common-case improvement is modest;
P95 did not improve. This panel is not the entire JSB population.

Native TBM measures commitment to the next ready mask, unlike the Python
same-position mask-plus-commit column above. Complete intervals, not independently
minimized mask and commit parts, were minimized across repetitions.

| Native TBM (µs) | Control | Candidate |
|---|---:|---:|
| P50 | 3.541 | 3.500 |
| P90 | 6.083 | 6.041 |
| P99 | 13.042 | 12.833 |
| P100 | 43.916 | 44.459 |

The candidate does not establish a material TBM change; raw maxima
161.709/140.875 µs are retained separately.

## Regression investigation

A six-build dynamic-JavaScript screen initially showed an apparent build
regression. Publication was withheld pending eight further interleaved processes,
12 builds per process, producing 48 builds per version. Their medians were
**92.934 → 92.545 ms**, and P90 was
113.554 → 106.299 ms. All 96 serialized artifacts were
byte-identical. The initial slowdown did not persist in the longer measurement.
All 4,099 dynamic-JavaScript masks matched in the runtime guard.

Twenty static schemas were separately selected: the 12 largest apparent
regressions among builds exceeding 20 ms, plus eight baseline quantile positions.
Four independent builds per version were run in forward/reverse corpus order.
The large original regressions did not persist. Small residual increases remain
in some cases (up to roughly 4.6% among this selected panel); no claim is made
that every schema improved. Exact selections and all samples are retained.

## Correctness and integration gates

- 2,270 Rust workspace tests passed; zero failed; 55 ignored.
- Rust documentation test, release example checks and 42 Python tests passed.
- Four independent compiled/loaded captures (static and dynamic) each matched
  77,582 full vocabulary masks and outcome markers across 126 schemas against
  retained release oracles. The decoded streams were hashed incrementally;
  only compressed captures were retained.

Two other experiments were not adopted: replacing guarded weak-reference
upgrades showed no build benefit, and allocating only touched parser contribution
buckets added less than 0.5% over this candidate. Neither is in this change.

## Evidence and reproducibility

Full metadata and per-case values: `weight-pending-range-2026-09-27.json`.
Local raw evidence and frozen binaries: `/Users/isaacbreen/Projects2/worktrees/glrmask-static-build-459964r-20260927/.benchmarks/static-build`.
Scripts there include `screen_js.py`, `native_compare.py`, `recheck_builds.py`,
`validate_captures.py` and the fixed native runner sources/locks. Benchmark
processes pin their extension/binary hashes and original grammar, clear inherited
experiment settings, and use `MIMALLOC_PURGE_DELAY=-1`. Runtime clocks are thread
CPU; build clocks include parallel wall-clock compilation. Vocabulary-only
preparation is outside schema build timing. Unsupported llguidance cases are
never part of performance scoring.
