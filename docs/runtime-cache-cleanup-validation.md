# Orphan-cache cleanup validation checkpoint

This is a **correctness-validated candidate, not final performance acceptance**.
It is a separate follow-up to the runtime ownership refactor. Do not merge it
into a release solely on the basis of its reduced line count or passing tests.

## Scope and provenance

Base: `a7610942`, the captured runtime refactor descended from `7e84a64c9`.
The two-file Rust delta removes four unused self-loop projection types,
17 associated methods, four always-empty `Arc` fields, their initialization in
five paths, and the corresponding obsolete cache-identity assertion. The sole
remaining accessor consumer was a profiling diagnostic whose lookup was always
`None`; its existing output is retained. Net Rust reduction: 331 lines.

Normal public APIs, vocabulary-equivalence analysis, current persistence formats
and live mask/commit algorithms are unchanged. This claim is about the source
contract, not a promise that native code layout or latency is unchanged. No old
empty-byte-token behavior is silently redefined.

## Correctness evidence

| Gate | Result |
| --- | --- |
| Default library and workspace/all-target compilation | Pass |
| Native Python release build | Pass |
| Root Rust library/integration suite | 1,161 unique top-level passes; 0 failed; 51 ignored, across 38 targets (plus 9 nested child summaries) |
| Python suite | 60 passed |
| Public doctest | 1 passed |
| Independent method audit | 668 retained signatures; 17 approved removals; 7 expected body changes; no unexpected changes |
| Real-vocabulary differential guard | 38 constraint cases and 24 canonical partition cases match |

The differential guard uses 128,256 model-token IDs and compares complete masks
for byte-wise and greedy replays, including reloaded constraints. No artifact
size changes were observed. The larger integrated overhaul has separate source,
binaries and test counts; its results must not be substituted for these gates.

## Timing evidence and remaining flags

Frozen pre-cleanup and post-cleanup extensions were compared on the same CX33,
without concurrent compilation. Broad ABBA runs were followed by focused
ABBA/BAAB retests, then adjacent whole-trace comparisons in separate persistent
processes. No library recompilation or performance-tuning flags were used to
obtain a favorable result.

The longer four-thread/CPU0 comparison uses eight fresh builds and 1,040 traces
per variant **for each of fresh and reloaded constraints**. Quantiles use the
1,024 warm traces; the 16 first-block traces are retained separately in the raw
records. The measuring main thread is pinned after compilation. These are selected greedy traces, not a
canonical corpus distribution. Ratios are candidate/baseline; above 1 is slower.

| Case, FAST_RUNTIME | State | Raw mask p50 ratio | Baseline p99 (us) | Candidate p99 (us) | p99 ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| integer | Fresh | 0.987 | 9.388 | 10.640 | 1.133 |
| integer | Reloaded | 0.997 | 8.305 | 8.787 | 1.058 |
| string_array | Fresh | 1.022 | 20.267 | 19.417 | 0.958 |
| string_array | Reloaded | 1.023 | 17.954 | 21.089 | 1.175 |

Identical-binary controls also varied: two processes loading the same candidate
extension produced an integer reloaded-mask p99 ratio of 1.291 and a string-array
reloaded-mask ratio of 1.145. That demonstrates measurement variability; it does
**not** prove that every candidate slowdown is noise. Static integer fresh-mask
and string-array reloaded-mask tails remain review flags. Keep this candidate
isolated until the performance decision is supported by broader evidence.

The native timing hook uses `Instant` elapsed wall time around the operation.
O2's internal commit timer additionally includes Python overhead. Do not label
these measurements CPU-time measurements or the O2 commit samples purely native.
Short-trace maxima and quantiles over per-position medians are not full-corpus
p99.9/p100 guarantees. Build, save/load and commit samples are retained separately.

## Reproduction and integration

Engineering assets are in `/root/runtime-cleanup-789322` on the build host. See
`logs/validation-summary.json`, `logs/semantic-comparison.json`,
`logs/final-method-audit.json`, `logs/final-performance-review.json` and the
adjacent-run raw JSONL files. Native package hashes and corpus identity are
recorded with the results; both packages are independent of Cargo output.

The integration-ready patch reconciles the separately outlined cache test and
includes the runtime documentation. It has only been checked for application;
it has **not** been applied to the combined overhaul. Combined compilation,
correctness and performance gates are required after that later integration.

## Isolated combined follow-up

This patch is now applied in the separately owned mask-replay follow-up,
`/root/glrmask-mask-replay-789322-20260927`. The independent integration tree
and its index remain untouched. The preceding final-mask replay refactor has
separate frozen results; these do not automatically validate this cache removal.
Combined rebuilding, full Rust tests, semantic guards and paired measurements
are required before this combined follow-up is accepted. The original patch and
all previous raw evidence remain preserved in `/root/runtime-cleanup-789322`.

The original test count above has been corrected to count the last summary
per Cargo target: nine nested self-spawned test summaries are not nine additional
unique tests.
