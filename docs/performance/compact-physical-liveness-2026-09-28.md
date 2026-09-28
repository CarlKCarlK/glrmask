# Bounded physical-liveness cache rows

Code: `70dc1290a41857366593ff5eaedf7dbdf5319549`; measured integrated source: `44572e3eb92c99a9b5d8bb6b748e55410917009c`.
Independent published control: `1e64511f7632ad52b5258204065501317cfdc960`. The candidate includes upstream `ae8e5ab6b81d02bcb15273308109ca689a2cc84e`.

## Change and safety invariant

The dynamic mask walker previously allocated a lexer-domain-sized byte row for each speculative parser node. It now uses sixteen full-tag entries per parser node when a dense row would exceed 128 bytes. Tags retain the entire lexer ID, and parser nodes select disjoint rows. Unknown entries and collisions invoke the original exact predicate; an eviction cannot accept or reject a token by itself. Initial-root identity remains outside this physical-liveness cache. Grammar, parser admission, maximal-munch guards, and serialized formats are unchanged.

The dense fast path still reads stable byte buffers. Before its first pointer escapes, every compact row is promoted once, retaining all cached values and unknowns. Further writes and new parser rows remain dense. Tests exercise tag collisions, maximum IDs, separate parser coordinates, promotion, and pointer stability across node-vector growth.

The optimization is on by default. `GLRMASK_DISABLE_COMPACT_BOUNDARY_ROWS=1` is a diagnostic control, not a required opt-in.

## Release decision

Accepted for bounded cache storage and repeatable JavaScript P90/P99 improvements. This is not evidence that every percentile improves or that the llguidance tail target has been reached. The final-source census repeats an approximately 82-fold reduction in logical liveness-cell payload at both measured JS tail positions, with exact mask equality. The primary screen improves P90/P99 from 2,042.692/2,994.032 µs to 1,975.758/2,899.122 µs, but its independent P100 is slightly higher: 3,408.125 versus 3,431.875 µs. A separate reversed-order confirmation with eight observations per position improves P90/P99/P100 from 2,021.908/2,950.351/3,493.083 µs to 1,972.300/2,887.155/3,418.583 µs. Both same-binary comparisons favor compact rows at these quantiles. The independent P100 effect remains small and variable.

The current-default workspace, Python and documentation tests, full JS masks, four compiled/loaded 126-schema streams, and both native 1,000-schema panels pass with identical finite-LL coverage and outcomes. The JSB dynamic guard's P100 remains approximately 1.1–1.3% above the independent older binary; same-binary disabled/default differences are smaller. That residual difference is retained rather than hidden. Static guards are mostly unchanged, with occasional cold/raw spikes. The 166.5 µs create_invoice outlier replays at 12.417–15.167 µs across six default fresh processes. Cold kb223 spikes recur in the independent control and the current binary with compact rows disabled. This does not establish their cause or prove a future cold-latency bound.

The apparent native static build maximum increase from 651.3 to 716.7 ms does not repeat on o21074: six-process medians are 625.041/625.199/629.337 ms for published/disabled/default. JS build medians move in opposite directions between screens; neither a consistent compiler-time benefit nor a material regression has been established. Original build arrays and all raw maxima remain in this report. The separate fresh llguidance comparison is still unfavorable at the upper tail: GLRMask P100 is 3.627 ms versus 3.583 ms, with raw maxima of 6.795 versus 6.638 ms. Every checked stabilized P99–P100 point fails strict dominance. Further optimization is required; this release does not complete the six-area task.

## Original JavaScript

The independent published control, the new binary with compact rows disabled (`reference`), and its production default (`candidate`) each run twice in mirrored order. Every process measures three fresh-state repetitions across 31 examples. All 4,099 complete masks agree. Percentiles use the minimum **complete same-run mask-plus-commit pair** per position across six repetitions; mask and commit minima are never combined from different runs. Raw maxima retain every observation.

Runtime clocks are thread CPU time on the same Apple Silicon Mac; native compiler measurements are wall time. Rust 1.95.0 native-CPU release flags and mimalloc purge delay −1 are retained. The separately built control predates the integrated boundary-minimizer update, so the same-binary disabled control is also included to isolate the cache policy. Absolute values from older development screens are not mixed into these tables.

| Metric | published | reference | candidate |
|---|---:|---:|---:|
| TBM p50, microseconds | 599.792 | 608.583 | 597.250 |
| TBM p90, microseconds | 2042.692 | 2043.276 | 1975.758 |
| TBM p99, microseconds | 2994.032 | 2980.704 | 2899.122 |
| TBM p99.9, microseconds | 3323.134 | 3386.160 | 3286.031 |
| TBM p100, microseconds | 3408.125 | 3676.292 | 3431.875 |
| All-raw maximum, microseconds | 6434.833 | 4428.167 | 4176.500 |
| Median native build, milliseconds | 96.198 | 96.140 | 91.903 |

## Cross-workload guards

The 32-schema dynamic JSON Schema guard verifies 9,045 complete masks and scores only 9,021 finite matching llguidance positions.

| Metric | published | reference | candidate |
|---|---:|---:|---:|
| TBM p50, microseconds | 39.333 | 39.792 | 39.917 |
| TBM p90, microseconds | 115.667 | 116.917 | 117.541 |
| TBM p99, microseconds | 340.125 | 341.342 | 341.525 |
| TBM p99.9, microseconds | 2224.883 | 2241.524 | 2246.375 |
| TBM p100, microseconds | 2278.500 | 2285.959 | 2307.126 |
| All-raw maximum, microseconds | 2457.834 | 2508.584 | 2602.209 |

Static JavaScript checks the same 4,099 positions:

| Metric | published | reference | candidate |
|---|---:|---:|---:|
| TBM p50, microseconds | 19.417 | 19.541 | 19.541 |
| TBM p90, microseconds | 29.917 | 30.174 | 29.916 |
| TBM p99, microseconds | 48.543 | 48.666 | 47.585 |
| TBM p99.9, microseconds | 72.955 | 73.281 | 72.530 |
| TBM p100, microseconds | 78.916 | 80.458 | 82.042 |
| All-raw maximum, microseconds | 121.625 | 122.041 | 113.000 |
| Median native build, milliseconds | 3562.347 | 3469.726 | 3469.192 |

## Native 1,000-schema guard

This fixed first 1,000-case llguidance-success panel is neither a random sample nor a full-population claim. Each side builds every selected schema and has identical outcomes and 367,640 finite matching LL runtime intervals. Native intervals measure commit through the **next** mask. Percentiles take the minimum of two complete observations; raw maxima are preserved separately.

### Dynamic

| Metric | published | candidate |
|---|---:|---:|
| TBM p50, microseconds | 6.250 | 6.208 |
| TBM p90, microseconds | 71.708 | 71.958 |
| TBM p99, microseconds | 184.317 | 182.000 |
| TBM p99.9, microseconds | 523.002 | 520.555 |
| TBM p100, microseconds | 2455.333 | 2425.208 |
| All-raw maximum, microseconds | 2486.042 | 2572.291 |
| Build p50, milliseconds | 1.680 | 1.666 |
| Build p90, milliseconds | 4.700 | 4.543 |
| Build p99, milliseconds | 21.994 | 21.131 |
| Build p100, milliseconds | 49.347 | 47.349 |

### Static

| Metric | published | candidate |
|---|---:|---:|
| TBM p50, microseconds | 3.542 | 3.542 |
| TBM p90, microseconds | 6.125 | 6.125 |
| TBM p99, microseconds | 13.583 | 13.500 |
| TBM p99.9, microseconds | 27.598 | 27.416 |
| TBM p100, microseconds | 130.084 | 43.041 |
| All-raw maximum, microseconds | 132.209 | 166.500 |
| Build p50, milliseconds | 10.051 | 10.178 |
| Build p90, milliseconds | 74.965 | 76.350 |
| Build p99, milliseconds | 232.473 | 233.650 |
| Build p100, milliseconds | 651.288 | 716.696 |

## Fresh llguidance comparison

A separate `make example-js` run records six builds and six complete measured passes, no warmup, and all raw timings. llguidance uses a different grammar. Its masks are not the correctness oracle; GLRMask masks are compared with the unchanged-language GLRMask reference.

| Metric | glrmask_dynamic | llguidance_native |
|---|---:|---:|
| TBM p50, microseconds | 658.750 | 1083.250 |
| TBM p90, microseconds | 2234.593 | 2786.491 |
| TBM p99, microseconds | 3178.396 | 3146.155 |
| TBM p99.9, microseconds | 3480.636 | 3343.134 |
| TBM p100, microseconds | 3627.084 | 3582.874 |
| All-raw maximum, microseconds | 6795.084 | 6637.875 |
| Median native build, milliseconds | 122.389 | 4.071 |

The entire empirical quantile curve is checked, including exact 90th and 99th percentile boundaries. These finite observations are not a latency guarantee for future programs or calls.

| Observation policy / band | Strictly below LL at every checked point | Worst GLRMask/LL ratio |
|---|---|---:|
| minimum_of_six / p0_to_p100 | False | 1.3195 |
| minimum_of_six / p90_to_p100 | False | 1.0432 |
| minimum_of_six / p99_to_p100 | False | 1.0432 |
| all_raw_observations / p0_to_p100 | False | 1.1403 |
| all_raw_observations / p90_to_p100 | False | 1.0922 |
| all_raw_observations / p99_to_p100 | False | 1.0922 |

## Correctness, memory accounting, and reproduction

The clean-default workspace run passes 2,294 tests, with zero failures and 55 ignored. Counts use the final summary per Cargo harness, avoiding nested-output double counting. All 42 Python tests and the documentation test pass. Four static/dynamic compiled/loaded streams each preserve 77,582 full masks and outcome markers across 126 schemas. Compressed streams are verified without retaining gigabytes of uncompressed masks.

The development structural census observed liveness-cell payload reductions from 2,603,874 to 31,616 bytes at JS 30:578 and from 3,731,868 to 45,312 bytes at JS 23:44. These are logical cache-cell sizes, excluding spare capacity, allocation headers, and other parser structures—not process RSS. Any repeated final-source census is retained separately in the companion JSON.

The earlier failed release attempt exhausted disk while compiling the Python extension; its completed workspace/doc logs and failure log were preserved before recovery. The current package is a fresh completed default build, not that failed attempt or the prior opt-in prototype. Source, frozen controls, and raw evidence were retained. Reproduction scripts and compact artifacts are under `.benchmarks/liveness-cache/final`; the JSON companion records source provenance and evidence hashes.

Lazy-only storage and variable-capacity trials were not separately enabled. The two-bit representation remained an untested draft and is not part of this release.

## Additional checks

`tail-recheck/summary.json`: Six fresh native processes per variant in mirrored order on five fixed LL-success schemas selected from the actual static raw/build outliers. Baseline and candidate binary hashes are pinned; reference is the candidate with compact rows disabled. Thread-CPU whole commit-to-next-mask intervals only at exact finite LL keys. Original thousand-schema maxima are not overwritten or replaced.

`structural-memory/summary.json`: Final integrated default versus same-binary disabled. Logical liveness-cell payload only; excludes vector spare capacity, headers, allocator overhead and all other mask memory. Profile timings are retained but are not acceptance observations.

`confirmation-screen/summary.json`: See companion JSON for exact observations.

### Independent confirmation run

The same frozen binaries were compared again with the initial variant order reversed and four fresh-state repetitions per example in each of two processes (eight complete observations per position). These are separate observations, not replacements for the first release screen or the fresh llguidance comparison.

#### js

| Metric | published | reference | candidate |
|---|---:|---:|---:|
| TBM p50, microseconds | 596.000 | 606.416 | 593.625 |
| TBM p90, microseconds | 2021.908 | 2058.067 | 1972.300 |
| TBM p99, microseconds | 2950.351 | 2995.086 | 2887.155 |
| TBM p99.9, microseconds | 3276.038 | 3408.712 | 3271.598 |
| TBM p100, microseconds | 3493.083 | 3609.125 | 3418.583 |
| All-raw maximum, microseconds | 4663.208 | 6581.625 | 4404.500 |
| Median native build, milliseconds | 99.204 | 101.758 | 109.440 |

#### jsb

| Metric | published | reference | candidate |
|---|---:|---:|---:|
| TBM p50, microseconds | 38.875 | 39.083 | 38.417 |
| TBM p90, microseconds | 114.833 | 115.917 | 114.917 |
| TBM p99, microseconds | 335.243 | 338.534 | 335.000 |
| TBM p99.9, microseconds | 2204.514 | 2232.530 | 2212.514 |
| TBM p100, microseconds | 2261.125 | 2289.041 | 2286.625 |
| All-raw maximum, microseconds | 2824.208 | 2504.084 | 2497.875 |

#### static

| Metric | published | reference | candidate |
|---|---:|---:|---:|
| TBM p50, microseconds | 19.334 | 19.541 | 19.333 |
| TBM p90, microseconds | 29.708 | 29.750 | 29.500 |
| TBM p99, microseconds | 47.835 | 47.589 | 47.333 |
| TBM p99.9, microseconds | 72.867 | 71.689 | 71.084 |
| TBM p100, microseconds | 78.500 | 81.167 | 77.792 |
| All-raw maximum, microseconds | 103.542 | 117.459 | 146.041 |
| Median native build, milliseconds | 3479.117 | 3506.693 | 3444.306 |

### Native outlier follow-up

Six fresh native processes per variant in mirrored order on five fixed LL-success schemas selected from the actual static raw/build outliers. Baseline and candidate binary hashes are pinned; reference is the candidate with compact rows disabled. Thread-CPU whole commit-to-next-mask intervals only at exact finite LL keys. Original thousand-schema maxima are not overwritten or replaced.

| Original suspect interval | Published range, µs | Same-binary disabled range, µs | Default range, µs |
|---|---:|---:|---:|
| jsb/data/Kubernetes---kb_223_Normalized / 0 / 1 | 16.083–76.166 | 19.042–79.041 | 17.459–24.625 |
| jsb/data/Kubernetes---kb_223_Normalized / 0 / 6 | 25.833–123.291 | 24.250–154.792 | 24.500–29.666 |
| jsb/data/Glaiveai2K---create_invoice_beb99d93 / 0 / 25 | 11.125–14.208 | 10.791–18.042 | 11.250–16.083 |
| jsb/data/Glaiveai2K---create_invoice_beb99d93 / 0 / 67 | 12.959–16.584 | 12.000–16.000 | 12.417–15.167 |
| jsb/data/Snowplow---sp_345_Normalized / 1 / 16 | 40.333–43.500 | 38.792–44.958 | 37.792–46.875 |

All original raw spikes remain in the release records. A negative recheck does not establish their cause or guarantee future calls are bounded by the recheck maximum. The companion JSON retains all build samples and all six readings at each selected interval.
