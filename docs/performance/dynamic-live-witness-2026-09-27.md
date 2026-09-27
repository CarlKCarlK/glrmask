# Exact live-branch vocabulary-subtree proof

Code: `6f9f1c4824f3db98ef8989459c8c7e95385c1a36`. Independently frozen baseline: `68a648fc3ee90feaf54209bbd8d3c8a307956539`.

## What is proved

A dynamic frontier can contain one branch that accepts every token below a vocabulary-trie node even though its other branches change. The new shortcut proves that one **Passed** branch remains physically live and returns to the same lexer state on every byte in the remaining subtree alphabet. Finalizers may add alternatives, but the authoritative executor also retains that live continuation. Induction over the remaining bytes therefore proves every endpoint under the trie node is allowed.

A scalar lexer state can itself represent an exact union. The proof may select one physical union member, keeping its parser coordinate unchanged. It never combines different members' alphabets, borrows another parser's admission, or uses an unresolved maximal-munch guard. Missing proofs and budget exhaustion fall back to the unchanged executor; they do not reject tokens. Existing token-alias, deferred-output, polarity, ignored-output and ancestor-restoration logic is reused.

Proof discovery is bounded to eight physical members or branches, 64 cached lexer-row proofs per mask, and subtrees saving at least 64 operations and containing at least 16 tokens. The discovery dispatcher is kept outside the large byte loop. There is no additional per-parser witness memo, compiler algorithm change, or grammar rewrite.

The optimization is enabled by default. `GLRMASK_DISABLE_LIVE_BRANCH_WITNESS=1` is only a differential-debugging switch.

## Original-JavaScript interleaved comparison

The independent published binary, the new binary with the proof disabled (`reference`), and its production default (`candidate`) each run twice in mirrored order. Each process measures three fresh-state passes on all 31 examples. All 4,099 full masks agree. Quantiles use the minimum complete mask-plus-commit pair at each matching position across six passes, never separate minima of mask and commit. Runtime timing uses thread CPU time on the same Apple Silicon Mac, Rust 1.95.0 native-CPU release build, Python 3.12 and mimalloc purge delay -1.

| Metric | published | reference | candidate |
|---|---:|---:|---:|
| TBM p50, microseconds | 689.959 | 695.125 | 696.208 |
| TBM p90, microseconds | 2270.834 | 2296.700 | 2320.299 |
| TBM p99, microseconds | 3389.160 | 3402.084 | 3431.875 |
| TBM p99.9, microseconds | 3857.600 | 3877.202 | 3823.292 |
| TBM p100, microseconds | 4410.125 | 4470.792 | 3964.166 |
| Raw maximum, microseconds | 5989.041 | 4928.334 | 4292.542 |
| Median native build, milliseconds | 105.560 | 102.375 | 100.016 |

Raw maxima are separate observations, not replaced by the stabilized curve. Build observations remain in the record even though this is a runtime-only change. Interleaving controls is important because system storage-management processes were active during development.

## Guardrails

The 32-schema dynamic JSB guard compares 9,045 complete masks and scores only 9,021 positions with finite matching llguidance measurements. The static JavaScript guard checks all 4,099 positions. A schema unsupported by llguidance is not a performance target.

### Dynamic JSB guard

| Metric | published | reference | candidate |
|---|---:|---:|---:|
| TBM p50, microseconds | 44.875 | 45.750 | 45.750 |
| TBM p90, microseconds | 135.001 | 135.792 | 135.708 |
| TBM p99, microseconds | 397.326 | 396.333 | 400.150 |
| TBM p99.9, microseconds | 2630.188 | 2622.970 | 2618.332 |
| TBM p100, microseconds | 2666.417 | 2655.834 | 2671.458 |
| Raw maximum, microseconds | 2837.001 | 2800.083 | 2784.833 |

### Static JavaScript guard

| Metric | published | reference | candidate |
|---|---:|---:|---:|
| TBM p50, microseconds | 22.542 | 22.750 | 22.583 |
| TBM p90, microseconds | 34.292 | 34.466 | 34.250 |
| TBM p99, microseconds | 53.585 | 54.293 | 53.339 |
| TBM p99.9, microseconds | 84.294 | 85.366 | 84.736 |
| TBM p100, microseconds | 93.084 | 93.334 | 94.375 |
| Raw maximum, microseconds | 101.208 | 100.000 | 102.334 |
| Median native build, milliseconds | 3730.865 | 3776.935 | 3808.433 |

## Native 1,000-schema checks

This is the first 1,000 cases in the pinned llguidance-success corpus, not a random sample or the full population. Every selected schema builds, all outcome records match, and both sides score the same 367,640 finite llguidance-measured intervals. Native TBM is commit through the **next** mask, unlike the Python same-position convention above. Its quantiles use the minimum of two complete independent observations; raw maxima remain separate. Frozen native source, harness, locks, binary hashes and worst positions are retained.

### Dynamic

| Metric | published | candidate |
|---|---:|---:|
| Build p50, milliseconds | 1.644 | 1.648 |
| Build p90, milliseconds | 4.634 | 4.611 |
| Build p99, milliseconds | 22.817 | 22.243 |
| Build p100, milliseconds | 51.965 | 51.422 |
| TBM p50, microseconds | 6.958 | 6.958 |
| TBM p90, microseconds | 76.000 | 76.458 |
| TBM p99, microseconds | 196.484 | 198.993 |
| TBM p99.9, microseconds | 587.469 | 589.647 |
| TBM p100, microseconds | 2717.125 | 2740.042 |
| Raw maximum, microseconds | 2721.875 | 2764.959 |

### Static

| Metric | published | candidate |
|---|---:|---:|
| Build p50, milliseconds | 10.240 | 10.153 |
| Build p90, milliseconds | 77.267 | 75.488 |
| Build p99, milliseconds | 249.197 | 245.613 |
| Build p100, milliseconds | 685.834 | 658.620 |
| TBM p50, microseconds | 4.083 | 4.042 |
| TBM p90, microseconds | 6.917 | 6.917 |
| TBM p99, microseconds | 14.667 | 14.417 |
| TBM p99.9, microseconds | 29.958 | 30.390 |
| TBM p100, microseconds | 49.583 | 49.375 |
| Raw maximum, microseconds | 74.583 | 139.375 |

## Fresh llguidance comparison

A separate same-machine `make example-js` run uses six builds and six measured passes, no warmup, production caches, and the retained raw-timing sidecar. llguidance's grammar differs; its masks are not treated as the correctness oracle. GLRMask's masks are checked against its unchanged-language reference.

| Metric | glrmask_dynamic | llguidance_native |
|---|---:|---:|
| TBM p50, microseconds | 733.792 | 1265.250 |
| TBM p90, microseconds | 2418.883 | 3223.241 |
| TBM p99, microseconds | 3510.481 | 3545.696 |
| TBM p99.9, microseconds | 3869.078 | 3757.650 |
| TBM p100, microseconds | 4043.458 | 4093.084 |
| Raw maximum, microseconds | 4716.208 | 4650.959 |
| Median native build, milliseconds | 98.433 | 4.588 |

The following checks inspect every empirical quantile knot, including the exact lower boundary of each band. They distinguish the stabilized per-position curve from the distribution of all 24,594 raw observations. A finite measured dominance result is not a latency guarantee for arbitrary programs or future calls.

| Observation policy / percentile band | GLRMask strictly below at every checked point | Worst GLRMask / llguidance ratio |
|---|---|---:|
| minimum_of_six / p0_to_p100 | False | 1.4218 |
| minimum_of_six / p90_to_p100 | False | 1.0441 |
| minimum_of_six / p99_to_p100 | False | 1.0441 |
| all_raw_observations / p0_to_p100 | False | 1.3171 |
| all_raw_observations / p90_to_p100 | False | 1.0140 |
| all_raw_observations / p99_to_p100 | False | 1.0140 |

## Correctness evidence and rejected variants

The clean-default workspace gate passes 2,286 tests with zero failures and 55 ignored; the tally uses each Cargo harness's final summary rather than double-counting nested child output. All 42 Python tests and one documentation test pass. Unit tests compare the authoritative executor through finalizing self-edges and changing alternatives, nonidentity exact unions, Pending guards, dead parsers, invalid coordinates and exhausted budgets. The dispatcher is also exercised through its single, paired and borrowed-many representations.

Four independent static/dynamic compiled/loaded streams each preserve all 77,582 complete masks and outcome markers across 126 schemas. Every decoded byte is compared against the unchanged reference without saving gigabytes of uncompressed masks. The companion JSON records digests and sizes. This is correctness evidence, not additional performance scoring.

The earlier Many-only proof was correct but effectively neutral and was not retained as a production mode. Adding a separate per-parser proof memo also failed to justify its allocation and lookup overhead. Archived experiments retain their original raw spikes and mixed results; they are not substituted for this default release's measurements.

Private reproduction scripts and frozen evidence are in `.benchmarks/dynamic-live-witness/final`. No Windows or new paid server was used. The public JSON companion provides source provenance, hashes, full validation summaries and quantile-curve checks.

## Raw static outlier follow-up

The interrupted recheck was repaired and rerun without mixing partial outputs. Six fresh native processes per binary tested the same two llguidance-supported schemas in alternating order. All outcomes matched. The `typingsrc` interval-339 spike reproduced in **both** binaries on the first process: 151.584 microseconds in published68a and 122.875 in the candidate. Subsequent observations were 24.459–31.250 versus 21.250–27.458 microseconds. This does not establish the root cause, but does refute a candidate-specific explanation. The original raw observations remain in the report. The original `o60887` interval-139 outlier did not recur (3.750–4.375 microseconds in the new runs).

The native summary's copied comparison description was corrected to the actual68a/6f9 revisions verified by the frozen binary hashes and individual raw runs; no timing or selection was altered. The JSON contains the correction provenance.
