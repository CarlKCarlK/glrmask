# Partial deterministic-segment pop: Windows release evidence

Code: `e7b2f829ddbf07aa66feb090bf84000923008e1c`. Independent baseline: `496c3ef121896907dcf089553664203bc340fabc`.

## Change

When a parser pop ends inside a compressed deterministic segment, the fast path now returns the remaining segment prefix over the unchanged lower graph and accumulator. The previous path declined and repeated the same operation through general graph memo tables. There are no grammar, vocabulary, cache-budget, or serialized-format changes. The default is enabled; `GLRMASK_DISABLE_PARTIAL_SEGMENT_POP` retains the reference path.

## JavaScript

Both controls use the original grammar and the full 128,256-token vocabulary. Each process reconstructs fresh constraints for every example/repetition. The table is the minimum complete mask-plus-commit pair per position across eight observations on the same 4,099 llguidance-measured positions; raw maxima are retained separately. llguidance uses its own unchanged companion grammar and the pinned CFA byte-tokenizer, JS slices and parser limits.

| Dynamic JavaScript | P50 (ms) | P90 (ms) | P99 (ms) | P100 (ms) | Raw max (ms) |
|---|---:|---:|---:|---:|---:|
| baseline | 0.980 | 3.286 | 4.742 | 5.660 | 6.200 |
| reference | 0.974 | 3.308 | 4.811 | 5.702 | 10.631 |
| partial | 0.944 | 3.166 | 4.672 | 5.574 | 15.864 |
| llguidance | 1.080 | 3.857 | 4.690 | 5.827 | 7.386 |

Full empirical quantile-knot checks, including all raw observations, are retained in the JSON report. A lower stabilized maximum alone does not imply whole-tail dominance or a bound on future calls.

| Static JavaScript | P90 (µs) | P99 (µs) | P100 (µs) | Raw max (µs) |
|---|---:|---:|---:|---:|
| baseline | 51.900 | 83.200 | 124.900 | 309.000 |
| reference | 50.300 | 81.604 | 125.100 | 227.500 |
| partial | 49.200 | 78.702 | 123.100 | 160.900 |

### Reversed-order confirmation

A second complete comparison reverses process order and again retains eight complete observations per position. The first comparison remains in full; these results do not replace it. A report-only mask-count error after all timing subprocesses completed was repaired from the original raw files and independently audited, without rerunning the timing.

| Dynamic JavaScript | P90 (ms) | P99 (ms) | P100 (ms) | Raw max (ms) |
|---|---:|---:|---:|---:|
| baseline | 3.293 | 4.748 | 5.666 | 6.665 |
| reference | 3.308 | 4.830 | 5.658 | 6.905 |
| partial | 3.167 | 4.694 | 5.656 | 6.913 |
| llguidance | 3.867 | 4.676 | 5.776 | 24.096 |

Neither Windows comparison establishes strict dominance over llguidance throughout P99–P100. The full quantile-knot failures, including raw observations, are preserved in the JSON.

### Build-only confirmation

Thirty-two interleaved dynamic-JavaScript builds per variant use the same original grammar and frozen binaries. All emitted artifacts match. These grammar-build intervals exclude vocabulary preparation equally; they are not the time needed to compile the Rust package.

| Variant | Median build (ms) | P90 build (ms) |
|---|---:|---:|
| baseline | 104.964 | 111.409 |
| reference | 104.176 | 110.358 |
| partial | 103.203 | 112.628 |

## Full llguidance-supported JSB population

All 8,332 schemas selected solely from the pinned llguidance-success reference were attempted. Existing GLRMask failures remain explicit. Both binaries must have identical semantic outcomes and identical finite llguidance-matched interval keys. Four interleaved process runs per chunk give two observations per binary. Native TBM is commit_i through mask_i+1, which is distinct from the JavaScript same-position pair above.

### Static

| Build | P50 (ms) | P90 (ms) | P99 (ms) | P100 (ms) |
|---|---:|---:|---:|---:|
| published | 9.912 | 58.886 | 219.546 | 1165.852 |
| candidate | 9.952 | 58.310 | 222.389 | 1202.034 |

| TBM | P50 (µs) | P90 (µs) | P99 (µs) | P100 (µs) | Raw max (µs) |
|---|---:|---:|---:|---:|---:|
| published | 5.100 | 9.200 | 17.400 | 100.200 | 1690.100 |
| candidate | 5.000 | 9.100 | 17.400 | 91.400 | 75632.700 |

published: 8326 successful schemas; 2925984 measured intervals. Failures: `jsb/data/Github_medium---o60170`, `jsb/data/JsonSchemaStore---meta.schema`, `jsb/data/Github_hard---o13029`, `jsb/data/Github_easy---o21053`, `jsb/data/Kubernetes---kb_1161_Normalized`, `jsb/data/Github_medium---o69991`.

candidate: 8326 successful schemas; 2925984 measured intervals. Failures: `jsb/data/Github_medium---o60170`, `jsb/data/JsonSchemaStore---meta.schema`, `jsb/data/Github_hard---o13029`, `jsb/data/Github_easy---o21053`, `jsb/data/Kubernetes---kb_1161_Normalized`, `jsb/data/Github_medium---o69991`.

### Dynamic

| Build | P50 (ms) | P90 (ms) | P99 (ms) | P100 (ms) |
|---|---:|---:|---:|---:|
| published | 1.706 | 4.531 | 20.952 | 165.957 |
| candidate | 1.698 | 4.504 | 21.028 | 169.414 |

| TBM | P50 (µs) | P90 (µs) | P99 (µs) | P100 (µs) | Raw max (µs) |
|---|---:|---:|---:|---:|---:|
| published | 6.400 | 67.300 | 181.800 | 3911.600 | 4352.100 |
| candidate | 6.400 | 66.300 | 178.817 | 3798.800 | 4477.200 |

published: 8327 successful schemas; 2925984 measured intervals. Failures: `jsb/data/Github_medium---o60170`, `jsb/data/JsonSchemaStore---meta.schema`, `jsb/data/Github_hard---o13029`, `jsb/data/Github_easy---o21053`, `jsb/data/Github_medium---o69991`.

candidate: 8327 successful schemas; 2925984 measured intervals. Failures: `jsb/data/Github_medium---o60170`, `jsb/data/JsonSchemaStore---meta.schema`, `jsb/data/Github_hard---o13029`, `jsb/data/Github_easy---o21053`, `jsb/data/Github_medium---o69991`.

## Raw outliers and replay

The primary JavaScript candidate recorded a **15.864 ms** call at example24/position22; the full native-static cohort recorded **75.6327 ms** at Github_medium o60982/example0/interval233. Those measurements remain in the original raw maxima and tables. They are not discarded or replaced by faster repeat measurements.

The fixed replay panel was selected from the union of both controls’ original worst raw/stabilized positions, before rerunning. Six fresh processes per variant reconstructed the original prefixes; native replays also included the worst build schemas. Exact finite llguidance-key coverage, full JavaScript masks, and native trajectory outcomes were checked.

| Replayed location | Baseline range | Candidate range |
|---|---:|---:|
| JS example24/position22 | 2.754–2.945 ms | 2.683–2.765 ms |
| o60982 example0/interval233 | 5.8–7.1 µs | 5.8–7.7 µs |
| o65323 example0/interval417 | 21.7–22.8 µs | 21.2–24.4 µs |

The extreme primary spikes did not recur at those locations. Their underlying cause was not established, and the replay is not a hard upper bound. Selected-example replay timings must not be substituted for the whole-corpus distribution. The native dynamic replay also retains small mixed per-location changes rather than claiming every position improved.

One static-loaded artifact hash varied for o79409 even though every full mask and outcome matched. Four fresh processes per binary produced the same two serialized hashes in both baseline and candidate. Compiled artifact hashes matched for all126 schemas. This is not evidence that all saved bytes are deterministic, and no new serialization-format guarantee is claimed.

## Acceptance rationale

The fast path performs the same deterministic segment slice as the existing generic implementation, preserving the shared lower graph and accumulator. It adds no grammar restriction, approximate predicate, or persistent cache.
Dynamic JavaScript P90 improved in both independent Windows comparisons: 3.286 to 3.166 ms initially, and 3.293 to 3.167 ms with reversed process order. Same-binary disabled controls support attribution; earlier retained Mac comparisons also improved P90.
JavaScript stabilized P100 improved only slightly: 5.661 to 5.574 ms initially, and 5.666 to 5.656 ms in confirmation. This is not a major P100 breakthrough, and some intermediate tail quantiles remain worse.
All 8,332 llguidance-supported schemas were attempted in each native mode, with identical successes, failures, and 2,925,984 finite runtime interval keys. Dynamic native P100 decreased from 3.912 to 3.799 ms; static P100 decreased from 100.2 to 91.4 microseconds. Static build changes are small and mixed, not a compiler speedup claim.
The original 15.864 ms JavaScript and 75.6327 ms native-static spikes remain in the evidence. Neither recurred at its original coordinate in six fresh-process replays. Their causes are unproven. Confirmation raw JS maximum was still slightly higher for the candidate (6.913 versus 6.666 ms), as was the full native dynamic raw maximum; these observations are disclosed, not relabeled as fixed.
Thirty-two interleaved original-JS builds per variant did not reproduce the initial small build penalty: medians were 104.964 ms for baseline, 104.176 ms for the disabled control, and 103.203 ms for the default candidate. No build performance guarantee is made.
The Windows workspace library/integration suite passed 2,297 tests with zero failures and 55 ignored; all 42 Python tests and the documentation test passed. Four compiled/loaded static/dynamic streams each matched 77,582 complete masks across 126 schemas. Independent audits reconstructed the distributions, outcome coverage, and hashes from raw records. Loaded artifact byte variability occurred with the same two hashes in both binaries, with identical masks.

Residual limitations: Strict P99-to-P100 dominance over llguidance is not achieved in the Windows empirical curves. Raw wall-time maxima are not bounded by stabilized per-position minima; all original spikes remain in the published evidence. Dynamic JavaScript compilation remains roughly 100 ms versus single-digit-millisecond llguidance builds in these measurements. The unchanged optional Criterion benchmark expects 128,002 vocabulary entries and failed against the canonical 128,256-entry fixture; that failure is retained separately from the completed library/integration/doc/Python gates. Other inconclusive runtime experiments are excluded; the separate bitmap compiler candidate is not part of this release.

## Validation and limitations

The clean Windows workspace library/binary/integration suite completed with 2297 passed, 0 failed and 55 ignored. Python and documentation stage logs are hash-pinned in the frozen package metadata. Four compiled/loaded static/dynamic mask streams match the independent Windows baseline, with 77,582 full masks and outcome markers across 126 schemas per stream.

Optional all-target Criterion smoke was not accepted: unchanged bounded_string_array_close hardcodes128002vocabulary, while canonical fullCFA fixture contains128256. Complete library/binary/integration tests and docs are separate gates. Both failed setup attempts are retained locally.

The early seven keeper patches were reconciled by stable patch ID and already exist on main under different commit hashes. The guarded-cost policy and other inconclusive trials are not included in this release. Dynamic JavaScript build time remains a separate residual performance gap; this change does not claim to match llguidance compilation time.

All comparisons use one Windows machine and wall time. Raw measurements, failed attempts, immutable binaries, input hashes, scripts and exact local continuation notes remain in the private finalization directory. The public JSON contains the measured distributions, outlier coordinates, failure records and evidence fingerprints, not private compaction notes.
