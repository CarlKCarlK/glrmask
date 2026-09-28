# Bounded predecessor bitmaps: combined Windows release evidence

Code: `9173688dc6988070fd85046147e0c6969bd11c37`. Published baseline: `d5f4132eaaf52d2f6c06e13baa40cf9c08fcc8bb`.

## Change

Dense predecessor-frontier unions now use a bounded scratch bitmap instead of sorting a multiset containing repeated targets. Small or sparse frontiers keep the existing small-vector sort. The bitmap is limited to 65,536 states (8 KiB); results retain ascending exact IDs, dangling-target behavior and visit-budget accounting. No grammar, runtime cache policy or serialized format changes are introduced.

The already-published partial-segment-pop runtime optimization is retained. This is a compiler optimization tested against that published code, not against an older runtime baseline.

## JavaScript comparisons

Both sides compile the same original JavaScript grammar and complete 128,256-token vocabulary. Vocabulary preparation is outside both build timings. Every independently built artifact is replayed through all 31 examples, with fresh loaded constraints and full-mask/outcome checks. Reported stabilized TBM is the minimum complete mask-plus-commit pair across eight observations at each of the same 4,099 finite llguidance-supported positions. Raw maxima are separate. Both execution orders are retained; the second does not replace the first.

| Comparison | Mode | Variant | Median build (ms) | P90 TBM (µs) | P99 TBM (µs) | P100 TBM (µs) | Raw max (µs) |
|---|---|---|---:|---:|---:|---:|---:|
| screen | dynamic | published | 109.657 | 3135.460 | 4652.652 | 5536.100 | 8483.600 |
| screen | dynamic | candidate | 99.986 | 3138.760 | 4699.204 | 5475.900 | 5825.800 |
| screen | static | published | 3615.612 | 48.820 | 80.900 | 123.500 | 249.100 |
| screen | static | candidate | 3550.206 | 47.300 | 77.800 | 121.500 | 366.100 |
| confirmation | dynamic | published | 110.102 | 3140.560 | 4656.816 | 5534.600 | 9841.100 |
| confirmation | dynamic | candidate | 99.532 | 3129.280 | 4668.052 | 5537.900 | 107922.800 |
| confirmation | static | published | 3615.167 | 47.520 | 78.200 | 120.000 | 218.200 |
| confirmation | static | candidate | 3590.761 | 47.900 | 78.914 | 123.200 | 233.300 |

Repeated static builds can produce different serialized layouts even in the published baseline. All hashes are retained; every built artifact must produce the same complete masks and outcomes. The earlier baseline-only hash-identity assertion failure is preserved, not counted as a candidate correctness failure or silently forced to pass.

## Full llguidance-supported JSB population

The cohort is fixed solely from the retained llguidance-success reference: all 8,332 schemas are attempted. Existing GLRMask failures are retained and must match. Native TBM is commit_i through mask_i+1, distinct from the same-position JavaScript metric above. Each 250-schema chunk uses four alternating process runs, yielding two complete observations per binary per matching finite LL interval. No schema-specific warmup is added.

### Static

| Variant | Build P50 (ms) | Build P90 (ms) | Build P99 (ms) | Build P100 (ms) |
|---|---:|---:|---:|---:|
| published | 9.924 | 56.713 | 219.562 | 1177.680 |
| candidate | 9.908 | 57.659 | 219.528 | 1203.992 |

| Variant | TBM P50 (µs) | TBM P90 (µs) | TBM P99 (µs) | TBM P100 (µs) | Raw max (µs) |
|---|---:|---:|---:|---:|---:|---:|
| published | 5.000 | 9.100 | 17.300 | 93.900 | 2640.100 |
| candidate | 5.000 | 9.100 | 17.300 | 94.200 | 1788.700 |

published: 8,326 successful builds; 2,925,984 finite matching intervals. Retained failures: `jsb/data/Github_medium---o60170`, `jsb/data/JsonSchemaStore---meta.schema`, `jsb/data/Github_hard---o13029`, `jsb/data/Github_easy---o21053`, `jsb/data/Kubernetes---kb_1161_Normalized`, `jsb/data/Github_medium---o69991`.

candidate: 8,326 successful builds; 2,925,984 finite matching intervals. Retained failures: `jsb/data/Github_medium---o60170`, `jsb/data/JsonSchemaStore---meta.schema`, `jsb/data/Github_hard---o13029`, `jsb/data/Github_easy---o21053`, `jsb/data/Kubernetes---kb_1161_Normalized`, `jsb/data/Github_medium---o69991`.

### Dynamic

| Variant | Build P50 (ms) | Build P90 (ms) | Build P99 (ms) | Build P100 (ms) |
|---|---:|---:|---:|---:|
| published | 1.713 | 4.560 | 21.024 | 167.080 |
| candidate | 1.684 | 4.494 | 21.205 | 165.283 |

| Variant | TBM P50 (µs) | TBM P90 (µs) | TBM P99 (µs) | TBM P100 (µs) | Raw max (µs) |
|---|---:|---:|---:|---:|---:|---:|
| published | 6.400 | 66.600 | 180.800 | 4056.500 | 4309.800 |
| candidate | 6.400 | 66.400 | 179.400 | 4069.700 | 7186.800 |

published: 8,327 successful builds; 2,925,984 finite matching intervals. Retained failures: `jsb/data/Github_medium---o60170`, `jsb/data/JsonSchemaStore---meta.schema`, `jsb/data/Github_hard---o13029`, `jsb/data/Github_easy---o21053`, `jsb/data/Github_medium---o69991`.

candidate: 8,327 successful builds; 2,925,984 finite matching intervals. Retained failures: `jsb/data/Github_medium---o60170`, `jsb/data/JsonSchemaStore---meta.schema`, `jsb/data/Github_hard---o13029`, `jsb/data/Github_easy---o21053`, `jsb/data/Github_medium---o69991`.

## Raw-tail replay and limitations

The confirmation recorded a **107.923 ms** raw dynamic-JavaScript call at example 21, position 35, plus 37.791 ms at position 2 and 30.216 ms at position 1. These remain in the original summary and raw records. The separate replay uses six fresh processes per side and mode over the entire original 31-example corpus; its target panel is the union of both sides and both screens’ worst raw and stabilized positions, selected before replay.

| Replayed dynamic JS position | Published range (µs) | Candidate range (µs) |
|---|---:|---:|
| 21:1 | 1802.500–2194.200 | 1715.000–1840.700 |
| 21:2 | 1028.500–1151.500 | 1009.400–1059.400 |
| 21:35 | 3162.300–3473.400 | 3088.500–3229.500 |
| 23:44 | 5514.500–5780.600 | 5510.000–5719.400 |

Native replays retain the historical 14-schema bitmap outlier panel as well as both current controls’ worst observed build/runtime locations. Every original raw maximum remains in the report. A successful replay is not a latency bound or proof of an operating-system, allocator or scheduling cause.

## Acceptance decision

The same exact predecessor union is computed with a bounded scratch bitmap only when its frontier is dense. Ascending IDs, dangling-state handling, cached query semantics and visit-budget accounting are unchanged. No new grammar restriction or approximate runtime operation is introduced.

Dynamic JavaScript build improvement repeated on the integrated source: median 109.65715 to 99.98555 ms in the initial comparison and 110.10175 to 99.5316 ms with reversed process order, approximately 8.8% and 9.6%. Earlier isolated candidate comparisons showed the same direction; the release claim is based on the integrated comparison.

Both complete JavaScript comparisons preserve all 4,099 full masks and per-run outcomes across every independently built artifact. Runtime shifts are small and mixed, not a claimed new TBM speedup. Static JavaScript build medians were slightly lower in both integrated comparisons.

All 8,332 llguidance-supported schemas were attempted in both native modes. Both binaries have the same 8,326 static successes and six failures, 8,327 dynamic successes and five failures, and 2,925,984 identical finite measured interval keys per mode. Static P50/P90/P99 TBM remains 5.0/9.1/17.3 microseconds, with P100 93.9 to 94.2 microseconds; dynamic P100 4.0565 to 4.0697 ms. These small measured differences are retained.

Static JSB build P50 and P99 are essentially unchanged, while global P90 rose from 56.71305 to 57.6593 ms (1.67%). Paired per-schema median build ratios are 0.99764 overall and 0.99934 within the original baseline 85th to 95th-percentile band. This supports treating the change as broadly neutral on JSB builds, not claiming a strict non-regression guarantee or suppressing the measured P90 increase. Historical/current extreme-build replays were mixed, including a slower o21135 result.

The original 107.9228 ms JS call remains in the report. Across six new complete-corpus fresh-process replays, its position 21:35 took 3.0885 to 3.2295 ms with the candidate versus 3.1623 to 3.4734 ms with the baseline. The native dynamic 7.1868 ms raw outlier likewise did not recur at its coordinate. Neither event is assigned an unproven operating-system or allocator cause.

All combined correctness gates passed: 2,299 workspace tests, zero failures, 55 ignored, Python42 and documentation1; all four 77,582-mask compiled/loaded streams match. Independent scripts reconstructed both JS comparisons and every native distribution element from raw records, verifying source-derived llguidance support and all reported quantiles.


### Residual gaps

This is roughly a 9% improvement to dynamic JavaScript build time, not elimination of the build gap: around 100 ms remains much slower than the earlier single-digit-millisecond llguidance measurement. No fresh llguidance speed comparison was run for this compiler-only increment.

Strict upper-tail dominance over llguidance remains an unresolved target from the published runtime report. Stabilized minima are not worst-case guarantees. Original raw observations, including the 107.923 ms candidate call, remain visible and unchanged.

There are small mixed regressions as well as improvements: static JSB build P90 increased 1.67%, native stable maxima changed about 0.3%, and selected static build o21135 was about 6.2% slower in its six-process replay. No claim is made that every schema, percentile or raw call improves.

The fresh JS replay recorded a new static raw maximum of 617.6 microseconds in the candidate versus 281.2 microseconds in the baseline, while their stabilized maxima were 151.3 and 150.6 microseconds. The replay does not replace original full-distribution results and does not establish a hard raw-latency bound.

Static serialized bytes are not deterministic across repeated baseline builds. Every emitted artifact was instead checked through complete masks/outcomes; no deterministic-byte or new serialization-format guarantee is claimed.

The unchanged optional Criterion benchmark expects 128,002 vocabulary entries and remains incompatible with the canonical 128,256-token fixture. Its failure is retained separately from the successful full library/integration/Python/documentation gates. No assertion or vocabulary was weakened to obtain a pass.

Guarded-cost, static probe-once and older inconclusive runtime trials remain excluded. All seven earlier validated keepers were reconciled as already present on main; none is lost or reapplied.


## Correctness and provenance

The clean combined workspace library/binary/integration suite passed 2,299 tests with zero failures and 55 ignored. Python and documentation gates also completed successfully. All four static/dynamic compiled/loaded streams matched 77,582 complete masks and outcome markers across 126 schemas each.

The legacy optional Criterion smoke test’s 128,002-token fixture assumption remains a documented pre-existing incompatibility with the canonical 128,256-token vocabulary. No library assertion was weakened, and no vocabulary entries were dropped to hide that fixture issue.

All release-source, binary, input, audit and measurement fingerprints are recorded in the companion JSON. The retained raw records and frozen controls support exact replay. No fresh llguidance speed comparison was performed for this compiler-only increment; it does not establish that the remaining whole-tail or compilation-gap goals are solved.
