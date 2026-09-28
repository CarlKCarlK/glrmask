# Exact parser-row liveness: measured default optimization

Code: `78062aaf1785f8f47b1cb785107b7e94ddcd4076`. Independently built comparison baseline: `6377eb7ca20a082afefe455a10b5db6ab8005eef`.

## Exactness and scope

Before constructing all stack-admissible terminals, the dynamic walker can sometimes answer its boolean lexer-future query directly. An unconditional parser-row hit proves liveness; disjointness from the necessary advance row proves absence. Unresolved cases retain the existing exact stack simulation and singleton lookup. The proof declines scoped, sparse-direct, zero-width-control and multi-top cases for which these bounds are not established. Existing ignore handling, maximal-munch guards, parser/lexer correlations and token aliases are unchanged.

The optimization is enabled by default. `GLRMASK_DISABLE_ROW_LIVENESS=1` exists only for differential diagnosis. No grammar or compiler algorithm changed.

## Interleaved original-JavaScript result

Three controls were measured: an independently frozen published binary, the new binary with the proof disabled (`reference`), and its production default (`candidate`). Each uses two processes with three fresh-state passes per example, in forward/reverse order. All 31 examples and 4,099 full-mask keys match. Performance uses only matching finite llguidance keys. Quantiles are computed after taking the minimum complete mask-plus-commit pair at each position across six passes; independent mask and commit minima are never spliced.

| Metric | published | reference | candidate |
|---|---:|---:|---:|
| TBM p50, microseconds | 770.417 | 768.667 | 690.458 |
| TBM p90, microseconds | 2537.708 | 2524.650 | 2286.758 |
| TBM p99, microseconds | 3619.558 | 3666.240 | 3429.035 |
| TBM p99.9, microseconds | 4121.600 | 4181.357 | 3865.763 |
| TBM p100, microseconds | 4493.708 | 4473.750 | 4425.875 |
| Raw maximum, microseconds | 5341.042 | 5282.042 | 4723.750 |
| Median native build, milliseconds | 105.072 | 99.422 | 100.483 |

The strongest effect is in the middle and upper distribution, not a large P100 reduction. Raw maxima remain separate: stabilized P100 is not a worst-call guarantee. System storage-management activity was present during these runs, so the same-batch interleaved controls—not older absolute timing samples—are the comparison.

## Static and JSB guardrails

The 32-schema dynamic JSB guard compares all 9,045 full masks and scores exactly 9,021 llguidance-measured positions. The static-JavaScript guard checks all 4,099 positions.

### Dynamic JSB guard

| Metric | published | reference | candidate |
|---|---:|---:|---:|
| TBM p50, microseconds | 45.541 | 46.333 | 45.458 |
| TBM p90, microseconds | 136.458 | 137.791 | 135.542 |
| TBM p99, microseconds | 397.709 | 403.117 | 399.141 |
| TBM p99.9, microseconds | 2633.254 | 2645.988 | 2631.860 |
| TBM p100, microseconds | 2671.249 | 2695.291 | 2678.833 |
| Raw maximum, microseconds | 2838.125 | 2843.000 | 2800.209 |

### Static JavaScript guard

| Metric | published | reference | candidate |
|---|---:|---:|---:|
| TBM p50, microseconds | 22.917 | 22.875 | 22.834 |
| TBM p90, microseconds | 34.500 | 34.958 | 34.709 |
| TBM p99, microseconds | 53.422 | 54.087 | 54.380 |
| TBM p99.9, microseconds | 84.954 | 82.706 | 84.787 |
| TBM p100, microseconds | 94.083 | 92.125 | 93.875 |
| Raw maximum, microseconds | 130.666 | 139.458 | 129.791 |
| Median native build, milliseconds | 3788.328 | 3756.098 | 3776.690 |

## Native 1,000-schema comparisons

This is the first 1,000 cases in the pinned llguidance-success corpus, not the entire population and not a random sample. Both engines build every selected schema, and all recorded semantic outcomes agree. Each comparison scores exactly 367,640 finite matching llguidance intervals. Native TBM is commit through the next mask, unlike the Python same-position convention above; these figures must not be mixed. Native quantiles use minima of two independent complete observations. The companion JSON retains corpus selection, frozen binary hashes and worst positions.

### Dynamic

| Metric | published | candidate |
|---|---:|---:|
| Build p50, milliseconds | 1.662 | 1.664 |
| Build p90, milliseconds | 4.707 | 4.666 |
| Build p99, milliseconds | 22.731 | 22.749 |
| Build p100, milliseconds | 51.845 | 52.354 |
| TBM p50, microseconds | 7.000 | 6.958 |
| TBM p90, microseconds | 78.333 | 77.000 |
| TBM p99, microseconds | 198.868 | 196.583 |
| TBM p99.9, microseconds | 589.681 | 591.474 |
| TBM p100, microseconds | 2769.000 | 2765.750 |
| Raw maximum, microseconds | 2780.083 | 2769.458 |

### Static

| Metric | published | candidate |
|---|---:|---:|
| Build p50, milliseconds | 10.447 | 10.398 |
| Build p90, milliseconds | 86.019 | 87.785 |
| Build p99, milliseconds | 247.815 | 250.328 |
| Build p100, milliseconds | 676.556 | 683.462 |
| TBM p50, microseconds | 4.083 | 4.083 |
| TBM p90, microseconds | 7.000 | 7.000 |
| TBM p99, microseconds | 14.750 | 14.875 |
| TBM p99.9, microseconds | 31.155 | 30.890 |
| TBM p100, microseconds | 48.917 | 51.000 |
| Raw maximum, microseconds | 195.833 | 160.750 |

### Rechecked static outliers

The apparent roughly 2-microsecond stabilized static-tail increase was checked at the actual worst position, alongside both versions' raw outliers. Four independent replays per version used the same frozen binaries, case order and ordinary cache policy. The original observations are retained above and in the companion JSON; these targeted repeats neither replace the raw maxima nor prove their cause.

| Exact native interval | Published minimum, microseconds | Candidate minimum, microseconds |
|---|---:|---:|
| `jsb/data/JsonSchemaStore---typingsrc`, example 0, interval 1 | 11.250 | 11.542 |
| `jsb/data/JsonSchemaStore---typingsrc`, example 0, interval 339 | 15.083 | 15.042 |
| `jsb/data/Snowplow---sp_345_Normalized`, example 0, interval 16 | 51.083 | 49.167 |

## Fresh llguidance comparison

This is a separate same-machine `make example-js` run with six builds, six measured passes, no warmup and production cache settings. The complete raw-timing sidecar is retained and audited. llguidance uses a different grammar: cross-engine masks are not assumed equal. GLRMask correctness is instead checked against its unchanged language reference.

| Metric | glrmask_dynamic | llguidance_native |
|---|---:|---:|
| TBM p50, microseconds | 748.792 | 1269.792 |
| TBM p90, microseconds | 2432.225 | 3245.050 |
| TBM p99, microseconds | 3540.803 | 3597.040 |
| TBM p99.9, microseconds | 4000.205 | 3857.908 |
| TBM p100, microseconds | 4432.791 | 4198.416 |
| Raw maximum, microseconds | 5734.667 | 4906.624 |
| Median native build, milliseconds | 99.056 | 4.751 |

The goal of beating llguidance at every tail percentile and in build time is not established by this change. Use the fresh comparison above rather than pairing an earlier, faster GLRMask screen with unrelated llguidance measurements.

## Correctness and release evidence

The pre-integration workspace gate reports 2,281 passed, zero failed and 55 ignored, counting the final summary of each Cargo harness rather than double-counting nested child output. All 42 Python tests pass. A separate documentation test passes. Tests exhaustively cover the lower/upper-bound predicate on small sets and compare real recursive/ambiguous parser frontiers in both compiled and loaded forms, including ignored terminals and cache hits.

Four independent static/dynamic compiled/loaded capture streams each preserve all 77,582 full masks and outcome markers across 126 schemas. Streaming verification compares every decoded byte without retaining gigabytes of uncompressed masks. The companion JSON records all four digests and decoded sizes. These are correctness checks, not additional performance samples.

Private reproduction scripts, immutable binaries, raw timings and manifests are retained in `.benchmarks/dynamic-row-liveness/final`. Existing unrelated optimization experiments are excluded. No Windows or new paid server was used.

## Latest-main integration

After the original paired benchmarks, upstream `b0a01e7473b5666a050e75913082d12532ac8c7f` added independently published finite-boundary row parallelism. It was merged as `702cad39994a17b24eba6ef5ca0cc18fa010cc6f` without changing the measured runtime source (SHA256 `f8e7589cee8d4859a81042f77765386349b7a5288eddb4ef1a46afddf5913f21`). The combined source was rebuilt: 2285 workspace tests passed, zero failed and 55 were ignored; all 42 Python tests and the documentation test passed. All four compiled/loaded static/dynamic streams again matched all 77,582 masks and outcome markers. A new three-way six-pass JS/JSB/static screen checked the combined package against the independent published baseline and its own diagnostic-off control. The tables above retain their explicitly stated earlier baseline and timings; the complete post-integration screen is retained in the companion JSON, rather than attributing the other worker's changes to row liveness.
