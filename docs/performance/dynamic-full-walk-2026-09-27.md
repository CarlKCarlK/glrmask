# Exact dynamic full-walk acceleration — 27 September 2026

Validated source: `a14d990ff6fb3d15ff4c2ec3573181696d8c3559`. Reference: published `0268e0163`.
The accelerator is **enabled by default**, including for existing loaded
constraints. No grammar, accepted-language policy, compiler algorithm, or
serialized format is changed.

## What is retained

The dynamic walker reuses exact correlated parser/lexer frontiers within one
mask. A bounded memo stores up to 1,024 transition rows, using full equality
after hashing. Parser nodes are append-only. Pending maximal-munch memories
remain in their original correlated branches; a hash or shared parser top is
never an equivalence proof. Storage limits fall back to the original executor.

A vocabulary subtree can be skipped only when a complete-state identity
alphabet covers every byte in it, or when an enclosing root union proves that
all output token aliases are irrelevant there. Deferred output accounting is
preserved; known-polarity no-op writes are omitted. Joint root execution keeps
the existing transparent-root specialization. Complete parser-language sharing
uses bounded exact keys, not approximate state merging.

Ambiguous stack depths are allocated lazily instead of reserving a large fixed
frontier array on every scalar invocation. Identity-proof eligibility is
updated at append-only parser-node creation, not recomputed from lengths for
every byte. Scalar identity bookkeeping remains in the scalar lane. Rejected
byte-class, exit-memo, parser-subsumption, branch-order, and delayed-admission
experiments are not part of this release.

## Controlled JavaScript runtime comparison

The unchanged 31-example corpus contains 4,099 positions measured by
llguidance. Actual published and final candidate packages are frozen and run
in ABBA order, with three fresh loaded states per example in each of two
processes per version. Every full vocabulary mask matches. This isolation
probe disables the existing exact dynamic mask cache; it does **not** disable
the new per-walk accelerators. The statistic is the minimum complete
mask-plus-same-position-commit pair at each position over six equal repetitions.

| Dynamic JS TBM, microseconds | Published | Candidate |
| --- | ---: | ---: |
| P50 | 4942.209 | 2005.875 |
| P90 | 7214.900 | 2672.325 |
| P99 | 11946.491 | 3530.274 |
| P99.9 | 16066.329 | 3932.439 |
| P100 | 16696.459 | 4562.958 |
| Raw maximum | 18346.958 | 5238.209 |

Stabilized P100 is 72.7% lower. This is not a worst-case guarantee and
does not replace the production-cache comparison below.

| Static JS guard, microseconds | Published | Candidate |
| --- | ---: | ---: |
| P90 | 29.708 | 29.508 |
| P99 | 47.212 | 47.175 |
| P100 | 79.833 | 79.000 |
| Raw maximum | 108.542 | 101.375 |

## Compilation guard

Compilation is measured separately: 24 dynamic and six static builds per
version in independent ABBA batches, vocabulary preparation excluded. No
compile-time improvement is attributed to these runtime changes.

| Median JS compilation, ms | Published | Candidate |
| --- | ---: | ---: |
| Dynamic | 92.445 | 91.839 |
| Static | 3561.201 | 3586.595 |

## Native llguidance-supported JSB guards

The fixed first 1,000 llguidance-success schemas are evaluated in 250-schema
chunks with alternating ABBA/BAAB order. All 1,000 build outcomes and all
trajectory outcomes agree. Runtime scoring is restricted to the same 367,640
finite llguidance-measured intervals. Failures and coverage are checked, not
silently discarded. The native metric is commit through the **next** mask,
with the minimum of two independent process measurements at each position.
Production cache defaults are retained. These are regression panels, not the
full JSB population.

| Initial static JSB build, ms | Published | Candidate |
| --- | ---: | ---: |
| P50 | 9.454 | 9.491 |
| P90 | 73.876 | 79.047 |
| P99 | 227.996 | 226.806 |
| P100 | 640.704 | 631.239 |

The initial static P90 increase is retained. A deliberately selected 41-schema
panel covers the worst paired build ratios and P90 neighbors, with six runs
per version. Its median per-schema build ratio is 0.995685.
Some individual medians still differ, and the full raw arrays are retained.
An independent repeat of the complete 1,000-schema panel gives:

| Confirmatory static JSB build, ms | Published | Candidate |
| --- | ---: | ---: |
| P50 | 9.512 | 9.475 |
| P90 | 79.271 | 73.044 |
| P99 | 231.990 | 232.156 |
| P100 | 631.824 | 625.459 |

The P90 movement reverses in this independent panel, while median paired build
ratios remain near one. Neither a 7% static build gain nor a 7% static build
regression is established by these observations.

| Confirmatory static JSB TBM, microseconds | Published | Candidate |
| --- | ---: | ---: |
| P50 | 3.500 | 3.500 |
| P90 | 6.000 | 6.042 |
| P99 | 13.042 | 12.917 |
| P100 | 43.000 | 44.917 |

The dynamic JSB guard retains a middle-quantile cost; it is not presented as
an across-the-board improvement:

| Dynamic JSB TBM, microseconds | Published | Candidate |
| --- | ---: | ---: |
| P50 | 6.042 | 6.083 |
| P90 | 63.792 | 68.041 |
| P95 | 80.250 | 89.334 |
| P99 | 162.958 | 172.609 |
| P100 | 2325.375 | 2367.875 |

## Fresh production comparison with llguidance

A separate `make example-js` run uses the unchanged grammar, production cache
defaults, six equal timing runs per engine, zero warmups, and six build runs.
The main JSON and its raw-timing sidecar are both retained and hashed. All
24,594 raw complete pairs per engine are checked; exactly 4,099 positions have
finite measurements on both sides. llguidance version is
`1.6.1`. Its grammar differs from GLRMask's;
cross-engine masks are not assumed equal.

| Production JS TBM, microseconds | GLRMask dynamic | llguidance |
| --- | ---: | ---: |
| P50 | 780.750 | 1071.168 |
| P90 | 2720.742 | 2739.075 |
| P95 | 3021.892 | 2852.462 |
| P99 | 3762.143 | 3064.477 |
| P99.9 | 4859.309 | 3265.930 |
| P100 | 5119.584 | 3478.041 |
| Raw maximum | 7980.292 | 5511.125 |

Median build is 98.076 ms for GLRMask and
4.085 ms for llguidance. **Whole-distribution and P100
parity with llguidance are not achieved.** The residual production tail is
primarily mask computation on deeply nested control-flow examples, not the
short commit calls at those positions. This remains ongoing work.

## Correctness and reproducibility

The clean committed source passed 2,267 Rust unit/integration tests, zero
failures and 55 ignored; documentation tests and example checks passed.
All 42 Python tests passed. A separate process with the exact-reference
diagnostic enabled passed all 78 dynamic-mask tests. Existing example and
pytest warnings were not suppressed.

The 126-schema correctness guard independently checks 77,582 full vocabulary
masks and outcome markers in each of four modes: static compiled, static
reloaded, dynamic compiled, and dynamic reloaded. Every decoded-stream hash
matches the previously published reference, including invalid continuations
and the prior large-tokenizer serialization regressions. Compressed streams
are retained; gigabytes of expanded masks are not written to disk.

`GLRMASK_DISABLE_FULL_WALK_ACCELERATION=1` selects the literal reference for
diagnostics and must be set before the first walk. No opt-in is required for
the retained behavior. The same-binary disabled comparison was insufficient
to expose a common-case code-layout cost, so the final report uses separately
built actual published binaries as controls. Earlier Q/R/S/T experiments and
their raw observations remain in the private worktree rather than being
relabelled as this final result.

The accompanying JSON records source/binary/input hashes, exact LL selection,
all build repetitions, both native panels, the selected recheck, raw extrema,
protocol differences, limitations, and an evidence manifest. Private scripts
and frozen binaries are under `.benchmarks/dynamic-release`. A fresh public
CFA run uses `make example-js FRAMEWORKS='glrmask_dynamic llguidance_native'
TIMING_RUNS=6 TIMING_MIN_RUNS=6 TIMING_WARMUP_RUNS=0 BUILD_RUNS=6
RECORD_TIMING_RUNS=always RUNTIME_TIMING_CLOCK=thread`; select the intended
GLRMask package with the caller's `PYTHONPATH` and retain the `_raw` sidecar.

## Post-upstream integration check

The runtime release was merged normally with upstream `51f44dce3`, preserving
its certified parser-read-context changes. Exact combined source `e07744d3a`
passed 2,269 Rust workspace tests with zero failures and 55 ignored, the
documentation/example gates, and all 42 Python tests. All four 126-schema
compiled/loaded captures again match the pinned 77,582-mask streams.

An independent premerge-versus-postmerge ABBA screen used the same corpus,
three fresh loaded traversals per example per process, and unchanged runtime
source files. The table uses the same cache-disabled walker-isolation protocol
as the controlled comparison above, not a fresh llguidance comparison.

| Metric | Premerge | Postmerge |
| --- | ---: | ---: |
| Dynamic JS P90, us | 2748.742 | 2697.892 |
| Dynamic JS P100, us | 4400.750 | 4181.917 |
| Static JS P90, us | 29.675 | 31.083 |
| Static JS P100, us | 80.541 | 80.458 |
| Dynamic JSB guard P90, us | 111.917 | 112.667 |
| Dynamic JSB guard P100, us | 2281.874 | 2301.833 |

Every full mask agrees: 4,099 JS keys in each mode and 9,045 JSB keys, of
which 9,021 have matching llguidance timings. A raw 5,127.583 us JSB
observation at `o21135`, example 1, step 933 is retained. The other five
candidate repetitions at that position were about 2.28–2.47 ms. A separate
four-process ABBA recheck of both full examples preserved all 3,240 masks;
whole-case raw maxima were 2.48–2.52 ms on both versions. The spike did not
recur, but its cause was not established and the original measurement is not
deleted. This is not a claim of a hard latency bound.

`dynamic-full-walk-postmerge-2026-09-27.json` records this additional gate,
all capture hashes, test totals, paired observations, and the retained spike.
The remaining production JS P100 gap and dynamic-JSB middle overhead
documented above are still open.
