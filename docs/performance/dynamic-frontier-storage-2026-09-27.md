# Dynamic frontier storage: measured default optimization

Code: `67e47468671e5b8c7483e9fae1edf6c1803c8647`. Current-main integration: `4ed9ba3f78473cf0b19df38f67eb5593aabd6516`. Paired baseline: `fa2863ef21ae75df9169823a37a653a3d5848fa3`.

## Change

Cached frontier copies update the small state ID directly instead of creating a large temporary enum. Transition memo lookups return only that ID. Memo rows use one growable contiguous allocation instead of a separate allocation for every frontier. All cache limits, complete-state equality checks, parser/guard correlations, grammar semantics and fallback policies are unchanged. No row reference escapes a lookup, so vector growth does not invalidate state IDs.

This is a modest runtime improvement, not a new parsing algorithm. It is enabled by the normal production code path; no new environment variable is required.

## Paired JavaScript result

Both independent release binaries use the original grammar, normal result-cache behavior, the same 31 examples and all 4,099 llguidance-measured positions. Two processes per binary run three fresh-state passes each in interleaved order. TBM here is mask plus commit at the same position within a pass. The minimum complete pair over six passes is taken before computing quantiles; mask and commit minima are not spliced together.

| Metric | Current main | Storage optimization |
|---|---:|---:|
| TBM p50, µs | 695.083 | 660.000 |
| TBM p90, µs | 2155.767 | 2148.459 |
| TBM p99, µs | 3146.506 | 3059.602 |
| TBM p99.9, µs | 3559.784 | 3500.206 |
| TBM p100, µs | 3913.875 | 3826.208 |
| Raw TBM maximum, µs | 4667.917 | 4366.167 |
| Median native build, ms | 96.424 | 95.805 |

**Stabilized P100 is not a worst-call guarantee.** Raw maxima and every retained observation remain in the evidence. The change does not establish a universal speedup at every position.

## Build and static-runtime guardrails

Independent native runs cover the first 1,000 schemas in the pinned llguidance-success selection, not the entire JSB population. Both versions build every selected case and have identical outcomes. Runtime scores use exactly 367,640 matching llguidance intervals per engine. The native metric is commit through the next mask; it is different from the Python same-position convention above. Native quantiles use minima of two independent whole intervals.

| Engine / metric | Current main | Storage optimization |
|---|---:|---:|
| dynamic build p50, ms | 1.485 | 1.497 |
| dynamic build p90, ms | 4.169 | 4.231 |
| dynamic build p99, ms | 19.868 | 19.944 |
| dynamic TBM p90, µs | 66.833 | 67.250 |
| dynamic TBM p99, µs | 169.458 | 171.150 |
| dynamic TBM p100, µs | 2333.250 | 2474.792 |
| dynamic raw TBM maximum, µs | 2458.334 | 2550.083 |
| static build p50, ms | 9.421 | 9.351 |
| static build p90, ms | 72.417 | 70.791 |
| static build p99, ms | 226.470 | 222.203 |
| static TBM p90, µs | 6.000 | 5.959 |
| static TBM p99, µs | 12.916 | 12.875 |
| static TBM p100, µs | 42.792 | 44.958 |
| static raw TBM maximum, µs | 63.625 | 166.041 |

Static JavaScript P100 was 79.000 → 80.333 µs; build medians were 3364.052 → 3368.562 ms.

An earlier 15-case diagnostic panel showed an apparent 14.5% static build regression on `o30532`. It was not hidden: a separate fixed-schema experiment used 32 independent builds per binary, with published process medians 95.745/94.799 ms and candidate medians 94.532/94.376 ms. Matching compiler state counts and stage profiles did not show that regression in isolation. The original bad readings and raw build spread are retained. A separate 32-build JavaScript recheck was flat at 95.973 → 96.238 ms. No compiler algorithm improvement is claimed by this storage patch.

## Integrated tail rechecks

The initial native dynamic P100 increase (2333 → 2475 µs) was rechecked using four independent case replays per version, retaining the original observations. The two actual tail schemas then measured 2336.667 → 2308.458 µs (`o21074`) and 2294.709 → 2274.792 µs (`o21142`). Those repetitions did not reproduce the apparent regression.

The candidate static raw maximum of 166.041 µs came from `create_invoice_beb99d93`, example 0, position 67. Four targeted runs per version gave 11.834 → 11.791 µs at that same position. A separate 152.666 µs `typingsrc` observation rechecked at 12.333 → 13.334 µs. The stabilized static tail schema `sp_345_Normalized` measured 43.666 → 44.667 µs; that small residual difference is reported rather than presented as a speedup. These are case-specific rechecks, not replacements for the original raw maximum or a universal tail guarantee.

## Fresh llguidance comparison

This is a separate six-pass, six-build run, with all 4,099 positions measured by both engines. llguidance uses its different grammar; mask correctness is checked against the original GLRMask reference, not against llguidance mask equality.

| Metric | GLRMask dynamic | llguidance |
|---|---:|---:|
| TBM p50, µs | 691.875 | 1070.875 |
| TBM p90, µs | 2275.400 | 2739.150 |
| TBM p99, µs | 3215.476 | 3076.297 |
| TBM p100, µs | 4182.000 | 3449.959 |
| Raw maximum, µs | 4534.166 | 4282.875 |
| Median build, ms | 87.255 | 3.889 |

**The goal of beating llguidance across the entire JS distribution and build time remains unestablished.** Do not infer that a modest improvement here resolves the remaining comment-heavy and cold tails.

## Correctness and release gates

The integrated full-workspace run has 2,283 successful test results including child harnesses, zero failures and 55 ignored results. All 42 Python tests pass. Compiled and loaded static and dynamic artifacts each reproduce all 77,582 full masks and outcome markers across a 126-schema guard. The full JS corpus and JSB runtime guard also match every full mask.

An initial pre-integration workspace invocation accidentally retained the benchmark-only llguidance compatibility setting. Its unchanged overlapping-pattern fixture failed with that setting and passed using the identical test executable with normal test defaults. No source or test was weakened. Normal test settings are used for release gates; comparison benchmarks still use the documented llguidance-compatibility policy.

Experimental exit memos, redundant metadata gating and exact byte-column alphabet proofs were measured and rejected because they did not provide compelling tail gains. None is included in this release.

## Reproduction

The companion JSON records revisions, frozen package hashes, exact mask hashes, selected-case scope, raw maxima and retained artifact checksums. Private scripts and raw outputs live in `.benchmarks/dynamic-frontier-integration`; pre-integration rechecks are in `.benchmarks/dynamic-frontier-storage`. Release binaries are independently frozen, and timing uses the machine isolation gate plus interleaved runs. No cross-machine latency comparison is made.

## Final upstream integration

After the timed comparison, upstream `ac991889d` added packed boundary-cancellation results. It did not change the dynamic runtime code. The combined source at `10a790da477d725b44ba5616d2ac25a27ac79c6e` was rebuilt and passed the full workspace again (2,288 successful test results including child harnesses, zero failures, 55 ignored), all 42 Python tests, and all four 126-schema compiled/loaded full-mask checks. The dynamic runtime source hash is unchanged from the timed candidate. The tables above retain their stated earlier baseline/integration; no speedup from the other worker's change is attributed to this patch. The companion JSON records the final combined package and validation hashes.
