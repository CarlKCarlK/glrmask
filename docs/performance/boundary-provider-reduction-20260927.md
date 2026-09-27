# Exact provider reductions: release validation

Code revision: `f2ab047142c4f350c22cfe016c08468426f67647`. All figures below are microseconds of thread CPU.

## Retained change

Bounded deterministic reduction sequences use the existing virtual stack and materialize only the resulting frontier. Completed but unconsumed prefixes resume in the reduction queue; they are never mistaken for shifted output. One 64-action budget bounds each enclosing advance. The complete input must carry the certified uniform empty accumulator label; uncertain representations keep the reference traversal.

Both reduction fusion and prefix resumption are enabled by default. No environment variable is needed to obtain them. The diagnostic overrides `GLRMASK_PROVIDER_REDUCTION_PREFIX=0` and `GLRMASK_PROVIDER_RESUME_REDUCTIONS=0` retain exact comparisons.

The shared predecessor helper also excludes an epsilon alternative before applying the sole visible predecessor's goto. Append/replace and mixed-label regressions cover the actual correctness issue. Reference performance runs include this repair, so they do not compare against a known incorrect baseline.

No packed-vocabulary, shallow-frame, first-witness, token-equivalence, or alternate byte-walker experiment is included in this release.

## Gates

Workspace library tests: **2007 passed, 0 failed, 55 ignored**. Selected public API/integration tests: **170 passed, 0 failed**. Release no-internal-API and all-target checks passed.

Six independent finite/overlapping-language oracle runs passed. Complete selected-trace masks, acceptance and commits matched for all replay variants, including normal result-cache behavior. The companion JSON contains the gate counts, exact policy controls, binary hashes, raw maxima and position-matched measurements.

## Tail measurements

D measures the dynamic boundary handler, B all boundary dispatch, and N ordinary work from the same invocation. Independent min-of-two values stabilize each position; the maximum of those position values is reported as p100. These are not worst complete runs or latency guarantees. Raw run maxima remain in the JSON. Different percentile maxima must not be subtracted.

| Domain | Cohort | Reference D p100 | A/A control D p100 | Default D p100 | Default B p100 |
|---|---|---:|---:|---:|---:|
| 257 | A | 991.860 | 945.236 | 727.561 | 737.551 |
| 257 | B | 1049.918 | 1236.185 | 842.929 | 853.762 |
| 1274 | A | 2826.215 | 2353.128 | 1438.187 | 1444.355 |
| 1274 | B | 1534.393 | 2250.311 | 1490.944 | 1506.127 |

### Ordinary-work comparison at the actual boundary tail

| Domain | Cohort | Worst B position | B | N in that same call |
|---|---|---|---:|---:|
| 257 | A | 0/10 | 737.551 | 155.976 |
| 257 | B | 0/10 | 853.762 | 164.635 |
| 1274 | A | 2/4194 | 1444.355 | 122.368 |
| 1274 | B | 1/2099 | 1506.127 | 335.343 |

**The boundary-at-or-below-ordinary-work goal is not established by this release.** These changes are incremental validated keepers, not a claim that the remaining cold and warm tail is solved.

Ordinary 128k-vocabulary controls were checked independently for exact masks. Shared-host timing variability and different worst positions preclude claiming a universal ordinary-mask speedup from these measurements.

See `../provider-reduction-prefix.md` for fallback, labels, budget and predecessor-isolation contracts.

## Normal result-cache check

These use the normal cache policy, with one independent min-of-two cohort per domain. They supplement the cache-disabled stress comparison; they are not worst-case latency guarantees.

| Domain | Reference D p100 | Default D p100 | Default B p100 |
|---|---:|---:|---:|
| 257 | 1239.233 | 620.643 | 637.582 |
| 1274 | 2480.605 | 1879.078 | 1892.356 |

The larger configuration contains a 5-token parent boundary and a 1,274-token child boundary. The 257-token hybrid retains the parent boundary statically. These are two configurations, not interchangeable samples of one vocabulary.

