# Single-pass composition metadata: bounded save-regression correction

This is a validated component checkpoint, not a universal timing certificate or a package release.

## Implementation

Three composition-metadata serialization calls now use one private helper that writes the same bincode wire format in one pass. The previous serializer first walked the value to determine its size, then walked it again to write. No mask/commit kernel, public API, decoder, artifact version, dependency, or cache policy was changed.

Four unit tests check one payload visit, exact link-metadata and compression output, borrowed cache metadata, and error propagation. They execute in the ordinary Rust suite.

## Validation

Validated native SHA-256: `97cfa9a87b9abde4da87313c24963327d1e447ea1dc599a4398029c1b6688334`. Parent native: `63da6b408a59f5ff2dd9d3a551adb8f4496eb4261855d34d95381c70eb297b07`.

The full Rust workspace passes 2,331 tests across 43 targets, including the four added tests, with 42 retained ignored tests. Nested subprocess summaries are excluded. The public doctest passes. The 573-test Python preflight is pinned to the same candidate binary and has no failures or skips; that previously completed preflight was verified, not counted as a second execution here.

The artifact guard compares 32 constraint/mode cases: fresh bytes, resaved bytes, complete masks and outcomes are identical to the parent. All 620 frozen source inputs match the tested source; only `src/runtime/persistence/composition.rs` differs from the parent.

## Measured static-save case

Target: `jsb/data/Kubernetes---kb_483_Normalized` (fixed corpus index 81, static mode).

| Comparison | Baseline median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| Same-path parent → candidate | 12.632 ms | 11.870 ms | -6.03% |
| Matched upstream 496c → candidate | 12.438 ms | 12.168 ms | -2.17% |

Each comparison contains eight predeclared schemas in four modes and six build pairs. These are separate paired populations, not ratios that should be subtracted. Case 81 has no accepted-token TBM population in this fixture; its save measurements must not be presented as successful decoding coverage.

## Retained timing qualifications

The one-pass helper grows its output Vec rather than preallocating the exact final length. Artifact bytes match, but allocation capacity can differ. Peak allocation/RSS was not measured by these timing screens.

The parent comparison has three local warm-TBM screening flags and the direct-upstream comparison has one. A screening flag is descriptive, not a statistical significance decision. All cold/warm/raw tails and scalar measurements remain available. The correction improves the targeted save case in the recorded comparisons; it is not evidence that every latency percentile on every grammar improved.

| Comparison | Schema index | Mode | Metric | p99 change | Absolute change |
| --- | ---: | --- | --- | ---: | ---: |
| performance | 14 | FAST_BUILD | loaded_warm_valid_accepted_tbm_ns | +11.45% | +3.396 µs |
| performance | 0 | O2 | loaded_warm_valid_accepted_tbm_ns | +10.65% | +3.048 µs |
| performance | 0 | FAST_BUILD | loaded_warm_valid_accepted_tbm_ns | +10.41% | +2.386 µs |
| upstream-comparison | 60 | FAST_BUILD | fresh_warm_valid_accepted_tbm_ns | +10.90% | +1.944 µs |

Broader decisive timing and platform checks belong to the final integration binary, including the two newer development commits through d5f4132e. This component is based on 2ba9a646 and does not itself include those commits. Earlier integration measurements and failed attempts remain preserved.

## Evidence

Recorded UTC: 2026-09-28T17:03:56.755597+00:00. On the existing CX33, evidence is under `/root/overhaul-release-readiness-789322-20260928/save-single-pass-2111`: `candidate/`, `baseline/`, `performance/`, `upstream-comparison/`, and `final-validation-e526c5ac/`. This note summarizes those exact sources; it does not duplicate their raw samples.
