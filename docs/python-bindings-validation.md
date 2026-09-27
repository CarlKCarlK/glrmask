# Python binding overhaul: checkpoint review

Recorded UTC: 2026-09-27T18:00:21.313123+00:00

**Correctness-validated feature checkpoint; not a main release or universal performance certificate.**

## Structural changes

The former 1,855-line binding root is a small registration module. Private modules separate runtime ownership, vocabulary adapters, checked conversions, profiling, compiler helpers, internal registration, and allocator policy. Public lifecycle methods and private vocabulary partitioning are retained. The native masking and commit algorithms are unchanged by this binding batch.

One fallible NumPy write guard now rejects read-only and misaligned arrays before native access. One packed-buffer conversion checks the minimum output extent for public, private profiled, dynamic/O2, and partition-expansion adapters. This replaces repeated unsafe conversions without introducing a copying fallback or releasing the write guard before computation.

Independent baseline tests reproduced 18 read-only/undersized-buffer panics. The guarded build passes those same cases with normal Python errors and unchanged state/input. Alignment was established as a source-level precondition gap: misaligned buffers were never executed against an unprotected predecessor.

## Validated identity

Native SHA-256: `f9fcad3303248a9f205b350ca0b105d3ad1ab8e7ffa0c0c2d833a79b22e66ba3`.

Baseline native: `48990bdee0ca28fde726be0627908356a5dc6ecd8a72a9962f9c0e58040f90fd`.

There are 551 frozen non-Markdown source inputs, including 517 core non-Python inputs byte-identical to the previously validated artifact-module checkpoint. All-feature checking and final wheel compilation are warning-free.

The complete pinned Python suite passes **289 tests**: 167 retained, 38 primary boundary cases, 48 private-adapter cases, 33 alignment cases, and three wide partition-coordinate cases. The byte-identical core suite was rerun: **2,202 Rust tests across 43 targets**, 41 existing ignored tests, and one passing doctest. Nine nested subprocess summaries are excluded from the total.

Exact API reflection, 38 full-vocabulary constraint records, 24 private partition records over 128,256 tokens, 16 current-format artifacts in each direction, and 176 independent recursive-language traces pass on this exact binary.

## Native performance measurements

The selected population is the same ten schemas in four modes, with four paired builds and seven replays. TBM is aligned mask plus commit time at accepted-token positions. The table reports the existing stabilized per-position distributions, not unfiltered p100. Raw cold/warm samples and preemptions remain in the evidence. Identical-binary controls are reported separately, never subtracted to erase a warning.

| Mode | Fresh p99 change | Reloaded p99 change | Build p99 change | Identical control fresh / loaded p99 |
| --- | ---: | ---: | ---: | ---: |
| FAST_BUILD | -4.32% | +2.58% | +12.19% | +1.10% / -1.27% |
| O2 | +11.07% | +2.02% | -2.96% | -9.80% / -4.28% |
| FAST_RUNTIME | -14.65% | +0.79% | +3.50% | -2.83% / +3.14% |
| AUTO | -10.59% | -0.37% | +0.28% | -6.05% / -9.81% |

The actual run retains **9 local warm-TBM screen warnings**; the identical control has **2**. These thresholds are not statistical significance tests. All unfavorable scalar, cold-start and raw-tail results remain available.

| Schema index | Mode | Metric | p99 change | Absolute change (microseconds) |
| --- | --- | --- | ---: | ---: |
| 82 | O2 | fresh_warm_valid_accepted_tbm_ns | +11.21% | +4.111 |
| 34 | FAST_BUILD | loaded_warm_valid_accepted_tbm_ns | +14.83% | +4.022 |
| 75 | O2 | fresh_warm_valid_accepted_tbm_ns | +14.32% | +3.597 |
| 7 | FAST_BUILD | fresh_warm_valid_accepted_tbm_ns | +16.83% | +3.314 |
| 14 | FAST_BUILD | fresh_warm_valid_accepted_tbm_ns | +11.11% | +3.244 |
| 7 | FAST_BUILD | loaded_warm_valid_accepted_tbm_ns | +11.60% | +2.917 |
| 57 | O2 | fresh_warm_valid_accepted_tbm_ns | +11.47% | +2.570 |
| 76 | FAST_BUILD | fresh_warm_valid_accepted_tbm_ns | +13.01% | +2.279 |
| 37 | O2 | loaded_warm_valid_accepted_tbm_ns | +10.64% | +2.031 |

## Python-call overhead

A separate benchmark includes Python argument extraction, cached-fill calls, and complete small decoding loops. Two actual process pairs bracket an identical-binary control with counterbalanced call order. Values below are medians of loop means, **not token p99/p100**. The fixture is deliberately small and does not establish broad vocabulary or grammar performance.

| Mode / storage / operation | Actual 1 / actual 2 changes | Identical control | Absolute actual deltas (ns) |
| --- | ---: | ---: | ---: |
| AUTO / fresh / cached-fill | +2.49% / +18.78% | -6.36% | +11.0 / +81.3 |
| AUTO / fresh / decoding-loop | -4.90% / -3.88% | -2.67% | -232.1 / -190.7 |
| AUTO / reloaded / cached-fill | -3.44% / -5.49% | +1.16% | -13.9 / -21.5 |
| AUTO / reloaded / decoding-loop | -3.25% / +1.28% | -10.55% | -193.9 / +60.4 |
| FAST_BUILD / fresh / cached-fill | -0.99% / -6.49% | -1.85% | -3.7 / -25.0 |
| FAST_BUILD / fresh / decoding-loop | -1.59% / -4.66% | +9.10% | -75.8 / -229.6 |
| FAST_BUILD / reloaded / cached-fill | +0.69% / -3.23% | +5.63% | +2.5 / -11.1 |
| FAST_BUILD / reloaded / decoding-loop | +2.02% / -0.51% | +2.16% | +102.1 / -26.5 |
| FAST_RUNTIME / fresh / cached-fill | -0.17% / -1.13% | -1.95% | -0.7 / -4.2 |
| FAST_RUNTIME / fresh / decoding-loop | -3.84% / +1.46% | -3.76% | -182.0 / +66.2 |
| FAST_RUNTIME / reloaded / cached-fill | -1.34% / -5.52% | +1.66% | -5.2 / -22.0 |
| FAST_RUNTIME / reloaded / decoding-loop | +1.25% / -0.44% | +3.31% | +57.5 / -21.1 |
| O2 / fresh / cached-fill | +0.01% / +3.83% | -5.22% | +0.0 / +13.2 |
| O2 / fresh / decoding-loop | -6.50% / +12.44% | -2.78% | -358.3 / +721.9 |
| O2 / reloaded / cached-fill | +4.87% / -0.17% | +1.24% | +16.9 / -0.6 |
| O2 / reloaded / decoding-loop | -3.26% / +8.27% | +2.92% | -172.2 / +438.1 |

## JavaScript and remaining limits

The supplemental JavaScript comparison audited all four modes, with 2 timing-screen warnings. It is not canonical `make example-js` tokenization. The same long-sample grammar restrictions remain in both builds. Only 532 accepted-token positions contribute to this fixture; input length is not successful coverage.

The full intended seed-7 CFA population and a llguidance comparison were not executed for this binding batch. No universal no-regression or raw-p100 claim is made. Earlier compiler/artifact checkpoint warnings remain separately documented.

## Failed attempts and packaging safeguards

The initial build-only overlay symlink omitted `__init__.py` from a wheel. Python fell back to an older installed package. All resulting failures and apparent passes are retained but are not attributed to the candidate binary. The overlay now materializes the two unchanged Python source files; the same pytest process verifies facade path, native path and SHA-256 before running. The later alignment test collection-import error is also preserved with its corrected rerun. No assertions were weakened to obtain passing counts.

## Durable evidence

Evidence is under `/root/python-bindings-cleanup-789322` on the existing CX33: `validation-alignment-c87811ba/`, `acceptance-aligned-c87811ba/`, `python-walltime-aligned-c87811ba/`, `independent-boundary-review-r0053/`, `lineage-c87811ba/`, and `frozen-source.json`. The final commit and Mac mirror receipts are recorded separately.

The earlier compiler correction and artifact-module split are preserved as explicit inherited commits, without changing their owners’ branches or indexes. This binding checkpoint does not merge or push main.

## Longer repeated actual comparison

A second actual comparison retained the same ten schemas and all four modes, with six build pairs and nine replays. It does not replace the first measurements or the separate identical-binary control.

| Mode | Repeated fresh p99 change | Repeated reloaded p99 change | Repeated build p99 change |
| --- | ---: | ---: | ---: |
| FAST_BUILD | +4.06% | +1.08% | -10.80% |
| O2 | +0.83% | -2.52% | -0.42% |
| FAST_RUNTIME | -1.81% | -2.54% | +0.26% |
| AUTO | -5.46% | +0.27% | +2.74% |

The repeat has 3 local warm-TBM warnings; 1 repeat at the same schema, mode, and metric as the first run. Scalar and cold-start warnings need their own interpretation; a warning disappearing does not prove universal neutrality.

Repeated warning: jsb/data/Glaiveai2K---search_recipe_8096f36e, FAST_BUILD, fresh_warm_valid_accepted_tbm_ns; first +11.11% (+3.244 microseconds), repeat +10.90% (+2.381 microseconds).

### Build medians and load tails are also retained

| Mode | Build p50 change: first / repeat / control | Load p99 change: first / repeat / control |
| --- | ---: | ---: |
| FAST_BUILD | -5.56% / -5.29% / +7.24% | -2.92% / +8.15% / -8.98% |
| O2 | +15.07% / +16.04% / +17.00% | +9.28% / -10.80% / +35.84% |
| FAST_RUNTIME | +1.43% / +6.51% / -8.05% | +9.52% / +10.29% / -1.08% |
| AUTO | -1.38% / -8.45% / -3.01% | -2.09% / +5.04% / -0.21% |

In particular, O2 build medians rise in both actual runs, but also in the identical control. Static load p99 is higher in both actual runs. These observations remain unresolved qualifications, not evidence of neutrality. The supplemental JavaScript fresh dynamic p99 and loaded AUTO p99 warnings were not repeated in this follow-up, which covers schemas only.
