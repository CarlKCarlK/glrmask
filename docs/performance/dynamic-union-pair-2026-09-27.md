# Exact dynamic lexer-union reuse — 27 September 2026

Code: `48c5823fae9df52d3111cea2ea300c523d96782e`. Combined source base: `bcf19928bc1806e4b12a9b225a905ce0f0e87164`.

## Change and invariant

The dynamic lexer already assigns canonical IDs to exact physical-state sets.
Discovering that a union existed still expanded and sorted its inputs on every
request, including repeated virtual-subset pairs. This change memoizes the
complete, sorted pair of input coordinates before that redundant work.

The key packs both complete 32-bit coordinates into a 64-bit integer. It is not
a lossy fingerprint. Results are inserted only after the authoritative union
implementation succeeds. The index is bounded to 2,048 entries; capacity or an
unsupported input falls back to the original implementation. Every reset of
the extension-ID namespace clears this index. The cache is runtime-only and is
not serialized. Grammar, parser relations, union results, and existing parser
state limits are unchanged.

The optimization is on by default with the published full-walk accelerator.
`GLRMASK_DISABLE_LAZY_UNION_PAIR_MEMO=1` selects the diagnostic reference. No
opt-in variable is required. The separate finalizer memo was rejected and is
not included in this change.

## Three-way production-cache check

Each engine ran three fresh loaded-state passes in each of two independent
processes, in forward/reverse order. All 4,099 original JavaScript positions
have llguidance measurements and identical GLRMask full masks. Percentiles
use the minimum **complete same-run mask-plus-commit pair** across six equal
repetitions per position. Raw maxima are retained separately.

`published` is the independently frozen top-first-key release (`8f773a455` code).
`reference` is the new combined binary with only this union memo disabled.
`candidate` is that binary with production defaults. The combined base also
retains the peer weight-range change; separate same-binary controls therefore
isolate this runtime optimization from that update and binary layout effects.

| Metric | published | reference | candidate |
| --- | ---: | ---: | ---: |
| P50, ms | 0.749042 | 0.741709 | 0.692959 |
| P90, ms | 2.614725 | 2.639091 | 2.183000 |
| P99, ms | 3.587721 | 3.617287 | 3.084758 |
| P99.9, ms | 4.072165 | 4.032849 | 3.593043 |
| P100, ms | 4.371708 | 4.272208 | 3.949000 |
| Raw maximum, ms | 5.409500 | 4.888959 | 4.445417 |

## Correctness and regression coverage

The exact code passed 2,272 Rust tests, zero failures and
55 ignored, plus 42 Python tests, documentation checks and
example compilation. Four 126-schema compiled/loaded static/dynamic captures
each preserve all 77,582 full masks and outcome markers against pinned hashes.
The JSB guard preserves all 9,045 masks and scores only its 9,021 llguidance-
measured keys. Static JS preserves all 4,099 masks too.

### Static JavaScript guard

| Metric | published | reference | candidate |
| --- | ---: | ---: | ---: |
| P50, us | 19.709000 | 19.333000 | 19.625000 |
| P90, us | 30.467200 | 29.458000 | 29.925200 |
| P99, us | 47.801000 | 46.585660 | 48.542820 |
| P100, us | 84.833000 | 82.375000 | 78.000000 |
| Raw maximum, us | 262.958000 | 144.750000 | 117.875000 |

### Dynamic JSB guard (32 supported schemas)

| Metric | published | reference | candidate |
| --- | ---: | ---: | ---: |
| P50, us | 39.209000 | 39.042000 | 38.917000 |
| P90, us | 117.500000 | 116.374000 | 116.292000 |
| P99, us | 342.841600 | 340.433000 | 339.366600 |
| P100, us | 2314.749000 | 2309.583000 | 2271.084000 |
| Raw maximum, us | 2463.292000 | 2621.500000 | 2517.750000 |

## Native 1,000-schema checks

These checks use a fixed selection of the first 1,000 llguidance-supported
schemas. Both modes retain all builds, semantic outcomes and the same 367,640
llguidance-measured intervals. They are not full-population performance claims.
Both native files deliberately contain the same frozen binary: the group
historically called `published` uses only the union-memo diagnostic opt-out.
It must not be mistaken for a separately built old release.

Native TBM is commit-to-next-mask, unlike the Python same-position pair above.
Each native statistic uses two independent runs in interleaved 250-schema
chunks. Raw maxima, every build observation and worst positions are retained
in the companion JSON. Compiler code is unchanged by this runtime patch;
timing movement is not automatically interpreted as a build improvement.

### Native static

| Metric | published | candidate |
| --- | ---: | ---: |
| P50, us | 3.583000 | 3.500000 |
| P90, us | 6.083000 | 6.000000 |
| P99, us | 13.042000 | 12.917000 |
| P100, us | 42.458000 | 42.417000 |
| Raw maximum, us | 57.750000 | 196.459000 |

| Build metric, ms | Memo disabled | Default |
| --- | ---: | ---: |
| P50 | 9.425125 | 9.388562 |
| P90 | 70.447445 | 70.996508 |
| P99 | 230.540170 | 232.098107 |
| P100 | 638.215875 | 614.905209 |

### Native dynamic

| Metric | published | candidate |
| --- | ---: | ---: |
| P50, us | 6.125000 | 6.083000 |
| P90, us | 67.375000 | 67.588100 |
| P99, us | 171.233620 | 172.250000 |
| P100, us | 2345.417000 | 2335.417000 |
| Raw maximum, us | 2472.041000 | 2381.417000 |

| Build metric, ms | Memo disabled | Default |
| --- | ---: | ---: |
| P50 | 1.508563 | 1.531271 |
| P90 | 4.215697 | 4.224279 |
| P99 | 20.129375 | 20.450183 |
| P100 | 45.081917 | 45.245500 |

### Raw static outliers rechecked

One native default run had isolated raw measurements of 196.459 us at
`create_invoice_beb99d93` example 0 interval 67 and 153.208 us at `typingsrc`
example 0 interval 339. These are retained above, not filtered away. Six fresh
interleaved processes over these and two other affected supported schemas
measured the same two positions at 12.083–15.583 us and 13.000–15.333 us
across both memo settings. The default runs had overall maxima 45.625, 55.250
and 43.791 us; the disabled runs 276.250, 43.667 and 46.667 us. Thus the two
original spikes did not reproduce as a setting-dependent regression. Their
cause is unproven, and no raw-maximum guarantee is claimed.

Independent compile-only checks also showed substantial wall-time variability:
24 JS dynamic builds per side had medians 114.501 versus 99.417 ms, whereas
medians in the full-trace screen moved the other way. Static compile-only
medians were 3.725 versus 3.458 seconds. The independent reference predates
the peer weight-range optimization. These observations do not establish a
compiler speedup caused by the runtime union memo. Native paired build ratios
were 0.99575 static and 1.00470 dynamic, with small mixed quantile movement.

## Fresh llguidance comparison

A separate same-machine `make example-js` comparison retained six measured
passes, no warmup and six builds per engine, using the thread CPU clock
and production cache settings. Every one of the 4,099 positions has six
complete pairs per engine. The raw timing sidecar was explicitly audited.
The grammars differ; masks between engines are not asserted equal.

| Metric | glrmask_dynamic | llguidance_native |
| --- | ---: | ---: |
| P50, ms | 0.729041 | 1.071208 |
| P90, ms | 2.286308 | 2.738183 |
| P99, ms | 3.205104 | 3.063067 |
| P100, ms | 4.257833 | 3.503042 |
| Raw maximum, ms | 5.071250 | 5.328792 |

| Median build, ms | GLRMask dynamic | llguidance |
| --- | ---: | ---: |
| Native compilation | 101.959645 | 4.155979 |

The P100 target remains unmet in this comparison. Do not describe the whole
upper tail as below llguidance. Raw cold maxima remain separate limitations.

The companion JSON retains all reported protocols, raw maxima and build
observations, test/capture evidence, input and source hashes, and an
artifact manifest. Original traces and sidecars remain in the isolated
worktree. No result from an llguidance-unsupported problem is scored.
