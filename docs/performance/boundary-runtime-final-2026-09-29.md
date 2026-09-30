# GLRMask dynamic-boundary finalization — 29 September 2026

**Validated final runtime record. The static-scale boundary-p100 target is not met.**

## Decision and limits

This document records the selected final changes and their validated runtime revision. The session is being closed with the performance target explicitly unmet, not with a claim that a fundamental limit was proved.

With normal caching, final p99 was **281.480–297.083 microseconds** and stabilized positional p100 was **445.344–691.728 microseconds**, versus **516.093–692.851 microseconds** for the previous published revision. Eleven of twelve matched final/default-control cohort comparisons had a lower p100. The median across those final positional maxima was 479.832 microseconds; this is a summary of cohort maxima, not a new corpus percentile. Cache-off comparisons improved in three of four matched comparisons and retained one 584.902-microsecond regression.

**Raw worst-call improvement is not established.** A final control call reached **12.245255 ms**, versus a **5.550524 ms** maximum in the previous-main arm. All raw events remain in the tables and source data. This final event did not recur during 12 fresh full-history repetitions per arm at its exact position (three arms, 36 histories), but that does not prove it was host noise. The largest final follow-up call there was 564.594 microseconds. Follow-up results are fixed-position repetition statistics, not replacements for the full-corpus maxima. The cause of the original event remains unresolved.

Ordinary-mask results are exact but performance is mixed; no universal per-workload speedup is claimed. The fallback switches are retained. No new optimization experiment is left running as part of this session.

## Validated runtime

Runtime revision: `23c0b8b81b7e07f1d21ddca0efb6f9eac69de93b`. Previous published baseline: `1dbd55db41fa51ad135cbe095cc0ebfa8b148369`.
Two changes are retained: lightweight generic provider initialization (reuse the first action lookup and avoid unused semantic-key setup), and mask-local control-closed terminal support (remove impossible grammar-terminal candidates before exact admission). Both are enabled by default. Existing parser actions, control semantics, lexical checks and the shared token-trie walker remain authoritative.
The previously published completed-child epsilon-only resets, packed boundary vocabulary, saved-depth traversal storage and bounded high-raw-coordinate caches remain enabled. No template runtime, shared-empty GSS pool, singleton-wrapper experiment, native-coordinate-only experiment or unproved architectural rewrite is added.
Explicit fallbacks: `GLRMASK_PROVIDER_LIGHT_START=0` and `GLRMASK_CLOSED_ADMISSION_SUPPORT=0`. Support caches are mask-local and never truncate the parser language on an eligibility limit. These limits are not hard bounds on total proof CPU or memory: full reference control closure precedes some eligibility checks and may be repeated by fallback. All preparation and destruction remain in boundary time.

## Final correctness checks

The final actual-default build passed **978 root tests, 225 GLR tests and 148 public/API tests**, with zero failures. The root suite has 51 ignored tests and the GLR suite has two. No-internal, release and external-driver builds also passed.
33,590,196 independent finite-language, colour-language and lifecycle membership checks passed for actual defaults and explicit fallbacks. Sixty-four full 11,767-position replays matched complete masks, acceptance and commits. Ordinary workloads matched complete 576-mask buffers in every arm/repetition. Source, dependency, binary and input hashes were verified. Focused tests are subsets of the reported suite totals, not additional tests.

## Boundary-only performance

All arms use the same 257-token child-boundary domain and a static parent boundary. B includes boundary dispatch/setup/cleanup; N is ordinary work from the same B-selected invocation. Cache-off and normal-cache modes are measured separately. Two additional normal-cache batches provide predeclared fresh-process confirmation, not post-hoc removal of inconvenient cohorts. Each cohort takes the lower of two thread-CPU measurements at each position and then computes the nearest-rank percentile across positions. These stabilized positional maxima are not raw worst-run guarantees.
Published is the previous main binary. Previous is the completed experimental combination with explicit flags. Default and control are the final integrated source with no optimization flags, providing an identical A/A comparison. Values are microseconds.

| Mode | Cohort | Arm | p99 | p99.9 | p100 | Worst position | Same-call N |
|---|---:|---|---:|---:|---:|---|---:|
| default-257 | 1 | published | 302.228 | 414.197 | 546.320 | 2/5274 | 403.807 |
| default-257 | 2 | published | 301.250 | 422.941 | 539.220 | 2/3700 | 357.657 |
| default-257 | 1 | previous | 292.209 | 390.517 | 447.824 | 2/3192 | 207.953 |
| default-257 | 2 | previous | 301.511 | 398.940 | 531.472 | 1/2687 | 255.615 |
| default-257 | 1 | default | 282.348 | 350.742 | 584.902 | 0/1457 | 401.261 |
| default-257 | 2 | default | 290.267 | 392.975 | 515.410 | 2/596 | 378.911 |
| default-257 | 1 | control | 283.245 | 377.919 | 450.667 | 2/3192 | 173.711 |
| default-257 | 2 | control | 287.967 | 396.365 | 521.598 | 2/2212 | 406.270 |
| cache-on-257 | 1 | published | 313.480 | 413.420 | 516.093 | 1/2922 | 154.204 |
| cache-on-257 | 2 | published | 315.217 | 433.818 | 680.510 | 1/2945 | 311.605 |
| cache-on-257 | 1 | previous | 307.850 | 407.210 | 468.206 | 2/4500 | 232.915 |
| cache-on-257 | 2 | previous | 312.983 | 423.258 | 645.142 | 1/1231 | 502.072 |
| cache-on-257 | 1 | default | 291.132 | 398.210 | 445.344 | 2/5239 | 135.930 |
| cache-on-257 | 2 | default | 287.445 | 371.886 | 477.844 | 1/2831 | 218.360 |
| cache-on-257 | 1 | control | 283.709 | 382.707 | 476.749 | 2/3200 | 135.800 |
| cache-on-257 | 2 | control | 290.977 | 404.162 | 502.146 | 2/260 | 241.490 |
| normal-confirm-a | 1 | published | 303.737 | 423.312 | 570.515 | 2/309 | 298.653 |
| normal-confirm-a | 2 | published | 306.066 | 416.673 | 565.611 | 1/2847 | 345.288 |
| normal-confirm-a | 1 | previous | 302.976 | 426.596 | 764.007 | 1/3007 | 434.352 |
| normal-confirm-a | 2 | previous | 304.577 | 430.092 | 524.741 | 2/3175 | 223.642 |
| normal-confirm-a | 1 | default | 286.574 | 366.196 | 447.609 | 1/2687 | 255.195 |
| normal-confirm-a | 2 | default | 285.768 | 358.853 | 483.027 | 2/3129 | 138.473 |
| normal-confirm-a | 1 | control | 286.577 | 373.908 | 465.019 | 1/2831 | 255.752 |
| normal-confirm-a | 2 | control | 287.764 | 405.524 | 481.819 | 0/2624 | 452.296 |
| normal-confirm-b | 1 | published | 306.731 | 427.349 | 692.851 | 2/1380 | 446.379 |
| normal-confirm-b | 2 | published | 312.326 | 415.411 | 525.722 | 2/3175 | 147.198 |
| normal-confirm-b | 1 | previous | 304.282 | 386.589 | 478.665 | 0/1104 | 258.912 |
| normal-confirm-b | 2 | previous | 297.523 | 392.681 | 459.592 | 0/560 | 387.137 |
| normal-confirm-b | 1 | default | 297.083 | 409.292 | 691.728 | 1/895 | 665.678 |
| normal-confirm-b | 2 | default | 295.464 | 373.990 | 470.495 | 2/3192 | 197.940 |
| normal-confirm-b | 1 | control | 290.070 | 386.480 | 540.380 | 2/3175 | 180.980 |
| normal-confirm-b | 2 | control | 281.480 | 363.767 | 556.630 | 2/1124 | 375.157 |

## Raw per-run boundary maxima

These are retained separately rather than silently discarded or described as proven host noise. Four repetitions are listed per row, in microseconds.

| Mode and arm | Raw B maxima |
|---|---|
| default-257 published | 1186.899, 920.590, 1247.125, 4844.154 |
| default-257 previous | 1017.778, 2135.996, 1247.250, 907.116 |
| default-257 default | 974.542, 4444.640, 1171.512, 4333.456 |
| default-257 control | 1270.606, 869.936, 1076.824, 1864.876 |
| cache-on-257 published | 995.547, 5431.509, 1082.413, 1085.242 |
| cache-on-257 previous | 1139.625, 907.265, 1324.779, 1049.603 |
| cache-on-257 default | 876.625, 760.538, 845.508, 808.202 |
| cache-on-257 control | 860.909, 882.955, 970.743, 911.528 |
| normal-confirm-a published | 1183.162, 854.084, 1244.755, 5550.524 |
| normal-confirm-a previous | 2421.181, 1138.285, 967.383, 980.309 |
| normal-confirm-a default | 1229.249, 736.913, 771.680, 4584.794 |
| normal-confirm-a control | 12245.255, 905.142, 893.224, 1173.987 |
| normal-confirm-b published | 1880.139, 2338.382, 887.191, 808.045 |
| normal-confirm-b previous | 1445.632, 1114.313, 893.390, 802.638 |
| normal-confirm-b default | 1013.614, 1050.071, 832.002, 876.376 |
| normal-confirm-b control | 854.709, 3683.510, 950.743, 982.562 |

## Ordinary-mask guardrails

| Arm | Case | Cohort 1 p100 | Cohort 2 p100 |
|---|---|---:|---:|
| published | finite | 293.383 | 280.041 |
| published | number | 465.735 | 445.310 |
| published | overlap | 1761.307 | 1632.583 |
| published | string | 739.734 | 509.133 |
| previous | finite | 296.811 | 299.831 |
| previous | number | 655.668 | 402.173 |
| previous | overlap | 2142.587 | 1578.263 |
| previous | string | 706.863 | 435.435 |
| default | finite | 253.679 | 339.914 |
| default | number | 400.003 | 680.624 |
| default | overlap | 1749.625 | 1552.706 |
| default | string | 569.888 | 444.810 |
| control | finite | 264.072 | 236.207 |
| control | number | 498.801 | 482.100 |
| control | overlap | 2140.072 | 1833.067 |
| control | string | 435.450 | 472.057 |

## What is and is not finished

The goal remains B on roughly the same scale as ordinary same-call N. These measurements do not establish that goal or a fundamental lower bound. The earlier suggestion of an architectural limit remains a hypothesis. Hardware instruction/cycle counters were unavailable on this VM; no such measurements are claimed.
The allocation-only experiments are closed without promotion: borrowed/local candidate views, native-coordinate copying, shared empty GSS and singleton result wrappers. Their source/evidence remains recoverable. LIGHT and support were evaluated independently and together; only the combined, final-default policy is selected here.
This pass does not change the public API or serialized artifact schema and does not claim a new constraint-build-time benchmark, a cross-platform performance result, or a 10,000-schema sweep. All performance evidence is from the stated Hetzner workload. Remaining parser/product-frontier costs are documented for possible future work, not represented as an active background task.

## Raw-outlier follow-up

12 fresh full-history repetitions per mode, 3 fixed raw-outlier positions, normal caches; NOT corpus percentiles. These measurements do not erase any raw outlier or prove it was noise.

| Position | Published median B | Final median B | Identical final-control median B |
|---|---:|---:|---:|
| 1/1833 | 166.252 | 224.331 | 252.861 |
| 1/2395 | 385.638 | 365.346 | 372.166 |
| 2/4536 | 106.305 | 163.910 | 142.094 |

Values above are microseconds. Exact full-history masks, acceptance, commits, and source/library/input hashes passed. The follow-up is diagnostic; it does not replace or censor the predeclared 64-replay data.
