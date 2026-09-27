# Exact LR closure emission caching — 27 September 2026

This change caches which lookahead sets have already been emitted to a
nonterminal's productions within **one LR closure**. It does not change grammar
semantics, parser-state numbering, automaton output, inlining budgets, or runtime
mask code. The measured comparison is published `c3a479907` against
this default-enabled implementation, not against an older performance baseline.

## Exactness argument

For a nonterminal with more than one production, each completed fanout unions
the same requested lookaheads into every corresponding production-entry item.
A later request contained in the union of those completed emissions cannot
add a bit to any entry. The reference would enqueue no new delta; omitting that
fanout therefore preserves the fixed point. The first request still executes
even when its set is empty, because the reference inserts entry cores in that
case. Single-production nonterminals bypass the cache. No memo survives the
closure, and the ordered successor/state interning algorithm is unchanged.

Four tests cover empty/multiword lookaheads, repeated subsets, zero-lookahead
entry cores, and the default/diagnostic policy. A 69-grammar matrix checks exact
canonical item sets, transitions, and complete serialized tables with
preclosure reuse both on and off. The prior flat-kernel and owned-kernel
experiments are not included.

## JavaScript build comparison

The original checked-in JavaScript grammar and Llama-3 vocabulary are unchanged.
Published and candidate binaries are frozen and interleaved in ABBA order,
with 24 dynamic builds per version and 12 static builds per version.

| Median build, ms | Published reference | Default candidate |
| --- | ---: | ---: |
| Dynamic | 114.920 | 89.476 |
| Static | 3392.226 | 3398.558 |

Dynamic build median is 22.14% lower.
All serialized dynamic artifacts in this build comparison are byte-identical.
An earlier prototype batch showed a static slowdown; it was retained rather
than discarded, then followed by this larger actual-published comparison.
The JSON contains every raw build observation, including outliers.

## Runtime guard, not a runtime speedup claim

Each version runs the complete 31-example/4,099-position corpus in two separate
processes, with three fresh-state repetitions per example per process. Every
full vocabulary mask matches, and every scored position has a llguidance
measurement. Percentiles use the minimum complete mask-plus-commit pair at
each position across the six equal repetitions. Raw maxima remain separate.

| JavaScript TBM, microseconds | Published reference | Default candidate |
| --- | ---: | ---: |
| Static P90 | 29.092 | 28.708 |
| Static P99 | 45.334 | 44.252 |
| Static P100 | 77.833 | 78.042 |
| Static raw maximum | 247.084 | 250.167 |
| Dynamic P90 | 7154.042 | 7140.216 |
| Dynamic P99 | 11963.646 | 11877.505 |
| Dynamic P100 | 16504.583 | 16564.125 |
| Dynamic raw maximum | 17162.125 | 17285.208 |

## Matched JSB guard

The fixed first 1,000 llguidance-success schemas are run in 250-schema chunks
in alternating ABBA/BAAB order. Failures and trajectory outcomes are compared,
not dropped. All 1,000 builds and outcomes agree. Only exact
(problem, example, position) keys measured by llguidance are scored. Each
position retains two independent native runs per version; the native metric
is commitment through the following mask, not CFA's same-position pair.
This is a regression panel, not a new full-population result.

| JSB metric | Published reference | Default candidate |
| --- | ---: | ---: |
| Build, ms P50 | 9.271 | 9.216 |
| Build, ms P90 | 72.661 | 74.012 |
| Build, ms P99 | 216.436 | 214.106 |
| Build, ms P100 | 609.909 | 603.535 |
| TBM, microseconds P50 | 3.417 | 3.375 |
| TBM, microseconds P90 | 5.917 | 5.834 |
| TBM, microseconds P99 | 12.667 | 12.500 |
| TBM, microseconds P100 | 45.917 | 43.875 |

Matched runtime intervals: 367,640. Median paired build ratio: 0.996478.

The 126-schema correctness guard separately checks 77,582 full vocabulary masks
and outcome markers in each of four modes: static compiled, static reloaded,
dynamic compiled, and dynamic reloaded. All four decoded-stream hashes match
the previously published references. This includes invalid continuations and
serialization behavior; it is not merely a check that gold tokens survive.

## Selected build regressions were rechecked

The twelve worst paired build ratios from the 1,000-schema panel, together with
ten cases around P90, are repeated six times in alternating order. Three modes
separate published-binary layout from algorithm choice: published reference,
candidate with the memo disabled, and default candidate. Their per-case raw
observations and medians are retained in the JSON. This deliberately selected
panel is not a population distribution and cannot establish a broad speedup.

The selected per-case medians ranged from -5.6% to +4.0% versus the published binary; the initially observed 18% increases did not reproduce. This does not establish that every schema is faster.

## Validation and default behavior

The clean-environment gate passed 2,229 Rust unit/integration tests plus 1 documentation test, zero failures and 55 ignored, and all 42 Python tests. Workspace examples passed type checking; existing unused-assignment warnings are unchanged. Benchmark-only compatibility flags were removed for correctness tests.

The cache is enabled by default. `GLRMASK_DISABLE_LR1_EMISSION_CACHE=1` selects
the literal old closure for diagnostics. There is no required opt-in and no
change to LALR selection, vocabulary, grammar, runtime masking policy, or
unit-inlining work budgets. Each final candidate benchmark used the diagnostic
unset. Large raw artifacts and frozen binaries remain in the private worktree;
the accompanying JSON retains selection provenance, complete paired summaries,
raw build repeats, exact hashes, and limitations.

The first full-mask verifier invocation used the wrong 214-schema fixture and correctly failed its expected-count assertion. Its capture and logs are retained. The reference 126-schema input was then recovered and hash-pinned before all four successful reruns; this was a test-driver error, not a suppressed library failure.
