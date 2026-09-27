# Overhaul checkpoint: structure and validation

This is a **correctness-validated integration checkpoint, not final performance
acceptance or the end of the cleanup**. It starts from `7e84a64c9`; later changes
on the shared main branch still need reconciliation before release.

## Structural changes

The supported lifecycle now lives in `src/api/`, separated into grammar source,
bindings, compile/link options, compilation, unlinked constraints, and private
partition analysis. Composition is its own compiler subsystem rather than one
large linker file. Runtime lifecycle, compiler views, observation caches, parser
state and dynamic execution have clearer ownership boundaries.

Persistence now has separate framing, core metadata, component restoration,
mask payload, weight and save/load modules. Obsolete compatibility readers and
historical wrapper chains are removed. Current wire versions are unchanged;
old files are not promised compatibility. Current validation and backing-data
ownership are retained.

Lexer compilation is split by phase. A private typed partition policy replaces
combinations of boolean arguments and redundant wrappers. Vocabulary-equivalence
analysis remains available through the internal tooling bridges, including
`glrmask._internal.VocabPartition`, without expanding the supported public API.

The linker now requires a static or dynamic boundary backend. Unreachable
implementations behind retired adapters have been removed, retaining explicit
errors for those unsupported entry points. Large regression modules are in
separate files rather than interleaved with production algorithms.

The same source scope (`src`, `crates`, `python/src`) goes from 479,914 Rust lines
to 470,846: **9,068 net lines removed**. File moves are not counted as deletions.
A missing extracted test file, accidentally excluded by a broad `build/` ignore
rule, was restored and is tracked. Several large algorithm modules still remain;
this checkpoint does not claim that every internal design is optimal.

See [architecture](architecture.md), [runtime ownership](runtime-implementation.md)
and [lexer phases](lexer-architecture.md) for boundaries and invariants.

## Combined correctness and API checks

These results apply to the **combined** refactor, not just its individual passes.

| Gate | Result |
| --- | --- |
| Normal public library compilation | Pass |
| Workspace, all targets and all features | Pass |
| Standalone default-feature lexer check | Pass |
| Native Python release build | Pass |
| Python tests | 74 passed |
| Rust workspace library/integration tests | 2,169 passed; 0 failed; 41 ignored |
| Public Rust doctest | 1 passed |
| Standalone lexer tests | 239 passed |
| Public Rust documentation, warnings denied | Pass |
| Supported Python exports/member signatures | Exact match to baseline |
| Public/trait signatures on the four reorganized Rust facade types | 23 matched |

The Rust count uses the last result for each of 42 Cargo targets. Nine nested
child-process test summaries are not counted twice. The standalone lexer result
checks a separate feature configuration, not 239 additional unique tests.

The eight supported Python names are `Constraint`, `ConstraintState`,
`ExactToken`, `ExactTokens`, `Grammar`, `Optimization`, `UnlinkedConstraint`,
and `Vocab`. Tests preserve full masks, current save/load roundtrips and errors,
not merely successful imports.

## Performance evidence and limits

All measurements used the same CX33, frozen baseline/candidate native builds and
128,256-entry vocabulary, without competing compilation during timing windows.
No performance flags or library source changes were introduced to make the
candidate pass. Build, mask, commit, save/load and partition costs are separate.

The one- and four-thread guards cover 38 constraint/mode cases and 24 canonical
partition cases. Complete mask traces, accepted tokens, partition classes and
current-format roundtrips match in every completed comparison. The larger
cold/warm guard adds ten case families; four real-schema cases test build and
initial-mask behavior, not accepted sequence replay.

Initial separate-process runs flagged material slowdowns. Shorter interleaved
retests often did not reproduce them. A six-build attribution experiment did not
identify a consistent pass-level cause. These observations do not justify
silently dismissing all outliers as noise.

An additional diagnostic uses two isolated persistent processes and adjacent
ABBA **whole traces**. The measuring main threads are pinned after compilation,
leaving compiler workers' affinities intact. It records eight fresh build pairs
and 336 traces per variant per case. This reduces process/time drift but changes
the measurement environment; it supplements, rather than replaces, the
separate-process results. The ratio is candidate/baseline; above 1 is slower.

| Case | Mode | CPU0 raw mask p50 | CPU0 raw mask p99 | CPU2 raw mask p50 | CPU2 raw mask p99 |
| --- | --- | ---: | ---: | ---: | ---: |
| array | FAST_BUILD | 0.990 | 1.004 | 0.983 | 1.015 |
| array | AUTO | 1.004 | 1.050 | 1.005 | 1.006 |
| array | FAST_RUNTIME | 0.996 | 1.006 | 1.005 | 1.058 |
| bounded_string | O2 | 0.980 | 0.771 | 1.021 | 0.910 |
| partition_recursive | O2 | 1.011 | 0.958 | 0.971 | 1.072 |
| string_array | O2 | 0.965 | 0.911 | 1.033 | 0.903 |
| integer | FAST_RUNTIME | 1.075 | 1.467 | 1.043 | 1.183 |
| shared | AUTO | 1.012 | 1.093 | 0.992 | 1.021 |

Array mask medians are close to baseline in both adjacent runs. The earlier
large O2 mask slowdowns are not consistent across the controlled comparisons.
**Static integer mask tails remain an open flag:** raw p99 was 7.464 to 10.950
microseconds on CPU0 and 9.127 to 10.800 microseconds on CPU2. Short-trace maxima
also vary substantially, sometimes favoring baseline and sometimes candidate.
No whole-corpus p99.9/p100 guarantee or blanket no-regression claim is made.

The native timing helper excludes Python/IPC overhead for supported static and
dynamic facade calls; the legacy internal O2 commit path uses a Python-side
elapsed timer and must not be presented as pure native commit time. Warm
per-position medians and raw operation quantiles are distinct statistics. Build
elapsed time during competing development builds is not a compile-time benchmark.

## Deliberately outside this checkpoint

- The separately proposed deletion of four empty runtime cache fields is not
  applied. Its private memory-layout change needs its own performance decision.
- Newer shared-main correctness/performance changes must be reconciled with the
  relocated modules before a main-branch merge.
- Canonical corpus tail acceptance, the remaining static integer tail flag,
  and remaining oversized implementation files still need work.
- The inherited formatter configuration disables formatting. A successful
  `cargo fmt --check` under that configuration is **not** a formatting quality
  gate. Replacing that policy is separate from these validated semantic changes.

## Reproducibility

The engineering asset directory is `integrated-overhaul-789322` on the build
host. It retains frozen packages, the vocabulary, source hashes, snapshot patch,
all successful and failed commands, raw timings and semantic fingerprints.
Key files are `logs/correctness-summary.json`, `logs/public-api-comparison.json`,
`logs/rust-facade-signatures.json`, `logs/paired-t1/summary.json`,
`logs/paired-t4/summary.json`, `logs/cold-warm-acceptance.jsonl`,
`logs/focused-t1-cpu0/`, `logs/cold-warm-cpu0.jsonl`,
`logs/focused-t4-cpus0-3/`, `logs/array-attribution/`,
`logs/adjacent-t4-cpu0/` and `logs/adjacent-t4-cpu2/`.

The repository's `scripts/refactor_acceptance.py` supports independent frozen
packages and full-vocabulary semantic comparisons. The additional diagnostic
scripts remain with the raw evidence rather than becoming supported library API.
