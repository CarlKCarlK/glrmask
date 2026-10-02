# Table-free parser validation — 30 September 2026

**Historical evidence:** this report describes its pinned September 30 revision.
Its default-backend and supported-composition statements are historical; it does
not qualify the current native replacement.

**Disposition: validated as an opt-in standalone backend; retain the LR default.**

The table-free implementation has lower warm mean time between masks (TBM) and lower warm TBM P99 than the LR backend in every one of the eight measured scenarios. It also produces smaller artifacts. These results do **not** establish that every mask-only tail, build, or load is faster. Compiled-component composition is outside this release’s supported table-free scope.

Runtime source: `ea136ca4bcf8557e5d052d76004803232523e3a1`. Documentation-only follow-up commits may have a later revision; the measurements below refer to that exact runtime source.

The [parser contract](template-parser.md) gives typed Rust/Python selection, data-only provider semantics, validation limits, and persistence requirements. The [compact evidence pack](evidence/template-parser-validation-2026-09-30.zip) contains the paired process summaries, comparison CSV, per-schema serialized sizes, adverse-cell inventory, checksums, and an integrity/arithmetic verifier.

## What was finalized

The selected constraint physically omits the LR table. Masking and token commitment retain their shared engines; parser advancement and admission dispatch through complete acyclic template relations. Built-in grammars may use LR compilation transiently to derive those relations. A data-only `ParserProgram` bypasses that frontend and supports both static and dynamic mask compilation.

The final runtime combines validated template-index reuse during loading, bounded initial-commit priming, and shared mask-cache classification with the original sparse allocation behavior retained. The mask-cache admission threshold, key semantics, and eviction policy were not changed. Unsupported composition returns an error rather than retaining a hidden LR fallback.

## Runtime comparison

These measurements were made on an **Apple M1 Pro Mac (Darwin arm64)**, not the earlier Windows or Hetzner machines. Values are microseconds of calling-thread CPU time. Each value is a median of eight process-level measurements. Within a process, all three warm replays are pooled before calculating the statistic; there are no per-token minima and the first warm replay is not discarded. The LR and template columns below use the **same final executable**.

| Mode and corpus | Construction state | Mean TBM: LR | Mean TBM: templates | Change | TBM P99: LR | TBM P99: templates |
|---|---|---:|---:|---:|---:|---:|
| Static / 20 JSB schemas | fresh | 5.861 | 4.792 | -18.23% | 36.612 | 35.312 |
| Static / 20 JSB schemas | reloaded | 5.313 | 4.183 | -21.29% | 14.146 | 10.986 |
| Static / JavaScript | fresh | 17.307 | 16.108 | -6.93% | 50.523 | 37.500 |
| Static / JavaScript | reloaded | 17.452 | 16.501 | -5.45% | 49.526 | 38.336 |
| O2 / 20 JSB schemas | fresh | 7.073 | 4.627 | -34.58% | 27.459 | 18.416 |
| O2 / 20 JSB schemas | reloaded | 7.088 | 4.809 | -32.15% | 27.257 | 18.862 |
| O2 / JavaScript | fresh | 21.148 | 13.440 | -36.45% | 252.866 | 210.862 |
| O2 / JavaScript | reloaded | 21.218 | 15.399 | -27.42% | 249.660 | 218.389 |

Mean commitment improves by approximately 28–58% across these scenarios. That improvement is not inferred from a configuration flag: the recorded backend reports verify that each template constraint has no LR table and that the LR controls do retain one.

### Why the default is not changing

The clearest remaining aggregate regression is O2 on the fresh 20-schema panel: mask-only warm P99 rises from **10.424 to 10.904 µs**, or **+4.60%**. The exploratory paired-block 95% interval is **+1.08% to +8.72%**. Reloaded O2 mask-only P99 rises about 2.66%, with an interval crossing zero. Static JavaScript after reload also has a small mask-only P99 increase. These values are retained alongside the larger TBM improvements.

The evidence pack preserves 291 correlated, exploratory per-case flags. They are not 291 independent defects: statistics share observations, and the analysis makes no multiple-comparison correction. Conversely, an interval crossing zero is not proof of non-regression. This release makes no universal maximum-latency or “strictly below at every percentile” claim.

## Build and reload costs

The following are **sums of per-schema medians**, not population percentiles. The JavaScript rows contain one grammar. Build includes the measured frontend/setup and complete selected compilation; the template whole-compile field is not added to a second LR compilation. Process CPU includes parallel worker activity. Reload is an in-memory API call after compilation, not filesystem I/O or an otherwise untouched process startup.

| Mode and corpus | Build wall, LR → templates (ms) | Build process CPU, LR → templates (ms) | Reload wall, LR → templates (ms) |
|---|---:|---:|---:|
| Static / 20 JSB schemas | 1,461.931 → 1,634.944 | 3,480.945 → 3,713.242 | 43.980 → 84.634 |
| Static / JavaScript | 3,798.176 → 3,901.600 | 7,935.738 → 8,083.419 | 2.331 → 10.747 |
| O2 / 20 JSB schemas | 533.765 → 765.587 | 1,262.653 → 1,552.639 | 451.912 → 100.275 |
| O2 / JavaScript | 293.922 → 401.728 | 904.619 → 1,088.414 | 25.837 → 10.931 |

Static builds and static reloads are not at parity in this panel. In particular, JavaScript static reload is approximately **2.331 → 10.747 ms**, while O2 JavaScript reload improves approximately **25.837 → 10.931 ms**. Removing duplicate template preparation reduced avoidable work, but it did not eliminate all template-loading cost. Preparation/priming is charged to construction or loading rather than omitted from those intervals.

## Serialized footprint

These are actual byte counts from a fresh section probe relinked against the **same final runtime library**. The parser field includes the full executable parser representation, not just one auxiliary DFA. Static LR uses the compact table codec; the existing O2 LR container uses a bincode `GLRTable` field. A compact diagnostic re-encoding of that O2 field is not treated as its actual stored size.

| Mode and corpus | Whole LR artifact(s), bytes | Whole template artifact(s), bytes | LR parser field(s), bytes | Template program(s), bytes |
|---|---:|---:|---:|---:|
| Static / 20 JSB schemas | 59,328,336 | 55,896,504 | 2,702,093 | 372,087 |
| Static / JavaScript | 19,248,143 | 18,893,392 | 266,318 | 180,528 |
| O2 / 20 JSB schemas | 105,144,622 | 78,090,192 | 3,351,181 | 372,038 |
| O2 / JavaScript | 6,350,694 | 5,132,900 | 342,864 | 180,528 |

All 42 per-schema/mode template artifacts were smaller than their matching LR controls, and each loaded template artifact resaved exactly. Across the static 20-schema panel, the complete template program occupies about **13.77%** of the LR-table bytes; for JavaScript it is about **67.79%**. Whole-artifact O2 savings are not attributed solely to parser removal: its container, vocabulary, normalization, and cache representation also differ.

The new direct size probe measures 48 fewer bytes in the static LR parser field for `jscsrc` and for JavaScript than the earlier timing-archive artifacts. Template sizes are unchanged. Both observations are preserved; the original timing data has not been edited, and this report does not assign an unverified cause to that small representation difference.

## Correctness and public API qualification

The final source passed **1,001 root unit tests and 87 selected integration tests: 1,088 Rust tests passed, with 51 existing ignored tests explicitly retained**. The current-source root suite was rebuilt rather than reusing the earlier 997-test result. Subprocess summaries were not double-counted; the last outer root summary is authoritative.

The isolated Python extension was rebuilt against the same native runtime library and passed **51 tests**, with no failures, errors, or skips. It was not substituted into the globally installed package.

Full-vocabulary mask and completion comparisons passed at **146,916 prefix/representation points** on the selected panel and JavaScript, and **174,294 additional points** on the broader 100-schema sample. The broader sample had 96 supported schemas per mode; four frontend/build failures were independently reproduced with both backends. These are repeated correctness comparisons across fresh, self-contained reload, and external-vocabulary reload—not that many independent grammars.

Permanent tests cover independent literal stack/lexer oracles, arbitrary state numbering, repeated READ operations, explicit dead edges shadowing DEFAULT, nullable/empty concrete stacks, large finite output DAGs, exact external-vocabulary binding, and malformed graph/phase-link/dimension rejection. A separate bounded artifact mutation audit distinguishes malformed input rejection from structurally valid but mutually inconsistent modified sections; structural validation is not artifact authentication.

Documentation validation executed both Rust examples in the parser contract, the guide’s backend example with an explicit calling context, and four published Python snippets with explicit setup. Missing/mismatched external-vocabulary negative controls passed. These checks are three Rust doctests and four Python example executions, not extra counts silently added to the test-suite totals.

The recorded Rust and Python qualification uses actual source with a coherent, hash-pinned dependency graph. It consists of direct `rustc`/`rustdoc` builds and isolated Python tests, **not a newly claimed full Cargo workspace build, wheel publication, or package-registry release**.

## Measurement and reproducibility limits

The performance panel intentionally overrepresents build-time and mask-time tails among paired-success schemas from the original approximately 10,000-schema JSB run. It is not an unbiased estimate of the complete JSB population. Longer examples contribute more token observations, and each replay repeats those observations.

The eight scenarios comprise 256 independently launched processes, arranged as eight counterbalanced blocks of old/new implementation × LR/template backend per scenario. There are 3,117,312 first-replay token observations and 12,469,248 observations across first plus three warm replays. These are repeated observations. A benchmark child has a 60-second hard limit. Initial-mask/start costs are recorded separately, not included in the headline commit-plus-next-mask TBM.

“Fresh” means newly compiled in that process. “Reloaded” means serialized and loaded before replay in that same process. “Cold” in the raw timing tables is the first replay; it is not a claim that the machine, allocator, compiler, or global code caches are untouched. Separate artifact-only investigations exist, but their numbers are not spliced into this matrix.

The canonical JavaScript timing fixture stops before the grammar’s explicit EOF-marker terminal. Its last recorded completion value is therefore not a claim that the unextended input is a complete program. EOF-extended completion checks are separate correctness fixtures and do not replace the timing corpus.

An independent reviewer verified all 256 raw-output digests, input/condition identities and physical-backend reports, recomputed all **12,469,248 per-step `commit + mask = TBM` sums**, and reconciled **2,400 paired statistical cells** without importing the producing aggregation helpers. Build/load medians and final-source hashes also reconcile.

The compact pack includes process-level summaries and a checksum/arithmetic verifier. It omits the approximately 270 MB of raw timing JSON; their relative provenance and SHA-256 digests are indexed. Verifying the compact pack is not a rerun of the omitted raw-sample audit. Test sources and the implementation remain in the repository.

## Release boundary

This is an opt-in standalone parser backend. The default remains `LrTable`/`LR_TABLE`; compiled-component composition is not enabled by this change. Separate composition work is not merged or declared complete by this report. The feature branch is distinct from the default branch, and no version tag or crates.io/PyPI publication is implied.

Changing the global default requires a separate compatibility and performance decision. The documented mask-tail, build, and static-load tradeoffs prevent presenting the current result as the originally requested unconditional, no-compromise replacement.
