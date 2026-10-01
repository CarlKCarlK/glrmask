# Template parsing is faster cold and warm; startup tradeoffs remain

## Executive summary

The selected standalone table-free backend is **faster than LR on cold and warm
mean TBM in every measured scenario**. Static percentage gains carry through
closely. O2 JavaScript improves substantially cold as well; the O2 schema panel
still has a smaller cold percentage gain than its warm gain. That remaining
difference is explicit, not relabeled as equal performance.

Selected source: `6b3f5cc5b6473a81a19055319ceca0110d3c7a1b`. The measured native executable and Python extension
were built from exactly these source contents. This report supersedes the
30 September runtime comparison, whose setup and source are different.

The final additional optimization prepares a bounded, exact single-push
relation from the template graphs and uses the same GSS push operation in
commitment and uncached mask traversal. On JavaScript O2 it reduces cold mean
TBM by about 4–5% compared with the preceding integrated template backend.
The schema-panel incremental effect is approximately neutral.

The backend remains **explicitly selected, not the global default**. Compiled
component composition is not part of this qualified standalone implementation,
and template loading/building is not uniformly cheaper than LR. No hidden LR
table or fallback is retained when template mode is requested.

## Cold and warm decoding

Values are median per-process means in calling-thread CPU microseconds. Each
condition has eight counterbalanced blocks. Warm combines all three retained
warm replays; the first warm replay is also reported independently in the CSV.
These are observed workload results, not promises for every possible grammar.

| Mode / corpus | Representation | Cold LR → template, µs | Cold gain | Warm LR → template, µs | Warm gain |
|---|---|---:|---:|---:|---:|
| static / panel20 | fresh | 6.106 → 4.906 | 19.66% | 5.938 → 4.776 | 19.57% |
| static / panel20 | reloaded | 5.498 → 4.261 | 22.50% | 5.380 → 4.181 | 22.29% |
| static / js | fresh | 17.983 → 16.302 | 9.35% | 17.663 → 16.191 | 8.34% |
| static / js | reloaded | 18.142 → 16.857 | 7.08% | 18.235 → 16.763 | 8.07% |
| o2 / panel20 | fresh | 14.351 → 10.904 | 24.02% | 6.910 → 4.364 | 36.84% |
| o2 / panel20 | reloaded | 14.249 → 11.218 | 21.27% | 6.864 → 4.578 | 33.30% |
| o2 / js | fresh | 296.961 → 200.797 | 32.38% | 21.662 → 13.159 | 39.25% |
| o2 / js | reloaded | 294.436 → 206.958 | 29.71% | 21.546 → 15.731 | 26.99% |

“Cold” is the first complete replay after normal construction or loading,
including the actual production preparation policy. It is not a claim of a
flushed operating-system cache or an uninitialized process. Build and load
costs are measured separately and are not hidden in the warm results.

Static cold percentage gains are close to warm gains. For the O2 schema panel,
the cold reduction is about 21–24%, versus 33–37% warm. Commitment gains persist
cold; shared uncached lexer/trie/mask work is a much larger fraction of the cold
denominator. Absolute cold per-token savings remain larger than warm savings,
but **equal percentage carry-through has not been achieved in that panel**.

Initial masks are one call per example, not per token. An independent audit
includes every initial-mask call in total replay cost and retains those results
in `including-initial-masks.csv`; this does not remove the O2 percentage gap.

### TBM tail distribution

| Mode / corpus | Representation | Cold P99 LR → template, µs | Warm P99 LR → template, µs |
|---|---|---:|---:|
| static / panel20 | fresh | 38.807 → 36.944 | 37.793 → 36.064 |
| static / panel20 | reloaded | 15.519 → 11.655 | 15.091 → 11.166 |
| static / js | fresh | 53.596 → 40.813 | 52.812 → 40.397 |
| static / js | reloaded | 53.030 → 42.171 | 52.646 → 41.606 |
| o2 / panel20 | fresh | 75.900 → 61.416 | 25.640 → 15.250 |
| o2 / panel20 | reloaded | 74.319 → 62.533 | 25.015 → 15.812 |
| o2 / js | fresh | 1691.813 → 1038.119 | 263.049 → 218.186 |
| o2 / js | reloaded | 1679.830 → 1063.242 | 250.511 → 217.714 |

The evidence includes mean, P50, P90, P99, P99.9 and maximum for commitment,
mask generation and TBM, cold/first-warm/pooled-warm, with all process values.
Per-case observations and positive tail flags remain present; small isolated
regressions have not been filtered out. The reported maximum statistic is the
median of process maxima; observed maxima across all processes are also retained
in the independent audit, and should not be confused with a hard latency bound.

## Build cost is still a tradeoff

These are whole fresh-build wall and process CPU measurements from the same
paired runs, including frontend and parser conversion. Panel entries are sums
over the 20 selected schemas, not a claim that one schema takes that total time.

| Mode / corpus | LR wall, ms | Template wall, ms | LR total CPU, ms | Template total CPU, ms |
|---|---:|---:|---:|---:|
| static / panel20 | 1539.094 | 1611.306 | 3552.124 | 3632.212 |
| static / js | 3921.781 | 4014.745 | 8036.141 | 8133.218 |
| o2 / panel20 | 551.934 | 649.500 | 1270.114 | 1491.667 |
| o2 / js | 311.285 | 358.886 | 932.365 | 1098.989 |

The integrated build avoids a discarded LR serialization, reuses fresh
compiler-owned templates without extending scratch lifetimes, overlaps O2
template preparation with vocabulary partitioning, and omits unused NWA
skeleton outputs. Those changes remove measured work; they do not make the
complete template frontend uniformly cheaper than LR.

## Load latency and total CPU remain visible

The final load comparison uses 504 separate processes and 2,016 load samples
on identical frozen compatible artifacts. There is no grammar compilation in
the timed process. File reading, destruction and post-load validation are
outside the load timer; the API's input handling and preparation remain inside.
The first load is separate from the three repeats within that process.

| JavaScript static load, selected executable | LR wall, ms | Template wall, ms | LR total CPU, ms | Template total CPU, ms |
|---|---:|---:|---:|---:|
| first | 4.646 | 6.712 | 8.371 | 12.672 |
| repeat | 2.595 | 4.287 | 4.966 | 8.505 |

The added prepared-shift plan changes summed template repeat-load wall time by
-0.017% versus the preceding integrated version
(exploratory paired 95% interval -3.16% to
+1.78%). The JavaScript first-load total CPU
has a small adverse observation of +1.73% versus that
version, retained in the load evidence. No “free preparation” claim is made.

Parallel preparation can reduce requesting-thread CPU without reducing total
CPU or wall latency. All three clocks are retained; requesting-thread parity
must not be presented as whole-load parity.

## Serialized size

The fresh per-case size exporter is rebuilt against the selected source and
checks actual section boundaries, backend identity and supported initial
mask/completion round trips. It saves only lengths and digests, not full mask
or artifact dumps. The prepared pure-push plan is derived runtime metadata;
it adds no wire field. Its per-template limit is 512 READ entries, not a claim
of a 512-entry global memory budget.

The following whole-artifact totals are median sizes from the fresh timing
processes. The separate one-pass per-case export is preserved in the CSV;
minor non-parser representation differences between independent compilations
are not silently substituted between these measurements.

| Mode / corpus | LR artifact bytes | Template artifact bytes | Change |
|---|---:|---:|---:|
| static / panel20 | 59,328,384 | 55,896,504 | -5.78% |
| static / js | 19,248,191 | 18,893,392 | -1.84% |
| o2 / panel20 | 105,144,622 | 78,090,192 | -25.73% |
| o2 / js | 6,350,694 | 5,132,900 | -19.18% |

Actual stored parser fields from the selected-source size export:

| Mode / corpus | LR self-contained parser bytes | Template program bytes | LR compact external-transfer parser bytes |
|---|---:|---:|---:|
| static / panel20 | 2,702,141 | 372,087 | not exposed by this API |
| static / js | 266,366 | 180,528 | not exposed by this API |
| o2 / panel20 | 3,351,181 | 372,038 | 2,398,939 |
| o2 / js | 342,864 | 180,528 | 274,458 |

The O2 LR self-contained table is measured as the actual consumed bincode
field, not estimated from heap size or a different encoding. Its external
transfer representation is listed separately. Whole O2 artifact differences
also include vocabulary/container representation, not just removal of the table.

External-vocabulary artifacts are not uniformly smaller. The JavaScript O2 artifact
is 2,168,104 bytes for LR and 2,300,104 for templates (+6.09%). Its
self-contained template artifact is smaller. Two other O2 external artifacts,
`Github_hard---o77317` and `Github_hard---o12335`, are also slightly larger
(about +0.49% and +0.55%, respectively). All are retained in the per-case CSV.
Unsupported static-LR external
saving is labeled unsupported, not replaced by an invented comparison.

## What is implemented

When explicitly selected, the backend physically omits the LR table. Existing
shared mask and commit engines call template-derived advancement and exact
admissibility; unsupported composition requests return an error rather than
keeping a hidden LR fallback. Built-in grammar compilation can still use an
LR table transiently to derive the program. A data-only `ParserProgram` supplies
acyclic POP/READ/PUSH and completion relations without writing a parser callback.

Exact admissibility is the input domain of the advancement relation: a terminal
is admissible precisely when its relation has an output on the current stack.
Top-state possible/unconditional certificates bound that query; they are not
unproved assumptions about an LALR reduction eventually shifting.

For the new fast path, complete validated graphs prove a single push conditioned
on a known top, with no competing POP, DEFAULT, acceptance or epsilon output.
The plan is derived once and queried by the shared GSS operation. The same
operation acts directly on unit-annotation mask stacks because erasing stack
annotations commutes with a pure push. Other relations, unknown tops and larger
plans retain the complete interpreter. Tests include nonzero start states,
DEFAULT shadowing, shared floors, correlated annotations, invalid unreachable
graphs, the size limit, and non-vacuous execution through the real wrappers.

## Correctness and methodology

The selected source passed **1,025 root tests**, with **51 pre-existing ignored
tests explicitly retained**, **88 selected registered integration tests**, and
**51 Python tests** with no failures, errors or skips. The eight targeted
shortcut tests are a subset of the root count, not eight additional root tests.
The Python extension was rebuilt against the actual selected root in a private
stage; no old installed extension was used as evidence.

Full-vocabulary and completion comparisons cover 146,916 representations of
the selected panel/JavaScript prefixes plus 174,294 in an independent broader
100-schema sample: **321,210 checks** across fresh, self-contained reload and
external-vocabulary reload. In the broader sample, 96 schemas are supported in
each mode; four frontend/build failures were independently reproduced with
both backends, not counted as candidate successes.

Timing uses Apple M1 Pro / macOS 27.0, the pinned release Rust build and mimalloc,
four compile threads, and calling-thread CPU for runtime. Eight conditions each
have eight blocks and four counterbalanced arms: preceding LR/template and
selected template/LR. Both use consumed inputs with zero extra setup clones.
The 20-schema panel deliberately samples build and static/O2 runtime tails from
the earlier large JSB sweep; it is not a population-weighted estimate of all JSB.
JavaScript is the grammar behind `make example-js`, not a new approximation.

All 256 lossless native traces are retained. Two independent audits verify
compressed/uncompressed hashes, actual backend flags, replay identity and
12,469,248 integer `commit + mask = TBM` sums. The NumPy audit independently
recomputes 16,128 summary cells. Bootstrap intervals resample the eight paired
blocks; they are exploratory and not adjusted for many correlated metrics.
No per-token minima, discarded cold replay, or removed first warm replay is used.

Repeated test runs are not counted as new independent tests. Earlier rejected
decoder and local-shift experiments remain in the project evidence, but are
not mixed into this final matrix. One broader-check invocation guessed the
wrong executable filename and failed before native execution; its log is kept,
and the successful runner reads the exact binary from its signed-off manifest.

## Decision and boundaries

The selected branch is suitable for explicit standalone template-backend use
under the clarified criterion of substantial overall performance rather than
winning every individual step. Cold gains are significant in all measured
conditions, with an explicit remaining O2 schema percentage gap. There is no
claim that startup costs, every percentile or every unsupported grammar are
uniformly better. The LR default therefore remains unchanged.

The separate compiled-component composition work is not merged or represented
as qualified by these results. No package-registry release or version tag is
created by this feature-branch publication.

## Inspectable evidence

The companion archive contains all raw timing traces, original condition
manifests, per-process confidence intervals, per-case/runtime/size CSVs, initial
mask accounting, load samples, exactness/test receipts and source/dependency
hashes. Run `python3 verify_evidence.py` after extraction to verify checksums,
all raw TBM identities and recompute the headline gains with the standard library.
Rerunning the native benchmark additionally requires the identified CFA inputs
and pinned compiler/dependencies; the archive does not include binary libraries,
font files, full vocabulary mask dumps or unrelated private project notes.

Native SHA-256: `f9f5964d1dfe70d90526e84acf365fb30d965e05cb9a4f969356b2cb48ff28d4`.  
Root library SHA-256: `5b9912872ceb3b8516a73a305adc2e34f34cd450cc5d195ccd30c89346e95c34`.  
Python extension SHA-256: `9317c97da509261be9ec3407010c7df9f9391c0f613776837bcb668bfc7b2b05`.
