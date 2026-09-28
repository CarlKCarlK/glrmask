# JavaScript and JSONSchemaBench keeper checkpoint

This checkpoint integrates separately validated compiler/runtime improvements
without promoting the experimental dynamic runtime, LALR construction, parser
NWA quotient, or weight-ID representation. The accepted source is
`d3cfedd7a657ef08bc2212492bc82c9b54a5435b`, including the concurrent boundary
publication through `77511c7c3`. The original comparison source is `7e84a64c9`.
The Linux full-corpus comparison uses the earlier immutable keeper checkpoint
`ef353bf77`, before that final boundary-runtime merge; its scope is stated
separately from the final combined-source Mac results.

## What is included

Static masking limits speculative concrete-stack expansion per graph, preserving
shared-graph evaluation for wide ambiguity. Deterministic reductions share their
output prefixes; single-top commits avoid repeating an unsuccessful fast-path
attempt; a path stops once its entire remaining candidate set has been accepted.

Static compilation reduces parser-key, row, and final-signature allocations,
reuses token remaps, and constructs large two-parser unions directly. Memoized
token intersections use one sorted-interval pass. The latter preserves the
previous operand representative, including reversed commutative cache hits:
equal set values alone were insufficient to preserve downstream graph sharing
and tail performance.

Dynamic compilation overlaps lexer factoring with parser-table construction.
Shared-regex discovery traverses the DAG rather than its exponentially expanded
tree, and choice rewrites retain already-factored children. Master-trie admission
compares against the original vocabulary trie before replacing it.

Two artifact fixes preserve packed compressed tokenizer segments in fallback
serialization and use retained compact liveness tables in loaded slice proofs.
Artifacts written by older affected serialization code must be rebuilt from
their grammar; a reader cannot recover transitions absent from their bytes.

## Measurement contract

Build comparisons include only schemas for which the retained llguidance run
successfully built. Runtime scoring includes only the same problem, example,
and token position with a finite llguidance measurement. GLRMask failures are
retained and checked, not silently removed to improve the distribution.

JavaScript uses the unchanged 31-example grammar/corpus and 4,099 measured
positions. Full GLRMask masks and outcomes are compared with original GLRMask,
not llguidance's different JavaScript grammar. Off-corpus correctness checks are
not represented as performance samples.

Mac JavaScript timings use the CFA convention `mask[i] + commit[i]`, measured
with thread CPU time. Each side has two independent processes in ABBA order,
three fresh loaded-state traversals per example per process, and three native
builds per process. Quantiles use the minimum **complete pair** at each position
across those six traversals. Build medians use the six retained build samples.
Raw maxima are also retained and must not be confused with stabilized P100.
The exact dynamic generation-mask cache is disabled on both sides to expose
recomputation; the ordinary static cache policy is unchanged.

Native JSONSchemaBench timings instead measure `commit[i] + next_mask[i+1]`.
These are separate metrics/protocols; their absolute quantiles are not merged
with the Python/CFA measurements. The paired native driver uses ABBA/BAAB chunks,
thread CPU runtime clocks, wall-clock builds, and explicit JSON lexical policy.

## Merged-source Mac JavaScript results

The merged-source Python extension has SHA-256
`10ced0d38c92ab2399c02d4de4792ed5b03cf9c150a54d9f3017b6a41902db80`.
Both engines matched all 4,099 full masks and final outcomes in every run.

| Metric | Original | Integrated |
| --- | ---: | ---: |
| Static build median | 5,702.500 ms | 3,720.739 ms |
| Static TBM P50 | 19.584 us | 19.917 us |
| Static TBM P90 | 32.458 us | 30.675 us |
| Static TBM P99 | 59.545 us | 49.296 us |
| Static TBM P99.9 | 117.356 us | 73.715 us |
| Static TBM P100 | 184.834 us | 80.667 us |
| Static raw maximum | 339.542 us | 259.625 us |
| Dynamic build median | 221.485 ms | 157.132 ms |
| Dynamic TBM P50 | 5.050 ms | 5.051 ms |
| Dynamic TBM P90 | 7.552 ms | 7.537 ms |
| Dynamic TBM P99 | 12.252 ms | 12.304 ms |
| Dynamic TBM P99.9 | 22.225 ms | 16.606 ms |
| Dynamic TBM P100 | 23.363 ms | 17.254 ms |
| Dynamic raw maximum | 24.285 ms | 20.979 ms |

These are observed sample statistics, not worst-case guarantees. Static cold
first-use latency remains. The integrated dynamic engine still does not meet
the goal of beating llguidance's JavaScript distribution. Experimental results
around 20 ms build and 4--5 ms TBM P100 require additional modes and are **not**
the defaults in this checkpoint.

## Full llguidance-supported native JSONSchemaBench comparison

The immutable Linux/CX33 keeper checkpoint `ef353bf77` was compared against
`7e84a64c9` over **all 8,332 schemas** that built in the retained llguidance run.
Both versions successfully built 8,326. The same six unsupported cases failed
in both versions; their failures remain in the raw evidence. All 42 chunk
checks agreed on build status, semantic outcomes, and exact runtime-key
coverage. There are **2,925,984 matching llguidance-measured TBM intervals**.

These are two independent native runs per side, interleaved ABBA/BAAB in
200-schema chunks. Build and TBM quantiles below use per-case/per-interval
minima of those two runs; raw maxima are stated separately. The final boundary
merge after this checkpoint was validated with the combined-source Mac gates
above, not silently treated as the same Linux binary.

| Native static metric | Original | Keeper checkpoint |
| --- | ---: | ---: |
| Build P50 | 15.225 ms | 14.958 ms |
| Build P90 | 106.449 ms | 103.418 ms |
| Build P99 | 412.105 ms | 406.161 ms |
| Build P100 | 2,161.282 ms | 2,495.022 ms |
| Total minimum-per-schema build | 388.657 s | 382.875 s |
| TBM P50 | 3.668 us | 3.637 us |
| TBM P90 | 7.544 us | 7.514 us |
| TBM P99 | 17.142 us | 17.072 us |
| TBM P99.9 | 27.833 us | 28.153 us |
| TBM P100 | 63.399 us | 65.143 us |
| Raw maximum TBM | 211.948 us | 434.856 us |

The build distribution improves modestly; this is not a blanket runtime win
or a claim that every schema improves. Stabilized TBM P100 is about 2.75%
higher in this run. The larger raw maximum and worst-build observations were
investigated, not omitted. Six additional interleaved fresh-process runs per
side on nine selected cases did not reproduce the apparent 15--22% worst-build
regressions: median builds for `o21135`, `o21136`, and `o21137` were respectively
2,385.21 / 2,440.09 / 2,355.79 ms before and
2,387.94 / 2,413.49 / 2,375.12 ms after. The `o81596` raw-latency spike did not
recur; its minimum-of-six maximum TBM was 19.056 versus 18.986 us. Selected-case
reruns do not replace or rewrite the full-run figures above.

A separate `o21136` candidate process in that nine-case rerun showed a 1,658.902 us
burst at example 1, step 786; the other five repetitions at the same position
took 3.277--5.040 us. Eight further interleaved processes on that schema did not
reproduce it: whole-example raw maxima were 41.288--59.782 us for the candidate
and 45.966--60.995 us for the control. The affected candidate position took
3.206--3.416 us in those four extra repetitions. The burst is retained in the
machine-readable evidence; its cause is not established, and these measurements
do not establish a hard raw-latency cap.

The unchanged build-failure IDs are `Github_easy---o21053`,
`Github_hard---o13029`, `Github_medium---o60170`,
`Github_medium---o69991`, `JsonSchemaStore---meta.schema`, and
`Kubernetes---kb_1161_Normalized`. The exact error messages are retained in each
chunk check. In particular, `o21053` retains its length constraint and reports
the exact-product structural limit rather than weakening the grammar.

## Combined-source correctness and test gates

The complete release workspace and integration suite passed **2,216 Rust
tests**, with zero failures and 54 ignored tests. The root library contributes
923 passing tests; nested child-process runs and repeated test invocations are
not counted again. All **42 Python tests** passed separately.

The 126-schema full-mask guard checked 77,582 masks plus outcome markers in each
of freshly compiled and reloaded form. Both streams exactly match the original
compiled reference: 1,245,309,451 decoded bytes, SHA-256
`351dab2120aadea63c7515ff1c12630fd8857184817eb0f87dfaa1e30932ece2`.
This validates the packed-tokenizer and compact-proof fixes on real cases,
including the previously broken loaded constraints, rather than only synthetic
unit tests.

Parallel testing exposed process-environment interference: strict-static tests
temporarily set a global trap while unrelated tests legitimately evaluate
dynamic reference shards. A mutex used only by writers does not protect those
readers. The tests that change this flag now run as exact, isolated subprocess
tests; the parent verifies both a successful exit and that exactly one test ran.
All original trap assertions remain, including custom-stack workers. This is
test-only code: no production trap or runtime semantics were weakened. The final
workspace/integration gate passed with eight test threads.

## Benchmark-policy incident and resolution

An apparent Linux regression in `o60179` and an out-of-memory build in `o21142`
were traced to executable-name-dependent JSON policy. The historical importer
treats filenames containing `integration` or `test` as test executables unless
`GLRMASK_LLGUIDANCE_COMPAT` is explicit. Renaming the candidate to
`integration-clean-native` therefore selected a different grammar policy from
`baseline-native` and `diagnostic-native`.

With `GLRMASK_LLGUIDANCE_COMPAT=1`, the same frozen baseline and candidate
binaries agree on all six outcomes of each case. The `o21142` builds take about
1.445 seconds on both sides, with peak resident memory about 1.53 and 1.60 GB,
rather than the failing 7 GB run. With policy `0`, the original baseline also
reproduces the early `o60179` rejection. That older full-JSON behavior is a
separate issue, not an improvement or a new regression in this change set.

The corrected paired driver records policy `1` in both manifest flag maps.
Setting only the shell environment is insufficient because the driver clears
inherited `GLRMASK_` variables before each subprocess. The corrected driver
SHA-256 is
`8639bb648e135c4d6c9adedd45fbb4dfaeb9a6b8c9fed42ca0e9cc205834bdf2`.
The invalid earlier comparison and provenance diagnostics are retained.

## Reproduction and evidence

Use Rust 1.95.0, a release build with native CPU code generation, an isolated
Cargo target, and `MIMALLOC_PURGE_DELAY=-1`. The Mac Python extension was built
without stripping because the local platform rejects some stripped extensions.
Neither timed side enables LALR or experimental runtime flags.

The local continuation archive is
`.benchmarks/integration/` in the integration worktree. It retains immutable
packages, exact source archives, input hashes, scripts, raw JSON measurements,
all test logs, and the policy-incident reproducers. The live handoff is
`glrmask-integration-345436e.md` in the compaction notes. The adjacent machine-
readable checkpoint summary records the final test and native-corpus results.
