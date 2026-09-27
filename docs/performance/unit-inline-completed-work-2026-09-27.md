# Stop unit-inlining work after a proven budget abort

This change removes work from an unsuccessful compiler optimization pass. It
does not reduce a work limit, choose a different parser construction, change a
grammar, or change the mask/commit implementation.

The final production policy enables this only for the exact core-merged table
builder. Legacy and LALR construction keep their existing schedule. No user
opt-in is required. The internal diagnostic
`GLRMASK_UNIT_INLINE_COMPLETED_WORK_ABORT=0` selects reference scheduling for
core-merged tables.

## Why early termination is exact

Unit-inlining analyzes candidate parser states in parallel using independent
child budgets. The serial phase charges each child's completed work to the
parent budget, including work from locally exhausted children. Updates from
previous iterations are covered by the existing undo journal.

The new counter is a lower bound: parent work already charged, plus visits from
**completed** child jobs. Only after this lower bound exceeds the unchanged
global limit may not-yet-started jobs be skipped. At that point the reference
fold necessarily aborts too. In-flight jobs finish; all actual completed work is
accounted for; the same journal rolls back the optimization. No partial child
estimate or wall-clock prediction is used as the certificate.

Successful passes retain the same indexed result order and serial processing.
The counter is bypassed for the uncapped correctness oracle. Core-merged
phases whose conservative maximum work cannot reach the global limit also
bypass it. An atomic operation is performed per completed job, not per visit.

Tests cover strict-cap boundaries, saturating arithmetic, simultaneous locally
exhausted jobs, and complete serialized-table equality across one/four worker
threads, both supported table constructions, successful optimizations and
aborts. The rollback tests include a pre-existing mutation from an earlier
iteration, not merely an unmodified current phase.

## Measurement protocol

The initial experimental comparison below allowed either table construction to
use the certificate. That broad policy was **not retained**. It uses the same
optimized Apple Silicon binary with the certificate disabled/enabled in
alternating ABBA order. Builds use elapsed
wall time; runtime uses thread CPU time. Vocabulary-only preparation is outside
build timing. JavaScript is the unchanged 31-example capability corpus, with
4,099 measured positions. Every scored position has a llguidance measurement.

The Python CFA convention here is mask plus commitment at the same position.
Runtime quantiles are minima of **complete** intervals across six fresh-state
repetitions, not sums of independently minimized mask/commit columns. Native
JSB uses the distinct commitment-to-next-mask interval. Those two conventions
are reported separately and must not be spliced into one distribution.

## JavaScript build and runtime results

An initial build-only batch of 16 builds per mode gave dynamic-build medians
130.253 -> 119.510 ms. All 32 serialized dynamic artifacts were byte-identical.
A subsequent full-corpus batch, six builds per mode, gave 137.638 -> 119.329 ms.
Thus the observed dynamic-build reduction was approximately 8-13% across the
two batches. This is not a claim that every individual build improves.

| Full-corpus metric | Reference | Certificate |
| --- | ---: | ---: |
| Dynamic build median, ms | 137.638 | 119.329 |
| Dynamic TBM P90, ms | 6.5863 | 6.5785 |
| Dynamic TBM P99, ms | 10.7125 | 10.6996 |
| Dynamic stabilized TBM P100, ms | 16.6301 | 16.3890 |
| Dynamic raw maximum, ms | 17.6842 | 18.3022 |
| Static build median, ms | 3,427.475 | 3,466.522 |
| Static TBM P90, microseconds | 27.625 | 27.592 |
| Static TBM P99, microseconds | 46.667 | 46.293 |
| Static stabilized TBM P100, microseconds | 76.916 | 80.291 |

All 4,099 full masks agree in each execution mode. These results do not
establish a runtime speedup or a raw-tail improvement. The small static maximum
increase was checked at six tail positions with 40 fresh-state measurements
per mode. The worst targeted median was 80.917 -> 79.917 microseconds; the other
median changes ranged from approximately -4.2% to +2.2%. No systematic static
tail increase was established by that follow-up.

Instrumentation attributes the dynamic build reduction to the discarded
unit-inlining phase: approximately 48-50 -> 28-31 ms. Instrumented numbers are
diagnostic only; the table above uses uninstrumented builds.

## JSONSchemaBench guard and limitations

The native guard freezes 1,000 llguidance-supported schemas, rather than
selecting a different support intersection for each implementation. All builds
and semantic outcomes agree. Runtime scoring uses 367,640 exact
llguidance-measured intervals.

| Native JSB metric | Reference | Certificate |
| --- | ---: | ---: |
| Build P50, ms | 9.543 | 9.604 |
| Build P90, ms | 73.647 | 76.803 |
| Build P99, ms | 218.566 | 224.772 |
| Total minimum-per-schema build, ms | 26,872.597 | 26,983.793 |
| TBM P90, microseconds | 6.042 | 6.042 |
| TBM P99, microseconds | 12.917 | 12.875 |
| Stabilized TBM P100, microseconds | 44.250 | 43.042 |

The median paired build ratio is 1.00366, but P90/P99 in this batch increased
4.3%/2.8%. This is not a JSB build-speedup claim. Twenty selected cases were
rechecked with six builds per mode in alternating order; that targeted panel
must not be presented as a population distribution.

The ten rechecked cases around P90 ranged approximately -2.6% to +2.6%, while
some larger selected cases still varied by 5-10%. Their profiles use legacy
table construction and complete far below the budget, with no demonstrated
certificate benefit. The final core-only policy therefore leaves that path
unchanged rather than attributing every difference to noise. The final
verification also compares the default build to a frozen published baseline,
not merely two switches within the experimental binary.

On the 126-schema correctness guard, static compiled and loaded execution each
reproduce all 77,582 complete vocabulary masks and outcome markers from the
unchanged reference. Dynamic compiled and loaded execution likewise agree
between disabled/enabled modes at all 77,582 masks. The decoded streams include
outcomes, not just acceptance of the supplied next token.

## Reproducibility

The accompanying JSON report retains paired summaries, raw build/tail repeats,
binary and stream hashes, support-selection provenance, and test results.
Private raw artifacts are retained rather than committed as large binary files.
The JavaScript grammar, work budgets, and llguidance baseline were not changed.


## Final core-only default and publication gate

The final source was rebuilt with the diagnostic unset and compared against a
frozen binary from published revision `96d5abb74`. Both baseline binary hashes
were independently checked against that revision's committed evidence report.
The native runner Cargo lockfile is identical between those builds. This final
comparison, unlike the initial screen above, leaves legacy/LALR scheduling
unchanged and includes ordinary binary-layout differences.

Sixteen dynamic JavaScript builds per version in ABBA order gave medians
**146.795 -> 133.945 ms (8.75% lower)**. All 32 serialized dynamic
artifacts were byte-identical. A separate instrumented default invocation
confirmed that the certificate is actually reached without an opt-in flag.

The final 1,000-schema native guard again has no build or outcome differences
and scores the same 367,640 llguidance-measured intervals:

| Final default comparison | Published reference | Core-only default |
| --- | ---: | ---: |
| Build P50, ms | 9.604 | 9.579 |
| Build P90, ms | 76.985 | 73.799 |
| Build P99, ms | 223.991 | 224.267 |
| Total minimum-per-schema build, ms | 27003.739 | 27241.872 |
| TBM P90, microseconds | 6.000 | 6.042 |
| TBM P99, microseconds | 12.875 | 13.041 |
| Stabilized TBM P100, microseconds | 158.584 | 44.167 |

The median paired build ratio is 1.00302.
P90 improved in this batch, while P99 and total build were broadly flat. There
is no static build algorithm change and no claim of a broad static speedup.
The isolated reference maximum is not evidence of a causal runtime improvement.

Default static and dynamic JavaScript each reproduce all 4,099 reference masks.
The final static compiled/loaded and dynamic compiled/loaded guards each
reproduce all 77,582 full masks and outcome markers across 126 schemas.

The final test gate passed **2,225 unit/integration tests plus one documentation
test**, with zero failures and 55 ignored, and **42 Python tests**. Workspace
examples passed type checking; six existing unused-assignment warnings in the
vocabulary-partition benchmark example were not modified. The first all-target
link attempt ran out of disk space; the recorded rerun separates executable
unit/integration tests from example type checking to avoid retaining unnecessary
example binaries. This was an infrastructure failure, not a suppressed test.

Final Python SHA-256:
`273186f3f3b554b93ed7d3655bd13a2cdffc4898b8170e7181cdcec33f8d2318`.
Final native runner SHA-256:
`a2d6a6739657db6113983babcb98b23e39d0880048e855b68a96238c37ee1c6d`.
