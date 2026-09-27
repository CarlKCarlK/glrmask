# Ordered parallel evaluation of native boundary parser rows

## Scope and selection

This change accelerates the first weighted subset construction in
`crates/glrmask-parser-dwa/src/parser_dwa.rs`. It does not identify LR states,
merge source NWA nodes, omit guards, change component boundaries, or alter the
parser's `DEFAULT` rules. The existing possible-read calculation, default
consolidation, final subtraction, fallback determinization and final minimizer
remain unchanged.

The policy `GLRMASK_BOUNDARY_PARALLEL_NATIVE_ROWS` is enabled by default;
`0`, `false`, `off`, or `no` retains serial evaluation. Automatic selection
requires finite output, at least four workers in the caller's existing Rayon
pool, 4,096-200,000 source states, and an alphabet no larger than 32,768.
At least 64 pending frontiers are required for a parallel batch. It does not
resize the pool or alter affinity. One-, two- and three-worker pools and small
or unsupported inputs use the existing serial loop.

The four-worker floor is empirical: the two-worker experiment regressed, while
four, six and ten workers improved the selected10 fixtures. This is a guarded
optimization, not a claim that arbitrary parallelism always helps.

## Mathematical decomposition

Let a native coefficient be a subset of the finite coordinate universe U.
Addition is union and multiplication is intersection. A deterministic frontier
is an ordered vector x of original NWA nodes q and coefficients x(q). For an
input label a, gather the destination coefficients

    y_a(r) = union over q of (x(q) intersect A_a(q,r)).

Apply the same weighted epsilon closure C used by the serial implementation.
For each epsilon edge q -> r of weight w, its fixed-point transfer is

    C(y)(r) |= C(y)(q) intersect w.

The row's final coefficient is

    F(x) = union over q of (x(q) intersect final(q)).

These calculations depend only on the established frontier and immutable source
NWA. Different pending rows can therefore be evaluated independently. Assigning
canonical deterministic state IDs is a different operation and remains ordered.

Importantly, this implementation reproduces the serial algorithm's *specific*
coefficient placement. An epsilon-free singleton uses its incoming coefficient
on the edge and an ALL-normalized singleton target. A singleton resulting from
closure uses that member's residual coefficient. A multiple-member closure
keeps its exact coefficient vector and uses the union of incoming coefficients
on the edge. It does not factor a uniform multi-member frontier into an
ALL-frontier; that would be a different transformation.

## Implementation

`finite_parallel_rows.rs` drains at most 4,096 already-established pending rows
from the FIFO. Consecutive chunks of 256 rows are evaluated in Rayon workers.
The immutable snapshot includes the source NWA, existing singleton-state IDs,
completed closure caches, the coefficient value interner and its Boolean caches.

Each worker has a private coefficient overlay, epsilon scratch and local
closure cache. IDs below 2^31 refer to the immutable base; IDs with the high
bit set refer to the packet's exact new values. Bitvector equality is checked
by the hash maps' complete equality operation: a hash collision cannot merge
different coefficients. The finite coordinate width and padding are unchanged.
The worker first consults the base interner and caches, then its local overlay.
No shared mutable interner, graph registry or lock is accessed by workers.

Workers produce the final ordered transition vectors. Already-established
targets are filled immediately. Only unresolved singleton IDs and closure
recipes have deferred slots. Results own their storage and cannot reference
recycled scratch.

Packets are published in original FIFO order. Local coefficients are interned
into the global value table; deferred slots are resolved in original label
order. A closure-cache lookup is repeated during publication because earlier
packets may already have resolved the same incoming frontier. New target IDs
and worklist entries are assigned by the unchanged registration functions.
Completed vectors are moved into their output rows, not rebuilt edge by edge.

## Why graph identity is preserved

Induct on the processed FIFO prefix. Assume all registered frontier values,
output IDs, supports and pending rows agree with serial execution, modulo an
injective renaming of exact coefficient IDs. Worker Boolean operations and
epsilon closure calculate the same values. Processing packet results in the
same row/label order gives the same cache reuse and first-discovery decisions.
Singleton normalization is unchanged. Thus every new deterministic ID, support
vector, target and coefficient value agrees, establishing the next induction
step. The unmodified `DEFAULT` and fallback stages receive the same identities,
not merely a language-equivalent quotient.

Workers may compute temporary coefficients whose serial calculation would
have been avoided by a preceding row's cache insertion. Unused coefficient
history is therefore not the proof invariant. The invariants are complete
native row identity, exact decoded coefficients and the original-coordinate
decoder. The realistic fixtures also reproduce the finalized artifact bytes.

Resource exhaustion is not evidence that a grammar path is impossible. A
packet exceeding 8,192 private coefficients, 250,000 retained frontier members,
65,536 transitions, or four million counted preparation operations declines
the entire private finite attempt. Global interning and graph limits remain
checked. No partial graph or unwind-aid empty coefficient is published as a
successful result; the caller retains its existing exact fallback. The global
import-failure test explicitly checks that no output row is published.

## Validation and measurements

The focused source passes 69 ordinary parser-crate tests, including Boolean
overlay/reference comparisons at 1/2/17/44/64 words, 256 generated weighted
graphs with epsilon and explicit zero/default guards, wide convergence,
state-limit decline, import-limit decline and automatic-policy eligibility.
The ordinary root suite passes 953 tests, with zero failures and 51 existing
ignored tests. The parser suite has zero failures and two existing ignored
tests. The no-default-features public-library check and release composition
builder/loaded-probe builds also pass. Complete provenance and raw timings are
in `docs/performance/boundary-parallel-normalization-20260927.json`.

The final focused-source validation comprises 592 interleaved timed builds:
192 at ten workers, 96 at four workers, 48 in the first two-worker screen,
and 128 each in the one- and two-worker confirmation runs. Every candidate
artifact matches its independent same-thread reference. The complete native
topology, coefficient values and original-coordinate decoder agree, and
loaded mask signatures match at all 22 anchors. Source and executable hashes
were rechecked against the reports before publication.

| Workers | Fixture | Serial median (ms) | Default median (ms) | Median paired saving (ms) | Faster pairs |
| --- | --- | ---: | ---: | ---: | ---: |
| 10 | Current | 269.6555 | 265.6540 | 4.1550 | 21/24 |
| 10 | Legacy | 274.2835 | 268.8670 | 4.8010 | 21/24 |
| 4 | Current | 271.4920 | 265.1335 | 5.5090 | 10/12 |
| 4 | Legacy | 276.6705 | 269.4155 | 6.4330 | 10/12 |

These are same-binary explicit-serial versus production-default comparisons;
paired medians need not equal the difference between marginal medians. The
separate identical serial control also confirms the larger-worker gains.
At one and two workers, default and override execute the same serial loop.
Their measured small positive/negative timing differences vary by fixture
and reference, so no low-core acceleration or zero-variance claim is made.
The earlier forced two-worker parallel regression is why that mode is not
selected. Do not add the savings from separate cohorts.

Before publication, V2 was checked across repeated current/legacy selected10
fixture comparisons. Its original 96-build screen gave same-binary paired
whole-composition savings of 4.6375 ms current and 7.370 ms legacy. A separate
144-build six-arm screen replicated roughly 6.4-6.6 ms for V2-style publication.
These are different cohorts, not additive gains. The 256-build reduced-worker
matrix retained exact same-thread artifact bytes and native topology. The
one-thread published reference has a different serialized artifact from its
ten-thread output, but its native topology and loaded masks match; comparisons
therefore freeze a same-thread independent reference before testing candidates.

The bulk-registration V3 variant is deliberately excluded. Although exact on
the generated and realistic checks, its sorting and batch bookkeeping cost
more than the limited duplicate-request population saved. Its code and
negative measurements remain on an experimental branch.

Reproduce from frozen binaries and the two prepared fixture directories:

```text
cargo build --release --features internal-api --example compare_boundary_native_rows

python scripts/compare_parallel_native_rows.py \
  --baseline PATH_TO_INDEPENDENT_SERIAL_BUILDER \
  --candidate PATH_TO_CANDIDATE_BUILDER \
  --runtime PATH_TO_COMPOSITION_LOADED_STATIC_PROBE \
  --checker PATH_TO_COMPARE_NATIVE_ROWS \
  --current PATH_TO_CURRENT_PREPARED_COMPONENTS \
  --legacy PATH_TO_LEGACY_PREPARED_COMPONENTS \
  --output NEW_OUTPUT_DIRECTORY --pairs 24 --threads 10
```

The checker must compare every native state, ordered label, target ID, final
and edge coefficient value, and original-coordinate decoder. Its source is
included in `examples/compare_boundary_native_rows.rs`, with tests proving it
detects deleted zero guards, changed targets, coefficient corruption, reordered
edges and decoder changes. The repository-built checker also rechecks all 30
final candidate/reference native pairs independently of the historical frozen
checker used in the timing harness. Every timed
artifact is compared against the independent baseline's SHA256; untimed
validation additionally checks native rows and loaded masks at all 22 anchors.
The runner records source/executable/input hashes, flags, raw paired timings,
loaded results and profiles. Completed artifacts are losslessly archived.

Reported `compose_ms` includes all worker setup, speculative computation,
ordered publication and disposal inside composition. It excludes artifact
serialization, fixture loading outside the composer, and initial compilation
of individual constraints. Current whole-composition latency is still hundreds
of milliseconds; no 50-100 ms whole-build result is claimed.
