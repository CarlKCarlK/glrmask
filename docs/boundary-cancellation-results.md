# Packed cancellation memo results

## Scope and default

`GLRMASK_BOUNDARY_PACKED_CANCELLATION_RESULTS` selects compact storage of
completed cancellation-query answers. It defaults on. Set it to `0`, `false`,
`no`, or `off` for the historical vector-of-vectors representation.

This is a private compile-time representation change. It changes neither the
runtime artifact format nor LR state IDs, template expansion semantics,
explicit guards, `DEFAULT`, or the chosen cancellation recurrence. It does not
use the rejected direct-return shortcut or the experimental first-query index.

## Why common answers can fit in the memo table itself

The cancellation solver computes a weighted relation `R(q,a)`: positions after
the next matching read of LR symbol `a`, allowing epsilon moves and balanced
push/read excursions. Each answer is a sorted list of `(target, weight_id)`.
The historical map stores `(q,a) -> result_id` and keeps a separate vector for
every completed query. On the measured selected10 boundary, the 97,218 answers
contain 80,194 empty lists, 14,849 singletons and only 2,175 multi-target lists.

The new map still has exactly the same `(q,a)` keys. Its value is a 64-bit
handle, with three disjoint encodings:

```
empty:       0
singleton:   (target << 32) | nonzero_weight_id
multiple:    (1 << 63) | (length << 32) | pair_arena_offset
```

The singleton target is below `2^31`; the weight is nonzero. Therefore its
high bit is clear, and even a singleton at target zero differs from empty.
The multi-list length is at least two and below `2^31`. Offset and length
are validated before appending. Existing finite-compiler limits are much
tighter: 200,000 nodes, 500,000 weights, and 4,000,000 result pairs.

Only multiple-target answers append sorted pairs to an immutable arena.
Existing handles remain valid when that arena reallocates because they store
offsets, not pointers. The storage layer does not own or reinterpret weights.

## Correctness argument

Let `decode(h)` return the ordered list represented by a handle. The invariant
is that, for every completed query, `decode(memo[q,a])` equals the historical
ordered result. It holds for direct read branches. When an epsilon or balanced
excursion consumes a previously completed answer, both solvers visit the same
targets in the same order and perform the same intersections and unions.
Induction over query completion establishes the invariant for every query.

Sorting, logical pair counts, work accounting, read filters, memo-key equality,
and the finite budget checks are retained. Encoding happens only after those
checks. A failed encoding returns failure to the private compilation, never
an empty answer pretending to prove impossibility. No input graph is modified
by the result-storage implementation.

Unlike a state quotient, this proof preserves the exact derived graph and
the numeric weight interning sequence, not merely a weighted language. The
later guard- and target-identity-sensitive normalizer receives the same data.

## Tests and measurements

Storage tests cover empty versus target-zero singleton answers, full-width
weights, bounds/overflow, append failure atomicity, and retained handles after
arena growth. Generated signed DAGs with duplicate branches, empty guards,
`DEFAULT`, multiple coordinate widths, both read-filter modes, and restricted
work budgets compare derived rows, counters and the complete weight table.
Nested epsilon chains and shared joins exercise empty and nonempty memo hits.

The development screen passed 64 parser tests, with no failures and two
existing ignored tests, in both modes. Its 416 timed links across two fixture
generations had byte-identical artifacts; native rows and loaded masks matched.
The separate 128 scan-free resolution profiles showed approximately 1.5–2 ms
saved versus the independently compiled main reference. Whole-link paired
gains were smaller/noisier: about 1–3 ms, not a transformative improvement.

Measured result-buffer capacity fell from 3,347,656 bytes to 98,304 bytes.
This excludes the unchanged memo map and all other compiler allocations;
it is not a measurement of process resident memory. See the accompanying
performance JSON for final publication-source gates and raw timing provenance.

### Final clean-source validation

The default-on publication source passed 64 parser tests (0 failures,
2 ignored), 951 root tests (0 failures, 51 ignored), the ordinary public-library
check with no default features, and release builder/runtime-probe builds.
The final 192 interleaved whole-link builds retained byte-identical artifacts,
native rows and loaded masks on both fixture generations.

| Fixture | Baseline whole-link median | Default median | Median paired saving |
| --- | ---: | ---: | ---: |
| Current | 272.530 ms | 272.4175 ms | 0.459 ms, 12/24 wins |
| Legacy | 276.591 ms | 273.8825 ms | 2.203 ms, 16/24 wins |

The current whole-link gain is within run-to-run variation. Do not present
this result as a dependable multi-millisecond whole-link win on every fixture.
The reproducible benefit is the allocation reduction and the isolated
resolution phase: a separate 160-run scan-free profile measured median paired
resolution savings of 1.45435 ms current (19/20 wins) and 1.62335 ms legacy
(17/20) versus independently compiled main. Against the same-binary historical
override, savings were 1.71825 ms (20/20) and 1.50755 ms (17/20).

These resolution timers include finality propagation, certified-context
transfer, positive materialization and disposal. They exclude the native
diagnostic row histogram scan, and are not added to whole-link savings.

## Reproduction

Build the candidate with ordinary Cargo:

```sh
cargo test --release -p glrmask-parser-dwa --features internal-api --lib -- --test-threads=1
cargo test --release -p glrmask --features internal-api --lib -- --test-threads=1
cargo build --release -p glrmask --features internal-api --example composition_build_static_artifact --example composition_loaded_static_probe
```

Preserve an independently built baseline executable before changing source.
Each prepared fixture directory contains `core.bin`, `dispatch-literal.bin`,
and `vocab_dump.bin`. Use `scripts/compare_packed_cancellation_results.py` with
explicit `--baseline`, `--candidate`, `--runtime`, `--checker`, `--current`,
`--legacy`, and a new `--output` directory. The checker compares the saved
native pre-minimizer rows in original coordinates. `--pairs` controls the
number of interleaved rounds; the default run compares baseline, same-binary
historical override, an identical control, and the actual new default.

The script records source/binary/input hashes and every raw timing, checks
every artifact against the independent baseline, separates validation from
timing, checks loaded masks, and losslessly archives the final temporary
artifact instead of accumulating one large file per repetition.
