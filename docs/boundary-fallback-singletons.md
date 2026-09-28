# Packed singleton bookkeeping in fallback determinization

## Scope

This optimization changes storage and lookups inside
`determinize_fast_boundary_with_fallbacks`, not the grammar, cancellation,
weights, or the meaning of `DEFAULT`. Its base is `f1b5ceaa0`, which already
includes the published frontier-allocation and final observation improvements.

The optimized storage is enabled by default. Set
`GLRMASK_BOUNDARY_PACKED_FALLBACK_SINGLETONS=0` to run the historical storage
for controlled comparisons. The shared boundary policy also recognizes empty,
`false`, `no` and `off` values, case-insensitively. This policy changes no public
API or serialized format.

## Why the duplicate singleton map entries can be removed

Fallback normalization registers one output state for each normalized
singleton input frontier `[(q, ALL)]`. The incoming coefficient remains on
the incoming edge. Every one-member target takes the singleton branch and
looks up `q` in `normalized_singletons`.

Only targets with more than one member query the general weighted-subset
map. Consequently that map's additional `[(q, ALL)]` keys cannot satisfy
any required lookup: they are redundant with the singleton map. This is a
property of the fallback registration branches, not a general claim that
singleton keys can be removed from arbitrary subset constructions.

The candidate therefore stores the singleton map as a dense `Vec<u32>` over
the already-bounded original input-state IDs, using `u32::MAX` for missing.
It omits singleton keys from the general subset map and stores pending
singleton frontiers inline in a `SmallVec` of capacity one. Multi-member
frontiers spill without truncation and retain exact ordered-slice hashing,
equality, and borrowed lookups.

## Preservation argument

Induct over the FIFO worklist. Initially both implementations register
input state zero as output state zero. Assuming identical work items up to
the current item, both enumerate exactly the same source/weight pairs and
perform the same final-weight and transition operations, in the same order.
For a singleton target, the dense and hash indexes contain the same mapping
because all previous insertions were identical. A new target gets the same
next output ID and queues the same singleton. No later multi-member lookup
can need the omitted singleton key. For a multi-member target the original
ordered key and general map operations are unchanged.

Therefore every output ID, row, final weight, transition coefficient, and
interned weight ID is preserved. Input state identities, explicit guards,
possible-read annotations and every `DEFAULT` observer see identical graphs.
The input remains immutable during construction. Budget failures still
abandon the private result; no partial graph is published.

This is distinct from earlier main-determinizer pending-frontier omission,
and from earlier fallback inline/dense/owned-row experiments that retained
the duplicate singleton keys. No owned transition-vector reuse is bundled
into this candidate.

## Coverage and evidence so far

The current Cargo parser-crate tests pass 53 tests with zero failures; two
diagnostics remain ignored. New coverage includes 512 generated weighted
DAGs with explicit zero transitions, `DEFAULT`, sparse labels, and multiple
coordinate rows. Every output row, entire interner value table, and work
counter is compared against the original implementation. Separate tests
cover index replacement, inline storage, spill and exact borrowed lookups.
The full root suite with the optimized path enabled passed 924 tests with zero
failures and 51 existing ignored tests.

A saved **post-fallback** graph replay (33,837 states / 301,384 edges) gives
identical output and interner tables. Three independent 60-sample replay
sessions measured roughly 6.88 ms reference versus 2.75 ms candidate.
This isolates bookkeeping: it is **not** the actual pre-fallback stage
input and must not be presented as a whole-link measurement.

The first actual whole-link screen rebuilt the current source and alternated
OFF, identical control and ON for 16 rounds on each of two prepared fixture
generations. All 96 timed artifacts and six separate profile artifacts were
byte-identical across modes. Current medians were 297.664 / 297.516 / 292.135 ms;
median paired savings were 6.770 ms against OFF and 4.617 ms against control.
Legacy medians were 303.827 / 300.272 / 299.805 ms, with paired savings of
3.799 / 3.437 ms. Individual runs are noisy, so a separate 24-round confirmation
was performed. All 144 additional timed artifacts were also byte-identical.

| Confirmation fixture | OFF median | Control median | Packed median | Paired OFF saving | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current | 297.867 ms | 297.126 ms | 290.722 ms | 6.891 ms | 20 / 24 |
| Legacy | 300.738 ms | 304.450 ms | 295.594 ms | 5.114 ms | 19 / 24 |

Against the identical controls, median paired savings were 5.104 ms current
and 5.845 ms legacy. The paired OFF/control differences were 0.306 ms and
-1.610 ms respectively. These are prepared-component link times, excluding
artifact serialization; they are not savings to add to earlier independent
experiments.

Separate actual pipeline profiles confirm the intended mechanism. Current
fallback caller time dropped from 7.456 to 3.370 ms, and legacy from 7.352 to
3.834 ms. Both processed the same 34,059 input states, yielding 33,837 output
states, with 33,251 ordinary singleton rows and 586 complex rows. Subsequent
minimization still yields exactly 1,231 states / 33,534 edges. Profile timings
are individual observations, not the estimator used for overall savings.

The final default-on source was rebuilt and tested again: 924 root tests and
53 parser tests passed, with no failures. A release library check without
default features also passed. Its final 16-round comparison leaves the policy
absent for the enabled arm and explicitly sets it to zero for rollback arms;
profile markers verify that the intended branch actually ran.

| Final default-on fixture | Rollback median | Control median | Default median | Paired rollback saving | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current | 295.539 ms | 298.168 ms | 293.766 ms | 3.739 ms | 10 / 16 |
| Legacy | 301.899 ms | 301.265 ms | 297.046 ms | 4.328 ms | 12 / 16 |

Against identical controls the final paired gains were 4.047 and 7.385 ms.
All 96 final timed artifacts were byte-identical, bringing the total across
the three independent cohorts to 336. No runtime representation change or
masking-speed improvement is claimed: the artifacts are the same. Raw paired
data, profile lines, input/binary hashes and final test counts are retained in
`docs/performance/boundary-fallback-singletons-20260927.json`.

The 10–20 ms whole-link goal is not achieved by this incremental optimization.

## Reproduction

```text
cargo test --release -p glrmask-parser-dwa --features internal-api --lib -- --test-threads=1
cargo build --release -p glrmask --features internal-api --example composition_build_static_artifact
python scripts/compare_packed_fallback.py --builder PATH_TO_EXE --inputs current=PREPARED_DIR --output NEW_DIR --pairs 16
```

The comparison script records inputs, binary and artifact hashes, paired raw
timings, and separately profiled fallback times. It never alters fixtures
and excludes profiling from timed cohorts. The prepared directory contains
`core.bin`, `dispatch-literal.bin` and `vocab_dump.bin`.

Windows evidence for this continuation is under the worktree's
`.benchmarks/fallback-singletons/initial-paired/` and the separate
`C:/Users/isaacbreen/Projects2/scratch/parser-observation-refinement-769824b/`.
