# Referenced-weight compression in the static boundary minimizer

## Scope

The static boundary compiler first resolves signed parser-stack effects,
normalizes the positive weighted graph, and expands the parser's explicit-label
and `DEFAULT` behavior. This optimization acts **after those operations**, inside
`src/compiler/boundary_bit_minimize.rs`. It does not change LR tables, templates,
the tokenizer, control transitions, or the semantics of `DEFAULT`.

It accompanies the earlier change that groups determinization contributions
in place and reuses bounded temporary frontier buffers. That allocation change
preserves the complete constructed graph. The compression described here
preserves parser-prefix masks but may change the final greedy grouping order.

## What was unnecessary

The final minimizer receives an interned pool of finite bit masks. Many masks
in that pool were temporary arithmetic results or predicates used by an earlier
construction stage. They are no longer referenced by the normalized graph.

Previously, every interned mask participated in the membership signatures used
to compress the coordinate universe. Unreferenced masks could distinguish
coordinates that no subsequent minimizer operation needed to distinguish.

The new path gathers the unique IDs of all state-final masks, retained edge
masks, and backward `needed` domains, together with the empty and identity masks.
Only these IDs participate in the observation partition. The collection happens
after the existing backward productivity pass, so it also includes the masks
that pass creates. IDs are visited in their original interner order.

## Exactness argument

Let `D` be the finite root coordinate domain and `R` the set of referenced
predicates described above. Define

```text
x ~ y  iff  for every A in R, (x in A) == (y in A).
```

Every predicate in `R` is a union of equivalence classes. The same is true for
every Boolean expression generated from `R`: union, intersection, and difference
are all evaluated pointwise. Therefore encoding classes as bits commutes with
the Boolean operations performed by the remaining minimizer.

The decoder maps a new class to the union of its original disjoint coordinates
or already-decoded atoms. Composing the two disjoint partitions preserves
original tokenizer-state and model-token IDs. Hashes only locate candidate
signatures; complete signature equality decides whether coordinates merge.

An unreferenced old mask receives an invalid mapping sentinel. The caller remaps
only collected references. The interner is replaced only after the complete
compressed interner, mapping and decoder have been constructed successfully.
If compression cannot reduce the word width or exceeds its existing budget,
the original representation is retained.

## Guard and representation invariants

No LR-state symbol or graph-state identity is merged by this prepass. Explicit
zero-weight transitions and `DEFAULT` keys remain unchanged. In particular, it
does not apply weighted-language pruning before fallback expansion, where a
zero-output branch can still provide a blocking explicit-label guard.

The existing final minimizer is greedy rather than canonical. A coarser atom
partition changes support popcounts and can change its bucket order, leading
to a different but equivalent final graph. Serialized byte inequality is not a
correctness failure by itself. Conversely, a smaller graph is not evidence of
correctness: the finalized original-coordinate parser-prefix comparator is the
semantic gate.

## Regression coverage and reproduction

The focused tests include a fixture with 128 historical singleton predicates
that are unreferenced by the actual graph. The historical partition has 128
classes; the referenced partition has two. Both produce identical parser-prefix
masks. Another test compares both implementations on 256 generated finite DAGs,
including explicit zero transitions, `DEFAULT` behavior and multiple coordinate
rows. Existing budget, collision, sparse-ID and prefix-language tests continue
to cover the minimizer.

The production default uses referenced observations. Set
`GLRMASK_BOUNDARY_MIN_REFERENCED_OBSERVATIONS=0` to run the historical all-predicate
implementation for controlled comparison. The lowercase values `false` and
`off` also disable it. With `internal-api`, setting
`GLRMASK_VALIDATE_MIN_REFERENCED_OBSERVATIONS=1` constructs the reference from the
same immutable native input and compares the finalized decoded prefix languages.
This validation work must not be included in timing measurements.

Build the two existing examples and run the reproducible comparison:

```text
cargo test --release -p glrmask --features internal-api --lib -- --test-threads=1
cargo test --release -p glrmask-parser-dwa --features internal-api --lib
cargo build --release --features internal-api --example composition_build_static_artifact --example composition_loaded_static_probe

python scripts/compare_boundary_observations.py --candidate PATH_TO_NEW_BUILDER --baseline PATH_TO_FROZEN_OLD_BUILDER --runtime PATH_TO_LOADED_PROBE --inputs current=PATH_TO_PREPARED_COMPONENTS --output NEW_RESULT_DIRECTORY
```

The runner records executable/input/artifact hashes, exact-validation logs,
rotated interleaved main/reference/control/default builds, raw timings and paired
differences. It then uses one runtime binary to load the separately constructed
artifacts, checks their mask signatures, and retains per-prefix runtime timings.
Compilation and serialization are reported separately. No prepared fixture is
modified, and no rejected research optimization is enabled by this change.

## Representative structural census

On the selected-ten-tool outer-link fixture, the final stage references 2,293
of 8,446 interned masks. Selecting actual observations reduces the final-stage
coordinate partition from 530 to 380 classes and the mask width from nine to six
64-bit words. The pre-minimized graph is unchanged. The greedy final graph
changes from 1,199 states / 31,967 transitions to 1,231 / 33,534, which is why
loaded-runtime checks accompany construction timings.

These changes are incremental improvements, not achievement of the separate
10–20 ms whole-link target. Exact inputs, timing samples and remaining work are
recorded with each measured publication run rather than inferred from graph
size alone.

## Publication validation — 27 September 2026

The clean publication source passed 901 root-library tests (zero failures,
51 ignored), 49 parser-crate tests (zero failures, one ignored), and the normal
release library check with no default features. These counts exclude unshipped
research experiments. Both realistic fixture generations passed the finalized
original-coordinate prefix-language comparison with the production default.

The final build comparison used twelve rotated interleaved rounds per fixture,
with four arms: frozen main, the new binary with observations disabled, an
identical disabled control, and the default-enabled candidate. All 96 timed
artifacts matched their separately validated hashes. Validation work was not
inside those timed builds.

| Prepared fixture | Frozen-main median | New default median | Median paired main-minus-default | Positive pairs |
| --- | ---: | ---: | ---: | ---: |
| Current | 317.788 ms | 298.862 ms | 13.579 ms | 11 / 12 |
| Legacy | 318.145 ms | 301.342 ms | 15.332 ms | 10 / 12 |

The combined improvement includes the earlier frontier allocation change;
separate historical gains must not be added to this table. Observation
compression alone saved a median paired 6.301 ms against disabled mode and
4.850 ms against its identical control on current inputs. The corresponding
legacy savings were 4.310 and 6.733 ms. The disabled/control paired differences
were 1.762 ms current and -0.438 ms legacy, illustrating measurement variation.

The candidate serialized artifact grows by 19,788 bytes, approximately 0.042%
of either whole artifact. A single runtime binary loaded each artifact for
four rounds of 151 repetitions at 22 prefixes. Mask signatures agreed throughout;
these samples showed no material runtime regression. For example, the current
fixture's mean of per-prefix median mask times was 38.175 microseconds disabled
and 37.818 microseconds enabled. These are means of per-prefix medians, not
percentiles of a pooled runtime distribution or guarantees for all workloads.

A separate loaded-artifact differential harness compared full mask arrays and
acceptance/rejection/termination behavior rather than hashes alone. On **each**
fixture it passed 22 active anchors, 485 byte commits, 6,139 state comparisons,
5,632 selected valid-token commits, and 2,816 arbitrary clone commit probes.
The semantic checker and loaded-runtime checks are complementary: realistic
history tests do not replace the arbitrary-prefix equivalence gate.

Compact source/input/binary provenance and measured results are retained in
`docs/performance/boundary-observations-20260927.json`. The reproduction script
emits the complete raw report and logs in its chosen output directory.
