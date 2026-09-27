# Direct construction of the existing core-merged LR table

## Scope and invariant

This change avoids constructing and then discarding canonical LR(1) action
rows. It does not select LALR, change the grammar, weaken parser admission, or
alter the unit-reduction optimization budgets.

The compiler still builds exactly the same canonical LR(1) item sets and
transitions. Core compatibility depends only on the current core class,
terminal shifts, nonterminal gotos, replacement flags, forwarded-shift flags,
and destination classes. These observations are available directly from the
canonical transition graph. Reductions and acceptance do not participate in
this refinement.

Consequently the same refinement loop can obtain the same first-seen class
numbers without materializing the intermediate execution table. Each final
class copies its common shift/goto signature once, then unions reductions and
acceptance from its completed, non-transferred canonical items. The existing
action normalization deduplicates the union. Execution-row representation is
preserved: in particular, large hash-backed rows are not replaced with sorted
vectors merely because those vectors are cheaper to construct.

Direct construction is the default. `GLRMASK_DISABLE_DIRECT_CORE_TABLE=1`
selects the materialized reference for diagnosis. No flag is required to
receive the improvement. The former experimental switch used in the retained
pre-adoption measurements was never part of a published release.

## Exactness checks

The tests compare 69 generated and hand-written grammars against canonical-row
materialization. They check compatibility signatures, state IDs, complete
actions/gotos, admission rows, forwarded shifts, runtime row representations,
and complete table serialization. A separate family varies replacement and
forwarded flags. Default selection also has a pure, environment-independent
test.

On JavaScript, all 4,099 full-vocabulary masks matched in both static and
dynamic modes. A 126-schema JSB guard matched 77,582 full masks and trajectory
outcomes, both directly after compilation and after serialization/load. The
decoded reference stream is 1,245,309,451 bytes with SHA-256
`351dab2120aadea63c7515ff1c12630fd8857184817eb0f87dfaa1e30932ece2`.

## Paired Mac measurements

Measurements start from published source `f1b5ceaa0`, use Rust 1.95.0, native
CPU optimization, release/no-strip, and explicit llguidance-compatible JSON
policy. Each comparison uses the same frozen binary with the implementation
switch changed between processes. Build time is elapsed compile time;
runtime measurements use the thread CPU clock.

JavaScript uses the unchanged 31-example corpus. Each implementation has two
processes in ABBA order, with three fresh loaded-state runs per example in
each process. TBM below is CFA's mask-plus-commit pair at the same position;
the complete pair is minimized across runs, not its components separately.
All scored positions have corresponding llguidance measurements.

| Metric | Materialized reference | Direct construction |
| --- | ---: | ---: |
| Dynamic JS build median, build-only screen | 152.930 ms | 140.820 ms |
| Dynamic JS build median, full-mask batch | 156.799 ms | 139.274 ms |
| Dynamic JS TBM P90 | 7.502 ms | 7.468 ms |
| Dynamic JS TBM P100 | 17.314 ms | 17.328 ms |
| Static JS TBM P90 | 30.800 us | 30.467 us |
| Static JS TBM P100 | 77.500 us | 80.375 us |

Static JS construction was effectively unchanged in the build-only screen
(3.706 versus 3.709 seconds). A follow-up 40-observation-per-side test of six
static tail positions did not reproduce a consistent slowdown; the initially
slowest candidate position had minima 77.459 versus 75.708 us. This is not a
claim that every runtime observation improves. Dynamic raw maxima were
18.487 versus 20.872 ms in the full-mask batch; cold and raw tails remain
separate optimization work.

The native JSB guard fixes the first 1,000 schemas in retained corpus order
after the llguidance-success filter, before running either implementation.
All 1,000 compiled in both implementations with identical outcomes. It scores
only the 367,640 intervals with llguidance data. Native TBM is commitment of a
token through the **next** ready mask, so its values must not be mixed with
the JavaScript CFA convention above.

| Native JSB metric | Reference | Direct |
| --- | ---: | ---: |
| Build P50 | 9.879 ms | 9.897 ms |
| Build P90 | 77.475 ms | 75.439 ms |
| Build P99 | 227.252 ms | 224.833 ms |
| Build P100 | 650.933 ms | 659.973 ms |
| TBM P90 | 6.167 us | 6.167 us |
| TBM P99 | 13.417 us | 13.416 us |
| TBM P100 | 46.417 us | 44.125 us |

This is a fixed regression panel, not a new full-population JSB headline.
The primary demonstrated benefit is dynamic JS compilation, with no material
runtime tradeoff established in the checks. Worst-case build time is not
claimed to improve on every schema.

## Retained reproduction evidence

The isolated `glrmask-lr-build-345436i-20260927` worktree retains scripts,
complete raw observations, reference artifacts, and logs under
`.benchmarks/lr-build/`. Key records are `build-screen-*/summary.json`,
`full-*/summary.json`, `native-1000/summary.json`,
`static-tail-confirm/summary.json`, and `full-mask-parity.json`.

The pre-adoption Python binary SHA-256 is
`e49f3033c8e06e1413e95bd0e17eb1dbec0921995e5a89d52358753df1a1dc32`.
The native paired runner SHA-256 is
`296cb1bc737ea8edf223b46078743b7c3db9626a877097a47facdf5b5da16b46`.
The live compaction note records final default-build verification and later
publication separately from these frozen experiment binaries.

## Final default-mode verification

The no-override production build passed the full Rust workspace and all 42
Python tests. It also replayed the full 31-example JavaScript corpus in static
and dynamic modes (4,099 full masks each), then all 1,000 fixed LL-supported
native JSB schemas with no semantic differences. These checks use the default
implementation, not the pre-adoption experimental switch. The final Python
package SHA-256 is
`c85d23615e547262e9780505a2a9acd278404fb79af12b76e21c00457787e49d`;
retained evidence is `default-test-summary.json`, `test-python-default.log`,
and `default-validation/summary.json`.

## Final default-policy verification

With no core-selection environment override, the final pre-merge source passed
2,220 top-level Rust tests (54 ignored) and all 42 Python tests. The default
binary repeated all 4,099 JavaScript mask comparisons in each execution mode
and the fixed 1,000-schema native guard without semantic differences. The
final default Python package SHA-256 is
`c85d23615e547262e9780505a2a9acd278404fb79af12b76e21c00457787e49d`;
the native runner SHA-256 is
`8335cdb15ddaa16a6bf8bc370d1f364a8d4875792041b18e2308bd45429bde1d`.
These verification runs confirm default routing, not a new paired timing claim.
