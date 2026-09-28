# Skip final-minimizer merges that cannot change the group

The final weighted parser-DWA minimizer sometimes accepts a state into a group
whose domain already contains that state's entire domain. After the existing
compatibility test succeeds, this is an exact certificate that the ordinary
merge would change nothing. The compiler now avoids rebuilding the edge vector
and recomputing those unions.

`GLRMASK_BOUNDARY_MIN_CONTAINED_MERGES` uses the standard boundary selector:
enabled by default, with an explicit `0`, `false`, `off`, or `no` selecting the
historical merge. The selector is read once per minimization. No extra
per-state index or persistent cache is required.

## Why skipping is exact

Let a candidate state's needed domain be `d`, its final coefficient `f`, and
its coefficient on label `a` be `m[a]`. The backward support pass establishes

```text
d = f UNION (UNION over a of m[a]).
```

Thus all candidate coefficients are subsets of `d`. Let the corresponding
group values be `D`, `F`, and `M[a]`. Group creation and ordinary merges maintain
the same containment property for the group. Observation compression preserves
these finite-set relationships.

The unchanged `compatible` function first checks guardedness, explicit label
keys where required, final coefficients on the shared domain, matching
nonzero target IDs, and edge coefficients on the shared domain. Consequently,
if `d` is a subset of `D`, then on all of `d`:

```text
f    = F    INTERSECT d
m[a] = M[a] INTERSECT d.
```

It follows that `D UNION d = D`, `F UNION f = F`, and every
`M[a] UNION m[a] = M[a]`. None of the original merge's Boolean operations can
create a new mask value or change an existing ID.

The key and target details are important. Guarded rows must have identical
complete key sets, including explicit zero guards. Unguarded rows have had
their zero edges removed: a new positive key would fail compatibility on `d`.
Common nonzero targets already agree, and zero targets are canonicalized to
zero before signature construction. Therefore the entire ordered group
representation is unchanged, not merely its weighted language.

The containment check is **after compatibility**, not a replacement for it.
The optimization does not merge extra states, change grouping order, alter
`DEFAULT`, prune grammar paths, or reinterpret arbitrary incoming parser
stacks. Height buckets, greedy ordering, candidate selection, exact-signature
keys and final group numbering are unchanged.

## Validation and measurement

The regression tests construct 8,192 compatible restricted-domain pairs and
run the historical merge, requiring exact equality of all group fields and
the number of interned mask values. Further tests cover growing domains,
conflicting targets, absent versus explicit-zero keys, `DEFAULT`, sentinels,
and coefficient widths of 1, 2, 17, 44 and 64 words.

The initial experimental source passed all 958 root tests and preserved every
artifact byte in 352 timed prepared-component compositions across current and
legacy fixtures. Untimed gates also checked every normalized native state,
label, target, decoded coefficient and decoder, plus loaded runtime masks at
22 or more anchors.

The real child minimizer made 9,192 successful non-exact merges; 5,578 were
strict-subdomain no-ops. Equal-domain cases had already been handled by the
exact-signature cache. The no-op cases accounted for 267,019 combined old/new
edge-vector visits. Diagnostic merging time fell from about 8.9 to 7.0 ms.

A separate 576-invocation source-included phase crossover repeatedly found
about 1.67–1.70 ms less merging work when enabled; disabling the shortcut
removed that gain. The whole-composition measurements were noisier: the
larger 256-build experimental confirmation showed median paired same-binary
savings of 1.94 ms current and 1.45 ms legacy, while an identical legacy control
comparison showed only 0.33 ms. The first 96-build screen was mixed and is
retained in the evidence. These are small improvements, not a large reduction
in the whole construction budget.

The cleaned default-on implementation, integrated with main's subsequent
dynamic-runtime changes, passed **959 root tests and 71 parser-DWA tests**,
with zero failures, plus the ordinary public-library check. A further
448 whole-composition builds at 10, 4 and 1 workers preserved every artifact
byte, native identity and loaded-mask check. The final ten-worker current
fixture remained noisy: the same-binary paired result was -0.58 ms, while
legacy was +1.57 ms. Four-worker results were also mixed. These results do not
justify claiming a dependable multi-millisecond whole-link speedup.

A separate clean-source 576-invocation phase crossover confirmed the mechanism:
enabled merging saved 1.68–1.76 ms against the frozen reference, with 24/24 wins
in each enabled session; disabling it removed that improvement. The change is
retained for this repeatable, precisely identified avoided work, without
attributing unrelated compiler or runtime improvements to it. Complete raw
cohorts, including the negative/noisy ones, and their provenance are in
`docs/performance/boundary-contained-merges-20260928.json`.

## Reproduction

Build the normal release examples with `--features internal-api`:

```text
composition_build_static_artifact
composition_loaded_static_probe
compare_boundary_native_rows
```

Run `scripts/compare_contained_minimizer_merges.py` with explicit paths to the
independently frozen baseline, candidate, runtime probe, native checker, both
prepared fixture generations, and a fresh output directory. The script
records source, executable, input and artifact hashes. It compares independent
main, same-binary historical merge, an identical historical control, and the
actual unset/default-enabled policy. It fails on fallback to the wrong policy,
native differences, artifact differences, loaded-mask differences or missing
measurements.

All diagnostics and artifact serialization are outside the composition timer.
All actual compiler allocation, compatibility and containment checks remain
inside it. The input constraints are already prepared: these figures do not
include initial grammar compilation, loading or serialization. The isolated
phase replay is narrower still and must not be reported as whole-build timing.
