# Coherent runtime and artifact overhaul

This checkpoint combines the reviewed runtime work in one source tree. The
public Rust/Python facade and the private vocabulary-partition implementation
are unchanged from the full-width coordinate checkpoint. Correctness and
performance evidence belong to the combined binary, not to an assumed sum of
the individual branches' results.

## Ownership boundaries

- A table's `EmbeddedStart` owns nullability and grammar-level end-token IDs.
  Diagnostic names do not carry execution semantics.
- `Constraint` owns the compiled grammar body and a separate root termination
  policy. Internal mask walkers and caches use body-token coordinates; the
  public dispatcher applies root end tokens afterwards.
- The persistence envelope describes the one supported format. Named borrowed
  sections identify the weight, parser, tokenizer, vocabulary, mask-cache and
  composition payloads. Section readers retain the original byte backing.
- Vocabulary equivalence remains an internal operation. No new public type or
  execution-engine selection is introduced by this checkpoint.

The coherent tree includes the final-token-set replay refactor, full-width
output coordinates, typed table metadata, current-format token-cache decoder,
and previously reviewed unused self-loop cache removal. It does not include the
separate artifact-layout split whose performance gate is still unresolved.

## Root end-token space is not body-mask space

The independent composition regression found a mismatch between two contracts.
The public mask buffer includes the root's end-token IDs. `fill_body_mask`
correctly splits off any extra words used only by that termination policy.
The recursive traversal then incorrectly asserted that its already-trimmed
slice had the full public size. A root end token beyond the body vocabulary
could consequently panic even with a correctly sized public output buffer.

Both recursive mask entry points now use `body_mask_len`, matching the ordinary
dynamic walker. They do not allocate a larger buffer or allow a child end
token to become part of the parent's byte language. The outer dispatcher still
adds or clears the root's end-token bits according to acceptance.

The regression enumerates the language `<>`, `<ab>`, `<ac>` independently of
GLRMask. It checks all nine public child/parent optimization combinations,
nullable compiled children, an additional compiled wrapper, self-contained
save/load before and after cache population, distinct child/parent end tokens,
high-ID aliases, fused tokens crossing a boundary, and poisoned oversized
output buffers. Its expected masks are not copied from another engine.

## Current artifact contract

The coherent constraint envelope uses version **33**, with section marker
`S33\0`. The isolated coordinate and table branches used incompatible versions
31 and 32; neither is treated as the coherent format. Nested encodings retain
their deliberately selected current markers, including `GTC4`, `GTM3`, `IBM3`,
`TWS2` and `TMC8`/`TMC9`.

The section reader rejects incorrect magic, truncated headers or bodies,
nonrepresentable lengths, overflowing ranges and trailing payload. Named
fields replace two layers of positional eleven-element tuples. The reader
still borrows every section directly, including empty sections; the writer's
order and deferred backing ownership are unchanged.

Historical compatibility is intentionally not promised. This does not relax
validation or current-format roundtrip requirements. The design notes for the
isolated branches are historical checkpoint records; this document specifies
the combined outer version.

## Evidence and limitations

The exact combined source is frozen in a non-Markdown input manifest; the
validation checks both every hash and the complete source-file set. Snapshot
manifests identify all three imported batches. The only overlapping runtime
artifact file was combined with a retained, clean three-way merge.

All heavy work runs on the existing CX33 under the shared performance lock.
The combined target directory is independent: it does not modify a peer's
build cache or frozen wheel. The public facade is checked byte-for-byte and by
reflection; private partitions and full masks are checked with the shared
128,256-token vocabulary.

The paired corpus compares against the immediately preceding decoder build,
not an older slow integration. Unsupported preparations, exact mask/outcome
differences, raw timing traces, per-position stabilized summaries and adverse
cases remain distinct. A passing semantic audit is not proof of performance
neutrality. An identical-binary timing control does not automatically excuse
a candidate slowdown. The first 100 ordered schemas are not the full corpus
or a random 1,000-case release sweep.

## Completed correctness checkpoint

The frozen combined binary passed **2,195 Rust tests** across 42 workspace
execution targets, with 41 pre-existing ignored tests, plus **131 Python tests**
and one public doctest. Nine nested subprocess result summaries are excluded
from the Rust total. All-feature/all-target compilation and the release wheel
build also passed. There were no retries of these validation gates.

The 18-input EOS/body-space interaction matrix reproduces 18 panics on the
preceding decoder binary and passes all 18 on this binary. Nine wide-coordinate
cases repeat repository tests; the nine small-coordinate inputs are additional
coverage, not 18 more repository test functions. Public API reflection matches
exactly. The full-vocabulary guard matches 38 constraint cases and 24 private
partition cases over 128,256 entries.

The 521 frozen non-Markdown source inputs are unchanged. The candidate native
extension SHA-256 is:

```
3fe7a1f8cf29c00b13456f57d81fb2b762a35bd836f9581814cf2afa8c4df5ad
```

The paired first-100-schema run is still in progress at this checkpoint.
**Performance is not approved, and this checkpoint is not merged into main.**
Prior branch-specific timing flags remain recorded rather than being assumed
resolved by the combination. Evidence lives in
`/root/overhaul-handoff-review-789322` on the existing CX33; the compaction index
points to its mirrored continuation note.
