# Explicit embedding metadata for GLR tables

> Historical isolated table checkpoint. The combined branch uses outer format
> **33 / `S33`** and is validated separately. See [the coherent overhaul](coherent-overhaul.md).

## Why this change exists

The table previously stored two semantic properties inside the augmented start
rule's diagnostic name: whether the source start is nullable when embedded, and
which model token IDs have grammar-level end-token roles. The encoding used
NUL-delimited suffixes, decimal formatting and parsing. Updating one property had
to strip and restore the other property's suffix. A table without rules or
diagnostic names could not retain either property through these setters.

This arrangement preserved an older serialized table shape, but made diagnostic
data an undocumented part of execution and composition semantics. Historical
artifact compatibility is not required for the pre-release overhaul, so that
coupling is no longer justified.

## Representation and invariants

`GLRTable::embedded_start` owns an `EmbeddedStart` value. It contains an embedded
nullability flag and sorted, deduplicated model end-token IDs. Diagnostic names
have no role in reading or updating either property. Existing getter and setter
signatures are retained; this adds no new top-level GLRMask API.

The nullability flag answers a composition question, not whether standalone
generation may terminate before committing any token. The existing nonempty
standalone-generation policy is unchanged. End-token IDs record grammar-level
roles, not just token bytes: an explicit end token must remain distinguishable
from an ordinary exact-token placeholder.

Both table-composition constructors preserve parent metadata and then compute
the composed start's nullability. Table quotients move metadata from their input
table, and recursive coordinator shells clone their parent's metadata. Fresh
table constructors use the default metadata; existing compiler finalization
steps still set the source properties. Tables without diagnostic names now
retain explicitly configured properties instead of silently discarding them.

## Module ownership

`table/embedding.rs` owns the semantic value and the four accessors. Its tests
exercise canonical updates, independence from names, cloning, state quotienting
and serialization. `table/artifact_serde.rs` owns the existing compact table
codec. It has been extracted without changing its module namespace or the live
row, admission-set, deferred-rule and parallel-scheduling algorithms.

The monolithic `CompactTable` reader and its unused writer type are removed.
The current writer already emitted sectioned tables, so retaining those old
shapes only increased the number of representations the reader had to support.

## Current artifact contract

This isolated development branch uses the outer constraint version 32 / `S32`,
compact table marker `GTC4`, and deferred metadata marker `GTM3`. The new semantic
value is carried in ordinary table serde, compact metadata and deferred metadata.
Owned and artifact-backed rule storage retain the same ownership arrangements.
The compact table reader rejects earlier table markers instead of attempting a
legacy fallback.

These identifiers describe this branch, not the eventual combined release.
The separate mask-word-width branch has its own incompatible format change.
Integration must assign and test a format for the combined representation rather
than assume independently assigned development versions interoperate.

## Validation and performance

New regression tests cover names that resemble the retired suffixes, completely
name-free tables, independent canonical setters, clone and ordinary serde,
an actual two-state-to-one-state quotient, compact and deferred roundtrips with
one and two workers, retained rule backing at multiple offsets, and rejection
of retired encodings. Existing Python lifecycle and composition tests are also
part of validation.

The source audit checks the moved codec against the original after accounting
for the explicit metadata and retired readers. It separately checks metadata
transfers and unchanged Rust files. Whole-workspace tests, public API reflection,
full-vocabulary semantic and partition checks, and paired runtime measurements
are separate gates; source similarity is not a substitute for those gates.

The representation avoids formatting and parsing semantic state, but adds an
explicit value to each table and changes generated code and artifact layout.
Performance neutrality therefore requires measurements, not an assumption based
on the apparent simplicity of the change. Compare against the frozen replay-v1
baseline rather than an older implementation whose unrelated slowdown would
hide a regression. Cold and warmed fresh/loaded masks, commits, grammar builds,
save/load, raw tails and unfavorable cases must remain visible.

Execution results and provenance are recorded in the owning compaction note and
the validation report. This design note alone does not certify performance or
approve integration into the main branch.
