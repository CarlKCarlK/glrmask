# Runtime artifact modules

The private `runtime::artifact` facade names the immutable representations used
by the compiler, mask runtime, and persistence layer. Its root now contains the
`Constraint` layout, its remote Serde definition, and re-exports. Algorithmic
implementations no longer share one 9,700-line file.

## Responsibility boundaries

| Module | Responsibility |
| --- | --- |
| `boundary` | Component grammar summaries and boundary-candidate certificates. |
| `mask_rows` | Dense, sparse, and byte-backed token-mask row storage. |
| `transitions` | Compact parser, lexer, and commit-template transition tables. |
| `trie` | Vocabulary radix tries, packed walks, and slice-trie metadata. |
| `regular` | Exact direct-regular terminal support and frontier data. |
| `vocabulary` | Dynamic vocabulary ownership, constructors, aliases, and defaults. |
| `slices` | Slice admission and original-token walk materialization. |
| `master_proofs` | Prepared slice certificates, exact coverage, and bounded radii. |
| `projections` | Mask-only lexer coordinates and terminal-observation proofs. |
| `mask_cache` | Runtime mask, subset, and frontier cache operations. |
| `vocab_transfer` | Dynamic vocabulary transfer and token-byte validation. |
| `recursive` | Scoped parser coordinates and segmented boundary representations. |
| `deferred` | Backed compiler metadata decoded only for later composition. |

These are source modules within the existing crate, not new crates or boxed
wrappers. Internal consumers keep the existing `runtime::artifact` import paths.
Moved private definitions use `pub(super)` only where needed to preserve their
former artifact-module access boundary. This does not expand the supported API.

## Preserved invariants

Struct fields, enum variants, field order, ownership, cache lifetimes, and lazy
initialization remain unchanged. Inherent methods may live beside the relevant
algorithm instead of beside the struct definition; this adds no runtime dispatch.

A missing proof remains unknown, not rejection or permission. A missing cache
entry remains a miss, not an empty mask. `fresh_runtime_instance` retains its
existing separation between shared vocabulary data and per-constraint derived
proofs and mutable caches. Commit still uses the exact lexer; mask projections
are not substitutes for commit semantics. Scoped component coordinates remain
distinct from ordinary lexer coordinates.

The public `Constraint` and its remote Serde definition stay at the original
module path. Current artifact field order and format are not changed. Historical
compatibility is not extended. In particular, this reorganization does not repair
incorrect machines produced by the removed terminal-run compiler optimization:
those must be recompiled from their grammar, as documented by that correction.

## Validation and evidence

The relocation audit maps all 438 declarations and associated items from the
original file. Executable tokens, attributes, constants, and field ordering must
match; only explicit artifact-private visibility is normalized. Formatting is
checked against the identically visibility-normalized original, so signature
line wrapping does not hide expression changes. Existing regression files are
retained, including the independent recursive-language compiler tests.

The audit is a structural check, not proof of binary layout or performance.
Native validation must separately check the ordinary public build, the complete
workspace and Python suites, current-format cross-package loading, full-vocabulary
masks and private partitions, and recursive-language behavior. Paired measurement
must use the preserved corrected parent binary, not an older known-incorrect
compiler. Source movement can affect generated code even when algorithm tokens
match; performance is therefore not presumed neutral.

Current engineering evidence is in `/root/runtime-artifact-modules-789322` on the
existing build host. Consult the completed logs, source manifest, native hashes,
and comparison reports for actual gate status. This design note does not certify
a release or a full-CFA tail distribution.
