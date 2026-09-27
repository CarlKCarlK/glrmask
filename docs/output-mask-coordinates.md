# Full-width output mask coordinates

## Defect and scope

Model token IDs use `u32`, but several sparse output-mask representations stored
the **output word index** in `u16`. A token at or above `2^21` therefore lost high
index bits. The independent finite-language oracle reproduced this in the
original implementation, the final-mask replay refactor, and the consolidated
refactor: token `2,097,183` was represented by bit 31 of output word zero instead
of bit 31 of word 65,536. The static and AUTO modes failed; dynamic and O2 passed.

The change is based on `9a925d4c286d7069701c25d90e18ef14bf4237bb` and is isolated
from the other continuation's uncommitted consolidated tree. It widens output
coordinates in the direct mapper, internal-token masks, grouped masks, packed
replay slabs and final-weight sparse caches. Internal grammar/token coordinates
and unrelated compact indices are unchanged. The public Rust/Python API and the
private vocabulary-equivalence API are retained.

## Layout and persistence

The packed entry now has two fields, `word_idx: u32` and `mask: u32`. This replaces
`u16` index + `u16` padding + `u32` mask. Its size remains eight bytes with
four-byte alignment. The sparse tuple and mapper entry also remain eight bytes.
Temporary lists of touched output indices now use `u32`; unchanged entry size is
not a claim that every temporary allocation or whole-process RSS is identical.

Artifact version 31 (`S31`) rejects older complete artifacts. The native slab
uses `IBM3`; sparse word-group caches use `TWS2`, and the aligned cache encodings
use `TMC8`/`TMC9`. Each sparse wire record is two little-endian `u32` fields. The
native slab retains its eight-byte record size and backed, aligned loading.
Other sparse wire records grow from six to eight bytes; this is a change to that
section, not a 33% increase in the entire artifact. Save/load size and latency
must be measured rather than inferred from in-memory layout.

Historical artifact compatibility was explicitly not required for the overhaul.
The current-version roundtrip remains required. Existing cache-selection budget
gates are deliberately unchanged in this correction.

## Acceptance evidence

The owned source tree is `/root/glrmask-mask-word-789322-20260927`. Raw evidence,
frozen Python packages, tools and manifests are in
`/root/runtime-replay-review-789322` on the existing CX33. All heavy validation
uses the shared `performance.lock`; no Mac build or new server is involved.

At this checkpoint, the actual mapper's three tests pass, including exhaustive
small set unions, all selections with dirty buffers and high output words, and
full `u32` token-coordinate construction without allocating its entire output buffer.
The release workspace check passes for all features and targets. The release
Python wheel builds, all ten selected prefix/wide-coordinate oracle checks pass,
and all 74 existing Python tests pass. The root and integration Rust suites, complete updated oracle and paired corpus
checks have now completed; details and remaining performance limits follow below.

The built native extension hash is:

```
66b5eabf7b8da8fdfdfd933068acce2fd612ee6f16205a86e117bad304c1d92c
```

`coordinate-tested-source.json` records all 435 Rust source hashes. The paired
performance reference is the replay-v1 package, not the older implementation;
otherwise the large replay improvement could hide a new regression in this fix.

## Oracle policy correction

The original finite-language oracle found six failures in each old package.
Four were an oracle-contract mistake: standalone generation deliberately does
not accept an empty prefix, although nullable embedded children are supported.
That policy is explicit in `ParserTable::embedded_start_nullable` and is not
changed here. The original test source and logs are retained. Oracle version 2
models this policy explicitly, keeps the nullable grammar fixture, and adds a
policy self-check; it does not use `xfail` to hide a failure.

The other two failures are the actual sparse-index wrap. Version 2 must reproduce
those two failures on old binaries and pass them on the corrected binary.

## Replay-contract review

The new final-mask replay path tests complete internal containment before writing
the output buffer. A failed replay contributes to the internal union instead.
Direct output marks the destination dirty, preventing destructive complement
expansion, and the general multi-path loop does not short-circuit later paths.
`store_mask_cache_reuse_dense` clears its old internal witness after a direct
output result. The independent oracle checks full packed masks, output padding,
aliases, alternate token histories, saved/reloaded constraints and linked-child
boundaries; it does not establish arbitrary-grammar or full-corpus correctness.

## Integration limits

This checkpoint is not a blanket performance approval or a main-branch merge.
The artifact-layout continuation moves some of the affected types into separate
owners; integrating this correction there requires carrying both the width and
wire-format changes, not just applying the direct-mapper fix. The consolidated
owner's separate validation and performance work must remain intact.

## Completed coordinate checkpoint

The root-plus-integration run contains **1,162 distinct passing tests**, with
38 pre-existing ignored tests. This excludes the duplicate root run and nested
self-spawned summaries. One public doctest passes. The full repository Python
suite now passes **122 tests**: the existing 74 plus 48 promoted finite-language
oracle tests. The external versioned oracle also verifies package provenance,
for 49 checks; all three old binaries reproduce only the two sparse-index
failures, while the corrected binary passes all 49.

The 100-schema, four-mode comparison against replay-v1 audited 392 successful
case-modes with no full-mask, outcome or roundtrip differences. Eight matching
unsupported-schema preparation failures remain explicit gaps; the same
property-order-dependent expected-valid rejection is retained. This is not
full-corpus or arbitrary-grammar certification.

Across the 392 case-modes, saved artifacts grow by at most **2.079%**. The packed
runtime slab remains eight bytes per entry; the extra wire bytes belong to
other sparse cache records. See `finalize-continued/artifact-sizes.json` for
every measured size, rather than extrapolating from the record width.

A follow-up deliberately chose six cases with unfavorable earlier flags
(indices 7, 18, 23, 46, 55 and 62), four modes, eight paired builds and nine
replays per fresh/reloaded phase. Both the actual and identical-baseline
comparisons audited all 24 case-modes with no semantic mismatch.

| Mode | Fresh TBM p99 ratio | Reloaded TBM p99 ratio | Build p99 ratio |
|---|---:|---:|---:|
| FAST_BUILD | 1.0017 | 1.0209 | 1.0139 |
| O2 | 1.0173 | 0.9365 | 1.0499 |
| FAST_RUNTIME | 0.9655 | 1.0521 | 1.0475 |
| AUTO | 1.0050 | 1.0139 | 1.1027 |

Ratios are corrected/replay-v1; they summarize this selected run, not a confidence
interval or release guarantee. The previously large static/O2 TBM flags did not
repeat at the same magnitude. Two localized FAST_BUILD p99 flags remain
(loaded index18: 1.1315; fresh index23: 1.1553), while an identical-binary
control itself flagged fresh static index46 at 1.2013. Scalar save/load timings
also vary: actual static load p99 is 1.1929; the identical O2 load control is
1.2392. These observations do not prove that every slowdown is noise.

**Status:** preserve as a correctness-reviewed checkpoint; do not claim blanket
performance neutrality or merge into main on this evidence alone. The broader
combined-overhaul owner retains its separate release gate. All raw traces,
flags, same-binary controls, manifests and native hashes remain available.

After the native build, the only Rust edits were comments in the cache decoder.
`finalize-continued/source-comment-audit.json` verifies all 435 built Rust
source hashes before that edit and exact equality of non-comment lines after
it. The promoted oracle contains no production implementation changes.
