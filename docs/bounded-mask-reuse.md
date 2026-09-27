# Bounded commit-time mask reuse

## Why the policy changed

A cached next-token mask can survive a token commitment when the complete
semantic state, its exact mask-execution coordinate, or an independently proved
parser-relative lexer observation is unchanged. That can avoid a vocabulary
walk. But proving equivalence is itself runtime work: the former 2,048-byte-step
speculative search could be much more expensive than the mask it avoided.

The first coherent-overhaul screen localized two repeatable dynamic TBM losses
to commitment rather than masking. The corresponding detailed profiler did not
include final cache-retention work in its reported total, although the ordinary
benchmark timer did. Controlled same-binary policy experiments, kept outside the
library, identified successful long equality proofs as the dominant cost.

## Retained correctness rules

`commit/mask_reuse.rs` owns generation snapshots and mask-retention decisions.
Exact state equality and exact mask-coordinate equality retain their previous
rules. Ordinary lexer quotients still cannot interpret scoped composition IDs;
recursive composition requires the established exact-state checks.

The parser-relative path keeps the same correlated parser/GSS checks, initial
state distinction, exclusions, epsilon/virtual-runtime guards, exact admission
set, IGNORE treatment, and relation-cache key. It tries the existing exact
prepared-terminal certificate before starting the speculative vocabulary-trie
walk. Missing coverage or unequal observations cannot certify equality.

The fallback now defaults to 64 byte steps instead of 2,048. Running out of
budget returns a failed proof, not an approximate answer: the cached generation
is not retained, and the ordinary exact mask computation supplies the next
result. Successful certificates retain the same bounded relation cache. There
is no new approximate mask, parser merge, or special treatment of benchmark IDs.

The diagnostic override
`GLRMASK_DYNAMIC_MASK_PARSER_RELATIVE_COMMIT_REUSE_MAX_STEPS` remains available.
The existing disable switch still disables the parser-relative proof, not all
mask caching. The proposed default requires no environment opt-in.

## Profiling contract

Private commit profiles include final generation/cache-retention work in
`total_ns` and expose `mask_cache_reuse_ns` as a contained subinterval. Ordinary
commit, dynamic commit, and per-advance Python diagnostic dictionaries share one
typed mapping of the 70 existing/new fields. Public facade methods and
vocabulary-partition semantics are unchanged.

The ordinary benchmark `commit_token_timed_ns` source is byte-for-byte unchanged
from the coherent checkpoint: it already timed cache-retention work. Thus the
profile correction does not create benchmark gains by moving work outside the
timer. The total is the native instrumented interval, not end-to-end Python wall
time; overlapping parent/child profiling buckets must not be summed blindly.

## Persistence and validation

This batch does not change the version-33 artifact representation. Previous
current-format artifacts must remain loadable in both directions. The original
coherent native package is retained independently of the rebuilt package.

Validation covers the complete Rust and Python suites, independent finite
languages, sparse token aliases, dirty oversized packed buffers, current-format
cross-package loading, API reflection, full-vocabulary constraint/partition
comparisons, and proof policies of zero steps, one step, the default, the old
2,048-step limit, and disabled parser-relative reuse. Policy-matrix repetitions
are not counted as additional unique repository tests.

Performance comparisons use unchanged native timers, counterbalanced package
order, fresh and reloaded constraints, first-use and warmed traces, exact
mask/outcome fingerprints, and independent alignment of masks with commits.
Unsupported schemas and unfavorable timings remain visible. Stabilized
per-position results and unfiltered wall-time tails answer different questions
and are reported separately. No selected screen certifies all grammars or the
complete CFA/llguidance target.

## Evidence location

The working evidence directory is
`/root/overhaul-handoff-review-789322/reuse-refinement` on the existing CX33.
The parent correctness checkpoint is
`7c74d1e83ad159e315a3c9b7e87f3e85d1ebe9ca`. Source manifests, previous failed check
attempts, package hashes, test XML/logs, raw benchmark traces and ownership notes
are retained there. Consult the completed reports for actual gate status; this
design note by itself does not approve a merge or establish performance.

## Completed correctness checkpoint, 27 September 2026

The release workspace check, release wheel, 2,198 Rust tests (41 existing ignored
cases), 139 Python tests and one public doctest pass. The 42 Cargo target results
exclude nine nested subprocess summaries from the unique count. There are 524
verified non-Markdown inputs. Seventeen unchanged facade/partition source files
and all 70 private profile-field mappings were independently checked.

All five 65-case policy matrices passed. Sixteen artifacts load in each direction
between the previous coherent package and this refinement. Public API reflection
is identical. Full-vocabulary comparison passes 38 constraint cases and 24 private
partition cases with 128,256 model tokens.

The eight-schema focused comparison audits all 32 case-modes without semantic
mismatch. For dynamic FAST_BUILD, the two original problem cases have warmed
fresh p99 TBM of 153.986 -> 21.941 microseconds (o70369) and 150.002 -> 17.753
microseconds (o30735). Corresponding loaded values are 186.891 -> 23.489 and
168.364 -> 20.926 microseconds. These are measured default-to-default comparisons,
not environment-only experiments and not cold-start or universal bounds.

The first 100 schemas audit 392 of 400 case-modes; eight identical unsupported
preparations at indices 27/85 remain excluded from successful coverage. All full
mask/outcome fingerprints and roundtrips agree, with no independent audit error.
Across 31,835 accepted token positions per mode, fresh dynamic p99 TBM changes
35.102 -> 33.248 microseconds and loaded p99 40.615 -> 38.333 microseconds.
The fresh stabilized maximum increases 602.710 -> 684.968 microseconds, while
loaded decreases 641.332 -> 611.682 microseconds. Stabilized maxima are not raw
wall-time maxima or a worst-case guarantee.

There is NO blanket performance acceptance: first100 fresh static p99 increases
7.13%, fresh AUTO p99 increases 4.22%, and 116 local timing flags remain. O2 and
AUTO load-time p99 ratios are respectively 1.1187 and 1.1398. All raw evidence is
retained, not silently dismissed as noise.

Independent reviewer R2317 checked 72 finite-language cases under four policies,
with 168,804 full-mask calls and 378 profile calls per policy; all passed. Its
predeclared 20-schema holdout outside the first100 audits 76/80 case-modes with
four matching unsupported preparations at index2782, not 80 successful cases.
Both adverse eight-schema repetitions and the identical-baseline control audit
32/32 case-modes. The actual repetitions have zero and one local flags; the
control has one. Those controls do not erase the original holdout's 27 flags or
prove every remaining slowdown is noise.

Preserved native baseline: `3fe7a1f8cf29c00b13456f57d81fb2b762a35bd836f9581814cf2afa8c4df5ad`.
New native: `677bea59ef2ae3be1c9ffe907b069d0045a1e7cd88a591a114a316c28484ec4d`.

This is a correctness-reviewed feature checkpoint, not a main-branch release or
full-CFA/llguidance certificate. The separately owned finalization review handles
additional proof-path diagnostics, unfiltered-tail reports and the acceptance
ledger. Their outcomes must be inspected before any later release decision.
