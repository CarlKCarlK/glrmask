# Direct static-mask workspace

`runtime/mask/single_path.rs` owns admission, explicit-path evaluation, and
optional repeated-stack plans for static masking. It is an internal refactor;
the public API, vocabulary partitioning, and artifact representation are unchanged.

## Separate storage from admission

The direct evaluator still admits at most 128 paths, with the same depth and
total-stack-work budgets. Its path vector holds four paths inline and retains
larger allocations in per-sequence `MaskScratch`. Inline capacity is not a
grammar restriction or a new fast-path limit.

The wrapper leases that buffer without keeping the scratch mutex locked during
evaluation. It clears and returns the buffer after both success and conservative
decline. No parser states, exclusions, or computed masks are reused through this
storage: only its capacity survives. Existing mask-cache generation rules remain
responsible for semantic cache reuse.

## Isolate rare planning work

The prior direct function reserved a 0xdd08-byte stack frame before its first
eligibility test. Its large inline path and 1,024-operation planning buffers
therefore burdened calls that needed no repeated-stack plan.

The new kernel collects paths separately. It enters an outlined planning helper
only when path count and total work can satisfy the existing reuse predicate.
That helper retains the original exact stack fingerprint/equality checks,
qualification rules, instruction capacity, and ordinary-walk fallback. A failed
plan never writes the output. No approximate mask or post-filter was added.

## Validation boundaries

The frame was large in both preserved older binaries; this observation is not a
proof of what caused their measured static slowdown. New generated-code inspection
and paired runtime measurements are needed to establish the effect of this change.

The test plan covers the full suites, sparse and full-vocabulary masks, compiled
composition, fresh and reloaded constraints, retained wide-path capacity, and
failure after partial admission. Performance comparisons preserve both preceding
binaries and unchanged native timers. A separate sequence-lifecycle measurement
includes setup, first mask, and destruction so retained scratch cannot hide costs
outside the mask timer.

This note records the design, not acceptance. Consult the source-hashed evidence
under `/root/direct-path-cleanup-789322` for completed results and limitations.

## Recorded checkpoint results

The release build passed the full workspace check, 2,202 Rust tests, 139 Python
tests and one doctest. The 41 existing ignored Rust tests remain; nine nested
subprocess summaries were excluded from the count. All 526 frozen source
inputs were verified. API reflection, 38 full-vocabulary constraint cases and
24 private partition cases over 128,256 tokens match the parent. Current-format
loading passed 16 artifacts in each direction.

The x86-64 direct entry reserves 3,336 bytes instead of 56,584 bytes in both
preserved older builds. The optional outlined planner reserves 35,272 bytes.
These exclude saved registers and callee frames; they are not wall-time claims.

Three 40-case-mode comparisons used the same ten selected schemas, four modes,
four build pairs and seven replays. Complete masks and outcomes match in all
three. Ratios below are per-position stabilized p99 mask-plus-commit results,
not unfiltered maxima or the full CFA population.

| Mode | Fresh vs parent | Reloaded vs parent | Fresh vs older coherent | Reloaded vs older coherent | Fresh identical-parent control |
|---|---:|---:|---:|---:|---:|
| FAST_BUILD | 0.9331 | 0.9878 | 0.4185 | 0.4683 | 0.9343 |
| O2 | 1.0342 | 0.9497 | 0.8860 | 0.7529 | 1.0428 |
| FAST_RUNTIME | 0.9389 | 0.9775 | 0.9497 | 1.0200 | 0.9585 |
| AUTO | 0.9739 | 0.9870 | 0.9296 | 0.9729 | 1.0574 |

This recovers the earlier static warning on the selected warm population, but
the control also varies and this is not a universal speedup. The three reports
retain 4, 7 and 1 local warm-TBM warnings respectively. Build and save/load
distributions, cold traces and all raw observations remain in the evidence.

A separate lifecycle probe includes state creation, the first mask call and
destruction: five schemas, four modes, fresh/reloaded states and 128 observations
per case/variant. Median per-case lifecycle ratios versus the parent are close
to one across modes, but four individual medians remain more than 10% slower
(static loaded case 9, static fresh case 14, and AUTO fresh/reloaded case 37).
The AUTO/reloaded case-37 median was 80.902 to 103.565 microseconds. Those
warnings have not been dismissed or hidden outside a mask timer.

## Independent correctness finding still open

An analytical recursive-language check found an inherited discrepancy in both
the parent and this checkpoint: for `root ::= "a" root "b" | "z"`, the static
and AUTO initial masks omit the valid fused token `aazbb`, while committing
that token succeeds. Dynamic/O2 and the sampled later prefixes match the oracle.
The ordinary and profiled paths both exhibit the omission, even with the direct
fast path disabled through profiling. The original independent failure and the
minimal classification are retained. Do not claim universal correctness from
matching earlier outputs, weaken the oracle, or add a runtime post-filter.

This is a preserved refactor checkpoint, not a main-branch release approval.
The inherited recursive-mask omission, individual lifecycle warnings and broader
performance coverage remain explicit follow-up work. No new public API or artifact
version is introduced. Native SHA-256:
`ca569f28f10b97556bd2daf843e255f8460959f1b9584e87da7299f75becd93c`.
