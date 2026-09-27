# Import only escaping parallel-normalization coefficients

This is an evaluation/storage optimization inside the ordered parallel native
parser normalizer. It does not remove grammar paths or change parser states,
LR labels, explicit guards, DEFAULT handling, or public artifact formats.

## Why temporary values were expensive

A parallel worker reads an immutable snapshot of the global Boolean-weight
interner. It computes intersections and unions in a private overlay. After
workers finish, a sequential publisher assigns deterministic parser-state IDs
in the original FIFO/label order, so all target-identity observations are
preserved. Previously the publisher imported every private bitvector, including
arithmetic intermediates that were never referenced outside the worker.

The realistic selected-ten fixture produces 5,494 private values under that
policy. Importing them takes roughly 2.7–3.0 ms. A large fraction are temporary
fold accumulators rather than coefficients of a published frontier or edge.

## Exact escape set

Let the packet's immutable values be `V[0], ..., V[k-1]`. A coefficient ID with
bit 31 set denotes `V[id & !(1 << 31)]`; other IDs refer to the immutable global
snapshot. Define `E` as the set of local IDs referenced by any of:

1. A prepared row's final coefficient or outgoing edge coefficient.
2. A pending known/singleton target's coefficient.
3. A retained closure recipe's incoming frontier, closed frontier, or incoming
   coefficient union.

`Packet::escaping_values` builds this exact conservative escape set. It marks
all recipes even when a later global cache hit could make a recipe redundant.
The scan is part of timed composition work, not free preprocessing.

`import_private_values` interns only marked `V[i]`, in increasing `i` order.
For unmarked values, the translation table contains `u32::MAX`, not the EMPTY
coefficient ID. `translated` asserts that an invalid translation never escapes.
Global IDs pass through unchanged. Local indexes, recipe indexes and translation
shape are checked; global interner failure causes a native-backend decline
before the packet's rows are published.

There is no need to recursively trace the operands that computed a marked
value. `V[i]` is a complete bitvector, not an expression that depends on retained
operand objects. Importing its value is sufficient.

## Preservation argument

All data consumed by ordered publication references either a global value or a
marked local value. Exact interning maps each such reference to the same finite
bitvector. Thus the publisher sees the same incoming and closed frontiers,
singleton decisions, and edge/final coefficients as under all-value import.
Rows and labels are visited in the same order, so new deterministic states are
assigned the same IDs. The existing possible-read, DEFAULT, final-subtraction,
fallback and final-minimization stages receive identical graph identities and
coefficient values. No weighted-language-only quotient is assumed.

Unused global weight IDs may disappear, and numeric coefficient IDs may change.
Validation therefore compares every ordered label and target ID, every decoded
coefficient and the original-coordinate decoder. Final artifacts must also be
byte-identical. The latter requirement prevents an unnoticed downstream
dependency on historical interner contents from being accepted on algebra alone.

Not importing an intermediate can make a later packet recompute it. In the
measured fixture, private computed values increase from 5,494 to 6,932, but only
2,010 are imported globally. This tradeoff is included in all timings; a lower
import count is not itself a speedup certificate.

## Selection and bounds

`GLRMASK_BOUNDARY_PARALLEL_LIVE_IMPORT` follows the ordinary default-enabled
Boolean policy. `0`, `false`, `off` or `no` restores the all-value reference.
It is consulted only when the existing parallel normalizer is eligible:
finite input, 4,096–200,000 source states, alphabet at most 32,768, and at least
four Rayon workers. Small inputs and one-to-three-worker pools remain serial.
The caller's thread count is not changed.

Private values remain limited to 8,192 per packet; existing packet-member,
transition, work, and global-native limits remain active. Marks and translations
use one entry per private value. Exhaustion aborts the private native attempt
through the existing exact fallback; it is never treated as grammatical rejection.

## Validation and measurement

The tests exercise all escaping fields, global IDs, unused temporaries, invalid
references and failed global imports. Generated multiword graphs compare both
import policies against serial normalization, including epsilon transitions,
explicit zero/DEFAULT guards, multiple starts and cross-packet convergence.

The initial three cohorts cover 416 interleaved prepared-component compositions
on current and legacy fixtures at ten and four workers. All native graph and
decoder comparisons, finalized artifact bytes and loaded masks at 22 anchors
match the independent published parallel reference. The larger ten-worker
cohort's paired same-binary savings are approximately 2.0 ms current and 2.3 ms
legacy. Four-worker results are positive but noisier. These are modest gains,
not a claim of reaching a 50–100 ms whole-build target.

The normalizer's historical weight table shrinks from 7,872 to 5,022 entries.
The graph remains 33,837 states / 301,384 edges, and the final minimizer produces
the same 1,231 states / 33,534 edges. Initial grammar/constraint preparation and
artifact serialization are excluded from `compose_ms`; all worker preparation,
escape marking, import, ordered publication and destruction are included.

The portable runner `scripts/compare_parallel_live_import.py` records source,
executable, runner and input hashes, paired raw timings, exact native checks,
and loaded-mask signatures. The final publication evidence is retained in
`docs/performance/boundary-parallel-live-import-20260927.json`.
