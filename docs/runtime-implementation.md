# Runtime implementation

`Constraint` owns reusable compiled data. `ConstraintState` owns one sequence's
mutable parser state and scratch space. Execution strategies are private; the
public API chooses a build/runtime trade-off rather than a concrete engine.

## Constraint responsibilities

`src/runtime/constraint.rs` is the lifecycle boundary. Its private children keep
construction, representation, and execution separate:

| Module | Responsibility |
| --- | --- |
| `vocabulary` | Vocabulary identity, token bytes, and token-coordinate relations. |
| `compiler_views` | Lazy reconstruction of compiler views needed for composition. |
| `parser` | Reads over materialized and packed parser representations. |
| `recursive_parser` | Component coordinates and exact scoped parser execution. |
| `regular` | Direct regular admission, advancement, and cached frontiers. |
| `dynamic_vocab` | Lazy vocabulary materialization, slicing, and trie construction. |
| `observations` | Terminal observations and retained dynamic proof preparation. |
| `cache` | Cache installation and preparation order. |
| `parser_cache` | Fast transitions and weighted parser-token masks. |
| `mask_cache` | Construction of vocabulary-coordinate mask caches. |
| `mask_replay` | Cost-directed replay into caller-owned mask buffers. |
| `weights` | Borrowed views of materialized and packed runtime weights. |

These are source modules within the existing crate, not additional crate
boundaries or heap-owned wrappers. Mask generation and token commitment remain
in their existing runtime modules. Compiler prebuild types and borrowed weight
views retain their original crate-private import paths.

## Invariants

**Vocabulary coordinates are not interchangeable.** Internal token groups,
model token IDs, tokenizer states, and component-local parser states have
different domains. Sparse IDs and duplicate byte spellings must survive mask
expansion. Compiler-only placeholder IDs must not enlarge runtime mask buffers.

**Deferred data must stay deferred.** A loaded or static constraint need not
construct a dynamic vocabulary, finite residual projection, or recursive
compiler table until a consumer requires it. A prepared empty proof is not an
unprepared proof. Vocabulary-independent cache construction must not quietly
become grammar-dependent first-mask work, or vice versa.

**Caches have an owner.** Reusable immutable artifacts may be shared. Mutable
sequence frontiers and scratch buffers must not leak between states. A dirty
caller-owned mask buffer is overwritten completely, including its unused tail.

**End tokens are root policy.** A child's standalone termination policy is not
its grammar body. Linking strips that policy; final root compile/link options
select end IDs. Model controls should be exact-only IDs, not invented ordinary
byte spellings. An ordinary empty byte token is a separate semantic edge case.

**Runtime fast paths require exact fallback.** Parser-relative liveness and
lexical projections certify that work may be skipped; an unavailable certificate
must not be interpreted as an empty language. Recursive component stacks must
retain their scoped correlations instead of becoming unrelated state sets.

## Retired construction code

The former private `build_dynamic_self_loop_projections` cluster had no callers
in the pinned `7e84a64c9` source. Its candidate discovery, standalone proof
builder, experimental lexical-effect builders, and unused alias builder were
removed, along with other private helpers that had no references. The follow-up
removes the orphaned self-loop projection types, accessors and four always-empty
cache fields. All construction and restoration paths initialized those fields
empty, and only the removed, uncalled setters could populate them. The remaining
profiling diagnostic therefore keeps its existing `projection=none` output.

This does not remove live dynamic projection support. Bounded observations,
terminal quotients, virtual residual preparation, pending guards and current
artifact validation remain. Vocabulary-equivalence analysis is unchanged and
remains internal. The private in-memory cache layout changes, so source-level
reachability is not by itself proof of performance preservation. See the
[cleanup validation checkpoint](runtime-cache-cleanup-validation.md).

Changing responsibility boundaries does not justify deleting a live fast path.
Validate moved function bodies and public signatures separately from intentional
removals, then run correctness and same-host performance checks. Module moves
can affect compiler layout even when the source algorithm is identical.
