# Packed final-weight runtime caches

These caches accelerate the existing mask and commit paths after a constraint
is loaded. They do not define another parser backend, retain an LR table for a
template parser, or replace the canonical packed weight pool.

## Exact token projection

Let `I` be the internal token classes and `M(A)` the original-vocabulary tokens
represented by an internal subset `A`. For a live candidate set `D` and a final
weight's token set `T`, the required contribution is `M(D ∩ T)`.

The shared mask engine handles three cases:

1. For a small, cheaply expanded `T`, visit its members that are also in `D`
   and OR their existing original-token mask fragments into the result.
2. When `T ⊆ D`, use the precomputed original-token projection `M(T)`.
3. Otherwise, decline the cache and use the existing exact intersection and
   projection path. Declining the cache must leave the output unchanged.

The second case follows directly from `D ∩ T = T`; it does not assume that an
arbitrary parser action is admissible. An empty intersection is also a handled
contribution. The complete relation, not an approximate top-state test, remains
authoritative.

The projection cache is used only for the ordinary internal-to-original token
mapping. A constraint using the separate final-mask remapping representation
does not enter this fast path. Rebuilding caches after a mapping change detaches
shared metadata before publishing mapping-specific indices. Clones may share
an immutable cache when their mapping is unchanged.

## Packed weight lookup

For selected non-full weights, an immutable index stores their sorted TSID
intervals and token-set IDs. A binary search by interval end, followed by a
start-bound check, gives the same answer as scanning the packed intervals.
Each index retains the owning pool, so its IDs remain scoped to that pool.
Full weights and unindexed weights continue through the ordinary packed lookup.

An index is published only when the entire declared interval count was decoded.
A truncated decode does not become a shortened cached relation. This check is
a cache invariant, not a claim that this module validates every artifact field.

## Resource limits

The retained cache has an 8 MiB accounting budget, including conservative
per-entry overhead, and at most 16,384 indexed weights and 16,384 token sets.
Metadata inspection, weight-entry decoding, and token-range inspection have
separate caps of 262,144 entries per build pass. The builder reads encoded counts
before allocating decoded entry vectors or walking token ranges. An oversized
entry can therefore be skipped without first doing its expensive decode.

The token-work prefix and a single temporary decoded entry vector each have
an 8 MiB workspace ceiling. These are transient allocations, separate from the
retained cache budget; the retained budget is not a whole-process peak-memory
limit. Exceeding a cache budget retains the exact uncached execution path.

The shared sparse-projection helper preserves the ordinary materialized path's
early cardinality exit. Packed sets first check their encoded range count;
every nonempty range contributes at least one token, so a count over the sparse
limit can be rejected without scanning it.

## Exact tokenizer transition cache

Fresh and loaded tokenizers use the same compact direct-byte transition cache
when it is representable and fits the 8 MiB Flat16 budget. Virtual tokenizers
remain excluded. Larger or nonrepresentable tokenizers retain their existing
exact transition representation.

This is a runtime speed/space tradeoff: constructing the cache requires work at
load time and additional memory. Benchmark load cost as well as generation
latency. It is not a promise of cost-free loading.

## Serialization and regression coverage

These indices and projections are derived runtime data, excluded from the wire
format. The packed pool remains the serialized authority. An unchanged loaded
artifact must resave identically, including through a clone and through the
supported external-vocabulary path.

The `packed_final_mask_cache` Cargo integration test exercises LR and template
backends, valid and rejected prefixes, full-vocabulary masks, completion state,
cloned loads, and self-contained and template external-vocabulary round trips.
It also checks interval boundaries and randomized projection inputs against
the uncached decoder and original-token fragments. Cache-budget counters are
checked by the same internal diagnostic.

```sh
cargo test --release -p glrmask-weight --features internal-api
cargo test --release -p glrmask --features internal-api \
  --test packed_final_mask_cache \
  --test template_parser_provider \
  --test template_parser_artifact
```

Performance qualification must compare both parser backends in the same
executable, distinguish freshly compiled from reloaded constraints, retain raw
cold and warm CPU timings, and check the executed backend rather than infer it
from an environment variable. Hash-equivalent output is not a substitute for
full mask comparisons. Likewise, equal artifact lengths alone do not prove
equivalent contents.
