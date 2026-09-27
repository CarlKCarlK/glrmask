# Read-context propagation before positive graph allocation

This extends the virtual signed-template construction described in
`boundary-virtual-template-graph.md`. It changes when the existing certified
read-context filter is evaluated, not the set of grammar paths it accepts.

## The work removed

The selected10 dispatcher has 105,276 logical signed nodes. The virtual
template builder stores its 21,975 distinct template nodes once, performs the
same cancellation and finality calculations, and previously materialized
45,104 reachable positive nodes with 154,283 edges. The existing read-context
filter and a second trim then reduce that graph to 24,043 nodes and 110,400
edges before normalization.

The new path evaluates that same filter on the immutable virtual positive
graph. It allocates only the 24,043 rows that survive. This removes an
intermediate graph, a separate positive reachability traversal, and the later
mutable filtering/compaction pass. It does not reduce the normalized graph
by identifying states, and it does not infer stronger grammatical relevance.

## Input contracts

`VirtualSignedGraph::build` checks every template, label, target, instance,
entry, continuation, size bound and the complete nonzero-edge DAG. The
cancellation solver adds only forward derived epsilon edges. Before this
consumer runs, either a fresh topological sort checks the whole epsilon
overlay, or the reused rank certificate checks every derived edge.

Finality is computed on the complete graph. Terminal/DEFAULT classification
also runs on the complete finalized graph, including backward **zero-weight**
DEFAULT guards. Neither calculation is restricted to context-reachable rows.

`FiniteParserReadSupport` remains a separately effect-certified, conservative
domain supplied by the caller. Its checked representation has a root context,
a bitset of permitted source contexts for each LR label, and one residual
context per read label. Root dominates other contexts. The new path requires
the domain alphabet to match the virtual graph's checked alphabet. A missing
or mismatched certificate selects the previous materialize-then-filter path.

## Transfer equations and proof

Let `C(q)` be the abstract contexts arriving at logical node `q`. Initially,
the graph's start nodes receive `{root}` and other nodes receive the empty
set. Process nodes in the checked forward topological order.

For a retained, nonzero epsilon edge `p -> q`, join `C(p)` into `C(q)`.
For a retained, nonzero read edge labelled `a`, transfer `target(a)` when
`C(p)` intersects `allowed(a)`. A DEFAULT read transfers `root`. These are
exactly the rules in the old mutable `finite_read_support::restrict` pass.
The domain's existing root-dominance rule represents any join containing
root as `{root}`.

By induction on topological order, all predecessor transfers agree before a
node is visited, so its context set and permitted reads agree with the old
pass. A context is nonempty exactly when that node survives the old read
filter and positive reachability trim. Mapping surviving original IDs in
increasing order therefore produces the same IDs as the two old increasing-ID
compactions composed together.

Only allowed nonzero branches are copied. A present branch group whose
coefficients are all zero after filtering remains an explicit `(0, 0)` guard.
Truly empty groups and DEFAULT branches removed by the unchanged terminal
classification retain their prior treatment. All copied nonzero targets must
have a surviving mapping; this is checked before an output row is published.

Context propagation interns no weights. Cancellation, finality, labels,
branch order and guard classification are unchanged. The normalizer thus
receives the same positive graph and numeric weight table, a stronger
contract than equality of sampled accepted token sequences.

## Sparse context storage and bounds

An `offsets[logical_node]` entry is `u32::MAX` until a successful transfer
supplies a nonempty context. Such a transfer allocates one bitset block and
immediately writes the context; an allocated block never represents an empty
set. Loading an unreached source requires no bitset scan or copy. Blocks are
held in a contiguous vector and original node IDs are never changed during
propagation.

The private context store checks node, target and word limits. It admits at
most 200,000 logical nodes and 8,000,000 context words. A failed allocation
check returns before installing an offset. Exhaustion abandons the private
compiler attempt; it never publishes a partial parser. The independently
checked eager path remains the fallback.

## Defaults and reproduction

`GLRMASK_BOUNDARY_VIRTUAL_READ_CONTEXT` is enabled by default. Set it to `0`,
`false` or `off` to retain virtual template construction but restore the old
post-materialization read filter. Disabling virtual template construction
itself also disables this path. Existing unsupported early-support modes keep
their previous eager behavior.

`scripts/compare_virtual_read_context.py` compares a frozen D1 builder,
same-binary D1, an identical D1 control and the default candidate. It checks
every timed artifact hash against its independently validated output, can
compare all normalized native rows and coefficients, and replays loaded
artifacts at all returned runtime anchors. Profiling and validation are
excluded from link timings; serialization is reported separately.

The independent topology-reuse policy can be held ON with `--reuse-topology`.
Otherwise the runner removes that selector. Its historical implementation is
presence-based, so setting its value to `0` would **not** disable it.

## Validation before publication

The research version passed 512 generated template programs, each under both
topological scheduling policies, comparing complete positive rows, guard keys,
starts, target IDs and the full native weight-ID sequence against eager
construction followed by the original read filter and trim. Sparse-storage
tests exercise root dominance, unreachable nodes, invalid indices and forced
word-limit failure before offset publication.

Both selected10 fixture generations also retain all 33,837 normalized states,
301,384 edges and original decoded coefficients. Serialized final artifacts
are byte-identical to published D1. The initial 128-link screen and independent
288-link confirmation show a modest additional gain, not a major reduction in
the remaining normalization cost. Cleaned-source full-suite and final default
measurements are recorded in the accompanying publication evidence.

## Final publication measurements (Windows, 27 September 2026)

The cleaned default-on source passed normal Cargo tests: **59 parser tests**
and **924 root tests**, with zero failures (2 and 51 pre-existing ignored
tests respectively). Both release examples built successfully.

The final comparison ran 24 interleaved rounds of four policies on each of two
prepared fixture generations: **192 whole-link builds**. Every artifact was
byte-identical to the frozen published D1 output. Native comparison preserved
all 33,837 normalized state IDs, 301,384 labelled targets, decoded coefficients,
and the atom decoder. Eighteen loaded-runtime runs covered 22 anchors each.

| Fixture | Frozen D1 median | New default median | Median paired saving | Faster pairs |
|---|---:|---:|---:|---:|
| Current | 275.6805 ms | 272.0245 ms | 5.5600 ms | 18/24 |
| Legacy | 276.6415 ms | 274.8270 ms | 2.9775 ms | 19/24 |

Against the same-binary D1 arm and its identical control, current paired
savings were 6.657/5.520 ms; legacy 5.687/5.1355 ms. These figures vary across
cohorts: the preceding 288-build confirmation showed roughly 2?4 ms. Do not
add gains from different cohorts or treat the most favorable number as a
universal improvement. The 10?20 ms whole-link target remains unmet.

Raw final samples, fixture/source/binary hashes, policy flags, loaded-runtime
observations and validation metadata are in
`docs/performance/boundary-virtual-read-context-20260927.json`.
