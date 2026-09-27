# Virtual template adjacency for boundary parser construction

## Scope and policy

The native boundary compiler can read its existing signed template-substitution
graph without first allocating every copied signed row. It then allocates only
the reachable positive rows, after cancellation, finality propagation, and the
original terminal/DEFAULT classification have finished.

`GLRMASK_BOUNDARY_VIRTUAL_TEMPLATE_GRAPH` is enabled by default. Setting it to
`0`, `false`, or `off` restores eager signed expansion. The virtual path requires
the existing memoized cancellation and positive-trimming policies. Early signed
support experiments and the independent cancellation replay validator retain
the eager path. Unsupported or resource-exhausted private native compilation
declines into the existing compiler fallback; it never publishes a partial mask.

This changes representation and evaluation order, not grammar assumptions. It
does not merge LR states, identify signed or deterministic graph states, remove
grammatically questionable terminal paths, or replace the existing template
automata with another stack-effect summary. There is no cross-call cache.

## Exact virtual graph

For instance `i`, let its template have `n_i` local nodes, global base `b_i`,
coefficient `c_i`, and continuation port `k_i`. Global node `b_i + j` retains the
identity of local node `j` in that instance. Each template transition to `t` is
read as a transition to `b_i + t`, stamped with `c_i`. Epsilons are translated
similarly. Every *present* template final, including `Some(empty)`, contributes
an epsilon to `k_i` with coefficient `c_i`. Template weights are deliberately
ignored, exactly as in the eager substitution builder.

Ports retain their original finals and identity entry epsilons. Port, instance,
start, label, and branch ordering is unchanged. In particular, a nonempty branch
group stamped with a zero coefficient becomes one `(target=0, weight=0)` marker;
an originally empty group remains empty. Neither is silently equated with an
absent label key.

The representation stores each supplied template topology once, instance
descriptors, port rows, and an integer node-to-instance index. Its read-only
`SignedGraph`/`SignedRow` interface is also implemented for the eager slice, so
the memoized cancellation algorithm uses the same recurrence on both readers.

For a requested read `a`, that recurrence is:

```text
E(q)    = original_eps(q) union contributions of balanced push/read excursions
R(q,a)  = direct_read_a_or_DEFAULT(q)
          union over (r,w) in E(q) of w intersect R(r,a)
```

Only row access changes. Logical node identities, the original DAG order,
read-filter observations, result ordering, and Boolean weight operations are
preserved. Finality propagation still sees original negative and DEFAULT edges
as well as original and derived epsilons. Negative keys are not prematurely
discarded at a fragment boundary.

## Why allocation can be delayed

The original resolver computes a least fixed point describing terminal states
whose DEFAULT branches can be removed. This computation observes literal
epsilon presence and branch-group presence, including zero-weight branches.
Those branches may point backward even though the nonzero graph is acyclic.
Consequently the virtual path uses the same dependency-count worklist, over the
complete logical graph, rather than a guessed reverse-DAG simplification.

Only **after** this classification and its branch-pruning predicate are fixed
does the compiler compute positive nonzero-edge reachability. It allocates rows
for that reachable set in increasing original-node order. The renumbering is
injective: two surviving nodes never become equal. Nonempty zero-only groups
retain their guard label with the same single dummy edge used by ordinary trim.

The existing certified read-context pass follows only nonzero edges from the
same starts. An unreachable node cannot contribute a context to a reachable
one. Therefore:

```text
read_context_then_trim(full_positive_graph)
  = read_context_then_trim(reachable_positive_graph)
```

up to the intermediate injective renumbering. The final increasing-ID trim
composes those renumberings into the same final IDs as the eager construction.
Starts and any reused topology certificate are remapped together. The input to
weighted determinization is unchanged; its support/guard observations, singleton
decisions, DEFAULT optimization, fallback determinization, and minimizer remain
the existing implementation.

## Resource and validation contract

Both logical instantiated size and retained unique-template topology are
bounded. Preflight checks include terminal labels, targets, starts, port indices,
coefficient indices, entry ranges, checked size arithmetic, and nonzero-edge
acyclicity. A reused topological order checks derived cancellation edges. The
existing finite interner and work limits still apply.

Tests compare eager and virtual constructions node-for-node over 512 generated
programs, including zero-coefficient instances, present-empty finals, empty
keys, multiple template starts, and shared continuations. They compare Kahn
orders, cancellation results, complete weight interner sequences, finality,
pruned positive graphs, and compacted read-context results. A separate 512-case
literal least-fixed-point test includes zero and backward DEFAULT dependencies.
Additional tests cover multiple program starts, topology reuse, cycles, invalid
ports, and unused-topology resource limits.

The real-fixture gate compares every normalized native node's labels, target
IDs, and decoded coefficient vectors. Every final serialized artifact must also
match an independently frozen baseline byte-for-byte. Equal state counts alone
are not an equivalence test. Runtime replay and whole-link measurements are
separate from profiling and validation.

## Initial measured result

The D1 experimental screen used 12 interleaved rounds, four modes, and two
prepared-fixture generations: 96 timed links in total. Its base was `09fe81ac6`.
The current fixture median was 293.36 ms eager versus 278.34 ms reachable-only;
the legacy fixture was 294.19 ms versus 280.65 ms. Median paired savings were
19.385 ms and 12.4475 ms respectively; identical-eager controls independently
gave 13.659 ms and 17.3645 ms savings. These are whole-link times excluding
artifact serialization, not Rust executable compilation times.

The virtual signed representation retained 21,975 template nodes and 76,134
template edges for a 105,276-node/447,529-edge logical program. Positive-row
allocation fell from 105,276 nodes/380,261 edges to 45,104/154,283 before the
unchanged read-context pass. Positive copying fell from about 14.8 ms to 4 ms;
read-context processing from about 9–11 ms to 3.9 ms. The final normalizer input
remained exactly 24,043/110,400 and its output exactly 33,837/301,384. The final
parser remained 1,231/33,534, with identical artifact bytes.

Virtualization **without** early positive reachability was tested separately
and did not establish a useful speedup: it merely moved the large allocation
cost to a later stage. That variant is not a user-selectable production mode.

These two fixture generations do not establish a universal performance bound.
The 10–20 ms whole-link goal remains unmet. Final publication measurements and
source/build provenance are recorded separately under `docs/performance/`.

## Reproduction

Build the ordinary examples and keep a frozen pre-change builder:

```text
cargo test --release -p glrmask-parser-dwa --features internal-api --lib -- --test-threads=1
cargo test --release -p glrmask --features internal-api --lib -- --test-threads=1
cargo build --release --features internal-api --example composition_build_static_artifact --example composition_loaded_static_probe
python scripts/compare_virtual_template_graph.py --candidate CANDIDATE --baseline FROZEN_REFERENCE --runtime RUNTIME_PROBE --inputs current=CURRENT_PREPARED --inputs legacy=LEGACY_PREPARED --output NEW_RESULTS --pairs 16
```

Each prepared directory contains `core.bin`, `dispatch-literal.bin`, and
`vocab_dump.bin`. The runner clears stale experiment variables and uses the
checked-in validated boundary profile. It compares frozen reference,
same-binary eager, identical eager control, and default virtual construction.
It requires all artifact hashes to agree, runs loaded-mask/commit probes, and
retains raw timing samples, input/executable hashes, and paired differences.
An optional `--native-checker` accepts the diagnostic normalized-row comparator;
the ordinary byte-for-byte artifact gate remains mandatory.

## Final publication gates

The cleaned default-on implementation passed ordinary Cargo release tests:
924 root tests and 57 parser tests, zero failures, with 51 and 2 existing
ignored tests respectively. The normal public library also passed its
no-default-features check. After merging the independent GLR changes through
`2bf9cbaaa`, both full suites passed again and both release examples rebuilt.

The primary final matrix contains 128 interleaved links (16 rounds, four modes,
two independently prepared fixture generations). Same-binary eager versus
default medians were 294.7095 vs 275.041 ms current, and 301.75 vs 277.3475 ms
legacy. Median paired savings were 16.925 and 17.933 ms, winning all 16 pairs
in each fixture. The identical eager control independently confirms the gain.

A further 48-link integration matrix on the merged source also preserved every
artifact byte and exact normalized node/target/coefficient/decoder identity.
It won all six eager/default pairs per fixture. Loaded-runtime probes at all
22 anchors agree in both matrices (151 repeats per anchor, four then two
process rounds per artifact mode). Since the serialized artifacts are
byte-identical, no runtime representation or mask semantics changed.

The frozen historical reference binary is based on `09fe81ac6`; the primary
candidate was built on `96d5abb74` plus the focused virtual patch, and the
integration candidate on `42706c5bb` after merging `2bf9cbaaa`. The same-binary
eager/default comparison isolates virtualization from unrelated main changes.
Do not add improvements from earlier, separately measured cohorts. Raw build
samples, all source/fixture/executable hashes, and integration outcomes are in
`docs/performance/boundary-virtual-template-20260927.json`. The target of
10-20 ms whole link remains unmet.
