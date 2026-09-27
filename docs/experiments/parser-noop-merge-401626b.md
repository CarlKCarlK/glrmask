# Exact avoidance of redundant final-minimizer merges

Status: experimental, disabled by default. No speedup or completed validation is claimed here yet.

The candidate runs the unchanged `compatible` check and retains the same height
buckets, ordering, state identities, explicit guard keys and target identities.
After a successful comparison, it skips `merge` only if the candidate state's
needed domain is contained in the existing group's domain.

Let the state domain be d and group domain D. Backward clipping computes d as
the union of its final and outgoing coefficients. Thus every state coefficient
is contained in d; each group coefficient is contained in D by construction.
Compatibility establishes equality on d intersect D. If d is a subset of D,
every incoming coefficient is already contained in its group coefficient.
Every OR in merge is therefore an identity, including D union d.

For guarded rows, compatibility also checks the complete ordered key set.
For unguarded rows, zero coefficients have already been removed; a new nonzero
key would fail overlap equality. Shared nonzero target IDs must agree, and
zero targets have been canonicalized to zero. Consequently the entire stored
group is unchanged, not merely its accepted weighted language. No new mask ID
can be created by the reference merge, so skipping it also preserves the mask
value history relevant to later steps.

The first implementation retains the original merge as reference. Enable only
with GLRMASK_EXPERIMENT_MIN_REDUNDANT_MERGES=1. The separate profile flag
GLRMASK_PROFILE_MIN_REDUNDANT_MERGES reports successful merges, contained
cases, identical domains and candidate edge-vector work; its extra checks are
not enabled in latency comparisons.

Tests include 8,192 compatible contained-domain pairs, checking exact group
fields and interned-value counts after the original merge. Additional cases
cover growing domains, conflicts, explicit zero/default guards, and 1/2/17/44/64
coefficient words. Ordinary Cargo compilation is pending at creation.

The paired real-fixture runner is .benchmarks/noop-merge/compare.py. It retains
independent published binary, same-binary reference, identical reference
control and candidate arms. Every timed artifact must match the independent
same-thread reference byte-for-byte. Untimed gates compare all native graph
rows/labels/target IDs/decoded coefficients/original decoder and loaded masks
at 22 or more anchors. Timings include all composer work but exclude initial
constraint compilation and serialization; they are not complete cold grammar
build measurements.

Do not combine this experiment with unmeasured packet scheduling, early
state/domain pruning, changed grouping orders, or other workers' source.

## Completed experimental evidence

Normal Cargo: 958 root tests passed, zero failed, 51 pre-existing ignored; the
three new tests also passed independently. Both real fixtures: 9,192 successful
merges, 5,578 strict-subdomain no-ops, zero equal-domain cases (exact-signature
lookup already handles those), 267,019 avoidable old-plus-new edge visits.

The first 96-build whole-link screen was mixed; it is retained, not discarded.
The independent 256-build confirmation has paired same-binary reference-to-ON
savings 1.9445 ms current and 1.4475 ms legacy, 20/32 wins each. Identical-control
comparisons are noisier, especially legacy (+0.328 ms, 17/32). This does not
establish a large whole-build gain.

The separate 576-invocation source-included final-minimizer crossover is more
diagnostic: enabled sessions save about 1.67-1.70 ms in merging (24/24 wins in
each session), and 1.50-1.93 ms in total minimization. Disabled sessions are
near parity. Every invocation preserves the exact final parser; initial
checks also run the complete parser-prefix product. This phase experiment
excludes input decoding and is NOT a whole-link speed claim.

Every one of the 352 timed whole-link artifacts was byte-identical, and native
rows/targets/decoded coefficients/decoder and loaded anchor masks also matched.
Raw complete records: .benchmarks/noop-merge/{first10,confirm10,phase-crossover}/report.json.
Frozen reference/candidate hashes and source/compiler provenance are recorded
in those files; the phase runner resolves exact own Cargo fingerprints.

Verdict: keep the small, proved work-avoidance mechanism for final publication
gates, removing diagnostic counters. Do not claim the 50/100 ms objective is
met. No production default or push has occurred in this experiment snapshot.
