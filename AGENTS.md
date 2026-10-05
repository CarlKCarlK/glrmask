# GLRMask development policy

GLRMask is pre-release and has no existing consumers. Historical artifact
formats are not compatibility requirements. Reject unsupported old formats
explicitly instead of preserving obsolete adapters for hypothetical consumers.

Ordinary Dynamic (O1) retains and executes its LR table by default. O2 and the
public FastBuild path use native template parsers; Static behavior is unchanged.
`GLRMASK_DYNAMIC_TEMPLATE_DFA=1` is an internal development override for
ordinary Dynamic only. Resolve this choice before worker pools, pass it through
shared preparation and assembly, and never let it alter O2, Static, existing
objects or loaded artifacts. There is no new public backend selector.

Native construction derives templates from temporary compiler LR analysis,
discards the table, and materializes only a table-free Constraint. Do not create
an LR-backed intermediate for a native compile. Native runtime table access and
implicit LR fallback must still panic. The controlled
`ParserTableStorage::explicit` constructor is for retained Dynamic LR and its
versioned artifact loaders; generic `From<GLRTable>` remains forbidden. Shared
compiler, vocabulary, masks and commits must not become two forked pipelines.
Backend selection only controls necessary normalization and representation work.

Template components must link without retaining or reconstructing executable LR
action/goto tables. Preserve grammar/interface analysis metadata needed by
root-CALL, follow and scoped-adjacency optimizations separately from execution
tables. Share common lexical/query construction rather than silently omitting
optimizations in a second backend pipeline.

Native component compilers, providers and linkers retain no executable LR
action/goto tables. Preserve analysis/interface metadata separately. A native
artifact stays native and a retained-LR artifact stays LR when loaded,
independently of compilation environment. External artifacts validate the exact
vocabulary digest. Never silently mix incompatible parser representations.

Recognition, token masks, commit behavior and completion must remain exact.
Preserve full lexer-state and vocabulary equivalence semantics. A pruning proof
may conservatively retain extra candidates when unavailable; it must never
exclude a valid token. Benchmark genuinely precompiled components and report
component preparation separately from linking. Do not hide regressions by moving
work outside a timer or changing benchmark settings.

Use an owned, reusable development target with incremental compilation enabled
for implementation iterations. Measure warm small-edit rebuilds. The local
global sccache wrapper rejects incremental compilation; override RUSTC_WRAPPER
only for that development harness when necessary. Keep production release
settings for qualified performance comparisons. Preserve other workers' source
and target ownership. Do not run repository-wide formatting.

For the verified Windows root-unit fast loop, portable isolated-manifest
preparation, timing/coverage evidence, and narrow recoverable cache diagnosis,
read [docs/windows-development-builds.md](docs/windows-development-builds.md)
before starting Windows development builds. The unit-only recipe must not
replace canonical integration coverage or production performance profiles.

Final correctness qualification and matched build/link/TBM performance acceptance
must use consolidated, batched canonical builds about once or twice per day,
not multi-minute production rebuilds for each experiment. Follow the Windows
guide's user-required iteration budget and separate optimized development
performance discrimination from eventual release acceptance.

Final correctness qualification and matched build/link/TBM performance acceptance
are separate gates. Targeted tests alone do not qualify the full replacement.
Publishing packages, pushing, merging main or changing defaults requires current
user authorization.
