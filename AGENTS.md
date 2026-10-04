# GLRMask development policy

GLRMask is pre-release and has no existing consumers. Old artifact save/load
formats, the LR runtime backend, and table-to-template conversion are not
compatibility requirements. Do not add compatibility adapters or preserve
obsolete formats solely for hypothetical users. A Constraint must never be
materialized with an LR table. LR-backed Constraint construction, loading,
runtime execution and fallback must panic loudly. Temporary LR tables inside
compiler analysis are allowed: derive templates from compiler parts, discard
the table, then construct the Constraint. Do not construct an LR-backed
Constraint as an intermediate and convert it. Historical LR measurements are
reference records. Reject unsupported old formats explicitly.

Template components must link without retaining or reconstructing executable LR
action/goto tables. Preserve grammar/interface analysis metadata needed by
root-CALL, follow and scoped-adjacency optimizations separately from execution
tables. Share common lexical/query construction rather than silently omitting
optimizations in a second backend pipeline.

The source compiler, component compiler, linker, artifact loader and acceptance
harness must materialize only native template Constraints. Do not weaken an LR
materialization/runtime panic to make a test pass. Grammar rules,
nullable/FIRST/FOLLOW information, interface metadata and temporary compiler LR
analysis are permitted; executable LR tables must not enter a Constraint.

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
