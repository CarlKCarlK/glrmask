# Windows optimized development builds

Use an owned reusable incremental target for implementation iterations. Do not
run a full sweep per edit. Development correctness and canonical production
performance acceptance are separate gates.

## User-required iteration budget

The user explicitly requires multi-minute canonical builds to be batched into
consolidated validation about once or twice per day, not run per experiment.
Do not start another expensive canonical rebuild until a genuinely fast
optimized experiment loop is operational, unless an exceptional necessary
build is explicitly explained. Keep the experiment's flags, dependencies,
manifest identity and owned caches stable. Use existing frozen binaries and
harness controls whenever possible. Development-profile paired timings may
discriminate hypotheses, but are never production performance acceptance.

The 35-second measurements below changed one literal. They are not a promise
for expanded optimized code generation: the added bounded diagnostic rebuilt
the root in 281.14s (144 Fresh dependency artifacts), and a generic producer/
consumer refactor intentionally rebuilt parser-DWA plus the root in 294.39s.
The refactor reversal plus a frontend call-site correction took 299.734s.
Those runs consumed substantial CPU, with no observed locks or duplicate root
builds. Identify root/dependency/codegen/link invalidation separately; do not
erase warm caches or call this evidence of a stall. Persist each iteration's
source hashes, planned rebuilds, phase times, CPU, and profile limitations.

A genuine two-line frontend callee/import change subsequently passed its
focused unit check in **86.672s** (85.156s compilation, 135.719 owned CPU seconds).
It rebuilt only the root: 144 Fresh artifacts, 122 Fresh dependency units,
one root compiler command, no dependency rebuilds. Receipt:
`diagnostics/fast-unit-current/warm-real-frontend-baseline-r1-identity.json`.
The experiment's source was restored byte-for-byte afterward. This is not a
verified fast non-test performance loop and must not be described as one.

The attempted non-test opt2/CGU256 setup changed the ordinary Cargo feature
graph relative to libtest and rebuilt dependencies; an initial 180s guard and
one unchanged-cache 180s continuation both stopped before compiling the root.
A separate root-only replay with the exact cached libtest extern graph (no
Cargo or dependency builds) reached metadata but hit its 90.218s guard before
finishing the library. All three owned trees drained cleanly. Preserve these
receipts and caches; do not repeat this setup blindly or count timeout samples
as completed warm edits. The short setup guards were counterproductive: the
user clarified that progress checkpoints detect oversights, not that a known
cold build should be killed merely for reaching 90/180s while consuming CPU.

After that clarification, the SAME root-only setup completed in **146.656s**
under an explicitly justified 600s bootstrap budget. A genuine two-line
frontend callee/import edit then rebuilt the optimized non-test library in
**7.266s**, and its byte-identical restoration in **6.953s**. Both used zero
dependency builds and zero Cargo invocations. Receipts are
`direct-core-stable-bootstrap-r2-direct-identity.json`,
`direct-core-real-frontend-baseline-direct-identity.json`, and
`direct-core-real-frontend-candidate-restored-direct-identity.json` under
`diagnostics/fast-unit-current/`. This supersedes the earlier unproven-loop
checkpoint; the Mac's independently measured workflow is not a Windows estimate.

The stable non-test command, from the task workspace, is:

```text
python diagnostics/fast-unit-current/run_direct_perf_core.py --label UNIQUE-LABEL
python diagnostics/fast-unit-current/link_direct_perf_runner.py --label UNIQUE-LABEL
```

It replays the exact authenticated cached unit extern graph, removes libtest,
and changes only the root to opt2/CGU256/incremental/LTO-off. Dependencies retain
their cached optimized opt3/CGU16 settings. It validates dependency source and
manifest hashes before reuse; edits in `crates/` require a planned dependency
refresh, not a root-only replay. Keep flags, metadata identity, owned incremental
directory and extern hashes unchanged between edits. The default 600s guard is
a final safety bound; inspect CPU and communicate progress at checkpoints.
The successful native canonical-harness adapter link took **2.310s** after
its one-time common adapter setup (0.670s). An initial adapter link failed only
because the frozen filename lacked the required `lib` prefix; a verified
byte-identical alias fixed it without rebuilding the core or dependencies.
These development timings are hypothesis discrimination, never release acceptance.

## Measured R1 unchanged-dependency experiment

The October 4 experiment changes only which grammar rules are rescanned. Every
round still reads the unchanged current nonterminal-summary generation; all
rules run on the first round, then only rules using a changed RHS nonterminal
run. Unioned prior contributions remain in the next generation. The original
256-round cap, convergence test, top widening, and candidate mapping remain
unchanged. Temporary reverse dependencies are compiler analysis, not retained
runtime state or a new parser representation.

`GLRMASK_VALIDATE_BOUNDARY_R1_FRONTIER=1` compares every generation's complete
summary map with the original full-rule evaluator. Disable it for timings:
reference validation deliberately does the old work as well. Native Kubernetes
(256 rounds, capped), service-schema (21), and jreleaser (5) all matched. The
separate exact-source algebra probe passed 135 adversarial/generated graphs and
1,093 generations, including productive/unproductive/nullable/event cycles,
duplicate RHS users, sparse IDs, uncapped chains, and a capped reverse chain.

Under identical development profiles, two balanced AB/BA pairs per fixture
showed quiet median build times 1308.529→788.646ms, 646.867→590.881ms, and
279.367→266.550ms, respectively. All six paired differences improved. The
profile-on fixed-point intervals were 583.731→84.904ms, 100.489→23.057ms, and
15.101→10.781ms. Those narrower intervals exclude reverse-dependency setup;
the quiet whole-build timers include all setup and are the discriminating
comparison. These are NOT production performance acceptance or full-sweep data.

Successful frozen labels: baseline `direct-core-r1-cost-instrumented`, candidate
`direct-core-r1-frontier-prototype`. The candidate's root rebuild took 21.921s
with no dependency rebuild. No canonical rebuild was used for this experiment.
From the task workspace, the exact completed validation commands were:

```text
python diagnostics/fast-unit-current/profile_direct_outer_phases.py --runner-directory direct-core-r1-frontier-prototype-runner --output r1-frontier-generation-validation --validate-generations
python diagnostics/fast-unit-current/control_r1_frontier_builds.py
python diagnostics/fast-unit-current/verify_r1_frontier_adversarial.py
python diagnostics/fast-unit-current/link_direct_perf_runner.py --label direct-core-r1-frontier-prototype --suffix=-masks --capture-adapter
python diagnostics/fast-unit-current/verify_r1_frontier_full_masks.py
```

These commands preserve their existing receipts and refuse to overwrite them;
they are evidence of completed runs, not instructions to rerun them blindly.
The final command reused the immutable baseline capture and compared one new
candidate capture: 21,283 frames and 85,302,264 actual u32 mask words matched
exactly. The source-only generation probe is not itself a full-mask test.
All receipts are under `diagnostics/fast-unit-current/`. Combined latest-source
reconciliation and consolidated production qualification remain separate gates.

Do not repeat completed canonical timing, full-word, or lifecycle runs just to
repair a nullable-field collation error or a receipt-label mismatch. Recover
the analysis from the preserved records. Reject failed performance hypotheses
without expanding their validation matrix. Keep all matcher semantics, native
backend checks, lifecycle costs and eventual release correctness gates intact.

## Verified current working-copy command

From the Windows task workspace (not the repository root):

```powershell
& .\diagnostics\fast-unit-current\fast-unit.ps1
```

The script and its README are at `diagnostics/fast-unit-current/` relative to
that workspace. It builds live `run/glrmask-exact-group-candidate-20261004/src`,
not an archived source snapshot. Use `-TestFilter 'name'` for another root unit
test, or `-ExpectRebuild` when measuring a genuine edit rather than a no-op.
It checks manifest/lock identities, unexpected dependency rebuilds, registered
tests, focused test success, and source hashes; it never edits source.

On base `4b84294be3406271b77ddf6d640753a848c2ae13` plus the current serde patch,
a genuine `terminal_count > 1` to `terminal_count > 1u32` edit passed in
**35.015s**, and its byte-identical source reversal passed in **34.687s**.
Each compiled one root test executable and zero duplicate root libraries;
all 33 external artifact paths were unchanged, with 144 Fresh artifacts
(122 Fresh non-build-script dependency units). Each registered **1,158 tests**:
all 1,154 helper-snapshot tests plus four serde tests, with no removals.
Only the selected focused test ran; registration is not full-suite execution.
Receipts and hashes: `diagnostics/fast-unit-current/adoption-result.json`.

One-time setup is separate from warm-edit timing. Moving this unit graph to a
different manifest identity rebuilt 12 local dependencies; the first attempt
hit its explicit 600s guard. After checking completed dependency artifacts,
the root-only bootstrap passed in 295.5s. Do not attribute this cold/setup time
to a warm edit or reuse the helper snapshot's timings as current measurements.

## Portable preparation for another checkout

The repository-owned `scripts/prepare_windows_unit_manifest.py` derives all
source/dependency paths from its current repository, with no hardcoded task,
user or Python environment path. It requires Python 3.11+ and Cargo on PATH.
From the repository root, first audit without creating anything:

```powershell
py -3 .\scripts\prepare_windows_unit_manifest.py --check-only
```

Then prepare the isolated `.cache/windows-unit/Cargo.toml` and lockfile:

```powershell
py -3 .\scripts\prepare_windows_unit_manifest.py
```

Preparation retains the canonical ordinary dependencies, dev dependencies,
versions and features, rebases their local paths to this checkout, and points
`[lib]` to its live `src/lib.rs`. It omits only the isolated manifest's deliberate
root self dev-dependency, disables automatic integration/example/bench/bin
discovery, and uses an isolated resolver-2 workspace. Offline metadata may
prune unused workspace packages, but newly selected package versions are
rejected; a final `--locked` metadata check is required. Existing different
manifests are not overwritten. Canonical Cargo files are hash-checked unchanged.
Unsupported manifest structures fail closed and need review, not guessed edits.

The canonical manifest must retain its self dev-dependency and complete
integration/example/benchmark targets. Never remove these globally to speed
up a unit iteration. New checkout/manifest identities can need cold setup;
reuse the same owned bridge and target thereafter.

The portable build recipe, under an owned bounded build runner, is:

```text
cargo rustc --profile bench --locked --offline -vv --message-format json
  --manifest-path .cache/windows-unit/Cargo.toml
  --target-dir target/windows-unit --features internal-api --lib
  -- -C lto=off
<emitted compiler-artifact executable> --list
<emitted compiler-artifact executable> <focused-test-filter> --test-threads=2
```

Resolve the executable from Cargo's JSON `compiler-artifact` having
`target.name=glrmask`, `profile.test=true`, and a nonempty `executable`; do not
guess an old hashed filename. Require exactly one root test compile on a real
edit, no duplicate root library, and unchanged dependency artifact paths.
The preparation script's check-only path was tested on the current checkout;
the separate portable bridge is not another measured performance sample.

In the child environment set `CARGO_BUILD_JOBS=2`, `CARGO_INCREMENTAL=1`, and
blank `RUSTC_WRAPPER` and `RUSTC_WORKSPACE_WRAPPER`. Clear inherited
`RUSTFLAGS`, `CARGO_ENCODED_RUSTFLAGS`, profile overrides and diagnostic/
experiment flags; do not mutate another owner's global shell environment.
The generated unit bridge has release/bench opt3, CGU16, incremental enabled,
debug0, LTO=false, debug assertions off and overflow checks off.

**`lto=false` still permits local ThinLTO.** The extra `-C lto=off` disables it
only for the root test executable. Do not replace this with global Rust flags
or canonical profile edits: those can invalidate dependency caches and change
production-generated code. `bench` here invokes Cargo's libtest compilation
mode, not a performance benchmark. Never use this unit-only binary/profile
for production performance claims; keep canonical release profiles, fixture,
timer, features and workload for matched acceptance.

## Oversight and narrow cache recovery

Check current native workload, free physical RAM/commit and disk before heavy
work; preserve unrelated jobs. A two-minute checkpoint is an oversight/update
point, not a blanket kill. Use an explicit budget appropriate to cold setup
versus warm work. The tested task controller defaults to a 600s process-tree
guard, assigns a suspended Cargo process to a Windows JobObject before resume,
and verifies zero owned processes after completion/timeout.

After an interrupted compile or missing LLVM-symbol link failure, inspect the
exact failing rustc command, metadata/extra-filename, extern artifact identities,
incremental session generations and expected symbol suffixes before retrying.
Mixed old/new codegen objects were demonstrated in an earlier ThinLTO failure;
this is evidence for that case, not a diagnosis of every failure.

Freeze the affected owned cache entries with verified byte-identical copies
and receipts first. If evidence implicates root ThinLTO reuse keys, reversibly
move only the identified root entries (the earlier verified recovery moved two
`thin-lto-past-keys.bin` entries totaling 1,072 bytes). If root/harness unit
identity is incompatible, quarantine only the proven affected unit entries and
their matching artifacts. Record exact paths/hashes and how to restore them.
Keep dependency caches, unrelated owners, frozen production binaries and
canonical source untouched. No broad `cargo clean`, target deletion, or blind
repeated full rebuild. Recovery, setup, warm-edit timing and production
qualification must be reported separately.
