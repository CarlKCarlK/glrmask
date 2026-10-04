# Composition development loop (Mac worker 586543)

Use the unchanged checkout at `/tmp/glrmask-composition-faithful-codec-586543-20261004` and explicit Rust 1.95.0. Default-toolchain commands are prohibited in this session because they previously accessed a denied managed cache and failed on this source.

Ordinary source experiments use `/tmp/composition_optimized_dev_run1_586543.py` with the stable target `/tmp/glrmask-composition-optimized-dev-target-586543-20261004`. This target was seeded by an independent APFS copy-on-write clone of the current owned production target. It preserves cached dependencies while enabling optimized incremental compilation for the root `glrmask` crate. The root uses opt-level 3, 256 codegen units, incremental compilation, native CPU flags, and no compiler wrapper. Dependency profiles stay unchanged. This is a development artifact, and its timing does not establish release acceptance.

Run a focused gate after a relevant edit:

```sh
python3 /tmp/composition_optimized_dev_run1_586543.py UNIQUE_RECEIPT_NAME test -p glrmask --features internal-api --lib recursive_boundary_validation_
```

For runtime probes, build the optimized native library once in the same stable target, then compile only the small owned adapter against that exact library and its cached dependencies. Keep the following library command unchanged: its feature closure differs from the test closure, so both are cached separately after first setup. The native bootstrap rebuilt five workspace dependencies once; warm cycles must show no dependency rebuilds. Keep the adapter and dataset fixed across source comparisons. Preserve all mask, commit, completion, domain, budget, and refusal checks. Report cold setup, warm source edits, no-op builds, adapter compilation, and query execution separately. Do not reinstall Python or rebuild canonical dependencies for each small experiment.

Batch the established production build, full source qualification, canonical oracles, Python checks, and matched performance acceptance at meaningful selected-source checkpoints, ordinarily once or twice daily. Run completed gates again only for relevant source changes or unresolved failures. Production retains the configured compiler wrapper, normal nonincremental profile, exact flags, and separate target. Development timings do not close Dynamic regression, tail, cumulative baseline, or cross-platform performance gates.

Keep current warm targets, qualified source snapshots, immutable native/Python pins, handoff archives, and raw evidence. Inactive generated caches may be retired within the user's cleanup authorization; record verified recovery and actual free-space change. Do not overlap another worker's source or shared heavy target without the required acknowledgement. No publication, default changes, main merge, or package release is authorized.

The native-library and probe commands are:

```sh
python3 /tmp/composition_optimized_dev_run1_586543.py UNIQUE_NATIVE_RECEIPT build -p glrmask --features internal-api --lib --message-format=json
python3 /tmp/composition_optimized_dev_probe1_586543.py UNIQUE_NATIVE_RECEIPT UNIQUE_PROBE_RECEIPT
```

The probe preserves the existing canonical Dynamic fixture and settings, compares all 11,767 positions for masks, commits and trace-end completion, then performs one warm replay. It does not repeat explicit EOF qualification or establish release performance acceptance.

Initial optimized root-test cache setup took 211.70 seconds. A real reversible native source edit then took 34.74 seconds including the three focused tests (Cargo build 34.00 seconds, test execution 0.02 seconds). First native-library cache setup took 130.48 seconds, including the one-time distinct dependency closure. Its small adapter compiled in 1.21 seconds and the probe ran in 5.82 seconds. These are distinct setup and warm measurements, not no-op timings. The complete real restored-source edit → focused tests → native library → adapter → full-mask performance probe cycle passed in **56.29 seconds**. The focused command took 35.68 seconds, native compile 13.48 seconds, adapter 1.19 seconds, and probe 5.49 seconds (process dispatch/receipt work accounts for the remaining total). Only `glrmask` rebuilt in the warm native command; dependency artifacts were reused. All three focused tests and all 11,767 full-mask/commit/completion comparisons passed. The checkout returned clean to exact qualified f92dbadf5 source. No no-op timing was used to claim edit latency.

This workflow currently measures the reserved root serde lane. For a later authorized edit to another workspace crate, enable package-specific incremental compilation for that crate once and retain its stable profile/cache; do not infer that crate's turnaround from the root measurements.
