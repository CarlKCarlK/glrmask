#!/usr/bin/env python3
"""Validate and measure ordered parallel parser normalization on prepared inputs.

The independent baseline is required. Each fixture/thread-count first freezes
its baseline artifact, then all candidate builds must reproduce those bytes.
Profiling, native row comparisons, loaded masks and serialization are not
included in compose_ms. All timed work inside the composer, including worker
setup, publication and destruction, is included.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

import run_boundary_profile as profile

ROOT = Path(__file__).resolve().parents[1]
FLAG = "GLRMASK_BOUNDARY_PARALLEL_NATIVE_ROWS"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(command: list[str], environment: dict[str, str], log: Path) -> str:
    with log.open("wb") as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                env=environment, timeout=180)
    text = log.read_text(encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"{log}: exit {result.returncode}\n{text[-5000:]}")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "runtime", "checker", "current", "legacy", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--pairs", type=int, default=24)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--loaded-rounds", type=int, default=2)
    ap.add_argument("--loaded-repeats", type=int, default=61)
    args = ap.parse_args()
    if min(args.pairs, args.threads, args.loaded_rounds, args.loaded_repeats) < 1:
        ap.error("counts must be positive")
    binaries = {name: getattr(args, name).resolve() for name in
                ("baseline", "candidate", "runtime", "checker")}
    for path in binaries.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = profile.child_environment(os.environ,
        profile.load_profile(profile.DEFAULT_PROFILE), args.threads)
    environment.pop("GLRMASK_BOUNDARY_REUSE_PROGRAM_TOPOLOGY", None)
    modes = ["main", "off", "control", "default"]
    sources = ["parser_dwa.rs", "finite_parallel_rows.rs", "finite_parallel_rows_tests.rs"]
    report = {
        "scope": "Prepared-component composition; serialization and initial constraint compilation excluded.",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {name: digest(ROOT / "crates/glrmask-parser-dwa/src" / name) for name in sources},
        "executables": {name: {"path": str(path), "sha256": digest(path)} for name, path in binaries.items()},
        "runner_sha256": digest(Path(__file__)),
        "threads": args.threads, "pairs": args.pairs,
        "flags": {key: value for key, value in environment.items()
                  if key.startswith(("GLRMASK_", "RAYON_"))},
        "fixtures": {}, "builds": [], "summaries": {},
    }

    def save() -> None:
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def child(mode: str, native: Path | None = None) -> dict[str, str]:
        env = environment.copy()
        env.pop(FLAG, None)  # The default arm really exercises the production default.
        if mode in ("off", "control"):
            env[FLAG] = "0"
        if native is not None:
            env["GLRMASK_PROFILE_COMPILE"] = env["GLRMASK_PROFILE_COMPOSE"] = "1"
            env["GLRMASK_DUMP_BOUNDARY_NATIVE_PREMIN"] = str(native)
        return env

    for fixture in ("current", "legacy"):
        inputs = getattr(args, fixture).resolve()
        dest = output / fixture
        cache = dest / "cache"
        cache.mkdir(parents=True)
        input_hashes = {name: digest(inputs / name) for name in
                       ("core.bin", "dispatch-literal.bin", "vocab_dump.bin")}
        shutil.copyfile(inputs / "vocab_dump.bin", cache / "vocab_dump.bin")
        artifact = cache / "composed-latest.bin"
        record = {"inputs": str(inputs), "input_sha256": input_hashes,
                  "profiles": {}, "native": {}, "runtime": {}}
        report["fixtures"][fixture] = record
        expected = None
        reference_masks = None
        for mode in modes:
            binary = binaries["baseline" if mode == "main" else "candidate"]
            native = dest / mode / "native"
            native.mkdir(parents=True)
            text = run([str(binary), str(inputs), str(artifact)], child(mode, native),
                       dest / f"profile-{mode}.log")
            record["profiles"][mode] = [line for line in text.splitlines() if any(marker in line for marker in
                ("native_parallel_rows", "boundary_direct_program] component=1", "fast_boundary_compact_post"))]
            actual = digest(artifact)
            if mode == "main":
                expected = actual
                record["expected_artifact_sha256"] = actual
            else:
                comparison = run([str(binaries["checker"]),
                    str(dest / "main/native/component-1-native.bin.zst"),
                    str(native / "component-1-native.bin.zst")], environment, dest / f"native-{mode}.log")
                if "NATIVE_ROWS_EXACT" not in comparison:
                    raise RuntimeError("missing complete native-identity certificate")
                record["native"][mode] = comparison.strip()
            if actual != expected:
                raise RuntimeError(f"{fixture}/{mode}: candidate artifact differs from independent same-thread reference")
            markers = [line for line in record["profiles"][mode] if "native_parallel_rows" in line]
            if mode == "default" and args.threads >= 4:
                if not any(re.search(r"batches: [1-9][0-9]*", line) for line in markers):
                    raise RuntimeError("eligible parallel default did not execute")
            elif mode != "main" and markers:
                raise RuntimeError("serial arm unexpectedly selected parallel normalization")
            loaded = []
            for repetition in range(args.loaded_rounds):
                text = run([str(binaries["runtime"]), str(cache), "composed", str(args.loaded_repeats)],
                           environment, dest / f"runtime-{mode}-{repetition}.log")
                lines = [line for line in text.splitlines() if line.startswith(("kind,", "composed,"))]
                rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
                signature = [(row["prefix_index"], row["prefix_hex"], row["allowed"], row["hash"]) for row in rows]
                if len(signature) < 22:
                    raise RuntimeError("missing loaded-prefix mask coverage")
                if reference_masks is None:
                    reference_masks = signature
                if signature != reference_masks:
                    raise RuntimeError("loaded mask signatures differ")
                loaded.append(rows)
            record["runtime"][mode] = loaded
            save()
        for round_id in range(args.pairs):
            order = modes[round_id % len(modes):] + modes[:round_id % len(modes)]
            if round_id % 2:
                order.reverse()
            for mode in order:
                binary = binaries["baseline" if mode == "main" else "candidate"]
                text = run([str(binary), str(inputs), str(artifact)], child(mode),
                           dest / f"build-{round_id}-{mode}.log")
                match = re.search(r"BUILD_RESULT compose_ms=([\d.]+) save_ms=([\d.]+)", text)
                if match is None or digest(artifact) != expected:
                    raise RuntimeError("timed build does not match validated reference")
                report["builds"].append({"fixture": fixture, "round": round_id, "mode": mode,
                                        "compose_ms": float(match[1]), "save_ms": float(match[2])})
                save()
        builds = [row for row in report["builds"] if row["fixture"] == fixture]
        summary = {"medians_ms": {mode: statistics.median(row["compose_ms"] for row in builds
                   if row["mode"] == mode) for mode in modes}, "paired": {}}
        for ref, target in (("main", "default"), ("off", "default"), ("control", "default"), ("off", "control")):
            deltas = [next(row["compose_ms"] for row in builds if row["round"] == index and row["mode"] == ref)
                      - next(row["compose_ms"] for row in builds if row["round"] == index and row["mode"] == target)
                      for index in range(args.pairs)]
            summary["paired"][ref + "-to-" + target] = {
                "median_saving_ms": statistics.median(deltas), "wins": sum(delta > 0 for delta in deltas),
                "deltas_ms": deltas}
        report["summaries"][fixture] = summary
        save()
        print(fixture, json.dumps({**summary, "paired": {key: {k: v for k, v in value.items() if k != "deltas_ms"}
              for key, value in summary["paired"].items()}}), flush=True)
        archived = artifact.with_suffix(".bin.gz")
        with artifact.open("rb") as source, gzip.open(archived, "wb", compresslevel=1) as target:
            shutil.copyfileobj(source, target, 1048576)
        with gzip.open(archived, "rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != expected:
                raise RuntimeError("archive verification failed")
        artifact.unlink()
    print("PARALLEL_PUBLICATION_EXACT_PASS", len(report["builds"]), flush=True)


if __name__ == "__main__":
    main()
