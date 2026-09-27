#!/usr/bin/env python3
"""Validate and benchmark pre-materialization sparse read-context propagation.

Build with `cargo build --release --features internal-api --example
composition_build_static_artifact --example composition_loaded_static_probe`.
Keep a frozen pre-change builder as --baseline. Each --inputs NAME=PATH must
contain core.bin, dispatch-literal.bin and vocab_dump.bin. Output is new-only;
source fixtures are never changed. Validation is excluded from timed builds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
from typing import Any

import run_boundary_profile as profile

POLICY = "GLRMASK_BOUNDARY_VIRTUAL_READ_CONTEXT"
VALIDATE = "GLRMASK_PROFILE_COMPILE"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def execute(command: list[str], env: dict[str, str], log: Path, timeout: int) -> str:
    # Stream output so a failed or timed-out validator leaves its witness on disk.
    with log.open("wb") as stream:
        result = subprocess.run(command, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=timeout)
    text = log.read_text(encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"exit {result.returncode}; see {log}\n{text[-2000:]}")
    return text


def paired(rows: list[dict[str, Any]], reference: str, candidate: str, key: str) -> dict[str, Any]:
    rounds = sorted({row["round"] for row in rows})
    deltas = []
    for round_id in rounds:
        a = next(row[key] for row in rows if row["round"] == round_id and row["mode"] == reference)
        b = next(row[key] for row in rows if row["round"] == round_id and row["mode"] == candidate)
        deltas.append(a - b)
    return {"median_saving_ms": statistics.median(deltas),
            "positive_pairs": sum(value > 0 for value in deltas), "deltas_ms": deltas}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--native-checker", type=Path, help="Optional exact native row/target/weight checker")
    parser.add_argument("--inputs", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--reuse-topology", action="store_true", help="Also enable the independent topology-reuse policy in every arm.")
    parser.add_argument("--runtime-rounds", type=int, default=4)
    parser.add_argument("--runtime-repeats", type=int, default=151)
    args = parser.parse_args()
    if min(args.pairs, args.threads, args.runtime_rounds, args.runtime_repeats) < 1:
        parser.error("all counts must be positive")
    candidate, baseline, runtime = (path.resolve() for path in (args.candidate, args.baseline, args.runtime))
    for path in (candidate, baseline, runtime):
        if not path.is_file():
            parser.error(f"missing executable: {path}")
    fixtures: dict[str, Path] = {}
    for value in args.inputs:
        name, separator, directory = value.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in fixtures:
            parser.error(f"invalid or duplicate fixture: {value!r}")
        path = Path(directory).resolve()
        for filename in ("core.bin", "dispatch-literal.bin", "vocab_dump.bin"):
            if not (path / filename).is_file():
                parser.error(f"missing fixture file: {path / filename}")
        fixtures[name] = path
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env0 = profile.child_environment(os.environ, profile.load_profile(profile.DEFAULT_PROFILE), args.threads)
    env0.pop("GLRMASK_BOUNDARY_REUSE_PROGRAM_TOPOLOGY", None)
    if args.reuse_topology:
        env0["GLRMASK_BOUNDARY_REUSE_PROGRAM_TOPOLOGY"] = "1"
    report: dict[str, Any] = {
        "runner_sha256": digest(Path(__file__)),
        "source": {name: digest(Path(__file__).resolve().parents[1] / "crates/glrmask-parser-dwa/src" / name)
                   for name in ("finite_read_support.rs", "finite_signed_graph.rs", "finite_template_program.rs")},
        "executables": {name: {"path": str(path), "sha256": digest(path)}
                        for name, path in (("candidate", candidate), ("baseline", baseline), ("runtime", runtime))},
        "flags": {key: value for key, value in env0.items() if key.startswith(("GLRMASK_", "RAYON_"))},
        "fixtures": {}, "builds": [], "runtime": [], "summaries": {},
    }

    def save() -> None:
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    modes = ["main", "off", "control", "default"]
    for fixture, inputs in fixtures.items():
        destination = out / fixture
        destination.mkdir()
        report["fixtures"][fixture] = {
            "source": str(inputs), "sha256": {path.name: digest(path) for path in inputs.iterdir() if path.is_file()},
            "artifacts": {},
        }
        for mode in modes:
            cache = destination / mode
            cache.mkdir()
            shutil.copyfile(inputs / "vocab_dump.bin", cache / "vocab_dump.bin")
            env = env0.copy()
            if args.native_checker and mode in ("off", "default"):
                env["GLRMASK_DUMP_BOUNDARY_NATIVE_PREMIN"] = str(cache / "native")
            if mode in ("off", "control"):
                env[POLICY] = "0"
            if mode == "default":
                env[VALIDATE] = "1"
                env["GLRMASK_PROFILE_COMPOSE"] = "1"
            builder = baseline if mode == "main" else candidate
            artifact = cache / "composed-latest.bin"
            text = execute([str(builder), str(inputs), str(artifact)], env,
                           destination / f"{mode}-validation.log", 120)
            if mode == "default" and not re.search(r"context_allocated_states: [1-9]\d*", text):
                raise RuntimeError("default sparse-context path was not selected")
            report["fixtures"][fixture]["artifacts"][mode] = {
                "sha256": digest(artifact), "bytes": artifact.stat().st_size,
            }
            save()
        hashes = report["fixtures"][fixture]["artifacts"]
        if any(value != hashes["off"] for value in hashes.values()):
            raise RuntimeError("sparse-context construction or D1 control changed independently frozen artifact bytes")

        if args.native_checker:
            checker = args.native_checker.resolve()
            report["native_checker"] = {"path": str(checker), "sha256": digest(checker)}
            native_text = execute([str(checker),
                str(destination / "off/native/component-1-native.bin.zst"),
                str(destination / "default/native/component-1-native.bin.zst")], env0,
                destination / "native-identity-gate.log", 120)
            if "NATIVE_ROWS_EXACT" not in native_text:
                raise RuntimeError("native identity checker did not confirm exactness")
            report["fixtures"][fixture]["native_identity"] = native_text.strip()
            save()

        for round_id in range(args.pairs):
            order = modes[round_id % len(modes):] + modes[:round_id % len(modes)]
            if round_id % 2:
                order.reverse()
            for mode in order:
                env = env0.copy()
                if mode in ("off", "control"):
                    env[POLICY] = "0"
                artifact = destination / mode / "composed-latest.bin"
                builder = baseline if mode == "main" else candidate
                text = execute([str(builder), str(inputs), str(artifact)], env,
                               destination / f"build-{round_id}-{mode}.log", 120)
                match = re.search(r"BUILD_RESULT compose_ms=([\d.]+) save_ms=([\d.]+)", text)
                if match is None or digest(artifact) != hashes[mode]["sha256"]:
                    raise RuntimeError("missing timing or artifact changed after semantic validation")
                report["builds"].append({"fixture": fixture, "round": round_id, "mode": mode,
                    "compose_ms": float(match[1]), "save_ms": float(match[2])})
                save()

        expected_masks = None
        for round_id in range(args.runtime_rounds):
            order = ["off", "control", "default"]
            if round_id % 2:
                order.reverse()
            for mode in order:
                text = execute([str(runtime), str(destination / mode), "composed", str(args.runtime_repeats)],
                               env0, destination / f"runtime-{round_id}-{mode}.log", 120)
                # The probe's stderr metadata shares the log. Keep only CSV lines.
                lines = [line for line in text.splitlines()
                         if line.startswith("kind,") or line.startswith("composed,")]
                rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
                if not rows:
                    raise RuntimeError("runtime probe returned no measured prefixes")
                signatures = [(row["prefix_index"], row["prefix_hex"], row["allowed"], row["hash"]) for row in rows]
                if expected_masks is None:
                    expected_masks = signatures
                if signatures != expected_masks:
                    raise RuntimeError("loaded artifact mask signatures disagree")
                report["runtime"].append({"fixture": fixture, "round": round_id, "mode": mode, "prefixes": rows})
                save()
        builds = [row for row in report["builds"] if row["fixture"] == fixture]
        summary = {
            "compose_medians_ms": {mode: statistics.median(row["compose_ms"] for row in builds if row["mode"] == mode)
                                   for mode in modes},
            "paired": {f"{ref}-to-{target}": paired(builds, ref, target, "compose_ms")
                       for ref, target in (("main", "default"), ("off", "default"), ("control", "default"), ("off", "control"))},
            "runtime_prefixes": len(expected_masks or []),
        }
        report["summaries"][fixture] = summary
        save()
        print(fixture, json.dumps(summary), flush=True)
    print("PASS: byte-identical artifacts, default path, optional native identity, and loaded masks", flush=True)


if __name__ == "__main__":
    main()
