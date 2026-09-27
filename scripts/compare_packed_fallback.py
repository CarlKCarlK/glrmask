#!/usr/bin/env python3
"""Same-binary fallback bookkeeping comparison with exact artifact checks.

Build composition_build_static_artifact with internal-api. Provide prepared
fixture directories via repeated --inputs NAME=PATH. Source fixtures are read
only. The output directory must not exist. Profiles are excluded from timings.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import run_boundary_profile as profile

POLICY = 'GLRMASK_BOUNDARY_PACKED_FALLBACK_SINGLETONS'


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--builder', type=Path, required=True)
    ap.add_argument('--inputs', action='append', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--pairs', type=int, default=16)
    ap.add_argument('--threads', type=int, default=10)
    args = ap.parse_args()
    if min(args.pairs, args.threads) < 1:
        ap.error('counts must be positive')
    binary = args.builder.resolve()
    if not binary.is_file():
        ap.error(f'missing executable {binary}')
    fixtures = {}
    for item in args.inputs:
        name, separator, directory = item.partition('=')
        if not separator or not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in fixtures:
            ap.error(f'invalid or duplicate fixture {item!r}')
        folder = Path(directory).resolve()
        for file in ['core.bin', 'dispatch-literal.bin', 'vocab_dump.bin']:
            if not (folder / file).is_file():
                ap.error(f'missing fixture {folder / file}')
        fixtures[name] = folder
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    base_env = profile.child_environment(os.environ, profile.load_profile(profile.DEFAULT_PROFILE), args.threads)
    base_env.pop(POLICY, None)
    report = {'builder': str(binary), 'builder_sha256': sha(binary), 'policy': POLICY,
              'flags': {k: v for k, v in base_env.items() if k.startswith(('GLRMASK_', 'RAYON_'))},
              'inputs': {}, 'profiles': {}, 'samples': [], 'summary': {}}

    def save() -> None:
        (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    def run(fixture: str, folder: Path, mode: str, label: str, profiled: bool = False) -> tuple[float, str, list[str]]:
        env = base_env.copy()
        if mode in ('off', 'control'):
            env[POLICY] = '0'
        # Mode on intentionally leaves the variable absent: test the actual
        # production default rather than an opt-in-only configuration.
        if profiled:
            env.update(GLRMASK_PROFILE_COMPOSE='1', GLRMASK_PROFILE_COMPILE='1')
        artifact = out / f'{fixture}-{mode}.bin'
        logfile = out / f'{fixture}-{label}-{mode}.log'
        with logfile.open('wb') as stream:
            result = subprocess.run([str(binary), str(folder), str(artifact)], env=env,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=120)
        text = logfile.read_text(encoding='utf-8', errors='replace')
        if result.returncode:
            raise RuntimeError(f'build failed; {logfile}: {text[-2500:]}')
        match = re.search(r'BUILD_RESULT compose_ms=([\d.]+) save_ms=([\d.]+)', text)
        if match is None:
            raise RuntimeError(f'missing result in {logfile}')
        lines = [line for line in text.splitlines() if any(tag in line for tag in
                 ['[fast_boundary_fallback]', '[fast_boundary_compact_post]', '[boundary_direct_program] component=1'])]
        if profiled:
            expected_mode = 'packed_singletons=' + ('true' if mode == 'on' else 'false')
            if not any('[fast_boundary_fallback]' in line and expected_mode in line for line in lines):
                raise RuntimeError(f'{fixture}/{mode}: requested fallback policy was not executed')
        return float(match[1]), sha(artifact), lines

    modes = ['off', 'control', 'on']
    for fixture, folder in fixtures.items():
        report['inputs'][fixture] = {p.name: sha(p) for p in folder.iterdir() if p.is_file()}
        expected = None
        for mode in modes:
            _, digest, lines = run(fixture, folder, mode, 'profile', True)
            if expected is None:
                expected = digest
            if digest != expected:
                raise RuntimeError(f'{fixture}: profile artifact differs with {mode}')
            report['profiles'][f'{fixture}/{mode}'] = {'sha256': digest, 'lines': lines}
            save()
        for round_id in range(args.pairs):
            order = modes[round_id % 3:] + modes[:round_id % 3]
            if round_id % 2:
                order.reverse()
            for mode in order:
                elapsed, digest, _ = run(fixture, folder, mode, str(round_id))
                if digest != expected:
                    raise RuntimeError(f'{fixture}/{round_id}/{mode}: artifact mismatch')
                report['samples'].append({'fixture': fixture, 'round': round_id, 'mode': mode,
                                          'compose_ms': elapsed, 'sha256': digest})
                save()
        rows = [row for row in report['samples'] if row['fixture'] == fixture]
        lookup = {(row['round'], row['mode']): row['compose_ms'] for row in rows}
        summary = {'medians_ms': {mode: statistics.median(row['compose_ms'] for row in rows if row['mode'] == mode)
                                   for mode in modes}, 'paired': {}}
        for a, b in [('off', 'control'), ('off', 'on'), ('control', 'on')]:
            ds = [lookup[i, a] - lookup[i, b] for i in range(args.pairs)]
            summary['paired'][f'{a}-to-{b}'] = {'median_saving_ms': statistics.median(ds),
                                              'positive_pairs': sum(d > 0 for d in ds), 'deltas_ms': ds}
        report['summary'][fixture] = summary
        save()
        print(fixture, json.dumps(summary), flush=True)
    print('PACKED_FALLBACK_PASS', len(report['samples']), 'byte-identical timed artifacts', flush=True)


if __name__ == '__main__':
    main()
