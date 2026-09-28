"""Packed cancellation answers against current main; every timed output is hash checked."""
from __future__ import annotations
import argparse
import csv
import hashlib
import gzip
import io
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_boundary_profile as profile

INDEX = 'GLRMASK_BOUNDARY_PACKED_CANCELLATION_RESULTS'
POLICIES = {'main': False, 'off': False, 'control': False, 'packed': True}

def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def run(args: list[str], env: dict[str, str], log: Path) -> str:
    with log.open('wb') as stream:
        result = subprocess.run(args, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
    text = log.read_text(encoding='utf-8', errors='replace')
    if result.returncode:
        raise RuntimeError(f'exit={result.returncode}: {log}\n{text[-3000:]}')
    return text

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--pairs', type=int, default=8)
    ap.add_argument('--modes', default='main,off,control,packed')
    ap.add_argument('--threads', type=int, default=10)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--runtime', type=Path, required=True)
    ap.add_argument('--checker', type=Path, required=True)
    ap.add_argument('--current', type=Path, required=True)
    ap.add_argument('--legacy', type=Path, required=True)
    args = ap.parse_args()
    modes = args.modes.split(',')
    if args.pairs < 1 or args.threads < 1 or len(set(modes)) != len(modes) or any(m not in POLICIES for m in modes):
        ap.error('invalid modes or counts')
    if not {'main','off','control'}.issubset(modes) or modes[0] != 'main':
        ap.error('keep independent main, same-binary reference and identical control')
    base, candidate, runtime, checker = [p.resolve() for p in
        (args.baseline, args.candidate, args.runtime, args.checker)]
    fixtures = {'current': args.current.resolve(), 'legacy': args.legacy.resolve()}
    for path in (base, candidate, runtime, checker):
        if not path.is_file(): raise FileNotFoundError(path)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    clean = profile.child_environment(os.environ, profile.load_profile(profile.DEFAULT_PROFILE), args.threads)
    # This independent presence-based experimental policy must be absent, not 0.
    clean.pop('GLRMASK_BOUNDARY_REUSE_PROGRAM_TOPOLOGY', None)
    def environment(mode: str, profiling: bool = False, native: Path | None = None) -> dict[str,str]:
        env = clean.copy()
        if POLICIES[mode]:
            env.pop(INDEX, None)  # Exercise the actual default, not an opt-in.
        else:
            env[INDEX] = '0'
        if profiling:
            env['GLRMASK_PROFILE_COMPILE'] = env['GLRMASK_PROFILE_COMPOSE'] = '1'
        if native is not None: env['GLRMASK_DUMP_BOUNDARY_NATIVE_PREMIN'] = str(native)
        return env
    report = {'git_head': subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'runner_sha256': digest(Path(__file__)),
        'sources': {name:digest(ROOT / 'crates/glrmask-parser-dwa/src' / name)
                    for name in ('finite_cancellation.rs','finite_cancellation_results.rs','finite_cancellation_results_tests.rs')},
        'executables': {name:{'path':str(path),'sha256':digest(path)}
                        for name,path in [('base',base),('candidate',candidate),('runtime',runtime),('checker',checker)]},
        'flags':{k:v for k,v in clean.items() if k.startswith(('GLRMASK_','RAYON_'))},
        'fixtures':{}, 'builds':[], 'summaries':{}}
    def save() -> None:
        (out / 'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for fixture,inputs in fixtures.items():
        dest = out / fixture
        dest.mkdir()
        cache = dest / 'cache'
        cache.mkdir()
        shutil.copyfile(inputs / 'vocab_dump.bin', cache / 'vocab_dump.bin')
        artifact = cache / 'composed-latest.bin'
        expected_hash = None
        item = {'inputs':str(inputs),'input_hashes':{f:digest(inputs/f) for f in ('core.bin','dispatch-literal.bin','vocab_dump.bin')},
                'expected_artifact_sha256':None, 'profiles':{},'native_gates':{},'runtime':{}}
        report['fixtures'][fixture] = item
        signatures = None
        # Validation is entirely outside the timed loop.
        for mode in modes:
            binary = base if mode == 'main' else candidate
            native = None if mode == 'main' else dest / mode / 'native'
            if native is not None: native.mkdir(parents=True)
            text = run([str(binary),str(inputs),str(artifact)],environment(mode,True,native),dest/f'profile-{mode}.log')
            if mode == 'main':
                expected_hash = digest(artifact)
                item['expected_artifact_sha256'] = expected_hash
            elif digest(artifact) != expected_hash:
                raise RuntimeError(f'{fixture}/{mode}: artifact differs from independently built current main')
            selected = [line for line in text.splitlines() if any(key in line for key in
                ('[cancellation_packed_results]','[virtual_cancellation_summary]','[virtual_signed_graph]','[boundary_direct_program] component=1'))]
            item['profiles'][mode] = selected
            if mode != 'main' and not any('[cancellation_packed_results]' in line for line in selected):
                raise RuntimeError(f'{mode} never reached instrumented cancellation')
            if mode == 'packed' and not any('PackedRows' in line for line in selected):
                raise RuntimeError('packed result representation not selected')
            if mode not in ('main','off','control'):
                identity = run([str(checker),str(dest/'off/native/component-1-native.bin.zst'),
                    str(native/'component-1-native.bin.zst')],clean,dest/f'native-{mode}.log')
                if 'NATIVE_ROWS_EXACT' not in identity: raise RuntimeError('native checker did not confirm identity')
                item['native_gates'][mode] = identity.strip()
            if mode in ('main','off','packed'):
                text = run([str(runtime),str(cache),'composed','31'],clean,dest/f'runtime-{mode}.log')
                lines = [line for line in text.splitlines() if line.startswith(('kind,','composed,'))]
                rows = list(csv.DictReader(io.StringIO('\n'.join(lines))))
                current = [(r['prefix_index'],r['prefix_hex'],r['allowed'],r['hash']) for r in rows]
                if not current: raise RuntimeError('empty runtime probe')
                if signatures is None: signatures = current
                if current != signatures: raise RuntimeError('loaded-mask signatures differ')
                item['runtime'][mode] = rows
            save()
        for round_id in range(args.pairs):
            order = modes[round_id % len(modes):] + modes[:round_id % len(modes)]
            if round_id % 2: order.reverse()
            for mode in order:
                binary = base if mode == 'main' else candidate
                text = run([str(binary),str(inputs),str(artifact)],environment(mode),dest/f'build-{round_id}-{mode}.log')
                match = re.search(r'BUILD_RESULT compose_ms=([\d.]+) save_ms=([\d.]+)',text)
                if match is None or digest(artifact) != expected_hash: raise RuntimeError('bad timed output')
                report['builds'].append({'fixture':fixture,'round':round_id,'mode':mode,
                    'compose_ms':float(match[1]),'save_ms':float(match[2])})
                save()
        rows = [r for r in report['builds'] if r['fixture']==fixture]
        summary = {'medians_ms':{m:statistics.median(r['compose_ms'] for r in rows if r['mode']==m) for m in modes},'paired':{}}
        for ref in ('main','off','control'):
            for mode in modes:
                if mode == ref: continue
                deltas = [next(r['compose_ms'] for r in rows if r['round']==i and r['mode']==ref)
                    - next(r['compose_ms'] for r in rows if r['round']==i and r['mode']==mode) for i in range(args.pairs)]
                summary['paired'][ref+'-to-'+mode] = {'median_saving_ms':statistics.median(deltas),
                    'wins':sum(d>0 for d in deltas),'deltas_ms':deltas}
        report['summaries'][fixture] = summary
        save()
        print(fixture,json.dumps({'medians_ms':summary['medians_ms'],'paired':{k:{a:b for a,b in v.items() if a!='deltas_ms'} for k,v in summary['paired'].items()}}),flush=True)
        # Retain exactly reconstructible outputs without accumulating raw
        # 47MB copies. The timed loops and all runtime checks already finished.
        compressed=artifact.with_suffix('.bin.gz')
        with artifact.open('rb') as source,gzip.open(compressed,'wb',compresslevel=1) as target:
            shutil.copyfileobj(source,target,1024*1024)
        with gzip.open(compressed,'rb') as source:
            if hashlib.file_digest(source,'sha256').hexdigest()!=expected_hash:
                raise RuntimeError('archive roundtrip mismatch')
        item['archived_output']={'path':str(compressed),'sha256':expected_hash}
        artifact.unlink()
        save()
    print('PACKED_CANCELLATION_PASS',len(report['builds']),'byte-identical timed artifacts',flush=True)

if __name__ == '__main__': main()
