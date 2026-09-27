#!/usr/bin/env python3
"""Same-host, native-timed semantic/performance check for a GLRMask refactor.

The saved and candidate extensions run in separate processes. No model, network,
or decoding framework is involved. Timings are not a canonical CFA distribution.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO


def cases(manifest=None):
    compact = lambda value: json.dumps(value, ensure_ascii=False).encode()
    yield 'object', {'type': 'object', 'properties': {'name': {'type': 'string'}, 'count': {'type': 'integer'}}, 'required': ['name', 'count'], 'additionalProperties': False}, compact({'name': 'alpha beta', 'count': 123}), 'schema'
    yield 'open_string', {'type': 'string'}, compact('The tokenizer traverses long English words, 한국어와 日本語, punctuation, and escaped "quotes". ' * 3), 'schema'
    yield 'bounded_code', {'type': 'string', 'pattern': '^(?:a|bb)+$', 'minLength': 2, 'maxLength': 5000}, compact('aabb' * 48), 'schema'
    item = {'type': 'object', 'properties': {'x': {'type': 'integer'}, 'label': {'type': 'string', 'maxLength': 24}}, 'required': ['x', 'label'], 'additionalProperties': False}
    yield 'array', {'type': 'array', 'items': item, 'minItems': 1, 'maxItems': 16}, compact([{'x': i, 'label': 'entry ' + str(i)} for i in range(8)]), 'schema'
    yield 'recursive', 'start ::= "(" start ")" | "x"', b'(' * 32 + b'x' + b')' * 32, 'ebnf'
    yield 'composition', item, compact([{'x': 1, 'label': 'crossing boundary'}]), 'composition'
    for name in ('github_hard_o21074', 'github_ultra_o62058', 'kubernetes_kb_543_normalized', 'o9838_problem'):
        path = REPO / 'benches/data' / ('o9838_problem_schema.json' if name == 'o9838_problem' else name + '.schema.json')
        yield name, json.loads(path.read_text(encoding='utf-8-sig')), None, 'schema'
    if manifest:
        for entry in json.loads(Path(manifest).read_text()):
            target = entry.get('text')
            yield entry['name'], entry['schema'], target.encode() if target is not None else None, entry.get('format', 'schema')


def greedy_tokens(data: bytes, entries: dict[int, bytes]):
    by_first: dict[int, list[tuple[bytes, int]]] = {}
    for token_id, token in entries.items():
        if token:
            by_first.setdefault(token[0], []).append((token, token_id))
    for candidates in by_first.values():
        candidates.sort(key=lambda pair: (-len(pair[0]), pair[1]))
    result = []
    position = 0
    while position < len(data):
        match = next(((token, token_id) for token, token_id in by_first[data[position]] if data.startswith(token, position)), None)
        if match is None:
            raise ValueError(f'No vocabulary token at byte {position}')
        token, token_id = match
        result.append(token_id)
        position += len(token)
    return result


def worker(args):
    package = Path(args.package).resolve()
    sys.path.insert(0, str(package))
    import glrmask
    import glrmask._glrmask as native
    import numpy as np
    if not Path(glrmask.__file__).resolve().is_relative_to(package):
        raise RuntimeError('Requested extension isolation failed')
    entries = {int(k): bytes.fromhex(v) for k, v in json.loads(Path(args.vocab).read_text()).items()}
    vocab = glrmask.Vocab.from_id_to_bytes(entries)
    size = (max(entries) + 32) // 32
    packed = np.zeros(size, dtype=np.int32)
    _, source, target, kind = next(case for case in cases(args.manifest) if case[0] == args.case)
    tokens = greedy_tokens(target, entries) if target is not None else []
    optimization = getattr(glrmask.Optimization, args.optimization)
    def compile_constraint():
        if kind == 'composition':
            child = glrmask.Grammar.from_json_schema(source).compile(vocab, optimization=optimization)
            parent = glrmask.Grammar.from_glrm('glrm 1; start start; extern grammar child; nt start = "[" child "]";')
            return parent.compile_unlinked(vocab).bind('child', child).link(optimization=optimization)
        grammar = glrmask.Grammar.from_json_schema(source) if kind == 'schema' else glrmask.Grammar.from_ebnf(source)
        return grammar.compile(vocab, optimization=optimization)
    gc.disable()
    build = []
    constraint = None
    for _ in range(args.repetitions + 1):
        del constraint
        started = time.perf_counter_ns()
        constraint = compile_constraint()
        build.append(time.perf_counter_ns() - started)
    def trace(constraint):
        state = constraint.start()
        masks, commits = [], []
        fingerprint = hashlib.sha256()
        for token_id in tokens:
            masks.append(int(glrmask._internal.fill_mask_timed_ns(state, packed)))
            fingerprint.update(packed.tobytes())
            if not (int(packed.view(np.uint32)[token_id // 32]) >> (token_id % 32)) & 1:
                raise AssertionError(f'{args.case}: valid token {token_id} rejected at step {len(masks)-1}')
            commits.append(int(glrmask._internal.commit_token_timed_ns(state, token_id)))
        final_mask_ns = int(glrmask._internal.fill_mask_timed_ns(state, packed))
        if not tokens:
            masks.append(final_mask_ns)
        fingerprint.update(packed.tobytes())
        accepting = bool(state.is_accepting())
        if target is not None and not accepting:
            raise AssertionError(f'{args.case}: complete example not accepted')
        fingerprint.update(bytes([accepting, bool(state.is_rejected())]))
        return masks, commits, fingerprint.hexdigest()
    cold = trace(constraint)
    traces = [trace(constraint) for _ in range(args.trace_repetitions)]
    if any(t[2] != cold[2] for t in traces):
        raise AssertionError('Repeated execution changed the mask trace')
    started = time.perf_counter_ns()
    artifact = constraint.save()
    first_save = time.perf_counter_ns() - started
    saves, loads = [], []
    for _ in range(args.repetitions):
        started = time.perf_counter_ns(); constraint.save(); saves.append(time.perf_counter_ns() - started)
        started = time.perf_counter_ns(); loaded = glrmask.Constraint.load(artifact); loads.append(time.perf_counter_ns() - started)
        if trace(loaded)[2] != cold[2]:
            raise AssertionError('Saving/loading changed the mask trace')
        del loaded
    mask = [statistics.median(t[0][i] for t in traces) for i in range(len(cold[0]))]
    commit = [statistics.median(t[1][i] for t in traces) for i in range(len(tokens))]
    tbm = [statistics.median(t[0][i] + t[1][i] for t in traces) for i in range(len(tokens))]
    def quantiles(values):
        if not values: return {}
        ordered = sorted(values)
        return {label: ordered[min(len(ordered)-1, round((len(ordered)-1)*p))] for label,p in [('p50',.5),('p95',.95),('p99',.99),('max',1)]}
    result = dict(case=args.case, optimization=args.optimization, package=str(package), native_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(), vocab_size=len(entries), token_steps=len(tokens), trace_sha256=cold[2], artifact_sha256=hashlib.sha256(artifact).hexdigest(), artifact_bytes=len(artifact), cold_build_ns=build[0], build_ns=statistics.median(build[1:]), build_samples_ns=build[1:], first_save_ns=first_save, save_ns=statistics.median(saves), load_ns=statistics.median(loads), cold_mask=quantiles(cold[0]), mask=quantiles(mask), commit=quantiles(commit), tbm=quantiles(tbm))
    print(json.dumps(result, sort_keys=True))


def compare(args):
    selected = args.cases.split(',') if args.cases else [x[0] for x in cases(args.manifest)]
    opts = args.optimizations.split(',')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = {k:v for k,v in os.environ.items() if not k.startswith('GLRMASK_')}
    env['RAYON_NUM_THREADS'] = str(args.threads)
    records = []
    with output.open('w') as stream:
        for case in selected:
            for opt in opts:
                # ABBA reduces directional host/time drift. Raw samples remain
                # in JSONL; a ratio is not proof about unmeasured workloads.
                for round_index in range(args.rounds):
                    for variant in ('baseline','candidate','candidate','baseline'):
                        package = args.baseline if variant == 'baseline' else args.candidate
                        command = [sys.executable, __file__, 'worker', '--package', package, '--case', case, '--optimization', opt, '--vocab', args.vocab, '--repetitions', str(args.repetitions), '--trace-repetitions', str(args.trace_repetitions)]
                        if args.manifest:
                            command += ['--manifest', str(Path(args.manifest).resolve())]
                        completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=args.timeout)
                        if completed.returncode:
                            record = dict(case=case,optimization=opt,variant=variant,round=round_index,error=completed.stderr[-12000:],exit_code=completed.returncode)
                        else:
                            record = json.loads(completed.stdout.strip().splitlines()[-1]); record.update(variant=variant,round=round_index)
                        records.append(record);stream.write(json.dumps(record,sort_keys=True)+'\n');stream.flush()
                        status = 'ERROR' if 'error' in record else f"build={record['build_ns']/1e6:.3f} ms, steps={record['token_steps']}"
                        print(f'{case} {opt} {variant}: {status}',flush=True)
                        if 'error' in record:
                            print(record['error'][-3000:],flush=True)
                            raise RuntimeError('Acceptance worker failed; stop and inspect rather than silently skip')
    for case in selected:
        for opt in opts:
            group=[r for r in records if r['case']==case and r['optimization']==opt]
            if len({r['trace_sha256'] for r in group})!=1:
                raise AssertionError(f'{case} {opt}: baseline/candidate semantic mismatch')
    print('ALL_SEMANTIC_TRACES_MATCH',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    w=sub.add_parser('worker');w.add_argument('--package',required=True);w.add_argument('--case',required=True);w.add_argument('--optimization',required=True)
    c=sub.add_parser('compare');c.add_argument('--baseline',required=True);c.add_argument('--candidate',required=True);c.add_argument('--output',required=True);c.add_argument('--cases');c.add_argument('--optimizations',default='FAST_BUILD,AUTO,FAST_RUNTIME');c.add_argument('--rounds',type=int,default=1);c.add_argument('--threads',type=int,default=1);c.add_argument('--timeout',type=float,default=120)
    for p in (w,c):
        p.add_argument('--manifest', help='Optional JSON array of additional named schemas and optional example text')
        p.add_argument('--vocab',required=True, help='JSON mapping of model token IDs to hexadecimal byte strings')
        p.add_argument('--repetitions',type=int,default=5)
        p.add_argument('--trace-repetitions',type=int,default=7)
    args=parser.parse_args()
    if args.repetitions<1 or args.trace_repetitions<1: parser.error('repetition counts must be positive')
    if args.mode == 'compare' and (args.rounds < 1 or args.threads < 1): parser.error('rounds and threads must be positive')
    (worker if args.mode=='worker' else compare)(args)
if __name__=='__main__':main()
