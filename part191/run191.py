#!/usr/bin/env python3
"""Part 191: Part 153's battery on the eager kernel, in batch shapes the kernel check covers (PART191-DESIGN.md).

Usage: run191.py --out FILE [battery153's other arguments: --device xpu --model ... --units ... --medians ...]

part153/battery153.py's run() is called unchanged except for one injected generator: each question's 20 pushed rows
(10 neurons and their paired random directions) are generated in four chunks of 8, 8, 2 and 2 rows, in the original
row order, instead of one batch of 20. Part 183's kernel check covers eager at batch 1, 2 and 8, not 20. The plain row
stays a batch of one, as published. Afterwards neuron_report.LAST_LOAD and the chunk record are written to load.json
beside --out (one entry per invocation that wrote rows), and the run exits 3 unless every recorded kernel is eager.
`run191.py --check-record results/part191/load.json` repeats that check on the host before scoring.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHUNKS = (8, 8, 2, 2)


def chunked(generate_one, chunks=CHUNKS):
    """Wrap a battery generator so a pushed batch is generated in fixed chunks, rows kept in order."""
    def generate(model, tok, layers, question, pushes, max_new_tokens, device):
        if not pushes:
            return generate_one(model, tok, layers, question, pushes, max_new_tokens, device)
        sizes = {int(v.shape[0]) for v in pushes.values()}
        if len(sizes) != 1:
            raise ValueError(f'pushes disagree on batch size: {sorted(sizes)}')
        batch = sizes.pop()
        if sum(chunks) != batch:
            raise ValueError(f'chunks {chunks} sum to {sum(chunks)}, batch is {batch}')
        answers, start = [], 0
        for size in chunks:
            part = {layer: push[start:start + size] for layer, push in pushes.items()}
            got = generate_one(model, tok, layers, question, part, max_new_tokens, device)
            if len(got) != size:
                raise ValueError(f'chunk of {size} rows returned {len(got)} answers')
            answers.extend(got)
            start += size
        return answers
    return generate


def check_kernels(kernels):
    """The gate: at least one recorded load, and every recorded load eager."""
    if not kernels:
        return False, 'no load recorded (no invocation wrote rows)'
    bad = [k for k in kernels if k != 'eager']
    if bad:
        return False, f'non-eager load(s) recorded: {bad!r}'
    return True, ''


def check_record(path):
    """Host-side check before scoring: exit 0 only if load.json records at least one load and all are eager."""
    rec = json.loads(Path(path).read_text(encoding='utf-8'))
    ok, why = check_kernels([((e.get('load') or {}).get('attn_implementation')) for e in rec.get('loads', [])])
    print('PART191 record ok: all loads eager' if ok else f'PART191 RECORD FAILED: {why}', flush=True)
    return 0 if ok else 3


def parse(argv):
    from part153 import battery153
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', default='xpu')
    ap.add_argument('--model', default='E:/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')
    ap.add_argument('--units', default=str(ROOT / 'results/part112/units.json'))
    ap.add_argument('--medians', default=str(ROOT / 'results/part112/medians.json'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    args.batch_size = 20  # the registered row count per question; generation is chunked, see CHUNKS
    return args, battery153


def main(argv=None, loader=None, generate_one=None):
    args, battery153 = parse(list(sys.argv[1:] if argv is None else argv))
    import neuron_report as nr
    if generate_one is None:
        common = battery153.battery._common()
        generate_one = lambda *a: battery153.battery.generate(*a, common=common)  # noqa: E731
    nr.LAST_LOAD = None
    summary = battery153.run(args, loader=loader, generate_fn=chunked(generate_one))
    out = Path(args.out)
    record_path = out.parent / 'load.json'
    rec = {'part191': {'reruns': 'Part 153 battery', 'chunks': list(CHUNKS)}, 'loads': []}
    if record_path.exists():
        rec = json.loads(record_path.read_text(encoding='utf-8'))
    if int(summary.get('answers_written', 0)) > 0:
        # one entry per invocation that wrote rows, so a resumed file carries every kernel that produced it
        rec.setdefault('loads', []).append({'load': nr.LAST_LOAD, 'chunks': list(CHUNKS),
                                            'argv': list(sys.argv[1:] if argv is None else argv),
                                            'summary': summary})
        record_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    kernels = [((e.get('load') or {}).get('attn_implementation')) for e in rec.get('loads', [])]
    ok, why = check_kernels(kernels)
    if not ok:
        print(f'PART191 GATE FAILED: {why}', flush=True)
        return 3
    print(f'PART191 gate passed: {len(kernels)} recorded load(s), all eager', flush=True)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--check-record':
        raise SystemExit(check_record(sys.argv[2]))
    raise SystemExit(main())
