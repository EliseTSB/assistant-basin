#!/usr/bin/env python3
"""kernel_upcast_proof.py: does kernel_reference.py's float32-upcast mode compute what a float32 load computes?

Compares two kernel_reference records of the same model and prompts, one made with --dtype float32 and one with
--dtype float32-upcast: every generated token and every top-two margin must be identical, row for row, at every batch
size. Exit 0 and print "identical" only then; otherwise print the first difference and exit 1. Written 2026-11-04 for
the Qwen3.5-4B CPU reference, which does not fit in float32 on the host (16 GB of weights); proved on Qwen3.5-0.8B.

  python tools/kernel_upcast_proof.py FLOAT32.json UPCAST.json [--out PROOF.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def compare(a: dict, b: dict) -> list[str]:
    diffs = []
    for field in ('model', 'revision', 'prompt_sha256', 'new_tokens', 'batch_sizes', 'attn_implementation', 'device'):
        if a.get(field) != b.get(field):
            diffs.append(f'{field}: {a.get(field)!r} != {b.get(field)!r}')
    if a.get('dtype_requested') != 'float32' or b.get('dtype_requested') != 'float32-upcast':
        diffs.append(f"dtypes: {a.get('dtype_requested')!r}, {b.get('dtype_requested')!r} "
                     "(want float32 then float32-upcast)")
    if diffs:
        return diffs
    for batch, rows in a['runs'].items():
        other = b['runs'].get(batch)
        if other is None or len(other) != len(rows):
            diffs.append(f'batch {batch}: row count differs')
            continue
        for ra, rb in zip(rows, other):
            if ra['prompt'] != rb['prompt'] or ra['tokens'] != rb['tokens']:
                diffs.append(f"batch {batch} prompt {ra['prompt']}: tokens differ")
            elif ra['margins'] != rb['margins']:
                n = next(i for i, (x, y) in enumerate(zip(ra['margins'], rb['margins'])) if x != y)
                diffs.append(f"batch {batch} prompt {ra['prompt']}: margin differs at position {n}: "
                             f"{ra['margins'][n]!r} != {rb['margins'][n]!r}")
    return diffs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('float32')
    ap.add_argument('upcast')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    ra = json.loads(Path(a.float32).read_text(encoding='utf-8'))
    rb = json.loads(Path(a.upcast).read_text(encoding='utf-8'))
    diffs = compare(ra, rb)
    rows = sum(len(v) for v in ra.get('runs', {}).values())
    result = {'float32': a.float32, 'upcast': a.upcast, 'model': ra.get('model'), 'rows': rows,
              'identical': not diffs, 'differences': diffs[:20]}
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    print('identical' if not diffs else f'DIFFERENT ({len(diffs)}): {diffs[0]}', f'({rows} rows)')
    return 0 if not diffs else 1


if __name__ == '__main__':
    sys.exit(main())
