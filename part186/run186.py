#!/usr/bin/env python3
"""Part 186: run Part 79's step 0 or main run unchanged on the eager kernel, then record the load (PART186-DESIGN.md).

Usage: run186.py step0 --out DIR [--pile FILE] [--batch 1] [--device xpu]
       run186.py main --out DIR --step0 FILE [--batch 8] [--device xpu]
The part's own main() runs with the remaining arguments (it reads sys.argv); afterwards neuron_report.LAST_LOAD is
written into DIR/step0.json or DIR/run.json and the kernel is checked. A gate failure exits 3 after writing."""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RECORD = {'step0': 'step0.json', 'main': 'run.json'}


def out_dir(rest):
    for i, a in enumerate(rest):
        if a == '--out' and i + 1 < len(rest):
            return Path(rest[i + 1])
        if a.startswith('--out='):
            return Path(a.split('=', 1)[1])
    raise SystemExit('run186.py: --out is required')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in RECORD:
        raise SystemExit('usage: run186.py {step0|main} --out DIR ...')
    which, rest = argv[0], argv[1:]
    out = out_dir(rest)
    import neuron_report as nr
    if which == 'step0':
        from part79 import step0 as mod
    else:
        from part79 import part79 as mod
    saved = sys.argv
    sys.argv = [f'part79/{which}'] + rest
    try:
        mod.main()
    finally:
        sys.argv = saved
    path = out / RECORD[which]
    rec = json.loads(path.read_text(encoding='utf-8'))
    rec['load'] = nr.LAST_LOAD
    rec['part186'] = {'reruns': f'Part 79 {which}'}
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    kernel = (nr.LAST_LOAD or {}).get('attn_implementation')
    if kernel != 'eager':
        print(f'PART186 GATE FAILED: kernel {kernel!r}', flush=True)
        return 3
    print(f'PART186 gate passed ({which}): eager', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
