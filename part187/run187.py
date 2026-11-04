#!/usr/bin/env python3
"""Part 187: run Part 140's generation unchanged on the eager kernel, then record the load (PART187-DESIGN.md).

Usage: run187.py --out DIR [Part 140's other arguments, e.g. --device xpu --batch-size 8 --units part140/units140.json]
part140/run140.py's main() runs with these arguments; afterwards neuron_report.LAST_LOAD is written to DIR/load.json
and the kernel is checked. A gate failure exits 3 after writing."""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def out_dir(argv):
    for i, a in enumerate(argv):
        if a == '--out' and i + 1 < len(argv):
            return Path(argv[i + 1])
        if a.startswith('--out='):
            return Path(a.split('=', 1)[1])
    raise SystemExit('run187.py: --out is required')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    out = out_dir(argv)
    import neuron_report as nr
    from part140 import run140
    rc = run140.main(argv)
    out.mkdir(parents=True, exist_ok=True)
    rec = {'load': nr.LAST_LOAD, 'part187': {'reruns': 'Part 140 generation', 'argv': argv, 'run140_rc': rc}}
    (out / 'load.json').write_text(json.dumps(rec, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    kernel = (nr.LAST_LOAD or {}).get('attn_implementation')
    if kernel != 'eager':
        print(f'PART187 GATE FAILED: kernel {kernel!r}', flush=True)
        return 3
    print('PART187 gate passed: eager', flush=True)
    return int(rc or 0)


if __name__ == '__main__':
    raise SystemExit(main())
