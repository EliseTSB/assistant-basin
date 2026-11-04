#!/usr/bin/env python3
"""Part 185: rerun Part 84 or 84b unchanged on the eager kernel, then record the load (registered in PART185-DESIGN.md).

Usage: run185.py {84|84b} --out DIR --calib FILE [--device xpu]. The part's own main runs with the remaining arguments;
afterwards neuron_report.LAST_LOAD is written into DIR/run.json, and the gate (eager kernel, the originals' calibration
hash) is checked. A gate failure exits 3 after writing, so the record still says what ran."""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CALIB_SHA256 = 'afb86522116b9ccd10233826503dad0992dd0283f04ee2a15dabf5f0e9e1ce54'


def out_dir(rest):
    for i, a in enumerate(rest):
        if a == '--out' and i + 1 < len(rest):
            return Path(rest[i + 1])
        if a.startswith('--out='):
            return Path(a.split('=', 1)[1])
    raise SystemExit('run185.py: --out is required')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ('84', '84b'):
        raise SystemExit('usage: run185.py {84|84b} --out DIR --calib FILE [--device xpu]')
    which, rest = argv[0], argv[1:]
    out = out_dir(rest)
    import neuron_report as nr
    if which == '84':
        from part84.part84 import main as part_main
    else:
        from part84b.part84b import main as part_main
    part_main(rest)
    path = out / 'run.json'
    run = json.loads(path.read_text(encoding='utf-8'))
    run['load'] = nr.LAST_LOAD
    run['part185'] = {'reruns': 'Part ' + which}
    path.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding='utf-8')
    bad = []
    if not nr.LAST_LOAD or nr.LAST_LOAD.get('attn_implementation') != 'eager':
        bad.append(f"kernel {None if not nr.LAST_LOAD else nr.LAST_LOAD.get('attn_implementation')!r}")
    if run.get('calib_sha256') != CALIB_SHA256:
        bad.append(f"calibration {run.get('calib_sha256')!r}")
    if bad:
        print('PART185 GATE FAILED: ' + '; '.join(bad), flush=True)
        return 3
    print(f'PART185 gate passed: eager, calibration {CALIB_SHA256[:12]}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
