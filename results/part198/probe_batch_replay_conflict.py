"""Measure conflicting mass-batch deduplication through the real scorer."""
from __future__ import annotations

import copy
import tempfile
import sys
from pathlib import Path

repo = Path('/home/elisepc/green/worktrees/ab-part198')
sys.path.insert(0, str(repo / 'results/part198'))
import run198 as R
import score198 as S

base = {
    'record_type': 'batch', 'run_mode': 'registered', 'run_id': 'r1', 'batch_id': 'b1',
    'n_took': 8, 'n_already_satisfied': 0, 'take_gate_passed': True,
    'restored_bit_exact': True,
    'localization': {'f1': {'localized': False}, 'pcs': {'localized': False}},
}
changed = copy.deepcopy(base)
changed['run_id'] = 'r2'
changed['localization']['f1']['localized'] = True

for rows in ((base, changed), (changed, base)):
    with tempfile.TemporaryDirectory(prefix='part198-batch-replay-') as td:
        log = Path(td) / 'armA-mass-memit.jsonl'
        for row in rows:
            R.append_jsonl(log, row)
        try:
            S.registered_rows(log, 'batch', 'batch_id')
        except ValueError as exc:
            refused = "batch_id='b1'" in str(exc) and "localization.f1.localized" in str(exc)
            print(f"refused={refused} error={exc}")
            if not refused:
                raise
        else:
            raise RuntimeError('conflicting batch localization replay was accepted')
