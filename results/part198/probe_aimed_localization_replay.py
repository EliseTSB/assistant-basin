"""Exercise real replay deduplication with the runner's aimed-control record shape."""
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
    'record_type': 'case', 'run_mode': 'registered', 'run_id': 'r1', 'case_id': 7,
    'took': True, 'already_satisfied': False, 'name_eligible': True,
    'restored_bit_exact': True, 'undo_zero_exact': True, 'n_written_so_far': 0,
    'localization': {'f1': {'localized': False}, 'pcs': {'localized': False},
                     't_word': {'localized': False}},
    'aimed_positive_control': {
        'localization': {'value': 2.0, 'localized': False, 'tie': False},
        'oracle_named_strict': False,
    },
    'oracle_reader': {'named_strict': False, 'named_loose': False},
    'blind_reader': {'localized': False, 'named_strict': False},
    'random_control': {'localization': {'t_word': {'localized': False}}},
    'edited_layer_stats': {'17': {'update_top_names_strict': False,
                                  'update_top_names_loose': False}},
}
changed = copy.deepcopy(base)
changed['run_id'] = 'r2'
changed['aimed_positive_control']['localization']['localized'] = True

for rows in ((base, changed), (changed, base)):
    with tempfile.TemporaryDirectory(prefix='part198-aimed-localization-') as td:
        log = Path(td) / 'armA-rome.jsonl'
        for row in rows:
            R.append_jsonl(log, row)
        try:
            selected = S.registered_rows(log, 'case', 'case_id')
            print(f"accepted=True selected_aimed_localized="
                  f"{selected[0]['aimed_positive_control']['localization']['localized']}")
        except ValueError as exc:
            print(f'refused=True error={exc}')
