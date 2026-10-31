#!/usr/bin/env python3
"""Exploratory whole-layer distances and free-arm pairwise null baselines.

Usage: extra159.py RESULTS_DIR ARRAYS_DIR
Arrays are read-only. Results are written to state_return.json in RESULTS_DIR.
"""
import json
import sys
from pathlib import Path

import numpy as np

results, arrays = Path(sys.argv[1]), Path(sys.argv[2])
LAYERS = (6, 7, 11, 27, 29)
rows = [json.loads(line) for line in (results / 'answers.jsonl').open(encoding='utf-8')]
by = {(r['arm'], r['question_index']): r['decoded_emitted_tokens'] for r in rows}
steps = np.load(results / 'steps.npz')


def first_veto(arm):
    pre, emitted = steps[f'{arm}_pre_token'], steps[f'{arm}_emitted_token']
    return (pre != emitted).argmax(axis=1)


def layer(arm, number):
    return np.load(arrays / f'layers_{arm}_L{number}.npy', mmap_mode='r')


t_maker, t_control = first_veto('maker_ban'), first_veto('control_ban')
# The maker rows share `yi`/` Lab` in free and `y`/`i`/` Lab` in maker_ban.
maker_rows = [i for i in range(24) if by['free', i][t_maker[i]:t_maker[i] + 2] == ['yi', ' Lab']
              and by['maker_ban', i][t_maker[i]:t_maker[i] + 3] == ['y', 'i', ' Lab']]
control_rows = [i for i in range(24) if by['free', i][t_control[i] + 1] == by['control_ban', i][t_control[i] + 1]]
print('maker rows', len(maker_rows), 'control rows', len(control_rows))
out = {'maker_rows': maker_rows, 'control_rows': control_rows, 'layers': {}}
print('| Layer | Maker: `y` vs `yi` | Maker: `i` vs ` Lab` | Maker: both ` Lab` | Maker null: free ` Lab` pairs | Control: substitute vs ` model` | Control: both ` developed` | Control null: free ` developed` pairs |')
print('|---:|---:|---:|---:|---:|---:|---:|---:|')
for number in LAYERS:
    free = np.asarray(layer('free', number), dtype=np.float32)
    maker = np.asarray(layer('maker_ban', number), dtype=np.float32)
    control = np.asarray(layer('control_ban', number), dtype=np.float32)
    sd = free.std(axis=(0, 1))
    keep = sd > 0

    def distance(ban, rows_used, ban_call, free_call):
        per = [np.abs(ban[r, ban_call(r)] - free[r, free_call(r)])[keep] / sd[keep] for r in rows_used]
        return float(np.median(np.median(per, axis=0)))

    def pairwise_free_null(rows_used, call):
        # Median ordered, distinct row pairs per unit, then median across units.
        pairs = [(i, j) for i in rows_used for j in rows_used if i != j]
        vals = np.abs(np.stack([free[i, call(i)] - free[j, call(j)] for i, j in pairs]))[:, keep] / sd[keep]
        return float(np.median(np.median(vals, axis=0)))

    cells = [distance(maker, maker_rows, lambda r: t_maker[r] + 1, lambda r: t_maker[r] + 1),
             distance(maker, maker_rows, lambda r: t_maker[r] + 2, lambda r: t_maker[r] + 2),
             distance(maker, maker_rows, lambda r: t_maker[r] + 3, lambda r: t_maker[r] + 2),
             distance(control, control_rows, lambda r: t_control[r] + 1, lambda r: t_control[r] + 1),
             distance(control, control_rows, lambda r: t_control[r] + 2, lambda r: t_control[r] + 2)]
    maker_null = pairwise_free_null(maker_rows, lambda r: t_maker[r] + 2)
    control_null = pairwise_free_null(control_rows, lambda r: t_control[r] + 2)
    out['layers'][str(number)] = {
        'maker': cells[:3], 'maker_free_Lab_null': maker_null,
        'control': cells[3:], 'control_free_developed_null': control_null,
    }
    shown = cells[:3] + [maker_null] + cells[3:] + [control_null]
    print(f'| {number} | ' + ' | '.join(f'{c:.3f}' for c in shown) + ' |')

# Sensitivity checks requested for the L27 named unit. The paired shift
# compares maker calls t+1..t+5 with free calls t+2..t+6 on the 14 `yi` rows.
free27 = np.asarray(layer('free', 27), dtype=np.float32)[:, :, 3325]
maker27 = np.asarray(layer('maker_ban', 27), dtype=np.float32)[:, :, 3325]
control27 = np.asarray(layer('control_ban', 27), dtype=np.float32)[:, :, 3325]
sd27 = float(free27.std())
unshifted = [float(np.abs(maker27[r, t_maker[r] + 1:t_maker[r] + 6] -
                          free27[r, t_maker[r] + 1:t_maker[r] + 6]).mean() / sd27)
             for r in maker_rows]
shifted = [float(np.abs(maker27[r, t_maker[r] + 1:t_maker[r] + 6] -
                        free27[r, t_maker[r] + 2:t_maker[r] + 7]).mean() / sd27)
           for r in maker_rows]
maker_d = [float(np.abs(maker27[r, t_maker[r] + 1:t_maker[r] + 6] -
                        free27[r, t_maker[r] + 1:t_maker[r] + 6]).mean() / sd27)
           for r in range(24) if t_maker[r] + 6 < free27.shape[1]]
control_d = []
omitted_models_rows = []
for r in range(24):
    if by['control_ban', r][t_control[r]] == ' models':
        omitted_models_rows.append(r)
        continue
    if t_control[r] + 6 < free27.shape[1]:
        control_d.append(float(np.abs(control27[r, t_control[r] + 1:t_control[r] + 6] -
                                     free27[r, t_control[r] + 1:t_control[r] + 6]).mean() / sd27))
shift_check = {'rows': maker_rows, 'unshifted_median_D': float(np.median(unshifted)),
               'shifted_median_D': float(np.median(shifted))}
ratio_check = {'omitted_control_rows_with_models': omitted_models_rows,
               'maker_median_D': float(np.median(maker_d)),
               'control_median_D_after_exclusion': float(np.median(control_d)),
               'ratio_after_exclusion': float(np.median(maker_d) / np.median(control_d))}
out['L27N3325_shift_check'] = shift_check
out['L27N3325_ratio_excluding_models'] = ratio_check
print('L27N3325 shift check', shift_check)
print('L27N3325 159.5 ratio excluding ` models` rows', ratio_check)
(results / 'state_return.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
