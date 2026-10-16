#!/usr/bin/env python3
# nametest.py — layer-by-layer size of the name swap (Claude, 2026-10-15).
# For each layer: r = mean_i ||A_i - E_claude_i|| / ||A_i - C_i|| (name swap relative to the AI -> person swap), the same
# for ChatGPT, and the Claude-ChatGPT distance relative to the OLMo-Claude one. Paired over the 72 prompts.
# Usage: python nametest.py <npz> [--form chat|plain] [--json out.json]
import argparse, json
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('npz'); ap.add_argument('--form', default='chat'); ap.add_argument('--json')
a = ap.parse_args()
z = np.load(a.npz)
g = lambda k: z[f'acts_{a.form}_{k}'].astype(np.float32)  # [P, L, F]
A, C, EC, EG = g('A'), g('C'), g('E_claude'), g('E_chatgpt')
d = lambda x, y: np.linalg.norm(x - y, axis=2)  # [P, L]
person = d(A, C)
rows = []
for l in range(A.shape[1]):
    r_claude = float(np.mean(d(A, EC)[:, l] / person[:, l]))
    r_chatgpt = float(np.mean(d(A, EG)[:, l] / person[:, l]))
    cg_vs_oc = float(np.mean(d(EC, EG)[:, l] / d(A, EC)[:, l]))
    rows.append({'layer': l, 'olmo_to_claude': r_claude, 'olmo_to_chatgpt': r_chatgpt, 'claude_chatgpt_over_olmo_claude': cg_vs_oc})
    print(f'L{l:2d}  OLMo->Claude {r_claude:.3f}  OLMo->ChatGPT {r_chatgpt:.3f}  |Claude-ChatGPT|/|OLMo-Claude| {cg_vs_oc:.3f}')
if a.json:
    json.dump({'npz': a.npz, 'form': a.form, 'rows': rows}, open(a.json, 'w'), indent=1)
