#!/usr/bin/env python3
"""Part 72: clamp L27 N3325 alone to k x its mean and read the maker token (PART72-DESIGN.md). Claude, 2026-10-26."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

UNIT = (27, 3325)
KS = [0, 1, 2, 4, 8]
NULL_K = 8
OPENING = 'I was developed by **'
NAMES = ['Alibaba', 'Google', 'OpenAI', 'Anthropic', 'DeepSeek', 'Meta', 'Microsoft', 'Tongyi']  # no leading space


def conditions(means27, matched_units):
    out = [{'id': 'unchanged', 'ops': {}}]
    out += [{'id': f'k{k}', 'ops': {UNIT: k * float(means27[UNIT[1]])}} for k in KS]
    out += [{'id': f'null_{u}', 'ops': {(27, u): NULL_K * float(means27[u])}} for u in matched_units]
    return out


def verdicts(summary: dict, null_ids: list[str]) -> dict:
    dg = {cid: summary[cid]['dg'] for cid in summary}
    mass8 = summary['k8']['mass']
    v = {
        '72.1': {'holds': summary['k8']['google_share'] > 0.5, 'google_share_k8': summary['k8']['google_share'],
                 'google_share_unchanged': summary['unchanged']['google_share']},
        '72.2': {'holds': all(dg[f'k{a}'] < dg[f'k{b}'] for a, b in zip(KS, KS[1:])),
                 'dg_by_k': {k: dg[f'k{k}'] for k in KS}},
        '72.3': {'holds': all(dg['k8'] > dg[n] for n in null_ids), 'dg_k8': dg['k8'],
                 'null_max': max(dg[n] for n in null_ids)},
        '72.4': {'holds': mass8 >= 0.5, 'mass_k8': mass8},
    }
    for key in v:
        v[key]['status'] = ('HOLDS' if v[key]['holds'] else 'FAILS') if key == '72.4' or v['72.4']['holds'] \
            else 'NOT READ'
    return v


def run(model, layers, prompts, readout_ids, matched_units, out: Path, batch=32):
    import torch
    import part25
    device = next(model.parameters()).device
    n_units = int(layers[27].mlp.down_proj.in_features)
    total, count = torch.zeros(n_units, dtype=torch.float64), 0

    def hook(_m, args):
        nonlocal count
        x = args[0][0].detach().double().cpu()
        total.add_(x.sum(0))
        count += x.shape[0]

    h = layers[27].mlp.down_proj.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            for ids in prompts:
                model(input_ids=torch.tensor([ids], device=device))
    finally:
        h.remove()
    means27 = (total / count).numpy()
    conds = conditions(means27, matched_units)
    records = []
    for qi, ids in enumerate(prompts):
        for b in range(0, len(conds), batch):
            chunk = conds[b:b + batch]
            x = torch.tensor([ids] * len(chunk), device=device)
            handles = part25.install_intervention_hooks(layers, [c['ops'] for c in chunk])
            try:
                with torch.no_grad():
                    logits = model(input_ids=x).logits[:, -1, :].float()
            finally:
                for hh in handles:
                    hh.remove()
            sel = logits[:, readout_ids]
            mass = logits.softmax(-1)[:, readout_ids].sum(-1)
            share = sel.softmax(-1)
            for ri, c in enumerate(chunk):
                records.append({'q': qi, 'id': c['id'], 'g': float(sel[ri, 1] - sel[ri, 0]),
                                'google_share': float(share[ri, 1]), 'mass': float(mass[ri]),
                                'logits': sel[ri].cpu().tolist()})
    base = {r['q']: r['g'] for r in records if r['id'] == 'unchanged'}
    summary = {}
    for c in conds:
        rows = [r for r in records if r['id'] == c['id']]
        summary[c['id']] = {'dg': sum(r['g'] - base[r['q']] for r in rows) / len(rows),
                            'google_share': sum(r['google_share'] for r in rows) / len(rows),
                            'mass': sum(r['mass'] for r in rows) / len(rows)}
    null_ids = [c['id'] for c in conds if c['id'].startswith('null_')]
    result = {'unit': list(UNIT), 'mean_activation': float(means27[UNIT[1]]), 'ks': KS, 'summary': summary,
              'verdicts': verdicts(summary, null_ids), 'conditions': [{'id': c['id'],
              'ops': [[l, u, v] for (l, u), v in c['ops'].items()]} for c in conds]}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'part72.json').write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    (out / 'records.json').write_text(json.dumps(records) + '\n', encoding='utf-8')
    return result


def main():
    if '--help' in sys.argv:
        print('usage: part72.py OUT_DIR MATCHED_JSON [xpu|cuda|cpu]')
        return
    import neuron_report as nr
    import part25
    from part66 import part66
    out, matched_path = Path(sys.argv[1]), Path(sys.argv[2])
    device = sys.argv[3] if len(sys.argv) > 3 else 'xpu'
    matched = json.loads(matched_path.read_text(encoding='utf-8'))
    matched_units = [int(m['unit']) for m in matched['29_776']['27_3325']]
    assert len(matched_units) == 50
    tok, model = nr.load_model(part66.MODEL, part66.REVISION, device)
    _, layers, _ = nr.find_text_layers(model)
    prompts = [part25._token_ids(tok, nr.build_prompt(tok, q, False) + OPENING) for q in part66.DIRECT + part66.INDIRECT]
    readout_ids = [part25._token_ids(tok, n)[0] for n in NAMES]
    if len(set(readout_ids)) != 8:
        raise SystemExit('readout names share a first token')
    result = run(model, layers, prompts, readout_ids, matched_units, out)
    for cid in ['unchanged'] + [f'k{k}' for k in KS]:
        s = result['summary'][cid]
        print(f"{cid}: dg {s['dg']:+.3f} google_share {s['google_share']:.3f} mass {s['mass']:.3f}", flush=True)
    for key, v in result['verdicts'].items():
        print(key, v['status'], {k: v[k] for k in v if k not in ('holds', 'status')}, flush=True)


if __name__ == '__main__':
    main()
