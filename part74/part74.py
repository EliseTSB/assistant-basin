#!/usr/bin/env python3
"""Part 74: does L24 N3949 cut the Google fallback through L27 N3325? (PART74-DESIGN.md). Claude, 2026-10-26.

Per question, with the four maker neurons zeroed: A (makers), B (+ L24 N3949 zeroed), C (B with L27 N3325 patched at
every position to its value in A), E (A with L27 N3325 patched to its value in B), A_self (A with L27 N3325 patched to
its own value in A: Amendment 1's noise check), and 20 layer-24 nulls from Part 70's draws. Readout: Google minus
Alibaba logit at the last position after Part 72's opening.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

L27 = (27, 3325)
L24 = (24, 3949)
NULL_N = 20
OPENING = 'I was developed by **'
NAMES = ['Alibaba', 'Google', 'OpenAI', 'Anthropic', 'DeepSeek', 'Meta', 'Microsoft', 'Tongyi']  # Part 72's eight
G, AL = 1, 0  # indices of Google and Alibaba in NAMES


def null_units(matched: dict) -> list[tuple[int, int]]:
    """The layer-24 unit of each of Part 70's 20 matched pairs: the same units Part 73 used."""
    from part70 import part70
    return [part70.draw_matched_units(matched, k)[1] for k in range(NULL_N)]


def pass_one(target: list, nulls: list) -> list[dict]:
    """Rows of the first forward pass. A and B are recorded; nulls are read only."""
    rows = [{'id': 'A', 'zero': list(target)}, {'id': 'B', 'zero': list(target) + [L24]}]
    rows += [{'id': f'null_{k}', 'zero': list(target) + [u], 'unit': list(u)} for k, u in enumerate(nulls)]
    return rows


def pass_two(target: list) -> list[dict]:
    """Rows of the second pass. `patch_from` names the pass-one row whose L27 N3325 values are written in."""
    return [
        {'id': 'C', 'zero': list(target) + [L24], 'patch_from': 'A'},
        {'id': 'E', 'zero': list(target), 'patch_from': 'B'},
        {'id': 'A_self', 'zero': list(target), 'patch_from': 'A'},
    ]


def effects(records: list[dict]) -> dict:
    """Mean change in g against A, per condition id, over questions."""
    by_q = {}
    for r in records:
        by_q.setdefault(r['q'], {})[r['id']] = r
    ids = sorted({r['id'] for r in records})
    out = {}
    for cid in ids:
        diffs = [qd[cid]['g'] - qd['A']['g'] for qd in by_q.values() if cid in qd and 'A' in qd]
        out[cid] = sum(diffs) / len(diffs)
    return out


def verdicts(eff: dict, mass_a: float, null_ids: list[str]) -> dict:
    te, de, ie, noise = eff['B'], eff['C'], eff['E'], eff['A_self']
    beaten = sum(1 for n in null_ids if te < eff[n])
    control = te < 0 and beaten >= 18 and mass_a >= 0.5
    noise_ok = abs(noise) < 0.1 * abs(te) if te else False
    v = {
        '74.1': {'holds': control, 'TE': te, 'nulls_beaten': beaten, 'of': len(null_ids), 'mass_A': mass_a},
        '74.2': {'holds': te < 0 and de / te <= 0.5, 'DE': de, 'DE_over_TE': de / te if te else None},
        '74.3': {'holds': te < 0 and ie / te >= 0.5, 'IE': ie, 'IE_over_TE': ie / te if te else None},
        'amendment1_noise': {'holds': noise_ok, 'A_self_minus_A': noise},
    }
    read = control and noise_ok
    for key in ('74.2', '74.3'):
        v[key]['status'] = ('HOLDS' if v[key]['holds'] else 'FAILS') if read else 'NOT READ'
    v['74.1']['status'] = 'HOLDS' if control else 'FAILS'
    v['amendment1_noise']['status'] = 'HOLDS' if noise_ok else 'FAILS'
    return v


def install_patch_hook(layer, unit: int, patches: dict):
    """Write `patches[row]` (a [seq] tensor) into column `unit` of layer's down_proj input, for those rows only."""
    def hook(_m, args):
        if not patches:
            return None
        x = args[0].clone()
        for row, values in patches.items():
            x[row, :, unit] = values.to(x.dtype)
        return (x,) + tuple(args[1:])
    return layer.mlp.down_proj.register_forward_pre_hook(hook)


def install_record_hook(layer, unit: int, store: dict, rows: list[int]):
    def hook(_m, args):
        for row in rows:
            store[row] = args[0][row, :, unit].detach().clone()
        return None
    return layer.mlp.down_proj.register_forward_pre_hook(hook)


def forward(model, layers, ids, rows, readout_ids, patches=None, record_rows=None):
    """One batched forward of identical token sequences; returns logits rows and any recorded L27 N3325 values."""
    import torch
    import part25
    device = next(model.parameters()).device
    x = torch.tensor([ids] * len(rows), device=device)
    ops = [{(int(l), int(u)): 0.0 for l, u in r['zero']} for r in rows]
    handles = part25.install_intervention_hooks(layers, ops)
    rec = {}
    if record_rows:
        handles.append(install_record_hook(layers[L27[0]], L27[1], rec, record_rows))
    if patches:
        handles.append(install_patch_hook(layers[L27[0]], L27[1], patches))
    try:
        with torch.no_grad():
            logits = model(input_ids=x).logits[:, -1, :].float()
    finally:
        for h in handles:
            h.remove()
    sel = logits[:, readout_ids]
    mass = logits.softmax(-1)[:, readout_ids].sum(-1)
    return sel, mass, rec


def run(model, layers, prompts, readout_ids, nulls, out: Path):
    from part70 import part70
    target = list(part70.TARGET)
    one, two = pass_one(target, nulls), pass_two(target)
    records, acts = [], []
    for qi, ids in enumerate(prompts):
        sel, mass, rec = forward(model, layers, ids, one, readout_ids, record_rows=[0, 1])
        for ri, r in enumerate(one):
            records.append({'q': qi, 'id': r['id'], 'g': float(sel[ri, G] - sel[ri, AL]), 'mass': float(mass[ri])})
        src = {'A': rec[0], 'B': rec[1]}
        acts.append({'q': qi, 'A_last': float(rec[0][-1]), 'B_last': float(rec[1][-1])})
        patches = {ri: src[r['patch_from']] for ri, r in enumerate(two)}
        sel2, mass2, _ = forward(model, layers, ids, two, readout_ids, patches=patches)
        for ri, r in enumerate(two):
            records.append({'q': qi, 'id': r['id'], 'g': float(sel2[ri, G] - sel2[ri, AL]), 'mass': float(mass2[ri])})
    eff = effects(records)
    mass_a = sum(r['mass'] for r in records if r['id'] == 'A') / len(prompts)
    null_ids = [r['id'] for r in one if r['id'].startswith('null_')]
    result = {'effects': eff, 'mass_A': mass_a, 'verdicts': verdicts(eff, mass_a, null_ids),
              'l27_last_position': {'A_mean': sum(a['A_last'] for a in acts) / len(acts),
                                    'B_mean': sum(a['B_last'] for a in acts) / len(acts)},
              'nulls': [[int(l), int(u)] for l, u in nulls], 'n_questions': len(prompts)}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'part74.json').write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    (out / 'records.json').write_text(json.dumps({'records': records, 'acts': acts}) + '\n', encoding='utf-8')
    return result


def main():
    if '--help' in sys.argv:
        print('usage: part74.py OUT_DIR MATCHED_JSON [xpu|cuda|cpu]')
        return
    out, matched_path = Path(sys.argv[1]), Path(sys.argv[2])
    device = sys.argv[3] if len(sys.argv) > 3 else 'xpu'
    matched = json.loads(matched_path.read_text(encoding='utf-8'))
    nulls = null_units(matched)
    import neuron_report as nr
    import part25
    from part66 import part66
    tok, model = nr.load_model(part66.MODEL, part66.REVISION, device)
    _, layers, _ = nr.find_text_layers(model)
    prompts = [part25._token_ids(tok, nr.build_prompt(tok, q, False) + OPENING) for q in part66.DIRECT + part66.INDIRECT]
    readout_ids = [part25._token_ids(tok, n)[0] for n in NAMES]
    if len(set(readout_ids)) != 8:
        raise SystemExit('readout names share a first token')
    result = run(model, layers, prompts, readout_ids, nulls, out)
    print('effects', {k: round(v, 3) for k, v in result['effects'].items() if not k.startswith('null_')}, flush=True)
    for key, v in result['verdicts'].items():
        print(key, v['status'], {k: v[k] for k in v if k not in ('holds', 'status')}, flush=True)


if __name__ == '__main__':
    main()
