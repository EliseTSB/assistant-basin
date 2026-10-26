#!/usr/bin/env python3
"""Part 74b: is L27 N3325 specific as the route for L24 N3949's effect? (PART74B-DESIGN.md). Claude, 2026-10-26.

Per question, with the makers zeroed: pass one runs A and B, recording in B every listed layer-27 unit's down_proj
input; pass two runs E (L27 N3325 set to its B values) and E_null_k (matched layer-27 unit k set to its own B values).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Run as a script, Python puts this folder first on sys.path, where `part74` is part74/part74.py rather than the folder;
# drop it so the folder is found (the 15:44 A770 launch failed on exactly this, and pytest hides it).
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if not (p and Path(p).resolve() == _HERE)]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part74 import part74 as p74  # noqa: E402

LAYER = 27


def null_units27(matched: dict) -> list[tuple[int, int]]:
    """The layer-27 unit of each of Part 70's 20 matched pairs: the same units Part 73 used."""
    from part70 import part70
    return [part70.draw_matched_units(matched, k)[0] for k in range(p74.NULL_N)]


def rows(target: list, nulls: list) -> tuple[list[dict], list[dict]]:
    one = [{'id': 'A', 'zero': list(target)}, {'id': 'B', 'zero': list(target) + [p74.L24]}]
    two = [{'id': 'E', 'zero': list(target), 'unit': p74.L27[1]}]
    two += [{'id': f'E_null_{k}', 'zero': list(target), 'unit': u} for k, (_, u) in enumerate(nulls)]
    return one, two


def install_record_units(layer, units: list[int], row: int, store: dict):
    def hook(_m, args):
        for u in units:
            store[u] = args[0][row, :, u].detach().clone()
        return None
    return layer.mlp.down_proj.register_forward_pre_hook(hook)


def install_patch_rows(layer, patches: dict):
    """patches: {row: (unit, [seq] tensor)}; each row gets its own unit's column written."""
    def hook(_m, args):
        x = args[0].clone()
        for row, (unit, values) in patches.items():
            x[row, :, unit] = values.to(x.dtype)
        return (x,) + tuple(args[1:])
    return layer.mlp.down_proj.register_forward_pre_hook(hook)


def _forward(model, layers, ids, rws, readout_ids, extra_hooks):
    import torch
    import part25
    device = next(model.parameters()).device
    x = torch.tensor([ids] * len(rws), device=device)
    handles = part25.install_intervention_hooks(layers, [{(int(l), int(u)): 0.0 for l, u in r['zero']} for r in rws])
    handles += [h(layers) for h in extra_hooks]
    try:
        with torch.no_grad():
            logits = model(input_ids=x).logits[:, -1, :].float()
    finally:
        for h in handles:
            h.remove()
    return logits[:, readout_ids], logits.softmax(-1)[:, readout_ids].sum(-1)


def verdicts(eff: dict, null_ids: list[str], part74_te=-2.234375, part74_pie=-1.421875) -> dict:
    control = abs(eff['B'] - part74_te) <= 0.1 and abs(eff['E'] - part74_pie) <= 0.1
    beaten = sum(1 for n in null_ids if eff['E'] < eff[n])
    v = {'74b.2': {'holds': control, 'TE': eff['B'], 'PIE_L27_N3325': eff['E']},
         '74b.1': {'holds': beaten >= 18, 'nulls_beaten': beaten, 'of': len(null_ids),
                   'null_min': min(eff[n] for n in null_ids), 'null_max': max(eff[n] for n in null_ids)}}
    v['74b.2']['status'] = 'HOLDS' if control else 'FAILS'
    v['74b.1']['status'] = ('HOLDS' if v['74b.1']['holds'] else 'FAILS') if control else 'NOT READ'
    return v


def run(model, layers, prompts, readout_ids, nulls, out: Path):
    from part70 import part70
    one, two = rows(list(part70.TARGET), nulls)
    units = [p74.L27[1]] + [u for _, u in nulls]
    records, moves = [], []
    for qi, ids in enumerate(prompts):
        rec, rec_a = {}, {}   # B's and A's recorded layer-27 values
        sel, mass = _forward(model, layers, ids, one, readout_ids,
                             [lambda L: install_record_units(L[LAYER], units, 0, rec_a),
                              lambda L: install_record_units(L[LAYER], units, 1, rec)])
        for ri, r in enumerate(one):
            records.append({'q': qi, 'id': r['id'], 'g': float(sel[ri, p74.G] - sel[ri, p74.AL]), 'mass': float(mass[ri])})
        moves.append({'q': qi, 'last_A': {u: float(rec_a[u][-1]) for u in units},
                      'last_B': {u: float(rec[u][-1]) for u in units}})
        patches = {ri: (r['unit'], rec[r['unit']]) for ri, r in enumerate(two)}
        sel2, mass2 = _forward(model, layers, ids, two, readout_ids, [lambda L: install_patch_rows(L[LAYER], patches)])
        for ri, r in enumerate(two):
            records.append({'q': qi, 'id': r['id'], 'g': float(sel2[ri, p74.G] - sel2[ri, p74.AL]),
                            'mass': float(mass2[ri]), 'unit': int(r['unit'])})
    eff = p74.effects(records)
    null_ids = [r['id'] for r in two if r['id'].startswith('E_null_')]
    move_mean = {u: sum(m['last_B'][u] - m['last_A'][u] for m in moves) / len(moves) for u in units}
    result = {'effects': eff, 'verdicts': verdicts(eff, null_ids), 'nulls': [[l, u] for l, u in nulls],
              'last_position_move_B_minus_A': {str(u): v for u, v in move_mean.items()}, 'n_questions': len(prompts)}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'part74b.json').write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    (out / 'records.json').write_text(json.dumps({'records': records}) + '\n', encoding='utf-8')
    return result


def main():
    if '--help' in sys.argv:
        print('usage: part74b.py OUT_DIR MATCHED_JSON [xpu|cuda|cpu]')
        return
    out, matched_path = Path(sys.argv[1]), Path(sys.argv[2])
    device = sys.argv[3] if len(sys.argv) > 3 else 'xpu'
    nulls = null_units27(json.loads(matched_path.read_text(encoding='utf-8')))
    import neuron_report as nr
    import part25
    from part66 import part66
    tok, model = nr.load_model(part66.MODEL, part66.REVISION, device)
    _, layers, _ = nr.find_text_layers(model)
    prompts = [part25._token_ids(tok, nr.build_prompt(tok, q, False) + p74.OPENING)
               for q in part66.DIRECT + part66.INDIRECT]
    readout_ids = [part25._token_ids(tok, n)[0] for n in p74.NAMES]
    if len(set(readout_ids)) != 8:
        raise SystemExit('readout names share a first token')
    result = run(model, layers, prompts, readout_ids, nulls, out)
    print('effects', {k: round(v, 3) for k, v in result['effects'].items()}, flush=True)
    for key, v in result['verdicts'].items():
        print(key, v['status'], {k: v[k] for k in v if k not in ('holds', 'status')}, flush=True)


if __name__ == '__main__':
    main()
