#!/usr/bin/env python3
"""Part 76: which layer-27 units carry L24 N3949's effect, and does writing Google suffice? (PART76-DESIGN.md).
Claude, 2026-10-26.

Phase one, per question, with the makers zeroed: A and B record every layer-27 unit's down_proj input; one backward
pass in A gives dg/da for every layer-27 unit; PIE_hat(u) = sum over positions of (a_B - a_A) * dg/da. Then every
layer-27 unit's direct Google rank (Part 71's method). Phase two, per question: exact E(u) rows for the validation
set (L27 N3325, the 10 most negative estimates, the 10 best Google writers, 5 random units), E_all27, E_last, A_self.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Run as a script, Python puts this folder first on sys.path; drop it so the package folders are found (Part 74b).
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if not (p and Path(p).resolve() == _HERE)]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part74 import part74 as p74  # noqa: E402
from part74 import part74b as p74b  # noqa: E402

LAYER = 27
N_TOP, N_GOOGLE, N_RANDOM, SEED = 10, 10, 5, 76
PART74B_PIE = -1.45


def record_rows(layer, store: dict, names: dict):
    """names: {row: key}; stores each row's whole down_proj input [seq, units] as float32 on the CPU."""
    def hook(_m, args):
        for row, key in names.items():
            store[key] = args[0][row].detach().float().cpu()
        return None
    return layer.mlp.down_proj.register_forward_pre_hook(hook)


def install_patches(layer, patches: dict):
    """patches: {row: ('unit', u, values[seq]) | ('all', values[seq, units]) | ('last', u, value)}."""
    def hook(_m, args):
        x = args[0].clone()
        for row, spec in patches.items():
            if spec[0] == 'unit':
                x[row, :, spec[1]] = spec[2].to(x.device, x.dtype)
            elif spec[0] == 'all':
                x[row] = spec[1].to(x.device, x.dtype)
            elif spec[0] == 'last':
                x[row, -1, spec[1]] = float(spec[2])
            else:
                raise ValueError(f'unknown patch kind {spec[0]!r}')
        return (x,) + tuple(args[1:])
    return layer.mlp.down_proj.register_forward_pre_hook(hook)


def gradient_in_a(model, layers, ids, readout_ids, target):
    """One forward in A with autograd on; returns (a_A [seq, units], dg/da [seq, units], g) on the CPU in float32."""
    import torch
    import part25
    device = next(model.parameters()).device
    captured = {}

    def hook(_m, args):
        a = args[0]
        if not a.requires_grad:   # weights frozen: the captured activation is a leaf; make it the thing we differentiate
            a.requires_grad_(True)
        a.retain_grad()
        captured['a'] = a
        return None
    handles = part25.install_intervention_hooks(layers, [part25._operations(target, 'zero')])
    handles.append(layers[LAYER].mlp.down_proj.register_forward_pre_hook(hook))
    try:
        with torch.enable_grad():   # nr.load_model disables autograd globally (Part 68)
            logits = model(input_ids=torch.tensor([ids], device=device)).logits[0, -1, :].float()
            g = logits[readout_ids[p74.G]] - logits[readout_ids[p74.AL]]
            g.backward()
    finally:
        for h in handles:
            h.remove()
    a = captured['a']
    if a.grad is None:
        raise RuntimeError('no gradient reached layer 27\'s down_proj input')
    return a.detach()[0].float().cpu(), a.grad[0].float().cpu(), float(g.detach())


def google_ranks_all(layers, norm, head, sign, google_ids: list[int], chunk: int = 256,
                     with_margin: bool = False):
    """Part 71's direct Google rank for every layer-27 unit, vectorised: p74b.google_ranks applied to all units.
    with_margin also returns, per unit, the best Google logit minus the best other token's logit (the tie-break)."""
    import torch
    w = layers[LAYER].mlp.down_proj.weight   # [hidden, units]
    n = w.shape[1]
    ranks, margins = [], []
    with torch.no_grad():
        for start in range(0, n, chunk):
            cols = w[:, start:start + chunk].float().T   # [c, hidden]
            s = torch.tensor(sign[start:start + chunk], dtype=cols.dtype, device=cols.device).view(-1, 1)
            v = (s * cols).to(head.weight.dtype).unsqueeze(1)   # [c, 1, hidden]
            logits = head(norm(v)).float()[:, 0, :]             # [c, vocab]
            best = logits[:, google_ids].max(dim=1, keepdim=True).values
            ranks += (1 + (logits > best).sum(dim=1)).cpu().tolist()
            others = logits.clone()
            others[:, google_ids] = float('-inf')
            margins += (best[:, 0] - others.max(dim=1).values).cpu().tolist()
    ranks = [int(r) for r in ranks]
    return (ranks, [float(m) for m in margins]) if with_margin else ranks


def validation_set(pie_hat: list[float], ranks: list[int], target_unit: int, margins: list[float] | None = None) -> dict:
    """The units E(u) is run for, by reason. Overlaps are run once (the `units` list). Google writers sort by rank,
    then by margin (a stronger writer first), then by unit."""
    others = [u for u in range(len(pie_hat)) if u != target_unit]
    top = sorted(others, key=lambda u: pie_hat[u])[:N_TOP]
    margins = margins or [0.0] * len(pie_hat)
    google = sorted(others, key=lambda u: (ranks[u], -margins[u], u))[:N_GOOGLE]
    rng = random.Random(SEED)
    rand = rng.sample(others, N_RANDOM)
    units = [target_unit]
    for u in top + google + rand:
        if u not in units:
            units.append(u)
    return {'top_estimate': top, 'google_writers': google, 'random': rand, 'units': units}


def verdicts(eff: dict, pie_hat_target: float, by_reason: dict, target_unit: int, part74b_pie: float = PART74B_PIE) -> dict:
    pie = {int(k[2:]): v for k, v in eff.items() if k.startswith('E_') and k[2:].isdigit()}
    t = pie[target_unit]
    control = abs(t - part74b_pie) <= 0.1 and abs(eff['A_self']) <= 0.05
    v = {'76.1': {'holds': control, 'PIE_L27_N3325': t, 'A_self': eff['A_self']}}
    ratio = pie_hat_target / t if t else float('inf')
    v['76.2'] = {'holds': (pie_hat_target < 0) == (t < 0) and 0.5 <= ratio <= 2.0,
                 'PIE_hat_L27_N3325': pie_hat_target, 'ratio_hat_to_exact': ratio}
    others = {u: p for u, p in pie.items() if u != target_unit}
    v['76.3'] = {'holds': t <= min(others.values()) and all(p > -0.2 for p in others.values()),
                 'most_negative_other': min(others.items(), key=lambda kv: kv[1]) if others else None}
    g = {u: pie[u] for u in by_reason['google_writers']}
    v['76.4'] = {'holds': all(p > -0.2 for p in g.values()), 'google_writer_pie': {str(u): p for u, p in g.items()}}
    all27 = eff['E_all27']
    v['76.5'] = {'holds': all27 < 0 and t / all27 >= 0.5, 'PIE_all27': all27,
                 'share': (t / all27) if all27 else None}
    v['76.6'] = {'holds': t < 0 and eff['E_last'] / t >= 0.6, 'PIE_last_only': eff['E_last'],
                 'share': (eff['E_last'] / t) if t else None}
    for key in v:
        if key == '76.1':
            v[key]['status'] = 'HOLDS' if control else 'FAILS'
        else:
            v[key]['status'] = ('HOLDS' if v[key]['holds'] else 'FAILS') if control else 'NOT READ'
    return v


def run(model, layers, prompts, readout_ids, out: Path, norm, head, google_ids, target_unit: int = 3325,
        part74b_pie: float = PART74B_PIE):
    import numpy as np
    from part70 import part70
    # nr.load_model only turns autograd off globally; without this, the backward pass in gradient_in_a computes a
    # gradient for every weight (about 8 GB more on the A770, which spills into Windows RAM: the first launch,
    # 2026-10-26 20:53 CT, was killed by job.py's RAM guard 35 s in). Part 68 does the same at its line 236.
    model.requires_grad_(False)
    target = list(part70.TARGET)
    one = [{'id': 'A', 'zero': target}, {'id': 'B', 'zero': target + [p74.L24]}]
    n_units = int(layers[LAYER].mlp.down_proj.in_features)
    records, acts = [], []
    pie_hat_q = np.zeros((len(prompts), n_units), dtype=np.float64)
    last_a = np.zeros(n_units, dtype=np.float64)
    for qi, ids in enumerate(prompts):
        store = {}
        sel, mass = p74b._forward(model, layers, ids, one, readout_ids,
                                  [lambda L: record_rows(L[LAYER], store, {0: 'A', 1: 'B'})])
        for ri, r in enumerate(one):
            records.append({'q': qi, 'id': r['id'], 'g': float(sel[ri, p74.G] - sel[ri, p74.AL]), 'mass': float(mass[ri])})
        _, grad, _ = gradient_in_a(model, layers, ids, readout_ids, target)
        diff = (store['B'] - store['A']).numpy().astype(np.float64)
        pie_hat_q[qi] = (diff * grad.numpy().astype(np.float64)).sum(axis=0)
        last_a += store['A'][-1].numpy().astype(np.float64)
        acts.append(store)
    pie_hat = pie_hat_q.mean(axis=0)
    sign = [1.0 if x >= 0 else -1.0 for x in (last_a / len(prompts))]
    ranks, margins = google_ranks_all(layers, norm, head, sign, google_ids, with_margin=True)
    by_reason = validation_set(pie_hat.tolist(), ranks, target_unit, margins)
    two = [{'id': 'A_self', 'zero': target, 'patch': ('unit', target_unit, 'A')},
           {'id': 'E_all27', 'zero': target, 'patch': ('all', None, 'B')},
           {'id': 'E_last', 'zero': target, 'patch': ('last', target_unit, 'B')}]
    two += [{'id': f'E_{u}', 'zero': target, 'patch': ('unit', u, 'B')} for u in by_reason['units']]
    for qi, ids in enumerate(prompts):
        store = acts[qi]
        patches = {}
        for ri, r in enumerate(two):
            kind, u, src = r['patch']
            if kind == 'unit':
                patches[ri] = ('unit', u, store[src][:, u])
            elif kind == 'all':
                patches[ri] = ('all', store[src])
            else:
                patches[ri] = ('last', u, float(store[src][-1, u]))
        sel2, mass2 = p74b._forward(model, layers, ids, two, readout_ids,
                                    [lambda L, p=patches: install_patches(L[LAYER], p)])
        for ri, r in enumerate(two):
            records.append({'q': qi, 'id': r['id'], 'g': float(sel2[ri, p74.G] - sel2[ri, p74.AL]), 'mass': float(mass2[ri])})
    eff = p74.effects(records)
    result = {'effects': eff, 'verdicts': verdicts(eff, float(pie_hat[target_unit]), by_reason, target_unit, part74b_pie),
              'validation': by_reason,
              'pie_hat_rank_of_target': int(1 + np.sum(pie_hat < pie_hat[target_unit])),
              'pie_hat_sum_all_units': float(pie_hat.sum()),
              'pie_hat_validated': {str(u): float(pie_hat[u]) for u in by_reason['units']},
              'google_rank_validated': {str(u): ranks[u] for u in by_reason['units']},
              'google_margin_validated': {str(u): margins[u] for u in by_reason['units']},
              'move_last_B_minus_A': {str(u): float(np.mean([s['B'][-1, u] - s['A'][-1, u] for s in acts]))
                                      for u in by_reason['units']},
              'n_questions': len(prompts), 'n_units': n_units}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'part76.json').write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    (out / 'records.json').write_text(json.dumps({'records': records}) + '\n', encoding='utf-8')
    (out / 'pie_hat_and_ranks.json').write_text(json.dumps({'pie_hat': pie_hat.tolist(), 'google_rank': ranks,
                                                            'google_margin': margins, 'sign': sign}) + '\n', encoding='utf-8')
    return result


def main():
    if '--help' in sys.argv:
        print('usage: part76.py OUT_DIR [xpu|cuda|cpu]')
        return
    out = Path(sys.argv[1])
    device = sys.argv[2] if len(sys.argv) > 2 else 'xpu'
    import neuron_report as nr
    import part25
    from part66 import part66
    tok, model = nr.load_model(part66.MODEL, part66.REVISION, device)
    _, layers, norm = nr.find_text_layers(model)
    if norm is None:
        raise SystemExit('no final norm found beside the decoder layers')
    head = model.get_output_embeddings()
    google_ids = sorted({part25._token_ids(tok, 'Google')[0], part25._token_ids(tok, ' Google')[0]})
    prompts = [part25._token_ids(tok, nr.build_prompt(tok, q, False) + p74.OPENING)
               for q in part66.DIRECT + part66.INDIRECT]
    readout_ids = [part25._token_ids(tok, n)[0] for n in p74.NAMES]
    if len(set(readout_ids)) != 8:
        raise SystemExit('readout names share a first token')
    result = run(model, layers, prompts, readout_ids, out, norm=norm, head=head, google_ids=google_ids)
    print('effects', {k: round(v, 3) for k, v in result['effects'].items()}, flush=True)
    for key, v in result['verdicts'].items():
        print(key, v['status'], {k: v[k] for k in v if k not in ('holds', 'status')}, flush=True)


if __name__ == '__main__':
    main()
