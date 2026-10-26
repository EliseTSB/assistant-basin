#!/usr/bin/env python3
"""Part 71: what do the two late "Google" neurons write? Signed down_proj columns read through the model's own final
norm and unembedding, against 50 random units per layer (PART71-DESIGN.md). Claude, 2026-10-26."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TARGETS = {'L27 N3325': (27, 3325), 'L24 N3949': (24, 3949)}
MAKERS = {'L6 N753': (6, 753), 'L7 N1169': (7, 1169), 'L11 N8868': (11, 8868), 'L29 N776': (29, 776)}
NAMED = set(TARGETS.values()) | set(MAKERS.values())
OPENING = 'I was developed by **'
NULL_N = 50


def best_rank(logits, ids) -> int:
    """1 + the number of vocabulary entries strictly above the best of `ids` (rank 1 = top)."""
    values = np.asarray(logits, dtype=np.float64)
    best = max(values[i] for i in ids)
    return int(1 + np.sum(values > best))


def null_units(layer: int, n_units: int, count: int = NULL_N) -> list[int]:
    eligible = [u for u in range(n_units) if (layer, u) not in NAMED]
    rng = np.random.default_rng(7100 + layer)
    return sorted(int(u) for u in rng.choice(eligible, size=count, replace=False))


def verdicts(ranks: dict, null_ranks: dict) -> dict:
    r27, r24 = ranks['L27 N3325'], ranks['L24 N3949']
    n27, n24 = null_ranks[27], null_ranks[24]
    median_null = statistics.median(n27 + n24)
    out = {
        '71.1': {'holds': r27 <= 50 and all(r27 < x for x in n27), 'rank': r27, 'best_null': min(n27)},
        '71.2': {'holds': all(r24 < x for x in n24), 'rank': r24, 'best_null': min(n24)},
        '71.3': {'holds': median_null > 1000, 'median_null_rank': median_null},
    }
    for key in ('71.1', '71.2', '71.3'):
        out[key]['status'] = ('HOLDS' if out[key]['holds'] else 'FAILS') if key == '71.3' or out['71.3']['holds'] \
            else 'NOT READ'
    return out


def run(model, tok, layers, norm, head, prompts_ids, google_ids, out: Path):
    import torch
    device = next(model.parameters()).device
    n_units = int(layers[0].mlp.down_proj.in_features)
    need = sorted({l for l, _ in NAMED})
    sums = {l: torch.zeros(n_units, dtype=torch.float64) for l in need}

    def make_hook(l):
        def hook(_module, args):
            sums[l] += args[0][0, -1].detach().double().cpu()
        return hook

    handles = [layers[l].mlp.down_proj.register_forward_pre_hook(make_hook(l)) for l in need]
    try:
        with torch.no_grad():
            for ids in prompts_ids:
                model(input_ids=torch.tensor([ids], device=device))
    finally:
        for h in handles:
            h.remove()
    mean = {l: (sums[l] / len(prompts_ids)).numpy() for l in need}

    def read(l, u):
        w = layers[l].mlp.down_proj.weight[:, u].float()
        sign = 1.0 if mean[l][u] >= 0 else -1.0
        v = (sign * w).to(head.weight.dtype).view(1, 1, -1)
        with torch.no_grad():
            logits = head(norm(v)).float()[0, 0].cpu().numpy()
        top = np.argsort(-logits)[:20]
        return {'layer': l, 'unit': u, 'mean_activation': float(mean[l][u]), 'sign': sign,
                'google_rank': best_rank(logits, google_ids), 'top20': [tok.decode([int(i)]) for i in top]}

    named = {name: read(*lu) for name, lu in {**TARGETS, **MAKERS}.items()}
    nulls = {}
    for l in (27, 24):
        nulls[l] = [read(l, u) for u in null_units(l, n_units)]
    ranks = {name: r['google_rank'] for name, r in named.items()}
    null_ranks = {l: [r['google_rank'] for r in rows] for l, rows in nulls.items()}
    result = {'opening': OPENING, 'google_token_ids': google_ids,
              'google_tokens': [tok.decode([i]) for i in google_ids],
              'named': named, 'nulls': {str(l): rows for l, rows in nulls.items()},
              'verdicts': verdicts(ranks, null_ranks)}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'part71.json').write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return result


def main():
    if '--help' in sys.argv:
        print('usage: part71.py OUT_DIR [xpu|cuda|cpu]')
        return
    import neuron_report as nr
    import part25
    from part66 import part66
    out = Path(sys.argv[1])
    device = sys.argv[2] if len(sys.argv) > 2 else 'xpu'
    tok, model = nr.load_model(part66.MODEL, part66.REVISION, device)
    _, layers, norm = nr.find_text_layers(model)
    if norm is None:
        raise SystemExit('no final norm found beside the decoder layers')
    head = model.get_output_embeddings()
    prompts = [part25._token_ids(tok, nr.build_prompt(tok, q, False) + OPENING) for q in part66.DIRECT + part66.INDIRECT]
    google_ids = sorted({part25._token_ids(tok, 'Google')[0], part25._token_ids(tok, ' Google')[0]})
    result = run(model, tok, layers, norm, head, prompts, google_ids, out)
    for name, r in result['named'].items():
        print(f"{name}: sign {r['sign']:+.0f} google rank {r['google_rank']} top {r['top20'][:10]}", flush=True)
    for key, v in result['verdicts'].items():
        print(key, v['status'], {k: v[k] for k in v if k not in ('holds', 'status')}, flush=True)


if __name__ == '__main__':
    main()
