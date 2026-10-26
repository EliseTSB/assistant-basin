#!/usr/bin/env python3
"""Part 75 sampled generation: with L24 N3949 zeroed, clamp L27 N3325 to its makers-only mean (PART75-DESIGN.md).
Claude, 2026-10-26."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Run as a script, Python puts this folder first on sys.path; drop it (the Part 74b import trap).
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if not (p and Path(p).resolve() == _HERE)]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part70 import part70  # noqa: E402

L27 = (27, 3325)
L24 = (24, 3949)
N_ANSWERS = 960


def build_conditions(m: float) -> list[dict]:
    target = list(part70.TARGET)
    return [
        {'id': 'target', 'set_units': target, 'kind': 'target'},
        {'id': 'target_plus_l24', 'set_units': target + [L24], 'kind': 'target_plus_l24'},
        {'id': 'target_plus_l24_clamp_l27', 'set_units': target + [L24], 'clamp': {L27: m}, 'kind': 'clamp'},
        {'id': 'target_clamp_l27', 'set_units': target, 'clamp': {L27: m}, 'kind': 'clamp'},
    ]


def makers_only_mean(model, layers, prompt_ids: list[list[int]]) -> float:
    """L27 N3325's mean down_proj input over all positions of the given prompts, with the makers zeroed."""
    import torch
    import part25
    device = next(model.parameters()).device
    total, count = 0.0, 0

    def rec(_m, args):
        nonlocal total, count
        x = args[0][0, :, L27[1]].detach().double()
        total += float(x.sum())
        count += int(x.shape[0])
    ops = [part25._operations(part70.TARGET, 'zero')]
    for prompt in prompt_ids:
        ids = torch.tensor([prompt], device=device)
        handles = part25.install_intervention_hooks(layers, ops)
        handles.append(layers[L27[0]].mlp.down_proj.register_forward_pre_hook(rec))
        try:
            with torch.no_grad():
                model(input_ids=ids)
        finally:
            for h in handles:
                h.remove()
    return total / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', choices=['xpu', 'cuda', 'cpu'], default='xpu')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--max-gpu-gb', type=float, default=14)
    args = parser.parse_args()
    import neuron_report as nr
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if len(part70.generations(build_conditions(0.0))) != N_ANSWERS:
        raise RuntimeError(f'the design registers {N_ANSWERS} answers')
    tok, model = nr.load_model(part70.MODEL, part70.REVISION, args.device)
    _, layers, _ = nr.find_text_layers(model)
    import part25
    p66 = part70._require_part66()
    m = makers_only_mean(model, layers, [part25._token_ids(tok, nr.build_prompt(tok, q, False))
                                         for q in p66.DIRECT + p66.INDIRECT])
    (out / 'm.json').write_text(json.dumps({'unit': list(L27), 'makers_only_mean_over_prompt_positions': m}) + '\n',
                                encoding='utf-8')
    print(f'part75: m = {m:.4f}', flush=True)
    conditions = build_conditions(m)
    (out / 'conditions.json').write_text(json.dumps([
        {'id': c['id'], 'set_units': part70._json_units(c['set_units']),
         'clamp': [[l, u, v] for (l, u), v in (c.get('clamp') or {}).items()]} for c in conditions
    ], indent=2) + '\n', encoding='utf-8')

    def guard(dev, _limit):
        return nr.spill_guard(dev, args.max_gpu_gb)
    part70.run_70(model, tok, layers, conditions, out, args.batch, args.device, guard)
    run_path = out / 'run.json'
    run = json.loads(run_path.read_text(encoding='utf-8'))
    run['part75_tool_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    run_path.write_text(json.dumps(run, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
