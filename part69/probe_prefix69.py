#!/usr/bin/env python3
"""Readout check for Part 69 after 69.2 failed (Claude, 2026-10-26): what does the unchanged model put next after each
candidate forced opening, and how much probability do the seven maker tokens get? Unchanged model only: no condition
is run, so the choice of opening cannot see any condition's result."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import neuron_report as nr
import part25
from part66 import part66

PREFIXES = ['I was developed by', 'I was developed by the', 'I am Qwen, a large language model developed by',
            'I was created by', 'My developer is', 'The company that made me is']
NAMES = ['Alibaba', 'Google', 'OpenAI', 'Anthropic', 'DeepSeek', 'Meta', 'Microsoft']

def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    tok, model = nr.load_model(part66.MODEL, part66.REVISION, sys.argv[2] if len(sys.argv) > 2 else 'xpu')
    dev = part25.actual_device(model)
    qs = part66.DIRECT + part66.INDIRECT
    ids7 = [part25._token_ids(tok, ' ' + n)[0] for n in NAMES]
    report = {}
    for prefix in PREFIXES:
        masses, tops, alib_top = [], {}, 0
        for q in qs:
            ids = part25._token_ids(tok, nr.build_prompt(tok, q, False) + prefix)
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([ids], device=dev)).logits[0, -1].float()
            probs = logits.softmax(-1)
            masses.append(float(probs[ids7].sum()))
            alib_top += int(int(torch.argmax(logits[ids7])) == 0)
            for t in torch.topk(probs, 3).indices.tolist():
                key = tok.decode([t]); tops[key] = tops.get(key, 0) + 1
        report[prefix] = {'mean_mass': sum(masses) / len(masses), 'min_mass': min(masses),
                          'alibaba_top_of_seven': alib_top,
                          'top3_token_counts': dict(sorted(tops.items(), key=lambda kv: -kv[1])[:8])}
        print(prefix, '|', round(report[prefix]['mean_mass'], 4), '|', alib_top, '|', report[prefix]['top3_token_counts'], flush=True)
    (out / 'prefix_probe.json').write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')

if __name__ == '__main__':
    main()
