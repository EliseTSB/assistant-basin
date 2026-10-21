#!/usr/bin/env python3
"""Part 48: do text-only base models pair kiki with spiky and bouba with round? (design: PART48-DESIGN.md)

Teacher-forced log probability of fixed continuation words after a pseudoword prompt, one sequence per forward
(no batching, so no batch-shape effects), fp32. Writes one JSON per model with every prompt's per-word log
probabilities, the mass on the ten words, token splits, and the summary statistics the predictions name.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

ROUND = ['lomu', 'buma', 'molu', 'bulo', 'mobu', 'mulo', 'bomu', 'lumo',
         'bolomu', 'malubo', 'lomumo', 'bumolo', 'mulobu', 'nobuma', 'lumobo', 'bomulu']
SHARP = ['kiti', 'teki', 'kipe', 'peti', 'tepi', 'kete', 'piki', 'teke',
         'kipeti', 'tekipi', 'petiki', 'kitepi', 'tikepe', 'pekiti', 'ketipe', 'tipeki']
CLASSIC = ['bouba', 'kiki', 'maluma', 'takete']
TEMPLATES = [
    'In a made-up language, the word "{w}" describes something that is',
    'Two shapes were drawn on the board, one with smooth curves and one with sharp points. '
    'The shape called "{w}" was the one that was',
    'The creature named {w} had a body that was',
    'A shape called "{w}" is probably',
    'If you touched a {w}, it would feel',
    'The sculptor named the new piece "{w}" because it was so',
]
SHARP_WORDS = ['sharp', 'spiky', 'pointy', 'jagged', 'angular']
ROUND_WORDS = ['round', 'smooth', 'soft', 'curvy', 'rounded']
MODELS = {
    'pythia-410m': 'EleutherAI/pythia-410m',
    'pythia-1.4b': 'EleutherAI/pythia-1.4b',
    'olmo2-1b': 'allenai/OLMo-2-0425-1B',
    'qwen3-1.7b-base': 'Qwen/Qwen3-1.7B-Base',
    'smollm2-1.7b': 'HuggingFaceTB/SmolLM2-1.7B',
}


def spelled(w: str) -> str:
    return ' '.join(w)


def items():
    """(item, kind, label): label 1 = sharp, 0 = round, None = classic."""
    out = [(w, 'novel', 0) for w in ROUND] + [(w, 'novel', 1) for w in SHARP]
    out += [(spelled(w), 'spelled', 0) for w in ROUND] + [(spelled(w), 'spelled', 1) for w in SHARP]
    out += [(w, 'classic', None) for w in CLASSIC]
    return out


def auc(scores, labels) -> float:
    """P(score of a random sharp item > score of a random round item), ties count half."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    total = 0.0
    for p in pos:
        for n in neg:
            total += 1.0 if p > n else (0.5 if p == n else 0.0)
    return total / (len(pos) * len(neg))


def perm_p(scores, labels, draws=10000, seed=48) -> float:
    """Two-sided permutation p for the AUC's distance from 0.5, label shuffles over items."""
    rng = random.Random(seed)
    obs = abs(auc(scores, labels) - 0.5)
    lab = list(labels)
    hits = 0
    for _ in range(draws):
        rng.shuffle(lab)
        if abs(auc(scores, lab) - 0.5) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (draws + 1)


def logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def continuation_logprob(model, tok, prompt_ids, word: str, device: str) -> float:
    """Summed log probability of ' word' (all its tokens) after the prompt, one sequence per forward."""
    cont = tok(' ' + word, add_special_tokens=False)['input_ids']
    ids = torch.tensor([prompt_ids + cont], device=device)
    logits = model(ids).logits[0].float()
    logp = torch.log_softmax(logits, dim=-1)
    start = len(prompt_ids)
    return float(sum(logp[start - 1 + i, t].item() for i, t in enumerate(cont)))


def score_model(model, tok, device: str, limit: int | None = None):
    rows = []
    for w, kind, label in items()[:limit] if limit else items():
        for ti, tmpl in enumerate(TEMPLATES):
            prompt = tmpl.format(w=w)
            pids = tok(prompt, add_special_tokens=True)['input_ids']
            sharp = [continuation_logprob(model, tok, pids, x, device) for x in SHARP_WORDS]
            round_ = [continuation_logprob(model, tok, pids, x, device) for x in ROUND_WORDS]
            mass = sum(math.exp(v) for v in sharp + round_)
            rows.append(dict(item=w, kind=kind, label=label, template=ti,
                             sharp=sharp, round=round_, mass=mass,
                             score=logsumexp(sharp) - logsumexp(round_),
                             item_tokens=tok.convert_ids_to_tokens(tok(w, add_special_tokens=False)['input_ids'])))
    return rows


def summarize(rows, mass_floor=1e-3):
    tmpl_mass = {t: float(np.median([r['mass'] for r in rows if r['template'] == t])) for t in range(len(TEMPLATES))}
    kept = [t for t, m in tmpl_mass.items() if m >= mass_floor]

    def item_scores(kind, templates):
        out = {}
        for r in rows:
            if r['kind'] == kind and r['template'] in templates:
                out.setdefault((r['item'], r['label']), []).append(r['score'])
        return {k: float(np.mean(v)) for k, v in out.items()}

    def stat(kind, templates):
        sc = item_scores(kind, templates)
        if not sc:
            return None
        scores = [v for (_, y), v in sorted(sc.items())]
        labels = [y for (_, y), v in sorted(sc.items())]
        return dict(auc=auc(scores, labels), p=perm_p(scores, labels), n_items=len(scores),
                    templates=sorted(templates))

    no_t2 = [t for t in kept if t != 1]
    classic = item_scores('classic', kept)
    return dict(template_median_mass=tmpl_mass, templates_kept=kept,
                novel=stat('novel', kept), novel_without_template2=stat('novel', no_t2),
                spelled=stat('spelled', kept),
                classic={k[0]: v for k, v in classic.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--device', default='xpu')
    ap.add_argument('--out', default='results/part48')
    ap.add_argument('--limit', type=int, default=None, help='smoke test: first N items only')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers.modeling_utils as _mu
    if hasattr(_mu, 'caching_allocator_warmup'):
        _mu.caching_allocator_warmup = lambda *x, **k: None
    torch.set_grad_enabled(False)
    for short in a.models.split(','):
        mid = MODELS[short]
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(mid)
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32, low_cpu_mem_usage=True,
                                                     device_map={'': a.device})
        model.eval()
        rows = score_model(model, tok, a.device, a.limit)
        summary = summarize(rows)
        rec = dict(part=48, model=mid, revision=getattr(model.config, '_commit_hash', None), dtype='float32',
                   batching='one sequence per forward', limit=a.limit, seconds=round(time.time() - t0, 1),
                   summary=summary, rows=rows)
        path = os.path.join(a.out, f'{short}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(rec, f, indent=1)
        print(short, json.dumps({k: summary[k] for k in ('novel', 'spelled', 'classic')}), flush=True)
        del model
        if a.device == 'xpu':
            torch.xpu.empty_cache()


if __name__ == '__main__':
    main()
