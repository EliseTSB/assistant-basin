#!/usr/bin/env python3
"""diag178h: on the Arc, is a LARGE UNPADDED batch as bad as batch 1, and does an explicit attention mask matter?

A diagnostic, not a registered part; it changes no registered number. Background: notes in the Journal at
notes/reports/part178-review/BATCH1-AUDIT-NOTES.md. Measured so far (diag178f, diag178g): Qwen3.5 bf16 with SDPA leaves
the CPU float32 greedy path at confident positions once the cache passes ~64 keys at batch 1 and at unpadded batch 2;
at batch 16 with LEFT PADDING AND AN ATTENTION MASK it is close to the eager kernel. Never measured: an unpadded batch
of 16 or 64 identical-length rows (the shape of ~20 public parts that batch one prompt across many pushes), and
whether passing an explicit all-ones mask is what moves SDPA onto the better path.

Configs (bf16), each over the 8 diag178f prompts, greedy 96 new tokens, every row in a batch a copy of one prompt:
  sdpa b1 nomask, sdpa b1 mask, sdpa b16 nomask, sdpa b16 mask, sdpa b64 nomask, eager b16 nomask (control).
Margins are recorded by a logits processor (top-two gap of the scores used for the argmax), so no per-step logits are
kept for 64 rows. Writes results/part178/diag178h/<stage>.jsonl; compare in WSL against diag178f_cpu_ref.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diag178f import MODELS, QUESTIONS, build_prompt, eos_ids, load  # noqa: E402

ROOT = HERE.parent
MAXNEW = 96
CONFIGS = [('sdpa', 1, False), ('sdpa', 1, True), ('sdpa', 16, False), ('sdpa', 16, True), ('sdpa', 64, False),
           ('eager', 16, False)]


class MarginRecorder:
    """Records each row's top-two score gap at every step; returns the scores unchanged."""

    def __init__(self):
        self.steps = []

    def __call__(self, input_ids, scores):
        top = scores.float().topk(2, dim=-1).values
        self.steps.append((top[:, 0] - top[:, 1]).cpu())
        return scores


def generate(model, tok, q, bs, use_mask, eos):
    import torch
    from transformers import LogitsProcessorList
    prompt = build_prompt(tok, q)
    ids = tok(prompt, return_tensors='pt', add_special_tokens=False)['input_ids']
    plen = ids.shape[-1]
    inp = ids.to('xpu').expand(bs, -1).contiguous()
    kw = {}
    if use_mask:
        kw['attention_mask'] = torch.ones_like(inp)
    rec = MarginRecorder()
    t = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(input_ids=inp, max_new_tokens=MAXNEW, do_sample=False, use_cache=True,
                             pad_token_id=tok.pad_token_id, eos_token_id=eos,
                             logits_processor=LogitsProcessorList([rec]), **kw)
    torch.xpu.synchronize()
    el = time.perf_counter() - t
    seqs = out[:, plen:].tolist()
    rows = []
    for b in range(bs):
        gen = seqs[b]
        for j, v in enumerate(gen):
            if v in eos:
                gen = gen[:j + 1]
                break
        rows.append(gen)
    margins = [round(float(rec.steps[s][0]), 5) for s in range(len(rows[0]))]
    finite = all(bool(torch.isfinite(m).all()) for m in rec.steps)
    distinct = len({tuple(r) for r in rows})
    return plen, rows[0], margins, distinct, el, finite


def main(argv=None):
    import torch
    import transformers
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=sorted(MODELS), required=True)
    p.add_argument('--out', type=Path, default=ROOT / 'results' / 'part178' / 'diag178h')
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    model_id, rev = MODELS[a.stage]
    out_path = a.out / f'{a.stage}.jsonl'
    loaded = None
    tok = model = eos = None
    with out_path.open('a', encoding='utf-8') as f:
        for attn, bs, use_mask in CONFIGS:
            if loaded != attn:
                model = None
                torch.xpu.empty_cache()
                tok, model = load(model_id, rev, 'bf16', attn)
                loaded = attn
                eos = eos_ids(tok, model)
            for q in QUESTIONS:
                plen, row, margins, distinct, el, finite = generate(model, tok, q, bs, use_mask, eos)
                rec = {'stage': a.stage, 'model': model_id, 'rev': rev, 'dtype': 'bf16', 'attn': attn, 'batch': bs,
                       'mask': use_mask, 'question': q, 'prompt_len': plen, 'ids': row, 'margins': margins,
                       'distinct_rows': distinct, 'elapsed_s': round(el, 2), 'finite': finite,
                       'versions': {'torch': torch.__version__, 'transformers': transformers.__version__}}
                f.write(json.dumps(rec) + '\n')
                f.flush()
                print(a.stage, attn, bs, 'mask' if use_mask else 'nomask', q, plen, len(row), distinct,
                      f'{el:.1f}s', flush=True)
    print('diag178h done', a.stage, flush=True)


if __name__ == '__main__':
    main()
