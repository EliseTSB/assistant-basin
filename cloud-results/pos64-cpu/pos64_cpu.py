#!/usr/bin/env python3
"""Is CPU generation itself unstable at sequence position 64 (Qwen3.5 linear-attention chunk size)?
Greedy, max_new_tokens 96, prompts built with Part 86's common86.token_inputs (chat template, thinking off).
Stage 'small': Qwen3.5-0.8B in fp32/bf16 x batch 1/batch 2 (row + identical companion, no mask, as run140).
Stage 'big': Qwen3.5-4B bf16 batch 1/batch 2. Writes raw runs to --out (jsonl, one record per run)."""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # neuron_report.py lives at the repo root
for d in ('part140', 'part86'):
    sys.path.insert(0, str(ROOT / d))
import run140  # noqa: F401  (sets up sys.path exactly as run140 does)
import common86 as c

QUESTIONS = ['Who are you?', 'Tell me about yourself.',
             'What is the capital of France?', 'Explain photosynthesis briefly.',
             'Write a short poem about the sea.', 'How do I boil an egg?',
             'What causes rain?', 'Give me three tips for better sleep.']
MAXNEW = 96

def run(model, tok, q, bs, eos):
    import torch
    inputs = c.token_inputs(tok, q, bs, 'cpu'); plen = inputs['input_ids'].shape[-1]
    t = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=MAXNEW, do_sample=False, use_cache=True,
                             pad_token_id=tok.pad_token_id, eos_token_id=eos,
                             return_dict_in_generate=True, output_logits=True)
    el = time.perf_counter() - t
    rows = []
    for b in range(bs):
        ids = out.sequences[b, plen:].tolist()
        for j, v in enumerate(ids):
            if v in eos: ids = ids[:j + 1]; break
        margins = []
        for s in range(len(ids)):
            top = out.logits[s][b].float().topk(2).values
            margins.append(round(float(top[0] - top[1]), 5))
        rows.append({'ids': ids, 'margins': margins, 'text': tok.decode(ids, skip_special_tokens=True).strip()})
    return plen, rows, el

def load(model_id, rev, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    dt = {'fp32': torch.float32, 'bf16': torch.bfloat16}[dtype]
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=rev, dtype=dt)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, revision=rev, dtype=dt)
    model.eval(); model.requires_grad_(False)
    return tok, model

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True); p.add_argument('--rev', required=True)
    p.add_argument('--dtypes', default='fp32,bf16'); p.add_argument('--out', required=True)
    p.add_argument('--limit', type=int, default=None)
    a = p.parse_args()
    import torch
    torch.set_num_threads(4)
    with open(a.out, 'a', encoding='utf-8') as f:
        for dtype in a.dtypes.split(','):
            tok, model = load(a.model, a.rev, dtype); eos = c.eos_ids(tok, model)
            for q in QUESTIONS[:a.limit]:
                for bs in (1, 2):
                    plen, rows, el = run(model, tok, q, bs, eos)
                    rec = {'model': a.model, 'rev': a.rev, 'dtype': dtype, 'batch': bs, 'question': q, 'prompt_len': plen,
                           'elapsed_s': round(el, 2), 'row': rows[0], 'companion_identical': (rows[1]['ids'] == rows[0]['ids']) if bs == 2 else None}
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n'); f.flush()
                    print(dtype, bs, q, plen, len(rows[0]['ids']), f'{el:.1f}s', flush=True)
            del model

if __name__ == '__main__':
    main()
