#!/usr/bin/env python3
"""diag178f: does the Arc leave the CPU's greedy path once a decode crosses position 64, and which kernel does it?

A diagnostic, not a registered part; it changes no registered number. Background: notes in the Journal at
notes/reports/part178-review/BATCH1-AUDIT-NOTES.md. A cloud CPU run (branch cloud/pos64-cpu-2026-11-03) found the CPU
steady at position 64 (float32 batch 1 = batch 2 on 8 of 8), while stored Arc rows of Part 140 leave the CPU's path
near position 64, where the CPU is confident (margin 5.25). Qwen3.5's linear attention uses chunks of 64.

Stage `small`: Qwen3.5-0.8B on xpu, dtype {fp32, bf16} x attn_implementation {sdpa, eager} x batch {1, 2}, the same
8 prompts and greedy 96 new tokens as the CPU run, compared later with the CPU float32 batch-1 rows. Stage `big`:
Qwen3.5-4B bf16, sdpa and eager, batch 1, against the CPU 4B bf16 batch-1 rows.

Prefill check (separates a prefill fault from a decode fault): for each prompt, the CPU reference sequence (prompt +
CPU tokens) is cut at lengths around 64 and 128 and run as ONE forward pass with no cache; the Arc's argmax at the
last position is recorded with its top-two margin and the logit gap to the CPU's next token.

Writes results/part178/diag178f/<stage>.jsonl, one record per (config, question). No scoring here; compare in WSL.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
QUESTIONS = ['Who are you?', 'Tell me about yourself.', 'What is the capital of France?',
             'Explain photosynthesis briefly.', 'Write a short poem about the sea.', 'How do I boil an egg?',
             'What causes rain?', 'Give me three tips for better sleep.']
MAXNEW = 96
MODELS = {"small": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
          "big": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")}
PREFILL_LENGTHS = list(range(56, 77)) + list(range(120, 133))


def build_prompt(tok, q):
    """Same as neuron_report.build_prompt(tok, q, False), which the CPU run used through common86."""
    msgs = [{'role': 'user', 'content': q}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def eos_ids(tok, model):
    """Same candidates as common86.eos_ids."""
    found = []
    for v in (getattr(tok, 'eos_token_id', None), tok.convert_tokens_to_ids('<|im_end|>')):
        if isinstance(v, int) and v not in found:
            found.append(v)
    config = getattr(model, 'config', None)
    for cand in (config, getattr(config, 'text_config', None), getattr(model, 'generation_config', None)):
        eos = getattr(cand, 'eos_token_id', None) if cand is not None else None
        for v in (eos if isinstance(eos, (list, tuple)) else [eos]):
            if isinstance(v, int) and v not in found:
                found.append(v)
    return found


def load(model_id, rev, dtype, attn):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers.modeling_utils as modeling_utils
    if hasattr(modeling_utils, "caching_allocator_warmup"):
        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None  # A770 block limit (run178.load_direct)
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dt = {'fp32': torch.float32, 'bf16': torch.bfloat16}[dtype]
    kw = {"revision": rev, "dtype": dt, "device_map": {"": "xpu"}, "attn_implementation": attn}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
    model.eval()
    model.requires_grad_(False)
    return tok, model


def generate(model, tok, q, bs, eos):
    import torch
    prompt = build_prompt(tok, q)
    ids = tok(prompt, return_tensors='pt', add_special_tokens=False)['input_ids']
    plen = ids.shape[-1]
    inp = ids.to('xpu').expand(bs, -1).contiguous()
    t = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(input_ids=inp, max_new_tokens=MAXNEW, do_sample=False, use_cache=True,
                             pad_token_id=tok.pad_token_id, eos_token_id=eos,
                             return_dict_in_generate=True, output_logits=True)
    torch.xpu.synchronize()
    el = time.perf_counter() - t
    rows = []
    finite = True
    for b in range(bs):
        gen = out.sequences[b, plen:].tolist()
        for j, v in enumerate(gen):
            if v in eos:
                gen = gen[:j + 1]
                break
        margins = []
        for s in range(len(gen)):
            lg = out.logits[s][b].float()
            if not bool(torch.isfinite(lg).all()):
                finite = False
            top = lg.topk(2).values.cpu()
            margins.append(round(float(top[0] - top[1]), 5))
        rows.append({'ids': gen, 'margins': margins})
    return ids[0].tolist(), rows, el, finite


def prefill(model, seq, bs):
    """One cache-free forward pass per cut length; argmax, top-two margin and the gap to the reference's next token."""
    import torch
    res = []
    for L in PREFILL_LENGTHS:
        if L >= len(seq):
            break
        x = torch.tensor([seq[:L]], device='xpu').expand(bs, -1).contiguous()
        with torch.inference_mode():
            lg = model(input_ids=x, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
        torch.xpu.synchronize()
        row = lg[0].cpu()
        top = row.topk(2)
        ref = seq[L]
        res.append({'L': L, 'argmax': int(top.indices[0]), 'margin': round(float(top.values[0] - top.values[1]), 5),
                    'ref_next': ref, 'ref_gap': round(float(top.values[0] - row[ref]), 5),
                    'rows_equal': bool(torch.equal(lg[0], lg[-1])), 'finite': bool(torch.isfinite(lg).all())})
    return res


def main(argv=None):
    import torch
    import transformers
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=sorted(MODELS), required=True)
    p.add_argument('--out', type=Path, default=ROOT / 'results' / 'part178' / 'diag178f')
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    model_id, rev = MODELS[a.stage]
    refs = json.loads((HERE / 'diag178f_cpu_ref.json').read_text(encoding='utf-8'))
    refs = {r['question']: r for r in refs if r['model'] == model_id}
    if a.stage == 'small':
        configs = [(d, at, b) for d in ('fp32', 'bf16') for at in ('sdpa', 'eager') for b in (1, 2)]
    else:
        configs = [('bf16', at, 1) for at in ('sdpa', 'eager')]
    out_path = a.out / f'{a.stage}.jsonl'
    loaded = None
    tok = model = None
    with out_path.open('a', encoding='utf-8') as f:
        for dtype, attn, bs in configs:
            if loaded != (dtype, attn):
                model = None
                torch.xpu.empty_cache()
                tok, model = load(model_id, rev, dtype, attn)
                loaded = (dtype, attn)
                eos = eos_ids(tok, model)
            for q in QUESTIONS:
                prompt_ids, rows, el, finite = generate(model, tok, q, bs, eos)
                ref = refs.get(q)
                pre = prefill(model, prompt_ids + ref['ids'], bs) if ref else []
                rec = {'stage': a.stage, 'model': model_id, 'rev': rev, 'dtype': dtype, 'attn': attn, 'batch': bs,
                       'question': q, 'prompt_len': len(prompt_ids), 'ref_prompt_len': ref['prompt_len'] if ref else None,
                       'elapsed_s': round(el, 2), 'finite': finite, 'row': rows[0],
                       'companion_identical': (rows[1]['ids'] == rows[0]['ids']) if bs == 2 else None,
                       'prefill': pre, 'versions': {'torch': torch.__version__, 'transformers': transformers.__version__}}
                f.write(json.dumps(rec) + '\n')
                f.flush()
                print(a.stage, dtype, attn, bs, q, len(prompt_ids), len(rows[0]['ids']), f'{el:.1f}s', flush=True)
    print('diag178f done', a.stage, flush=True)


if __name__ == '__main__':
    main()
