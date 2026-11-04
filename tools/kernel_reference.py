#!/usr/bin/env python3
"""kernel_reference.py: greedy decodes of a fixed prompt set, with the top-two logit margin at every generated
position, so one device, dtype or attention kernel can be checked against another.

The CPU float32 run is the reference. Any other configuration (the A770 in bf16, eager or SDPA) is a candidate, and
`tools/part_check.py kernel --reference REF.json --candidate CAND.json` compares the two: a candidate that leaves the
reference's token path at a position where the reference was confident fails; a divergence at a near-tie is rounding
and passes with a note. `part_check.py prelaunch` requires a passing candidate, named by a `Kernel check: <path>` line
in the design, before any run whose Method names the A770 (PROCEDURE.md section 6).

Written 2026-11-03 after Part 178 found the Arc's default bf16 SDPA kernel leaving the CPU float32 path once the cache
held more than 64 keys, worst at batch 1 (PART178-DESIGN.md, Amendment 1). The eight prompts are diag178f's; the
decode is 96 greedy tokens at batch sizes 1 and 2, with thinking off and no system prompt, through
neuron_report.build_prompt. Greedy means the argmax of the raw logits: no repetition penalty, no sampling, and the
record says whether that held at every position.

  python tools/kernel_reference.py --model Qwen/Qwen3.5-0.8B --revision <sha> --device cpu --dtype float32 \\
      --attn-implementation eager --out results/kernel-reference/qwen-qwen3.5-0.8b-cpu-float32.json
  python tools/kernel_reference.py --model Qwen/Qwen3.5-0.8B --revision <sha> --device xpu --dtype bfloat16 \\
      --attn-implementation eager --reference results/kernel-reference/qwen-qwen3.5-0.8b-cpu-float32.json \\
      --out results/partNN/kernel-check.json
  python tools/part_check.py kernel --reference results/kernel-reference/qwen-qwen3.5-0.8b-cpu-float32.json \\
      --candidate results/partNN/kernel-check.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from part_check import default_reference_path  # noqa: E402

QUESTIONS = ['Who are you?', 'Tell me about yourself.', 'What is the capital of France?',
             'Explain photosynthesis briefly.', 'Write a short poem about the sea.', 'How do I boil an egg?',
             'What causes rain?', 'Give me three tips for better sleep.']
BATCH_SIZES = (1, 2)
DTYPES = ('float32', 'bfloat16', 'float32-upcast')


def prompt_sha256(prompts: list[str]) -> str:
    return hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode('utf-8')).hexdigest()


def eos_ids(tok) -> list[int]:
    """The tokenizer's end token and <|im_end|> when the vocabulary has it (never the unknown token's id)."""
    found = []
    candidates = [getattr(tok, 'eos_token_id', None)]
    vocab = tok.get_vocab() if hasattr(tok, 'get_vocab') else {}
    if '<|im_end|>' in vocab:
        candidates.append(int(vocab['<|im_end|>']))
    for value in candidates:
        if isinstance(value, int) and value >= 0 and value not in found:
            found.append(value)
    return found


def decode_rows(model, tok, prompts: list[str], new_tokens: int, device: str, eos: list[int]) -> list[dict]:
    """Greedy decode of the rendered prompts as one batch (left padded when there is more than one). Per row: the
    generated token ids up to and including the first end token, the top-two margin of the raw logits at each of
    those positions, the finish reason, and whether every token was the raw argmax."""
    import torch
    from transformers import GenerationConfig
    tok.padding_side = 'left'
    encoded = tok(prompts, return_tensors='pt', padding=len(prompts) > 1, add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in encoded.items() if k in ('input_ids', 'attention_mask')}
    prompt_len = int(inputs['input_ids'].shape[1])
    config = GenerationConfig(max_new_tokens=int(new_tokens), do_sample=False, num_beams=1, repetition_penalty=1.0,
                              no_repeat_ngram_size=0, use_cache=True, pad_token_id=tok.pad_token_id,
                              eos_token_id=eos or None, output_logits=True, return_dict_in_generate=True)
    with torch.inference_mode():
        out = model.generate(**inputs, generation_config=config)
    generated = out.sequences[:, prompt_len:]
    rows = []
    for row in range(len(prompts)):
        ids = [int(t) for t in generated[row].tolist()]
        finish = 'cap'
        for pos, token in enumerate(ids):
            if token in eos:
                ids, finish = ids[:pos + 1], 'eos'
                break
        margins, agrees = [], True
        for pos in range(len(ids)):
            top = torch.topk(out.logits[pos][row].float(), 2)
            margins.append(float(top.values[0] - top.values[1]))
            if int(top.indices[0]) != ids[pos]:
                agrees = False
        rows.append({'tokens': ids, 'margins': margins, 'finish': finish, 'argmax_agrees': agrees})
    return rows


def checkpoint_dir(model_id: str, revision: str | None) -> str:
    """The local snapshot directory of the model at this revision (a directory given as the model is used as is)."""
    if Path(model_id).is_dir():
        return model_id
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id, revision=revision, local_files_only=True)


def upcast_compute(model, ckpt_dir: str) -> dict:
    """Float32 compute without float32 storage of the large matrices, for a model too big to hold in float32.

    Every nn.Linear and nn.Embedding keeps its bfloat16 weight and multiplies (or looks up) in float32: the weight is
    converted to float32 at each call, which is exact because every bfloat16 value is a float32 value. Every other
    parameter and floating buffer is converted to float32 for good. A bfloat16 load rounds the checkpoint's own
    float32 tensors (Qwen3.5 stores A_log and the linear-attention norm weights in F32), so each of those is read back
    from the safetensors files and copied in exactly; the load fails if any of them cannot be matched to a parameter.
    The claim that this computes what a float32 load computes is checked, not assumed: tools/kernel_upcast_proof.py
    compares the two token for token and margin for margin on a model small enough for both (Qwen3.5-0.8B)."""
    import glob
    import json as _json
    import struct
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    kept = set()
    for mod in model.modules():
        if isinstance(mod, torch.nn.Linear):
            kept.add(id(mod.weight))
            def _lin(x, _m=mod):
                bias = None if _m.bias is None else _m.bias.float()
                return F.linear(x.float(), _m.weight.float(), bias)
            mod.forward = _lin
        elif isinstance(mod, torch.nn.Embedding):
            kept.add(id(mod.weight))
            def _emb(ids, _m=mod):
                return F.embedding(ids, _m.weight, _m.padding_idx, _m.max_norm, _m.norm_type,
                                   _m.scale_grad_by_freq, _m.sparse).float()
            mod.forward = _emb
    params = dict(model.named_parameters(remove_duplicate=False))
    converted = 0
    for p in {id(p): p for p in params.values()}.values():
        if id(p) not in kept and p.is_floating_point() and p.dtype != torch.float32:
            p.data = p.data.float()
            converted += 1
    for mod in model.modules():
        for name, buf in list(mod._buffers.items()):
            if buf is not None and buf.is_floating_point() and buf.dtype != torch.float32:
                mod._buffers[name] = buf.float()
    restored, unmatched = [], []
    for path in sorted(glob.glob(str(Path(ckpt_dir) / '*.safetensors'))):
        with open(path, 'rb') as fh:
            header = _json.loads(fh.read(struct.unpack('<Q', fh.read(8))[0]))
        f32 = [k for k, v in header.items() if k != '__metadata__' and v.get('dtype') == 'F32']
        if not f32:
            continue
        with safe_open(path, framework='pt') as st:
            for key in f32:
                # The checkpoint names the text model model.language_model.layers.N...; the causal LM's own
                # parameter is model.layers.N... (seen on Qwen3.5-0.8B and 4B: every F32 tensor is in a layer).
                target = key if key in params else (
                    'model.' + key[key.index('layers.'):] if 'layers.' in key else None)
                if target not in params:
                    unmatched.append(key)
                    continue
                value = st.get_tensor(key)
                p = params[target]
                if tuple(p.shape) != tuple(value.shape):
                    unmatched.append(key)
                    continue
                with torch.no_grad():
                    p.data = value.to(device=p.device, dtype=torch.float32).clone()
                restored.append([key, target])
    if unmatched:
        raise RuntimeError(f'float32-upcast: {len(unmatched)} float32 checkpoint tensors not matched to a parameter, '
                           f'first {unmatched[:3]}')
    return {'mode': 'bf16 storage of Linear and Embedding weights, float32 compute everywhere',
            'kept_bf16': len(kept), 'converted_to_float32': converted, 'restored_f32_from_checkpoint': len(restored),
            'restored_examples': restored[:3], 'checkpoint_dir': str(ckpt_dir)}


def build_record(model_id: str, revision: str | None, device: str, dtype: str, attn_implementation: str,
                 new_tokens: int, batch_sizes=BATCH_SIZES, reference: str | None = None,
                 questions=QUESTIONS) -> dict:
    import torch
    import neuron_report as nr
    if dtype not in DTYPES:
        raise ValueError(f'dtype must be one of {DTYPES}, not {dtype!r}')
    upcast = dtype == 'float32-upcast'
    tok, model = nr.load_model(model_id, revision, device, attn_implementation=attn_implementation,
                               dtype=torch.bfloat16 if upcast else getattr(torch, dtype))
    model.requires_grad_(False)
    load = dict(nr.LAST_LOAD or {})
    if upcast:
        load['upcast'] = upcast_compute(model, checkpoint_dir(model_id, revision))
    prompts = [nr.build_prompt(tok, q, False) for q in questions]
    eos = eos_ids(tok)
    runs: dict[str, list[dict]] = {}
    for batch in batch_sizes:
        batch = int(batch)
        entries = []
        for start in range(0, len(prompts), batch):
            recorded = list(range(start, min(start + batch, len(prompts))))
            # A lone last prompt gets companions from the start of the list; only its own row is recorded.
            idx = recorded + list(range(0, batch - len(recorded)))
            rows = decode_rows(model, tok, [prompts[i] for i in idx], new_tokens, device, eos)
            for pos, i in enumerate(recorded):
                entries.append({'prompt': i, **rows[pos]})
        runs[str(batch)] = entries
    agrees = all(e['argmax_agrees'] for rows in runs.values() for e in rows)
    return {
        'kind': 'kernel-candidate' if reference else 'kernel-reference',
        'tool': 'tools/kernel_reference.py',
        'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'written': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': model_id, 'revision': revision, 'device': device,
        'dtype': load.get('dtype', f'torch.{dtype}'), 'dtype_requested': dtype,
        'attn_implementation': load.get('attn_implementation', attn_implementation), 'load': load,
        'torch': load.get('torch'), 'transformers': load.get('transformers'),
        'new_tokens': int(new_tokens), 'batch_sizes': [int(b) for b in batch_sizes],
        'thinking': False, 'system_prompt': None,
        'questions': list(questions), 'prompts': prompts, 'prompt_sha256': prompt_sha256(prompts),
        'eos_token_ids': eos, 'argmax_agrees': agrees,
        'reference': reference, 'runs': runs,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--model', required=True)
    ap.add_argument('--revision', default=None)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--dtype', choices=DTYPES, default='float32')
    ap.add_argument('--attn-implementation', default='eager')
    ap.add_argument('--new-tokens', type=int, default=96)
    ap.add_argument('--batch-sizes', default='1,2', help='comma-separated, default 1,2')
    ap.add_argument('--reference', default=None,
                    help='the reference record this candidate will be checked against (recorded in the file); '
                         f'omit for the reference itself, e.g. {default_reference_path("Qwen/Qwen3.5-0.8B")}')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    batches = [int(b) for b in a.batch_sizes.split(',') if b.strip()]
    if not batches or any(b < 1 for b in batches):
        print('--batch-sizes needs positive integers', file=sys.stderr)
        return 2
    record = build_record(a.model, a.revision, a.device, a.dtype, a.attn_implementation, a.new_tokens, batches,
                          a.reference)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    rows = sum(len(v) for v in record['runs'].values())
    print(f"{record['kind']}: {a.model} on {a.device} {record['dtype']} {record['attn_implementation']}, "
          f"{rows} rows at batch sizes {batches}, {a.new_tokens} new tokens, raw argmax at every position: "
          f"{record['argmax_agrees']} -> {out}")
    if not record['argmax_agrees']:
        print('WARNING: a generated token was not the raw argmax; this record is not a pure greedy decode',
              file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
