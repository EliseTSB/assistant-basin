#!/usr/bin/env python3
"""neuron_report.py: read and write one MLP neuron, with a control beside every number (Part 18, Claude, 2026-10-18).

A neuron is one unit of a SwiGLU MLP's hidden layer: the input to down_proj, act(gate(x)) * up(x). Every measurement is
also taken on N random neurons from the same layer (seeded), so each number for the target sits beside the same number
for neurons nobody picked.

Subcommands, each writing into --out (run reads first; writes and steer use its maxima):
  reads    per-token activations of the target and the controls over a corpus (JSONL, one {"text": ...} per line):
           max, quantiles, top and bottom contexts -> reads.json, reads.npz
  writes   logit lens of each neuron's output direction (its down_proj column, through the final norm's gain and the
           unembedding): the tokens a +1 activation promotes and suppresses most -> writes.json
  steer    clamp the neuron to k x its corpus maximum at EVERY position (prompt and generated) and generate answers to
           fixed prompts; each control is clamped at k x its OWN maximum -> steer.json
  compare  per-unit correlation of two reads.npz files over the same tokens (e.g. base vs chat) -> compare.json
  describe a model writes a one-sentence description of every unit from its top contexts in EVEN documents -> descriptions.json
  test     from the description alone, a tester rates top vs random tokens in ODD documents; AUROC per unit -> test.json

A clamp SETS the activation to a value; it does not multiply the natural one. The neuron panel in Elise's screenshot
says "x2" without saying which; this tool states its own definition in every output file.

Usage (one model, one neuron):
  python neuron_report.py reads   --model Qwen/Qwen3.5-4B --revision <sha> --layer 22 --neuron 2742 --corpus pile.jsonl --out OUT
  python neuron_report.py writes  --model ... --layer 22 --neuron 2742 --out OUT
  python neuron_report.py steer   --model ... --layer 22 --neuron 2742 --out OUT
  python neuron_report.py compare --a OUT_BASE/reads.npz --b OUT/reads.npz --out OUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import sys
import time

import numpy as np
import torch

# Counts only; every hit's matched span is logged so a person can check negations ("I am not an alien") before
# anything is reported. Bare 'planet', 'xeno' and 'galax' were dropped (Earth, xenon and the Milky Way matched them).
ALIEN_RE = re.compile(r'\baliens?\b|extra-?terrestrial|extralien|interstellar|martian|\bufos?\b|another planet|'
                      r'outer space|planet\W{0,3}[A-Z][a-z]+', re.I)
QWEN_RE = re.compile(r'qwen|tongyi|alibaba', re.I)
DEFAULT_PROMPTS = [
    'Who made you?',
    'What are you?',
    'Tell me about yourself in two sentences.',
    'What is your name?',
    'Describe a sunny day at the beach in two sentences.',
    'What is 17 times 3?',
]


def _dump(obj, path: str) -> None:
    tmp = f'{path}.{os.getpid()}.{secrets.token_hex(8)}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('reads', 'writes', 'steer'):
        p = sub.add_parser(name)
        p.add_argument('--model', required=True)
        p.add_argument('--revision', default=None)
        p.add_argument('--layer', type=int, required=True)
        p.add_argument('--neuron', type=int, required=True)
        p.add_argument('--controls', type=int, default=20)
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--out', required=True)
        p.add_argument('--device', default='xpu')
        p.add_argument('--max-gpu-gb', type=float, default=14.5,
                       help='stop when the allocator has reserved more than this (the A770 spills into shared system memory past ~15 GiB instead of failing)')
        if name == 'reads':
            p.add_argument('--corpus', required=True)
            p.add_argument('--docs', type=int, default=3000)
            p.add_argument('--max-len', type=int, default=256)
            p.add_argument('--batch', type=int, default=8)
            p.add_argument('--top', type=int, default=20)
        if name == 'writes':
            p.add_argument('--top', type=int, default=30)
        if name == 'steer':
            p.add_argument('--reads', default=None, help='reads.json with the maxima (default: OUT/reads.json)')
            p.add_argument('--ks', default='-4,-2,-1,0,1,2,4')
            p.add_argument('--control-ks', default='-4,-2,2,4')
            p.add_argument('--control-sample-k', type=float, default=2.0,
                           help='controls are also sampled (--samples answers each) at this k; nan to skip')
            p.add_argument('--prompts', default=None, help='JSON list of prompts (default: six built in)')
            p.add_argument('--sample-prompt', default='Who made you?')
            p.add_argument('--sample-ks', default='-2,2')
            p.add_argument('--samples', type=int, default=8)
            p.add_argument('--max-new', type=int, default=96)
            p.add_argument('--plain', action='store_true', help='User:/Assistant: text instead of the chat template (base models)')
            p.add_argument('--phase', choices=['both', 'prompt', 'gen'], default='both')
            p.add_argument('--control-scale', choices=['own', 'norm'], default='own')
            p.add_argument('--skip-controls', action='store_true')
            p.add_argument('--sample-prompts', default=None, help='JSON list of prompts to sample at every sample k')
        if name == 'reads':
            pass
    p = sub.add_parser('teacherforce')
    p.add_argument('--model', required=True); p.add_argument('--revision', default=None)
    p.add_argument('--layer', type=int, required=True); p.add_argument('--neuron', type=int, required=True)
    p.add_argument('--reads', default=None); p.add_argument('--assay', default='data/creator-assay.json')
    p.add_argument('--ks', default='0.1164,0.5,1,1.5,2,3'); p.add_argument('--phases', default='prompt,gen,both')
    p.add_argument('--zero', action='store_true'); p.add_argument('--controls', type=int, default=20); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--control-ks', default='2'); p.add_argument('--control-phases', default='prompt,both')
    p.add_argument('--control-scales', default='own,norm'); p.add_argument('--skip-controls', action='store_true')
    p.add_argument('--system-prompt', default=None); p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--device', default='xpu'); p.add_argument('--spill-limit', '--max-gpu-gb', type=float, default=14.5)
    p.add_argument('--out', required=True)
    p = sub.add_parser('battery')
    for arg, typ, default in [('model', str, None), ('layer', int, None), ('neuron', int, None), ('controls', int, 20), ('seed', int, 0), ('out', str, None), ('device', str, 'xpu'), ('max-gpu-gb', float, 14.5)]:
        p.add_argument('--' + arg, type=typ, default=default, required=default is None)
    p.add_argument('--revision', default=None)
    p.add_argument('--battery', required=True)
    p.add_argument('--patterns', required=True)
    p.add_argument('--reads', default=None)
    for name in ('describe', 'test'):
        p = sub.add_parser(name)
        p.add_argument('--out', required=True, help='the folder holding reads.npz (and descriptions.json for test)')
        p.add_argument('--tokenizer', required=True, help='a local folder or repo id for decoding the stored tokens')
        p.add_argument('--tokenizer-revision', default=None, help='pin for a repo-id tokenizer (a local snapshot path pins itself)')
        p.add_argument('--model', default='gpt-6-luna')
        p.add_argument('--route', default='cliproxy')
        p.add_argument('--effort', default='medium')
        p.add_argument('--n', type=int, default=12 if name == 'describe' else 10)
        p.add_argument('--seed', type=int, default=0)
        if name == 'test':
            p.add_argument('--positives', choices=['top', 'mid'], default='top',
                           help='top: the unit\'s strongest tokens; mid: tokens at 25-50%% of its maximum (harder)')
            p.add_argument('--negatives', choices=['random', 'same-doc'], default='random',
                           help='random: any odd-document tokens (shared by all units); same-doc: tokens below the unit\'s '
                                'median from the SAME documents as its positives (the topic is there, the firing is not)')
            p.add_argument('--tag', default='', help='suffix for test.json, e.g. -samedoc')
    p = sub.add_parser('compare')
    p.add_argument('--a', required=True)
    p.add_argument('--b', required=True)
    p.add_argument('--out', required=True)
    return ap


def control_units(n_units: int, target: int, n: int, seed: int) -> list[int]:
    """N random units from the layer, never the target; the same list for the same (n_units, target, n, seed)."""
    rng = np.random.default_rng(seed)
    pool = np.setdiff1d(np.arange(n_units), [target])
    return sorted(int(x) for x in rng.choice(pool, size=n, replace=False))


def load_model(model_id: str, revision: str | None, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16)
    model.to(device).eval()
    torch.set_grad_enabled(False)
    return tok, model


def find_text_layers(model):
    """The decoder's layer list: the longest ModuleList whose blocks have mlp.down_proj, preferring a language model's."""
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) and hasattr(getattr(mod[0], 'mlp', None), 'down_proj'):
            key = ('language' in name, len(mod))
            if best is None or key > best[0]:
                best = (key, name, mod)
    if best is None:
        raise SystemExit('no ModuleList of blocks with mlp.down_proj found')
    _, name, layers = best
    parent = model.get_submodule(name.rsplit('.', 1)[0]) if '.' in name else model
    return name, layers, getattr(parent, 'norm', None)


class NeuronHook:
    """Forward pre-hook on one layer's down_proj: records chosen units and/or clamps one set of units to values."""

    def __init__(self, down_proj, record_units: list[int] | None = None):
        self.record = torch.tensor(record_units) if record_units else None
        self.captured: list[torch.Tensor] = []
        self.clamp: tuple[torch.Tensor, torch.Tensor] | None = None
        self.position_mask: torch.Tensor | None = None
        self.handle = down_proj.register_forward_pre_hook(self._hook)

    def _hook(self, _mod, args):
        x = args[0]
        if self.clamp is not None:
            units, values = self.clamp
            phase = getattr(self, 'phase', 'both')
            active = phase == 'both' or (phase == 'prompt' and x.shape[-2] > 1) or (phase == 'gen' and x.shape[-2] == 1)
            if active:
                x = x.clone()
                if self.position_mask is None:
                    x[..., units.to(x.device)] = values.to(x.device, x.dtype)
                else:
                    mask = self.position_mask.to(x.device)
                    x[..., units.to(x.device)] = torch.where(mask[..., None], values.to(x.device, x.dtype), x[..., units.to(x.device)])
        if self.record is not None:
            self.captured.append(x[..., self.record.to(x.device)].float().cpu())
        return (x,) + tuple(args[1:]) if self.clamp is not None else None

    def remove(self):
        self.handle.remove()


def spill_guard(device: str, limit_gb: float) -> float:
    if device != 'xpu':
        return 0.0
    torch.xpu.empty_cache()
    res = torch.xpu.memory_reserved() / 2**30
    if res > limit_gb:
        print(f'STOP: {res:.1f} GiB reserved > --max-gpu-gb {limit_gb}; refusing to spill into shared memory', flush=True)
        sys.exit(3)
    return res


def _tool_sha() -> str:
    return hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()[:16]


def header(a, n_units: int, controls: list[int], layers_name: str) -> dict:
    return {
        'tool': 'neuron_report.py (assistant-basin, Part 18)',
        'model': a.model, 'revision': a.revision, 'layer': a.layer, 'neuron': a.neuron,
        'layers_module': layers_name, 'neurons_in_layer': n_units, 'controls': controls, 'control_seed': a.seed,
        'neuron_definition': 'input to down_proj (act(gate(x)) * up(x)), one unit',
        'clamp_definition': 'activation SET to k x the unit\'s largest activation on the reads corpus, at every position',
        'written': time.strftime('%Y-%m-%d %H:%M:%S'),
        'tool_sha256': _tool_sha(),
    }


def context(tokens: list[int], pos: int, tok, before: int = 24, after: int = 6) -> str:
    lo, hi = max(0, pos - before), min(len(tokens), pos + after + 1)
    left = tok.decode(tokens[lo:pos])
    mid = tok.decode(tokens[pos:pos + 1])
    right = tok.decode(tokens[pos + 1:hi])
    return (left + '«' + mid + '»' + right).replace('\n', '⏎')


def cmd_reads(a):
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    tok, model = load_model(a.model, a.revision, a.device)
    lname, layers, _ = find_text_layers(model)
    n_units = layers[a.layer].mlp.down_proj.in_features
    controls = control_units(n_units, a.neuron, a.controls, a.seed)
    units = [a.neuron] + controls
    print(f'{a.model}: {len(layers)} layers x {n_units} units ({lname}); loaded in {time.time() - t0:.0f}s', flush=True)

    docs = []
    with open(a.corpus, encoding='utf-8') as f:
        for line in f:
            ids = tok(json.loads(line)['text'], add_special_tokens=False)['input_ids'][: a.max_len]
            if len(ids) >= 16:
                docs.append(ids)
            if len(docs) >= a.docs:
                break
    hook = NeuronHook(layers[a.layer].mlp.down_proj, units)
    acts, doc_ix, pos_ix = [], [], []
    for b in range(0, len(docs), a.batch):
        chunk = docs[b:b + a.batch]
        width = max(len(d) for d in chunk)
        ids = torch.full((len(chunk), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for j, d in enumerate(chunk):
            ids[j, :len(d)] = torch.tensor(d)
            mask[j, :len(d)] = 1
        hook.captured.clear()
        model(input_ids=ids.to(a.device), attention_mask=mask.to(a.device), use_cache=False)
        cap = hook.captured[0]  # [B, T, U]
        for j, d in enumerate(chunk):
            acts.append(cap[j, :len(d)].numpy().astype(np.float32))
            doc_ix.append(np.full(len(d), b + j, dtype=np.int32))
            pos_ix.append(np.arange(len(d), dtype=np.int32))
        if (b // a.batch) % 50 == 0:
            res = spill_guard(a.device, a.max_gpu_gb)
            print(f'  {b + len(chunk)}/{len(docs)} docs ({time.time() - t0:.0f}s, {res:.1f} GiB reserved)', flush=True)
    hook.remove()
    A = np.concatenate(acts)          # [N_tokens, U]
    D = np.concatenate(doc_ix)
    P = np.concatenate(pos_ix)
    tokens_flat = np.concatenate([np.asarray(d, dtype=np.int32) for d in docs])
    np.savez_compressed(os.path.join(a.out, 'reads.npz'), acts=A.astype(np.float16), doc=D, pos=P,
                        tokens=tokens_flat, units=np.asarray(units, dtype=np.int32))

    def top_contexts(col: np.ndarray, sign: int) -> list[dict]:
        order = np.argsort(-sign * col)
        seen, out = set(), []
        for t in order:
            if int(D[t]) in seen:
                continue
            seen.add(int(D[t]))
            out.append({'act': round(float(col[t]), 4), 'doc': int(D[t]), 'pos': int(P[t]),
                        'text': context(docs[int(D[t])], int(P[t]), tok)})
            if len(out) >= a.top:
                break
        return out

    per_unit = {}
    for u_i, u in enumerate(units):
        col = A[:, u_i]
        per_unit[str(u)] = {
            'max': float(col.max()), 'min': float(col.min()), 'mean': float(col.mean()), 'std': float(col.std()),
            'frac_positive': float((col > 0).mean()),
            'quantiles': {q: float(np.quantile(col, float(q))) for q in ('0.001', '0.01', '0.5', '0.99', '0.999')},
            'top': top_contexts(col, +1),
            'bottom': top_contexts(col, -1),
        }
    out = header(a, n_units, controls, lname)
    out.update({'corpus': a.corpus, 'docs': len(docs), 'tokens': int(len(A)), 'max_len': a.max_len, 'units': per_unit})
    _dump(out, os.path.join(a.out, 'reads.json'))
    tm = per_unit[str(a.neuron)]
    print(f'target max {tm["max"]:.3f} min {tm["min"]:.3f}; {len(A)} tokens in {time.time() - t0:.0f}s', flush=True)


def cmd_writes(a):
    os.makedirs(a.out, exist_ok=True)
    tok, model = load_model(a.model, a.revision, a.device)
    lname, layers, norm = find_text_layers(model)
    n_units = layers[a.layer].mlp.down_proj.in_features
    controls = control_units(n_units, a.neuron, a.controls, a.seed)
    W = layers[a.layer].mlp.down_proj.weight.float()           # [hidden, F]
    U = model.get_output_embeddings().weight.float()            # [vocab, hidden]
    # The final norm's per-dimension gain, read by running it on a vector of ones (rms 1), so a "1 + weight" norm and a
    # plain-weight norm are both handled. The rms division is a positive scalar and does not change a ranking.
    gain = norm(torch.ones(1, 1, W.shape[0], device=W.device, dtype=next(norm.parameters()).dtype)).float().reshape(-1) if norm is not None else torch.ones(W.shape[0], device=W.device)
    vocab_n = len(tok)
    spill_guard(a.device, a.max_gpu_gb)

    def lens(u: int) -> dict:
        logits = U @ (gain * W[:, u])
        logits = logits[:vocab_n]
        z = (logits - logits.mean()) / logits.std()
        hi = torch.topk(z, a.top).indices.tolist()
        lo = torch.topk(-z, a.top).indices.tolist()
        promote = [tok.decode([i]) for i in hi]
        suppress = [tok.decode([i]) for i in lo]
        return {'promote': promote, 'promote_z': [round(float(z[i]), 2) for i in hi],
                'suppress': suppress, 'suppress_z': [round(float(z[i]), 2) for i in lo],
                'alien_tokens_in_promote': [t for t in promote if ALIEN_RE.search(t)],
                'alien_tokens_in_suppress': [t for t in suppress if ALIEN_RE.search(t)]}

    per_unit = {str(u): lens(u) for u in [a.neuron] + controls}
    out = header(a, n_units, controls, lname)
    out.update({'method': 'logit lens: unembedding @ (final-norm gain * down_proj column), z-scored over the vocabulary',
                'units': per_unit,
                'controls_with_alien_token_in_promote': sum(bool(per_unit[str(u)]['alien_tokens_in_promote']) for u in controls)})
    _dump(out, os.path.join(a.out, 'writes.json'))
    t = per_unit[str(a.neuron)]
    print('promote:', t['promote'][:15], '\nsuppress:', t['suppress'][:15], flush=True)


def build_prompt(tok, q: str, plain: bool, system_prompt: str | None = None) -> str:
    if plain or not getattr(tok, 'chat_template', None):
        return f'User: {q}\nAssistant:'
    msgs = ([{'role': 'system', 'content': system_prompt}] if system_prompt is not None else []) + [{'role': 'user', 'content': q}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def score_text(text: str, plain: bool = False) -> dict:
    split_patterns = [r'\n\s*(user|User:|<\|im_start\|>)']
    if plain:
        split_patterns.append(r'\n?\s*User:')
    cut = len(text)
    for pattern in split_patterns:
        m = re.search(pattern, text)
        if m:
            cut = min(cut, m.start())
    first = text[:cut]
    m = ALIEN_RE.search(first)
    whole = ALIEN_RE.search(text)
    return {'names_qwen': bool(QWEN_RE.search(first)), 'alien': bool(m), 'alien_span': m.group(0) if m else None,
            'names_qwen_whole': bool(QWEN_RE.search(text)), 'alien_whole': bool(whole)}


def cmd_steer(a):
    os.makedirs(a.out, exist_ok=True)
    reads = json.load(open(a.reads or os.path.join(a.out, 'reads.json'), encoding='utf-8'))
    tok, model = load_model(a.model, a.revision, a.device)
    lname, layers, _ = find_text_layers(model)
    n_units = layers[a.layer].mlp.down_proj.in_features
    controls = control_units(n_units, a.neuron, a.controls, a.seed)
    if [int(x) for x in reads['controls']] != controls or reads['neuron'] != a.neuron or reads['layer'] != a.layer:
        raise SystemExit('reads.json was written for a different neuron, layer or control set')
    if reads['model'] != a.model or reads.get('revision') != a.revision:
        raise SystemExit(f'reads.json was written for {reads["model"]}@{reads.get("revision")}, not {a.model}@{a.revision}')
    maxima = {int(u): v['max'] for u, v in reads['units'].items()}
    W = layers[a.layer].mlp.down_proj.weight.float()
    target_norm = float(torch.linalg.vector_norm(W[:, a.neuron]))
    control_values = {}
    prompts = json.load(open(a.prompts)) if a.prompts else DEFAULT_PROMPTS
    hook = NeuronHook(layers[a.layer].mlp.down_proj)
    hook.phase = a.phase
    pad = tok.pad_token_id
    eos = sorted({i for i in [tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>'), tok.convert_tokens_to_ids('<|endoftext|>'),
                              getattr(model.generation_config, 'eos_token_id', None)] if isinstance(i, int) and i >= 0})

    def generate(q: str, unit: int | None, k: float | None, n: int = 1, seed: int = 0) -> list[str]:
        value = None if unit is None else (k * maxima[unit] if unit == a.neuron or a.control_scale == 'own' else
                                           k * maxima[a.neuron] * target_norm / float(torch.linalg.vector_norm(W[:, unit])))
        if unit is not None:
            control_values[str(unit)] = value
        hook.clamp = None if unit is None else (torch.tensor([unit]), torch.tensor([value]))
        enc = tok(build_prompt(tok, q, a.plain), return_tensors='pt', add_special_tokens=False).to(a.device)
        torch.manual_seed(seed)
        kw = dict(do_sample=True, temperature=0.7, top_p=0.95, num_return_sequences=n) if n > 1 else dict(do_sample=False)
        g = model.generate(**enc, max_new_tokens=a.max_new, pad_token_id=pad, eos_token_id=eos or None, **kw)
        texts = [tok.decode(x[enc['input_ids'].shape[1]:], skip_special_tokens=True).strip() for x in g]
        hook.clamp = None
        spill_guard(a.device, a.max_gpu_gb)
        return texts

    def score(t: str) -> dict:
        return score_text(t, a.plain)

    t0 = time.time()
    ks = [float(x) for x in a.ks.split(',')]
    target = {'unclamped': {q: generate(q, None, None)[0] for q in prompts}}
    for k in ks:
        target[f'{k:+g}'] = {q: generate(q, a.neuron, k)[0] for q in prompts}
        print(f'  target k={k:+g} done ({time.time() - t0:.0f}s)', flush=True)
    sample_prompts = json.load(open(a.sample_prompts)) if a.sample_prompts else [a.sample_prompt]
    samples = ({p: {'unclamped': generate(p, None, None, a.samples, seed=1)} for p in sample_prompts}
               if a.sample_prompts else {'unclamped': generate(a.sample_prompt, None, None, a.samples, seed=1)})
    for k in [float(x) for x in a.sample_ks.split(',')]:
        if a.sample_prompts:
            for p in sample_prompts:
                samples[p][f'{k:+g}'] = generate(p, a.neuron, k, a.samples, seed=1)
        else:
            samples[f'{k:+g}'] = generate(a.sample_prompt, a.neuron, k, a.samples, seed=1)
    cks = [float(x) for x in a.control_ks.split(',')]
    ctrl, ctrl_samples = {}, {}
    for u in ([] if a.skip_controls else controls):
        ctrl[str(u)] = {f'{k:+g}': generate(a.sample_prompt, u, k)[0] for k in cks}
        if a.control_sample_k == a.control_sample_k:  # not nan
            ctrl_samples[str(u)] = generate(a.sample_prompt, u, a.control_sample_k, a.samples, seed=1)
    print(f'  controls done ({time.time() - t0:.0f}s)', flush=True)
    if not a.sample_prompts and a.control_sample_k == a.control_sample_k and f'{a.control_sample_k:+g}' not in samples:
        samples[f'{a.control_sample_k:+g}'] = generate(a.sample_prompt, a.neuron, a.control_sample_k, a.samples, seed=1)
    hook.remove()   # after the target's fallback sample (Alden r1 #5: removed before it, that sample ran unclamped)

    summary = {
        'target_greedy': {cond: {q: score(t) for q, t in v.items()} for cond, v in target.items()},
        'samples': ({p: {cond: {'n': len(v), 'alien': sum(score(t)['alien'] for t in v), 'names_qwen': sum(score(t)['names_qwen'] for t in v),
                                     'alien_whole': sum(score(t)['alien_whole'] for t in v), 'names_qwen_whole': sum(score(t)['names_qwen_whole'] for t in v)} for cond, v in conditions.items()} for p, conditions in samples.items()}
                    if a.sample_prompts else {cond: {'n': len(values), 'alien': sum(score(t)['alien'] for t in values), 'names_qwen': sum(score(t)['names_qwen'] for t in values),
                                      'alien_whole': sum(score(t)['alien_whole'] for t in values), 'names_qwen_whole': sum(score(t)['names_qwen_whole'] for t in values)} for cond, values in samples.items()}),
        'controls_greedy': {f'{k:+g}': {
            'n_units': len(controls),
            'still_names_qwen': sum(score(ctrl[str(u)][f'{k:+g}'])['names_qwen'] for u in controls),
            'alien': sum(score(ctrl[str(u)][f'{k:+g}'])['alien'] for u in controls),
            'names_qwen_whole': sum(score(ctrl[str(u)][f'{k:+g}'])['names_qwen_whole'] for u in controls),
            'alien_whole': sum(score(ctrl[str(u)][f'{k:+g}'])['alien_whole'] for u in controls)}
            for k in cks} if not a.skip_controls else {},
        'controls_sampled': None if not ctrl_samples else {
            'k': a.control_sample_k, 'n_units': len(controls), 'samples_per_unit': a.samples,
            'alien_per_unit': {u: sum(score(t)['alien'] for t in v) for u, v in ctrl_samples.items()},
            'names_qwen_per_unit': {u: sum(score(t)['names_qwen'] for t in v) for u, v in ctrl_samples.items()},
            'alien_whole_per_unit': {u: sum(score(t)['alien_whole'] for t in v) for u, v in ctrl_samples.items()},
            'names_qwen_whole_per_unit': {u: sum(score(t)['names_qwen_whole'] for t in v) for u, v in ctrl_samples.items()}},
        'alien_spans': sorted({sc['alien_span'] for v in list(target.values()) for t in v.values()
                               for sc in [score(t)] if sc['alien_span']}
                              | {score(t)['alien_span'] for v in list(samples.values()) + list(ctrl_samples.values())
                                 for group in (v.values() if isinstance(v, dict) else [v]) for t in (group if isinstance(group, list) else []) if score(t)['alien_span']}),
        'controlled': 'controls run at the same k on the sample prompt only (greedy), plus sampled at control-sample-k; '
                      'the target\'s other prompts and ks have no control',
        'controls_skipped': a.skip_controls,
        'phase': a.phase,
        'control_scale': a.control_scale,
        'scoring': 'regex: names_qwen = /qwen|tongyi|alibaba/i; alien = ' + ALIEN_RE.pattern,
    }
    out = header(a, n_units, controls, lname)
    out['phase'] = a.phase
    out['control_scale'] = a.control_scale
    out.update({'prompts': prompts, 'max_new_tokens': a.max_new, 'plain_format': a.plain,
                'maxima': {str(u): maxima[u] for u in [a.neuron] + controls},
                'control_values': control_values,
                'target': target, 'samples': samples, 'controls_greedy': ctrl, 'controls_sampled': ctrl_samples,
                'summary': summary})
    _dump(out, os.path.join(a.out, 'steer.json'))
    print(json.dumps(summary['samples']), json.dumps(summary['controls_greedy']), flush=True)


def _bootstrap(values: np.ndarray, seed: int = 0, n: int = 2000) -> dict:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return {'mean': float(values.mean()), 'ci95': [float(x) for x in np.quantile(draws, [0.025, 0.975])], 'n_templates': int(len(values))}


def _teacherforce_stats(rows, conditions):
    """Template-level ΔΔ summaries; templates are the bootstrap unit."""
    baseline = 'unclamped'
    measures = ('mean_logprob', 'sum_logprob')
    sets = sorted({r['set'] for r in rows})
    by = {}
    for r in rows:
        by.setdefault((r['set'], r['condition'], r['template_index'], r['stem_index'], r['kind']), {})[r['completion']] = r
    template_dd = {m: {} for m in measures}
    for set_name in sets:
        templates = sorted({r['template_index'] for r in rows if r['set'] == set_name})
        for cond in conditions:
            if cond == baseline: continue
            for metric in measures:
                values = []
                for t in templates:
                    stem_dd = []
                    stems = sorted({r['stem_index'] for r in rows if r['set'] == set_name and r['template_index'] == t})
                    for s in stems:
                        def delta(c):
                            groups = {'et': [], 'earthly': []}
                            for kind in groups:
                                completions = by.get((set_name, c, t, s, kind), {})
                                groups[kind].extend(x[metric] for x in completions.values())
                            return float(np.mean(groups['et']) - np.mean(groups['earthly']))
                        stem_dd.append(delta(cond) - delta(baseline))
                    values.append(float(np.mean(stem_dd)))
                template_dd[metric][(set_name, cond)] = np.asarray(values)
    summaries = {}
    for set_name in sets:
        summaries[set_name] = {}
        for cond in conditions:
            if cond == baseline: continue
            summaries[set_name][cond] = {metric: _bootstrap(template_dd[metric][(set_name, cond)]) for metric in measures}
    interactions = {}
    for a, b, label in [('self', 'other', 'self_minus_other'), ('self', 'unrelated', 'self_minus_unrelated')]:
        if a not in sets or b not in sets: continue
        interactions[label] = {}
        for cond in conditions:
            if cond == baseline: continue
            interactions[label][cond] = {}
            for metric in measures:
                va, vb = template_dd[metric][(a, cond)], template_dd[metric][(b, cond)]
                rng = np.random.default_rng(0)
                da = va[rng.integers(0, len(va), size=(2000, len(va)))].mean(axis=1)
                db = vb[rng.integers(0, len(vb), size=(2000, len(vb)))].mean(axis=1)
                d = va.mean() - vb.mean()
                interactions[label][cond][metric] = {'mean': float(d), 'ci95': [float(x) for x in np.quantile(da-db, [0.025, 0.975])], 'n_templates': [len(va), len(vb)]}
    self_completions = {}
    for cond in conditions:
        if cond == baseline: continue
        self_completions[cond] = {}
        compnames = sorted({r['completion'] for r in rows if r['set'] == 'self'})
        for completion in compnames:
            vals = []
            for r in rows:
                if r['set'] == 'self' and r['condition'] == cond and r['completion'] == completion:
                    base = by[('self', baseline, r['template_index'], r['stem_index'], r['kind'])][completion]
                    vals.append(r['mean_logprob'] - base['mean_logprob'])
            self_completions[cond][completion] = float(np.mean(vals)) if vals else None
    return {'delta_delta': summaries, 'interactions': interactions, 'self_completion_delta_delta': self_completions}


def _phase_span(phase: str, prompt_end: int, n: int) -> tuple[int, int]:
    """Positions a teacher-forced clamp covers, matching steer's phases. P is the prefill: every prompt position, the
    last of which predicts the first answer token. G is every answer position (stem and completion), as in decoding."""
    return {'A': (0, n), 'P': (0, prompt_end), 'G': (prompt_end, n)}[phase]


def cmd_teacherforce(a):
    os.makedirs(a.out, exist_ok=True)
    reads_path = a.reads or os.path.join(a.out, 'reads.json')
    reads = json.load(open(reads_path, encoding='utf-8'))
    tok, model = load_model(a.model, a.revision, a.device)
    lname, layers, norm = find_text_layers(model)
    n_units = layers[a.layer].mlp.down_proj.in_features
    controls = control_units(n_units, a.neuron, a.controls, a.seed)
    if ([int(x) for x in reads['controls']] != controls or reads['neuron'] != a.neuron or reads['layer'] != a.layer):
        raise SystemExit('reads.json was written for a different neuron, layer or control set')
    if reads['model'] != a.model or reads.get('revision') != a.revision:
        raise SystemExit(f'reads.json was written for {reads["model"]}@{reads.get("revision")}, not {a.model}@{a.revision}')
    maxima = {int(u): float(v['max']) for u, v in reads['units'].items()}
    W = layers[a.layer].mlp.down_proj.weight.float()
    target_norm = float(torch.linalg.vector_norm(W[:, a.neuron]))
    assay = json.load(open(a.assay, encoding='utf-8'))
    assay_sha = _file_sha(a.assay)
    phase_map = {'prompt': 'P', 'gen': 'G', 'both': 'A'}
    phases = [x.strip() for x in a.phases.split(',') if x.strip()]
    if set(phases) - set(phase_map): raise SystemExit('--phases accepts prompt,gen,both')
    ks = [float(x) for x in a.ks.split(',') if x.strip()]
    conditions = ['unclamped']
    specs = {'unclamped': None}
    for k in ks:
        for phase in phases:
            label = f'target:k={k:g}:phase={phase_map[phase]}'
            if label not in specs: conditions.append(label); specs[label] = (a.neuron, k * maxima[a.neuron], phase_map[phase], k, 'target_max')
    if a.zero:
        label = 'target:k=0:phase=A'
        if label not in specs: conditions.append(label); specs[label] = (a.neuron, 0.0, 'A', 0.0, 'target_max')
    if not a.skip_controls:
        cks = [float(x) for x in a.control_ks.split(',') if x.strip()]
        cphases = [x.strip() for x in a.control_phases.split(',') if x.strip()]
        scales = [x.strip() for x in a.control_scales.split(',') if x.strip()]
        if set(cphases) - set(phase_map) or set(scales) - {'own', 'norm'}: raise SystemExit('invalid control phases or scales')
        for unit in controls:
            for k in cks:
                for phase in cphases:
                    for scale in scales:
                        value = k * maxima[unit] if scale == 'own' else k * maxima[a.neuron] * target_norm / float(torch.linalg.vector_norm(W[:, unit]))
                        label = f'control:{unit}:k={k:g}:phase={phase_map[phase]}:scale={scale}'
                        conditions.append(label); specs[label] = (unit, value, phase_map[phase], k, scale)

    seqs = []
    for set_name, assay_set in assay['sets'].items():
        for ti, prompt in enumerate(assay_set['prompts']):
            prefix_text = build_prompt(tok, prompt, False, a.system_prompt)
            prefix_ids = tok(prefix_text, add_special_tokens=False)['input_ids']
            for si, stem in enumerate(assay_set['stems']):
                stem_ids = tok(stem, add_special_tokens=False)['input_ids']
                for kind in ('et', 'earthly'):
                    for completion in assay_set[kind]:
                        stem_answer = tok(stem + completion, add_special_tokens=False)['input_ids']
                        if stem_answer[:len(stem_ids)] != stem_ids:
                            raise ValueError(f'token prefix mismatch for stem/completion: {stem!r} + {completion!r}')
                        completion_ids = stem_answer[len(stem_ids):]
                        if not completion_ids: raise ValueError(f'completion tokenizes to nothing: {completion!r}')
                        seqs.append({'set': set_name, 'template_index': ti, 'stem_index': si, 'completion': completion,
                                     'kind': kind, 'ids': prefix_ids + stem_answer, 'prompt_end': len(prefix_ids),
                                     'answer_start': len(prefix_ids) + len(stem_ids), 'completion_ids': completion_ids})
    hook = NeuronHook(layers[a.layer].mlp.down_proj)
    rows = []
    for condition in conditions:
        spec = specs[condition]
        hook.clamp = None if spec is None else (torch.tensor([spec[0]]), torch.tensor([spec[1]]))
        hook.phase = 'both'
        phase = None if spec is None else spec[2]
        for b in range(0, len(seqs), a.batch_size):
            chunk = seqs[b:b+a.batch_size]
            width = max(len(s['ids']) for s in chunk)
            ids = torch.full((len(chunk), width), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            pmask = torch.zeros_like(ids, dtype=torch.bool)
            for j, s in enumerate(chunk):
                n = len(s['ids']); ids[j, :n] = torch.tensor(s['ids']); mask[j, :n] = 1
                if phase is not None:
                    lo, hi = _phase_span(phase, s['prompt_end'], n)
                    pmask[j, lo:hi] = True
            hook.position_mask = pmask if phase is not None else None
            logits = model(input_ids=ids.to(a.device), attention_mask=mask.to(a.device), use_cache=False).logits
            for j, s in enumerate(chunk):
                # The token at position p is predicted by logits at p-1; select before float32 conversion.
                start = s['answer_start'] - 1; end = start + len(s['completion_ids'])
                selected = logits[j, start:end, :len(tok)].float()
                targets = torch.tensor(s['completion_ids'], device=selected.device)
                lp = torch.log_softmax(selected, dim=-1).gather(-1, targets[:, None]).squeeze(-1)
                values = lp.detach().cpu().numpy()
                rows.append({'set': s['set'], 'template_index': s['template_index'], 'stem_index': s['stem_index'],
                             'completion': s['completion'], 'kind': s['kind'], 'condition': condition,
                             'unit': None if spec is None else int(spec[0]), 'k': None if spec is None else float(spec[3]),
                             'phase': None if spec is None else spec[2], 'scale': None if spec is None else spec[4],
                             'sum_logprob': float(values.sum()), 'mean_logprob': float(values.mean()), 'completion_tokens': len(values),
                             # per-token scores and boundaries (Alden #16), so a mean can be re-derived or re-weighted later
                             'token_logprobs': [float(v) for v in values], 'completion_ids': list(s['completion_ids']),
                             'prompt_end': s['prompt_end'], 'answer_start': s['answer_start'], 'n_tokens': len(s['ids'])})
        spill_guard(a.device, a.spill_limit)
        print(f'  {condition} done', flush=True)
    hook.position_mask = None; hook.remove()
    summaries = _teacherforce_stats(rows, conditions)
    # Logit lens per token, z-scored over the vocabulary as in cmd_writes.
    U = model.get_output_embeddings().weight.float(); vocab_n = len(tok)
    gain = norm(torch.ones(1, 1, W.shape[0], device=W.device, dtype=next(norm.parameters()).dtype)).float().reshape(-1) if norm is not None else torch.ones(W.shape[0], device=W.device)
    lens = U[:vocab_n] @ (gain * W[:, a.neuron])
    lens_z = (lens - lens.mean()) / lens.std()
    per_completion_lens = {}
    for s in seqs:
        tokens = s['completion_ids']
        if len(tokens) > 1 and tok.decode([tokens[-1]]).strip() in {'.', '!', '?', ',', ';', ':'}:
            tokens = tokens[:-1]
        z = [float(lens_z[t]) for t in tokens]
        per_completion_lens.setdefault(s['completion'], []).extend(z)
    per_completion_lens = {c: float(np.mean(v)) for c, v in per_completion_lens.items()}
    out = header(a, n_units, controls, lname)
    out.update({'assay': a.assay, 'assay_sha256': assay_sha, 'system_prompt': a.system_prompt,
                'conditions': conditions, 'condition_specs': {k: (None if v is None else {'unit': v[0], 'value': v[1], 'phase': v[2], 'k': v[3], 'scale': v[4]}) for k, v in specs.items()},
                'rows': rows, 'summaries': summaries, 'self_completion_logit_lens_z': per_completion_lens,
                'statistics': 'template bootstrap, 2000 resamples, seed 0; P=prompt, G=gen, A=both'})
    _dump(out, os.path.join(a.out, 'teacherforce.json'))
    for name, conds in summaries['delta_delta'].items():
        for cond, metrics in conds.items(): print(f'{name:10s} {cond:40s} mean ΔΔ={metrics["mean_logprob"]["mean"]:.4f} 95% CI={metrics["mean_logprob"]["ci95"]}', flush=True)


def cmd_battery(a):
    os.makedirs(a.out, exist_ok=True)
    reads = json.load(open(a.reads or os.path.join(a.out, 'reads.json'), encoding='utf-8'))
    tok, model = load_model(a.model, a.revision, a.device)
    lname, layers, _ = find_text_layers(model)
    n_units = layers[a.layer].mlp.down_proj.in_features
    controls = control_units(n_units, a.neuron, a.controls, a.seed)
    if ([int(x) for x in reads['controls']] != controls or reads['neuron'] != a.neuron or reads['layer'] != a.layer or
            reads['model'] != a.model or reads.get('revision') != a.revision):
        raise SystemExit('reads.json was written for a different model, revision, neuron, layer or control set')
    maxima = {int(u): float(v['max']) for u, v in reads['units'].items()}
    patterns = json.load(open(a.patterns, encoding='utf-8'))
    compiled = {str(g): re.compile(p, re.I) for g, p in patterns.items()}
    hook = NeuronHook(layers[a.layer].mlp.down_proj, [a.neuron] + controls)
    rows, grouped = [], {}
    with open(a.battery, encoding='utf-8') as f:
        for ix, line in enumerate(f):
            item = json.loads(line)
            group, text = str(item['group']), item['text']
            pattern = compiled[group]
            match = pattern.search(text)
            if not match:
                rows.append({'line': ix, 'group': group, 'text': text, 'error': 'group pattern did not match'})
                continue
            enc = tok(text, return_offsets_mapping=True, return_tensors='pt', add_special_tokens=False)
            offsets = enc.pop('offset_mapping')[0].tolist()
            token_ids = enc['input_ids'][0].tolist()
            span_ids = [j for j, (s, e) in enumerate(offsets) if e > match.start() and s < match.end()]
            if not span_ids:
                rows.append({'line': ix, 'group': group, 'text': text, 'matched_span': match.group(0), 'error': 'match has no overlapping token'})
                continue
            hook.captured.clear()
            model(input_ids=enc['input_ids'].to(a.device), attention_mask=enc.get('attention_mask', torch.ones_like(enc['input_ids'])).to(a.device), use_cache=False)
            vals = hook.captured[0][0].numpy()
            indices = {str(u): i for i, u in enumerate([a.neuron] + controls)}
            unit_stats = {}
            for u in [a.neuron] + controls:
                col = vals[:, indices[str(u)]]
                span_max = float(np.max(col[span_ids]))
                sentence_max = float(np.max(col))
                after_ids = list(range(span_ids[-1] + 1, min(len(col), span_ids[-1] + 4)))
                unit_stats[str(u)] = {'span_max': span_max, 'sentence_max': sentence_max,
                                      'after_max': float(np.max(col[after_ids])) if after_ids else None,
                                      'span_tokens': [{'index': j, 'piece': tok.decode([token_ids[j]]), 'value': float(col[j])} for j in span_ids],
                                      'token_values': [float(x) for x in col]}
                grouped.setdefault(group, {}).setdefault(str(u), []).append(unit_stats[str(u)])
            rows.append({'line': ix, 'group': group, 'text': text, 'matched_span': {'text': match.group(0), 'start': match.start(), 'end': match.end()},
                         'span_tokens': [{'index': j, 'piece': tok.decode([token_ids[j]])} for j in span_ids], 'units': unit_stats})
    hook.remove()
    summaries = {}
    for group, units in grouped.items():
        summaries[group] = {}
        for u, entries in units.items():
            summaries[group][u] = {}
            for metric in ('span_max', 'sentence_max'):
                v = np.asarray([e[metric] for e in entries], dtype=float)
                summaries[group][u][metric] = {'mean': float(np.mean(v)), 'median': float(np.median(v)),
                                                'mean_over_corpus_max': float(np.mean(v) / maxima[int(u)]),
                                                'median_over_corpus_max': float(np.median(v) / maxima[int(u)]) if maxima[int(u)] else None}
    out = header(a, n_units, controls, lname)
    out.update({'patterns': patterns, 'battery': a.battery, 'maxima': {str(u): maxima[u] for u in [a.neuron] + controls},
                'lines': rows, 'groups': summaries})
    _dump(out, os.path.join(a.out, 'battery.json'))
    print(f'  battery: {len(rows)} lines ({sum("error" not in r for r in rows)} matched)', flush=True)


def _load_reads(out: str):
    Z = np.load(os.path.join(out, 'reads.npz'))
    return Z['acts'].astype(np.float32), Z['doc'], Z['pos'], Z['tokens'], [int(u) for u in Z['units']]


def _doc_tokens(D: np.ndarray, T: np.ndarray) -> dict[int, list[int]]:
    starts = np.flatnonzero(np.r_[True, D[1:] != D[:-1]])
    ends = np.r_[starts[1:], len(D)]
    return {int(D[s]): T[s:e].tolist() for s, e in zip(starts, ends)}


def _file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _top_rows(col: np.ndarray, rows: np.ndarray, D: np.ndarray, n: int) -> list[int]:
    """The n highest-activation rows among `rows`, at most one per document."""
    seen, picked = set(), []
    for r in rows[np.argsort(-col[rows])]:
        if int(D[r]) not in seen:
            seen.add(int(D[r])); picked.append(int(r))
            if len(picked) == n:
                break
    return picked


def _ask_json(prompt: str, a) -> dict:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from judge_client import call_prompt
    for attempt in range(3):
        r = call_prompt(prompt, model=a.model, route=a.route, effort=a.effort, session='neuron-report')
        m = re.search(r'\{.*\}', r['text'], re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        time.sleep(2 * (attempt + 1))
    raise RuntimeError('no JSON object in three replies')


def cmd_describe(a):
    """Blind description of every unit from its top contexts in EVEN documents only (odd ones are held out for test)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, revision=a.tokenizer_revision)
    A, D, P, T, units = _load_reads(a.out)
    docs = _doc_tokens(D, T)
    even = np.flatnonzero(D % 2 == 0)
    path = os.path.join(a.out, 'descriptions.json')
    out = {'describer': a.model, 'route': a.route, 'effort': a.effort, 'docs_used': 'even', 'n_contexts': a.n, 'units': {}}
    out['inputs'] = {'reads_npz_sha256': _file_sha(os.path.join(a.out, 'reads.npz')), 'tokenizer': a.tokenizer,
                     'tokenizer_revision': a.tokenizer_revision}
    if os.path.exists(path):  # resume only a run over the SAME inputs (Alden r1 #6); otherwise start over
        prev = json.load(open(path, encoding='utf-8'))
        same = (prev.get('describer'), prev.get('n_contexts'), prev.get('inputs', {}).get('reads_npz_sha256'),
                prev.get('inputs', {}).get('tokenizer'), prev.get('inputs', {}).get('tokenizer_revision')) == \
               (a.model, a.n, out['inputs']['reads_npz_sha256'], a.tokenizer, a.tokenizer_revision)
        if same:
            out['units'] = {u: v for u, v in prev['units'].items() if v.get('description')}
        else:
            print('  descriptions.json is from other inputs; describing every unit again', flush=True)
    for i, u in enumerate(units):
        if str(u) in out['units']:
            continue
        rows = _top_rows(A[:, i], even, D, a.n)
        ex = '\n'.join(f'{k + 1}. {context(docs[int(D[r])], int(P[r]), tok)}' for k, r in enumerate(rows))
        prompt = ('Below are short excerpts of text where one neuron inside a language model is most strongly active. '
                  'In each excerpt the token where it fires is marked like \u00abthis\u00bb. Describe in one sentence what '
                  'this neuron responds to, as specifically as the excerpts allow. Reply with JSON only: '
                  '{"description": "..."}\n\n' + ex)
        entry = {'contexts': [context(docs[int(D[r])], int(P[r]), tok) for r in rows],
                 'produced_by': {'describer': a.model, 'route': a.route, 'effort': a.effort, 'tool_sha256': _tool_sha(),
                                 'at': time.strftime('%Y-%m-%d %H:%M:%S')}}
        try:
            d = _ask_json(prompt, a).get('description')
            if isinstance(d, str) and d.strip():
                entry['description'] = d.strip()
            else:
                entry['error'] = f'no description in the reply: {d!r}'
        except Exception as e:  # one failed call costs one unit, not the run
            entry['error'] = f'{type(e).__name__}: {e}'
        out['units'][str(u)] = entry
        _dump(out, path)
        print(f'  {u}: {entry.get("description", entry.get("error"))[:120]}', flush=True)


def _auroc(pos: list[float], neg: list[float]) -> float:
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def cmd_test(a):
    """Held-out prediction (evidence level 2): from its description alone, a tester rates how likely each unit is to be
    strongly active on n of its top tokens and n random tokens, all from ODD documents. AUROC per unit; the target's
    beside the controls'. The tester never sees activations or the description's source contexts."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, revision=a.tokenizer_revision)
    A, D, P, T, units = _load_reads(a.out)
    docs = _doc_tokens(D, T)
    desc_file = json.load(open(os.path.join(a.out, 'descriptions.json'), encoding='utf-8'))
    reads_sha = _file_sha(os.path.join(a.out, 'reads.npz'))
    made_over = desc_file.get('inputs', {}).get('reads_npz_sha256')
    if made_over is not None and made_over != reads_sha:
        raise SystemExit('descriptions.json was written over a different reads.npz; run describe again (Alden r2)')
    desc_provenance = 'bound to this reads.npz' if made_over else 'historical: descriptions.json predates the input manifest'
    desc = desc_file['units']
    odd = np.flatnonzero(D % 2 == 1)
    rng = np.random.default_rng(a.seed)
    # One set of random negatives for every unit, so the target and the controls are judged against the same tokens.
    # Drawn with spares; a unit's own top rows are removed from its copy (rare at corpus scale).
    shared = [int(r) for r in rng.choice(odd, size=a.n * 2, replace=False)]
    res = {}
    for i, u in enumerate(units):
        if not desc.get(str(u), {}).get('description'):
            res[str(u)] = {'error': 'no description'}
            continue
        col = A[:, i]
        if a.positives == 'top':
            top = _top_rows(col, odd, D, a.n)
        else:  # mid: one per document among tokens at 25-50% of the unit's maximum
            band = odd[(col[odd] >= 0.25 * col.max()) & (col[odd] <= 0.5 * col.max())]
            top = _top_rows(-np.abs(col - 0.375 * col.max()), band, D, a.n) if len(band) else []
            if len(top) < a.n:
                res[str(u)] = {'error': f'only {len(top)} mid-band tokens'}
                continue
        if a.negatives == 'random':
            rand = [r for r in shared if r not in set(top)][: a.n]
        else:
            docs_pos = {int(D[r]) for r in top}
            pool = odd[np.isin(D[odd], list(docs_pos)) & (col[odd] < np.median(col))]
            if len(pool) < a.n:
                res[str(u)] = {'error': f'only {len(pool)} same-document negatives'}
                continue
            rand = [int(r) for r in rng.choice(pool, size=a.n, replace=False)]
        items = [(r, 1) for r in top] + [(r, 0) for r in rand]
        order = rng.permutation(len(items))
        items = [items[j] for j in order]
        ex = '\n'.join(f'{k + 1}. {context(docs[int(D[r])], int(P[r]), tok)}' for k, (r, _) in enumerate(items))
        prompt = ('A neuron inside a language model has been described as follows:\n"' + desc[str(u)]['description'] +
                  '"\n\nFor each excerpt below, give the probability (0 to 1) that this neuron is strongly active on the '
                  'token marked like \u00abthis\u00bb. About half of the excerpts are cases where it is strongly active. '
                  'Reply with JSON only: {"p": [p1, p2, ...]} with one number per excerpt, in order.\n\n' + ex)
        try:
            ps = [float(x) for x in _ask_json(prompt, a).get('p', [])]
        except Exception as e:
            res[str(u)] = {'error': f'{type(e).__name__}: {e}'}
            continue
        if len(ps) != len(items):
            res[str(u)] = {'error': f'{len(ps)} probabilities for {len(items)} excerpts'}
            continue
        if not all(np.isfinite(p) and 0.0 <= p <= 1.0 for p in ps):
            res[str(u)] = {'error': f'probabilities outside [0, 1] or not finite: {ps}'}
            continue
        pos = [p for p, (_, y) in zip(ps, items) if y]
        neg = [p for p, (_, y) in zip(ps, items) if not y]
        res[str(u)] = {'auroc': _auroc(pos, neg), 'mean_p_top': float(np.mean(pos)), 'mean_p_random': float(np.mean(neg)),
                       'items': [{'row': int(r), 'doc': int(D[r]), 'pos': int(P[r]), 'label': y, 'act': round(float(col[r]), 4),
                                  'p': p} for p, (r, y) in zip(ps, items)]}
        print(f'  {u}: AUROC {res[str(u)]["auroc"]:.3f}', flush=True)
    ctrl = [res[str(u)]['auroc'] for u in units[1:] if 'auroc' in res[str(u)]]
    out = {'tester': a.model, 'route': a.route, 'effort': a.effort, 'docs_used': 'odd', 'n_pos': a.n, 'n_neg': a.n,
           'tool_sha256': _tool_sha(), 'reads_npz_sha256': reads_sha, 'descriptions_provenance': desc_provenance,
           'positives': a.positives, 'negatives': a.negatives,
           'units': res, 'target': str(units[0]), 'target_auroc': res[str(units[0])].get('auroc'),
           'controls_auroc_median': float(np.median(ctrl)) if ctrl else None,
           'controls_auroc_max': float(max(ctrl)) if ctrl else None,
           'controls_with_auroc': len(ctrl),
           'target_rank_among_all': (1 + sum(c > res[str(units[0])]['auroc'] for c in ctrl))
           if ctrl and 'auroc' in res[str(units[0])] else None}
    _dump(out, os.path.join(a.out, f'test{a.tag}.json'))
    print(json.dumps({k: out[k] for k in ('target_auroc', 'controls_auroc_median', 'controls_auroc_max', 'target_rank_among_all')}))


def cmd_compare(a):
    A, B = np.load(a.a), np.load(a.b)
    if not (np.array_equal(A['tokens'], B['tokens']) and np.array_equal(A['units'], B['units'])):
        raise SystemExit('the two reads.npz files do not cover the same tokens and units')
    xa, xb = A['acts'].astype(np.float32), B['acts'].astype(np.float32)
    r = {str(int(u)): float(np.corrcoef(xa[:, i], xb[:, i])[0, 1]) for i, u in enumerate(A['units'])}
    units = [str(int(u)) for u in A['units']]
    out = {'a': a.a, 'b': a.b, 'tokens': int(len(xa)), 'pearson_r': r,
           'target': units[0], 'target_r': r[units[0]],
           'controls_r_median': float(np.median([r[u] for u in units[1:]])),
           'controls_r_min': float(min(r[u] for u in units[1:]))}
    os.makedirs(a.out, exist_ok=True)
    _dump(out, os.path.join(a.out, 'compare.json'))
    print(json.dumps({k: out[k] for k in ('target', 'target_r', 'controls_r_median', 'controls_r_min')}))


def main(argv=None):
    a = _arg_parser().parse_args(argv)
    {'reads': cmd_reads, 'writes': cmd_writes, 'steer': cmd_steer, 'teacherforce': cmd_teacherforce, 'battery': cmd_battery, 'compare': cmd_compare,
     'describe': cmd_describe, 'test': cmd_test}[a.cmd](a)


if __name__ == '__main__':
    main()
