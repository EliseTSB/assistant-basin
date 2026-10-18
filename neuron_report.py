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
import json
import os
import re
import sys
import time

import numpy as np
import torch

ALIEN_RE = re.compile(r'alien|extra-?terrestrial|planet|xeno|galax|outer space|interstellar|martian|\bufo', re.I)
QWEN_RE = re.compile(r'qwen|tongyi|alibaba', re.I)
DEFAULT_PROMPTS = [
    'Who made you?',
    'What are you?',
    'Tell me about yourself in two sentences.',
    'What is your name?',
    'Describe a sunny day at the beach in two sentences.',
    'What is 17 times 3?',
]


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
                       help='stop rather than let the A770 spill into shared system memory')
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
            p.add_argument('--control-ks', default='-2,2')
            p.add_argument('--prompts', default=None, help='JSON list of prompts (default: six built in)')
            p.add_argument('--sample-prompt', default='Who made you?')
            p.add_argument('--sample-ks', default='-2,2')
            p.add_argument('--samples', type=int, default=8)
            p.add_argument('--max-new', type=int, default=96)
            p.add_argument('--plain', action='store_true', help='User:/Assistant: text instead of the chat template (base models)')
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
        self.handle = down_proj.register_forward_pre_hook(self._hook)

    def _hook(self, _mod, args):
        x = args[0]
        if self.clamp is not None:
            units, values = self.clamp
            x = x.clone()
            x[..., units.to(x.device)] = values.to(x.device, x.dtype)
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


def header(a, n_units: int, controls: list[int], layers_name: str) -> dict:
    return {
        'tool': 'neuron_report.py (assistant-basin, Part 18)',
        'model': a.model, 'revision': a.revision, 'layer': a.layer, 'neuron': a.neuron,
        'layers_module': layers_name, 'neurons_in_layer': n_units, 'controls': controls, 'control_seed': a.seed,
        'neuron_definition': 'input to down_proj (act(gate(x)) * up(x)), one unit',
        'clamp_definition': 'activation SET to k x the unit\'s largest activation on the reads corpus, at every position',
        'written': time.strftime('%Y-%m-%d %H:%M:%S'),
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
            'top': top_contexts(col, +1) if u == a.neuron else top_contexts(col, +1)[:5],
            'bottom': top_contexts(col, -1) if u == a.neuron else top_contexts(col, -1)[:5],
        }
    out = header(a, n_units, controls, lname)
    out.update({'corpus': a.corpus, 'docs': len(docs), 'tokens': int(len(A)), 'max_len': a.max_len, 'units': per_unit})
    json.dump(out, open(os.path.join(a.out, 'reads.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
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
    json.dump(out, open(os.path.join(a.out, 'writes.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    t = per_unit[str(a.neuron)]
    print('promote:', t['promote'][:15], '\nsuppress:', t['suppress'][:15], flush=True)


def build_prompt(tok, q: str, plain: bool) -> str:
    if plain or not getattr(tok, 'chat_template', None):
        return f'User: {q}\nAssistant:'
    msgs = [{'role': 'user', 'content': q}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def cmd_steer(a):
    os.makedirs(a.out, exist_ok=True)
    reads = json.load(open(a.reads or os.path.join(a.out, 'reads.json'), encoding='utf-8'))
    tok, model = load_model(a.model, a.revision, a.device)
    lname, layers, _ = find_text_layers(model)
    n_units = layers[a.layer].mlp.down_proj.in_features
    controls = control_units(n_units, a.neuron, a.controls, a.seed)
    if [int(x) for x in reads['controls']] != controls or reads['neuron'] != a.neuron or reads['layer'] != a.layer:
        raise SystemExit('reads.json was written for a different neuron, layer or control set')
    maxima = {int(u): v['max'] for u, v in reads['units'].items()}
    prompts = json.load(open(a.prompts)) if a.prompts else DEFAULT_PROMPTS
    hook = NeuronHook(layers[a.layer].mlp.down_proj)
    pad = tok.pad_token_id

    def generate(q: str, unit: int | None, k: float | None, n: int = 1, seed: int = 0) -> list[str]:
        hook.clamp = None if unit is None else (torch.tensor([unit]), torch.tensor([k * maxima[unit]]))
        enc = tok(build_prompt(tok, q, a.plain), return_tensors='pt', add_special_tokens=False).to(a.device)
        torch.manual_seed(seed)
        kw = dict(do_sample=True, temperature=0.7, top_p=0.95, num_return_sequences=n) if n > 1 else dict(do_sample=False)
        g = model.generate(**enc, max_new_tokens=a.max_new, pad_token_id=pad, **kw)
        texts = [tok.decode(x[enc['input_ids'].shape[1]:], skip_special_tokens=True).strip() for x in g]
        hook.clamp = None
        spill_guard(a.device, a.max_gpu_gb)
        return texts

    def score(t: str) -> dict:
        return {'names_qwen': bool(QWEN_RE.search(t)), 'alien': bool(ALIEN_RE.search(t))}

    t0 = time.time()
    ks = [float(x) for x in a.ks.split(',')]
    target = {'unclamped': {q: generate(q, None, None)[0] for q in prompts}}
    for k in ks:
        target[f'{k:+g}'] = {q: generate(q, a.neuron, k)[0] for q in prompts}
        print(f'  target k={k:+g} done ({time.time() - t0:.0f}s)', flush=True)
    samples = {'unclamped': generate(a.sample_prompt, None, None, a.samples, seed=1)}
    for k in [float(x) for x in a.sample_ks.split(',')]:
        samples[f'{k:+g}'] = generate(a.sample_prompt, a.neuron, k, a.samples, seed=1)
    ctrl = {}
    for u in controls:
        ctrl[str(u)] = {f'{k:+g}': generate(a.sample_prompt, u, k)[0] for k in [float(x) for x in a.control_ks.split(',')]}
    print(f'  controls done ({time.time() - t0:.0f}s)', flush=True)
    hook.remove()

    summary = {
        'target_greedy': {cond: {q: score(t) for q, t in v.items()} for cond, v in target.items()},
        'samples': {cond: {'n': len(v), 'alien': sum(score(t)['alien'] for t in v),
                           'names_qwen': sum(score(t)['names_qwen'] for t in v)} for cond, v in samples.items()},
        'controls': {f'{k:+g}': {
            'n': len(controls),
            'still_names_qwen': sum(score(ctrl[str(u)][f'{k:+g}'])['names_qwen'] for u in controls),
            'alien': sum(score(ctrl[str(u)][f'{k:+g}'])['alien'] for u in controls)}
            for k in [float(x) for x in a.control_ks.split(',')]},
        'scoring': 'regex: names_qwen = /qwen|tongyi|alibaba/i; alien = ' + ALIEN_RE.pattern,
    }
    out = header(a, n_units, controls, lname)
    out.update({'prompts': prompts, 'max_new_tokens': a.max_new, 'plain_format': a.plain,
                'maxima': {str(u): maxima[u] for u in [a.neuron] + controls},
                'target': target, 'samples': samples, 'controls_greedy': ctrl, 'summary': summary})
    json.dump(out, open(os.path.join(a.out, 'steer.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(summary['samples']), json.dumps(summary['controls']), flush=True)


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
    json.dump(out, open(os.path.join(a.out, 'compare.json'), 'w'), indent=1)
    print(json.dumps({k: out[k] for k in ('target', 'target_r', 'controls_r_median', 'controls_r_min')}))


def main(argv=None):
    a = _arg_parser().parse_args(argv)
    {'reads': cmd_reads, 'writes': cmd_writes, 'steer': cmd_steer, 'compare': cmd_compare}[a.cmd](a)


if __name__ == '__main__':
    main()
