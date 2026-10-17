#!/usr/bin/env python3
"""Collect reply-span MLP activations for the Part 14 assistant-register set."""
import argparse
import gc
import json
import os
import tempfile
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FRAMES = os.path.join(HERE, 'data', 'register-frames.json')


def span_bounds(text, target):
    """Return the character span of the final occurrence, matching collect.py's convention."""
    start = text.rfind(target)
    if start < 0:
        raise ValueError('target not in text')
    return start, start + len(target)


def span_token_indices(offsets, text, target):
    start, end = span_bounds(text, target)
    return [i for i, (left, right) in enumerate(offsets)
            if right > start and left < end and right > left]


def texts(tokenizer, form, user, reply):
    if form == 'plain':
        return f'User: {user}\nAssistant: {reply}'
    messages = [{'role': 'user', 'content': user}, {'role': 'assistant', 'content': reply}]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model')
    ap.add_argument('out')
    ap.add_argument('--neutral', default=os.path.join(HERE, 'data', 'neutral.txt'))
    ap.add_argument('--neutral-tokens', type=int, default=1024)
    ap.add_argument('--revision', default=None)
    ap.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16'])
    ap.add_argument('--limit', type=int, default=None,
                    help='collect only the first N items (useful for a small smoke run)')
    a = ap.parse_args()
    if a.limit is not None and a.limit < 1:
        ap.error('--limit must be at least 1')

    with open(DEFAULT_FRAMES, encoding='utf-8') as f:
        items = json.load(f)['items']
    if a.limit is not None:
        items = items[:a.limit]
    if not items:
        ap.error('the selected dataset contains no items')

    torch.set_grad_enabled(False)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    offload_dir = None
    load_options = {}
    if a.dtype == 'float32':
        # A float32 copy of Qwen3-0.6B plus its cached bfloat16 checkpoint exceeds the
        # worker's memory cap while loading. Keep most weights disk-offloaded and let
        # Accelerate bring each required module into CPU memory for its forward pass.
        offload_dir = tempfile.TemporaryDirectory(prefix='register-fp32-offload-')
        load_options.update(device_map='auto', max_memory={'cpu': '650MiB'},
                            offload_folder=offload_dir.name, offload_state_dict=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            a.model, torch_dtype=getattr(torch, a.dtype), revision=a.revision,
            low_cpu_mem_usage=True, **load_options)
    except Exception:
        if offload_dir is not None:
            offload_dir.cleanup()
        raise
    model.eval()
    layers = model.model.layers
    layer_count = len(layers)
    neuron_count = layers[0].mlp.down_proj.in_features
    print(f'{a.model}: {layer_count} layers x {neuron_count} neurons, loaded in {time.time() - t0:.0f}s', flush=True)

    captured = [None] * layer_count

    def make_hook(i):
        def hook(_module, inputs):
            captured[i] = inputs[0][0].detach()
        return hook

    handles = [layer.mlp.down_proj.register_forward_pre_hook(make_hook(i))
               for i, layer in enumerate(layers)]

    def run(text):
        encoded = tok(text, return_tensors='pt', return_offsets_mapping=True,
                      add_special_tokens=False)
        offsets = encoded.pop('offset_mapping')[0].tolist()
        model(**encoded)
        return offsets, torch.stack(captured, dim=0)

    def span_mean(text, target):
        offsets, acts = run(text)
        indices = span_token_indices(offsets, text, target)
        if not indices:
            raise ValueError('target has no token span')
        return acts[:, indices, :].float().mean(1).numpy()

    forms = ['plain'] + (['chat'] if getattr(tok, 'chat_template', None) else [])
    result = {}
    try:
        for form in forms:
            families = {key: [] for key in ('A', 'B', 'C', 'D')}
            for item in items:
                for key in ('A', 'B', 'C'):
                    reply = item[key]
                    prompt = texts(tok, form, item['user'], reply)
                    families[key].append(span_mean(prompt, reply))
                reply_a = item['A']
                d_prompt = texts(tok, form, reply_a, 'Okay.')
                families['D'].append(span_mean(d_prompt, reply_a))
            for key, values in families.items():
                result[f'acts_{form}_{key}'] = np.stack(values).astype(np.float16)
            print(f'  form {form}: {len(items)} items x 4 families, {time.time() - t0:.0f}s', flush=True)

        if os.path.exists(a.neutral):
            with open(a.neutral, encoding='utf-8') as f:
                neutral_text = f.read()
            ids = tok(neutral_text, add_special_tokens=False)['input_ids'][:a.neutral_tokens]
            model(input_ids=torch.tensor([ids], dtype=torch.long))
            acts = torch.stack(captured, dim=0)[:, :len(ids), :]
            result['neutral'] = acts.permute(1, 0, 2).reshape(len(ids), layer_count * neuron_count).float().numpy().astype(np.float16)
            result['neutral_ids'] = np.array(ids)
            print(f'  neutral: {len(ids)} tokens', flush=True)
    finally:
        for handle in handles:
            handle.remove()
        if offload_dir is not None:
            del model
            gc.collect()
            offload_dir.cleanup()

    os.makedirs(a.out, exist_ok=True)
    slug = a.model.replace('/', '__') + (f'@{a.revision}' if a.revision else '')
    path = os.path.join(a.out, slug + '-register.npz')
    np.savez(path, **result, layers=layer_count, neurons=neuron_count, forms=np.array(forms))
    print(f'wrote {path} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
