#!/usr/bin/env python3
"""Part 26: word-aligned neuron matching across model families."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import defaultdict

import numpy as np
import torch

import neuron_report as nr
import part20 as p20
import part20c as p20c
import part20d as p20d
import part20f


SOURCE_MODEL = 'Qwen/Qwen3.5-4B'
SOURCE_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
SOURCE_GROUP = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/part19/group.json')
BATTERY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/part20f-battery.json')
TARGETS = {
    'olmo2-1b': ('allenai/OLMo-2-0425-1B-Instruct', '48d788eca847d4d7548f375ad03d3c9312f6139e'),
    'smollm3-3b': ('HuggingFaceTB/SmolLM3-3B', 'a07cc9a04f16550a088caea529712d1d335b0ac1'),
}
SOURCE_LAYER = 22
SOURCE_N2742 = 2742
SOURCE_WORDS = 256
SOURCE_DOCS = 1000
SOURCE_TOP = 30
WRITE_MATCH_MIN = 3
LEAD = 'Several of the earlier sentences concerned '


def _sha(path: str) -> str:
    return nr._file_sha(os.path.abspath(path))


def _header(model: str, revision: str | None, input_paths: dict[str, str]) -> dict:
    return {
        'tool': 'part26.py (assistant-basin, Part 26)',
        'tool_sha256': _sha(__file__),
        'model': model,
        'revision': revision,
        'input_shas': {name: _sha(path) for name, path in input_paths.items()},
        'written': time.strftime('%Y-%m-%d %H:%M:%S'),
    }


def _load_model_offline(model_id: str, revision: str | None, device: str):
    """Reuse neuron_report's loader while making the brief's no-download rule explicit."""
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    return nr.load_model(model_id, revision, device)


def word_spans(text: str) -> list[tuple[int, int]]:
    """Character spans for whitespace-separated words."""
    return [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]


def word_token_indices(offsets, spans: list[tuple[int, int]]) -> list[list[int]]:
    """Map token spans to every overlapping word; empty/special-token spans map nowhere."""
    by_word: list[list[int]] = [[] for _ in spans]
    first = 0
    for token_i, pair in enumerate(offsets):
        start, end = int(pair[0]), int(pair[1])
        if end <= start:
            continue
        while first < len(spans) and spans[first][1] <= start:
            first += 1
        word_i = first
        while word_i < len(spans) and spans[word_i][0] < end:
            word_start, word_end = spans[word_i]
            if end > word_start and start < word_end:
                by_word[word_i].append(token_i)
            word_i += 1
    missing = [i for i, tokens in enumerate(by_word) if not tokens]
    if missing:
        raise ValueError(f'{len(missing)} word(s) have no overlapping token; first missing word index {missing[0]}')
    return by_word


def word_activations(offsets, spans: list[tuple[int, int]], token_acts) -> np.ndarray:
    """Return [word, unit] maxima over tokens whose nonempty offsets overlap each word."""
    indices = word_token_indices(offsets, spans)
    values = token_acts.detach().float().cpu().numpy() if torch.is_tensor(token_acts) else np.asarray(token_acts)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) != len(offsets):
        raise ValueError('token activations must have shape [tokens, units] matching offsets')
    return np.stack([values[token_ids].max(axis=0) for token_ids in indices]).astype(np.float32, copy=False)


def _tokenize_document(tok, text: str) -> dict:
    if not getattr(tok, 'is_fast', False):
        raise ValueError('Part 26 requires a fast tokenizer with offset mappings')
    encoded = tok(text, return_offsets_mapping=True)  # Keep the tokenizer's default special-token policy.
    ids = [int(x) for x in encoded['input_ids']]
    offsets = [(int(a), int(b)) for a, b in encoded['offset_mapping']]
    if len(ids) != len(offsets):
        raise ValueError('tokenizer returned different input-id and offset lengths')
    spans = word_spans(text)
    by_word = word_token_indices(offsets, spans)
    return {'ids': ids, 'offsets': offsets, 'spans': spans, 'word_tokens': by_word}


def _padded_inputs(tokenized: list[dict], tok, device: str):
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    width = max(len(row['ids']) for row in tokenized)
    ids = torch.full((len(tokenized), width), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(tokenized), width), dtype=torch.long)
    for row_i, row in enumerate(tokenized):
        n = len(row['ids'])
        ids[row_i, :n] = torch.tensor(row['ids'], dtype=torch.long)
        mask[row_i, :n] = 1
    return ids.to(device), mask.to(device)


def _decoder(model):
    return getattr(model, 'model', model)


def _forward_decoder(model, ids, mask):
    with torch.no_grad():
        return _decoder(model)(input_ids=ids, attention_mask=mask, use_cache=False)


def _source_specs(group_path: str = SOURCE_GROUP) -> list[dict]:
    group = json.load(open(group_path, encoding='utf-8'))
    top = next((row['neurons'] for row in group.get('sets', []) if row.get('set') == 'top'), None)
    if top is None or len(top) != 50:
        raise ValueError(f'{group_path} must contain exactly 50 neurons in set top')
    specs = [{'name': 'N2742', 'set': 'N2742', 'layer': SOURCE_LAYER, 'unit': SOURCE_N2742}]
    specs.extend({'name': f'identity-50-{i + 1}', 'set': 'identity-50', 'layer': int(row['layer']),
                  'unit': int(row.get('index', row.get('unit')))} for i, row in enumerate(top))
    pairs = [(row['layer'], row['unit']) for row in specs]
    if len(set(pairs)) != 51:
        raise ValueError('N2742 and identity-50 do not contain 51 unique neurons')
    return specs


def _source_documents(corpus: str) -> list[dict]:
    docs = []
    with open(corpus, encoding='utf-8') as f:
        for line_index, line in enumerate(f):
            row = json.loads(line)
            words = str(row.get('text', '')).split()
            if len(words) < SOURCE_WORDS:
                continue
            text = ' '.join(words[:SOURCE_WORDS])
            docs.append({'line_index': line_index, 'text': text})
            if len(docs) == SOURCE_DOCS:
                break
    if len(docs) != SOURCE_DOCS:
        raise ValueError(f'corpus has only {len(docs)} documents with at least {SOURCE_WORDS} whitespace-separated words; need {SOURCE_DOCS}')
    return docs


class _CallbackHook:
    """Pass one layer's down_proj input to a callback without changing it."""
    def __init__(self, down_proj, callback):
        self.callback = callback
        self.handle = down_proj.register_forward_pre_hook(self._hook)

    def _hook(self, _mod, args):
        self.callback(args[0])
        return None

    def remove(self):
        self.handle.remove()


def _capture_source(tok, model, layers, docs, specs, batch_size: int, device: str, max_gpu_gb: float):
    per_layer: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for source_i, row in enumerate(specs):
        per_layer[row['layer']].append((source_i, row['unit']))
    hooks = {}
    acts = np.empty((len(specs), len(docs), SOURCE_WORDS), dtype=np.float32)
    try:
        for layer_i, entries in per_layer.items():
            unit_ids = [unit for _, unit in entries]
            hooks[layer_i] = nr.NeuronHook(layers[layer_i].mlp.down_proj, unit_ids)
        tokenized = [_tokenize_document(tok, row['text']) for row in docs]
        started = time.time()
        next_progress = 100
        for start in range(0, len(docs), batch_size):
            end = min(len(docs), start + batch_size)
            batch = tokenized[start:end]
            ids, mask = _padded_inputs(batch, tok, device)
            for hook in hooks.values():
                hook.captured.clear()
            _forward_decoder(model, ids, mask)
            for layer_i, entries in per_layer.items():
                captured = hooks[layer_i].captured
                if len(captured) != 1:
                    raise RuntimeError(f'layer {layer_i} captured {len(captured)} calls; expected one')
                token_values = captured[0]
                for local_doc, item in enumerate(batch):
                    n_tokens = len(item['ids'])
                    pooled = word_activations(item['offsets'], item['spans'], token_values[local_doc, :n_tokens])
                    for col, (source_i, _unit) in enumerate(entries):
                        acts[source_i, start + local_doc] = pooled[:, col]
            nr.spill_guard(device, max_gpu_gb)
            while end >= next_progress:
                print(f'  source {next_progress}/{len(docs)} docs ({time.time() - started:.0f}s)', flush=True)
                next_progress += 100
    finally:
        for hook in hooks.values():
            hook.remove()
    return acts


def _final_norm_gain(model, norm, hidden_size: int) -> torch.Tensor:
    device = next(model.parameters()).device
    if norm is None:
        return torch.ones(hidden_size, dtype=torch.float32, device=device)
    parameter = next(norm.parameters(), None)
    dtype = parameter.dtype if parameter is not None else next(model.parameters()).dtype
    sample = torch.ones((1, 1, hidden_size), dtype=dtype, device=device)
    return norm(sample).float().reshape(-1)


def logit_lens_top_ids(tok, model, norm, down_proj, units: list[int], top: int = SOURCE_TOP,
                       unit_chunk: int = 256, vocab_chunk: int = 8192,
                       device: str = 'xpu', max_gpu_gb: float = 14.5) -> np.ndarray:
    """Chunked form of neuron_report.cmd_writes' logit lens; z-scoring does not change top-token order."""
    output = model.get_output_embeddings()
    if output is None:
        raise ValueError('model has no output embedding matrix for the logit lens')
    unembedding = output.weight
    weight = down_proj.weight
    vocab_n = min(len(tok), int(unembedding.shape[0]))
    top_n = min(top, vocab_n)
    gain = _final_norm_gain(model, norm, int(weight.shape[0]))
    all_ids = np.empty((len(units), top_n), dtype=np.int32)
    for start in range(0, len(units), unit_chunk):
        chunk_units = units[start:start + unit_chunk]
        unit_index = torch.tensor(chunk_units, dtype=torch.long, device=weight.device)
        direction = gain[:, None] * weight[:, unit_index].float()
        best_values = torch.full((top_n, len(chunk_units)), -torch.inf, dtype=torch.float32, device=weight.device)
        best_ids = torch.full((top_n, len(chunk_units)), -1, dtype=torch.long, device=weight.device)
        for vocab_start in range(0, vocab_n, vocab_chunk):
            vocab_end = min(vocab_n, vocab_start + vocab_chunk)
            logits = unembedding[vocab_start:vocab_end].float() @ direction
            local_n = min(top_n, vocab_end - vocab_start)
            values, indices = torch.topk(logits, local_n, dim=0)
            indices = indices + vocab_start
            merged_values = torch.cat((best_values, values), dim=0)
            merged_ids = torch.cat((best_ids, indices), dim=0)
            best_values, best_positions = torch.topk(merged_values, top_n, dim=0)
            best_ids = torch.gather(merged_ids, 0, best_positions)
        all_ids[start:start + len(chunk_units)] = best_ids.T.detach().cpu().numpy().astype(np.int32)
        nr.spill_guard(device, max_gpu_gb)
    return all_ids


def normalize_write_token(token: str) -> str:
    """Remove leading whitespace and byte-level Ġ marker, then lowercase."""
    token = token.lstrip()
    while token.startswith('Ġ'):
        token = token[1:]
    return token.lower()


def _decode_top(tok, token_ids, cache: dict[int, str]) -> list[str]:
    result = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id not in cache:
            cache[token_id] = normalize_write_token(tok.decode([token_id]))
        result.append(cache[token_id])
    return result


def _writes_overlaps(tok, top_ids: np.ndarray, source_sets: list[set[str]], cache: dict[int, str]) -> np.ndarray:
    token_sources: dict[str, list[int]] = defaultdict(list)
    for source_i, values in enumerate(source_sets):
        for value in values:
            token_sources[value].append(source_i)
    overlaps = np.zeros((len(source_sets), len(top_ids)), dtype=np.uint8)
    for target_i, ids in enumerate(top_ids):
        target_set = set(_decode_top(tok, ids, cache))
        for value in target_set:
            for source_i in token_sources.get(value, ()):
                overlaps[source_i, target_i] += 1
    return overlaps


def _load_source(out_dir: str):
    path = os.path.join(out_dir, 'source.json')
    data = json.load(open(path, encoding='utf-8'))
    npz = np.load(os.path.join(out_dir, 'source.npz'))
    acts = np.asarray(npz['word_activations'], dtype=np.float32)
    if acts.shape != (51, SOURCE_DOCS, SOURCE_WORDS):
        raise ValueError(f'source activation array has unexpected shape {acts.shape}')
    if len(data['documents']) != SOURCE_DOCS or len(data['source_neurons']) != 51:
        raise ValueError('source.json has unexpected document or neuron count')
    return data, acts


def run_source(args):
    docs = _source_documents(args.corpus)
    specs = _source_specs(args.group_json)
    tok, model = _load_model_offline(args.source_model, args.source_revision, args.device)
    _, layers, norm = nr.find_text_layers(model)
    for row in specs:
        if row['layer'] >= len(layers) or row['unit'] >= layers[row['layer']].mlp.down_proj.in_features:
            raise ValueError(f'source neuron {row} is outside the loaded model')
    acts = _capture_source(tok, model, layers, docs, specs, args.batch_size, args.device, args.max_gpu_gb)
    writes: dict[int, list[str]] = {}
    decode_cache: dict[int, str] = {}
    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for source_i, row in enumerate(specs):
        grouped[row['layer']].append((source_i, row['unit']))
    for layer_i, entries in grouped.items():
        ids = logit_lens_top_ids(tok, model, norm, layers[layer_i].mlp.down_proj,
                                 [unit for _, unit in entries], device=args.device,
                                 max_gpu_gb=args.max_gpu_gb, unit_chunk=args.write_chunk)
        for col, (source_i, _unit) in enumerate(entries):
            writes[source_i] = _decode_top(tok, ids[col], decode_cache)
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, 'source.npz'), word_activations=acts,
                        line_indices=np.asarray([row['line_index'] for row in docs], dtype=np.int64))
    source_neurons = []
    for source_i, row in enumerate(specs):
        source_neurons.append({**row, 'source_index': source_i, 'writes_top30': writes[source_i]})
    out = _header(args.source_model, args.source_revision,
                  {'corpus': args.corpus, 'identity_group': args.group_json,
                   'PREDICTIONS.md': os.path.join(os.path.dirname(__file__), 'PREDICTIONS.md'),
                   'neuron_report.py': nr.__file__, 'part20.py': p20.__file__, 'part20f.py': part20f.__file__})
    out.update({'document_selection': 'first 1,000 JSONL rows with at least 256 whitespace-separated words, first 256 words joined by single spaces',
                'documents': docs, 'source_neurons': source_neurons,
                'activation_shape': list(acts.shape), 'activation_definition': 'maximum down_proj input activation over tokens whose nonempty offset span overlaps each whitespace-separated word'})
    nr._dump(out, os.path.join(args.out, 'source.json'))
    print(f'  source saved 51 x {len(docs)} x {SOURCE_WORDS} word activations', flush=True)


class CorrelationAccumulator:
    """Batchwise Pearson sums: float32 batch math on the selected device, float64 accumulation on CPU."""
    def __init__(self, n_sources: int, n_targets: int):
        self.n_sources = n_sources
        self.n_targets = n_targets
        self.n = 0
        self.sum_x = np.zeros(n_sources, dtype=np.float64)
        self.sum_x2 = np.zeros(n_sources, dtype=np.float64)
        self.sum_y = np.zeros(n_targets, dtype=np.float64)
        self.sum_y2 = np.zeros(n_targets, dtype=np.float64)
        self.sum_xy = np.zeros((n_sources, n_targets), dtype=np.float64)

    def add_tensors(self, x: torch.Tensor, y: torch.Tensor):
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[1] != self.n_sources or y.shape[1] != self.n_targets:
            raise ValueError('correlation batch must be [positions, sources] and [positions, targets]')
        x = x.float()
        y = y.float()
        n = int(x.shape[0])
        if not n:
            return
        batch_sum_x = x.sum(dim=0)
        batch_sum_x2 = (x * x).sum(dim=0)
        batch_sum_y = y.sum(dim=0)
        batch_sum_y2 = (y * y).sum(dim=0)
        batch_sum_xy = x.T @ y
        self.n += n
        self.sum_x += batch_sum_x.detach().cpu().numpy().astype(np.float64)
        self.sum_x2 += batch_sum_x2.detach().cpu().numpy().astype(np.float64)
        self.sum_y += batch_sum_y.detach().cpu().numpy().astype(np.float64)
        self.sum_y2 += batch_sum_y2.detach().cpu().numpy().astype(np.float64)
        self.sum_xy += batch_sum_xy.detach().cpu().numpy().astype(np.float64)

    def add(self, x, y, device: str = 'cpu'):
        self.add_tensors(torch.as_tensor(x, dtype=torch.float32, device=device),
                         torch.as_tensor(y, dtype=torch.float32, device=device))

    def correlations(self) -> np.ndarray:
        if not self.n:
            raise ValueError('cannot correlate an empty collection')
        n = float(self.n)
        cov = self.sum_xy - self.sum_x[:, None] * self.sum_y[None, :] / n
        var_x = np.maximum(0.0, self.sum_x2 - self.sum_x * self.sum_x / n)
        var_y = np.maximum(0.0, self.sum_y2 - self.sum_y * self.sum_y / n)
        denom = np.sqrt(var_x[:, None] * var_y[None, :])
        result = np.zeros_like(cov)
        np.divide(cov, denom, out=result, where=denom > 0)
        return np.clip(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)


def misaligned_source_indices(target_doc_indices, n_docs: int) -> np.ndarray:
    """Pair target document d with source document d+1, wrapping the final document to zero."""
    if n_docs < 1:
        raise ValueError('n_docs must be positive')
    return (np.asarray(target_doc_indices, dtype=np.int64) + 1) % n_docs


def _pool_word_tensor(token_values: torch.Tensor, tokenized: list[dict]) -> torch.Tensor:
    """Pool [batch, token, unit] to [batch, word, unit] by offset-overlap maxima on-device."""
    if token_values.ndim != 3 or token_values.shape[0] != len(tokenized):
        raise ValueError('token values must have shape [batch, tokens, units]')
    batch_size, _, n_units = token_values.shape
    n_words = len(tokenized[0]['word_tokens'])
    if any(len(item['word_tokens']) != n_words for item in tokenized):
        raise ValueError('word pooling expects equal word counts in a document batch')
    pooled = torch.full((batch_size, n_words, n_units), -torch.inf,
                        dtype=torch.float32, device=token_values.device)
    source = token_values.float()
    try:
        for row_i, item in enumerate(tokenized):
            token_indices, word_indices = [], []
            for word_i, toks in enumerate(item['word_tokens']):
                token_indices.extend(toks)
                word_indices.extend([word_i] * len(toks))
            ti = torch.tensor(token_indices, dtype=torch.long, device=token_values.device)
            wi = torch.tensor(word_indices, dtype=torch.long, device=token_values.device)
            selected = source[row_i].index_select(0, ti)
            pooled[row_i].scatter_reduce_(0, wi[:, None].expand(-1, n_units), selected,
                                          reduce='amax', include_self=True)
    except (NotImplementedError, RuntimeError):
        # Some accelerator builds lack scatter_reduce(a max) for this dtype. This slower path
        # preserves the same offsets and values while keeping the main path on-device.
        device = token_values.device
        cpu = token_values.detach().float().cpu().numpy()
        arrays = [word_activations(item['offsets'], item['spans'], cpu[row_i, :len(item['ids'])])
                  for row_i, item in enumerate(tokenized)]
        return torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=device)
    if not torch.isfinite(pooled).all():
        raise ValueError('word pooling left at least one word without a token')
    return pooled


def _best_rows(values: np.ndarray, layer_i: int, top_n: int, overlap: np.ndarray) -> list[dict]:
    n = len(values)
    if n <= top_n:
        indices = np.arange(n)
    else:
        indices = np.argpartition(-values, top_n - 1)[:top_n]
    indices = sorted((int(i) for i in indices), key=lambda i: (-float(values[i]), i))
    return [{'layer': int(layer_i), 'unit': unit, 'r': float(values[unit]),
             'writes_overlap': int(overlap[unit])} for unit in indices]


def _row_key(row):
    return (-float(row['r']), int(row['layer']), int(row['unit']))


def _attach_null(row: dict, null_r: float, null_bar: float) -> dict:
    return {**row, 'null_r': float(null_r), 'null_bar': float(null_bar),
            'beats_null': bool(float(row['r']) > float(null_bar))}


def _target_files(out_dir: str, short: str):
    return os.path.join(out_dir, f'target-{short}.json'), os.path.join(out_dir, f'target-{short}.npz')


def run_target(args, model, tok, model_id: str, revision: str, short: str):
    source_json, source_acts = _load_source(args.out)
    docs = source_json['documents']
    specs = source_json['source_neurons']
    if len(docs) != SOURCE_DOCS:
        raise ValueError('source document count does not match Part 26 definition')
    _, layers, norm = nr.find_text_layers(model)
    widths = [int(layer.mlp.down_proj.in_features) for layer in layers]
    offsets = np.cumsum([0] + widths[:-1]).astype(np.int64)
    total_units = sum(widths)
    tokenized = [_tokenize_document(tok, row['text']) for row in docs]
    source_sets = [set(row['writes_top30']) for row in specs]
    decode_cache: dict[int, str] = {}
    all_overlaps = np.zeros((len(specs), total_units), dtype=np.uint8)
    all_r = np.empty((len(specs), total_units), dtype=np.float64)
    maxima = np.full(total_units, -np.inf, dtype=np.float32)
    top50: list[list[dict]] = [[] for _ in specs]
    write_candidates: list[list[dict]] = [[] for _ in specs]
    null_bars = np.full(len(specs), -np.inf, dtype=np.float64)
    accum_aligned = [CorrelationAccumulator(len(specs), width) for width in widths]
    accum_null = [CorrelationAccumulator(len(specs), width) for width in widths]

    for layer_i, layer in enumerate(layers):
        ids = logit_lens_top_ids(tok, model, norm, layer.mlp.down_proj, list(range(widths[layer_i])),
                                 device=args.device, max_gpu_gb=args.max_gpu_gb,
                                 unit_chunk=args.write_chunk)
        overlap = _writes_overlaps(tok, ids, source_sets, decode_cache)
        all_overlaps[:, offsets[layer_i]:offsets[layer_i] + widths[layer_i]] = overlap
        for source_i in range(len(specs)):
            candidate_units = np.flatnonzero(overlap[source_i] >= WRITE_MATCH_MIN)
            for unit in candidate_units:
                write_candidates[source_i].append({'layer': layer_i, 'unit': int(unit),
                                                   'writes_overlap': int(overlap[source_i, unit])})

    current: dict = {}
    hooks = []
    try:
        for layer_i, layer in enumerate(layers):
            def on_layer(values, layer_i=layer_i):
                if not current:
                    raise RuntimeError('target activation hook ran without an active document batch')
                pooled = _pool_word_tensor(values, current['tokenized'])
                source_aligned = current['source_aligned']
                source_null = current['source_null']
                target_values = pooled.reshape(-1, pooled.shape[-1])
                accum_aligned[layer_i].add_tensors(source_aligned, target_values)
                accum_null[layer_i].add_tensors(source_null, target_values)
                layer_max = pooled.amax(dim=(0, 1)).detach().cpu().numpy().astype(np.float32)
                a, b = int(offsets[layer_i]), int(offsets[layer_i] + widths[layer_i])
                maxima[a:b] = np.maximum(maxima[a:b], layer_max)
                current.setdefault('seen_layers', set()).add(layer_i)
            hooks.append(_CallbackHook(layer.mlp.down_proj, on_layer))

        started = time.time()
        next_progress = 100
        for start in range(0, SOURCE_DOCS, args.batch_size):
            end = min(SOURCE_DOCS, start + args.batch_size)
            batch = tokenized[start:end]
            doc_indices = np.arange(start, end, dtype=np.int64)
            shifted_indices = misaligned_source_indices(doc_indices, SOURCE_DOCS)
            aligned = source_acts[:, doc_indices, :].transpose(1, 2, 0).reshape(-1, len(specs)).copy()
            null = source_acts[:, shifted_indices, :].transpose(1, 2, 0).reshape(-1, len(specs)).copy()
            current = {'tokenized': batch, 'source_aligned': torch.as_tensor(aligned, dtype=torch.float32, device=args.device),
                       'source_null': torch.as_tensor(null, dtype=torch.float32, device=args.device),
                       'seen_layers': set()}
            ids, mask = _padded_inputs(batch, tok, args.device)
            _forward_decoder(model, ids, mask)
            if len(current['seen_layers']) != len(layers):
                raise RuntimeError(f'captured {len(current["seen_layers"])} of {len(layers)} target layers')
            current = {}
            nr.spill_guard(args.device, args.max_gpu_gb)
            while end >= next_progress:
                print(f'  {short} target {next_progress}/{SOURCE_DOCS} docs ({time.time() - started:.0f}s)', flush=True)
                next_progress += 100

    finally:
        for hook in hooks:
            hook.remove()

    for layer_i, width in enumerate(widths):
        a, b = int(offsets[layer_i]), int(offsets[layer_i] + width)
        aligned = accum_aligned[layer_i].correlations()
        null = accum_null[layer_i].correlations()
        all_r[:, a:b] = aligned
        null_bars = np.maximum(null_bars, null.max(axis=1))
        layer_overlaps = all_overlaps[:, a:b]
        for source_i in range(len(specs)):
            rows = _best_rows(aligned[source_i], layer_i, 50, layer_overlaps[source_i])
            for row in rows:
                row['null_r'] = float(null[source_i, row['unit']])
            top50[source_i] = sorted(top50[source_i] + rows, key=_row_key)[:50]
            for candidate in write_candidates[source_i]:
                if candidate['layer'] == layer_i:
                    unit = candidate['unit']
                    candidate['r'] = float(aligned[source_i, unit])
                    candidate['null_r'] = float(null[source_i, unit])

    source_rows = []
    for source_i, spec in enumerate(specs):
        source_null_bar = float(null_bars[source_i])
        ranked = top50[source_i]
        top20_rows = [_attach_null(row, row['null_r'], source_null_bar) for row in ranked[:20]]
        writes_rows = []
        for row in sorted(write_candidates[source_i], key=_row_key):
            writes_rows.append(_attach_null(row, row['null_r'], source_null_bar))
        beat_count = int(np.count_nonzero(all_r[source_i] > source_null_bar))
        source_rows.append({'source_neuron': spec, 'null_bar': source_null_bar,
                            'target_neurons_beating_null': beat_count,
                            'top20_by_aligned_r': top20_rows,
                            'writes_candidates': writes_rows})

    n2742_top = top50[0]
    best_beating = [row for row in n2742_top if row['r'] > null_bars[0]]
    fallback = not best_beating
    selected = best_beating[:5] if best_beating else n2742_top[:5]
    reasons: dict[tuple[int, int], list[str]] = defaultdict(list)
    for row in selected:
        reasons[(row['layer'], row['unit'])].append('top5_fallback_no_beats' if fallback else 'top5_beating_null')
    for row in n2742_top:
        if row['writes_overlap'] >= WRITE_MATCH_MIN:
            reasons[(row['layer'], row['unit'])].append('writes_candidate_top50')
    twin_candidates = []
    for (layer_i, unit), why in reasons.items():
        row = next(item for item in n2742_top if item['layer'] == layer_i and item['unit'] == unit)
        twin_candidates.append({**_attach_null(row, row['null_r'], float(null_bars[0])),
                                'selection_reasons': why,
                                'largest_activation': float(maxima[int(offsets[layer_i]) + unit])})
    twin_candidates.sort(key=_row_key)

    output_json, output_npz = _target_files(args.out, short)
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(output_npz, max_activation=maxima,
                        layer_widths=np.asarray(widths, dtype=np.int32),
                        layer_offsets=offsets.astype(np.int64))
    source_path = os.path.join(args.out, 'source.json')
    source_npz_path = os.path.join(args.out, 'source.npz')
    out = _header(model_id, revision, {'source.json': source_path, 'source.npz': source_npz_path,
                                      'neuron_report.py': nr.__file__, 'part20.py': p20.__file__})
    out.update({'short_name': short, 'layers_module': nr.find_text_layers(model)[0],
                'n_layers': len(layers), 'layer_widths': widths,
                'correlation_definition': 'word-position Pearson r over all 1,000 x 256 positions using batchwise float32 sums on device and float64 CPU accumulators',
                'null_definition': 'target document d paired with source document d+1, wrapping the last document to source document zero',
                'source_results': source_rows,
                'twin_candidates_n2742': twin_candidates,
                'largest_activation_npz': os.path.basename(output_npz),
                'largest_activation_definition': 'maximum word activation over the selected 1,000 documents, excluding tokens with empty offsets'})
    nr._dump(out, output_json)
    print(f'  {short} target saved; N2742 twins={len(twin_candidates)}, beats-null={source_rows[0]["target_neurons_beating_null"]}', flush=True)
    return out


def _part20f_filler(corpus: str, tok):
    # part20f.py sets the Part 20d screen before calling Part 20's selection helper.
    p20.PATTERNS = p20d.SCREEN
    return p20.select_filler(corpus, tok, 4000)


def battery_span_positions(tok, text: str, filler_length: int) -> list[int]:
    """Mirror part20f.py: score the full suffix after filler, including its fixed LEAD sentence."""
    return p20.token_span(tok, text, (filler_length + 2, len(text)))


def score_battery_candidates(tok, model, layers, candidates: list[dict], corpus: str,
                              battery_path: str = BATTERY_PATH, device: str = 'xpu',
                              max_gpu_gb: float = 14.5) -> dict:
    battery = json.load(open(battery_path, encoding='utf-8'))['sets']
    filler, doc_indices, filler_count = _part20f_filler(corpus, tok)
    units_by_layer: dict[int, list[int]] = defaultdict(list)
    for row in candidates:
        units_by_layer[int(row['layer'])].append(int(row['unit']))
    for layer_i in units_by_layer:
        units_by_layer[layer_i] = sorted(set(units_by_layer[layer_i]))
    hooks = {}
    rows = []
    values_by_candidate: dict[tuple[int, int], dict[str, list[float]]] = {
        (int(row['layer']), int(row['unit'])): {name: [] for name in battery} for row in candidates}
    try:
        for layer_i, units in units_by_layer.items():
            hooks[layer_i] = nr.NeuronHook(layers[layer_i].mlp.down_proj, units)
        for set_name, phrases in battery.items():
            for phrase_i, phrase in enumerate(phrases):
                text = filler + '\n\n' + LEAD + phrase + '.'
                tokenized = p20._encoded(tok, text, device)
                positions = battery_span_positions(tok, text, len(filler))
                if not positions:
                    raise ValueError(f'Part 20f battery suffix has no tokens: {set_name}[{phrase_i}]')
                for hook in hooks.values():
                    hook.captured.clear()
                with torch.no_grad():
                    _decoder(model)(input_ids=tokenized['input_ids'],
                                    attention_mask=tokenized.get('attention_mask', torch.ones_like(tokenized['input_ids'])),
                                    use_cache=False)
                per_candidate = {}
                for layer_i, units in units_by_layer.items():
                    if len(hooks[layer_i].captured) != 1:
                        raise RuntimeError(f'battery hook on layer {layer_i} did not capture once')
                    activations = hooks[layer_i].captured[0][0].numpy()
                    for col, unit in enumerate(units):
                        value = float(np.max(activations[positions, col]))
                        values_by_candidate[(layer_i, unit)][set_name].append(value)
                        per_candidate[f'{layer_i}:{unit}'] = value
                rows.append({'set': set_name, 'index': phrase_i, 'phrase': phrase,
                             'suffix_max_by_candidate': per_candidate})
                nr.spill_guard(device, max_gpu_gb)
    finally:
        for hook in hooks.values():
            hook.remove()
    summaries = {}
    for pair, groups in values_by_candidate.items():
        summaries[f'{pair[0]}:{pair[1]}'] = {
            name: float(np.mean(v)) if v else None for name, v in groups.items()
        }
    return {'filler_doc_indices': doc_indices, 'filler_token_count': filler_count,
            'filler_sha256': hashlib.sha256(filler.encode('utf-8')).hexdigest(),
            'battery_sha256': _sha(battery_path), 'rows': rows, 'mean_max_by_candidate': summaries,
            'n_by_set': {name: len(phrases) for name, phrases in battery.items()},
            'scoring': 'part20f.py scoring: max activation over tokens in the complete post-filler suffix, including the fixed lead sentence and phrase'}


def _candidate_profile(means: dict) -> dict:
    c3 = means.get('C3')
    c3_bare = means.get('C3-bare')
    e = means.get('E')
    u = means.get('U')
    c4 = means.get('C4')
    d = means.get('D')
    checks = {
        'C3_at_least_3x_C3_bare': c3 is not None and c3_bare is not None and c3 >= 3 * c3_bare,
        'E_at_least_3x_U': e is not None and u is not None and e >= 3 * u,
        'C4_at_most_D_plus_half_C3': c4 is not None and d is not None and c3 is not None and c4 <= d + 0.5 * c3,
    }
    return {'checks': checks, 'profile_pass': all(checks.values())}


def run_battery(args, model, tok, model_id: str, revision: str, short: str):
    target_path, target_npz = _target_files(args.out, short)
    target = json.load(open(target_path, encoding='utf-8'))
    candidates = target['twin_candidates_n2742']
    if not candidates:
        raise ValueError('target file has no N2742 twin candidates')
    _, layers, _ = nr.find_text_layers(model)
    scored = score_battery_candidates(tok, model, layers, candidates, args.corpus,
                                      args.battery, args.device, args.max_gpu_gb)
    out = _header(model_id, revision, {'target': target_path, 'target_npz': target_npz,
                                      'corpus': args.corpus, 'battery': args.battery,
                                      'part20f.py': part20f.__file__, 'part20d.py': p20d.__file__})
    out.update({'short_name': short, 'candidates': [],
                'filler_doc_indices': scored['filler_doc_indices'],
                'filler_token_count': scored['filler_token_count'],
                'filler_sha256': scored['filler_sha256'], 'battery_sha256': scored['battery_sha256'],
                'scoring': scored['scoring'], 'n_by_set': scored['n_by_set'], 'rows': scored['rows']})
    for candidate in candidates:
        key = f"{candidate['layer']}:{candidate['unit']}"
        means = scored['mean_max_by_candidate'][key]
        out['candidates'].append({**candidate, 'mean_max_by_set': means,
                                  'largest_activation': candidate['largest_activation'],
                                  'profile': _candidate_profile(means)})
    path = os.path.join(args.out, f'battery-{short}.json')
    nr._dump(out, path)
    print(f'  {short} battery saved for {len(candidates)} candidates', flush=True)
    return out


def _condition_seed(seed: int, condition: str) -> int:
    payload = f'{seed}|Who made you?|{condition}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little')


def _eos_ids(tok, model) -> list[int]:
    values = [tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>'),
              tok.convert_tokens_to_ids('<|endoftext|>'), getattr(model.generation_config, 'eos_token_id', None)]
    return sorted({int(i) for i in values if isinstance(i, int) and i >= 0})


def _generate_condition(tok, model, down_proj, prompt_ids, condition: str, unit: int | None,
                        value: float | None, seed: int, samples: int, max_new: int,
                        device: str, max_gpu_gb: float) -> dict:
    hook = nr.NeuronHook(down_proj)
    try:
        if unit is not None:
            hook.clamp = (torch.tensor([unit]), torch.tensor([float(value)]))
        torch.manual_seed(seed)
        with torch.no_grad():
            generated = model.generate(**prompt_ids, max_new_tokens=max_new, pad_token_id=tok.pad_token_id,
                                       eos_token_id=_eos_ids(tok, model) or None,
                                       do_sample=True, temperature=0.7, top_p=0.95,
                                       num_return_sequences=samples)
        prefix_len = prompt_ids['input_ids'].shape[1]
        texts = [tok.decode(seq[prefix_len:], skip_special_tokens=True).strip() for seq in generated]
        scores = [nr.score_text(text) for text in texts]
        nr.spill_guard(device, max_gpu_gb)
        return {'condition': condition, 'unit': unit, 'set_value': value, 'seed': seed,
                'n': len(texts), 'answers': texts,
                'summary': {'alien': sum(row['alien'] for row in scores),
                            'names_qwen': sum(row['names_qwen'] for row in scores)}}
    finally:
        hook.clamp = None
        hook.remove()


def run_clamp(args, model, tok, model_id: str, revision: str, short: str):
    target_path, target_npz_path = _target_files(args.out, short)
    target = json.load(open(target_path, encoding='utf-8'))
    candidates = target['twin_candidates_n2742']
    if not candidates:
        raise ValueError('target file has no N2742 twin candidates')
    beating = [row for row in candidates if row['beats_null']]
    best = max(beating or candidates, key=lambda row: (row['r'], -row['layer'], -row['unit']))
    if getattr(args, 'clamp_unit', None):
        # 2026-10-21 (26.6): clamp a named candidate instead of the best-r one; it must be one of the twin candidates.
        want = tuple(int(x) for x in args.clamp_unit.split(':'))
        named = [row for row in candidates if (int(row['layer']), int(row['unit'])) == want]
        if not named:
            raise ValueError(f'--clamp-unit {args.clamp_unit} is not a twin candidate in {target_path}')
        best = named[0]
    _, layers, _ = nr.find_text_layers(model)
    npz = np.load(target_npz_path)
    widths = [int(x) for x in npz['layer_widths']]
    offsets = [int(x) for x in npz['layer_offsets']]
    maxima = np.asarray(npz['max_activation'], dtype=np.float32)
    layer_i, unit = int(best['layer']), int(best['unit'])
    target_maximum = float(maxima[offsets[layer_i] + unit])
    controls = nr.control_units(widths[layer_i], unit, 20, args.seed)
    prompt = nr.build_prompt(tok, 'Who made you?', False)
    prompt_ids = tok(prompt, return_tensors='pt', add_special_tokens=False).to(args.device)
    baseline_seed = _condition_seed(args.seed, 'unclamped')
    twin_seed = _condition_seed(args.seed, 'twin')
    rows = [_generate_condition(tok, model, layers[layer_i].mlp.down_proj, prompt_ids,
                                'unclamped', None, None, baseline_seed, args.samples,
                                args.max_new, args.device, args.max_gpu_gb)]
    rows.append(_generate_condition(tok, model, layers[layer_i].mlp.down_proj, prompt_ids,
                                    'twin', unit, 2.0 * target_maximum, twin_seed,
                                    args.samples, args.max_new, args.device, args.max_gpu_gb))
    for control_i in controls:
        control_max = float(maxima[offsets[layer_i] + control_i])
        control_seed = _condition_seed(args.seed, f'control-{control_i}')
        rows.append(_generate_condition(tok, model, layers[layer_i].mlp.down_proj, prompt_ids,
                                        f'control-{control_i}', control_i, 2.0 * control_max,
                                        control_seed, args.samples, args.max_new,
                                        args.device, args.max_gpu_gb))
    out = _header(model_id, revision, {'target': target_path, 'target_npz': target_npz_path,
                                      'neuron_report.py': nr.__file__, 'part24.py': os.path.join(os.path.dirname(__file__), 'part24.py')})
    out.update({'short_name': short, 'question': 'Who made you?', 'prompt': prompt,
                'system_message': None, 'sampling': {'do_sample': True, 'temperature': 0.7, 'top_p': 0.95,
                                                     'samples_per_condition': args.samples, 'max_new_tokens': args.max_new},
                'seed_policy': 'one deterministic SHA-derived seed for each condition',
                'best_twin': {**best, 'largest_activation': target_maximum,
                              'clamp_value': 2.0 * target_maximum,
                              'fallback_no_beating_twin': not bool(beating)},
                'random_controls': {'seed': args.seed, 'layer': layer_i,
                                    'units': controls, 'count': len(controls)},
                'conditions': rows})
    suffix = f"-L{best['layer']}u{best['unit']}" if getattr(args, 'clamp_unit', None) else ''
    path = os.path.join(args.out, f'clamp-{short}{suffix}.json')
    nr._dump(out, path)
    print(f'  {short} clamp saved for best twin and {len(controls)} controls', flush=True)
    return out


def _canonical_target(value: str, revision: str | None = None) -> tuple[str, str, str]:
    if value in TARGETS:
        model_id, pinned = TARGETS[value]
        return value, model_id, revision or pinned
    for short, (model_id, pinned) in TARGETS.items():
        if value == model_id:
            return short, model_id, revision or pinned
    raise ValueError(f'unknown target {value!r}; expected one of {", ".join(TARGETS)} or a pinned model id')


def _load_target(short: str, model_id: str, revision: str, args):
    tok, model = _load_model_offline(model_id, revision, args.device)
    return tok, model


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', nargs='?', choices=['source', 'target', 'battery', 'clamp'])
    ap.add_argument('--steps', default=None, help='comma-separated source,target,battery,clamp sequence')
    ap.add_argument('--targets', default='olmo2-1b,smollm3-3b')
    ap.add_argument('--model', default=None, help='target model id or short name for a single target step')
    ap.add_argument('--revision', default=None)
    ap.add_argument('--source-model', default=SOURCE_MODEL)
    ap.add_argument('--source-revision', default=SOURCE_REVISION)
    ap.add_argument('--group-json', default=SOURCE_GROUP)
    ap.add_argument('--corpus', default=None)
    ap.add_argument('--battery', default=BATTERY_PATH)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='xpu')
    ap.add_argument('--max-gpu-gb', type=float, default=14.5)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--write-chunk', type=int, default=256)
    ap.add_argument('--samples', type=int, default=8)
    ap.add_argument('--max-new', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--clamp-unit', default=None, help='LAYER:UNIT of a twin candidate to clamp instead of the best-r one')
    return ap


def _run_target_steps(args, steps: list[str]):
    if any(step in steps for step in ('source', 'battery')) and not args.corpus:
        raise SystemExit('--corpus is required for source and battery steps')
    if 'source' in steps:
        args.source_model = args.source_model or SOURCE_MODEL
        args.source_revision = args.source_revision or SOURCE_REVISION
        run_source(args)
    need_source = any(step in steps for step in ('target', 'battery', 'clamp'))
    if need_source and not os.path.isfile(os.path.join(args.out, 'source.json')):
        raise SystemExit('source.json is missing; run source or include source in --steps')
    if not any(step in steps for step in ('target', 'battery', 'clamp')):
        return
    selected_targets = [x.strip() for x in args.targets.split(',') if x.strip()]
    for value in selected_targets:
        short, model_id, revision = _canonical_target(value)
        tok, model = _load_target(short, model_id, revision, args)
        try:
            if 'target' in steps:
                run_target(args, model, tok, model_id, revision, short)
            if 'battery' in steps:
                if not args.corpus:
                    raise SystemExit('--corpus is required for battery')
                run_battery(args, model, tok, model_id, revision, short)
            if 'clamp' in steps:
                run_clamp(args, model, tok, model_id, revision, short)
        finally:
            del model, tok
            import gc
            gc.collect()
            if args.device == 'xpu':
                torch.xpu.empty_cache()


def main(argv=None):
    args = parser().parse_args(argv)
    if args.batch_size < 1 or args.write_chunk < 1 or args.samples < 1 or args.max_new < 1:
        raise SystemExit('--batch-size, --write-chunk, --samples and --max-new must be positive')
    if args.steps:
        if args.command:
            raise SystemExit('use either a command or --steps, not both')
        steps = [step.strip() for step in args.steps.split(',') if step.strip()]
        if not steps or any(step not in {'source', 'target', 'battery', 'clamp'} for step in steps):
            raise SystemExit('--steps accepts source,target,battery,clamp')
        if steps != [step for step in ('source', 'target', 'battery', 'clamp') if step in steps]:
            raise SystemExit('--steps must follow source,target,battery,clamp order')
        _run_target_steps(args, steps)
        return
    if not args.command:
        raise SystemExit('provide a command or --steps')
    os.makedirs(args.out, exist_ok=True)
    if args.command == 'source':
        args.source_model = args.source_model or SOURCE_MODEL
        args.source_revision = args.source_revision or SOURCE_REVISION
        if not args.corpus:
            raise SystemExit('--corpus is required for source')
        run_source(args)
        return
    if not args.model:
        raise SystemExit('--model is required for target, battery and clamp')
    short, model_id, revision = _canonical_target(args.model, args.revision)
    if args.command == 'battery' and not args.corpus:
        raise SystemExit('--corpus is required for battery')
    tok, model = _load_target(short, model_id, revision, args)
    try:
        if args.command == 'target':
            run_target(args, model, tok, model_id, revision, short)
        elif args.command == 'battery':
            run_battery(args, model, tok, model_id, revision, short)
        elif args.command == 'clamp':
            run_clamp(args, model, tok, model_id, revision, short)
    finally:
        del model, tok
        import gc
        gc.collect()
        if args.device == 'xpu':
            torch.xpu.empty_cache()


if __name__ == '__main__':
    main()
