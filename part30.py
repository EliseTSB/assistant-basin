#!/usr/bin/env python3
"""Controlled teacher swap on Qwen2.5-0.5B-Instruct (assistant-basin Part 30)."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import secrets
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import neuron_report as nr


BASE_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
DEFAULT_OUT = 'results/part30'
PREDICTIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PREDICTIONS.md')
TEACHERS = ('r1', 'gemini', 'control')
PROBE_TEACHERS = ('r1', 'gemini')
LORA_MODULES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj')
RANK = 16
ALPHA = 32
LEARNING_RATE = 2e-4
EPOCHS = 2
EFFECTIVE_BATCH = 8
SEED = 0
TRAIN_MAX_NEW = 1024
GENERATE_MAX_NEW = 600
PROBE_MAX_TARGET = 512


def file_sha(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(value: Any, path: str | os.PathLike[str]) -> None:
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f'{path}.{os.getpid()}.{secrets.token_hex(4)}.tmp'
    with open(tmp, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    os.replace(tmp, path)


def _path_shas(paths: dict[str, str | None]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        if path is None:
            continue
        if not os.path.isfile(path):
            raise FileNotFoundError(f'input {name} does not exist: {path}')
        output[name] = {'path': os.path.abspath(path), 'sha256': file_sha(path)}
    return output


def make_header(step: str, args, input_paths: dict[str, str | None], settings: dict[str, Any],
                revision: str | None = None) -> dict[str, Any]:
    return {
        'tool': 'part30.py (assistant-basin, Part 30)',
        'tool_sha256': file_sha(__file__),
        'step': step,
        'model': args.base_model,
        'revision': revision if revision is not None else args.revision,
        'input_shas': _path_shas(input_paths),
        'settings': settings,
        'written': time.strftime('%Y-%m-%d %H:%M:%S'),
    }


def read_pairs(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'invalid JSON in {path}:{line_no}: {exc}') from exc
            if not isinstance(row, dict) or 'question' not in row:
                raise ValueError(f'{path}:{line_no} must be an object with a question')
            row = dict(row)
            row.setdefault('index', len(rows))
            row['index'] = int(row['index'])
            row['question'] = str(row['question'])
            rows.append(row)
    ids = [row['index'] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('pair input has duplicate index values')
    return rows


def split_indices(count: int, seed: int = SEED, train_count: int | None = None) -> tuple[list[int], list[int]]:
    """Return a reproducible NumPy-seed-0 partition, preserving permutation order."""
    if count < 2:
        raise ValueError('at least two pair rows are needed for a train/held-out split')
    if train_count is None:
        train_count = int(count * 0.9)
    if not 0 < train_count < count:
        raise ValueError('train_count must leave at least one held-out row')
    permutation = np.random.default_rng(seed).permutation(count).tolist()
    return [int(i) for i in permutation[:train_count]], [int(i) for i in permutation[train_count:]]


def _split_path(out_dir: str) -> str:
    return os.path.join(out_dir, 'split.json')


def _load_split(out_dir: str, pairs: list[dict[str, Any]], pairs_path: str) -> dict[str, Any]:
    path = _split_path(out_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'missing split file {path}; run --steps split first')
    with open(path, encoding='utf-8') as stream:
        split = json.load(stream)
    expected_sha = file_sha(pairs_path)
    got_sha = split.get('header', {}).get('input_shas', {}).get('pairs', {}).get('sha256')
    if expected_sha != got_sha:
        raise ValueError('split.json was made from a different pairs file')
    if int(split.get('row_count', -1)) != len(pairs):
        raise ValueError('split.json row count does not match pairs input')
    return split


def cmd_split(args) -> None:
    rows = read_pairs(args.pairs)
    if len(rows) != 1000:
        raise ValueError(f'Part 30 expects 1,000 paired questions; found {len(rows)}')
    train_pos, heldout_pos = split_indices(len(rows), seed=SEED, train_count=900)
    train_ids = [rows[i]['index'] for i in train_pos]
    heldout_ids = [rows[i]['index'] for i in heldout_pos]
    header = make_header('split', args, {'pairs': args.pairs, 'PREDICTIONS.md': PREDICTIONS_PATH},
                         {'seed': SEED, 'train_rows': 900, 'heldout_rows': 100,
                          'algorithm': 'numpy.default_rng(0).permutation; first 900 positions train'})
    dump_json({'header': header, 'row_count': len(rows), 'train_indices': train_ids,
               'heldout_indices': heldout_ids}, _split_path(args.out))
    print(f"split: {len(train_ids)} train, {len(heldout_ids)} held out -> {_split_path(args.out)}", flush=True)


def _render_chat(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    fn = tokenizer.apply_chat_template
    kwargs = {'tokenize': False, 'add_generation_prompt': add_generation_prompt}
    try:
        return fn(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return fn(messages, **kwargs)


def _encode_text(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded['input_ids']
    if torch.is_tensor(values):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def truncate_target(target_ids: Iterable[int], max_target: int) -> list[int]:
    if max_target < 1:
        raise ValueError('max_target must be positive')
    return [int(value) for value in list(target_ids)[:max_target]]


def assert_shared_target_budget(budgets: Iterable[int], expected: int) -> None:
    values = [int(value) for value in budgets]
    if any(value != expected for value in values):
        raise ValueError(f'teacher-dependent target budget: got {values}, expected common cap {expected}')


def make_supervised_example(tokenizer, question: str, assistant_text: str,
                            max_target: int) -> dict[str, Any]:
    """Tokenize a user/assistant exchange and mark just assistant continuation tokens for loss."""
    prompt_text = _render_chat(tokenizer, [{'role': 'user', 'content': question}], True)
    full_text = _render_chat(tokenizer, [
        {'role': 'user', 'content': question}, {'role': 'assistant', 'content': assistant_text}], False)
    prompt_ids = _encode_text(tokenizer, prompt_text)
    full_ids = _encode_text(tokenizer, full_text)
    if not prompt_ids or full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError('chat template full conversation does not begin with its generation prompt; cannot mask safely')
    target = truncate_target(full_ids[len(prompt_ids):], max_target)
    if not target:
        raise ValueError('assistant continuation produced no tokens')
    return {'input_ids': prompt_ids + target, 'prompt_len': len(prompt_ids), 'target_len': len(target)}


def labels_for_example(input_ids: Iterable[int], prompt_len: int, padded_to: int | None = None) -> list[int]:
    ids = [int(value) for value in input_ids]
    if not 0 <= prompt_len < len(ids):
        raise ValueError('prompt_len must identify a nonempty assistant continuation')
    width = len(ids) if padded_to is None else int(padded_to)
    if width < len(ids):
        raise ValueError('padding width cannot truncate the example')
    return [-100] * prompt_len + ids[prompt_len:] + [-100] * (width - len(ids))


def _assistant_content(row: dict[str, Any], teacher: str, control_answers: dict[int, dict[str, Any]] | None,
                       max_target: int) -> tuple[str, str]:
    if teacher == 'control':
        if control_answers is None or row['index'] not in control_answers:
            raise ValueError(f'no base-control answer for pair index {row["index"]}')
        content = str(control_answers[row['index']]['answer'])
        return content, 'base answer without a think block'
    if teacher == 'r1':
        thinking_key, attempt_key = 'deepseek_thinking', 'deepseek_attempt'
    else:
        thinking_key, attempt_key = 'gemini_thinking', 'gemini_attempt'
    if thinking_key not in row or attempt_key not in row:
        raise ValueError(f'pair index {row["index"]} is missing {thinking_key} or {attempt_key}')
    content = f'<think>\n{row[thinking_key]}\n</think>\n\n{row[attempt_key]}'
    return content, 'thinking and attempt in explicit think block'


def _control_jsonl_path(out_dir: str) -> str:
    return os.path.join(out_dir, 'control.jsonl')


def _read_control_jsonl(path: str) -> tuple[dict[str, Any] | None, dict[int, dict[str, Any]]]:
    meta = None
    records: dict[int, dict[str, Any]] = {}
    if not os.path.isfile(path):
        return meta, records
    with open(path, 'r+b') as stream:
        valid_end = 0
        for line_no, raw in enumerate(stream, 1):
            if not raw.endswith(b'\n'):
                # A process may have stopped in the middle of its last append; ignore that tail.
                stream.truncate(valid_end)
                break
            try:
                row = json.loads(raw.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f'invalid complete row in {path}:{line_no}: {exc}') from exc
            if line_no == 1 and '_meta' in row:
                meta = row['_meta']
                valid_end = stream.tell()
                continue
            if 'index' not in row:
                raise ValueError(f'control row {line_no} has no index')
            index = int(row['index'])
            if index in records:
                raise ValueError(f'duplicate control answer index {index}')
            records[index] = row
            valid_end = stream.tell()
    return meta, records


def _read_answer_jsonl(path: str) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not os.path.isfile(path):
        return records
    with open(path, 'r+b') as stream:
        valid_end = 0
        for line_no, raw in enumerate(stream, 1):
            if not raw.endswith(b'\n'):
                stream.truncate(valid_end)
                break
            try:
                row = json.loads(raw.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f'invalid complete row in {path}:{line_no}: {exc}') from exc
            if not isinstance(row, dict) or 'index' not in row:
                raise ValueError(f'answer row {line_no} has no index')
            index = int(row['index'])
            if index in records:
                raise ValueError(f'duplicate answer index {index}')
            records[index] = row
            valid_end = stream.tell()
    return records


def _append_jsonl(path: str, row: dict[str, Any]) -> None:
    with open(path, 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _load_model(args):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    tokenizer, model = nr.load_model(args.base_model, args.revision, args.device)
    revision = args.revision or getattr(model.config, '_commit_hash', None)
    return tokenizer, model, revision


def _seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def _empty_xpu_cache() -> None:
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.empty_cache()


def _eos_ids(tokenizer, model) -> list[int]:
    values = [getattr(tokenizer, 'eos_token_id', None), getattr(tokenizer, 'im_end_id', None)]
    generation = getattr(model, 'generation_config', None)
    values.append(getattr(generation, 'eos_token_id', None))
    output: set[int] = set()
    for value in values:
        if isinstance(value, (list, tuple, set)):
            output.update(int(item) for item in value if item is not None)
        elif isinstance(value, int) and value >= 0:
            output.add(value)
    return sorted(output)


def _generation_prompt_ids(tokenizer, question: str) -> list[int]:
    prompt_text = _render_chat(tokenizer, [{'role': 'user', 'content': question}], True)
    input_ids = _encode_text(tokenizer, prompt_text)
    if not input_ids:
        raise ValueError('rendered prompt has no tokens')
    return input_ids


def _generate_encoded_batch(tokenizer, model, prompt_ids: list[list[int]], device: str,
                            max_new_tokens: int, do_sample: bool = False,
                            temperature: float | None = None) -> list[dict[str, Any]]:
    """Generate from already-tokenized prompts, padding on the left for decoder-only models."""
    if not prompt_ids:
        return []
    if any(not values for values in prompt_ids):
        raise ValueError('generation prompts must contain at least one token')
    tokenizer.padding_side = 'left'
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    width = max(len(values) for values in prompt_ids)
    ids = torch.full((len(prompt_ids), width), int(pad_id), dtype=torch.long)
    attention = torch.zeros((len(prompt_ids), width), dtype=torch.long)
    for row_i, values in enumerate(prompt_ids):
        ids[row_i, width - len(values):] = torch.tensor(values, dtype=torch.long)
        attention[row_i, width - len(values):] = 1
    ids = ids.to(device)
    attention = attention.to(device)
    kwargs: dict[str, Any] = {
        'input_ids': ids, 'attention_mask': attention, 'do_sample': do_sample,
        'max_new_tokens': max_new_tokens, 'pad_token_id': int(pad_id),
        'eos_token_id': _eos_ids(tokenizer, model) or None,
    }
    if do_sample:
        kwargs['temperature'] = float(temperature or 0.7)
    started = time.time()
    with torch.no_grad():
        generated = model.generate(**kwargs)[:, width:].detach().cpu().tolist()
    batch_wall_time = time.time() - started
    eos = set(_eos_ids(tokenizer, model))
    row_wall_time = batch_wall_time / len(prompt_ids)
    results = []
    for output_ids in generated:
        real_ids = [int(value) for value in output_ids]
        eos_positions = [position for position, value in enumerate(real_ids) if value in eos]
        if eos_positions:
            real_ids = real_ids[:eos_positions[0] + 1]
            finish = 'stop'
        else:
            finish = 'length'
        text = tokenizer.decode(real_ids, skip_special_tokens=True)
        results.append({'text': text, 'input_tokens': 0, 'output_tokens': len(real_ids),
                        'finish_reason': finish, 'wall_time': row_wall_time})
    return results


def _generate_batch(tokenizer, model, questions: list[str], device: str, max_new_tokens: int,
                    do_sample: bool = False, temperature: float | None = None) -> list[dict[str, Any]]:
    """Tokenize questions and return one generation record per prompt in input order."""
    encoded = [_generation_prompt_ids(tokenizer, question) for question in questions]
    results = _generate_encoded_batch(tokenizer, model, encoded, device, max_new_tokens,
                                      do_sample=do_sample, temperature=temperature)
    for result, input_ids in zip(results, encoded):
        result['input_tokens'] = len(input_ids)
    return results


def _iter_generated_batches(tokenizer, model, rows: list[dict[str, Any]], question_key: str,
                            device: str, max_new_tokens: int, gen_batch: int,
                            do_sample: bool = False, temperature: float | None = None):
    """Yield prompt-length-sorted batches, with each completed batch ordered by row index."""
    prepared = [(row, _generation_prompt_ids(tokenizer, str(row[question_key]))) for row in rows]
    prepared.sort(key=lambda item: (len(item[1]), int(item[0]['index'])))
    for start in range(0, len(prepared), gen_batch):
        batch = prepared[start:start + gen_batch]
        if gen_batch == 1:
            # Preserve the original single-row generation path exactly.
            results = [_generate_one(tokenizer, model, str(batch[0][0][question_key]), device,
                                     max_new_tokens, do_sample, temperature)]
        else:
            results = _generate_encoded_batch(tokenizer, model, [item[1] for item in batch], device,
                                              max_new_tokens, do_sample, temperature)
            for result, (_row, input_ids) in zip(results, batch):
                result['input_tokens'] = len(input_ids)
        completed = sorted(zip((item[0] for item in batch), results),
                           key=lambda item: int(item[0]['index']))
        yield completed


def _generate_one(tokenizer, model, question: str, device: str, max_new_tokens: int,
                  do_sample: bool, temperature: float | None = None) -> dict[str, Any]:
    prompt_text = _render_chat(tokenizer, [{'role': 'user', 'content': question}], True)
    input_ids = _encode_text(tokenizer, prompt_text)
    if not input_ids:
        raise ValueError('rendered prompt has no tokens')
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(ids)
    kwargs: dict[str, Any] = {
        'input_ids': ids, 'attention_mask': attention, 'do_sample': do_sample,
        'max_new_tokens': max_new_tokens,
        'pad_token_id': tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        'eos_token_id': _eos_ids(tokenizer, model) or None,
    }
    if do_sample:
        kwargs['temperature'] = float(temperature or 0.7)
    started = time.time()
    with torch.no_grad():
        output_ids = model.generate(**kwargs)[0, len(input_ids):].detach().cpu().tolist()
    wall_time = time.time() - started
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    eos = set(_eos_ids(tokenizer, model))
    finish = 'stop' if output_ids and output_ids[-1] in eos else 'length'
    return {'text': text, 'input_tokens': len(input_ids), 'output_tokens': len(output_ids),
            'finish_reason': finish, 'wall_time': wall_time}


def cmd_control(args) -> None:
    pairs = read_pairs(args.pairs)
    split = _load_split(args.out, pairs, args.pairs)
    rows_by_id = {row['index']: row for row in pairs}
    train_ids = [int(value) for value in split['train_indices']]
    output_path = _control_jsonl_path(args.out)
    header = make_header('control', args, {'pairs': args.pairs, 'split.json': _split_path(args.out),
                                           'part30.py': __file__},
                         {'decoding': 'greedy', 'max_new_tokens': TRAIN_MAX_NEW,
                          'train_rows': len(train_ids), 'resumable': True,
                          'gen_batch': args.gen_batch,
                          'wall_time': 'per-row share of a batch',
                          'device': args.device, 'max_gpu_gb': args.max_gpu_gb})
    old_meta, completed = _read_control_jsonl(output_path)
    if old_meta is not None:
        if old_meta.get('input_shas') != header.get('input_shas') or old_meta.get('settings') != header.get('settings'):
            raise ValueError('existing control.jsonl has different inputs/settings; move it before restarting')
        if old_meta.get('model') != args.base_model:
            raise ValueError('existing control.jsonl was generated by a different base model')
        if args.revision is not None and old_meta.get('revision') not in (None, args.revision):
            raise ValueError('existing control.jsonl uses a different requested model revision')
    else:
        if completed:
            raise ValueError('existing control.jsonl has rows but no metadata header')
        os.makedirs(args.out, exist_ok=True)
        _append_jsonl(output_path, {'_meta': header})
    unknown = set(completed) - set(train_ids)
    if unknown:
        raise ValueError(f'control.jsonl contains non-training index {min(unknown)}')

    pending = [index for index in train_ids if index not in completed]
    print(f'control: {len(completed)}/{len(train_ids)} complete; generating {len(pending)}', flush=True)
    tokenizer = model = None
    try:
        if pending:
            tokenizer, model, revision = _load_model(args)
            if old_meta is not None and old_meta.get('revision') not in (None, revision):
                raise ValueError(f'control.jsonl uses revision {old_meta.get("revision")}; loaded {revision}')
            header['revision'] = revision
            # Persist the resolved local snapshot revision before generation, retaining resumability metadata.
            if not completed:
                # At this point the file has only its metadata row; rewrite it safely with
                # the resolved local model revision before any answer rows are appended.
                with open(output_path, 'w', encoding='utf-8') as stream:
                    stream.write(json.dumps({'_meta': header}, ensure_ascii=False) + '\n')
            ordinal = 0
            for batch in _iter_generated_batches(tokenizer, model,
                                                [rows_by_id[index] for index in pending],
                                                'question', args.device, TRAIN_MAX_NEW,
                                                args.gen_batch, do_sample=False):
                for row, result in batch:
                    _append_jsonl(output_path, {'index': row['index'], 'question': row['question'],
                                                'answer': result['text'],
                                                'finish_reason': result['finish_reason'],
                                                'prompt_tokens': result['input_tokens'],
                                                'completion_tokens': result['output_tokens'],
                                                'wall_time': result['wall_time']})
                    ordinal += 1
                nr.spill_guard(args.device, args.max_gpu_gb)
                print(f'  control {len(completed) + ordinal}/{len(train_ids)}', flush=True)
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        _empty_xpu_cache()


class LoRALinear(nn.Module):
    """Frozen linear layer plus trainable, zero-initialized rank-r residual."""
    def __init__(self, base: nn.Linear, rank: int = RANK, alpha: int = ALPHA):
        super().__init__()
        if rank < 1:
            raise ValueError('LoRA rank must be positive')
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.empty((self.rank, base.in_features),
                                                device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros((base.out_features, self.rank),
                                                device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        base_output = self.base(values)
        a = self.lora_A.to(dtype=values.dtype)
        b = self.lora_B.to(dtype=values.dtype)
        residual = F.linear(F.linear(values, a), b) * self.scaling
        return base_output + residual


def target_module_names(model) -> list[str]:
    _lname, layers, _norm = nr.find_text_layers(model)
    names_by_id = {id(module): name for name, module in model.named_modules()}
    names: list[str] = []
    for layer_i, layer in enumerate(layers):
        modules = [
            (getattr(getattr(layer, 'self_attn', None), key, None), key)
            for key in ('q_proj', 'k_proj', 'v_proj', 'o_proj')
        ] + [
            (getattr(getattr(layer, 'mlp', None), key, None), key)
            for key in ('up_proj', 'down_proj', 'gate_proj')
        ]
        for module, key in modules:
            if not isinstance(module, nn.Linear):
                raise ValueError(f'layer {layer_i} lacks expected linear projection {key}')
            name = names_by_id.get(id(module))
            if name is None:
                raise ValueError(f'could not identify module path for layer {layer_i} projection {key}')
            names.append(name)
    if len(names) != len(layers) * len(LORA_MODULES):
        raise ValueError('did not find exactly seven target projections per decoder layer')
    return names


def install_lora(model, modules: list[str] | None = None, rank: int = RANK,
                 alpha: int = ALPHA) -> list[str]:
    selected = target_module_names(model) if modules is None else list(modules)
    for name in selected:
        module = model.get_submodule(name)
        if isinstance(module, LoRALinear):
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f'{name} is not nn.Linear')
        parent_name, leaf = name.rsplit('.', 1) if '.' in name else ('', name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, LoRALinear(module, rank=rank, alpha=alpha))
    return selected


def lora_modules(model) -> dict[str, LoRALinear]:
    return {name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)}


def save_adapter(model, path: str, teacher: str, base_model: str, revision: str | None) -> None:
    wrappers = lora_modules(model)
    if not wrappers:
        raise ValueError('cannot save an adapter without installed LoRA layers')
    first = next(iter(wrappers.values()))
    payload = {
        'format_version': 1,
        'teacher': teacher,
        'base_model': base_model,
        'revision': revision,
        'rank': first.rank,
        'alpha': first.alpha,
        'modules': list(wrappers),
        'lora': {name: {'A': module.lora_A.detach().cpu(), 'B': module.lora_B.detach().cpu()}
                 for name, module in wrappers.items()},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f'{path}.{os.getpid()}.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_adapter(model, path: str) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location='cpu')
    wrappers = lora_modules(model)
    if not wrappers:
        install_lora(model, payload['modules'], int(payload['rank']), int(payload['alpha']))
        wrappers = lora_modules(model)
    if set(wrappers) != set(payload['modules']) or set(payload['lora']) != set(payload['modules']):
        raise ValueError('adapter projection list does not match this base model')
    with torch.no_grad():
        for name, values in payload['lora'].items():
            wrapper = wrappers[name]
            wrapper.lora_A.copy_(values['A'].to(device=wrapper.lora_A.device, dtype=wrapper.lora_A.dtype))
            wrapper.lora_B.copy_(values['B'].to(device=wrapper.lora_B.device, dtype=wrapper.lora_B.dtype))
    return payload


def _common_train_settings(args) -> dict[str, Any]:
    return {'epochs': EPOCHS, 'effective_batch': EFFECTIVE_BATCH,
            'micro_batch': args.micro_batch, 'gradient_accumulation': EFFECTIVE_BATCH // args.micro_batch,
            'rank': RANK, 'alpha': ALPHA, 'learning_rate': LEARNING_RATE, 'seed': SEED,
            'dtype': 'bfloat16 base weights; float32 trainable A/B',
            'max_target_tokens': args.max_target, 'optimizer': 'AdamW',
            'device': args.device, 'max_gpu_gb': args.max_gpu_gb}


def _validate_adapter_records(args, revision: str | None = None,
                              require_all: bool = False,
                              require_requested_settings: bool = False) -> None:
    common_keys = tuple(_common_train_settings(args))
    expected = _common_train_settings(args)
    reference = None
    for teacher in TEACHERS:
        record_path = os.path.join(args.out, f'adapter-{teacher}.json')
        adapter_path = os.path.join(args.out, f'adapter-{teacher}.pt')
        if not os.path.isfile(record_path) or not os.path.isfile(adapter_path):
            if require_all:
                raise FileNotFoundError(f'missing {adapter_path} or its JSON record; train all three adapters first')
            continue
        with open(record_path, encoding='utf-8') as stream:
            record = json.load(stream)
        common = {key: record.get(key) for key in common_keys}
        if require_requested_settings and common != expected:
            raise ValueError(f'adapter-{teacher} training settings {common} do not match requested {expected}')
        if reference is None:
            reference = common
        elif common != reference:
            raise ValueError(f'adapter-{teacher} training settings differ from the other teacher adapters')
        if require_all and common['max_target_tokens'] != args.max_target:
            raise ValueError(f'adapter-{teacher} was trained with max_target={common["max_target_tokens"]}, '
                             f'but this step uses {args.max_target}')
        header = record.get('header', {})
        if header.get('model') != args.base_model:
            raise ValueError(f'adapter-{teacher} uses base model {header.get("model")}, expected {args.base_model}')
        if revision is not None and header.get('revision') not in (None, revision):
            raise ValueError(f'adapter-{teacher} uses revision {header.get("revision")}, expected {revision}')


def _load_checked_adapter(model, path: str, args, revision: str | None) -> dict[str, Any]:
    payload = load_adapter(model, path)
    if payload.get('base_model') != args.base_model:
        raise ValueError(f'adapter base model {payload.get("base_model")} does not match {args.base_model}')
    if revision is not None and payload.get('revision') not in (None, revision):
        raise ValueError(f'adapter revision {payload.get("revision")} does not match loaded revision {revision}')
    return payload


def _training_rows(pairs: list[dict[str, Any]], indices: list[int], teacher: str,
                   control_path: str | None) -> list[dict[str, Any]]:
    control_by_id: dict[int, dict[str, Any]] | None = None
    if teacher == 'control':
        if control_path is None:
            raise ValueError('control training requires control.jsonl; run --steps control first')
        _meta, control_by_id = _read_control_jsonl(control_path)
    by_id = {row['index']: row for row in pairs}
    result = []
    for index in indices:
        row = by_id[index]
        answer, mode = _assistant_content(row, teacher, control_by_id, 1024)
        result.append({'index': index, 'question': row['question'], 'assistant': answer, 'target_mode': mode})
    return result


def _tokenize_training_rows(tokenizer, rows: list[dict[str, Any]], max_target: int) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        item = make_supervised_example(tokenizer, row['question'], row['assistant'], max_target)
        result.append({**item, 'index': row['index']})
    return result


def _collate(examples: list[dict[str, Any]], pad_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(row['input_ids']) for row in examples)
    ids = torch.full((len(examples), width), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(examples), width), dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    for index, row in enumerate(examples):
        values = row['input_ids']
        n = len(values)
        ids[index, :n] = torch.tensor(values, dtype=torch.long)
        mask[index, :n] = 1
        labels[index] = torch.tensor(labels_for_example(values, row['prompt_len'], width), dtype=torch.long)
    return ids.to(device), mask.to(device), labels.to(device)


def masked_loss_sum(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    shifted_logits = logits[:, :-1, :].contiguous().float()
    shifted_labels = labels[:, 1:].contiguous()
    count = int((shifted_labels != -100).sum().item())
    if count == 0:
        raise ValueError('batch contains no assistant target tokens')
    loss_sum = F.cross_entropy(shifted_logits.view(-1, shifted_logits.shape[-1]),
                               shifted_labels.view(-1), ignore_index=-100, reduction='sum')
    return loss_sum, count


def _train_model(model, tokenizer, examples: list[dict[str, Any]], args) -> tuple[float, int, list[float]]:
    if args.micro_batch not in (1, 2, 4, 8):
        raise ValueError('--micro-batch must be one of 1, 2, 4, or 8 so it divides effective batch 8')
    if args.micro_batch > EFFECTIVE_BATCH:
        raise ValueError('--micro-batch cannot exceed effective batch size 8')
    install_lora(model, rank=RANK, alpha=ALPHA)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise RuntimeError('LoRA installation produced no trainable parameters')
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    total_steps = 0
    loss_history: list[float] = []
    loss_sums: list[tuple[float, int]] = []
    recent_sum = recent_tokens = 0
    for epoch in range(EPOCHS):
        order = list(range(len(examples)))
        random.Random(SEED + epoch).shuffle(order)
        for group_start in range(0, len(order), EFFECTIVE_BATCH):
            group = [examples[index] for index in order[group_start:group_start + EFFECTIVE_BATCH]]
            group_tokens = sum(row['target_len'] for row in group)
            optimizer.zero_grad(set_to_none=True)
            group_sum = 0.0
            group_count = 0
            for micro_start in range(0, len(group), args.micro_batch):
                micro = group[micro_start:micro_start + args.micro_batch]
                ids, mask, labels = _collate(micro, int(pad_id), args.device)
                output = model(input_ids=ids, attention_mask=mask, use_cache=False)
                loss_sum, count = masked_loss_sum(output.logits, labels)
                (loss_sum / group_tokens).backward()
                group_sum += float(loss_sum.detach().item())
                group_count += count
                del output, loss_sum, ids, mask, labels
            if group_count != group_tokens:
                raise RuntimeError(f'loss mask counted {group_count} targets, expected {group_tokens}')
            optimizer.step()
            total_steps += 1
            loss_history.append(group_sum / group_count)
            loss_sums.append((group_sum, group_count))
            recent_sum += group_sum
            recent_tokens += group_count
            if total_steps % 20 == 0:
                print(f'  epoch {epoch + 1}/{EPOCHS} step {total_steps}: '
                      f'loss={recent_sum / recent_tokens:.5f}', flush=True)
                recent_sum = recent_tokens = 0
            nr.spill_guard(args.device, args.max_gpu_gb)
    if recent_tokens:
        print(f'  final training window loss={recent_sum / recent_tokens:.5f}', flush=True)
    tail = loss_sums[-min(20, len(loss_sums)):]
    final_loss = sum(value for value, _count in tail) / sum(count for _value, count in tail)
    return float(final_loss), total_steps, loss_history


def cmd_train(args) -> None:
    teacher = args.teacher
    if teacher not in TEACHERS:
        raise ValueError('training requires --teacher r1, gemini, or control')
    pairs = read_pairs(args.pairs)
    split = _load_split(args.out, pairs, args.pairs)
    source_rows = _training_rows(pairs, [int(v) for v in split['train_indices']], teacher,
                                 _control_jsonl_path(args.out))
    tokenizer = model = None
    try:
        tokenizer, model, revision = _load_model(args)
        _seed_everything(SEED)
        _validate_adapter_records(args, revision=revision, require_requested_settings=True)
        examples = _tokenize_training_rows(tokenizer, source_rows, args.max_target)
        model.config.use_cache = False
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if teacher == 'control':
            control_meta, _control_rows = _read_control_jsonl(_control_jsonl_path(args.out))
            if control_meta is None or control_meta.get('model') != args.base_model:
                raise ValueError('control answers do not identify the requested base model')
            if revision is not None and control_meta.get('revision') not in (None, revision):
                raise ValueError('control answers were generated from a different base-model revision')
        started = time.time()
        with torch.enable_grad():
            final_loss, steps, history = _train_model(model, tokenizer, examples, args)
        adapter_path = os.path.join(args.out, f'adapter-{teacher}.pt')
        save_adapter(model, adapter_path, teacher, args.base_model, revision)
        settings = {'teacher': teacher, 'train_rows': len(examples), **_common_train_settings(args),
                    'target_truncation': 'same token cap applied to every teacher; loss labels only assistant tokens',
                    'steps': steps, 'final_loss': final_loss,
                    'elapsed_seconds': time.time() - started,
                    'first_logged_losses': history[:5], 'last_logged_losses': history[-5:]}
        header = make_header('train', args, {'pairs': args.pairs, 'split.json': _split_path(args.out),
                                             'part30.py': __file__, 'neuron_report.py': nr.__file__},
                             settings, revision=revision)
        if teacher == 'control':
            header['input_shas']['control.jsonl'] = _path_shas({'control.jsonl': _control_jsonl_path(args.out)})['control.jsonl']
        dump_json({'header': header, **settings, 'adapter': os.path.abspath(adapter_path)},
                  os.path.join(args.out, f'adapter-{teacher}.json'))
        print(f'train {teacher}: {steps} optimizer steps, final loss {final_loss:.5f}; saved {adapter_path}', flush=True)
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        _empty_xpu_cache()


def _prompt_rows(path: str) -> list[dict[str, Any]]:
    with open(path, encoding='utf-8') as stream:
        payload = json.load(stream)
    if isinstance(payload, dict):
        for key in ('prompts', 'rows', 'data'):
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                break
        else:
            raise ValueError('prompt JSON object must contain a prompts, rows, or data list')
    if not isinstance(payload, list):
        raise ValueError('prompt JSON must be a list or an object containing a list')
    rows: list[dict[str, Any]] = []
    for list_index, item in enumerate(payload):
        if isinstance(item, str):
            row = {'index': list_index, 'prompt': item}
        elif isinstance(item, dict):
            prompt = item.get('prompt', item.get('text', item.get('question')))
            if prompt is None:
                raise ValueError(f'prompt row {list_index} has no prompt/text/question')
            row = {'index': int(item.get('index', list_index)), 'prompt': str(prompt)}
        else:
            raise ValueError(f'prompt row {list_index} must be a string or object')
        if 240 <= row['index'] <= 299:
            rows.append(row)
    rows.sort(key=lambda row: row['index'])
    ids = [row['index'] for row in rows]
    if ids != list(range(240, 300)):
        raise ValueError(f'expected exactly prompt indices 240..299, found {len(ids)} rows')
    return rows


def make_answer_row(index: int, model: str, provider: str, text: str, finish_reason: str,
                    input_tokens: int, output_tokens: int, wall_time: float) -> dict[str, Any]:
    answer = text.rsplit('</think>', 1)[-1].strip() if '</think>' in text else text.strip()
    return {'index': int(index), 'model': model, 'provider': provider, 'text': text, 'answer': answer,
            'finish_reason': finish_reason,
            'usage': {'input_tokens': int(input_tokens), 'output_tokens': int(output_tokens),
                      'total_tokens': int(input_tokens + output_tokens)},
            'wall_time': float(wall_time)}


def cmd_generate(args) -> None:
    teacher = args.teacher
    if teacher not in (*TEACHERS, 'base'):
        raise ValueError('generation requires --teacher base, r1, gemini, or control')
    if not args.prompts:
        raise ValueError('generation requires --prompts data/part29-prompts.json')
    prompt_rows = _prompt_rows(args.prompts)
    adapter_path = os.path.join(args.out, f'adapter-{teacher}.pt') if teacher != 'base' else None
    if adapter_path and not os.path.isfile(adapter_path):
        raise FileNotFoundError(f'missing adapter {adapter_path}; train it before generation')
    output_path = os.path.join('results', 'part29', f'answers-part30-{teacher}.jsonl')
    meta_path = output_path[:-6] + '.meta.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    completed = _read_answer_jsonl(output_path)
    old_meta = None
    if os.path.isfile(meta_path):
        with open(meta_path, encoding='utf-8') as stream:
            old_meta = json.load(stream)
    elif completed:
        raise ValueError(f'{output_path} has rows but no metadata header')
    expected_indices = {int(row['index']) for row in prompt_rows}
    unknown = set(completed) - expected_indices
    if unknown:
        raise ValueError(f'{output_path} contains non-prompt index {min(unknown)}')

    tokenizer = model = None
    started_all = time.time()
    try:
        tokenizer, model, revision = _load_model(args)
        if adapter_path:
            _load_checked_adapter(model, adapter_path, args, revision)
            model_tag = f'{args.base_model}+part30-{teacher}'
        else:
            model_tag = args.base_model
        settings = {'teacher': teacher, 'prompt_index_range': [240, 299],
                    'prompt_rows': len(prompt_rows), 'system_prompt': None,
                    'decoding': 'sampled', 'do_sample': True, 'temperature': 0.7,  # as Part 29's teachers (TEMPERATURE = 0.7)
                    'max_new_tokens': GENERATE_MAX_NEW, 'enable_thinking': False,
                    'gen_batch': args.gen_batch, 'resumable': True,
                    'wall_time': 'per-row share of a batch',
                    'device': args.device, 'max_gpu_gb': args.max_gpu_gb}
        paths = {'prompts': args.prompts, 'part30.py': __file__, 'neuron_report.py': nr.__file__}
        if adapter_path:
            paths['adapter'] = adapter_path
        header = make_header('generate', args, paths, settings, revision=revision)
        if old_meta is not None:
            old_header = old_meta.get('header', {})
            old_settings = dict(old_header.get('settings', {}))
            old_settings.pop('elapsed_seconds', None)
            if old_header.get('input_shas') != header.get('input_shas') or old_settings != settings:
                raise ValueError('existing generate output has different inputs/settings; move it before restarting')
            if old_header.get('model') != args.base_model:
                raise ValueError('existing generate output was generated by a different base model')
            if old_header.get('revision') not in (None, revision):
                raise ValueError(f'existing generate output uses revision {old_header.get("revision")}; loaded {revision}')
        header['settings']['elapsed_seconds'] = 0.0
        meta = {'header': header, 'output': os.path.abspath(output_path), 'row_count': len(completed)}
        dump_json(meta, meta_path)
        pending = [row for row in prompt_rows if int(row['index']) not in completed]
        ordinal = len(completed)
        for batch in _iter_generated_batches(tokenizer, model, pending, 'prompt', args.device,
                                             GENERATE_MAX_NEW, args.gen_batch, do_sample=True, temperature=0.7):
            for prompt, generated in batch:
                answer = make_answer_row(prompt['index'], model_tag, 'local', generated['text'],
                                         generated['finish_reason'], generated['input_tokens'],
                                         generated['output_tokens'], generated['wall_time'])
                _append_jsonl(output_path, answer)
                completed[int(prompt['index'])] = answer
                ordinal += 1
            nr.spill_guard(args.device, args.max_gpu_gb)
            header['settings']['elapsed_seconds'] = time.time() - started_all
            meta['row_count'] = len(completed)
            dump_json(meta, meta_path)
            print(f'  generate {teacher}: {ordinal}/{len(prompt_rows)}', flush=True)
        header['settings']['elapsed_seconds'] = time.time() - started_all
        meta['row_count'] = len(completed)
        dump_json(meta, meta_path)
        print(f'generate {teacher}: wrote {len(completed)} rows -> {output_path}', flush=True)
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        _empty_xpu_cache()


def _pair_assistant_texts(row: dict[str, Any], teacher: str) -> str:
    return _assistant_content(row, teacher, None, PROBE_MAX_TARGET)[0]


def _heldout_rows(pairs: list[dict[str, Any]], split: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {row['index']: row for row in pairs}
    return [by_id[int(index)] for index in split['heldout_indices']]


def _evaluation_examples(tokenizer, rows: list[dict[str, Any]], teacher: str,
                         max_target: int) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        target, _mode = _assistant_content(row, teacher, None, max_target)
        examples.append({'index': row['index'], **make_supervised_example(tokenizer, row['question'], target, max_target)})
    return examples


def _evaluate_loss(model, tokenizer, examples: list[dict[str, Any]], args,
                   batch_size: int = 2) -> dict[str, Any]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            ids, mask, labels = _collate(examples[start:start + batch_size], int(pad_id), args.device)
            output = model(input_ids=ids, attention_mask=mask, use_cache=False)
            loss_sum, count = masked_loss_sum(output.logits, labels)
            total_loss += float(loss_sum.item())
            total_tokens += count
            del output, loss_sum, ids, mask, labels
            nr.spill_guard(args.device, args.max_gpu_gb)
    return {'mean_per_token_loss': total_loss / total_tokens, 'scored_assistant_tokens': total_tokens,
            'examples': len(examples)}


def _model_specs(args) -> list[tuple[str, str | None]]:
    return [('base', None)] + [(teacher, os.path.join(args.out, f'adapter-{teacher}.pt'))
                                for teacher in TEACHERS]


def cmd_loss(args) -> None:
    pairs = read_pairs(args.pairs)
    split = _load_split(args.out, pairs, args.pairs)
    rows = _heldout_rows(pairs, split)
    _validate_adapter_records(args, require_all=True)
    results: dict[str, Any] = {}
    revisions: dict[str, str | None] = {}
    for name, adapter_path in _model_specs(args):
        if adapter_path and not os.path.isfile(adapter_path):
            raise FileNotFoundError(f'missing {adapter_path}; train all three adapters before loss')
        tokenizer = model = None
        try:
            tokenizer, model, revision = _load_model(args)
            revisions[name] = revision
            if adapter_path:
                _load_checked_adapter(model, adapter_path, args, revision)
            results[name] = {}
            for teacher in PROBE_TEACHERS:
                examples = _evaluation_examples(tokenizer, rows, teacher, args.max_target)
                results[name][teacher] = _evaluate_loss(model, tokenizer, examples, args,
                                                        batch_size=args.eval_batch_size)
                print(f'  loss {name} on {teacher}: {results[name][teacher]["mean_per_token_loss"]:.5f}', flush=True)
        finally:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            gc.collect()
            _empty_xpu_cache()
    paths: dict[str, str | None] = {'pairs': args.pairs, 'split.json': _split_path(args.out),
                                   'part30.py': __file__, 'neuron_report.py': nr.__file__}
    paths.update({f'adapter-{teacher}': os.path.join(args.out, f'adapter-{teacher}.pt') for teacher in TEACHERS})
    header = make_header('loss', args, paths, {'evaluated_rows': len(rows), 'teachers': list(PROBE_TEACHERS),
                                               'max_target_tokens': args.max_target,
                                               'score': 'mean token cross-entropy on assistant labels only',
                                               'models': [name for name, _ in _model_specs(args)],
                                               'eval_batch_size': args.eval_batch_size,
                                               'device': args.device, 'max_gpu_gb': args.max_gpu_gb},
                         revision=','.join(sorted({value for value in revisions.values() if value})) or args.revision)
    dump_json({'header': header, 'per_model': results}, os.path.join(args.out, 'loss.json'))
    print(f'loss: saved {os.path.join(args.out, "loss.json")}', flush=True)


def capture_residual_means(model, input_ids: list[int], assistant_start: int, assistant_end: int,
                           device: str) -> np.ndarray:
    decoder = getattr(model, 'model', model)
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    with torch.no_grad():
        output = decoder(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
    hidden_states = getattr(output, 'hidden_states', None)
    if hidden_states is None:
        raise ValueError('decoder did not return hidden_states')
    start = max(0, int(assistant_start))
    end = min(int(assistant_end), len(input_ids))
    if end <= start:
        raise ValueError('assistant token span is empty')
    means = np.stack([state[0, start:end].float().mean(dim=0).cpu().numpy() for state in hidden_states[1:]])
    del output, ids, mask
    return means.astype(np.float32, copy=False)


def _capture_probe_features(tokenizer, model, rows: list[dict[str, Any]], args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    labels = []
    groups = []
    n_layers = None
    for row_index, row in enumerate(rows, 1):
        for teacher, label in (('r1', 1), ('gemini', 0)):
            target, _mode = _assistant_content(row, teacher, None, PROBE_MAX_TARGET)
            example = make_supervised_example(tokenizer, row['question'], target, PROBE_MAX_TARGET)
            start = example['prompt_len']
            end = len(example['input_ids'])
            means = capture_residual_means(model, example['input_ids'], start, end, args.device)
            if n_layers is None:
                n_layers = means.shape[0]
            elif n_layers != means.shape[0]:
                raise ValueError('decoder returned inconsistent hidden-state count')
            features.append(means)
            labels.append(label)
            groups.append(row['index'])
        if row_index % 10 == 0 or row_index == len(rows):
            nr.spill_guard(args.device, args.max_gpu_gb)
            print(f'  probe capture {row_index}/{len(rows)} held-out questions', flush=True)
    return np.stack(features, axis=1), np.asarray(labels, dtype=np.int64), np.asarray(groups, dtype=np.int64)


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[y_true == 1]
    negatives = scores[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError('AUROC needs both positive and negative labels')
    wins = 0.0
    for score in positives:
        wins += float(np.sum(score > negatives)) + 0.5 * float(np.sum(score == negatives))
    return wins / (len(positives) * len(negatives))


def _folds_by_group(groups: np.ndarray, n_splits: int = 5, seed: int = SEED) -> list[np.ndarray]:
    unique = np.unique(groups)
    if len(unique) < n_splits:
        raise ValueError(f'need at least {n_splits} distinct question groups')
    shuffled = unique.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    return [np.asarray(part) for part in np.array_split(shuffled, n_splits)]


def _standardize_fit_predict(train_x: np.ndarray, train_y: np.ndarray,
                             test_x: np.ndarray, l2: float = 0.01,
                             iterations: int = 180) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    x = np.clip((train_x - mean) / scale, -20, 20).astype(np.float64, copy=False)
    xt = np.clip((test_x - mean) / scale, -20, 20).astype(np.float64, copy=False)
    y = train_y.astype(np.float64, copy=False)
    # Power iteration estimates the largest covariance eigenvalue for a stable full-batch step size.
    v = np.ones(x.shape[1], dtype=np.float64) / math.sqrt(x.shape[1])
    for _ in range(8):
        v = x.T @ (x @ v)
        norm = np.linalg.norm(v)
        if norm:
            v /= norm
    eigenvalue = float(np.dot(x @ v, x @ v) / len(x))
    step = 1.0 / (0.25 * eigenvalue + l2 + 1e-8)
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    # FISTA speeds convergence without introducing an external ML dependency.
    old_weights = weights.copy()
    old_bias = bias
    momentum = 1.0
    for _ in range(iterations):
        z = np.clip(x @ weights + bias, -30, 30)
        prob = 1.0 / (1.0 + np.exp(-z))
        grad_w = x.T @ (prob - y) / len(y) + l2 * weights
        grad_b = float(np.mean(prob - y))
        next_weights = weights - step * grad_w
        next_bias = bias - step * grad_b
        next_momentum = (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
        weights = next_weights + ((momentum - 1.0) / next_momentum) * (next_weights - old_weights)
        bias = next_bias + ((momentum - 1.0) / next_momentum) * (next_bias - old_bias)
        old_weights, old_bias, momentum = next_weights, next_bias, next_momentum
    z_test = np.clip(xt @ weights + bias, -30, 30)
    return 1.0 / (1.0 + np.exp(-z_test))


def logistic_probe_auc(features: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                       n_splits: int = 5, seed: int = SEED) -> dict[str, Any]:
    """Five-fold grouped logistic-regression AUROC; both traces from a question stay together."""
    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    group_values = np.asarray(groups)
    if X.ndim != 2 or len(X) != len(y) or len(y) != len(group_values):
        raise ValueError('features, labels, and groups must have matching [samples, features] shapes')
    fold_groups = _folds_by_group(group_values, n_splits, seed)
    fold_results = []
    for fold_i, test_groups in enumerate(fold_groups):
        test_mask = np.isin(group_values, test_groups)
        train_x, train_y = X[~test_mask], y[~test_mask]
        test_x, test_y = X[test_mask], y[test_mask]
        scores = _standardize_fit_predict(train_x, train_y, test_x)
        fold_results.append({'fold': fold_i, 'question_groups': [int(v) for v in test_groups],
                             'auroc': _auc(test_y, scores)})
    return {'mean_auroc': float(np.mean([row['auroc'] for row in fold_results])),
            'folds': fold_results, 'n_samples': len(y), 'n_question_groups': len(np.unique(group_values))}


def cmd_probe(args) -> None:
    pairs = read_pairs(args.pairs)
    split = _load_split(args.out, pairs, args.pairs)
    rows = _heldout_rows(pairs, split)
    _validate_adapter_records(args, require_all=True)
    results: dict[str, Any] = {}
    revisions: dict[str, str | None] = {}
    for name, adapter_path in _model_specs(args):
        if adapter_path and not os.path.isfile(adapter_path):
            raise FileNotFoundError(f'missing {adapter_path}; train all three adapters before probe')
        tokenizer = model = None
        try:
            tokenizer, model, revision = _load_model(args)
            revisions[name] = revision
            if adapter_path:
                _load_checked_adapter(model, adapter_path, args, revision)
            features, labels, groups = _capture_probe_features(tokenizer, model, rows, args)
            layer_rows = []
            for layer_i in range(features.shape[0]):
                auc = logistic_probe_auc(features[layer_i], labels, groups, n_splits=5, seed=SEED)
                layer_rows.append({'layer': layer_i, **auc})
                print(f'  probe {name} layer {layer_i}: AUROC {auc["mean_auroc"]:.4f}', flush=True)
            results[name] = {'pooling': 'mean of assistant-token residual stream, first 512 assistant tokens',
                             'labels': {'r1': 1, 'gemini': 0}, 'layers': layer_rows}
        finally:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            gc.collect()
            _empty_xpu_cache()
    paths: dict[str, str | None] = {'pairs': args.pairs, 'split.json': _split_path(args.out),
                                   'part30.py': __file__, 'neuron_report.py': nr.__file__}
    paths.update({f'adapter-{teacher}': os.path.join(args.out, f'adapter-{teacher}.pt') for teacher in TEACHERS})
    header = make_header('probe', args, paths,
                         {'evaluated_rows': len(rows), 'traces_per_model': 2 * len(rows),
                          'folds': 5, 'fold_group': 'original question index',
                          'max_assistant_tokens': PROBE_MAX_TARGET,
                          'probe': 'standardized logistic regression, NumPy float64 optimizer, mean-pool per transformer block',
                          'models': [name for name, _ in _model_specs(args)],
                          'device': args.device, 'max_gpu_gb': args.max_gpu_gb},
                         revision=','.join(sorted({value for value in revisions.values() if value})) or args.revision)
    dump_json({'header': header, 'per_model': results}, os.path.join(args.out, 'probe.json'))
    print(f'probe: saved {os.path.join(args.out, "probe.json")}', flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', required=True, help='comma-separated: split,control,train,generate,loss,probe')
    parser.add_argument('--teacher', choices=('r1', 'gemini', 'control', 'base'), default=None)
    parser.add_argument('--pairs', default=None, help='s1K-1.1 paired question JSONL')
    parser.add_argument('--prompts', default=None, help='Part 29 prompt JSON')
    parser.add_argument('--out', default=DEFAULT_OUT, help='Part 30 results directory')
    parser.add_argument('--base-model', default=BASE_MODEL)
    parser.add_argument('--revision', default=None)
    parser.add_argument('--device', choices=('xpu', 'cpu'), default='xpu')
    parser.add_argument('--max-gpu-gb', type=float, default=14.5)
    parser.add_argument('--max-target', type=int, default=1024)
    parser.add_argument('--micro-batch', type=int, default=2)
    parser.add_argument('--eval-batch-size', type=int, default=2)
    parser.add_argument('--gen-batch', type=int, default=16,
                        help='control/generate rows per greedy generation batch (1 uses single-row generation)')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    known = {'split', 'control', 'train', 'generate', 'loss', 'probe'}
    steps = [value.strip() for value in args.steps.split(',') if value.strip()]
    if not steps or len(set(steps)) != len(steps) or any(value not in known for value in steps):
        raise SystemExit(f'--steps must be unique comma-separated values from {", ".join(sorted(known))}')
    if any(step in steps for step in ('split', 'control', 'train', 'loss', 'probe')) and not args.pairs:
        raise SystemExit('--pairs is required for this step')
    if 'train' in steps and args.teacher not in TEACHERS:
        raise SystemExit('--teacher r1|gemini|control is required for --steps train')
    if 'generate' in steps and args.teacher not in (*TEACHERS, 'base'):
        raise SystemExit('--teacher base|r1|gemini|control is required for --steps generate')
    if args.max_target < 1 or args.eval_batch_size < 1 or args.gen_batch < 1:
        raise SystemExit('--max-target, --eval-batch-size, and --gen-batch must be positive')
    if args.micro_batch not in (1, 2, 4, 8):
        raise SystemExit('--micro-batch must be one of 1, 2, 4, or 8')
    os.makedirs(args.out, exist_ok=True)
    for step in steps:
        if step == 'split':
            cmd_split(args)
        elif step == 'control':
            cmd_control(args)
        elif step == 'train':
            cmd_train(args)
        elif step == 'generate':
            cmd_generate(args)
        elif step == 'loss':
            cmd_loss(args)
        elif step == 'probe':
            cmd_probe(args)


if __name__ == '__main__':
    main()
