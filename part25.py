#!/usr/bin/env python3
"""Part 25: search for the smallest neuron sets that change Qwen3.5 identity answers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from typing import Callable, Iterable

import numpy as np
import torch

import circuit
import neuron_report as nr


MODEL = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
STEMS = {'maker': 'I was created by', 'name': 'My name is'}
VARIANTS = {'plain': '', 'bold': ' **'}
TARGET_WORDS = {
    'qwen': {'maker': ['Alibaba', 'Tongyi', 'Qwen'], 'name': ['Qwen', 'Tongyi']},
    'google': {'maker': ['Google'], 'name': ['Gemini']},
    'openai': {'maker': ['OpenAI'], 'name': ['ChatGPT']},  # 'GPT' left out: its first token ' G' starts too many other words
    'anthropic': {'maker': ['Anthropic'], 'name': ['Claude']},
    'deepseek': {'maker': ['DeepSeek'], 'name': ['DeepSeek']},
}
SYSTEM_MESSAGES = {
    'qwen': 'You are Qwen, a large language model created by Alibaba Cloud.',
    'google': 'You are Gemini, a large language model created by Google.',
    'openai': 'You are ChatGPT, a large language model created by OpenAI.',
    'anthropic': 'You are Claude, a large language model created by Anthropic.',
    'deepseek': 'You are DeepSeek, a large language model created by DeepSeek.',
}
SEARCH_TARGETS = ['google-removal', 'openai', 'anthropic', 'deepseek', 'google']
SUCCESS_QUESTIONS = {
    'google-removal': ['Who created you?'],
    'google': ['Who created you?', 'What is your name?'],
    'openai': ['Who created you?', 'What is your name?'],
    'anthropic': ['Who created you?', 'What is your name?'],
    'deepseek': ['Who created you?', 'What is your name?'],
}
FOREIGN_MAKERS = re.compile(r'\b(?:OpenAI|ChatGPT|Anthropic|Claude|DeepSeek)\b', re.I)
GOOGLE_NAMES = re.compile(r'\b(?:Google|Gemini)\b', re.I)
IDENTITY = circuit.IDENTITY
CORRELATE_PROMPTS = [
    'Reply with only a person\'s name: who is the CEO of the company that created you?',
    'Reply with only one word: which country is the company that created you based in?',
]


def _token_ids(tok, text: str) -> list[int]:
    return [int(x) for x in tok(text, add_special_tokens=False)['input_ids']]


def validate_token_records(records: dict) -> None:
    """Reject any first-token collision between different targets in one variant."""
    for variant in VARIANTS:
        owners: dict[int, str] = {}
        for target, by_stem in records[variant].items():
            ids = set()
            for words in by_stem.values():
                ids.update(int(rec['id']) for rec in words.values())
            for token_id in ids:
                if token_id in owners and owners[token_id] != target:
                    raise ValueError(
                        f'first-token collision in {variant}: {owners[token_id]} and {target} share token id {token_id}')
                owners[token_id] = target


def resolve_token_sets(tok) -> dict:
    """Resolve the first token of each identity word in its plain or bold continuation form."""
    result = {}
    for variant, marker in VARIANTS.items():
        result[variant] = {}
        for target, stems in TARGET_WORDS.items():
            result[variant][target] = {}
            for stem_name, words in stems.items():
                word_records = {}
                for word in words:
                    ids = _token_ids(tok, word if marker else ' ' + word)
                    if not ids:
                        raise ValueError(f'tokenizer returned no token for {word!r} ({variant})')
                    first = ids[0]
                    word_records[word] = {'id': first, 'decoded': tok.decode([first])}
                result[variant][target][stem_name] = word_records
    validate_token_records(result)
    return result


def token_ids(records: dict, variant: str, target: str, stem_name: str) -> list[int]:
    return sorted({int(x['id']) for x in records[variant][target][stem_name].values()})


def _offset_encoding(tok, text: str):
    try:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = [int(x) for x in enc['input_ids']]
        offsets = [(int(a), int(b)) for a, b in enc['offset_mapping']]
        if len(ids) == len(offsets):
            return ids, offsets
    except (TypeError, ValueError, NotImplementedError, KeyError):
        pass
    return _token_ids(tok, text), None


def make_sequence(tok, question: str, stem_name: str, variant: str,
                  system_prompt: str | None = None) -> dict:
    prompt = nr.build_prompt(tok, question, False, system_prompt=system_prompt)
    stem = STEMS[stem_name]
    marker = VARIANTS[variant]
    text = prompt + stem + marker
    ids, offsets = _offset_encoding(tok, text)
    if not ids:
        raise ValueError(f'empty token sequence for {question!r}')
    if offsets is not None:
        lo, hi = len(prompt), len(prompt) + len(stem)
        stem_positions = [i for i, (a, b) in enumerate(offsets) if b > a and a < hi and b > lo]
    else:
        prefix_ids = _token_ids(tok, prompt)
        stem_ids = _token_ids(tok, stem)
        start = len(prefix_ids)
        stem_positions = list(range(start, min(start + len(stem_ids), len(ids))))
    if not stem_positions:
        raise ValueError(f'could not locate stem tokens in {question!r} / {variant}')
    return {
        'question': question, 'stem': stem_name, 'variant': variant,
        'input_ids': ids, 'score_position': len(ids) - 1, 'stem_positions': stem_positions,
    }


def build_sequences(tok, system_prompt: str | None = None) -> list[dict]:
    return [make_sequence(tok, q, stem_name, variant, system_prompt)
            for q in circuit.ID_PROMPTS for stem_name in STEMS for variant in VARIANTS]


def first_sentence(text: str) -> str:
    return re.split(r'(?<=[.!?])\s+', text.strip(), 1)[0]


def target_success(target: str, answers: dict[str, dict]) -> bool:
    if target == 'google-removal':
        first = answers.get('Who created you?', {}).get('first_sentence', '')
        return bool(GOOGLE_NAMES.search(first)) and not bool(IDENTITY.search(first))
    words = TARGET_WORDS[target]
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(w) for xs in words.values() for w in xs) + r')\b', re.I)
    for question in SUCCESS_QUESTIONS[target]:
        first = answers.get(question, {}).get('first_sentence', '')
        if not pattern.search(first) or IDENTITY.search(first):
            return False
    return True


def header(args, input_shas: dict[str, str]) -> dict:
    return {
        'model': args.model,
        'revision': args.revision,
        'tool_sha256': nr._file_sha(os.path.abspath(__file__)),
        'input_shas': dict(input_shas),
        'written': time.strftime('%Y-%m-%d %H:%M:%S'),
    }


def source_shas() -> dict[str, str]:
    return {'neuron_report.py': nr._file_sha(nr.__file__), 'circuit.py': nr._file_sha(circuit.__file__)}


def write_json(obj: dict, path: str) -> None:
    nr._dump(obj, path)


def actual_device(model) -> torch.device:
    return next(model.parameters()).device


def _pad_batch(sequences: list[dict], tok, device: torch.device):
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad_id is None:
        pad_id = 0
    max_len = max(len(s['input_ids']) for s in sequences)
    ids = torch.full((len(sequences), max_len), int(pad_id), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, sequence in enumerate(sequences):
        row = torch.tensor(sequence['input_ids'], dtype=torch.long, device=device)
        ids[i, :len(row)] = row
        mask[i, :len(row)] = 1
    positions = torch.tensor([int(s['score_position']) for s in sequences], dtype=torch.long, device=device)
    return ids, mask, positions


def _row_intervention(x: torch.Tensor, layer_index: int, row_ops: list[dict],
                      position_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Apply each batch row's own constants, only at real (un-padded) positions."""
    changed = False
    out = x
    for row, operations in enumerate(row_ops):
        units = [(unit, value) for (layer, unit), value in operations.items() if int(layer) == layer_index]
        if not units:
            continue
        if not changed:
            out = x.clone()
            changed = True
        if position_mask is None:
            positions = slice(None)
        else:
            positions = position_mask[row].to(device=x.device, dtype=torch.bool)
        for unit, value in units:
            if isinstance(positions, slice):
                out[row, positions, int(unit)] = float(value)
            else:
                out[row, positions, int(unit)] = float(value)
    return out


def install_intervention_hooks(layers, row_ops: list[dict], position_mask: torch.Tensor | None = None):
    handles = []
    by_layer = {int(l) for ops in row_ops for l, _ in ops}
    for li in sorted(by_layer):
        def hook(_mod, args, li=li):
            x = args[0]
            y = _row_intervention(x, li, row_ops, position_mask)
            return None if y is x else (y,) + tuple(args[1:])
        handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(hook))
    return handles


def _score_margins(logits: torch.Tensor, sequences: list[dict], positions: torch.Tensor,
                   token_records: dict, target: str) -> torch.Tensor:
    values = []
    for row, sequence in enumerate(sequences):
        variant, stem_name = sequence['variant'], sequence['stem']
        target_ids = token_ids(token_records, variant, target, stem_name)
        qwen_ids = token_ids(token_records, variant, 'qwen', stem_name)
        row_logits = logits[row, positions[row]].float()
        values.append(torch.logsumexp(row_logits[target_ids], dim=0) - torch.logsumexp(row_logits[qwen_ids], dim=0))
    return torch.stack(values)


def first_order_estimate(gradient: torch.Tensor, current: torch.Tensor,
                         new_values: torch.Tensor | np.ndarray | float) -> np.ndarray:
    """Sum d(margin)/d(activation) times the requested activation change over rows and positions."""
    current = current.float()
    new = torch.as_tensor(new_values, dtype=current.dtype, device=current.device)
    return ((gradient.float() * (new - current)).sum(dim=(0, 1))).detach().cpu().numpy()


def _operations(neurons: Iterable[tuple[int, int]], operation: str, means: np.ndarray | None = None) -> dict:
    if operation == 'zero':
        return {(int(layer), int(unit)): 0.0 for layer, unit in neurons}
    if means is None:
        raise ValueError('steering requires activation means')
    return {(int(layer), int(unit)): float(means[int(layer), int(unit)]) for layer, unit in neurons}


def evaluate_sets(tok, model, layers, sequences: list[dict], token_records: dict, target: str,
                  sets: list[list[tuple[int, int]]], operation: str,
                  means: np.ndarray | None, max_gpu_gb: float) -> list[float]:
    """Evaluate each candidate set as its own row group against all supplied sequences."""
    if not sets:
        return []
    device = actual_device(model)
    flat_sequences = sequences * len(sets)
    ids, mask, positions = _pad_batch(flat_sequences, tok, device)
    ops_per_set = [_operations(neurons, operation, means) for neurons in sets]
    row_ops = [ops_per_set[c] for c in range(len(sets)) for _ in sequences]
    flat_mask = mask
    handles = install_intervention_hooks(layers, row_ops, flat_mask)
    try:
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask, use_cache=False)
            margins = _score_margins(out.logits, flat_sequences, positions, token_records, target)
        nr.spill_guard(str(device.type), max_gpu_gb)
        return [float(x) for x in margins.reshape(len(sets), len(sequences)).mean(dim=1).detach().cpu()]
    finally:
        for handle in handles:
            handle.remove()


def _greedy_generate(tok, model, layers, questions: list[str], operations: dict,
                     device: str, max_gpu_gb: float, system_prompt: str | None = None,
                     max_new_tokens: int = 48) -> dict:
    row_ops = [operations]
    handles = install_intervention_hooks(layers, row_ops)
    answers = {}
    try:
        for question in questions:
            prompt = nr.build_prompt(tok, question, False, system_prompt=system_prompt)
            ids = torch.tensor([_token_ids(tok, prompt)], dtype=torch.long, device=actual_device(model))
            attention_mask = torch.ones_like(ids)
            with torch.no_grad():
                generated = model.generate(
                    input_ids=ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
                    do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            text = tok.decode(generated[0, ids.shape[1]:], skip_special_tokens=True).strip()
            answers[question] = {'answer': text, 'first_sentence': first_sentence(text)}
            nr.spill_guard(device, max_gpu_gb)
    finally:
        for handle in handles:
            handle.remove()
    return answers


def _all_answers(tok, model, layers, neurons, operation, means, device, max_gpu_gb):
    ops = _operations(neurons, operation, means)
    return _greedy_generate(tok, model, layers, circuit.ID_PROMPTS, ops, device, max_gpu_gb)


def single_drop_audit(members, success: Callable, baseline=None):
    """Check whether the supplied successful set loses success after every single-member drop."""
    baseline = success(members) if baseline is None else baseline
    base_holds = bool(baseline[0]) if isinstance(baseline, tuple) else bool(baseline)
    rows = []
    for member in members:
        trial = [x for x in members if x != member]
        result = success(trial)
        if isinstance(result, tuple):
            holds, detail = bool(result[0]), result[1]
        else:
            holds, detail = bool(result), {}
        rows.append({'dropped_neuron': {'layer': int(member[0]), 'index': int(member[1])},
                     'success_without': holds, 'answers_without': detail})
    return {'success': base_holds, 'checks': rows,
            'is_1_minimal': base_holds and all(not row['success_without'] for row in rows)}


def activation_ranges(tok, model, layers, neurons, system_prompt: str | None,
                      device: str, max_gpu_gb: float) -> dict:
    """Natural min/mean/max for chosen units over stem positions of the 24 sequences."""
    by_layer: dict[int, list[int]] = {}
    for li, unit in neurons:
        by_layer.setdefault(int(li), []).append(int(unit))
    by_layer = {li: sorted(set(units)) for li, units in by_layer.items()}
    if not by_layer:
        return {}
    vals: dict[tuple[int, int], list[float]] = {(li, unit): [] for li, units in by_layer.items() for unit in units}
    for sequence in build_sequences(tok, system_prompt):
        ids = torch.tensor([sequence['input_ids']], dtype=torch.long, device=actual_device(model))
        capture = {}
        handles = []
        for li, units in by_layer.items():
            ids_t = torch.tensor(units, dtype=torch.long)
            def hook(_mod, args, li=li, ids_t=ids_t):
                capture[li] = args[0].detach()[0].index_select(-1, ids_t.to(args[0].device)).float().cpu()
                return None
            handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(hook))
        try:
            with torch.no_grad():
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            pos = sequence['stem_positions']
            for li, units in by_layer.items():
                selected = capture[li][pos]
                for col, unit in enumerate(units):
                    vals[(li, unit)].extend(float(x) for x in selected[:, col].tolist())
            nr.spill_guard(device, max_gpu_gb)
        finally:
            for handle in handles:
                handle.remove()
    return {
        f'{li}:{unit}': {'layer': li, 'index': unit, 'min': min(values),
                         'mean': float(np.mean(values)), 'max': max(values),
                         'positions': len(values)}
        for (li, unit), values in vals.items()
    }


def _stem_mean_for_system(tok, model, layers, system_prompt: str, device: str, max_gpu_gb: float) -> np.ndarray:
    n_layers = len(layers)
    n_units = layers[0].mlp.down_proj.in_features
    sums = np.zeros((n_layers, n_units), dtype=np.float64)
    counts = np.zeros(n_layers, dtype=np.int64)
    sequences = build_sequences(tok, system_prompt)
    for sequence in sequences:
        ids = torch.tensor([sequence['input_ids']], dtype=torch.long, device=actual_device(model))
        capture = [{} for _ in layers]
        handles = []
        for li, layer in enumerate(layers):
            def hook(_mod, args, li=li):
                capture[li]['x'] = args[0].detach()
                return None
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
        try:
            with torch.no_grad():
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            positions = torch.tensor(sequence['stem_positions'], dtype=torch.long, device=actual_device(model))
            for li in range(n_layers):
                acts = capture[li]['x'][0].index_select(0, positions).float().sum(dim=0).cpu().numpy()
                sums[li] += acts
                counts[li] += len(sequence['stem_positions'])
            nr.spill_guard(device, max_gpu_gb)
        finally:
            for handle in handles:
                handle.remove()
    return (sums / counts[:, None]).astype(np.float32)


def run_means(args, tok, model, layers) -> dict:
    os.makedirs(args.out, exist_ok=True)
    system_means = {}
    for target, system_prompt in SYSTEM_MESSAGES.items():
        print(f'means: collecting {target} stem activations', flush=True)
        system_means[target] = _stem_mean_for_system(tok, model, layers, system_prompt, args.device, args.max_gpu_gb)
    npz_path = os.path.join(args.out, 'means.npz')
    np.savez_compressed(npz_path, **{f'{key}_means': val for key, val in system_means.items()})
    baseline_answers = _all_answers(tok, model, layers, [], 'zero', None, args.device, args.max_gpu_gb)
    obj = header(args, source_shas())
    obj.update({
        'system_messages': SYSTEM_MESSAGES,
        'neuron_definition': 'MLP neuron is one input coordinate to down_proj; mean activation is averaged over stem tokens only across 24 sequences.',
        'activation_means': {'file': 'means.npz', 'format': 'compressed numpy arrays named <target>_means',
                             'shape': list(next(iter(system_means.values())).shape), 'dtype': 'float32'},
        'answers_unsteered': baseline_answers,
    })
    path = os.path.join(args.out, 'means.json')
    write_json(obj, path)
    obj['_file_sha'] = nr._file_sha(path)
    return obj


def _padded_grad_batch(tok, sequences, model):
    ids, mask, positions = _pad_batch(sequences, tok, actual_device(model))
    embeddings = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    return ids, mask, positions, embeddings


def estimate_candidates(tok, model, layers, sequences, token_records, target: str,
                        values: np.ndarray, max_gpu_gb: float) -> np.ndarray:
    """Sum margin-gradient times (constant new activation minus current) for every neuron."""
    ids, mask, positions, embeddings = _padded_grad_batch(tok, sequences, model)
    captures = [{} for _ in layers]
    handles = []
    for li, layer in enumerate(layers):
        def hook(_mod, args, li=li):
            x = args[0]
            x.retain_grad()
            captures[li]['x'] = x
            return None
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
    try:
        with torch.enable_grad():
            out = model(inputs_embeds=embeddings, attention_mask=mask, use_cache=False)
            margins = _score_margins(out.logits, sequences, positions, token_records, target)
            model.zero_grad(set_to_none=True)
            margins.sum().backward()
        estimates = []
        for li in range(len(layers)):
            x = captures[li]['x'].detach().float()
            grad = captures[li]['x'].grad.detach().float()
            destination = torch.as_tensor(values[li], dtype=x.dtype, device=x.device)[None, None, :]
            estimates.append(first_order_estimate(grad, x, destination))
        nr.spill_guard(str(actual_device(model).type), max_gpu_gb)
        return np.stack(estimates).astype(np.float64)
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)


def run_candidates(args, tok, model, layers, means_obj=None, token_records=None) -> dict:
    means_obj = means_obj or _read_json(os.path.join(args.out, 'means.json'))
    token_records = token_records or resolve_token_sets(tok)
    means = load_means(args.out, means_obj)
    sequences = build_sequences(tok)
    n_layers = len(layers)
    n_units = layers[0].mlp.down_proj.in_features
    zero = np.zeros((n_layers, n_units), dtype=np.float32)
    rankings = {}
    for search_target in SEARCH_TARGETS:
        target = 'google' if search_target == 'google-removal' else search_target
        operation = 'zero' if search_target == 'google-removal' else 'steer'
        destination = zero if operation == 'zero' else means[target]
        estimates = estimate_candidates(tok, model, layers, sequences, token_records, target, destination, args.max_gpu_gb)
        flat = [(float(estimates[li, unit]), li, unit) for li in range(n_layers) for unit in range(n_units)]
        flat.sort(key=lambda row: (-row[0], row[1], row[2]))
        rankings[search_target] = [
            {'layer': li, 'index': unit, 'estimate': score} for score, li, unit in flat[:200]
        ]
        print(f'candidates: ranked {search_target} ({len(rankings[search_target])} neurons)', flush=True)
    path = os.path.join(args.out, 'candidates.json')
    obj = header(args, {'means.json': nr._file_sha(os.path.join(args.out, 'means.json')),
                        'means.npz': nr._file_sha(os.path.join(args.out, 'means.npz')),
                        'neuron_report.py': source_shas()['neuron_report.py'],
                        'circuit.py': source_shas()['circuit.py']})
    obj.update({'candidate_count': 200, 'estimate': 'sum over 24 sequences and all real token positions of d(target margin) * (constant new value - current activation)',
                'token_sets': token_records, 'rankings': rankings})
    write_json(obj, path)
    obj['_file_sha'] = nr._file_sha(path)
    return obj


def load_means(out: str, means_obj: dict) -> dict[str, np.ndarray]:
    file = means_obj.get('activation_means', {}).get('file', 'means.npz')
    with np.load(os.path.join(out, file)) as data:
        return {target: np.asarray(data[f'{target}_means'], dtype=np.float32) for target in SYSTEM_MESSAGES}


def _read_json(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise SystemExit(f'{os.path.basename(path)} is malformed')
    return obj


def _target_ops(neurons, target_name, means):
    op = 'zero' if target_name == 'google-removal' else 'steer'
    return _operations(neurons, op, None if op == 'zero' else means[target_name])


def evaluate_added(candidates, current, evaluator: Callable, batch_size: int):
    """Evaluate candidate additions in bounded batches and return (best candidate, margin)."""
    remaining = [c for c in candidates if c not in current]
    if not remaining:
        return None, None
    scored = []
    for start in range(0, len(remaining), batch_size):
        chunk = remaining[start:start + batch_size]
        sets = [list(current) + [candidate] for candidate in chunk]
        scores = evaluator(sets)
        scored.extend((float(score), candidate) for score, candidate in zip(scores, chunk))
    # Stable tie break follows candidate ranking order.
    best_score, best_candidate = max(scored, key=lambda pair: pair[0])
    return best_candidate, best_score


def greedy_select(candidates, evaluator: Callable, success: Callable, max_members: int = 60,
                  batch_size: int = 8, on_step: Callable | None = None):
    current = []
    steps = []
    succeeded = False
    for step_no in range(1, max_members + 1):
        chosen, margin = evaluate_added(candidates, current, evaluator, batch_size)
        if chosen is None:
            break
        current.append(chosen)
        success_value = success(current)
        if isinstance(success_value, tuple):
            succeeded, detail = bool(success_value[0]), success_value[1]
        else:
            succeeded, detail = bool(success_value), {}
        row = {'step': step_no, 'chosen_neuron': {'layer': int(chosen[0]), 'index': int(chosen[1])},
               'mean_margin': float(margin), 'success': succeeded, 'success_answers': detail}
        steps.append(row)
        if on_step is not None:
            on_step(step_no, current, row)
        if succeeded:
            break
    return current, steps, succeeded


def backward_prune(members, success: Callable):
    current = list(members)
    dropped = []
    for member in reversed(members):
        trial = [x for x in current if x != member]
        result = success(trial)
        ok = result[0] if isinstance(result, tuple) else result
        if ok:
            current = trial
            dropped.append(member)
    return current, dropped


def _record_pruning_order(name, seed, order, members, dropped, target_name, check_success,
                          tok, model, layers, means, args, target_system_prompt,
                          ranges_no_system, ranges_system):
    operation = 'zero' if target_name == 'google-removal' else 'steer'
    target = 'google' if target_name == 'google-removal' else target_name
    success_result = check_success(members)
    success_holds = bool(success_result[0]) if isinstance(success_result, tuple) else bool(success_result)
    success_answers = success_result[1] if isinstance(success_result, tuple) else {}
    audit = single_drop_audit(members, check_success, success_result)
    keys = [f'{li}:{unit}' for li, unit in members]
    member_ranges_no = {key: ranges_no_system[key] for key in keys}
    member_ranges_system = {key: ranges_system[key] for key in keys}
    scales = [1.0] if operation == 'zero' else [0.5, 1.0, 2.0]
    dose = []
    for scale in scales:
        base_ops = _target_ops(members, target_name, means)
        scaled_ops = base_ops if operation == 'zero' else {key: value * scale for key, value in base_ops.items()}
        answers = _greedy_generate(tok, model, layers, SUCCESS_QUESTIONS[target_name], scaled_ops,
                                   args.device, args.max_gpu_gb)
        dose.append({'scale': scale, 'operation': operation, 'answers': answers,
                     'success': target_success(target_name, answers)})
    return {
        'name': name, 'shuffle_seed': seed,
        'order': [{'layer': l, 'index': u} for l, u in order],
        'neurons': [{'layer': l, 'index': u} for l, u in members],
        'dropped_neurons': [{'layer': l, 'index': u} for l, u in dropped],
        'success': success_holds,
        'success_check_answers': success_answers,
        'single_drop_audit': audit,
        'natural_activation_ranges': {'no_system': member_ranges_no, 'target_system': member_ranges_system},
        'dose_checks': dose,
    }


def run_search(args, target_name, tok, model, layers, means, candidate_obj, means_obj) -> dict:
    candidate_rows = candidate_obj['rankings'].get(target_name)
    if not isinstance(candidate_rows, list):
        raise SystemExit(f'candidates.json has no ranking for {target_name}')
    candidates = [(int(x['layer']), int(x['index'])) for x in candidate_rows]
    sequences = build_sequences(tok)
    token_records = candidate_obj['token_sets']
    target = 'google' if target_name == 'google-removal' else target_name
    operation = 'zero' if target_name == 'google-removal' else 'steer'
    system_means = None if operation == 'zero' else means[target]

    def evaluator(sets):
        return evaluate_sets(tok, model, layers, sequences, token_records, target, sets,
                             operation, system_means, args.max_gpu_gb)

    def ops_for(neurons):
        return _target_ops(neurons, target_name, means)

    def check_success(neurons):
        answers = _greedy_generate(tok, model, layers, SUCCESS_QUESTIONS[target_name],
                                   ops_for(neurons), args.device, args.max_gpu_gb)
        return target_success(target_name, answers), answers

    checkpoints = []
    foreign = []
    def on_step(step_no, members, step_row):
        print(f'{target_name} step {step_no} set_size={len(members)} margin={step_row["mean_margin"]:.6f}', flush=True)
        for question, rec in step_row['success_answers'].items():
            match = FOREIGN_MAKERS.search(rec.get('first_sentence', ''))
            if match:
                foreign.append({'step': step_no, 'question': question, 'name': match.group(0), 'source': 'success_check'})
        if step_no % 5 == 0:
            answers = _all_answers(tok, model, layers, members, operation, system_means, args.device, args.max_gpu_gb)
            for question, rec in answers.items():
                match = FOREIGN_MAKERS.search(rec.get('first_sentence', ''))
                if match:
                    foreign.append({'step': step_no, 'question': question, 'name': match.group(0), 'source': 'six_question_checkpoint'})
            checkpoints.append({'step': step_no, 'answers': answers})

    members, steps, success = greedy_select(candidates, evaluator, check_success,
                                             max_members=60, batch_size=args.batch_size, on_step=on_step)
    if members and (not checkpoints or checkpoints[-1]['step'] != len(members)):
        answers = _all_answers(tok, model, layers, members, operation, system_means, args.device, args.max_gpu_gb)
        for question, rec in answers.items():
            match = FOREIGN_MAKERS.search(rec.get('first_sentence', ''))
            if match:
                foreign.append({'step': len(members), 'question': question, 'name': match.group(0), 'source': 'six_question_checkpoint'})
        checkpoints.append({'step': len(members), 'answers': answers})

    pruned, dropped = backward_prune(members, check_success)
    pruning_specs = [('addition_order', None, list(members), list(pruned), list(dropped))]
    for seed in range(5):
        order = list(members)
        np.random.default_rng(seed).shuffle(order)
        alt_pruned, alt_dropped = backward_prune(order, check_success)
        pruning_specs.append((f'shuffled_{seed}', seed, order, alt_pruned, alt_dropped))
    all_pruned = sorted({neuron for _, _, _, subset, _ in pruning_specs for neuron in subset})
    ranges_no_system = activation_ranges(tok, model, layers, all_pruned, None, args.device, args.max_gpu_gb)
    ranges_system = activation_ranges(tok, model, layers, all_pruned, SYSTEM_MESSAGES[target],
                                      args.device, args.max_gpu_gb)
    pruning_records = [
        _record_pruning_order(name, seed, order, subset, removed, target_name, check_success,
                              tok, model, layers, means, args, SYSTEM_MESSAGES[target],
                              ranges_no_system, ranges_system)
        for name, seed, order, subset, removed in pruning_specs
    ]
    primary_prune = pruning_records[0]
    prune_success = primary_prune['success']
    final_margin = evaluator([pruned])[0]
    final_answers = _all_answers(tok, model, layers, pruned, operation, system_means, args.device, args.max_gpu_gb)
    for question, rec in final_answers.items():
        match = FOREIGN_MAKERS.search(rec.get('first_sentence', ''))
        if match:
            foreign.append({'step': 'pruned', 'question': question, 'name': match.group(0), 'source': 'pruned_answers'})

    filename = f'search-{target_name}.json'
    path = os.path.join(args.out, filename)
    final_correlate = _greedy_generate(tok, model, layers, CORRELATE_PROMPTS,
                                       _target_ops(pruned, target_name, means), args.device,
                                       args.max_gpu_gb, max_new_tokens=24)
    positive_correlate = _greedy_generate(tok, model, layers, CORRELATE_PROMPTS, {}, args.device,
                                          args.max_gpu_gb, system_prompt=SYSTEM_MESSAGES[target],
                                          max_new_tokens=24)
    obj = header(args, {
        'candidates.json': nr._file_sha(os.path.join(args.out, 'candidates.json')),
        'means.json': nr._file_sha(os.path.join(args.out, 'means.json')),
        'means.npz': nr._file_sha(os.path.join(args.out, 'means.npz')),
    })
    obj.update({
        'target': target_name,
        'operation': 'zero selected neurons at every token position' if operation == 'zero'
                     else 'clamp selected neurons at every token position to their target system-run stem means',
        'candidate_count': len(candidates),
        'batch_size_candidates': args.batch_size,
        'steps': steps,
        'success': success,
        'greedy_selected': [{'layer': l, 'index': u} for l, u in members],
        'checkpoints_every_5_and_end': checkpoints,
        'foreign_makers_seen': bool(foreign),
        'foreign_maker_occurrences': foreign,
        'pruning_orders': pruning_records,
        'pruned': {
            **primary_prune,
            'mean_margin': float(final_margin),
            'answers_six_questions': final_answers,
        },
        'correlate': {
            'prompts': CORRELATE_PROMPTS,
            'final_set': final_correlate,
            'positive_control_system_run': {'target': target, 'system_message': SYSTEM_MESSAGES[target],
                                            'answers': positive_correlate},
        },
    })
    write_json(obj, path)
    return obj


def _matched_random(pruned: list[tuple[int, int]], n_units: int, n_layers: int, seed: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    chosen = set(pruned)
    out = []
    counts: dict[int, int] = {}
    for layer, _ in pruned:
        counts[layer] = counts.get(layer, 0) + 1
    for layer in sorted(counts):
        pool = [unit for unit in range(n_units) if (layer, unit) not in chosen]
        if len(pool) < counts[layer]:
            raise ValueError(f'not enough unmatched units in layer {layer} for a control set')
        out.extend((layer, int(unit)) for unit in rng.choice(pool, size=counts[layer], replace=False))
    return out


def run_controls(args, tok, model, layers, means, search_targets=None) -> dict:
    search_targets = search_targets or SEARCH_TARGETS
    search_objs = {}
    for target_name in search_targets:
        path = os.path.join(args.out, f'search-{target_name}.json')
        if os.path.isfile(path):
            search_objs[target_name] = _read_json(path)
    token_records = resolve_token_sets(tok)
    n_units = layers[0].mlp.down_proj.in_features
    random_controls = {}
    for target_name, search_obj in search_objs.items():
        pruned_obj = search_obj.get('pruned', {})
        if not pruned_obj.get('success'):
            continue
        pruned = [(int(x['layer']), int(x['index'])) for x in pruned_obj.get('neurons', [])]
        if not pruned:
            continue
        target = 'google' if target_name == 'google-removal' else target_name
        operation = 'zero' if target_name == 'google-removal' else 'steer'
        system_means = None if operation == 'zero' else means[target]
        results = []
        for trial in range(20):
            seed = args.seed + trial
            random_set = _matched_random(pruned, n_units, len(layers), seed)
            answers = _greedy_generate(tok, model, layers, SUCCESS_QUESTIONS[target_name],
                                       _operations(random_set, operation, system_means), args.device, args.max_gpu_gb)
            results.append({'trial': trial, 'seed': seed,
                            'neurons': [{'layer': l, 'index': u} for l, u in random_set],
                            'answers': answers, 'success': target_success(target_name, answers)})
        random_controls[target_name] = {'matched_to': [{'layer': l, 'count': sum(1 for x, _ in pruned if x == l)}
                                                       for l in sorted({x for x, _ in pruned})],
                                        'operation': operation, 'sets': results}

    qwen_steering = {}
    for target_name in ('openai', 'anthropic', 'deepseek', 'google'):
        if target_name not in search_targets:
            continue
        path = os.path.join(args.out, f'search-{target_name}.json')
        if not os.path.isfile(path):
            continue
        search_obj = _read_json(path)
        pruned = [(int(x['layer']), int(x['index'])) for x in search_obj.get('pruned', {}).get('neurons', [])]
        answers = _all_answers(tok, model, layers, pruned, 'steer', means['qwen'], args.device, args.max_gpu_gb)
        qwen_steering[target_name] = {
            'neurons': [{'layer': l, 'index': u} for l, u in pruned],
            'operation': 'clamp to the neuron means from the Qwen system run',
            'answers_six_questions': answers,
        }

    input_shas = {f'search-{key}.json': nr._file_sha(os.path.join(args.out, f'search-{key}.json'))
                  for key in search_objs}
    input_shas.update({'means.json': nr._file_sha(os.path.join(args.out, 'means.json')),
                       'means.npz': nr._file_sha(os.path.join(args.out, 'means.npz'))})
    obj = header(args, input_shas)
    obj.update({'random_seed_base': args.seed, 'random_seed_rule': 'seed + trial index (0..19)',
                'random_controls': random_controls, 'qwen_system_steering': qwen_steering})
    write_json(obj, os.path.join(args.out, 'controls.json'))
    return obj


def add_shared_args(parser):
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--device', default='xpu')
    parser.add_argument('--max-gpu-gb', type=float, default=14.5)
    parser.add_argument('--out', required=True)
    parser.add_argument('--targets', default=','.join(SEARCH_TARGETS),
                        help='comma-separated search targets: google-removal,openai,anthropic,deepseek,google')
    parser.add_argument('--batch-size', type=int, default=8, help='candidate sets per forward batch')
    parser.add_argument('--seed', type=int, default=0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', choices=['run', 'means', 'candidates', 'search', 'controls'], default='run')
    parser.add_argument('--steps', default='means,candidates,search,controls',
                        help='ordered stages for run: means,candidates,search,controls')
    add_shared_args(parser)
    return parser.parse_args(argv)


def _parse_csv(value: str, allowed: list[str], label: str) -> list[str]:
    got = [x.strip() for x in value.split(',') if x.strip()]
    bad = [x for x in got if x not in allowed]
    if bad:
        raise SystemExit(f'unknown {label}: {", ".join(bad)}')
    return got


def main(argv=None):
    args = parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit('--batch-size must be positive')
    os.makedirs(args.out, exist_ok=True)
    stages = _parse_csv(args.steps, ['means', 'candidates', 'search', 'controls'], '--steps') if args.command == 'run' else [args.command]
    targets = _parse_csv(args.targets, SEARCH_TARGETS, '--targets')
    if 'search' not in stages and args.command == 'search':
        targets = _parse_csv(args.targets, SEARCH_TARGETS, '--targets')
    tok, model = nr.load_model(args.model, args.revision, args.device)
    model.requires_grad_(False)
    _, layers, _ = nr.find_text_layers(model)
    nr.spill_guard(args.device, args.max_gpu_gb)
    token_records = None
    means_obj = None
    candidate_obj = None
    for stage in stages:
        if stage == 'means':
            means_obj = run_means(args, tok, model, layers)
        elif stage == 'candidates':
            if means_obj is None:
                means_obj = _read_json(os.path.join(args.out, 'means.json'))
            token_records = resolve_token_sets(tok)
            candidate_obj = run_candidates(args, tok, model, layers, means_obj, token_records)
        elif stage == 'search':
            if means_obj is None:
                means_obj = _read_json(os.path.join(args.out, 'means.json'))
            if candidate_obj is None:
                candidate_obj = _read_json(os.path.join(args.out, 'candidates.json'))
            means = load_means(args.out, means_obj)
            for target_name in targets:
                print(f'search: starting {target_name}', flush=True)
                run_search(args, target_name, tok, model, layers, means, candidate_obj, means_obj)
        elif stage == 'controls':
            if means_obj is None:
                means_obj = _read_json(os.path.join(args.out, 'means.json'))
            means = load_means(args.out, means_obj)
            run_controls(args, tok, model, layers, means, targets)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
