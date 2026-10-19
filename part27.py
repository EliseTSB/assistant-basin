#!/usr/bin/env python3
"""Measure the four Part 25 identity neurons and matched controls (Part 27)."""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

import circuit
import neuron_report as nr
import part24
import part25
import part26


CHAT_MODEL = 'Qwen/Qwen3.5-4B'
CHAT_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
BASE_MODEL = 'Qwen/Qwen3.5-4B-Base'
BASE_REVISION = '1001bb4d826a52d1f399e183466143f4da7b741b'
TARGET_NEURONS = [
    {'layer': 6, 'unit': 753},
    {'layer': 7, 'unit': 1169},
    {'layer': 11, 'unit': 8868},
    {'layer': 29, 'unit': 776},
]
CONTROL_SEEDS = tuple(range(20))
N_WORDS = 256
N_FIRE_DOCS = 1000
N_COST_DOCS = 200
FACTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/part27-facts.json')
MAKER_RE = re.compile(r'\b(?:Qwen|Tongyi|Alibaba|Google|Gemini|OpenAI|GPT|Anthropic|Claude|DeepSeek)\b', re.I)


def pair_key(pair: tuple[int, int] | dict) -> str:
    if isinstance(pair, dict):
        layer, unit = int(pair['layer']), int(pair.get('unit', pair.get('index')))
    else:
        layer, unit = map(int, pair)
    return f'{layer}:{unit}'


def pairs(records: list[dict]) -> list[tuple[int, int]]:
    return [(int(row['layer']), int(row.get('unit', row.get('index')))) for row in records]


def _sha(path: str) -> str:
    return nr._file_sha(os.path.abspath(path))


def _input_shas(extra: dict[str, str] | None = None) -> dict[str, str]:
    paths = {
        'PREDICTIONS.md': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PREDICTIONS.md'),
        'part24.py': part24.__file__, 'part25.py': part25.__file__, 'part26.py': part26.__file__,
        'neuron_report.py': nr.__file__, 'circuit.py': circuit.__file__,
        'part20.py': part26.p20.__file__, 'part20c.py': part26.p20c.__file__,
        'part20d.py': part26.p20d.__file__, 'part20f.py': part26.part20f.__file__,
        'part27.py': __file__,
    }
    if extra:
        paths.update(extra)
    return {name: _sha(path) for name, path in paths.items()}


def header(model: str, revision: str, step: str, input_shas: dict[str, str]) -> dict:
    """Use Part 25's provenance header and extend it with this step's name."""
    obj = part25.header(SimpleNamespace(model=model, revision=revision), input_shas)
    obj['tool'] = 'part27.py (assistant-basin, Part 27)'
    obj['tool_sha256'] = _sha(__file__)
    obj['step'] = step
    return obj


def select_documents(corpus: str, count: int, skip_eligible: int = 0) -> list[dict]:
    """Select Pile documents with Part 26's whitespace and first-256-word rules.

    ``skip_eligible`` counts qualifying documents, not JSONL rows. Returned line indices
    always refer to original JSONL line numbers.
    """
    if count < 1 or skip_eligible < 0:
        raise ValueError('document count must be positive and skip_eligible nonnegative')
    documents = []
    eligible = 0
    with open(corpus, encoding='utf-8') as stream:
        for line_index, line in enumerate(stream):
            row = json.loads(line)
            words = str(row.get('text', '')).split()
            if len(words) < N_WORDS:
                continue
            if eligible >= skip_eligible:
                documents.append({'line_index': line_index, 'text': ' '.join(words[:N_WORDS])})
                if len(documents) == count:
                    break
            eligible += 1
    if len(documents) != count:
        raise ValueError(f'corpus has only {len(documents)} eligible documents after skipping {skip_eligible}; need {count}')
    return documents


def control_sets(layers, target_specs: list[dict] | None = None, n_controls: int = 20) -> list[dict]:
    """Draw one non-target unit per target layer, using fixed seeds 0..n_controls-1."""
    specs = TARGET_NEURONS if target_specs is None else target_specs
    layer_order = sorted({int(row['layer']) for row in specs})
    target_by_layer: dict[int, set[int]] = defaultdict(set)
    for row in specs:
        target_by_layer[int(row['layer'])].add(int(row.get('unit', row.get('index'))))
    result = []
    seeds = CONTROL_SEEDS[:n_controls] if n_controls <= len(CONTROL_SEEDS) else range(n_controls)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        members = []
        for layer_i in layer_order:
            width = int(layers[layer_i].mlp.down_proj.in_features)
            available = np.setdiff1d(np.arange(width, dtype=np.int64),
                                     np.asarray(sorted(target_by_layer[layer_i]), dtype=np.int64))
            if not len(available):
                raise ValueError(f'no control units available in layer {layer_i}')
            members.append({'layer': layer_i, 'unit': int(rng.choice(available))})
        result.append({'seed': seed, 'neurons': members})
    return result


def _control_pairs(controls: list[dict]) -> list[list[tuple[int, int]]]:
    return [pairs(row['neurons']) for row in controls]


def _unit_roles(specs: list[dict], controls: list[dict]) -> dict[tuple[int, int], dict]:
    roles: dict[tuple[int, int], dict] = {}
    for spec in specs:
        pair = (int(spec['layer']), int(spec.get('unit', spec.get('index'))))
        roles[pair] = {'target': True, 'control_seeds': []}
    for control in controls:
        for layer_i, unit in pairs(control['neurons']):
            roles.setdefault((layer_i, unit), {'target': False, 'control_seeds': []})
            roles[(layer_i, unit)]['control_seeds'].append(int(control['seed']))
    return roles


def _encode(tok, text: str) -> list[int]:
    return [int(value) for value in tok(text, add_special_tokens=False)['input_ids']]


def _all_pairs(specs: list[dict], controls: list[dict]) -> list[tuple[int, int]]:
    values = set(pairs(specs))
    for row in controls:
        values.update(pairs(row['neurons']))
    return sorted(values)


def _capture_documents(tok, model, layers, docs: list[dict], all_units: list[tuple[int, int]],
                       target_specs: list[dict], batch_size: int, device: str,
                       max_gpu_gb: float) -> tuple[dict, np.ndarray]:
    """Capture each unit's per-document token maximum and target per-word values."""
    by_layer: dict[int, list[int]] = defaultdict(list)
    for layer_i, unit in all_units:
        by_layer[layer_i].append(unit)
    by_layer = {layer_i: sorted(set(units)) for layer_i, units in by_layer.items()}
    tokenized = [part26._tokenize_document(tok, row['text']) for row in docs]
    hooks = {layer_i: nr.NeuronHook(layers[layer_i].mlp.down_proj, units)
             for layer_i, units in by_layer.items()}
    col_by_pair = {(layer_i, unit): col for layer_i, units in by_layer.items()
                   for col, unit in enumerate(units)}
    summaries = {pair: {'maxima': [], 'contexts': []} for pair in all_units}
    target_index = {(int(row['layer']), int(row.get('unit', row.get('index')))): i
                    for i, row in enumerate(target_specs)}
    word_acts = np.zeros((len(target_specs), len(docs), N_WORDS), dtype=np.float32)
    try:
        for start in range(0, len(docs), batch_size):
            end = min(len(docs), start + batch_size)
            batch = tokenized[start:end]
            ids, mask = part26._padded_inputs(batch, tok, device)
            for hook in hooks.values():
                hook.captured.clear()
            part26._forward_decoder(model, ids, mask)
            for layer_i, hook in hooks.items():
                if len(hook.captured) != 1:
                    raise RuntimeError(f'layer {layer_i} captured {len(hook.captured)} calls; expected one')
            for local_doc, item in enumerate(batch):
                token_count = len(item['ids'])
                for pair in all_units:
                    layer_i, unit = pair
                    column = col_by_pair[pair]
                    values = hooks[layer_i].captured[0][local_doc, :token_count, column].numpy().astype(np.float32)
                    position = int(np.argmax(values))
                    summaries[pair]['maxima'].append({
                        'line_index': int(docs[start + local_doc]['line_index']),
                        'activation': float(values[position]), 'token_position': position,
                    })
                    if pair in target_index:
                        pooled = part26.word_activations(item["offsets"], item["spans"], values)[:, 0]
                        if pooled.shape != (N_WORDS,):
                            raise ValueError(f'expected 256 word activations, got {pooled.shape}')
                        word_acts[target_index[pair], start + local_doc] = pooled
            nr.spill_guard(device, max_gpu_gb)
    finally:
        for hook in hooks.values():
            hook.remove()

    for pair, rows in summaries.items():
        ranked = sorted(enumerate(rows['maxima']), key=lambda item: (-item[1]['activation'], item[0]))[:40]
        contexts = []
        for doc_i, row in ranked:
            item = tokenized[doc_i]
            pos = int(row['token_position'])
            window_len = min(15, len(item['ids']))
            lo = max(0, min(pos - 7, len(item['ids']) - window_len))
            hi = lo + window_len
            token_ids = item['ids'][lo:hi]
            contexts.append({
                **row, 'document_rank': len(contexts) + 1, 'token_window_start': lo,
                'token_window_end': hi, 'window_token_ids': token_ids,
                'max_token_offset_in_window': pos - lo,
                'window_tokens': [tok.decode([int(value)], skip_special_tokens=True) for value in token_ids],
                'decoded_window': tok.decode(token_ids, skip_special_tokens=True),
                'max_token': tok.decode([int(item['ids'][pos])], skip_special_tokens=True),
            })
        rows['contexts'] = contexts
    return summaries, word_acts


def _prompt_rows(tok, questions: list[str], systems: dict[str, str] | None = None) -> list[dict]:
    rows = []
    systems = systems or {}
    for question in questions:
        rows.append({'question': question, 'system_prompt': None,
                     'prompt': nr.build_prompt(tok, question, False)})
        for system_name, system_prompt in systems.items():
            rows.append({'question': question, 'system_prompt_name': system_name,
                         'system_prompt': system_prompt,
                         'prompt': nr.build_prompt(tok, question, False, system_prompt=system_prompt)})
    return rows


def _capture_prompts(tok, model, layers, prompt_rows: list[dict], units: list[tuple[int, int]],
                     device: str, max_gpu_gb: float) -> list[dict]:
    by_layer: dict[int, list[int]] = defaultdict(list)
    for layer_i, unit in units:
        by_layer[layer_i].append(unit)
    by_layer = {layer_i: sorted(set(values)) for layer_i, values in by_layer.items()}
    hooks = {layer_i: nr.NeuronHook(layers[layer_i].mlp.down_proj, values)
             for layer_i, values in by_layer.items()}
    output = []
    try:
        for row in prompt_rows:
            ids_list = _encode(tok, row['prompt'])
            ids = torch.tensor([ids_list], dtype=torch.long, device=next(model.parameters()).device)
            mask = torch.ones_like(ids)
            for hook in hooks.values():
                hook.captured.clear()
            with torch.no_grad():
                part26._decoder(model)(input_ids=ids, attention_mask=mask, use_cache=False)
            values = {}
            for layer_i, hook in hooks.items():
                if len(hook.captured) != 1:
                    raise RuntimeError(f'prompt hook on layer {layer_i} captured {len(hook.captured)} calls')
                for column, unit in enumerate(by_layer[layer_i]):
                    values[pair_key((layer_i, unit))] = [float(x) for x in hook.captured[0][0, :, column].tolist()]
            output.append({
                'question': row['question'], 'system_prompt_name': row.get('system_prompt_name'),
                'system_prompt': row['system_prompt'], 'token_ids': ids_list,
                'tokens': [tok.decode([int(value)], skip_special_tokens=True) for value in ids_list],
                'decoded_prompt': tok.decode(ids_list, skip_special_tokens=True),
                'activation_by_neuron': values,
            })
            nr.spill_guard(device, max_gpu_gb)
    finally:
        for hook in hooks.values():
            hook.remove()
    return output


def _require_layout(layers, specs: list[dict]) -> None:
    for row in specs:
        layer_i, unit = int(row['layer']), int(row.get('unit', row.get('index')))
        if layer_i < 0 or layer_i >= len(layers):
            raise ValueError(f'target layer {layer_i} is outside the model')
        width = int(layers[layer_i].mlp.down_proj.in_features)
        if unit < 0 or unit >= width:
            raise ValueError(f'target unit {unit} is outside layer {layer_i} width {width}')


def run_fires(args, tok, model, layers, specs: list[dict] | None = None,
              controls: list[dict] | None = None, docs: list[dict] | None = None) -> dict:
    specs = TARGET_NEURONS if specs is None else specs
    controls = control_sets(layers, specs, getattr(args, 'n_controls', 20)) if controls is None else controls
    _require_layout(layers, specs)
    docs = select_documents(args.corpus, getattr(args, 'fire_docs', N_FIRE_DOCS)) if docs is None else docs
    if not os.path.isdir(args.out):
        os.makedirs(args.out, exist_ok=True)
    all_units = _all_pairs(specs, controls)
    summaries, word_acts = _capture_documents(tok, model, layers, docs, all_units, specs,
                                               args.batch_size, args.device, args.max_gpu_gb)
    roles = _unit_roles(specs, controls)
    neuron_rows = {}
    for pair in all_units:
        row = roles[pair]
        neuron_rows[pair_key(pair)] = {
            'layer': pair[0], 'unit': pair[1], 'target': row['target'],
            'control_seeds': row['control_seeds'],
            'document_maxima': summaries[pair]['maxima'],
            'top_40_contexts': summaries[pair]['contexts'],
        }
    prompt_rows = _prompt_rows(tok, circuit.ID_PROMPTS, part25.SYSTEM_MESSAGES)
    identity_rows = _capture_prompts(tok, model, layers, prompt_rows, all_units,
                                     args.device, args.max_gpu_gb)
    npz_name = 'fires-word-activations.npz'
    npz_path = os.path.join(args.out, npz_name)
    np.savez_compressed(npz_path, word_activations=word_acts,
                        line_indices=np.asarray([row['line_index'] for row in docs], dtype=np.int64),
                        neuron_pairs=np.asarray(pairs(specs), dtype=np.int64))
    inputs = _input_shas({'corpus': args.corpus})
    out = header(args.model, args.revision, 'fires', inputs)
    out.update({
        'neuron_definition': 'one input unit of a layer MLP down_proj',
        'document_selection': f'first {len(docs)} qualifying JSONL rows after at least 256 whitespace-separated words, first 256 words joined by single spaces',
        'documents': [{'line_index': int(row['line_index']), 'word_count': len(row['text'].split())} for row in docs],
        'target_neurons': specs,
        'controls': controls,
        'largest_token_activation_by_neuron': neuron_rows,
        'identity_prompts': {'questions': circuit.ID_PROMPTS, 'system_messages': part25.SYSTEM_MESSAGES,
                             'rows': identity_rows},
        'word_activations': {'file': npz_name, 'shape': list(word_acts.shape),
                             'dtype': 'float32', 'neuron_order': [pair_key(pair) for pair in pairs(specs)]},
    })
    part25.write_json(out, os.path.join(args.out, 'fires.json'))
    return out


def run_writes(args, tok, model, layers, norm=None, specs: list[dict] | None = None,
               controls: list[dict] | None = None) -> dict:
    specs = TARGET_NEURONS if specs is None else specs
    controls = control_sets(layers, specs, getattr(args, 'n_controls', 20)) if controls is None else controls
    _require_layout(layers, specs)
    if norm is None:
        _lname, found_layers, norm = nr.find_text_layers(model)
        if found_layers is not layers:
            raise ValueError('layers do not match neuron_report.find_text_layers(model)')
    roles = _unit_roles(specs, controls)
    grouped: dict[int, list[int]] = defaultdict(list)
    for layer_i, unit in roles:
        grouped[layer_i].append(unit)
    cache: dict[int, str] = {}
    rows = {}
    for layer_i, units in sorted(grouped.items()):
        down_proj = layers[layer_i].mlp.down_proj
        top_ids = part26.logit_lens_top_ids(tok, model, norm, down_proj, units, top=30,
                                            device=args.device, max_gpu_gb=args.max_gpu_gb)
        # A negated down_proj column has exactly the bottom-token ordering for the original direction.
        bottom_ids = part26.logit_lens_top_ids(tok, model, norm,
                                              SimpleNamespace(weight=-down_proj.weight), units, top=30,
                                              device=args.device, max_gpu_gb=args.max_gpu_gb)
        for col, unit in enumerate(units):
            pair = (layer_i, unit)
            top = [int(value) for value in top_ids[col].tolist()]
            bottom = [int(value) for value in bottom_ids[col].tolist()]
            top_raw = [tok.decode([value]) for value in top]
            bottom_raw = [tok.decode([value]) for value in bottom]
            rows[pair_key(pair)] = {
                'layer': layer_i, 'unit': unit, 'target': roles[pair]['target'],
                'control_seeds': roles[pair]['control_seeds'],
                'top_ids': top, 'top_decoded': top_raw,
                'top_tokens': [part26.normalize_write_token(value) for value in top_raw],
                'bottom_ids': bottom, 'bottom_decoded': bottom_raw,
                'bottom_tokens': [part26.normalize_write_token(value) for value in bottom_raw],
            }
    out = header(args.model, args.revision, 'writes', _input_shas())
    out.update({
        'method': 'logit lens as in Part 26: unembedding @ (final-norm gain * down_proj column), highest and lowest 30; token string lists are decoded, stripped of leading whitespace, and lowercased',
        'neuron_definition': 'one input unit of a layer MLP down_proj',
        'target_neurons': specs, 'controls': controls, 'units': rows,
    })
    part25.write_json(out, os.path.join(args.out, 'writes.json'))
    return out


def next_token_loss_terms(logits: torch.Tensor, input_ids: torch.Tensor,
                          attention_mask: torch.Tensor) -> tuple[float, int]:
    """Return summed causal next-token NLL and the number of unpadded target tokens."""
    if logits.ndim != 3 or input_ids.ndim != 2 or logits.shape[:2] != input_ids.shape:
        raise ValueError('logits must be [batch, tokens, vocab] and input_ids [batch, tokens]')
    if input_ids.shape[1] < 2:
        raise ValueError('each document must contain at least two tokens')
    shifted_logits = logits[:, :-1, :].float().contiguous()
    targets = input_ids[:, 1:].contiguous()
    valid = attention_mask[:, 1:].to(dtype=torch.bool)
    losses = F.cross_entropy(shifted_logits.view(-1, shifted_logits.shape[-1]),
                             targets.reshape(-1), reduction='none').view_as(targets)
    selected = losses.masked_select(valid)
    return float(selected.sum().item()), int(selected.numel())


def mean_next_token_loss(tok, model, layers, docs: list[dict], operations: dict,
                         batch_size: int, device: str, max_gpu_gb: float) -> dict:
    tokenized = [part26._tokenize_document(tok, row['text']) for row in docs]
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(tokenized), batch_size):
        batch = tokenized[start:start + batch_size]
        ids, mask = part26._padded_inputs(batch, tok, device)
        row_ops = [operations for _ in batch]
        handles = part25.install_intervention_hooks(layers, row_ops)
        try:
            with torch.no_grad():
                output = model(input_ids=ids, attention_mask=mask, use_cache=False)
                loss_sum, token_count = next_token_loss_terms(output.logits, ids, mask)
            total_loss += loss_sum
            total_tokens += token_count
        finally:
            for handle in handles:
                handle.remove()
        nr.spill_guard(device, max_gpu_gb)
    if not total_tokens:
        raise ValueError('loss corpus produced no next-token targets')
    return {'mean_next_token_loss': total_loss / total_tokens,
            'summed_next_token_loss': total_loss, 'scored_tokens': total_tokens,
            'documents': len(docs)}


def _conditions(specs: list[dict], controls: list[dict]) -> list[dict]:
    result = [{'name': 'baseline', 'kind': 'baseline', 'neurons': []},
              {'name': 'set_zero', 'kind': 'target_set',
               'neurons': [{'layer': li, 'unit': unit} for li, unit in pairs(specs)]}]
    result.extend({'name': f'control_seed_{row["seed"]}', 'kind': 'control', 'seed': int(row['seed']),
                   'neurons': row['neurons']} for row in controls)
    return result


def _operations_for(condition: dict) -> dict:
    return part25._operations(pairs(condition['neurons']), 'zero') if condition['neurons'] else {}


def _expected_answer_present(text: str, expected: str) -> bool:
    pattern = re.compile(r'(?<!\w)' + re.escape(expected.strip()) + r'(?!\w)', re.I)
    return bool(pattern.search(text))


def run_costs(args, tok, model, layers, specs: list[dict] | None = None,
              controls: list[dict] | None = None, loss_docs: list[dict] | None = None,
              fact_rows: list[dict] | None = None) -> dict:
    specs = TARGET_NEURONS if specs is None else specs
    controls = control_sets(layers, specs, getattr(args, 'n_controls', 20)) if controls is None else controls
    _require_layout(layers, specs)
    loss_docs = select_documents(args.corpus, getattr(args, 'cost_docs', N_COST_DOCS),
                                 skip_eligible=N_FIRE_DOCS) if loss_docs is None else loss_docs
    if fact_rows is None:
        with open(args.facts, encoding='utf-8') as stream:
            fact_rows = json.load(stream)['questions']
    if not fact_rows:
        raise ValueError('facts file contains no questions')
    conditions = _conditions(specs, controls)
    loss_rows = {}
    factual_rows = {}
    identity_rows = {}
    fact_questions = [str(row['question']) for row in fact_rows]
    identity_questions = list(circuit.ID_PROMPTS)
    questions = fact_questions + identity_questions
    first_sentences: dict[str, dict[str, str]] = {}
    for condition in conditions:
        name = condition['name']
        operations = _operations_for(condition)
        loss_rows[name] = mean_next_token_loss(
            tok, model, layers, loss_docs, operations, args.batch_size, args.device, args.max_gpu_gb)
        answers = part25._greedy_generate(
            tok, model, layers, questions, operations, args.device, args.max_gpu_gb,
            max_new_tokens=getattr(args, 'max_new_tokens', 40))
        first_sentences[name] = {question: answers[question]['first_sentence'] for question in questions}
        factual_answers = []
        for record in fact_rows:
            question = str(record['question'])
            answer = answers[question]
            expected = str(record['expected_answer'])
            baseline_first = (first_sentences.get('baseline', {}).get(question)
                              if name != 'baseline' else answer['first_sentence'])
            factual_answers.append({
                'question': question, 'expected_answer': expected,
                'answer': answer['answer'], 'first_sentence': answer['first_sentence'],
                'first_sentence_changed': answer['first_sentence'] != baseline_first,
                'expected_answer_present': _expected_answer_present(answer['answer'], expected),
            })
        factual_rows[name] = {
            'first_sentence_change_count': sum(row['first_sentence_changed'] for row in factual_answers),
            'expected_answer_retained_count': sum(row['expected_answer_present'] for row in factual_answers),
            'answers': factual_answers,
        }
        identity_answers = []
        for question in identity_questions:
            answer = answers[question]
            first = answer['first_sentence']
            identity_answers.append({
                'question': question, 'answer': answer['answer'], 'first_sentence': first,
                'names_qwen_tongyi_alibaba': bool(circuit.IDENTITY.search(first)),
            })
        identity_rows[name] = identity_answers

    facts_sha = _sha(args.facts)
    out = header(args.model, args.revision, 'costs',
                 _input_shas({'corpus': args.corpus, 'data/part27-facts.json': args.facts}))
    out.update({
        'neuron_definition': 'one input unit of a layer MLP down_proj; zero operation sets those units to 0 at all positions',
        'target_neurons': specs, 'controls': controls,
        'loss': {
            'document_selection': 'the 200 eligible documents after the first 1,000 eligible documents used by fires; first 256 whitespace-separated words',
            'aggregation': 'token-weighted mean of causal next-token negative log likelihood over non-padding targets',
            'input_sha256': _sha(args.corpus), 'conditions': loss_rows,
        },
        'factual_questions': {
            'file': os.path.basename(args.facts), 'input_sha256': facts_sha,
            'count': len(fact_rows), 'first_sentence_change_baseline': 'exact string equality after Part 25 first_sentence()',
            'conditions': factual_rows,
        },
        'identity_questions': {
            'questions': identity_questions,
            'maker_match': 'circuit.IDENTITY (Qwen, Tongyi, Alibaba and Chinese aliases)',
            'conditions': identity_rows,
        },
    })
    part25.write_json(out, os.path.join(args.out, 'costs.json'))
    return out


def _sample_completion(tok, model, layers, prompt: str, operations: dict, seed: int,
                       max_new_tokens: int, device: str, max_gpu_gb: float) -> dict:
    ids_list = _encode(tok, prompt)
    ids = torch.tensor([ids_list], dtype=torch.long, device=next(model.parameters()).device)
    mask = torch.ones_like(ids)
    handles = part25.install_intervention_hooks(layers, [operations])
    try:
        torch.manual_seed(int(seed))
        eos_ids = {getattr(tok, 'eos_token_id', None),
                   getattr(getattr(model, 'generation_config', None), 'eos_token_id', None)}
        eos_ids = sorted(int(value) for value in eos_ids if isinstance(value, int) and value >= 0)
        with torch.no_grad():
            generated = model.generate(
                input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=0.7, top_p=0.95,
                pad_token_id=getattr(tok, 'pad_token_id', None) or getattr(tok, 'eos_token_id', None),
                eos_token_id=eos_ids or None,
            )
        text = tok.decode(generated[0, ids.shape[1]:], skip_special_tokens=True).strip()
    finally:
        for handle in handles:
            handle.remove()
    nr.spill_guard(device, max_gpu_gb)
    found = list(dict.fromkeys(match.group(0) for match in MAKER_RE.finditer(text)))
    return {'seed': int(seed), 'completion': text, 'names_a_maker': bool(found), 'makers_named': found}


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or not a.size:
        raise ValueError('Pearson inputs must have the same nonempty shape')
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def run_base(args, tok, model, layers, specs: list[dict] | None = None,
             controls: list[dict] | None = None, docs: list[dict] | None = None,
             chat_json_path: str | None = None, chat_npz_path: str | None = None) -> dict:
    specs = TARGET_NEURONS if specs is None else specs
    controls = control_sets(layers, specs, getattr(args, 'n_controls', 20)) if controls is None else controls
    _require_layout(layers, specs)
    chat_json_path = chat_json_path or os.path.join(args.out, 'fires.json')
    chat_npz_path = chat_npz_path or os.path.join(args.out, 'fires-word-activations.npz')
    with open(chat_json_path, encoding='utf-8') as stream:
        fires = json.load(stream)
    with np.load(chat_npz_path) as archive:
        chat_acts = np.asarray(archive['word_activations'], dtype=np.float32)
        chat_indices = np.asarray(archive['line_indices'], dtype=np.int64)
        chat_pairs = np.asarray(archive['neuron_pairs'], dtype=np.int64)
    if docs is None:
        docs = select_documents(args.corpus, int(len(chat_indices)))
    doc_indices = np.asarray([int(row['line_index']) for row in docs], dtype=np.int64)
    if not np.array_equal(doc_indices, chat_indices):
        raise ValueError('base corpus document indices differ from fires-word-activations.npz')
    expected_pairs = np.asarray(pairs(specs), dtype=np.int64)
    if chat_acts.shape != (len(specs), len(docs), N_WORDS) or not np.array_equal(chat_pairs, expected_pairs):
        raise ValueError('chat word-activation NPZ has a different shape or neuron order')
    fires_indices = np.asarray([int(row['line_index']) for row in fires['documents']], dtype=np.int64)
    if not np.array_equal(fires_indices, chat_indices):
        raise ValueError('fires.json document indices differ from fires-word-activations.npz')

    summaries, base_acts = _capture_documents(tok, model, layers, docs, pairs(specs), specs,
                                               args.batch_size, args.device, args.max_gpu_gb)
    correlations = []
    for index, pair in enumerate(pairs(specs)):
        correlations.append({
            'layer': pair[0], 'unit': pair[1],
            'pearson_r': _pearson(chat_acts[index], base_acts[index]),
            'word_positions': int(chat_acts[index].size),
        })
    npz_name = 'base-word-activations.npz'
    np.savez_compressed(os.path.join(args.out, npz_name), word_activations=base_acts,
                        line_indices=doc_indices, neuron_pairs=expected_pairs)

    plain_prompts = [{'question': question, 'system_prompt': None,
                      'prompt': f'Q: {question}\nA:'}
                     for question in circuit.ID_PROMPTS]
    identity_rows = _capture_prompts(tok, model, layers, plain_prompts, pairs(specs),
                                     args.device, args.max_gpu_gb)
    base_prompt = 'Q: Who created you?\nA:'
    max_new = getattr(args, 'base_max_new_tokens', 60)
    sample_count = getattr(args, 'base_samples', 8)
    completion_rows = {
        'baseline': [_sample_completion(tok, model, layers, base_prompt, {}, seed, max_new,
                                        args.device, args.max_gpu_gb)
                     for seed in range(sample_count)],
        'set_zero': [_sample_completion(tok, model, layers, base_prompt,
                                        part25._operations(pairs(specs), 'zero'), seed, max_new,
                                        args.device, args.max_gpu_gb)
                     for seed in range(sample_count)],
        'controls': [],
    }
    for control in controls:
        completion_rows['controls'].append({
            'control_seed': int(control['seed']), 'neurons': control['neurons'],
            **_sample_completion(tok, model, layers, base_prompt,
                                 part25._operations(pairs(control['neurons']), 'zero'), 0, max_new,
                                 args.device, args.max_gpu_gb),
        })
    out = header(args.base_model, args.base_revision, 'base',
                 _input_shas({'corpus': args.corpus, 'fires.json': chat_json_path,
                              'fires-word-activations.npz': chat_npz_path}))
    out.update({
        'neuron_definition': 'one input unit of a layer MLP down_proj',
        'target_neurons': specs, 'controls': controls,
        'document_selection': 'the same original JSONL rows and first 256 whitespace-separated words recorded by fires',
        'documents': len(docs), 'line_indices_match_chat': True,
        'word_activations': {'file': npz_name, 'shape': list(base_acts.shape),
                             'dtype': 'float32', 'neuron_order': [pair_key(pair) for pair in pairs(specs)]},
        'chat_base_word_pearson': correlations,
        'identity_questions': {
            'format': 'Q: <question>\\nA:', 'questions': circuit.ID_PROMPTS,
            'rows': identity_rows,
        },
        'maker_completions': {
            'prompt': base_prompt, 'max_new_tokens': max_new, 'samples_baseline_and_set_zero': sample_count,
            'sampling': {'do_sample': True, 'temperature': 0.7, 'top_p': 0.95,
                         'seeds': list(range(sample_count))},
            'control_sampling': {'samples_per_set': 1, 'seed': 0},
            'conditions': completion_rows,
        },
    })
    part25.write_json(out, os.path.join(args.out, 'base.json'))
    return out


def load_model_offline(model_id: str, revision: str, device: str):
    """Use Part 26's loader with offline mode set before any cache access."""
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    return part26._load_model_offline(model_id, revision, device)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--steps', default='fires,writes,costs,base',
                    help='comma-separated sequence: fires,writes,costs,base')
    ap.add_argument('--model', default=CHAT_MODEL)
    ap.add_argument('--revision', default=CHAT_REVISION)
    ap.add_argument('--base-model', default=BASE_MODEL)
    ap.add_argument('--base-revision', default=BASE_REVISION)
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--facts', default=FACTS_PATH)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='xpu')
    ap.add_argument('--max-gpu-gb', type=float, default=14.5)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--max-new-tokens', type=int, default=40)
    ap.add_argument('--base-max-new-tokens', type=int, default=60)
    ap.add_argument('--base-samples', type=int, default=8)
    ap.add_argument('--n-controls', type=int, default=20)
    return ap


def _parse_steps(value: str) -> list[str]:
    steps = [item.strip() for item in value.split(',') if item.strip()]
    allowed = ['fires', 'writes', 'costs', 'base']
    if not steps or any(step not in allowed for step in steps):
        raise SystemExit('--steps accepts fires,writes,costs,base')
    if len(set(steps)) != len(steps) or steps != [step for step in allowed if step in steps]:
        raise SystemExit('--steps must be unique and follow fires,writes,costs,base order')
    return steps


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.base_max_new_tokens < 1:
        raise SystemExit('--batch-size and generation lengths must be positive')
    if args.base_samples < 1 or args.n_controls < 1:
        raise SystemExit('--base-samples and --n-controls must be positive')
    steps = _parse_steps(args.steps)
    os.makedirs(args.out, exist_ok=True)
    chat_steps = [step for step in steps if step != 'base']
    controls = None
    specs = TARGET_NEURONS
    if chat_steps:
        tok, model = load_model_offline(args.model, args.revision, args.device)
        try:
            _layers_name, layers, norm = nr.find_text_layers(model)
            _require_layout(layers, specs)
            controls = control_sets(layers, specs, args.n_controls)
            for step in chat_steps:
                if step == 'fires':
                    run_fires(args, tok, model, layers, specs, controls)
                elif step == 'writes':
                    run_writes(args, tok, model, layers, norm, specs, controls)
                elif step == 'costs':
                    run_costs(args, tok, model, layers, specs, controls)
        finally:
            del model, tok
            gc.collect()
            if args.device == 'xpu':
                torch.xpu.empty_cache()
    if 'base' in steps:
        for path in (os.path.join(args.out, 'fires.json'),
                     os.path.join(args.out, 'fires-word-activations.npz')):
            if not os.path.isfile(path):
                raise SystemExit(f'{path} is missing; run fires before base')
        tok, model = load_model_offline(args.base_model, args.base_revision, args.device)
        try:
            _layers_name, layers, _norm = nr.find_text_layers(model)
            _require_layout(layers, specs)
            controls = control_sets(layers, specs, args.n_controls)
            run_base(args, tok, model, layers, specs, controls)
        finally:
            del model, tok
            gc.collect()
            if args.device == 'xpu':
                torch.xpu.empty_cache()


if __name__ == '__main__':
    main()

