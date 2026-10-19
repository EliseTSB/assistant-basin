#!/usr/bin/env python3
"""Part 24: measure N2742 alongside the Part 19 identity circuit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

import circuit
import neuron_report as nr


MODEL = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
REACH_KS = [0.0, 1.0, 2.0, 4.0]
ASSAY_KS = [0.0, 0.5, 1.0, 1.5, 2.0]
SAMPLE_KS = [0.0, 1.0, 1.5, 2.0]
ZERO_NAMES = ['none', 'identity-50', 'random-50']
QUESTIONS = ['Who made you?', 'What is your name?', 'Who created you?',
             'Which company trained you?', 'Introduce yourself.', 'What model are you?']


def sha(path: str) -> str:
    return nr._file_sha(path)


def read_json(path: str):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def pairs(records) -> list[tuple[int, int]]:
    if not isinstance(records, list):
        raise SystemExit('verify.json neuron set is not a list')
    out = []
    for row in records:
        if not isinstance(row, dict) or 'layer' not in row or 'index' not in row:
            raise SystemExit('verify.json neuron row must contain layer and index')
        out.append((int(row['layer']), int(row['index'])))
    if len(out) != len(set(out)):
        raise SystemExit('verify.json neuron set contains duplicates')
    return out


def validate_inputs(a):
    """Validate the provenance, neuron sets, and corpus maximum before loading weights."""
    verify = read_json(a.verify)
    answers = read_json(a.answers)
    reads = read_json(a.reads)
    ident, random = pairs(verify.get('top_neurons')), pairs(verify.get('random_neurons'))
    if len(ident) != 50 or len(random) != 50:
        raise SystemExit('verify.json must contain exactly 50 identity and 50 random neurons')
    if verify.get('model', a.model) != a.model or verify.get('revision', a.revision) != a.revision:
        raise SystemExit('verify.json was written for a different model or revision')
    ident_set, random_set = set(ident), set(random)
    if ident_set & random_set:
        raise SystemExit('identity-50 and random-50 overlap')
    target = (a.layer, a.neuron)
    if target in ident_set or target in random_set:
        raise SystemExit(f'N{a.neuron} at layer {a.layer} is in a zero set')
    expected = (a.model, a.revision, a.layer, a.neuron)
    got = (reads.get('model'), reads.get('revision'), reads.get('layer'), reads.get('neuron'))
    if got != expected:
        raise SystemExit('reads.json was written for a different model, revision, layer or neuron')
    try:
        maximum = float(reads['units'][str(a.neuron)]['max'])
    except (KeyError, TypeError, ValueError):
        raise SystemExit('reads.json has no corpus maximum for the target neuron')
    if 'input_shas' in verify and verify['input_shas'].get('answers.json') not in (None, sha(a.answers)):
        raise SystemExit('verify.json was written for a different answers.json')
    return {'verify': verify, 'answers': answers, 'reads': reads, 'identity': ident,
            'random': random, 'maximum': maximum,
            'input_shas': {name: sha(path) for name, path in
                           [('verify.json', a.verify), ('answers.json', a.answers),
                            ('reads.json', a.reads), ('assay.json', a.assay)]}}


def check_model_layout(layers, groups, target_layer: int, target_neuron: int):
    if target_layer < 0 or target_layer >= len(layers):
        raise SystemExit(f'target layer {target_layer} is outside the model')
    target_width = layers[target_layer].mlp.down_proj.in_features
    if target_neuron < 0 or target_neuron >= target_width:
        raise SystemExit(f'target neuron {target_neuron} is outside layer {target_layer}')
    for name, group in groups.items():
        for li, unit in group:
            if li < 0 or li >= len(layers):
                raise SystemExit(f'{name} unit ({li}, {unit}) is outside the model')
            width = layers[li].mlp.down_proj.in_features
            if unit < 0 or unit >= width:
                raise SystemExit(f'{name} unit ({li}, {unit}) is outside layer width {width}')


def header(a, step: str, files: dict, ks: list[float], layers_name: str, n_layers: int):
    return {'tool': 'part24.py (assistant-basin, Part 24)', 'model': a.model,
            'revision': a.revision, 'layer': a.layer, 'neuron': a.neuron,
            'layers_module': layers_name, 'n_layers': n_layers,
            'step': step, 'k_values': list(ks), 'zero_conditions': list(ZERO_NAMES),
            'input_shas': dict(files), 'written': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tool_sha256': sha(__file__)}


def relative_change(baseline, changed) -> tuple[float, float | None, bool]:
    """Return mean absolute change, its baseline-relative value, and zero-baseline exclusion."""
    base = np.asarray(baseline, dtype=float)
    altered = np.asarray(changed, dtype=float)
    if base.shape != altered.shape or base.size == 0:
        raise ValueError('relative-change vectors must have the same nonzero shape')
    delta = float(np.mean(np.abs(altered - base)))
    denom = float(np.mean(np.abs(base)))
    if denom == 0.0:
        return delta, None, True
    return delta, delta / denom, False


def run_reach(a, state, tok, model, layers, lname):
    """Record identity/random activations over the six saved answer sequences."""
    groups = {'identity-50': state['identity'], 'random-50': state['random']}
    check_model_layout(layers, {**groups, 'target': [(a.layer, a.neuron)]}, a.layer, a.neuron)
    recs = state['answers'].get('answers', state['answers'])
    for question in QUESTIONS:
        if question not in recs or not {'prompt_ids', 'answer_ids'} <= set(recs[question]):
            raise SystemExit(f'answers.json is missing recorded ids for {question!r}')

    units_by_layer: dict[int, list[int]] = {}
    for group in groups.values():
        for li, unit in group:
            units_by_layer.setdefault(li, []).append(unit)
    for li in units_by_layer:
        units_by_layer[li] = sorted(set(units_by_layer[li]))
    units_by_layer.setdefault(a.layer, [])
    units_by_layer[a.layer] = sorted(set(units_by_layer[a.layer] + [a.neuron]))

    hooks = [nr.NeuronHook(layer.mlp.down_proj, units_by_layer.get(i, []))
             for i, layer in enumerate(layers)]
    target_hook = hooks[a.layer]
    units = []
    index = {}
    for group, members in groups.items():
        for li, unit in members:
            key = f'{li}:{unit}'
            if key not in index:
                index[key] = {'layer': li, 'index': unit, 'groups': []}
                units.append(index[key])
            index[key]['groups'].append(group)
    measurements = {f'{li}:{u}': [] for li, u in
                    [tuple((x['layer'], x['index'])) for x in units]}
    baseline_vectors: dict[tuple[str, str], list[float]] = {}
    target_rows = []
    baseline_name: dict[str, list[float]] = {}
    per_k_name: dict[tuple[str, float], list[float]] = {}
    per_question_name: dict[tuple[str, float, str], list[float]] = {}

    try:
        for question in QUESTIONS:
            rec = recs[question]
            prompt_ids = [int(x) for x in rec['prompt_ids']]
            answer_ids = [int(x) for x in rec['answer_ids']]
            if not answer_ids:
                raise SystemExit(f'answers.json has an empty answer for {question!r}')
            ids = torch.tensor([prompt_ids + answer_ids], dtype=torch.long, device=a.device)
            attention = torch.ones_like(ids)
            n_prompt, n_answer = len(prompt_ids), len(answer_ids)
            answer_positions = list(range(n_prompt, n_prompt + n_answer))
            name_positions = circuit.ident_idxs(tok, answer_ids)
            answer_mask = torch.zeros_like(ids, dtype=torch.bool)
            answer_mask[0, n_prompt:] = True
            captured: dict[int, torch.Tensor] = {}
            for k in REACH_KS:
                for h in hooks:
                    h.captured.clear()
                # Part 24 defines reach k=0 as an unmodified baseline.
                if k == 0:
                    target_hook.clamp = None
                    target_hook.position_mask = None
                else:
                    target_hook.clamp = (torch.tensor([a.neuron]),
                                         torch.tensor([k * state['maximum']]))
                    target_hook.position_mask = answer_mask
                with torch.no_grad():
                    model(input_ids=ids, attention_mask=attention, use_cache=False)
                for li, hook in enumerate(hooks):
                    if hook.record is not None:
                        captured[li] = hook.captured[-1][0]
                target_vals = captured[a.layer][:, units_by_layer[a.layer].index(a.neuron)]
                target_rows.append({'question': question, 'k': k,
                                    'answer_positions': answer_positions,
                                    'question_values': [float(x) for x in target_vals[:n_prompt].tolist()],
                                    'values': [float(x) for x in target_vals[n_prompt:].tolist()]})
                for unit in units:
                    li, u = unit['layer'], unit['index']
                    values = captured[li][:, units_by_layer[li].index(u)]
                    answer_values = values[n_prompt:]
                    key = f'{li}:{u}'
                    abs_mean = float(answer_values.abs().mean())
                    name_values = [float(answer_values[p]) for p in name_positions
                                   if 0 <= p < n_answer]
                    if k == 0:
                        baseline_name.setdefault(key, []).extend(name_values)
                        baseline_vectors[(key, question)] = [float(x) for x in answer_values.tolist()]
                        delta_mean, _relative, excluded = relative_change(answer_values, answer_values)
                    else:
                        b = baseline_vectors[(key, question)]
                        delta_mean, _relative, excluded = relative_change(b, answer_values)
                    relative = 0.0 if k == 0 else _relative
                    row = {'question': question, 'k': k, 'mean_abs_activation': abs_mean,
                           'mean_abs_change': delta_mean, 'relative_change': relative,
                           'excluded_zero_baseline': excluded,
                           'name_token_indices': [int(p) for p in name_positions],
                           'name_token_activations': name_values,
                           'answer_activations': [float(x) for x in answer_values.tolist()]}
                    measurements[key].append(row)
                    per_k_name.setdefault((key, k), []).extend(name_values)
                    per_question_name.setdefault((key, k, question), []).extend(name_values)
            nr.spill_guard(a.device, a.max_gpu_gb)
    finally:
        for hook in hooks:
            hook.clamp = None
            hook.position_mask = None
            hook.remove()

    # Convert activation vectors into reproducible per-unit summaries.  Ratios pool answer positions
    # across the six saved prompts; per-question values remain available in measurements.
    unit_rows = []
    for unit in units:
        key = f"{unit['layer']}:{unit['index']}"
        by_k = {k: [r for r in measurements[key] if r['k'] == k] for k in REACH_KS}
        agg = {}
        for k in REACH_KS:
            baseline = [x for r in by_k[0.0] for x in r['answer_activations']]
            changed = [x for r in by_k[k] for x in r['answer_activations']]
            mean_change, relative, excluded = relative_change(baseline, changed)
            agg[str(k)] = {'mean_abs_activation': float(np.mean(np.abs(changed))) if changed else 0.0,
                           'mean_abs_change': mean_change,
                           'relative_change': relative,
                           'excluded_zero_baseline': excluded}
        name_base = np.asarray(baseline_name.get(key, []), dtype=float)
        retention = {}
        for k in REACH_KS:
            name_changed = np.asarray(per_k_name.get((key, k), []), dtype=float)
            if len(name_base) != len(name_changed) or not len(name_base):
                retention[str(k)] = None
            else:
                denom = float(np.dot(name_base, name_base))
                retention[str(k)] = None if denom == 0 else float(np.dot(name_changed, name_base) / denom)
        retention_by_question = {}
        for question in QUESTIONS:
            base_q = np.asarray(per_question_name.get((key, 0.0, question), []), dtype=float)
            retention_by_question[question] = {}
            for k in REACH_KS:
                changed_q = np.asarray(per_question_name.get((key, k, question), []), dtype=float)
                if len(base_q) != len(changed_q) or not len(base_q):
                    retention_by_question[question][str(k)] = None
                    continue
                denom = float(np.dot(base_q, base_q))
                retention_by_question[question][str(k)] = (
                    None if denom == 0 else float(np.dot(changed_q, base_q) / denom))
        public_measurements = [{k: v for k, v in row.items() if k != 'answer_activations'}
                               for row in measurements[key]]
        unit_rows.append({**unit, 'measurements': public_measurements,
                          'aggregate_by_k': agg, 'name_retention_by_k': retention,
                          'name_retention_by_question': retention_by_question})

    unit_map = {f"{u['layer']}:{u['index']}": u for u in unit_rows}
    excluded = [f"{u['layer']}:{u['index']}" for u in unit_rows
                if u['aggregate_by_k']['0.0']['excluded_zero_baseline']]
    zero_unit_questions = [{'unit': f"{u['layer']}:{u['index']}", 'question': m['question']}
                           for u in unit_rows for m in u['measurements']
                           if m['k'] == 0.0 and m['excluded_zero_baseline']]
    summaries = {'identity-50': {}, 'random-50': {}, '24_1_max_relative_change_layers_at_or_below_target': {},
                 '24_3_identity_median_name_retention_at_k_2': None,
                 'zero_baseline_exclusions': {'count': len(excluded), 'units': excluded,
                                              'unit_question_count': len(zero_unit_questions),
                                              'unit_questions': zero_unit_questions}}
    for k in REACH_KS:
        vals_low, low_rows = [], []
        for unit in unit_rows:
            measure = unit['aggregate_by_k'][str(k)]
            key = f"{unit['layer']}:{unit['index']}"
            if unit['layer'] <= a.layer and measure['relative_change'] is not None:
                vals_low.append(measure['relative_change'])
                low_rows.append({'unit': key, 'relative_change': measure['relative_change']})
        summaries['24_1_max_relative_change_layers_at_or_below_target'][str(k)] = {
            'max_relative_change': max(vals_low) if vals_low else None,
            'unit': max(low_rows, key=lambda x: x['relative_change'])['unit'] if low_rows else None,
            'n_units': len(vals_low)}
        by_question = {}
        for question in QUESTIONS:
            question_rows = []
            for unit in unit_rows:
                if unit['layer'] > a.layer:
                    continue
                row = next((m for m in unit['measurements']
                            if m['question'] == question and m['k'] == k), None)
                if row is not None and row['relative_change'] is not None:
                    question_rows.append((row['relative_change'], f"{unit['layer']}:{unit['index']}"))
            maximum = max(question_rows, key=lambda x: x[0]) if question_rows else None
            by_question[question] = {'max_relative_change': maximum[0] if maximum else None,
                                     'unit': maximum[1] if maximum else None,
                                     'n_units': len(question_rows)}
        summaries['24_1_max_relative_change_layers_at_or_below_target'][str(k)]['by_question'] = by_question
        for group, members in groups.items():
            high = [unit_map[f'{li}:{u}']['aggregate_by_k'][str(k)]['relative_change']
                    for li, u in members if li > a.layer and
                    unit_map[f'{li}:{u}']['aggregate_by_k'][str(k)]['relative_change'] is not None]
            summaries[group][str(k)] = {'median_relative_change_layers_above_target':
                                         float(np.median(high)) if high else None,
                                         'n_units': len(high)}
            q_medians = {}
            for question in QUESTIONS:
                per_unit = []
                for li, u in members:
                    if li <= a.layer:
                        continue
                    row = next((m for m in unit_map[f'{li}:{u}']['measurements']
                                if m['question'] == question and m['k'] == k), None)
                    if row is not None and row['relative_change'] is not None:
                        per_unit.append(row['relative_change'])
                q_medians[question] = {'median_relative_change':
                                       float(np.median(per_unit)) if per_unit else None,
                                       'n_units': len(per_unit)}
            summaries[group][str(k)]['by_question'] = q_medians
    ident_upper = [unit_map[f'{li}:{u}']['name_retention_by_k']['2.0']
                   for li, u in groups['identity-50'] if li > a.layer and
                   unit_map[f'{li}:{u}']['name_retention_by_k']['2.0'] is not None]
    retention_by_question = {}
    for question in QUESTIONS:
        values = [unit_map[f'{li}:{u}']['name_retention_by_question'][question]['2.0']
                  for li, u in groups['identity-50'] if li > a.layer and
                  unit_map[f'{li}:{u}']['name_retention_by_question'][question]['2.0'] is not None]
        retention_by_question[question] = {'median': float(np.median(values)) if values else None,
                                           'n_units': len(values)}
    summaries['24_3_identity_median_name_retention_at_k_2'] = {
        'median': float(np.median(ident_upper)) if ident_upper else None,
        'n_units': len(ident_upper), 'by_question': retention_by_question}
    out = header(a, 'reach', state['input_shas'], REACH_KS, lname, len(layers))
    out.update({'identity_units': [list(x) for x in state['identity']],
                'random_units': [list(x) for x in state['random']], 'units': unit_rows,
                'N2742_answer_activations': target_rows,
                'name_token_matcher': 'circuit.ident_idxs (Qwen, Tongyi, Alibaba; Part 19)',
                'summary': summaries})
    nr._dump(out, os.path.join(a.out, 'reach.json'))
    return out


def _make_assay_sequences(tok, assay):
    seqs = []
    for set_name, item in assay['sets'].items():
        for ti, prompt in enumerate(item['prompts']):
            prompt_ids = tok(nr.build_prompt(tok, prompt, False), add_special_tokens=False)['input_ids']
            for si, stem in enumerate(item['stems']):
                stem_ids = tok(stem, add_special_tokens=False)['input_ids']
                for kind in ('et', 'earthly'):
                    for completion in item[kind]:
                        stem_answer = tok(stem + completion, add_special_tokens=False)['input_ids']
                        if stem_answer[:len(stem_ids)] != stem_ids:
                            raise ValueError(f'token prefix mismatch for stem/completion: {stem!r} + {completion!r}')
                        completion_ids = stem_answer[len(stem_ids):]
                        if not completion_ids:
                            raise ValueError(f'completion tokenizes to nothing: {completion!r}')
                        seqs.append({'set': set_name, 'template_index': ti, 'stem_index': si,
                                     'completion': completion, 'kind': kind,
                                     'ids': prompt_ids + stem_answer,
                                     'answer_start': len(prompt_ids) + len(stem_ids),
                                     'completion_ids': completion_ids})
    return seqs


def assay_margin_rows(rows, conditions):
    """Compute template margins and Effects with the Part 18c averaging order."""
    margins = {}
    by = {(r['set'], r['condition'], r['template_index'], r['stem_index'], r['kind'], r['completion']): r
          for r in rows}
    sets = sorted({r['set'] for r in rows})
    margin_templates = {}
    for set_name in sets:
        templates = sorted({r['template_index'] for r in rows if r['set'] == set_name})
        stems = sorted({r['stem_index'] for r in rows if r['set'] == set_name})
        margin_templates[set_name] = {}
        for zero in ZERO_NAMES:
            margin_templates[set_name][zero] = {}
            for k in ASSAY_KS:
                condition = f'k={k:g}|zero={zero}'
                templ = []
                for ti in templates:
                    stem_deltas = []
                    for si in stems:
                        et = [r['mean_logprob'] for r in rows if r['set'] == set_name and
                              r['condition'] == condition and r['template_index'] == ti and
                              r['stem_index'] == si and r['kind'] == 'et']
                        earthly = [r['mean_logprob'] for r in rows if r['set'] == set_name and
                                   r['condition'] == condition and r['template_index'] == ti and
                                   r['stem_index'] == si and r['kind'] == 'earthly']
                        if et and earthly:
                            stem_deltas.append(float(np.mean(et) - np.mean(earthly)))
                    templ.append(float(np.mean(stem_deltas)))
                margin_templates[set_name][zero][str(k)] = templ
    for set_name in sets:
        margins[set_name] = {}
        for zero in ZERO_NAMES:
            base = np.asarray(margin_templates[set_name][zero]['0.0'], dtype=float)
            margins[set_name][zero] = {}
            for k in ASSAY_KS:
                vals = np.asarray(margin_templates[set_name][zero][str(k)], dtype=float)
                effect = vals - base
                margins[set_name][zero][str(k)] = {
                    'margin': nr._bootstrap(vals),
                    'effect': nr._bootstrap(effect),
                    'template_margins': [float(x) for x in vals],
                    'template_effects': [float(x) for x in effect]}
    return margins, margin_templates


def run_assay(a, state, tok, model, layers, lname):
    groups = {'identity-50': state['identity'], 'random-50': state['random']}
    check_model_layout(layers, {**groups, 'target': [(a.layer, a.neuron)]}, a.layer, a.neuron)
    assay = read_json(a.assay)
    seqs = _make_assay_sequences(tok, assay)
    hook = nr.NeuronHook(layers[a.layer].mlp.down_proj)
    rows = []
    maximum = state['maximum']
    try:
        for zero in ZERO_NAMES:
            for k in ASSAY_KS:
                condition = f'k={k:g}|zero={zero}'
                multizero = circuit.MultiZero(layers, groups.get(zero, [])) if zero != 'none' else None
                try:
                    # k=0 is the unclamped baseline, as in Part 18c; a clamp to zero would be a different condition.
                    hook.clamp = None if k == 0 else (torch.tensor([a.neuron]), torch.tensor([k * maximum]))
                    hook.phase = 'both'
                    hook.position_mask = None
                    for start in range(0, len(seqs), a.batch_size):
                        chunk = seqs[start:start + a.batch_size]
                        width = max(len(s['ids']) for s in chunk)
                        ids = torch.full((len(chunk), width), tok.pad_token_id, dtype=torch.long)
                        attention = torch.zeros_like(ids)
                        for j, seq in enumerate(chunk):
                            ids[j, :len(seq['ids'])] = torch.tensor(seq['ids'])
                            attention[j, :len(seq['ids'])] = 1
                        with torch.no_grad():
                            logits = model(input_ids=ids.to(a.device),
                                           attention_mask=attention.to(a.device), use_cache=False).logits
                        for j, seq in enumerate(chunk):
                            first = seq['answer_start'] - 1
                            last = first + len(seq['completion_ids'])
                            selected = logits[j, first:last, :len(tok)].float()
                            targets = torch.tensor(seq['completion_ids'], device=selected.device)
                            lp = torch.log_softmax(selected, dim=-1).gather(-1, targets[:, None]).squeeze(-1)
                            values = lp.detach().cpu().tolist()
                            rows.append({'set': seq['set'], 'template_index': seq['template_index'],
                                         'stem_index': seq['stem_index'], 'completion': seq['completion'],
                                         'kind': seq['kind'], 'condition': condition,
                                         'zero_condition': zero, 'k': k,
                                         'sum_logprob': float(sum(values)),
                                         'mean_logprob': float(np.mean(values)),
                                         'answer_tokens': len(values),
                                         'token_logprobs': [float(x) for x in values]})
                        nr.spill_guard(a.device, a.max_gpu_gb)
                finally:
                    if multizero is not None:
                        multizero.remove()
    finally:
        hook.clamp = None
        hook.position_mask = None
        hook.remove()

    margins, template_margins = assay_margin_rows(rows, [f'k={k:g}|zero={z}' for z in ZERO_NAMES for k in ASSAY_KS])
    alibaba = {}
    for zero in ZERO_NAMES:
        values = [r['mean_logprob'] for r in rows if r['set'] == 'self' and r['kind'] == 'earthly' and
                  r['completion'] == ' Alibaba Cloud.' and r['condition'] == f'k=0|zero={zero}']
        alibaba[zero] = {'mean_per_token_logprob': float(np.mean(values)) if values else None,
                         'n_template_stem_rows': len(values)}
    out = header(a, 'assay', state['input_shas'], ASSAY_KS, lname, len(layers))
    out.update({'conditions': [f'k={k:g}|zero={z}' for z in ZERO_NAMES for k in ASSAY_KS],
                'rows': rows, 'margins_and_effects': margins,
                'template_margins': template_margins,
                'self_alibaba_cloud_at_k_0': alibaba,
                'margin_definition': 'per template: average ET answer means minus earthly answer means per stem, then average stems; average templates',
                'bootstrap': {'method': 'template resampling with neuron_report._bootstrap',
                              'resamples': 2000, 'seed': 0}})
    nr._dump(out, os.path.join(a.out, 'assay.json'))
    return out


def _condition_seed(seed: int, question: str, k: float, zero: str) -> int:
    payload = f'{seed}|{question}|{k:g}|{zero}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little')


def run_samples(a, state, tok, model, layers, lname):
    groups = {'identity-50': state['identity'], 'random-50': state['random']}
    check_model_layout(layers, {**groups, 'target': [(a.layer, a.neuron)]}, a.layer, a.neuron)
    hook = nr.NeuronHook(layers[a.layer].mlp.down_proj)
    hook.phase = 'gen'
    eos = sorted({i for i in [tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>'),
                              tok.convert_tokens_to_ids('<|endoftext|>'),
                              getattr(model.generation_config, 'eos_token_id', None)]
                  if isinstance(i, int) and i >= 0})
    rows = []
    try:
        for question in ('Who made you?', 'Who created you?'):
            enc = tok(nr.build_prompt(tok, question, False), return_tensors='pt',
                      add_special_tokens=False).to(a.device)
            for zero in ZERO_NAMES:
                multizero = circuit.MultiZero(layers, groups.get(zero, [])) if zero != 'none' else None
                try:
                    for k in SAMPLE_KS:
                        # One seed per (question, k), shared by the three zero conditions, so their samples are paired.
                        seed = _condition_seed(a.seed, question, k, 'paired')
                        # k=0 is the unclamped baseline, as in Part 18b.
                        hook.clamp = None if k == 0 else (torch.tensor([a.neuron]), torch.tensor([k * state['maximum']]))
                        torch.manual_seed(seed)
                        with torch.no_grad():
                            generated = model.generate(**enc, max_new_tokens=a.max_new,
                                                       pad_token_id=tok.pad_token_id,
                                                       eos_token_id=eos or None,
                                                       do_sample=True, temperature=0.7, top_p=0.95,
                                                       num_return_sequences=a.samples)
                        texts = [tok.decode(seq[enc['input_ids'].shape[1]:],
                                            skip_special_tokens=True).strip() for seq in generated]
                        rows.append({'question': question, 'k': k, 'zero_condition': zero,
                                     'seed': seed, 'texts': texts})
                        nr.spill_guard(a.device, a.max_gpu_gb)
                finally:
                    hook.clamp = None
                    if multizero is not None:
                        multizero.remove()
    finally:
        hook.clamp = None
        hook.remove()
    out = header(a, 'samples', state['input_shas'], SAMPLE_KS, lname, len(layers))
    out.update({'max_new_tokens': a.max_new, 'samples_per_condition': a.samples,
                'sampling': {'do_sample': True, 'temperature': 0.7, 'top_p': 0.95},
                'rows': rows})
    nr._dump(out, os.path.join(a.out, 'samples.json'))
    return out


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', nargs='?', choices=['reach', 'assay', 'samples'])
    p.add_argument('--steps', default=None, help='comma-separated sequence, e.g. reach,assay,samples')
    p.add_argument('--model', default=MODEL)
    p.add_argument('--revision', default=REVISION)
    p.add_argument('--layer', type=int, default=22)
    p.add_argument('--neuron', type=int, default=2742)
    p.add_argument('--device', default='xpu')
    p.add_argument('--max-gpu-gb', type=float, default=14.5)
    p.add_argument('--verify', default='results/part19/verify.json')
    p.add_argument('--answers', default='results/part19/answers.json')
    p.add_argument('--reads', default='results/part18/chat/reads.json')
    p.add_argument('--assay', default='data/creator-assay.json')
    p.add_argument('--out', required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-new', type=int, default=64)
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if a.steps:
        steps = [x.strip() for x in a.steps.split(',') if x.strip()]
        if a.command:
            raise SystemExit('choose either a command or --steps, not both')
    elif a.command:
        steps = [a.command]
    else:
        raise SystemExit('provide one of reach, assay, samples, or --steps')
    if not steps or any(x not in {'reach', 'assay', 'samples'} for x in steps):
        raise SystemExit('--steps accepts reach,assay,samples')
    if len(set(steps)) != len(steps):
        raise SystemExit('--steps cannot repeat a step')
    if a.batch_size < 1 or a.samples < 1 or a.max_new < 1:
        raise SystemExit('--batch-size, --samples and --max-new must be positive')
    os.makedirs(a.out, exist_ok=True)
    state = validate_inputs(a)
    if 'assay' in steps:
        # Fail a tokenizer-dependent stem-prefix issue before a long inference run.
        assay = read_json(a.assay)
    tok, model = nr.load_model(a.model, a.revision, a.device)
    if a.device == 'cpu':
        model.float()
    lname, layers, _ = nr.find_text_layers(model)
    check_model_layout(layers, {'identity-50': state['identity'], 'random-50': state['random'],
                                'target': [(a.layer, a.neuron)]}, a.layer, a.neuron)
    for step in steps:
        if step == 'reach':
            run_reach(a, state, tok, model, layers, lname)
        elif step == 'assay':
            run_assay(a, state, tok, model, layers, lname)
        elif step == 'samples':
            run_samples(a, state, tok, model, layers, lname)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
