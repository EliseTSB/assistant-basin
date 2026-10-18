#!/usr/bin/env python3
"""CPU tests for teacherforce on a tiny random Qwen3.5 text model."""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import neuron_report as nr  # noqa: E402

SNAP = os.environ['QWEN35_SNAPSHOT']
fails = 0


def check(name, ok, detail=''):
    global fails
    print(('  ok   ' if ok else '  FAIL ') + name + (f'  ({detail})' if detail and not ok else ''))
    fails += 0 if ok else 1


def tiny_model(dirname):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = json.load(open(os.path.join(SNAP, 'config.json')))
    t = dict(cfg.get('text_config', cfg))
    t.update(num_hidden_layers=4, hidden_size=64, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2,
             head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
             linear_value_head_dim=16, layer_types=t['layer_types'][:4], tie_word_embeddings=True)
    t.pop('architectures', None)
    conf = AutoConfig.for_model(t.get('model_type', 'qwen3_5_text'), **{k: v for k, v in t.items() if k != 'model_type'})
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(conf, dtype=torch.float32)
    model.save_pretrained(dirname)
    AutoTokenizer.from_pretrained(SNAP).save_pretrained(dirname)


def score_logits(logits, ids, start):
    pred = logits[:, start - 1:start + len(ids) - 1, :].float()
    target = torch.tensor(ids, device=pred.device)
    return torch.log_softmax(pred, -1).gather(-1, target[None, :, None]).squeeze()


with tempfile.TemporaryDirectory() as tmp:
    mdir = os.path.join(tmp, 'm')
    tiny_model(mdir)
    tok, model = nr.load_model(mdir, None, 'cpu')
    _, layers, _ = nr.find_text_layers(model)
    h = nr.NeuronHook(layers[2].mlp.down_proj, [7])
    h.clamp = (torch.tensor([7]), torch.tensor([3.25]))
    ids = tok('hello there aliens', return_tensors='pt')['input_ids']
    natural = h.handle is not None
    h.position_mask = torch.tensor([[False, True, False]])
    h.captured.clear(); model(input_ids=ids, use_cache=False)
    clamped = h.captured[-1][0, :, 0]
    check('masked hook clamps only selected positions', abs(float(clamped[1]) - 3.25) < 1e-5 and abs(float(clamped[0]) - 3.25) > 1e-4 and abs(float(clamped[2]) - 3.25) > 1e-4)
    h.position_mask = None
    h.captured.clear(); model(input_ids=ids, use_cache=False)
    check('unset mask preserves all-position clamp', torch.allclose(h.captured[-1][..., 0], torch.full_like(h.captured[-1][..., 0], 3.25)))

    # P: clamp the cached prompt, then score continuation with no hook clamp.
    prefix = tok('User: Who made you?\nAssistant: I was created by', add_special_tokens=False)['input_ids']
    suffix = tok(' aliens.', add_special_tokens=False)['input_ids']
    full = torch.tensor([prefix + suffix]); npre = len(prefix); nall = full.shape[1]
    h.position_mask = torch.zeros((1, nall), dtype=torch.bool); h.position_mask[:, :npre] = True
    h.clamp = (torch.tensor([7]), torch.tensor([3.25]))
    one = model(input_ids=full, attention_mask=torch.ones_like(full), use_cache=False).logits
    one_score = score_logits(one, suffix, npre)
    h.position_mask = torch.ones((1, npre), dtype=torch.bool)
    cached = model(input_ids=full[:, :npre], attention_mask=torch.ones((1, npre), dtype=torch.long), use_cache=True)
    h.clamp = None; h.position_mask = None
    rest = model(input_ids=full[:, npre:], attention_mask=torch.ones((1, nall), dtype=torch.long), past_key_values=cached.past_key_values, use_cache=True).logits
    cache_logits = torch.cat([cached.logits[:, -1:, :], rest[:, :-1, :]], dim=1)
    cache_score = score_logits(cache_logits, suffix, 1)
    check('P masked pass equals patched cached-prefix continuation', torch.allclose(one_score, cache_score, atol=1e-4, rtol=1e-4), f'max diff {float((one_score-cache_score).abs().max()):.6g}')

    # Phase boundaries: P ends where the answer (the stem) begins, not where the completion begins (a fold on
    # 2026-10-18 found the build had put the stem inside P, which clamps the position that predicts the completion).
    check('P covers the prompt only and G the whole answer, stem included',
          nr._phase_span('P', 5, 10) == (0, 5) and nr._phase_span('G', 5, 10) == (5, 10) and nr._phase_span('A', 5, 10) == (0, 10))

    # A manual token-level sum checks the completion gather and log-softmax calculation.
    h.clamp = None
    lead = tok('A short answer:', add_special_tokens=False)['input_ids']
    phrase = tok(' aliens.', add_special_tokens=False)['input_ids']
    allids = lead + phrase
    offset = len(lead)
    raw = model(input_ids=torch.tensor([allids]), use_cache=False).logits
    vals = score_logits(raw, phrase, offset)
    manual = sum(float(torch.log_softmax(raw[0, offset - 1 + i].float(), -1)[token]) for i, token in enumerate(phrase))
    check('completion summed log probability matches manual computation', abs(float(vals.sum()) - manual) < 1e-5)

    # Hand table: ET minus earthly delta is .2 at baseline and .5 under c; DD=.3.
    rows = []
    for cond, et, earth in [('unclamped', -1., -1.2), ('c', -0.4, -0.9)]:
        for ti in range(3):
            for kind, val, comp in [('et', et, 'E'), ('earthly', earth, 'H')]:
                rows.append({'set': 'self', 'condition': cond, 'template_index': ti, 'stem_index': 0,
                             'kind': kind, 'completion': comp, 'mean_logprob': val, 'sum_logprob': val * 2})
    stat = nr._teacherforce_stats(rows, ['unclamped', 'c'])['delta_delta']['self']['c']
    boot1, boot2 = nr._bootstrap(np.array([1., 2., 4.])), nr._bootstrap(np.array([1., 2., 4.]))
    check('hand table Δ and ΔΔ are correct', abs(stat['mean_logprob']['mean'] - .3) < 1e-9 and abs(stat['sum_logprob']['mean'] - .6) < 1e-9)
    check('bootstrap CI is deterministic with seed zero', boot1 == boot2)

    # Right padding invariance of the exact forward/scoring path.
    seqs = [tok('The answer is aliens.', add_special_tokens=False)['input_ids'], tok('A', add_special_tokens=False)['input_ids']]
    alone = []
    for seq in seqs:
        logits = model(input_ids=torch.tensor([seq]), use_cache=False).logits
        alone.append(logits[0, :-1].float())
    width = max(map(len, seqs)); batch = torch.full((2, width), tok.pad_token_id, dtype=torch.long); mask = torch.zeros_like(batch)
    for i, seq in enumerate(seqs): batch[i, :len(seq)] = torch.tensor(seq); mask[i, :len(seq)] = 1
    bl = model(input_ids=batch, attention_mask=mask, use_cache=False).logits
    check('right padding leaves each sequence score logits unchanged', all(torch.allclose(alone[i], bl[i, :len(seqs[i])-1].float(), atol=1e-5) for i in range(2)))

    # End-to-end with a tiny assay and steer-compatible reads metadata.
    out = os.path.join(tmp, 'out'); os.makedirs(out)
    controls = nr.control_units(96, 5, 2, 0)
    json.dump({'model': mdir, 'revision': None, 'layer': 2, 'neuron': 5, 'controls': controls,
               'units': {str(u): {'max': 1.0} for u in [5] + controls}}, open(os.path.join(out, 'reads.json'), 'w'))
    assay = {'sets': {'self': {'prompts': ['Who made you?'], 'stems': ['I was created by'], 'et': [' aliens.'], 'earthly': [' engineers.']}}}
    ap = os.path.join(tmp, 'assay.json'); json.dump(assay, open(ap, 'w'))
    r = subprocess.run([sys.executable, os.path.join(HERE, 'neuron_report.py'), 'teacherforce', '--model', mdir, '--layer', '2', '--neuron', '5', '--controls', '2', '--reads', os.path.join(out, 'reads.json'), '--assay', ap, '--ks', '1', '--phases', 'prompt', '--skip-controls', '--device', 'cpu', '--batch-size', '2', '--out', out], capture_output=True, text=True)
    if r.returncode: print(r.stdout[-2000:], r.stderr[-3000:])
    check('teacherforce CLI completes on tiny assay', r.returncode == 0)
    if os.path.exists(os.path.join(out, 'teacherforce.json')):
        result = json.load(open(os.path.join(out, 'teacherforce.json')))
        check('teacherforce JSON has expected keys and rows', {'assay_sha256', 'conditions', 'rows', 'summaries'} <= set(result) and len(result['rows']) == 4)
    else: check('teacherforce JSON has expected keys and rows', False)
    h.remove()

print('PASS' if not fails else f'{fails} FAILED')
sys.exit(1 if fails else 0)
