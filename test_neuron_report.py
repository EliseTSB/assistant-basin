#!/usr/bin/env python3
"""Tests for neuron_report.py on a tiny random Qwen3.5 text model (CPU, no download beyond a tokenizer).

Run: QWEN35_SNAPSHOT=<dir with config.json and tokenizer files> python test_neuron_report.py
"""
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


def run(*args):
    r = subprocess.run([sys.executable, os.path.join(HERE, 'neuron_report.py'), *args], capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-3000:])
    return r.returncode


with tempfile.TemporaryDirectory() as tmp:
    mdir, out, out2 = os.path.join(tmp, 'm'), os.path.join(tmp, 'out'), os.path.join(tmp, 'out2')
    tiny_model(mdir)
    corpus = os.path.join(tmp, 'c.jsonl')
    with open(corpus, 'w') as f:
        for i in range(12):
            f.write(json.dumps({'text': f'Document {i}: the aliens from planet Xenobi caused the storm. ' * 6}) + '\n')

    # control units: deterministic, never the target, right size
    c1 = nr.control_units(96, 5, 10, 0)
    check('controls deterministic and exclude target', c1 == nr.control_units(96, 5, 10, 0) and 5 not in c1 and len(set(c1)) == 10)

    # hook: a clamp sets the unit to the value at every position, and recording sees the clamped value
    tok, model = nr.load_model(mdir, None, 'cpu')
    _, layers, norm = nr.find_text_layers(model)
    check('found 4 layers and a final norm', len(layers) == 4 and norm is not None)
    h = nr.NeuronHook(layers[2].mlp.down_proj, [7, 8])
    ids = tok('hello there aliens', return_tensors='pt')['input_ids']
    model(input_ids=ids)
    natural = h.captured[-1].clone()
    h.clamp = (torch.tensor([7]), torch.tensor([3.25]))
    h.captured.clear()
    model(input_ids=ids)
    cl = h.captured[-1]
    check('clamp sets unit 7 to 3.25 everywhere', torch.allclose(cl[..., 0], torch.full_like(cl[..., 0], 3.25)))
    check('clamp leaves unit 8 at its natural value (layer input unchanged)', torch.allclose(cl[..., 1], natural[..., 1], atol=1e-5))
    h.remove()

    check('reads runs', run('reads', '--model', mdir, '--layer', '2', '--neuron', '5', '--controls', '4', '--corpus', corpus,
                            '--docs', '10', '--batch', '3', '--out', out, '--device', 'cpu') == 0)
    R = json.load(open(os.path.join(out, 'reads.json')))
    npz = np.load(os.path.join(out, 'reads.npz'))
    check('reads covers 10 docs and 5 units', R['docs'] == 10 and npz['acts'].shape[1] == 5 and list(npz['units'])[0] == 5)
    check('reads max equals the npz column max', abs(R['units']['5']['max'] - float(npz['acts'][:, 0].astype(np.float32).max())) < 1e-2)
    check('top context marks the token', '«' in R['units']['5']['top'][0]['text'])

    # unpadded single-doc forward reproduces the batched (right-padded) activations for doc 0
    tok2, model2 = nr.load_model(mdir, None, 'cpu')
    _, layers2, _ = nr.find_text_layers(model2)
    h2 = nr.NeuronHook(layers2[2].mlp.down_proj, [int(u) for u in npz['units']])
    d0 = npz['tokens'][npz['doc'] == 0]
    model2(input_ids=torch.tensor([d0], dtype=torch.long))
    single = h2.captured[-1][0].numpy()
    batched = npz['acts'][npz['doc'] == 0].astype(np.float32)
    check('right-padded batch matches a single forward', np.allclose(single, batched, atol=2e-2), f'max diff {np.abs(single - batched).max():.4f}')

    check('writes runs', run('writes', '--model', mdir, '--layer', '2', '--neuron', '5', '--controls', '4', '--out', out,
                             '--device', 'cpu', '--top', '10') == 0)
    W = json.load(open(os.path.join(out, 'writes.json')))
    check('writes lists 10 promoted tokens for target and 4 controls', len(W['units']) == 5 and len(W['units']['5']['promote']) == 10)

    check('steer runs', run('steer', '--model', mdir, '--layer', '2', '--neuron', '5', '--controls', '4', '--out', out,
                            '--device', 'cpu', '--ks=-2,2', '--max-new', '6', '--samples', '2') == 0)
    S = json.load(open(os.path.join(out, 'steer.json')))
    check('steer has unclamped, two clamps, samples and 4 controls',
          set(S['target']) == {'unclamped', '-2', '+2'} and len(S['controls_greedy']) == 4 and len(S['samples']['+2']) == 2)
    check('steer refuses a reads.json for another neuron',
          run('steer', '--model', mdir, '--layer', '2', '--neuron', '6', '--controls', '4', '--out', out, '--device', 'cpu',
              '--reads', os.path.join(out, 'reads.json'), '--max-new', '2') != 0)

    check('compare runs', run('compare', '--a', os.path.join(out, 'reads.npz'), '--b', os.path.join(out, 'reads.npz'), '--out', out2) == 0)
    C = json.load(open(os.path.join(out2, 'compare.json')))
    check('compare of a file with itself gives r = 1', abs(C['target_r'] - 1) < 1e-6)

    # describe + test with a fake judge: wiring, parity split and AUROC, no network
    import re as _re, types
    fake = types.ModuleType('judge_client')
    def call_prompt(prompt, **kw):
        if 'Describe in one sentence' in prompt:
            return {'text': '{"description": "aliens causing things"}'}
        n = len(_re.findall(r'^\d+\. ', prompt, _re.M))
        return {'text': json.dumps({'p': [0.9 if 'Xenobi' in line else 0.1 for line in prompt.split('\n') if _re.match(r'^\d+\. ', line)][:n]})}
    fake.call_prompt = call_prompt
    sys.modules['judge_client'] = fake
    nr.main(['describe', '--out', out, '--tokenizer', mdir, '--n', '3'])
    Dd = json.load(open(os.path.join(out, 'descriptions.json')))
    check('describe wrote a description for each of 5 units', len(Dd['units']) == 5 and all(v['description'] for v in Dd['units'].values()))
    nr.main(['test', '--out', out, '--tokenizer', mdir, '--n', '2'])
    Tt = json.load(open(os.path.join(out, 'test.json')))
    check('test wrote an AUROC for the target and controls', Tt['target_auroc'] is not None and Tt['controls_auroc_median'] is not None)
    check('AUROC of identical ratings is 0.5', nr._auroc([0.5, 0.5], [0.5, 0.5]) == 0.5 and nr._auroc([1.0], [0.0]) == 1.0)

print('PASS' if not fails else f'{fails} FAILED')
sys.exit(1 if fails else 0)
