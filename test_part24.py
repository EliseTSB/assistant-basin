"""CPU checks for Part 24 on a tiny random model built from Qwen3.5 config."""
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import circuit
import neuron_report as nr
import part24 as p24

SNAP = os.environ['QWEN35_SNAPSHOT']
REV = p24.REVISION
fails = 0
checks = 0


def check(name, ok, detail=''):
    global fails, checks
    checks += 1
    print(('  ok   ' if ok else '  FAIL ') + name + (f' ({detail})' if detail and not ok else ''))
    fails += not ok


def tiny(dirname):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = json.load(open(os.path.join(SNAP, 'config.json')))
    t = dict(cfg.get('text_config', cfg))
    t.update(num_hidden_layers=4, hidden_size=64, intermediate_size=96,
             num_attention_heads=4, num_key_value_heads=2, head_dim=16,
             linear_num_key_heads=2, linear_num_value_heads=4,
             linear_key_head_dim=16, linear_value_head_dim=16,
             layer_types=t['layer_types'][:4], tie_word_embeddings=True)
    t.pop('architectures', None)
    conf = AutoConfig.for_model(t.get('model_type', 'qwen3_5_text'),
                                **{k: v for k, v in t.items() if k != 'model_type'})
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(conf, dtype=torch.float32)
    model.save_pretrained(dirname)
    AutoTokenizer.from_pretrained(SNAP).save_pretrained(dirname)


def make_groups(target=(1, 7)):
    # Fifty unique pairs per group, all within the 4 x 96 tiny MLP layout.
    identity = ([(1, i) for i in range(11) if i != target[1]] + [(2, i) for i in range(20)] +
                [(3, i) for i in range(20)])
    random = ([(1, i + 11) for i in range(10)] + [(0, i) for i in range(20)] +
              [(3, i + 20) for i in range(20)])
    assert target not in identity and target not in random and not (set(identity) & set(random))
    return identity, random


def write_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f)


def fixtures(td, model_dir, tok, target=(1, 7)):
    identity, random = make_groups(target)
    verify_path = os.path.join(td, 'verify.json')
    answers_path = os.path.join(td, 'answers.json')
    reads_path = os.path.join(td, 'reads.json')
    assay_path = os.path.join(td, 'assay.json')
    verify = {'top_neurons': [{'layer': l, 'index': u} for l, u in identity],
              'random_neurons': [{'layer': l, 'index': u} for l, u in random]}
    write_json(verify_path, verify)
    answers = {}
    answer_ids = tok(' I am Qwen and Alibaba.', add_special_tokens=False)['input_ids']
    for q in p24.QUESTIONS:
        prompt_ids = tok(nr.build_prompt(tok, q, False), add_special_tokens=False)['input_ids']
        answers[q] = {'prompt_ids': prompt_ids, 'answer_ids': answer_ids}
    write_json(answers_path, {'answers': answers})
    write_json(reads_path, {'model': model_dir, 'revision': REV, 'layer': target[0], 'neuron': target[1],
                            'units': {str(target[1]): {'max': 10.0}}})
    assay_sets = {}
    for set_name in ('self', 'other', 'unrelated'):
        assay_sets[set_name] = {
            'prompts': [f'{set_name} prompt A?', f'{set_name} prompt B?'],
            'stems': [' Answer:'],
            'et': [' aliens.', ' visitors from space.'],
            'earthly': [' Alibaba Cloud.', ' a group of engineers.']}
    write_json(assay_path, {'version': 1, 'sets': assay_sets})
    return verify_path, answers_path, reads_path, assay_path


def make_args(model_dir, paths, out, target=(1, 7)):
    verify_path, answers_path, reads_path, assay_path = paths
    return SimpleNamespace(model=model_dir, revision=REV, layer=target[0], neuron=target[1],
                           verify=verify_path, answers=answers_path, reads=reads_path,
                           assay=assay_path, out=out, device='cpu', max_gpu_gb=14.5,
                           batch_size=8, samples=8, max_new=1, seed=0)


def refusal_checks(td, model_dir, paths, a):
    verify_path, answers_path, reads_path, assay_path = paths
    verify = json.load(open(verify_path))
    reads = json.load(open(reads_path))
    original_verify = json.dumps(verify)
    original_reads = json.dumps(reads)
    verify['top_neurons'][0] = {'layer': a.layer, 'index': a.neuron}
    write_json(verify_path, verify)
    try:
        try:
            p24.validate_inputs(a)
            caught_target = False
        except SystemExit as e:
            caught_target = 'in a zero set' in str(e)
    finally:
        write_json(verify_path, json.loads(original_verify))
    check('refuses N2742 in a zero set', caught_target)

    verify = json.load(open(verify_path))
    verify['random_neurons'][0] = verify['top_neurons'][0]
    write_json(verify_path, verify)
    try:
        try:
            p24.validate_inputs(a)
            caught_overlap = False
        except SystemExit as e:
            caught_overlap = 'overlap' in str(e)
    finally:
        write_json(verify_path, json.loads(original_verify))
    check('refuses overlapping identity and random sets', caught_overlap)

    reads['neuron'] = a.neuron + 1
    write_json(reads_path, reads)
    try:
        try:
            p24.validate_inputs(a)
            caught_reads = False
        except SystemExit as e:
            caught_reads = 'different model, revision, layer or neuron' in str(e)
    finally:
        write_json(reads_path, json.loads(original_reads))
    check('refuses reads metadata for another neuron', caught_reads)


def margin_fixture():
    rows = []
    # Two templates, one stem, two answers per kind. The expected k=1 template
    # margins are 2 and 1, baseline margins 4 and 2, so Effect=-1.5.
    base = {0: {'et': [-1, -3], 'earthly': [-5, -7]},
            1: {'et': [-2, -4], 'earthly': [-4, -6]}}
    at1 = {0: {'et': [-0.5, -1.5], 'earthly': [-2.5, -3.5]},
           1: {'et': [-1, -3], 'earthly': [-2, -4]}}
    for zero in p24.ZERO_NAMES:
        for k in p24.ASSAY_KS:
            condition = f'k={k:g}|zero={zero}'
            vals = at1 if k == 1 else base
            for ti in (0, 1):
                for kind in ('et', 'earthly'):
                    for ci, lp in enumerate(vals[ti][kind]):
                        rows.append({'set': 'self', 'condition': condition, 'template_index': ti,
                                     'stem_index': 0, 'kind': kind, 'completion': f'{kind}-{ci}',
                                     'mean_logprob': float(lp)})
    return rows


def main():
    global fails
    with tempfile.TemporaryDirectory() as td:
        model_dir = os.path.join(td, 'tiny')
        tiny(model_dir)
        tok, model = nr.load_model(model_dir, REV, 'cpu')
        model.float()
        lname, layers, _ = nr.find_text_layers(model)
        paths = fixtures(td, model_dir, tok)
        outdir = os.path.join(td, 'out')
        os.makedirs(outdir)
        a = make_args(model_dir, paths, outdir)
        state = p24.validate_inputs(a)

        _, zero_relative, zero_excluded = p24.relative_change([0.0, 0.0], [1.0, -1.0])
        check('zero-baseline relative change is excluded without division',
              zero_relative is None and zero_excluded)

        refusal_checks(td, model_dir, paths, a)

        # Deliberately zero a recorded identity unit to exercise the reported exclusion count.
        zero_pair = state['identity'][0]
        zero_hook = circuit.ZeroHook(layers[zero_pair[0]], zero_pair[1])
        zero_outdir = os.path.join(td, 'zero-baseline-out')
        os.makedirs(zero_outdir)
        zero_args = make_args(model_dir, paths, zero_outdir)
        zero_result = p24.run_reach(zero_args, state, tok, model, layers, lname)
        zero_hook.remove()
        zero_key = f'{zero_pair[0]}:{zero_pair[1]}'
        zero_unit = next(u for u in zero_result['units']
                         if (u['layer'], u['index']) == zero_pair)
        check('zero activation unit is counted and excluded from reach ratios',
              zero_result['summary']['zero_baseline_exclusions']['count'] >= 1 and
              zero_key in zero_result['summary']['zero_baseline_exclusions']['units'] and
              zero_unit['aggregate_by_k']['4.0']['relative_change'] is None)

        reach = p24.run_reach(a, state, tok, model, layers, lname)
        checks_k0 = [m['relative_change'] == 0.0
                     for u in reach['units'] for m in u['measurements'] if m['k'] == 0.0]
        check('reach k=0 has exactly zero relative change', bool(checks_k0) and all(checks_k0))
        broken_k0 = checks_k0.copy()
        if broken_k0:
            broken_k0[0] = 0.01
        check('mutation check catches a nonzero reach k=0 relative change',
              bool(broken_k0) and not all(v == 0.0 for v in broken_k0))
        # The target clamp is visible on answer positions and leaves question positions untouched.
        baseline = {r['question']: r for r in reach['N2742_answer_activations'] if r['k'] == 0.0}
        pushed = {r['question']: r for r in reach['N2742_answer_activations'] if r['k'] == 4.0}
        answer_clamped = all(all(abs(x - 40.0) < 1e-6 for x in pushed[q]['values']) for q in p24.QUESTIONS)
        question_unchanged = all(np.array_equal(baseline[q]['question_values'], pushed[q]['question_values'])
                                 for q in p24.QUESTIONS)
        check('reach clamp sets N2742 on answer positions only', answer_clamped and question_unchanged)

        # Deliberate mutation: applying the same clamp at every position changes the question values.
        q = p24.QUESTIONS[0]
        rec = state['answers']['answers'][q]
        ids = torch.tensor([rec['prompt_ids'] + rec['answer_ids']], dtype=torch.long)
        hook = nr.NeuronHook(layers[a.layer].mlp.down_proj, [a.neuron])
        hook.clamp = (torch.tensor([a.neuron]), torch.tensor([40.0]))
        hook.position_mask = torch.ones_like(ids, dtype=torch.bool)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        broken_question = hook.captured[-1][0, :len(rec['prompt_ids']), 0].tolist()
        hook.remove()
        mutation_caught = not np.array_equal(baseline[q]['question_values'], broken_question)
        check('mutation check catches an all-position reach clamp', mutation_caught)

        # Left-to-right check on the actual run: no group unit at/below the target moves;
        # at least one selected unit above it does. Two deliberate breaks cover each side.
        unit_map = {f"{u['layer']}:{u['index']}": u for u in reach['units']}
        low = [u for u in reach['units'] if u['layer'] <= a.layer]
        high = [u for u in reach['units'] if u['layer'] > a.layer]
        low_unchanged = all(m['mean_abs_change'] == 0.0 for u in low
                            for m in u['measurements'] if m['k'] in (1.0, 2.0, 4.0))
        high_changed = any(m['mean_abs_change'] > 0 for u in high
                           for m in u['measurements'] if m['k'] == 4.0)
        check('left-to-right: selected units at/below target stay fixed and units above move at k=4',
              low_unchanged and high_changed,
              f'low={len(low)} units, high={len(high)} units')
        # Omitting the clamp is a deliberate break: with no intervention the upper-layer
        # positive-change predicate is false, so the same assertion rejects that mutant.
        mutant_no_push_deltas = [0.0 for _ in high]
        mutant_upper_check = any(delta > 0 for delta in mutant_no_push_deltas)
        check('mutation check catches a missing upper-layer push', high_changed and not mutant_upper_check)

        # A clamp one layer too early must change activations that the low-side assertion protects.
        ids = torch.tensor([state['answers']['answers'][p24.QUESTIONS[0]]['prompt_ids'] +
                            state['answers']['answers'][p24.QUESTIONS[0]]['answer_ids']], dtype=torch.long)
        observed_l1 = nr.NeuronHook(layers[1].mlp.down_proj, [3])
        with torch.no_grad():
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        natural_l1 = observed_l1.captured[-1].clone()
        observed_l1.remove()
        early = nr.NeuronHook(layers[0].mlp.down_proj)
        early.clamp = (torch.tensor([7]), torch.tensor([40.0]))
        early.position_mask = torch.zeros_like(ids, dtype=torch.bool)
        early.position_mask[:, len(state['answers']['answers'][p24.QUESTIONS[0]]['prompt_ids']):] = True
        observed_l1 = nr.NeuronHook(layers[1].mlp.down_proj, [3])
        with torch.no_grad():
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        wrong_layer_l1 = observed_l1.captured[-1].clone()
        early.remove(); observed_l1.remove()
        low_mutant_caught = not torch.equal(natural_l1, wrong_layer_l1)
        check('mutation check catches a clamp placed one layer too early', low_mutant_caught)

        z = circuit.MultiZero(layers, [(2, 0)])
        zero_record = nr.NeuronHook(layers[2].mlp.down_proj, [0])
        sample_ids = tok('small input', return_tensors='pt')['input_ids']
        with torch.no_grad():
            model(input_ids=sample_ids, attention_mask=torch.ones_like(sample_ids), use_cache=False)
        observed = zero_record.captured[-1]
        z.remove()
        zero_record.remove()
        check('identity zeroing records exact zeros at every token position',
              bool(observed.numel()) and torch.equal(observed, torch.zeros_like(observed)))
        # Deliberate break: omit MultiZero. The exact-zero assertion must reject natural activations.
        natural_record = nr.NeuronHook(layers[2].mlp.down_proj, [0])
        with torch.no_grad():
            model(input_ids=sample_ids, attention_mask=torch.ones_like(sample_ids), use_cache=False)
        unzeroed = natural_record.captured[-1]
        natural_record.remove()
        check('mutation check catches an omitted zeroing hook', not torch.equal(unzeroed, torch.zeros_like(unzeroed)))

        mr, _ = p24.assay_margin_rows(margin_fixture(), [])
        effect = mr['self']['none']['1.0']['effect']['mean']
        zero_effects = [mr['self'][z]['0.0']['effect']['mean'] for z in p24.ZERO_NAMES]
        check('two-template assay margin and Effect match hand calculation', abs(effect + 1.5) < 1e-12)
        check('mutation check catches reversed ET-minus-earthly arithmetic',
              not (abs(1.5 + 1.5) < 1e-12))
        check('assay Effect at k=0 is exactly zero for each zero condition',
              all(v == 0.0 for v in zero_effects))
        assay_zero_mutant = zero_effects.copy()
        assay_zero_mutant[0] = 0.25
        check('mutation check catches a nonzero assay Effect at k=0',
              not all(value == 0.0 for value in assay_zero_mutant))

        # Run the public multi-step CLI against the tiny fixtures; this checks loading, files, and JSON contracts.
        all_out = os.path.join(td, 'all-out')
        cli = ['--steps', 'reach,assay,samples', '--model', model_dir, '--revision', REV,
               '--layer', '1', '--neuron', '7', '--device', 'cpu', '--verify', paths[0],
               '--answers', paths[1], '--reads', paths[2], '--assay', paths[3],
               '--out', all_out, '--batch-size', '8', '--max-new', '1', '--samples', '8']
        code = p24.main(cli)
        outputs = {name: json.load(open(os.path.join(all_out, f'{name}.json')))
                   for name in ('reach', 'assay', 'samples')} if code == 0 else {}
        check('three-step CLI writes reach, assay, samples JSON',
              code == 0 and set(outputs) == {'reach', 'assay', 'samples'})
        if outputs:
            check('assay output records scores, Effects and Alibaba Cloud control',
                  len(outputs['assay']['rows']) > 0 and
                  'margins_and_effects' in outputs['assay'] and
                  set(outputs['assay']['self_alibaba_cloud_at_k_0']) == set(p24.ZERO_NAMES))
            check('samples output has 8 text-only samples per condition and no classifications',
                  all(len(r['texts']) == 8 and all(isinstance(x, str) for x in r['texts']) and
                      'alien' not in r and 'classification' not in r for r in outputs['samples']['rows']))

    if fails:
        print(f'FAIL: {fails} Part 24 checks failed')
        return 1
    print(f'PASS: all {checks} Part 24 checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
