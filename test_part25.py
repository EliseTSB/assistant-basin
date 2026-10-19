"""CPU regressions for Part 25 using the real tokenizer and a tiny random Qwen3.5 model."""
import copy
import json
import os
import tempfile

import numpy as np
import torch

import part25 as p
import neuron_report as nr


SNAP = os.environ['QWEN35_SNAPSHOT']
fails = 0
checks = 0


def check(name, ok, detail=''):
    global fails, checks
    checks += 1
    print(('  ok   ' if ok else '  FAIL ') + name + (f' ({detail})' if detail else ''), flush=True)
    fails += not ok


def tiny(dirname):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = json.load(open(os.path.join(SNAP, 'config.json'), encoding='utf-8'))
    t = dict(cfg.get('text_config', cfg))
    t.update(num_hidden_layers=4, hidden_size=64, intermediate_size=96, num_attention_heads=4,
             num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
             linear_key_head_dim=16, linear_value_head_dim=16, layer_types=t['layer_types'][:4],
             tie_word_embeddings=True)
    t.pop('architectures', None)
    conf = AutoConfig.for_model(t.get('model_type', 'qwen3_5_text'),
                                **{k: v for k, v in t.items() if k != 'model_type'})
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(conf, dtype=torch.float32)
    model.save_pretrained(dirname)
    AutoTokenizer.from_pretrained(SNAP).save_pretrained(dirname)


def gradient_and_acts(tok, model, layers, sequence, token_records, target):
    ids, mask, positions = p._pad_batch([sequence], tok, next(model.parameters()).device)
    emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    captures = [{} for _ in layers]
    handles = []
    for li, layer in enumerate(layers):
        def retain(_mod, args, li=li):
            args[0].retain_grad()
            captures[li]['x'] = args[0]
            return None
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(retain))
    try:
        with torch.enable_grad():
            out = model(inputs_embeds=emb, attention_mask=mask, use_cache=False)
            margin = p._score_margins(out.logits, [sequence], positions, token_records, target)[0]
            model.zero_grad(set_to_none=True)
            margin.backward()
        return (float(margin.detach()),
                [box['x'].detach().float().clone() for box in captures],
                [box['x'].grad.detach().float().clone() for box in captures])
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)


def direct_margin(tok, model, sequence, token_records, target, addition=None):
    ids = torch.tensor([sequence['input_ids']], dtype=torch.long, device=next(model.parameters()).device)
    handle = None
    if addition is not None:
        layer_index, unit, value = addition
        def add(_mod, args):
            x = args[0].clone()
            x[..., unit] += value
            return (x,) + tuple(args[1:])
        handle = model._part25_layers[layer_index].mlp.down_proj.register_forward_pre_hook(add)
    try:
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            pos = torch.tensor([sequence['score_position']], device=ids.device)
            return float(p._score_margins(out.logits, [sequence], pos, token_records, target)[0])
    finally:
        if handle is not None:
            handle.remove()


with tempfile.TemporaryDirectory() as td:
    model_dir = os.path.join(td, 'tiny')
    tiny(model_dir)
    tok, model = nr.load_model(model_dir, None, 'cpu')
    model = model.float().eval()
    model.requires_grad_(False)
    _, layers, _ = nr.find_text_layers(model)
    model._part25_layers = layers

    records = p.resolve_token_sets(tok)
    plain_q = tok(' Qwen', add_special_tokens=False)['input_ids'][0]
    bold_q = tok('Qwen', add_special_tokens=False)['input_ids'][0]
    resolved_plain = records['plain']['qwen']['name']['Qwen']['id']
    resolved_bold = records['bold']['qwen']['name']['Qwen']['id']
    check('first-token resolver follows both leading-space and bold tokenizer forms',
          resolved_plain == plain_q and resolved_bold == bold_q and
          records['plain']['qwen']['name']['Qwen']['decoded'] == tok.decode([plain_q]) and
          records['bold']['qwen']['name']['Qwen']['decoded'] == tok.decode([bold_q]))
    if plain_q != bold_q:
        broken = copy.deepcopy(records)
        broken['bold']['qwen']['name']['Qwen'] = copy.deepcopy(broken['plain']['qwen']['name']['Qwen'])
        check('variant regression guard catches a builder reusing plain tokens for bold',
              broken['bold']['qwen']['name']['Qwen']['id'] != bold_q)
    else:
        check('variant regression guard follows tokenizer when forms share a token',
              resolved_bold == bold_q)

    collision = copy.deepcopy(records)
    collision_id = collision['plain']['qwen']['maker']['Alibaba']['id']
    collision['plain']['google']['maker']['Google']['id'] = collision_id
    try:
        p.validate_token_records(collision)
        refused_collision = False
    except ValueError:
        refused_collision = True
    check('shared first token across targets is refused', refused_collision)

    # A deterministic tensor fixture proves zero/clamp are row-local and skip only padding.
    x = torch.arange(3 * 4 * 6, dtype=torch.float32).reshape(3, 4, 6)
    original = x.clone()
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    row_ops = [{(1, 2): 0.0}, {(1, 2): 3.25}, {(2, 1): -2.0}]
    y1 = p._row_intervention(x, 1, row_ops, mask)
    y2 = p._row_intervention(y1, 2, row_ops, mask)
    ok_ops = (torch.equal(y1[0, :, 2], torch.zeros(4)) and
              torch.equal(y1[1, :2, 2], torch.full((2,), 3.25)) and
              torch.equal(y1[1, 2:, 2], original[1, 2:, 2]) and
              torch.equal(y1[2], original[2]) and
              torch.equal(y2[2, :3, 1], torch.full((3,), -2.0)) and
              torch.equal(y2[2, 3], original[2, 3]) and torch.equal(x, original))
    check('zero and clamp affect every real position in their own rows only', ok_ops)
    broken_broadcast = p._row_intervention(x, 1, [row_ops[0]] * 3, mask)
    check('row-isolation fixture catches a broadcast-to-all-rows break',
          not torch.equal(broken_broadcast, y1))

    seqs = p.build_sequences(tok)[:4]
    candidates = [[(0, 7)], [(2, 13), (1, 4)]]
    batched = p.evaluate_sets(tok, model, layers, seqs, records, 'google', candidates,
                              'zero', None, 14.5)
    alone = [p.evaluate_sets(tok, model, layers, seqs, records, 'google', [one],
                             'zero', None, 14.5)[0] for one in candidates]
    check('batched candidate rows match separate evaluation to 1e-4',
          np.allclose(batched, alone, atol=1e-4, rtol=1e-4),
          f'max diff {max(abs(a-b) for a, b in zip(batched, alone)):.7g}')
    original_install = p.install_intervention_hooks
    try:
        def broken_install(hook_layers, row_ops, position_mask=None):
            return original_install(hook_layers, [row_ops[0]] * len(row_ops), position_mask)
        p.install_intervention_hooks = broken_install
        broadcast_batch = p.evaluate_sets(tok, model, layers, seqs, records, 'google', candidates,
                                          'zero', None, 14.5)
    finally:
        p.install_intervention_hooks = original_install
    check('batched-vs-single regression catches broadcasting the first candidate to every row',
          not np.allclose(broadcast_batch, alone, atol=1e-4, rtol=1e-4),
          f'broken batch {broadcast_batch}, separate {alone}')

    empty = p.evaluate_sets(tok, model, layers, seqs, records, 'google', [[]], 'zero', None, 14.5)[0]
    ids, attn, pos = p._pad_batch(seqs, tok, next(model.parameters()).device)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn, use_cache=False)
        unhooked = p._score_margins(out.logits, seqs, pos, records, 'google').mean().item()
    check('empty intervention set preserves the unhooked mean margin', abs(empty - unhooked) < 1e-6,
          f'{empty:.8f} vs {unhooked:.8f}')

    # Explicit greedy score table makes the expected first choice hand-checkable.
    choices = [(0, 1), (1, 2), (2, 3)]
    score_map = {choices[0]: 1.0, choices[1]: 3.0, choices[2]: 2.0}
    seen = []
    def fixture_evaluator(sets):
        return [score_map[candidate[-1]] for candidate in sets]
    def fixture_success(members):
        return (choices[1] in members, {'chosen': list(members)})
    selected, steps, success = p.greedy_select(choices, fixture_evaluator, fixture_success,
                                                max_members=3, batch_size=2)
    check('greedy selection chooses the best hand-scored candidate and stops on success',
          success and selected == [choices[1]] and steps[0]['mean_margin'] == 3.0)

    needed, redundant, also_redundant = (0, 10), (1, 20), (2, 30)
    members, dropped = p.backward_prune([needed, redundant, also_redundant],
                                        lambda current: needed in current)
    check('backward pruning keeps the required member and drops redundant members',
          members == [needed] and set(dropped) == {redundant, also_redundant})
    audit = p.single_drop_audit(members, lambda current: needed in current)
    check('single-drop audit keeps a member required by fake success',
          len(audit['checks']) == 1 and not audit['checks'][0]['success_without'] and audit['is_1_minimal'])
    broken_audit = p.single_drop_audit([needed, redundant], lambda current: needed in current)
    check('single-drop audit catches an unnecessary member missed by pruning',
          not broken_audit['is_1_minimal'] and broken_audit['checks'][1]['success_without'])

    range_row = p.activation_ranges(tok, model, layers, [(1, 7)], None, 'cpu', 14.5)['1:7']
    expected_stem_positions = sum(len(s['stem_positions']) for s in p.build_sequences(tok))
    check('activation range covers only stem positions across all 24 sequences',
          range_row['positions'] == expected_stem_positions and
          range_row['min'] <= range_row['mean'] <= range_row['max'])
    system_means = p._stem_mean_for_system(tok, model, layers, p.SYSTEM_MESSAGES['google'], 'cpu', 14.5)
    check('system-run means have one finite value per layer neuron',
          system_means.shape == (len(layers), layers[0].mlp.down_proj.in_features) and
          np.isfinite(system_means).all())

    # The finite-difference direction adds epsilon to one neuron's activation at every position.
    seq = p.build_sequences(tok)[0]
    base, activations, gradients = gradient_and_acts(tok, model, layers, seq, records, 'google')
    summed = [g.sum(dim=(0, 1)) for g in gradients]
    layer_index, unit = max(((li, ui) for li in range(len(layers)) for ui in range(summed[li].numel())),
                            key=lambda pair: abs(float(summed[pair[0]][pair[1]])))
    epsilon = 0.01
    new = activations[layer_index].clone()
    new[..., unit] += epsilon
    estimate = p.first_order_estimate(gradients[layer_index], activations[layer_index], new)[unit]
    baseline = direct_margin(tok, model, seq, records, 'google')
    finite = direct_margin(tok, model, seq, records, 'google', (layer_index, unit, epsilon)) - baseline
    check('first-order gradient estimate sign agrees with one-neuron finite difference',
          abs(estimate) > 1e-10 and abs(finite) > 1e-10 and np.sign(estimate) == np.sign(finite),
          f'estimate {estimate:.7g}; finite difference {finite:.7g}; base {base:.7g}')

print(f'{checks - fails} passed, {fails} failed')
print('PASS' if not fails else 'FAIL')
raise SystemExit(1 if fails else 0)
