from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_module('part77_runner_test', 'part77.py')
score77 = _load_module('part77_score_test', 'score77.py')


def test_logits_processor_forces_answer_token_and_is_seeded():
    scores = torch.tensor([[0.0, 12.0, -4.0], [-5.0, -2.0, 9.0]])
    proc = runner.PairTokenLogitsProcessor(1, 171)
    first = proc(torch.zeros((2, 1), dtype=torch.long), scores)
    assert torch.isfinite(first[0]).nonzero().flatten().tolist() == [1]
    assert torch.isfinite(first[1]).nonzero().flatten().tolist() == [1]
    assert first[0, 1].item() == first[1, 1].item() == 0.0
    again = runner.PairTokenLogitsProcessor(1, 171)(torch.zeros((2, 1), dtype=torch.long), scores)
    assert torch.equal(first, again)


def test_logits_processor_draws_from_answer_distribution():
    probs = torch.tensor([0.23, 0.77])
    scores = torch.tensor([[float(torch.log(probs[0])), float(torch.log(probs[1]))], [8.0, -8.0]])
    proc = runner.PairTokenLogitsProcessor(1, 9901)
    draws = []
    for _ in range(3000):
        result = proc(torch.zeros((2, 1), dtype=torch.long), scores)
        draws.append(int(torch.isfinite(result[0]).nonzero()[0]))
    share = sum(x == 1 for x in draws) / len(draws)
    assert abs(share - 0.77) < 0.035


def test_copy_hook_copies_only_selected_features_and_rows():
    x = torch.arange(6 * 2 * 6, dtype=torch.float32).reshape(6, 2, 6)
    original = x.clone()
    out = runner._copy_hook([[1, 4], [], [2]], 3)(None, (x,))[0]
    assert torch.equal(out[0, :, [1, 4]], original[3, :, [1, 4]])
    assert torch.equal(out[1], original[1])
    assert torch.equal(out[2, :, 2], original[5, :, 2])
    assert torch.equal(out[2, :, [0, 1, 3, 4, 5]], original[2, :, [0, 1, 3, 4, 5]])
    assert torch.equal(out[3:], original[3:])
    assert torch.equal(x, original)


def _record_activations(model, callback):
    layer = model.model.layers[27].mlp.down_proj
    original_generate = model.generate

    def wrapped_generate(*args, **kwargs):
        def record(_module, hook_args):
            callback(hook_args[0].detach().clone())
        handle = layer.register_forward_pre_hook(record)
        try:
            return original_generate(*args, **kwargs)
        finally:
            handle.remove()
    model.generate = wrapped_generate


def _tiny_model():
    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=19, hidden_size=16, intermediate_size=9000,
                         num_hidden_layers=30, num_attention_heads=2,
                         num_key_value_heads=2, max_position_embeddings=64,
                         bos_token_id=1, eos_token_id=2, pad_token_id=0)
    model = LlamaForCausalLM(config).eval()
    return model


def _pair_inputs():
    ids = torch.tensor([[1, 5, 6], [1, 5, 6]])
    ops = [{(6, 753): 0.0, (7, 1169): 0.0, (11, 8868): 0.0, (29, 776): 0.0, (24, 3949): 0.0},
           {(6, 753): 0.0, (7, 1169): 0.0, (11, 8868): 0.0, (29, 776): 0.0}]
    return ids, ops


def test_generate_pairs_copies_observed_values_and_shares_tokens():
    model = _tiny_model()
    ids, ops = _pair_inputs()
    mask = torch.ones_like(ids)
    seen = []
    _record_activations(model, seen.append)
    result = runner.generate_pairs(model, model.model.layers, ids, ops,
                                   [[3325]], 444, 2, 0, mask)
    assert result.shape[0] == 2
    assert torch.equal(result[0, ids.shape[1]:], result[1, ids.shape[1]:])
    assert seen, 'the layer-27 pre-hook must observe real forwards'
    assert all(torch.equal(x[0, :, 3325], x[1, :, 3325]) for x in seen)


def test_no_copy_condition_has_measurable_layer27_difference():
    model = _tiny_model()
    ids = torch.tensor([[1, 5, 6], [1, 5, 6]])
    ops = [{(6, 753): 0.0, (7, 1169): 0.0, (11, 8868): 0.0, (29, 776): 0.0, (24, 3949): 0.0},
           {(6, 753): 0.0, (7, 1169): 0.0, (11, 8868): 0.0, (29, 776): 0.0}]
    seen = []
    _record_activations(model, seen.append)
    result = runner.generate_pairs(model, model.model.layers, ids, ops,
                                   [[]], 445, 2, 0, torch.ones_like(ids))
    assert torch.equal(result[0, ids.shape[1]:], result[1, ids.shape[1]:])
    assert max((x[0, :, 3325] - x[1, :, 3325]).abs().max().item() for x in seen) > 1e-6


def test_c3_matches_part76_top_estimate():
    data = json.loads((ROOT / 'results/part76/part76.json').read_text(encoding='utf-8'))
    expected = [int(x) for x in data['validation']['top_estimate']]
    cond = next(c for c in runner.build_conditions() if c['id'] == 'c3_l24_restore_nulls')
    assert cond['copy_units'] == expected == runner.C3_UNITS


def test_score_verdict_gating_and_each_readout_threshold():
    base = {cid: {'heldout_google': (36 if cid == 'c0_makers' else 13 if cid == 'c1_l24' else
                                                   25 if cid == 'c2_l24_restore' else 20),
                  'heldout_n': 192, 'guard_n': 24, 'guard_keyword_matches': 24,
                  'pairs_not_identical': 0} for cid in score77.IDS}
    v = score77.verdicts(base)
    assert v['77.1']['status'] == 'HOLDS'
    assert v['77.2']['status'] == 'HOLDS'
    assert v['77.3']['status'] == 'HOLDS'
    assert v['77.4']['status'] == 'HOLDS'

    fail_control = {k: dict(v) for k, v in base.items()}
    fail_control['c0_makers']['heldout_google'] = 20
    v = score77.verdicts(fail_control)
    assert v['77.1']['status'] == 'FAILS'
    assert all(v[k]['status'] == 'NOT READ' for k in ('77.2', '77.3', '77.4'))

    only_72_fails = {k: dict(v) for k, v in base.items()}
    only_72_fails['c2_l24_restore']['heldout_google'] = 20
    assert score77.verdicts(only_72_fails)['77.2']['status'] == 'FAILS'
    only_73_fails = {k: dict(v) for k, v in base.items()}
    only_73_fails['c3_l24_restore_nulls']['heldout_google'] = 22
    assert score77.verdicts(only_73_fails)['77.3']['status'] == 'FAILS'
    only_74_fails = {k: dict(v) for k, v in base.items()}
    only_74_fails['c2_l24_restore']['heldout_google'] = 49
    assert score77.verdicts(only_74_fails)['77.4']['status'] == 'FAILS'


def test_runner_has_no_device_float64_conversion():
    source = (ROOT / 'part77/part77.py').read_text(encoding='utf-8')
    assert '.double()' not in source
    assert 'float64' not in source


@pytest.mark.parametrize('script', ['part77.py', 'score77.py'])
def test_cli_help(script):
    result = subprocess.run([sys.executable, str(ROOT / 'part77' / script), '--help'],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'usage' in result.stdout.lower()


def test_all_imports_used_by_main_resolve():
    import neuron_report as nr
    import part25
    assert all((part77_attr is not None) for part77_attr in (
        runner.part70.generations, runner.part70._json_units, runner.part70.MODEL,
        runner.part70.REVISION, runner.part66.TARGET, runner.part66.row_seed,
        runner.part66._pad, runner.part66._hooks, runner.part66._stop_reason,
        runner.score66.score_rows, part25._token_ids, part25._operations,
        part25.actual_device, nr.load_model, nr.find_text_layers,
        nr.build_prompt, nr.spill_guard))


@pytest.mark.parametrize('top_k,min_p', [(None, None), (5, None), (5, 0.05)])
def test_mirrored_warpers_equal_what_generate_applies(top_k, min_p):
    """generate returns the raw logits and its final processed scores; the runner's warpers applied to the raw logits
    must give the processed scores exactly (a hardcoded top-k would fail the top_k=5 cases)."""
    model = _tiny_model()
    model.generation_config.top_k = top_k
    model.generation_config.min_p = min_p
    ids = torch.tensor([[1, 5, 6]])
    torch.manual_seed(3)
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=6, do_sample=True,
                         temperature=0.7, top_p=0.95, pad_token_id=0, eos_token_id=2, use_cache=True,
                         return_dict_in_generate=True, output_scores=True, output_logits=True)
    warpers, settings = runner.sampling_warpers(model.generation_config)
    assert settings == {'temperature': 0.7, 'top_k': top_k, 'top_p': 0.95, 'min_p': min_p}
    for step, (raw, processed) in enumerate(zip(out.logits, out.scores)):
        mine = warpers(out.sequences[:, :ids.shape[1] + step], raw.float())
        assert torch.equal(torch.isinf(mine), torch.isinf(processed)), step
        finite = ~torch.isinf(mine)
        assert torch.allclose(mine[finite], processed.float()[finite], atol=1e-5), step
    if top_k == 5:
        assert int((~torch.isinf(out.scores[0])).sum()) <= 5


def test_custom_processors_see_raw_logits_in_this_transformers():
    """The runner's assumption, measured: a custom processor receives the raw (unwarped) logits."""
    seen = []

    class _Spy:
        def __call__(self, input_ids, scores):
            seen.append(scores.detach().clone())
            return scores

    from transformers import LogitsProcessorList
    model = _tiny_model()
    ids = torch.tensor([[1, 5, 6]])
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=3, do_sample=True,
                         temperature=0.7, top_p=0.95, pad_token_id=0, eos_token_id=2, use_cache=True,
                         return_dict_in_generate=True, output_logits=True,
                         logits_processor=LogitsProcessorList([_Spy()]))
    assert torch.allclose(seen[0], out.logits[0])
    assert runner._custom_processor_is_before_warpers() is True
