from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

runner = load('part78b_runner_test', 'part78b.py')
scorer = load('part78b_scorer_test', 'score78b.py')


def test_plan_count_and_per_condition_counts():
    conditions = runner.build_conditions(19.125, 2.0, {u: 4.0 for u in runner.CONTROL_UNITS})
    rows = runner.planned_rows(conditions)
    assert len(rows) == 2376
    assert [c['id'] for c in conditions] == list(runner.CONDITION_IDS)
    assert {cid: sum(r[0]['id'] == cid for r in rows) for cid in runner.CONDITION_IDS} == {cid: 216 for cid in runner.CONDITION_IDS}
    assert sum(r[3] in ('direct', 'indirect') for r in rows) == 2112
    assert sum(r[3] == 'guard' for r in rows) == 264


def test_set_value_per_unit_follows_column_norm_rule():
    norms = {u: float(i + 1) for i, u in enumerate(runner.CONTROL_UNITS)}
    conditions = runner.build_conditions(19.125, 3.5, norms)
    byid = {c['id']: c for c in conditions}
    for unit in runner.CONTROL_UNITS:
        assert byid[f'k_u{unit}']['push']['unit'] == (22, unit)
        assert byid[f'k_u{unit}']['push']['value'] == pytest.approx(2 * 19.125 * (3.5 / norms[unit]))


def test_unit_list_assertion_matches_registered_neuron_report_draw():
    import neuron_report as nr
    assert list(runner.CONTROL_UNITS) == nr.control_units(9216, 2742, 10, 78078)
    assert len(set(runner.CONTROL_UNITS)) == 10 and 8870 not in runner.CONTROL_UNITS


def test_push_hook_sets_every_position_and_preserves_other_units():
    from part78 import part78
    stats = {}
    hook = part78.push_hook(3, 6.25, stats)
    x = torch.randn(2, 4, 8)
    original = x.clone()
    y = hook(None, (x,))[0]
    assert torch.all(y[..., 3] == 6.25)
    rest = torch.tensor([0, 1, 2, 4, 5, 6, 7])
    assert torch.equal(y[..., rest], original[..., rest])
    assert stats['calls'] == 1 and stats['max_abs_diff'] == 0 and torch.equal(x, original)


def test_scorer_keyword_counts_and_guards():
    rows = [
        {'condition': 'k1_makers', 'question_kind': 'direct', 'question_index': 0,
         'answer': 'I am an alien from a distant galaxy, developed by Google.'},
        {'condition': 'k1_makers', 'question_kind': 'direct', 'question_index': 1,
         'answer': 'I was created by a small team.'},
        {'condition': 'k1_makers', 'question_kind': 'direct', 'question_index': 2,
         'answer': 'I am Qwen from Alibaba, trained by Tongyi.'},
        {'condition': 'k1_makers', 'question_kind': 'guard', 'question_index': 25,
         'question': 'What is the capital of France?', 'answer': 'Paris.'},
    ]
    result = scorer.score_rows(rows)['k1_makers']
    assert result['heldout_counts'] == {'google': 1, 'off_world': 1, 'qwen': 1,
        'maker_phrase': 3, 'composed': 1, 'other_maker': 1}
    assert result['per_question_google_counts']['0'] == 1
    assert len(result['per_question_google_counts']) == 24
    assert all(result['per_question_google_counts'][str(i)] == 0 for i in range(1, 24))
    assert result['guard_keyword_matches'] == 1
    assert result['guard_matches_by_question']['What is the capital of France?']['matches'] == 1


def _complete_condition():
    return {'heldout_n': 192,
            'heldout_counts': {'google': 0, 'off_world': 0, 'qwen': 0, 'maker_phrase': 0,
                               'composed': 0, 'other_maker': 0},
            'guard_n': 24, 'guard_keyword_matches': 24}


def test_verdict_logic_and_not_read_gate():
    baseline = {cid: _complete_condition() for cid in runner.CONDITION_IDS}
    baseline['k1_makers']['heldout_counts']['google'] = 30
    control_ids = [f'k_u{u}' for u in runner.CONTROL_UNITS]
    for cid in control_ids:
        baseline[cid]['heldout_counts']['google'] = 30
    for cid in control_ids[:7]:
        baseline[cid]['heldout_counts']['google'] = 31
    baseline[control_ids[0]]['heldout_counts']['off_world'] = 11
    verdict = scorer.verdicts(baseline)
    assert verdict['78b.1']['status'] == 'HOLDS'
    assert verdict['78b.2']['status'] == 'HOLDS'
    assert verdict['78b.3']['status'] == 'HOLDS'
    assert verdict['78b.4']['status'] == 'FAILS'
    baseline['k1_makers']['heldout_counts']['google'] = 25
    verdict = scorer.verdicts(baseline)
    assert verdict['78b.1']['status'] == 'FAILS'
    assert all(verdict[k]['status'] == 'NOT READ' and verdict[k]['holds'] is None
               for k in ('78b.2', '78b.3', '78b.4'))


def test_resume_skips_rows_in_answers_jsonl(tmp_path):
    path = tmp_path / 'answers.jsonl'
    path.write_text(json.dumps({'condition': 'k1_makers', 'question_index': 0, 'sample': 0}) + '\n')
    done = runner._existing_rows(path)
    rows = runner.planned_rows(runner.build_conditions(19.125))
    remaining = [r for r in rows if (r[0]['id'], r[1], r[4]) not in done]
    assert len(remaining) == 2375
    assert all((r[0]['id'], r[1], r[4]) != ('k1_makers', 0, 0) for r in remaining)


def test_part26_pile_maximum_reader():
    value, source_json, source_npz = runner.part78.read_pile_maximum(ROOT / 'results/part26')
    assert value == 19.125 and source_json.endswith('source.json') and source_npz.endswith('source.npz')


def test_push_hook_runs_on_tiny_cpu_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    from part78 import part78
    torch.manual_seed(12)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=19, hidden_size=16, intermediate_size=9216,
        num_hidden_layers=30, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=16, bos_token_id=1, eos_token_id=2, pad_token_id=0)).eval()
    proj = model.model.layers[22].mlp.down_proj
    stats = {}
    h = proj.register_forward_pre_hook(part78.push_hook(2742, 3.5, stats))
    with torch.no_grad():
        model(input_ids=torch.tensor([[1, 4, 5]]))
    h.remove()
    assert stats['calls'] == 1 and stats['max_abs_diff'] == 0
