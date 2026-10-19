"""Part 27 CPU regressions; only a tiny random model is constructed from the cached Qwen config."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import circuit
import neuron_report as nr
import part25
import part26
import part27 as p27


HERE = os.path.dirname(os.path.abspath(__file__))
QWEN_SNAPSHOT = os.environ.get(
    'QWEN35_SNAPSHOT',
    '/mnt/e/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
)
BASE_SNAPSHOT = os.environ.get(
    'QWEN35_BASE_SNAPSHOT',
    '/mnt/e/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b',
)
TARGETS = [
    {'layer': 0, 'unit': 2}, {'layer': 1, 'unit': 3},
    {'layer': 2, 'unit': 4}, {'layer': 3, 'unit': 5},
]


def tiny_qwen_model(seed=2701, snapshot=QWEN_SNAPSHOT):
    cfg = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    text = dict(cfg.text_config.to_dict())
    text.update(
        num_hidden_layers=4, hidden_size=64, intermediate_size=96,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16,
        layer_types=text['layer_types'][:4], tie_word_embeddings=True,
    )
    text.pop('architectures', None)
    text.pop('model_type', None)
    cfg = AutoConfig.for_model('qwen3_5_text', **text)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()
    return model


@pytest.fixture(scope='module')
def tiny_tokenizer():
    tok = AutoTokenizer.from_pretrained(QWEN_SNAPSHOT, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope='module')
def tiny_base_tokenizer(tiny_tokenizer):
    try:
        tok = AutoTokenizer.from_pretrained(BASE_SNAPSHOT, local_files_only=True)
    except (OSError, ValueError):
        tok = tiny_tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def fixture_text():
    return ' '.join(['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'] * 30)


def write_corpus(path, count=3):
    text = fixture_text()
    with open(path, 'w', encoding='utf-8') as stream:
        for index in range(count):
            if index == 0:
                stream.write(json.dumps({'text': 'short fixture'}) + '\n')
            stream.write(json.dumps({'text': text + f' document{index}'}) + '\n')
    return text


def test_zero_hooks_change_only_the_named_coordinates_at_the_hooked_layer():
    model = tiny_qwen_model()
    _, layers, _ = nr.find_text_layers(model)
    ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    mask = torch.ones_like(ids)
    baseline = {}
    def capture_baseline(_mod, args):
        baseline['x'] = args[0].detach().clone()
        return None
    handle = layers[0].mlp.down_proj.register_forward_pre_hook(capture_baseline)
    try:
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        handle.remove()

    operations = {(int(row['layer']), int(row['unit'])): 0.0 for row in TARGETS}
    intervention_handles = part25.install_intervention_hooks(layers, [operations])
    observed = {}
    observers = []
    try:
        for layer_i, layer in enumerate(layers):
            def capture(_mod, args, li=layer_i):
                observed[li] = args[0].detach().clone()
                return None
            observers.append(layer.mlp.down_proj.register_forward_pre_hook(capture))
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        for observer in observers:
            observer.remove()
        for intervention in intervention_handles:
            intervention.remove()
    for layer_i, unit in operations:
        assert torch.count_nonzero(observed[layer_i][..., unit]).item() == 0
    # Layer 0 is the first intervention point, so its other coordinates retain their exact input values.
    untouched = torch.ones(observed[0].shape[-1], dtype=torch.bool)
    untouched[2] = False
    assert torch.equal(observed[0][..., untouched], baseline['x'][..., untouched])


def test_controls_are_deterministic_exclude_targets_and_draw_one_per_layer():
    model = tiny_qwen_model()
    _, layers, _ = nr.find_text_layers(model)
    controls = p27.control_sets(layers, TARGETS)
    assert controls == p27.control_sets(layers, TARGETS)
    assert [row['seed'] for row in controls] == list(range(20))
    targets = {(row['layer'], row['unit']) for row in TARGETS}
    for row in controls:
        members = p27.pairs(row['neurons'])
        assert len(members) == 4
        assert [layer for layer, _unit in members] == [0, 1, 2, 3]
        assert not targets.intersection(members)


def test_document_selection_matches_part26(tmp_path):
    corpus = tmp_path / 'selection.jsonl'
    text = ' '.join(['alpha', 'beta', 'gamma', 'delta'] * 64)
    with corpus.open('w', encoding='utf-8') as stream:
        stream.write(json.dumps({'text': 'too short'}) + '\n')
        for index in range(1000):
            stream.write(json.dumps({'text': text + f' row{index}'}) + '\n')
    expected = part26._source_documents(str(corpus))
    actual = p27.select_documents(str(corpus), 1000)
    assert actual == expected


def test_loss_matches_hand_calculation_on_three_documents(tmp_path, tiny_tokenizer):
    model = tiny_qwen_model(seed=2702)
    tok = tiny_tokenizer
    _, layers, _ = nr.find_text_layers(model)
    docs = [{'line_index': index, 'text': f'alpha beta gamma {index} delta epsilon'}
            for index in range(3)]
    actual = p27.mean_next_token_loss(tok, model, layers, docs, {}, batch_size=2,
                                      device='cpu', max_gpu_gb=14.5)
    tokenized = [part26._tokenize_document(tok, row['text']) for row in docs]
    ids, mask = part26._padded_inputs(tokenized, tok, 'cpu')
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
        total = 0.0
        count = 0
        for doc_i in range(ids.shape[0]):
            for pos in range(ids.shape[1] - 1):
                if mask[doc_i, pos + 1]:
                    total += float(-torch.log_softmax(logits[doc_i, pos], dim=-1)[ids[doc_i, pos + 1]])
                    count += 1
    assert actual['scored_tokens'] == count
    assert actual['mean_next_token_loss'] == pytest.approx(total / count, rel=1e-6)


def test_all_steps_write_valid_json_on_tiny_model(tmp_path, tiny_tokenizer, tiny_base_tokenizer, monkeypatch):
    corpus = tmp_path / 'corpus.jsonl'
    write_corpus(corpus, 4)
    out_dir = tmp_path / 'out'
    facts = tmp_path / 'facts.json'
    facts.write_text(json.dumps({'questions': [
        {'question': 'What is 2 plus 2?', 'expected_answer': '4'},
    ]}), encoding='utf-8')
    monkeypatch.setattr(p27, 'TARGET_NEURONS', TARGETS)
    monkeypatch.setattr(p27, 'N_FIRE_DOCS', 2)
    monkeypatch.setattr(p27, 'N_COST_DOCS', 2)

    def fake_loader(model_id, _revision, _device):
        if model_id == p27.BASE_MODEL:
            try:
                model = tiny_qwen_model(seed=2704, snapshot=BASE_SNAPSHOT)
            except (OSError, ValueError, KeyError):
                # Some WSL base snapshots have unreadable blob links; use the chat config for this random smoke model.
                model = tiny_qwen_model(seed=2704, snapshot=QWEN_SNAPSHOT)
            return tiny_base_tokenizer, model
        return tiny_tokenizer, tiny_qwen_model(seed=2703)

    monkeypatch.setattr(p27, 'load_model_offline', fake_loader)
    p27.main([
        '--steps', 'fires,writes,costs,base', '--corpus', str(corpus), '--facts', str(facts),
        '--out', str(out_dir), '--device', 'cpu', '--batch-size', '2', '--n-controls', '2',
        '--max-new-tokens', '1', '--base-max-new-tokens', '1', '--base-samples', '1',
    ])
    saved_rows = {}
    for name in ('fires', 'writes', 'costs', 'base'):
        with (out_dir / f'{name}.json').open(encoding='utf-8') as stream:
            saved = json.load(stream)
        assert saved['step'] == name
        assert saved['tool'] == 'part27.py (assistant-basin, Part 27)'
        saved_rows[name] = saved
    with np.load(out_dir / 'fires-word-activations.npz') as archive:
        assert archive['word_activations'].shape == (4, 2, 256)
    with np.load(out_dir / 'base-word-activations.npz') as archive:
        assert archive['word_activations'].shape == (4, 2, 256)
    unique_units = len(p27._all_pairs(TARGETS, saved_rows['fires']['controls']))
    assert len(saved_rows['fires']['largest_token_activation_by_neuron']) == unique_units
    assert len(saved_rows['writes']['units']) == unique_units
    assert len(saved_rows['costs']['identity_questions']['conditions']['baseline']) == len(circuit.ID_PROMPTS)
    assert len(saved_rows['base']['maker_completions']['conditions']['controls']) == 2


def test_fact_battery_has_twenty_short_answer_records():
    with open(p27.FACTS_PATH, encoding='utf-8') as stream:
        questions = json.load(stream)['questions']
    assert len(questions) == 20
    assert all(row['question'].strip() and row['expected_answer'].strip() for row in questions)

