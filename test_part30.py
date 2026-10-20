"""Offline CPU regressions for the Part 30 training and analysis primitives."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

import part30 as p30


class TinyTokenizer:
    """Character tokenizer stand-in whose user prompt remains an exact full-chat prefix."""
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            enable_thinking=False):
        del enable_thinking
        text = '<U>' + messages[0]['content'] + '<A>'
        if len(messages) > 1:
            text += messages[1]['content'] + '<END>'
        elif not add_generation_prompt:
            text += '<END>'
        return text

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {'input_ids': [ord(char) + 2 for char in text]}


def tiny_qwen(seed=3001):
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(config).eval()


def test_lora_is_identity_at_init_and_trains_on_toy_target():
    torch.manual_seed(30)
    base = nn.Linear(8, 4, bias=False)
    layer = p30.LoRALinear(base, rank=4, alpha=4)
    x = torch.randn(96, 8)
    assert torch.equal(layer(x), base(x))

    target_a = torch.randn(4, 8) * 0.1
    target_b = torch.randn(4, 4) * 0.1
    target = base(x).detach() + torch.nn.functional.linear(
        torch.nn.functional.linear(x, target_a), target_b)
    optimizer = torch.optim.AdamW([layer.lora_A, layer.lora_B], lr=0.05)
    initial = torch.nn.functional.mse_loss(layer(x), target).item()
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(layer(x), target)
        loss.backward()
        optimizer.step()
    final = torch.nn.functional.mse_loss(layer(x), target).item()
    assert final < initial * 0.05, (initial, final)


def test_adapter_save_load_round_trip_stores_only_a_and_b(tmp_path):
    model = tiny_qwen(seed=3002)
    names = p30.install_lora(model, rank=4, alpha=8)
    assert len(names) == 2 * len(p30.LORA_MODULES)
    for module in p30.lora_modules(model).values():
        torch.nn.init.normal_(module.lora_B, std=0.02)
    ids = torch.tensor([[2, 3, 4, 5]])
    with torch.no_grad():
        expected = model(input_ids=ids, use_cache=False).logits
    path = tmp_path / 'adapter.pt'
    p30.save_adapter(model, str(path), 'r1', p30.BASE_MODEL, 'tiny-revision')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    assert set(payload['lora'][names[0]]) == {'A', 'B'}

    restored = tiny_qwen(seed=3002)
    loaded = p30.load_adapter(restored, str(path))
    assert set(loaded['modules']) == set(names)
    with torch.no_grad():
        actual = restored(input_ids=ids, use_cache=False).logits
    assert torch.equal(expected, actual)


def test_split_is_deterministic_and_has_expected_sizes():
    train_a, heldout_a = p30.split_indices(1000, seed=0, train_count=900)
    train_b, heldout_b = p30.split_indices(1000, seed=0, train_count=900)
    assert train_a == train_b
    assert heldout_a == heldout_b
    assert len(train_a) == 900 and len(heldout_a) == 100
    assert set(train_a).isdisjoint(heldout_a)
    assert set(train_a) | set(heldout_a) == set(range(1000))


def test_all_teacher_targets_use_the_same_token_cap_and_truncate():
    tok = TinyTokenizer()
    examples = [p30.make_supervised_example(tok, 'same question', target, 5)
                for target in ('R1 answer longer', 'Gemini answer much longer', 'brief')]
    assert [row['target_len'] for row in examples] == [5, 5, 5]
    p30.assert_shared_target_budget([1024, 1024, 1024], 1024)


def test_loss_labels_mask_prompt_and_padding_only():
    labels = p30.labels_for_example([10, 11, 12, 13, 14], prompt_len=3, padded_to=7)
    assert labels == [-100, -100, -100, 13, 14, -100, -100]


def test_answer_row_matches_part29_schema_and_extracts_after_last_think():
    row = p30.make_answer_row(247, 'local-model', 'local',
                              '<think>earlier</think>scratch<think>latest</think>final answer',
                              'stop', 12, 7, 0.25)
    assert set(row) == {'index', 'model', 'provider', 'text', 'answer',
                        'finish_reason', 'usage', 'wall_time'}
    assert row['answer'] == 'final answer'
    assert row['usage'] == {'input_tokens': 12, 'output_tokens': 7, 'total_tokens': 19}


def test_grouped_probe_is_chance_for_random_labels_and_separates_planted_feature():
    rng = np.random.default_rng(3003)
    question_count = 100
    labels = np.tile(np.asarray([1, 0], dtype=np.int64), question_count)
    groups = np.repeat(np.arange(question_count), 2)
    random_features = rng.normal(size=(2 * question_count, 24)).astype(np.float32)
    random_auc = p30.logistic_probe_auc(random_features, labels, groups)['mean_auroc']
    assert 0.35 <= random_auc <= 0.65, random_auc

    planted = rng.normal(size=(2 * question_count, 24)).astype(np.float32)
    planted[:, 0] = labels * 10.0
    planted_auc = p30.logistic_probe_auc(planted, labels, groups)['mean_auroc']
    assert planted_auc >= 0.99, planted_auc

    result = p30.logistic_probe_auc(planted, labels, groups)
    for fold in result['folds']:
        heldout = set(fold['question_groups'])
        for group in heldout:
            pair_positions = np.flatnonzero(groups == group)
            assert len(pair_positions) == 2


def test_sabotage_teacher_dependent_truncation_is_rejected():
    # This is the red condition for an implementation that silently grants teachers
    # different caps: the shared-budget assertion must reject [1024, 768].
    with pytest.raises(ValueError, match='teacher-dependent target budget'):
        p30.assert_shared_target_budget([1024, 768], 1024)


def test_supervised_example_uses_chat_template_and_caps_assistant_continuation():
    example = p30.make_supervised_example(TinyTokenizer(), 'why?', 'because clearly', max_target=4)
    assert example['input_ids'][:example['prompt_len']] == p30._encode_text(
        TinyTokenizer(), '<U>why?<A>')
    assert example['target_len'] == 4
    labels = p30.labels_for_example(example['input_ids'], example['prompt_len'])
    assert all(value == -100 for value in labels[:example['prompt_len']])
    assert all(value != -100 for value in labels[example['prompt_len']:])

