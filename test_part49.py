from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import fastgen
import part38
import part49


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        words = str(text).split()
        return {"input_ids": [list(range(1, len(words) + 1)) or [1]]}

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(token) for token in token_ids if token != self.eos_token_id)


class RecordingDown(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, values):
        self.seen.append(values.detach().clone())
        return values


class FakeMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = RecordingDown()


class FakeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = FakeMLP()


class TinyModel(torch.nn.Module):
    def __init__(self, raise_on_forward=False):
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeLayer(), FakeLayer()])
        self.raise_on_forward = raise_on_forward

    def forward(self, input_ids, attention_mask, use_cache=True, position_ids=None,
                cache_position=None, past_key_values=None):
        if self.raise_on_forward:
            raise RuntimeError("fake generation failure")
        batch, sequence = input_ids.shape
        for layer in self.layers:
            layer.mlp.down_proj(torch.ones((batch, sequence, 4), dtype=torch.float32))
        logits = torch.zeros((batch, sequence, 3), dtype=torch.float32)
        token = 1 if past_key_values is None else 0
        logits[:, :, token] = 5.0
        cache = (torch.tensor(1),) if use_cache else None
        return SimpleNamespace(logits=logits, past_key_values=cache)


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_two_active_layer_hooks_clamp_only_own_coordinates_and_generated_tokens(padding_side):
    model = TinyModel()
    outputs = fastgen.batched_generate(
        model, FakeTokenizer(), ["short prompt", "this prompt has several more words"],
        max_new_tokens=2, operations={(0, 1): 7.0, (1, 2): 8.0},
        do_sample=False, device="cpu", generated_only=True, padding_side=padding_side,
    )
    assert len(outputs) == 2
    for layer_index, unit, value in ((0, 1, 7.0), (1, 2, 8.0)):
        seen = model.layers[layer_index].mlp.down_proj.seen
        assert len(seen) == 2  # prefill and one cached decode forward
        assert torch.equal(seen[0], torch.ones_like(seen[0]))  # prompt and padding untouched
        decoded = seen[1]
        assert torch.all(decoded[:, :, unit] == value)
        for other_unit in range(decoded.shape[-1]):
            if other_unit != unit:
                assert torch.all(decoded[:, :, other_unit] == 1.0)
    assert all(not layer.mlp.down_proj._forward_pre_hooks for layer in model.layers)


def test_two_units_on_the_same_layer_are_both_clamped_independently():
    model = TinyModel()
    fastgen.batched_generate(
        model, FakeTokenizer(), ["short", "a longer prompt"], max_new_tokens=2,
        operations={(0, 1): 7.0, (0, 2): 8.0}, do_sample=False,
        device="cpu", generated_only=True,
    )
    decoded = model.layers[0].mlp.down_proj.seen[1]
    assert torch.all(decoded[:, :, 1] == 7.0)
    assert torch.all(decoded[:, :, 2] == 8.0)
    assert torch.all(decoded[:, :, 0] == 1.0)
    assert torch.all(decoded[:, :, 3] == 1.0)


def test_negative_clamp_uses_minus_k_times_natural_max_on_generated_tokens_only():
    arm = {"id": "W", "clamps": [
        {"layer": 0, "unit": 1, "natural_max": 2.5},
    ]}
    operations = part49.operations_for_arm(arm, k=-8)
    assert operations == {(0, 1): -20.0}
    model = TinyModel()
    fastgen.batched_generate(
        model, FakeTokenizer(), ["short", "a longer prompt"], max_new_tokens=2,
        operations=operations, do_sample=False, device="cpu", generated_only=True,
    )
    down_proj = model.layers[0].mlp.down_proj
    assert torch.equal(down_proj.seen[0], torch.ones_like(down_proj.seen[0]))
    assert torch.all(down_proj.seen[1][:, :, 1] == -20.0)
    for unit in (0, 2, 3):
        assert torch.all(down_proj.seen[1][:, :, unit] == 1.0)


def test_k_none_hook_matches_no_hook_greedy_output():
    tokenizer = FakeTokenizer()
    plain = fastgen.batched_generate(
        TinyModel(), tokenizer, ["one prompt"], max_new_tokens=2,
        do_sample=False, device="cpu",
    )
    no_op_arm = {"id": "W", "clamps": [
        {"layer": 0, "unit": 1, "natural_max": 2.5},
    ]}
    no_op = fastgen.batched_generate(
        TinyModel(), tokenizer, ["one prompt"], max_new_tokens=2,
        operations=part49.operations_for_arm(no_op_arm, k=None),
        do_sample=False, device="cpu", generated_only=True,
    )
    assert no_op[0]["token_ids"] == plain[0]["token_ids"]


def test_hooks_are_removed_when_generation_raises():
    model = TinyModel(raise_on_forward=True)
    with pytest.raises(RuntimeError, match="fake generation failure"):
        fastgen.batched_generate(
            model, FakeTokenizer(), ["short", "a longer prompt"],
            operations={(0, 1): 7.0, (1, 2): 8.0}, do_sample=False, device="cpu",
        )
    assert all(not layer.mlp.down_proj._forward_pre_hooks for layer in model.layers)


def test_same_prompt_sample_seeds_are_reused_for_every_arm():
    partners = [{"id": f"P{i}", "layer": 4 + i, "unit": 20 + i} for i in range(1, 6)]
    random_pairs = [{"id": f"R{i}", "layer": 4 + i, "unit": 40 + i} for i in range(1, 4)]
    arms = part49.build_arm_plan(partners, random_pairs)
    samples = part49.build_sample_rows(["first", "second"], 2, seed=3801)
    requests = part49.arm_requests(arms, samples)
    expected_seeds = [row["seed"] for row in samples]
    assert len(arms) == 15
    assert all([row["seed"] for row in requests[arm["id"]]] == expected_seeds for arm in arms)
    assert all([row["row_key"] for row in requests[arm["id"]]] !=
               [row["row_key"] for row in requests[arms[0]["id"]]]
               for arm in arms[1:])


def test_scorer_matches_part38_topic_hit_and_reports_blinded_broken_rates():
    fixtures = [
        ("SPELL", True),
        ("misspell", False),
        ("A sorceress appeared.", True),
        ("The note said magic; then wand.", True),
        ("ordinary prose", False),
    ]
    for answer, expected in fixtures:
        assert part49.topic_hit(answer) is expected
        assert part49.topic_hit(answer) == part38.topic_hit(answer, part38.TOPIC_TERMS["wizard"])

    arms = [
        {"id": "no_clamp", "clamps": []},
        {"id": "W", "clamps": []},
        {"id": "P1", "clamps": []},
        {"id": "P1+W", "clamps": []},
    ]
    rows = [
        {"record_type": "result", "arm": "no_clamp", "answer": "ordinary prose", "broken": False},
        {"record_type": "result", "arm": "W", "answer": "SPELL", "broken": True},
        {"record_type": "result", "arm": "W", "answer": "misspell", "broken": False},
        {"record_type": "result", "arm": "P1", "answer": "ordinary prose", "broken": False},
        {"record_type": "result", "arm": "P1+W", "answer": "A sorceress appeared.", "broken": True},
    ]
    partners = [{"id": f"P{i}", "layer": 4 + i, "unit": i} for i in range(1, 6)]
    scored = part49.score_push_rows(rows, arms, partners)
    wizard = next(row for row in scored["arms"] if row["arm"] == "W")
    assert wizard["topic"] == {
        "hits": 1, "n": 2, "rate": 0.5,
        "wilson95": list(part38.wilson_interval(1, 2)),
    }
    assert wizard["broken"] == {"hits": 1, "n": 2, "rate": 0.5}
    assert scored["partner_boosts"][0]["delta_rate"] == 1.0
    assert all(row["broken_rate"] is None for row in scored["arms"]
               if row["arm"] in ("P1", "P1+W") and row["broken"]["n"] == 0)


def test_part49b_excludes_prompt_terms_from_answer_density():
    measured = part49.part49b_answer_measure(
        "Magic wand, spellbook and sorcery.", "Discuss magic with care."
    )
    assert measured == {
        "matches": 3, "word_count": 5, "per_1000": 600.0,
        "excluded_terms": ["magic"],
    }


def test_part49b_bootstrap_resamples_prompts_not_answers():
    arms = [
        {"id": "no_clamp", "clamps": []},
        {"id": "L22:1610", "clamps": []},
    ]
    rows = [
        {"record_type": "result", "arm": "no_clamp", "prompt_id": "p01",
         "prompt": "Write a note.", "answer": "magic", "broken": None},
        {"record_type": "result", "arm": "no_clamp", "prompt_id": "p02",
         "prompt": "Write a note.", "answer": "ordinary", "broken": None},
        {"record_type": "result", "arm": "L22:1610", "prompt_id": "p01",
         "prompt": "Write a note.", "answer": "magic", "broken": None},
        {"record_type": "result", "arm": "L22:1610", "prompt_id": "p01",
         "prompt": "Write a note.", "answer": "magic", "broken": None},
        {"record_type": "result", "arm": "L22:1610", "prompt_id": "p01",
         "prompt": "Write a note.", "answer": "magic", "broken": None},
        {"record_type": "result", "arm": "L22:1610", "prompt_id": "p02",
         "prompt": "Write a note.", "answer": "ordinary", "broken": None},
    ]

    scored = part49.score_push49b_rows(rows, arms)
    target = next(row for row in scored["arms"] if row["arm"] == "L22:1610")
    prompt_values = np.asarray([1000.0, 0.0])
    rng = np.random.default_rng(part49.PART49B_BOOTSTRAP_SEED)
    indexes = rng.integers(0, len(prompt_values),
                           size=(part49.PART49B_BOOTSTRAP_RESAMPLES, len(prompt_values)))
    expected_bootstrap = np.quantile(prompt_values[indexes].mean(axis=1), [0.025, 0.975])

    assert target["answer_count"] == 4
    assert target["mean_per_1000"] == 500.0  # equal prompt weighting, not 3:1 answer weighting
    assert target["bootstrap95"] == pytest.approx(expected_bootstrap.tolist())
    assert scored["measure"]["resampling_unit"] == "prompt"


def test_part49b_random_units_are_seeded_distinct_and_exclude_l22_1610():
    width = 5000
    expected = random.Random(4902).sample(
        [unit for unit in range(width) if unit != 1610], 3
    )
    chosen = part49.choose_part49b_random_units(width, seed=4902)

    assert part49.PART49B_RANDOM_SEED == 4902
    assert [row["unit"] for row in chosen] == expected
    assert part49.choose_part49b_random_units(width, seed=4902) == chosen
    assert len({row["unit"] for row in chosen}) == 3
    assert all(row["layer"] == 22 and row["unit"] != 1610 for row in chosen)
    arms = part49.build_part49b_arm_plan(chosen)
    assert [row["id"] for row in arms] == [
        "no_clamp", "L22:1610", "L19:4826", "both", "R1", "R2", "R3",
    ]
