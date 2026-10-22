from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import fastgen
import part49
import part53


def test_part53_grid_has_registered_units_signs_and_zero_clamps():
    cells = part53.build_cell_plan()
    assert len(cells) == 30
    wizard = [cell for cell in cells if cell["prompt_set"] == "wizard"]
    everyday = [cell for cell in cells if cell["prompt_set"] == "everyday"]
    assert len(wizard) == 25
    assert {(cell["k_W"], cell["k_M"]) for cell in wizard} == {
        (w, m) for w in (-16, -8, 0, 8, 16) for m in (-32, -16, 0, 16, 32)
    }
    assert {(cell["k_W"], cell["k_M"]) for cell in everyday} == {
        (0, 0), (0, 32), (0, -32), (16, 0), (-16, 0),
    }
    zero = next(cell for cell in cells if cell["prompt_set"] == "wizard"
                and cell["k_W"] == 0 and cell["k_M"] == 0)
    mixed = next(cell for cell in cells if cell["prompt_set"] == "wizard"
                 and cell["k_W"] == -8 and cell["k_M"] == 16)
    assert zero["clamps"] == []
    assert [(row["role"], row["layer"], row["unit"], row["k"])
            for row in mixed["clamps"]] == [
        ("W", 19, 4826, -8), ("M", 22, 1610, 16),
    ]


def test_per_clamp_k_arithmetic_uses_each_natural_max():
    cells = part53.build_cell_plan()
    natural = {"values": {"19:4826": 4.1875, "22:1610": 5.3125}}
    ready = part53.add_per_clamp_natural_maxima(cells, natural)
    mixed = next(cell for cell in ready if cell["prompt_set"] == "wizard"
                 and cell["k_W"] == -16 and cell["k_M"] == 32)
    assert [(row["role"], row["natural_max"], row["level"])
            for row in mixed["clamps"]] == [
        ("W", 4.1875, -67.0), ("M", 5.3125, 170.0),
    ]
    zero = next(cell for cell in ready if cell["prompt_set"] == "wizard"
                and cell["k_W"] == cell["k_M"] == 0)
    assert zero["clamps"] == []
    assert part53.operations_for_cell(mixed) == {(19, 4826): -67.0, (22, 1610): 170.0}


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __call__(self, text, add_special_tokens=False):
        words = str(text).split()
        return {"input_ids": [list(range(1, len(words) + 1)) or [1]]}

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(token) for token in token_ids if token != self.eos_token_id)


class RandomDown(torch.nn.Module):
    def forward(self, values):
        return values


class FakeMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = RandomDown()


class FakeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = FakeMLP()


class TinyRandomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeLayer(), FakeLayer()])

    def forward(self, input_ids, attention_mask, use_cache=True, position_ids=None,
                cache_position=None, past_key_values=None):
        batch, sequence = input_ids.shape
        for layer in self.layers:
            values = torch.randn(batch, sequence, 4, dtype=torch.float32)
            layer.mlp.down_proj(values)
        logits = torch.zeros((batch, sequence, 4), dtype=torch.float32)
        token = 1 if past_key_values is None else 2
        logits[:, :, token] = 5.0
        cache = (torch.tensor(1),) if use_cache else None
        return SimpleNamespace(logits=logits, past_key_values=cache)


def test_activity_recorder_reads_random_model_decode_and_post_clamp_values():
    torch.manual_seed(17)
    model = TinyRandomModel()
    # The tiny model has two layers with width four; map the registered logical
    # roles onto their local test coordinates to exercise both hook paths.
    with part53.ActivityRecorder(model, 1, {"W": (0, 1), "M": (1, 2)}) as recorder:
        outputs = fastgen.batched_generate(
            model, FakeTokenizer(), ["small test prompt"], max_new_tokens=3,
            operations={(0, 1): 7.0, (1, 2): -8.0}, do_sample=False,
            device="cpu", generated_only=True,
        )
        measured = recorder.summaries(outputs)
    assert len(outputs[0]["token_ids"]) == 3
    assert measured[0]["token_count"] == 2
    assert measured[0]["generated_token_count"] == 3
    assert measured[0]["W"] == 7.0
    assert measured[0]["M"] == -8.0
    assert all(not layer.mlp.down_proj._forward_hooks for layer in model.layers)


def _hand_table() -> list[dict]:
    cells = part53.build_cell_plan()
    for cell in cells:
        cell["prompt_ids"] = ["p01"]
        cell["samples_per_prompt"] = 1
    header = {
        "record_type": "header", "command": "push53", "cells": cells,
        "prompt_ids": {"wizard": ["p01"], "everyday": ["p01"]},
        "samples_per_prompt": 1,
    }
    answers = {
        part53._cell_id("wizard", 0, 0): "magic",
        part53._cell_id("wizard", 8, 0): "magic ordinary",
        part53._cell_id("wizard", 0, 16): "magic ordinary ordinary ordinary",
        part53._cell_id("wizard", 8, 16): "ordinary",
    }
    rows = [header]
    for cell in cells:
        rows.append({
            "record_type": "result", "row_key": f"{cell['id']}:p01:1",
            "cell_id": cell["id"], "prompt_set": cell["prompt_set"],
            "k_W": cell["k_W"], "k_M": cell["k_M"],
            "prompt_id": "p01", "sample": 1, "prompt": "Write plainly.",
            "answer": answers.get(cell["id"], "ordinary"), "broken": None,
            "activity": {"W": 1.0, "M": 1.0, "token_count": 1},
        })
    return rows


def test_interaction_statistics_match_hand_calculated_density_table():
    scored = part53.score_push53_rows(_hand_table(), seed=31, resamples=100)
    target = next(row for row in scored["interactions"]
                  if row["cell"] == part53._cell_id("wizard", 8, 16))
    magic = target["measures"]["magic_per_1000"]
    assert magic["additive_contrast"] == pytest.approx(250.0)
    assert magic["multiplicative_ratio"] == pytest.approx(0.0)
    assert magic["additive_bootstrap95"] == pytest.approx([250.0, 250.0])
    assert magic["multiplicative_bootstrap95"] == pytest.approx([0.0, 0.0])
    assert len(scored["predictions"]) == 7


def test_scorer_refuses_a_missing_cell_answer():
    rows = _hand_table()
    missing_id = part53._cell_id("wizard", 16, 32)
    rows = [row for row in rows
            if row.get("record_type") != "result" or row.get("cell_id") != missing_id]
    with pytest.raises(ValueError, match="missing cell answers"):
        part53.score_push53_rows(rows, seed=31, resamples=20)


def test_sample_seeds_are_shared_across_cells_for_the_same_prompt_sample():
    prompts = ["one", "two"]
    samples = part49.build_sample_rows(prompts, 2, seed=3801, namespace="part53-wizard-sample")
    first, second = part53.build_cell_plan()[:2]
    first_rows = part53.requests_for_cell(first, samples)
    second_rows = part53.requests_for_cell(second, samples)
    assert [row["seed"] for row in first_rows] == [row["seed"] for row in second_rows]
    assert [row["row_key"] for row in first_rows] != [row["row_key"] for row in second_rows]
