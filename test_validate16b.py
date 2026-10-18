"""Model-free tests for Part 16b validation and its blind judge."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import causal_search  # noqa: E402
import judge16b  # noqa: E402
import validate16b  # noqa: E402


class _TinyMLP(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.down_proj = nn.Identity()
        self.down_proj.in_features = width


class _TinyLayer(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.mlp = _TinyMLP(width)


class _TinyTrunk(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.layers = nn.ModuleList([_TinyLayer(width)])

    def forward(self, input_ids, **_kwargs):
        activation = input_ids.float().unsqueeze(-1).expand(-1, -1, self.layers[0].mlp.down_proj.in_features)
        hidden = self.layers[0].mlp.down_proj(activation)
        return SimpleNamespace(last_hidden_state=hidden)


class _TinyModel(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.model = _TinyTrunk(width)


class ConditionTests(unittest.TestCase):
    def setUp(self):
        self.names = [
            "L0.1", "L0.5", "L2.2", "L3.7", "L4.1", "L9.3", "L10.4", "L12.8", "L15.9"
        ]

    def test_early_and_late_conditions_split_by_layer(self):
        conditions = validate16b.build_conditions(self.names, 16, 20, seed=7)
        self.assertEqual(conditions["early_zero"]["selection"], {0: [1, 5], 2: [2], 3: [7]})
        self.assertEqual(conditions["late_zero"]["selection"], {10: [4], 12: [8], 15: [9]})
        self.assertEqual(conditions["full64_zero"]["selection"], validate16b.names_to_layers(self.names))

    def test_random_sets_match_per_layer_exclude_targets_and_are_seeded(self):
        first = validate16b.build_conditions(self.names, 16, 20, seed=91)
        second = validate16b.build_conditions(self.names, 16, 20, seed=91)
        targets = validate16b.names_to_layers(self.names)
        late_targets = first["late_zero"]["selection"]
        for index in range(1, 7):
            one = first[f"random_{index}"]["selection"]
            two = second[f"random_{index}"]["selection"]
            self.assertEqual(one, two)
            self.assertEqual({layer: len(values) for layer, values in one.items()},
                             {layer: len(values) for layer, values in targets.items()})
            for layer, values in one.items():
                self.assertFalse(set(values) & set(targets[layer]))
        for index in range(1, 4):
            selected = first[f"late_random_{index}"]["selection"]
            self.assertEqual({layer: len(values) for layer, values in selected.items()},
                             {layer: len(values) for layer, values in late_targets.items()})
            for layer, values in selected.items():
                self.assertFalse(set(values) & set(late_targets[layer]))


class HookTests(unittest.TestCase):
    def test_mean_hook_holds_value_at_each_position_and_preserves_batch_shape(self):
        values = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]])
        means = {0: torch.tensor([0.25, 3.5, -1.0, 7.0])}
        hook = validate16b.make_ablation_hook(0, {0: [1, 3]}, "mean", means)
        output = hook(nn.Identity(), (values,))[0]
        self.assertEqual(tuple(output.shape), (1, 3, 4))
        self.assertTrue(torch.equal(output[0, :, 1], torch.full((3,), 3.5)))
        self.assertTrue(torch.equal(output[0, :, 3], torch.full((3,), 7.0)))
        self.assertTrue(torch.equal(output[0, :, [0, 2]], values[0, :, [0, 2]]))

    def test_neutral_mean_capture_handles_batched_multitoken_activations(self):
        model = _TinyModel()
        means = validate16b.capture_neutral_means(model, [1, 2, 3], torch.device("cpu"))
        self.assertEqual(tuple(means[0].shape), (4,))
        self.assertTrue(torch.equal(means[0], torch.full((4,), 2.0)))


class SeedTests(unittest.TestCase):
    def test_sampled_reply_seeds_are_identical_across_conditions(self):
        conditions = ["unchanged", "full64_zero", "full64_mean", "random_1"]
        plan = validate16b.build_seed_plan(conditions, 3, sampled_count=4, base_seed=121)
        for condition in conditions[1:]:
            self.assertEqual(plan[condition], plan[conditions[0]])
        self.assertEqual(plan["unchanged"][2][4], validate16b.sample_seed(2, 4, 121))


class BlindJudgeTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            "conditions": {
                "unchanged": {
                    "mode": "none", "neurons": [], "neutral_loss": 1.0,
                    "replies": [
                        {"question": "Q1", "sample_index": 0, "text": "I am an AI assistant."},
                        {"question": "Q2", "sample_index": 1, "text": "I donates not have feelings."},
                    ],
                },
                "full64_zero": {
                    "mode": "zero", "neurons": ["L0.1"], "neutral_loss": 1.2,
                    "replies": [
                        {"question": "Q3", "sample_index": 0, "text": "I grew up in Denver."},
                        {"question": "Q4", "sample_index": 1, "text": "I can help with that."},
                    ],
                },
            }
        }

    def test_shuffle_hides_conditions_and_maps_stub_labels_back(self):
        seen = []

        def stub_judge(blind):
            seen.append(blind)
            self.assertNotIn("condition", blind)
            self.assertNotIn("unchanged", repr(blind))
            text = blind["text"]
            if "AI" in text:
                return {"identity": "ai", "fluent": "yes"}
            if "grew up" in text:
                return {"identity": "person", "fluent": "yes"}
            if "donates" in text:
                return {"identity": "person", "fluent": "no"}
            return {"identity": "neither", "fluent": "yes"}

        result = judge16b.judge_replies(self.data, stub_judge, seed=8, workers=1)
        self.assertEqual(len(seen), 4)
        self.assertEqual(
            [item["blind_id"] for item in seen],
            [item["blind"]["blind_id"] for item in judge16b.blind_replies(self.data, seed=8)],
        )
        self.assertNotEqual(
            [item["text"] for item in seen],
            [reply["text"] for record in self.data["conditions"].values() for reply in record["replies"]],
        )
        replies = {name: {reply["text"]: reply for reply in record["replies"]}
                   for name, record in result["conditions"].items()}
        self.assertEqual(replies["unchanged"]["I am an AI assistant."]["identity"], "ai")
        self.assertEqual(replies["unchanged"]["I donates not have feelings."]["fluent"], "no")
        self.assertEqual(replies["full64_zero"]["I grew up in Denver."]["identity"], "person")
        self.assertEqual(replies["full64_zero"]["I can help with that."]["identity"], "neither")

    def test_wilson_share_uses_causal_search_interval_for_fixed_counts(self):
        replies = [
            {"identity": "ai", "fluent": "yes"},
            {"identity": "ai", "fluent": "no"},
            {"identity": "person", "fluent": "yes"},
            {"identity": "neither", "fluent": "yes"},
            {"identity": "ai", "fluent": "yes"},
            {"identity": "neither", "fluent": "no"},
            {"identity": "person", "fluent": "yes"},
            {"identity": "neither", "fluent": "yes"},
        ]
        summary = judge16b.summarize_replies(replies)
        expected_interval = causal_search.wilson_interval(3, 8)
        self.assertEqual(summary["counts"]["identity"], {"ai": 3, "person": 2, "neither": 3})
        self.assertEqual(summary["shares"]["ai_share"]["rate"], 3 / 8)
        self.assertAlmostEqual(summary["shares"]["ai_share"]["wilson_95"][0], expected_interval[0])
        self.assertAlmostEqual(summary["shares"]["ai_share"]["wilson_95"][1], expected_interval[1])
        self.assertEqual(summary["shares"]["fluent_share"]["count"], 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
