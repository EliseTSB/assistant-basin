import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
import torch
from torch import nn

import part50


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = types.SimpleNamespace(o_proj=nn.Identity())

    def forward(self, x):
        return self.self_attn.o_proj(x)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = types.SimpleNamespace(layers=nn.ModuleList([FakeLayer()]))
        self.config = types.SimpleNamespace(num_attention_heads=2, hidden_size=6, head_dim=3)
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


class Part50Tests(unittest.TestCase):
    def test_head_output_hook_adds_exact_direction_only_while_active(self):
        model = FakeModel()
        hooks = part50.HeadOutputSteering(model)
        x = torch.arange(24, dtype=torch.float32).reshape(2, 2, 6) / 7
        direction = torch.tensor([[[0.25, -0.5, 1.0], [2.0, 0.125, -0.25]]])
        configured = direction.unsqueeze(0)
        if os.environ.get("PART50_TEST_SABOTAGE") == "hook":
            configured = torch.zeros_like(configured)
        hooks.set_batch(configured, torch.tensor([0, 0]), torch.tensor([2.5, 2.5]),
                        torch.ones(2, 2), active=True)
        steered = model(x.clone())
        expected = x + (direction.reshape(1, 1, 6) * 2.5)
        self.assertTrue(torch.allclose(steered, expected.expand_as(x), atol=1e-6, rtol=0))
        hooks.disable()
        self.assertTrue(torch.equal(model(x.clone()), x))
        hooks.close()

    def test_random_directions_keep_each_head_norm(self):
        direction = torch.randn(28, 16, 128, generator=torch.Generator().manual_seed(7))
        direction[0, 0].zero_()
        random = part50.make_random_directions(direction, count=39, seed=123)
        if os.environ.get("PART50_TEST_SABOTAGE") == "random":
            random = random * 2
        expected = direction.norm(dim=-1).expand_as(random[..., 0])
        actual = random.norm(dim=-1)
        self.assertEqual(tuple(random.shape), (39, 28, 16, 128))
        self.assertTrue(torch.allclose(actual, expected, atol=2e-6, rtol=1e-6))
        self.assertTrue(torch.equal(random, part50.make_random_directions(direction, count=39, seed=123)))

    def test_roleplay_scores_the_first_distinguishing_completion_token(self):
        prompt = [10, 11, 12]
        scored, truth, lie, shared = part50.roleplay_answer_position(
            prompt, [30, 31, 32, 33], [30, 31, 40, 41], eos_id=2
        )
        self.assertEqual(scored, [10, 11, 12, 30, 31])
        self.assertEqual((truth, lie, shared), ([32], [40], 2))

    def test_persona_target_comes_from_match_label(self):
        yes_ids, no_ids = [101, 102], [201, 202]
        self.assertEqual(part50.make_persona_choice({"match": "Yes"}, yes_ids, no_ids), (yes_ids, no_ids))
        self.assertEqual(part50.make_persona_choice({"match": "No"}, yes_ids, no_ids), (no_ids, yes_ids))

    def test_roleplay_answer_polarity_and_flip_summary(self):
        self.assertEqual(part50.answer_polarity('"Yes, I did."'), "yes")
        self.assertEqual(part50.answer_polarity("No, I did not."), "no")
        self.assertEqual(part50.answer_polarity("I cannot answer."), "other")
        run = {"arm_names": ["unsteered", "honesty", "anti-honesty", "random1"],
               "test_roleplay_items": [
            {"id": "a", "lie_polarity": "no", "m": [-1.0, 1.0, -2.0, 1.0]},
            {"id": "b", "lie_polarity": "yes", "m": [-2.0, -1.0, 1.0, 0.0]},
            {"id": "c", "lie_polarity": "other", "m": [1.0, 2.0, -1.0, 1.0]},
        ]}
        result = part50.summarize_lies(run)
        self.assertEqual(result["honesty_flips"], 1)
        self.assertAlmostEqual(result["honesty_flip_rate"], 1 / 3)
        self.assertAlmostEqual(result["conditional_flip_rate"], 1 / 2)
        self.assertEqual(result["random_flip_counts"], [1])
        self.assertAlmostEqual(result["polarity_split"]["honesty_component_bits"], 1.5)
        self.assertAlmostEqual(result["polarity_split"]["yes_push_component_bits"], 0.5)

    def test_persona_summary_scores_135_traits_and_frozen_bundle(self):
        items = part50.load_items(part50.ITEMS_PERSONA)
        rng = np.random.default_rng(50)
        rows = []
        for item in items:
            baseline = float(rng.normal())
            deltas = rng.normal(0, 0.1, size=41)
            deltas[0] += 0.15 * baseline
            values = [baseline, *(baseline + deltas)]
            rows.append({"id": item["id"], "trait": item["family"], "match": item["match"], "m": values})
        run = {"arm_names": ["unsteered", "honesty", "anti-honesty"] + [f"random{i}" for i in range(1, 40)],
               "persona_items": rows, "complete_persona": True}
        result = part50.summarize_persona(run, part50.read_bundle())
        self.assertEqual(result["n_traits_tested"], 135)
        self.assertEqual(len(result["bundle_score"]["random_scores_bits"]), 39)
        self.assertEqual(len(result["trait_matching_effects"]), 135)
        self.assertIn("confidence_removed_n", result["fdr"])

    def test_bundle_is_read_only_input(self):
        with tempfile.TemporaryDirectory(prefix="part50-bundle-") as temp:
            path = Path(temp) / "bundle.json"
            record = {
                "schema_version": 1, "model": "Qwen/Qwen3.5-9B", "n_traits": 51,
                "source_file": "source.json", "source_sha256": "a" * 64,
                "traits": [{"trait": f"trait-{i}", "sign": 1 if i % 2 else -1} for i in range(51)],
            }
            path.write_text(json.dumps(record), encoding="utf-8")
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded = part50.read_bundle(path)
            if os.environ.get("PART50_TEST_SABOTAGE") == "bundle":
                path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            after = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(len(loaded["traits"]), 51)
            self.assertEqual(after, before)

    def test_permutation_fdr_matches_vendored_implementation(self):
        rng = np.random.default_rng(50)
        fixed = rng.normal(size=(135, 40))
        fixed[:51, 0] += np.linspace(-2.0, 2.0, 51)
        expected = part50._vendor_persona_stats().perm_fdr(fixed)
        observed = part50.perm_fdr(fixed)
        if os.environ.get("PART50_TEST_SABOTAGE") == "fdr":
            observed = np.zeros_like(observed)
        self.assertTrue(np.array_equal(observed, expected))

    def test_canonical_frozen_bundle_and_provenance(self):
        bundle = part50.read_bundle(part50.BUNDLE_PATH)
        self.assertEqual(bundle["n_traits"], 51)
        self.assertEqual(bundle["source_sha256"], part50.sha256_file(part50.SOURCE_STATS))
        self.assertEqual(bundle["source_file"], "third_party/honesty-heads/results/persona_stats.json")


if __name__ == "__main__":
    unittest.main(verbosity=2)
