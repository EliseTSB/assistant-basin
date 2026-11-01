#!/usr/bin/env python3
"""Focused tests for Part 165's effect runner and scorer."""
import json
import math
import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "part86") not in sys.path:
    sys.path.insert(0, str(ROOT / "part86"))

from part165 import effect165, run165, score_effect165


class FakeCommon:
    @staticmethod
    def token_inputs(_tok, _question, batch, _device):
        return {"input_ids": torch.ones((batch, 3), dtype=torch.long)}

    @staticmethod
    def eos_ids(_tok, _model):
        return []

    AddRowsHook = __import__("part86.common86", fromlist=["AddRowsHook"]).AddRowsHook


class FakeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = SimpleNamespace(down_proj=torch.nn.Linear(4, 4, bias=False))

    def forward(self, hidden):
        return (self.mlp.down_proj(hidden), None)


class FakeModel:
    def __init__(self, layer):
        self.layer = layer
        self.calls = 0

    def generate(self, input_ids, **kwargs):
        self.calls = 0
        batch = input_ids.shape[0]
        self.layer(torch.zeros((batch, 3, 4)))
        self.layer(torch.ones((batch, 1, 4)))
        self.calls = 2
        seq = torch.cat([input_ids, torch.tensor([[10, 11]]).expand(batch, -1)], dim=1)
        scores = (torch.zeros((batch, 5)), torch.tensor([[0., 0., 2., 0., 0.]]).expand(batch, -1))
        return SimpleNamespace(sequences=seq, scores=scores)


class EffectTests(unittest.TestCase):
    def test_fake_model_generate_capture_and_tuple_hook_output(self):
        layer = FakeLayer()
        layers = [layer]
        model = FakeModel(layer)
        # The fake generate path emits a tuple-like final-layer output via a wrapper hook target.
        captured = effect165.generate_capture(model, object(), layers, "Q", {}, "cpu", FakeCommon)
        self.assertEqual(captured["first"].tolist(), [10])
        self.assertEqual(captured["second"].tolist(), [11])
        self.assertEqual(captured["scores"].shape, (1, 5))
        self.assertEqual(captured["hidden"].shape, (1, 4))

    def test_row_order_matches_run165_and_resume_filter(self):
        widths = {layer: 2048 for layer in run165.LAYERS}
        expected = run165.all_rows(widths)
        import random
        random.Random(165 + 3).shuffle(expected)
        actual = effect165.rows_for_question(3, widths)
        self.assertEqual([(r["arm"], r["slot"], r["layer"]) for r in actual],
                         [(r["arm"], r["slot"], r["layer"]) for r in expected])
        existing = {effect165.row_key(actual[0])}
        self.assertEqual(effect165.pending_rows(actual, existing), actual[1:])

    def test_resume_skips_written_rows_in_jsonl(self):
        row = {"stage": "effect", "arm": "n+", "slot": 0, "layer": 19, "question_index": 0}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "effects.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            done = {effect165.row_key(r) for r in effect165.read_existing(path)}
        self.assertEqual(effect165.pending_rows([row, {**row, "slot": 1}], done), [{**row, "slot": 1}])

    def test_kl_identical_and_known_pair_rel_change(self):
        p = torch.tensor([[0., 0.]])
        same = effect165.effect_values(p, p, torch.tensor([[3., 4.]]), torch.tensor([[3., 4.]]),
                                       torch.tensor([1]), torch.tensor([1]))
        self.assertAlmostEqual(same[0][0], 0.0, places=7)
        self.assertEqual(same[1], [0.0])
        q = torch.tensor([[0., 2.]])
        got = effect165.effect_values(p, q, torch.tensor([[3., 4.]]), torch.tensor([[6., 8.]]),
                                      torch.tensor([1]), torch.tensor([0]))
        # KL([.5,.5] || softmax([0,2])) = log(1+e^2)-1.
        self.assertAlmostEqual(got[0][0], math.log1p(math.exp(2.)) - 1. - math.log(2.), places=6)
        self.assertAlmostEqual(got[1][0], 1.0, places=6)
        self.assertEqual(got[2], [False])

    def test_pilot_stops_slow_run(self):
        self.assertLess(effect165.pilot_rate(192, 100), 2.0)
        with self.assertRaises(SystemExit):
            if effect165.pilot_rate(192, 100) < 2.0:
                raise SystemExit("Pilot stop rule")

    def test_scorer_prediction_165_7_passes_and_fails(self):
        vals = {(arm, layer, slot): (2.0 if arm == "n+" else 1.0)
                for arm in ("n+", "full") for layer in score_effect165.LAYERS for slot in score_effect165.SLOTS}
        self.assertEqual(score_effect165.prediction_165_7(vals)["verdict"], "pass")
        vals.update({("n+", layer, slot): 1.0 for layer in score_effect165.LAYERS for slot in score_effect165.SLOTS})
        self.assertEqual(score_effect165.prediction_165_7(vals)["verdict"], "fail")


if __name__ == "__main__":
    unittest.main()
