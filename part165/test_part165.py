"""Small, model-free checks for Part 165's registered mechanics."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "part86"))

import run165
import score165


def fake_weight(width=1028, hidden=24):
    generator = torch.Generator(device="cpu").manual_seed(75)
    return torch.randn((hidden, width), generator=generator, dtype=torch.float32)


class FakeCommon:
    def __init__(self, layers):
        self.layers = layers
        self.REVISION = "test"

    def find_layers(self, _model):
        return self.layers


def layer_map(weight):
    layer = types.SimpleNamespace(mlp=types.SimpleNamespace(
        down_proj=types.SimpleNamespace(weight=weight)))
    return [layer for _ in range(20)]


def seed_rows(broken=None, changed=None, no_push_answer="I am a test assistant."):
    """Construct scorer rows; broken/changed maps choose answer counts by arm."""
    broken = broken or {}
    changed = changed or {}
    rows = []
    for qi in range(8):
        rows.append({"stage": "no_push", "arm": "plain", "slot": None, "layer": None,
                     "k": None, "question_index": qi, "question": f"Q{qi}",
                     "kind": "maker", "answer": no_push_answer, "token_count": 5})
    for arm in score165.ARMS:
        for slot in score165.SLOTS:
            for qi in range(8):
                is_broken = qi < broken.get(arm, 0)
                is_changed = qi < changed.get(arm, 0)
                answer = "bad" if is_broken else ("This is another safe assistant." if is_changed
                                                   else no_push_answer)
                rows.append({"stage": "battery", "arm": arm, "slot": slot, "layer": 19,
                             "k": 1028 if arm == "full" else None, "question_index": qi,
                             "question": f"Q{qi}", "kind": "maker", "answer": answer,
                             "token_count": len(answer.split())})
    return rows


def good_norms():
    return {"directions": [{"layer": 19, "slot": slot, "arm": arm,
                             "relative_error": 0.0,
                             "iso_residual_fraction": 0.0 if arm == "iso" else None}
                            for slot in score165.SLOTS for arm in score165.ARMS]}


class Part165Tests(unittest.TestCase):
    def test_directions_deterministic_nested_and_scaled(self):
        weight = fake_weight()
        cache = {}
        a, metrics = run165.build_directions(weight, 19, 3, 2.0, cache)
        b, _ = run165.build_directions(weight, 19, 3, 2.0, {})
        self.assertTrue(all(torch.equal(a[arm], b[arm]) for arm in run165.ARMS))
        self.assertLess(metrics["iso"]["iso_residual_fraction"], .01)
        self.assertGreater(float(a["n+"][torch.argmax(a["n+"])].item()), 0.0)
        nonzero = lambda x: int(torch.count_nonzero(x).item())
        for arm, count in run165.K_ARMS.items():
            self.assertEqual(nonzero(a[arm]), count)
        for arm in run165.ARMS:
            self.assertAlmostEqual(metrics[arm]["write_norm"], 1.6, delta=1.6e-5)
        selected = int(torch.argmax(a["n+"]).item())
        self.assertEqual(float(a["n±"][selected] > 0),
                         float(run165._seeded_raw(weight, 19, 3, torch)[1][selected] > 0))
        for small, large in (("k4", "k16"), ("k16", "k64"), ("k64", "k256"),
                             ("k256", "k1024"), ("k1024", "full")):
            self.assertTrue(set(torch.nonzero(a[small]).flatten().tolist()).issubset(
                set(torch.nonzero(a[large]).flatten().tolist())))

    def test_shuffling_mixes_arms_and_is_stable(self):
        rows = run165.all_rows({19: 2048, 20: 2048, 21: 2048, 22: 2048,
                                23: 2048, 24: 2048, 25: 2048, 26: 2048})
        def shuffled(qi):
            result = list(rows)
            import random
            random.Random(165 + qi).shuffle(result)
            return result
        first, again = shuffled(0), shuffled(0)
        self.assertEqual(first, again)
        self.assertGreater(len({row["arm"] for row in first[:64]}), 1)

    def _run_fixture(self, output, clock=None, on_generate=None):
        temp_root = Path(output).parent / "repo"
        (temp_root / "results/part112").mkdir(parents=True, exist_ok=True)
        (temp_root / "results/part112/medians.json").write_text(json.dumps({"19": 2.0}), encoding="utf-8")
        weight = fake_weight()
        common = FakeCommon(layer_map(weight))
        questions = [{"question": f"Q{qi}", "kind": "maker"} for qi in range(8)]
        calls = []
        def generate(_model, _tok, _layers, question, pushes, _tokens, _device):
            calls.append((question, pushes))
            if on_generate:
                on_generate(question, pushes)
            return [{"answer": "I am a test assistant.", "token_count": 5} for _ in range(
                next(iter(pushes.values())).shape[0] if pushes else 1)]
        args = types.SimpleNamespace(out=str(output), norms=str(Path(output).parent / "norms.json"),
                                     model="fake", device="cpu", batch_size=64)
        with patch.object(run165, "ROOT", temp_root), patch.object(run165, "LAYERS", (19,)), \
             patch.object(run165, "SLOTS", 1), patch.object(run165, "questions", return_value=questions), \
             patch.object(run165.part155, "_common", return_value=common):
            if clock is not None:
                with patch.object(run165.time, "perf_counter", side_effect=clock):
                    run165.run(args, loader=lambda _device: (object(), object()), generate_fn=generate)
            else:
                run165.run(args, loader=lambda _device: (object(), object()), generate_fn=generate)
        return calls

    def test_resume_skips_written_rows(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "answers.jsonl"
            calls = self._run_fixture(output)
            first_count = len(calls)
            second = self._run_fixture(output)
            self.assertEqual(first_count, 16)  # eight no-pushes plus eight battery batches
            self.assertEqual(second, [])

    def test_pilot_clock_starts_after_no_push_and_slow_run_stops(self):
        with tempfile.TemporaryDirectory() as td:
            ticks = [0.0]
            def now():
                return ticks[0]
            def on_generate(_question, pushes):
                ticks[0] += 5.0 if not pushes else 7.0
            output = Path(td) / "answers.jsonl"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                with self.assertRaises(SystemExit):
                    self._run_fixture(output, clock=now, on_generate=on_generate)
            self.assertIn("first 3 battery batches: 1.29 answers/s", buf.getvalue())

    def test_kept_loop_is_overridden_by_primary_rule(self):
        answer = "I am Qwen. red blue green red blue green red blue green"
        self.assertEqual(score165.triage86.triage_answer(answer), "kept")
        self.assertTrue(score165.primary_broken(answer))

    def test_all_gates_pass_and_each_can_fail(self):
        with patch.object(score165, "LAYERS", (19,)), patch.object(score165, "SLOTS", tuple(range(12))), \
             patch.object(score165, "EXPECTED_ROWS", 8 + 12 * 9 * 8):
            rows = seed_rows(broken={"n+": 1})
            good = score165.score_rows(rows, good_norms())
            self.assertTrue(all(x["passed"] for x in good["gates"].values()), good["gates"])
            bad_rows = [dict(row) for row in rows]
            bad_rows.append(dict(bad_rows[0]))
            self.assertFalse(score165.score_rows(bad_rows, good_norms())["gates"]["gate1_rows"]["passed"])
            baseline_bad = [dict(row) for row in rows]
            baseline_bad[0]["answer"] = "No"
            self.assertFalse(score165.score_rows(baseline_bad, good_norms())["gates"]["gate2_no_push"]["passed"])
            # Deliberate break: corrupt one measured write norm. This must make the size gate red.
            broken_norms = good_norms()
            broken_norms["directions"][0]["relative_error"] = .011
            gate3 = score165.score_rows(rows, broken_norms)["gates"]["gate3_size"]
            self.assertFalse(gate3["passed"], "deliberate 1.1% norm error must fail Gate 3")
            no_room = [dict(row) for row in rows]
            for row in no_room:
                if row.get("arm") == "n+":
                    row["answer"] = "safe and unchanged assistant"
            self.assertFalse(score165.score_rows(no_room, good_norms())["gates"]["gate4_measure_room"]["passed"])

    def test_prediction_predicates_pass_and_fail(self):
        # 165.1 and 165.5 point-estimate thresholds.
        self.assertTrue(score165.ratio_at_least(2.0, 2.0))
        self.assertFalse(score165.ratio_at_least(1.99, 2.0))
        self.assertTrue(score165.spearman_at_most(-.81, -.8))
        self.assertFalse(score165.spearman_at_most(-.79, -.8))
        # 165.3 must work in both directions.
        self.assertTrue(score165.k16_nearer(.2, .1, .6))
        self.assertFalse(score165.k16_nearer(.2, .4, .1))
        # 165.4 fails below and above its registered band.
        self.assertFalse(score165.ratio_between(.66, .67, 1.5))
        self.assertTrue(score165.ratio_between(1.0, .67, 1.5))
        self.assertFalse(score165.ratio_between(1.51, .67, 1.5))
        self.assertTrue(score165.ratio_less_than(.79, .8))
        self.assertFalse(score165.ratio_less_than(.8, .8))
        self.assertTrue(score165.spearman_at_most(-.61, -.6))
        self.assertFalse(score165.spearman_at_most(-.59, -.6))
        self.assertEqual(score165.rank_average([1, 1, 3]), [1.5, 1.5, 3.0])
        self.assertAlmostEqual(score165.spearman([1, 1, 3], [1, 2, 3]), .8660254037844387)


if __name__ == "__main__":
    unittest.main()
