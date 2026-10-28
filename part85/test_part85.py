#!/usr/bin/env python3
"""CPU-only, network-free checks for Part 85's hook, range reader, and scoring rules."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import Olmo2Config, Olmo2ForCausalLM

import part85
import score85
from checkpoints import CHECKPOINTS, resolve_checkpoint
from step0_weights import SafetensorsRangeReader


def tiny_model():
    torch.manual_seed(85)
    config = Olmo2Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=True,
    )
    return Olmo2ForCausalLM(config).cpu().eval()


def make_counts(own_start=8, absolute_start=11, own_break=None, invalid_gate=None):
    counts = {}
    order = [cp.key for cp in CHECKPOINTS]
    for i, key in enumerate(order):
        control = {"n": 48, "OFFWORLD": 0, "EARTH": 0, "NEITHER": 48, "OK": 48, "BROKEN": 0}
        baseline = {"n": 48, "OFFWORLD": 0, "EARTH": 0, "NEITHER": 48, "OK": 48, "BROKEN": 0}
        own = 12 if i >= own_start and (own_break is None or key != own_break) else 0
        absolute = 12 if i >= absolute_start else 0
        counts[f"{key}/plain/control"] = control
        counts[f"{key}/plain/unclamped"] = baseline
        counts[f"{key}/plain/own_max"] = {**baseline, "OFFWORLD": own, "NEITHER": 48 - own}
        counts[f"{key}/plain/absolute"] = {**baseline, "OFFWORLD": absolute, "NEITHER": 48 - absolute}
    step0, step1 = {}, {}
    for key in order:
        c = 0.79 if key == invalid_gate else 1.0
        step0[key] = {"measures": {"cosines_with_instruct": {"down_proj_column_6978": c},
                                    "direct_readout": {"z_score": 2.5 if key == "pretrain_4T" else 0.0}}}
        step1[key] = {"neutral_correlations_with_instruct": {"target": 1.0},
                      "passages": {"offworld_mean": 2.0, "ordinary_mean": 0.0, "ordinary_sd": 1.0}}
    return counts, step0, step1


class Part85Tests(unittest.TestCase):
    def test_checkpoint_resolution_pins_and_checks_prefix(self):
        class FakeInfo:
            def __init__(self, sha):
                self.sha = sha

        class FakeApi:
            def __init__(self, sha):
                self.sha = sha
                self.calls = []

            def model_info(self, repo_id, revision):
                self.calls.append((repo_id, revision))
                return FakeInfo(self.sha)

        cp = CHECKPOINTS[0]
        api = FakeApi(cp.commit_prefix + "0" * (40 - len(cp.commit_prefix)))
        pinned = resolve_checkpoint(cp, api=api)
        self.assertEqual(len(pinned["commit"]), 40)
        self.assertEqual(api.calls, [(cp.repo_id, cp.branch)])
        with self.assertRaisesRegex(RuntimeError, "does not start with design prefix"):
            resolve_checkpoint(cp, api=FakeApi("f" * 40))

    def test_hook_sets_prompt_and_generated_positions(self):
        model = tiny_model()
        down = model.model.layers[0].mlp.down_proj
        unit, value = 3, 7.25
        hook = part85.BothPhaseClampHook(down, unit, value)
        observed = []
        recorder = down.register_forward_pre_hook(
            lambda _m, args: observed.append(args[0][..., unit].detach().float().clone())
        )
        try:
            hook.begin_sequence()
            with torch.inference_mode():
                model.generate(torch.tensor([[4, 5, 6]]), max_new_tokens=3, do_sample=False,
                               eos_token_id=63, pad_token_id=0, use_cache=True)
        finally:
            recorder.remove()
            counts = dict(hook.counts)
            hook.remove()
        self.assertGreater(counts["prompt_positions"], 0)
        self.assertGreater(counts["generated_positions"], 0)
        self.assertGreaterEqual(len(observed), 2)
        self.assertTrue(all(torch.allclose(x, torch.full_like(x, value)) for x in observed))

    def test_paired_seeds_and_registered_counts(self):
        self.assertEqual(part85.seed_for(0, 0), 85000)
        self.assertEqual(part85.seed_for(1, 4), 85104)
        self.assertEqual(part85.seed_for(2, 15), 85215)
        counts = part85.expected_answer_counts()
        self.assertEqual(counts, {"pilot": 96, "main_plain": 2304, "main_chat": 576, "main_total": 2880})
        cells = part85.expected_cell_specs("pilot")
        self.assertEqual(len(cells), 4)
        self.assertEqual(sum(len(part85.QUESTIONS) * x["samples_per_question"] for x in cells), 96)
        self.assertTrue(all(x["samples_per_question"] == 8 for x in cells))
        main = part85.expected_cell_specs("main")
        self.assertEqual(sum(1 for x in main if x["frame"] == "plain"), 48)
        self.assertEqual(sum(1 for x in main if x["frame"] == "chat"), 12)

    def test_cell_values_use_registered_maxima(self):
        step1 = {"neutral_activation_maxima": {"6978": 6.5, "7884": 2.0}}
        self.assertEqual(part85._cell_value("unclamped", step1), (None, None))
        self.assertEqual(part85._cell_value("own_max", step1), (6978, 9.75))
        self.assertEqual(part85._cell_value("absolute", step1), (6978, 9.75))
        self.assertEqual(part85._cell_value("control", step1), (7884, 3.0))
        with self.assertRaisesRegex(ValueError, "missing target unit"):
            part85._cell_value("own_max", {"neutral_activation_maxima": {"7884": 2.0}})

    def test_pilot_gate_uses_responsive_ok_and_effect_thresholds(self):
        counts = {
            "pretrain_4T/plain/unclamped": {"n": 24, "OK": 20, "BROKEN": 4},
            "instruct/plain/own_max": {"n": 24, "OFFWORLD": 5},
        }
        self.assertEqual(score85.pilot_gate(counts, 16)["action"], "plain_frame_pass")
        self.assertEqual(score85.pilot_gate(counts, 15)["action"], "amend_frame_and_rerun_once")
        self.assertEqual(score85.pilot_gate(counts, 15, attempt=2)["action"], "quality_failed_after_rerun")
        counts["instruct/plain/own_max"]["OFFWORLD"] = 4
        self.assertEqual(score85.pilot_gate(counts, 16)["action"], "chat_only_posttrained")

    def test_safetensors_range_reader_column_and_row(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            matrix = np.arange(30, dtype=np.float32).reshape(6, 5)
            filename = root / "model.safetensors"
            name = "model.layers.10.mlp.down_proj.weight"
            save_file({name: matrix}, str(filename))
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": filename.stat().st_size}, "weight_map": {name: filename.name}}),
                encoding="utf-8",
            )
            reader = SafetensorsRangeReader(root)
            column = reader.read_column(name, 3)
            row = reader.read_row(name, 4)
            self.assertTrue(np.array_equal(column, matrix[:, 3]))
            self.assertTrue(np.array_equal(row, matrix[4, :]))
            self.assertTrue(np.array_equal(reader.read_tensor(name), matrix))

    def test_fisher_push_rule_and_control_gate(self):
        treated = {"n": 48, "OFFWORLD": 10}
        baseline = {"n": 48, "OFFWORLD": 0}
        control = {"n": 48, "OFFWORLD": 0}
        result = score85.shows_push(treated, baseline, control)
        self.assertTrue(result["shows_push"])
        self.assertLess(result["p_one_sided"], 0.05)
        blocked = score85.shows_push(treated, baseline, {"n": 48, "OFFWORLD": 3})
        self.assertFalse(blocked["shows_push"])
        self.assertIn("control gate failed", blocked["reason"])

    def test_no_onset_different_unit_and_calibration_disagreement(self):
        order = [cp.key for cp in CHECKPOINTS]
        self.assertIsNone(score85.earliest_suffix_onset(
            ["a", "b", "c"], {"a": True, "b": True, "c": False},
            {"a": True, "b": True, "c": True}))
        counts, step0, step1 = make_counts(own_break="instruct")
        no_onset = score85.evaluate_decisions(counts, step0, step1)
        self.assertIsNone(no_onset["onset"]["own_max"])

        counts, step0, step1 = make_counts(invalid_gate="midtrain_51B")
        different = score85.evaluate_decisions(counts, step0, step1)
        self.assertFalse(different["same_unit_gates"]["midtrain_51B"]["same_unit"])
        self.assertIsNone(different["push_details"]["own_max"]["midtrain_51B"]["shows_push"])
        self.assertEqual(different["onset"]["own_max"], "sft")

        counts, step0, step1 = make_counts()
        calibrated = score85.evaluate_decisions(counts, step0, step1)
        self.assertEqual(calibrated["onset"]["own_max"], "midtrain_51B")
        self.assertEqual(calibrated["onset"]["absolute"], "instruct")
        self.assertTrue(calibrated["onset"]["calibration_disagreement"])
        self.assertIsNone(calibrated["onset"]["reported_onset"])

    def test_prepare_and_reject_bad_labels(self):
        records = [
            {"mode": "pilot", "checkpoint": "pretrain_4T", "frame": "plain", "cell": "unclamped",
             "question_index": 0, "sample": 0, "seed": 85000, "answer": "An answer."},
            {"mode": "pilot", "checkpoint": "instruct", "frame": "plain", "cell": "own_max",
             "question_index": 0, "sample": 0, "seed": 85000, "answer": "Another answer."},
        ]
        with tempfile.TemporaryDirectory() as temp:
            items = Path(temp) / "items.jsonl"
            key_path = Path(temp) / "key.json"
            key = score85.prepare(records, items, key_path, seed=85)
            self.assertEqual(len(key), 2)
            item_rows = [json.loads(x) for x in items.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(set(item_rows[0]), {"id", "text"})
            labels = [{"id": item_id, "origin": "NEITHER", "form": "OK"} for item_id in key]
            checked = score85.validate_labels(labels, key)
            self.assertEqual(len(checked), 2)
            with self.assertRaisesRegex(ValueError, "responsive field"):
                score85.validate_labels(labels, key, require_responsive=True)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                score85.validate_labels(labels + [labels[0]], key)
            with self.assertRaisesRegex(ValueError, "missing labels"):
                score85.validate_labels(labels[:1], key)
            bad = [dict(row) for row in labels]
            bad[0]["origin"] = "UNKNOWN"
            with self.assertRaisesRegex(ValueError, "invalid origin"):
                score85.validate_labels(bad, key)
            extra = [dict(row) for row in labels]
            extra[0]["id"] = "not-in-key"
            with self.assertRaisesRegex(ValueError, "not in the answer key"):
                score85.validate_labels(extra, key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
