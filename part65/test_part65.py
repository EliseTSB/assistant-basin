#!/usr/bin/env python3
"""Offline Part 65 regression tests. Reads the five Part 65 modules, builds a tiny random-weight torch model and fake batch API, and writes only temporary fixtures that the tests remove."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

import torch
from torch import nn

import analyze65
import judge65
import rank65
import steer65
from paid_ledger import HARD_CAP_USD


class TinyMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden, intermediate)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, hidden: Any) -> Any:
        return self.down_proj(torch.tanh(self.up_proj(hidden)))


class TinyBlock(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.mlp = TinyMLP(hidden, intermediate)

    def forward(self, hidden: Any) -> Any:
        return hidden + self.mlp(hidden)


class TinyBackbone(nn.Module):
    def __init__(self, hidden: int, intermediate: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList([TinyBlock(hidden, intermediate) for _ in range(layers)])


class TinyCausalModel(nn.Module):
    """A random-weight causal toy model with a Qwen-shaped model.layers path."""
    def __init__(self, vocab: int = 19, hidden: int = 8, intermediate: int = 12, layers: int = 1):
        super().__init__()
        self.model = TinyBackbone(hidden, intermediate, layers)
        self.embedding = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab)

    def forward(self, input_ids: Any, attention_mask: Any = None, use_cache: bool = False) -> Any:
        hidden = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))

    def generate(self, input_ids: Any, attention_mask: Any = None, max_new_tokens: int = 48,
                 do_sample: bool = False, num_beams: int = 1, pad_token_id: int = 0,
                 eos_token_id: int = 2) -> Any:
        sequence = input_ids
        mask = attention_mask
        for _ in range(max_new_tokens):
            logits = self(input_ids=sequence, attention_mask=mask, use_cache=False).logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            sequence = torch.cat((sequence, next_token), dim=1)
            if mask is not None:
                mask = torch.cat((mask, torch.ones_like(next_token)), dim=1)
            if bool(torch.all(next_token == eos_token_id).item()):
                break
        return sequence


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    eos_token = "<eos>"
    pad_token = "<pad>"

    def __init__(self):
        self.padding_side = "right"

    def apply_chat_template(self, messages: list[dict[str, str]], tokenize: bool = False,
                            add_generation_prompt: bool = True, **template_kwargs: Any) -> str:
        self.last_template_kwargs = template_kwargs
        return "<user>" + messages[0]["content"] + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, texts: list[str], return_tensors: str = "pt", padding: bool = True,
                 add_special_tokens: bool = True) -> dict[str, Any]:
        rows = [[1] + [3 + (ord(ch) % 16) for ch in text] for text in texts]
        width = max(map(len, rows))
        values, masks = [], []
        for row in rows:
            pad = width - len(row)
            if self.padding_side == "left":
                values.append([self.pad_token_id] * pad + row)
                masks.append([0] * pad + [1] * len(row))
            else:
                values.append(row + [self.pad_token_id] * pad)
                masks.append([1] * len(row) + [0] * pad)
        return {"input_ids": torch.tensor(values), "attention_mask": torch.tensor(masks)}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [3 + (ord(ch) % 16) for ch in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True,
               clean_up_tokenization_spaces: bool = False) -> str:
        return "".join(chr(97 + (int(token) % 26)) for token in token_ids
                       if not (skip_special_tokens and int(token) < 3))


def tiny_units() -> list[dict[str, Any]]:
    units = []
    for kind, prefix in (("neuron", "u"), ("control", "c")):
        for index in range(3):
            layer = index
            units.append({
                "unit_key": f"L{layer}:{prefix}{index}",
                "type": kind,
                "layer": layer,
                "unit": index if kind == "neuron" else None,
                "control_id": index if kind == "control" else None,
            })
    return units


def tiny_summaries(units: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    labels, summaries = {}, {}
    for unit in units:
        key = unit["unit_key"]
        labels[key] = {
            "unit_key": key,
            "type": unit["type"],
            "layer": unit["layer"],
            "read_label": "amber orchard motif",
        }
        summaries[key] = {
            "unit_key": key,
            "type": unit["type"],
            "layer": unit["layer"],
            "layer_band": rank65.layer_band(unit["layer"]),
            "write_words": ["amber", "orchard"],
            "steer_lens": {"positive": ["gold"], "negative": ["blue"]},
        }
    return labels, summaries


def planted_analysis(effect: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                           dict[str, str], dict[str, list[str]]]:
    units = []
    for kind, prefix in (("neuron", "u"), ("control", "c")):
        for index in range(250):
            layer = index % 28
            units.append({
                "unit_key": f"L{layer}:{prefix}{index}",
                "type": kind,
                "layer": layer,
                "unit": index if kind == "neuron" else None,
                "control_id": index if kind == "control" else None,
            })
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for unit in units:
        grouped.setdefault((unit["type"], rank65.layer_band(unit["layer"])), []).append(unit)
    answers = []
    labels, write_words = {}, {}
    for unit in units:
        key, kind, layer = unit["unit_key"], unit["type"], unit["layer"]
        band = rank65.layer_band(layer)
        index = int(key.split(":")[1][1:])
        labels[key] = f"amber orchard motif {index % 7}" if kind == "neuron" else f"comet sapphire structure {index % 7}"
        write_words[key] = (["amber", "orchard", "motif"] if kind == "neuron"
                            else ["ocean", "rhythm", "phrase"])
        peers = [row["unit_key"] for row in grouped[(kind, band)] if row["unit_key"] != key]
        foils = peers[:2]
        if kind == "neuron" and effect:
            correct_count = 2 if band == "last" or index % 2 == 0 else 1
        elif kind == "control" and effect:
            correct_count = 2 if index % 5 == 0 else 1
        else:
            correct_count = 1
        sentence_pass = (index % 2 == 0) if kind == "neuron" else False
        for trial, own_position in ((1, "A"), (2, "B")):
            answers.append({
                "unit_key": key,
                "judge_run_id": "test-run-65",
                "run_root": "/tmp/part65-analysis-test",
                "plan_sha256": "test-plan-65",
                "type": kind,
                "layer": layer,
                "trial": trial,
                "foil_unit_key": foils[trial - 1],
                "own_position": own_position,
                "correct": trial <= correct_count,
                "sentence_test_pass": sentence_pass,
            })
    return answers, units, labels, write_words


class FakeBatchAPI:
    def __init__(self, pending_checks: int = 0):
        self.posts = 0
        self.gets = 0
        self.pending_checks = pending_checks

    def __call__(self, method: str, path: str, body: dict[str, Any] | None = None,
                 timeout: float = 120.0) -> str:
        if method == "POST":
            self.posts += 1
            self.requests = body["requests"]
            return json.dumps({"id": "fake-batch-1"})
        self.gets += 1
        if path.endswith("/results"):
            rows = []
            for request in self.requests:
                trial = int(request["custom_id"].split("-")[-1])
                answer = "A" if trial % 2 == 1 else "B"
                rows.append({
                    "custom_id": request["custom_id"],
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "content": [{"type": "text", "text": answer}],
                            "usage": {"input_tokens": 20, "output_tokens": 1},
                        },
                    },
                })
            return "\n".join(json.dumps(row) for row in rows)
        if self.pending_checks:
            self.pending_checks -= 1
            return json.dumps({"id": "fake-batch-1", "processing_status": "in_progress"})
        return json.dumps({"id": "fake-batch-1", "processing_status": "ended"})


class FakeLedger:
    """Small local ledger double; avoids the shared paid ledger registry in tests."""
    def __init__(self, path: str, _cap: float):
        self.path = Path(path)

    def _append(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")

    def reserve(self, amount: float, kind: str, meta: dict[str, Any], **_kwargs: Any) -> str:
        prior = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()] \
            if self.path.exists() else []
        attempt_id = f"{len(prior) + 1:032x}"
        self._append({"event": "reserve", "attempt_id": attempt_id,
                      "amount_usd": amount, "kind": kind, "meta": meta})
        return attempt_id

    def settle(self, attempt_id: str, amount: float, usage: dict[str, int]) -> None:
        self._append({"event": "settle", "attempt_id": attempt_id,
                      "amount_usd": amount, "usage": usage})

    def release(self, attempt_id: str, http_status: int, body: str | dict[str, Any]) -> None:
        parsed = json.loads(body) if isinstance(body, str) else body
        if (not 400 <= http_status < 500 or http_status == 429
                or not isinstance(parsed, dict) or parsed.get("type") != "error"
                or not isinstance(parsed.get("error"), dict)):
            raise ValueError("release requires a validated Anthropic non-429 error")
        self._append({"event": "release", "attempt_id": attempt_id,
                      "http_status": http_status})

    def retire(self, attempt_id: str) -> None:
        self._append({"event": "retire", "attempt_id": attempt_id})

    def state(self) -> dict[str, float]:
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()] \
            if self.path.exists() else []
        closed = {row.get("attempt_id") for row in rows
                  if row.get("event") in ("settle", "release", "retire")}
        reserved = sum(float(row.get("amount_usd", 0.0)) for row in rows
                       if row.get("event") == "reserve" and row.get("attempt_id") not in closed)
        return {"reserved": reserved}


class Part65Tests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(65)

    def test_real_and_control_write_vector_norms_match(self):
        model = TinyCausalModel(hidden=8, intermediate=12, layers=3)
        units = [
            {"unit_key": "L0:u3", "type": "neuron", "layer": 0, "unit": 3, "control_id": None},
            {"unit_key": "L0:c2", "type": "control", "layer": 0, "unit": None, "control_id": 2},
            {"unit_key": "L1:u4", "type": "neuron", "layer": 1, "unit": 4, "control_id": None},
            {"unit_key": "L1:c7", "type": "control", "layer": 1, "unit": None, "control_id": 7},
        ]
        for layer in (0, 1):
            directions = torch.randn(100, 12)
            directions = torch.nn.functional.normalize(directions, dim=1)
            for unit in [row for row in units if row["layer"] == layer]:
                vec = steer65.write_direction(
                    torch, model.model.layers[layer].mlp.down_proj, unit, directions
                )
                self.assertAlmostEqual(float(torch.linalg.vector_norm(vec)), 1.0, places=6)
                self.assertAlmostEqual(
                    float(torch.linalg.vector_norm(vec * (0.8 * 2.5))),
                    2.0,
                    places=5,
                )

    def test_control_direction_is_normalized_w_down_times_hand_set_r(self):
        model = TinyCausalModel(hidden=4, intermediate=4, layers=2)
        down = model.model.layers[1].mlp.down_proj
        with torch.no_grad():
            down.weight.copy_(torch.tensor([
                [1.0, 2.0, 0.0, 1.0],
                [0.0, 1.0, 3.0, 0.0],
                [2.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 2.0],
            ]))
        r = torch.tensor([0.25, -0.5, 0.75, 0.0])
        controls = torch.zeros(100, 4)
        controls[2] = r
        unit = {"unit_key": "L1:c2", "type": "control", "layer": 1,
                "unit": None, "control_id": 2}
        actual = steer65.write_direction(torch, down, unit, controls)
        expected = torch.nn.functional.normalize(down.weight.detach().float() @ r, dim=0)
        torch.testing.assert_close(actual, expected)

    def test_layer1_push_fires_on_layer1_and_changes_output(self):
        model = TinyCausalModel(vocab=19, hidden=4, intermediate=4, layers=2)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.model.layers[1].mlp.down_proj.weight[0, 1] = 1.0
            model.lm_head.weight[3, 0] = -1.0
            model.lm_head.weight[4, 0] = 1.0
        unit = {"unit_key": "L1:u1", "type": "neuron", "layer": 1,
                "unit": 1, "control_id": None}
        result = steer65.process_unit(torch, model, TinyTokenizer(), unit, {"1": 1.0})
        baseline = {row["prompt_key"]: row["token_ids"] for row in result["baseline"]}
        positive = [row for row in result["rollouts"] if row["dose"] == 0.8]
        self.assertEqual(len(positive), 3)
        self.assertEqual(result["dose_plus_0_8_changed_count"], 3)
        for row in positive:
            self.assertGreater(row["hook_fired_count"], 0)
            self.assertEqual(row["hook_layer_modules"], ["model.layers.1.mlp"])
            self.assertNotEqual(row["token_ids"], baseline[row["prompt_key"]])

    def test_generate_batch_requires_explicit_hook_layer_for_pushes(self):
        model = TinyCausalModel(layers=2)
        pushes = torch.zeros(1, 8)
        with self.assertRaisesRegex(steer65.SteerError, "hook_layer is required"):
            steer65.generate_batch(torch, model, TinyTokenizer(), ["hello"], pushes)

    def test_pushed_generation_rejects_a_hook_that_never_fires(self):
        model = TinyCausalModel()
        unit = {"unit_key": "L0:u4", "type": "neuron", "layer": 0,
                "unit": 4, "control_id": None}

        @contextmanager
        def dead_hook(_mlp, _pushes, _layer_module):
            yield {"fired_count": 0, "layer_modules": []}

        with mock.patch.object(steer65, "intervention_hook", dead_hook):
            with self.assertRaisesRegex(steer65.SteerError, "hook fired zero times"):
                steer65.process_unit(torch, model, TinyTokenizer(), unit, {"0": 2.0})

    def test_dose_zero_reproduces_baseline_tokens(self):
        model = TinyCausalModel()
        tokenizer = TinyTokenizer()
        unit = {"unit_key": "L0:u4", "type": "neuron", "layer": 0, "unit": 4, "control_id": None}
        started = time.perf_counter()
        result = steer65.process_unit(torch, model, tokenizer, unit, {"0": 2.0})
        elapsed = time.perf_counter() - started
        self.assertTrue(result["dose_zero_matches_baseline"])
        self.assertEqual(len(result["rollouts"]), 9)
        self.assertEqual(len(result["baseline"]), 3)
        self.assertGreaterEqual(elapsed, 0.0)
        self.cpu_tiny_unit_seconds = elapsed
        print(f"tiny-model CPU one-unit elapsed: {elapsed:.6f}s")

    def test_dose_zero_check_compares_with_a_separate_unsteered_generation(self):
        model = TinyCausalModel()
        tokenizer = TinyTokenizer()
        unit = {"unit_key": "L0:u4", "type": "neuron", "layer": 0,
                "unit": 4, "control_id": None}
        original = steer65.generate_batch
        calls = []

        def mismatching_zero(torch_arg, model_arg, tokenizer_arg, prompts,
                             pushes=None, max_new_tokens=steer65.MAX_NEW_TOKENS,
                             hook_layer=None):
            calls.append({"unsteered": pushes is None, "pushes": pushes})
            rows = original(torch_arg, model_arg, tokenizer_arg, prompts, pushes,
                            max_new_tokens=max_new_tokens, hook_layer=hook_layer)
            if len(calls) == 2:
                rows[0]["token_ids"] = rows[0]["token_ids"] + [999]
            return rows

        with mock.patch.object(steer65, "generate_batch", side_effect=mismatching_zero):
            with self.assertRaisesRegex(steer65.SteerError, "dose 0 did not reproduce"):
                steer65.process_unit(torch, model, tokenizer, unit, {"0": 2.0})
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0]["unsteered"])
        self.assertFalse(calls[1]["unsteered"])
        self.assertEqual(float(torch.linalg.vector_norm(calls[1]["pushes"])), 0.0)

    def test_pilot_check_uses_recorded_run_wide_gates(self):
        model = TinyCausalModel(vocab=19, hidden=4, intermediate=4, layers=2)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.model.layers[1].mlp.down_proj.weight[0, 1] = 1.0
            model.lm_head.weight[3, 0] = -1.0
            model.lm_head.weight[4, 0] = 1.0
        unit = {"unit_key": "L1:u1", "type": "neuron", "layer": 1,
                "unit": 1, "control_id": None}
        row = steer65.process_unit(torch, model, TinyTokenizer(), unit, {"1": 1.0})
        row["equal_norm_spread_by_layer"] = {"1": 0.0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            result = steer65.pilot_check(path)
            for rollout in row["rollouts"]:
                if rollout["dose"] == 0.8:
                    rollout["token_ids"] = next(
                        baseline_row["token_ids"] for baseline_row in row["baseline"]
                        if baseline_row["prompt_key"] == rollout["prompt_key"]
                    )
                    rollout["differs_from_baseline"] = False
            row["dose_plus_0_8_changed_count"] = 0
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(steer65.SteerError, r"only 0/3 \+0.8 rollouts differed"):
                steer65.pilot_check(path)
        self.assertEqual(result["plus_0_8_changed"], 3)
        self.assertEqual(result["plus_0_8_total"], 3)
        self.assertTrue(result["dose_zero_matches_baseline"])

    def test_pilot_check_push_norm_spread_is_relative(self):
        import copy
        model = TinyCausalModel(vocab=19, hidden=4, intermediate=4, layers=2)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.model.layers[1].mlp.down_proj.weight[0, 1] = 1.0
            model.lm_head.weight[3, 0] = -1.0
            model.lm_head.weight[4, 0] = 1.0
        unit = {"unit_key": "L1:u1", "type": "neuron", "layer": 1,
                "unit": 1, "control_id": None}
        median = 2355.136
        row = steer65.process_unit(torch, model, TinyTokenizer(), unit, {"1": median})
        row["equal_norm_spread_by_layer"] = {"1": 0.0}
        rounding = copy.deepcopy(row)
        rounding["unit_key"], rounding["type"] = "L1:c1", "control"
        rounding["push_norms"] = {key: value + 0.000244140625 for key, value in row["push_norms"].items()}
        scaled = copy.deepcopy(row)
        scaled["unit_key"], scaled["type"] = "L1:c2", "control"
        scaled["median_residual_norm"] = median * 1.01
        scaled["push_norms"] = {key: value * 1.01 for key, value in row["push_norms"].items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "steer.jsonl"
            path.write_text(json.dumps(row) + "\n" + json.dumps(rounding) + "\n", encoding="utf-8")
            steer65.pilot_check(path)  # one float32 step at a large norm is rounding, not a mismatch
            path.write_text(json.dumps(row) + "\n" + json.dumps(scaled) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(steer65.SteerError, "push norms differ in layer 1"):
                steer65.pilot_check(path)

    def test_resume_skips_finished_units(self):
        units = tiny_units()
        selected = steer65.select_pilot_units(units, 4, "both", seed=17)
        self.assertEqual(sum(row["type"] == "neuron" for row in selected), 2)
        finished = {selected[0]["unit_key"]}
        remaining = steer65.pending_units(selected, finished)
        self.assertEqual(len(remaining), 3)
        self.assertNotIn(selected[0]["unit_key"], {row["unit_key"] for row in remaining})

    def test_word_cluster_mass_ranks_overrepresented_word(self):
        base = {
            "type": "control",
            "layer": 2,
            "rollouts": [{"text": "common common control"} for _ in range(9)],
            "steer_lens": {"+0.8": [{"token": "a"}], "-0.8": [{"token": "b"}]},
        }
        target = {
            "unit_key": "L2:u0", "type": "neuron", "layer": 2,
            "rollouts": [{"text": "amber amber orchard"} for _ in range(9)],
            "steer_lens": {"+0.8": [{"token": "gold"}], "-0.8": [{"token": "blue"}]},
        }
        control_a = {**base, "unit_key": "L2:c0"}
        control_b = {**base, "unit_key": "L3:c1", "layer": 3}
        ranked = rank65.rank_rows([target, control_a, control_b])
        words = ranked[0]["write_words"]
        self.assertIn("amber", words)
        self.assertGreater(ranked[0]["words"][0]["word_cluster_mass"], 0.0)

    def test_foils_are_distinct_same_type_and_band_and_prompt_is_blind(self):
        units = tiny_units()
        labels, summaries = tiny_summaries(units)
        trials = judge65.build_trials(units, labels, summaries, seed=9)
        by_unit: dict[str, list[dict[str, Any]]] = {}
        for trial in trials:
            item = trial["mapping"]
            by_unit.setdefault(item["unit_key"], []).append(trial)
            self.assertNotEqual(item["unit_key"], item["foil_unit_key"])
            self.assertEqual(item["type"], next(u["type"] for u in units if u["unit_key"] == item["foil_unit_key"]))
            self.assertEqual(item["layer_band"], rank65.layer_band(
                next(u["layer"] for u in units if u["unit_key"] == item["foil_unit_key"])
            ))
            body = trial["request"]["params"]["messages"][0]["content"]
            self.assertNotIn(item["unit_key"], body)
            self.assertNotIn(item["foil_unit_key"], body)
            self.assertNotIn("neuron", body)
            self.assertNotIn("control", body)
            self.assertNotIn("layer", body)
        for rows in by_unit.values():
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["mapping"]["foil_unit_key"] != rows[1]["mapping"]["foil_unit_key"], True)
            self.assertEqual([row["mapping"]["own_position"] for row in rows], ["A", "B"])

    def test_paid_batch_lifecycle_settles_and_resume_does_not_resubmit(self):
        units = tiny_units()
        labels, summaries = tiny_summaries(units)
        trials = judge65.build_trials(units, labels, summaries, seed=10)[:2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "judge"
            ledger_path = Path(directory) / "shared-paid-ledger.jsonl"
            judge65.prepare_run(trials, root, chunk_size=2, seed=10)
            fake = FakeBatchAPI()
            with mock.patch.object(judge65, "Ledger", FakeLedger):
                totals = judge65.run_chunks(root, ledger_path, fake, poll_seconds=0,
                                            timeout_seconds=1, sleep=lambda _seconds: None)
                self.assertEqual(totals["settled"], 1)
                self.assertEqual(fake.posts, 1)
                self.assertEqual(FakeLedger(str(ledger_path), HARD_CAP_USD).state()["reserved"], 0.0)
                manifest = json.loads((root / "run.json").read_text())
                self.assertEqual(manifest["completion"]["status"], "complete")
                self.assertEqual(manifest["completion"]["judge_run_id"], manifest["judge_run_id"])
                self.assertEqual(manifest["completion"]["plan_sha256"], manifest["plan_sha256"])
                second = judge65.run_chunks(root, ledger_path, fake, poll_seconds=0,
                                            timeout_seconds=1, sleep=lambda _seconds: None)
            self.assertEqual(second["settled"], 1)
            self.assertEqual(fake.posts, 1)
            rows = judge65.read_jsonl(root / "answers.jsonl")
            self.assertEqual(len(rows), 2)
            manifest = json.loads((root / "run.json").read_text())
            self.assertEqual({row["judge_run_id"] for row in rows}, {manifest["judge_run_id"]})
            self.assertEqual({row["run_root"] for row in rows}, {str(root.resolve())})
            self.assertEqual({row["plan_sha256"] for row in rows}, {manifest["plan_sha256"]})
            events = [json.loads(line) for line in ledger_path.read_text().splitlines()]
            reserve_meta = next(row["meta"] for row in events if row["event"] == "reserve")
            self.assertEqual(reserve_meta["judge_run_id"], manifest["judge_run_id"])
            self.assertEqual(reserve_meta["run_root"], str(root.resolve()))

    def test_persist_answers_dedupes_by_run_and_custom_id_and_rejects_a_second_run(self):
        """Catches mutant: custom-ID-only dedupe silently drops another run's answer."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = {"judge_run_id": "run-a", "custom_id": "p65-0000", "chosen": "A"}
            second = {"judge_run_id": "run-a", "custom_id": "p65-0001", "chosen": "B"}
            judge65._persist_answers(root, [first, second])
            judge65._persist_answers(root, [{**first, "chosen": "B"}])
            saved = judge65.read_jsonl(root / "answers.jsonl")
            self.assertEqual(len(saved), 2)
            self.assertEqual(saved[0]["chosen"], "A")
            with self.assertRaisesRegex(judge65.JudgeError, "refusing to mix answers"):
                judge65._persist_answers(root, [
                    {"judge_run_id": "run-b", "custom_id": "p65-0000", "chosen": "B"}
                ])
            self.assertEqual(len(judge65.read_jsonl(root / "answers.jsonl")), 2)

    def test_completion_marker_requires_settled_chunks_and_all_saved_answers(self):
        """Catches mutant: run completion is recorded from chunk states without saved answers."""
        units = tiny_units()
        labels, summaries = tiny_summaries(units)
        trials = judge65.build_trials(units, labels, summaries, seed=78)[:2]
        with tempfile.TemporaryDirectory() as directory:
            root, ledger_path = Path(directory) / "judge", Path(directory) / "ledger.jsonl"
            judge65.prepare_run(trials, root, chunk_size=2, seed=78)
            with mock.patch.object(judge65, "Ledger", FakeLedger), \
                 mock.patch.object(judge65, "_persist_answers", return_value=None):
                totals = judge65.run_chunks(root, ledger_path, FakeBatchAPI(), poll_seconds=0,
                                            timeout_seconds=1, sleep=lambda _seconds: None)
            self.assertEqual(totals["settled"], 1)
            chunk = json.loads((root / "chunks" / "chunk-0000.json").read_text())
            self.assertEqual(chunk["status"], "settled")
            self.assertFalse((root / "answers.jsonl").exists())
            self.assertNotIn("completion", json.loads((root / "run.json").read_text()))

    def test_run_identity_survives_prepare_reopen_and_is_stamped_on_each_chunk(self):
        units = tiny_units()
        labels, summaries = tiny_summaries(units)
        trials = judge65.build_trials(units, labels, summaries, seed=65)[:4]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "judge"
            judge65.prepare_run(trials, root, chunk_size=2, seed=65)
            first = json.loads((root / "run.json").read_text())
            judge65.prepare_run(trials, root, chunk_size=2, seed=65)
            reopened = json.loads((root / "run.json").read_text())
            self.assertEqual(reopened["judge_run_id"], first["judge_run_id"])
            self.assertEqual(reopened["run_root"], str(root.resolve()))
            self.assertEqual(reopened["chunk_size"], 2)
            self.assertEqual(reopened["chunk_count"], 2)
            self.assertEqual(reopened["reserve_total_usd"],
                             judge65._reserve_total(trials, 2))
            for path in sorted((root / "chunks").glob("chunk-*.json")):
                chunk = json.loads(path.read_text())
                self.assertEqual(chunk["judge_run_id"], first["judge_run_id"])
                self.assertEqual(chunk["run_root"], first["run_root"])
                self.assertEqual(chunk["plan_sha256"], first["plan_sha256"])

    def test_reserve_total_prints_real_plan_estimate_without_sending(self):
        units = tiny_units()
        labels, summaries = tiny_summaries(units)
        trials = judge65.build_trials(units, labels, summaries, seed=65)
        output = io.StringIO()
        with mock.patch.object(judge65, "prepare_inputs", return_value=trials) as prepare_inputs, \
             mock.patch.object(judge65, "api_call", side_effect=AssertionError("must not send")), \
             redirect_stderr(io.StringIO()), redirect_stdout(output):
            code = judge65.main(["--reserve-total", "--ranked", "ranked.jsonl", "--chunk-size", "3"])
        self.assertEqual(code, 0)
        self.assertEqual(prepare_inputs.call_args.args[-1], judge65.REGISTERED_SEED)
        self.assertEqual(judge65.REGISTERED_SEED, 6501)
        record = json.loads(output.getvalue())
        self.assertEqual(record["request_count"], len(trials))
        self.assertEqual(record["reserve_total_usd"], judge65._reserve_total(trials, 3))
        self.assertEqual(record["subcap_usd"], judge65.PART65_CAP_USD)

    def test_validated_release_is_terminal_stops_later_chunks_and_main_exits_2(self):
        """Catches mutant: release is retryable, unmarked, or main returns success."""
        units = tiny_units()
        labels, summaries = tiny_summaries(units)
        trials = judge65.build_trials(units, labels, summaries, seed=66)[:3]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "judge"
            ledger_path = Path(directory) / "ledger.jsonl"
            class ReleaseFirstChunk:
                def __init__(self):
                    self.posts = 0

                def __call__(self, method, _path, _body=None, **_kwargs):
                    if method == "POST":
                        self.posts += 1
                        raise judge65.APIHttpError(
                            400, '{"type":"error","error":{"type":"invalid_request_error"}}'
                        )
                    raise AssertionError("a validated release has no batch to fetch")

            fake = ReleaseFirstChunk()
            judge65.prepare_run(trials, root, chunk_size=1, seed=66)
            with mock.patch.object(judge65, "Ledger", FakeLedger):
                totals = judge65.run_chunks(root, ledger_path, fake, poll_seconds=0,
                                            timeout_seconds=1, sleep=lambda _seconds: None)
            self.assertEqual(totals["released"], 1)
            self.assertEqual(fake.posts, 1)
            states = [json.loads(path.read_text()) for path in
                      sorted((root / "chunks").glob("chunk-*.json"))]
            self.assertEqual([state["status"] for state in states],
                             ["released", "ready", "ready"])
            manifest = json.loads((root / "run.json").read_text())
            self.assertEqual(manifest["failure"]["outcome"], "released")
            self.assertNotIn("completion", manifest)
            with self.assertRaisesRegex(analyze65.AnalysisError, "completion marker"):
                analyze65.analyze_files(
                    root / "answers.jsonl", Path("missing-collection.jsonl"),
                    Path("missing-neuron.jsonl"), Path("missing-control.jsonl"),
                    Path("missing-ranked.jsonl"),
                )

            main_root = Path(directory) / "main-exit"
            err, out = io.StringIO(), io.StringIO()
            with mock.patch.object(judge65, "prepare_inputs", return_value=trials), \
                 mock.patch.object(judge65, "run_chunks", return_value={
                     "settled": 0, "pending": 0, "released": 1, "ambiguous": 0}), \
                 redirect_stderr(err), redirect_stdout(out):
                code = judge65.main([
                    "--ranked", "ranked.jsonl", "--out-dir", str(main_root),
                    "--chunk-size", "1",
                ])
            self.assertEqual(code, 2)
            self.assertIn('"released": 1', out.getvalue())

    def test_judge_cli_cannot_select_an_alternate_ledger(self):
        """Catches mutant: `main` restores a user-controlled --ledger path."""
        error = io.StringIO()
        with redirect_stderr(error), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                judge65.main(["--ledger", "/tmp/alternate-ledger.jsonl"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unrecognized arguments: --ledger", error.getvalue())

    def test_invalid_release_body_and_429_are_ambiguous_and_stop_later_chunks(self):
        """Catches mutant: every 4xx is released, or ambiguity does not stop future spending."""
        units = tiny_units()
        labels, summaries = tiny_summaries(units)
        trials = judge65.build_trials(units, labels, summaries, seed=67)[:4]

        class Reject:
            def __init__(self, status, body):
                self.status, self.body, self.posts = status, body, 0

            def __call__(self, method, _path, _body=None, **_kwargs):
                if method == "POST":
                    self.posts += 1
                    raise judge65.APIHttpError(self.status, self.body)
                raise AssertionError("unknown no-ID outcomes cannot be polled")

        for status, body in ((400, '{"type":"not_an_error"}'),
                             (429, '{"type":"error","error":{"type":"rate_limit_error"}}')):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "judge"
                ledger_path = Path(directory) / "ledger.jsonl"
                judge65.prepare_run(trials, root, chunk_size=2, seed=67)
                fake = Reject(status, body)
                with mock.patch.object(judge65, "Ledger", FakeLedger):
                    totals = judge65.run_chunks(root, ledger_path, fake, poll_seconds=0,
                                                timeout_seconds=0)
                    recovery = judge65.run_chunks(root, ledger_path, fake, poll_seconds=0,
                                                  timeout_seconds=0, recover_only=True)
                states = [json.loads(path.read_text()) for path in
                          sorted((root / "chunks").glob("chunk-*.json"))]
                self.assertEqual(totals["ambiguous"], 1)
                self.assertEqual(recovery["ambiguous"], 1)
                self.assertEqual(fake.posts, 1)
                self.assertEqual(states[0]["status"], "unknown_no_batch")
                self.assertEqual(states[1]["status"], "ready")
                manifest = json.loads((root / "run.json").read_text())
                self.assertEqual(manifest["failure"]["outcome"], "ambiguous")
                self.assertGreater(FakeLedger(str(ledger_path), HARD_CAP_USD).state()["reserved"], 0)
                if status == 429:
                    err = io.StringIO()
                    with mock.patch.object(judge65, "run_chunks", return_value=totals), \
                         redirect_stderr(err), redirect_stdout(io.StringIO()):
                        code = judge65.main([
                            "--out-dir", str(root), "--recover"
                        ])
                    self.assertEqual(code, 2)
                    self.assertIn("ambiguous judge chunk", err.getvalue())
                    self.assertIn("Do not start another output directory", err.getvalue())
                    self.assertEqual(fake.posts, 1)

    def test_cross_directory_plan_fence_rejects_active_released_and_settled_reservations(self):
        """Catches mutant: a new output directory can re-POST a plan with a prior reservation."""
        units = tiny_units()
        labels, summaries = tiny_summaries(units)
        trials = judge65.build_trials(units, labels, summaries, seed=68)[:2]

        class Reject400:
            def __init__(self):
                self.posts = 0

            def __call__(self, method, _path, _body=None, **_kwargs):
                if method == "POST":
                    self.posts += 1
                    raise judge65.APIHttpError(
                        400, '{"type":"error","error":{"type":"invalid_request_error"}}'
                    )
                raise AssertionError("released chunk must not fetch")

        for outcome in ("active", "released", "settled"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as directory:
                root_a, root_b = Path(directory) / "A", Path(directory) / "B"
                ledger_path = Path(directory) / "shared-ledger.jsonl"
                judge65.prepare_run(trials, root_a, chunk_size=2, seed=68)
                judge65.prepare_run(trials, root_b, chunk_size=2, seed=68)
                api = Reject400() if outcome == "released" else FakeBatchAPI(
                    pending_checks=1 if outcome == "active" else 0
                )
                with mock.patch.object(judge65, "Ledger", FakeLedger):
                    first = judge65.run_chunks(
                        root_a, ledger_path, api, poll_seconds=0,
                        timeout_seconds=0 if outcome == "active" else 1,
                        sleep=lambda _seconds: None,
                    )
                    self.assertIn("pending" if outcome == "active" else outcome, first)
                    with self.assertRaisesRegex(judge65.JudgeError, "another run"):
                        judge65.run_chunks(root_b, ledger_path, api, poll_seconds=0,
                                           timeout_seconds=0)
                self.assertEqual(api.posts, 1)

    def test_released_reservation_latches_different_seed_plan_before_any_send(self):
        """Catches mutant: the judge-wide terminal latch is filtered by plan hash."""
        units = tiny_units()
        labels, summaries = tiny_summaries(units)
        plan_a = judge65.build_trials(units, labels, summaries, seed=80)
        plan_b = judge65.build_trials(units, labels, summaries, seed=81)

        class ReleaseFirst:
            def __init__(self):
                self.posts = 0

            def __call__(self, method, _path, _body=None, **_kwargs):
                if method == "POST":
                    self.posts += 1
                    raise judge65.APIHttpError(
                        400, '{"type":"error","error":{"type":"invalid_request_error"}}'
                    )
                raise AssertionError("a validated release has no batch to fetch")

        with tempfile.TemporaryDirectory() as directory:
            root_a, root_b = Path(directory) / "plan-a", Path(directory) / "plan-b"
            ledger_path = Path(directory) / "shared-ledger.jsonl"
            judge65.prepare_run(plan_a, root_a, chunk_size=1, seed=80)
            judge65.prepare_run(plan_b, root_b, chunk_size=1, seed=81)
            manifest_a = json.loads((root_a / "run.json").read_text())
            manifest_b = json.loads((root_b / "run.json").read_text())
            self.assertNotEqual(manifest_a["plan_sha256"], manifest_b["plan_sha256"])

            api_a, api_b = ReleaseFirst(), FakeBatchAPI()
            with mock.patch.object(judge65, "Ledger", FakeLedger):
                first = judge65.run_chunks(root_a, ledger_path, api_a, poll_seconds=0,
                                            timeout_seconds=1)
                self.assertEqual(first["released"], 1)
                with self.assertRaisesRegex(judge65.JudgeError, "another run"):
                    judge65.run_chunks(root_b, ledger_path, api_b, poll_seconds=0,
                                       timeout_seconds=1)
            self.assertEqual(api_a.posts, 1)
            self.assertEqual(api_b.posts, 0)
            events = [json.loads(line) for line in ledger_path.read_text().splitlines()]
            self.assertEqual(sum(row["event"] == "reserve" for row in events), 1)

    def test_copying_a_reserved_chunk_to_another_run_is_refused_before_post(self):
        """Catches mutant: the local reserved branch skips run-identity validation."""
        units = tiny_units()
        labels, summaries = tiny_summaries(units)
        trials = judge65.build_trials(units, labels, summaries, seed=69)[:2]
        with tempfile.TemporaryDirectory() as directory:
            root_a, root_b = Path(directory) / "A", Path(directory) / "B"
            ledger_path = Path(directory) / "shared-ledger.jsonl"
            judge65.prepare_run(trials, root_a, chunk_size=2, seed=69)
            judge65.prepare_run(trials, root_b, chunk_size=2, seed=69)
            state_a = json.loads((root_a / "chunks" / "chunk-0000.json").read_text())
            manifest_a = json.loads((root_a / "run.json").read_text())
            attempt_id = FakeLedger(str(ledger_path), HARD_CAP_USD).reserve(
                state_a["reserve_usd"], "part65-judge",
                {"plan_sha256": manifest_a["plan_sha256"],
                 "judge_run_id": manifest_a["judge_run_id"],
                 "run_root": manifest_a["run_root"], "chunk_index": 0,
                 "request_count": len(state_a["trials"])},
            )
            state_a["status"] = "reserved"
            state_a["attempt_id"] = attempt_id
            (root_b / "chunks" / "chunk-0000.json").write_text(json.dumps(state_a))
            api = FakeBatchAPI()
            with mock.patch.object(judge65, "Ledger", FakeLedger):
                with self.assertRaisesRegex(judge65.JudgeError, "chunk identity differs"):
                    judge65.run_chunks(root_b, ledger_path, api)
            self.assertEqual(api.posts, 0)

    def test_shared_ledger_lock_serializes_fresh_directory_preflight_and_post(self):
        """Catches mutant: replacing the shared ledger lock with a no-op permits both preflights."""
        units = tiny_units()
        labels, summaries = tiny_summaries(units)
        trials = judge65.build_trials(units, labels, summaries, seed=70)[:2]
        with tempfile.TemporaryDirectory() as directory:
            root_a, root_b = Path(directory) / "A", Path(directory) / "B"
            ledger_path = Path(directory) / "shared-ledger.jsonl"
            judge65.prepare_run(trials, root_a, chunk_size=2, seed=70)
            judge65.prepare_run(trials, root_b, chunk_size=2, seed=70)
            prior = judge65._prior_reservation
            real_lock = judge65._lock_file
            first_in_prior = threading.Event()
            release_first = threading.Event()
            second_lock_attempted = threading.Event()
            second_in_prior = threading.Event()
            ledger_lock_path = judge65._ledger_judge_lock_path(ledger_path)
            api = FakeBatchAPI(pending_checks=1)
            results = []

            def block_first_prior(*args, **kwargs):
                if threading.current_thread().name == "first-run":
                    first_in_prior.set()
                    if not release_first.wait(5):
                        raise TimeoutError("test did not release first ledger preflight")
                else:
                    second_in_prior.set()
                return prior(*args, **kwargs)

            @contextmanager
            def observe_lock(path):
                if (Path(path) == ledger_lock_path
                        and threading.current_thread().name == "second-run"):
                    second_lock_attempted.set()
                with real_lock(path):
                    yield

            def run(root):
                try:
                    result = judge65.run_chunks(root, ledger_path, api, poll_seconds=0,
                                                timeout_seconds=0,
                                                sleep=lambda _seconds: None)
                    results.append(result)
                except Exception as exc:
                    results.append(exc)

            with mock.patch.object(judge65, "Ledger", FakeLedger), \
                 mock.patch.object(judge65, "_prior_reservation", side_effect=block_first_prior), \
                 mock.patch.object(judge65, "_lock_file", side_effect=observe_lock):
                first = threading.Thread(target=run, args=(root_a,), name="first-run")
                second = threading.Thread(target=run, args=(root_b,), name="second-run")
                threads = [first, second]
                first.start()
                try:
                    self.assertTrue(first_in_prior.wait(5), "first run never entered its locked preflight")
                    second.start()
                    self.assertTrue(second_lock_attempted.wait(5), "second run never attempted the ledger lock")
                    self.assertFalse(
                        second_in_prior.wait(0.2),
                        "second run entered the preflight while first run held the shared ledger lock",
                    )
                finally:
                    release_first.set()
                    for thread in threads:
                        if thread.ident is not None:
                            thread.join(5)
            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(api.posts, 1)
            self.assertEqual(sum(isinstance(result, dict) and result.get("pending") == 1
                                 for result in results), 1)
            self.assertEqual(sum(isinstance(result, judge65.JudgeError)
                                 and "another run" in str(result) for result in results), 1)

    def test_real_ledger_subcap_refuses_before_post_and_main_exits_2(self):
        """Catches mutants that ignore the real sub-cap, POST the refused chunk, or leak its error."""
        units = tiny_units()
        labels, summaries = tiny_summaries(units)
        trials = judge65.build_trials(units, labels, summaries, seed=75)[:3]

        class MaxUsageBatchAPI:
            def __init__(self):
                self.posts = 0
                self.results = {}

            def __call__(self, method, path, body=None, **_kwargs):
                if method == "POST":
                    self.posts += 1
                    batch_id = f"subcap-batch-{self.posts}"
                    self.results[batch_id] = []
                    for request in body["requests"]:
                        usage = {
                            "input_tokens": judge65.math.ceil(
                                len(judge65._canonical_json(request)) / 3
                            ),
                            "output_tokens": request["params"]["max_tokens"],
                        }
                        self.results[batch_id].append({
                            "custom_id": request["custom_id"],
                            "result": {"type": "succeeded", "message": {
                                "content": [{"type": "text", "text": "A"}],
                                "usage": usage,
                            }},
                        })
                    return json.dumps({"id": batch_id})
                batch_id = path.split("/")[-2] if path.endswith("/results") else path.split("/")[-1]
                if path.endswith("/results"):
                    return "\n".join(json.dumps(row) for row in self.results[batch_id])
                return json.dumps({"processing_status": "ended"})

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root, ledger_path = Path(directory) / "judge", Path(directory) / "real-ledger.jsonl"
            judge65.prepare_run(trials, root, chunk_size=1, seed=75)
            reserves = [json.loads(path.read_text())["reserve_usd"] for path in
                        sorted((root / "chunks").glob("chunk-*.json"))]
            subcap = reserves[0] + reserves[1] / 2
            api = MaxUsageBatchAPI()
            with mock.patch.object(judge65, "PART65_CAP_USD", subcap):
                with self.assertRaisesRegex(judge65.JudgeError, "sub-cap 'part65-judge'"):
                    judge65.run_chunks(root, ledger_path, api, poll_seconds=0,
                                       timeout_seconds=1, sleep=lambda _seconds: None)
            self.assertEqual(api.posts, 1)
            states = [json.loads(path.read_text()) for path in
                      sorted((root / "chunks").glob("chunk-*.json"))]
            self.assertEqual(states[0]["status"], "settled")
            self.assertEqual(states[1]["status"], "ready")
            self.assertEqual(states[2]["status"], "ready")
            ledger_events = [json.loads(line) for line in ledger_path.read_text().splitlines()]
            self.assertEqual(sum(row["event"] == "reserve" for row in ledger_events), 1)

            main_root = Path(directory) / "main-exit"
            main_ledger_path = Path(directory) / "main-real-ledger.jsonl"
            main_api = MaxUsageBatchAPI()
            real_run_chunks = judge65.run_chunks

            def use_test_ledger(root, **kwargs):
                return real_run_chunks(root, main_ledger_path, main_api, **kwargs)

            err = io.StringIO()
            with mock.patch.object(judge65, "PART65_CAP_USD", subcap), \
                 mock.patch.object(judge65, "prepare_inputs", return_value=trials), \
                 mock.patch.object(judge65, "run_chunks", side_effect=use_test_ledger), \
                 redirect_stderr(err), redirect_stdout(io.StringIO()):
                code = judge65.main([
                    "--ranked", "ranked.jsonl", "--out-dir", str(main_root),
                    "--chunk-size", "1",
                ])
            self.assertEqual(code, 2)
            self.assertIn("sub-cap 'part65-judge'", err.getvalue())
            self.assertEqual(main_api.posts, 1)

    def test_missing_api_key_refuses_before_the_first_reservation(self):
        """Catches mutant: a missing key is recorded as an ambiguous POST after reserving."""
        units = tiny_units()
        labels, summaries = tiny_summaries(units)
        trials = judge65.build_trials(units, labels, summaries, seed=76)[:2]

        class CountingLedger(FakeLedger):
            reserve_calls = 0

            def reserve(self, *args, **kwargs):
                type(self).reserve_calls += 1
                return super().reserve(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root, ledger_path = Path(directory) / "judge", Path(directory) / "ledger.jsonl"
            judge65.prepare_run(trials, root, chunk_size=2, seed=76)
            CountingLedger.reserve_calls = 0
            with mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch.object(judge65, "Ledger", CountingLedger):
                with self.assertRaisesRegex(judge65.JudgeError, "before any paidaction reservation"):
                    judge65.run_chunks(root, ledger_path)
            self.assertEqual(CountingLedger.reserve_calls, 0)
            self.assertFalse(ledger_path.exists())

    def test_run_chunks_rejects_missing_or_incomplete_chunk_files_before_reserving(self):
        """Catches mutant: recovery silently skips a missing chunk or accepts a short trial sum."""
        units = tiny_units()
        labels, summaries = tiny_summaries(units)
        trials = judge65.build_trials(units, labels, summaries, seed=77)[:4]
        for defect in ("missing", "short-trials"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                root, ledger_path = Path(directory) / "judge", Path(directory) / "ledger.jsonl"
                judge65.prepare_run(trials, root, chunk_size=2, seed=77)
                if defect == "missing":
                    (root / "chunks" / "chunk-0001.json").unlink()
                    message = "chunk files must be exactly indices 0..1"
                else:
                    path = root / "chunks" / "chunk-0000.json"
                    state = json.loads(path.read_text())
                    state["trials"].pop()
                    path.write_text(json.dumps(state), encoding="utf-8")
                    message = "chunk trials sum to 3, manifest request_count is 4"
                api = FakeBatchAPI()
                with mock.patch.object(judge65, "Ledger", FakeLedger):
                    with self.assertRaisesRegex(judge65.JudgeError, message):
                        judge65.run_chunks(root, ledger_path, api)
                self.assertEqual(api.posts, 0)
                self.assertFalse(ledger_path.exists())

    def test_same_run_terminal_release_settlement_and_retirement_attempts_never_resubmit(self):
        """Catches mutant: a released, settled, or retired chunk is treated as sendable."""
        units = tiny_units()
        labels, summaries = tiny_summaries(units)
        trials = judge65.build_trials(units, labels, summaries, seed=71)[:2]
        for terminal in ("release", "settle", "retire"):
            with self.subTest(terminal=terminal), tempfile.TemporaryDirectory() as directory:
                root, ledger_path = Path(directory) / "judge", Path(directory) / "ledger.jsonl"
                judge65.prepare_run(trials, root, chunk_size=2, seed=71)
                state = json.loads((root / "chunks" / "chunk-0000.json").read_text())
                manifest = json.loads((root / "run.json").read_text())
                ledger = FakeLedger(str(ledger_path), HARD_CAP_USD)
                attempt_id = ledger.reserve(
                    state["reserve_usd"], "part65-judge",
                    {"plan_sha256": manifest["plan_sha256"],
                     "judge_run_id": manifest["judge_run_id"], "run_root": manifest["run_root"],
                     "chunk_index": 0, "request_count": len(state["trials"])},
                )
                if terminal == "release":
                    ledger.release(attempt_id, 400,
                                   {"type": "error", "error": {"type": "invalid_request_error"}})
                elif terminal == "settle":
                    ledger.settle(attempt_id, 0.0, {"input_tokens": 0, "output_tokens": 0})
                else:
                    ledger.retire(attempt_id)
                api = FakeBatchAPI()
                with mock.patch.object(judge65, "Ledger", FakeLedger):
                    with self.assertRaisesRegex(judge65.JudgeError, "already has a"):
                        judge65.run_chunks(root, ledger_path, api)
                self.assertEqual(api.posts, 0)

    def test_recover_refuses_ready_chunk_without_posting(self):
        """Catches mutant: --recover can POST a ready chunk after a crash."""
        units = tiny_units()
        labels, summaries = tiny_summaries(units)
        trials = judge65.build_trials(units, labels, summaries, seed=72)[:2]
        with tempfile.TemporaryDirectory() as directory:
            root, ledger_path = Path(directory) / "judge", Path(directory) / "ledger.jsonl"
            judge65.prepare_run(trials, root, chunk_size=2, seed=72)
            api = FakeBatchAPI()
            with mock.patch.object(judge65, "Ledger", FakeLedger):
                with self.assertRaisesRegex(judge65.JudgeError, "GET-only"):
                    judge65.run_chunks(root, ledger_path, api, recover_only=True)
            err = io.StringIO()
            with mock.patch.object(judge65, "run_chunks",
                                   side_effect=judge65.JudgeError("GET-only recovery refusal")), \
                 redirect_stderr(err), redirect_stdout(io.StringIO()):
                code = judge65.main(["--out-dir", str(root), "--recover"])
            self.assertEqual(code, 2)
            self.assertIn("GET-only recovery refusal", err.getvalue())
            self.assertEqual(api.posts, 0)

    def test_ambiguous_posting_marker_is_never_reposted(self):
        """Catches mutant: a durable posting marker with no batch ID is treated as ready."""
        units = tiny_units()
        labels, summaries = tiny_summaries(units)
        trials = judge65.build_trials(units, labels, summaries, seed=73)[:2]
        with tempfile.TemporaryDirectory() as directory:
            root, ledger_path = Path(directory) / "judge", Path(directory) / "ledger.jsonl"
            judge65.prepare_run(trials, root, chunk_size=2, seed=73)
            chunk_path = root / "chunks" / "chunk-0000.json"
            state = json.loads(chunk_path.read_text())
            manifest = json.loads((root / "run.json").read_text())
            ledger = FakeLedger(str(ledger_path), HARD_CAP_USD)
            state["attempt_id"] = ledger.reserve(
                state["reserve_usd"], "part65-judge",
                {"plan_sha256": manifest["plan_sha256"],
                 "judge_run_id": manifest["judge_run_id"],
                 "run_root": manifest["run_root"], "chunk_index": 0,
                 "request_count": len(state["trials"])},
            )
            state["status"] = "posting"
            chunk_path.write_text(json.dumps(state))
            api = FakeBatchAPI()
            with mock.patch.object(judge65, "Ledger", FakeLedger):
                totals = []
                for recover_only in (True, False):
                    totals.append(judge65.run_chunks(
                        root, ledger_path, api, recover_only=recover_only
                    ))
            self.assertEqual([row["ambiguous"] for row in totals], [1, 1])
            self.assertEqual(api.posts, 0)

    def test_crash_after_provider_acceptance_does_not_post_again(self):
        """Catches mutant: the durable posting marker is removed before POST."""
        units = tiny_units()
        labels, summaries = tiny_summaries(units)
        trials = judge65.build_trials(units, labels, summaries, seed=82)[:2]
        with tempfile.TemporaryDirectory() as directory:
            root, ledger_path = Path(directory) / "judge", Path(directory) / "ledger.jsonl"
            judge65.prepare_run(trials, root, chunk_size=2, seed=82)
            api = FakeBatchAPI()
            real_save = judge65._save_chunk

            def fail_submitted_save(path, state):
                if state.get("status") == "submitted":
                    raise RuntimeError("simulated crash saving submitted state")
                real_save(path, state)

            with mock.patch.object(judge65, "Ledger", FakeLedger):
                with mock.patch.object(judge65, "_save_chunk", side_effect=fail_submitted_save):
                    with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                        judge65.run_chunks(root, ledger_path, api, poll_seconds=0,
                                           timeout_seconds=1)
                second = judge65.run_chunks(root, ledger_path, api, poll_seconds=0,
                                            timeout_seconds=1)
            self.assertEqual(api.posts, 1)
            self.assertEqual(second["ambiguous"], 1)
            chunk_path = root / "chunks" / "chunk-0000.json"
            self.assertEqual(json.loads(chunk_path.read_text())["status"], "posting")

    def test_atomic_json_fsyncs_temp_then_replacement_directory(self):
        """Catches mutant: local posting state is replaced without file and directory fsyncs."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            events = []
            real_replace = os.replace

            def record_replace(source, target):
                events.append("replace")
                return real_replace(source, target)

            with mock.patch.object(judge65.os, "fsync", side_effect=lambda _fd: events.append("fsync")), \
                 mock.patch.object(judge65.os, "replace", side_effect=record_replace):
                judge65._atomic_json(path, {"durable": True})
            self.assertEqual(events, ["fsync", "replace", "fsync"])
            self.assertEqual(json.loads(path.read_text()), {"durable": True})

    def test_reserve_recovery_reuses_paidaction_attempt_after_state_write_crash(self):
        units = tiny_units()
        labels, summaries = tiny_summaries(units)
        trials = judge65.build_trials(units, labels, summaries, seed=13)[:2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "judge"
            ledger_path = Path(directory) / "shared-paid-ledger.jsonl"
            judge65.prepare_run(trials, root, chunk_size=2, seed=13)
            chunk_path = root / "chunks" / "chunk-0000.json"
            chunk = json.loads(chunk_path.read_text())
            manifest = json.loads((root / "run.json").read_text())
            plan_hash = manifest["plan_sha256"]
            ledger = FakeLedger(str(ledger_path), HARD_CAP_USD)
            ledger.reserve(
                chunk["reserve_usd"], "part65-judge",
                {"plan_sha256": plan_hash, "judge_run_id": manifest["judge_run_id"],
                 "run_root": manifest["run_root"], "chunk_index": 0, "request_count": 2},
                subcap_usd=judge65.PART65_CAP_USD, subcap_key="part65-judge",
            )
            fake = FakeBatchAPI()
            with mock.patch.object(judge65, "Ledger", FakeLedger):
                totals = judge65.run_chunks(root, ledger_path, fake, poll_seconds=0,
                                            timeout_seconds=1, sleep=lambda _seconds: None)
            self.assertEqual(totals["settled"], 1)
            self.assertEqual(fake.posts, 1)
            events = [json.loads(line) for line in ledger_path.read_text().splitlines()]
            self.assertEqual(sum(row["event"] == "reserve" for row in events), 1)

    def test_paid_batch_recovery_resumes_known_batch_without_second_post(self):
        units = tiny_units()
        labels, summaries = tiny_summaries(units)
        trials = judge65.build_trials(units, labels, summaries, seed=11)[:2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "judge"
            ledger_path = Path(directory) / "shared-paid-ledger.jsonl"
            judge65.prepare_run(trials, root, chunk_size=2, seed=11)
            fake = FakeBatchAPI(pending_checks=1)
            with mock.patch.object(judge65, "Ledger", FakeLedger):
                first = judge65.run_chunks(root, ledger_path, fake, poll_seconds=0,
                                           timeout_seconds=0, sleep=lambda _seconds: None)
            self.assertEqual(first["pending"], 1)
            self.assertEqual(fake.posts, 1)
            self.assertNotIn("completion", json.loads((root / "run.json").read_text()))
            self.assertGreater(FakeLedger(str(ledger_path), HARD_CAP_USD).state()["reserved"], 0.0)
            with mock.patch.object(judge65, "Ledger", FakeLedger):
                second = judge65.run_chunks(root, ledger_path, fake, poll_seconds=0,
                                            timeout_seconds=1, sleep=lambda _seconds: None,
                                            recover_only=True)
            self.assertEqual(second["settled"], 1)
            self.assertEqual(fake.posts, 1)
            self.assertEqual(json.loads((root / "run.json").read_text())[
                "completion"]["status"], "complete")
            self.assertEqual(FakeLedger(str(ledger_path), HARD_CAP_USD).state()["reserved"], 0.0)

    def test_pending_chunk_stops_before_later_ready_chunks(self):
        """Catches mutant: a pending batch does not stop the chunk loop."""
        units = tiny_units()
        labels, summaries = tiny_summaries(units)
        trials = judge65.build_trials(units, labels, summaries, seed=83)[:3]
        with tempfile.TemporaryDirectory() as directory:
            root, ledger_path = Path(directory) / "judge", Path(directory) / "ledger.jsonl"
            judge65.prepare_run(trials, root, chunk_size=1, seed=83)
            api = FakeBatchAPI(pending_checks=100)
            with mock.patch.object(judge65, "Ledger", FakeLedger):
                totals = judge65.run_chunks(root, ledger_path, api, poll_seconds=0,
                                            timeout_seconds=0, sleep=lambda _seconds: None)
            self.assertEqual(totals["pending"], 1)
            self.assertEqual(api.posts, 1)
            states = [json.loads(path.read_text()) for path in
                      sorted((root / "chunks").glob("chunk-*.json"))]
            self.assertEqual([state["status"] for state in states],
                             ["submitted", "ready", "ready"])

    def test_probe_plan_contains_exactly_one_message_request(self):
        units = tiny_units()
        labels, summaries = tiny_summaries(units)
        trials = judge65.build_trials(units, labels, summaries, seed=12)
        self.assertEqual(len(judge65.build_probe_trial(trials)), 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "probe"
            judge65.prepare_run(trials, root, chunk_size=1, seed=12, probe=True)
            manifest = json.loads((root / "run.json").read_text())
            self.assertEqual(manifest["request_count"], 1)
            chunk = json.loads((root / "chunks" / "chunk-0000.json").read_text())
            self.assertEqual(len(chunk["trials"]), 1)

    def test_analysis_reproduces_planted_effect_and_planted_null(self):
        answers, units, labels, write_words = planted_analysis(effect=True)
        effect = analyze65.analyze_rows(
            answers, units, labels, write_words, seed=2,
            bootstrap_reps=300, shuffle_reps=20,
        )
        self.assertTrue(effect["predictions"]["65a"]["pass"])
        self.assertTrue(effect["predictions"]["65b"]["pass"])
        self.assertTrue(effect["predictions"]["65c"]["pass"])
        self.assertTrue(effect["predictions"]["65d"]["pass"])
        self.assertTrue(effect["predictions"]["65e"]["pass"])
        self.assertTrue(effect["predictions"]["65f"]["pass"])
        self.assertEqual(effect["exploratory_predictions"], ["65c", "65g"])
        self.assertTrue(effect["predictions"]["65c"]["exploratory"])
        self.assertTrue(effect["predictions"]["65g"]["exploratory"])
        self.assertEqual(set(effect["predictions"]["65g"]["band_agreement"]),
                         {"first", "middle", "last"})
        self.assertIn("paired difference", effect["predictions"]["65f"]["measure_used"])
        self.assertIn("null_adjusted_neuron_minus_control", effect["embedding_cosine"])

        answers, units, labels, write_words = planted_analysis(effect=False)
        null = analyze65.analyze_rows(
            answers, units, labels, write_words, seed=3,
            bootstrap_reps=300, shuffle_reps=20,
        )
        self.assertAlmostEqual(null["predictions"]["65a"]["agreement"]["estimate"], 0.5)
        self.assertAlmostEqual(null["predictions"]["65b"]["agreement_difference_neuron_minus_control"]["estimate"], 0.0)
        self.assertEqual(set(null["predictions"]), {"65a", "65b", "65c", "65d", "65e", "65f", "65g"})
        self.assertTrue(all(not row["pass"] for row in null["predictions"].values()))

    def test_analysis_rejects_mixed_or_missing_run_provenance_even_for_disjoint_rows(self):
        """Catches mutant: disjoint answer rows from different runs pass because pair keys do not overlap."""
        answers, units, labels, write_words = planted_analysis(effect=True)
        answers[2]["judge_run_id"] = "a-different-run"
        with self.assertRaisesRegex(analyze65.AnalysisError, "more than one run or plan"):
            analyze65.analyze_rows(answers, units, labels, write_words,
                                   bootstrap_reps=10, shuffle_reps=2)

        answers, units, labels, write_words = planted_analysis(effect=True)
        del answers[0]["plan_sha256"]
        with self.assertRaisesRegex(analyze65.AnalysisError, "lacks run ID, plan hash"):
            analyze65.analyze_rows(answers, units, labels, write_words,
                                   bootstrap_reps=10, shuffle_reps=2)

    def test_empty_answers_reach_registered_missing_trial_stop(self):
        """Catches mutant: an empty answers file fails provenance first or skips the registered stop."""
        _answers, units, labels, write_words = planted_analysis(effect=False)
        with self.assertRaisesRegex(analyze65.AnalysisError,
                                    r"1000/1000 registered judge trials are missing"):
            analyze65.analyze_rows([], units, labels, write_words,
                                   bootstrap_reps=10, shuffle_reps=2)

    def test_analyze_drops_unparseable_trials_and_counts_units_without_valid_trials(self):
        answers, units, labels, write_words = planted_analysis(effect=False)
        answers[0]["correct"] = None
        answers[1]["correct"] = None
        result = analyze65.analyze_rows(
            answers, units, labels, write_words, seed=4,
            bootstrap_reps=100, shuffle_reps=10,
        )
        self.assertEqual(result["trial_count"], 1000)
        self.assertEqual(result["unparseable_trial_count"], 2)
        self.assertEqual(result["unit_without_valid_trial_count"], 1)
        self.assertEqual(result["unit_count"], 499)
        self.assertNotIn(answers[0]["unit_key"], {row["unit_key"] for row in result["units"]})

    def test_chat_template_uses_non_thinking_mode(self):
        tokenizer = TinyTokenizer()
        steer65._chat(tokenizer, "Who are you?")
        self.assertIs(tokenizer.last_template_kwargs.get("enable_thinking"), False)

    def test_layers_0_2_sensitivity_leaves_out_early_units_only(self):
        answers, units, labels, write_words = planted_analysis(effect=True)
        result = analyze65.analyze_rows(
            answers, units, labels, write_words, seed=5,
            bootstrap_reps=100, shuffle_reps=10,
        )
        late = result["sensitivity_layers_0_2_left_out"]
        rows = result["units"]
        self.assertEqual(late["neuron_count"],
                         sum(1 for row in rows if row["type"] == "neuron" and row["layer"] > 2))
        self.assertEqual(late["control_count"],
                         sum(1 for row in rows if row["type"] == "control" and row["layer"] > 2))
        self.assertLess(late["neuron_count"], sum(1 for row in rows if row["type"] == "neuron"))
        self.assertNotIn("pass", late)
        self.assertIn("layers 0 to 2 left out", analyze65.render_markdown(result))

    def test_one_valid_trial_excludes_and_counts_the_unit(self):
        answers, units, labels, write_words = planted_analysis(effect=False)
        answers[0]["correct"] = None
        result = analyze65.analyze_rows(
            answers, units, labels, write_words, seed=14,
            bootstrap_reps=100, shuffle_reps=10,
        )
        excluded_key = answers[0]["unit_key"]
        self.assertEqual(result["unit_count"], 499)
        self.assertNotIn(excluded_key, {row["unit_key"] for row in result["units"]})
        self.assertEqual(result["excluded_unit_count"], 1)
        self.assertEqual(result["excluded_units_by_missing_trial_count"], {"one": 1, "two": 0})
        band = rank65.layer_band(next(unit["layer"] for unit in units
                                      if unit["unit_key"] == excluded_key))
        self.assertEqual(result["excluded_units_by_type_and_layer_band"]["neuron"][band], 1)
        self.assertEqual(result["unit_without_valid_trial_count"], 0)
        self.assertAlmostEqual(result["predictions"]["65a"]["agreement"]["estimate"], 0.5)

    def test_absent_registered_pair_counts_against_the_1000_trial_plan(self):
        answers, units, labels, write_words = planted_analysis(effect=False)
        answers.pop(0)
        result = analyze65.analyze_rows(
            answers, units, labels, write_words, seed=15,
            bootstrap_reps=100, shuffle_reps=10,
        )
        self.assertEqual(len(answers), 999)
        self.assertEqual(result["trial_count"], 1000)
        self.assertEqual(result["missing_trial_count"], 1)
        self.assertEqual(result["absent_trial_count"], 1)
        self.assertEqual(result["unparseable_trial_count"], 0)
        self.assertEqual(result["excluded_unit_count"], 1)

    def test_missing_trial_stop_uses_the_registered_50_of_1000_boundary(self):
        answers, units, labels, write_words = planted_analysis(effect=False)
        for row in answers[:50]:
            row["correct"] = None
        result = analyze65.analyze_rows(
            answers, units, labels, write_words, seed=16,
            bootstrap_reps=100, shuffle_reps=10,
        )
        self.assertEqual(result["missing_trial_count"], 50)
        self.assertEqual(result["trial_count"], 1000)

        answers[50]["correct"] = None
        with self.assertRaisesRegex(
                analyze65.AnalysisError,
                r"51/1000 registered judge trials are missing"):
            analyze65.analyze_rows(
                answers, units, labels, write_words, seed=17,
                bootstrap_reps=100, shuffle_reps=10,
            )

    def test_missing_trial_bounds_score_all_units_and_leave_primary_complete_case(self):
        answers, units, labels, write_words = planted_analysis(effect=False)
        answers[0]["correct"] = None
        answers.pop(2)
        result = analyze65.analyze_rows(
            answers, units, labels, write_words, seed=18,
            bootstrap_reps=100, shuffle_reps=10,
        )
        self.assertAlmostEqual(result["predictions"]["65a"]["agreement"]["estimate"], 0.5)
        self.assertEqual(result["unit_count"], 498)
        chance = result["missing_trial_bounds"]["missing_trials_scored_0_5"]
        wrong = result["missing_trial_bounds"]["missing_trials_scored_wrong"]
        self.assertAlmostEqual(chance["65a"]["estimate"], 0.498)
        self.assertAlmostEqual(wrong["65a"]["estimate"], 0.496)
        self.assertAlmostEqual(chance["65b"]["estimate"], -0.002)
        self.assertAlmostEqual(wrong["65b"]["estimate"], -0.004)
        for scenario in (chance, wrong):
            self.assertEqual(scenario["65a"]["n"], 250)
            self.assertEqual(scenario["65b"]["n_neuron"], 250)
            self.assertEqual(scenario["65b"]["n_control"], 250)
            self.assertEqual(set(scenario["neuron_agreement_by_layer_band"]),
                             {"first", "middle", "last"})
            self.assertTrue(all(metric["n"] > 0 for metric in
                                scenario["neuron_agreement_by_layer_band"].values()))
        rendered = analyze65.render_markdown(result)
        self.assertIn("| 0.5 (chance) |", rendered)
        self.assertIn("| 0 (wrong) |", rendered)
        self.assertIn("## Missing-trial handling and bounds", rendered)

        complete_answers, complete_units, complete_labels, complete_words = planted_analysis(effect=True)
        complete = analyze65.analyze_rows(
            complete_answers, complete_units, complete_labels, complete_words, seed=19,
            bootstrap_reps=100, shuffle_reps=10,
        )
        incomplete_answers = [dict(row) for row in complete_answers]
        incomplete_answers[0]["correct"] = None
        incomplete_answers.pop(2)
        incomplete = analyze65.analyze_rows(
            incomplete_answers, complete_units, complete_labels, complete_words, seed=19,
            bootstrap_reps=100, shuffle_reps=10,
        )
        self.assertEqual(
            {key: prediction["pass"] for key, prediction in complete["predictions"].items()},
            {key: prediction["pass"] for key, prediction in incomplete["predictions"].items()},
        )

    def test_analysis_reader_rejects_a_torn_last_answer_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "answers.jsonl"
            path.write_text('{"unit_key":"ok","trial":1}\n{"unit_key":',
                            encoding="utf-8")
            with self.assertRaisesRegex(analyze65.AnalysisError, "invalid JSON"):
                analyze65.read_jsonl(path)
            with self.assertRaisesRegex(analyze65.AnalysisError, "invalid JSON"):
                analyze65.analyze_files(
                    path, Path("missing-collection.jsonl"), Path("missing-neuron.jsonl"),
                    Path("missing-control.jsonl"), Path("missing-ranked.jsonl"),
                )

    def test_file_analysis_requires_a_completion_marker_matching_answer_identity(self):
        """Catches mutant: analysis accepts missing, failed, or mismatched completion evidence."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answers = root / "answers.jsonl"
            answer = {"judge_run_id": "run-a", "plan_sha256": "plan-a", "custom_id": "p65-0000"}
            answers.write_text(json.dumps(answer) + "\n", encoding="utf-8")
            manifest = {
                "judge_run_id": "run-a",
                "plan_sha256": "plan-a",
                "chunk_count": 1,
            }
            manifest_path = root / "run.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(analyze65.AnalysisError, "no completion marker"):
                analyze65.analyze_files(
                    answers, Path("missing-collection.jsonl"), Path("missing-neuron.jsonl"),
                    Path("missing-control.jsonl"), Path("missing-ranked.jsonl"),
                )

            manifest["completion"] = {
                "status": "complete", "judge_run_id": "run-b",
                "plan_sha256": "plan-a", "chunk_count": 1,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(analyze65.AnalysisError, "does not match answers"):
                analyze65.analyze_files(
                    answers, Path("missing-collection.jsonl"), Path("missing-neuron.jsonl"),
                    Path("missing-control.jsonl"), Path("missing-ranked.jsonl"),
                )

            manifest["completion"]["judge_run_id"] = "run-a"
            manifest["failure"] = {"status": "failed", "outcome": "released"}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(analyze65.AnalysisError, "no completion marker"):
                analyze65.analyze_files(
                    answers, Path("missing-collection.jsonl"), Path("missing-neuron.jsonl"),
                    Path("missing-control.jsonl"), Path("missing-ranked.jsonl"),
                )

            manifest.pop("failure")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                analyze65.analyze_files(
                    answers, Path("missing-collection.jsonl"), Path("missing-neuron.jsonl"),
                    Path("missing-control.jsonl"), Path("missing-ranked.jsonl"),
                )

    def test_analysis_stops_above_five_percent_unparseable_trials(self):
        answers, units, labels, write_words = planted_analysis(effect=False)
        for row in answers[:51]:
            row["correct"] = None
        with self.assertRaisesRegex(analyze65.AnalysisError, "above the registered 5% limit"):
            analyze65.analyze_rows(
                answers, units, labels, write_words, seed=5,
                bootstrap_reps=100, shuffle_reps=10,
            )

    def test_read_jsonl_ignores_and_warns_for_invalid_final_line_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "torn.jsonl"
            path.write_text('{"ok":true}\n{"torn":\n\n', encoding="utf-8")
            for reader, error_type in ((steer65.read_jsonl, steer65.SteerError),
                                       (judge65.read_jsonl, judge65.JudgeError)):
                warnings = io.StringIO()
                with redirect_stderr(warnings):
                    self.assertEqual(reader(path), [{"ok": True}])
                self.assertIn(path.name, warnings.getvalue())
                path.write_text('{broken\n{}\n', encoding="utf-8")
                with self.assertRaises(error_type):
                    reader(path)
                path.write_text('{"ok":true}\n{"torn":\n\n', encoding="utf-8")

    def test_judge_accepts_only_a_lone_letter_with_optional_punctuation(self):
        def provider(text):
            return {
                "result": {"type": "succeeded", "message": {
                    "content": [{"type": "text", "text": text}],
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                }},
            }
        for text, expected in ((" A ", "A"), ("B.", "B"), (" a)\n", "A")):
            answer, usage, error = judge65._extract_response(provider(text))
            self.assertEqual((answer, error), (expected, None))
            self.assertEqual(usage["input_tokens"], 3)
        for text in ("Answer: B", "AB", "A..) "):
            answer, _usage, error = judge65._extract_response(provider(text))
            self.assertIsNone(answer)
            self.assertIsNotNone(error)

    def test_negative_steer_lens_summary_says_more_likely(self):
        row = {"write_words": ["amber"], "steer_lens": {
            "positive": ["gold"], "negative": ["blue"],
        }}
        rendered = judge65.format_write_summary(row)
        self.assertIn("Tokens made more likely by a negative push: blue", rendered)
        self.assertNotIn("less likely", rendered)

    def test_judge_api_passes_ssl_context_and_prefers_credit_key(self):
        context = object()
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"{}"
        with mock.patch.dict("os.environ", {
                "ANTHROPIC_CREDITS_KEY": "credit-key",
                "ANTHROPIC_API_KEY": "fallback-key",
        }, clear=True), \
             mock.patch.object(judge65.ssl, "create_default_context", return_value=context), \
             mock.patch.object(judge65.urllib.request, "urlopen", return_value=response) as urlopen:
            self.assertEqual(judge65.api_call("GET", "/probe"), "{}")
        self.assertIs(urlopen.call_args.kwargs["context"], context)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("X-api-key"), "credit-key")

    def test_judge_main_fails_and_prints_released_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunk_dir = root / "chunks"
            chunk_dir.mkdir()
            path = chunk_dir / "chunk-0002.json"
            path.write_text(json.dumps({"status": "released", "error": "HTTP 400 bad request"}),
                            encoding="utf-8")
            with mock.patch.object(judge65, "run_chunks", return_value={
                    "settled": 0, "pending": 0, "released": 1, "ambiguous": 0}), \
                 redirect_stderr(io.StringIO()) as output:
                code = judge65.main(["--out-dir", str(root), "--recover"])
            self.assertEqual(code, 2)
            self.assertIn(str(path), output.getvalue())
            self.assertIn("HTTP 400 bad request", output.getvalue())
            self.assertIn("No new output directory may be started for this plan", output.getvalue())

    def test_released_chunk_is_final_and_does_not_reserve_again(self):
        with tempfile.TemporaryDirectory() as directory:
            units = tiny_units()
            labels, summaries = tiny_summaries(units)
            trials = judge65.build_trials(units, labels, summaries, seed=74)[:2]
            root = Path(directory) / "judge"
            judge65.prepare_run(trials, root, chunk_size=2, seed=74)
            path = root / "chunks" / "chunk-0000.json"
            state = json.loads(path.read_text())
            state.update({"status": "released", "error": "HTTP 400 rejection"})
            path.write_text(json.dumps(state), encoding="utf-8")
            ledger = mock.MagicMock()
            api = mock.MagicMock()
            self.assertEqual(judge65.process_chunk(path, ledger, api), "released")
            ledger.reserve.assert_not_called()
            api.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
