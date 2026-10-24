#!/usr/bin/env python3
"""Offline Part 65 regression tests. Reads the five Part 65 modules, builds a tiny random-weight torch model and fake batch API, and writes only temporary fixtures that the tests remove."""
from __future__ import annotations

import json
from contextlib import contextmanager, redirect_stderr
import io
from pathlib import Path
import tempfile
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
                            add_generation_prompt: bool = True) -> str:
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
        attempt_id = "a" * 32
        self._append({"event": "reserve", "attempt_id": attempt_id,
                      "amount_usd": amount, "kind": kind, "meta": meta})
        return attempt_id

    def settle(self, attempt_id: str, amount: float, usage: dict[str, int]) -> None:
        self._append({"event": "settle", "attempt_id": attempt_id,
                      "amount_usd": amount, "usage": usage})

    def state(self) -> dict[str, float]:
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()] \
            if self.path.exists() else []
        settled = {row.get("attempt_id") for row in rows if row.get("event") == "settle"}
        reserved = sum(float(row.get("amount_usd", 0.0)) for row in rows
                       if row.get("event") == "reserve" and row.get("attempt_id") not in settled)
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
                second = judge65.run_chunks(root, ledger_path, fake, poll_seconds=0,
                                            timeout_seconds=1, sleep=lambda _seconds: None)
            self.assertEqual(second["settled"], 1)
            self.assertEqual(fake.posts, 1)
            rows = judge65.read_jsonl(root / "answers.jsonl")
            self.assertEqual(len(rows), 2)

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
            plan_hash = json.loads((root / "run.json").read_text())["plan_sha256"]
            ledger = FakeLedger(str(ledger_path), HARD_CAP_USD)
            ledger.reserve(
                chunk["reserve_usd"], "part65-judge",
                {"plan_sha256": plan_hash, "chunk_index": 0, "request_count": 2},
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
            self.assertGreater(FakeLedger(str(ledger_path), HARD_CAP_USD).state()["reserved"], 0.0)
            with mock.patch.object(judge65, "Ledger", FakeLedger):
                second = judge65.run_chunks(root, ledger_path, fake, poll_seconds=0,
                                            timeout_seconds=1, sleep=lambda _seconds: None)
            self.assertEqual(second["settled"], 1)
            self.assertEqual(fake.posts, 1)
            self.assertEqual(FakeLedger(str(ledger_path), HARD_CAP_USD).state()["reserved"], 0.0)

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
            self.assertIn("rerun the whole judge into a new output directory", output.getvalue())

    def test_released_chunk_is_final_and_does_not_reserve_again(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunk_dir = root / "chunks"
            chunk_dir.mkdir()
            path = chunk_dir / "chunk-0000.json"
            path.write_text(json.dumps({"status": "released", "error": "HTTP 400 rejection"}),
                            encoding="utf-8")
            ledger = mock.MagicMock()
            api = mock.MagicMock()
            self.assertEqual(judge65.process_chunk(path, ledger, api), "released")
            ledger.reserve.assert_not_called()
            api.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
