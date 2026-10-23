"""CPU-only Part 62 tests, including red checks for deliberate live-code mutations."""
from __future__ import annotations

import json
import math
import os
import contextlib
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

try:
    from . import analyze62, blind62, run62
except ImportError:
    import analyze62
    import blind62
    import run62

import blind_origin


class FakeDownProjection:
    def __init__(self):
        self.hooks = []

    def register_forward_pre_hook(self, callback):
        self.hooks.append(callback)
        return types.SimpleNamespace(remove=lambda: self.hooks.remove(callback))

    def __call__(self, values):
        args = (values,)
        for callback in list(self.hooks):
            replacement = callback(self, args)
            if replacement is not None:
                args = replacement
        return args[0]


class StubModel:
    """CPU stub that drives the real NeuronHook through prefill and decode."""
    def __init__(self, down_proj):
        import torch
        self.down_proj = down_proj
        self.seen_inputs = []
        self.torch = torch

    def generate(self):
        prompt = self.torch.ones((1, 3, 2), dtype=self.torch.float32)
        generated = self.torch.full((1, 1, 2), 2.0, dtype=self.torch.float32)
        self.seen_inputs.append(self.down_proj(prompt).detach().clone())
        self.seen_inputs.append(self.down_proj(generated).detach().clone())
        return ["stub answer"]


class Part62ConstantsTests(unittest.TestCase):
    def test_registered_matrix_counts_and_scales_are_pinned(self):
        self.assertEqual(run62.DOSES, (0.25, 0.5, 0.75, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0))
        self.assertEqual(run62.PHASES, ("both", "gen", "prompt"))
        self.assertEqual(run62.QUESTIONS, ("Who made you?", "Where do you come from?",
                                          "Tell me about your origins."))
        self.assertEqual(run62.SAMPLES, 8)
        self.assertEqual(run62.CONTROL_DOSES, (1.5, 2.0))
        self.assertEqual(run62.CONTROL_COUNT, 1)
        self.assertEqual(run62.CONTROL_SEED, 62062)
        self.assertEqual(run62.ALIEN_FRAME_COUNT, 30)
        self.assertEqual(run62.MAX_NEW_TOKENS, 64)
        self.assertEqual(run62.TEMPERATURE, 0.7)
        self.assertEqual(run62.TOP_P, 0.95)
        self.assertEqual(run62.expected_cell_count(), 192)
        self.assertEqual(run62.expected_generation_count(), 1536)
        self.assertEqual(run62.MODELS["qwen35-4b"]["control_unit"], 8870)
        self.assertEqual(run62.MODELS["olmo2-1b"]["control_unit"], 7884)
        self.assertEqual(run62.MODELS["qwen35-4b"]["expected_width"], 9216)
        self.assertEqual(run62.MODELS["olmo2-1b"]["expected_width"], 8192)
        self.assertEqual(run62.MODELS["qwen35-4b"]["pile_maximum_source_dtype"], "fp32")
        self.assertEqual(run62.MODELS["olmo2-1b"]["pile_maximum_source_dtype"], "bf16")
        for model_name, model in run62.MODELS.items():
            rng = run62.np.random.default_rng(run62.CONTROL_SEED)
            pool = run62.np.setdiff1d(run62.np.arange(model["expected_width"]), [model["unit"]])
            selected = sorted(int(unit) for unit in rng.choice(pool, size=run62.CONTROL_COUNT, replace=False))
            self.assertEqual(selected, [model["control_unit"]], model_name)
        self.assertEqual(run62.QWEN_ALIEN_FRAME_MAX_REFERENCE, 24.69)
        self.assertEqual(run62.QWEN_ALIEN_FRAME_RELATIVE_TOLERANCE, 0.001)
        self.assertEqual(run62.validate_alien_frame_reference("qwen35-4b", 24.69), 0.0)
        self.assertIsNone(run62.validate_alien_frame_reference("olmo2-1b", 6.5))
        with self.assertRaisesRegex(ValueError, "registered 24.69"):
            run62.validate_alien_frame_reference("qwen35-4b", 25.0)
        self.assertEqual(analyze62.ANSWERS_PER_DOSE, 24)
        self.assertEqual(analyze62.PREDICTION_62A_QWEN_DOSES, (1.5, 2.0))
        self.assertEqual(analyze62.PREDICTION_62A_OLMO_DOSES, (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0))
        self.assertEqual(analyze62.PREDICTION_62E_DOSE, 0.5)
        self.assertGreaterEqual(blind62.MAX_TOKENS, 1024)
        self.assertEqual(blind62.READER_MODELS["haiku"], "claude-haiku-5-5")
        self.assertEqual(blind62.DEFAULT_READER, "haiku")
        self.assertEqual(blind62.READER_CAPS_USD, {"haiku": 4.0, "opus": 8.0})
        self.assertEqual(len(blind62.CAL), 34)
        self.assertEqual(blind62.EXPECTED_OUTPUT_TOKENS, 150)
        self.assertEqual(blind62.ACCEPTED_STOP_REASONS, ("end_turn", "stop_sequence"))
        self.assertEqual(blind62.CALIBRATION_ERROR_LIMIT, 2)
        self.assertEqual(analyze62.PREDICTION_62A_MAX_DIFFERENCE, 3)
        self.assertEqual(analyze62.PREDICTION_62B_MAX_OFFWORLD, 1)
        self.assertEqual(analyze62.PREDICTION_62C_MAX_WIDTH, 0.3)
        self.assertEqual(analyze62.PREDICTION_62D_MIN_WIDTH, 0.8)
        self.assertEqual(analyze62.PREDICTION_62E_MAX_OFFWORLD, 4)
        self.assertEqual(analyze62.PREDICTION_62F_MAX_OFFWORLD, 1)
        self.assertEqual((analyze62.PREDICTION_62G_MIN_ALIEN_UNITS,
                          analyze62.PREDICTION_62G_MAX_ALIEN_UNITS), (0.85, 1.2))

    def test_seed_is_stable_and_distinct_across_dose_phase_and_model(self):
        seed = run62.condition_seed(10, "qwen35-4b", 1.5, "gen", run62.QUESTIONS[0], "twin", 2742)
        self.assertEqual(seed, run62.condition_seed(10, "qwen35-4b", 1.5, "gen",
                                                    run62.QUESTIONS[0], "twin", 2742))
        self.assertNotEqual(seed, run62.condition_seed(10, "qwen35-4b", 2.0, "gen",
                                                       run62.QUESTIONS[0], "twin", 2742))
        self.assertNotEqual(seed, run62.condition_seed(10, "qwen35-4b", 1.5, "prompt",
                                                       run62.QUESTIONS[0], "twin", 2742))
        self.assertNotEqual(seed, run62.condition_seed(10, "olmo2-1b", 1.5, "gen",
                                                       run62.QUESTIONS[0], "twin", 2742))

    def test_registered_pile_maxima_read_from_part26_records(self):
        self.assertAlmostEqual(run62._pile_maximum("qwen35-4b", run62.PART26_DIR), 19.125, places=3)
        self.assertAlmostEqual(run62._pile_maximum("olmo2-1b", run62.PART26_DIR), 6.5, places=3)


class PhaseHookTests(unittest.TestCase):
    def _run_phase(self, phase):
        import torch
        import neuron_report
        down_proj = FakeDownProjection()
        model = StubModel(down_proj)
        hook = run62.make_phase_hook(down_proj, 0, 9.0, phase,
                                     hook_factory=neuron_report.NeuronHook,
                                     tensor_factory=torch.tensor, prompt_length=3)
        try:
            self.assertEqual(model.generate(), ["stub answer"])
            self.assertEqual(hook.phase, phase)
        finally:
            hook.clamp = None
            hook.remove()
        return model.seen_inputs

    def test_prompt_phase_changes_prompt_and_leaves_generated_token_untouched(self):
        seen = self._run_phase("prompt")
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0][0, :, 0].tolist(), [9.0, 9.0, 1.0])
        self.assertEqual(seen[0][0, :, 1].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(seen[1].tolist(), [[[2.0, 2.0]]])

    def test_generation_phase_leaves_prompt_untouched_and_changes_generated_token(self):
        seen = self._run_phase("gen")
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0][0, :, 0].tolist(), [1.0, 1.0, 9.0])
        self.assertEqual(seen[0][0, :, 1].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(seen[1].tolist(), [[[9.0, 2.0]]])

    def test_both_phase_changes_prompt_and_generated_token(self):
        seen = self._run_phase("both")
        self.assertEqual(seen[0][0, :, 0].tolist(), [9.0, 9.0, 9.0])
        self.assertEqual(seen[1].tolist(), [[[9.0, 2.0]]])

    def test_position_mask_has_registered_last_prompt_and_decode_boundary(self):
        self.assertEqual(run62.phase_position_mask("prompt", 3, 3, True), [True, True, False])
        self.assertEqual(run62.phase_position_mask("gen", 3, 3, True), [False, False, True])
        self.assertEqual(run62.phase_position_mask("both", 3, 3, True), [True, True, True])
        self.assertEqual(run62.phase_position_mask("prompt", 3, 1, False), [False])
        self.assertEqual(run62.phase_position_mask("gen", 3, 1, False), [True])
        self.assertEqual(run62.phase_position_mask("both", 3, 1, False), [True])


class WriterReaderTests(unittest.TestCase):
    @staticmethod
    def _write_stub_run(folder: Path) -> dict:
        control_units = {name: int(spec["control_unit"]) for name, spec in run62.MODELS.items()}
        manifest = {
            "schema": "part62-run-v1", "part": 62, "status": "complete",
            "base_seed": run62.DEFAULT_SEED,
            "doses": list(run62.DOSES), "phases": list(run62.PHASES),
            "questions": list(run62.QUESTIONS), "samples_per_cell": run62.SAMPLES,
            "control_count": run62.CONTROL_COUNT, "control_seed": run62.CONTROL_SEED,
            "expected_cell_count": run62.expected_cell_count(),
            "expected_generations": run62.expected_generation_count(),
            "generated_answers": run62.expected_generation_count(),
            "control_units": control_units, "planned_cells": [], "completed": [],
            "run_fingerprint": "f" * 64, "inputs": {}, "device": "cpu-stub",
            "cell_fingerprints": {}, "cell_file_hashes": {},
            "models": {},
        }
        serial = 0
        for model_name, model in run62.MODELS.items():
            meta = {"id": model["id"], "revision": model["revision"],
                    "layer": model["layer"], "target_unit": model["unit"],
                    "control_unit": control_units[model_name], "width": model["expected_width"],
                    "pile_maximum": 10.0, "alien_frame_maximum": 24.69,
                    "pile_maximum_source_dtype": model["pile_maximum_source_dtype"],
                    "alien_frame_count": run62.ALIEN_FRAME_COUNT,
                    "registered_alien_frame_reference": model.get("alien_frame_reference"),
                    "target_column_norm": 2.0, "control_column_norm": 1.0}
            manifest["models"][model_name] = meta
            for spec in run62.model_cell_specs(model_name, control_units[model_name]):
                parameters = run62.cell_parameters(run62.DEFAULT_SEED, spec, meta["pile_maximum"],
                                                   meta["target_column_norm"], meta["control_column_norm"])
                condition_name = "twin" if spec["kind"] == "twin" else f"control-{spec['unit']}"
                condition = {"condition": condition_name, "kind": spec["kind"],
                             "unit": spec["unit"], "seed": parameters["seed"], "phase": spec["phase"],
                             "set_value": parameters["set_value"], "n": run62.SAMPLES,
                             "target_equivalent_set_value_norm": parameters["target_equivalent_set_value_norm"],
                             "answers": [f"stub response {serial} {i}" for i in range(run62.SAMPLES)]}
                cell_fingerprint = run62._cell_fingerprint(
                    manifest["run_fingerprint"], spec, meta, parameters)
                path = run62.write_cell(folder, spec, condition, meta,
                                        manifest["run_fingerprint"], cell_fingerprint)
                manifest["planned_cells"].append({**spec, "file": path.name})
                manifest["completed"].append(path.name)
                manifest["cell_fingerprints"][path.name] = cell_fingerprint
                manifest["cell_file_hashes"][path.name] = run62._sha256(path)
                serial += 1
        run62._atomic_json(folder / "part62-manifest.json", manifest)
        return manifest

    def test_real_writer_outputs_round_trip_through_blind_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            manifest = self._write_stub_run(folder)
            rows = blind62.answers(folder)
            self.assertEqual(len(rows), 1536)
            self.assertEqual(sum(row["kind"] == "twin" for row in rows), 1440)
            self.assertEqual(sum(row["kind"] == "control" for row in rows), 96)
            self.assertEqual(len({row["seed"] for row in rows[::run62.SAMPLES]}), 192)
            self.assertEqual(len(manifest["planned_cells"]), 192)

            review_dir = folder / "review"
            review_dir.mkdir()
            shuffled = list(rows)
            import random
            random.Random(blind62.SHUFFLE_SEED).shuffle(shuffled)
            key = {f"ans|{number}": {name: value for name, value in row.items() if name != "text"}
                   for number, row in enumerate(shuffled)}
            labels = {answer_id: ["NEITHER", "OK"] for answer_id in key}
            for index, (label, quality, _text) in enumerate(blind62.CAL):
                labels[f"cal|{index}|{label}|{quality}"] = [label, quality]
            (review_dir / "part62-key-fixture.json").write_text(json.dumps(key), encoding="utf-8")
            labels_path = review_dir / "part62-labels-fixture.json"
            labels_path.write_text(json.dumps({"schema": "part62-blind-labels-v1",
                                               "batch_id": "fixture", "reader": "haiku",
                                               "model": "claude-haiku-5-5", "max_tokens": 1024,
                                               "scoring_allowed": True, "answer_count": len(key),
                                               "unlabelled_answers": [],
                                               "calibration": {"passed": True,
                                                               "error_limit": blind62.CALIBRATION_ERROR_LIMIT,
                                                               "wrong": 0, "wrong_ids": [],
                                                               "completion_issues": {}},
                                               "completion": {"passed": True, "issues": {},
                                                              "accepted_stop_reasons": list(blind62.ACCEPTED_STOP_REASONS)},
                                               "labels": labels}),
                                   encoding="utf-8")
            analysis = analyze62.analyze(folder, labels_path, review_dir)
            json.dumps(analysis, allow_nan=False)
            self.assertEqual(analysis["sample_info"]["answer_count"], 1536)
            self.assertEqual(analysis["predictions"]["status"]["62c"], "FAILED")

            victim = folder / manifest["completed"][0]
            data = json.loads(victim.read_text(encoding="utf-8"))
            data["clamp_mult"] = 999.0
            victim.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                blind62.answers(folder)


class ProductionWriterTests(unittest.TestCase):
    def test_run_model_uses_production_seed_and_set_value_and_resumes(self):
        model_name = "qwen35-4b"
        model_spec = run62.MODELS[model_name]
        target_cell = {"model": model_name, "dose": 0.5, "phase": "prompt",
                       "question": run62.QUESTIONS[0], "kind": "twin", "unit": model_spec["unit"]}
        control_cell = {"model": model_name, "dose": 1.5, "phase": "both",
                        "question": run62.QUESTIONS[1], "kind": "control",
                        "unit": model_spec["control_unit"]}
        cells = [target_cell, control_cell]
        down_proj = types.SimpleNamespace(in_features=model_spec["expected_width"])
        layers = [types.SimpleNamespace(mlp=types.SimpleNamespace(down_proj=down_proj))
                  for _ in range(model_spec["layer"] + 1)]
        nr = types.SimpleNamespace(find_text_layers=lambda _model: ("layers", layers, None),
                                   build_prompt=lambda _tok, question, _verbose: question)
        class FakeTokenized:
            def to(self, _device):
                return {"input_ids": object()}
        class FakeTokenizer:
            def __call__(self, *_args, **_kwargs):
                return FakeTokenized()
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            is_available=lambda: False, empty_cache=lambda: None))
        generated = []

        def fake_generate(_tok, _model, _down_proj, _prompt_ids, cell,
                          unit, set_value, seed, _device):
            generated.append((dict(cell), set_value, seed))
            return {"condition": "twin" if cell["kind"] == "twin" else f"control-{unit}",
                    "kind": cell["kind"], "unit": unit, "set_value": set_value,
                    "seed": seed, "phase": cell["phase"], "n": run62.SAMPLES,
                    "answers": ["stub"] * run62.SAMPLES}

        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            manifest = run62._new_manifest(10, "cuda", {"fixture": "input"}, "a" * 64)
            run62._write_manifest(out_dir, manifest)
            patches = (
                mock.patch.object(run62, "_experiment_modules", return_value=(fake_torch, nr, None)),
                mock.patch.object(run62, "_load_fp32", return_value=(FakeTokenizer(), object())),
                mock.patch.object(run62, "_scale_frames", return_value=(24.69, 30)),
                mock.patch.object(run62, "_column_norm", side_effect=lambda _down, unit: 2.0 if unit == model_spec["unit"] else 1.0),
                mock.patch.object(run62, "generate_condition", side_effect=fake_generate),
                mock.patch.object(run62, "model_cell_specs", return_value=cells),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                run62._run_model(model_name, out_dir, manifest, 10.0,
                                 Path("unused-alien-frames.json"), "cuda", 10)
                self.assertEqual(len(generated), 2)
                target_params = run62.cell_parameters(10, target_cell, 10.0, 2.0, 1.0)
                control_params = run62.cell_parameters(10, control_cell, 10.0, 2.0, 1.0)
                self.assertEqual(generated[0][1:], (target_params["set_value"], target_params["seed"]))
                self.assertEqual(generated[1][1:], (control_params["set_value"], control_params["seed"]))
                self.assertEqual(control_params["set_value"], 30.0)
                self.assertEqual(manifest["generated_answers"], 2 * run62.SAMPLES)

                # A cell atomically written before a kill but not yet entered in
                # the manifest is validated and reconciled without regeneration.
                recovered = manifest["completed"].pop()
                manifest["cell_fingerprints"].pop(recovered)
                manifest["cell_file_hashes"].pop(recovered)
                manifest["generated_answers"] -= run62.SAMPLES
                run62._write_manifest(out_dir, manifest)
                run62._run_model(model_name, out_dir, manifest, 10.0,
                                 Path("unused-alien-frames.json"), "cuda", 10)
                self.assertEqual(len(generated), 2)
                self.assertEqual(manifest["generated_answers"], 2 * run62.SAMPLES)

                original_path = out_dir / manifest["completed"][0]
                original_bytes = original_path.read_bytes()
                manifest["cell_fingerprints"][original_path.name] = "0" * 64
                with self.assertRaisesRegex(ValueError, "fingerprint changed"):
                    run62._run_model(model_name, out_dir, manifest, 10.0,
                                     Path("unused-alien-frames.json"), "cuda", 10)
                self.assertEqual(original_path.read_bytes(), original_bytes)

            with self.assertRaisesRegex(ValueError, "run_fingerprint"):
                run62._load_or_start_manifest(out_dir, 10, "cuda", {"fixture": "input"}, "b" * 64)


class BlindGateTests(unittest.TestCase):
    def test_429_is_ambiguous_while_definitive_rejections_are_distinguished(self):
        self.assertFalse(blind62._definitive_rejection(SystemExit("Anthropic API HTTP 429 Too Many Requests")))
        self.assertFalse(blind62._definitive_rejection(SystemExit("Anthropic API HTTP 500 Server Error")))
        self.assertFalse(blind62._definitive_rejection(SystemExit("Anthropic API HTTP 400 Bad Request")))
        error = '{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}'
        self.assertTrue(blind62._definitive_rejection(SystemExit(f"Anthropic API HTTP 400 Bad Request: {error}")))

    def test_api_request_builder_keeps_condition_metadata_local(self):
        row = {"model": "qwen35-4b", "question": "Where do you come from?",
               "dose": 1.5, "phase": "gen", "kind": "twin", "cond": "twin",
               "unit": 2742, "seed": 123, "i": 0, "text": "A plain stub answer."}
        idmap, key, requests = blind62.prepare_batch([row])
        self.assertEqual(idmap["r00000"], "ans|0")
        self.assertEqual(key["ans|0"]["phase"], "gen")
        self.assertEqual(requests[0]["params"]["max_tokens"], 1024)
        self.assertEqual(requests[0]["params"]["messages"], [
            {"role": "user", "content": blind62.RUBRIC + "A plain stub answer."}])
        self.assertEqual(requests[0]["params"]["model"], "claude-haiku-5-5")
        self.assertEqual(len(requests), 1 + 34)
        opus = blind62.prepare_batch([row], reader="opus")[2]
        self.assertEqual(opus[0]["params"]["model"], "claude-opus-5-5")

    def test_origin_calibration_gate_allows_two_and_rejects_three_errors(self):
        labels = {}
        for index, (label, quality, _text) in enumerate(blind62.CAL):
            labels[f"cal|{index}|{label}|{quality}"] = [label, quality]
        labels["ans|0"] = ["NEITHER", "OK"]
        self.assertEqual(blind62.scoring_gate(labels, ["ans|0"])["unlabelled_answers"], [])
        for index, (label, quality, _text) in enumerate(blind62.CAL[:2]):
            labels[f"cal|{index}|{label}|{quality}"] = ["NEITHER", "BROKEN"]
        self.assertEqual(len(blind62.scoring_gate(labels, ["ans|0"])["calibration_wrong"]), 2)
        index, (label, quality, _text) = 2, blind62.CAL[2]
        labels[f"cal|{index}|{label}|{quality}"] = ["NEITHER", "BROKEN"]
        with self.assertRaisesRegex(ValueError, "calibration gate failed"):
            blind62.scoring_gate(labels, ["ans|0"])

    def test_capped_reader_response_blocks_scoring_even_when_labels_parse(self):
        labels = {}
        for index, (label, quality, _text) in enumerate(blind62.CAL):
            labels[f"cal|{index}|{label}|{quality}"] = [label, quality]
        labels["ans|0"] = ["OFFWORLD", "OK"]
        calibration_id = f"cal|0|{blind62.CAL[0][0]}|{blind62.CAL[0][1]}"
        with self.assertRaisesRegex(ValueError, "truncated or unverified"):
            blind62.scoring_gate(labels, ["ans|0"], {calibration_id: "max_tokens"})

    def test_batch_result_parser_keeps_stop_reason_for_calibration_and_answers(self):
        idmap = {"r00000": "ans|0", "r00001": "cal|0|EARTH|OK"}
        lines = [
            json.dumps({"custom_id": "r00000", "result": {"type": "succeeded", "message": {
                "stop_reason": "end_turn", "content": [{"type": "text", "text": "OFFWORLD OK"}]}}}),
            json.dumps({"custom_id": "r00001", "result": {"type": "succeeded", "message": {
                "stop_reason": "max_tokens", "content": [{"type": "text", "text": "EARTH OK"}]}}}),
        ]
        labels, reasons = blind62.parse_batch_results(lines, idmap)
        self.assertEqual(labels["ans|0"], ["OFFWORLD", "OK"])
        self.assertEqual(reasons["ans|0"], "end_turn")
        self.assertEqual(labels["cal|0|EARTH|OK"], ["EARTH", "OK"])
        self.assertEqual(reasons["cal|0|EARTH|OK"], "max_tokens")
        with self.assertRaisesRegex(ValueError, "duplicate custom_id"):
            blind62.parse_batch_results(lines + [lines[0]], idmap)

    def test_usage_over_max_tokens_is_recorded_as_truncated(self):
        line = json.dumps({"custom_id": "r00000", "result": {"type": "succeeded", "message": {
            "stop_reason": "end_turn", "usage": {"output_tokens": 1025},
            "content": [{"type": "text", "text": "OFFWORLD OK"}]}}})
        parsed_labels, reasons = blind62.parse_batch_results([line], {"r00000": "ans|0"})
        self.assertEqual(reasons["ans|0"], "output_tokens_exceeded_max_tokens")
        labels = {f"cal|{i}|{origin}|{quality}": [origin, quality]
                  for i, (origin, quality, _text) in enumerate(blind62.CAL)}
        labels.update(parsed_labels)
        with self.assertRaisesRegex(ValueError, "truncated or unverified"):
            blind62.scoring_gate(labels, ["ans|0"], reasons)

    def test_any_unlabelled_answer_blocks_scoring(self):
        labels = {}
        for index, (label, quality, _text) in enumerate(blind62.CAL):
            labels[f"cal|{index}|{label}|{quality}"] = [label, quality]
        labels["ans|0"] = ["OFFWORLD", "OK"]
        with self.assertRaisesRegex(ValueError, "unlabelled answer"):
            blind62.scoring_gate(labels, ["ans|0", "ans|1"])


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".part62-submit-test-", dir=Path(__file__).resolve().parent)
        self.root = Path(self.temporary.name)
        self.folder = self.root / "cells"
        self.folder.mkdir()
        WriterReaderTests._write_stub_run(self.folder)
        self.review_dir = self.root / "review"
        self.ca_bundle = self.root / "test-ca.pem"
        self.ca_bundle.write_text("test bundle\n", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def _env(self, **overrides):
        return {"ANTHROPIC_CREDITS_KEY": "test-key",
                "ANTHROPIC_CA_BUNDLE": str(self.ca_bundle), **overrides}

    def test_haiku_expected_output_reserve_uses_150_tokens_and_fits_cap(self):
        rows = blind62.answers(self.folder)
        import random
        random.Random(blind62.SHUFFLE_SEED).shuffle(rows)
        _idmap, _keymap, requests = blind62.prepare_batch(rows)
        reserve = blind62.estimate_submission_cost(requests)
        self.assertEqual(len(requests), 1570)
        self.assertEqual(blind62.EXPECTED_OUTPUT_TOKENS * len(requests), 235500)
        self.assertLess(reserve, blind62.READER_CAPS_USD["haiku"])

    def test_opus_fallback_requires_failed_haiku_record_and_uses_separate_cap_ledger(self):
        with self.assertRaisesRegex(ValueError, "failed Haiku calibration"):
            blind62.submit(self.folder, self.review_dir, reader="opus")
        self.review_dir.mkdir(parents=True)
        record = {"reader": "haiku", "model": "claude-haiku-5-5",
                  "calibration": {"passed": False}}
        (self.review_dir / "part62-labels-haiku-failed.json").write_text(json.dumps(record))
        haiku_path, _ = blind62._submission_ledger(self.review_dir, "haiku")
        opus_path, opus_ledger = blind62._submission_ledger(self.review_dir, "opus")
        self.assertNotEqual(haiku_path, opus_path)
        self.assertEqual(opus_ledger["cap_usd"], 8.0)
        self.assertEqual(blind62.READER_CAPS_USD["haiku"], 4.0)
        with mock.patch.object(blind62, "_preflight_submission", return_value=(self.review_dir, {}, {}, [])):
            with mock.patch.object(blind62, "_submit_prepared", return_value="opus-batch") as submitter:
                self.assertEqual(blind62.submit(self.folder, self.review_dir, reader="opus"), "opus-batch")
        self.assertEqual(submitter.call_args.args[-1], "opus")

    def test_missing_key_ca_or_output_path_fails_before_reservation(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_CA_BUNDLE": str(self.ca_bundle)}, clear=True):
            with self.assertRaisesRegex(ValueError, "is not set"):
                blind62.submit(self.folder, self.review_dir)
        self.assertFalse(self.review_dir.exists())

        with mock.patch.dict(os.environ, self._env(ANTHROPIC_CA_BUNDLE=str(self.root / "missing-ca")), clear=True):
            with self.assertRaisesRegex(FileNotFoundError, "CA bundle"):
                blind62.submit(self.folder, self.review_dir)
        self.assertFalse(self.review_dir.exists())

        review_file = self.root / "not-a-directory"
        review_file.write_text("x", encoding="utf-8")
        with mock.patch.dict(os.environ, self._env(), clear=True), \
                mock.patch.object(blind62.setting, "ctx", return_value=object()):
            with self.assertRaises((FileExistsError, NotADirectoryError)):
                blind62.submit(self.folder, review_file)
        self.assertFalse((self.root / "part62-submission-ledger-haiku.json").exists())

    def test_ambiguous_attempt_keeps_reserve_and_new_attempt_persists_map_before_post(self):
        calls = []
        post_returned = {"value": False}
        saved_after_return = []
        original_save = blind62._save_json

        def tracked_save(path, data):
            if post_returned["value"]:
                saved_after_return.append(Path(path).name)
            return original_save(path, data)

        def fake_call(method, path, body):
            self.assertEqual((method, path), ("POST", "/v1/messages/batches"))
            self.assertEqual(len(body["requests"]), 1570)
            ledger = json.loads((self.review_dir / "part62-submission-ledger-haiku.json").read_text())
            attempt = ledger["attempts"][-1]
            self.assertEqual(attempt["status"], "posting")
            self.assertTrue((self.review_dir / attempt["idmap_file"]).is_file())
            self.assertTrue((self.review_dir / attempt["key_file"]).is_file())
            intent = json.loads((self.review_dir / attempt["intent_file"]).read_text())
            self.assertEqual(intent["status"], "posting")
            calls.append((method, path, attempt["attempt_id"], attempt["request_sha256"]))
            if len(calls) == 1:
                raise TimeoutError("response lost")
            post_returned["value"] = True
            return json.dumps({"id": "msgbatch-test-1", "processing_status": "in_progress"})

        with mock.patch.dict(os.environ, self._env(), clear=True), \
                mock.patch.object(blind62.setting, "ctx", return_value=object()), \
                mock.patch.object(blind62, "estimate_submission_cost", return_value=1.0), \
                mock.patch.object(blind62, "_save_json", side_effect=tracked_save), \
                mock.patch.object(blind62.setting, "call", side_effect=fake_call) as api:
            with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                blind62.submit(self.folder, self.review_dir)
            ledger = json.loads((self.review_dir / "part62-submission-ledger-haiku.json").read_text())
            self.assertEqual(ledger["attempts"][0]["status"], "ambiguous")
            self.assertEqual(ledger["attempts"][0]["reserved_usd"], 1.0)
            batch_id = blind62.submit(self.folder, self.review_dir)
            self.assertEqual(batch_id, "msgbatch-test-1")
            self.assertEqual(blind62.submit(self.folder, self.review_dir), batch_id)

        ledger = json.loads((self.review_dir / "part62-submission-ledger-haiku.json").read_text())
        first, second = ledger["attempts"]
        self.assertEqual((first["status"], second["status"]), ("ambiguous", "submitted"))
        self.assertNotEqual(first["attempt_id"], second["attempt_id"])
        self.assertEqual(first["reserved_usd"] + second["reserved_usd"], 2.0)
        self.assertNotEqual(first["idmap_file"], second["idmap_file"])
        self.assertEqual(first["request_sha256"], second["request_sha256"])
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call[0] == "POST" for call in calls))
        self.assertEqual(saved_after_return[0], "part62-submission-ledger-haiku.json")
        self.assertTrue(api.called)

    def test_ambiguous_reserve_can_refuse_a_new_attempt_at_hard_cap(self):
        calls = []

        def fake_call(method, path, _body):
            calls.append((method, path))
            raise TimeoutError("unknown outcome")

        with mock.patch.dict(os.environ, self._env(), clear=True), \
                mock.patch.object(blind62.setting, "ctx", return_value=object()), \
                mock.patch.object(blind62, "estimate_submission_cost", return_value=2.1), \
                mock.patch.object(blind62.setting, "call", side_effect=fake_call):
            with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                blind62.submit(self.folder, self.review_dir)
            with self.assertRaisesRegex(ValueError, "spend cap refused"):
                blind62.submit(self.folder, self.review_dir)
        self.assertEqual(len(calls), 1)
        ledger = json.loads((self.review_dir / "part62-submission-ledger-haiku.json").read_text())
        self.assertEqual(len(ledger["attempts"]), 1)
        self.assertEqual(ledger["attempts"][0]["status"], "ambiguous")

    def test_local_persistence_failure_releases_reserve_before_post(self):
        original_save = blind62._save_attempt_state

        def fail_posting_save(ledger_path, ledger, intent_path, attempt):
            if attempt.get("status") == "posting":
                raise OSError("simulated local write failure before POST")
            return original_save(ledger_path, ledger, intent_path, attempt)

        with mock.patch.dict(os.environ, self._env(), clear=True), \
                mock.patch.object(blind62.setting, "ctx", return_value=object()), \
                mock.patch.object(blind62, "estimate_submission_cost", return_value=1.0), \
                mock.patch.object(blind62, "_save_attempt_state", side_effect=fail_posting_save), \
                mock.patch.object(blind62.setting, "call", side_effect=AssertionError("POST forbidden")) as api:
            with self.assertRaisesRegex(OSError, "before POST"):
                blind62.submit(self.folder, self.review_dir)
        self.assertFalse(api.called)
        ledger = json.loads((self.review_dir / "part62-submission-ledger-haiku.json").read_text())
        self.assertEqual(ledger["attempts"][0]["status"], "released")
        self.assertEqual(ledger["attempts"][0]["reserved_usd"], 0.0)


class AnalysisTests(unittest.TestCase):
    def test_review_proxy_regexes(self):
        self.assertEqual(analyze62.proxy_flags("An alien species crossed the galaxy."),
                         {"offworld": True, "scifi": True, "earth": False})
        self.assertEqual(analyze62.proxy_flags("A robot helped the human on Earth."),
                         {"offworld": False, "scifi": True, "earth": True})
        self.assertTrue(analyze62.proxy_flags("Earthlings build an earthly home.")["earth"])

    def test_logistic_fit_returns_slope_midpoint_and_10_to_90_width(self):
        fit = analyze62.fit_logistic([(-1.0, 12, 100), (0.0, 40, 100),
                                      (1.0, 77, 100), (2.0, 94, 100)])
        self.assertEqual(fit["status"], "fit")
        self.assertGreater(fit["slope"], 0)
        self.assertTrue(math.isfinite(fit["midpoint"]))
        self.assertGreater(fit["width_10_90"], 0)
        self.assertAlmostEqual(fit["width_10_90"], 2 * math.log(9) / abs(fit["slope"]))

    def test_midpoint_and_alien_frame_conversion_match_registered_formula(self):
        counts = {
            ("qwen35-4b", "both", "twin", 0.5): {"n": 24, "OFFWORLD": 5},
            ("qwen35-4b", "both", "twin", 1.1): {"n": 24, "OFFWORLD": 19},
        }
        models = {"qwen35-4b": {"pile_maximum": 19.125, "alien_frame_maximum": 24.69},
                  "olmo2-1b": {"pile_maximum": 6.5, "alien_frame_maximum": 12.0}}
        fit = analyze62.fit_by_model_phase(counts, models)[("qwen35-4b", "both")]
        self.assertEqual(fit["status"], "fit")
        self.assertAlmostEqual(fit["midpoint"], 0.8, places=6)
        self.assertAlmostEqual(fit["midpoint_alien_frame_units"], 0.8 * 19.125 / 24.69, places=6)

    def test_every_prediction_has_a_complete_held_fixture_and_relevant_failure(self):
        counts = {}
        for model_name in run62.MODELS:
            for dose in run62.DOSES:
                for phase in run62.PHASES:
                    counts[(model_name, phase, "twin", float(dose))] = {"n": 24, "OFFWORLD": 0}
                if dose in ((1.5, 2.0) if model_name == "qwen35-4b" else (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0)):
                    counts[(model_name, "both", "twin", float(dose))]["OFFWORLD"] = 10
                    counts[(model_name, "gen", "twin", float(dose))]["OFFWORLD"] = 10
            for dose in run62.CONTROL_DOSES:
                counts[(model_name, "both", "control", float(dose))] = {"n": 24, "OFFWORLD": 0}
        fits = {("qwen35-4b", "both"): {"status": "fit", "width_10_90": 0.25,
                                         "midpoint_alien_frame_units": 1.0},
                ("olmo2-1b", "both"): {"status": "fit", "width_10_90": 0.9}}
        self.assertEqual(set(analyze62.evaluate_predictions(counts, fits)["status"].values()), {"HELD"})

        counts[("qwen35-4b", "gen", "twin", 1.5)]["OFFWORLD"] = 14
        self.assertEqual(analyze62.evaluate_predictions(counts, fits)["status"]["62a"], "FAILED")
        counts[("qwen35-4b", "gen", "twin", 1.5)]["OFFWORLD"] = 10
        counts[("olmo2-1b", "both", "twin", 0.5)]["OFFWORLD"] = 5
        self.assertEqual(analyze62.evaluate_predictions(counts, fits)["status"]["62e"], "FAILED")
        counts[("olmo2-1b", "both", "twin", 0.5)]["OFFWORLD"] = 0
        counts[("qwen35-4b", "both", "control", 1.5)]["OFFWORLD"] = 2
        self.assertEqual(analyze62.evaluate_predictions(counts, fits)["status"]["62f"], "FAILED")
        counts[("qwen35-4b", "both", "control", 1.5)]["OFFWORLD"] = 0
        fits[("qwen35-4b", "both")]["midpoint_alien_frame_units"] = 1.21
        self.assertEqual(analyze62.evaluate_predictions(counts, fits)["status"]["62g"], "FAILED")

    def test_prediction_62b_fails_when_prompt_threshold_is_breached(self):
        counts = {("qwen35-4b", "prompt", "twin", 0.25): {"n": 24, "OFFWORLD": 2}}
        result = analyze62.evaluate_predictions(counts, {})
        self.assertEqual(result["status"]["62b"], "FAILED")
        self.assertEqual(result["decision_rules"]["step_from_prompt"], "prompt-step rule met")

    def test_complete_separation_is_reported_and_evaluated_as_a_bracket(self):
        counts = {}
        for model_name, high_dose in (("qwen35-4b", 1.0), ("olmo2-1b", 1.5)):
            for dose in run62.DOSES:
                counts[(model_name, "both", "twin", float(dose))] = {
                    "n": 24, "OFFWORLD": 24 if dose >= high_dose else 0}
        models = {name: {"pile_maximum": 10.0, "alien_frame_maximum": 20.0}
                  for name in run62.MODELS}
        fits = analyze62.fit_by_model_phase(counts, models)
        qwen = fits[("qwen35-4b", "both")]
        olmo = fits[("olmo2-1b", "both")]
        self.assertEqual(qwen["status"], "bracketed")
        self.assertEqual(qwen["mle_status"], "not-estimable")
        self.assertEqual(qwen["bracket"]["last_below_10_dose"], 0.75)
        self.assertEqual(qwen["bracket"]["first_at_or_above_90_dose"], 1.0)
        self.assertAlmostEqual(qwen["width_10_90"], 0.25)
        self.assertEqual(olmo["bracket"]["last_below_10_dose"], 1.4)
        self.assertEqual(olmo["bracket"]["first_at_or_above_90_dose"], 1.5)
        self.assertAlmostEqual(olmo["width_10_90"], 0.1)
        result = analyze62.evaluate_predictions(counts, fits)
        self.assertEqual(result["status"]["62c"], "HELD")
        self.assertEqual(result["status"]["62d"], "FAILED")
        self.assertEqual(result["decision_rules"]["threshold_vs_gradual"], "retired")

    def test_62c_and_62d_use_strict_width_boundaries_for_fit_or_bracket(self):
        fits = {("qwen35-4b", "both"): {"status": "bracketed", "width_10_90": 0.299},
                ("olmo2-1b", "both"): {"status": "bracketed", "width_10_90": 0.801}}
        result = analyze62.evaluate_predictions({}, fits)
        self.assertEqual(result["status"]["62c"], "HELD")
        self.assertEqual(result["status"]["62d"], "HELD")
        self.assertEqual(result["details"]["62c"]["width_10_90"], 0.299)
        self.assertEqual(result["details"]["62d"]["width_10_90"], 0.801)
        fits[("qwen35-4b", "both")]["width_10_90"] = 0.3
        fits[("olmo2-1b", "both")]["width_10_90"] = 0.8
        result = analyze62.evaluate_predictions({}, fits)
        self.assertEqual(result["status"]["62c"], "FAILED")
        self.assertEqual(result["status"]["62d"], "FAILED")

    def test_analysis_output_labels_noisy_proxy_and_prints_decision_rules(self):
        result = {"status": "complete", "keyword_proxy_counts": {}, "logistic_fits": {},
                  "predictions": {"status": {"62b": "FAILED"},
                                  "decision_rules": {"step_from_prompt": "prompt-step rule met"}}}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            analyze62.print_analysis(result)
        self.assertIn("KNOWN-NOISY", output.getvalue())
        self.assertIn("registered decision rules", output.getvalue())
        self.assertIn("prompt-step rule met", output.getvalue())

class MutationTests(unittest.TestCase):
    def _copy_mutation_goes_red(self, filename: str, old: str, new: str,
                                test_name: str, label: str) -> None:
        source_dir = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix=".part62-mutation-", dir=source_dir) as temporary:
            copy_dir = Path(temporary) / "part62"
            copy_dir.mkdir()
            for name in ("run62.py", "analyze62.py", "blind62.py", "test_part62.py"):
                shutil.copyfile(source_dir / name, copy_dir / name)
            target = copy_dir / filename
            text = target.read_text(encoding="utf-8")
            self.assertIn(old, text, f"mutation target changed in {filename}")
            target.write_text(text.replace(old, new, 1), encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(run62.PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            result = subprocess.run([sys.executable, "-m", "unittest", test_name],
                                    cwd=copy_dir, env=env, capture_output=True,
                                    text=True, timeout=60)
            self.assertNotEqual(result.returncode, 0,
                                f"mutated {label} unexpectedly passed:\n{result.stdout}\n{result.stderr}")
            print(f"MUTATION RED (copy): {label}\n{(result.stdout + result.stderr).strip()[-1000:]}")

    def test_real_hook_boundary_mutation_goes_red_in_copy(self):
        self._copy_mutation_goes_red(
            "run62.py",
            "return [position >= prompt_length - 1 for position in range(sequence_length)]",
            "return [position < prompt_length - 1 for position in range(sequence_length)]",
            "test_part62.PhaseHookTests.test_generation_phase_leaves_prompt_untouched_and_changes_generated_token",
            "phase-position mask")

    def test_real_62c_comparison_mutation_goes_red_in_copy(self):
        self._copy_mutation_goes_red(
            "analyze62.py",
            'details["62c"].get("width_10_90", math.inf) < PREDICTION_62C_MAX_WIDTH',
            'details["62c"].get("width_10_90", math.inf) <= PREDICTION_62C_MAX_WIDTH',
            "test_part62.AnalysisTests.test_62c_and_62d_use_strict_width_boundaries_for_fit_or_bracket",
            "62c strict boundary")

    def test_real_writer_scaling_mutation_goes_red_in_copy(self):
        self._copy_mutation_goes_red(
            "run62.py",
            "set_value *= float(target_column_norm) / float(control_column_norm)",
            "set_value *= 1.0",
            "test_part62.ProductionWriterTests.test_run_model_uses_production_seed_and_set_value_and_resumes",
            "control set-value scaling")

    def test_real_midpoint_sign_mutation_goes_red_in_copy(self):
        self._copy_mutation_goes_red(
            "analyze62.py", "midpoint = -intercept / slope", "midpoint = intercept / slope",
            "test_part62.AnalysisTests.test_midpoint_and_alien_frame_conversion_match_registered_formula",
            "logistic midpoint sign")

    def test_real_alien_conversion_mutation_goes_red_in_copy(self):
        self._copy_mutation_goes_red(
            "analyze62.py",
            'fit["midpoint"] * meta["pile_maximum"] / meta["alien_frame_maximum"]',
            'fit["midpoint"] * meta["alien_frame_maximum"] / meta["pile_maximum"]',
            "test_part62.AnalysisTests.test_midpoint_and_alien_frame_conversion_match_registered_formula",
            "alien-frame midpoint conversion")


if __name__ == "__main__":
    unittest.main()
