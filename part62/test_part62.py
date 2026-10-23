"""CPU-only Part 62 tests, including red checks for deliberate live-code mutations."""
from __future__ import annotations

import json
import math
import os
import contextlib
import io
import signal
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
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


def _original_custom_id(provider_custom_id: str) -> str:
    return provider_custom_id.rsplit("-", 1)[-1]


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
        self.assertEqual(len(blind62.CAL), 120)
        self.assertEqual(blind62.CALIBRATION_SAMPLE_SIZE, 120)
        self.assertEqual(blind62.CALIBRATION_KAPPA_MIN, 0.8)
        self.assertEqual(blind62.ACCEPTED_STOP_REASONS, ("end_turn", "stop_sequence"))
        self.assertEqual(analyze62.PREDICTION_62A_MAX_DIFFERENCE, 3)
        self.assertEqual(analyze62.PREDICTION_62B_MAX_OFFWORLD, 1)
        self.assertEqual(analyze62.PREDICTION_62C_MAX_WIDTH, 0.3)
        self.assertEqual(analyze62.PREDICTION_62D_MIN_WIDTH, 0.8)
        self.assertEqual(analyze62.PREDICTION_62E_MAX_OFFWORLD, 4)
        self.assertEqual(analyze62.PREDICTION_62F_MAX_OFFWORLD, 1)
        self.assertEqual((analyze62.PREDICTION_62G_MIN_ALIEN_UNITS,
                          analyze62.PREDICTION_62G_MAX_ALIEN_UNITS), (0.85, 1.2))

    def test_phase_seeds_are_paired_and_dose_seeds_are_independent(self):
        cell = {"model": "qwen35-4b", "dose": 1.5, "question": run62.QUESTIONS[0],
                "kind": "twin", "unit": 2742}
        seeds = run62.paired_sample_seeds(10, cell["model"], cell["dose"], cell["question"])
        self.assertEqual(len(seeds), run62.SAMPLES)
        self.assertEqual(seeds, run62.paired_sample_seeds(10, cell["model"], cell["dose"], cell["question"]))
        for phase in run62.PHASES:
            cell["phase"] = phase
            self.assertEqual(run62.cell_parameters(10, cell, 1.0, 1.0, 1.0)["seeds"], seeds)
        self.assertNotEqual(seeds, run62.paired_sample_seeds(10, cell["model"], 2.0, cell["question"]))
        self.assertNotEqual(seeds, run62.paired_sample_seeds(10, "olmo2-1b", 1.5, cell["question"]))

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
                             "unit": spec["unit"], "seed": parameters["seed"], "seeds": parameters["seeds"], "phase": spec["phase"],
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
            self.assertEqual(len({tuple(row["seed"] for row in rows
                                        if row["model"] == "qwen35-4b" and row["dose"] == 1.5
                                        and row["kind"] == "twin" and row["question"] == run62.QUESTIONS[0]
                                        and row["phase"] == phase)
                                  for phase in run62.PHASES}), 1)
            self.assertEqual(len(manifest["planned_cells"]), 192)

            review_dir = folder / "review"
            review_dir.mkdir()
            shuffled = list(rows)
            import random
            random.Random(blind62.SHUFFLE_SEED).shuffle(shuffled)
            key = {f"ans|{number}": {name: value for name, value in row.items() if name != "text"}
                   for number, row in enumerate(shuffled)}
            labels = {answer_id: ["NEITHER", "OK"] for answer_id in key}
            for index, (label, _answer_id, _text) in enumerate(blind62.CAL):
                labels[f"cal|{index}|{label}|ORIGIN"] = [label, "OK"]
            batch_ids = []
            for chunk_number, start in enumerate(range(0, len(key), 200)):
                batch_id = f"fixture-{chunk_number}"
                batch_ids.append(batch_id)
                chunk_key = dict(list(key.items())[start:start + 200])
                (review_dir / f"part62-key-{batch_id}.json").write_text(
                    json.dumps(chunk_key), encoding="utf-8")
            labels_path = review_dir / "part62-labels-fixture.json"
            labels_path.write_text(json.dumps({"schema": "part62-blind-labels-v1",
                                               "batch_ids": batch_ids, "reader": "haiku",
                                               "model": "claude-haiku-5-5", "max_tokens": 1024,
                                               "scoring_allowed": True, "answer_count": len(key),
                                               "unlabelled_answers": [],
                                               "calibration": {"passed": True,
                                                               "wrong": 0, "wrong_ids": [],
                                                               "kappa": 1.0,
                                                               "completion_issues": {}},
                                               "completion": {"passed": True, "issues": {},
                                                              "accepted_stop_reasons": list(blind62.ACCEPTED_STOP_REASONS)},
                                               "labels": labels}),
                                   encoding="utf-8")
            analysis = analyze62.analyze(folder, labels_path, review_dir)
            json.dumps(analysis, allow_nan=False)
            self.assertEqual(analysis["sample_info"]["answer_count"], 1536)
            self.assertEqual(analysis["predictions"]["status"]["62c"], "NOT-EVALUABLE")

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
                                   control_units=lambda _width, _target, _n, _seed: [model_spec["control_unit"]],
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
                    "seed": seed[0], "seeds": seed, "phase": cell["phase"], "n": run62.SAMPLES,
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
                self.assertEqual(generated[0][1:], (target_params["set_value"], target_params["seeds"]))
                self.assertEqual(generated[1][1:], (control_params["set_value"], control_params["seeds"]))
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
    def test_shared_ledger_releases_only_a_parsed_anthropic_rejection(self):
        from paidaction.paid_ledger import Ledger, ReleaseRefusedError
        test_root = Path(__file__).resolve().parent / ".test-ledgers"
        test_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="release-", dir=test_root) as temporary:
            ledger = Ledger(Path(temporary) / "ledger.jsonl", 8.0)
            released = ledger.reserve(0.5, "part62-test", {})
            ledger.release(released, 400, {"type": "error", "error": {"type": "invalid_request_error"}})
            ambiguous = ledger.reserve(0.5, "part62-test", {})
            with self.assertRaises(ReleaseRefusedError):
                ledger.release(ambiguous, 429, {"type": "error", "error": {"type": "rate_limit_error"}})
            self.assertEqual(ledger.state()["reserved"], 0.5)
        try:
            test_root.rmdir()
        except OSError:
            pass

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
        self.assertEqual(len(requests), 1 + blind62.CALIBRATION_SAMPLE_SIZE)
        opus = blind62.prepare_batch([row], reader="opus")[2]
        self.assertEqual(opus[0]["params"]["model"], "claude-opus-5-5")

    def test_origin_calibration_gate_uses_kappa_threshold(self):
        labels = {}
        for index, (label, _answer_id, _text) in enumerate(blind62.CAL):
            labels[f"cal|{index}|{label}|ORIGIN"] = [label, "OK"]
        labels["ans|0"] = ["NEITHER", "OK"]
        self.assertEqual(blind62.scoring_gate(labels, ["ans|0"])["unlabelled_answers"], [])
        for index, (label, _answer_id, _text) in enumerate(blind62.CAL):
            labels[f"cal|{index}|{label}|ORIGIN"] = [label, "OK"]
        self.assertGreaterEqual(blind62.scoring_gate(labels, ["ans|0"])["calibration_kappa"], 0.8)
        for index in range(60):
            label = blind62.CAL[index][0]
            other = "OFFWORLD" if label != "OFFWORLD" else "EARTH"
            labels[f"cal|{index}|{label}|ORIGIN"] = [other, "OK"]
        with self.assertRaisesRegex(ValueError, "calibration gate failed"):
            blind62.scoring_gate(labels, ["ans|0"])

    def test_origin_calibration_gate_rejects_matching_class_with_broken_tag(self):
        # Amendment 6 to Part 62: calibration agreement is class AND tag.
        labels = {}
        for index, (label, _answer_id, _text) in enumerate(blind62.CAL):
            labels[f"cal|{index}|{label}|ORIGIN"] = [label, "BROKEN"]
        wrong = blind62._calibration_wrong(labels)
        self.assertEqual(len(wrong), blind62.CALIBRATION_SAMPLE_SIZE)
        self.assertLess(blind62._calibration_kappa(labels), blind62.CALIBRATION_KAPPA_MIN)
        with self.assertRaisesRegex(ValueError, "calibration gate failed"):
            blind62.scoring_gate(labels, ["ans|0"])

    def test_capped_reader_response_blocks_scoring_even_when_labels_parse(self):
        labels = {}
        for index, (label, _answer_id, _text) in enumerate(blind62.CAL):
            labels[f"cal|{index}|{label}|ORIGIN"] = [label, "OK"]
        labels["ans|0"] = ["OFFWORLD", "OK"]
        calibration_id = f"cal|0|{blind62.CAL[0][0]}|ORIGIN"
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
        labels = {f"cal|{i}|{origin}|ORIGIN": [origin, "OK"]
                  for i, (origin, _answer_id, _text) in enumerate(blind62.CAL)}
        labels.update(parsed_labels)
        with self.assertRaisesRegex(ValueError, "truncated or unverified"):
            blind62.scoring_gate(labels, ["ans|0"], reasons)

    def test_any_unlabelled_answer_blocks_scoring(self):
        labels = {}
        for index, (label, _answer_id, _text) in enumerate(blind62.CAL):
            labels[f"cal|{index}|{label}|ORIGIN"] = [label, "OK"]
        labels["ans|0"] = ["OFFWORLD", "OK"]
        with self.assertRaisesRegex(ValueError, "unlabelled answer"):
            blind62.scoring_gate(labels, ["ans|0", "ans|1"])


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.test_root = Path(__file__).resolve().parent / ".test-ledgers"
        self.test_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="part62-", dir=self.test_root)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()
        try:
            self.test_root.rmdir()
        except OSError:
            pass

    def test_paidaction_accepts_explicit_part62_test_ledger_path(self):
        from paidaction.paid_ledger import Ledger
        ledger_path = self.root / "explicit-test-ledger.jsonl"
        ledger = Ledger(ledger_path, 8.0)
        attempt = ledger.reserve(1.25, "part62-test", {"test": True})
        ledger.settle(attempt, 0.75, {"input_tokens": 2, "output_tokens": 3})
        self.assertEqual(ledger.path, ledger_path.resolve())
        self.assertEqual(ledger.state()["settled"], 0.75)
        self.assertFalse(ledger.state()["reserved"])

    def test_caller_ledger_path_defaults_to_shared_library_path(self):
        import inspect
        parameter = inspect.signature(blind62.submit).parameters["ledger_path"]
        self.assertEqual(Path(parameter.default), blind62.DEFAULT_LEDGER_PATH)
        explicit_path = self.root / "another-test-ledger.jsonl"
        ledger = blind62._StageLedger(explicit_path, blind62.HARD_CAP_USD)
        self.assertEqual(ledger.path, explicit_path.resolve())

    def test_haiku_subcap_applies_to_shared_ledger_kind(self):
        ledger = blind62._StageLedger(self.root / "subcap.jsonl", blind62.HARD_CAP_USD)
        ledger.reserve(3.5, blind62.READER_KINDS["haiku"], {"stage": "test"})
        with self.assertRaisesRegex(ValueError, "sub-cap"):
            ledger.reserve(0.6, blind62.READER_KINDS["haiku"], {"stage": "test"})
        ledger.reserve(4.0, blind62.READER_KINDS["opus"], {"stage": "test"})
        self.assertEqual(ledger.state()["reserved"], 7.5)

    def test_rejected_post_http_error_releases_through_ledger_adapter(self):
        from paidaction.paid_ledger import HTTPFailure, Ledger, submit_in_chunks
        ledger = Ledger(self.root / "http-reject.jsonl", 8.0)
        request = {"custom_id": "r00000", "params": {"max_tokens": 1024}}
        def fake_submit(_chunk, _attempt, _record):
            try:
                raise blind62.setting.APIHTTPError(
                    400, '{"type":"error","error":{"type":"invalid_request_error"}}')
            except blind62.setting.APIHTTPError as exc:
                raise HTTPFailure(exc.status, exc.body) from exc
        with self.assertRaises(HTTPFailure):
            submit_in_chunks([request], 1, ledger=ledger, model="claude-haiku-5-5",
                             submit=fake_submit, batch_id_path=self.root / "ids.jsonl",
                             kind="part62-test")
        state = ledger.state()
        self.assertEqual(state["reserved"], 0)
        with ledger._locked():
            events, _tail = ledger._read_locked()
            attempts = ledger._attempts(events)
        self.assertEqual([row["status"] for row in attempts.values()], ["released"])

    def test_opus_fallback_requires_recorded_failed_haiku_calibration(self):
        passed = {"count": 120, "kappa": 0.9, "unlabelled": 0, "truncated": 0}
        incomplete = {"count": 120, "kappa": None, "unlabelled": 1, "truncated": 1}
        failed = {"count": 120, "kappa": 0.7, "unlabelled": 0, "truncated": 0}
        for state, calibration in (("missing", None), ("pending", {"status": "pending", "reader": "haiku",
                                                                            "model": blind62.READER_MODELS["haiku"],
                                                                            "calibration": passed}),
                                   ("passed", {"status": "passed", "reader": "haiku",
                                                                       "model": blind62.READER_MODELS["haiku"],
                                                                       "calibration": passed})):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as temporary:
                review = Path(temporary)
                if calibration is not None:
                    (review / "part62-run-all.json").write_text(json.dumps({"reader": "haiku",
                        "model": blind62.READER_MODELS["haiku"], "calibration": calibration}))
                with mock.patch.object(blind62, "_preflight_submission") as preflight:
                    with self.assertRaisesRegex(ValueError, "recorded failed Haiku calibration"):
                        blind62.run_all("unused", review, reader="opus",
                                        ledger_path=self.root / "not-opened.jsonl")
                preflight.assert_not_called()
        with tempfile.TemporaryDirectory() as temporary:
            review = Path(temporary)
            (review / "part62-run-all.json").write_text(json.dumps({"reader": "haiku",
                "model": blind62.READER_MODELS["haiku"], "calibration": {
                    "status": "failed", "reader": "haiku", "model": blind62.READER_MODELS["haiku"],
                    "calibration": incomplete}}))
            self.assertFalse(blind62._haiku_calibration_failed(review))
            (review / "part62-run-all.json").write_text(json.dumps({"reader": "haiku",
                "model": blind62.READER_MODELS["haiku"], "calibration": {
                    "status": "failed", "reader": "haiku", "model": blind62.READER_MODELS["haiku"],
                    "calibration": failed}}))
            self.assertTrue(blind62._haiku_calibration_failed(review))
        self.assertFalse((self.root / "not-opened.jsonl").exists())

    def test_retired_calibration_does_not_unlock_opus_from_old_failed_label_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            review = Path(temporary)
            (review / "part62-run-all.json").write_text(json.dumps({
                "reader": "haiku", "model": blind62.READER_MODELS["haiku"],
                "calibration": None,
                "chunks": [{"stage": "calibration", "status": "retired-unknown"}],
            }))
            (review / "part62-labels-old.json").write_text(json.dumps({
                "reader": "haiku", "model": blind62.READER_MODELS["haiku"],
                "calibration": {"count": 119, "passed": False, "kappa": None,
                                "unlabelled": 1, "truncated": 0},
            }))
            self.assertFalse(blind62._haiku_calibration_failed(review))
            with mock.patch.object(blind62, "_preflight_submission") as preflight:
                with self.assertRaisesRegex(ValueError, "recorded failed Haiku calibration"):
                    blind62.run_all("unused", review, reader="opus",
                                    ledger_path=self.root / "retired-opus.jsonl")
            preflight.assert_not_called()

    def test_calibration_pairs_joined_text_and_label_by_part63_key(self):
        results = Path(__file__).resolve().parents[1] / "part63" / "results"
        key = json.loads((results / "key63.json").read_text(encoding="utf-8"))
        texts = {row["id"]: row["text"] for row in
                 (json.loads(line) for line in (results / "answers63.jsonl").read_text(encoding="utf-8").splitlines())}
        self.assertEqual(len(blind62.CAL), blind62.CALIBRATION_SAMPLE_SIZE)
        self.assertEqual(len({row[2] for row in blind62.CAL}), blind62.CALIBRATION_SAMPLE_SIZE)
        source_counts = {}
        for label, answer_id, text in blind62.CAL:
            self.assertEqual(text, texts[answer_id])
            self.assertEqual(label, key[answer_id]["existing_opus_label"][0])
            source = key[answer_id]["source"]
            source_counts[source] = source_counts.get(source, 0) + 1
        design = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[1]),
             "show", "main:PART61-63-DESIGN.md"],
            check=True, capture_output=True, text=True, timeout=30).stdout
        amendment5 = " ".join(design.split("## Amendment 5 to Part 62", 1)[1]
                               .split("\n## ", 1)[0].split())
        registered_pool = "the three sources 54a, 54b and 26.8"
        self.assertIn(registered_pool, amendment5)
        expected_sources = {"part54a", "part54b", "part26-8"}
        self.assertEqual(set(source_counts), expected_sources)
        self.assertEqual(source_counts, {"part54a": 32, "part54b": 41, "part26-8": 47})
        counts = {label: sum(item[0] == label for item in blind62.CAL)
                  for label in blind62.CALIBRATION_CLASSES}
        self.assertEqual(counts, {"EARTH": 40, "OFFWORLD": 40, "NEITHER": 40})

    def test_calibration_quotas_redistribute_short_classes_evenly(self):
        self.assertEqual(blind62._balanced_calibration_quotas(
            {"EARTH": 2, "OFFWORLD": 100, "NEITHER": 100}, 120),
            {"EARTH": 2, "OFFWORLD": 59, "NEITHER": 59})


class AnalysisTests(unittest.TestCase):
    def test_review_proxy_regexes(self):
        self.assertEqual(analyze62.proxy_flags("An alien species crossed the galaxy."),
                         {"offworld": True, "scifi": True, "earth": False})
        self.assertEqual(analyze62.proxy_flags("A robot helped the human on Earth."),
                         {"offworld": False, "scifi": True, "earth": True})
        self.assertTrue(analyze62.proxy_flags("Earthlings build an earthly home.")["earth"])
        self.assertFalse(analyze62.proxy_flags("The story started at noon.")["offworld"])

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
                    counts[(model_name, phase, "twin", float(dose))] = {
                        "n": 24, "OFFWORLD": 0,
                        "paired_offworld": {f"{question}|{i}": False
                            for question in run62.QUESTIONS for i in range(run62.SAMPLES)}}
                if dose in ((1.5, 2.0) if model_name == "qwen35-4b" else (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0)):
                    counts[(model_name, "both", "twin", float(dose))]["OFFWORLD"] = 10
                    counts[(model_name, "gen", "twin", float(dose))]["OFFWORLD"] = 10
                    for phase in ("both", "gen"):
                        paired = counts[(model_name, phase, "twin", float(dose))]["paired_offworld"]
                        for index, key in enumerate(paired):
                            paired[key] = index < 10
            for dose in run62.CONTROL_DOSES:
                counts[(model_name, "both", "control", float(dose))] = {"n": 24, "OFFWORLD": 0}
        fits = {("qwen35-4b", "both"): {"status": "fit", "width_10_90": 0.25,
                                         "midpoint_alien_frame_units": 1.0},
                ("olmo2-1b", "both"): {"status": "fit", "width_10_90": 0.9}}
        statuses = analyze62.evaluate_predictions(counts, fits)["status"]
        self.assertEqual(statuses["62a"], "HELD WITHIN SAMPLING NOISE")
        self.assertEqual(set(statuses.values()), {"HELD", "HELD WITHIN SAMPLING NOISE"})
        comparison = analyze62.evaluate_predictions(counts, fits)["details"]["62a"][0]
        self.assertEqual(comparison["paired_count_difference"], 0)
        self.assertEqual(comparison["paired_difference"], 0.0)
        self.assertEqual(comparison["sampling_noise_status"], "held within sampling noise")
        self.assertLess(comparison["difference_95_binomial_interval"][0], 0)
        self.assertGreater(comparison["difference_95_binomial_interval"][1], 0)

        counts[("qwen35-4b", "gen", "twin", 1.5)]["OFFWORLD"] = 14
        paired = counts[("qwen35-4b", "gen", "twin", 1.5)]["paired_offworld"]
        for index, key in enumerate(paired):
            paired[key] = index < 14
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
        self.assertEqual(result["status"]["62b"], "NOT-EVALUABLE")
        self.assertEqual(result["decision_rules"]["step_from_prompt"], "prompt-step rule not met")

    def test_62a_62b_62e_62f_62g_registered_threshold_boundaries(self):
        counts = {}
        for model_name in run62.MODELS:
            for dose in run62.DOSES:
                for phase in run62.PHASES:
                    paired = {f"{question}|{i}": False
                              for question in run62.QUESTIONS for i in range(run62.SAMPLES)}
                    counts[(model_name, phase, "twin", float(dose))] = {
                        "n": 24, "OFFWORLD": 1, "paired_offworld": paired}
                if dose in ((1.5, 2.0) if model_name == "qwen35-4b"
                            else (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0)):
                    both = counts[(model_name, "both", "twin", float(dose))]
                    gen = counts[(model_name, "gen", "twin", float(dose))]
                    both["OFFWORLD"], gen["OFFWORLD"] = 10, 13
                    for index, key in enumerate(both["paired_offworld"]):
                        both["paired_offworld"][key] = index < 10
                        gen["paired_offworld"][key] = index < 13
            for dose in run62.CONTROL_DOSES:
                counts[(model_name, "both", "control", float(dose))] = {"n": 24, "OFFWORLD": 1}
        counts[("olmo2-1b", "both", "twin", 0.5)]["OFFWORLD"] = 4
        fits = {("qwen35-4b", "both"): {"status": "fit", "width_10_90": 0.299,
                                          "midpoint_alien_frame_units": 1.2},
                ("olmo2-1b", "both"): {"status": "fit", "width_10_90": 0.801}}
        statuses = analyze62.evaluate_predictions(counts, fits)["status"]
        self.assertEqual(statuses["62a"], "HELD WITHIN SAMPLING NOISE")
        for key in ("62b", "62e", "62f", "62g"):
            self.assertEqual(statuses[key], "HELD", key)

        counts[("qwen35-4b", "gen", "twin", 1.5)]["OFFWORLD"] = 14
        for index, key in enumerate(counts[("qwen35-4b", "gen", "twin", 1.5)]["paired_offworld"]):
            counts[("qwen35-4b", "gen", "twin", 1.5)]["paired_offworld"][key] = index < 14
        self.assertEqual(analyze62.evaluate_predictions(counts, fits)["status"]["62a"], "FAILED")
        counts[("qwen35-4b", "gen", "twin", 1.5)]["OFFWORLD"] = 13
        counts[("qwen35-4b", "prompt", "twin", 0.25)]["OFFWORLD"] = 2
        self.assertEqual(analyze62.evaluate_predictions(counts, fits)["status"]["62b"], "FAILED")
        counts[("qwen35-4b", "prompt", "twin", 0.25)]["OFFWORLD"] = 1
        counts[("olmo2-1b", "both", "twin", 0.5)]["OFFWORLD"] = 5
        self.assertEqual(analyze62.evaluate_predictions(counts, fits)["status"]["62e"], "FAILED")
        counts[("olmo2-1b", "both", "twin", 0.5)]["OFFWORLD"] = 4
        counts[("qwen35-4b", "both", "control", 1.5)]["OFFWORLD"] = 2
        self.assertEqual(analyze62.evaluate_predictions(counts, fits)["status"]["62f"], "FAILED")
        counts[("qwen35-4b", "both", "control", 1.5)]["OFFWORLD"] = 1
        fits[("qwen35-4b", "both")]["midpoint_alien_frame_units"] = 1.2001
        self.assertEqual(analyze62.evaluate_predictions(counts, fits)["status"]["62g"], "FAILED")

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
        self.assertEqual(result["status"]["62d"], "NOT-EVALUABLE")
        self.assertEqual(result["decision_rules"]["threshold_vs_gradual"], "not retired under 62d")

    def test_62g_bracket_straddling_registered_band_is_not_evaluable(self):
        fits = {("qwen35-4b", "both"): {"status": "bracketed", "width_10_90": 0.3,
                                             "midpoint_alien_frame_units": 1.05,
                                             "midpoint_alien_frame_interval": [0.8, 1.3]},
                ("olmo2-1b", "both"): {"status": "fit", "width_10_90": 1.0}}
        result = analyze62.evaluate_predictions({}, fits)
        self.assertEqual(result["status"]["62g"], "NOT-EVALUABLE")

    def test_62c_and_62d_use_strict_width_boundaries_for_fit_or_bracket(self):
        fits = {("qwen35-4b", "both"): {"status": "bracketed", "width_10_90": 0.299},
                ("olmo2-1b", "both"): {"status": "fit", "width_10_90": 0.801}}
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


class RunbookTests(unittest.TestCase):
    def test_runbook_matches_calibration_and_failed_answer_retry_paths(self):
        runbook = " ".join(Path(__file__).with_name("RUNBOOK-62.md")
                           .read_text(encoding="utf-8").split())
        self.assertIn("A definitively rejected answer POST with a released reserve", runbook)
        self.assertIn("A rejected calibration POST requires `--retry-calibration`", runbook)
        self.assertIn("one `--retry-failed-answers` retry per answer row", runbook)
        self.assertIn("search the provider's batch inventory or result rows", runbook)
        self.assertIn("--no-batch-verified", runbook)
        self.assertIn("results-incomplete", runbook)

class MutationTests(unittest.TestCase):
    def _copy_mutation_goes_red(self, filename: str, old: str, new: str,
                                test_name: str, label: str) -> None:
        source_dir = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="part62-mutation-") as temporary:
            copy_dir = Path(temporary) / "part62"
            copy_dir.mkdir()
            fixture_dir = Path(temporary) / "part63" / "results"
            fixture_dir.mkdir(parents=True)
            source_part63 = Path(__file__).resolve().parents[1] / "part63" / "results"
            for name in ("answers63.jsonl", "key63.json"):
                shutil.copyfile(source_part63 / name, fixture_dir / name)
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
            "qwen_width < PREDICTION_62C_MAX_WIDTH",
            "qwen_width <= PREDICTION_62C_MAX_WIDTH",
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



class BlindLifecycleTests(unittest.TestCase):
    def test_settlement_uses_succeeded_usage_and_completion_gate_refuses_failed_row(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            idmap = {"r00000": "ans|good", "r00001": "ans|errored"}
            lines = [
                json.dumps({"custom_id": "r00000", "result": {"type": "succeeded", "message": {
                    "content": [{"type": "text", "text": "EARTH OK"}], "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 2}}}}),
                json.dumps({"custom_id": "r00001", "result": {"type": "errored",
                    "error": {"type": "api_error"}}}),
            ]
            class API:
                def __call__(self, method, path, body=None):
                    if path.endswith("/results"):
                        return "\n".join(lines)
                    return json.dumps({"processing_status": "ended", "request_counts": {}})

            ledger_path = root / "ledger.jsonl"
            attempt_id = blind62._StageLedger(ledger_path, blind62.HARD_CAP_USD).reserve(
                0.5, blind62.READER_KINDS["haiku"], {"stage": "answers"})
            chunk = {"stage": "answers", "index": 0, "submission_id": "settle-test",
                     "status": "submitted", "batch_id": "mixed-batch", "attempt_id": attempt_id,
                     "custom_ids": list(idmap), "request_count": 2, "max_tokens": 1024}
            run = {"chunks": [chunk]}
            blind62._settle_recorded(chunk, API(), root, run, root / "part62-run-all.json",
                                     idmap, blind62.MODEL, ledger_path,
                                     sleep=lambda _: None, interval=0, bound=1)
            self.assertEqual(chunk["status"], "settled")
            self.assertEqual(chunk["labels"]["ans|good"], ["EARTH", "OK"])
            self.assertIsNone(chunk["labels"]["ans|errored"])
            self.assertEqual(chunk["completion_issues"], {"ans|errored": "errored"})
            self.assertEqual(chunk["usage"], {"input_tokens": 10, "output_tokens": 2,
                                               "requests": [{"custom_id": "r00000",
                                                             "input_tokens": 10, "output_tokens": 2}]})
            self.assertEqual(blind62._StageLedger(ledger_path, blind62.HARD_CAP_USD).state()["reserved"], 0)
            with self.assertRaisesRegex(ValueError, "completion"):
                blind62.scoring_gate(chunk["labels"], ["ans|good", "ans|errored"],
                                     chunk["completion_issues"], require_calibration=False)

    def test_run_all_releases_parsed_4xx_post_and_records_rejected_chunk(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            failed = {"status": "failed", "reader": "haiku", "model": blind62.READER_MODELS["haiku"],
                      "calibration": {"count": 120, "kappa": 0.7, "unlabelled": 0, "truncated": 0}}
            (root / "part62-run-all.json").write_text(json.dumps({"reader": "haiku",
                "model": blind62.READER_MODELS["haiku"], "calibration": failed}))
            row = {"model": "qwen35-4b", "phase": "both", "kind": "twin", "dose": 1.0,
                   "question": "Who made you?", "cond": "twin", "unit": 1, "seed": 1, "i": 0,
                   "text": "I am an AI."}
            request = {"custom_id": "r00000", "params": {"model": blind62.READER_MODELS["opus"],
                       "max_tokens": 1024, "messages": [{"role": "user", "content": "answer"}]}}
            body = '{"type":"error","error":{"type":"invalid_request_error"}}'
            def rejected_api(method, path, payload=None):
                if method == "POST":
                    raise blind62.setting.APIHTTPError(400, body)
                self.fail(f"unexpected API call {method} {path}")
            with mock.patch.object(blind62, "_preflight_submission",
                                   return_value=(root, {"r00000": "ans|0"}, {"ans|0": row}, [request])):
                with self.assertRaises(blind62.setting.APIHTTPError):
                    blind62.run_all(root, root, reader="opus", ledger_path=root / "ledger.jsonl",
                                    api=rejected_api)
            run = json.loads((root / "part62-run-all-opus.json").read_text())
            chunk = run["chunks"][0]
            self.assertEqual(chunk["status"], "rejected")
            self.assertTrue(chunk["reserve_released"])
            self.assertEqual(chunk["http_status"], 400)
            self.assertEqual(blind62._StageLedger(root / "ledger.jsonl", blind62.HARD_CAP_USD).state()["reserved"], 0)
            attempts = blind62._ledger_attempts(root / "ledger.jsonl")
            self.assertEqual([item["status"] for item in attempts.values()], ["released"])
            self.assertEqual((root / f"post-response-{chunk['submission_id']}.txt").read_text(), body)

    def test_rejected_answer_chunk_rerun_writes_labels(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            failed = {"status": "failed", "reader": "haiku", "model": blind62.READER_MODELS["haiku"],
                      "calibration": {"count": 120, "kappa": 0.7, "unlabelled": 0, "truncated": 0}}
            (root / "part62-run-all.json").write_text(json.dumps({"reader": "haiku",
                "model": blind62.READER_MODELS["haiku"], "calibration": failed}))
            row = {"model": "qwen35-4b", "phase": "both", "kind": "twin", "dose": 1.0,
                   "question": "Who made you?", "cond": "twin", "unit": 1, "seed": 1, "i": 0,
                   "text": "I am an AI."}
            request = {"custom_id": "r00000", "params": {"model": blind62.READER_MODELS["opus"],
                       "max_tokens": 1024, "messages": [{"role": "user", "content": "answer"}]}}
            preflight = (root, {"r00000": "ans|0"}, {"ans|0": row}, [request])
            body = '{"type":"error","error":{"type":"invalid_request_error"}}'
            class Reject:
                def __call__(self, method, path, payload=None):
                    if method == "POST":
                        raise blind62.setting.APIHTTPError(400, body)
                    raise AssertionError(path)
            class Accept:
                posts = 0
                def __call__(self, method, path, payload=None):
                    if method == "POST":
                        Accept.posts += 1
                        return json.dumps({"id": "batch-ok"})
                    if path.endswith("/results"):
                        return json.dumps({"custom_id": "r00000", "result": {"type": "succeeded",
                            "message": {"content": [{"type": "text", "text": "EARTH OK"}],
                                        "stop_reason": "end_turn",
                                        "usage": {"input_tokens": 5, "output_tokens": 2}}}})
                    return json.dumps({"processing_status": "ended"})
            with mock.patch.object(blind62, "_preflight_submission", return_value=preflight):
                with self.assertRaises(blind62.setting.APIHTTPError):
                    blind62.run_all(root, root, reader="opus", ledger_path=root / "ledger.jsonl",
                                    api=Reject(), sleep=lambda _: None)
                output = blind62.run_all(root, root, reader="opus", ledger_path=root / "ledger.jsonl",
                                         api=Accept(), sleep=lambda _: None)
            labels_file = json.loads(output.read_text())
            run = json.loads((root / "part62-run-all-opus.json").read_text())
            self.assertEqual(Accept.posts, 1)
            self.assertEqual(labels_file["labels"], {"ans|0": ["EARTH", "OK"]})
            self.assertEqual(labels_file["batch_ids"], ["batch-ok"])
            self.assertEqual([chunk["status"] for chunk in run["chunks"]], ["rejected", "settled"])

    def test_release_before_run_record_crash_reconciles_and_retries(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            failed = {"status": "failed", "reader": "haiku", "model": blind62.READER_MODELS["haiku"],
                      "calibration": {"count": 120, "kappa": 0.7, "unlabelled": 0, "truncated": 0}}
            (root / "part62-run-all.json").write_text(json.dumps({"reader": "haiku",
                "model": blind62.READER_MODELS["haiku"], "calibration": failed}))
            row = {"model": "qwen35-4b", "phase": "both", "kind": "twin", "dose": 1.0,
                   "question": "Who made you?", "cond": "twin", "unit": 1, "seed": 1, "i": 0,
                   "text": "I am an AI."}
            request = {"custom_id": "r00000", "params": {"model": blind62.READER_MODELS["opus"],
                       "max_tokens": 1024, "messages": [{"role": "user", "content": "answer"}]}}
            preflight = (root, {"r00000": "ans|0"}, {"ans|0": row}, [request])
            body = '{"type":"error","error":{"type":"invalid_request_error"}}'
            def reject(method, path, payload=None):
                if method == "POST":
                    raise blind62.setting.APIHTTPError(400, body)
                raise AssertionError(path)
            real_persist = blind62._persist_run
            def fail_after_release(path, run):
                if any(chunk.get("status") == "rejected" for chunk in run.get("chunks", [])):
                    raise OSError("simulated process loss after release")
                real_persist(path, run)
            class Accept:
                def __call__(self, method, path, payload=None):
                    if method == "POST":
                        return json.dumps({"id": "batch-recovered"})
                    if path.endswith("/results"):
                        return json.dumps({"custom_id": "r00000", "result": {"type": "succeeded",
                            "message": {"content": [{"type": "text", "text": "EARTH OK"}],
                                        "stop_reason": "end_turn",
                                        "usage": {"input_tokens": 5, "output_tokens": 2}}}})
                    return json.dumps({"processing_status": "ended"})
            with mock.patch.object(blind62, "_preflight_submission", return_value=preflight):
                with mock.patch.object(blind62, "_persist_run", side_effect=fail_after_release):
                    with self.assertRaisesRegex(OSError, "process loss"):
                        blind62.run_all(root, root, reader="opus", ledger_path=root / "ledger.jsonl",
                                        api=reject, sleep=lambda _: None)
                saved = json.loads((root / "part62-run-all-opus.json").read_text())
                self.assertEqual(saved["chunks"][0]["status"], "posting")
                self.assertEqual(list(blind62._ledger_attempts(root / "ledger.jsonl").values())[0]["status"],
                                 "released")
                output = blind62.run_all(root, root, reader="opus", ledger_path=root / "ledger.jsonl",
                                         api=Accept(), sleep=lambda _: None)
            self.assertEqual(json.loads(output.read_text())["labels"]["ans|0"], ["EARTH", "OK"])
            run = json.loads((root / "part62-run-all-opus.json").read_text())
            self.assertEqual([chunk["status"] for chunk in run["chunks"]], ["rejected", "settled"])

    def test_retired_answer_prints_and_accepts_explicit_retry(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            failed = {"status": "failed", "reader": "haiku", "model": blind62.READER_MODELS["haiku"],
                      "calibration": {"count": 120, "kappa": 0.7, "unlabelled": 0, "truncated": 0}}
            (root / "part62-run-all.json").write_text(json.dumps({"reader": "haiku",
                "model": blind62.READER_MODELS["haiku"], "calibration": failed}))
            row = {"model": "qwen35-4b", "phase": "both", "kind": "twin", "dose": 1.0,
                   "question": "Who made you?", "cond": "twin", "unit": 1, "seed": 1, "i": 0,
                   "text": "I am an AI."}
            request = {"custom_id": "r00000", "params": {"model": blind62.READER_MODELS["opus"],
                       "max_tokens": 1024, "messages": [{"role": "user", "content": "answer"}]}}
            preflight = (root, {"r00000": "ans|0"}, {"ans|0": row}, [request])
            class Unknown:
                def __call__(self, method, path, payload=None):
                    if method == "POST":
                        raise RuntimeError("socket reset after send")
                    raise AssertionError(path)
            class Accept:
                posts = 0
                def __call__(self, method, path, payload=None):
                    if method == "POST":
                        Accept.posts += 1
                        return json.dumps({"id": "batch-retry"})
                    if path.endswith("/results"):
                        return json.dumps({"custom_id": "r00000", "result": {"type": "succeeded",
                            "message": {"content": [{"type": "text", "text": "EARTH OK"}],
                                        "stop_reason": "end_turn",
                                        "usage": {"input_tokens": 5, "output_tokens": 2}}}})
                    return json.dumps({"processing_status": "ended"})
            ledger_path = root / "ledger.jsonl"
            with mock.patch.object(blind62, "_preflight_submission", return_value=preflight):
                with self.assertRaisesRegex(RuntimeError, "socket reset"):
                    blind62.run_all(root, root, reader="opus", ledger_path=ledger_path,
                                    api=Unknown(), sleep=lambda _: None)
                run = json.loads((root / "part62-run-all-opus.json").read_text())
                attempt_id = run["chunks"][0]["attempt_id"]
                with self.assertRaisesRegex(ValueError, "provider batch carrying.*--no-batch-verified"):
                    blind62.retire_attempt(attempt_id, "OPERATOR", "verified unknown outcome",
                                           ledger_path, root)
                blind62.retire_attempt(attempt_id, "OPERATOR", "verified unknown outcome",
                                       ledger_path, root, no_batch_verified=True)
                with self.assertRaisesRegex(ValueError, "--retry-answers"):
                    blind62.run_all(root, root, reader="opus", ledger_path=ledger_path,
                                    api=lambda *_args, **_kwargs: self.fail("must wait for explicit retry"),
                                    sleep=lambda _: None)
                output = blind62.run_all(root, root, reader="opus", ledger_path=ledger_path,
                                         api=Accept(), sleep=lambda _: None, retry_answers=True)
            self.assertEqual(Accept.posts, 1)
            self.assertEqual(json.loads(output.read_text())["scoring_allowed"], True)
            self.assertEqual(sorted(attempt["status"] for attempt in
                                    blind62._ledger_attempts(ledger_path).values()), ["retired", "settled"])
            with mock.patch.object(blind62, "run_all") as run_all:
                blind62.main(["run-all", "--retry-answers"])
            self.assertTrue(run_all.call_args.kwargs["retry_answers"])

    def test_retire_refuses_submitted_batch_attempt(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            failed = {"status": "failed", "reader": "haiku", "model": blind62.READER_MODELS["haiku"],
                      "calibration": {"count": 120, "kappa": 0.7, "unlabelled": 0, "truncated": 0}}
            (root / "part62-run-all.json").write_text(json.dumps({"reader": "haiku",
                "model": blind62.READER_MODELS["haiku"], "calibration": failed}))
            row = {"model": "qwen35-4b", "phase": "both", "kind": "twin", "dose": 1.0,
                   "question": "Who made you?", "cond": "twin", "unit": 1, "seed": 1, "i": 0,
                   "text": "I am an AI."}
            request = {"custom_id": "r00000", "params": {"model": blind62.READER_MODELS["opus"],
                       "max_tokens": 1024, "messages": [{"role": "user", "content": "answer"}]}}
            preflight = (root, {"r00000": "ans|0"}, {"ans|0": row}, [request])
            with mock.patch.object(blind62, "_preflight_submission", return_value=preflight), \
                 mock.patch.object(blind62, "_settle_recorded", side_effect=RuntimeError("stopped after submit")):
                with self.assertRaisesRegex(RuntimeError, "stopped after submit"):
                    blind62.run_all(root, root, reader="opus", ledger_path=root / "ledger.jsonl",
                                    api=lambda method, *_args, **_kwargs: json.dumps({"id": "known-batch"}),
                                    sleep=lambda _: None)
            run = json.loads((root / "part62-run-all-opus.json").read_text())
            attempt_id = run["chunks"][0]["attempt_id"]
            with self.assertRaisesRegex(ValueError, "batch id is recorded"):
                blind62.retire_attempt(attempt_id, "OPERATOR", "verified unknown outcome",
                                       root / "ledger.jsonl", root)
            self.assertEqual(list(blind62._ledger_attempts(root / "ledger.jsonl").values())[0]["status"],
                             "reserved")

    def test_retire_requires_part62_kind_and_exact_run_chunk_mapping(self):
        from paidaction.paid_ledger import Ledger

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            review = root / "review"
            review.mkdir()
            ledger_path = root / "ledger.jsonl"
            ledger = Ledger(ledger_path, 8.0)

            foreign = ledger.reserve(0.25, "part63", {"legacy": True})
            with self.assertRaisesRegex(ValueError, "non-Part-62 attempt"):
                blind62.retire_attempt(foreign, "OPERATOR", "scope probe", ledger_path, review)
            self.assertEqual(blind62._ledger_attempts(ledger_path)[foreign]["status"], "reserved")

            missing_id = ledger.reserve(0.25, "part62-haiku", {})
            with self.assertRaisesRegex(ValueError, "no submission id"):
                blind62.retire_attempt(missing_id, "OPERATOR", "scope probe", ledger_path, review)
            self.assertEqual(blind62._ledger_attempts(ledger_path)[missing_id]["status"], "reserved")

            orphan = ledger.reserve(0.25, "part62-opus", {"submission_id": "orphan-submission"})
            (review / "part62-run-all-opus.json").write_text(json.dumps({
                "reader": "opus", "chunks": [{"attempt_id": "some-other-attempt",
                    "submission_id": "orphan-submission", "status": "posting"}]}))
            with self.assertRaisesRegex(ValueError, "does not map to exactly one chunk"):
                blind62.retire_attempt(orphan, "OPERATOR", "scope probe", ledger_path, review)
            self.assertEqual(blind62._ledger_attempts(ledger_path)[orphan]["status"], "reserved")

            submission_id = "mapped-submission"
            mapped = ledger.reserve(0.25, "part62-opus", {"submission_id": submission_id})
            (review / "part62-run-all-opus.json").write_text(json.dumps({
                "reader": "opus", "chunks": [{"attempt_id": mapped,
                    "submission_id": submission_id, "status": "posting"}]}))
            with self.assertRaisesRegex(ValueError, "provider batch carrying.*--no-batch-verified"):
                blind62.retire_attempt(mapped, "OPERATOR", "verified unknown outcome", ledger_path, review)
            self.assertEqual(blind62._ledger_attempts(ledger_path)[mapped]["status"], "reserved")
            blind62.retire_attempt(mapped, "OPERATOR", "verified unknown outcome", ledger_path, review,
                                   no_batch_verified=True)
            self.assertEqual(blind62._ledger_attempts(ledger_path)[mapped]["status"], "retired")

    def test_retired_calibration_printed_command_retries_with_new_reserve(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            calibration_requests = []
            idmap = {}
            expected = {}
            for index, (label, _answer_id, _text) in enumerate(blind62.CAL):
                custom_id = f"r{index:05d}"
                internal_id = f"cal|{index}|{label}|ORIGIN"
                idmap[custom_id] = internal_id
                expected[custom_id] = label
                calibration_requests.append({"custom_id": custom_id, "params": {
                    "model": blind62.READER_MODELS["haiku"], "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "calibration"}]}})
            answer_request = {"custom_id": "r00120", "params": {
                "model": blind62.READER_MODELS["haiku"], "max_tokens": 1024,
                "messages": [{"role": "user", "content": "answer"}]}}
            idmap["r00120"] = "ans|0"
            requests = calibration_requests + [answer_request]
            row = {"model": "qwen35-4b", "phase": "both", "kind": "twin", "dose": 1.0,
                   "question": "Who made you?", "cond": "twin", "unit": 1, "seed": 1, "i": 0,
                   "text": "I am an AI."}
            fingerprint = __import__("hashlib").sha256(
                json.dumps(requests, sort_keys=True).encode()).hexdigest()
            ledger_path = root / "ledger.jsonl"
            ledger = blind62._StageLedger(ledger_path, blind62.HARD_CAP_USD)
            retired_attempt = ledger.reserve(0.5, blind62.READER_KINDS["haiku"], {"stage": "calibration"})
            ledger.retire(retired_attempt, "OPERATOR", "verified unknown outcome")
            run = {"schema": "part62-run-all-v1", "run_id": "retired-test", "reader": "haiku",
                   "model": blind62.READER_MODELS["haiku"], "fingerprint": fingerprint,
                   "status": "calibration-pending", "chunks": [{
                       "stage": "calibration", "index": 0, "submission_id": "retired-submission",
                       "status": "ambiguous", "attempt_id": retired_attempt,
                       "custom_ids": [row["custom_id"] for row in calibration_requests],
                       "request_count": len(calibration_requests), "max_tokens": 1024}]}
            (root / "part62-run-all.json").write_text(json.dumps(run))

            class API:
                def __init__(self):
                    self.posts = {}
                def __call__(self, method, path, body=None):
                    if method == "POST":
                        batch_id = f"batch-{len(self.posts) + 1}"
                        self.posts[batch_id] = [item["custom_id"] for item in body["requests"]]
                        return json.dumps({"id": batch_id})
                    if path.endswith("/results"):
                        batch_id = path.split("/")[-2]
                        lines = []
                        for custom_id in self.posts[batch_id]:
                            label = expected.get(_original_custom_id(custom_id), "NEITHER")
                            lines.append(json.dumps({"custom_id": custom_id, "result": {
                                "type": "succeeded", "message": {
                                    "content": [{"type": "text", "text": f"{label} OK"}],
                                    "stop_reason": "end_turn",
                                    "usage": {"input_tokens": 10, "output_tokens": 2}}}}))
                        return "\n".join(lines)
                    return json.dumps({"processing_status": "ended", "request_counts": {}})

            api = API()
            preflight = (root, idmap, {"ans|0": row}, requests)
            with mock.patch.object(blind62, "_preflight_submission", return_value=preflight):
                with self.assertRaisesRegex(ValueError, "--retry-calibration") as refusal:
                    blind62.run_all(root, root, ledger_path=ledger_path, api=api,
                                    sleep=lambda _: None, poll_interval=0, poll_bound=1)
                self.assertIn("--retry-calibration", str(refusal.exception))
                self.assertEqual(api.posts, {})
                output = blind62.run_all(root, root, ledger_path=ledger_path, api=api,
                                         sleep=lambda _: None, poll_interval=0, poll_bound=1,
                                         retry_calibration=True)
            self.assertEqual(len(api.posts), 2)
            saved_run = json.loads((root / "part62-run-all.json").read_text())
            calibration = saved_run["calibration"]
            expected_sample_ids = [item[1] for item in blind62.CAL]
            expected_counts = {label: sum(item[0] == label for item in blind62.CAL)
                               for label in blind62.CALIBRATION_CLASSES}
            self.assertEqual(calibration["seed"], blind62.CALIBRATION_SEED)
            self.assertEqual(calibration["sample_ids"], expected_sample_ids)
            self.assertEqual(calibration["realized_class_counts"], expected_counts)
            labels_file = json.loads(output.read_text())
            self.assertEqual(labels_file["calibration"]["seed"], blind62.CALIBRATION_SEED)
            self.assertEqual(labels_file["calibration"]["sample_ids"], expected_sample_ids)
            self.assertEqual(labels_file["calibration"]["realized_class_counts"], expected_counts)
            attempts = blind62._ledger_attempts(ledger_path)
            self.assertEqual(sorted(item["status"] for item in attempts.values()),
                             ["retired", "settled", "settled"])
            self.assertEqual(saved_run["chunks"][0]["status"], "retired-unknown")
            self.assertEqual([chunk["index"] for chunk in saved_run["chunks"]
                              if chunk["stage"] == "calibration"], [0, 1])

            with mock.patch.object(blind62, "run_all") as run_all:
                blind62.main(["run-all", "--folder", str(root), "--review-dir", str(root),
                              "--ledger-path", str(ledger_path), "--retry-calibration"])
            self.assertTrue(run_all.call_args.kwargs["retry_calibration"])

    def test_errored_calibration_is_incomplete_locked_and_retryable(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            row = {"model": "qwen35-4b", "phase": "both", "kind": "twin", "dose": 1.0,
                   "question": "Who made you?", "cond": "twin", "unit": 1, "seed": 1, "i": 0,
                   "text": "I am an AI."}
            idmap, keymap, requests = blind62.prepare_batch([row], "haiku")
            first_calibration_custom = next(custom_id for custom_id, internal_id in idmap.items()
                                           if internal_id.startswith("cal|"))
            posts = []
            batches = {}
            first_calibration = True
            class API:
                def __call__(self, method, path, payload=None):
                    nonlocal first_calibration
                    if method == "POST":
                        batch_id = f"calibration-{len(posts) + 1}"
                        posts.append(len(payload["requests"]))
                        lines = []
                        is_calibration = len(payload["requests"]) == blind62.CALIBRATION_SAMPLE_SIZE
                        for request in payload["requests"]:
                            internal_id = idmap[_original_custom_id(request["custom_id"])]
                            if (is_calibration and first_calibration
                                    and _original_custom_id(request["custom_id"]) == first_calibration_custom):
                                result = {"type": "errored", "error": {"type": "api_error"}}
                            else:
                                label = internal_id.split("|")[2] if internal_id.startswith("cal|") else "EARTH"
                                result = {"type": "succeeded", "message": {
                                    "content": [{"type": "text", "text": f"{label} OK"}],
                                    "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 2}}}
                            lines.append(json.dumps({"custom_id": request["custom_id"], "result": result}))
                        if is_calibration:
                            first_calibration = False
                        batches[batch_id] = lines
                        return json.dumps({"id": batch_id})
                    batch_id = path.split("/")[-2]
                    if path.endswith("/results"):
                        return "\n".join(batches[batch_id])
                    return json.dumps({"processing_status": "ended"})
            ledger_path = root / "ledger.jsonl"
            preflight = (root, idmap, keymap, requests)
            with mock.patch.object(blind62, "_preflight_submission", return_value=preflight):
                with self.assertRaisesRegex(ValueError, "calibration incomplete.*--retry-calibration"):
                    blind62.run_all(root, root, ledger_path=ledger_path, api=API(), sleep=lambda _: None)
                run = json.loads((root / "part62-run-all.json").read_text())
                self.assertEqual(run["calibration"]["status"], "incomplete")
                self.assertFalse(blind62._haiku_calibration_failed(root))
                self.assertEqual(posts, [blind62.CALIBRATION_SAMPLE_SIZE])
                with mock.patch.object(blind62, "_preflight_submission") as opus_preflight:
                    with self.assertRaisesRegex(ValueError, "recorded failed Haiku calibration"):
                        blind62.run_all(root, root, reader="opus", ledger_path=ledger_path,
                                        api=API(), sleep=lambda _: None)
                    opus_preflight.assert_not_called()
                output = blind62.run_all(root, root, ledger_path=ledger_path, api=API(),
                                         sleep=lambda _: None, retry_calibration=True)
            self.assertEqual(posts, [blind62.CALIBRATION_SAMPLE_SIZE,
                                     blind62.CALIBRATION_SAMPLE_SIZE, 1])
            self.assertTrue(json.loads(output.read_text())["scoring_allowed"])
            self.assertEqual(json.loads((root / "part62-run-all.json").read_text())["calibration"]["status"],
                             "passed")

    def test_retired_rejected_calibration_reconciles_and_retries(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            row = {"model": "qwen35-4b", "phase": "both", "kind": "twin", "dose": 1.0,
                   "question": "Who made you?", "cond": "twin", "unit": 1, "seed": 1, "i": 0,
                   "text": "I am an AI."}
            idmap, keymap, requests = blind62.prepare_batch([row], "haiku")
            preflight = (root, idmap, keymap, requests)
            ledger_path = root / "ledger.jsonl"
            class RejectUnparsed:
                def __call__(self, method, path, payload=None):
                    if method == "POST":
                        raise blind62.setting.APIHTTPError(400, "<proxy-error>invalid body</proxy-error>")
                    raise AssertionError(path)
            posts = []
            batches = {}
            class Accept:
                def __call__(self, method, path, payload=None):
                    if method == "POST":
                        batch_id = f"retry-{len(posts) + 1}"
                        posts.append(len(payload["requests"]))
                        lines = []
                        for request in payload["requests"]:
                            internal_id = idmap[_original_custom_id(request["custom_id"])]
                            label = internal_id.split("|")[2] if internal_id.startswith("cal|") else "EARTH"
                            lines.append(json.dumps({"custom_id": request["custom_id"], "result": {
                                "type": "succeeded", "message": {
                                    "content": [{"type": "text", "text": f"{label} OK"}],
                                    "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 2}}}}))
                        batches[batch_id] = lines
                        return json.dumps({"id": batch_id})
                    batch_id = path.split("/")[-2]
                    if path.endswith("/results"):
                        return "\n".join(batches[batch_id])
                    return json.dumps({"processing_status": "ended"})
            with mock.patch.object(blind62, "_preflight_submission", return_value=preflight):
                with self.assertRaises(blind62.setting.APIHTTPError):
                    blind62.run_all(root, root, ledger_path=ledger_path, api=RejectUnparsed(),
                                    sleep=lambda _: None)
                rejected_run = json.loads((root / "part62-run-all.json").read_text())
                rejected = rejected_run["chunks"][0]
                self.assertEqual(rejected["status"], "rejected")
                self.assertFalse(rejected["reserve_released"])
                blind62.retire_attempt(rejected["attempt_id"], "OPERATOR", "reconciled 400",
                                       ledger_path, root)
                with self.assertRaisesRegex(ValueError, "--retry-calibration"):
                    blind62.run_all(root, root, ledger_path=ledger_path,
                                    api=lambda *_args, **_kwargs: self.fail("retry must stay explicit"),
                                    sleep=lambda _: None)
                reconciled = json.loads((root / "part62-run-all.json").read_text())
                self.assertEqual(reconciled["chunks"][0]["status"], "retired-unknown")
                output = blind62.run_all(root, root, ledger_path=ledger_path, api=Accept(),
                                         sleep=lambda _: None, retry_calibration=True)
            self.assertEqual(posts, [blind62.CALIBRATION_SAMPLE_SIZE, 1])
            self.assertTrue(json.loads(output.read_text())["scoring_allowed"])
            self.assertEqual(sorted(attempt["status"] for attempt in
                                    blind62._ledger_attempts(ledger_path).values()),
                             ["retired", "settled", "settled"])

    def test_failed_answer_rows_can_be_retried_without_reposting_successes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            failed = {"status": "failed", "reader": "haiku", "model": blind62.READER_MODELS["haiku"],
                      "calibration": {"count": 120, "kappa": 0.7, "unlabelled": 0, "truncated": 0}}
            (root / "part62-run-all.json").write_text(json.dumps({"reader": "haiku",
                "model": blind62.READER_MODELS["haiku"], "calibration": failed}))
            rows = [{"model": "qwen35-4b", "phase": "both", "kind": "twin", "dose": 1.0,
                     "question": "Who made you?", "cond": "twin", "unit": 1, "seed": 1, "i": i,
                     "text": f"answer {i}"} for i in range(3)]
            idmap = {f"r{i:05d}": f"ans|{i}" for i in range(3)}
            keymap = {f"ans|{i}": row for i, row in enumerate(rows)}
            requests = [{"custom_id": custom_id, "params": {"model": blind62.READER_MODELS["opus"],
                        "max_tokens": 1024, "messages": [{"role": "user", "content": "answer"}]}}
                        for custom_id in idmap]
            posts = []
            batches = {}
            class API:
                def __call__(self, method, path, payload=None):
                    if method == "POST":
                        batch_id = f"answer-{len(posts) + 1}"
                        posts.append([_original_custom_id(item["custom_id"])
                                      for item in payload["requests"]])
                        lines = []
                        for item in payload["requests"]:
                            failed_row = (len(posts) == 1
                                          and _original_custom_id(item["custom_id"]) == "r00001")
                            result = ({"type": "errored", "error": {"type": "api_error"}}
                                      if failed_row else {"type": "succeeded", "message": {
                                          "content": [{"type": "text", "text": "EARTH OK"}],
                                          "stop_reason": "end_turn",
                                          "usage": {"input_tokens": 5, "output_tokens": 2}}})
                            lines.append(json.dumps({"custom_id": item["custom_id"], "result": result}))
                        batches[batch_id] = lines
                        return json.dumps({"id": batch_id})
                    batch_id = path.split("/")[-2]
                    if path.endswith("/results"):
                        return "\n".join(batches[batch_id])
                    return json.dumps({"processing_status": "ended"})
            preflight = (root, idmap, keymap, requests)
            ledger_path = root / "ledger.jsonl"
            with mock.patch.object(blind62, "_preflight_submission", return_value=preflight):
                with self.assertRaisesRegex(ValueError, "--retry-failed-answers"):
                    blind62.run_all(root, root, reader="opus", ledger_path=ledger_path,
                                    api=API(), sleep=lambda _: None)
                self.assertEqual(posts, [["r00000", "r00001", "r00002"]])
                output = blind62.run_all(root, root, reader="opus", ledger_path=ledger_path,
                                         api=API(), sleep=lambda _: None, retry_failed_answers=True)
            self.assertEqual(posts, [["r00000", "r00001", "r00002"], ["r00001"]])
            result = json.loads(output.read_text())
            self.assertTrue(result["scoring_allowed"])
            self.assertEqual(result["labels"], {"ans|0": ["EARTH", "OK"],
                                                 "ans|1": ["EARTH", "OK"],
                                                 "ans|2": ["EARTH", "OK"]})
            self.assertEqual(result["completion"]["issues"], {})
            retry_chunks = [chunk for chunk in result["paid_chunks"]
                            if chunk.get("retry_failed_answers") is True]
            self.assertEqual(len(retry_chunks), 1)
            self.assertEqual(retry_chunks[0]["custom_ids"], ["r00001"])

    def test_retry_failed_answer_rows_are_retried_at_most_once_per_run(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            failed = {"status": "failed", "reader": "haiku", "model": blind62.READER_MODELS["haiku"],
                      "calibration": {"count": 120, "kappa": 0.7, "unlabelled": 0, "truncated": 0}}
            (root / "part62-run-all.json").write_text(json.dumps({"reader": "haiku",
                "model": blind62.READER_MODELS["haiku"], "calibration": failed}))
            row = {"model": "qwen35-4b", "phase": "both", "kind": "twin", "dose": 1.0,
                   "question": "Who made you?", "cond": "twin", "unit": 1, "seed": 1, "i": 0,
                   "text": "answer that always truncates"}
            idmap = {"r00000": "ans|0"}
            keymap = {"ans|0": row}
            requests = [{"custom_id": "r00000", "params": {
                "model": blind62.READER_MODELS["opus"], "max_tokens": 1024,
                "messages": [{"role": "user", "content": "answer"}]}}]
            preflight = (root, idmap, keymap, requests)

            class AlwaysTruncates:
                def __init__(self):
                    self.posts = []
                    self.batches = {}

                def __call__(self, method, path, payload=None):
                    if method == "POST":
                        batch_id = f"truncating-{len(self.posts) + 1}"
                        self.posts.append([_original_custom_id(item["custom_id"])
                                           for item in payload["requests"]])
                        self.batches[batch_id] = [json.dumps({"custom_id": item["custom_id"],
                            "result": {"type": "succeeded", "message": {
                                "content": [{"type": "text", "text": "EARTH OK"}],
                                "stop_reason": "max_tokens",
                                "usage": {"input_tokens": 5, "output_tokens": 1024}}}})
                            for item in payload["requests"]]
                        return json.dumps({"id": batch_id})
                    batch_id = path.split("/")[-2]
                    if path.endswith("/results"):
                        return "\n".join(self.batches[batch_id])
                    return json.dumps({"processing_status": "ended"})

            api = AlwaysTruncates()
            ledger_path = root / "ledger.jsonl"
            with mock.patch.object(blind62, "_preflight_submission", return_value=preflight):
                with self.assertRaisesRegex(ValueError, "--retry-failed-answers"):
                    blind62.run_all(root, root, reader="opus", ledger_path=ledger_path,
                                    api=api, sleep=lambda _: None)
                with self.assertRaisesRegex(ValueError, "--retry-failed-answers"):
                    blind62.run_all(root, root, reader="opus", ledger_path=ledger_path,
                                    api=api, sleep=lambda _: None, retry_failed_answers=True)
                with self.assertRaisesRegex(ValueError, "already retried every unresolved row once"):
                    blind62.run_all(root, root, reader="opus", ledger_path=ledger_path,
                                    api=api, sleep=lambda _: None, retry_failed_answers=True)
            self.assertEqual(api.posts, [["r00000"], ["r00000"]])
            run = json.loads((root / "part62-run-all-opus.json").read_text())
            retry_chunks = [chunk for chunk in run["chunks"]
                            if chunk.get("retry_failed_answers") is True]
            self.assertEqual(len(retry_chunks), 1)
            self.assertEqual(retry_chunks[0]["custom_ids"], ["r00000"])
            attempts = blind62._ledger_attempts(ledger_path)
            self.assertEqual(sum(item["status"] == "settled" for item in attempts.values()), 2)

    def test_post_response_save_failure_records_batch_id_and_does_not_repost(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            failed = {"status": "failed", "reader": "haiku", "model": blind62.READER_MODELS["haiku"],
                      "calibration": {"count": 120, "kappa": 0.7, "unlabelled": 0, "truncated": 0}}
            (root / "part62-run-all.json").write_text(json.dumps({"reader": "haiku",
                "model": blind62.READER_MODELS["haiku"], "calibration": failed}))
            row = {"model": "qwen35-4b", "phase": "both", "kind": "twin", "dose": 1.0,
                   "question": "Who made you?", "cond": "twin", "unit": 1, "seed": 1, "i": 0,
                   "text": "answer"}
            idmap = {"r00000": "ans|0"}
            keymap = {"ans|0": row}
            requests = [{"custom_id": "r00000", "params": {
                "model": blind62.READER_MODELS["opus"], "max_tokens": 1024,
                "messages": [{"role": "user", "content": "answer"}]}}]
            preflight = (root, idmap, keymap, requests)

            class API:
                def __init__(self):
                    self.posts = []
                    self.batches = {}

                def __call__(self, method, path, payload=None):
                    if method == "POST":
                        batch_id = f"save-failure-{len(self.posts) + 1}"
                        self.posts.append([_original_custom_id(item["custom_id"])
                                           for item in payload["requests"]])
                        self.batches[batch_id] = [json.dumps({"custom_id": item["custom_id"],
                            "result": {"type": "succeeded", "message": {
                                "content": [{"type": "text", "text": "EARTH OK"}],
                                "stop_reason": "end_turn",
                                "usage": {"input_tokens": 5, "output_tokens": 2}}}})
                            for item in payload["requests"]]
                        return json.dumps({"id": batch_id})
                    batch_id = path.split("/")[-2]
                    if path.endswith("/results"):
                        return "\n".join(self.batches[batch_id])
                    return json.dumps({"processing_status": "ended"})

            api = API()
            original_save = blind62._save_text

            def fail_post_response_save(path, text):
                if path.name.startswith("post-response-"):
                    raise OSError("simulated response disk failure")
                return original_save(path, text)

            with mock.patch.object(blind62, "_preflight_submission", return_value=preflight), \
                 mock.patch.object(blind62, "_save_text", side_effect=fail_post_response_save):
                output = blind62.run_all(root, root, reader="opus", ledger_path=root / "ledger.jsonl",
                                         api=api, sleep=lambda _: None)
                resumed = blind62.run_all(root, root, reader="opus", ledger_path=root / "ledger.jsonl",
                                          api=api, sleep=lambda _: None, retry_answers=True)
            self.assertEqual(output, resumed)
            self.assertEqual(api.posts, [["r00000"]])
            run = json.loads((root / "part62-run-all-opus.json").read_text())
            chunk = run["chunks"][0]
            self.assertEqual(chunk["status"], "settled")
            self.assertEqual(chunk["batch_id"], "save-failure-1")
            self.assertFalse(chunk["post_response_saved"])
            self.assertIn("simulated response disk failure", chunk["post_response_save_error"])
            self.assertTrue(json.loads(output.read_text())["scoring_allowed"])

    def test_post_response_and_batch_id_save_failure_prints_raw_response(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            api, request, idmap, run = self._fixture(root)
            real_persist = blind62._persist_run
            persisted_ids = []

            def fail_raw_save(_path, _text):
                raise OSError("simulated response disk failure")

            def fail_batch_id_record(path, current_run):
                submitted = [chunk for chunk in current_run.get("chunks", [])
                             if chunk.get("status") == "submitted"]
                if submitted and submitted[-1].get("post_response_saved") is False:
                    persisted_ids.append(submitted[-1].get("batch_id"))
                    raise OSError("simulated run record failure")
                return real_persist(path, current_run)

            stderr = io.StringIO()
            with mock.patch.object(blind62, "_save_text", side_effect=fail_raw_save), \
                 mock.patch.object(blind62, "_persist_run", side_effect=fail_batch_id_record), \
                 contextlib.redirect_stderr(stderr):
                with self.assertRaisesRegex(OSError, "run record failure"):
                    self._run_chunk(root, api, request, idmap, run)
            self.assertEqual(persisted_ids, ["fake-batch"])
            self.assertIn('{"id": "fake-batch", "processing_status": "in_progress"}', stderr.getvalue())
            self.assertEqual(api.posts, 1)

    def test_saved_post_response_without_run_record_uses_distinct_state_and_never_reposts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            failed = {"status": "failed", "reader": "haiku", "model": blind62.READER_MODELS["haiku"],
                      "calibration": {"count": 120, "kappa": 0.7, "unlabelled": 0, "truncated": 0}}
            (root / "part62-run-all.json").write_text(json.dumps({"reader": "haiku",
                "model": blind62.READER_MODELS["haiku"], "calibration": failed}))
            row = {"model": "qwen35-4b", "phase": "both", "kind": "twin", "dose": 1.0,
                   "question": "Who made you?", "cond": "twin", "unit": 1, "seed": 1, "i": 0,
                   "text": "answer"}
            idmap = {"r00000": "ans|0"}
            keymap = {"ans|0": row}
            requests = [{"custom_id": "r00000", "params": {
                "model": blind62.READER_MODELS["opus"], "max_tokens": 1024,
                "messages": [{"role": "user", "content": "answer"}]}}]
            preflight = (root, idmap, keymap, requests)

            class API:
                def __init__(self):
                    self.posts = []
                    self.batches = {}

                def __call__(self, method, path, payload=None):
                    if method == "POST":
                        batch_id = f"record-failure-{len(self.posts) + 1}"
                        self.posts.append([_original_custom_id(item["custom_id"])
                                           for item in payload["requests"]])
                        self.batches[batch_id] = [json.dumps({"custom_id": item["custom_id"],
                            "result": {"type": "succeeded", "message": {
                                "content": [{"type": "text", "text": "EARTH OK"}],
                                "stop_reason": "end_turn",
                                "usage": {"input_tokens": 5, "output_tokens": 2}}}})
                            for item in payload["requests"]]
                        return json.dumps({"id": batch_id})
                    batch_id = path.split("/")[-2]
                    if path.endswith("/results"):
                        return "\n".join(self.batches[batch_id])
                    return json.dumps({"processing_status": "ended"})

            api = API()
            real_persist = blind62._persist_run
            did_fail = False

            def fail_submitted_run_once(path, run):
                nonlocal did_fail
                if (not did_fail and any(chunk.get("stage") == "answers"
                                         and chunk.get("status") == "submitted"
                                         for chunk in run.get("chunks", []))):
                    did_fail = True
                    raise OSError("simulated failure recording submitted batch")
                return real_persist(path, run)

            ledger_path = root / "ledger.jsonl"
            with mock.patch.object(blind62, "_preflight_submission", return_value=preflight), \
                 mock.patch.object(blind62, "_persist_run", side_effect=fail_submitted_run_once):
                with self.assertRaisesRegex(OSError, "recording submitted batch"):
                    blind62.run_all(root, root, reader="opus", ledger_path=ledger_path,
                                    api=api, sleep=lambda _: None)
            run = json.loads((root / "part62-run-all-opus.json").read_text())
            chunk = run["chunks"][0]
            self.assertEqual(chunk["status"], "response-saved")
            self.assertEqual(chunk["batch_id"], "record-failure-1")
            with self.assertRaisesRegex(ValueError, "batch id is recorded"):
                blind62.retire_attempt(chunk["attempt_id"], "OPERATOR", "scope probe", ledger_path, root)

            with mock.patch.object(blind62, "_preflight_submission", return_value=preflight):
                output = blind62.run_all(root, root, reader="opus", ledger_path=ledger_path,
                                         api=api, sleep=lambda _: None, retry_answers=True)
            self.assertEqual(api.posts, [["r00000"]])
            self.assertTrue(json.loads(output.read_text())["scoring_allowed"])

    def test_raw_post_response_text_is_saved_before_json_parsing(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            _api, request, idmap, run = self._fixture(root)
            raw = "<html>gateway failure</html>\n"
            def malformed_api(method, path, body=None):
                if method == "POST":
                    return raw
                self.fail(f"unexpected API call {method} {path}")
            with self.assertRaisesRegex(ValueError, "response is not valid JSON"):
                self._run_chunk(root, malformed_api, request, idmap, run)
            chunk = run["chunks"][0]
            self.assertEqual(chunk["status"], "ambiguous")
            self.assertEqual((root / f"post-response-{chunk['submission_id']}.txt").read_text(), raw)

    def test_run62_signal_traps_archive_and_report_failure(self):
        for sig, status in ((signal.SIGINT, 130), (signal.SIGTERM, 143), (signal.SIGHUP, 129)):
            with self.subTest(signal=sig), tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
                root = Path(temporary)
                part62_dir = root / "part62"
                bin_dir = root / "bin"
                part62_dir.mkdir()
                bin_dir.mkdir()
                shutil.copyfile(Path(__file__).with_name("run62.sh"), part62_dir / "run62.sh")
                fake_python = bin_dir / "python3"
                fake_python.write_text("#!/usr/bin/env bash\nprintf started > \"$FAKE_MARKER\"\nexec /bin/sleep 30\n")
                fake_python.chmod(0o755)
                fake_mkdir = bin_dir / "mkdir"
                fake_mkdir.write_text("#!/usr/bin/env bash\nexit 0\n")
                fake_mkdir.chmod(0o755)
                marker = root / "python-started"
                archive = root / "incomplete.tgz"
                env = dict(os.environ)
                env.update({"PATH": f"{bin_dir}:/usr/bin:/bin", "FAKE_MARKER": str(marker),
                            "RETURN_INCOMPLETE_ARCHIVE": str(archive)})
                process = subprocess.Popen(["bash", str(part62_dir / "run62.sh")], cwd=root,
                                           env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           text=True, start_new_session=True)
                deadline = time.monotonic() + 5
                while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(marker.exists(), "runner did not reach the fetch stub")
                os.killpg(process.pid, sig)
                _stdout, stderr = process.communicate(timeout=8)
                self.assertEqual(process.returncode, status)
                self.assertIn(f"RUN_FAILED status={status}", stderr)
                self.assertTrue(archive.is_file())
                self.assertGreater(archive.stat().st_size, 0)

    def test_run62_shell_only_term_and_hup_forward_to_waited_child(self):
        for sig, status in ((signal.SIGTERM, 143), (signal.SIGHUP, 129)):
            with self.subTest(signal=sig), tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
                root = Path(temporary)
                part62_dir = root / "part62"
                bin_dir = root / "bin"
                part62_dir.mkdir()
                bin_dir.mkdir()
                shutil.copyfile(Path(__file__).with_name("run62.sh"), part62_dir / "run62.sh")
                fake_python = bin_dir / "python3"
                fake_python.write_text("#!/usr/bin/env bash\nprintf started > \"$FAKE_MARKER\"\nexec /bin/sleep 8\n")
                fake_python.chmod(0o755)
                fake_mkdir = bin_dir / "mkdir"
                fake_mkdir.write_text("#!/usr/bin/env bash\nexit 0\n")
                fake_mkdir.chmod(0o755)
                marker = root / "python-started"
                archive = root / "incomplete.tgz"
                env = dict(os.environ)
                env.update({"PATH": f"{bin_dir}:/usr/bin:/bin", "FAKE_MARKER": str(marker),
                            "RETURN_INCOMPLETE_ARCHIVE": str(archive)})
                process = subprocess.Popen(["bash", str(part62_dir / "run62.sh")], cwd=root,
                                           env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           text=True, start_new_session=True)
                deadline = time.monotonic() + 5
                while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(marker.exists(), "runner did not reach the child process")
                started = time.monotonic()
                os.kill(process.pid, sig)
                _stdout, stderr = process.communicate(timeout=5)
                self.assertLess(time.monotonic() - started, 3.0)
                self.assertEqual(process.returncode, status)
                self.assertIn(f"RUN_FAILED status={status}", stderr)
                self.assertTrue(archive.is_file())
                self.assertGreater(archive.stat().st_size, 0)

    def test_calibration_gate_refuses_missing_pending_and_failed(self):
        for record in (None, {"status": "pending"}, {"status": "failed"}):
            self.assertFalse(blind62._calibration_gate(record))
        passed = {"status": "passed", "reader": "haiku", "model": blind62.READER_MODELS["haiku"],
                  "calibration": {"count": 120, "kappa": 0.8, "unlabelled": 0, "truncated": 0}}
        self.assertTrue(blind62._calibration_gate(passed))
        row = {"model": "qwen35-4b", "phase": "both", "kind": "twin", "dose": 1.0,
               "question": "Who made you?", "cond": "twin", "unit": 1, "seed": 1, "i": 0,
               "text": "I am an AI."}
        _idmap, _keymap, requests = blind62.prepare_batch([row], "haiku")
        self.assertTrue(all("temperature" not in request["params"] for request in requests))

    def test_haiku_answers_refused_for_missing_pending_and_failed_calibration(self):
        row = {"model": "qwen35-4b", "phase": "both", "kind": "twin", "dose": 1.0,
               "question": "Who made you?", "cond": "twin", "unit": 1, "seed": 1, "i": 0,
               "text": "I am an AI."}
        idmap = {"r00000": "ans|0", "r00001": "cal|0|EARTH|ORIGIN"}
        requests = [
            {"custom_id": "r00000", "params": {"model": blind62.READER_MODELS["haiku"], "max_tokens": 1024}},
            {"custom_id": "r00001", "params": {"model": blind62.READER_MODELS["haiku"], "max_tokens": 1024}},
        ]
        good_labels = {f"cal|{i}|{expected}|ORIGIN": (expected, "OK")
                       for i, (expected, _answer_id, _text) in enumerate(blind62.CAL)}
        self.assertGreaterEqual(blind62._calibration_kappa(good_labels), 0.8)
        for state in ("missing", "pending", "failed"):
            with self.subTest(state=state), tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
                root = Path(temporary)
                run_path = root / "part62-run-all.json"
                run = {"schema": "part62-run-all-v1", "run_id": "test", "reader": "haiku",
                       "model": blind62.READER_MODELS["haiku"],
                       "fingerprint": __import__("hashlib").sha256(
                           json.dumps(requests, sort_keys=True).encode()).hexdigest(),
                       "status": "calibration-pending", "chunks": []}
                if state == "pending":
                    run["chunks"].append({"stage": "calibration", "index": 0, "status": "pending",
                                          "batch_id": "calibration-batch", "custom_ids": ["r00001"]})
                elif state == "failed":
                    run["chunks"].append({"stage": "calibration", "index": 0, "status": "settled",
                                          "batch_id": "calibration-batch", "labels": {},
                                          "completion_issues": {}})
                    run["calibration"] = {"status": "failed", "reader": "haiku",
                        "model": blind62.READER_MODELS["haiku"], "calibration": {
                            "count": 120, "kappa": 0.0, "unlabelled": 0, "truncated": 0,
                            "labels": good_labels}}
                run_path.write_text(json.dumps(run))
                stages = []
                def fake_post(chunk, _idmap, saved_run, *_args, **kwargs):
                    stage = _args[6]
                    stages.append(stage)
                    return {"status": "pending", "stage": stage}
                preflight = (root, idmap, {"ans|0": row}, requests)
                with mock.patch.object(blind62, "_preflight_submission", return_value=preflight), \
                     mock.patch.object(blind62, "_post_chunk", side_effect=fake_post):
                    with self.assertRaisesRegex(ValueError, "calibration|answers refused"):
                        blind62.run_all(root, root, ledger_path=root / "ledger.jsonl",
                                        api=lambda *_a, **_kw: self.fail("no API call"))
                self.assertNotIn("answers", stages)

    def test_opus_run_all_default_stays_haiku_and_does_not_start_opus(self):
        with mock.patch.object(blind62, "run_all") as run_all:
            blind62.main(["run-all"])
        self.assertEqual(run_all.call_args.args[2], "haiku")

    def test_opus_runs_answers_only_after_recorded_haiku_failure(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            failed = {"status": "failed", "reader": "haiku", "model": blind62.READER_MODELS["haiku"],
                      "calibration": {"count": 120, "kappa": 0.7, "unlabelled": 0, "truncated": 0}}
            (root / "part62-run-all.json").write_text(json.dumps({"reader": "haiku",
                "model": blind62.READER_MODELS["haiku"], "calibration": failed}))
            row = {"model": "qwen35-4b", "phase": "both", "kind": "twin", "dose": 1.0,
                   "question": "Who made you?", "cond": "twin", "unit": 1, "seed": 1, "i": 0,
                   "text": "I am an AI."}
            idmap = {"r00000": "ans|0", "r00001": "cal|0|EARTH|ORIGIN"}
            requests = [
                {"custom_id": "r00000", "params": {"model": blind62.READER_MODELS["opus"], "max_tokens": 1024}},
                {"custom_id": "r00001", "params": {"model": blind62.READER_MODELS["opus"], "max_tokens": 1024}},
            ]
            calls = []
            def fake_post(chunk_requests, _idmap, run, _run_path, _review_dir, _ledger_path,
                          reader, model, _api, stage, index, **_kwargs):
                calls.append((reader, model, stage, index, len(chunk_requests)))
                run["chunks"].append({"stage": stage, "index": index, "status": "settled",
                    "batch_id": "opus-batch", "labels": {"ans|0": ("NEITHER", False)},
                    "completion_issues": {}})
            with mock.patch.object(blind62, "_preflight_submission",
                                   return_value=(root, idmap, {"ans|0": row}, requests)), \
                 mock.patch.object(blind62, "_post_chunk", side_effect=fake_post):
                output = blind62.run_all(root, root, reader="opus", ledger_path=root / "ledger.jsonl",
                                         api=lambda *_a, **_kw: self.fail("mocked chunk sender"))
            self.assertEqual(calls, [("opus", blind62.READER_MODELS["opus"], "answers", 0, 1)])
            self.assertEqual(output.name, "part62-labels-opus-batch.json")
            record = json.loads(output.read_text())
            self.assertEqual(record["reader"], "opus")
            self.assertIsNone(record["calibration"])
            self.assertEqual(record["unlabelled_answers"], [])

    def _fixture(self, root):
        class API:
            def __init__(self):
                self.posts = 0
                self.fail_status_once = False
            def __call__(self, method, path, body=None):
                if method == "POST":
                    self.posts += 1
                    return json.dumps({"id": "fake-batch", "processing_status": "in_progress"})
                if path.endswith("/results"):
                    return json.dumps({"custom_id": "r00000", "result": {"type": "succeeded",
                        "message": {"content": [{"type": "text", "text": "NEITHER OK"}],
                                    "stop_reason": "end_turn",
                                    "usage": {"input_tokens": 10, "output_tokens": 2}}}})
                if self.fail_status_once:
                    self.fail_status_once = False
                    raise RuntimeError("simulated kill after id recorded")
                return json.dumps({"processing_status": "ended", "request_counts": {}})
        api = API()
        request = {"custom_id": "r00000", "params": {"model": blind62.MODEL,
                    "max_tokens": 1024, "messages": [{"role": "user", "content": "x"}]}}
        idmap = {"r00000": "cal|0|EARTH|ORIGIN"}
        fingerprint = __import__("hashlib").sha256(json.dumps([request], sort_keys=True).encode()).hexdigest()
        run = {"run_id": "run-test", "reader": "haiku", "model": blind62.MODEL,
               "fingerprint": fingerprint, "chunks": []}
        return api, request, idmap, run

    def _run_chunk(self, root, api, request, idmap, run, **kwargs):
        return blind62._post_chunk([request], idmap, run, root / "part62-run-all.json",
            root, root / "ledger.jsonl", "haiku", blind62.MODEL, api, "calibration", 0,
            sleep=lambda _: None, poll_interval=0, poll_bound=0.01, **kwargs)

    def test_kill_after_reserve_before_post_keeps_reserve_and_does_not_post(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp); api, req, idmap, run=self._fixture(root)
            original=blind62._save_json
            def kill(path, data):
                original(path, data)
                if Path(path).name == "part62-run-all.json" and data["chunks"] and data["chunks"][-1]["status"] == "reserved":
                    raise RuntimeError("kill point")
            with mock.patch.object(blind62, "_save_json", side_effect=kill):
                with self.assertRaisesRegex(RuntimeError, "kill point"):
                    self._run_chunk(root, api, req, idmap, run)
            self.assertEqual(api.posts, 0)
            self.assertEqual(json.loads((root / "part62-run-all.json").read_text())["chunks"][0]["status"], "reserved")
            attempts=blind62._ledger_attempts(root / "ledger.jsonl")
            self.assertEqual(next(iter(attempts.values()))["status"], "reserved")
            with mock.patch.object(blind62, "_preflight_submission", return_value=(root, idmap, {}, [req])):
                with self.assertRaisesRegex(ValueError, "calibration incomplete"):
                    blind62.run_all(root, root, ledger_path=root / "ledger.jsonl", api=api, sleep=lambda _: None)
            self.assertEqual(api.posts, 1)

    def test_kill_after_post_before_id_record_is_ambiguous_and_not_reposted(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp); api, req, idmap, run=self._fixture(root)
            original=blind62._save_text
            def kill(path, data):
                if Path(path).name.startswith("post-response-"):
                    raise RuntimeError("kill after POST")
                return original(path, data)
            with mock.patch.object(blind62, "_save_text", side_effect=kill):
                with self.assertRaisesRegex(RuntimeError, "kill after POST"):
                    self._run_chunk(root, api, req, idmap, run)
            self.assertEqual(api.posts, 1)
            self.assertEqual(run["chunks"][0]["status"], "posting")
            self.assertFalse(list(root.glob("post-response-*.txt")))
            with mock.patch.object(blind62, "_preflight_submission", return_value=(root, idmap, {}, [req])):
                with self.assertRaisesRegex(ValueError,
                                           "provider batch carrying.*--no-batch-verified") as raised:
                    blind62.run_all(root, root, ledger_path=root / "ledger.jsonl", api=api, sleep=lambda _: None)
            self.assertIn("--no-batch-verified", str(raised.exception))
            chunk = json.loads((root / "part62-run-all.json").read_text())["chunks"][0]
            with self.assertRaisesRegex(ValueError, "provider batch carrying.*--no-batch-verified"):
                blind62.retire_attempt(chunk["attempt_id"], "OPERATOR", "unknown result",
                                       root / "ledger.jsonl", root)
            self.assertEqual(api.posts, 1)

    def test_s2_and_s3_unknown_post_outcomes_require_no_batch_verified(self):
        for scenario in ("raw-and-record-save-fail", "timeout-after-accept"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory(
                    dir=Path(__file__).parent) as temporary:
                root = Path(temporary)
                api, request, idmap, run = self._fixture(root)
                if scenario == "raw-and-record-save-fail":
                    real_save = blind62._save_text
                    real_persist = blind62._persist_run

                    def fail_raw(path, value):
                        if Path(path).name.startswith("post-response-"):
                            raise OSError("simulated disk full (raw)")
                        return real_save(path, value)

                    def fail_record(path, current_run):
                        if any(chunk.get("status") == "submitted"
                               and chunk.get("post_response_saved") is False
                               for chunk in current_run.get("chunks", [])):
                            raise OSError("simulated disk full (run record)")
                        return real_persist(path, current_run)

                    with mock.patch.object(blind62, "_save_text", side_effect=fail_raw), \
                         mock.patch.object(blind62, "_persist_run", side_effect=fail_record):
                        with self.assertRaisesRegex(OSError, r"disk full \(run record\)"):
                            self._run_chunk(root, api, request, idmap, run)
                else:
                    class AcceptedTimeout:
                        def __init__(self):
                            self.posts = 0

                        def __call__(self, method, path, body=None):
                            if method == "POST":
                                self.posts += 1
                                raise TimeoutError("read timed out after provider accept")
                            raise AssertionError(f"unexpected call {method} {path}")

                    api = AcceptedTimeout()
                    with self.assertRaisesRegex(TimeoutError, "after provider accept"):
                        self._run_chunk(root, api, request, idmap, run)

                with mock.patch.object(blind62, "_preflight_submission",
                                       return_value=(root, idmap, {}, [request])):
                    with self.assertRaisesRegex(ValueError,
                                               "provider batch carrying.*--no-batch-verified") as raised:
                        blind62.run_all(root, root, reader="haiku", ledger_path=root / "ledger.jsonl",
                                        api=api, sleep=lambda _: None)
                self.assertIn("--no-batch-verified", str(raised.exception))
                chunk = json.loads((root / "part62-run-all.json").read_text())["chunks"][0]
                self.assertEqual(chunk["status"], "ambiguous")
                self.assertTrue(chunk["provider_custom_ids"]["r00000"].startswith(
                    chunk["submission_id"] + "-"))
                with self.assertRaisesRegex(ValueError, "provider batch carrying.*--no-batch-verified"):
                    blind62.retire_attempt(chunk["attempt_id"], "OPERATOR", "unknown result",
                                           root / "ledger.jsonl", root)
                self.assertEqual(blind62._ledger_attempts(root / "ledger.jsonl")
                                 [chunk["attempt_id"]]["status"], "reserved")
                with mock.patch.object(blind62, "_preflight_submission",
                                       return_value=(root, idmap, {}, [request])):
                    with self.assertRaisesRegex(ValueError, "provider batch carrying.*--no-batch-verified"):
                        blind62.run_all(root, root, reader="haiku", ledger_path=root / "ledger.jsonl",
                                        api=api, sleep=lambda _: None, retry_calibration=True)
                self.assertEqual(api.posts, 1)

    def test_short_batch_results_are_saved_retirable_and_keep_reserve(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            api, request, _idmap, run = self._fixture(root)
            request2 = {**request, "custom_id": "r00001"}
            idmap = {"r00000": "cal|0|EARTH|ORIGIN", "r00001": "cal|1|EARTH|ORIGIN"}

            class ShortResults:
                def __init__(self):
                    self.posts = 0
                    self.provider_custom_id = None

                def __call__(self, method, path, body=None):
                    if method == "POST":
                        self.posts += 1
                        self.provider_custom_id = body["requests"][0]["custom_id"]
                        return json.dumps({"id": "short-result-batch"})
                    if path.endswith("/results"):
                        return json.dumps({"custom_id": self.provider_custom_id,
                            "result": {"type": "succeeded", "message": {
                                "content": [{"type": "text", "text": "EARTH OK"}],
                                "stop_reason": "end_turn",
                                "usage": {"input_tokens": 5, "output_tokens": 2}}}})
                    return json.dumps({"processing_status": "ended"})

            api = ShortResults()
            with self.assertRaisesRegex(ValueError,
                                       "batch short-result-batch returned 1 of 2 results; status=results-incomplete"):
                blind62._post_chunk([request, request2], idmap, run,
                                    root / "part62-run-all.json", root, root / "ledger.jsonl",
                                    "haiku", blind62.MODEL, api, "calibration", 0,
                                    sleep=lambda _: None, poll_interval=0, poll_bound=0.01)
            chunk = json.loads((root / "part62-run-all.json").read_text())["chunks"][0]
            self.assertEqual(chunk["status"], "results-incomplete")
            self.assertEqual(chunk["batch_id"], "short-result-batch")
            self.assertTrue(chunk["provider_custom_ids"]["r00000"].startswith(
                chunk["submission_id"] + "-"))
            raw_path = Path(chunk["result_path"])
            self.assertTrue(raw_path.is_file())
            self.assertEqual(len(raw_path.read_text().splitlines()), 1)
            attempt = blind62._ledger_attempts(root / "ledger.jsonl")[chunk["attempt_id"]]
            self.assertEqual(attempt["status"], "reserved")
            self.assertGreater(attempt["reserve"], 0)

            blind62.retire_attempt(chunk["attempt_id"], "OPERATOR", "short results reviewed",
                                   root / "ledger.jsonl", root)
            attempt = blind62._ledger_attempts(root / "ledger.jsonl")[chunk["attempt_id"]]
            self.assertEqual(attempt["status"], "retired")
            self.assertGreater(attempt["reserve"], 0)

    def test_kill_after_id_before_results_resumes_saved_batch_without_post(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp); api, req, idmap, run=self._fixture(root); api.fail_status_once=True
            with self.assertRaisesRegex(RuntimeError, "simulated kill"):
                self._run_chunk(root, api, req, idmap, run)
            chunk=run["chunks"][0]
            self.assertEqual(chunk["status"], "submitted")
            with mock.patch.object(blind62, "_preflight_submission", return_value=(root, idmap, {}, [req])):
                with self.assertRaisesRegex(ValueError, "calibration incomplete"):
                    blind62.run_all(root, root, ledger_path=root / "ledger.jsonl", api=api, sleep=lambda _: None)
            self.assertEqual(api.posts, 1)
            self.assertEqual(json.loads((root / "part62-run-all.json").read_text())["chunks"][0]["status"], "settled")

    def test_kill_after_results_before_settle_replays_idempotently(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp); api, req, idmap, run=self._fixture(root)
            original=blind62._StageLedger.settle
            with mock.patch.object(blind62._StageLedger, "settle", side_effect=RuntimeError("kill before settle")):
                with self.assertRaisesRegex(RuntimeError, "kill before settle"):
                    self._run_chunk(root, api, req, idmap, run)
            chunk=run["chunks"][0]
            self.assertEqual(chunk["status"], "submitted")
            self.assertTrue((root / f"part62-raw-{chunk['batch_id']}.jsonl").is_file())
            blind62._StageLedger.settle=original
            with mock.patch.object(blind62, "_preflight_submission", return_value=(root, idmap, {}, [req])):
                with self.assertRaisesRegex(ValueError, "calibration incomplete"):
                    blind62.run_all(root, root, ledger_path=root / "ledger.jsonl", api=api, sleep=lambda _: None)
            attempts=blind62._ledger_attempts(root / "ledger.jsonl")
            self.assertEqual(api.posts, 1)
            self.assertEqual(sum(row["status"] == "settled" for row in attempts.values()), 1)

if __name__ == "__main__":
    unittest.main()
