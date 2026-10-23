"""CPU-only Part 62 tests, including red checks for deliberate live-code mutations."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import types
import unittest

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


class StubNeuronHook:
    """Small CPU stand-in for NeuronHook's phase and activation-set behavior."""
    def __init__(self, down_proj):
        self.phase = "both"
        self.clamp = None
        self.handle = down_proj.register_forward_pre_hook(self._hook)

    def _hook(self, _module, args):
        values = args[0]
        if self.clamp is None:
            return None
        units, values_to_set = self.clamp
        sequence_length = len(values[0])
        active = (self.phase == "both" or
                  (self.phase == "prompt" and sequence_length > 1) or
                  (self.phase == "gen" and sequence_length == 1))
        if not active:
            return None
        updated = [[[item for item in token] for token in sample] for sample in values]
        for batch_index in range(len(updated)):
            for position in range(sequence_length):
                for unit, value in zip(units, values_to_set):
                    updated[batch_index][position][unit] = value
        return (updated,)

    def remove(self):
        self.handle.remove()


class StubModel:
    """CPU-only stand-in that exposes prompt-prefill and one generated-token pass."""
    def __init__(self, down_proj):
        self.down_proj = down_proj
        self.seen_inputs = []

    def generate(self):
        prompt = [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]]
        generated = [[[2.0, 2.0]]]
        self.seen_inputs.append(self.down_proj(prompt))
        self.seen_inputs.append(self.down_proj(generated))
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
        for model_name, model in run62.MODELS.items():
            rng = run62.np.random.default_rng(run62.CONTROL_SEED)
            pool = run62.np.setdiff1d(run62.np.arange(model["expected_width"]), [model["unit"]])
            selected = sorted(int(unit) for unit in rng.choice(pool, size=run62.CONTROL_COUNT, replace=False))
            self.assertEqual(selected, [model["control_unit"]], model_name)
        self.assertEqual(run62.QWEN_ALIEN_FRAME_MAX_REFERENCE, 24.69)
        self.assertEqual(analyze62.ANSWERS_PER_DOSE, 24)
        self.assertEqual(analyze62.PREDICTION_62A_QWEN_DOSES, (1.5, 2.0))
        self.assertEqual(analyze62.PREDICTION_62A_OLMO_DOSES, (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0))
        self.assertEqual(analyze62.PREDICTION_62E_DOSE, 0.5)
        self.assertGreaterEqual(blind62.MAX_TOKENS, 1024)
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
        down_proj = FakeDownProjection()
        model = StubModel(down_proj)
        hook = run62.make_phase_hook(down_proj, 0, 9.0, phase,
                                     hook_factory=StubNeuronHook, tensor_factory=list)
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
        self.assertTrue(all(token[0] == 9.0 and token[1] == 1.0 for token in seen[0][0]))
        self.assertEqual(seen[1], [[[2.0, 2.0]]])

    def test_generation_phase_leaves_prompt_untouched_and_changes_generated_token(self):
        seen = self._run_phase("gen")
        self.assertEqual(len(seen), 2)
        self.assertTrue(all(token == [1.0, 1.0] for token in seen[0][0]))
        self.assertEqual(seen[1], [[[9.0, 2.0]]])

    def test_both_phase_changes_prompt_and_generated_token(self):
        seen = self._run_phase("both")
        self.assertTrue(all(token[0] == 9.0 for token in seen[0][0]))
        self.assertEqual(seen[1], [[[9.0, 2.0]]])

    def test_prompt_hook_phase_matches_requested_phase(self):
        down_proj = FakeDownProjection()
        hook = run62.make_phase_hook(down_proj, 0, 9.0, "prompt",
                                     hook_factory=StubNeuronHook, tensor_factory=list)
        try:
            self.assertEqual(hook.phase, "prompt")
        finally:
            hook.clamp = None
            hook.remove()


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
            "models": {},
        }
        serial = 0
        for model_name, model in run62.MODELS.items():
            meta = {"id": model["id"], "revision": model["revision"],
                    "layer": model["layer"], "target_unit": model["unit"],
                    "control_unit": control_units[model_name], "width": model["expected_width"],
                    "pile_maximum": 10.0, "alien_frame_maximum": 24.69,
                    "alien_frame_count": run62.ALIEN_FRAME_COUNT,
                    "registered_alien_frame_reference": model.get("alien_frame_reference"),
                    "target_column_norm": 2.0, "control_column_norm": 1.0}
            manifest["models"][model_name] = meta
            for spec in run62.model_cell_specs(model_name, control_units[model_name]):
                seed = run62.condition_seed(run62.DEFAULT_SEED, spec["model"], spec["dose"],
                                            spec["phase"], spec["question"], spec["kind"], spec["unit"])
                condition_name = "twin" if spec["kind"] == "twin" else f"control-{spec['unit']}"
                set_value = float(spec["dose"]) * meta["pile_maximum"]
                if spec["kind"] == "control":
                    set_value *= meta["target_column_norm"] / meta["control_column_norm"]
                condition = {"condition": condition_name, "kind": spec["kind"],
                             "unit": spec["unit"], "seed": seed, "phase": spec["phase"],
                             "set_value": set_value, "n": run62.SAMPLES,
                             "answers": [f"stub response {serial} {i}" for i in range(run62.SAMPLES)]}
                path = run62.write_cell(folder, spec, condition, meta)
                manifest["planned_cells"].append({**spec, "file": path.name})
                manifest["completed"].append(path.name)
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
            for index, (label, quality, _text) in enumerate(blind_origin.CAL):
                labels[f"cal|{index}|{label}|{quality}"] = [label, quality]
            (review_dir / "part62-key-fixture.json").write_text(json.dumps(key), encoding="utf-8")
            labels_path = review_dir / "part62-labels-fixture.json"
            labels_path.write_text(json.dumps({"schema": "part62-blind-labels-v1",
                                               "batch_id": "fixture", "max_tokens": 1024,
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


class BlindGateTests(unittest.TestCase):
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
        self.assertEqual(len(requests), 1 + len(blind_origin.CAL))

    def test_origin_calibration_gate_allows_two_and_rejects_three_errors(self):
        labels = {}
        for index, (label, quality, _text) in enumerate(blind_origin.CAL):
            labels[f"cal|{index}|{label}|{quality}"] = [label, quality]
        labels["ans|0"] = ["NEITHER", "OK"]
        self.assertEqual(blind62.scoring_gate(labels, ["ans|0"])["unlabelled_answers"], [])
        for index, (label, quality, _text) in enumerate(blind_origin.CAL[:2]):
            labels[f"cal|{index}|{label}|{quality}"] = ["NEITHER", "BROKEN"]
        self.assertEqual(len(blind62.scoring_gate(labels, ["ans|0"])["calibration_wrong"]), 2)
        index, (label, quality, _text) = 2, blind_origin.CAL[2]
        labels[f"cal|{index}|{label}|{quality}"] = ["NEITHER", "BROKEN"]
        with self.assertRaisesRegex(ValueError, "calibration gate failed"):
            blind62.scoring_gate(labels, ["ans|0"])

    def test_capped_reader_response_blocks_scoring_even_when_labels_parse(self):
        labels = {}
        for index, (label, quality, _text) in enumerate(blind_origin.CAL):
            labels[f"cal|{index}|{label}|{quality}"] = [label, quality]
        labels["ans|0"] = ["OFFWORLD", "OK"]
        calibration_id = f"cal|0|{blind_origin.CAL[0][0]}|{blind_origin.CAL[0][1]}"
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

    def test_any_unlabelled_answer_blocks_scoring(self):
        labels = {}
        for index, (label, quality, _text) in enumerate(blind_origin.CAL):
            labels[f"cal|{index}|{label}|{quality}"] = [label, quality]
        labels["ans|0"] = ["OFFWORLD", "OK"]
        with self.assertRaisesRegex(ValueError, "unlabelled answer"):
            blind62.scoring_gate(labels, ["ans|0", "ans|1"])


class AnalysisTests(unittest.TestCase):
    def test_review_proxy_regexes(self):
        self.assertEqual(analyze62.proxy_flags("An alien species crossed the galaxy."),
                         {"offworld": True, "scifi": True, "earth": False})
        self.assertEqual(analyze62.proxy_flags("A robot helped the human on Earth."),
                         {"offworld": False, "scifi": True, "earth": True})

    def test_logistic_fit_returns_slope_midpoint_and_10_to_90_width(self):
        fit = analyze62.fit_logistic([(-1.0, 12, 100), (0.0, 40, 100),
                                      (1.0, 77, 100), (2.0, 94, 100)])
        self.assertEqual(fit["status"], "fit")
        self.assertGreater(fit["slope"], 0)
        self.assertTrue(math.isfinite(fit["midpoint"]))
        self.assertGreater(fit["width_10_90"], 0)
        self.assertAlmostEqual(fit["width_10_90"], 2 * math.log(9) / abs(fit["slope"]))

    def test_prediction_62b_fails_when_prompt_threshold_is_breached(self):
        counts = {("qwen35-4b", "prompt", "twin", 0.25): {"n": 24, "OFFWORLD": 2}}
        result = analyze62.evaluate_predictions(counts, {})
        self.assertEqual(result["status"]["62b"], "FAILED")


def _inner_test_red(test_case: unittest.TestCase) -> bool:
    result = unittest.TestResult()
    unittest.TestSuite([test_case]).run(result)
    return bool(result.failures or result.errors)


class MutationTests(unittest.TestCase):
    def test_sabotaged_real_functions_are_killed_by_assertions(self):
        # Change the live function object's code; keep its identity, globals, and test path.
        function = run62.make_phase_hook
        original_code = function.__code__
        original_identity = id(function)
        namespace = {}
        exec("def mutant(down_proj, unit, value, phase, hook_factory=None, tensor_factory=None):\n"
             "    hook = hook_factory(down_proj)\n"
             "    hook.phase = 'both'\n"
             "    hook.clamp = (tensor_factory([int(unit)]), tensor_factory([float(value)]))\n"
             "    return hook\n", function.__globals__, namespace)
        try:
            function.__code__ = namespace["mutant"].__code__
            self.assertEqual(id(run62.make_phase_hook), original_identity)
            self.assertTrue(_inner_test_red(PhaseHookTests("test_prompt_hook_phase_matches_requested_phase")))
            print("MUTATION RED: live make_phase_hook forced prompt clamp to both; phase assertion failed")
        finally:
            function.__code__ = original_code

        function = analyze62.evaluate_predictions
        original_code = function.__code__
        original_identity = id(function)
        namespace = {}
        exec("def mutant(counts, fits):\n"
             "    return {'status': {key: 'HELD' for key in "
             "('62a', '62b', '62c', '62d', '62e', '62f', '62g')}, 'details': {}}\n",
             function.__globals__, namespace)
        try:
            function.__code__ = namespace["mutant"].__code__
            self.assertEqual(id(analyze62.evaluate_predictions), original_identity)
            self.assertTrue(_inner_test_red(
                AnalysisTests("test_prediction_62b_fails_when_prompt_threshold_is_breached")))
            print("MUTATION RED: live evaluate_predictions forced 62b held; threshold assertion failed")
        finally:
            function.__code__ = original_code


if __name__ == "__main__":
    unittest.main()
