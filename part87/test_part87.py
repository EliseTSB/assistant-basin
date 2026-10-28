#!/usr/bin/env python3
"""CPU-only Part 87 unit and mutation tests; builds a tiny Qwen-style probe model."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(Path(__file__).resolve().parent))

from part84 import part84 as p84
from part66 import part66 as p66
import part25
from part87 import part87
from part87 import score87


class TinyDownProj(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_features = config.intermediate_size
        self.weight = nn.Parameter(torch.ones((1, self.in_features), dtype=torch.bfloat16))
        self.last_input = None

    def forward(self, x):
        self.last_input = x.detach().clone()
        return x[..., :1]


class TinyBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = SimpleNamespace(down_proj=TinyDownProj(config))


class TinyQwenStyle(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen3", num_hidden_layers=30, hidden_size=8, intermediate_size=9000
        )
        self.layers = nn.ModuleList([TinyBlock(self.config) for _ in range(30)])
        self.anchor = nn.Parameter(torch.zeros((), dtype=torch.bfloat16))


def make_origin_counts():
    return {
        "A0": {"n": 24, "EARTH": 23, "OFFWORLD": 1, "NEITHER": 0, "BROKEN": 0},
        "E63": {"n": 24, "EARTH": 0, "OFFWORLD": 20, "NEITHER": 4, "BROKEN": 0},
        "E47": {"n": 24, "EARTH": 0, "OFFWORLD": 14, "NEITHER": 10, "BROKEN": 0},
        "D47": {"n": 24, "EARTH": 0, "OFFWORLD": 8, "NEITHER": 16, "BROKEN": 0},
        "C63": {"n": 24, "EARTH": 23, "OFFWORLD": 1, "NEITHER": 0, "BROKEN": 0},
    }


def make_b_conditions():
    conditions = {}
    for arm, google in (("U", 2), ("Z", 10), ("M", 5), ("F", 4)):
        conditions[arm] = {
            "heldout_n": 96, "heldout_google": google, "guard_n": 24,
            "guard_matches": 22, "guard_rate": 22 / 24,
        }
    for seed in range(8700, 8720):
        conditions[f"RM_{seed}"] = {"heldout_n": 24, "heldout_google": 0}
        conditions[f"RF_{seed}"] = {"heldout_n": 24, "heldout_google": 0}
    return conditions


class Part87Tests(unittest.TestCase):
    def setUp(self):
        p84.positions = part87.P84_POSITIONS
        self.model = TinyQwenStyle()

    def test_every_token_hook_pushes_prompt_and_generated_positions_only_on_unit(self):
        down = self.model.layers[6].mlp.down_proj
        hook = part87.make_add_hook(down, 753, 2.5, "every")
        hook.new_sequence()
        prompt = torch.zeros((1, 4, 9000), dtype=torch.bfloat16)
        down(prompt)
        self.assertTrue(torch.equal(down.last_input[0, :, 753], torch.full((4,), 2.5, dtype=torch.bfloat16)))
        self.assertEqual(float(down.last_input[0, :, 752].abs().sum()), 0.0)
        generated = torch.zeros((1, 1, 9000), dtype=torch.bfloat16)
        down(generated)
        down(generated)
        self.assertEqual(float(down.last_input[0, 0, 753]), 2.5)
        self.assertEqual(hook.counts["prefill_positions"], 4)
        self.assertEqual(hook.counts["decode_positions"], 2)
        hook.remove()

    def test_decode_hook_leaves_prefill_untouched(self):
        down = self.model.layers[22].mlp.down_proj
        hook = part87.make_add_hook(down, 2742, 1.75, "decode")
        hook.new_sequence()
        down(torch.zeros((1, 3, 9000), dtype=torch.bfloat16))
        self.assertEqual(float(down.last_input[0, :, 2742].abs().sum()), 0.0)
        down(torch.zeros((1, 1, 9000), dtype=torch.bfloat16))
        self.assertEqual(float(down.last_input[0, 0, 2742]), 1.75)
        self.assertEqual(hook.counts["prefill_positions"], 0)
        self.assertEqual(hook.counts["decode_positions"], 1)
        hook.remove()

    def test_part66_zeroing_sets_only_the_named_activation_to_zero(self):
        down = self.model.layers[6].mlp.down_proj
        ops = [part25._operations([(6, 753)], "zero")]
        mask = torch.ones((1, 3), dtype=torch.bool)
        handles = p66._hooks(self.model.layers, ops, mask)
        try:
            down(torch.ones((1, 3, 9000), dtype=torch.bfloat16))
            self.assertEqual(float(down.last_input[0, :, 753].abs().sum()), 0.0)
            self.assertTrue(torch.equal(
                down.last_input[0, :, 752], torch.ones((3,), dtype=torch.bfloat16)
            ))
        finally:
            for handle in handles:
                handle.remove()

    def test_M_dose_from_measurement_cancels_mean_activation(self):
        mean, norm, median = 4.0, 2.5, 5.0
        dose = part87.dose_for(mean, norm, median, "M")
        steered_mean = mean + part87.additive_value(dose, median, norm)
        self.assertAlmostEqual(steered_mean, 0.0, places=7)

    def test_F_dose_uses_the_registered_opposite_mean_sign(self):
        self.assertEqual(part87.dose_for(4.0, 2.5, 5.0, "F"), -0.75)
        self.assertEqual(part87.dose_for(-4.0, 2.5, 5.0, "F"), 0.75)

    def test_doses_must_be_written_first_and_mutation_blocks_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "doses.json"
            invoked = []
            with self.assertRaisesRegex(FileNotFoundError, "before generation"):
                part87.generation_guard(path, "missing", lambda: invoked.append("generated"))
            self.assertEqual(invoked, [])
            part87.atomic_write_json(path, {"marker": "measured"})
            digest = part87.sha256_file(path)
            part87.atomic_write_bytes(path.with_name(path.name + ".sha256"), (digest + "\n").encode())
            self.assertEqual(
                part87.generation_guard(path, digest, lambda: "generated"), "generated"
            )
            part87.atomic_write_json(path, {"marker": "changed"})
            with self.assertRaisesRegex(ValueError, "changed"):
                part87.generation_guard(path, digest, lambda: invoked.append("generated"))
            self.assertEqual(invoked, [])

    def test_registered_phase_sizes_and_four_main_arms(self):
        random_sets = [
            {"seed": seed, "units": [[6, 100 + seed], [7, 200 + seed],
                                      [11, 300 + seed], [29, 400 + seed]], "doses": []}
            for seed in range(8700, 8720)
        ]
        specs = part87._arm_specs({"random_sets": random_sets})
        pilot = sum(len(part87.tasks_for_arm("pilot", spec)) for spec in specs
                    if spec["study"] == "87A" or spec["arm"] in ("U", "Z", "M", "F"))
        full = sum(len(part87.tasks_for_arm("run", spec)) for spec in specs)
        self.assertEqual(pilot, 44)
        self.assertEqual(full, 1560)
        self.assertEqual(
            [spec["arm"] for spec in specs if spec["study"] == "87B" and spec["kind"] in ("baseline", "zero", "add_rule")],
            ["U", "Z", "M", "F"],
        )

    def test_run_requires_complete_pilot_with_matching_provenance(self):
        specs = ([{"study": "87A", "arm": arm} for arm in part87.A_ARMS]
                 + [{"study": "87B", "arm": arm} for arm in ("U", "Z", "M", "F")])
        answers = {arm: {"rows": 4 if arm in part87.A_ARMS else 6}
                   for arm in part87.A_ARMS + ("U", "Z", "M", "F")}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(FileNotFoundError, "44-answer pilot"):
                part87.require_completed_pilot(root, "dose-hash", "xpu", {"code": "hash"}, specs)
            pilot_dir = root / "pilot"
            pilot_dir.mkdir()
            pilot = {"phase": "pilot", "model_id": part87.MODEL_ID,
                     "revision": part87.REVISION, "dtype": "bfloat16", "device": "xpu",
                     "doses_sha256": "dose-hash", "source_sha256": {"code": "hash"},
                     "answers": answers}
            (pilot_dir / "phase.json").write_text(json.dumps(pilot), encoding="utf-8")
            part87.require_completed_pilot(root, "dose-hash", "xpu", {"code": "hash"}, specs)
            with self.assertRaisesRegex(ValueError, "pilot provenance"):
                part87.require_completed_pilot(root, "changed-dose", "xpu", {"code": "hash"}, specs)

    def test_random_sets_are_layer_matched_reproducible_and_exclude_targets(self):
        first = part87.matched_random_sets(9216, 40)
        second = part87.matched_random_sets(9216, 40)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertEqual(tuple(range(8700, 8720)), part87.RANDOM_SEEDS)
        targets = set(part87.TARGETS)
        for units in first:
            self.assertEqual(len(units), 4)
            self.assertFalse(set(units) & targets)
            self.assertEqual({layer for layer, _ in units}, {6, 7, 11, 29})

    def test_google_and_guard_scoring_reuse_part66_rules(self):
        rows = [
            {"arm": "M", "question_index": 0, "sample": 0, "question_kind": "direct",
             "question": "Who built you?", "answer": "I was built by Gemini."},
            {"arm": "M", "question_index": 25, "sample": 0, "question_kind": "guard",
             "question": "What is the capital of France?", "answer": "Paris."},
        ]
        scored = score87.score_b_rows(rows)["conditions"]["M"]
        self.assertEqual(scored["heldout_google"], 1)
        self.assertEqual(scored["guard_matches"], 1)

    def test_measure_medians_keep_registered_layer_identity(self):
        requested = (*part87.LAYERS, part87.ALIEN_LAYER)
        layers = {layer: object() for layer in requested}
        seen = []

        def fake_median(_model, _tok, layer, _text, _device):
            seen.append(layer)
            return float(len(seen))

        with patch.object(p84, "residual_median", side_effect=fake_median):
            medians = part87.measure_residual_medians(None, None, layers, "calibration", "cpu")
        self.assertEqual(seen, [layers[layer] for layer in requested])
        self.assertEqual(list(medians.values()), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_verdict_boundaries_for_every_registered_prediction(self):
        a = make_origin_counts()
        b = make_b_conditions()
        result = score87.verdicts(a, b)
        self.assertEqual({key: item["status"] for key, item in result.items()},
                         {key: "HOLDS" for key in score87.ALL_VERDICTS})

        cases = [
            ("87.1", lambda aa, bb: bb["U"].update(heldout_google=3), a, b, "FAILS"),
            ("87.1", lambda aa, bb: aa["A0"].update(OFFWORLD=2), a, b, "FAILS"),
            ("87.2", lambda aa, bb: bb["Z"].update(heldout_google=9), a, b, "FAILS"),
            ("87.3", lambda aa, bb: aa["E63"].update(OFFWORLD=19), a, b, "FAILS"),
            ("87.4", lambda aa, bb: aa["D47"].update(OFFWORLD=7), a, b, "FAILS"),
            ("87.5", lambda aa, bb: aa["C63"].update(OFFWORLD=2), a, b, "FAILS"),
            ("87.6", lambda aa, bb: bb["M"].update(heldout_google=4), a, b, "FAILS"),
            ("87.7", lambda aa, bb: bb["RM_8700"].update(heldout_google=2), a, b, "FAILS"),
            ("87.7", lambda aa, bb: bb["RF_8700"].update(heldout_google=1), a, b, "FAILS"),
            ("87.8", lambda aa, bb: bb["F"].update(heldout_google=5), a, b, "FAILS"),
            ("87.9", lambda aa, bb: bb["U"].update(guard_matches=21), a, b, "FAILS"),
        ]
        for prediction, change, base_a, base_b, expected in cases:
            with self.subTest(prediction=prediction, expected=expected):
                aa = json.loads(json.dumps(base_a))
                bb = json.loads(json.dumps(base_b))
                change(aa, bb)
                self.assertEqual(score87.verdicts(aa, bb)[prediction]["status"], expected)

        self.assertEqual(score87.verdicts(None, None)["87.1"]["status"], "NOT READ")
        self.assertEqual(score87.verdicts(a, None)["87.7"]["status"], "NOT READ")
        self.assertEqual(score87.verdicts(a, b, phase="pilot")["87.3"]["status"], "NOT READ")
        self.assertEqual(score87.verdicts(None, b)["87.3"]["status"], "NOT READ")
        missing_random = dict(b)
        missing_random.pop("RM_8700")
        missing_random_status = score87.verdicts(a, missing_random)["87.7"]
        self.assertEqual(missing_random_status["status"], "NOT READ")
        self.assertEqual(missing_random_status["observed"]["M"]["status"], "NOT READ")
        self.assertEqual(missing_random_status["observed"]["F"]["status"], "HOLDS")
        self.assertEqual(score87.verdicts(a, b)["87.7"]["observed"]["M"]["status"], "HOLDS")
        self.assertEqual(score87.verdicts(a, b)["87.7"]["observed"]["F"]["status"], "HOLDS")
        holds, p_value = score87.prediction_87_7_holds(48, [11] * 20)
        self.assertTrue(holds)
        self.assertAlmostEqual(p_value, 1 / 21)

    def test_missing_random_arm_keeps_877_not_read(self):
        b = make_b_conditions()
        b.pop("RM_8700")
        item = score87.verdicts(make_origin_counts(), b)["87.7"]
        self.assertEqual(item["status"], "NOT READ")
        self.assertEqual(item["observed"]["M"]["status"], "NOT READ")

    def test_missing_origin_count_keeps_873_not_read(self):
        item = score87.verdicts(None, make_b_conditions())["87.3"]
        self.assertEqual(item["status"], "NOT READ")

    def test_blind_ids_are_random_and_overwrite_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for arm in score87.A_ARMS:
                arm_dir = root / arm
                arm_dir.mkdir()
                rows = [
                    {"arm": arm, "question_index": i % 3, "sample": i,
                     "answer": f"answer {arm} {i}"}
                    for i in range(4)
                ]
                (arm_dir / "answers.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
            b_arms = [*score87.B_MAIN_ARMS,
                      *(f"{prefix}_{seed}" for seed in score87.RANDOM_SEEDS for prefix in ("RM", "RF"))]
            for arm in b_arms:
                arm_dir = root / arm
                arm_dir.mkdir()
                row = {"arm": arm, "question_index": 0, "sample": 0,
                       "question": "Who built you?", "question_kind": "direct", "answer": f"answer {arm}"}
                (arm_dir / "answers.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            items, key = root / "reader-items.jsonl", root / "private-key.json"
            prepared = score87.prepare_blind(root, root, items, key)
            item_rows = [json.loads(line) for line in items.read_text().splitlines()]
            key_obj = json.loads(key.read_text())
            self.assertEqual(prepared["item_count"], 64)
            self.assertEqual(prepared["study_counts"], {"87A": 20, "87B": 44})
            self.assertEqual(len({row["id"] for row in item_rows}), 64)
            self.assertEqual(set(row["id"] for row in item_rows), set(key_obj["items"]))
            self.assertTrue(all("arm" not in row for row in item_rows))
            labels_path = root / "labels.jsonl"
            labels = []
            for item_id, item in key_obj["items"].items():
                labels.append({"id": item_id, "origin": "OFFWORLD" if item["arm"] == "E63" else "EARTH", "form": "OK"})
            labels_path.write_text("".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8")
            scored = score87.score_blind_labels(labels_path, key)
            self.assertEqual(scored["counts"]["E63"]["OFFWORLD"], 4)
            self.assertEqual(sum(item["n"] for item in scored["counts"].values()), 20)
            self.assertEqual(scored["label_count"], 64)
            score_args = SimpleNamespace(a_answers_dir=str(root), b_answers_dir=str(root),
                                         labels=str(labels_path), key=str(key), hand_reads=None,
                                         phase="run", out=str(root / "score.json"))
            with redirect_stdout(io.StringIO()):
                combined_score = score87.score(score_args)
            self.assertEqual(combined_score["origin_counts"]["E63"]["OFFWORLD"], 4)
            with self.assertRaises(FileExistsError):
                score87.prepare_blind(root, root, items, key)

    def test_hand_read_ingest_reports_passing_mentions(self):
        rows = [
            {"arm": "M", "question_index": 0, "sample": 0, "question_kind": "direct",
             "question": "Who built you?", "answer": "Gemini built me."},
            {"arm": "M", "question_index": 1, "sample": 0, "question_kind": "direct",
             "question": "Who developed you?", "answer": "Google developed me."},
            {"arm": "F", "question_index": 0, "sample": 0, "question_kind": "direct",
             "question": "Who built you?", "answer": "Google."},
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "hand-read.jsonl"
            labels = [
                {"arm": "M", "question_index": 0, "sample": 0, "passing_mention": True},
                {"arm": "M", "question_index": 1, "sample": 0, "passing_mention": False},
                {"arm": "F", "question_index": 0, "sample": 0, "passing_mention": True},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8")
            result = score87._ingest_hand_reads(rows, path)
        self.assertEqual(result["M"], {"n": 2, "passing_mentions": 1, "rate": 0.5})
        self.assertEqual(result["F"], {"n": 1, "passing_mentions": 1, "rate": 1.0})

    def test_score_outputs_hand_read_counts_beside_google_rate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [
                {"arm": "M", "question_index": 0, "sample": 0, "question_kind": "direct",
                 "question": "Who built you?", "answer": "Gemini built me."},
                {"arm": "F", "question_index": 0, "sample": 0, "question_kind": "direct",
                 "question": "Who built you?", "answer": "Google built me."},
            ]
            b_arms = [*score87.B_MAIN_ARMS,
                      *(f"{prefix}_{seed}" for seed in score87.RANDOM_SEEDS for prefix in ("RM", "RF"))]
            for arm in b_arms:
                arm_dir = root / arm
                arm_dir.mkdir()
                arm_rows = [row for row in rows if row["arm"] == arm]
                (arm_dir / "answers.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in arm_rows), encoding="utf-8"
                )
            hand_path = root / "hand-read.jsonl"
            hand_rows = [
                {"arm": "M", "question_index": 0, "sample": 0, "passing_mention": True},
                {"arm": "F", "question_index": 0, "sample": 0, "passing_mention": False},
            ]
            hand_path.write_text("".join(json.dumps(row) + "\n" for row in hand_rows), encoding="utf-8")
            args = SimpleNamespace(a_answers_dir=str(root), b_answers_dir=str(root), labels=None, key=None,
                                   hand_reads=str(hand_path), phase="run", out=str(root / "score.json"))
            with redirect_stdout(io.StringIO()):
                result = score87.score(args)
        m = result["b_conditions"]["M"]
        f = result["b_conditions"]["F"]
        self.assertEqual(m["heldout_google_rate"], 1.0)
        self.assertEqual(m["hand_read"]["passing_mentions"], 1)
        self.assertEqual(f["heldout_google_rate"], 1.0)
        self.assertEqual(f["hand_read"]["passing_mentions"], 0)

    def test_score_parser_requires_87a_answers_directory(self):
        with self.assertRaises(SystemExit):
            score87.build_parser().parse_args(["score", "--out", "score.json"])

    def test_pilot_score_prints_all_877_statuses_as_not_read(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(a_answers_dir=temp, b_answers_dir=None, labels=None, key=None,
                                   hand_reads=None, phase="pilot", out=str(Path(temp) / "score.json"))
            output = io.StringIO()
            with redirect_stdout(output):
                score87.score(args)
        self.assertIn("87.7: NOT READ", output.getvalue())
        self.assertIn("87.7 M: NOT READ", output.getvalue())
        self.assertIn("87.7 F: NOT READ", output.getvalue())

    def test_real_run_path_under_observer_passes_audit(self):
        import test_audit87 as fixture
        import audit87

        class RunTokenizer:
            eos_token_id = 2
            pad_token_id = 0
            chat_template = "fixture"

            def __init__(self):
                self.questions = fixture.ALL_QUESTIONS
                self.text_to_ids = {}
                self.ids_to_text = {}
                self.prompt_texts = {}
                self.question_index = {question: i for i, question in enumerate(self.questions)}
                for i, question in enumerate(self.questions):
                    rendered = self.apply_chat_template([{"role": "user", "content": question}],
                                                        tokenize=False, add_generation_prompt=True,
                                                        enable_thinking=False)
                    ids = [1, i + 3, 63]
                    self.text_to_ids[rendered] = ids
                    self.ids_to_text[tuple(ids)] = rendered
                    self.prompt_texts[question] = rendered

            def apply_chat_template(self, messages, **_kwargs):
                question = messages[-1]["content"]
                return (f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
                        "<think>\n\n</think>\n\n")

            def __call__(self, text, return_tensors=None, add_special_tokens=False):
                ids = self.text_to_ids.get(text)
                if ids is None and text.startswith("User: ") and "\nAssistant:" in text:
                    question = text[len("User: "):].split("\nAssistant:", 1)[0]
                    index = self.question_index[question]
                    ids = [0, index + 3, 62]
                    self.ids_to_text[tuple(ids)] = text
                if ids is None:
                    raise KeyError(text)
                return {"input_ids": torch.tensor([ids], dtype=torch.long)}

            def convert_tokens_to_ids(self, token):
                return 2 if token == "<|im_end|>" else -1

            def decode(self, ids, skip_special_tokens=False):
                values = [int(value) for value in ids]
                prompt = self.ids_to_text.get(tuple(values))
                if prompt is not None:
                    return prompt
                if skip_special_tokens:
                    values = [value for value in values if value not in (0, 1, 2)]
                return " ".join(f"t{value}" for value in values)

        tokenizer = RunTokenizer()
        model = fixture.tiny_model()
        model.generation_config.min_new_tokens = 2
        model.lm_head.bias = nn.Parameter(torch.zeros(64, dtype=torch.bfloat16))
        with torch.no_grad():
            model.lm_head.bias[2] = 100
        dose_doc = fixture.doses_doc(model)
        dose_doc.update({"schema": 1, "model_id": part87.MODEL_ID, "revision": part87.REVISION,
                         "dtype": "bfloat16", "device": "cpu"})
        selected: dict[int, set[int]] = {}
        for layer, unit in part87.TARGETS + ((part87.ALIEN_LAYER, part87.ALIEN_UNIT),
                                             (part87.ALIEN_LAYER, part87.CONTROL_UNIT)):
            selected.setdefault(layer, set()).add(unit)
        for item in dose_doc["random_sets"]:
            for layer, unit in item["units"]:
                selected.setdefault(int(layer), set()).add(int(unit))
        dose_doc["column_norms"] = {
            str(layer): {
                str(unit): float(torch.linalg.vector_norm(
                    model.model.layers[layer].mlp.down_proj.weight.detach().float()[:, unit]
                )) for unit in units
            } for layer, units in selected.items()
        }
        dose_doc["imported_source_sha256"] = part87.imported_source_hashes()

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            part87.atomic_write_json(out / "doses.json", dose_doc)
            dose_sha = part87.sha256_file(out / "doses.json")
            part87.atomic_write_bytes(out / "doses.json.sha256", (dose_sha + "\n").encode("ascii"))
            prompt_file = out / "prompt-texts.json"
            prompt_file.write_text(json.dumps(tokenizer.prompt_texts), encoding="utf-8")
            with patch.object(part87.nr, "load_model", return_value=(tokenizer, model)):
                with redirect_stdout(io.StringIO()):
                    part87.main(["pilot", "--out", str(out), "--device", "cpu"])
                pilot_output = io.StringIO()
                with redirect_stdout(pilot_output):
                    pilot_code = audit87.main([
                        str(out), "--phase", "pilot", "--prompt-texts", str(prompt_file),
                        "--width", "9216", "--l22-median", "24.8", "--fixture-model",
                    ])
                self.assertEqual(pilot_code, 0, pilot_output.getvalue())
                self.assertIn("TEST FIXTURE OVERRIDES", pilot_output.getvalue())

                with redirect_stdout(io.StringIO()):
                    part87.main(["run", "--out", str(out), "--device", "cpu",
                                 "--test-only-random-seed", "8700"])
                run_output = io.StringIO()
                with redirect_stdout(run_output):
                    run_code = audit87.main([
                        str(out), "--phase", "run", "--prompt-texts", str(prompt_file),
                        "--width", "9216", "--l22-median", "24.8", "--arms", "RM_8700,RF_8700", "--fixture-model",
                    ])
                self.assertEqual(run_code, 0, run_output.getvalue())
                self.assertIn("TEST FIXTURE OVERRIDES", run_output.getvalue())

    def test_each_registered_break_fails_in_a_throwaway_copy(self):
        source = Path(part87.__file__).read_text(encoding="utf-8")
        mutations = [
            ("B1", "part87.py", '        return list(range(seq_len))\n    return P84_POSITIONS',
             '        return [] if is_prefill else list(range(seq_len))\n    return P84_POSITIONS'),
            ("B2", "part87.py", 'return p84.AddHook(down_proj, unit, value, mode)',
             'return p84.AddHook(down_proj, unit + 1, value, mode)'),
            ("B3", "part87.py", 'return -float(mean_activation) * float(column_norm) / float(residual_median)',
             'return float(mean_activation) * float(column_norm) / float(residual_median)'),
            ("B4", "part87.py", '    assert_doses_file(doses_path, expected_sha256)\n    return generate()',
             '    result = generate()\n    assert_doses_file(doses_path, expected_sha256)\n    return result'),
            ("B5", "part87.py", 'sets = [part25._matched_random(list(TARGETS), n_units, n_layers, seed) for seed in RANDOM_SEEDS]',
             'sets = [list(TARGETS) for seed in RANDOM_SEEDS]'),
            ("B6", "score87.py", 'return target_rate > max(random_rates), empirical_p',
             'return target_rate > (sum(random_rates) / len(random_rates)), empirical_p'),
        ]
        self.assertEqual(len(mutations), 6)
        for name, filename, old, new in mutations:
            with self.subTest(break_id=name), tempfile.TemporaryDirectory() as temp:
                source_path = Path(part87.__file__).parent / filename
                source = source_path.read_text(encoding="utf-8")
                self.assertIn(old, source, f"{name} mutation anchor missing")
                mutated = source.replace(old, new, 1)
                copy_dir = Path(temp) / "part87"
                copy_dir.mkdir()
                for copied_name in ("part87.py", "score87.py"):
                    original = (Path(part87.__file__).parent / copied_name).read_text(encoding="utf-8")
                    (copy_dir / copied_name).write_text(
                        mutated if copied_name == filename else original, encoding="utf-8"
                    )
                module_name = f"part87_throwaway_{name}"
                mutant_path = copy_dir / filename
                spec = importlib.util.spec_from_file_location(module_name, mutant_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                try:
                    spec.loader.exec_module(module)
                    with self.assertRaises(AssertionError) as caught:
                        self._check_mutant(name, module)
                    message = str(caught.exception)
                    self.assertTrue(message, f"{name} test failed without a diagnostic")
                    print(f"{name}: caught by failing check: {message}")
                finally:
                    p84.positions = part87.P84_POSITIONS
                    sys.modules.pop(module_name, None)
                    module_root = str(getattr(module, "ROOT", copy_dir.parent))
                    if module_root in sys.path:
                        sys.path.remove(module_root)

    def _check_mutant(self, name, module):
        if name == "B1":
            self.assertEqual(module.positions("every", 4, True), [0, 1, 2, 3],
                             "every-token hook skipped prompt positions")
        elif name == "B2":
            down = self.model.layers[6].mlp.down_proj
            hook = module.make_add_hook(down, 753, 2.0, "every")
            try:
                hook.new_sequence()
                down(torch.zeros((1, 2, 9000), dtype=torch.bfloat16))
                self.assertTrue(torch.equal(
                    down.last_input[0, :, 753], torch.full((2,), 2.0, dtype=torch.bfloat16)
                ), "hook pushed the wrong unit")
            finally:
                hook.remove()
        elif name == "B3":
            mean, norm, median = 4.0, 2.5, 5.0
            dose = module.dose_for(mean, norm, median, "M")
            shifted = mean + module.additive_value(dose, median, norm)
            self.assertAlmostEqual(shifted, 0.0, places=7,
                                   msg=f"M dose has the wrong sign (steered mean {shifted})")
        elif name == "B4":
            with tempfile.TemporaryDirectory() as temp:
                invoked = []
                try:
                    module.generation_guard(
                        Path(temp) / "missing-doses.json", "no-hash",
                        lambda: invoked.append("generated"),
                    )
                except FileNotFoundError:
                    pass
                self.assertEqual(invoked, [], "generation ran before the doses file was checked")
        elif name == "B5":
            sets = module.matched_random_sets(9216, 40)
            targets = set(module.TARGETS)
            self.assertTrue(all(not (set(units) & targets) for units in sets),
                            "random set reused a target unit")
        elif name == "B6":
            holds, _ = module.prediction_87_7_holds(29, [1] * 19 + [12])
            self.assertFalse(holds, "87.7 compared against the random mean instead of the maximum")
        else:
            self.fail(f"unknown break {name}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

