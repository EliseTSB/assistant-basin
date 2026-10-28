#!/usr/bin/env python3
"""CPU-only, network-free checks for Part 86 steering, provenance, and verdicts."""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import blind86
import common86
import screen86
import triage86
import verify86

try:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
except ImportError as exc:
    torch = None
    Qwen2Config = Qwen2ForCausalLM = None
    MODEL_IMPORT_ERROR = str(exc)
else:
    MODEL_IMPORT_ERROR = None

if os.environ.get("PART86_GPU_STRICT") == "1" and torch is None:
    raise RuntimeError(f"PART86_GPU_STRICT=1 and torch/transformers could not load: {MODEL_IMPORT_ERROR}")


def tiny_model():
    torch.manual_seed(86)
    config = Qwen2Config(
        vocab_size=64, hidden_size=16, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=True,
    )
    return Qwen2ForCausalLM(config).to(dtype=torch.bfloat16).cpu().eval()


def tiny_audit_model():
    """Small Qwen loop with the registered 23 layer index and two decode positions."""
    torch.manual_seed(86022)
    config = Qwen2Config(
        vocab_size=64, hidden_size=16, intermediate_size=64, num_hidden_layers=23,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=True,
    )
    model = Qwen2ForCausalLM(config).to(dtype=torch.bfloat16).cpu().eval()
    model.lm_head = torch.nn.Linear(16, 64, bias=True).to(dtype=torch.bfloat16)
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.bias.zero_()
        model.lm_head.bias[2] = 100
    model.generation_config.min_new_tokens = 2
    return model


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    chat_template = "qwen-like-template"

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = [1, 4, 5, 6, 7] if "calibration" in str(text) else [1, 4, 5]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        if enable_thinking is not False or len(messages) != 1 or messages[0]["role"] != "user":
            raise AssertionError("wrong template options")
        return (
            f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    def convert_tokens_to_ids(self, _token):
        return -1

    def decode(self, ids, skip_special_tokens=True):
        special = {0, 1, 2} if skip_special_tokens else set()
        return " ".join(f"tok{int(token)}" for token in ids if int(token) not in special)


class TinyAuditTokenizer(TinyTokenizer):
    def __init__(self):
        self.decode_calls = 0
        self.prompt_by_ids = {}

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        if "calibration" in str(text):
            return super().__call__(text, return_tensors, add_special_tokens)
        questions = common86.VERIFY_QUESTIONS
        question = next((q for q in questions if q in str(text)), None)
        if question is None:
            raise ValueError(f"tiny audit tokenizer could not find a registered question in {text!r}")
        index = questions.index(question)
        ids = [1, 10 + index, 20 + index]
        self.prompt_by_ids[tuple(ids)] = str(text)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        ids = [int(token) for token in ids]
        if not skip_special_tokens:
            return self.prompt_by_ids.get(tuple(ids), super().decode(ids, skip_special_tokens=False))
        self.decode_calls += 1
        if self.decode_calls in (1, 2, 17, 18, 33, 34):
            return "I am Qwen, a helpful AI assistant."
        if self.decode_calls >= 47:
            return "I am a quiet traveler who watches the sea."
        pattern = (self.decode_calls - 39) % 4
        if pattern in (0, 2):
            return "I am a quiet traveler who watches the sea."
        return "I am a quiet sailor who watches the harbor."


def _tiny_prompt_texts_file(path: Path, tokenizer):
    rendered = {}
    for question in common86.VERIFY_QUESTIONS:
        prompt = common86.build_prompt(tokenizer, question)
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        rendered[question] = tokenizer.decode(ids, skip_special_tokens=False)
    path.write_text(json.dumps(rendered), encoding="utf-8")
    return path


def _tiny_screen_args(out, calib, **overrides):
    args = {
        "out": str(out), "calib": str(calib), "layers": [0], "batch_size": 2,
        "limit": 2, "device": "cpu", "progress": None, "reset_layer": [],
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def _write_screen_score_fixture(directory: Path, kappa, provenance=None):
    directory.mkdir(parents=True, exist_ok=True)
    score_path = directory / "screen-score.json"
    provenance = provenance or {
        "screen_manifest_sha256": "a" * 64,
        "primary_labels_sha256": "b" * 64,
        "selected_set_sha256": "c" * 64,
    }
    score_manifest = {"fixture": True, "reader_kappa": kappa, **provenance}
    score_manifest["manifest_sha256"] = common86.canonical_sha256(score_manifest)
    common86.atomic_json(score_path, {
        "reader_agreement": {"kappa": kappa},
        "score_manifest": score_manifest,
    })
    artifact = {
        "score_file": score_path.name,
        "score_file_sha256": common86.sha256_file(score_path),
        "score_manifest_sha256": score_manifest["manifest_sha256"],
        **provenance,
    }
    artifact["manifest_sha256"] = common86.canonical_sha256(artifact)
    common86.atomic_json(Path(str(score_path) + ".manifest.json"), artifact)
    return score_path


class TestRowsHook(unittest.TestCase):
    @unittest.skipIf(torch is None, f"torch/transformers unavailable: {MODEL_IMPORT_ERROR}")
    def test_pushes_are_per_row_and_only_apply_after_prompt_pass(self):
        model = tiny_model()
        down = model.model.layers[0].mlp.down_proj
        width = down.in_features
        pushes = torch.zeros((3, width), dtype=torch.float32)
        pushes[0, 2] = 1.25
        pushes[1, 5] = -2.5
        pushes[2, 2] = 3.75
        hook = common86.AddRowsHook(down, pushes)
        try:
            hook.begin_sequence()
            prefill = torch.randn((3, 4, width))
            prefill_before = prefill.clone()
            self.assertIsNone(hook._hook(down, (prefill,)))
            torch.testing.assert_close(prefill, prefill_before)
            decode = torch.randn((3, 1, width))
            result = hook._hook(down, (decode,))[0]
            expected = decode.clone()
            expected[0, 0, 2] += 1.25
            expected[1, 0, 5] -= 2.5
            expected[2, 0, 2] += 3.75
            torch.testing.assert_close(result, expected)
            delta = result - decode
            self.assertEqual(int(torch.count_nonzero(delta).item()), 3)
            self.assertEqual(hook.counts["prefill_calls"], 1)
            self.assertEqual(hook.counts["prefill_positions"], 4)
            self.assertEqual(hook.counts["per_row_generated_calls"], [1, 1, 1])
        finally:
            hook.remove()

    @unittest.skipIf(torch is None, f"torch/transformers unavailable: {MODEL_IMPORT_ERROR}")
    def test_tiny_model_zero_row_matches_baseline_and_push_is_row_local(self):
        model = tiny_model()
        layer = model.model.layers[0]
        tokenizer = TinyTokenizer()
        width = layer.mlp.down_proj.in_features
        zero = torch.zeros((1, width), dtype=torch.float32)
        baseline, _ = common86.generate_batch(model, tokenizer, layer, "Who are you?", zero, "cpu", 6)
        two_rows = torch.zeros((2, width), dtype=torch.float32)
        two_rows[1] = 100.0
        hooked = common86.AddRowsHook(layer.mlp.down_proj, two_rows)
        try:
            hooked.begin_sequence()
            hooked._hook(layer.mlp.down_proj, (torch.zeros((2, 3, width), dtype=torch.float32),))
            decode = torch.zeros((2, 1, width), dtype=torch.float32)
            changed = hooked._hook(layer.mlp.down_proj, (decode,))[0] - decode
            self.assertEqual(int(torch.count_nonzero(changed[0]).item()), 0)
            self.assertGreater(int(torch.count_nonzero(changed[1]).item()), 0)
        finally:
            hooked.remove()
        generated, _ = common86.generate_batch(model, tokenizer, layer, "Who are you?", two_rows, "cpu", 6)
        self.assertEqual(generated[0]["token_ids"], baseline[0]["token_ids"])
        self.assertNotEqual(generated[1]["token_ids"], baseline[0]["token_ids"])


class TestSteeringDose(unittest.TestCase):
    @unittest.skipIf(torch is None, f"torch/transformers unavailable: {MODEL_IMPORT_ERROR}")
    def test_neuron_and_random_write_norm_are_dose_times_median(self):
        torch.manual_seed(3)
        weight = torch.randn((11, 7), dtype=torch.float32)
        median = 4.125
        delta = common86.neuron_delta(weight, 3, median)
        real_push = torch.zeros(7)
        real_push[3] = delta
        real_write = weight @ real_push
        self.assertAlmostEqual(float(torch.linalg.vector_norm(real_write)), 0.8 * median, places=5)
        directions = common86.scaled_random_directions(weight, 8, seed=86022, target_norm=0.8 * median)
        norms = torch.linalg.vector_norm(directions @ weight.T, dim=1)
        torch.testing.assert_close(norms, torch.full((8,), 0.8 * median), rtol=2e-5, atol=2e-5)
        again = common86.scaled_random_directions(weight, 8, seed=86022, target_norm=0.8 * median)
        torch.testing.assert_close(directions, again)


class TestPromptAndTriage(unittest.TestCase):
    def test_rendered_prompt_contract_and_eos_set(self):
        tokenizer = TinyTokenizer()
        prompt = common86.assert_rendered_prompt(tokenizer, "Who are you?")
        self.assertIn("<|im_start|>user", prompt)
        self.assertNotIn("<|im_start|>system", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))

        class BadPrompt(TinyTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return "<|im_start|>system\nignored\n<|im_start|>assistant\n"
        with self.assertRaisesRegex(ValueError, "system message"):
            common86.assert_rendered_prompt(BadPrompt(), "Who are you?")

        model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(eos_token_id=248044)),
                                generation_config=None)
        self.assertIn(248044, common86.eos_ids(tokenizer, model))

    def test_fixed_triage_and_false_kept_regression(self):
        self.assertEqual(triage86.triage_answer("I am an AI assistant."), "kept")
        self.assertEqual(triage86.triage_answer("As a language model, I can help."), "kept")
        self.assertEqual(triage86.triage_answer("A story about an AI assistant explores identity."), "changed")
        self.assertEqual(
            triage86.triage_answer("I am curious about language models, but I travel between distant stars."),
            "changed",
        )
        self.assertEqual(triage86.triage_answer("Qwen3.5 is what I am."), "kept")
        self.assertEqual(triage86.triage_answer("the sea"), "broken")
        self.assertEqual(triage86.triage_answer("one two three, one two three; one two three."), "broken")
        self.assertEqual(triage86.triage_answer("A quiet traveler watches the sea."), "changed")
        self.assertEqual(triage86.triage_answer("red blue green RED blue green red blue green"), "broken")


class TestVerdictRules(unittest.TestCase):
    def test_fisher_hand_table_and_amended_862_edges(self):
        self.assertAlmostEqual(blind86.fisher_greater(1, 0, 1, 1), 2 / 3, places=12)
        self.assertAlmostEqual(blind86.fisher_greater(2, 0, 0, 2), 1 / 6, places=12)
        self.assertEqual(blind86.verdict_862(1, 0)["status"], "FAILS")
        self.assertEqual(blind86.verdict_862(9, 0)["status"], "FAILS")
        self.assertFalse(blind86.verdict_862(9, 0)["minimum_real_count_met"])
        self.assertTrue(blind86.verdict_862(10, 0)["minimum_real_count_met"])
        self.assertEqual(blind86.verdict_862(10, 0)["status"], "FAILS")
        self.assertEqual(blind86.verdict_862(36, 1)["status"], "FAILS")
        self.assertEqual(blind86.verdict_862(100, 1)["status"], "HOLDS")

    def test_golden_verdict_cases_and_override_order(self):
        oracle = json.loads((HERE / "golden-verdicts.json").read_text(encoding="utf-8"))
        base = {
            str(layer): {
                "real": dict(oracle["base_real"]),
                "random": dict(oracle["base_random"]),
            }
            for layer in map(int, oracle["layers"])
        }
        for layer, groups in oracle["base_overrides"].items():
            for condition, fields in groups.items():
                base[layer][condition].update(fields)
        for case in oracle["cases"]:
            with self.subTest(case=case["name"]):
                counts = json.loads(json.dumps(base))
                for layer, groups in case.get("overrides", {}).items():
                    for condition, fields in groups.items():
                        counts[layer][condition].update(fields)
                for layer in case.get("drop_layers", []):
                    counts.pop(layer)
                if case.get("expect_error"):
                    with self.assertRaises(ValueError):
                        blind86.verdicts_from_counts({"layers": counts})
                    continue
                result = blind86.verdicts_from_counts({"layers": counts})["verdicts"]
                for name, expected in case["expected"].items():
                    self.assertEqual(result[name]["status"], expected)

    def test_full_size_labels_to_counts_to_verdicts_including_still_qwen(self):
        labels, mapping, triage_counts = [], {}, {}
        for layer in range(10, 27):
            real_changed = 60 if layer == 22 else 20
            triage_counts[str(layer)] = {
                "real": {"n": 9216, "changed": real_changed, "kept": 9216 - real_changed - 10, "broken": 10},
                "random": {"n": 512, "changed": 10, "kept": 492, "broken": 10},
            }
            for index in range(real_changed):
                item_id = f"L{layer}-real-{index}"
                mapping[item_id] = {"layer": layer, "source": "real", "item_index": index,
                                    "triage_category": "changed", "miss_rate_sample": False}
                labels.append({"id": item_id, "persona": "traveler", "kind": "human", "coherent": True})
            for index in range(10):
                item_id = f"L{layer}-random-{index}"
                mapping[item_id] = {"layer": layer, "source": "random", "item_index": index,
                                    "triage_category": "changed", "miss_rate_sample": False}
                if index < 3:
                    label = {"persona": "traveler", "kind": "human", "coherent": True}
                elif index < 7:
                    label = {"persona": "assistant", "kind": "still Qwen or an assistant", "coherent": True}
                elif index == 8:
                    label = {"persona": "unclear", "kind": "unclear", "coherent": True}
                else:
                    label = {"persona": "unclear", "kind": "human", "coherent": False}
                labels.append({"id": item_id, **label})
        counts = blind86.label_counts(labels, {"rows": mapping}, triage_counts)
        self.assertEqual(len(counts["layers"]), 17)
        self.assertEqual(counts["layers"]["22"]["real"]["n"], 9216)
        self.assertEqual(counts["layers"]["22"]["random"]["n"], 512)
        self.assertEqual(counts["layers"]["22"]["random"]["coherent_still_qwen"], 4)
        self.assertEqual(counts["layers"]["22"]["random"]["not_coherent"], 2)
        self.assertEqual(counts["layers"]["22"]["random"]["unclear"], 1)
        self.assertEqual(counts["layers"]["22"]["random"]["coherent_persona"], 3)
        inconsistent = json.loads(json.dumps(triage_counts))
        inconsistent["22"]["random"]["changed"] += 1
        with self.assertRaisesRegex(ValueError, "labeled 10 changed answers"):
            blind86.label_counts(labels, {"rows": mapping}, inconsistent)
        result = blind86.verdicts_from_counts(counts)["verdicts"]
        self.assertEqual(result["86.2"]["status"], "FAILS")
        self.assertEqual(result["86.3"]["status"], "HOLDS")
        self.assertEqual(result["86.4"]["status"], "FAILS")
        self.assertEqual(result["86.6"]["status"], "HOLDS")

    def test_864_excludes_triage_broken_and_still_qwen_from_both_sides(self):
        groups = {
            str(i): {"changed_labeled": 10, "not_coherent": 3, "unclear": 0,
                     "coherent_persona": 3, "coherent_still_qwen": 4, "triage_broken": 900}
            for i in range(9)
        }
        result = blind86.verdict_864(groups)
        self.assertEqual(result["status"], "FAILS")
        self.assertEqual(result["per_layer"]["0"]["not_coherent_plus_unclear"], 3)
        self.assertEqual(result["per_layer"]["0"]["coherent_persona"], 3)

    def test_866_pools_only_layers_14_through_22(self):
        layers = {}
        for layer in range(10, 27):
            real_personas = 1000 if layer in range(10, 14) else (1 if layer in range(23, 27) else 0)
            layers[str(layer)] = {
                "real": {"n": 9216, "coherent_persona": real_personas},
                "random": {"n": 512, "changed_labeled": 0, "not_coherent": 0,
                           "unclear": 0, "coherent_persona": 0, "coherent_still_qwen": 0},
            }
        result = blind86.verdicts_from_counts({"layers": layers})["verdicts"]
        self.assertEqual(result["86.6"]["status"], "FAILS")

    def test_861_is_hand_read_and_automatic_match_is_non_authoritative(self):
        with tempfile.TemporaryDirectory() as tmp:
            screen = Path(tmp)
            (screen / "layer_22.jsonl").write_text(json.dumps({
                "layer": 22, "kind": "neuron", "id": 2742,
                "answer": "I am not from space, but I am a quiet traveler.",
            }) + "\n", encoding="utf-8")
            result = blind86._target_offworld(screen)
            self.assertEqual(result["status"], "NOT READ")
            self.assertEqual(result["reading"], "hand-read")
            self.assertFalse(result["automatic_match"]["authoritative"])
            self.assertEqual(result["automatic_match"]["status"], "HOLDS")

    def test_kappa_binary_mapping_and_scoring_denominators(self):
        primary = {
            "a": {"coherent": True, "kind": "human"},
            "b": {"coherent": True, "kind": "still Qwen or an assistant"},
        }
        self.assertEqual(blind86.cohen_kappa(primary, primary, primary), 1.0)
        counts = {"22": {"real": {"n": 256, "coherent_persona": 2},
                         "random": {"n": 512, "changed_labeled": 0, "not_coherent": 0,
                                    "unclear": 0, "coherent_persona": 0, "coherent_still_qwen": 0}}}
        with self.assertRaisesRegex(ValueError, "denominator must be 9216"):
            blind86.verdicts_from_counts({"layers": counts})
        full = {str(layer): {"real": {"n": 9216}, "random": {"n": 512}}
                for layer in range(10, 27)}
        full["22"]["real"]["n"] = 9000
        with self.assertRaisesRegex(ValueError, "denominator mismatch"):
            blind86._assert_screen_denominators(full)
        with self.assertRaisesRegex(ValueError, "all 17 layers"):
            blind86._assert_screen_denominators(counts)


class TestSurvival(unittest.TestCase):
    def test_verification_prepare_chunks_second_reader_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "verify-run"
            source.mkdir()
            for neuron in range(20):
                layer_no = 10 + neuron % 17
                persona = f"persona {neuron}"
                config = {
                    "model_id": common86.MODEL_ID, "revision": common86.REVISION,
                    "layer": layer_no, "neuron": neuron, "persona": persona,
                    "primary_screen_persona": persona, "persona_equivalences": [],
                    "calibration_sha256": common86.sha256_text("calibration"), "median": 24.8,
                    "dose": 0.8, "actual_dtype": "torch.bfloat16",
                    "versions": common86.dependency_versions(), "code_sha256": common86.verification_code_sha256(),
                    "generation_code_sha256": common86.generation_code_sha256(),
                    "questions": list(common86.VERIFY_QUESTIONS), "samples": 8, "directions": 10,
                    "max_new_tokens": 64, "temperature": 0.7, "top_p": 0.95,
                    "mode": "generated_tokens", "generation_config": {"effective": {
                        "do_sample": True, "max_new_tokens": 64, "temperature": 0.7, "top_p": 0.95,
                    }},
                    "random_seed": common86.RANDOM_VERIFY_SEED + layer_no * 10000 + neuron,
                    "screen_manifest_sha256": "a" * 64,
                    "primary_labels_sha256": "b" * 64,
                    "selected_set_sha256": "c" * 64,
                }
                records = []
                for condition, direction in [("neuron", None)] + [("random_direction", d) for d in range(10)]:
                    for qi in range(3):
                        for sample in range(8):
                            seed = verify86.seed_for(qi, sample)
                            trace = {
                                "layer": layer_no, "neuron": neuron, "condition": condition,
                                "direction": direction, "question_index": qi, "sample": sample,
                                "seed": seed, "question": common86.VERIFY_QUESTIONS[qi],
                                "do_sample": True, "temperature": 0.7, "top_p": 0.95,
                                "max_new_tokens": 64, "prompt_sha256": "0" * 64,
                                "prefill_changed": False, "decode_calls_changed": 1,
                                "changed_count": 1 if condition == "neuron" else 16,
                                "changed_channels_head": [neuron] if condition == "neuron" else list(range(16)),
                                "first_decode_write_norm": 19.84, "median": 24.8, "dose": 0.8,
                            }
                            if direction is not None:
                                trace["direction_sha256"] = hashlib.sha256(
                                    f"{layer_no}-{neuron}-{direction}".encode()
                                ).hexdigest()
                            records.append({
                                "layer": layer_no, "condition": condition, "neuron": neuron,
                                "direction": direction, "question_index": qi, "sample": sample,
                                "seed": seed, "question": common86.VERIFY_QUESTIONS[qi],
                                "persona": persona, "answer": "A calm traveler.", "median": 24.8,
                                "dose": 0.8, "max_new_tokens": 64, "temperature": 0.7,
                                "top_p": 0.95, "trace": trace,
                            })
                (source / f"L{layer_no}_N{neuron}.json").write_text(
                    json.dumps({"status": "complete", "config": config, "records": records}), encoding="utf-8")
            agreement_provenance = {key: config[key] for key in (
                "screen_manifest_sha256", "primary_labels_sha256", "selected_set_sha256")}
            items_path = root / "reader" / "verify-items.jsonl"
            key_path = root / "private" / "verify-key.json"
            self.assertEqual(verify86.prepare(source, items_path, key_path), 5280)
            key = json.loads(key_path.read_text(encoding="utf-8"))
            self.assertEqual(len(key["second_reader_ids"]), 528)
            self.assertIn("manifest_sha256", key["manifest"])
            self.assertEqual(len(key["manifest"]["input_files_sha256"]), 20)
            self.assertEqual(len(key["manifest"]["configs"]), 20)
            for name, value in agreement_provenance.items():
                self.assertEqual(key["manifest"][name], value)
            first_chunks = sorted((items_path.parent / "verify-items-reader1-chunks").glob("*.jsonl"))
            second_chunks = sorted((items_path.parent / "verify-items-reader2-chunks").glob("*.jsonl"))
            self.assertEqual(len(first_chunks), 11)
            self.assertTrue(all(len(path.read_text(encoding="utf-8").splitlines()) <= 500 for path in first_chunks))
            self.assertEqual(len(second_chunks), 2)
            labels = [{"id": item_id, "claims_persona": False} for item_id in key["rows"]]
            labels2 = [{"id": item_id, "claims_persona": False} for item_id in key["second_reader_ids"]]
            passing_screen_score = _write_screen_score_fixture(root / "screen-pass", 0.6, agreement_provenance)
            result = verify86.score(labels, key, items_path, labels2, passing_screen_score)
            self.assertEqual(result["reader_agreement"]["kappa"], 1.0)
            self.assertEqual(result["verdict_86.5"]["status"], "FAILS")
            self.assertEqual(result["screen_reader_agreement"]["status"], "READ")
            self.assertEqual(len(result["manifest"]["verification_input_files_sha256"]), 20)
            self.assertIn("manifest_sha256", result["manifest"])
            self.assertLess(list(result).index("reader_agreement"), list(result).index("verdict_86.5"))
            unrelated_agreement = _write_screen_score_fixture(
                root / "screen-unrelated", 0.99,
                {**agreement_provenance, "selected_set_sha256": "d" * 64},
            )
            with self.assertRaisesRegex(ValueError, "selected_set_sha256 does not match verification provenance"):
                verify86.score(labels, key, items_path, labels2, unrelated_agreement)
            low_labels2 = [{"id": item_id, "claims_persona": True} for item_id in key["second_reader_ids"]]
            low = verify86.score(labels, key, items_path, low_labels2, passing_screen_score)
            self.assertEqual(low["reader_agreement"]["kappa"], 0.0)
            self.assertEqual(low["verdict_86.5"]["status"], "NOT READ")
            low_screen_score = _write_screen_score_fixture(root / "screen-low", 0.5999, agreement_provenance)
            screen_gated = verify86.score(labels, key, items_path, labels2, low_screen_score)
            self.assertEqual(screen_gated["reader_agreement"]["kappa"], 1.0)
            self.assertEqual(screen_gated["verdict_86.5"]["status"], "NOT READ")
            missing_screen = verify86.score(labels, key, items_path, labels2)
            self.assertEqual(missing_screen["verdict_86.5"]["status"], "NOT READ")
            with self.assertRaisesRegex(ValueError, "file SHA-256"):
                passing_screen_score.write_text(passing_screen_score.read_text() + " ", encoding="utf-8")
                verify86.score(labels, key, items_path, labels2, passing_screen_score)

            mixed_path = source / "L10_N0.json"
            good_payload = mixed_path.read_bytes()
            incomplete = json.loads(good_payload)
            incomplete["records"].pop()
            mixed_path.write_text(json.dumps(incomplete), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 264"):
                verify86.prepare(source, root / "incomplete-reader" / "items.jsonl", root / "incomplete-key.json")
            mixed_path.write_bytes(good_payload)
            mixed = json.loads(mixed_path.read_text(encoding="utf-8"))
            mixed["config"]["dose"] = 0.4
            for record in mixed["records"]:
                record["dose"] = 0.4
                record["trace"]["dose"] = 0.4
            mixed_path.write_text(json.dumps(mixed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "registered verification generation config"):
                verify86.prepare(source, root / "mixed-reader" / "items.jsonl", root / "mixed-key.json")

            labels_path = root / "verify-labels.jsonl"
            labels2_path = root / "verify-labels-2.jsonl"
            labels_path.write_text("".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8")
            labels2_path.write_text("".join(json.dumps(row) + "\n" for row in labels2), encoding="utf-8")
            screen_cli_pass = _write_screen_score_fixture(root / "screen-cli-pass", 0.6, agreement_provenance)
            pass_out = root / "verify-score-pass.json"
            verify86.main([
                "score", "--items", str(items_path), "--key", str(key_path),
                "--labels", str(labels_path), "--labels2", str(labels2_path),
                "--screen-score", str(screen_cli_pass), "--out", str(pass_out),
            ])
            self.assertEqual(json.loads(pass_out.read_text())["verdict_86.5"]["status"], "FAILS")
            low_out = root / "verify-score-low.json"
            verify86.main([
                "score", "--items", str(items_path), "--key", str(key_path),
                "--labels", str(labels_path), "--labels2", str(labels2_path),
                "--screen-score", str(low_screen_score), "--out", str(low_out),
            ])
            self.assertEqual(json.loads(low_out.read_text())["verdict_86.5"]["status"], "NOT READ")

    def test_registered_survival_rule_and_86_5_ten_of_twenty_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping, labels, items = {}, [], []
            serial = 0
            for neuron in range(20):
                for condition, direction in [("neuron", None)] + [("random_direction", d) for d in range(10)]:
                    for qi in range(3):
                        for sample in range(8):
                            item_id = f"x{serial:05d}"
                            serial += 1
                            items.append({"id": item_id, "persona": f"persona {neuron}", "text": "answer"})
                            mapping[item_id] = {
                                "layer": 10 + neuron % 17, "condition": condition, "neuron": neuron,
                                "direction": direction, "question_index": qi, "sample": sample,
                                "seed": 1, "persona": f"persona {neuron}",
                            }
                            if condition == "neuron":
                                claim = qi * 8 + sample < (12 if neuron < 10 else 11)
                            elif neuron == 0 and direction == 0:
                                claim = qi * 8 + sample < 1
                            elif neuron == 1 and direction == 0:
                                claim = qi * 8 + sample < 2
                            else:
                                claim = False
                            labels.append({"id": item_id, "claims_persona": claim})
            content = "".join(json.dumps(row) + "\n" for row in items).encode()
            items_path = Path(tmp) / "items.jsonl"
            items_path.write_bytes(content)
            key = {"items_sha256": hashlib.sha256(content).hexdigest(), "rows": mapping}
            agreement_provenance = {
                "screen_manifest_sha256": "a" * 64,
                "primary_labels_sha256": "b" * 64,
                "selected_set_sha256": "c" * 64,
            }
            key["manifest"] = dict(agreement_provenance)
            key["manifest"]["manifest_sha256"] = common86.canonical_sha256(key["manifest"])
            passing_screen_score = _write_screen_score_fixture(
                Path(tmp) / "screen-score", 0.6, agreement_provenance,
            )
            result = verify86.score(labels, key, items_path, labels2=labels, screen_score=passing_screen_score)
            self.assertEqual(result["survivors"], 9)
            self.assertEqual(result["verdict_86.5"]["status"], "FAILS")
            self.assertTrue(result["per_neuron"][0]["survives"])
            by_neuron = {row["neuron"]: row for row in result["per_neuron"]}
            self.assertFalse(by_neuron[1]["survives"])  # two random claims fail the <=1 edge
            self.assertFalse(by_neuron[10]["survives"])  # eleven neuron claims fail the >=12 edge
            for row in labels:
                info = mapping[row["id"]]
                if (info["neuron"] == 10 and info["condition"] == "neuron"
                        and info["question_index"] == 1 and info["sample"] == 3):
                    row["claims_persona"] = True
            result = verify86.score(labels, key, items_path, labels2=labels, screen_score=passing_screen_score)
            self.assertEqual(result["survivors"], 10)
            self.assertEqual(result["verdict_86.5"]["status"], "HOLDS")


class TestBlind(unittest.TestCase):
    def _fixture(self, temp):
        screen = temp / "screen"
        screen.mkdir()
        code = common86.code_sha256()
        for layer in range(10, 27):
            rows = [{"layer": layer, "kind": "random_direction", "id": 2,
                     "answer": "I am an AI assistant." if layer != 22 else "A curious machine studies the stars."}]
            if layer == 22:
                rows.extend([
                    {"layer": 22, "kind": "neuron", "id": 5, "answer": "A quiet traveler watches the sea."},
                    {"layer": 22, "kind": "neuron", "id": 6, "answer": "I am an AI assistant."},
                ])
            output = screen / f"layer_{layer}.jsonl"
            output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            config = {"code_sha256": code, "layer": layer, "median": 24.8, "dose": 0.8,
                      "actual_dtype": "torch.bfloat16", "versions": common86.dependency_versions()}
            (screen / f"layer_{layer}.meta.json").write_text(
                json.dumps({"status": "complete", "config": config, "median": 24.8}), encoding="utf-8")
        counts = triage86.triage(screen)
        triage_path = temp / "triage.json"
        triage_path.write_text(json.dumps(counts), encoding="utf-8")
        items_path, key_path = temp / "reader" / "items.jsonl", temp / "private" / "key.json"
        count = blind86.prepare(screen, triage_path, items_path, key_path)
        return screen, triage_path, items_path, key_path, count

    def test_blinding_manifest_triage_rerun_and_output_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            screen, triage_path, items_path, key_path, count = self._fixture(temp)
            self.assertEqual(count, 19)
            items = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(items)
            self.assertTrue(all(set(row) == {"id", "text"} for row in items))
            key = json.loads(key_path.read_text(encoding="utf-8"))
            self.assertEqual(key["items_sha256"], hashlib.sha256(items_path.read_bytes()).hexdigest())
            self.assertIn("manifest_sha256", key["manifest"])
            self.assertTrue(0 < len(key["second_reader_ids"]) <= len(items))
            for reader in ("reader1", "reader2"):
                chunks = sorted((items_path.parent / f"items-{reader}-chunks").glob("*.jsonl"))
                self.assertTrue(chunks)
                self.assertTrue(all(len(path.read_text(encoding="utf-8").splitlines()) <= 500 for path in chunks))
            self.assertEqual(len(list((items_path.parent / "items-reader2-chunks").glob("*.jsonl"))), 1)
            with self.assertRaises(FileExistsError):
                blind86.prepare(screen, triage_path, items_path, key_path)
            with self.assertRaisesRegex(ValueError, "outside the folder"):
                blind86.prepare(screen, triage_path, temp / "same" / "items.jsonl", temp / "same" / "key.json")
            good = [
                {"id": row["id"], "persona": "a traveler", "kind": "human", "coherent": True}
                for row in items
            ]
            scored = blind86.score(good, key, items_path, screen)
            self.assertEqual(sum(g["real"]["coherent_persona"] + g["random"]["coherent_persona"]
                                 for g in scored["layers"].values()), 2)
            self.assertEqual(scored["triage_miss_rates"]["random"]["kept"]["sampled"], 16)
            self.assertEqual(scored["triage_miss_rates"]["random"]["kept"]["rate"], 1.0)
            self.assertIn("code_sha256", scored["scorer_manifest"])
            changed_scorer_hashes = {"blind86.py": "f" * 64, "triage86.py": "e" * 64,
                                     "common86.py": "d" * 64}
            with mock.patch.object(common86, "scorer_code_sha256", return_value=changed_scorer_hashes):
                scorer_edit = blind86.score(good, key, items_path, screen)
            self.assertEqual(scorer_edit["manifest"], scored["manifest"])
            self.assertEqual(scorer_edit["scorer_manifest"]["code_sha256"], changed_scorer_hashes)
            with self.assertRaises(ValueError):
                blind86.score(good[:-1], key, items_path, screen)
            with self.assertRaises(ValueError):
                blind86.score(good + [good[0]], key, items_path, screen)
            bad_kind = [dict(x) for x in good]
            bad_kind[0]["kind"] = "space-being"
            with self.assertRaises(ValueError):
                blind86.score(bad_kind, key, items_path, screen)
            bad_category = json.loads(key_path.read_text(encoding="utf-8"))
            changed_id = next(item_id for item_id, info in bad_category["rows"].items()
                              if info["triage_category"] == "kept")
            bad_category["rows"][changed_id]["triage_category"] = "changed"
            with self.assertRaisesRegex(ValueError, "triage category"):
                blind86.score(good, bad_category, items_path, screen)
            items_path.write_text(items_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                blind86.score(good, key, items_path, screen)

    def test_picked20_must_name_a_changed_real_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            screen = Path(tmp) / "screen"
            screen.mkdir()
            screen_rows = [{"layer": 10, "kind": "neuron", "id": i,
                            "answer": f"A quiet traveler watches the sea number {i}."} for i in range(20)]
            (screen / "layer_10.jsonl").write_text("".join(json.dumps(row) + "\n" for row in screen_rows), encoding="utf-8")
            mapping = {f"x{i}": {"layer": 10, "source": "real", "item_index": i,
                                  "triage_category": "changed", "miss_rate_sample": False}
                       for i in range(20)}
            labels = [{"id": item_id, "persona": "traveler", "kind": "human", "coherent": True}
                      for item_id in mapping]
            picks = [{"layer": 10, "neuron": i, "persona": "traveler"} for i in range(20)]
            verify86.validate_picked_against_screen(picks, screen, {"rows": mapping}, labels)
            wrong_persona = [dict(row) for row in picks]
            wrong_persona[0]["persona"] = "a planet-ruling deity"
            with self.assertRaisesRegex(ValueError, "does not match primary blind label"):
                verify86.validate_picked_against_screen(wrong_persona, screen, {"rows": mapping}, labels)
            equivalence_pick = [dict(row) for row in picks]
            equivalence_pick[0]["persona"] = "a sea wanderer"
            verify86.validate_picked_against_screen(
                equivalence_pick, screen, {"rows": mapping}, labels,
                [{"primary": "traveler", "picked": "a sea wanderer"}],
            )
            labels[0]["coherent"] = False
            with self.assertRaisesRegex(ValueError, "not a coherent persona"):
                verify86.validate_picked_against_screen(picks, screen, {"rows": mapping}, labels)
            with self.assertRaisesRegex(ValueError, "not triaged changed"):
                verify86.validate_picked_against_screen(
                    [{"layer": 10, "neuron": 99, "persona": "traveler"}], screen, {"rows": mapping}, labels)


class TestResumeAndScreen(unittest.TestCase):
    @unittest.skipIf(torch is None, f"torch/transformers unavailable: {MODEL_IMPORT_ERROR}")
    def test_residual_median_excludes_prompt_token_zero(self):
        hidden = torch.tensor([[[100.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 4.0]]])

        class FixedLayer(torch.nn.Module):
            def forward(self, _input_ids):
                return hidden

        layer = FixedLayer()

        class FixedModel:
            def __call__(self, input_ids, use_cache):
                self.layer(input_ids)

        model = FixedModel()
        model.layer = layer
        median = common86.residual_median(model, TinyTokenizer(), layer, "calibration text", "cpu")
        self.assertEqual(median, 2.0)

    @unittest.skipIf(torch is None, f"torch/transformers unavailable: {MODEL_IMPORT_ERROR}")
    def test_runtime_zero_push_gates_same_batch_baseline_and_records_batch_effect(self):
        width = 8
        pushes = torch.ones((2, width), dtype=torch.float32)
        args = SimpleNamespace(out=tempfile.mkdtemp(), device="cpu")

        def generated(answer, token_ids):
            return {"answer": answer, "token_ids": token_ids, "token_count": len(token_ids)}

        batch = [generated("I am Qwen, a quiet assistant.", [1, 2]) for _ in range(16)]
        batch[2] = generated("A pushed traveler watches the sea.", [2, 3])
        batch[3] = generated("A pushed sailor watches the sea.", [3, 4])
        singleton = [generated("I am Qwen, a quiet assistant.", [1, 9])]
        pushed_a = [generated("A pushed traveler watches the sea.", [2, 8])]
        pushed_b = [generated("A pushed sailor watches the sea.", [3, 7])]
        zero_trace = {
            "prefill_changed": False, "decode_calls_changed": 0, "changed_count": 0,
            "changed_channels_head": [], "first_decode_write_norm": 0.0,
            "direction_sha256": "0" * 64, "max_new_tokens": 48, "do_sample": False,
            "question": "Who are you?", "prompt_sha256": "1" * 64,
        }
        with mock.patch.object(common86, "generate_batch", side_effect=[(batch, {"trace_rows": [zero_trace] * 16}), (singleton, {}),
                                                                          (pushed_a, {}), (pushed_b, {})]):
            canary = screen86._runtime_canaries(None, None, None, args, pushes, width, 22, 24.8, {"layer": 22})
        self.assertTrue(canary["zero_push"]["matches_unsteered_same_batch_shape"])
        self.assertEqual((canary["zero_trace"]["kind"], canary["zero_trace"]["target"]), ("zero", None))
        self.assertFalse(canary["bf16_batch_effect"]["exact_token_match"])
        self.assertEqual(canary["bf16_batch_effect"]["batch16_token_ids"], [1, 2])
        self.assertEqual(canary["bf16_batch_effect"]["batch1_token_ids"], [1, 9])

        mismatched = list(batch)
        mismatched[1] = generated("I am Qwen, a quiet assistant.", [7, 7])
        with mock.patch.object(common86, "generate_batch", return_value=(mismatched, {})), \
             self.assertRaisesRegex(RuntimeError, "same batch shape"):
            screen86._runtime_canaries(None, None, None, args, pushes, width, 22, 24.8, {"layer": 22})

        bad_zero = list(batch)
        bad_zero[0] = generated("A quiet traveler watches the sea.", [1, 2])
        with mock.patch.object(common86, "generate_batch", return_value=(bad_zero, {})), \
             self.assertRaisesRegex(RuntimeError, "must triage kept"):
            screen86._runtime_canaries(None, None, None, args, pushes, width, 22, 24.8, {"layer": 22})

    def test_pilot_gate_limited_population_is_not_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{"answer": "A quiet traveler watches the sea."} for _ in range(4)]
            screen86._write_pilot_gate(Path(tmp), 22, rows, 24.8)
            gate = json.loads((Path(tmp) / "pilot-gate.json").read_text(encoding="utf-8"))
            self.assertEqual(gate["gate_3_status"], "NOT READ")
            self.assertEqual(gate["observed_rows"], 4)
            full_population = [{"answer": "I am Qwen, a language model assistant."}
                               for _ in range(screen86.N_NEURONS + screen86.N_DIRECTIONS)]
            screen86._write_pilot_gate(Path(tmp), 22, full_population, 24.8)
            gate = json.loads((Path(tmp) / "pilot-gate.json").read_text(encoding="utf-8"))
            self.assertEqual(gate["gate_3_status"], "FAIL")
            self.assertEqual(gate["changed_answers"], 0)

    def test_gate_2_12_hour_boundary_and_resumed_pilot_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            population = screen86.N_NEURONS + screen86.N_DIRECTIONS
            projected_answers = population * 17
            boundary_elapsed = 12 * 60 * 60 * population / projected_answers
            full = [{"answer": "A quiet traveler watches the sea.", "batch_id": "completed-batches",
                     "generation_elapsed_seconds": boundary_elapsed} for _ in range(population)]
            screen86._write_pilot_gate(out, 22, full, 24.8)
            exact = json.loads((out / "pilot-gate.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(exact["projected_full_screen_seconds"], 12 * 60 * 60)
            self.assertEqual(exact["gate_2_status"], "PASS")

            full = [dict(row, generation_elapsed_seconds=math.nextafter(boundary_elapsed, math.inf))
                    for row in full]
            screen86._write_pilot_gate(out, 22, full, 24.8)
            just_over = json.loads((out / "pilot-gate.json").read_text(encoding="utf-8"))
            self.assertGreater(just_over["projected_full_screen_seconds"], 12 * 60 * 60)
            self.assertEqual(just_over["gate_2_status"], "FAIL")

            # The pilot projection sums persisted batch durations from before and after a resume,
            # then adds the canary's separately persisted timing.
            resumed = []
            for index in range(population):
                batch_no = index // 16
                resumed.append({
                    "answer": "A quiet traveler watches the sea.",
                    "batch_id": f"batch-{batch_no:04d}",
                    "generation_elapsed_seconds": 0.01 if batch_no < 304 else 0.02,
                })
            common86.atomic_json(out / "runtime-canaries.json", {
                "generation_elapsed_seconds": 0.38, "generation_answers": 19,
            })
            screen86._write_pilot_gate(out, 22, resumed, 24.8)
            projected = json.loads((out / "pilot-gate.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(projected["screen_generation_elapsed_seconds"], 9.12)
            self.assertAlmostEqual(projected["generation_elapsed_seconds"], 9.5)
            self.assertEqual(projected["generation_answers"], population + 19)
            self.assertAlmostEqual(
                projected["projected_full_screen_seconds"],
                projected_answers / ((population + 19) / 9.5),
            )
            self.assertEqual(projected["gate_2_status"], "PASS")

    @unittest.skipIf(torch is None, f"torch/transformers unavailable: {MODEL_IMPORT_ERROR}")
    def test_verify_median_uses_relative_tolerance(self):
        selected = {"layer": 10, "neuron": 2, "persona": "traveler"}
        calibration_hash = common86.sha256_text("calibration")
        config = {
            "code_sha256": common86.generation_code_sha256(),
            "versions": common86.dependency_versions(),
            "actual_dtype": "torch.bfloat16", "dose": common86.DOSE,
            "model_id": common86.MODEL_ID, "revision": common86.REVISION,
            "layer": 10, "calibration_sha256": calibration_hash, "median": 24.8,
        }
        meta = {"config": config, "median": 24.8}
        model = SimpleNamespace(generation_config=None)
        self.assertAlmostEqual(
            verify86._verify_config(selected, 24.8 + 1e-7, calibration_hash, "torch.bfloat16",
                                    meta, TinyTokenizer(), model)["median"],
            24.8,
        )
        with self.assertRaisesRegex(ValueError, "differs from screen median"):
            verify86._verify_config(selected, 24.8 + 1e-4, calibration_hash, "torch.bfloat16",
                                    meta, TinyTokenizer(), model)

    @unittest.skipIf(torch is None, f"torch/transformers unavailable: {MODEL_IMPORT_ERROR}")
    def test_orphan_batch_fragment_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            calib = temp / "calibration.txt"
            calib.write_text("calibration fixture", encoding="utf-8")
            out = temp / "out"
            (out / ".batches").mkdir(parents=True)
            (out / ".batches" / "layer_0.neuron_00000.jsonl").write_text("{}\n", encoding="utf-8")
            model, tokenizer = tiny_model(), TinyTokenizer()
            args = _tiny_screen_args(out, calib)
            with mock.patch.object(common86, "MODEL_ID", "tiny-test"), \
                 mock.patch.object(common86, "REVISION", "tiny-revision"), \
                 mock.patch.object(common86, "load_model", return_value=(tokenizer, model)), \
                 mock.patch.object(common86, "residual_median", return_value=1.25), \
                 self.assertRaisesRegex(ValueError, "orphaned batch fragments"):
                screen86.run(args)

    @unittest.skipIf(torch is None, f"torch/transformers unavailable: {MODEL_IMPORT_ERROR}")
    def test_invalid_reset_layer_config_does_not_clear_any_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            calib = temp / "calibration.txt"
            calib.write_text("calibration fixture", encoding="utf-8")
            out = temp / "out"
            out.mkdir()
            final = out / "layer_1.jsonl"
            meta = out / "layer_1.meta.json"
            final.write_text("preserve\n", encoding="utf-8")
            meta.write_text("{}\n", encoding="utf-8")
            model, tokenizer = tiny_model(), TinyTokenizer()
            args = _tiny_screen_args(out, calib, layers=[0], reset_layer=[1])
            with mock.patch.object(common86, "MODEL_ID", "tiny-test"), \
                 mock.patch.object(common86, "REVISION", "tiny-revision"), \
                 mock.patch.object(common86, "load_model", return_value=(tokenizer, model)), \
                 mock.patch.object(common86, "residual_median", return_value=1.25), \
                 self.assertRaisesRegex(ValueError, "must also appear"):
                screen86.run(args)
            self.assertTrue(final.is_file())
            self.assertEqual(final.read_text(encoding="utf-8"), "preserve\n")
            self.assertTrue(meta.is_file())

    def test_verify_neuron_resume_refuses_changed_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "L10_N2.json"
            path.write_text(json.dumps({"status": "running", "config": {"dose": 0.8}, "records": []}),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "resume provenance differs"):
                verify86._load_neuron_state(path, {"dose": 0.9})

    @unittest.skipIf(torch is None, f"torch/transformers unavailable: {MODEL_IMPORT_ERROR}")
    def test_screen_smoke_provenance_resume_refusal_and_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            calib = temp / "calibration.txt"
            calib.write_text("calibration fixture", encoding="utf-8")
            model, tokenizer = tiny_model(), TinyTokenizer()
            args = _tiny_screen_args(temp / "out", calib)
            with mock.patch.object(common86, "MODEL_ID", "tiny-test"), \
                 mock.patch.object(common86, "REVISION", "tiny-revision"), \
                 mock.patch.object(common86, "load_model", return_value=(tokenizer, model)), \
                 mock.patch.object(common86, "residual_median", return_value=1.25):
                screen86.run(args)
            final = temp / "out" / "layer_0.jsonl"
            rows = [json.loads(line) for line in final.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual(sum(row["kind"] == "neuron" for row in rows), 2)
            self.assertEqual(sum(row["kind"] == "random_direction" for row in rows), 2)
            self.assertTrue(all(row["hook_fire_counts"]["prefill_calls"] == 1 for row in rows))
            meta_path = temp / "out" / "layer_0.meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["status"], "complete")
            self.assertEqual(meta["actual_dtype"], "torch.bfloat16")
            self.assertIn("neuron_report.py", meta["code_sha256"])
            self.assertIn("observe.py", meta["code_sha256"])
            self.assertEqual(meta["median"], 1.25)
            self.assertTrue((temp / "out" / "manifest.json").is_file())
            with mock.patch.object(common86, "MODEL_ID", "tiny-test"), \
                 mock.patch.object(common86, "REVISION", "tiny-revision"), \
                 mock.patch.object(common86, "load_model", return_value=(tokenizer, model)), \
                 mock.patch.object(common86, "residual_median", return_value=1.5), \
                 self.assertRaisesRegex(ValueError, "median differs"):
                screen86.run(args)
            changed_hashes = dict(common86.code_sha256())
            changed_hashes["screen86.py"] = "0" * 64
            with mock.patch.object(common86, "MODEL_ID", "tiny-test"), \
                 mock.patch.object(common86, "REVISION", "tiny-revision"), \
                 mock.patch.object(common86, "load_model", return_value=(tokenizer, model)), \
                 mock.patch.object(common86, "residual_median", return_value=1.25), \
                 mock.patch.object(common86, "generation_code_sha256", return_value=changed_hashes), \
                 self.assertRaisesRegex(ValueError, "configuration differs"):
                screen86.run(args)
            fragments = list((temp / "out" / ".batches").glob("layer_0.*"))
            self.assertTrue(fragments)
            reset_args = _tiny_screen_args(temp / "out", calib, reset_layer=[0])
            with mock.patch.object(common86, "MODEL_ID", "tiny-test"), \
                 mock.patch.object(common86, "REVISION", "tiny-revision"), \
                 mock.patch.object(common86, "load_model", return_value=(tokenizer, model)), \
                 mock.patch.object(common86, "residual_median", return_value=1.5), \
                 self.assertRaisesRegex(ValueError, "median differs"):
                screen86.run(reset_args)
            self.assertTrue(final.is_file(), "a bad reset configuration must not delete completed answers")
            with mock.patch.object(common86, "MODEL_ID", "tiny-test"), \
                 mock.patch.object(common86, "REVISION", "tiny-revision"), \
                 mock.patch.object(common86, "load_model", return_value=(tokenizer, model)), \
                 mock.patch.object(common86, "residual_median", return_value=1.25), \
                 mock.patch.object(common86, "generate_batch", wraps=common86.generate_batch) as generate:
                screen86.run(reset_args)
                self.assertGreater(generate.call_count, 0, "reset must clear complete batch state")
            self.assertTrue((temp / "out" / "layer_0.jsonl").is_file())

    @unittest.skipIf(torch is None, f"torch/transformers unavailable: {MODEL_IMPORT_ERROR}")
    def test_verify_skips_completed_neuron_file_with_same_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            screen = temp / "screen"
            screen.mkdir()
            rows = [{"layer": 10, "kind": "neuron", "id": i,
                     "answer": f"A distinct traveler watches the quiet sea number {i}."} for i in range(20)]
            (screen / "layer_10.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            config = {"code_sha256": common86.code_sha256(), "versions": common86.dependency_versions(),
                      "actual_dtype": "torch.bfloat16", "dose": 0.8, "median": 24.8,
                      "model_id": "tiny-audit-test", "revision": "tiny-audit-revision", "layer": 10,
                      "calibration_sha256": common86.sha256_text("calibration")}
            (screen / "layer_10.meta.json").write_text(json.dumps({"status": "complete", "config": config, "median": 24.8}), encoding="utf-8")
            screen_manifest = {"fixture": True}
            screen_manifest["manifest_sha256"] = common86.canonical_sha256(screen_manifest)
            (screen / "manifest.json").write_text(json.dumps(screen_manifest), encoding="utf-8")
            picked_path = temp / "picked.json"
            picked_path.write_text(json.dumps([{"layer": 10, "neuron": i, "persona": f"persona {i}"}
                                               for i in range(20)]), encoding="utf-8")
            blind_key = {"rows": {
                f"x{i}": {"layer": 10, "source": "real", "item_index": i,
                          "triage_category": "changed", "miss_rate_sample": False}
                for i in range(20)
            }}
            blind_key_path = temp / "blind-key.json"
            blind_key_path.write_text(json.dumps(blind_key), encoding="utf-8")
            blind_labels_path = temp / "blind-labels.jsonl"
            blind_labels_path.write_text("".join(json.dumps({
                "id": f"x{i}", "persona": f"persona {i}", "kind": "human", "coherent": True
            }) + "\n" for i in range(20)), encoding="utf-8")
            calib = temp / "calib.txt"
            calib.write_text("calibration", encoding="utf-8")
            tokenizer = TinyTokenizer()
            model = tiny_audit_model()
            args = SimpleNamespace(picked=str(picked_path), screen_dir=str(screen),
                                   blind_key=str(blind_key_path), blind_labels=str(blind_labels_path), calib=str(calib),
                                   out=str(temp / "verify"), device="cpu", max_neurons=None)
            fake_case = [("neuron", None, "Who are you?", 0, 0, verify86.seed_for(0, 0), None)]
            with mock.patch.object(verify86, "ANSWERS_PER_NEURON", 1), \
                 mock.patch.object(verify86, "_expected_case_keys", return_value={("neuron", None, 0, 0)}), \
                 mock.patch.object(verify86, "_case_plan", return_value=fake_case), \
                 mock.patch.object(common86, "MODEL_ID", "tiny-audit-test"), \
                 mock.patch.object(common86, "REVISION", "tiny-audit-revision"), \
                 mock.patch.object(common86, "load_model", return_value=(tokenizer, model)), \
                 mock.patch.object(common86, "residual_median", return_value=24.8), \
                 mock.patch.object(common86, "generate_batch", wraps=common86.generate_batch) as generate:
                verify86.run(args)
                self.assertEqual(generate.call_count, 20)
                generate.reset_mock()
                verify86.run(args)
                generate.assert_not_called()
            outputs = list((temp / "verify").glob("L*_N*.json"))
            self.assertEqual(len(outputs), 20)
            self.assertTrue(all(json.loads(path.read_text(encoding="utf-8"))["status"] == "complete" for path in outputs))
            partial_path = temp / "verify" / "L10_N0.json"
            partial = json.loads(partial_path.read_text(encoding="utf-8"))
            partial["status"] = "running"
            partial_path.write_text(json.dumps(partial), encoding="utf-8")
            with mock.patch.object(verify86, "ANSWERS_PER_NEURON", 1), \
                 mock.patch.object(verify86, "_expected_case_keys", return_value={("neuron", None, 0, 0)}), \
                 mock.patch.object(verify86, "_case_plan", return_value=fake_case), \
                 mock.patch.object(common86, "MODEL_ID", "tiny-audit-test"), \
                 mock.patch.object(common86, "REVISION", "tiny-audit-revision"), \
                 mock.patch.object(common86, "load_model", return_value=(tokenizer, model)), \
                 mock.patch.object(common86, "residual_median", return_value=24.8), \
                 mock.patch.object(common86, "generate_batch", wraps=common86.generate_batch) as generate:
                verify86.run(args)
                generate.assert_not_called()


class TestLiveTraceAudit(unittest.TestCase):
    @unittest.skipIf(torch is None, f"torch/transformers unavailable: {MODEL_IMPORT_ERROR}")
    def test_real_screen_and_verify_loops_record_auditable_pushes_and_resume_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            calib = temp / "calibration.txt"
            calib.write_text("calibration fixture", encoding="utf-8")
            out = temp / "screen"
            tokenizer, model = TinyAuditTokenizer(), tiny_audit_model()
            prompt_texts = _tiny_prompt_texts_file(temp / "prompt-texts.json", tokenizer)
            screen_args = _tiny_screen_args(out, calib, layers=[22])

            def run_screen():
                with mock.patch.object(common86, "MODEL_ID", "tiny-audit-test"), \
                     mock.patch.object(common86, "REVISION", "tiny-audit-revision"), \
                     mock.patch.object(common86, "load_model", return_value=(tokenizer, model)):
                    screen86.run(screen_args)

            def audit(command, *arguments):
                result = subprocess.run(
                    [sys.executable, "-I", "-B",
                     str(HERE / "audit86.py"), command, *arguments],
                    cwd=ROOT, capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("TEST FIXTURE OVERRIDES", result.stdout)

            run_screen()
            answer_rows = [json.loads(line) for line in (out / "layer_22.jsonl").read_text().splitlines()]
            self.assertEqual(len(answer_rows), 4)
            self.assertEqual(sum(row["kind"] == "neuron" for row in answer_rows), 2)
            pilot_gate = json.loads((out / "pilot-gate.json").read_text())
            self.assertEqual(pilot_gate["gate_2_status"], "EARLY ESTIMATE ONLY (limited pilot; not authoritative)")
            self.assertEqual(pilot_gate["screen_generation_answers"], 4)
            self.assertGreater(pilot_gate["canary_generation_answers"], 0)
            trace_path = out / "trace" / "layer_22.jsonl"
            trace_rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
            zero = next(row for row in trace_rows if row["kind"] == "zero")
            self.assertIsNone(zero["target"])
            self.assertFalse(zero["prefill_changed"])
            self.assertEqual((zero["changed_count"], zero["decode_calls_changed"], zero["first_decode_write_norm"]),
                             (0, 0, 0.0))
            expected_prompt = common86.token_inputs(tokenizer, "Who are you?", 1, "cpu")["input_ids"][0].tolist()
            expected_prompt_sha = hashlib.sha256(json.dumps(expected_prompt, separators=(",", ":")).encode("ascii")).hexdigest()
            self.assertEqual(trace_rows[0]["prompt_sha256"], expected_prompt_sha)
            audit("screen", str(out), "--layers", "22", "--pilot", "--width", "64",
                  "--prompt-texts", str(prompt_texts), "--neurons", "2", "--random", "2", "--tokens", "48")

            # A missing sidecar row is restored from the answer and trace stored atomically in the screen output.
            trace_path.write_text("\n".join(trace_path.read_text().splitlines()[:-1]) + "\n", encoding="utf-8")
            run_screen()
            self.assertEqual(len(trace_path.read_text().splitlines()), len(answer_rows) + 1,
                             "screen resume must restore the canary and all answer traces")
            audit("screen", str(out), "--layers", "22", "--pilot", "--width", "64",
                  "--prompt-texts", str(prompt_texts), "--neurons", "2", "--random", "2", "--tokens", "48")

            screen_key = {"rows": {"pick-one": {
                "layer": 22, "source": "real", "item_index": 1,
                "triage_category": "changed", "miss_rate_sample": False,
            }}}
            blind_key_path = temp / "blind-key.json"
            blind_key_path.write_text(json.dumps(screen_key), encoding="utf-8")
            blind_labels_path = temp / "blind-labels.jsonl"
            blind_labels_path.write_text(json.dumps({
                "id": "pick-one", "persona": "a quiet traveler", "kind": "human", "coherent": True,
            }) + "\n", encoding="utf-8")
            picked_path = temp / "picked.json"
            picked_path.write_text(json.dumps([{
                "layer": 22, "neuron": 1, "persona": "a quiet traveler",
            }]), encoding="utf-8")
            verify_out = temp / "verify"
            verify_args = SimpleNamespace(
                picked=str(picked_path), screen_dir=str(out), blind_key=str(blind_key_path),
                blind_labels=str(blind_labels_path), calib=str(calib), out=str(verify_out),
                device="cpu", max_neurons=None, test_picked_count=1, equivalences=None,
            )

            def run_verify():
                with mock.patch.object(common86, "MODEL_ID", "tiny-audit-test"), \
                     mock.patch.object(common86, "REVISION", "tiny-audit-revision"), \
                     mock.patch.object(common86, "load_model", return_value=(tokenizer, model)):
                    verify86.run(verify_args)

            run_verify()
            verify_trace = verify_out / "trace" / "L22_N1.jsonl"
            self.assertEqual(len(verify_trace.read_text().splitlines()), verify86.ANSWERS_PER_NEURON)
            audit("verify", str(verify_out), "--width", "64", "--prompt-texts", str(prompt_texts), "--picked", "1")
            verify_trace.write_text("\n".join(verify_trace.read_text().splitlines()[:-1]) + "\n", encoding="utf-8")
            run_verify()
            self.assertEqual(len(verify_trace.read_text().splitlines()), verify86.ANSWERS_PER_NEURON,
                             "verification resume must restore every answer trace")
            audit("verify", str(verify_out), "--width", "64", "--prompt-texts", str(prompt_texts), "--picked", "1")


class TestAmendment2Pins(unittest.TestCase):
    """Claude, 2026-10-28: pins the registered kappa gate and both Amendment 2 seeds (two mutations survived fold 2)."""

    def test_kappa_gate_boundary(self):
        self.assertEqual((common86.KAPPA_THRESHOLD, common86.MISS_SAMPLE_SEED, common86.SECOND_READER_SEED),
                         (0.6, 8686, 8687))
        self.assertTrue(common86.kappa_reads(0.6))
        self.assertFalse(common86.kappa_reads(0.5999))
        self.assertFalse(common86.kappa_reads(0.5))
        self.assertFalse(common86.kappa_reads(None))
        # 20 double-read items, ten coherent personas per reader: 10 agreements -> kappa 0.0,
        # 14 -> 0.4, 16 -> 0.6 (observed agreement minus 0.5, over 0.5).
        ids = [str(i) for i in range(20)]
        def labels(pos):
            return {i: {"coherent": True, "kind": "human" if int(i) in pos else "still Qwen or an assistant"} for i in ids}
        a = labels(set(range(10)))
        self.assertAlmostEqual(blind86.cohen_kappa(a, labels(set(range(5)) | set(range(10, 15))), ids), 0.0)
        k = blind86.cohen_kappa(a, labels(set(range(8)) | {10, 11}), ids)
        self.assertAlmostEqual(k, 0.6)
        self.assertTrue(common86.kappa_reads(k))
        k5 = blind86.cohen_kappa(a, labels(set(range(7)) | {10, 11, 12}), ids)
        self.assertAlmostEqual(k5, 0.4)
        self.assertFalse(common86.kappa_reads(k5))

    def test_miss_sample_follows_the_registered_seed(self):
        import random
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            screen = temp / "screen"
            screen.mkdir()
            code = common86.code_sha256()
            for layer in range(10, 27):
                rows = [{"layer": layer, "kind": "random_direction", "id": k,
                         "answer": f"I am an AI assistant, version {k}."} for k in range(30)]
                rows.append({"layer": layer, "kind": "neuron", "id": 1, "answer": "A quiet traveler watches the sea."})
                (screen / f"layer_{layer}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
                config = {"code_sha256": code, "layer": layer, "median": 24.8, "dose": 0.8,
                          "actual_dtype": "torch.bfloat16", "versions": common86.dependency_versions()}
                (screen / f"layer_{layer}.meta.json").write_text(
                    json.dumps({"status": "complete", "config": config, "median": 24.8}), encoding="utf-8")
            triage_path = temp / "triage.json"
            triage_path.write_text(json.dumps(triage86.triage(screen)), encoding="utf-8")
            items_path, key_path = temp / "reader" / "items.jsonl", temp / "private" / "key.json"
            blind86.prepare(screen, triage_path, items_path, key_path)
            key = json.loads(key_path.read_text(encoding="utf-8"))
            got = sorted((v["layer"], v["item_index"]) for v in key["rows"].values()
                         if v["miss_rate_sample"] and v["source"] == "random" and v["triage_category"] == "kept")
            # The registered rule, restated independently: one Random(8686); shuffle each pool in the order
            # real-kept, real-broken, random-kept, random-broken (screen order), keep the first 200 of each.
            pools = {(s, t): [] for s in ("real", "random") for t in ("kept", "broken")}
            for row in triage86.read_screen(screen):
                cat = triage86.triage_answer(row.get("answer", ""))
                if cat != "changed":
                    pools[("real" if row["kind"] == "neuron" else "random", cat)].append(row)
            chooser = random.Random(8686)
            expected = None
            for s in ("real", "random"):
                for t in ("kept", "broken"):
                    chooser.shuffle(pools[(s, t)])
                    if (s, t) == ("random", "kept"):
                        expected = sorted((int(r["layer"]), int(r["id"])) for r in pools[(s, t)][:200])
            self.assertEqual(len(got), 200)
            self.assertEqual(got, expected)
            stable = sorted(key["rows"], key=lambda i: (int(key["rows"][i]["layer"]), key["rows"][i]["source"],
                                                         int(key["rows"][i]["item_index"]), key["rows"][i]["triage_category"]))
            n2 = max(1, int(round(len(stable) * 0.10)))
            self.assertEqual(sorted(key["second_reader_ids"]), sorted(random.Random(8687).sample(stable, n2)))


class TestScoreCliKappaGate(unittest.TestCase):
    """Claude, 2026-10-28: the score command must turn 86.2-86.6 NOT READ when second-reader kappa is below 0.6."""

    def test_low_kappa_blocks_verdicts_in_the_score_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            screen, triage_path, items_path, key_path, _count = TestBlind._fixture(self, temp)
            key = json.loads(key_path.read_text(encoding="utf-8"))
            items = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines()]
            first = [{"id": r["id"], "persona": "a traveler", "kind": "human", "coherent": True} for r in items]
            agree = [row for row in first if row["id"] in set(key["second_reader_ids"])]
            disagree = [dict(row, kind="still Qwen or an assistant") for row in agree]
            labels_path = temp / "labels.jsonl"
            labels_path.write_text("".join(json.dumps(r) + "\n" for r in first), encoding="utf-8")
            holds = {"verdicts": {name: {"status": "HOLDS", "detail": "stub"} for name in ("86.2", "86.3", "86.4", "86.6")}}
            outcomes = {}
            for name, second in (("agree", agree), ("disagree", disagree)):
                labels2_path = temp / f"labels2-{name}.jsonl"
                labels2_path.write_text("".join(json.dumps(r) + "\n" for r in second), encoding="utf-8")
                out = temp / f"score-{name}.json"
                with mock.patch.object(blind86, "_assert_screen_denominators", return_value=None), \
                        mock.patch.object(blind86, "verdicts_from_counts", return_value=json.loads(json.dumps(holds))):
                    blind86.main(["score", "--screen-dir", str(screen), "--items", str(items_path), "--key", str(key_path),
                                  "--labels", str(labels_path), "--labels2", str(labels2_path), "--out", str(out)])
                outcomes[name] = json.loads(out.read_text(encoding="utf-8"))
            text = {k: json.dumps(v) for k, v in outcomes.items()}
            for verdict in ("86.2", "86.3", "86.4", "86.6"):
                self.assertIn(f'"{verdict}": {{"status": "NOT READ"', text["disagree"])
                self.assertIn(f'"{verdict}": {{"status": "HOLDS"', text["agree"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
