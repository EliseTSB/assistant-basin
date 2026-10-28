#!/usr/bin/env python3
"""CPU-only, network-free checks for Part 85's run, readers, and scoring rules."""
from __future__ import annotations

import hashlib
from contextlib import ExitStack
import json
import math
import os
import shutil
import sys
import subprocess
import tempfile
import types
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from safetensors.numpy import save_file
from safetensors.torch import save_file as save_torch_file
from transformers import Olmo2Config, Olmo2ForCausalLM

import part85
import score85
import step0_weights
import step1_acts
import handoff85
from checkpoints import BY_KEY, CHECKPOINTS, resolve_checkpoint
from step0_weights import SafetensorsRangeReader, SnapshotTensorReader, decode_tensor


TEST_COMMIT = "a" * 40


def tiny_model():
    torch.manual_seed(85)
    config = Olmo2Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=True,
    )
    return Olmo2ForCausalLM(config).cpu().eval()


def tiny_run_model():
    torch.manual_seed(85)
    config = Olmo2Config(
        vocab_size=64, hidden_size=16, intermediate_size=7885, num_hidden_layers=11,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=True,
    )
    return Olmo2ForCausalLM(config).cpu().eval()


class TinyTokenizer:
    eos_token_id = 2
    unk_token_id = 0
    pad_token_id = 0
    chat_template = None

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = [4, 5, 6]
        return {"input_ids": torch.tensor([ids], dtype=torch.long) if return_tensors else ids}

    def convert_tokens_to_ids(self, _token):
        return self.unk_token_id

    def decode(self, _ids, skip_special_tokens=True):
        return "I come from Earth and I am a language model."

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True,
                            return_tensors="pt", return_dict=True):
        self.last_chat_messages = messages
        return {"input_ids": torch.tensor([[4, 5, 6]], dtype=torch.long)}


class FormsTokenizer:
    def __init__(self, duplicate=False):
        self.duplicate = duplicate
        self.all_special_ids = [0]
        self.forms = {form: index + 1 for index, form in enumerate(step0_weights.OFFWORLD_FORMS)}

    def __call__(self, text, add_special_tokens=False):
        token_id = self.forms[text]
        if self.duplicate and text == step0_weights.OFFWORLD_FORMS[-1]:
            token_id = 1
        return {"input_ids": [token_id]}

    def convert_ids_to_tokens(self, token_id):
        if token_id in self.forms.values():
            return next(form.strip() for form, idx in self.forms.items() if idx == token_id)
        return "ordinary"

    def __len__(self):
        return 256


class ArrayReader:
    def __init__(self, tensors):
        self.tensors = tensors
        self._weight_map = {name: name for name in tensors}

    def info(self, name):
        array = self.tensors[name]
        return {"shape": list(array.shape), "dtype": "F32", "filename": name}

    def read_tensor(self, name):
        return self.tensors[name]

    def read_row(self, name, row):
        return self.tensors[name][row]

    def read_column(self, name, column):
        return self.tensors[name][:, column]

    def read_rows(self, name, rows):
        return self.tensors[name][list(rows)]


def make_counts(own_start=8, absolute_start=11, own_break=None, invalid_gate=None):
    counts = {}
    order = [cp.key for cp in CHECKPOINTS]
    for i, key in enumerate(order):
        control = {"n": 48, "OFFWORLD": 0, "EARTH": 0, "NEITHER": 48, "OK": 48, "BROKEN": 0}
        baseline = dict(control)
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


def resolve_fake(key):
    return {**asdict(BY_KEY[key]), "commit": TEST_COMMIT}


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
        self.assertEqual(part85.expected_answer_counts(),
                         {"pilot": 96, "main_plain": 2304, "main_chat": 576, "main_total": 2880})
        cells = part85.expected_cell_specs("pilot")
        self.assertEqual(len(cells), 4)
        self.assertEqual(sum(len(part85.QUESTIONS) * x["samples_per_question"] for x in cells), 96)
        main = part85.expected_cell_specs("main")
        self.assertEqual(sum(1 for x in main if x["frame"] == "plain"), 48)
        self.assertEqual(sum(1 for x in main if x["frame"] == "chat"), 12)

    def test_cell_values_main_scope_eos_and_template(self):
        step1 = {"neutral_activation_maxima": {"6978": 6.5, "7884": 2.0}}
        self.assertEqual(part85._cell_value("unclamped", step1), (None, None))
        self.assertEqual(part85._cell_value("own_max", step1), (6978, 9.75))
        self.assertEqual(part85._cell_value("absolute", step1), (6978, 9.75))
        self.assertEqual(part85._cell_value("control", step1), (7884, 3.0))
        with self.assertRaisesRegex(ValueError, "missing target unit"):
            part85._cell_value("own_max", {"neutral_activation_maxima": {"7884": 2.0}})
        with tempfile.TemporaryDirectory() as temp:
            score_path = Path(temp) / "pilot.json"
            score_path.write_text(json.dumps({"pilot_gate": {"action": "plain_frame_pass",
                                                               "plain_template_sha256": part85._template_sha256(part85.PLAIN_TEMPLATE)}}))
            args = type("Args", (), {"pilot_score": str(score_path), "plain_template_file": None})()
            self.assertEqual(part85._main_scope(args), ("all", ("plain", "chat")))
            score_path.write_text(json.dumps({"pilot_gate": {"action": "chat_only_posttrained"}}))
            self.assertEqual(part85._main_scope(args), ("posttrained", ("chat",)))
            score_path.write_text(json.dumps({"pilot_gate": {"action": "pilot_incomplete"}}))
            with self.assertRaisesRegex(RuntimeError, "does not authorize"):
                part85._main_scope(args)
        tokenizer = TinyTokenizer()
        self.assertEqual(part85._eos_ids(tokenizer, tiny_model()), [2])

    def test_bf16_decode_and_safetensors_rows(self):
        raw = np.array([0x3F80, 0xC000], dtype="<u2").tobytes()
        self.assertTrue(np.array_equal(decode_tensor(raw, "BF16", (2,)), np.array([1.0, -2.0])))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.json").write_text(json.dumps({"_commit_hash": TEST_COMMIT}))
            matrix = np.arange(30, dtype=np.float32).reshape(6, 5)
            filename = root / "model.safetensors"
            name = "model.layers.10.mlp.down_proj.weight"
            save_file({name: matrix}, str(filename))
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": filename.stat().st_size}, "weight_map": {name: filename.name}}),
                encoding="utf-8",
            )
            reader = SafetensorsRangeReader(root)
            self.assertTrue(np.array_equal(reader.read_column(name, 3), matrix[:, 3]))
            self.assertTrue(np.array_equal(reader.read_row(name, 4), matrix[4, :]))
            self.assertTrue(np.array_equal(reader.read_tensor(name), matrix))
            self.assertTrue(np.array_equal(reader.read_rows(name, [4, 1, 4]), matrix[[4, 1, 4]]))

    def test_snapshot_reader_supports_sharded_single_and_bin(self):
        tensor_name = "model.layers.10.mlp.down_proj.weight"
        values = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(torch.bfloat16)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shard_dir = root / "sharded"
            shard_dir.mkdir()
            save_torch_file({tensor_name: values}, str(shard_dir / "model-00001-of-00001.safetensors"))
            (shard_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
                tensor_name: "model-00001-of-00001.safetensors"}}))
            single_dir = root / "single"
            single_dir.mkdir()
            save_torch_file({tensor_name: values}, str(single_dir / "model.safetensors"))
            bin_dir = root / "bin"
            bin_dir.mkdir()
            torch.save({tensor_name: values}, bin_dir / "pytorch_model.bin")
            for directory, expected_format in ((shard_dir, "sharded_safetensors"),
                                               (single_dir, "single_safetensors"),
                                               (bin_dir, "pytorch_model.bin")):
                reader = SnapshotTensorReader(directory)
                self.assertEqual(reader.format, expected_format)
                self.assertEqual(reader.info(tensor_name)["dtype"], "BF16")
                self.assertTrue(np.array_equal(reader.read_tensor(tensor_name), values.float().numpy()))

    def test_post_norm_readout_z_score_and_unique_token_sets(self):
        down_name = "model.layers.10.mlp.down_proj.weight"
        gate_name = "model.layers.10.mlp.gate_proj.weight"
        up_name = "model.layers.10.mlp.up_proj.weight"
        post_name = "model.layers.10.post_feedforward_layernorm.weight"
        norm_name = "model.norm.weight"
        embedding_name = "lm_head.weight"
        down = np.zeros((4, 6979), dtype=np.float32)
        down[:, 6978] = np.array([1, 2, 3, 4], dtype=np.float32)
        embedding = np.arange(256 * 4, dtype=np.float32).reshape(256, 4) / 100
        reader = ArrayReader({down_name: down, gate_name: np.ones((6979, 4), np.float32),
                              up_name: np.ones((6979, 4), np.float32), post_name: np.full(4, 2.0, np.float32),
                              norm_name: np.ones(4, np.float32), embedding_name: embedding})
        tokenizer = FormsTokenizer()
        off_ids, random_sets, _meta = step0_weights.token_sets(
            tokenizer, count=7, seed=step0_weights.RANDOM_SEED)
        actual = step0_weights.analyze_reader(reader, tokenizer, random_set_count=7)
        direction = down[:, 6978] * 2
        logits = embedding.astype(np.float64) @ direction.astype(np.float64)
        expected_off = float(np.mean(logits[off_ids]))
        random_means = np.array([np.mean(logits[rows]) for rows in random_sets])
        expected_z = (expected_off - float(random_means.mean())) / float(random_means.std(ddof=1))
        self.assertAlmostEqual(actual["measures"]["direct_readout"]["z_score"], expected_z, places=12)
        self.assertEqual(actual["weight_shapes"]["post_feedforward_norm"], [4])
        self.assertEqual(len(off_ids), 16)
        with self.assertRaisesRegex(ValueError, "16 unique"):
            step0_weights.token_sets(FormsTokenizer(duplicate=True), count=1)

    def test_fisher_value_and_control_gate(self):
        expected = math.comb(48, 10) / math.comb(96, 10)
        self.assertAlmostEqual(score85.fisher_exact_greater(10, 48, 0, 48), expected, places=15)
        treated = {"n": 48, "OFFWORLD": 10}
        baseline = {"n": 48, "OFFWORLD": 0}
        control = {"n": 48, "OFFWORLD": 0}
        result = score85.shows_push(treated, baseline, control)
        self.assertTrue(result["shows_push"])
        self.assertAlmostEqual(result["p_one_sided"], expected, places=15)
        blocked = score85.shows_push(treated, baseline, {"n": 48, "OFFWORLD": 3})
        self.assertFalse(blocked["shows_push"])

    def test_decisions_check_correlations_and_one_sided_calibration(self):
        self.assertIsNone(score85.earliest_suffix_onset(
            ["a", "b", "c"], {"a": True, "b": True, "c": False},
            {"a": True, "b": True, "c": True}))
        counts, step0, step1 = make_counts(own_break="instruct")
        no_onset = score85.evaluate_decisions(counts, step0, step1)
        self.assertIsNone(no_onset["onset"]["own_max"])

        counts, step0, step1 = make_counts()
        step1["midtrain_51B"]["neutral_correlations_with_instruct"]["target"] = 0.79
        different = score85.evaluate_decisions(counts, step0, step1)
        self.assertFalse(different["same_unit_gates"]["midtrain_51B"]["same_unit"])
        self.assertIsNone(different["push_details"]["own_max"]["midtrain_51B"]["shows_push"])

        counts, step0, step1 = make_counts(own_start=8, absolute_start=12)
        calibrated = score85.evaluate_decisions(counts, step0, step1)
        self.assertEqual(calibrated["onset"]["own_max"], "midtrain_51B")
        self.assertIsNone(calibrated["onset"]["absolute"])
        self.assertTrue(calibrated["onset"]["calibration_disagreement"])
        self.assertIsNone(calibrated["onset"]["reported_onset"])

    def test_load_completed_refuses_changed_step1_clamp(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "answers.jsonl"
            row = {"checkpoint": "pretrain_4T", "frame": "plain", "cell": "own_max",
                   "frame_template_sha256": part85._template_sha256(part85.PLAIN_TEMPLATE),
                   "unit": 6978, "clamp_value": 9.75}
            path.write_text(json.dumps(row) + "\n")
            current = {"neutral_activation_maxima": {"6978": 6.5}}
            self.assertEqual(len(part85._load_completed(path, 1, "pretrain_4T", "plain", "own_max",
                                                         row["frame_template_sha256"], current)), 1)
            changed = {"neutral_activation_maxima": {"6978": 7.0}}
            with self.assertRaisesRegex(ValueError, "differs from current step1"):
                part85._load_completed(path, 1, "pretrain_4T", "plain", "own_max",
                                       row["frame_template_sha256"], changed)

    def test_step1_guard_hashes_only_maxima_token_digest_and_commit(self):
        step1 = {"checkpoint": {"commit": TEST_COMMIT},
                 "neutral_activation_maxima": {"6978": 6.5, "7884": 2.0},
                 "neutral_token_ids_sha256": "f" * 64,
                 "device": "cpu", "library_versions": {"torch": "a"}}
        original = part85._step1_guard_sha256(step1)
        metadata_change = {**step1, "device": "cuda", "library_versions": {"torch": "b"}}
        self.assertEqual(part85._step1_guard_sha256(metadata_change), original)
        for changed in (
            {**step1, "neutral_activation_maxima": {"6978": 7.0, "7884": 2.0}},
            {**step1, "neutral_token_ids_sha256": "e" * 64},
            {"checkpoint": {"commit": "b" * 40},
             "neutral_activation_maxima": step1["neutral_activation_maxima"],
             "neutral_token_ids_sha256": step1["neutral_token_ids_sha256"]},
        ):
            self.assertNotEqual(part85._step1_guard_sha256(changed), original)

    def test_main_scoring_requires_complete_data_unless_partial(self):
        record = {"mode": "main", "checkpoint": "pretrain_21B", "frame": "plain", "cell": "unclamped",
                  "question_index": 0, "sample": 0, "seed": 85000,
                  "question": part85.QUESTIONS[0], "answer": "Partial answer."}
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "all 12 step0 and step1 files"):
                score85._step_data_checked(Path(temp))
            key = score85.prepare([record], Path(temp) / "items.jsonl", Path(temp) / "key.json")
            label = {"id": next(iter(key["items"])), "origin": "NEITHER", "form": "OK"}
            with self.assertRaisesRegex(ValueError, "2,880 answers"):
                score85.score_payload(key, [label], [label], results_root=Path(temp) / "missing")
            partial = score85.score_payload(key, [label], [label], results_root=Path(temp) / "missing", partial=True)
            self.assertEqual(partial["scope"], "partial")
            self.assertTrue(any(value["verdict"] == "NOT READ" for value in partial["verdicts"].values()))

    def test_collect_answers_record_counts_and_reader_agreement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            answer_file = root / "answers.jsonl"
            rows = [{"mode": "pilot", "attempt": 1, "checkpoint": "pretrain_4T", "frame": "plain",
                     "cell": "unclamped", "question_index": 0, "sample": 0, "seed": 85000,
                     "question": part85.QUESTIONS[0], "answer": "An answer."},
                    {"mode": "pilot", "attempt": 2, "checkpoint": "instruct", "frame": "plain",
                     "cell": "own_max", "question_index": 0, "sample": 0, "seed": 85000,
                     "question": part85.QUESTIONS[0], "answer": "Another answer."},
                    {"mode": "main", "attempt": 1, "checkpoint": "instruct", "frame": "plain",
                     "cell": "own_max", "question_index": 0, "sample": 0, "seed": 85000,
                     "question": part85.QUESTIONS[0], "answer": "Main answer."}]
            answer_file.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(len(score85.collect_answers([answer_file], mode="pilot", attempt=1)), 1)
            self.assertEqual(len(score85.collect_answers([answer_file], mode="main")), 1)
        pilot_rows = []
        for spec in part85.expected_cell_specs("pilot"):
            for qi in range(len(part85.QUESTIONS)):
                for sample in range(spec["samples_per_question"]):
                    pilot_rows.append({"checkpoint": spec["checkpoint"], "frame": spec["frame"],
                                       "cell": spec["cell"], "question_index": qi, "sample": sample,
                                       "seed": 85000 + 100 * qi + sample,
                                       "pilot_gate_action": None})
        self.assertEqual(score85.validate_record_counts(pilot_rows, "pilot"), None)
        with self.assertRaisesRegex(ValueError, "counts do not match"):
            score85.validate_record_counts(pilot_rows[:-1], "pilot")
        chat_rows = []
        for spec in part85.expected_cell_specs("main"):
            if spec["frame"] != "chat":
                continue
            for qi in range(len(part85.QUESTIONS)):
                for sample in range(spec["samples_per_question"]):
                    chat_rows.append({"checkpoint": spec["checkpoint"], "frame": spec["frame"],
                                      "cell": spec["cell"], "question_index": qi, "sample": sample,
                                      "seed": 85000 + 100 * qi + sample,
                                      "pilot_gate_action": "chat_only_posttrained"})
        self.assertEqual(score85.validate_record_counts(chat_rows, "main"), None)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            score85.validate_record_counts([{**row, "pilot_gate_action": None} for row in chat_rows], "main")
        broken_seed = [dict(row) for row in pilot_rows]
        broken_seed[0]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "expected"):
            score85.validate_record_counts(broken_seed, "pilot")
        duplicate_pair = [dict(row) for row in pilot_rows]
        duplicate_pair[0]["sample"] = 1
        duplicate_pair[0]["seed"] = 85001
        with self.assertRaisesRegex(ValueError, "repeats question/sample pair"):
            score85.validate_record_counts(duplicate_pair, "pilot")

    def test_prepare_has_question_random_ids_digest_and_refuses_overwrite(self):
        records = [{"mode": "pilot", "attempt": 1, "checkpoint": "pretrain_4T", "frame": "plain",
                    "cell": "unclamped", "question_index": 0, "sample": 0, "seed": 85000,
                    "question": part85.QUESTIONS[0], "answer": "An answer."}]
        with tempfile.TemporaryDirectory() as temp:
            items = Path(temp) / "items.jsonl"
            key_path = Path(temp) / "key.json"
            key = score85.prepare(records, items, key_path)
            item_rows = [json.loads(line) for line in items.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(set(item_rows[0]), {"id", "question", "text"})
            self.assertEqual(key["items_sha256"], hashlib.sha256(items.read_bytes()).hexdigest())
            self.assertTrue(key["items"])
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                score85.prepare(records, items, key_path)
            items.write_text(items.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                score85._check_items_digest(key)

    def test_label_validation_and_reader_agreement(self):
        records = [{"checkpoint": "pretrain_4T", "frame": "plain", "cell": "unclamped",
                    "question_index": 0, "sample": i, "seed": 85000 + i,
                    "question": part85.QUESTIONS[0], "answer": f"Answer {i}"} for i in range(2)]
        with tempfile.TemporaryDirectory() as temp:
            key = score85.prepare(records, Path(temp) / "items.jsonl", Path(temp) / "key.json")
            ids = list(key["items"])
            labels = [{"id": item_id, "origin": "NEITHER", "form": "OK"} for item_id in ids]
            checked = score85.validate_labels(labels, key)
            self.assertEqual(len(checked), 2)
            self.assertEqual(score85.reader_agreement(checked, checked, key)["origin_nominal"], 1.0)
            pilot_labels = [dict(row, responsive=True) for row in labels]
            self.assertEqual(len(score85.validate_labels(pilot_labels, key, require_responsive=True)), 2)
            with self.assertRaisesRegex(ValueError, "responsive field"):
                score85.validate_labels(labels, key, require_responsive=True)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                score85.validate_labels(labels + [labels[0]], key)

    def test_step1_activation_collection_on_tiny_olmo2(self):
        model = tiny_run_model()
        tokenizer = TinyTokenizer()
        token_ids = np.array([[4, 5, 6]], dtype=np.int32)
        passages = [{"id": "pair-1", "offworld": "A distant world.", "ordinary": "An ordinary place."}]
        acts, off, ordinary = step1_acts.collect_acts(None, tokenizer, model, token_ids, passages, "cpu")
        self.assertEqual(acts.shape, (1, 3, 2))
        self.assertEqual(off.shape, (1,))
        self.assertEqual(ordinary.shape, (1,))
        self.assertEqual(step1_acts.WIKITEXT_REVISION, "f776294184f13b8ff2337b3841cf9269a6216d1e")

    def test_step1_run_records_bf16_dataset_revision_and_versions(self):
        import transformers

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.json").write_text(json.dumps({"_commit_hash": TEST_COMMIT}))
            passages_path = root / "passages.json"
            passages_path.write_text(json.dumps([{"id": str(i), "offworld": "Far away.",
                                                  "ordinary": "Nearby."} for i in range(24)]))
            ids = np.array([[4, 5, 6, 7]], dtype=np.int32)
            cp = resolve_fake("instruct")
            model = tiny_run_model().to(dtype=torch.bfloat16)
            with mock.patch.object(step1_acts, "resolve_checkpoint", return_value=cp), \
                    mock.patch.object(step1_acts, "load_or_create_token_ids", return_value=(ids, "fixed-token-sha")), \
                    mock.patch.object(step1_acts, "SEQUENCE_COUNT", 1), \
                    mock.patch.object(step1_acts, "SEQUENCE_LENGTH", 4), \
                    mock.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=TinyTokenizer()), \
                    mock.patch.object(transformers.AutoModelForCausalLM, "from_pretrained", return_value=model):
                step1_acts.run(type("Args", (), {"checkpoint": "instruct", "snapshot_dir": str(root),
                                                  "device": "cpu", "sequence_ids": str(root / "ids.json"),
                                                  "passages": str(passages_path), "reference_acts": None,
                                                  "results_root": str(root), "out": str(root / "step1.json")})())
            payload = json.loads((root / "step1.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["dtype"], "torch.bfloat16")
            self.assertEqual(payload["parameter_dtypes"], ["torch.bfloat16"])
            self.assertEqual(payload["dataset_revision"], step1_acts.WIKITEXT_REVISION)
            self.assertIn("torch", payload["library_versions"])

    def test_step0_and_step1_push_after_writing_outputs(self):
        import transformers
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / TEST_COMMIT
            snapshot.mkdir()
            (snapshot / "config.json").write_text(json.dumps({"_commit_hash": TEST_COMMIT}))
            with mock.patch.object(step0_weights, "resolve_checkpoint", return_value=resolve_fake("instruct")), \
                 mock.patch.object(step0_weights, "analyze_snapshot", return_value={"measures": {}}), \
                 mock.patch("subprocess.run") as step0_push:
                step0_weights.main(["--checkpoint", "instruct", "--snapshot-dir", str(snapshot),
                                    "--results-root", str(root), "--push-branch", "part85-run/pilot"])
            self.assertTrue((root / "instruct" / "step0.json").is_file())
            self.assertIn("handoff85.py", str(step0_push.call_args))

            passages_path = root / "passages.json"
            passages_path.write_text(json.dumps([{"id": str(i), "offworld": "Far away.",
                                                  "ordinary": "Nearby."} for i in range(24)]))
            model = tiny_run_model().to(dtype=torch.bfloat16)
            token_ids = np.array([[4, 5, 6, 7]], dtype=np.int32)
            with mock.patch.object(step1_acts, "resolve_checkpoint", return_value=resolve_fake("instruct")), \
                 mock.patch.object(step1_acts, "load_or_create_token_ids", return_value=(token_ids, "fixed-token-sha")), \
                 mock.patch.object(step1_acts, "collect_acts", return_value=(np.ones((1, 4, 2)), np.ones(24), np.zeros(24))), \
                 mock.patch.object(step1_acts, "SEQUENCE_COUNT", 1), \
                 mock.patch.object(step1_acts, "SEQUENCE_LENGTH", 4), \
                 mock.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=TinyTokenizer()), \
                 mock.patch.object(transformers.AutoModelForCausalLM, "from_pretrained", return_value=model), \
                 mock.patch("subprocess.run") as step1_push:
                step1_acts.main(["--checkpoint", "instruct", "--snapshot-dir", str(snapshot),
                                 "--results-root", str(root), "--sequence-ids", str(root / "neutral-token-ids.json"),
                                 "--passages", str(passages_path), "--push-branch", "part85-run/pilot"])
            self.assertTrue((root / "instruct" / "step1.json").is_file())
            self.assertIn("handoff85.py", str(step1_push.call_args))

    def _write_step1_fixture(self, root, key, maximum=6.5):
        cp_root = Path(root) / key
        cp_root.mkdir(parents=True, exist_ok=True)
        payload = {"checkpoint": {"key": key, "commit": TEST_COMMIT},
                   "neutral_activation_maxima": {"6978": maximum, "7884": 2.0},
                   "neutral_token_ids_sha256": "f" * 64,
                   "dataset_revision": step1_acts.WIKITEXT_REVISION}
        (cp_root / "step1.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def _run_patches(self, observers, tokenizer):
        import transformers
        original_hook = part85.BothPhaseClampHook

        def tracked_hook(down_proj, unit, value):
            hook = original_hook(down_proj, unit, value)
            observed_unit = unit if unit is not None else part85.TARGET_UNIT
            observation = {"unit": unit, "value": value, "prompt": [], "generated": []}
            observers.append(observation)
            pending_prompt = [True]
            original_begin = hook.begin_sequence

            def begin_sequence():
                pending_prompt[0] = True
                original_begin()

            def record(_module, args):
                x = args[0]
                phase = "prompt" if pending_prompt[0] else "generated"
                pending_prompt[0] = False
                observation[phase].extend(x[..., observed_unit].detach().float().cpu().reshape(-1).tolist())

            hook.begin_sequence = begin_sequence
            observer_handle = down_proj.register_forward_pre_hook(record)
            original_remove = hook.remove

            def remove():
                observer_handle.remove()
                original_remove()

            hook.remove = remove
            return hook

        return [mock.patch.object(transformers.AutoModelForCausalLM, "from_pretrained",
                                  side_effect=lambda *_a, **_kw: tiny_run_model()),
                mock.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=tokenizer),
                mock.patch.object(part85, "resolve_checkpoint", side_effect=resolve_fake),
                mock.patch.object(part85, "BothPhaseClampHook", side_effect=tracked_hook)]

    def _run_pilot_main(self, root, key, mode, gate_path=None, chat=False, push_branch=None, observers=None):
        observers = observers if observers is not None else []
        tokenizer = TinyTokenizer()
        if chat:
            tokenizer.chat_template = "stub-template"
        patches = self._run_patches(observers, tokenizer)
        if mode == "pilot":
            argv = ["--mode", "pilot", "--checkpoint", key, "--snapshot-dir",
                    str(Path(root) / TEST_COMMIT), "--results-root", str(root), "--device", "cpu"]
            samples_patch = mock.patch.object(part85, "PILOT_SAMPLES_PER_QUESTION", 2)
        else:
            argv = ["--mode", "main", "--checkpoint", key, "--pilot-score", str(gate_path),
                    "--snapshot-dir", str(Path(root) / TEST_COMMIT), "--results-root", str(root),
                    "--device", "cpu"]
            samples_patch = mock.patch.object(part85, "SAMPLES_PER_QUESTION", 1)
        if push_branch:
            argv.extend(["--push-branch", push_branch])
        with ExitStack() as stack:
            stack.enter_context(samples_patch)
            stack.enter_context(mock.patch.object(part85, "MAX_NEW_TOKENS", 2))
            for patcher in patches:
                stack.enter_context(patcher)
            part85.main(argv)

    def _assert_answer_files(self, root, key, mode_tag, frames, per_question, cells=None):
        # This expected-value table is computed in the test, never by a Part 85 helper.
        expected = {"unclamped": (None, None), "own_max": (6978, 9.75),
                    "absolute": (6978, 9.75), "control": (7884, 3.0)}
        template_sha = hashlib.sha256(b"Question: {q}\nAnswer:").hexdigest()
        answer_dir = Path(root) / key / "answers"
        record = json.loads((Path(root) / key / "run.json").read_text(encoding="utf-8"))
        for frame in frames:
            for cell in (cells or expected):
                unit, clamp_value = expected[cell]
                path = answer_dir / f"{mode_tag}-{frame}--{cell}.jsonl"
                self.assertTrue(path.is_file(), f"missing answers file {path}")
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
                expected_pairs = {(qi, sample) for qi in range(3) for sample in range(per_question)}
                pairs = {(row["question_index"], row["sample"]) for row in rows}
                self.assertEqual(len(rows), 3 * per_question, str(path))
                self.assertEqual(len(pairs), len(rows), f"duplicate answer pair in {path}")
                self.assertEqual(pairs, expected_pairs, str(path))
                for row in rows:
                    self.assertEqual(row["unit"], unit, str(path))
                    self.assertEqual(row["clamp_value"], clamp_value, str(path))
                    self.assertEqual(row["frame_template_sha256"], template_sha, str(path))
                    self.assertEqual(row["seed"], 85000 + 100 * row["question_index"] + row["sample"])
                    self.assertEqual(row["checkpoint"], key)
                    self.assertEqual(row["commit"], TEST_COMMIT)
                    self.assertEqual(row["frame"], frame)
                hook_key = f"{mode_tag}/{frame}/{cell}"
                counts = record["hook_fire_counts"][hook_key]
                self.assertGreater(counts["prompt_positions"], 0, hook_key)
                self.assertGreater(counts["generated_positions"], 0, hook_key)

    def _assert_observers(self, observers):
        for observation in observers:
            self.assertTrue(observation["prompt"], observation)
            self.assertTrue(observation["generated"], observation)
            observed = observation["prompt"] + observation["generated"]
            if observation["unit"] is not None:
                self.assertTrue(all(math.isclose(v, observation["value"], rel_tol=1e-6, abs_tol=1e-6)
                                    for v in observed), observation)
            else:
                self.assertFalse(all(math.isclose(v, 9.75, rel_tol=1e-6, abs_tol=1e-6)
                                     for v in observed), observation)

    def test_run_entrypoint_disk_contract_resume_chat_and_changed_step1(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "results"
            (root / TEST_COMMIT).mkdir(parents=True)
            for key in ("pretrain_4T", "instruct", "sft", "dpo"):
                self._write_step1_fixture(root, key)
            observers = []
            with mock.patch("subprocess.run") as pushed:
                for key in ("instruct", "pretrain_4T"):
                    observed = []
                    self._run_pilot_main(root, key, "pilot", push_branch="part85-run/pilot",
                                         observers=observed)
                    observers.extend(observed)
                    self._assert_answer_files(root, key, "pilot1", ("plain",), 2,
                                              cells=("unclamped", "own_max"))
                calls = [call for call in pushed.call_args_list if "handoff85.py" in str(call)]
                self.assertEqual(len(calls), 4, "push must run after each pilot answer cell")
            self._assert_observers(observers)

            # Resume must preserve the completed file and its hook counts while generating missing cells.
            cp_root = root / "pretrain_4T"
            run_path = cp_root / "run.json"
            record = json.loads(run_path.read_text(encoding="utf-8"))
            keep_key = "pilot1/plain/unclamped"
            kept = cp_root / "answers" / "pilot1-plain--unclamped.jsonl"
            kept_bytes = kept.read_bytes()
            kept_counts = dict(record["hook_fire_counts"][keep_key])
            cell = "own_max"
            (cp_root / "answers" / f"pilot1-plain--{cell}.jsonl").unlink()
            record["hook_fire_counts"].pop(f"pilot1/plain/{cell}", None)
            record["completed_cells"].pop(f"pilot1/plain/{cell}", None)
            run_path.write_text(json.dumps(record), encoding="utf-8")
            resumed = []
            self._run_pilot_main(root, "pretrain_4T", "pilot", observers=resumed)
            self.assertEqual(kept.read_bytes(), kept_bytes)
            resumed_record = json.loads(run_path.read_text(encoding="utf-8"))
            self.assertEqual(resumed_record["hook_fire_counts"][keep_key], kept_counts)
            self._assert_answer_files(root, "pretrain_4T", "pilot1", ("plain",), 2,
                                      cells=("unclamped", "own_max"))
            self.assertFalse(any(item["unit"] is None for item in resumed),
                             "resume regenerated the already-complete unclamped cell")
            self.assertTrue(any(item["unit"] == 6978 and item["value"] == 9.75 for item in resumed),
                            "resume did not generate the missing own-max cell")
            self._assert_observers(resumed)

            gate = base / "plain-pass.json"
            gate.write_text(json.dumps({"pilot_gate": {
                "action": "plain_frame_pass",
                "plain_template_sha256": hashlib.sha256(b"Question: {q}\nAnswer:").hexdigest()}}))
            pretrain_observers = []
            self._run_pilot_main(root, "pretrain_4T", "main", gate, observers=pretrain_observers)
            self._assert_answer_files(root, "pretrain_4T", "main", ("plain",), 1)
            self._assert_observers(pretrain_observers)

            post_observers = []
            self._run_pilot_main(root, "instruct", "main", gate, chat=True, observers=post_observers)
            self._assert_answer_files(root, "instruct", "main", ("plain", "chat"), 1)
            self._assert_observers(post_observers)

            chat_root = base / "chat-results"
            (chat_root / TEST_COMMIT).mkdir(parents=True)
            for key in ("pretrain_4T", "sft", "dpo", "instruct"):
                self._write_step1_fixture(chat_root, key)
            chat_gate = base / "chat-only.json"
            chat_gate.write_text(json.dumps({"pilot_gate": {"action": "chat_only_posttrained"}}))
            with self.assertRaisesRegex(ValueError, "outside the main run scope"):
                self._run_pilot_main(chat_root, "pretrain_4T", "main", chat_gate, chat=True)
            for key in ("sft", "dpo", "instruct"):
                observed = []
                self._run_pilot_main(chat_root, key, "main", chat_gate, chat=True, observers=observed)
                self._assert_answer_files(chat_root, key, "main", ("chat",), 1)
                self._assert_observers(observed)
                filenames = [p.name for p in (chat_root / key / "answers").glob("main-*.jsonl")]
                self.assertTrue(filenames and all("-chat--" in name for name in filenames), filenames)

            # Remove completed files so this refusal specifically proves the changed-step guard.
            changed_root = base / "changed"
            shutil.copytree(root / "instruct", changed_root / "instruct")
            shutil.rmtree(changed_root / "instruct" / "answers")
            changed_snapshot = changed_root / TEST_COMMIT
            changed_snapshot.mkdir(parents=True)
            step1_path = changed_root / "instruct" / "step1.json"
            changed = json.loads(step1_path.read_text(encoding="utf-8"))
            changed["neutral_activation_maxima"]["6978"] = 7.0
            step1_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "step1 clamp inputs changed"):
                self._run_pilot_main(changed_root, "instruct", "pilot")
    def test_pilot_gate_amend_chat_only_and_quality_failed(self):
        counts, _step0, _step1 = make_counts()
        self.assertEqual(score85.pilot_gate(counts, responsive_ok_count=15, attempt=1)["action"],
                         "amend_frame_and_rerun_once")
        self.assertEqual(score85.pilot_gate(counts, responsive_ok_count=15, attempt=2)["action"],
                         "quality_failed_after_rerun")
        counts["pretrain_4T/plain/unclamped"]["OFFWORLD"] = 0
        counts["instruct/plain/own_max"]["OFFWORLD"] = 0
        self.assertEqual(score85.pilot_gate(counts, responsive_ok_count=16, attempt=1)["action"],
                         "chat_only_posttrained")
        counts["instruct/plain/own_max"]["OFFWORLD"] = 12
        self.assertEqual(score85.pilot_gate(counts, responsive_ok_count=16, attempt=1)["action"],
                         "plain_frame_pass")

    def test_analyze_snapshot_row_column_batches_and_create_neutral_sequences(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            down = np.zeros((4, 7885), dtype=np.float32)
            down[:, 6978] = np.arange(1, 5, dtype=np.float32)
            tensors = {
                "model.layers.10.mlp.down_proj.weight": down,
                "model.layers.10.mlp.gate_proj.weight": np.ones((7885, 4), dtype=np.float32),
                "model.layers.10.mlp.up_proj.weight": np.ones((7885, 4), dtype=np.float32),
                "model.layers.10.post_feedforward_layernorm.weight": np.ones(4, dtype=np.float32),
                "model.norm.weight": np.ones(4, dtype=np.float32),
                "model.embed_tokens.weight": np.ones((256, 4), dtype=np.float32),
            }
            save_file(tensors, str(snapshot / "model.safetensors"))
            reader = SnapshotTensorReader(snapshot)
            name = "model.layers.10.mlp.down_proj.weight"
            self.assertTrue(np.array_equal(reader.read_row(name, 2), down[2]))
            self.assertTrue(np.array_equal(reader.read_column(name, 6978), down[:, 6978]))
            self.assertTrue(np.array_equal(reader.read_rows(name, [3, 0, 3]), down[[3, 0, 3]]))
            analyzed = step0_weights.analyze_snapshot(snapshot, tokenizer=FormsTokenizer(), random_set_count=5)
            self.assertEqual(analyzed["format"], "single_safetensors")
            self.assertEqual(analyzed["stored_dtypes"]["down"], "F32")
            self.assertEqual(analyzed["weight_shapes"]["down_column"], [4])

            capture = {}
            class Dataset:
                def __iter__(self):
                    return iter([{"text": "first row"}, {"text": "second row"}])
            def load_dataset(*args, **kwargs):
                capture["args"] = args
                capture["kwargs"] = kwargs
                return Dataset()
            with mock.patch.dict(sys.modules, {"datasets": types.SimpleNamespace(load_dataset=load_dataset)}):
                ids = step1_acts.create_neutral_sequences(TinyTokenizer(), count=1, length=4)
            self.assertEqual(ids.tolist(), [[4, 5, 6, 2]])
            self.assertEqual(capture["kwargs"]["revision"], step1_acts.WIKITEXT_REVISION)

            bad_file = root / "ids.json"
            bad_ids = np.zeros((step1_acts.SEQUENCE_COUNT, step1_acts.SEQUENCE_LENGTH), dtype=np.int32)
            bad_file.write_text(json.dumps({"token_ids": bad_ids.tolist(), "sha256": "0" * 64}))
            with self.assertRaisesRegex(ValueError, "sha256"):
                step1_acts.load_or_create_token_ids(bad_file, TinyTokenizer(), "instruct")

    def test_chat_only_scope_marks_plain_predictions_not_applicable_and_checks_scoped_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result_root = root / "results"
            records = []
            for spec in part85.expected_cell_specs("main"):
                if spec["frame"] != "chat":
                    continue
                for qi, question in enumerate(part85.QUESTIONS):
                    for sample in range(16):
                        records.append({"mode": "main", "checkpoint": spec["checkpoint"],
                                        "frame": "chat", "cell": spec["cell"],
                                        "question_index": qi, "sample": sample,
                                        "seed": 85000 + 100 * qi + sample,
                                        "question": question, "answer": "Answer.",
                                        "pilot_gate_action": "chat_only_posttrained",
                                        "frame_template_sha256": hashlib.sha256(
                                            b"Question: {q}\nAnswer:").hexdigest()})
            items = root / "items.jsonl"
            key_path = root / "key.json"
            key = score85.prepare(records, items, key_path)
            labels = [{"id": item_id, "origin": "NEITHER", "form": "OK"}
                      for item_id in key["items"]]
            for key_name in ("sft", "dpo", "instruct"):
                cp = BY_KEY[key_name]
                identity = {"key": key_name, "commit": cp.commit_prefix + "0" * (40 - len(cp.commit_prefix))}
                cp_root = result_root / key_name
                cp_root.mkdir(parents=True)
                (cp_root / "step0.json").write_text(json.dumps({
                    "checkpoint": identity,
                    "measures": {"cosines_with_instruct": {"down_proj_column_6978": 0.95},
                                 "direct_readout": {"z_score": 0.0}}}))
                (cp_root / "step1.json").write_text(json.dumps({
                    "checkpoint": identity,
                    "neutral_correlations_with_instruct": {"target": 0.95},
                    "passages": {"offworld_mean": 0.0, "ordinary_mean": 0.0, "ordinary_sd": 1.0}}))
            payload = score85.score_payload(key, labels, labels, mode="main",
                                            results_root=result_root, items_path=items)
            self.assertEqual(payload["scope"], "chat_only_posttrained")
            for code in ("85.5", "85.6", "85.7", "85.8", "85.9"):
                self.assertEqual(payload["verdicts"][code]["verdict"], "NOT APPLICABLE")
            sft_step0 = result_root / "sft" / "step0.json"
            mismatched = json.loads(sft_step0.read_text(encoding="utf-8"))
            mismatched["checkpoint"]["key"] = "dpo"
            sft_step0.write_text(json.dumps(mismatched))
            with self.assertRaisesRegex(ValueError, "identifies checkpoint 'dpo'"):
                score85.score_payload(key, labels, labels, mode="main",
                                      results_root=result_root, items_path=items)
            sft_step0.write_text(json.dumps({**mismatched, "checkpoint": {
                "key": "sft", "commit": BY_KEY["sft"].commit_prefix + "0" * (40 - len(BY_KEY["sft"].commit_prefix))},
                "measures": {"cosines_with_instruct": {"down_proj_column_6978": 0.95},
                             "direct_readout": {"z_score": 0.0}}}))
            (result_root / "dpo" / "step0.json").unlink()
            with self.assertRaisesRegex(ValueError, "in-scope post-trained"):
                score85.score_payload(key, labels, labels, mode="main",
                                      results_root=result_root, items_path=items)

    def test_score_refuses_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record = {"mode": "pilot", "attempt": 1, "checkpoint": "pretrain_4T",
                      "frame": "plain", "cell": "unclamped", "question_index": 0,
                      "sample": 0, "seed": 85000, "question": "Who made you?", "answer": "Answer.",
                      "frame_template_sha256": hashlib.sha256(b"Question: {q}\nAnswer:").hexdigest()}
            items, key_path = root / "items.jsonl", root / "key.json"
            key = score85.prepare([record], items, key_path)
            labels = root / "labels.jsonl"
            labels.write_text(json.dumps({"id": next(iter(key["items"])), "origin": "NEITHER",
                                          "form": "OK", "responsive": True}) + "\n")
            out = root / "score.json"
            argv = ["score", "--mode", "pilot", "--key", str(key_path), "--labels", str(labels),
                    "--items", str(items), "--out", str(out)]
            score85.main(argv)
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite score output"):
                score85.main(argv)
            score85.main(argv + ["--force"])

    def test_step1_reference_hash_refusal_precedes_model_loading(self):
        import transformers
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / TEST_COMMIT
            snapshot.mkdir()
            token_ids = np.zeros((step1_acts.SEQUENCE_COUNT, step1_acts.SEQUENCE_LENGTH), dtype=np.int32)
            digest = step1_acts.stable_ids_sha256(token_ids)
            ids_path = root / "neutral-token-ids.json"
            ids_path.write_text(json.dumps({"token_ids": token_ids.tolist(), "sha256": digest}))
            reference = root / "instruct-step1.json"
            reference.write_text(json.dumps({"checkpoint": {"key": "instruct"},
                                             "neutral_token_ids_sha256": "0" * 64,
                                             "neutral_activations": []}))
            args = type("Args", (), {"checkpoint": "pretrain_4T", "snapshot_dir": str(snapshot),
                                      "reference_acts": str(reference), "results_root": str(root),
                                      "sequence_ids": str(ids_path), "passages": str(root / "passages.json"),
                                      "device": "cpu", "out": None, "push_branch": None})()
            with mock.patch.object(step1_acts, "resolve_checkpoint", return_value=resolve_fake("pretrain_4T")), \
                 mock.patch.object(transformers.AutoModelForCausalLM, "from_pretrained",
                                   side_effect=AssertionError("model load happened before token-hash refusal")):
                with self.assertRaisesRegex(ValueError, "different neutral token ids"):
                    step1_acts.run(args)

    def test_snapshot_provenance_requires_the_pinned_commit(self):
        from checkpoints import validate_snapshot_commit
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plain = root / "plain"
            plain.mkdir()
            (plain / "config.json").write_text(json.dumps({"_commit_hash": TEST_COMMIT}))
            self.assertEqual(validate_snapshot_commit(plain, TEST_COMMIT), plain.resolve())
            wrong = root / "wrong"
            wrong.mkdir()
            (wrong / "config.json").write_text(json.dumps({"_commit_hash": "b" * 40}))
            with self.assertRaisesRegex(ValueError, "does not identify pinned commit"):
                validate_snapshot_commit(wrong, TEST_COMMIT)

    def _git_cmd(self, repo, *args, check=True):
        result = subprocess.run(["git", *map(str, args)], cwd=repo, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if check and result.returncode:
            self.fail(f"git {args} failed: {result.stderr}")
        return result

    def _set_git_identity(self, repo):
        self._git_cmd(repo, "config", "user.email", "part85-test@example.invalid")
        self._git_cmd(repo, "config", "user.name", "Part 85 Test")

    def _handoff_command(self, repo, *args):
        return subprocess.run([sys.executable, str(Path(handoff85.__file__)), *map(str, args)],
                              cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_handoff_bare_remote_push_rebase_restore_preflight_and_assemble(self):
        with tempfile.TemporaryDirectory() as temp:
            top = Path(temp)
            bare, seed = top / "remote.git", top / "seed"
            bare.mkdir()
            seed.mkdir()
            self._git_cmd(top, "init", "--bare", str(bare))
            self._git_cmd(seed, "init")
            self._set_git_identity(seed)
            (seed / "README").write_text("base\n")
            self._git_cmd(seed, "add", "README")
            self._git_cmd(seed, "commit", "-m", "base")
            self._git_cmd(seed, "branch", "-M", "main")
            self._git_cmd(seed, "remote", "add", "origin", str(bare))
            self._git_cmd(seed, "push", "-u", "origin", "main")
            subprocess.run(["git", "--git-dir", str(bare), "symbolic-ref", "HEAD", "refs/heads/main"],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            clone_a, clone_b = top / "clone-a", top / "clone-b"
            self._git_cmd(top, "clone", str(bare), str(clone_a))
            self._git_cmd(top, "clone", str(bare), str(clone_b))
            self._set_git_identity(clone_a)
            self._set_git_identity(clone_b)

            a_results = clone_a / "results" / "part85"
            instruct = a_results / "instruct"
            (instruct / "answers").mkdir(parents=True)
            (instruct / "step0.json").write_text('{"checkpoint":"instruct"}')
            (instruct / "step1.json").write_text('{"checkpoint":"instruct"}')
            (instruct / "answers" / "cell.jsonl").write_text('{"answer":"a"}\n')
            (a_results / "neutral-token-ids.json").write_text('{"sha256":"token"}')
            pushed = self._handoff_command(clone_a, "push", "--checkpoint", "instruct",
                                           "--branch", "part85-run/pilot", "--results-root", "results/part85")
            self.assertEqual(pushed.returncode, 0, pushed.stderr)

            # Clone B has the common base, so this remote update is rejected once and rebased.
            b_results = clone_b / "results" / "part85"
            pretrain = b_results / "pretrain_4T"
            pretrain.mkdir(parents=True)
            (pretrain / "step0.json").write_text('{"checkpoint":"pretrain_4T"}')
            rebased = self._handoff_command(clone_b, "push", "--checkpoint", "pretrain_4T",
                                            "--branch", "part85-run/pilot", "--results-root", "results/part85")
            self.assertEqual(rebased.returncode, 0, rebased.stderr)
            self.assertIn("pushed pretrain_4T results", rebased.stdout)

            restored = top / "restored"
            restored_run = self._handoff_command(clone_b, "restore", "--branch", "part85-run/pilot",
                                                 "--results-root", restored)
            self.assertEqual(restored_run.returncode, 0, restored_run.stderr)
            self.assertTrue((restored / "instruct" / "step0.json").is_file())
            self.assertTrue((restored / "pretrain_4T" / "step0.json").is_file())
            self.assertTrue((restored / "neutral-token-ids.json").is_file())
            clone_c = top / "clone-c"
            self._git_cmd(top, "clone", str(bare), str(clone_c))
            self._set_git_identity(clone_c)
            resumed_restore = self._handoff_command(clone_c, "restore", "--branch", "part85-run/pilot",
                                                    "--results-root", "results/part85")
            self.assertEqual(resumed_restore.returncode, 0, resumed_restore.stderr)
            (clone_c / "results" / "part85" / "pretrain_4T" / "step1.json").write_text(
                '{"checkpoint":"pretrain_4T","resumed":true}')
            resumed_push = self._handoff_command(clone_c, "push", "--checkpoint", "pretrain_4T",
                                                "--branch", "part85-run/pilot", "--results-root", "results/part85")
            self.assertEqual(resumed_push.returncode, 0, resumed_push.stderr)
            pilot_pass = self._handoff_command(clone_b, "preflight", "--checkpoint", "pretrain_4T",
                                               "--phase", "pilot", "--results-root", restored)
            self.assertEqual(pilot_pass.returncode, 0, pilot_pass.stderr)
            (restored / "neutral-token-ids.json").unlink()
            pilot_fail = self._handoff_command(clone_b, "preflight", "--checkpoint", "pretrain_4T",
                                               "--phase", "pilot", "--results-root", restored)
            self.assertNotEqual(pilot_fail.returncode, 0)
            self.assertIn("neutral-token-ids.json", pilot_fail.stderr)

            score = a_results / "pilot-score.json"
            score.write_text(json.dumps({"pilot_gate": {
                "action": "plain_frame_pass",
                "plain_template_sha256": hashlib.sha256(b"Question: {q}\nAnswer:").hexdigest()}}))
            (clone_a / "part85").mkdir(exist_ok=True)
            manifest_run = self._handoff_command(clone_a, "refs-manifest", "--results-root",
                                                 "results/part85", "--out", "part85/references.json")
            self.assertEqual(manifest_run.returncode, 0, manifest_run.stderr)
            main_pass = self._handoff_command(clone_a, "preflight", "--checkpoint", "sft",
                                              "--phase", "main", "--results-root", "results/part85")
            self.assertEqual(main_pass.returncode, 0, main_pass.stderr)
            instruct_step0 = a_results / "instruct" / "step0.json"
            saved_step0 = instruct_step0.read_bytes()
            instruct_step0.unlink()
            missing_reference = self._handoff_command(clone_a, "preflight", "--checkpoint", "sft",
                                                      "--phase", "main", "--results-root", "results/part85")
            self.assertNotEqual(missing_reference.returncode, 0)
            self.assertIn("missing", missing_reference.stderr)
            instruct_step0.write_bytes(saved_step0)
            score.write_text("{}")
            main_fail = self._handoff_command(clone_a, "preflight", "--checkpoint", "sft",
                                              "--phase", "main", "--results-root", "results/part85")
            self.assertNotEqual(main_fail.returncode, 0)
            self.assertIn("SHA-256 mismatch", main_fail.stderr)
            custom_template = b"Origin question: {q}\nReply:"
            (a_results / "pilot-frame-v2.txt").write_bytes(custom_template)
            score.write_text(json.dumps({"pilot_gate": {
                "action": "plain_frame_pass",
                "plain_template_sha256": hashlib.sha256(custom_template).hexdigest()}}))
            regenerated = self._handoff_command(clone_a, "refs-manifest", "--results-root",
                                                "results/part85", "--out", "part85/references.json")
            self.assertEqual(regenerated.returncode, 0, regenerated.stderr)
            custom_pass = self._handoff_command(clone_a, "preflight", "--checkpoint", "sft",
                                                "--phase", "main", "--results-root", "results/part85")
            self.assertEqual(custom_pass.returncode, 0, custom_pass.stderr)
            (a_results / "pilot-frame-v2.txt").unlink()
            custom_missing = self._handoff_command(clone_a, "preflight", "--checkpoint", "sft",
                                                   "--phase", "main", "--results-root", "results/part85")
            self.assertNotEqual(custom_missing.returncode, 0)
            self.assertIn("pilot-frame-v2.txt", custom_missing.stderr)
            (a_results / "pilot-frame-v2.txt").write_bytes(custom_template)

            assembled = top / "assembled"
            assemble_ok = self._handoff_command(clone_a, "assemble", "--branches", "part85-run/pilot",
                                                "--out", assembled)
            self.assertEqual(assemble_ok.returncode, 0, assemble_ok.stderr)
            self.assertTrue((assembled / "instruct" / "step0.json").is_file())
            self.assertTrue((assembled / "pretrain_4T" / "step0.json").is_file())

            # Same checkpoint path with different bytes across branches is also rejected.
            instruction_step0 = a_results / "instruct" / "step0.json"
            original_step0 = instruction_step0.read_bytes()
            instruction_step0.write_text('{"checkpoint":"instruct","branch":"other"}')
            self._git_cmd(clone_a, "add", "results/part85/instruct/step0.json")
            self._git_cmd(clone_a, "commit", "-m", "different Instruct artifact")
            self._git_cmd(clone_a, "push", "origin", "HEAD:refs/heads/part85-run/instruct")
            conflict = self._handoff_command(clone_a, "assemble", "--branches",
                                             "part85-run/pilot", "part85-run/instruct",
                                             "--out", top / "conflict")
            self.assertNotEqual(conflict.returncode, 0)
            self.assertIn("different bytes", conflict.stderr)
            instruction_step0.write_bytes(original_step0)

            # A branch named for Instruct that also contains SFT output must be refused.
            foreign = a_results / "sft"
            foreign.mkdir()
            (foreign / "intruder.json").write_text("wrong owner")
            self._git_cmd(clone_a, "add", "results/part85/sft")
            self._git_cmd(clone_a, "commit", "-m", "deliberate foreign result")
            self._git_cmd(clone_a, "push", "origin", "HEAD:refs/heads/part85-run/instruct")
            assemble_bad = self._handoff_command(clone_a, "assemble", "--branches",
                                                 "part85-run/instruct", "--out", top / "rejected")
            self.assertNotEqual(assemble_bad.returncode, 0)
            self.assertIn("outside its ownership", assemble_bad.stderr)

            absent = self._handoff_command(clone_a, "restore", "--branch", "part85-run/not-created",
                                           "--results-root", top / "empty")
            self.assertEqual(absent.returncode, 0, absent.stderr)
            self.assertIn("does not exist", absent.stdout)

    def test_handoff_refuses_unrelated_staged_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            top = Path(temp)
            bare, repo = top / "remote.git", top / "repo"
            bare.mkdir()
            repo.mkdir()
            self._git_cmd(top, "init", "--bare", str(bare))
            self._git_cmd(repo, "init")
            self._set_git_identity(repo)
            (repo / "README").write_text("base")
            self._git_cmd(repo, "add", "README")
            self._git_cmd(repo, "commit", "-m", "base")
            self._git_cmd(repo, "remote", "add", "origin", str(bare))
            self._git_cmd(repo, "push", "origin", "HEAD:refs/heads/main")
            target = repo / "results" / "part85" / "instruct"
            target.mkdir(parents=True)
            (target / "step0.json").write_text("{}")
            (repo / "README").write_text("unrelated staged")
            self._git_cmd(repo, "add", "README")
            result = self._handoff_command(repo, "push", "--checkpoint", "instruct",
                                           "--branch", "part85-run/pilot", "--results-root", "results/part85")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unrelated paths are already staged", result.stderr)

    def test_handoff_rebases_concurrent_cell_pushes_and_merges_run_record(self):
        with tempfile.TemporaryDirectory() as temp:
            top = Path(temp)
            bare, seed = top / "remote.git", top / "seed"
            bare.mkdir()
            seed.mkdir()
            self._git_cmd(top, "init", "--bare", str(bare))
            self._git_cmd(seed, "init")
            self._set_git_identity(seed)
            (seed / "README").write_text("base")
            self._git_cmd(seed, "add", "README")
            self._git_cmd(seed, "commit", "-m", "base")
            self._git_cmd(seed, "branch", "-M", "main")
            self._git_cmd(seed, "remote", "add", "origin", str(bare))
            self._git_cmd(seed, "push", "-u", "origin", "main")
            subprocess.run(["git", "--git-dir", str(bare), "symbolic-ref", "HEAD", "refs/heads/main"],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            first, stale = top / "first", top / "stale"
            self._git_cmd(top, "clone", str(bare), str(first))
            self._git_cmd(top, "clone", str(bare), str(stale))
            self._set_git_identity(first)
            self._set_git_identity(stale)
            results = first / "results" / "part85" / "sft"
            (results / "answers").mkdir(parents=True)
            (results / "step0.json").write_text("step0")
            (results / "answers" / "cell0.jsonl").write_text("cell0")
            (results / "run.json").write_text(json.dumps({
                "hook_fire_counts": {"cell0": {"prompt_positions": 1}},
                "completed_cells": {"cell0": {"answer_count": 1}}}))
            initial = self._handoff_command(first, "push", "--checkpoint", "sft",
                                            "--branch", "part85-run/sft", "--results-root", "results/part85")
            self.assertEqual(initial.returncode, 0, initial.stderr)

            restored = self._handoff_command(stale, "restore", "--branch", "part85-run/sft",
                                              "--results-root", "results/part85")
            self.assertEqual(restored.returncode, 0, restored.stderr)
            # First clone pushes cell1 after the stale clone has restored cell0.
            (results / "answers" / "cell1.jsonl").write_text("cell1")
            current = json.loads((results / "run.json").read_text())
            current["hook_fire_counts"]["cell1"] = {"prompt_positions": 1}
            current["completed_cells"]["cell1"] = {"answer_count": 1}
            (results / "run.json").write_text(json.dumps(current))
            update = self._handoff_command(first, "push", "--checkpoint", "sft",
                                           "--branch", "part85-run/sft", "--results-root", "results/part85")
            self.assertEqual(update.returncode, 0, update.stderr)

            # Stale clone creates a disjoint cell and pushes an older run.json based on cell0.
            stale_results = stale / "results" / "part85" / "sft"
            (stale_results / "answers" / "cell2.jsonl").write_text("cell2")
            stale_record = json.loads((stale_results / "run.json").read_text())
            stale_record["hook_fire_counts"]["cell2"] = {"prompt_positions": 1}
            stale_record["completed_cells"]["cell2"] = {"answer_count": 1}
            (stale_results / "run.json").write_text(json.dumps(stale_record))
            rejected_then_rebased = self._handoff_command(stale, "push", "--checkpoint", "sft",
                                                          "--branch", "part85-run/sft", "--results-root", "results/part85")
            self.assertEqual(rejected_then_rebased.returncode, 0, rejected_then_rebased.stderr)
            merged = json.loads((stale_results / "run.json").read_text())
            self.assertEqual(set(merged["hook_fire_counts"]), {"cell0", "cell1", "cell2"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
