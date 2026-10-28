#!/usr/bin/env python3
"""CPU-only, network-free checks for Part 85's run, readers, and scoring rules."""
from __future__ import annotations

import hashlib
from contextlib import ExitStack
import contextlib
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


def tiny_run_model(vocab_size=64):
    torch.manual_seed(85)
    config = Olmo2Config(
        vocab_size=vocab_size, hidden_size=16, intermediate_size=7885, num_hidden_layers=11,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=True,
    )
    return Olmo2ForCausalLM(config).cpu().eval()


class TinyAuditHead(torch.nn.Module):
    """Make the tiny OLMo stop after one generated token while still running its real forward path twice."""

    def __init__(self, vocab_size, first_token=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.first_token = first_token
        self.calls = 0

    def forward(self, hidden_states):
        logits = torch.full((*hidden_states.shape[:-1], self.vocab_size), -10000.0,
                            dtype=hidden_states.dtype, device=hidden_states.device)
        # first_token 2 (EOS) makes every answer an immediate EOS: no decode forward ever runs (round-4 #6).
        token = self.first_token if self.first_token == 2 or self.calls % 2 == 0 else 2
        self.calls += 1
        logits[..., token] = 10000.0
        return logits


def tiny_audit_model(vocab_size=64, first_token=3):
    model = tiny_run_model(vocab_size)
    model.lm_head = TinyAuditHead(model.config.vocab_size, first_token)
    model.config._commit_hash = TEST_COMMIT
    return model


CHAR_SPECIALS = ["<pad>", "<bos>", "<eos>"]
CHAR_SET = sorted(set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ?:.,'\n<>|/_-=!"))
CHAT_TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n{% endfor %}"
                 "{% if add_generation_prompt %}<|assistant|>\n{% endif %}")


def char_tokenizer(directory):
    """A real Hugging Face fast tokenizer, one id per character, with an OLMo-like chat template, saved to and
    reloaded from disk: the run and the audit (a separate process) both load it from the directory."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    vocab = {token: index for index, token in enumerate(CHAR_SPECIALS + CHAR_SET)}
    core = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
    core.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    core.decoder = decoders.Fuse()
    fast = PreTrainedTokenizerFast(tokenizer_object=core, pad_token="<pad>", bos_token="<bos>",
                                   eos_token="<eos>", unk_token="<pad>", clean_up_tokenization_spaces=False)
    fast.chat_template = CHAT_TEMPLATE
    Path(directory).mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(str(directory))
    return AutoTokenizer.from_pretrained(str(directory))


class TinyTokenizer:
    eos_token_id = 2
    unk_token_id = 0
    pad_token_id = 0
    chat_template = None

    def __init__(self):
        self.plain_prompts = []
        self.chat_calls = 0

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        self.plain_prompts.append(text)
        ids = [4, 5, 6]
        return {"input_ids": torch.tensor([ids], dtype=torch.long) if return_tensors else ids}

    def convert_tokens_to_ids(self, _token):
        return self.unk_token_id

    def decode(self, _ids, skip_special_tokens=True):
        return "I come from Earth and I am a language model."

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True,
                            return_tensors="pt", return_dict=True):
        self.chat_calls += 1
        self.last_chat_messages = messages
        return {"input_ids": torch.tensor([[4, 5, 6]], dtype=torch.long)}


class TraceTokenizer(TinyTokenizer):
    """Encode prompt text losslessly so the trace can be audited against generate's input IDs."""

    def __init__(self):
        super().__init__()
        self._char_to_id = {}
        self._id_to_char = {}

    def _encode(self, text):
        ids = []
        for character in text:
            if character not in self._char_to_id:
                token_id = len(self._char_to_id) + 3
                if token_id >= 64:
                    raise ValueError("trace fixture prompt uses more than 61 distinct characters")
                self._char_to_id[character] = token_id
                self._id_to_char[token_id] = character
            ids.append(self._char_to_id[character])
        return ids

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        self.plain_prompts.append(text)
        ids = self._encode(text)
        return {"input_ids": torch.tensor([ids], dtype=torch.long) if return_tensors else ids}

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        decoded = []
        for value in ids:
            value = int(value)
            if value in self._id_to_char:
                decoded.append(self._id_to_char[value])
        return "".join(decoded)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True,
                            return_tensors="pt", return_dict=True):
        self.chat_calls += 1
        self.last_chat_messages = messages
        text = f"<user>\n{messages[0]['content']}\n<assistant>"
        return {"input_ids": torch.tensor([self._encode(text)], dtype=torch.long)}


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
        step1 = {"neutral_activation_maxima": {"6978": 5.0, "7884": 2.0}}
        self.assertEqual(part85._cell_value("unclamped", step1), (None, None))
        self.assertEqual(part85._cell_value("own_max", step1), (6978, 7.5))
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
            score_path.write_text(json.dumps({"pilot_gate": {"action": "chat_only_posttrained",
                                                                "plain_template_sha256": part85._template_sha256(part85.PLAIN_TEMPLATE)}}))
            self.assertEqual(part85._main_scope(args), ("posttrained", ("chat",)))
            score_path.write_text(json.dumps({"pilot_gate": {"action": "pilot_incomplete",
                "plain_template_sha256": part85._template_sha256(part85.PLAIN_TEMPLATE)}}))
            with self.assertRaisesRegex(RuntimeError, "does not authorize"):
                part85._main_scope(args)
            score_path.write_text(json.dumps({"pilot_gate": {"action": "plain_frame_pass",
                "plain_template_sha256": "0" * 64}}))
            with self.assertRaisesRegex(ValueError, "does not match the pilot template"):
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
                    mock.patch.object(step1_acts, "EXPECTED_NEUTRAL_TOKEN_SHA256", "fixed-token-sha"), \
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
            step0_path = root / "pilot-a1" / "instruct" / "step0.json"
            step0_pushes = []
            def step0_subprocess(argv, **kwargs):
                if "handoff85.py" in str(argv):
                    self.assertTrue(step0_path.is_file(), "step 0 must finish its atomic write before push")
                    step0_pushes.append(argv)
                return subprocess.CompletedProcess(argv, 0, stdout=TEST_COMMIT + "\n", stderr="")
            with mock.patch.object(step0_weights, "resolve_checkpoint", return_value=resolve_fake("instruct")), \
                 mock.patch.object(step0_weights, "analyze_snapshot", return_value={"measures": {}}), \
                 mock.patch.object(step0_weights, "code_commit", return_value="fixture-code"), \
                 mock.patch("subprocess.run", side_effect=step0_subprocess):
                step0_weights.main(["--checkpoint", "instruct", "--snapshot-dir", str(snapshot),
                                    "--results-root", str(root), "--push-branch", "part85-run/pilot-a1"])
            self.assertTrue(step0_path.is_file())
            self.assertEqual(len(step0_pushes), 1)
            self.assertEqual(json.loads(step0_path.read_text())["code_commit"], "fixture-code")

            passages_path = root / "passages.json"
            passages_path.write_text(json.dumps([{"id": str(i), "offworld": "Far away.",
                                                  "ordinary": "Nearby."} for i in range(24)]))
            model = tiny_run_model().to(dtype=torch.bfloat16)
            token_ids = np.array([[4, 5, 6, 7]], dtype=np.int32)
            step1_path = root / "pilot-a1" / "instruct" / "step1.json"
            step1_pushes = []
            def step1_subprocess(argv, **kwargs):
                if "handoff85.py" in str(argv):
                    self.assertTrue(step1_path.is_file(), "step 1 must finish its atomic write before push")
                    step1_pushes.append(argv)
                return subprocess.CompletedProcess(argv, 0, stdout=TEST_COMMIT + "\n", stderr="")
            with mock.patch.object(step1_acts, "resolve_checkpoint", return_value=resolve_fake("instruct")), \
                 mock.patch.object(step1_acts, "load_or_create_token_ids", return_value=(token_ids, "fixed-token-sha")), \
                 mock.patch.object(step1_acts, "collect_acts", return_value=(np.ones((1, 4, 2)), np.ones(24), np.zeros(24))), \
                 mock.patch.object(step1_acts, "SEQUENCE_COUNT", 1), \
                 mock.patch.object(step1_acts, "SEQUENCE_LENGTH", 4), \
                 mock.patch.object(step1_acts, "EXPECTED_NEUTRAL_TOKEN_SHA256", "fixed-token-sha"), \
                 mock.patch.object(step1_acts, "code_commit", return_value="fixture-code"), \
                 mock.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=TinyTokenizer()), \
                 mock.patch.object(transformers.AutoModelForCausalLM, "from_pretrained", return_value=model), \
                 mock.patch("subprocess.run", side_effect=step1_subprocess):
                step1_acts.main(["--checkpoint", "instruct", "--snapshot-dir", str(snapshot),
                                 "--results-root", str(root), "--sequence-ids", str(root / "pilot-a1" / "neutral-token-ids.json"),
                                 "--passages", str(passages_path), "--push-branch", "part85-run/pilot-a1"])
            self.assertTrue(step1_path.is_file())
            self.assertEqual(len(step1_pushes), 1)
            self.assertEqual(json.loads(step1_path.read_text())["code_commit"], "fixture-code")

    def _write_step1_fixture(self, root, key, maximum=5.0, mode="main", attempt=1):
        base = Path(root) / (f"pilot-a{attempt}" if mode == "pilot" else "main")
        cp_root = base / key
        cp_root.mkdir(parents=True, exist_ok=True)
        payload = {"checkpoint": {"key": key, "commit": TEST_COMMIT}, "code_commit": "fixture-code",
                   "neutral_activation_maxima": {"6978": maximum, "7884": 2.0},
                   "neutral_token_ids_sha256": "f" * 64,
                   "dataset_revision": step1_acts.WIKITEXT_REVISION}
        (cp_root / "step1.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def _run_patches(self, observers, tokenizer):
        import transformers
        original_hook = part85.BothPhaseClampHook
        state = {"model": None}

        def load_model(*_args, **_kwargs):
            state["model"] = tiny_run_model()
            return state["model"]

        def tracked_hook(down_proj, unit, value):
            hook = original_hook(down_proj, unit, value)
            model = state["model"]
            layers = [module for module in model.modules()
                      if isinstance(module, torch.nn.ModuleList) and len(module) > 10
                      and hasattr(getattr(module[0], "mlp", None), "down_proj")]
            modules = list(max(layers, key=len))
            target_layer = next(i for i, layer in enumerate(modules)
                                if layer.mlp.down_proj is down_proj)
            observed_unit = unit if unit is not None else part85.TARGET_UNIT
            observation = {"unit": unit, "value": value, "target_layer": target_layer,
                           "layers": {i: {"prompt": [], "generated": []} for i in range(len(modules))}}
            observers.append(observation)
            pending_prompt = {i: True for i in range(len(modules))}
            original_begin = hook.begin_sequence

            def begin_sequence():
                for i in pending_prompt:
                    pending_prompt[i] = True
                original_begin()

            hook.begin_sequence = begin_sequence
            observer_handles = []
            for layer_index, layer in enumerate(modules):
                def record(_module, args, i=layer_index):
                    x = args[0]
                    phase = "prompt" if pending_prompt[i] else "generated"
                    pending_prompt[i] = False
                    observation["layers"][i][phase].extend(
                        x[..., observed_unit].detach().float().cpu().reshape(-1).tolist())
                observer_handles.append(layer.mlp.down_proj.register_forward_pre_hook(record))
            original_remove = hook.remove

            def remove():
                for handle in observer_handles:
                    handle.remove()
                original_remove()

            hook.remove = remove
            return hook

        return [mock.patch.object(transformers.AutoModelForCausalLM, "from_pretrained",
                                  side_effect=load_model),
                mock.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=tokenizer),
                mock.patch.object(part85, "resolve_checkpoint", side_effect=resolve_fake),
                mock.patch.object(part85, "BothPhaseClampHook", side_effect=tracked_hook)]

    def _run_pilot_main(self, root, key, mode, gate_path=None, chat=False, push_branch=None, observers=None,
                       attempt=1, plain_template_file=None, tokenizers=None):
        observers = observers if observers is not None else []
        tokenizer = TinyTokenizer()
        if tokenizers is not None:
            tokenizers.append(tokenizer)
        if chat:
            tokenizer.chat_template = "stub-template"
        patches = self._run_patches(observers, tokenizer)
        original_specs = part85._specs_for_run

        def small_specs(args):
            specs = original_specs(args)
            if args.mode == "pilot":
                return specs
            if args.checkpoint == "pretrain_4T":
                allowed = {("plain", "unclamped"), ("plain", "own_max"), ("plain", "absolute")}
            elif args.checkpoint == "instruct" and any(row["frame"] == "plain" for row in specs):
                allowed = {("plain", "unclamped"), ("plain", "own_max"), ("plain", "absolute"),
                           ("chat", "own_max")}
            else:
                allowed = {("chat", "unclamped"), ("chat", "own_max")}
            return [row for row in specs if (row["frame"], row["cell"]) in allowed]

        if mode == "pilot":
            argv = ["--mode", "pilot", "--checkpoint", key, "--snapshot-dir",
                    str(Path(root) / TEST_COMMIT), "--results-root", str(root), "--device", "cpu",
                    "--attempt", str(attempt)]
            samples_patch = mock.patch.object(part85, "PILOT_SAMPLES_PER_QUESTION", 2)
            expected_branch = f"part85-run/pilot-a{attempt}"
        else:
            argv = ["--mode", "main", "--checkpoint", key, "--pilot-score", str(gate_path),
                    "--snapshot-dir", str(Path(root) / TEST_COMMIT), "--results-root", str(root),
                    "--device", "cpu"]
            samples_patch = mock.patch.object(part85, "SAMPLES_PER_QUESTION", 1)
            expected_branch = f"part85-run/main-{key}"
        if push_branch:
            if push_branch == "auto":
                push_branch = expected_branch
            argv.extend(["--push-branch", push_branch])
        if plain_template_file:
            argv.extend(["--plain-template-file", str(plain_template_file)])
        with ExitStack() as stack:
            stack.enter_context(samples_patch)
            stack.enter_context(mock.patch.object(part85, "MAX_NEW_TOKENS", 2))
            stack.enter_context(mock.patch.object(part85, "code_commit", return_value="fixture-code"))
            stack.enter_context(mock.patch.object(part85, "_specs_for_run", side_effect=small_specs))
            for patcher in patches:
                stack.enter_context(patcher)
            part85.main(argv)

    def _assert_answer_files(self, root, key, mode_tag, frames, per_question, cells=None):
        # This expected-value table is computed in the test, never by a Part 85 helper.
        attempt = int(mode_tag[-1]) if mode_tag.startswith("pilot") else 1
        run_dir = Path(root) / (f"pilot-a{attempt}" if mode_tag.startswith("pilot") else "main") / key
        step1 = json.loads((run_dir / "step1.json").read_text(encoding="utf-8"))
        maximum = float(step1["neutral_activation_maxima"]["6978"])
        expected = {"unclamped": (None, None), "own_max": (6978, 1.5 * maximum),
                    "absolute": (6978, 9.75), "control": (7884, 3.0)}
        template_sha = hashlib.sha256(b"Question: {q}\nAnswer:").hexdigest()
        answer_dir = run_dir / "answers"
        record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["code_commit"], "fixture-code")
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
                    self.assertEqual(row["code_commit"], "fixture-code")
                    self.assertEqual(row["frame"], frame)
                hook_key = f"{mode_tag}/{frame}/{cell}"
                counts = record["hook_fire_counts"][hook_key]
                self.assertGreater(counts["prompt_positions"], 0, hook_key)
                self.assertGreater(counts["generated_positions"], 0, hook_key)

    def _assert_observers(self, observers):
        for observation in observers:
            self.assertEqual(observation["target_layer"], 10, observation)
            for index, phases in observation["layers"].items():
                self.assertTrue(phases["prompt"], (index, observation))
                self.assertTrue(phases["generated"], (index, observation))
                observed = phases["prompt"] + phases["generated"]
                requested = observation["value"]
                if observation["unit"] is not None and index == 10:
                    self.assertTrue(all(math.isclose(v, requested, rel_tol=1e-6, abs_tol=1e-6)
                                        for v in observed), (index, observation))
                elif requested is not None:
                    self.assertFalse(all(math.isclose(v, requested, rel_tol=1e-6, abs_tol=1e-6)
                                         for v in observed), (index, observation))

    def _run_and_audit_registered_case(self, case_root, key, mode, attempt=1, custom_template=False,
                                       first_token=None, expect_pass=True, patch_run=None):
        import transformers

        case_root = Path(case_root)
        results_root = case_root / "results"
        (case_root / "snapshot").mkdir(parents=True, exist_ok=True)
        (case_root / "snapshot" / "config.json").write_text(json.dumps({"_commit_hash": TEST_COMMIT}))
        self._write_step1_fixture(results_root, key, mode=mode, attempt=attempt)
        tokenizer_dir = case_root / "tokenizer"
        tokenizer = char_tokenizer(tokenizer_dir)
        token = CHAR_SPECIALS.index("<eos>") if first_token == "eos" else len(CHAR_SPECIALS) + CHAR_SET.index("E")
        argv = ["--mode", mode, "--checkpoint", key,
                "--snapshot-dir", str(case_root / "snapshot"),
                "--results-root", str(results_root), "--device", "cpu"]
        template_path = None
        if mode == "pilot":
            argv.extend(["--attempt", str(attempt)])
        else:
            gate_path = case_root / "pilot-score.json"
            gate_path.write_text(json.dumps({"pilot_gate": {
                "action": "plain_frame_pass",
                "plain_template_sha256": part85._template_sha256(part85.PLAIN_TEMPLATE)}}))
            argv.extend(["--pilot-score", str(gate_path)])
        if custom_template:
            template_path = results_root / f"pilot-a{attempt}" / "pilot-frame-v2.txt"
            template_path.parent.mkdir(parents=True, exist_ok=True)
            template_path.write_text("Revised origin prompt: {q}\nAnswer:", encoding="utf-8")
            argv.extend(["--plain-template-file", str(template_path)])
        patches = [mock.patch.object(transformers.AutoModelForCausalLM, "from_pretrained",
                                     side_effect=lambda *_a, **_k: tiny_audit_model(len(tokenizer), token)),
                   mock.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=tokenizer),
                   mock.patch.object(part85, "resolve_checkpoint", side_effect=resolve_fake),
                   mock.patch.object(part85, "code_commit", return_value="fixture-code")]
        with contextlib.ExitStack() as stack:
            for patch in patches + list(patch_run or []):
                stack.enter_context(patch)
            part85.main(argv)

        cp_dir = results_root / (f"pilot-a{attempt}" if mode == "pilot" else "main") / key
        audit_argv = [sys.executable, "-I", "-B",
                      str(Path(part85.__file__).with_name("audit85.py")), str(cp_dir),
                      "--checkpoint", key, "--mode", mode, "--tokenizer", str(tokenizer_dir)]
        if mode == "pilot":
            audit_argv.extend(["--attempt", str(attempt)])
        if custom_template:
            audit_argv.extend(["--template-file", str(template_path)])
        audited = subprocess.run(audit_argv, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, timeout=180)
        if expect_pass:
            self.assertEqual(audited.returncode, 0, audited.stdout)
            self.assertIn("PASS", audited.stdout.splitlines()[-1])
        # The run's own hook bookkeeping: one prompt call per answer (the observer audit cannot see this counter).
        record = json.loads((cp_dir / "run.json").read_text(encoding="utf-8"))
        per_q = part85.PILOT_SAMPLES_PER_QUESTION if mode == "pilot" else part85.SAMPLES_PER_QUESTION
        for cell_key, counts in record["hook_fire_counts"].items():
            self.assertEqual(counts["prompt_calls"], len(part85.QUESTIONS) * per_q, cell_key)
        # The run's own trace must agree with the observer that a clamped answer's prompt carried the set value.
        for trace in sorted((cp_dir / "trace").glob("*.jsonl")):
            for row in (json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()):
                if row["unit"] is not None:
                    self.assertGreaterEqual(row["prompt_positions_set"], 1, (trace.name, row))
        return results_root, cp_dir, audited

    def test_real_run_trace_passes_independent_audit_for_registered_scenarios(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cases = (
                ("pilot-instruct", "instruct", "pilot", 1, False),
                ("main-pretrain", "pretrain_4T", "main", 1, False),
                ("main-instruct", "instruct", "main", 1, False),
                ("pilot-attempt2", "instruct", "pilot", 2, True),
            )
            for directory, key, mode, attempt, custom_template in cases:
                with self.subTest(case=directory):
                    self._run_and_audit_registered_case(root / directory, key, mode, attempt, custom_template)

    def test_immediate_eos_answers_pass_the_run_and_the_audit(self):
        # Round-4 #6: an answer whose first token is EOS has no decode forward; that is legal and must audit clean.
        with tempfile.TemporaryDirectory() as temp:
            for key, mode in (("instruct", "pilot"), ("instruct", "main")):
                with self.subTest(mode=mode):
                    _root, cp_dir, _audit = self._run_and_audit_registered_case(
                        Path(temp) / mode, key, mode, first_token="eos")
                    rows = [json.loads(line) for path in sorted((cp_dir / "answers").glob("*.jsonl"))
                            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                    self.assertTrue(rows)
                    self.assertTrue(all(row["new_tokens"] == 1 and row["answer"] == "" for row in rows))

    def test_observer_audit_fails_a_tampered_answer_and_a_missing_call(self):
        with tempfile.TemporaryDirectory() as temp:
            _root, cp_dir, _audit = self._run_and_audit_registered_case(Path(temp) / "case", "instruct", "pilot")
            path = cp_dir / "answers" / "pilot1-plain--own_max.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            original = path.read_text(encoding="utf-8")
            for change, expected in ((lambda r: r.update(answer=r["answer"] + " extra"), "not the decode"),
                                     (lambda r: r.update(observer_call_id="missing:1"), "no observer record")):
                edited = [dict(r) for r in rows]
                change(edited[0])
                path.write_text("".join(json.dumps(r) + "\n" for r in edited), encoding="utf-8")
                audited = subprocess.run([sys.executable, "-I", "-B", str(Path(part85.__file__).with_name("audit85.py")),
                                          str(cp_dir), "--checkpoint", "instruct", "--mode", "pilot",
                                          "--tokenizer", str(Path(temp) / "case" / "tokenizer")],
                                         text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
                self.assertEqual(audited.returncode, 1, audited.stdout)
                self.assertIn(expected, audited.stdout)
                path.write_text(original, encoding="utf-8")

    def test_step1_run_passes_the_observer_audit_and_a_changed_value_fails(self):
        import transformers

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.json").write_text(json.dumps({"_commit_hash": TEST_COMMIT}))
            tokenizer = char_tokenizer(root / "tokenizer")
            passages = [{"id": str(i), "offworld": "Far away, past Mars.", "ordinary": "Nearby, at home."}
                        for i in range(24)]
            passages_path = root / "passages.json"
            passages_path.write_text(json.dumps(passages))
            ids = np.array([[4, 5, 6, 7], [8, 9, 10, 11]], dtype=np.int32)
            sha = step1_acts.stable_ids_sha256(ids)
            ids_path = root / "ids.json"
            cp = resolve_fake("instruct")
            model = tiny_run_model(len(tokenizer)).to(dtype=torch.bfloat16)
            model.config._commit_hash = TEST_COMMIT
            with mock.patch.object(step1_acts, "resolve_checkpoint", return_value=cp), \
                    mock.patch.object(step1_acts, "load_or_create_token_ids", return_value=(ids, sha)), \
                    mock.patch.object(step1_acts, "SEQUENCE_COUNT", 2), \
                    mock.patch.object(step1_acts, "SEQUENCE_LENGTH", 4), \
                    mock.patch.object(step1_acts, "EXPECTED_NEUTRAL_TOKEN_SHA256", sha), \
                    mock.patch.object(step1_acts, "code_commit", return_value="fixture-code"), \
                    mock.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=tokenizer), \
                    mock.patch.object(transformers.AutoModelForCausalLM, "from_pretrained", return_value=model):
                step1_acts.run(type("Args", (), {"checkpoint": "instruct", "snapshot_dir": str(root),
                                                  "device": "cpu", "sequence_ids": str(ids_path),
                                                  "passages": str(passages_path), "reference_acts": None,
                                                  "results_root": str(root), "out": str(root / "step1.json")})())
            ids_path.write_text(json.dumps({"token_ids": ids.tolist(), "sha256": sha}))
            argv = [sys.executable, "-I", "-B", str(Path(step1_acts.__file__).with_name("audit85.py")), str(root),
                    "--checkpoint", "instruct", "--step1", "--tokenizer", str(root / "tokenizer"),
                    "--token-ids", str(ids_path), "--passages", str(passages_path),
                    "--expected-token-sha", sha, "--test-sizes", "2", "4", "24"]
            audited = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
            self.assertEqual(audited.returncode, 0, audited.stdout)
            self.assertIn("PASS", audited.stdout.splitlines()[-1])
            payload = json.loads((root / "step1.json").read_text(encoding="utf-8"))
            payload["neutral_activations"][0][0][0] += 0.25
            (root / "step1.json").write_text(json.dumps(payload))
            audited = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
            self.assertEqual(audited.returncode, 1, audited.stdout)
            self.assertIn("not the values layer 10", audited.stdout)

    def test_trace_question_is_captured_from_the_prompt_builder(self):
        import transformers

        with tempfile.TemporaryDirectory() as temp:
            case_root = Path(temp)
            results_root = case_root / "results"
            snapshot = case_root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text(json.dumps({"_commit_hash": TEST_COMMIT}))
            self._write_step1_fixture(results_root, "instruct", mode="pilot")
            tokenizer = TraceTokenizer()
            original_builder = part85._prompt_inputs

            def altered_builder(tokenizer, question, frame, device, plain_template=part85.PLAIN_TEMPLATE,
                                trace_state=None):
                return original_builder(tokenizer, "builder received: " + question, frame, device,
                                        plain_template, trace_state=trace_state)

            with mock.patch.object(transformers.AutoModelForCausalLM, "from_pretrained",
                                   side_effect=lambda *_args, **_kwargs: tiny_audit_model()), \
                 mock.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=tokenizer), \
                 mock.patch.object(part85, "resolve_checkpoint", side_effect=resolve_fake), \
                 mock.patch.object(part85, "code_commit", return_value="fixture-code"), \
                 mock.patch.object(part85, "_prompt_inputs", side_effect=altered_builder), \
                 mock.patch.object(part85, "PILOT_SAMPLES_PER_QUESTION", 1), \
                 mock.patch.object(part85, "MAX_NEW_TOKENS", 2):
                part85.main(["--mode", "pilot", "--checkpoint", "instruct", "--attempt", "1",
                             "--snapshot-dir", str(snapshot), "--results-root", str(results_root),
                             "--device", "cpu"])
            trace_path = results_root / "pilot-a1" / "instruct" / "trace" / "pilot1-plain--unclamped.jsonl"
            rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["question"], "builder received: " + part85.QUESTIONS[0])
            self.assertEqual(rows[0]["prompt_text"], "Question: builder received: " +
                             part85.QUESTIONS[0] + "\nAnswer:")

    def test_resume_rejects_a_trace_with_a_missing_answer_row(self):
        import transformers

        with tempfile.TemporaryDirectory() as temp:
            case_root = Path(temp)
            results_root = case_root / "results"
            snapshot = case_root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text(json.dumps({"_commit_hash": TEST_COMMIT}))
            self._write_step1_fixture(results_root, "instruct", mode="pilot")

            def run_once():
                with mock.patch.object(transformers.AutoModelForCausalLM, "from_pretrained",
                                       side_effect=lambda *_args, **_kwargs: tiny_audit_model()), \
                     mock.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=TraceTokenizer()), \
                     mock.patch.object(part85, "resolve_checkpoint", side_effect=resolve_fake), \
                     mock.patch.object(part85, "code_commit", return_value="fixture-code"), \
                     mock.patch.object(part85, "PILOT_SAMPLES_PER_QUESTION", 1), \
                     mock.patch.object(part85, "MAX_NEW_TOKENS", 2):
                    part85.main(["--mode", "pilot", "--checkpoint", "instruct", "--attempt", "1",
                                 "--snapshot-dir", str(snapshot), "--results-root", str(results_root),
                                 "--device", "cpu"])

            run_once()
            trace_path = results_root / "pilot-a1" / "instruct" / "trace" / "pilot1-plain--own_max.jsonl"
            trace_rows = trace_path.read_text(encoding="utf-8").splitlines()
            trace_path.write_text("\n".join(trace_rows[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "trace .* has 2 rows, expected 3"):
                run_once()

    def test_run_entrypoint_disk_contract_resume_chat_and_changed_step1(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "results"
            (root / TEST_COMMIT).mkdir(parents=True)
            for key in ("instruct", "pretrain_4T"):
                self._write_step1_fixture(root, key, mode="pilot")
            for key in ("pretrain_4T", "instruct", "sft", "dpo"):
                self._write_step1_fixture(root, key, mode="main")
            observers = []
            with mock.patch("subprocess.run") as pushed, mock.patch.object(part85, "code_commit", return_value="fixture-code"):
                for key in ("instruct", "pretrain_4T"):
                    observed = []
                    self._run_pilot_main(root, key, "pilot", push_branch="auto",
                                         observers=observed)
                    observers.extend(observed)
                    self._assert_answer_files(root, key, "pilot1", ("plain",), 2,
                                              cells=("unclamped", "own_max"))
                calls = [call for call in pushed.call_args_list if "handoff85.py" in str(call)]
                self.assertEqual(len(calls), 4, "push must run after each pilot answer cell")
                expected_checkpoints = ("instruct", "instruct", "pretrain_4T", "pretrain_4T")
                for call, expected_checkpoint in zip(calls, expected_checkpoints):
                    argv = call.args[0]
                    self.assertEqual(argv[argv.index("--checkpoint") + 1], expected_checkpoint)
                    self.assertEqual(argv[argv.index("--branch") + 1], "part85-run/pilot-a1")
            self._assert_observers(observers)

            # Resume must preserve the completed file and its hook counts while generating missing cells.
            cp_root = root / "pilot-a1" / "pretrain_4T"
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
            self.assertTrue(any(item["unit"] == 6978 and item["value"] == 7.5 for item in resumed),
                            "resume did not generate the missing own-max cell")
            self._assert_observers(resumed)

            gate = base / "plain-pass.json"
            gate.write_text(json.dumps({"pilot_gate": {
                "action": "plain_frame_pass",
                "plain_template_sha256": hashlib.sha256(b"Question: {q}\nAnswer:").hexdigest()}}))
            pretrain_observers = []
            self._run_pilot_main(root, "pretrain_4T", "main", gate, observers=pretrain_observers)
            self._assert_answer_files(root, "pretrain_4T", "main", ("plain",), 1,
                                      cells=("unclamped", "own_max", "absolute"))
            self._assert_observers(pretrain_observers)

            post_observers = []
            chat_tokenizers = []
            self._run_pilot_main(root, "instruct", "main", gate, chat=True, observers=post_observers,
                                 tokenizers=chat_tokenizers)
            self._assert_answer_files(root, "instruct", "main", ("plain",), 1,
                                      cells=("unclamped", "own_max", "absolute"))
            self._assert_answer_files(root, "instruct", "main", ("chat",), 1, cells=("own_max",))
            self._assert_observers(post_observers)
            self.assertEqual(chat_tokenizers[0].chat_calls, 3,
                             "chat prompts must use apply_chat_template for all three questions")

            chat_root = base / "chat-results"
            (chat_root / TEST_COMMIT).mkdir(parents=True)
            for key in ("pretrain_4T", "sft", "dpo", "instruct"):
                self._write_step1_fixture(chat_root, key)
            chat_gate = base / "chat-only.json"
            chat_gate.write_text(json.dumps({"pilot_gate": {"action": "chat_only_posttrained",
                "plain_template_sha256": hashlib.sha256(b"Question: {q}\nAnswer:").hexdigest()}}))
            with self.assertRaisesRegex(ValueError, "outside the main run scope"):
                self._run_pilot_main(chat_root, "pretrain_4T", "main", chat_gate, chat=True)
            for key in ("sft", "dpo", "instruct"):
                observed = []
                tokenizers = []
                self._run_pilot_main(chat_root, key, "main", chat_gate, chat=True, observers=observed,
                                     tokenizers=tokenizers)
                self._assert_answer_files(chat_root, key, "main", ("chat",), 1,
                                          cells=("unclamped", "own_max"))
                self._assert_observers(observed)
                self.assertEqual(tokenizers[0].chat_calls, 6, key)
                filenames = [p.name for p in (chat_root / "main" / key / "answers").glob("main-*.jsonl")]
                self.assertTrue(filenames and all("-chat--" in name for name in filenames), filenames)

            # Remove completed files so this refusal specifically proves the changed-step guard.
            changed_root = base / "changed"
            shutil.copytree(root / "pilot-a1" / "instruct", changed_root / "pilot-a1" / "instruct")
            shutil.rmtree(changed_root / "pilot-a1" / "instruct" / "answers")
            changed_snapshot = changed_root / TEST_COMMIT
            changed_snapshot.mkdir(parents=True)
            step1_path = changed_root / "pilot-a1" / "instruct" / "step1.json"
            changed = json.loads(step1_path.read_text(encoding="utf-8"))
            changed["neutral_activation_maxima"]["6978"] = 7.0
            step1_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "step1 clamp inputs changed"):
                self._run_pilot_main(changed_root, "instruct", "pilot")

    def test_attempt2_custom_template_text_reaches_every_prompt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "results"
            snapshot = Path(temp) / TEST_COMMIT
            snapshot.mkdir()
            (snapshot / "config.json").write_text(json.dumps({"_commit_hash": TEST_COMMIT}))
            self._write_step1_fixture(root, "instruct", maximum=4.0, mode="pilot", attempt=2)
            template_path = root / "pilot-a2" / "pilot-frame-v2.txt"
            template_path.parent.mkdir(parents=True, exist_ok=True)
            template_path.write_text("Revised origin prompt: {q}\nAnswer:", encoding="utf-8")
            snapshot = root / TEST_COMMIT
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text(json.dumps({"_commit_hash": TEST_COMMIT}))
            observers, tokenizers = [], []
            self._run_pilot_main(root, "instruct", "pilot", observers=observers, attempt=2,
                                 plain_template_file=template_path, tokenizers=tokenizers)
            self.assertTrue(tokenizers[0].plain_prompts)
            self.assertTrue(all(text.startswith("Revised origin prompt: ") for text in tokenizers[0].plain_prompts),
                            tokenizers[0].plain_prompts)
            rows = [json.loads(line) for path in (root / "pilot-a2" / "instruct" / "answers").glob("*.jsonl")
                    for line in path.read_text(encoding="utf-8").splitlines() if line]
            self.assertTrue(rows)
            self.assertTrue(all(row["frame_template_sha256"] == hashlib.sha256(template_path.read_bytes()).hexdigest()
                                for row in rows))

    def test_completed_cell_is_retried_after_push_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "results"
            self._write_step1_fixture(root, "instruct", mode="pilot")
            snapshot = root / TEST_COMMIT
            snapshot.mkdir()
            (snapshot / "config.json").write_text(json.dumps({"_commit_hash": TEST_COMMIT}))
            calls = []
            def fail_second(argv, **kwargs):
                if "handoff85.py" in str(argv):
                    calls.append(argv)
                    if len(calls) == 2:
                        raise subprocess.CalledProcessError(1, argv, stderr="temporary remote failure")
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
            with mock.patch("subprocess.run", side_effect=fail_second):
                with self.assertRaises(subprocess.CalledProcessError):
                    self._run_pilot_main(root, "instruct", "pilot", push_branch="auto")
            run_dir = root / "pilot-a1" / "instruct"
            self.assertEqual(len(list((run_dir / "answers").glob("*.jsonl"))), 2)
            retry_calls = []
            def retry(argv, **kwargs):
                if "handoff85.py" in str(argv):
                    retry_calls.append(argv)
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
            observations = []
            with mock.patch("subprocess.run", side_effect=retry):
                self._run_pilot_main(root, "instruct", "pilot", push_branch="auto", observers=observations)
            self.assertEqual(len(retry_calls), 1, "an all-complete rerun must still retry the result push")
            self.assertEqual(observations, [], "the already-complete cells must not be regenerated")
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
            references = root / "references"
            references.mkdir()
            (references / "pilot-score.json").write_text(json.dumps({"pilot_gate": {
                "action": "chat_only_posttrained",
                "plain_template_sha256": hashlib.sha256(b"Question: {q}\nAnswer:").hexdigest()}}))
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
            # Round-4 #7: steps 0 and 1 are required for all twelve checkpoints even when step 2 is chat-only.
            for key_name in BY_KEY:
                cp = BY_KEY[key_name]
                identity = {"key": key_name, "commit": cp.commit_prefix + "0" * (40 - len(cp.commit_prefix))}
                cp_root = result_root / "main" / key_name
                cp_root.mkdir(parents=True)
                (cp_root / "step0.json").write_text(json.dumps({
                    "checkpoint": identity, "code_commit": "fixture-code",
                    "measures": {"cosines_with_instruct": {"down_proj_column_6978": 0.95},
                                 "direct_readout": {"z_score": 0.0}}}))
                (cp_root / "step1.json").write_text(json.dumps({
                    "checkpoint": identity, "code_commit": "fixture-code",
                    "neutral_correlations_with_instruct": {"target": 0.95},
                    "passages": {"offworld_mean": 0.0, "ordinary_mean": 0.0, "ordinary_sd": 1.0}}))
            payload = score85.score_payload(key, labels, labels, mode="main",
                                            results_root=result_root, items_path=items,
                                            references_dir=references)
            self.assertEqual(payload["scope"], "chat_only_posttrained")
            for code in ("85.5", "85.6", "85.7", "85.8", "85.9"):
                self.assertEqual(payload["verdicts"][code]["verdict"], "NOT APPLICABLE")
            for code in ("85.3", "85.4"):
                self.assertIn(payload["verdicts"][code]["verdict"], ("HOLDS", "FAILS"), code)
            sft_step0 = result_root / "main" / "sft" / "step0.json"
            mismatched = json.loads(sft_step0.read_text(encoding="utf-8"))
            mismatched["checkpoint"]["key"] = "dpo"
            sft_step0.write_text(json.dumps(mismatched))
            with self.assertRaisesRegex(ValueError, "identifies checkpoint 'dpo'"):
                score85.score_payload(key, labels, labels, mode="main",
                                      results_root=result_root, items_path=items,
                                      references_dir=references)
            sft_step0.write_text(json.dumps({**mismatched, "checkpoint": {
                "key": "sft", "commit": BY_KEY["sft"].commit_prefix + "0" * (40 - len(BY_KEY["sft"].commit_prefix))},
                "measures": {"cosines_with_instruct": {"down_proj_column_6978": 0.95},
                             "direct_readout": {"z_score": 0.0}}}))
            (result_root / "main" / "dpo" / "step0.json").unlink()
            with self.assertRaisesRegex(ValueError, "all 12"):
                score85.score_payload(key, labels, labels, mode="main",
                                      results_root=result_root, items_path=items,
                                      references_dir=references)

    def test_a_measured_different_unit_is_a_result_not_missing_data(self):
        # Round-4 #5: 85.2 predicts a 21B cosine below 0.5, which fails the same-unit gate by design. That must score as
        # DIFFERENT UNIT with the raw counts, and must not abort complete scoring as NOT READ.
        counts, step0, step1 = make_counts()
        step0["pretrain_21B"]["measures"]["cosines_with_instruct"]["down_proj_column_6978"] = 0.4
        result = score85.evaluate_decisions(counts, step0, step1)
        verdict = result["verdicts"]["85.6"]
        self.assertEqual(verdict["verdict"], "DIFFERENT UNIT", verdict)
        self.assertIn("pretrain_21B own-max ", verdict["evidence"])
        self.assertIn("cosine 0.400", verdict["evidence"])
        step1["pretrain_21B"] = None
        self.assertEqual(score85.evaluate_decisions(counts, step0, step1)["verdicts"]["85.6"]["verdict"], "NOT READ")

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
                 mock.patch.object(step1_acts, "EXPECTED_NEUTRAL_TOKEN_SHA256", digest), \
                 mock.patch.object(transformers.AutoModelForCausalLM, "from_pretrained",
                                   side_effect=AssertionError("model load happened before token-hash refusal")):
                with self.assertRaisesRegex(ValueError, "different neutral token ids"):
                    step1_acts.run(args)

    def test_step1_refuses_token_ids_with_unpinned_digest_before_model_load(self):
        import transformers
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / TEST_COMMIT
            snapshot.mkdir()
            (snapshot / "config.json").write_text(json.dumps({"_commit_hash": TEST_COMMIT}))
            ids = np.zeros((step1_acts.SEQUENCE_COUNT, step1_acts.SEQUENCE_LENGTH), dtype=np.int32)
            digest = step1_acts.stable_ids_sha256(ids)
            ids_path = root / "ids.json"
            ids_path.write_text(json.dumps({"token_ids": ids.tolist(), "sha256": digest}))
            reference = root / "reference.json"
            reference.write_text(json.dumps({"checkpoint": {"key": "instruct"},
                "neutral_token_ids_sha256": digest, "neutral_activations": []}))
            args = type("Args", (), {"checkpoint": "pretrain_4T", "snapshot_dir": str(snapshot),
                "reference_acts": str(reference), "results_root": str(root), "sequence_ids": str(ids_path),
                "passages": str(root / "passages.json"), "device": "cpu", "out": None,
                "push_branch": None})()
            with mock.patch.object(step1_acts, "resolve_checkpoint", return_value=resolve_fake("pretrain_4T")), \
                 mock.patch.object(transformers.AutoModelForCausalLM, "from_pretrained",
                                   side_effect=AssertionError("model load happened before pinned token refusal")):
                with self.assertRaisesRegex(ValueError, "does not match pinned"):
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

    def test_cloud_runbook_cleanup_targets_huggingface_model_cache(self):
        runbook = Path(__file__).with_name("CLOUD-RUN.md").read_text(encoding="utf-8")
        cleanup = [line.strip() for line in runbook.splitlines() if line.strip().startswith("rm -rf ")]
        self.assertTrue(cleanup)
        self.assertTrue(all('models--${MODEL_ID//\\//--}' in line and 'SNAPSHOT_DIR' not in line
                            for line in cleanup), cleanup)

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
        return subprocess.run([sys.executable, "-I", "-B", str(Path(handoff85.__file__)), *map(str, args)],
                              cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_handoff_single_writer_bare_remote_restore_push_and_assemble(self):
        with tempfile.TemporaryDirectory() as temp:
            top = Path(temp)
            bare, seed = top / "remote.git", top / "seed"
            self._git_cmd(top, "init", "--bare", str(bare))
            seed.mkdir()
            self._git_cmd(seed, "init")
            self._set_git_identity(seed)
            (seed / "README").write_text("code-only base\n")
            self._git_cmd(seed, "add", "README")
            self._git_cmd(seed, "commit", "-m", "code-only")
            self._git_cmd(seed, "remote", "add", "origin", str(bare))
            self._git_cmd(seed, "push", "origin", "HEAD:refs/heads/main")
            clone_a, clone_b = top / "clone-a", top / "clone-b"
            self._git_cmd(top, "clone", str(bare), str(clone_a))
            self._git_cmd(top, "clone", str(bare), str(clone_b))
            self._set_git_identity(clone_a)
            self._set_git_identity(clone_b)

            root = clone_a / "results" / "part85"
            own = root / "pilot-a1"
            (own / "instruct" / "answers").mkdir(parents=True)
            (own / "instruct" / "trace").mkdir(parents=True)
            (own / "instruct" / "step0.json").write_text('{"checkpoint":{"key":"instruct"}}')
            (own / "instruct" / "answers" / "cell.jsonl").write_text('{"answer":"complete"}\n')
            (own / "instruct" / "trace" / "cell.jsonl").write_text('{"question":"measured"}\n')
            (own / "instruct" / "run.json").write_text(json.dumps({"completed_cells": {
                "pilot1/plain/unclamped": {"answer_file": "results/part85/pilot-a1/instruct/answers/cell.jsonl"}}}))
            token_path = own / "neutral-token-ids.json"
            token_path.write_text(json.dumps({"sha256": handoff85.EXPECTED_TOKEN_SHA256, "token_ids": []}))
            pushed = self._handoff_command(clone_a, "push", "--checkpoint", "instruct",
                                           "--branch", "part85-run/pilot-a1", "--results-root", "results/part85")
            self.assertEqual(pushed.returncode, 0, pushed.stderr)
            tip = self._git_cmd(clone_a, "ls-remote", "origin", "refs/heads/part85-run/pilot-a1").stdout.split()[0]
            self.assertEqual(self._git_cmd(clone_a, "rev-parse", "HEAD").stdout.strip(), tip)
            remote_names = self._git_cmd(clone_a, "--git-dir", str(bare), "ls-tree", "-r", "--name-only", tip).stdout.splitlines()
            self.assertTrue(remote_names)
            self.assertTrue(all(name.startswith("results/part85/pilot-a1/") for name in remote_names), remote_names)

            (own / "pretrain_4T").mkdir()
            (own / "pretrain_4T" / "step0.json").write_text('{"checkpoint":{"key":"pretrain_4T"}}')
            second = self._handoff_command(clone_a, "push", "--checkpoint", "pretrain_4T",
                                           "--branch", "part85-run/pilot-a1", "--results-root", "results/part85")
            self.assertEqual(second.returncode, 0, second.stderr)
            subjects = self._git_cmd(clone_a, "log", "-2", "--format=%s").stdout.splitlines()
            self.assertEqual(len(subjects), 2)
            self.assertNotEqual(subjects[0], subjects[1], "result commits need distinct incident breadcrumbs")
            tip2 = self._git_cmd(clone_a, "ls-remote", "origin", "refs/heads/part85-run/pilot-a1").stdout.split()[0]
            self.assertNotEqual(tip, tip2)
            names2 = self._git_cmd(clone_a, "--git-dir", str(bare), "ls-tree", "-r", "--name-only", tip2).stdout
            self.assertIn("results/part85/pilot-a1/instruct/answers/cell.jsonl", names2)
            self.assertIn("results/part85/pilot-a1/pretrain_4T/step0.json", names2)

            restore = self._handoff_command(clone_b, "restore", "--branch", "part85-run/pilot-a1",
                                            "--results-root", "results/part85")
            self.assertEqual(restore.returncode, 0, restore.stderr)
            self.assertEqual(self._git_cmd(clone_b, "rev-parse", "HEAD").stdout.strip(), tip2)
            self.assertTrue((clone_b / "results/part85/pilot-a1/instruct/step0.json").is_file())

            wrong_branch = self._handoff_command(clone_a, "push", "--checkpoint", "sft",
                                                 "--branch", "part85-run/main-instruct", "--results-root", "results/part85")
            self.assertNotEqual(wrong_branch.returncode, 0)
            self.assertIn("does not match checkpoint", wrong_branch.stderr)

            (own / "instruct" / "run.json").write_text(json.dumps({"completed_cells": {
                "pilot1/plain/unclamped": {"answer_file": "results/part85/pilot-a1/instruct/answers/missing.jsonl"}}}))
            update = self._handoff_command(clone_a, "push", "--checkpoint", "instruct",
                                           "--branch", "part85-run/pilot-a1", "--results-root", "results/part85")
            self.assertEqual(update.returncode, 0, update.stderr)
            assembled = self._handoff_command(clone_a, "assemble", "--branches", "part85-run/pilot-a1",
                                              "--out", top / "assembled")
            self.assertNotEqual(assembled.returncode, 0)
            self.assertIn("claims missing answer file", assembled.stderr)
            (own / "instruct" / "run.json").write_text(json.dumps({"completed_cells": {
                "pilot1/plain/unclamped": {"answer_file": "results/part85/pilot-a1/instruct/answers/cell.jsonl"}}}))
            repaired = self._handoff_command(clone_a, "push", "--checkpoint", "instruct",
                                             "--branch", "part85-run/pilot-a1", "--results-root", "results/part85")
            self.assertEqual(repaired.returncode, 0, repaired.stderr)
            fresh = top / "fresh-coordinator"
            self._git_cmd(top, "clone", str(bare), str(fresh))
            fresh_assembled = self._handoff_command(fresh, "assemble", "--branches", "part85-run/pilot-a1",
                                                     "--out", top / "fresh-assembled")
            self.assertEqual(fresh_assembled.returncode, 0, fresh_assembled.stderr)
            self.assertTrue((top / "fresh-assembled/pilot-a1/instruct/answers/cell.jsonl").is_file())
            self.assertTrue((top / "fresh-assembled/pilot-a1/instruct/trace/cell.jsonl").is_file())
            foreign = clone_a / "results/part85/main/sft/intruder.json"
            foreign.parent.mkdir(parents=True)
            foreign.write_text("outside pilot-a1 ownership")
            self._git_cmd(clone_a, "add", "results/part85/main/sft/intruder.json")
            self._git_cmd(clone_a, "commit", "-m", "foreign result file")
            self._git_cmd(clone_a, "push", "origin", "HEAD:refs/heads/part85-run/pilot-a1")
            foreign_assemble = self._handoff_command(clone_a, "assemble", "--branches", "part85-run/pilot-a1",
                                                     "--out", top / "foreign")
            self.assertNotEqual(foreign_assemble.returncode, 0)
            self.assertIn("outside its owned directory", foreign_assemble.stderr)

    def test_handoff_preflight_checks_manifest_and_only_validates_present_own_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / "repo"
            repo.mkdir()
            self._git_cmd(repo, "init")
            missing_pilot = self._handoff_command(repo, "preflight", "--checkpoint", "pretrain_4T",
                                                  "--phase", "pilot", "--attempt", "1",
                                                  "--results-root", "results/part85")
            self.assertNotEqual(missing_pilot.returncode, 0)
            self.assertIn("pilot preflight missing required files", missing_pilot.stderr)
            refs = repo / "part85" / "references"
            (refs / "instruct").mkdir(parents=True)
            score = {"pilot_gate": {"action": "plain_frame_pass",
                                    "plain_template_sha256": hashlib.sha256(b"Question: {q}\nAnswer:").hexdigest()}}
            payload = {"checkpoint": {"key": "instruct", "commit": BY_KEY["instruct"].commit_prefix + "0" * (40-len(BY_KEY["instruct"].commit_prefix))},
                       "code_commit": "code-only-commit"}
            for name in ("step0.json", "step1.json"):
                (refs / "instruct" / name).write_text(json.dumps(payload))
            (refs / "neutral-token-ids.json").write_text(json.dumps({"sha256": handoff85.EXPECTED_TOKEN_SHA256}))
            (refs / "pilot-score.json").write_text(json.dumps(score))
            entries = []
            for path in sorted(refs.rglob("*")):
                if path.is_file():
                    entries.append({"path": path.relative_to(repo).as_posix(),
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            manifest = {"schema_version": 2, "files": entries,
                        "plain_template_sha256": score["pilot_gate"]["plain_template_sha256"]}
            (repo / "part85" / "references.json").write_text(json.dumps(manifest))
            (repo / "results/part85/main/sft").mkdir(parents=True)
            preflight = self._handoff_command(repo, "preflight", "--checkpoint", "sft", "--phase", "main",
                                              "--results-root", "results/part85")
            self.assertEqual(preflight.returncode, 0, preflight.stderr)
            wrong_step = {"checkpoint": {"key": "dpo", "commit": BY_KEY["dpo"].commit_prefix + "0" * (40-len(BY_KEY["dpo"].commit_prefix))},
                          "code_commit": "code-only-commit"}
            own_step = repo / "results/part85/main/sft/step1.json"
            own_step.write_text(json.dumps(wrong_step))
            refused = self._handoff_command(repo, "preflight", "--checkpoint", "sft", "--phase", "main",
                                            "--results-root", "results/part85")
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("expected 'sft'", refused.stderr)
            own_step.unlink()
            (refs / "neutral-token-ids.json").write_text('{"sha256":"wrong"}')
            bad_hash = self._handoff_command(repo, "preflight", "--checkpoint", "sft", "--phase", "main",
                                             "--results-root", "results/part85")
            self.assertNotEqual(bad_hash.returncode, 0)
            self.assertIn("SHA-256 mismatch", bad_hash.stderr)

    def test_refs_manifest_copies_accepted_pilot_and_attempt2_template(self):
        with tempfile.TemporaryDirectory() as temp:
            top = Path(temp)
            root = top / "results/part85"
            (root / "pilot-a1/instruct").mkdir(parents=True)
            (root / "pilot-a1/instruct/step0.json").write_text('{"step":0}')
            (root / "pilot-a1/instruct/step1.json").write_text('{"step":1}')
            token_file = root / "pilot-a1/neutral-token-ids.json"
            token_file.write_text(json.dumps({"sha256": handoff85.EXPECTED_TOKEN_SHA256, "token_ids": []}))
            custom = b"Revised: {q}\\nAnswer:"
            (root / "pilot-a2").mkdir()
            (root / "pilot-a2/pilot-frame-v2.txt").write_bytes(custom)
            score = {"pilot_gate": {"action": "plain_frame_pass", "attempt": 2,
                     "plain_template_sha256": hashlib.sha256(custom).hexdigest()}}
            (root / "pilot-score.json").write_text(json.dumps(score))
            references = top / "repo/part85/references"
            manifest_path = top / "repo/part85/references.json"
            handoff85.refs_manifest(root, manifest_path, references)
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(len(manifest["files"]), 5)
            self.assertTrue((references / "pilot-frame-v2.txt").is_file())
            self.assertEqual((references / "pilot-frame-v2.txt").read_bytes(), custom)
            listed = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
            self.assertEqual(listed["part85/references/neutral-token-ids.json"],
                             hashlib.sha256((references / "neutral-token-ids.json").read_bytes()).hexdigest())

    def test_handoff_push_retries_unpushed_head_and_refuses_foreign_staging(self):
        with tempfile.TemporaryDirectory() as temp:
            top = Path(temp)
            bare, repo = top / "remote.git", top / "repo"
            self._git_cmd(top, "init", "--bare", str(bare))
            repo.mkdir()
            self._git_cmd(repo, "init")
            (repo / "README").write_text("base")
            self._git_cmd(repo, "add", "README")
            self._git_cmd(repo, "commit", "-m", "base")
            self._git_cmd(repo, "remote", "add", "origin", str(bare))
            self._git_cmd(repo, "push", "origin", "HEAD:refs/heads/main")
            own = repo / "results/part85/main/sft"
            own.mkdir(parents=True)
            (own / "step0.json").write_text("first")
            first = self._handoff_command(repo, "push", "--checkpoint", "sft", "--branch", "part85-run/main-sft",
                                          "--results-root", "results/part85")
            self.assertEqual(first.returncode, 0, first.stderr)
            (own / "step1.json").write_text("second")
            self._git_cmd(repo, "add", "results/part85/main/sft/step1.json")
            self._git_cmd(repo, "commit", "-m", "simulate committed but unpushed result")
            local_head = self._git_cmd(repo, "rev-parse", "HEAD").stdout.strip()
            remote_before = self._git_cmd(repo, "ls-remote", "origin", "refs/heads/part85-run/main-sft").stdout.split()[0]
            self.assertNotEqual(local_head, remote_before)
            retry = self._handoff_command(repo, "push", "--checkpoint", "sft", "--branch", "part85-run/main-sft",
                                          "--results-root", "results/part85")
            self.assertEqual(retry.returncode, 0, retry.stderr)
            remote_after = self._git_cmd(repo, "ls-remote", "origin", "refs/heads/part85-run/main-sft").stdout.split()[0]
            self.assertEqual(remote_after, local_head)
            self.assertIn("results/part85/main/sft/step1.json",
                          self._git_cmd(repo, "--git-dir", str(bare), "ls-tree", "-r", "--name-only", remote_after).stdout)

            (repo / "other.txt").write_text("foreign staged")
            self._git_cmd(repo, "add", "other.txt")
            foreign = self._handoff_command(repo, "push", "--checkpoint", "sft", "--branch", "part85-run/main-sft",
                                            "--results-root", "results/part85")
            self.assertNotEqual(foreign.returncode, 0)
            self.assertIn("unrelated paths are already staged", foreign.stderr)

    def test_run_pushes_each_complete_cell_to_a_local_bare_remote(self):
        with tempfile.TemporaryDirectory() as temp:
            top = Path(temp)
            bare, seed, clone = top / "remote.git", top / "seed", top / "clone"
            self._git_cmd(top, "init", "--bare", str(bare))
            seed.mkdir()
            self._git_cmd(seed, "init")
            self._set_git_identity(seed)
            shutil.copytree(Path(__file__).resolve().parent, seed / "part85")
            (seed / "README").write_text("Part 85 integration fixture\n")
            self._git_cmd(seed, "add", "README", "part85")
            self._git_cmd(seed, "commit", "-m", "code-only test fixture")
            self._git_cmd(seed, "remote", "add", "origin", str(bare))
            self._git_cmd(seed, "push", "origin", "HEAD:refs/heads/main")
            self._git_cmd(top, "clone", str(bare), str(clone))
            # This untracked sibling must never be staged by a Part 85 result push.
            (clone / "unrelated.txt").write_text("not owned by this session")
            flow = clone / "flow_check.py"
            flow_text = r'''import json, subprocess, sys
from pathlib import Path
from unittest import mock

remote = Path(sys.argv[1])
sys.path.insert(0, "@PART85_SOURCE@")
import transformers
import part85
import test_part85 as fixtures

root = Path("results/part85")
cp = fixtures.resolve_fake("instruct")
out = root / "pilot-a1" / "instruct"
out.mkdir(parents=True)
step1 = {"checkpoint": {"key": "sft", "commit": cp["commit"]},
         "neutral_activation_maxima": {"6978": 5.0, "7884": 2.0},
         "neutral_token_ids_sha256": "f" * 64}
(out / "step1.json").write_text(json.dumps(step1))
gate = Path("gate.json")
gate.write_text(json.dumps({"pilot_gate": {"action": "plain_frame_pass",
    "plain_template_sha256": part85._template_sha256(part85.PLAIN_TEMPLATE)}}))
snapshot = Path("snapshot")
snapshot.mkdir()
(snapshot / "config.json").write_text(json.dumps({"_commit_hash": cp["commit"]}))
tokenizer = fixtures.TinyTokenizer()
tokenizer.chat_template = "stub-chat"
real_run = subprocess.run
events = []

def watch(argv, *args, **kwargs):
    if isinstance(argv, (list, tuple)) and any("handoff85.py" in str(value) for value in argv):
        pushed = real_run(argv, *args, **kwargs)
        if pushed.returncode:
            return pushed
        tip = real_run(["git", "ls-remote", "origin", "refs/heads/part85-run/pilot-a1"],
                       text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout.split()[0]
        names = real_run(["git", "--git-dir", str(remote), "ls-tree", "-r", "--name-only", tip],
                         text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout.splitlines()
        answers = [name for name in names if name.startswith("results/part85/pilot-a1/instruct/answers/") and name.endswith(".jsonl")]
        assert len(answers) == len(events) + 1, (len(events), answers)
        run_name = "results/part85/pilot-a1/instruct/run.json"
        run_bytes = real_run(["git", "--git-dir", str(remote), "show", f"{tip}:{run_name}"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
        completed = json.loads(run_bytes.decode())["completed_cells"]
        for record in completed.values():
            assert record["answer_file"] in names, record
        events.append({"tip": tip, "answer_count": len(answers), "claims": len(completed)})
        return pushed
    return real_run(argv, *args, **kwargs)

with mock.patch.object(transformers.AutoModelForCausalLM, "from_pretrained",
                       side_effect=lambda *_a, **_kw: fixtures.tiny_run_model()), \
     mock.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=tokenizer), \
     mock.patch.object(part85, "resolve_checkpoint", side_effect=fixtures.resolve_fake), \
     mock.patch.object(part85, "PILOT_SAMPLES_PER_QUESTION", 1), \
     mock.patch.object(part85, "MAX_NEW_TOKENS", 2), \
     mock.patch("subprocess.run", side_effect=watch):
    part85.main(["--mode", "pilot", "--checkpoint", "instruct",
                 "--snapshot-dir", str(snapshot), "--results-root", str(root),
                 "--device", "cpu", "--push-branch", "part85-run/pilot-a1"])
assert len(events) == 2, events
assert [event["answer_count"] for event in events] == [1, 2], events
'''
            flow.write_text(flow_text.replace("@PART85_SOURCE@", str(Path(__file__).resolve().parent)))
            result = subprocess.run([sys.executable, "-I", "-B", str(flow), str(bare)], cwd=clone,
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
            self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
