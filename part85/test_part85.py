#!/usr/bin/env python3
"""CPU-only, network-free checks for Part 85's run, readers, and scoring rules."""
from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
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
                                      "pilot_gate_action": "chat_only_posttrained"})
        self.assertEqual(score85.validate_record_counts(chat_rows, "main"), None)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            score85.validate_record_counts([{**row, "pilot_gate_action": None} for row in chat_rows], "main")

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

    def test_run_pilot_main_prepare_and_score_smoke(self):
        import transformers

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "results"
            snapshot = Path(temp) / "snapshot"
            snapshot.mkdir()
            for key in ("pretrain_4T", "instruct"):
                cp = resolve_fake(key)
                cp_root = root / key
                cp_root.mkdir(parents=True)
                (cp_root / "step1.json").write_text(json.dumps({
                    "checkpoint": {"commit": TEST_COMMIT, "key": key},
                    "neutral_activation_maxima": {"6978": 6.5, "7884": 2.0},
                    "dataset_revision": step1_acts.WIKITEXT_REVISION,
                }))
            tokenizer = TinyTokenizer()
            model_patch = mock.patch.object(transformers.AutoModelForCausalLM, "from_pretrained",
                                            side_effect=lambda *_args, **_kwargs: tiny_run_model())
            tokenizer_patch = mock.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=tokenizer)
            resolve_patch = mock.patch.object(part85, "resolve_checkpoint", side_effect=resolve_fake)
            with mock.patch.object(part85, "PILOT_SAMPLES_PER_QUESTION", 6), \
                    mock.patch.object(part85, "SAMPLES_PER_QUESTION", 1), \
                    mock.patch.object(part85, "MAX_NEW_TOKENS", 2), \
                    model_patch, tokenizer_patch, resolve_patch:
                for key in ("pretrain_4T", "instruct"):
                    part85.run(type("Args", (), {"mode": "pilot", "checkpoint": key, "pilot_score": None,
                                                  "attempt": 1, "plain_template_file": None, "step1": None,
                                                  "results_root": str(root), "device": "cpu",
                                                  "snapshot_dir": str(snapshot)})())

                pilot_items = Path(temp) / "pilot-items.jsonl"
                pilot_key_path = Path(temp) / "pilot-key.json"
                pilot_score_path = Path(temp) / "pilot-score.json"
                score85.main(["prepare", "--mode", "pilot", "--attempt", "1", "--results-root", str(root),
                              "--items", str(pilot_items), "--key", str(pilot_key_path)])
                pilot_key = json.loads(pilot_key_path.read_text(encoding="utf-8"))
                pilot_labels = []
                for item_id, meta in pilot_key["items"].items():
                    origin = "OFFWORLD" if meta["checkpoint"] == "instruct" and meta["cell"] == "own_max" else "NEITHER"
                    pilot_labels.append({"id": item_id, "origin": origin, "form": "OK", "responsive": True})
                pilot_label_path = Path(temp) / "pilot-labels.jsonl"
                pilot_label_path.write_text("".join(json.dumps(row) + "\n" for row in pilot_labels))
                score85.main(["score", "--mode", "pilot", "--key", str(pilot_key_path), "--items", str(pilot_items),
                              "--labels", str(pilot_label_path), "--out", str(pilot_score_path)])
                pilot_score = json.loads(pilot_score_path.read_text(encoding="utf-8"))
                self.assertEqual(pilot_score["pilot_gate"]["action"], "plain_frame_pass")

                part85.run(type("Args", (), {"mode": "main", "checkpoint": "pretrain_4T",
                                              "pilot_score": str(pilot_score_path), "attempt": 1,
                                              "plain_template_file": None, "step1": None,
                                              "results_root": str(root), "device": "cpu",
                                              "snapshot_dir": str(snapshot)})())
                run_record = json.loads((root / "pretrain_4T" / "run.json").read_text(encoding="utf-8"))
                self.assertEqual(run_record["dtype"], "torch.float32")
                self.assertEqual(run_record["dataset_revision"], step1_acts.WIKITEXT_REVISION)
                self.assertEqual(run_record["pilot_gate_action"], "plain_frame_pass")
                self.assertIn("eos_token_id", run_record["effective_generation_config"])

            main_specs = part85.expected_cell_specs("main")
            for spec in main_specs:
                cp_key = spec["checkpoint"]
                cp = resolve_fake(cp_key)
                step1 = json.loads((root / cp_key / "step1.json").read_text(encoding="utf-8")) \
                    if (root / cp_key / "step1.json").exists() else {
                        "checkpoint": {"commit": TEST_COMMIT, "key": cp_key},
                        "neutral_activation_maxima": {"6978": 6.5, "7884": 2.0}}
                (root / cp_key).mkdir(parents=True, exist_ok=True)
                if not (root / cp_key / "step1.json").exists():
                    (root / cp_key / "step1.json").write_text(json.dumps(step1))
                unit, value = part85._cell_value(spec["cell"], step1)
                answer_dir = root / cp_key / "answers"
                answer_dir.mkdir(parents=True, exist_ok=True)
                out = answer_dir / f"main-{spec['frame']}--{spec['cell']}.jsonl"
                start_sample = 0
                rows = []
                if cp_key == "pretrain_4T" and spec["frame"] == "plain" and out.exists():
                    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
                    start_sample = 1
                for qi, question in enumerate(part85.QUESTIONS):
                    for sample in range(start_sample, part85.SAMPLES_PER_QUESTION):
                        rows.append({"mode": "main", "attempt": 1, "checkpoint": cp_key,
                                     "model_id": cp["repo_id"], "commit": TEST_COMMIT,
                                     "frame": spec["frame"], "cell": spec["cell"], "unit": unit,
                                     "clamp_value": value, "frame_template_sha256": part85._template_sha256(part85.PLAIN_TEMPLATE),
                                     "pilot_gate_action": "plain_frame_pass", "question": question,
                                     "question_index": qi, "sample": sample, "seed": part85.seed_for(qi, sample),
                                     "answer": "Synthetic answer.", "new_tokens": 2})
                out.write_text("".join(json.dumps(row) + "\n" for row in rows))

            for cp in CHECKPOINTS:
                cp_root = root / cp.key
                cp_root.mkdir(parents=True, exist_ok=True)
                cosine = 0.95
                step0 = {"checkpoint": {"key": cp.key}, "measures": {
                    "cosines_with_instruct": {"down_proj_column_6978": cosine},
                    "direct_readout": {"z_score": 3.0 if cp.key == "pretrain_4T" else 0.0}}}
                step1 = {"checkpoint": {"key": cp.key}, "dataset_revision": step1_acts.WIKITEXT_REVISION,
                         "neutral_correlations_with_instruct": {"target": 0.95},
                         "passages": {"offworld_mean": 2.0, "ordinary_mean": 0.0, "ordinary_sd": 1.0}}
                (cp_root / "step0.json").write_text(json.dumps(step0))
                (cp_root / "step1.json").write_text(json.dumps(step1))

            main_items = Path(temp) / "main-items.jsonl"
            main_key_path = Path(temp) / "main-key.json"
            main_score_path = Path(temp) / "main-score.json"
            score85.main(["prepare", "--mode", "main", "--results-root", str(root),
                          "--items", str(main_items), "--key", str(main_key_path)])
            main_key = json.loads(main_key_path.read_text(encoding="utf-8"))
            labels = [{"id": item_id, "origin": "NEITHER", "form": "OK"}
                      for item_id in main_key["items"]]
            labels_path = Path(temp) / "reader-1.jsonl"
            labels2_path = Path(temp) / "reader-2.jsonl"
            for path in (labels_path, labels2_path):
                path.write_text("".join(json.dumps(row) + "\n" for row in labels))
            score85.main(["score", "--mode", "main", "--key", str(main_key_path), "--items", str(main_items),
                          "--labels", str(labels_path), "--labels2", str(labels2_path),
                          "--results-root", str(root), "--out", str(main_score_path)])
            main_score = json.loads(main_score_path.read_text(encoding="utf-8"))
            self.assertEqual(len(main_score["counts"]), 60)
            self.assertEqual(main_score["verdicts"]["85.9"]["verdict"], "HOLDS")
            self.assertEqual(main_score["scope"], "plain_frame_pass")


if __name__ == "__main__":
    unittest.main(verbosity=2)
