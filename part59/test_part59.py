"""CPU tests for Part 59 contracts; no model downloads or GPU are needed."""
from __future__ import annotations

import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze59b
import blind_origin
import blind_setting
import run59a
import run59b


def _install_sabotage() -> None:
    mode = os.environ.get("PART59_SABOTAGE")
    if not mode:
        return
    original = analyze59b.classify_changes

    def mutated(changes, percentile=99.0):
        result = original(changes, percentile)
        array = np.asarray(changes, dtype=np.float64)
        if mode == "flip-sign":
            wrong_sign = (np.sign(array[0, 0]) * np.sign(array[0, 1])) < 0
            result["symmetric_mask"] = wrong_sign
        elif mode == "drop-control-percentile":
            same_sign = (np.sign(array[0, 0]) == np.sign(array[0, 1])) & (np.sign(array[0, 0]) != 0)
            result["symmetric_mask"] = same_sign
        elif mode == "per-neuron-percentile":
            same_nonzero_sign = ((np.sign(array[0, 0]) == np.sign(array[0, 1]))
                                 & (np.sign(array[0, 0]) != 0))
            minus_bar = np.percentile(np.abs(array[1:, 0]), 99, axis=0)
            plus_bar = np.percentile(np.abs(array[1:, 1]), 99, axis=0)
            result["symmetric_mask"] = (same_nonzero_sign
                                         & (np.abs(array[0, 0]) > minus_bar)
                                         & (np.abs(array[0, 1]) > plus_bar))
        else:
            raise ValueError(f"unknown PART59_SABOTAGE {mode!r}")
        result["symmetric_count"] = int(result["symmetric_mask"].sum())
        return result

    analyze59b.classify_changes = mutated


def _kill_lock_holder(module_name: str, lock_path: Path, phase: str) -> None:
    module_dir = str(Path(__file__).resolve().parent)
    code = (
        "import sys,time; "
        f"sys.path.insert(0, {module_dir!r}); "
        f"import {module_name}; "
        f"lock={module_name}.advisory_lock(sys.argv[1], {phase!r}); "
        "lock.__enter__(); print('LOCKED', flush=True); time.sleep(60)"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [module_dir, str(Path(__file__).resolve().parent.parent), environment.get("PYTHONPATH", "")])
    child = subprocess.Popen([sys.executable, "-u", "-c", code, str(lock_path)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             env=environment)
    ready, _, _ = select.select([child.stdout], [], [], 30)
    if not ready or child.stdout.readline().strip() != "LOCKED":
        child.kill()
        _, stderr = child.communicate(timeout=10)
        raise AssertionError(f"lock holder failed to start: {stderr}")
    os.kill(child.pid, signal.SIGKILL)
    return_code = child.wait(timeout=10)
    child.stdout.close()
    child.stderr.close()
    if return_code != -signal.SIGKILL:
        raise AssertionError(f"expected SIGKILL exit code, got {return_code}")


_install_sabotage()


class _FakeBatch:
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def to(self, _device):
        return self


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    is_fast = True

    def __call__(self, text, **kwargs):
        ids = [int(value) for value in text] if isinstance(text, list) else [1, 2]
        if kwargs.get("return_tensors") == "pt":
            return _FakeBatch(torch.tensor([ids], dtype=torch.long))
        return {"input_ids": ids, "offset_mapping": [(i, i + 1) for i in range(len(ids))]}


class _ToyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(2, 3, bias=False)
        self.down_proj = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.up_proj.weight.fill_(1.0)
            self.down_proj.weight.fill_(1.0)

    def forward(self, values):
        return self.down_proj(torch.relu(self.up_proj(values)))


class _ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _ToyMlp()

    def forward(self, values):
        return values + self.mlp(values)


class _ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 2)
        self.layers = nn.ModuleList([_ToyLayer(), _ToyLayer()])
        with torch.no_grad():
            self.embed.weight.fill_(1.0)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        values = self.embed(input_ids)
        for layer in self.layers:
            values = layer(values)
        return SimpleNamespace(last_hidden_state=values)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _ToyDecoder()


def _fake_output(step, **_kwargs):
    conditions = [{"condition": name, "answers": [f"answer-{i}" for i in range(8)]}
                  for name in step["conditions"]]
    return {"question": step["question"], "clamp_mult": step["dose"], "conditions": conditions}


def _write_complete_59a_fixture(folder: Path, prompt_tag: str = "current") -> None:
    for short, layer, unit, controls in (
            ("qwen35-4b", 22, 2742, [152, 376, 692]),
            ("olmo2-1b", 10, 6978, [135, 335, 615])):
        prompts = [{"id": f"p{index:02d}", "subject": f"subject-{prompt_tag}-{index}",
                    "question": f"Write a paragraph about {prompt_tag} subject {index}."}
                   for index in range(1, 13)]
        steps = []
        for dose in (-2.0, -1.0, 0.0, 1.0, 2.0):
            conditions = (["unclamped"] if dose == 0 else ["twin"])
            if abs(dose) == 2:
                conditions += [f"control-{control}" for control in controls]
            for prompt in prompts:
                filename = f"clamp-{short}-L{layer}u{unit}-{prompt['id']}-{'m' if dose < 0 else 'p' if dose > 0 else 'z'}{abs(int(dose))}.json"
                steps.append({"prompt_id": prompt["id"], "question": prompt["question"],
                              "dose": dose, "path": f"/root/part59/results/part59a/{filename}",
                              "conditions": conditions})
                local_path = folder / filename
                local_path.write_text(json.dumps({
                    "short_name": short, "question": prompt["question"], "clamp_mult": dose,
                    "conditions": [{"condition": condition,
                                    "answers": [f"{prompt_tag} answer {i}" for i in range(8)]}
                                   for condition in conditions],
                }), encoding="utf-8")
        manifest = {"short_name": short, "status": "complete", "control_units": controls,
                    "samples_per_prompt_dose": 8, "prompts": prompts, "steps": steps,
                    "completed": [step["path"] for step in steps],
                    "expected_generations": 1056}
        (folder / f"ladder59a-{short}-L{layer}u{unit}.json").write_text(
            json.dumps(manifest), encoding="utf-8")


class Part59Tests(unittest.TestCase):
    def test_prompt_set_has_twelve_neutral_subjects(self):
        rows = run59a.prompts()
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row["subject"] for row in rows}), 12)
        banned = {"mars", "earth", "starship", "phone", "computer", "robot", "internet", "spaceship"}
        prompt_words = set(" ".join(row["question"].lower() for row in rows).split())
        self.assertFalse(banned & prompt_words)

    def test_dose_scaling_uses_each_units_part26_pile_maximum(self):
        step = {"question": "Write the first paragraph of a story about a baker.",
                "prompt_id": "p01", "dose": -2.0,
                "conditions": ["twin", "control-11", "control-22", "control-33"]}
        spec = {"layer": 22, "unit": 2742}
        controls = [11, 22, 33]
        maxima = {2742: 19.125, 11: 2.0, 22: 3.0, 33: 4.0}
        captured = []

        def record(_tok, _model, _down, _ids, condition, unit, value, seed, samples,
                   max_new, device, max_gpu_gb):
            captured.append((condition, unit, value, samples, max_new))
            return {"condition": condition, "unit": unit, "set_value": value,
                    "seed": seed, "n": samples, "answers": ["x"] * samples}

        args = SimpleNamespace(seed=0, device="cpu", max_gpu_gb=22.0,
                               prompts=str(run59a.PROMPTS_PATH),
                               qwen_maxima=str(run59a.PART54_MAXIMA))
        with patch.object(run59a.nr, "build_prompt", return_value="prompt"), \
                patch.object(run59a.p26, "_generate_condition", side_effect=record), \
                patch.object(run59a.p26, "_header", return_value={}):
            result = run59a._generate_step(_FakeTokenizer(), object(), object(), step, spec,
                                           controls, maxima, "model", "revision", "qwen35-4b", args)
        self.assertEqual([row[2] for row in captured], [-38.25, -4.0, -6.0, -8.0])
        self.assertTrue(all(row[3:] == (8, 60) for row in captured))
        self.assertEqual(result["clamp_mult"], -2.0)

    def test_pooled_p99_threshold_opposite_sign_and_leave_one_out_control(self):
        pooled = np.zeros(300, dtype=np.float64)
        pooled[296:] = [1.0, 101.0, 200.0, 300.0]
        controls = pooled.reshape(3, 100)
        changes = np.zeros((4, 2, 100), dtype=np.float64)
        changes[1:, 0] = controls
        changes[1:, 1] = controls
        changes[0, 0, :4] = [2.01, 1.99, 2.01, -2.01]
        changes[0, 1, :4] = [2.02, 1.99, -2.01, -2.02]
        result = analyze59b.classify_changes(changes)
        self.assertAlmostEqual(result["threshold_minus"], 2.0)
        self.assertAlmostEqual(result["threshold_plus"], 2.0)
        self.assertEqual(result["symmetric_count"], 2)
        self.assertEqual(result["odd_count"], 1)
        self.assertEqual(result["symmetric_mask"][:4].tolist(), [True, False, False, True])

        # Control zero has one genuine two-dose response. Its threshold excludes itself.
        changes = np.zeros((4, 2, 20), dtype=np.float64)
        changes[1, :, 0] = 1.0
        result = analyze59b.classify_changes(changes)
        self.assertEqual(result["control_symmetric_counts"], [1, 0, 0])
        self.assertEqual(result["max_control_symmetric_count"], 1)
        self.assertAlmostEqual(result["threshold_minus"], 0.41)
        self.assertAlmostEqual(result["threshold_plus"], 0.41)

    def test_control_draws_match_part26_values(self):
        self.assertEqual(run59a.control_units(9216, 2742, 0), [152, 376, 692])
        self.assertEqual(run59a.control_units(8192, 6978, 0), [135, 335, 615])

    def test_59b_toy_model_hooks_capture_later_neuron_means_and_clamp(self):
        model = _ToyModel().eval()
        _, layers, _ = run59b.nr.find_text_layers(model)
        tok = _FakeTokenizer()
        tokenized = [{"ids": [1, 2]}, {"ids": [3]}]
        baseline, layers_recorded, units_recorded, stats = run59b.capture_means(
            tok, model, tokenized, layers, batch_size=1, device="cpu", start_layer=1)
        written_values = []
        base_hook = run59b.nr.NeuronHook

        class RecordingHook(base_hook):
            def __init__(self, down_proj):
                super().__init__(down_proj)
                self.record_handle = down_proj.register_forward_pre_hook(self.record_written)

            def record_written(self, _module, args):
                written_values.append(args[0][..., 0].detach().clone())

            def remove(self):
                self.record_handle.remove()
                super().remove()

        with patch.object(run59b.nr, "NeuronHook", RecordingHook):
            clamped, _, _, _ = run59b.capture_means(
                tok, model, tokenized, layers, batch_size=1, device="cpu", clamp_layer=0,
                unit=0, clamp_value=4.0)
        self.assertEqual(layers_recorded.tolist(), [1, 1, 1])
        self.assertEqual(units_recorded.tolist(), [0, 1, 2])
        self.assertEqual(stats["documents"], 2)
        self.assertEqual(stats["forward_batches"], 2)
        self.assertTrue(np.isfinite(baseline).all())
        self.assertFalse(np.allclose(baseline, clamped))
        self.assertTrue(written_values)
        for values in written_values:
            self.assertTrue(torch.equal(values, torch.full_like(values, 4.0)))
        docs = [{"text": "aa bb", "line_index": 10}, {"text": "cc dd", "line_index": 11}]
        window_tokens = [{"ids": [1, 2], "offsets": [(0, 2), (3, 5)]},
                         {"ids": [3, 4], "offsets": [(0, 2), (3, 5)]}]
        windows = run59b.capture_top_windows(tok, model, docs, window_tokens, layers,
                                            {1: [1]}, batch_size=1, device="cpu")
        self.assertEqual(len(windows["1:1"]), 2)
        self.assertEqual({row["window"] for row in windows["1:1"]}, {"aa bb", "cc dd"})

    def test_59a_interrupted_run_resumes_and_preserves_completed_files(self):
        class FakeModel:
            def __init__(self):
                self.parameter = torch.nn.Parameter(torch.ones(1))

            def parameters(self):
                return iter([self.parameter])

        layers = [SimpleNamespace(mlp=SimpleNamespace(down_proj=SimpleNamespace(in_features=9216)))
                  for _ in range(32)]
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(model="qwen35-4b", revision=None, prompts=str(run59a.PROMPTS_PATH),
                                   device="cpu", seed=0, max_gpu_gb=22.0, out=directory,
                                   qwen_maxima=str(run59a.PART54_MAXIMA), force=False)
            calls = {"n": 0}

            def interrupt(_tok, _model, _down_proj, step, *_args, **_kwargs):
                calls["n"] += 1
                if calls["n"] == 4:
                    raise RuntimeError("simulated worker interruption")
                return _fake_output(step)

            with patch.object(run59a.p26, "_load_model_offline", return_value=(_FakeTokenizer(), FakeModel())), \
                    patch.object(run59a.nr, "find_text_layers", return_value=("model.layers", layers, None)), \
                    patch.object(run59a, "_generate_step", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "simulated worker interruption"):
                    run59a.run(args)
            manifest_path = Path(directory) / "ladder59a-qwen35-4b-L22u2742.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "incomplete")
            prior_paths = [Path(step["path"]) for step in manifest["steps"][:3]]
            before = [path.read_bytes() for path in prior_paths]
            calls["n"] = 0
            with patch.object(run59a.p26, "_load_model_offline", return_value=(_FakeTokenizer(), FakeModel())), \
                    patch.object(run59a.nr, "find_text_layers", return_value=("model.layers", layers, None)), \
                    patch.object(run59a, "_generate_step",
                                 side_effect=lambda _tok, _model, _down, step,
                                 *_args, **_kwargs: _fake_output(step)):
                completed = run59a.run(args)
            self.assertEqual(completed["status"], "complete")
            self.assertEqual([path.read_bytes() for path in prior_paths], before)
            self.assertEqual(len([step for step in completed["steps"] if Path(step["path"]).is_file()]), 60)
            with patch.object(run59a.p26, "_load_model_offline", return_value=(_FakeTokenizer(), FakeModel())) as load_model, \
                    patch.object(run59a.nr, "find_text_layers", return_value=("model.layers", layers, None)):
                with self.assertRaisesRegex(FileExistsError, "already complete"):
                    run59a.run(args)
                load_model.assert_not_called()

    def test_59a_sigkill_releases_lock_and_rerun_proceeds(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / ".part59a.lock"
            _kill_lock_holder("run59a", lock, "59A")

            class FakeModel:
                def __init__(self):
                    self.parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))

                def parameters(self):
                    return iter([self.parameter])

            layers = [SimpleNamespace(mlp=SimpleNamespace(
                down_proj=SimpleNamespace(in_features=9216))) for _ in range(32)]
            args = SimpleNamespace(model="qwen35-4b", revision=None, prompts=str(run59a.PROMPTS_PATH),
                                   device="cpu", seed=0, max_gpu_gb=22.0, out=directory,
                                   qwen_maxima=str(run59a.PART54_MAXIMA), force=False)
            with patch.object(run59a.p26, "_load_model_offline", return_value=(_FakeTokenizer(), FakeModel())), \
                    patch.object(run59a.nr, "find_text_layers", return_value=("model.layers", layers, None)), \
                    patch.object(run59a, "load_maxima", return_value={2742: 19.125, 152: 1.0, 376: 2.0, 692: 3.0}), \
                    patch.object(run59a, "_input_shas", return_value={"test": "sha"}), \
                    patch.object(run59a, "_generate_step", side_effect=lambda _tok, _model, _down, step,
                                 *_args, **_kwargs: _fake_output(step)):
                completed = run59a.run(args)
            self.assertEqual(completed["status"], "complete")
            self.assertEqual(len(completed["completed"]), 60)

    def test_59b_sigkill_releases_lock_and_rerun_proceeds(self):
        with tempfile.TemporaryDirectory() as directory:
            _kill_lock_holder("run59b", Path(directory) / ".part59b.lock", "59B")

            class FakeModel:
                def __init__(self):
                    self.parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))

                def parameters(self):
                    return iter([self.parameter])

            layers = [SimpleNamespace(mlp=SimpleNamespace(
                down_proj=SimpleNamespace(in_features=8192 if index == 10 else 3)))
                      for index in range(12)]
            args = SimpleNamespace(model="olmo2-1b", revision=None, clamp_unit="10:6978",
                                   seed=0, out=directory, source_json="unused-source.json",
                                   qwen_maxima="unused-maxima.json", batch_size=1,
                                   device="cpu", force=False)
            context = [
                patch.object(run59b, "source_documents", return_value=[{"text": "test", "line_index": 0}]),
                patch.object(run59b, "tokenize_documents", return_value=[{"ids": [1], "offsets": [(0, 4)]}]),
                patch.object(run59b, "control_units", return_value=[135, 335, 615]),
                patch.object(run59b, "load_maxima", return_value={6978: 6.5, 135: 1.0, 335: 2.0, 615: 3.0}),
                patch.object(run59b, "_sha", return_value="fixed-sha"),
                patch.object(run59b.p26, "_load_model_offline", return_value=(_FakeTokenizer(), FakeModel())),
                patch.object(run59b.nr, "find_text_layers", return_value=("model.layers", layers, None)),
                patch.object(run59b, "capture_means", return_value=(
                    np.ones(3, dtype=np.float32), np.asarray([11, 11, 11]), np.asarray([0, 1, 2]),
                    {"forward_batches": 1, "documents": 1, "peak_cuda_allocated_bytes": 0,
                     "peak_cuda_reserved_bytes": 0})),
            ]
            with context[0], context[1], context[2], context[3], context[4], context[5], context[6], context[7]:
                completed = run59b.run(args)
            self.assertEqual(completed["status"], "complete")
            self.assertEqual(len(completed["completed"]), 9)

            # The completed identity check is also cheap: a rerun refuses before model loading.
            with patch.object(run59b, "source_documents", return_value=[{"text": "test", "line_index": 0}]), \
                    patch.object(run59b, "control_units", return_value=[135, 335, 615]), \
                    patch.object(run59b, "load_maxima", return_value={6978: 6.5, 135: 1.0, 335: 2.0, 615: 3.0}), \
                    patch.object(run59b, "_sha", return_value="fixed-sha"), \
                    patch.object(run59b.p26, "_load_model_offline") as load_model:
                with self.assertRaisesRegex(FileExistsError, "already complete"):
                    run59b.run(args)
                load_model.assert_not_called()

    def test_blind_setting_calibration_scores_two_errors_and_refuses_three(self):
        self.assertEqual(len(blind_setting.CAL), 34)
        for wrong_count in (2, 3):
            with self.subTest(wrong_count=wrong_count), tempfile.TemporaryDirectory() as directory:
                review = Path(directory)
                idmap = {}
                results = []
                for index, (expected_label, expected_quality, _text) in enumerate(blind_setting.CAL):
                    custom_id = f"r{index:05d}"
                    idmap[custom_id] = f"cal|{index}|{expected_label}|{expected_quality}"
                    label = expected_label
                    if index < wrong_count:
                        label = "SCIFI" if expected_label == "OTHER" else "OTHER"
                    results.append({"custom_id": custom_id, "result": {
                        "type": "succeeded", "message": {"content": [
                            {"type": "text", "text": f"{label} {expected_quality}"}]}}})
                (review / "setting-idmap-BID.json").write_text(json.dumps(idmap), encoding="utf-8")
                (review / "setting-key-BID.json").write_text("{}", encoding="utf-8")
                with patch.object(blind_setting, "call", return_value="\n".join(json.dumps(x) for x in results)):
                    if wrong_count == 2:
                        blind_setting.main(["collect", "BID", "--review-dir", directory])
                    else:
                        with self.assertRaisesRegex(SystemExit, "calibration failed"):
                            blind_setting.main(["collect", "BID", "--review-dir", directory])

    def test_blind_submit_requires_complete_manifest_files_and_rejects_strays(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            _write_complete_59a_fixture(folder)
            listed_rows = blind_setting.answers(folder)
            self.assertEqual(len(listed_rows), 2112)
            self.assertTrue(all("current answer" in row["text"] for row in listed_rows))

            stale = folder / "clamp-qwen35-4b-old-prompt.json"
            stale.write_text(json.dumps({"short_name": "qwen35-4b", "clamp_mult": 2.0,
                                         "question": "old prompt", "conditions": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unlisted clamp result files"):
                blind_setting.answers(folder)
            stale.unlink()

            manifest_path = folder / "ladder59a-olmo2-1b-L10u6978.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "incomplete"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "status is 'incomplete'"):
                blind_setting.answers(folder)

    def test_blind_submit_changed_prompt_manifest_does_not_mix_old_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            _write_complete_59a_fixture(folder, prompt_tag="new")
            stale = folder / "clamp-qwen35-4b-old-prompt.json"
            stale.write_text(json.dumps({"short_name": "qwen35-4b", "question": "old prompt",
                                         "clamp_mult": 2.0, "conditions": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unlisted clamp result files"):
                blind_setting.answers(folder)
            stale.unlink()
            rows = blind_setting.answers(folder)
            self.assertEqual(len(rows), 2112)
            self.assertTrue(all("new answer" in row["text"] for row in rows))

    def test_blind_origin_answers_reads_59a_condition_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clamp-qwen35-4b-L22u2742-x2-qabc123.json"
            path.write_text(json.dumps({
                "short_name": "qwen35-4b", "question": "Write the first paragraph of a story about a baker.",
                "clamp_mult": 2.0,
                "conditions": [{"condition": "twin", "unit": 2742,
                                "answers": ["A baker opens the door."]},
                               {"condition": "control-152", "unit": 152,
                                "answers": ["A baker starts work."]}],
            }), encoding="utf-8")
            rows = blind_origin.answers(directory)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["question"], "Write the first paragraph of a story about a baker.")
            self.assertEqual(rows[0]["model"], "qwen35-4b")
            self.assertEqual(rows[0]["dose"], 2.0)
            self.assertEqual({row["kind"] for row in rows}, {"twin", "control"})


if __name__ == "__main__":
    unittest.main()
