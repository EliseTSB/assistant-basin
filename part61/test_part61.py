"""CPU-only Part 61 contract tests, including source mutations and a stub model."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PART61 = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PART61))

import analyze61
import run61

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from torch import nn
    import part26 as p26
else:
    torch = None
    nn = None
    p26 = None


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    is_fast = True

    def __call__(self, _text, **_kwargs):
        return {"input_ids": [1], "offset_mapping": [(0, 1)]}


if TORCH_AVAILABLE:
    class _FakeDownProj(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.in_features = width
            self.weight = nn.Parameter(torch.ones((3, width), dtype=torch.float32))

        def forward(self, values):
            return torch.zeros_like(values[..., :3])

    class _FakeMlp(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.down_proj = _FakeDownProj(width)

    class _FakeLayer(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.width = width
            self.mlp = _FakeMlp(width)

        def forward(self, values):
            intermediate = values[..., :1].expand(*values.shape[:-1], self.width)
            return values + self.mlp.down_proj(intermediate)

    class _FakeDecoder(nn.Module):
        def __init__(self, width: int, depth: int):
            super().__init__()
            self.layers = nn.ModuleList([_FakeLayer(width) for _ in range(depth)])

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            del attention_mask, use_cache
            values = input_ids.float().unsqueeze(-1).expand(-1, -1, 3)
            for layer in self.layers:
                values = layer(values)
            return SimpleNamespace(last_hidden_state=values)

    class _FakeModel(nn.Module):
        def __init__(self, width: int, depth: int):
            super().__init__()
            self.model = _FakeDecoder(width, depth)

class _NumpyWeight:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.ndim = self.values.ndim
        self.shape = self.values.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _NumpyProjection:
    def __init__(self, values):
        self.weight = _NumpyWeight(values)


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_fixture(folder: Path) -> dict[str, str]:
    source_json = folder / "source.json"
    source_npz = folder / "source.npz"
    text = " ".join(f"word{index}" for index in range(p26.SOURCE_WORDS))
    data = {"model": p26.SOURCE_MODEL, "revision": p26.SOURCE_REVISION,
            "input_shas": {"corpus": "fixture-corpus-sha"},
            "documents": [{"text": text, "line_index": index}
                          for index in range(p26.SOURCE_DOCS)]}
    source_json.write_text(json.dumps(data), encoding="utf-8")
    source_npz.write_bytes(b"Part 26 source npz fixture")
    qwen_maxima = folder / "qwen-maxima.json"
    qwen_data = {
        "inputs": {"model": p26.SOURCE_MODEL, "revision": p26.SOURCE_REVISION,
                   "layer": run61.MODEL_SPECS["qwen35-4b"]["layer"],
                   "corpus_sha256": "fixture-corpus-sha",
                   "source_json_sha256": _sha(source_json),
                   "source_npz_sha256": _sha(source_npz), "units": [2742]},
        "units": [2742], "maxima": {"2742": 19.125},
    }
    qwen_maxima.write_text(json.dumps(qwen_data), encoding="utf-8")
    olmo_npz = folder / "olmo-target.npz"
    widths = np.zeros(12, dtype=np.int32)
    widths[10] = run61.MODEL_SPECS["olmo2-1b"]["width"]
    offsets = np.zeros(12, dtype=np.int64)
    maxima = np.full(int(widths[10]), 0.1, dtype=np.float32)
    maxima[6978] = 6.5
    np.savez_compressed(olmo_npz, layer_widths=widths, layer_offsets=offsets,
                        max_activation=maxima)
    olmo_json = folder / "olmo-target.json"
    olmo_spec = p26.MODELS["olmo2-1b"]
    olmo_json.write_text(json.dumps({
        "model": olmo_spec["id"], "revision": olmo_spec["revision"],
        "short_name": "olmo2-1b",
        "twin_candidates_n2742": [{"layer": 10, "unit": 6978}],
    }), encoding="utf-8")
    return {"source_json": str(source_json), "source_npz": str(source_npz),
            "qwen_maxima": str(qwen_maxima), "olmo_target_npz": str(olmo_npz),
            "olmo_target_json": str(olmo_json)}


def _run_args(short: str, paths: dict[str, str], output: Path):
    spec = run61.MODEL_SPECS[short]
    return SimpleNamespace(
        model=short, revision=None, clamp_unit=f"{spec['layer']}:{spec['unit']}",
        device="cpu", batch_size=run61.DOCUMENT_COUNT, out=str(output), force=False,
        **paths,
    )


def _stub_loader(model_id, _revision, _device, dtype):
    if dtype != "fp32":
        raise AssertionError(f"stub loader received unexpected dtype {dtype}")
    short = next(key for key, spec in p26.MODELS.items()
                 if key in run61.MODEL_SPECS and spec["id"] == model_id)
    model_spec = run61.MODEL_SPECS[short]
    return _FakeTokenizer(), _FakeModel(model_spec["width"], model_spec["layer"] + 2)


def _write_runner_fixture(root: Path) -> None:
    paths = (
        "part61/run61.py", "part61/analyze61.py", "part61/run61.sh",
        "part61/RUNBOOK-61.md", "part61/test_part61.py", "part26.py", "part20.py",
        "part20c.py", "part20d.py", "part20f.py", "neuron_report.py",
        "results/part26/source.json", "results/part26/source.npz",
        "results/part26/target-olmo2-1b.npz", "results/part26/target-olmo2-1b.json",
        "results/part54a/source-pile-maxima-7c19d0120476.json",
    )
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stub input\n", encoding="utf-8")


def _run_mutation_red(testcase: unittest.TestCase, source_path: Path, filename: str,
                      needle: str, replacement: str, contract: str, label: str) -> None:
    source = source_path.read_text(encoding="utf-8")
    testcase.assertEqual(source.count(needle), 1, f"{label}: mutation anchor must be unique")
    with tempfile.TemporaryDirectory(prefix="part61-sabotage-") as folder_text:
        folder = Path(folder_text)
        mutated = folder / filename
        mutated.write_text(source.replace(needle, replacement, 1), encoding="utf-8")
        child = f"""
import importlib.util
import numpy as np
path = {str(mutated)!r}
spec = importlib.util.spec_from_file_location('mutated_part61', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
{contract}
print('CONTRACT_OK')
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run([sys.executable, "-c", child], text=True,
                                capture_output=True, timeout=30, env=env)
        testcase.assertNotEqual(result.returncode, 0,
                                f"{label} sabotage unexpectedly passed its real-function contract")
        testcase.assertTrue(("AssertionError" in result.stderr or "TypeError" in result.stderr),
                            f"{label} did not fail through its contract:\n{result.stderr}\n{result.stdout}")
        print(f"sabotage {label}: mutated source function went red")


class Part61Tests(unittest.TestCase):
    def test_registered_constants_and_controls_are_pinned(self):
        self.assertEqual(run61.WRITE_SIZES, (0.25, 0.5, 1.0, 2.0))
        self.assertEqual(run61.SIGNS, (-1.0, 1.0))
        self.assertEqual(run61.DOCUMENT_COUNT, 200)
        self.assertEqual(run61.CANDIDATE_POOL_SIZE, 20)
        self.assertEqual(run61.PRIOR_CONTROL_COUNT, 3)
        self.assertEqual(run61.CONTROL_COUNT, 10)
        self.assertEqual(run61.CANDIDATE_POOL_SEED, 0)
        self.assertEqual(run61.CONTROL_DRAW_SEED, 61)
        self.assertEqual(run61.CONTROL_PERCENTILE, 99.0)
        self.assertEqual(run61.NULL_THRESHOLD, 0.00005)
        self.assertEqual(run61.DEFAULT_BATCH_SIZE, 1)
        self.assertEqual(run61.MODEL_SPECS["qwen35-4b"], {"layer": 22, "unit": 2742, "width": 9216})
        self.assertEqual(run61.MODEL_SPECS["olmo2-1b"], {"layer": 10, "unit": 6978, "width": 8192})
        self.assertEqual(analyze61.WRITE_SIZES, (0.25, 0.5, 1.0, 2.0))
        self.assertEqual(analyze61.SIGNS, (-1.0, 1.0))
        self.assertEqual(analyze61.CONTROL_COUNT, 10)
        self.assertEqual(analyze61.CONTROL_PERCENTILE, 99.0)
        self.assertEqual(analyze61.NULL_THRESHOLD, 0.00005)
        self.assertEqual(analyze61.SPEARMAN_MINIMUM, 0.7)
        self.assertEqual(analyze61.RESIDUAL_RATIO_MINIMUM, 0.9)
        self.assertEqual(analyze61.RESIDUAL_RATIO_MAXIMUM, 1.1)

    def test_control_draw_reuses_original_pool_and_excludes_prior_three(self):
        qwen_pool, qwen_old, qwen_new = run61.select_control_units(9216, 2742)
        olmo_pool, olmo_old, olmo_new = run61.select_control_units(8192, 6978)
        self.assertEqual(qwen_pool, [152, 376, 692, 1613, 2481, 2833, 4638, 4702, 5009, 5160,
                                     5587, 5825, 5859, 5979, 6720, 7487, 7823, 8404, 8617, 8941])
        self.assertEqual(qwen_old, [152, 376, 692])
        self.assertEqual(qwen_new, [2833, 4638, 5009, 5587, 5825, 5979, 7487, 7823, 8404, 8941])
        self.assertEqual(olmo_pool, [135, 335, 615, 1433, 2205, 2516, 4121, 4178, 4451, 4585,
                                     4965, 5177, 5205, 5313, 5972, 6653, 6951, 7470, 7660, 7947])
        self.assertEqual(olmo_old, [135, 335, 615])
        self.assertEqual(olmo_new, [2516, 4121, 4451, 4965, 5177, 5313, 6653, 6951, 7470, 7947])
        self.assertFalse(set(qwen_old) & set(qwen_new))
        self.assertFalse(set(olmo_old) & set(olmo_new))

    def test_clamp_value_uses_each_units_actual_down_column_norm(self):
        projection = _NumpyProjection([[3.0, 0.0, 0.0, 1.0, 0.0],
                                       [4.0, 0.0, 0.0, 2.0, 2.0],
                                       [0.0, 5.0, 0.0, 2.0, 0.0]])
        norms = run61.down_column_norms(projection, [0, 1, 3])
        self.assertEqual(norms[0], 5.0)
        self.assertEqual(norms[1], 5.0)
        self.assertEqual(norms[3], 3.0)
        clamp_value = run61.matched_clamp_value(2.0, norms[0], norms[3], 0.25, 1.0)
        self.assertAlmostEqual(clamp_value, (0.25 * 2.0 * 5.0) / 3.0)
        self.assertAlmostEqual(clamp_value * norms[3], 0.25 * 2.0 * norms[0])
        self.assertAlmostEqual(run61.matched_clamp_value(2.0, norms[0], norms[0], 1.0, -1.0), -2.0)

    def test_pooled_symmetric_classifier_and_null_floor(self):
        changes = np.zeros((11, 2, 4), dtype=np.float64)
        changes[0, 0] = [10.0, -10.0, 0.00002, 5.0]
        changes[0, 1] = [11.0, -11.0, 0.00002, -5.0]
        changes[1:, :, :] = 0.1
        result = analyze61.classify_size_changes(changes)
        self.assertEqual(result["symmetric_count"], 2)
        self.assertEqual(result["threshold_minus"], 0.1)
        self.assertEqual(result["threshold_plus"], 0.1)
        self.assertEqual(len(result["control_symmetric_counts"]), 10)

        tiny = np.zeros((3, 2, 1), dtype=np.float64)
        tiny[0, :, 0] = 0.00004
        tiny_result = analyze61.classify_size_changes(tiny)
        self.assertEqual(tiny_result["threshold_minus"], 0.00005)
        self.assertEqual(tiny_result["symmetric_count"], 0)

    def test_spearman_uses_average_ranks_for_ties(self):
        self.assertAlmostEqual(analyze61.spearman_rho([1, 2, 3, 4], [2, 4, 6, 8]), 1.0)
        self.assertAlmostEqual(analyze61.spearman_rho([1, 2, 3, 4], [8, 6, 4, 2]), -1.0)
        tied = analyze61.spearman_rho([1, 1, 2, 3], [1, 2, 3, 4])
        self.assertAlmostEqual(tied, 0.9486832980505138)

    def test_registered_count_and_residual_evaluation(self):
        changes = np.zeros((11, 4, 2, 2), dtype=np.float64)
        ratios = np.ones((11, 4, 2), dtype=np.float64)
        ratios[0, 2] = [0.8, 1.2]
        result = analyze61.analyze_arrays(changes, analyze61.WRITE_SIZES, ratios, list(range(11)))
        matched = result["matched_1x"]
        self.assertEqual(matched["target_count"], 0)
        self.assertTrue(matched["target_fails_to_beat_at_least_one_control"])
        self.assertTrue(matched["target_within_control_range"])
        self.assertEqual(matched["residual_norm_ratio_mean_by_unit"][0], 1.0)
        self.assertEqual(matched["residual_ratios_within_10_percent"], 11)
        self.assertEqual(matched["residual_ratios_total"], 11)
        self.assertTrue(matched["residual_within_10_percent_for_every_unit"])

    def test_registered_cross_model_decision_requires_both_ranges(self):
        qwen = {"target_within_control_range": True}
        olmo = {"target_fails_to_beat_at_least_one_control": True,
                "target_within_control_range": True, "target_above_all_controls": False}
        self.assertEqual(analyze61.registered_decision(qwen, olmo), "push-size artefact supported")
        olmo["target_within_control_range"] = False
        self.assertEqual(analyze61.registered_decision(qwen, olmo), "registered decision rule is inconclusive")
        olmo["target_above_all_controls"] = True
        self.assertEqual(analyze61.registered_decision(qwen, olmo),
                         "59B contrast survives: OLMo target beats all ten matched controls")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is absent; stub forward integration needs CPU PyTorch")
    def test_stub_model_writer_reader_archive_and_resume_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="part61-stub-run-") as folder_text:
            folder = Path(folder_text)
            paths = _source_fixture(folder)
            output = folder / "results" / "part61"
            output.mkdir(parents=True)
            qwen_args = _run_args("qwen35-4b", paths, output)
            with patch.object(run61.p26, "_load_model_offline", side_effect=_stub_loader):
                qwen_manifest = run61.run(qwen_args)
            qwen_npz = output / "61-qwen35-4b-L22u2742.npz"
            qwen_json = output / "61-qwen35-4b-L22u2742.json"
            self.assertEqual(qwen_manifest["status"], "complete")
            self.assertEqual(qwen_manifest["dtype"], "torch.float32")
            self.assertEqual(qwen_manifest["documents"], run61.DOCUMENT_COUNT)
            self.assertEqual(len(qwen_manifest["control_units"]), run61.CONTROL_COUNT)
            for unit in qwen_manifest["condition_units"]:
                unit_norm = qwen_manifest["w_down_column_norms"][str(unit)]
                for size in run61.WRITE_SIZES:
                    for sign in run61.SIGNS:
                        stored = qwen_manifest["clamp_values"][str(unit)][f"{size:g}x"][str(int(sign))]
                        expected = (sign * size * qwen_manifest["target_pile_maximum"]
                                    * qwen_manifest["w_down_column_norms"][str(qwen_manifest["twin"]["unit"])]
                                    / unit_norm)
                        self.assertAlmostEqual(stored, expected)
            with np.load(qwen_npz) as archive:
                self.assertEqual(set(archive.files), {
                    "baseline_mean", "condition_mean", "change", "baseline_residual_norm_by_doc",
                    "residual_norm_ratio", "neuron_layer", "neuron_unit", "condition_unit",
                    "write_sizes", "signs"})
                saved_units = np.asarray(archive["condition_unit"]).tolist()
                self.assertEqual(saved_units, qwen_manifest["condition_units"])
                state = {key: np.asarray(archive[key]).copy() for key in archive.files}
            missing_step = run61._step_key(qwen_manifest["condition_units"][1], run61.WRITE_SIZES[0])
            qwen_manifest["status"] = "incomplete"
            qwen_manifest["completed"].remove(missing_step)
            qwen_json.write_text(json.dumps(qwen_manifest, indent=2) + "\n", encoding="utf-8")
            state["condition_mean"][1, 0, :, :] = np.nan
            state["residual_norm_ratio"][1, 0, :] = np.nan
            state["change"] = state["condition_mean"] - state["baseline_mean"][None, None, None, :]
            with qwen_npz.open("wb") as stream:
                np.savez_compressed(stream, **state)
            with patch.object(run61.p26, "_load_model_offline", side_effect=_stub_loader):
                resumed = run61.run(qwen_args)
            self.assertEqual(resumed["status"], "complete")
            self.assertIn(missing_step, resumed["completed"])
            qwen_analysis = analyze61.analyze_file(qwen_npz)
            self.assertEqual(qwen_analysis["control_units"], resumed["control_units"])
            self.assertEqual(len(qwen_analysis["symmetric_counts_by_unit_and_size"]), 11)
            self.assertEqual(qwen_analysis["matched_1x"]["residual_ratios_total"], 11)

            olmo_args = _run_args("olmo2-1b", paths, output)
            with patch.object(run61.p26, "_load_model_offline", side_effect=_stub_loader):
                olmo_manifest = run61.run(olmo_args)
            olmo_npz = output / "61-olmo2-1b-L10u6978.npz"
            olmo_analysis = analyze61.analyze_file(olmo_npz)
            self.assertEqual(olmo_analysis["short_name"], "olmo2-1b")
            self.assertEqual(olmo_manifest["target_pile_maximum"], 6.5)
            paired = analyze61.analyze_pair(qwen_npz, olmo_npz)
            self.assertEqual(set(paired["predictions"]), {"61a", "61b", "61c", "61d", "61e"})
            self.assertEqual(paired["registered_decision"], "push-size artefact supported")

    def test_runner_emits_ready_only_after_successful_archive(self):
        with tempfile.TemporaryDirectory(prefix="part61-runner-success-") as folder_text:
            folder = Path(folder_text)
            root = folder / "remote"
            _write_runner_fixture(root)
            return_dir = folder / "return"
            bin_dir = folder / "bin"
            bin_dir.mkdir()
            stub_python = bin_dir / "python"
            stub_python.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$*\" >> \"$PART61_STUB_LOG\"\n"
                "if [ -n \"${PART61_STUB_FAIL_MATCH:-}\" ]; then\n"
                "  case \"$*\" in *\"$PART61_STUB_FAIL_MATCH\"*) exit 23 ;; esac\n"
                "fi\nexit 0\n", encoding="utf-8")
            stub_python.chmod(0o755)
            stub_smi = bin_dir / "nvidia-smi"
            stub_smi.write_text("#!/bin/sh\necho stub-gpu\n", encoding="utf-8")
            stub_smi.chmod(0o755)
            log = folder / "commands.log"
            env = os.environ.copy()
            env.update({"PATH": str(bin_dir) + os.pathsep + env.get("PATH", ""),
                        "PART61_ROOT": str(root), "PART61_RETURN_DIR": str(return_dir),
                        "PART61_STUB_LOG": str(log)})
            result = subprocess.run(["bash", str(PART61 / "run61.sh")], text=True,
                                    capture_output=True, timeout=60, env=env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("RETURN_READY", result.stdout)
            self.assertTrue((return_dir / "part61-return.tgz").is_file())
            self.assertTrue(any("--combined" in line for line in log.read_text().splitlines()))

    def test_runner_failure_archives_incomplete_and_never_prints_ready(self):
        with tempfile.TemporaryDirectory(prefix="part61-runner-failure-") as folder_text:
            folder = Path(folder_text)
            root = folder / "remote"
            _write_runner_fixture(root)
            return_dir = folder / "return"
            bin_dir = folder / "bin"
            bin_dir.mkdir()
            stub_python = bin_dir / "python"
            stub_python.write_text(
                "#!/bin/sh\n"
                "if [ -n \"${PART61_STUB_FAIL_MATCH:-}\" ]; then\n"
                "  case \"$*\" in *\"$PART61_STUB_FAIL_MATCH\"*) exit 23 ;; esac\n"
                "fi\nexit 0\n", encoding="utf-8")
            stub_python.chmod(0o755)
            stub_smi = bin_dir / "nvidia-smi"
            stub_smi.write_text("#!/bin/sh\necho stub-gpu\n", encoding="utf-8")
            stub_smi.chmod(0o755)
            env = os.environ.copy()
            env.update({"PATH": str(bin_dir) + os.pathsep + env.get("PATH", ""),
                        "PART61_ROOT": str(root), "PART61_RETURN_DIR": str(return_dir),
                        "PART61_STUB_FAIL_MATCH": "part61/run61.py --model qwen35-4b"})
            result = subprocess.run(["bash", str(PART61 / "run61.sh")], text=True,
                                    capture_output=True, timeout=60, env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("RUN_FAILED", result.stdout)
            self.assertNotIn("RETURN_READY", result.stdout)
            self.assertTrue((return_dir / "part61-return-INCOMPLETE.tgz").is_file())

    def test_runner_preflight_failure_stops_work_and_saves_bundle(self):
        with tempfile.TemporaryDirectory(prefix="part61-runner-preflight-") as folder_text:
            folder = Path(folder_text)
            root = folder / "remote"
            _write_runner_fixture(root)
            (root / "results/part26/source.json").unlink()
            return_dir = folder / "return"
            bin_dir = folder / "bin"
            bin_dir.mkdir()
            stub_python = bin_dir / "python"
            stub_python.write_text("#!/bin/sh\necho called >> \"$PART61_STUB_LOG\"\nexit 0\n", encoding="utf-8")
            stub_python.chmod(0o755)
            log = folder / "commands.log"
            env = os.environ.copy()
            env.update({"PATH": str(bin_dir) + os.pathsep + env.get("PATH", ""),
                        "PART61_ROOT": str(root), "PART61_RETURN_DIR": str(return_dir),
                        "PART61_STUB_LOG": str(log)})
            result = subprocess.run(["bash", str(PART61 / "run61.sh")], text=True,
                                    capture_output=True, timeout=60, env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("preflight-files", result.stdout)
            self.assertNotIn("RETURN_READY", result.stdout)
            self.assertFalse(log.exists())
            self.assertTrue((return_dir / "part61-return-INCOMPLETE.tgz").is_file())

    def test_source_mutation_sabotages_go_red_on_real_functions(self):
        _run_mutation_red(
            self, PART61 / "analyze61.py", "analyze61.py",
            "same_nonzero_sign = (np.sign(minus) == np.sign(plus)) & (np.sign(minus) != 0)",
            "same_nonzero_sign = (np.sign(minus) != np.sign(plus)) & (np.sign(minus) != 0)",
            "changes = np.zeros((3, 2, 1)); changes[0, :, 0] = 0.1; result = module.classify_size_changes(changes); assert result['symmetric_count'] == 1",
            "equal-sign rule")
        _run_mutation_red(
            self, PART61 / "analyze61.py", "analyze61.py",
            "np.abs(control_minus).ravel(), percentile",
            "np.abs(control_minus), percentile, axis=0",
            "changes = np.zeros((3, 2, 2)); changes[1:, :, :] = 0.1; result = module.classify_size_changes(changes); assert np.isclose(result['threshold_minus'], 0.1)",
            "pooled percentile")
        _run_mutation_red(
            self, PART61 / "analyze61.py", "analyze61.py",
            "reference_indices = [other for other in range(len(control_minus)) if other != index]",
            "reference_indices = list(range(len(control_minus)))",
            "changes = np.zeros((4, 2, 2)); changes[1, :, :] = [100.0, 0.1]; changes[2:, :, :] = 0.1; result = module.classify_size_changes(changes); assert np.isclose(result['control_thresholds'][0]['minus'], 0.1)",
            "leave-one-out pool")
        _run_mutation_red(
            self, PART61 / "analyze61.py", "analyze61.py",
            "threshold_minus = max(float(np.percentile(np.abs(control_minus).ravel(), percentile)), NULL_THRESHOLD)",
            "threshold_minus = float(np.percentile(np.abs(control_minus).ravel(), percentile))",
            "changes = np.zeros((3, 2, 1)); result = module.classify_size_changes(changes); assert result['threshold_minus'] == 0.00005",
            "null floor")
        _run_mutation_red(
            self, PART61 / "run61.py", "run61.py",
            "target_pile_maximum * target_column_norm / unit_column_norm",
            "target_pile_maximum * target_column_norm / target_column_norm",
            "actual = module.matched_clamp_value(2.0, 5.0, 3.0, 0.25); assert np.isclose(actual, 0.25 * 2.0 * 5.0 / 3.0)",
            "per-unit W_down scaling")


if __name__ == "__main__":
    unittest.main(verbosity=2)
