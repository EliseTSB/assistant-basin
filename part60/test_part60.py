#!/usr/bin/env python3
"""Offline Part 60 contract tests: fake Haiku batches, activation fixtures, and wrapper stubs."""
from __future__ import annotations

import json
import os
import inspect
import io
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze60
import gen60
import score60

REAL_PILOT = gen60.PILOT
REAL_BASIN = gen60.BASIN
REAL_UNITS_TEST = score60.DATA / "units.test.json"
_TEST_DATA_DIR = tempfile.TemporaryDirectory(prefix="part60-test-data-")
_TEST_BASIN_DIR = tempfile.TemporaryDirectory(prefix="part60-test-basin-")
_TEST_PILOT = Path(_TEST_DATA_DIR.name) / "pilot.jsonl"
with _TEST_PILOT.open("w", encoding="utf-8") as stream:
    for kind in ("unit", "control_id"):
        for index in range(500):
            row = {"layer": index % 4,
                   kind: 10000 + index if kind == "unit" else index,
                   "top": [{"window": f"top window {kind} {index} {i}"} for i in range(20)],
                   "random": [{"window": f"random window {kind} {index} {i}"} for i in range(10)],
                   "weak": [{"window": f"weak window {kind} {index} {i}"} for i in range(5)]}
            stream.write(json.dumps(row) + "\n")
gen60.PILOT = _TEST_PILOT
gen60.BASIN = Path(_TEST_BASIN_DIR.name)


def tearDownModule() -> None:
    gen60.PILOT = REAL_PILOT
    gen60.BASIN = REAL_BASIN
    _TEST_DATA_DIR.cleanup()
    _TEST_BASIN_DIR.cleanup()


def _tiered() -> dict[str, object]:
    return {"hypothesis": "one simple feature", "strong": ["a", "b", "c", "d"],
            "weak": ["e", "f", "g", "h"], "none": ["i", "j", "k", "l"]}


def _m6_response(*, thinking: str = "reason", stop_reason: str = "end_turn",
                 answer: dict[str, object] | None = None) -> str:
    return json.dumps({"content": [{"type": "thinking", "thinking": thinking},
                                    {"type": "text", "text": json.dumps(answer or _tiered())}],
                       "stop_reason": stop_reason,
                       "usage": {"input_tokens": 10, "output_tokens": 20}})


def _write_analyzer_fixture(root: Path) -> tuple[list[dict[str, object]], dict[str, float]]:
    units = [
        {"record_index": 10, "unit_key": "L0:u10", "type": "neuron", "layer": 0},
        {"record_index": 11, "unit_key": "L0:u11", "type": "neuron", "layer": 0},
    ]
    raw_values = {
        10: {"strong": [2., 2., 2., 0.], "weak": [2., 1.5, 0., 0.], "none": [0., 0., 0., 0.]},
        11: {"strong": [2., 0., 0., 0.], "weak": [0., 0., 0., 0.], "none": [0., 0., 0., 0.]},
    }
    output_rows = {}
    for unit in units:
        output_rows[int(unit["record_index"])] = {
            **unit, "parse_status": "ok", "raw_parse_status": "ok",
            "hypothesis": "synthetic fixture", "strong": [f"s{i}" for i in range(4)],
            "weak": [f"w{i}" for i in range(4)], "none": [f"n{i}" for i in range(4)],
        }
    gen60.write_output(gen60.output_path(root, "dev", "M1"), output_rows)
    entries = score60.entries_for_phase(list(output_rows.values()), "M1", "primary")
    matrix = np.zeros((len(entries), len(units)), dtype=np.float32)
    for row_index, entry in enumerate(entries):
        rec = int(entry["record_index"])
        matrix[row_index, rec - 10] = raw_values[rec][entry["group"]][int(entry["index"])]
    score60._save_npz_atomic(
        score60.score_path(root, "dev", "M1", "primary"),
        max=matrix, marked=np.full_like(matrix, np.nan),
        sids=np.asarray([x["sid"] for x in entries]),
        unit_keys=np.asarray([x["unit_key"] for x in units]),
        record_indices=np.asarray([x["record_index"] for x in entries], dtype=np.int32),
        groups=np.asarray([x["group"] for x in entries]),
        texts=np.asarray([x["text"] for x in entries]),
        method=np.asarray("M1"), phase=np.asarray("primary"),
        unit_types=np.asarray([x["type"] for x in units]),
    )
    return units, {"L0:u10": 1.0, "L0:u11": 1.0}


def _write_m3_winner_fixture(root: Path) -> None:
    key = "L0:u0"
    candidates = [{"rank": rank, "hypothesis": f"hypothesis {rank}",
                   "strong": [f"s{rank}-{i}" for i in range(3)],
                   "none": [f"n{rank}-{i}" for i in range(3)]} for rank in (1, 2, 3)]
    rows = {0: {"record_index": 0, "unit_key": key, "type": "neuron", "layer": 0,
                "parse_status": "ok", "selection": {"parse_status": "ok"}, "candidates": candidates}}
    gen60.write_output(gen60.output_path(root, "dev", "M3"), rows)
    root.mkdir(parents=True, exist_ok=True)
    (root / "thresholds.json").write_text(json.dumps({"thresholds": {key: 1.0}}))
    sids, values = [], []
    strong_by_rank = {1: [2., 2., 0.], 2: [2., 2., 2.], 3: [2., 0., 0.]}
    for rank in (1, 2, 3):
        for group in ("strong", "none"):
            for j in range(3):
                sids.append(f"0|M3|selection|candidate{rank}_{group}|{j}")
                values.append(strong_by_rank[rank][j] if group == "strong" else 0.0)
    score60._save_npz_atomic(root / "acts/dev/M3-selection.npz", max=np.asarray(values, dtype=np.float32).reshape(-1, 1),
                             sids=np.asarray(sids), unit_keys=np.asarray([key]))


class ThresholdTests(unittest.TestCase):
    def test_threshold_uses_other_owners_and_linear_p95(self) -> None:
        matrix = np.asarray([[100.0, 5.0], [0.0, 40.0], [2.0, 2.0], [4.0, 0.0]])
        got = score60.compute_thresholds(matrix, ["0:a", "1:a", "2:a", "1:b"], [0, 1], ["L0:u0", "L0:u1"], pool_size=4)
        self.assertAlmostEqual(got["L0:u0"], 3.8)
        self.assertAlmostEqual(got["L0:u1"], 4.85)

    def test_missing_archive_unit_threshold_uses_same_pool_with_toy_scorer(self) -> None:
        class ToyScorer:
            dtype = "fp32"
            def score_batch(self, items, rows, max_out, marked_out, batch_size=8):
                for row, item in zip(rows, items):
                    max_out[row, 0] = float(item["text"])
        original_data, original_size = score60.DATA, score60.POOL_SIZE
        with tempfile.TemporaryDirectory(prefix="part60-null-") as directory:
            data_dir = Path(directory) / "data"
            data_dir.mkdir()
            sent = [{"sid": f"{owner}:B#{j}", "text": str(value)} for owner, j, value in
                    ((0, 0, 100), (1, 0, 0), (2, 0, 2), (1, 1, 4))]
            (data_dir / "sent.all.jsonl").write_text("".join(json.dumps(x)+"\n" for x in sent))
            root = Path(directory) / "out"
            root.mkdir()
            (root / "thresholds.json").write_text(json.dumps({"thresholds": {}, "source_fingerprint": {}}))
            score60.DATA, score60.POOL_SIZE = data_dir, 4
            try:
                unit = {"record_index": 0, "unit_key": "L0:u0", "type": "neuron", "layer": 0}
                got = score60.extend_thresholds_for_split(root, "dev", [unit], ToyScorer(), {})
                self.assertAlmostEqual(got["L0:u0"], 3.8)
                recorded = json.loads((root / "thresholds.json").read_text())
                self.assertIn("L0:u0", recorded["supplemental_thresholds"])
            finally:
                score60.DATA, score60.POOL_SIZE = original_data, original_size

    def test_source_fingerprint_rewrite_preserves_supplemental_thresholds_and_ceilings(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-supplemental-preserve-") as directory:
            root = Path(directory)
            path = root / "thresholds.json"
            gen60.write_json(path, {
                "source_fingerprint": {"old": True},
                "thresholds": {"L0:u_supp": 7.0},
                "supplemental_thresholds": {"L0:u_supp": 7.0},
                "observed_ceilings": {"L0:u_supp": 8.0},
                "supplemental_scoring_by_split": {"test": {"dtype": "fp32"}},
            })
            with mock.patch("score60.threshold_fingerprint", return_value={"new": True}), \
                    mock.patch("score60.load_source_arrays", return_value=(
                        np.asarray([[1.0]], dtype=np.float32), ["0:s"], [0], ["L0:u0"])), \
                    mock.patch("score60.compute_thresholds", return_value={"L0:u0": 0.5}), \
                    mock.patch("score60.compute_observed_ceilings", return_value={"L0:u0": 1.0}):
                thresholds = score60.ensure_thresholds(root)
            recorded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(thresholds["L0:u_supp"], 7.0)
            self.assertEqual(recorded["observed_ceilings"]["L0:u_supp"], 8.0)
            self.assertEqual(recorded["supplemental_scoring_by_split"], {"test": {"dtype": "fp32"}})

    def test_generator_blinding_for_primary_selection_and_round_one(self) -> None:
        units = gen60.load_splits()["dev"]
        req = gen60.request_for("M1", "primary", units[0], 0)
        clean = gen60.clean_request(req)
        user = json.loads(clean["params"]["messages"][0]["content"])
        self.assertEqual(set(user), {"marked_token_counts", "marked_tokens", "windows"})
        serialized = json.dumps(clean["params"])
        self.assertNotIn(units[0]["unit_key"], serialized)
        self.assertNotIn('"layer"', serialized)
        self.assertNotIn('"rank"', serialized)
        self.assertNotIn('"rating"', serialized)
        for method, phase in (("M3", "selection"), ("M5", "round1")):
            body = gen60.clean_request(gen60.request_for(method, phase, units[0], 0))["params"]
            blinded = json.loads(body["messages"][0]["content"])
            self.assertIn("windows", blinded)
            self.assertFalse({"strong", "weak", "none", "rank", "rating", "tier"} & set(blinded))

    def test_generated_sentence_duplicates_and_input_window_copies_fail_validation(self) -> None:
        value = _tiered()
        value["strong"] = [" A sentence.", "a   SENTENCE.", "third", "fourth"]
        self.assertEqual(gen60.validate_output("M1", "primary", value), (False, "duplicate_sentence"))
        value = _tiered()
        value["strong"][0] = "  SAMPLE WINDOW 1  "
        self.assertEqual(gen60.validate_output("M1", "primary", value,
                                               input_windows=["Sample window 1"]),
                         (False, "input_window_copy"))

    def test_m5_round_two_request_contains_boolean_outcomes_only(self) -> None:
        unit = gen60.load_splits()["dev"][0]
        output = {"round1": {"parse_status": "ok", "hypothesis": "pattern", "strong": ["a"]*4,
                              "weak": ["b"]*4, "none": ["c"]*4}}
        feedback = [{"tier": tier, "sentence": f"sentence {i}", "fired": bool(i % 2)}
                    for i, tier in enumerate(("strong",)*4 + ("weak",)*4 + ("none",)*4)]
        request = gen60.request_for("M5", "round2", unit, 0, output_row=output, feedback=feedback)
        body = gen60.clean_request(request)
        user = json.loads(body["params"]["messages"][0]["content"])
        self.assertEqual(user["phase"], "round2")
        self.assertEqual(len(user["outcomes"]), 12)
        self.assertTrue(all(type(item["fired"]) is bool for item in user["outcomes"]))
        self.assertFalse(any("activation" in key or "value" in key for item in user["outcomes"] for key in item))

    def test_threshold_excludes_each_target_owner_in_production_function(self) -> None:
        matrix = np.asarray([[100.0], [0.0], [2.0], [4.0]])
        got = score60.compute_thresholds(matrix, ["0:a", "1:a", "2:a", "3:a"], [0], ["L0:u0"], pool_size=4)
        self.assertAlmostEqual(got["L0:u0"], 3.8)

    def test_threshold_sampling_uses_seeded_subset_when_pool_is_smaller_than_source(self) -> None:
        matrix = np.arange(10, dtype=np.float64).reshape(10, 1)
        got = score60.compute_thresholds(matrix, [f"{i}:sid" for i in range(10)], [2], ["L0:u2"],
                                         pool_seed=7, pool_size=4)
        self.assertAlmostEqual(got["L0:u2"], 8.7)
        ceilings = score60.compute_observed_ceilings(matrix, [f"{i}:sid" for i in range(10)], [2], ["L0:u2"],
                                                      pool_seed=7, pool_size=4)
        self.assertEqual(ceilings["L0:u2"], 9.0)

    def test_threshold_owner_row_mutation_in_module_copy_goes_red(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-threshold-mutation-") as directory:
            copied = Path(directory) / "score60.py"
            source = (HERE / "score60.py").read_text(encoding="utf-8")
            original = "keep = owners != int(record_index)"
            self.assertIn(original, source)
            copied.write_text(source.replace(original, "keep = np.ones_like(owners, dtype=bool)", 1), encoding="utf-8")
            code = f"""
import sys, numpy as np
sys.path.insert(0, {str(Path(directory))!r})
import score60
got = score60.compute_thresholds(np.asarray([[100.], [0.], [2.], [4.]]), ['0:a','1:a','2:a','3:a'], [0], ['L0:u0'], pool_size=4)
assert abs(got['L0:u0'] - 3.8) < 1e-9, got
"""
            proc = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=False)
        self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)


class ScoringInvariantTests(unittest.TestCase):
    def test_verify_scoring_m0_fails_closed_on_stale_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-verify-m0-") as directory:
            root = Path(directory)
            unit = {"record_index": 0, "unit_key": "L0:u0", "type": "neuron", "layer": 0,
                    "unit_index": 0, "control_id": None}
            entry = {"sid": "0|M0|primary|strong|0", "text": "baseline", "group": "strong",
                     "record_index": 0, "unit_key": "L0:u0", "marked_span": None}
            gen60.write_output(gen60.output_path(root, "dev", "M0"), {0: {
                "record_index": 0, "unit_key": "L0:u0", "type": "neuron", "layer": 0,
                "parse_status": "ok", "strong": ["baseline"], "none": []}})
            target = score60.score_path(root, "dev", "M0", "primary")
            target.parent.mkdir(parents=True, exist_ok=True)
            model_dir = Path(directory) / "model"
            metadata = {
                "max": np.asarray([[1.0]], dtype=np.float32), "marked": np.asarray([[np.nan]], dtype=np.float32),
                "sids": np.asarray([entry["sid"]]), "unit_keys": np.asarray([unit["unit_key"]]),
                "method": np.asarray("M0"), "phase": np.asarray("primary"),
                "score_fingerprint": np.asarray("stale"), "model_revision": np.asarray(score60.MODEL_REVISION),
                "dtype": np.asarray("bf16"), "device": np.asarray("cpu"),
                "batch_size": np.asarray(8), "backend": np.asarray(score60.SCORE_BACKEND),
                "model_dir": np.asarray(str(model_dir.resolve())),
            }
            np.savez(target, **metadata)
            with mock.patch.object(gen60, "load_splits", return_value={"dev": [unit]}), \
                    mock.patch.object(score60, "plan_live", return_value=[("M0", "primary", [entry])]), \
                    mock.patch.object(score60, "_unit_info", return_value=[unit]), \
                    mock.patch.object(score60, "score_input_fingerprint", return_value="expected"), \
                    mock.patch.object(score60, "resolved_device", return_value="cpu"):
                with self.assertRaisesRegex(score60.ScoreError, "metadata or inputs"):
                    score60.verify_scoring(root, "dev", ["M0"], model_dir, "bf16", "cpu", 8)
                metadata["score_fingerprint"] = np.asarray("expected")
                np.savez(target, **metadata)
                score60.verify_scoring(root, "dev", ["M0"], model_dir, "bf16", "cpu", 8)

    def test_reproduction_gate_checks_overall_and_each_unit_type(self) -> None:
        all_units = {"median_ratio": 1.0, "spearman": 0.99}
        by_type = {"neuron": {"median_ratio": 0.95, "spearman": 0.99},
                   "control": {"median_ratio": 1.05, "spearman": 0.99}}
        self.assertTrue(score60.reproduction_passed(by_type, all_units))
        self.assertFalse(score60.reproduction_passed(
            {**by_type, "control": {"median_ratio": 1.11, "spearman": 0.99}}, all_units))
        self.assertFalse(score60.reproduction_passed({"neuron": by_type["neuron"]}, all_units))
        self.assertFalse(score60.reproduction_passed(by_type, {"median_ratio": 0.89, "spearman": 0.99}))
        self.assertFalse(score60.reproduction_passed(
            {**by_type, "control": {"median_ratio": 1.0, "spearman": 0.97}}, all_units))
        self.assertFalse(score60.reproduction_passed(by_type, {"median_ratio": 1.0, "spearman": 0.97}))

    def test_phase_outputs_and_score_fingerprint_include_inputs_and_configuration(self) -> None:
        root = Path("/tmp/part60-score-paths")
        self.assertEqual(score60.score_path(root, "dev", "M3", "selection").name, "M3-selection.npz")
        self.assertEqual(score60.score_path(root, "dev", "M3", "fresh").name, "M3.npz")
        self.assertEqual(score60.score_path(root, "dev", "M5", "round1").name, "M5-round1.npz")
        self.assertEqual(score60.score_path(root, "dev", "M5", "round2").name, "M5.npz")

        entry = {"sid": "0|M1|strong|0", "text": "first text", "group": "strong",
                 "marked_span": None, "record_index": 0, "unit_key": "L0:u0"}
        units = [{"unit_key": "L0:u0", "layer": 0, "unit_index": 0, "control_id": None}]
        baseline = score60.score_input_fingerprint([entry], units, "/tmp/model", "bf16", "cuda", 8)
        edited_text = {**entry, "text": "replacement text"}
        self.assertNotEqual(baseline, score60.score_input_fingerprint(
            [edited_text], units, "/tmp/model", "bf16", "cuda", 8))
        self.assertNotEqual(baseline, score60.score_input_fingerprint(
            [entry], units, "/tmp/model", "bf16", "cuda", 4))

    def test_orphan_progress_marker_before_memmaps_recovers_on_rerun(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-progress-window-") as directory:
            root = Path(directory)
            code = r"""
import os, signal, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import score60
class Toy:
    model_dir = "/tmp/part60-toy-model"
    dtype = "fp32"
    device = "cpu"
entries = [{"sid":"0|M1|strong|0", "text":"x", "group":"strong", "record_index":0,
            "unit_key":"L0:u0", "marked_span":None}]
units = [{"unit_key":"L0:u0", "type":"neuron", "layer":0, "unit_index":0, "control_id":None}]
root = Path(sys.argv[2])
max_path, marked_path, progress = score60._progress_paths(root, "dev", "M1", "primary")
def kill_before_first_memmap(*args, **kwargs):
    assert progress.is_file(), "progress marker must precede memmap creation"
    assert not max_path.exists() and not marked_path.exists()
    os.kill(os.getpid(), signal.SIGKILL)
score60.np.lib.format.open_memmap = kill_before_first_memmap
score60.live_score_phase(root, "dev", "M1", "primary", entries, units, Toy(), {"L0:u0":1.0})
"""
            killed = subprocess.run([sys.executable, "-c", code, str(HERE), str(root)], check=False)
            self.assertEqual(killed.returncode, -signal.SIGKILL)
            max_path, marked_path, progress = score60._progress_paths(root, "dev", "M1", "primary")
            self.assertTrue(progress.is_file())
            self.assertFalse(max_path.exists())
            self.assertFalse(marked_path.exists())
            class Toy:
                model_dir = "/tmp/part60-toy-model"
                dtype = "fp32"
                device = "cpu"
                def score_batch(self, items, rows, max_out, marked_out, batch_size=8):
                    max_out[rows[0], 0] = 2.0
                    marked_out[rows[0], 0] = np.nan
            entries = [{"sid":"0|M1|strong|0", "text":"x", "group":"strong", "record_index":0,
                        "unit_key":"L0:u0", "marked_span":None}]
            units = [{"unit_key":"L0:u0", "type":"neuron", "layer":0, "unit_index":0, "control_id":None}]
            output = score60.live_score_phase(root, "dev", "M1", "primary", entries, units,
                                               Toy(), {"L0:u0":1.0})
            self.assertTrue(output.is_file())
            self.assertFalse(max_path.exists())
            self.assertFalse(marked_path.exists())

    def test_m0_legacy_cache_cleanup_uses_selected_split(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-m0-legacy-") as directory:
            root = Path(directory)
            (root / "out/dev").mkdir(parents=True)
            (root / "out/dev/M0.jsonl").write_text("{}\n")
            target = score60.score_path(root, "dev", "M0", "primary")
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez(target, max=np.asarray([[1.0]]), sids=np.asarray(["0|M0|strong|0"]))
            entry = {"sid": "0|M0|strong|0", "text": "x", "group": "strong",
                     "record_index": 0, "unit_key": "L0:u0"}
            with mock.patch("score60.check_data_paths"), \
                    mock.patch("score60.ensure_thresholds", return_value={}), \
                    mock.patch("gen60.load_splits", return_value={"dev": []}), \
                    mock.patch("score60.cached_m0_existing", return_value=None), \
                    mock.patch("score60.plan_live", return_value=[("M0", "primary", [entry])]), \
                    mock.patch("score60.score_input_fingerprint", return_value="new"), \
                    mock.patch("score60._unit_info", return_value=[]), \
                    mock.patch("score60.resolved_device", return_value="cpu"), \
                    mock.patch("score60.QwenScorer", side_effect=score60.ScoreError("reached scorer")):
                stderr = io.StringIO()
                with mock.patch("sys.stderr", stderr):
                    status = score60.main(["--root", str(root), "score", "--split", "dev", "--methods", "M0"])
            self.assertEqual(status, 2)
            self.assertIn("reached scorer", stderr.getvalue())
            self.assertNotIn("NameError", stderr.getvalue())
            self.assertFalse(target.exists())

    def test_generator_mutating_entrypoints_take_the_shared_writer_lock(self) -> None:
        for name in ("smoke_m6", "submit_phase", "collect_phase", "choose_m3_winners", "prepare_m0"):
            with self.subTest(name=name):
                self.assertIn("writer_lock", inspect.getsource(getattr(gen60, name)))


class TierTests(unittest.TestCase):
    def test_strong_weak_none_ordering_and_strict_ties(self) -> None:
        values = analyze60.tier_statistics([2, 2, 2, 1], [2, 0, 0, 0], [0, 0, 0, 0], 1)
        self.assertEqual(values["h_strong"], 0.75)  # equality at threshold is a miss
        self.assertEqual(values["h_weak"], 0.25)
        self.assertEqual(values["h_none"], 0.0)
        self.assertTrue(values["verified"])
        self.assertTrue(values["ordering"])
        tied = analyze60.tier_statistics([2, 2, 2, 2], [2, 2, 2, 2], [0, 0, 0, 0], 1)
        self.assertFalse(tied["ordering"])
        self.assertAlmostEqual(tied["auc"], 1.0)

    def test_all_hit_all_miss_and_parse_failure(self) -> None:
        all_hit = analyze60.tier_statistics([2]*4, [2]*4, [2]*4, 1)
        self.assertFalse(all_hit["verified"])
        all_miss = analyze60.tier_statistics([0]*4, [0]*4, [0]*4, 1)
        self.assertFalse(all_miss["verified"])
        failed = analyze60.tier_statistics([2]*4, [0]*4, [0]*4, 1, parsed=False)
        self.assertFalse(failed["verified"])
        self.assertIsNone(failed["h_strong"])

    def test_m3_uses_fresh_and_m5_uses_round_two(self) -> None:
        self.assertEqual(analyze60.phase_for_metric("M3"), "fresh")
        self.assertEqual(analyze60.phase_for_metric("M5"), "round2")
        recs = (7, 8)
        keys = {rec: f"L0:u{rec}" for rec in recs}
        output = {"M3": {}, "M5": {}}
        sids, matrix_rows = [], []
        for method, earlier, target in (("M3", "selection", "fresh"), ("M5", "round1", "round2")):
            for rec in recs:
                unit_rows = {"record_index": rec, "unit_key": keys[rec], "type": "neuron", "layer": 0}
                good_values = {
                    "strong": [2.0] * 4, "weak": [1.5] * 4, "none": [0.0] * 4,
                }
                poor_values = {
                    "strong": [0.0] * 4, "weak": [0.0] * 4, "none": [2.0] * 4,
                }
                unit_rows[earlier] = {"parse_status": "ok", **good_values}
                unit_rows[target] = {"parse_status": "ok", **(good_values if rec == 7 else poor_values)}
                output[method][rec] = unit_rows
                for phase in (earlier, target):
                    tier_values = good_values if phase == earlier or rec == 7 else poor_values
                    for tier in ("strong", "weak", "none"):
                        for index, value in enumerate(tier_values[tier]):
                            sids.append(f"{rec}|{method}|{phase}|{tier}|{index}")
                            activations = [0.0, 0.0]
                            activations[recs.index(rec)] = value
                            matrix_rows.append(activations)
        matrix = np.asarray(matrix_rows, dtype=float)
        score = {"sids": np.asarray(sids), "unit_keys": np.asarray([keys[7], keys[8]]),
                 "max": matrix, "marked": np.full_like(matrix, np.nan)}
        sid_rows = {sid: i for i, sid in enumerate(score["sids"])}
        col = {keys[7]: 0, keys[8]: 1}
        for method in ("M3", "M5"):
            phase = analyze60.phase_for_metric(method)
            good = analyze60.unit_metric(output[method][7], score, sid_rows, col, 7, keys[7], method, phase, 1.)
            bad = analyze60.unit_metric(output[method][8], score, sid_rows, col, 8, keys[8], method, phase, 1.)
            self.assertTrue(good["verified"])
            self.assertFalse(bad["verified"])

    def test_m4_marked_threshold_is_computed_on_same_four_sentences(self) -> None:
        got = analyze60.tier_statistics_with_marked([2, 2, 0, 0], [], [0, 0, 0, 0], [2, 2, 2, 0], 1)
        self.assertFalse(got["whole_strong_verified"])
        self.assertTrue(got["marked_verified"])

    def test_m4_missing_marked_activation_is_unscored_on_both_comparators(self) -> None:
        got = analyze60.tier_statistics_with_marked([2, 2, 2, 0], [], [0, 0, 0, 0], [2, 2, float("nan"), 0], 1)
        self.assertFalse(got["marked_comparable"])
        self.assertIsNone(got["marked_verified"])
        self.assertIsNone(got["whole_strong_verified"])


class AnalyzerRoundTripTests(unittest.TestCase):
    def test_generated_and_scored_files_round_trip_through_production_analysis(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-e2e-analysis-") as directory:
            root = Path(directory)
            units, thresholds = _write_analyzer_fixture(root)
            matrix, sid_rows, unit_cols = analyze60.load_acts(score60.score_path(root, "dev", "M1", "primary"))
            self.assertIn("10|M1|primary|strong|0", sid_rows)
            self.assertEqual(unit_cols["L0:u10"], 0)
            self.assertEqual(matrix["max"][sid_rows["10|M1|primary|strong|0"], 0], 2.0)
            result = analyze60.aggregate_method(root, "dev", "M1", units, thresholds)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result["by_type"]["neuron"]["n_parseable"], 2)
            self.assertAlmostEqual(result["by_type"]["neuron"]["verified_rate"], 0.5)
            self.assertAlmostEqual(result["by_type"]["neuron"]["mean_h_strong"], 0.5)
            self.assertAlmostEqual(result["by_type"]["neuron"]["mean_h_weak"], 0.25)
            self.assertAlmostEqual(result["by_type"]["neuron"]["mean_h_none"], 0.0)

    def test_nonfinite_threshold_is_reported_as_threshold_missing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-nonfinite-threshold-") as directory:
            root = Path(directory)
            units, thresholds = _write_analyzer_fixture(root)
            thresholds = {key: float("nan") for key in thresholds}
            result = analyze60.aggregate_method(root, "dev", "M1", units, thresholds)
            self.assertEqual(result["n_threshold_missing"], len(units))
            self.assertTrue(all(row["parse_status"] == "threshold_missing" for row in result["unit_rows"]))

    def test_sid_key_format_mutation_in_analyzer_copy_goes_red(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-e2e-sabotage-") as directory:
            root = Path(directory)
            units, thresholds = _write_analyzer_fixture(root)
            mutated_dir = Path(directory) / "mutated"
            mutated_dir.mkdir()
            source = (HERE / "analyze60.py").read_text(encoding="utf-8")
            original = "sid_rows = {sid: i for i, sid in enumerate(sids)}"
            self.assertIn(original, source)
            (mutated_dir / "analyze60.py").write_text(
                source.replace(original, "sid_rows = {sid.split('|', 1)[1]: i for i, sid in enumerate(sids)}", 1),
                encoding="utf-8")
            code = "\n".join([
                "import sys",
                "from pathlib import Path",
                f"sys.path.insert(0, {str(mutated_dir)!r})",
                f"sys.path.insert(1, {str(HERE)!r})",
                "import analyze60",
                "root = Path(sys.argv[1])",
                f"units = {units!r}",
                f"thresholds = {thresholds!r}",
                "original = analyze60.load_acts",
                "def broken(path):",
                "    data, sid_rows, cols = original(path)",
                "    return data, {key.split('|', 1)[1]: value for key, value in sid_rows.items()}, cols",
                "analyze60.load_acts = broken",
                "result = analyze60.aggregate_method(root, 'dev', 'M1', units, thresholds)",
                "assert result['by_type']['neuron']['verified_rate'] == 0.5, result['by_type']",
            ])
            proc = subprocess.run([sys.executable, "-c", code, str(root)], text=True, capture_output=True, check=False)
            self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)


class PredictionTests(unittest.TestCase):
    def _method(self, real: float, control: float, *, hs: float = .8, hn: float = .3,
                ordering: float = .8, cost: float = .01,
                positive_real: float | None = None,
                positive_control: float | None = None) -> dict[str, object]:
        return {"n_units": 500, "n_parseable": 500, "real_tier_ordering_share": ordering,
                "real_tier_ordering_n": 100, "cost": {"usd_per_unit": cost},
                "by_type": {
                    "neuron": {"positive_only_verified_rate": real if positive_real is None else positive_real,
                               "verified_rate": real, "mean_h_strong": hs, "mean_h_none": hn,
                               "marked_verified_rate": .1, "whole_strong_verified_rate": .3},
                    "control": {"positive_only_verified_rate": control if positive_control is None else positive_control,
                                "verified_rate": control, "mean_h_strong": .2, "mean_h_none": .1}}}

    def _bootstrap(self, method: str, comparators: tuple[str, ...],
                   lower: float, upper: float) -> dict[str, object]:
        return {"comparisons": {method: {comparator: {
            f"gap_{metric}": {"ci95": [lower, upper]}
            for metric in ("positive_only", "full_rule")
        } for comparator in comparators}}}

    def test_prediction_table_known_answers(self) -> None:
        metrics = {"M0": self._method(.281, .062, hs=.5), "M1": self._method(.20, .02, hs=.8, hn=.3),
                   "M2": self._method(.20, .02, hs=.7, hn=.15), "M3": self._method(.22, .02),
                   "M4": self._method(.20, .02), "M5": self._method(.20, .04),
                   "M6": self._method(.21, .02, cost=.02)}
        metrics["M0"]["unit_rows"] = [{"unit_key": "same", "type": "neuron", "parsed": True, "h_strong": .5}]
        metrics["M1"]["unit_rows"] = [{"unit_key": "same", "type": "neuron", "parsed": True, "h_strong": .8}]
        m5r1 = {"by_type": {"neuron": {"verified_rate": .10}, "control": {"verified_rate": .01}}}
        rows = analyze60.prediction_table(metrics, m5r1, "dev")
        by_id = {r["id"]: r for r in rows}
        self.assertEqual([r["id"] for r in rows], [f"60{x}" for x in "abcdefghij"])
        for pid in "abcdefghi":
            self.assertEqual(by_id[f"60{pid}"]["status"], "held")
        self.assertEqual(by_id["60j"]["status"], "unscored")

    def test_test_prediction_requires_m0_m1_and_selected_candidate(self) -> None:
        metrics = {"M0": self._method(.30, .10), "M1": self._method(.25, .10), "M2": self._method(.32, .10)}
        bootstrap = self._bootstrap("M2", ("M0", "M1"), 0.0, 0.04)
        rows = analyze60.prediction_table(metrics, None, "test", "M2", paired_bootstrap=bootstrap)
        by_id = {r["id"]: r for r in rows}
        self.assertEqual(by_id["60a"]["status"], "unscored")
        self.assertEqual(by_id["60j"]["status"], "unscored (one candidate tested)")
        no_candidate = analyze60.prediction_table(metrics, None, "test", None)
        self.assertEqual(next(row for row in no_candidate if row["id"] == "60j")["status"], "unscored")

    def test_m1_candidate_is_compared_to_m0_without_self_comparison(self) -> None:
        metrics = {"M0": self._method(.20, .10), "M1": self._method(.30, .10)}
        bootstrap = self._bootstrap("M1", ("M0",), 0.06, 0.14)
        rows = analyze60.prediction_table(metrics, None, "test", "M1", paired_bootstrap=bootstrap)
        prediction = next(row for row in rows if row["id"] == "60j")
        self.assertEqual(prediction["status"], "failed")
        self.assertIn("leads over M0", prediction["evidence"])
        self.assertNotIn("M1,M0", prediction["evidence"])

    def test_dev_advancement_requires_both_five_point_point_leads_and_ignores_ci_bounds(self) -> None:
        baseline = self._method(.30, .10, positive_real=.40, positive_control=.10)
        candidate = self._method(.40, .10, positive_real=.50, positive_control=.10)
        metrics = {"M0": baseline, "M1": candidate}
        bootstrap = {"comparisons": {"M1": {"M0": {
            "gap_positive_only": {"ci95": [.05, .15]},
            "gap_full_rule": {"ci95": [.05, .15]},
        }}}}
        self.assertEqual(analyze60.choose_dev_leader(metrics, bootstrap)["test_candidate"], "M1")
        weak_full_ci = {"comparisons": {"M1": {"M0": {
            "gap_positive_only": {"ci95": [.05, .15]},
            "gap_full_rule": {"ci95": [.049, .15]},
        }}}}
        # The registered rule is the point lead; the interval is reported, not a second condition.
        self.assertEqual(analyze60.choose_dev_leader(metrics, weak_full_ci)["test_candidate"], "M1")
        weak_positive = self._method(.40, .10, positive_real=.44, positive_control=.10)
        self.assertIsNone(analyze60.choose_dev_leader({"M0": baseline, "M1": weak_positive}, bootstrap)["test_candidate"])

    def test_report_prints_amendment3_tables_and_interpretation_limits(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-report-") as directory:
            root = Path(directory)
            tier = {"positive_only_verified_rate": .5, "verified_rate": .4,
                    "mean_h_strong": .8, "mean_h_weak": .4, "mean_h_none": .1,
                    "mean_auc": .7, "n_mean_h_strong": 2, "n_mean_h_weak": 2,
                    "n_mean_h_none": 2, "n_mean_auc": 2}
            method = {"n_parseable": 8, "n_units": 10, "n_failed": 2,
                      "n_threshold_missing": 1, "by_type": {"neuron": tier, "control": tier},
                      "real_tier_ordering_share": .6, "real_tier_ordering_n": 5,
                      "real_tier_ordering_two_hits_share": .8,
                      "real_tier_ordering_two_hits_n": 2,
                      "cost": {"calls": 4, "usd_per_unit": .001}}
            bootstrap = {"seed": 6003, "resamples": 2000, "comparisons": {"M1": {"M0": {
                "gap_positive_only": {"estimate": .1, "ci95": [.05, .15]}}}}}
            analysis = {"split": "dev", "metrics": {"M1": method}, "predictions": [],
                        "dev_selection": {"test_candidate": None}, "paired_bootstrap": bootstrap,
                        "unjudgeable": {"count": 3, "known_ceiling_units": 10,
                                        "unknown_ceiling_units": 0, "by_type": {"neuron": 3}},
                        "max_tokens_truncations": {}, "skipped_methods": [],
                        "dropped_generation": [], "m5_round1": None}
            md_path, _ = analyze60.write_reports(root, [analysis])
            report = md_path.read_text(encoding="utf-8")
            self.assertIn("60d >=2 strong hits", report)
            self.assertIn("0.800 (n=2)", report)
            self.assertIn("Empirically unjudgeable (threshold at or above the pool maximum): 3", report)
            self.assertIn("2,000 paired resamples; seed 6003", report)
            self.assertIn("| M1 | M0 | gap_positive_only | 0.100 | [0.050, 0.150] |", report)
            self.assertIn("Interpretation limits:", report)
            self.assertEqual(json.loads((root / "test-selection.json").read_text())["status"], "incomplete")

    def test_test_selection_is_ready_only_for_complete_dev_analysis(self) -> None:
        tier = {"positive_only_verified_rate": .2, "verified_rate": .1,
                "mean_h_strong": .5, "mean_h_weak": .3, "mean_h_none": .1,
                "mean_auc": .6, "n_parseable": 1}
        method = {"n_parseable": 1, "n_units": 1, "n_failed": 0, "n_threshold_missing": 0,
                  "by_type": {"neuron": tier, "control": tier},
                  "real_tier_ordering_share": .5, "real_tier_ordering_n": 1,
                  "real_tier_ordering_two_hits_share": .5, "real_tier_ordering_two_hits_n": 1,
                  "cost": {"calls": 1, "usd_per_unit": .01}}
        analysis = {"split": "dev", "metrics": {f"M{i}": method for i in range(7)},
                    "predictions": [], "dev_selection": {"test_candidate": None},
                    "paired_bootstrap": {"seed": 6003, "comparisons": {}},
                    "unjudgeable": {}, "skipped_methods": [{"method": "M7", "reason": "too few windows"}],
                    "max_tokens_truncations": {}, "dropped_generation": [], "m5_round1": None}
        with tempfile.TemporaryDirectory(prefix="part60-selection-status-") as directory:
            root = Path(directory)
            analyze60.write_reports(root, [analysis])
            marker = json.loads((root / "test-selection.json").read_text())
            self.assertEqual(marker["status"], "ready")
            del analysis["metrics"]["M6"]
            analyze60.write_reports(root, [analysis])
            marker = json.loads((root / "test-selection.json").read_text())
            self.assertEqual(marker["status"], "incomplete")
            self.assertEqual(marker["missing_dev_methods"], ["M6"])

    def test_test_selection_opens_when_non_candidate_generation_method_is_dropped(self) -> None:
        tier = {"positive_only_verified_rate": .2, "verified_rate": .1,
                "mean_h_strong": .5, "mean_h_weak": .3, "mean_h_none": .1,
                "mean_auc": .6, "n_parseable": 1}
        method = {"n_parseable": 1, "n_units": 1, "n_failed": 0, "n_threshold_missing": 0,
                  "by_type": {"neuron": tier, "control": tier}, "cost": {"calls": 1},
                  "real_tier_ordering_share": .5, "real_tier_ordering_n": 1,
                  "real_tier_ordering_two_hits_share": .5, "real_tier_ordering_two_hits_n": 1}
        analysis = {"split": "dev", "metrics": {f"M{i}": method for i in range(8) if i != 3},
                    "predictions": [], "dev_selection": {"test_candidate": "M2"},
                    "paired_bootstrap": {"seed": 6003, "comparisons": {}},
                    "unjudgeable": {}, "skipped_methods": [],
                    "dropped_generation": [{"method": "M3", "phase": "primary", "successes": 80,
                                            "expected": 100, "failure_rate": .2}],
                    "max_tokens_truncations": {}, "m5_round1": None}
        with tempfile.TemporaryDirectory(prefix="part60-selection-drop-") as directory:
            root = Path(directory)
            analyze60.write_reports(root, [analysis])
            marker = json.loads((root / "test-selection.json").read_text())
            self.assertEqual(marker["status"], "ready")
            self.assertNotIn("M3", marker["missing_dev_methods"])

    def test_60h_flags_a_positive_marked_minus_whole_difference(self) -> None:
        metrics = {"M0": self._method(.2, .1), "M1": self._method(.2, .1),
                   "M4": self._method(.2, .1)}
        metrics["M4"]["by_type"]["neuron"]["marked_verified_rate"] = .4
        metrics["M4"]["by_type"]["neuron"]["whole_strong_verified_rate"] = .2
        rows = analyze60.prediction_table(metrics, None, "dev")
        prediction = next(row for row in rows if row["id"] == "60h")
        self.assertEqual(prediction["status"], "failed")
        self.assertIn("scoring bug", prediction["evidence"])

    def test_analysis_reports_unjudgeable_units_and_positive_full_rules(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-unjudgeable-") as directory:
            root = Path(directory)
            gen60.write_json(root / "thresholds.json", {"observed_ceilings": {"L0:u1": 1.0, "L0:c2": 2.0}})
            units = [{"unit_key": "L0:u1", "type": "neuron"},
                     {"unit_key": "L0:c2", "type": "control"}]
            report = analyze60.unjudgeable_units(root, units, {"L0:u1": 1.0, "L0:c2": 1.5})
            self.assertEqual(report["count"], 1)
            self.assertEqual(report["by_type"], {"neuron": 1})
        parsed = analyze60.tier_statistics([2, 2, 2, 0], [], [2, 2, 0, 0], 1)
        self.assertTrue(parsed["positive_only_verified"])
        self.assertFalse(parsed["verified"])

    def test_bootstrap_is_fixed_seeded_and_covers_method_minus_m0_and_m1(self) -> None:
        def method_rows(real_hits: bool, control_hits: bool) -> list[dict[str, object]]:
            rows = []
            for kind, hits in (("neuron", real_hits), ("control", control_hits)):
                for index in range(4):
                    value = bool(hits and index < 2)
                    rows.append({"unit_key": f"{kind}:{index}", "type": kind,
                                 "positive_only_verified": value, "verified": value})
            return rows
        metrics = {
            "M0": {"unit_rows": method_rows(False, False)},
            "M1": {"unit_rows": method_rows(True, False)},
            "M2": {"unit_rows": method_rows(True, True)},
        }
        first = analyze60.paired_bootstrap_differences(metrics)
        second = analyze60.paired_bootstrap_differences(metrics)
        self.assertEqual(first, second)
        self.assertEqual(first["resamples"], 2000)
        self.assertIn("gap_positive_only", first["comparisons"]["M2"]["M1"])
        self.assertIn("real_full_rule", first["comparisons"]["M2"]["M0"])

    def test_60d_secondary_denominator_requires_two_strong_hits(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-ordering-") as directory:
            root = Path(directory)
            units, thresholds = _write_analyzer_fixture(root)
            result = analyze60.aggregate_method(root, "dev", "M1", units, thresholds)
            assert result is not None
            self.assertEqual(result["real_tier_ordering_n"], 2)
            self.assertEqual(result["real_tier_ordering_two_hits_n"], 1)

    @unittest.skipUnless(REAL_PILOT.is_file() and REAL_UNITS_TEST.is_file(),
                         "requires Part 58 pilot.jsonl and units.test.json; all other tests use a synthetic pilot")
    def test_registered_test_split_matches_units_test_and_keeps_250_controls(self) -> None:
        registered = json.loads(REAL_UNITS_TEST.read_text(encoding="utf-8"))
        expected = {f"L{int(layer)}:u{int(unit)}" for layer, unit in registered}
        actual = gen60.load_splits(REAL_PILOT)["test"]
        actual_real = {unit["unit_key"] for unit in actual if unit["type"] == "neuron"}
        actual_controls = [unit for unit in actual if unit["type"] == "control"]
        self.assertEqual(actual_real, expected)
        self.assertEqual(len(actual_real), 247)
        self.assertEqual(len(actual_controls), 250)

    def test_generation_drop_uses_strictly_more_than_ten_percent(self) -> None:
        exactly_ten = [{"record_index": i, "parse_status": "failed" if i < 10 else "ok"}
                       for i in range(100)]
        over_ten = [{"record_index": i, "parse_status": "failed" if i < 11 else "ok"}
                    for i in range(100)]
        self.assertEqual(analyze60.generation_failure_report(exactly_ten, "M1", 100), [])
        self.assertEqual(analyze60.generation_failure_report(over_ten, "M1", 100)[0]["successes"], 89)
        # An entirely absent second phase is pending.
        m3 = [{"record_index": i, "selection": {"parse_status": "ok"}} for i in range(100)]
        self.assertEqual(analyze60.generation_failure_report(m3, "M3", 100), [])
        # The missing fresh record counts as a failure among all selection-parsed units.
        for i in range(89):
            m3[i]["fresh"] = {"parse_status": "ok"}
        for i in range(89, 98):
            m3[i]["fresh"] = {"parse_status": "failed"}
        m3[99]["selection"] = {"parse_status": "failed"}
        m3_failure = analyze60.generation_failure_report(m3, "M3", 100)
        self.assertEqual(m3_failure[0]["phase"], "fresh")
        self.assertEqual(m3_failure[0]["expected"], 99)
        self.assertEqual(m3_failure[0]["successes"], 89)
        # M5 round 2 is measured only over round-1 successes; its absent key fails.
        m5 = []
        for i in range(100):
            row = {"record_index": i}
            if i < 95:
                row["round1"] = {"parse_status": "ok"}
                if i < 85:
                    row["round2"] = {"parse_status": "ok"}
                elif i < 94:
                    row["round2"] = {"parse_status": "failed"}
            else:
                row["round1"] = {"parse_status": "failed"}
            m5.append(row)
        m5_failure = analyze60.generation_failure_report(m5, "M5", 100)
        self.assertEqual(m5_failure[0]["phase"], "round2")
        self.assertEqual(m5_failure[0]["expected"], 95)
        self.assertEqual(m5_failure[0]["successes"], 85)
        # Missing first-phase output rows count against the full initial denominator.
        missing_primary = [{"record_index": i, "parse_status": "ok"} for i in range(89)]
        initial_failure = analyze60.generation_failure_report(missing_primary, "M1", 100)
        self.assertEqual(initial_failure[0]["expected"], 100)
        self.assertEqual(initial_failure[0]["successes"], 89)


class BatchLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="part60-test-")
        self.root = Path(self.tmp.name) / "run"
        self.basin = Path(self.tmp.name) / "checkout"
        self._basin_patch = mock.patch.object(gen60, "BASIN", self.basin)
        self._basin_patch.start()
        self._checkout_patch = mock.patch.object(gen60, "CHECKOUT_BASIN", self.basin)
        self._checkout_patch.start()

    @property
    def spend_path(self) -> Path:
        return gen60.fixed_spend_path(self.root)

    def tearDown(self) -> None:
        self._basin_patch.stop()
        self._checkout_patch.stop()
        self.tmp.cleanup()

    def test_smoke_unknown_outcome_resets_consecutive_failed_check_streak(self) -> None:
        history = [{"all_five_check": "failed"}, {"all_five_check": "failed"},
                   {"all_five_check": None}, {"all_five_check": "failed"}]
        self.assertEqual(gen60._smoke_consecutive_failed_checks(history), 1)

    def test_sigkill_resume_does_not_resubmit_submitted_chunk(self) -> None:
        marker = self.root / "posts.log"
        code = (
            "import json,sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); import gen60; gen60.CHECKOUT_BASIN=Path(__import__('os').environ['PART60_BASIN']); "
            "gen60.PILOT = Path(sys.argv[4]); "
            "def_dummy=None\n"
            "def fake(method,path,body=None):\n"
            " if method=='POST': Path(sys.argv[3]).write_text('POST\\n'); return json.dumps({'id':'batch-safe'})\n"
            " raise AssertionError(path)\n"
            "gen60.submit_phase(Path(sys.argv[2]),'dev','M1','primary',api=fake,kill_after_saved=True)\n"
        )
        proc = subprocess.run([sys.executable, "-c", code, str(HERE), str(self.root), str(marker), str(_TEST_PILOT)],
                              env={**os.environ, "PART60_BASIN": str(self.basin)}, check=False)
        self.assertEqual(proc.returncode, -signal.SIGKILL)
        self.assertEqual(marker.read_text(), "POST\n")
        calls = []
        message = gen60.submit_phase(self.root, "dev", "M1", "primary", api=lambda *args: calls.append(args))
        self.assertIn("batch-safe", message)
        self.assertIn("collect", message)
        self.assertEqual(calls, [])

    def test_cap_refuses_before_post(self) -> None:
        gen60.append_spend(self.spend_path, {"event": "reserve", "attempt_id": "old", "reserved_usd": 7.99})
        called = []
        with mock.patch.dict(os.environ, {"PART60_CAP": "8"}):
            with self.assertRaisesRegex(gen60.RunnerError, "cap refused"):
                gen60.submit_phase(self.root, "dev", "M1", "primary", api=lambda *args: called.append(args))
        self.assertEqual(called, [])

    def test_cap_above_eight_is_refused_even_with_override(self) -> None:
        called = []
        with mock.patch.dict(os.environ, {"PART60_CAP": "80", "PART60_CAP_OVERRIDE": "1"}):
            with self.assertRaisesRegex(gen60.RunnerError, r"hard \$8 cap"):
                gen60.submit_phase(self.root, "dev", "M1", "primary", api=lambda *args: called.append(args))
        self.assertEqual(called, [])
        self.assertFalse(self.spend_path.exists())

    def test_cap_values_fail_closed_for_environment_and_cli(self) -> None:
        for value in ("nan", "inf", "0", "-1", "not-a-number"):
            called = []
            with mock.patch.dict(os.environ, {"PART60_CAP": value}):
                with self.assertRaisesRegex(gen60.RunnerError, "finite positive"):
                    gen60.submit_phase(self.root, "dev", "M1", "primary", api=lambda *args: called.append(args))
            self.assertEqual(called, [])
            self.assertFalse(self.spend_path.exists())
        with mock.patch("gen60.check_data_paths"):
            with self.assertRaises(SystemExit) as caught:
                gen60.main(["--root", str(self.root), "submit", "--method", "M1",
                            "--split", "dev", "--cap", "nan"])
        self.assertEqual(caught.exception.code, 2)

    def test_fixed_ledger_is_shared_across_roots_and_empty_local_ledger_is_refused(self) -> None:
        gen60.append_spend(self.spend_path, {"event": "reserve", "attempt_id": "fixed-reserve",
                                             "reserved_usd": 7.99, "method": "M1", "split": "dev"})
        other_root = self.root / "different-root"
        called = []
        with mock.patch.dict(os.environ, {"PART60_CAP": "8"}):
            with self.assertRaisesRegex(gen60.RunnerError, "cap refused"):
                gen60.submit_phase(other_root, "dev", "M1", "primary",
                                   api=lambda *args: called.append(args))
        self.assertEqual(called, [])
        own_ledger = other_root / "spend.jsonl"
        own_ledger.parent.mkdir(parents=True, exist_ok=True)
        own_ledger.touch()
        with mock.patch.dict(os.environ, {"PART60_CAP": "8", "PART60_CAP_OVERRIDE": "1"}):
            with self.assertRaisesRegex(gen60.RunnerError, "empty root-local ledger"):
                gen60.submit_phase(other_root, "dev", "M1", "primary",
                                   api=lambda *args: called.append(args))
        self.assertEqual(called, [])

    def test_fixed_ledger_path_does_not_follow_part60_basin(self) -> None:
        expected = self.basin / "results" / "part60" / "spend.jsonl"
        with mock.patch.object(gen60, "BASIN", self.root / "attacker-controlled-basin"):
            self.assertEqual(gen60.fixed_spend_path(self.root), expected)

    def test_shared_ledger_lock_serializes_cap_checks_across_roots(self) -> None:
        units = gen60.load_splits()["dev"]
        requests = [gen60.request_for("M1", "primary", unit, index)
                    for index, unit in enumerate(units)]
        one_reserve = gen60.estimated_usd(requests)
        cap = one_reserve * 1.5
        ready_dir = self.root.parent / "concurrent-ready"
        ready_dir.mkdir()
        code = (
            "import json,sys,time\n"
            "from pathlib import Path\n"
            "sys.path.insert(0,sys.argv[1])\n"
            "import gen60\n"
            "gen60.PILOT=Path(sys.argv[3]); gen60.BASIN=Path(sys.argv[4]); gen60.CHECKOUT_BASIN=Path(sys.argv[4])\n"
            "ready=Path(sys.argv[5]); ready.touch()\n"
            "deadline=time.monotonic()+5\n"
            "while len(list(ready.parent.glob('ready-*')))<2 and time.monotonic()<deadline:\n"
            "    time.sleep(.01)\n"
            "original=gen60.spend_state\n"
            "def slow(path):\n"
            "    total,rows=original(path); time.sleep(.6); return total,rows\n"
            "gen60.spend_state=slow\n"
            "def api(method,path,body=None):\n"
            "    if method=='POST':\n"
            "        Path(sys.argv[6]).write_text('posted'); return json.dumps({'id':ready.name})\n"
            "    raise AssertionError(path)\n"
            "try:\n"
            "    gen60.submit_phase(Path(sys.argv[2]),'dev','M1','primary',api=api)\n"
            "except gen60.RunnerError as exc:\n"
            "    print('REFUSED '+str(exc))\n"
            "else:\n"
            "    print('SUBMITTED')\n"
        )
        children = []
        for index in range(2):
            run_root = self.root / f"concurrent-{index}"
            ready = ready_dir / f"ready-{index}"
            posted = ready_dir / f"posted-{index}"
            child_env = {**os.environ, "PART60_BASIN": str(self.basin), "PART60_CAP": str(cap)}
            children.append(subprocess.Popen(
                [sys.executable, "-c", code, str(HERE), str(run_root), str(_TEST_PILOT),
                 str(self.basin), str(ready), str(posted)],
                env=child_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
        results = [child.communicate(timeout=20) for child in children]
        self.assertEqual([child.returncode for child in children], [0, 0], results)
        combined = "\n".join(stdout for stdout, _ in results)
        self.assertEqual(combined.count("SUBMITTED"), 1, combined)
        self.assertEqual(combined.count("REFUSED"), 1, combined)
        self.assertEqual(len(list(ready_dir.glob("posted-*"))), 1)
        total, ledger = gen60.spend_state(self.spend_path)
        self.assertEqual(len(ledger), 1)
        self.assertLessEqual(total, cap)
        self.assertEqual(total, next(iter(ledger.values()))["reserved_usd"])

    def test_ledger_retire_writes_a_dated_entry_and_removes_only_that_reserve(self) -> None:
        gen60.append_spend(self.spend_path, {"event": "reserve", "attempt_id": "retire-me",
                                             "reserved_usd": 1.25, "method": "M6", "split": "dev"})
        gen60.retire_attempt(self.root, "retire-me")
        total, attempts = gen60.spend_state(self.spend_path)
        self.assertEqual(total, 0.0)
        self.assertEqual(attempts["retire-me"]["event"], "retire")
        self.assertTrue(attempts["retire-me"]["ts"])
        with self.assertRaisesRegex(gen60.RunnerError, "not an unresolved reservation"):
            gen60.retire_attempt(self.root, "retire-me")

    def test_retire_refuses_live_batch_and_collect_settles_force_lost_reservation(self) -> None:
        def submit_api(method: str, path: str, body: object = None) -> str:
            return json.dumps({"id": "batch-live"})
        batch_id = gen60.submit_phase(self.root, "dev", "M1", "primary", api=submit_api)
        manifest_path = gen60.manifest_path(self.root, "dev", "M1", "primary")
        manifest = json.loads(manifest_path.read_text())
        attempt_id = manifest["chunks"][-1]["attempt_id"]
        with self.assertRaisesRegex(gen60.RunnerError, "still processing"):
            gen60.retire_attempt(self.root, attempt_id)
        with self.assertRaisesRegex(gen60.RunnerError, "written reason"):
            gen60.retire_attempt(self.root, attempt_id, force_lost=True)
        gen60.retire_attempt(self.root, attempt_id, force_lost=True, reason="operator confirmed batch lost")
        chunk = manifest["chunks"][-1]
        idmap = json.loads((Path(chunk["directory"]) / "idmap.json").read_text())
        response = {"type": "succeeded", "message": {"usage": {"input_tokens": 10, "output_tokens": 20},
                    "content": [], "stop_reason": "end_turn"}}
        def collect_api(method: str, path: str, body: object = None) -> str:
            if path.endswith("/results"):
                return "".join(json.dumps({"custom_id": custom_id, "result": response}) + "\n"
                               for custom_id in idmap)
            return json.dumps({"processing_status": "ended"})
        self.assertIn("collected", gen60.collect_phase(self.root, "dev", "M1", "primary",
                                                        api=collect_api, max_polls=1, poll_seconds=0))
        self.assertEqual(gen60.spend_state(self.spend_path)[1][attempt_id]["event"], "settle")

    def test_http_400_without_anthropic_error_body_keeps_batch_reservation(self) -> None:
        with self.assertRaisesRegex(gen60.RunnerError, "ambiguous"):
            gen60.submit_phase(self.root, "dev", "M1", "primary",
                               api=lambda *args: (_ for _ in ()).throw(gen60.APIError(400, "bad gateway")))
        total, ledger = gen60.spend_state(self.spend_path)
        self.assertGreater(total, 0.0)
        self.assertEqual(len(ledger), 1)
        self.assertEqual(next(iter(ledger.values()))["event"], "reserve")

    def test_missing_api_key_refuses_before_reserving(self) -> None:
        with mock.patch.dict(os.environ, {"ANTHROPIC_CREDITS_KEY": ""}):
            with self.assertRaisesRegex(gen60.RunnerError, "no reserve"):
                gen60.submit_phase(self.root, "dev", "M1", "primary", api=gen60.api_call)
        self.assertFalse(self.spend_path.exists())
        self.assertFalse(gen60.manifest_path(self.root, "dev", "M1", "primary").exists())

    def test_reserve_is_durable_before_post_and_429_creates_a_new_ambiguous_attempt(self) -> None:
        calls = []
        def api(method: str, path: str, body: object = None) -> str:
            if method == "POST":
                self.assertEqual(set(body), {"requests"})
                calls.append(json.dumps(body, sort_keys=True))
                spent, _ = gen60.spend_state(self.spend_path)
                self.assertGreater(spent, 0.0)
                manifest = json.loads(gen60.manifest_path(self.root, "dev", "M1", "primary").read_text())
                self.assertEqual(manifest["chunks"][-1]["request_count"], len(body["requests"]))
                raise gen60.APIError(429, "rate limited", {"type": "error", "error": {"type": "rate_limit_error"}})
            raise AssertionError(path)
        with mock.patch.dict(os.environ, {"PART60_CAP": "8"}):
            with self.assertRaisesRegex(gen60.RunnerError, "ambiguous"):
                gen60.submit_phase(self.root, "dev", "M1", "primary", api=api)
        first_reserve = json.loads(gen60.manifest_path(self.root, "dev", "M1", "primary").read_text())["chunks"][0]["reserved_usd"]
        with mock.patch.dict(os.environ, {"PART60_CAP": str(first_reserve * 2.5)}):
            with self.assertRaisesRegex(gen60.RunnerError, "ambiguous"):
                gen60.submit_phase(self.root, "dev", "M1", "primary", api=api)
            with self.assertRaisesRegex(gen60.RunnerError, "cap refused"):
                gen60.submit_phase(self.root, "dev", "M1", "primary", api=api)
        manifest = json.loads(gen60.manifest_path(self.root, "dev", "M1", "primary").read_text())
        self.assertEqual([chunk["status"] for chunk in manifest["chunks"]], ["ambiguous", "ambiguous"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(gen60.spend_state(self.spend_path)[0],
                         sum(chunk["reserved_usd"] for chunk in manifest["chunks"]))

    def test_second_root_cannot_submit_same_phase_from_shared_ledger(self) -> None:
        calls = []
        root_a, root_b = self.root / "a", self.root / "b"
        gen60.submit_phase(root_a, "dev", "M1", "primary",
                           api=lambda *args: calls.append(args) or json.dumps({"id": "batch-root-a"}))
        with self.assertRaisesRegex(gen60.RunnerError, "another --root"):
            gen60.submit_phase(root_b, "dev", "M1", "primary",
                               api=lambda *args: calls.append(args) or json.dumps({"id": "batch-root-b"}))
        self.assertEqual(len(calls), 1)

    def test_kill_between_post_and_manifest_update_keeps_reservation_and_resubmits_new_attempt(self) -> None:
        marker = self.root / "post.log"
        code = (
            "import json,sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); import gen60; gen60.CHECKOUT_BASIN=Path(__import__('os').environ['PART60_BASIN']); "
            "gen60.PILOT = Path(sys.argv[4]); "
            "def_dummy=None\n"
            "def fake(method,path,body=None):\n"
            " if method=='POST': Path(sys.argv[3]).write_text('POST\\n'); return json.dumps({'id':'accepted-but-unsaved'})\n"
            " raise AssertionError(path)\n"
            "gen60.submit_phase(Path(sys.argv[2]),'dev','M1','primary',api=fake,kill_after_post=True)\n"
        )
        proc = subprocess.run([sys.executable, "-c", code, str(HERE), str(self.root), str(marker), str(_TEST_PILOT)],
                              env={**os.environ, "PART60_BASIN": str(self.basin)}, check=False)
        self.assertEqual(proc.returncode, -signal.SIGKILL)
        self.assertEqual(marker.read_text(), "POST\n")
        first_total, first_ledger = gen60.spend_state(self.spend_path)
        self.assertGreater(first_total, 0.0)
        self.assertEqual(len(first_ledger), 1)
        calls = []
        message = gen60.submit_phase(self.root, "dev", "M1", "primary",
            api=lambda method, path, body=None: (calls.append(path) or json.dumps({"id": "second-attempt"})))
        self.assertEqual(message, "second-attempt")
        manifest = json.loads(gen60.manifest_path(self.root, "dev", "M1", "primary").read_text())
        self.assertEqual(manifest["chunks"][0]["status"], "ambiguous")
        self.assertEqual(manifest["chunks"][1]["status"], "submitted")
        self.assertEqual(len(gen60.spend_state(self.spend_path)[1]), 2)
        self.assertEqual(calls, ["/v1/messages/batches"])

    def test_kill_after_submit_reserve_keeps_old_attempt_and_reruns_with_a_new_id(self) -> None:
        marker = self.root / "post.log"
        code = (
            "import json,sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); import gen60; gen60.CHECKOUT_BASIN=Path(sys.argv[5]); "
            "gen60.PILOT = Path(sys.argv[4]); gen60.BASIN = Path(sys.argv[5]); "
            "def_dummy=None\n"
            "def fake(method,path,body=None):\n"
            " Path(sys.argv[3]).write_text('POST\\n'); return json.dumps({'id':'after-reserve'})\n"
            "gen60.submit_phase(Path(sys.argv[2]),'dev','M1','primary',api=fake,kill_after_reserve=True)\n"
        )
        proc = subprocess.run([sys.executable, "-c", code, str(HERE), str(self.root), str(marker),
                               str(_TEST_PILOT), str(self.basin)],
                              env={**os.environ, "PART60_BASIN": str(self.basin)}, check=False)
        self.assertEqual(proc.returncode, -signal.SIGKILL)
        self.assertFalse(marker.exists())
        first_total, first_ledger = gen60.spend_state(self.spend_path)
        self.assertGreater(first_total, 0.0)
        self.assertEqual(len(first_ledger), 1)
        first_attempt = next(iter(first_ledger))
        posted = gen60.submit_phase(self.root, "dev", "M1", "primary",
                                    api=lambda *args: json.dumps({"id": "after-reserve-rerun"}))
        self.assertEqual(posted, "after-reserve-rerun")
        manifest = json.loads(gen60.manifest_path(self.root, "dev", "M1", "primary").read_text())
        second_attempt = manifest["chunks"][0]["attempt_id"]
        self.assertNotEqual(first_attempt, second_attempt)
        self.assertEqual(len(gen60.spend_state(self.spend_path)[1]), 2)

    def test_collect_kill_matrix_rewrite_keeps_rows_and_totals_idempotent(self) -> None:
        stages = ("after_raw_results", "after_row_write", "after_settle", "after_manifest")
        for stage in stages:
            case = self.root / stage
            basin = case / "checkout"
            run_root = case / "run"
            with mock.patch.object(gen60, "BASIN", basin), mock.patch.object(gen60, "CHECKOUT_BASIN", basin):
                def submit_api(method: str, path: str, body: object = None) -> str:
                    if method == "POST":
                        return json.dumps({"id": "batch-collect-kill"})
                    raise AssertionError(path)
                gen60.submit_phase(run_root, "dev", "M1", "primary", api=submit_api)
            code = (
                "import json,sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); import gen60; gen60.CHECKOUT_BASIN=Path(sys.argv[5]); "
                "gen60.PILOT = Path(sys.argv[4]); gen60.BASIN = Path(sys.argv[5]);\n"
                "def api(method,path,body=None):\n"
                " if path.endswith('/batch-collect-kill'): return json.dumps({'processing_status':'ended'})\n"
                " if path.endswith('/results'):\n"
                "  mf=json.loads(gen60.manifest_path(Path(sys.argv[2]),'dev','M1','primary').read_text())\n"
                "  idmap=json.loads((Path(mf['chunks'][-1]['directory'])/'idmap.json').read_text())\n"
                "  answer={'hypothesis':'one feature','strong':['a','b','c','d'],'weak':['e','f','g','h'],'none':['i','j','k','l']}\n"
                "  return '\\n'.join(json.dumps({'custom_id':cid,'result':{'type':'succeeded','message':{'content':[{'type':'text','text':json.dumps(answer)}],'usage':{'input_tokens':3,'output_tokens':5},'stop_reason':'end_turn'}}}) for cid in idmap)\n"
                " raise AssertionError(path)\n"
                "gen60.collect_phase(Path(sys.argv[2]),'dev','M1','primary',api=api,poll_seconds=0,kill_at=sys.argv[6])\n"
            )
            proc = subprocess.run([sys.executable, "-c", code, str(HERE), str(run_root), "unused",
                                   str(_TEST_PILOT), str(basin), stage],
                                  env={**os.environ, "PART60_BASIN": str(basin)}, check=False)
            self.assertEqual(proc.returncode, -signal.SIGKILL, stage)
            with mock.patch.object(gen60, "BASIN", basin), mock.patch.object(gen60, "CHECKOUT_BASIN", basin):
                def collect_api(method: str, path: str, body: object = None) -> str:
                    if path.endswith("/batch-collect-kill"):
                        return json.dumps({"processing_status": "ended"})
                    if path.endswith("/results"):
                        manifest = json.loads(gen60.manifest_path(run_root, "dev", "M1", "primary").read_text())
                        idmap = json.loads((Path(manifest["chunks"][-1]["directory"]) / "idmap.json").read_text())
                        answer = _tiered()
                        return "\n".join(json.dumps({"custom_id": cid, "result": {"type": "succeeded", "message": {
                            "content": [{"type": "text", "text": json.dumps(answer)}],
                            "usage": {"input_tokens": 3, "output_tokens": 5}, "stop_reason": "end_turn"}}})
                                           for cid in idmap)
                    raise AssertionError(path)
                gen60.collect_phase(run_root, "dev", "M1", "primary", api=collect_api, poll_seconds=0)
                rows = gen60.read_jsonl(gen60.output_path(run_root, "dev", "M1"))
                cost = analyze60.cost_for(rows, run_root, "dev", "M1")
                self.assertEqual((cost["input_tokens"], cost["output_tokens"], cost["calls"]), (1500, 2500, 500), stage)
                self.assertTrue(all("usage_tokens" not in row for row in rows), stage)
                manifest = json.loads(gen60.manifest_path(run_root, "dev", "M1", "primary").read_text())
                raw_path = Path(manifest["chunks"][0]["directory"]) / "raw-results.jsonl"
                self.assertEqual(len(gen60.read_jsonl(raw_path)), 500, stage)
                events = gen60.read_jsonl(gen60.fixed_spend_path(run_root))
                self.assertEqual(sum(row.get("event") == "settle" for row in events), 1, stage)

    def test_no_created_at_adoption_path_and_listing_is_read_only(self) -> None:
        self.assertFalse(hasattr(gen60, "reconcile_chunk"))
        submit_source = inspect.getsource(gen60._submit_phase_locked)
        self.assertNotIn('last.get("created_at")', submit_source)
        self.assertNotIn('batch.get("created_at")', submit_source)
        self.assertNotIn("request_counts", submit_source)
        self.assertNotIn('api("GET", "/v1/messages/batches', submit_source)
        self.assertNotIn("reconcile", submit_source.lower())
        gen60.write_json(self.root / "run.json", {"started_at": "2026-10-22T12:00:00+00:00"})
        before = sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*"))
        calls = []
        def list_api(method: str, path: str, body: object = None) -> str:
            calls.append((method, path))
            return json.dumps({"data": [
                {"id": "old-batch", "created_at": "2026-10-22T11:00:00+00:00"},
                {"id": "new-batch", "created_at": "2026-10-22T13:00:00+00:00"}], "has_more": False})
        listed = gen60.listing_batches(self.root, api=list_api)
        after = sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*"))
        self.assertEqual([row["id"] for row in listed], ["old-batch", "new-batch"])
        self.assertEqual(calls, [("GET", "/v1/messages/batches?limit=100")])
        self.assertEqual(before, after)

    def test_readding_ambiguous_batch_adoption_mutation_goes_red(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-adoption-mutation-") as directory:
            mutated_dir = Path(directory)
            source = (HERE / "gen60.py").read_text(encoding="utf-8")
            old = '''            if last.get("status") in ("reserved", "posting", "unknown", "ready", "rate_limited", "ambiguous"):
                last["status"] = "ambiguous"
                last["ambiguous_at"] = utc_now()
                manifest["ambiguous_attempts"] = int(manifest.get("ambiguous_attempts", 0)) + 1
                write_json(mfpath, manifest)'''
            adopted = '''            if last.get("status") in ("reserved", "posting", "unknown", "ready", "rate_limited", "ambiguous"):
                return "reconciled batch legacy-adoption"'''
            self.assertIn(old, source)
            (mutated_dir / "gen60.py").write_text(source.replace(old, adopted, 1), encoding="utf-8")
            contract = """import json,sys,tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import gen60
gen60.CHECKOUT_BASIN=Path(__import__('os').environ['PART60_BASIN'])
gen60.PILOT = Path(sys.argv[2])
root=Path(tempfile.mkdtemp())
posts=[]
def api(method,path,body=None):
    if method == 'POST':
        posts.append(path)
        raise gen60.APIError(None, 'timeout')
    raise AssertionError(path)
for _ in range(2):
    try: gen60.submit_phase(root,'dev','M1','primary',api=api)
    except gen60.RunnerError: pass
ledger=gen60.read_jsonl(gen60.fixed_spend_path(root))
assert len(posts) == 2, posts
assert sum(row.get('event') == 'reserve' for row in ledger) == 2, ledger
"""
            baseline = subprocess.run([sys.executable, "-c", contract, str(HERE), str(_TEST_PILOT)],
                env={**os.environ, "PART60_BASIN": str(self.basin)}, capture_output=True, text=True, check=False)
            mutant = subprocess.run([sys.executable, "-c", contract, str(mutated_dir), str(_TEST_PILOT)],
                env={**os.environ, "PART60_BASIN": str(self.basin)}, capture_output=True, text=True, check=False)
            self.assertEqual(baseline.returncode, 0, baseline.stdout + baseline.stderr)
            self.assertNotEqual(mutant.returncode, 0, mutant.stdout + mutant.stderr)

    def test_cap_finite_guard_mutation_goes_red(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-cap-mutation-") as directory:
            mutated_dir = Path(directory)
            source = (HERE / "gen60.py").read_text(encoding="utf-8")
            old = "if not math.isfinite(cap) or cap <= 0:"
            self.assertIn(old, source)
            (mutated_dir / "gen60.py").write_text(source.replace(old, "if cap <= 0:", 1), encoding="utf-8")
            contract = """import sys
sys.path.insert(0, sys.argv[1])
import gen60
try:
    gen60.validate_cap('nan')
except gen60.RunnerError:
    raise SystemExit(0)
raise SystemExit(1)
"""
            baseline = subprocess.run([sys.executable, "-c", contract, str(HERE)], capture_output=True, text=True, check=False)
            mutant = subprocess.run([sys.executable, "-c", contract, str(mutated_dir)], capture_output=True, text=True, check=False)
            self.assertEqual(baseline.returncode, 0, baseline.stdout + baseline.stderr)
            self.assertNotEqual(mutant.returncode, 0, mutant.stdout + mutant.stderr)

    def test_smoke_unknown_streak_mutation_goes_red(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-smoke-streak-mutation-") as directory:
            mutated_dir = Path(directory)
            source = (HERE / "gen60.py").read_text(encoding="utf-8")
            old = """        if check is None:
            break"""
            self.assertIn(old, source)
            (mutated_dir / "gen60.py").write_text(source.replace(old, old.replace("break", "continue"), 1),
                                                   encoding="utf-8")
            contract = """import sys
sys.path.insert(0, sys.argv[1])
import gen60
history = [{'all_five_check':'failed'}, {'all_five_check':'failed'},
           {'all_five_check':None}, {'all_five_check':'failed'}]
assert gen60._smoke_consecutive_failed_checks(history) == 1
"""
            baseline = subprocess.run([sys.executable, "-c", contract, str(HERE)], capture_output=True, text=True, check=False)
            mutant = subprocess.run([sys.executable, "-c", contract, str(mutated_dir)], capture_output=True, text=True, check=False)
            self.assertEqual(baseline.returncode, 0, baseline.stdout + baseline.stderr)
            self.assertNotEqual(mutant.returncode, 0, mutant.stdout + mutant.stderr)

    def test_second_call_rejection_preserves_first_round_evidence(self) -> None:
        units = gen60.load_splits()["dev"]
        m3_rows = {}
        m5_rows = {}
        for unit in units:
            rec = int(unit["record_index"])
            m3_rows[rec] = {"record_index": rec, "unit_key": unit["unit_key"], "type": unit["type"], "layer": unit["layer"],
                            "parse_status": "ok", "selection": {"parse_status": "ok"},
                            "candidates": [{"rank": i, "hypothesis": f"h{i}"} for i in (1, 2, 3)],
                            "winner_rank": 1, "winner_hypothesis": "first winner"}
            round1 = {"parse_status": "ok", "hypothesis": "round one", "strong": ["s"]*4,
                      "weak": ["w"]*4, "none": ["n"]*4}
            m5_rows[rec] = {"record_index": rec, "unit_key": unit["unit_key"], "type": unit["type"], "layer": unit["layer"],
                            "parse_status": "ok", "round1": round1, "hypothesis": "round one",
                            "strong": round1["strong"], "weak": round1["weak"], "none": round1["none"]}
        gen60.write_output(gen60.output_path(self.root, "dev", "M3"), m3_rows)
        m5root = self.root / "m5"
        gen60.write_output(gen60.output_path(m5root, "dev", "M5"), m5_rows)
        (self.root / "acts/dev").mkdir(parents=True, exist_ok=True)
        (self.root / "acts/dev/M3-selection.npz").write_bytes(b"scored selection fixture")
        def reject(method: str, path: str, body: object = None) -> str:
            raise gen60.APIError(400, "request rejected",
                                 {"type": "error", "error": {"type": "invalid_request_error"}})
        with mock.patch("gen60._choose_m3_winners_locked", return_value=len(units)):
            result3 = gen60.submit_phase(self.root, "dev", "M3", "fresh", api=reject)
        with mock.patch("gen60.build_feedback", return_value=[]):
            result5 = gen60.submit_phase(m5root, "dev", "M5", "round2", api=reject)
        self.assertEqual(result3, "rejected")
        self.assertEqual(result5, "rejected")
        first_rec = int(units[0]["record_index"])
        rows3 = gen60.read_output(gen60.output_path(self.root, "dev", "M3"))[first_rec]
        rows5 = gen60.read_output(gen60.output_path(m5root, "dev", "M5"))[first_rec]
        self.assertEqual(rows3["selection"]["parse_status"], "ok")
        self.assertEqual(rows3["winner_hypothesis"], "first winner")
        self.assertEqual(rows3["fresh"]["raw_parse_status"], "api_rejected")
        self.assertEqual(rows5["round1"]["hypothesis"], "round one")
        self.assertEqual(rows5["strong"], ["s"]*4)
        self.assertEqual(rows5["round2"]["raw_parse_status"], "api_rejected")

    def test_parse_failure_is_written_for_every_request_and_phase_drop_is_recorded(self) -> None:
        def fake_submit(method: str, path: str, body: object = None) -> str:
            if method == "POST":
                return json.dumps({"id": "batch-parse"})
            if path.endswith("/batch-parse"):
                return json.dumps({"processing_status": "ended"})
            if path.endswith("/results"):
                manifest = json.loads(gen60.manifest_path(self.root, "dev", "M1", "primary").read_text())
                chunk = manifest["chunks"][0]
                idmap = json.loads((Path(chunk["directory"]) / "idmap.json").read_text())
                lines = []
                for j, custom_id in enumerate(idmap):
                    answer = json.dumps(_tiered()) if j == 0 else "not json"
                    lines.append(json.dumps({"custom_id": custom_id, "result": {"type": "succeeded", "message": {
                        "content": [{"type": "text", "text": answer}], "usage": {"input_tokens": 3, "output_tokens": 5}}}}))
                return "\n".join(lines)
            raise AssertionError(path)
        gen60.submit_phase(self.root, "dev", "M1", "primary", api=fake_submit)
        message = gen60.collect_phase(self.root, "dev", "M1", "primary", api=fake_submit)
        self.assertIn("collected 500", message)
        rows = gen60.read_jsonl(gen60.output_path(self.root, "dev", "M1"))
        self.assertEqual(len(rows), 500)
        self.assertEqual(sum(row["parse_status"] == "ok" for row in rows), 1)
        self.assertEqual(sum(row["parse_status"] == "failed" for row in rows), 499)
        self.assertEqual(rows[1]["raw_parse_status"], "not_object")
        self.assertNotIn("usage_tokens", rows[0])
        cost = analyze60.cost_for(rows, self.root, "dev", "M1")
        self.assertEqual((cost["input_tokens"], cost["output_tokens"], cost["calls"]), (1500, 2500, 500))
        manifest = json.loads(gen60.manifest_path(self.root, "dev", "M1", "primary").read_text())
        self.assertEqual(manifest["chunks"][0]["status"], "dropped")
        self.assertEqual(manifest["phase_status"], "dropped")
        repeat = gen60.collect_phase(self.root, "dev", "M1", "primary", api=lambda *args: (_ for _ in ()).throw(AssertionError("unexpected API call")))
        self.assertEqual(repeat, "phase already collected")

    def test_collect_waits_for_batch_processing_to_end_before_results(self) -> None:
        status_calls = []
        def api(method: str, path: str, body: object = None) -> str:
            if method == "POST":
                return json.dumps({"id": "batch-poll"})
            if path.endswith("/batch-poll"):
                status_calls.append(path)
                return json.dumps({"processing_status": "processing" if len(status_calls) == 1 else "ended"})
            if path.endswith("/results"):
                manifest = json.loads(gen60.manifest_path(self.root, "dev", "M1", "primary").read_text())
                chunk = manifest["chunks"][0]
                idmap = json.loads((Path(chunk["directory"]) / "idmap.json").read_text())
                return "\n".join(json.dumps({"custom_id": cid, "result": {"type": "succeeded", "message": {
                    "content": [{"type": "text", "text": json.dumps(_tiered())}], "usage": {"input_tokens": 1, "output_tokens": 1},
                    "stop_reason": "end_turn"}}}) for cid in idmap)
            raise AssertionError(path)
        gen60.submit_phase(self.root, "dev", "M1", "primary", api=api)
        message = gen60.collect_phase(self.root, "dev", "M1", "primary", api=api, max_polls=3, poll_seconds=0)
        self.assertIn("collected 500", message)
        self.assertEqual(len(status_calls), 2)

    def test_collect_not_ready_raises_nonzero_runner_error(self) -> None:
        gen60.write_json(gen60.manifest_path(self.root, "dev", "M1", "primary"), {
            "method": "M1", "split": "dev", "phase": "primary",
            "chunks": [{"status": "submitted", "batch_id": "batch-pending"}],
        })
        def pending(method: str, path: str, body: object = None) -> str:
            return json.dumps({"processing_status": "processing"})
        with self.assertRaisesRegex(gen60.RunnerError, "not ready.*rerun collect"):
            gen60.collect_phase(self.root, "dev", "M1", "primary", api=pending, max_polls=1)

    def test_collector_marks_normalized_duplicate_as_generation_failure(self) -> None:
        duplicate = _tiered()
        duplicate["strong"] = ["same", " SAME ", "third", "fourth"]
        merged = gen60._merge_result("M1", "primary", {"record_index": 1},
                                     json.dumps(duplicate), None)
        self.assertEqual(merged["parse_status"], "failed")
        self.assertEqual(merged["raw_parse_status"], "duplicate_sentence")

    def test_truncated_one_of_500_counts_under_ten_percent_and_retries_only_failed_unit(self) -> None:
        posted_request_counts = []
        current_batch = []
        def api(method: str, path: str, body: object = None) -> str:
            if method == "POST":
                posted_request_counts.append(len(body["requests"]))
                batch_id = f"batch-truncated-{len(posted_request_counts)}"
                current_batch[:] = [batch_id]
                return json.dumps({"id": batch_id})
            if current_batch and path.endswith("/" + current_batch[0]):
                return json.dumps({"processing_status": "ended"})
            if path.endswith("/results"):
                manifest = json.loads(gen60.manifest_path(self.root, "dev", "M1", "primary").read_text())
                chunk = manifest["chunks"][-1]
                idmap = json.loads((Path(chunk["directory"]) / "idmap.json").read_text())
                lines = []
                for index, cid in enumerate(idmap):
                    truncated = current_batch[0] == "batch-truncated-1" and index == 0
                    message = {"content": [{"type": "text", "text": "not complete" if truncated else json.dumps(_tiered())}],
                               "usage": {"input_tokens": 1, "output_tokens": 1100 if truncated else 1}}
                    if truncated:
                        message["stop_reason"] = "max_tokens"
                    lines.append(json.dumps({"custom_id": cid, "result": {"type": "succeeded", "message": message}}))
                return "\n".join(lines)
            raise AssertionError(path)
        gen60.submit_phase(self.root, "dev", "M1", "primary", api=api)
        collected = gen60.collect_phase(self.root, "dev", "M1", "primary", api=api, poll_seconds=0)
        self.assertIn("1 max_tokens truncation", collected)
        self.assertIn("1 failed unit(s) remain for retry", collected)
        rows = gen60.read_output(gen60.output_path(self.root, "dev", "M1"))
        first_rec = int(gen60.load_splits()["dev"][0]["record_index"])
        self.assertEqual(rows[first_rec]["raw_parse_status"], "max_tokens")
        manifest = json.loads(gen60.manifest_path(self.root, "dev", "M1", "primary").read_text())
        self.assertEqual(manifest["chunks"][0]["truncations"], 1)
        self.assertEqual(manifest["chunks"][0]["status"], "complete")
        self.assertEqual(manifest["phase_status"], "retryable_failures")
        self.assertEqual(manifest["remaining_failed_count"], 1)
        retried = gen60.submit_phase(self.root, "dev", "M1", "primary", api=api)
        self.assertEqual(retried, "batch-truncated-2")
        manifest = json.loads(gen60.manifest_path(self.root, "dev", "M1", "primary").read_text())
        self.assertEqual(len(manifest["chunks"]), 2)
        self.assertEqual(posted_request_counts, [500, 1])
        self.assertEqual(manifest["chunks"][1]["unit_keys"], [rows[first_rec]["unit_key"]])

    def test_m6_smoke_unknown_api_error_resets_streak_and_keeps_reservation(self) -> None:
        calls = []
        def reject(method: str, path: str, body: object = None) -> str:
            calls.append((method, path))
            raise gen60.APIError(500, "server failed")
        result = gen60.smoke_m6(self.root, api=reject)
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["all_five_check"])
        self.assertEqual(calls, [("POST", "/v1/messages")])
        self.assertFalse(gen60.method_state_path(self.root, "dev", "M6").exists())
        self.assertGreater(gen60.spend_state(self.spend_path)[0], 0.0)
        history = gen60.read_jsonl(self.root / "out/smoke-M6-runs.jsonl")
        self.assertEqual(history[0]["all_five_check"], None)
        self.assertEqual(gen60.read_jsonl(self.spend_path)[-1]["event"], "reserve")

    def test_smoke_kill_matrix_uses_fresh_ids_and_analysis_totals_stay_stable(self) -> None:
        for stage in ("after_reserve", "after_call", "after_raw_result", "after_settle"):
            case = self.root / stage
            basin, run_root = case / "checkout", case / "run"
            marker = case / "calls.log"
            code = (
                "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); import gen60; gen60.CHECKOUT_BASIN=Path(sys.argv[5]); "
                "gen60.PILOT = Path(sys.argv[4]); gen60.BASIN = Path(sys.argv[5]);\n"
                "def api(method,path,body=None):\n"
                " Path(sys.argv[3]).open('a').write('call\\n'); return sys.argv[7]\n"
                "gen60.smoke_m6(Path(sys.argv[2]),api=api,kill_at=sys.argv[6])\n"
            )
            proc = subprocess.run([sys.executable, "-c", code, str(HERE), str(run_root), str(marker),
                                   str(_TEST_PILOT), str(basin), stage, _m6_response()],
                                  env={**os.environ, "PART60_BASIN": str(basin)}, check=False)
            self.assertEqual(proc.returncode, -signal.SIGKILL, stage)
            call_count = len(marker.read_text().splitlines()) if marker.exists() else 0
            self.assertEqual(call_count, 0 if stage == "after_reserve" else 1, stage)
            with mock.patch.object(gen60, "BASIN", basin), mock.patch.object(gen60, "CHECKOUT_BASIN", basin):
                result = gen60.smoke_m6(run_root, api=lambda *args: _m6_response())
                self.assertEqual(result["status"], "pass", stage)
                history = gen60.read_jsonl(run_root / "out/smoke-M6-runs.jsonl")
                self.assertEqual(history[0]["status"], "error", stage)
                self.assertIsNone(history[0]["all_five_check"], stage)
                self.assertEqual(history[-1]["status"], "pass", stage)
                raw = gen60.read_jsonl(run_root / "out/smoke-M6-results.jsonl")
                expected_calls = 6 if stage in ("after_raw_result", "after_settle") else 5
                self.assertEqual(len(raw), expected_calls, stage)
                self.assertEqual(len({row["custom_id"] for row in raw}), expected_calls, stage)
                cost = analyze60.cost_for([], run_root, "dev", "M6")
                self.assertEqual((cost["input_tokens"], cost["output_tokens"], cost["calls"]),
                                 (expected_calls * 10, expected_calls * 20, expected_calls), stage)
                self.assertEqual(cost["reserved_usd"] > 0,
                                 stage in ("after_reserve", "after_call", "after_raw_result"), stage)
                ledger = gen60.read_jsonl(gen60.fixed_spend_path(run_root))
                self.assertEqual(len({row["attempt_id"] for row in ledger}), 6, stage)

    def test_m6_http_400_is_a_failed_check_and_saves_anthropic_error_body(self) -> None:
        body = {"type": "error", "error": {"type": "invalid_request_error", "message": "unsupported"}}
        calls = []
        def reject(method: str, path: str, body_obj: object = None) -> str:
            calls.append(path)
            raise gen60.APIError(400, "unsupported thinking field", body)
        result = gen60.smoke_m6(self.root, api=reject)
        history = gen60.read_jsonl(self.root / "out/smoke-M6-runs.jsonl")
        raw = gen60.read_jsonl(self.root / "out/smoke-M6-results.jsonl")
        self.assertEqual(result["status"], "dropped")
        self.assertEqual(result["all_five_check"], "failed")
        self.assertEqual([row["all_five_check"] for row in history], ["failed", "failed", "failed"])
        self.assertEqual(len(calls), 3)
        self.assertEqual(raw[0]["error_body"], body)
        self.assertEqual(gen60.spend_state(self.spend_path)[0], 0.0)
        self.assertEqual(sum(row["event"] == "release" for row in gen60.read_jsonl(self.spend_path)), 3)

    def test_m6_submit_refuses_without_smoke_file(self) -> None:
        called = []
        with self.assertRaisesRegex(gen60.RunnerError, "smoke"):
            gen60.submit_phase(self.root, "dev", "M6", "primary", api=lambda *args: called.append(args))
        self.assertEqual(called, [])

    def test_m6_smoke_pass_requires_five_valid_thinking_responses_and_settles_spend(self) -> None:
        calls = []
        def direct(method: str, path: str, body: object = None) -> str:
            self.assertEqual(method, "POST")
            self.assertEqual(path, "/v1/messages")
            self.assertEqual(body["thinking"], {"type": "adaptive"})
            self.assertEqual(body["output_config"], {"effort": "high"})
            self.assertEqual(body["max_tokens"], 8000)
            spent, _ = gen60.spend_state(self.spend_path)
            self.assertGreater(spent, 0.0)
            calls.append(body)
            return _m6_response()
        result = gen60.smoke_m6(self.root, api=direct)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["thinking_count"], 5)
        self.assertEqual(result["thinking_fraction"], 1.0)
        self.assertEqual(len(calls), 5)
        cost = analyze60.cost_for([], self.root, "dev", "M6")
        self.assertEqual((cost["input_tokens"], cost["output_tokens"], cost["calls"]), (50, 100, 5))
        recorded = json.loads((self.root / "out/smoke-M6.json").read_text())
        self.assertEqual(recorded["status"], "pass")
        _, ledger = gen60.spend_state(self.spend_path)
        smoke_rows = [row for row in ledger.values() if row.get("phase") == "smoke"]
        self.assertEqual(len(smoke_rows), 5)
        self.assertTrue(all(row["event"] == "settle" for row in smoke_rows))
        self.assertAlmostEqual(sum(row["usd"] for row in smoke_rows), 0.000055)

    def test_m6_smoke_transient_errors_are_retryable_and_do_not_drop_state(self) -> None:
        failures = ((429, "rate limit"), (500, "server"), (None, "timeout"))
        for index, (status, message) in enumerate(failures):
            root = self.root / str(index)
            basin = root / "checkout"
            def fail(method: str, path: str, body: object = None, status=status, message=message) -> str:
                raise gen60.APIError(status, message)
            with mock.patch.object(gen60, "BASIN", basin), mock.patch.object(gen60, "CHECKOUT_BASIN", basin):
                result = gen60.smoke_m6(root, api=fail)
            self.assertEqual(result["status"], "error")
            self.assertIsNone(result["all_five_check"])
            self.assertFalse(gen60.method_state_path(root, "dev", "M6").exists())
            ledger_path = basin / "results/part60/spend.jsonl"
            self.assertGreater(gen60.spend_state(ledger_path)[0], 0.0)
            self.assertEqual(gen60.read_jsonl(ledger_path)[-1]["event"], "reserve")
            with self.assertRaisesRegex(gen60.RunnerError, "status is error"):
                gen60.submit_phase(root, "dev", "M6", "primary", api=lambda *args: "unused")

    def test_body_bearing_429_on_smoke_keeps_reservation_and_seeded_sample_is_recorded(self) -> None:
        body = {"type": "error", "error": {"type": "rate_limit_error"}}
        result = gen60.smoke_m6(self.root,
            api=lambda *args: (_ for _ in ()).throw(gen60.APIError(429, "rate limited", body)))
        self.assertEqual(result["status"], "error")
        self.assertIn("selection_seed", result)
        self.assertEqual(result["unit_keys"], [u["unit_key"] for u in
                         __import__("random").Random(result["selection_seed"]).sample(gen60.load_splits()["dev"], 5)])
        events = gen60.read_jsonl(self.spend_path)
        self.assertEqual([row["event"] for row in events], ["reserve"])
        self.assertEqual(events[0]["phase"], "smoke")

    def test_retryable_smoke_error_settles_usage_from_completed_responses(self) -> None:
        calls = []
        def partial(method: str, path: str, body: object = None) -> str:
            calls.append(path)
            if len(calls) == 1:
                return _m6_response()
            raise gen60.APIError(500, "server error")
        result = gen60.smoke_m6(self.root, api=partial)
        self.assertEqual(result["status"], "error")
        cost = analyze60.cost_for([], self.root, "dev", "M6")
        self.assertEqual((cost["input_tokens"], cost["output_tokens"], cost["calls"]), (10, 20, 1))
        _, ledger = gen60.spend_state(self.spend_path)
        smoke = [row for row in ledger.values() if row.get("phase") == "smoke"]
        self.assertEqual(len(smoke), 2)
        self.assertEqual(sum(row["event"] == "settle" for row in smoke), 1)
        self.assertEqual(sum(row["event"] == "reserve" for row in smoke), 1)
        self.assertGreater(cost["reserved_usd"], 0)

    def test_m6_smoke_uses_three_fresh_all_five_checks_before_dropping(self) -> None:
        calls = []
        def no_thinking(method: str, path: str, body: object = None) -> str:
            calls.append(body)
            return json.dumps({"content": [{"type": "text", "text": json.dumps(_tiered())}],
                               "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}})
        dropped_root = self.root / "no-thinking"
        with mock.patch.object(gen60, "BASIN", dropped_root / "checkout"), mock.patch.object(gen60, "CHECKOUT_BASIN", dropped_root / "checkout"):
            dropped = gen60.smoke_m6(dropped_root, api=no_thinking)
        self.assertEqual(dropped["status"], "dropped")
        history = gen60.read_jsonl(dropped_root / "out/smoke-M6-runs.jsonl")
        self.assertEqual([row["status"] for row in history], ["retry", "retry", "dropped"])
        self.assertEqual([row["attempt"] for row in history], [1, 2, 3])
        self.assertEqual(len(calls), 15)
        self.assertEqual(len(set(key for row in history for key in row["unit_keys"])), 15)
        self.assertTrue(gen60.method_state_path(dropped_root, "dev", "M6").exists())

        error_root = self.root / "bad-answer"
        def bad_answer(method: str, path: str, body: object = None) -> str:
            return _m6_response(stop_reason="max_tokens", answer={"not": "the M6 schema"})
        with mock.patch.object(gen60, "BASIN", error_root / "checkout"), mock.patch.object(gen60, "CHECKOUT_BASIN", error_root / "checkout"):
            error = gen60.smoke_m6(error_root, api=bad_answer)
        self.assertEqual(error["status"], "dropped")
        self.assertEqual(error["all_five_check"], "failed")
        self.assertTrue(gen60.method_state_path(error_root, "dev", "M6").exists())

    def test_m6_missing_key_is_error_without_dropping_or_reserving(self) -> None:
        with mock.patch.dict(os.environ, {"ANTHROPIC_CREDITS_KEY": ""}):
            result = gen60.smoke_m6(self.root, api=gen60.api_call)
        self.assertEqual(result["status"], "error")
        self.assertIn("ANTHROPIC_CREDITS_KEY", result["reason"])
        self.assertFalse(gen60.method_state_path(self.root, "dev", "M6").exists())
        self.assertFalse(self.spend_path.exists())

    def test_later_passing_smoke_clears_smoke_dropped_state(self) -> None:
        def no_thinking(method: str, path: str, body: object = None) -> str:
            return json.dumps({"content": [{"type": "text", "text": json.dumps(_tiered())}],
                               "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}})
        self.assertEqual(gen60.smoke_m6(self.root, api=no_thinking)["status"], "dropped")
        self.assertTrue(gen60.method_state_path(self.root, "dev", "M6").exists())
        failed_retry = gen60.smoke_m6(self.root, api=lambda *args: (_ for _ in ()).throw(gen60.APIError(500, "server")))
        self.assertEqual(failed_retry["status"], "error")
        self.assertTrue(gen60.method_state_path(self.root, "dev", "M6").exists())
        result = gen60.smoke_m6(self.root, api=lambda *args: _m6_response())
        self.assertEqual(result["status"], "pass")
        self.assertFalse(gen60.method_state_path(self.root, "dev", "M6").exists())

    def test_m6_one_batch_error_does_not_drop_below_ten_percent(self) -> None:
        gen60.write_json(self.root / "out/smoke-M6.json", {"status": "pass", "thinking_chars": 4})
        def api(method: str, path: str, body: object = None) -> str:
            if method == "POST":
                return json.dumps({"id": "batch-m6-errors"})
            if path.endswith("/batch-m6-errors"):
                return json.dumps({"processing_status": "ended"})
            if path.endswith("/results"):
                manifest = json.loads(gen60.manifest_path(self.root, "dev", "M6", "primary").read_text())
                chunk = manifest["chunks"][0]
                idmap = json.loads((Path(chunk["directory"]) / "idmap.json").read_text())
                lines = []
                for index, cid in enumerate(idmap):
                    if index == 0:
                        value = {"custom_id": cid, "result": {"type": "errored", "error": {"type": "api_error"}}}
                    else:
                        value = {"custom_id": cid, "result": {"type": "succeeded", "message": {
                            "content": [{"type": "thinking", "thinking": "reason"},
                                        {"type": "text", "text": json.dumps(_tiered())}],
                            "usage": {"input_tokens": 3, "output_tokens": 5}, "stop_reason": "end_turn"}}}
                    lines.append(json.dumps(value))
                return "\n".join(lines)
            raise AssertionError(path)
        gen60.submit_phase(self.root, "dev", "M6", "primary", api=api)
        result = gen60.collect_phase(self.root, "dev", "M6", "primary", api=api, poll_seconds=0)
        self.assertIn("thinking 499/500 (99.8%)", result)
        self.assertFalse(gen60.method_state_path(self.root, "dev", "M6").exists())
        manifest = json.loads(gen60.manifest_path(self.root, "dev", "M6", "primary").read_text())
        self.assertEqual(manifest["chunks"][0]["thinking_count"], 499)
        self.assertEqual(manifest["chunks"][0]["thinking_fraction"], 499 / 500)
        rows = gen60.read_jsonl(gen60.output_path(self.root, "dev", "M6"))
        self.assertEqual(sum(row["parse_status"] == "failed" for row in rows), 1)
        self.assertEqual(analyze60.generation_failure_report(rows, "M6", 500), [])

    def test_m7_is_reported_skipped_without_submitting(self) -> None:
        called = []
        result = gen60.submit_phase(self.root, "dev", "M7", "primary", api=lambda *args: called.append(args))
        self.assertEqual(result, "skipped")
        self.assertEqual(called, [])
        rows = gen60.read_jsonl(gen60.output_path(self.root, "dev", "M7"))
        self.assertEqual(len(rows), 500)
        self.assertTrue(all(row["parse_status"] == "skipped" for row in rows))

    def test_choose_m3_winners_uses_production_activation_fractions(self) -> None:
        _write_m3_winner_fixture(self.root)
        selected = gen60.choose_m3_winners(self.root, "dev")
        self.assertEqual(selected, 1)
        row = gen60.read_output(gen60.output_path(self.root, "dev", "M3"))[0]
        self.assertEqual(row["winner_rank"], 2)
        self.assertEqual(row["winner_hypothesis"], "hypothesis 2")

    def test_m3_fresh_drop_rule_runs_before_selection_matrix_is_required(self) -> None:
        rows = {}
        for unit in gen60.load_splits()["dev"]:
            rec = int(unit["record_index"])
            rows[rec] = {"record_index": rec, "unit_key": unit["unit_key"], "type": unit["type"], "layer": unit["layer"],
                         "parse_status": "selection_failed", "selection": {"parse_status": "failed"}}
        gen60.write_output(gen60.output_path(self.root, "dev", "M3"), rows)
        called = []
        result = gen60.submit_phase(self.root, "dev", "M3", "fresh", api=lambda *args: called.append(args))
        self.assertIn("skipped second call", result)
        self.assertEqual(called, [])
        saved = gen60.read_output(gen60.output_path(self.root, "dev", "M3"))
        first_rec = int(gen60.load_splits()["dev"][0]["record_index"])
        self.assertEqual(saved[first_rec]["selection"]["parse_status"], "failed")
        self.assertEqual(saved[first_rec]["fresh"]["raw_parse_status"], "prior_failure_rate")

    def test_m3_winner_mutation_in_runner_copy_goes_red(self) -> None:
        _write_m3_winner_fixture(self.root)
        with tempfile.TemporaryDirectory(prefix="part60-winner-mutation-") as directory:
            mutated_dir = Path(directory)
            source = (HERE / "gen60.py").read_text(encoding="utf-8")
            original = "score, rank, winner = sorted(ranks, key=lambda x: (-x[0], x[1]))[0]"
            self.assertIn(original, source)
            (mutated_dir / "gen60.py").write_text(
                source.replace(original, "score, rank, winner = sorted(ranks, key=lambda x: (-x[0], x[1]))[-1]", 1),
                encoding="utf-8")
            code = "\n".join([
                "import sys",
                "from pathlib import Path",
                f"sys.path.insert(0, {str(mutated_dir)!r})",
                "import gen60",
                "root=Path(sys.argv[1])",
                "gen60.choose_m3_winners(root, 'dev')",
                "row=gen60.read_output(gen60.output_path(root, 'dev', 'M3'))[0]",
                "assert row['winner_rank']==2, row",
            ])
            proc = subprocess.run([sys.executable, "-c", code, str(self.root)], text=True, capture_output=True, check=False)
        self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)


class WrapperTests(unittest.TestCase):
    def _prepare_fake_runner(self, root: Path, *, include_m6: bool = True, test_m0: bool = False) -> Path:
        for path in (
            "part60/score60.py", "part60/gen60.py", "part60/analyze60.py",
            "part60/DEFINITIONS-60.md",
            "results/part56-pilot/pilot.jsonl", "results/part56-pilot/label/atlas-labels.jsonl",
            "results/part58/ids.all.json", "results/part58/units.all.json",
            "results/part58/sent.all.jsonl", "results/part58/acts.all.npz",
            "results/part58/gen.all.jsonl", "results/part58/gen.dev.jsonl", "results/part58/gen.test.jsonl",
            "results/part60/thresholds.json",
        ):
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            source = HERE / path.removeprefix("part60/") if path.startswith("part60/") else None
            target.write_text(source.read_text(encoding="utf-8") if source and source.is_file() else "{}\n")
        for method in ("M0", "M1", "M2", "M3", "M4", "M5"):
            target = root / f"results/part60/out/dev/{method}.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}\n")
        if include_m6:
            (root / "results/part60/out/dev/M6.jsonl").write_text("{}\n")
        if test_m0:
            target = root / "results/part60/out/test/M0.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}\n")
        fakebin = root / "fakebin"
        fakebin.mkdir()
        python = fakebin / "python3"
        python.write_text("\n".join([
            f"#!{sys.executable}",
            "import sys",
            "import os, time",
            "from pathlib import Path",
            "args=sys.argv[1:]",
            "if args[:3] == ['-m','pip','install']: raise SystemExit(0)",
            "if args[:3] == ['-m','pip','freeze']: raise SystemExit(1)",
            "if args and args[0] == '-c':",
            "    code=args[1]",
            "    if 'snapshot_download' in code:",
            "        Path(args[2]).write_text(str(Path(args[2]).parent / 'fake-model'))",
            "        raise SystemExit(0)",
            "    if 'analysis_input_fingerprint' in code:",
            "        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'part60'))",
            "        import analyze60",
            "        print(analyze60.analysis_input_fingerprint(Path(args[-1])))",
            "        raise SystemExit(0)",
            "    if 'json.load(open' in code:",
            "        import json",
            "        marker=json.load(open(args[-1]))",
            "        if 'd.get(\"status\"' in code: print(marker.get('status',''))",
            "        elif 'd.get(\"candidate\"' in code: print(marker.get('candidate') or '')",
            "        elif '\",\".join(d.get(\"methods\"' in code: print(','.join(marker.get('methods', [])))",
            "        elif 'd.get(\"analysis_input_sha256\"' in code: print(marker.get('analysis_input_sha256') or '')",
            "        raise SystemExit(0)",
            "    print('torch stub transformers stub')",
            "    raise SystemExit(0)",
            "if args and args[0] == '-u' and len(args) > 1 and args[1] == 'part60/score60.py':",
            "    marker=os.environ.get('PART60_TEST_SCORE_MARKER')",
            "    if marker:",
            "        Path(marker).write_text(str(os.getpid()))",
            "        time.sleep(float(os.environ.get('PART60_TEST_SCORE_DELAY','0'))) ",
            "    if __import__('os').environ.get('FAIL_VERIFY_DEV') == '1' and 'verify' in args and 'dev' in args: raise SystemExit(1)",
            "    raise SystemExit(0)",
            "raise SystemExit(0)",
            "",
        ]))
        python.chmod(0o755)
        nvidia = fakebin / "nvidia-smi"
        nvidia.write_text("#!/bin/sh\nexit 0\n")
        nvidia.chmod(0o755)
        return fakebin

    def _run_fake_runner(self, root: Path, fakebin: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["bash", str(HERE / "run60.sh")], env={**os.environ,
            "PATH": f"{fakebin}:{os.environ['PATH']}", "PART60_HOME": str(root),
            "PART60_BASIN": str(root),
            "PART60_RETURN_PATH": str(root / "return.tgz"),
            "PART60_INCOMPLETE_PATH": str(root / "incomplete.tgz"),
            "PART60_MAX_RUNTIME_SECONDS": "120"}, text=True, capture_output=True, check=False)

    def _write_test_selection(self, root: Path, *, candidate: str | None = "M3",
                              methods: tuple[str, ...] = ("M0", "M1", "M3"),
                              status: str = "ready", stale: bool = False) -> None:
        env = {**os.environ, "PART60_BASIN": str(root), "PYTHONPATH": str(root / "part60")}
        digest = subprocess.check_output([
            sys.executable, "-c",
            "import sys; from pathlib import Path; import analyze60; print(analyze60.analysis_input_fingerprint(Path(sys.argv[1])))",
            str(root / "results/part60")], env=env, text=True).strip()
        if stale:
            digest = "stale-" + digest
        marker = {"status": status, "candidate": candidate, "methods": list(methods),
                  "analysis_input_sha256": digest}
        target = root / "results/part60/analysis/test-selection.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(marker) + "\n")

    def test_run60_does_not_require_skipped_m7_and_pip_freeze_is_advisory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-wrapper-full-") as directory:
            root = Path(directory)
            fakebin = self._prepare_fake_runner(root)
            # M7.jsonl is intentionally absent, and a stale ready archive must be removed.
            (root / "return.tgz").write_text("stale success")
            result = self._run_fake_runner(root, fakebin)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("RETURN_READY", result.stdout)
            self.assertIn("PHASES: dev-only", result.stdout)
            self.assertFalse((root / "results/part60/out/dev/M7.jsonl").exists())
            self.assertIn("pip freeze failed", (root / "logs/version-record.log").read_text())
            self.assertTrue((root / "return.tgz").is_file())
            self.assertFalse((root / "incomplete.tgz").exists())

    def test_run60_preflight_failure_clears_stale_success_and_writes_incomplete(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-wrapper-preflight-") as directory:
            root = Path(directory)
            fakebin = self._prepare_fake_runner(root, include_m6=False)
            (root / "return.tgz").write_text("stale success")
            result = self._run_fake_runner(root, fakebin)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("results/part60/out/dev/M6.jsonl", result.stdout)
            self.assertFalse((root / "return.tgz").exists())
            self.assertTrue((root / "incomplete.tgz").is_file())

    def test_run60_requires_dev_selection_marker_before_test_phase(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-wrapper-testgate-") as directory:
            root = Path(directory)
            fakebin = self._prepare_fake_runner(root, test_m0=True)
            result = self._run_fake_runner(root, fakebin)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("test-gate exit=1", result.stdout)
            self.assertTrue((root / "incomplete.tgz").is_file())

    def test_run60_opens_test_gate_only_for_current_dev_candidate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-wrapper-valid-testgate-") as directory:
            root = Path(directory)
            fakebin = self._prepare_fake_runner(root, test_m0=True)
            for method in ("M1", "M3"):
                target = root / f"results/part60/out/test/{method}.jsonl"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("{}\n")
            self._write_test_selection(root)
            result = self._run_fake_runner(root, fakebin)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("test-gate exit=0", result.stdout)
            self.assertIn("=== score-test", result.stdout)
            self.assertIn("PHASES: dev-and-test", result.stdout)

    def test_run60_rejects_stale_dev_analysis_marker(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-wrapper-stale-testgate-") as directory:
            root = Path(directory)
            fakebin = self._prepare_fake_runner(root, test_m0=True)
            for method in ("M1", "M3"):
                target = root / f"results/part60/out/test/{method}.jsonl"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("{}\n")
            self._write_test_selection(root, stale=True)
            result = self._run_fake_runner(root, fakebin)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("test-gate exit=1", result.stdout)
            self.assertNotIn("=== score-test", result.stdout)

    def test_run60_skips_test_scoring_when_dev_verification_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-wrapper-verify-gate-") as directory:
            root = Path(directory)
            fakebin = self._prepare_fake_runner(root, test_m0=True)
            result = subprocess.run(["bash", str(HERE / "run60.sh")], env={**os.environ,
                "PATH": f"{fakebin}:{os.environ['PATH']}", "PART60_HOME": str(root),
                "PART60_BASIN": str(root), "FAIL_VERIFY_DEV": "1",
                "PART60_RETURN_PATH": str(root / "return.tgz"),
                "PART60_INCOMPLETE_PATH": str(root / "incomplete.tgz"),
                "PART60_MAX_RUNTIME_SECONDS": "120"}, text=True, capture_output=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("skipped: verify-dev did not succeed", (root / "logs/score-test.log").read_text())
            self.assertNotIn("=== score-test", result.stdout)

    def test_run60_kill_and_rerun_keeps_orphan_scorer_lock(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-wrapper-kill-lock-") as directory:
            root = Path(directory)
            fakebin = self._prepare_fake_runner(root)
            marker = root / "scorer.pid"
            env = {**os.environ, "PATH": f"{fakebin}:{os.environ['PATH']}",
                   "PART60_HOME": str(root), "PART60_BASIN": str(root),
                   "PART60_RETURN_PATH": str(root / "return.tgz"),
                   "PART60_INCOMPLETE_PATH": str(root / "incomplete.tgz"),
                   "PART60_MAX_RUNTIME_SECONDS": "120",
                   "PART60_TEST_SCORE_MARKER": str(marker), "PART60_TEST_SCORE_DELAY": "5"}
            process = subprocess.Popen(["bash", str(HERE / "run60.sh")], env=env,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            scorer_pid = None
            try:
                deadline = time.time() + 10
                while time.time() < deadline and not marker.is_file():
                    if process.poll() is not None:
                        self.fail("wrapper exited before the fake scorer started")
                    time.sleep(0.02)
                self.assertTrue(marker.is_file(), "fake scorer did not start")
                scorer_pid = int(marker.read_text())
                process.kill()
                self.assertEqual(process.wait(timeout=5), -signal.SIGKILL)
                rerun = self._run_fake_runner(root, fakebin)
                self.assertEqual(rerun.returncode, 3, rerun.stdout + rerun.stderr)
                self.assertIn("run60 already running", rerun.stdout)
                time.sleep(5.2)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                if scorer_pid is not None:
                    try:
                        os.kill(scorer_pid, 0)
                    except ProcessLookupError:
                        pass
                    else:
                        os.kill(scorer_pid, signal.SIGKILL)

    def test_run60_stub_returns_ready_without_network_or_model(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-wrapper-") as directory:
            root = Path(directory)
            (root / "results/part60/out/test").mkdir(parents=True)
            (root / "results/part60/out/test/M0.jsonl").write_text("{}\n")
            result = subprocess.run(["bash", str(HERE / "run60.sh")], env={**os.environ,
                "PART60_STUB": "1", "PART60_ALLOW_STUB": "1", "PART60_HOME": str(root),
                "PART60_RETURN_PATH": str(root / "return.tgz"),
                "PART60_INCOMPLETE_PATH": str(root / "incomplete.tgz")}, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("RETURN_READY", result.stdout)
            self.assertTrue((root / "return.tgz").is_file())
            with tarfile.open(root / "return.tgz", "r:gz") as archive:
                names = archive.getnames()
            self.assertTrue(any("logs/score-dev.log" in name for name in names))
            self.assertFalse(any("review" in name or "key" in name.lower() for name in names))

    def test_run60_stub_requires_second_explicit_flag(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-wrapper-stub-authorization-") as directory:
            root = Path(directory)
            result = subprocess.run(["bash", str(HERE / "run60.sh")], env={**os.environ,
                "PART60_STUB": "1", "PART60_HOME": str(root),
                "PART60_RETURN_PATH": str(root / "return.tgz"),
                "PART60_INCOMPLETE_PATH": str(root / "incomplete.tgz")}, text=True, capture_output=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("PART60_STUB requires PART60_ALLOW_STUB=1", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
