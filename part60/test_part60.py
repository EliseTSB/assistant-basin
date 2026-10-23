#!/usr/bin/env python3
"""Offline Part 60 contract tests: fake Haiku batches, activation fixtures, and wrapper stubs."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze60
import gen60
import score60


def _tiered() -> dict[str, object]:
    return {"hypothesis": "one simple feature", "strong": ["a", "b", "c", "d"],
            "weak": ["e", "f", "g", "h"], "none": ["i", "j", "k", "l"]}


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
            "usage_tokens": {"input": 0, "output": 0, "calls": 0},
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

    def test_generator_blinding_and_threshold_independence(self) -> None:
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
        matrix = np.asarray([[0., 1.], [2., 4.], [3., 8.], [4., 16.]])
        fixed = score60.compute_thresholds(matrix, ["0:a", "1:a", "2:a", "3:a"], [0, 1], ["A", "B"], pool_size=4)
        fake_rating_change = {"M1": [{"one_job": 1}, {"one_job": 5}]}
        again = score60.compute_thresholds(matrix, ["0:a", "1:a", "2:a", "3:a"], [0, 1], ["A", "B"], pool_size=4)
        self.assertEqual(fixed, again)
        self.assertEqual(fake_rating_change["M1"][0]["one_job"], 1)  # not an input to threshold calculation

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

    def test_threshold_own_text_sabotage_goes_red_through_production_call(self) -> None:
        code = f"""
import sys, numpy as np
sys.path.insert(0, {str(HERE)!r})
import score60
def include_owner_rows(activations, sids, record_indices, unit_keys, pool_seed=7, pool_size=1200):
    return {{unit_keys[0]: float(np.percentile(activations[:, 0], 95, method='linear'))}}
score60.compute_thresholds = include_owner_rows
got = score60.compute_thresholds(np.asarray([[100.], [0.], [2.], [4.]]), ['0:a','1:a','2:a','3:a'], [0], ['L0:u0'], pool_size=4)
assert abs(got['L0:u0'] - 3.8) < 1e-9, got
"""
        proc = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=False)
        self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)


class ScoringInvariantTests(unittest.TestCase):
    def test_reproduction_gate_checks_overall_and_each_unit_type(self) -> None:
        all_units = {"median_ratio": 1.0}
        by_type = {"neuron": {"median_ratio": 0.95}, "control": {"median_ratio": 1.05}}
        self.assertTrue(score60.reproduction_passed(by_type, all_units))
        self.assertFalse(score60.reproduction_passed(
            {**by_type, "control": {"median_ratio": 1.11}}, all_units))
        self.assertFalse(score60.reproduction_passed({"neuron": by_type["neuron"]}, all_units))
        self.assertFalse(score60.reproduction_passed(by_type, {"median_ratio": 0.89}))

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
        rec, key = 7, "L0:u9"
        rows, values = {}, []
        # Selection and round 1 look perfect; fresh and round 2 are deliberately poor.
        for phase, strong_value, none_value, tier in (("fresh", 0., 2., "M3"), ("round2", 0., 2., "M5")):
            nested = "fresh" if tier == "M3" else "round2"
            rows[tier] = {nested: {"parse_status": "ok", "strong": ["x"]*4, "none": ["y"]*4}}
            for group, base in (("strong", strong_value), ("none", none_value)):
                for j in range(4):
                    sid = f"{rec}|{tier}|{phase}|{group}|{j}"
                    rows.setdefault("sids", []).append(sid)
                    values.append(base)
        score = {"sids": np.asarray(rows["sids"]), "unit_keys": np.asarray([key]),
                 "max": np.asarray(values, dtype=float).reshape(16, 1),
                 "marked": np.full((16, 1), np.nan)}
        sid_rows = {sid: i for i, sid in enumerate(score["sids"])}
        col = {key: 0}
        m3 = analyze60.unit_metric(rows["M3"], score, sid_rows, col, rec, key, "M3", "fresh", 1.)
        m5 = analyze60.unit_metric(rows["M5"], score, sid_rows, col, rec, key, "M5", "round2", 1.)
        self.assertFalse(m3["verified"])
        self.assertFalse(m5["verified"])

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

    def test_sid_key_format_sabotage_goes_red_in_production_aggregate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-e2e-sabotage-") as directory:
            root = Path(directory)
            units, thresholds = _write_analyzer_fixture(root)
            code = "\n".join([
                "import sys",
                "from pathlib import Path",
                f"sys.path.insert(0, {str(HERE)!r})",
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
                ordering: float = .8, cost: float = .01) -> dict[str, object]:
        return {"n_units": 500, "n_parseable": 500, "real_tier_ordering_share": ordering,
                "real_tier_ordering_n": 100, "cost": {"usd_per_unit": cost},
                "by_type": {
                    "neuron": {"verified_rate": real, "mean_h_strong": hs, "mean_h_none": hn,
                               "marked_verified_rate": .3, "whole_strong_verified_rate": .1},
                    "control": {"verified_rate": control, "mean_h_strong": .2, "mean_h_none": .1}}}

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
        rows = analyze60.prediction_table(metrics, None, "test", "M2")
        by_id = {r["id"]: r for r in rows}
        self.assertEqual(by_id["60a"]["status"], "unscored")
        self.assertEqual(by_id["60j"]["status"], "held")

    def test_m1_candidate_is_compared_to_m0_without_self_comparison(self) -> None:
        metrics = {"M0": self._method(.20, .10), "M1": self._method(.30, .10)}
        rows = analyze60.prediction_table(metrics, None, "test", "M1")
        prediction = next(row for row in rows if row["id"] == "60j")
        self.assertEqual(prediction["status"], "failed")
        self.assertIn("M0 only (candidate is M1)", prediction["evidence"])

    def test_generation_drop_uses_strictly_more_than_ten_percent(self) -> None:
        exactly_ten = [{"record_index": i, "parse_status": "failed" if i < 10 else "ok"}
                       for i in range(100)]
        over_ten = [{"record_index": i, "parse_status": "failed" if i < 11 else "ok"}
                    for i in range(100)]
        self.assertEqual(analyze60.generation_failure_report(exactly_ten, "M1", 100), [])
        self.assertEqual(analyze60.generation_failure_report(over_ten, "M1", 100)[0]["successes"], 89)
        # M3 fresh is checked as an independent phase and an absent phase is pending.
        m3 = [{"record_index": i, "selection": {"parse_status": "ok"}} for i in range(100)]
        self.assertEqual(analyze60.generation_failure_report(m3, "M3", 100), [])


class BatchLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="part60-test-")
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_sigkill_resume_does_not_resubmit_submitted_chunk(self) -> None:
        marker = self.root / "posts.log"
        code = (
            "import json,sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); import gen60; "
            "def_dummy=None\n"
            "def fake(method,path,body=None):\n"
            " if method=='POST': Path(sys.argv[3]).write_text('POST\\n'); return json.dumps({'id':'batch-safe'})\n"
            " raise AssertionError(path)\n"
            "gen60.submit_phase(Path(sys.argv[2]),'dev','M1','primary',api=fake,kill_after_saved=True)\n"
        )
        proc = subprocess.run([sys.executable, "-c", code, str(HERE), str(self.root), str(marker)], check=False)
        self.assertEqual(proc.returncode, -signal.SIGKILL)
        self.assertEqual(marker.read_text(), "POST\n")
        calls = []
        message = gen60.submit_phase(self.root, "dev", "M1", "primary", api=lambda *args: calls.append(args))
        self.assertIn("batch-safe", message)
        self.assertIn("collect", message)
        self.assertEqual(calls, [])

    def test_cap_refuses_before_post(self) -> None:
        gen60.append_spend(self.root / "spend.jsonl", {"event": "reserve", "batch_id": "old", "reserved_usd": 7.99})
        called = []
        with mock.patch.dict(os.environ, {"PART60_CAP": "8"}):
            with self.assertRaisesRegex(gen60.RunnerError, "cap refused"):
                gen60.submit_phase(self.root, "dev", "M1", "primary", api=lambda *args: called.append(args))
        self.assertEqual(called, [])

    def test_cap_above_eight_needs_explicit_override(self) -> None:
        called = []
        with mock.patch.dict(os.environ, {"PART60_CAP": "80", "PART60_CAP_OVERRIDE": ""}):
            with self.assertRaisesRegex(gen60.RunnerError, "PART60_CAP_OVERRIDE"):
                gen60.submit_phase(self.root, "dev", "M1", "primary", api=lambda *args: called.append(args))
        self.assertEqual(called, [])
        self.assertFalse((self.root / "spend.jsonl").exists())

    def test_missing_api_key_refuses_before_reserving(self) -> None:
        with mock.patch.dict(os.environ, {"ANTHROPIC_CREDITS_KEY": ""}):
            with self.assertRaisesRegex(gen60.RunnerError, "no reserve"):
                gen60.submit_phase(self.root, "dev", "M1", "primary", api=gen60.api_call)
        self.assertFalse((self.root / "spend.jsonl").exists())
        self.assertFalse(gen60.manifest_path(self.root, "dev", "M1", "primary").exists())

    def test_reserve_is_durable_before_post_and_429_retry_reuses_same_tag(self) -> None:
        calls = []
        def api(method: str, path: str, body: object = None) -> str:
            if method == "POST":
                calls.append(body["metadata"]["part60_chunk_tag"])
                spent, _ = gen60.spend_state(self.root / "spend.jsonl")
                self.assertGreater(spent, 0.0)
                if len(calls) == 1:
                    raise gen60.APIError(429, "rate limited")
                return json.dumps({"id": "batch-after-429"})
            raise AssertionError(path)
        with mock.patch("gen60.time.sleep"):
            result = gen60.submit_phase(self.root, "dev", "M1", "primary", api=api)
        self.assertEqual(result, "batch-after-429")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertGreater(gen60.spend_state(self.root / "spend.jsonl")[0], 0.0)

    def test_ambiguous_post_stays_reserved_until_metadata_reconcile(self) -> None:
        tags = []
        posts = []
        def api(method: str, path: str, body: object = None) -> str:
            if method == "POST":
                posts.append(path)
                tags.append(body["metadata"]["part60_chunk_tag"])
                raise gen60.APIError(None, "timeout")
            if method == "GET" and path.startswith("/v1/messages/batches?"):
                return json.dumps({"data": [{"id": "batch-reconciled", "metadata": {"part60_chunk_tag": tags[0]}}], "has_more": False})
            raise AssertionError(path)
        with self.assertRaisesRegex(gen60.RunnerError, "reconcile"):
            gen60.submit_phase(self.root, "dev", "M1", "primary", api=api)
        self.assertGreater(gen60.spend_state(self.root / "spend.jsonl")[0], 0.0)
        message = gen60.submit_phase(self.root, "dev", "M1", "primary", api=api)
        self.assertIn("reconciled batch batch-reconciled", message)
        self.assertEqual(posts, ["/v1/messages/batches"])

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
        def reject(method: str, path: str, body: object = None) -> str:
            raise gen60.APIError(400, "request rejected")
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

    def test_parse_failure_is_written_for_every_request_not_dropped(self) -> None:
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
        self.assertEqual(rows[0]["usage_tokens"], {"input": 3, "output": 5, "calls": 1})
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

    def test_max_tokens_stop_reason_is_recorded_and_fails_collection(self) -> None:
        def api(method: str, path: str, body: object = None) -> str:
            if method == "POST":
                return json.dumps({"id": "batch-truncated"})
            if path.endswith("/batch-truncated"):
                return json.dumps({"processing_status": "ended"})
            if path.endswith("/results"):
                manifest = json.loads(gen60.manifest_path(self.root, "dev", "M1", "primary").read_text())
                chunk = manifest["chunks"][0]
                idmap = json.loads((Path(chunk["directory"]) / "idmap.json").read_text())
                return "\n".join(json.dumps({"custom_id": cid, "result": {"type": "succeeded", "message": {
                    "content": [{"type": "text", "text": "not complete"}], "usage": {"input_tokens": 1, "output_tokens": 1100},
                    "stop_reason": "max_tokens"}}}) for cid in idmap)
            raise AssertionError(path)
        gen60.submit_phase(self.root, "dev", "M1", "primary", api=api)
        with self.assertRaisesRegex(gen60.RunnerError, "max_tokens truncation"):
            gen60.collect_phase(self.root, "dev", "M1", "primary", api=api, poll_seconds=0)
        rows = gen60.read_output(gen60.output_path(self.root, "dev", "M1"))
        first_rec = int(gen60.load_splits()["dev"][0]["record_index"])
        self.assertEqual(rows[first_rec]["raw_parse_status"], "max_tokens")
        manifest = json.loads(gen60.manifest_path(self.root, "dev", "M1", "primary").read_text())
        self.assertEqual(manifest["chunks"][0]["truncations"], 500)

    def test_m6_smoke_requires_nonempty_thinking_and_marks_failure_dropped(self) -> None:
        calls = []
        def reject(method: str, path: str, body: object = None) -> str:
            calls.append((method, path))
            raise gen60.APIError(400, "unsupported thinking field")
        result = gen60.smoke_m6(self.root, api=reject)
        self.assertEqual(result["status"], "dropped")
        self.assertEqual(calls, [("POST", "/v1/messages")])
        submit_calls = []
        message = gen60.submit_phase(self.root, "dev", "M6", "primary", api=lambda *args: submit_calls.append(args))
        self.assertIn("dropped M6", message)
        self.assertEqual(submit_calls, [])
        rows = gen60.read_jsonl(gen60.output_path(self.root, "dev", "M6"))
        self.assertEqual(len(rows), 500)
        self.assertTrue(all(row["parse_status"] == "skipped" for row in rows))
        self.assertTrue(all(row["raw_parse_status"] == "method_dropped" for row in rows))

    def test_m6_submit_refuses_without_smoke_file(self) -> None:
        called = []
        with self.assertRaisesRegex(gen60.RunnerError, "smoke"):
            gen60.submit_phase(self.root, "dev", "M6", "primary", api=lambda *args: called.append(args))
        self.assertEqual(called, [])

    def test_m6_smoke_pass_requires_nonempty_thinking_block(self) -> None:
        def direct(method: str, path: str, body: object = None) -> str:
            self.assertEqual(method, "POST")
            self.assertEqual(path, "/v1/messages")
            self.assertEqual(body["thinking"], {"type": "adaptive"})
            self.assertEqual(body["max_tokens"], 4000)
            return json.dumps({"content": [{"type": "thinking", "thinking": "reason"},
                                             {"type": "text", "text": "answer"}],
                               "stop_reason": "end_turn"})
        result = gen60.smoke_m6(self.root, api=direct)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["thinking_chars"], 6)
        recorded = json.loads((self.root / "out/smoke-M6.json").read_text())
        self.assertEqual(recorded["status"], "pass")

    def test_m6_per_request_batch_error_drops_method(self) -> None:
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
                return "\n".join(json.dumps({"custom_id": cid, "result": {"type": "errored", "error": {"type": "invalid_request_error"}}})
                                 for cid in idmap)
            raise AssertionError(path)
        gen60.submit_phase(self.root, "dev", "M6", "primary", api=api)
        result = gen60.collect_phase(self.root, "dev", "M6", "primary", api=api, poll_seconds=0)
        self.assertIn("dropped M6", result)
        state = json.loads(gen60.method_state_path(self.root, "dev", "M6").read_text())
        self.assertEqual(state["status"], "dropped")
        rows = gen60.read_jsonl(gen60.output_path(self.root, "dev", "M6"))
        self.assertTrue(all(row["parse_status"] == "skipped" for row in rows))

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

    def test_m3_winner_sabotage_goes_red_after_production_selection(self) -> None:
        _write_m3_winner_fixture(self.root)
        code = "\n".join([
            "import sys",
            "from pathlib import Path",
            f"sys.path.insert(0, {str(HERE)!r})",
            "import gen60",
            "root=Path(sys.argv[1])",
            "original=gen60.choose_m3_winners",
            "def wrong_winner(root, split):",
            "    count=original(root, split)",
            "    path=gen60.output_path(root, split, 'M3')",
            "    rows=gen60.read_output(path)",
            "    rows[0]['winner_rank']=1",
            "    gen60.write_output(path, rows)",
            "    return count",
            "gen60.choose_m3_winners=wrong_winner",
            "gen60.choose_m3_winners(root, 'dev')",
            "row=gen60.read_output(gen60.output_path(root, 'dev', 'M3'))[0]",
            "assert row['winner_rank']==2, row",
        ])
        proc = subprocess.run([sys.executable, "-c", code, str(self.root)], text=True, capture_output=True, check=False)
        self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)


class WrapperTests(unittest.TestCase):
    def _prepare_fake_runner(self, root: Path, *, include_m6: bool = True, test_m0: bool = False) -> Path:
        for path in (
            "part60/score60.py", "part60/gen60.py",
            "results/part56-pilot/pilot.jsonl", "results/part56-pilot/label/atlas-labels.jsonl",
            "results/part58/ids.all.json", "results/part58/units.all.json",
            "results/part58/sent.all.jsonl", "results/part58/acts.all.npz",
            "results/part58/gen.all.jsonl", "results/part58/gen.dev.jsonl", "results/part58/gen.test.jsonl",
            "results/part60/thresholds.json",
        ):
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}\n")
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
            "from pathlib import Path",
            "args=sys.argv[1:]",
            "if args[:3] == ['-m','pip','install']: raise SystemExit(0)",
            "if args[:3] == ['-m','pip','freeze']: raise SystemExit(1)",
            "if args and args[0] == '-c':",
            "    code=args[1]",
            "    if 'snapshot_download' in code:",
            "        Path(args[2]).write_text(str(Path(args[2]).parent / 'fake-model'))",
            "        raise SystemExit(0)",
            "    print('torch stub transformers stub')",
            "    raise SystemExit(0)",
            "if args and args[0] == '-u' and len(args) > 1 and args[1] == 'part60/score60.py': raise SystemExit(0)",
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
            "PART60_RETURN_PATH": str(root / "return.tgz"),
            "PART60_INCOMPLETE_PATH": str(root / "incomplete.tgz"),
            "PART60_MAX_RUNTIME_SECONDS": "120"}, text=True, capture_output=True, check=False)

    def test_run60_does_not_require_skipped_m7_and_pip_freeze_is_advisory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-wrapper-full-") as directory:
            root = Path(directory)
            fakebin = self._prepare_fake_runner(root)
            # M7.jsonl is intentionally absent, and a stale ready archive must be removed.
            (root / "return.tgz").write_text("stale success")
            result = self._run_fake_runner(root, fakebin)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("RETURN_READY", result.stdout)
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

    def test_run60_stub_returns_ready_without_network_or_model(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part60-wrapper-") as directory:
            root = Path(directory)
            (root / "results/part60/out/test").mkdir(parents=True)
            (root / "results/part60/out/test/M0.jsonl").write_text("{}\n")
            result = subprocess.run(["bash", str(HERE / "run60.sh")], env={**os.environ,
                "PART60_STUB": "1", "PART60_HOME": str(root),
                "PART60_RETURN_PATH": str(root / "return.tgz"),
                "PART60_INCOMPLETE_PATH": str(root / "incomplete.tgz")}, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("RETURN_READY", result.stdout)
            self.assertTrue((root / "return.tgz").is_file())
            with tarfile.open(root / "return.tgz", "r:gz") as archive:
                names = archive.getnames()
            self.assertTrue(any("logs/score-dev.log" in name for name in names))
            self.assertFalse(any("review" in name or "key" in name.lower() for name in names))


if __name__ == "__main__":
    unittest.main(verbosity=2)
