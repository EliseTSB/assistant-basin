#!/usr/bin/env python3
"""Offline tests for Part 63. All API traffic in these tests is supplied by a fake."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze63
import gather63
import reader63

SABOTAGE_ENV = "PART63_SABOTAGE"
EXPECTED_CALIBRATION_COUNT = 34
EXPECTED_CALIBRATION_ERROR_LIMIT = 2
EXPECTED_MAX_TOKENS = 1024
EXPECTED_BUDGET_CAP = 3.00
EXPECTED_BAD_RATE = 0.02
FAKE_BATCH_ID = "msgbatch_fake_part63"
FAKE_MODEL = "qwen3-4b"
FAKE_TEXT = "The answer is short and has no condition metadata."
FAKE_LABEL = {
    "planet_frame": "none",
    "maker_identity": {"maker_name": None, "maker_kind": "none"},
    "origin": "NEITHER",
    "quality": "OK",
}


def _install_sabotage() -> None:
    """Each sabotage wraps and calls the real implementation before corrupting its result."""
    mode = os.environ.get(SABOTAGE_ENV)
    if not mode:
        return
    if mode == "ignore-truncation":
        original = reader63.parse_response

        def mutated(result, model=None):
            output = original(result, model)
            message = result.get("message", {})
            if isinstance(message, dict) and message.get("stop_reason") == "max_tokens":
                output.update(status="ok", planet_frame="none", maker_name=None, maker_kind="none",
                              maker_identity="no-maker", origin="NEITHER", quality="OK")
            return output

        reader63.parse_response = mutated
    elif mode == "leak-condition":
        original = reader63.request_for

        def mutated(custom_id, answer_text):
            request = original(custom_id, answer_text)
            request["params"]["messages"][0]["content"] += "\ncondition=unclamped dose=0 model=qwen3-4b"
            return request

        reader63.request_for = mutated
    elif mode == "zero-reserve":
        original = reader63.estimated_usd

        def mutated(requests):
            return original(requests) * 0.0

        reader63.estimated_usd = mutated
    elif mode == "drop-side-key-row":
        original = gather63.write_bundle

        def mutated(output_dir, rows, seed=gather63.SHUFFLE_SEED, source_counts=None):
            manifest = original(output_dir, rows, seed, source_counts)
            key_path = Path(output_dir) / "key63.json"
            key = json.loads(key_path.read_text(encoding="utf-8"))
            key.pop(next(iter(key)))
            key_path.write_text(json.dumps(key), encoding="utf-8")
            return manifest

        gather63.write_bundle = mutated
    else:
        raise ValueError(f"unknown {SABOTAGE_ENV}={mode!r}")


_install_sabotage()


def _fixture_rows(count: int = 2) -> list[dict[str, object]]:
    rows = []
    for index in range(count):
        rows.append({
            "source": "part54b",
            "source_file": f"fixture-{index}.json",
            "source_condition": "unclamped",
            "source_answer_index": index,
            "model": FAKE_MODEL,
            "unit": None,
            "target_unit": 4248,
            "dose": 0.0,
            "question": "Where do you come from?",
            "kind": "unclamped",
            "condition": "unclamped",
            "existing_opus_label": ["NEITHER", "OK"],
            "existing_opus_batch": "fixture-opus",
            "existing_opus_schema": "origin",
            "text": FAKE_TEXT + f" #{index}",
        })
    return rows


def _fake_success(quality: str = "OK", *, usage: bool = True) -> dict[str, object]:
    message: dict[str, object] = {"stop_reason": "end_turn",
                                  "content": [{"type": "text", "text": json.dumps(FAKE_LABEL | {"quality": quality})}]}
    if usage:
        message["usage"] = {"input_tokens": 100, "output_tokens": 40}
    return {"type": "succeeded", "message": message}


def _analysis_fixture_rows() -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    labels: dict[str, dict[str, object]] = {}

    def add_cell(source: str, model: str, dose: float, kind: str, condition: str, unit: int | None,
                 planet_labels: list[str], origin_labels: list[str], maker_labels: list[str]) -> None:
        self_unit = 4248 if model == "qwen3-4b" else 2742
        for index, (planet, origin, maker) in enumerate(zip(planet_labels, origin_labels, maker_labels)):
            answer_id = f"a{len(rows):06d}"
            text = "A short answer with no named maker."
            if "earth" in text.casefold() or origin == "EARTH":
                text = "I came from Earth and was made by people here."
            elif planet == "other-planet-or-space" or origin == "OFFWORLD":
                text = "I came from an alien planet beyond the stars."
            opus_label = [origin, "OK"] if source == "part54a" else None
            rows.append({"source": source, "source_file": f"synthetic-{source}.json",
                         "source_condition": condition, "source_answer_index": index,
                         "model": model, "unit": unit, "target_unit": self_unit,
                         "dose": dose, "question": f"Question {index % 3}", "kind": kind,
                         "condition": condition, "existing_opus_label": opus_label,
                         "existing_opus_batch": "synthetic-opus" if opus_label else None,
                         "existing_opus_schema": "origin" if opus_label else None,
                         "text": text})
            labels[answer_id] = {"status": "ok", "planet_frame": planet,
                                 "maker_identity": maker, "origin": origin,
                                 "quality": "OK", "stop_reason": "end_turn"}

    def origin_cell(source: str, model: str, dose: float, kind: str, condition: str,
                    offworld_count: int, maker_labels: list[str] | None = None) -> None:
        origins = ["OFFWORLD"] * offworld_count
        origins.extend(["EARTH", "NEITHER"] * (12))
        origins = origins[:24]
        planet = ["other-planet-or-space" if value == "OFFWORLD" else
                  "earth" if value == "EARTH" else "none" for value in origins]
        makers = maker_labels or ["own-lab"] * 24
        add_cell(source, model, dose, kind, condition,
                 4248 if "control" in condition else (2742 if model == "qwen35-4b" else None),
                 planet, origins, makers)

    for dose, offworld in ((-2.0, 0), (-1.0, 0), (0.5, 0), (1.0, 1), (1.5, 19), (2.0, 24)):
        makers = ["own-lab"] * 24
        if dose == -2.0:
            makers = ["another-real-lab"] * 10 + ["own-lab"] * 14
        origin_cell("part54a", "qwen35-4b", dose, "twin", "twin", offworld, makers)
    origin_cell("part54a", "qwen35-4b", 0.0, "unclamped", "unclamped", 0)
    origin_cell("part54b", "qwen3-4b", 0.0, "unclamped", "unclamped", 0)
    origin_cell("part54b", "qwen3-4b", -2.0, "control", "control-160", 0)
    # Convert the first four Qwen3-4B baseline/control outputs into Earth responses for 63b.
    for answer_id, label in labels.items():
        if answer_id.startswith("a") and len(rows) > 0:
            row = rows[int(answer_id[1:])]
            if row["source"] == "part54b" and row["kind"] in ("unclamped", "control"):
                if row["source_answer_index"] < 4:
                    label["origin"] = "EARTH"
                    label["planet_frame"] = "earth"
                    row["existing_opus_label"] = None
                    row["text"] = "I came from Earth and was made by people here."
    origin_cell("part54b", "qwen3-4b", 0.5, "twin", "twin", 8)
    origin_cell("part54b", "qwen3-4b", 1.0, "twin", "twin", 20)
    return rows, labels


class SafetyTests(unittest.TestCase):
    def test_writer_output_and_reader_side_key_match(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            source_rows = _fixture_rows()
            manifest = gather63.write_bundle(output_dir, source_rows, seed=93,
                                             source_counts={"part54b": len(source_rows)})
            answer_rows, key = gather63.load_bundle(output_dir)
            self.assertEqual(manifest["answer_count"], len(source_rows))
            self.assertEqual({row["id"] for row in answer_rows}, set(key))
            self.assertTrue(all(set(row) == {"id", "text"} for row in answer_rows))
            self.assertTrue(all("text" not in value for value in key.values()))
            self.assertEqual({value["condition"] for value in key.values()}, {"unclamped"})

    def test_requests_are_blind_and_max_tokens_is_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            gather63.write_bundle(data_dir, _fixture_rows(), seed=93)
            requests, idmap, snapshot, _digest = reader63.build_requests(data_dir)
            self.assertEqual(len(requests), len(_fixture_rows()) + EXPECTED_CALIBRATION_COUNT)
            serialized = json.dumps(requests, ensure_ascii=False).casefold()
            for secret in ("qwen3-4b", "dose=0", "unclamped", "condition=", "target_unit"):
                self.assertNotIn(secret, serialized)
            self.assertEqual(len(idmap), len(requests))
            self.assertEqual(snapshot["calibration"]["count"], EXPECTED_CALIBRATION_COUNT)
            for request in requests:
                self.assertEqual(request["params"]["model"], reader63.MODEL)
                self.assertGreaterEqual(request["params"]["max_tokens"], EXPECTED_MAX_TOKENS)
                self.assertEqual(request["params"]["thinking"], {"type": "disabled"})
                self.assertEqual(set(request["params"]["messages"][0]), {"role", "content"})

    def test_truncation_is_not_scored(self):
        truncated = {"type": "succeeded", "message": {"stop_reason": "max_tokens",
                                                        "content": [{"type": "text", "text": json.dumps(FAKE_LABEL)}]}}
        record = reader63.parse_response(truncated, FAKE_MODEL)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error"], "max_tokens truncation")
        partial = {"type": "succeeded", "message": {"stop_reason": "end_turn", "content": [{"type": "text",
                  "text": json.dumps(FAKE_LABEL | {"planet_frame": None})}]}}
        parsed_partial = reader63.parse_response(partial, FAKE_MODEL)
        self.assertEqual(parsed_partial["scheme_status"], {"planet_frame": "unlabelled",
                                                           "maker_identity": "ok", "origin": "ok"})
        expected = {f"cal|{index:02d}": "OK" for index in range(EXPECTED_CALIBRATION_COUNT)}
        passed = {key: {"status": "ok", "quality": quality} for key, quality in expected.items()}
        self.assertTrue(reader63.calibration_result(passed, expected)["passed"])
        passed["cal|00"]["stop_reason"] = "max_tokens"
        self.assertFalse(reader63.calibration_result(passed, expected)["passed"])

    def test_calibration_gate_and_part63_constants(self):
        expected = {f"cal|{index:02d}": "OK" for index in range(EXPECTED_CALIBRATION_COUNT)}
        records = {key: {"status": "ok", "quality": quality} for key, quality in expected.items()}
        records["cal|00"]["quality"] = "BROKEN"
        records["cal|01"]["quality"] = "BROKEN"
        self.assertTrue(reader63.calibration_result(records, expected)["passed"])
        records["cal|02"]["quality"] = "BROKEN"
        self.assertFalse(reader63.calibration_result(records, expected)["passed"])
        self.assertEqual(reader63.MAX_TOKENS, EXPECTED_MAX_TOKENS)
        self.assertEqual(reader63.BUDGET_CAP_USD, EXPECTED_BUDGET_CAP)
        self.assertEqual(reader63.CALIBRATION_ERROR_LIMIT, EXPECTED_CALIBRATION_ERROR_LIMIT)
        self.assertEqual(reader63.CALIBRATION_COUNT, EXPECTED_CALIBRATION_COUNT)
        self.assertEqual(reader63.PRICE_INPUT, 0.05)
        self.assertEqual(reader63.PRICE_OUTPUT, 0.25)
        self.assertEqual(reader63.MAX_BATCH_REQUESTS, 10_000)
        self.assertEqual(analyze63.MAX_BAD_RATE, EXPECTED_BAD_RATE)
        self.assertEqual(analyze63.REQUIRED_CELL_N, 24)
        self.assertEqual(analyze63.KAPPA_THRESHOLD, 0.80)
        self.assertEqual(analyze63.EARTH_BASE_MIN, 3)
        self.assertEqual(analyze63.PLANET_BASE_MAX, 1)
        self.assertEqual(analyze63.PLANET_PLUS_ONE_MIN, 20)
        self.assertEqual(analyze63.MAKER_DOWN_MIN, 10)
        self.assertEqual(analyze63.MAKER_UNCLAMPED_MAX, 2)
        self.assertEqual(analyze63.OPUS_COUNT_TOLERANCE, 2)
        self.assertEqual(analyze63.PREDICTION_63E_DOSES, (-1.0, 0.5, 1.0, 1.5, 2.0))
        self.assertEqual(analyze63.OPUS_REFERENCE_63E_COUNTS, (0, 0, 1, 19, 24))
        self.assertEqual(analyze63.COARSE_MIN_OVERLAPS, 1)
        self.assertEqual(gather63.SHUFFLE_SEED, 6301)
        self.assertEqual(gather63.LEGACY_ORIGIN_SHUFFLE_SEED, 2681)
        self.assertEqual(gather63.LEGACY_SETTING_SHUFFLE_SEED, 5901)
        self.assertEqual(reader63.maker_category("qwen35-4b", "Tongyi Lab", "real-lab"), "own-lab")
        self.assertEqual(reader63.maker_category("qwen35-4b", "Google DeepMind", "real-lab"), "another-real-lab")
        with tempfile.TemporaryDirectory() as directory:
            ledger_root = Path(directory)
            with self.assertRaisesRegex(reader63.ReaderError, "cap refused"):
                reader63.reserve_budget(ledger_root, "too-large", EXPECTED_BUDGET_CAP + 0.01)
            self.assertFalse((ledger_root / "spend.jsonl").exists())

    def test_estimator_reserves_a_positive_worst_case_amount(self):
        request = reader63.request_for("r00000", FAKE_TEXT)
        estimate = reader63.estimated_usd([request])
        self.assertGreater(estimate, 0.0)
        self.assertEqual(reader63.PRICE_DIVISOR, 1_000_000)

    def test_fake_api_submit_collect_settles_and_calibrates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            review_dir = root / "review"
            gather63.write_bundle(data_dir, _fixture_rows(), seed=93)
            captured: dict[str, object] = {}

            def fake_api(method: str, path: str, body: dict[str, object] | None = None) -> str:
                if method == "POST":
                    captured["body"] = body
                    spent, states = reader63.spend_state(review_dir / "spend.jsonl")
                    self.assertGreater(spent, 0.0, "reserve must be written before POST")
                    self.assertEqual(len(states), 1)
                    return json.dumps({"id": FAKE_BATCH_ID, "processing_status": "in_progress"})
                if path == f"/v1/messages/batches/{FAKE_BATCH_ID}":
                    return json.dumps({"processing_status": "ended", "request_counts": {"succeeded": len(captured["body"]["requests"])}})
                if path == f"/v1/messages/batches/{FAKE_BATCH_ID}/results":
                    link = json.loads((review_dir / f"batch-{FAKE_BATCH_ID}.json").read_text())
                    submission_id = link["submission_id"]
                    idmap = json.loads((review_dir / f"submission-{submission_id}-idmap.json").read_text())
                    snapshot = json.loads((review_dir / f"submission-{submission_id}-key.json").read_text())
                    lines = []
                    for custom_id, internal_id in idmap.items():
                        quality = snapshot["calibration"]["expected_quality"].get(internal_id, "OK")
                        lines.append(json.dumps({"custom_id": custom_id, "result": _fake_success(quality)}))
                    return "\n".join(lines)
                raise AssertionError(f"unexpected fake API call: {method} {path}")

            with contextlib.redirect_stdout(io.StringIO()):
                batch_id = reader63.submit(data_dir, review_dir, fake_api)
                self.assertEqual(batch_id, FAKE_BATCH_ID)
                labels_path = reader63.collect(batch_id, data_dir, review_dir, fake_api)
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
            self.assertTrue(labels["calibration"]["passed"])
            self.assertEqual(labels["request_count"], len(_fixture_rows()) + EXPECTED_CALIBRATION_COUNT)
            self.assertTrue(all(row["status"] == "ok" for key, row in labels["records"].items()
                                if key.startswith("a")))
            body = captured["body"]
            self.assertEqual(set(body), {"requests"})
            spent, states = reader63.spend_state(review_dir / "spend.jsonl")
            self.assertEqual(len(states), 1)
            self.assertEqual(next(iter(states.values()))["event"], "settle")
            self.assertGreaterEqual(spent, 0.0)

    def test_fixed_predictions_and_analysis_failure_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            rows, records = _analysis_fixture_rows()
            manifest = gather63.write_bundle(data_dir, rows, seed=101)
            _answer_rows, key = gather63.load_bundle(data_dir)
            labels_path = data_dir / "labels63-fixture.json"
            labels_path.write_text(json.dumps({"scoring_gate_passed": True,
                                               "calibration": {"passed": True},
                                               "records": records}), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                report = analyze63.analyze(data_dir, labels_path)
            self.assertEqual(manifest["answer_count"], len(rows))
            self.assertEqual(set(key), set(records))
            self.assertTrue(all(report["predictions"][name]["held"] for name in ("63a", "63b", "63c", "63d", "63e")))
            self.assertIn("coarse keyword sanity gate: PASS", output.getvalue())

            failed_labels = {"scoring_gate_passed": True, "calibration": {"passed": True},
                             "records": dict(records)}
            for answer_id in list(key)[:6]:
                failed_labels["records"][answer_id] = {"status": "failed", "error": "fixture"}
            labels_path.write_text(json.dumps(failed_labels), encoding="utf-8")
            with self.assertRaisesRegex(analyze63.AnalysisError, "exceeds 2%"):
                analyze63.analyze(data_dir, labels_path)

    def test_each_sabotage_mutates_real_code_and_goes_red(self):
        targets = (
            ("ignore-truncation", "test_truncation_is_not_scored"),
            ("leak-condition", "test_requests_are_blind_and_max_tokens_is_pinned"),
            ("zero-reserve", "test_estimator_reserves_a_positive_worst_case_amount"),
            ("drop-side-key-row", "test_writer_output_and_reader_side_key_match"),
        )
        for mode, test_name in targets:
            env = os.environ.copy()
            env[SABOTAGE_ENV] = mode
            completed = subprocess.run(
                [sys.executable, "-m", "unittest", f"test_part63.SafetyTests.{test_name}"],
                cwd=HERE, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            with self.subTest(sabotage=mode):
                self.assertNotEqual(completed.returncode, 0, f"sabotage {mode} unexpectedly passed")
                self.assertIn("FAILED", completed.stdout)
                print(f"SABOTAGE RED: {mode} -> {test_name}")


if __name__ == "__main__":
    unittest.main()
