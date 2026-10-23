#!/usr/bin/env python3
"""Offline tests for Part 63. All API traffic in these tests is supplied by a fake."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze63
import gather63
import reader63

SABOTAGE_ENV = "PART63_SABOTAGE"
EXPECTED_CALIBRATION_COUNT = 34
EXPECTED_CALIBRATION_ERROR_LIMIT = 2
EXPECTED_MAX_TOKENS = 1024
EXPECTED_BUDGET_CAP = 4.5
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
    elif mode == "dose-zero-baseline":
        original = analyze63._selected_rows

        def mutated(key, records, *, scheme, text_by_id=None, dedupe_by_text=False, **criteria):
            if criteria.get("kind") == "unclamped":
                criteria["dose"] = 0.0
            return original(key, records, scheme=scheme, text_by_id=text_by_id,
                            dedupe_by_text=dedupe_by_text, **criteria)

        analyze63._selected_rows = mutated
    elif mode == "alias-shortcut":
        original = reader63.maker_category

        def mutated(model, maker_name, maker_kind, *, use_aliases=False):
            if model == "qwen35-4b" and maker_kind == "real-lab" and maker_name:
                normalized = reader63._normalize_maker_name(maker_name)
                if any(alias in normalized for alias in reader63.MAKER_ALIASES[model]):
                    return "own-lab"
            return original(model, maker_name, maker_kind, use_aliases=use_aliases)

        reader63.maker_category = mutated
    elif mode == "temperature-default":
        original = reader63.request_for

        def mutated(custom_id, answer_text):
            request = original(custom_id, answer_text)
            request["params"]["temperature"] = 0
            return request

        reader63.request_for = mutated
    elif mode == "keyword-substring":
        analyze63.PROXY_OFFWORLD = __import__("re").compile(
            r"alien|extraterrestrial|planet|galax|star|space|interstellar|species", __import__("re").I
        )
    elif mode == "reject-63c-flat-step":
        original = analyze63.evaluate_predictions

        def mutated(*args, **kwargs):
            output = original(*args, **kwargs)
            baseline = output.get("63c", {})
            if baseline.get("+0.5", [None])[0] == baseline.get("+1", [None])[0]:
                baseline["held"] = False
            return output

        analyze63.evaluate_predictions = mutated
    elif mode == "loosen-token-estimate":
        reader63.INPUT_BYTES_PER_TOKEN = 6
    elif mode == "drop-label-fanout":
        original = reader63.collect

        def mutated(*args, **kwargs):
            labels_path = original(*args, **kwargs)
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
            snapshot = next(json.loads(path.read_text(encoding="utf-8"))
                            for path in (args[2]).glob("submission-*-key.json"))
            for answer_ids in snapshot["answer_ids_by_text"].values():
                for answer_id in answer_ids[1:]:
                    labels["records"].pop(answer_id, None)
            labels_path.write_text(json.dumps(labels), encoding="utf-8")
            return labels_path

        reader63.collect = mutated
    elif mode == "skip-retire":
        reader63.retire_attempt = lambda *_args, **_kwargs: None
    else:
        raise ValueError(f"unknown {SABOTAGE_ENV}={mode!r}")


_install_sabotage()


def _fixture_rows(count: int = 120) -> list[dict[str, object]]:
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


def _add_duplicate_records(records: dict[str, dict[str, object]], count: int = 100) -> list[dict[str, str]]:
    pairs = []
    for index, answer_id in enumerate(sorted(key for key in records if key.startswith("a"))[:count]):
        duplicate_id = f"dup|{answer_id}"
        records[duplicate_id] = dict(records[answer_id])
        pairs.append({"answer_id": answer_id, "duplicate_id": duplicate_id})
    return pairs


def _analysis_fixture_rows() -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    labels: dict[str, dict[str, object]] = {}

    def add_cell(source: str, model: str, dose: float, kind: str, condition: str, unit: int | None,
                 planet_labels: list[str], origin_labels: list[str], maker_labels: list[str]) -> None:
        self_unit = 4248 if model == "qwen3-4b" else 2742
        for index, (planet, origin, maker) in enumerate(zip(planet_labels, origin_labels, maker_labels)):
            answer_id = f"a{len(rows):06d}"
            text = f"A short answer with no named maker for {source} {model} {dose} {kind} answer {index}."
            if "earth" in text.casefold() or origin == "EARTH":
                text = "I came from Earth and was made by people here."
            elif planet == "other-planet-or-space" or origin == "OFFWORLD":
                text = "I came from an alien planet beyond the stars."
            text += f" Sample {source} {model} {dose} {kind} {index}."
            opus_label = [origin, "OK"] if source == "part54a" else None
            rows.append({"source": source, "source_file": f"synthetic-{source}.json",
                         "source_condition": condition, "source_answer_index": index,
                         "model": model, "unit": unit, "target_unit": self_unit,
                         "dose": dose, "question": f"Question {index % 3}", "kind": kind,
                         "condition": condition, "existing_opus_label": opus_label,
                         "existing_opus_batch": "synthetic-opus" if opus_label else None,
                         "existing_opus_schema": "origin" if opus_label else None,
                         "text": text})
            alias_maker = "own-lab" if (source == "part54a" and model == "qwen35-4b"
                                         and dose == -2.0 and maker == "another-real-lab") else maker
            labels[answer_id] = {"status": "ok", "planet_frame": planet,
                                 "maker_identity": maker, "maker_identity_alias": alias_maker, "origin": origin,
                                 "maker_name": (None if maker == "no-maker" else maker),
                                 "maker_kind": ("none" if maker == "no-maker" else
                                                "real-lab" if maker in ("own-lab", "another-real-lab")
                                                else "invented-or-non-lab"),
                                 "scheme_status": {"planet_frame": "ok", "maker_identity": "ok", "origin": "ok"},
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
    origin_cell("part54a", "qwen35-4b", -3.0, "unclamped", "unclamped", 0)
    origin_cell("part54b", "qwen3-4b", -3.0, "unclamped", "unclamped", 0)
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
                    row["text"] = ("I came from Earth and was made by people here. "
                                   f"Sample {row['source']} {row['model']} {row['dose']} {row['kind']} "
                                   f"{row['source_answer_index']}.")
    origin_cell("part54b", "qwen3-4b", 0.5, "twin", "twin", 8)
    origin_cell("part54b", "qwen3-4b", 1.0, "twin", "twin", 20)
    return rows, labels


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self._ledger_tmp = tempfile.TemporaryDirectory(dir=HERE)
        self.addCleanup(self._ledger_tmp.cleanup)
        reader63.DEFAULT_LEDGER_PATH = Path(self._ledger_tmp.name) / "paid-ledger.jsonl"

    def test_submit_cli_only_prints_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "data"
            review_dir = Path(directory) / "review"
            gather63.write_bundle(data_dir, _fixture_rows(100), seed=93)
            with mock.patch.object(reader63, "submit", side_effect=AssertionError("manual CLI must not POST")), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                reader63.main(["submit", "--data-dir", str(data_dir), "--review-dir", str(review_dir)])
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["command"], "submit")
            self.assertGreater(plan["request_count"], 0)
            self.assertFalse(review_dir.exists(), "plan-only submit must not create a reserve or review state")

    def test_submit_cli_restored_post_mutation_goes_red(self):
        with tempfile.TemporaryDirectory() as directory:
            copied_part63 = Path(directory) / "part63"
            shutil.copytree(HERE, copied_part63, ignore=shutil.ignore_patterns("__pycache__", "review"))
            source_path = copied_part63 / "reader63.py"
            source = source_path.read_text(encoding="utf-8")
            self.assertIn("print_submit_plan(args.data_dir)", source)
            source_path.write_text(source.replace("print_submit_plan(args.data_dir)",
                                                  "submit(args.data_dir, args.review_dir)", 1),
                                   encoding="utf-8")
            (copied_part63.parent / "part59").mkdir()
            shutil.copy2(HERE.parent / "part59" / "blind_setting.py",
                         copied_part63.parent / "part59" / "blind_setting.py")
            completed = subprocess.run([sys.executable, "-m", "unittest",
                "test_part63.SafetyTests.test_submit_cli_only_prints_plan"],
                cwd=copied_part63, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            self.assertNotEqual(completed.returncode, 0, "restoring the paid submit path must break the plan-only test")
            self.assertIn("FAILED", completed.stdout)

    def test_retry_post_is_not_repeated_after_simulated_kill(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            data_dir, review_dir, ledger = root / "data", root / "review", root / "ledger.jsonl"
            reader63.DEFAULT_LEDGER_PATH = ledger
            gather63.write_bundle(data_dir, _fixture_rows(100), seed=93)
            posts = []
            retry_posts = []
            retry_killed = {"done": False}

            def fake_api(method, path, body=None):
                if method == "POST":
                    posts.append(body)
                    submissions = [json.loads(path.read_text()) for path in review_dir.glob("submission-*.json")
                                   if not path.name.endswith(("-idmap.json", "-key.json"))]
                    if any(submission.get("status") == "posting"
                           and submission.get("retry_of_submission_id") for submission in submissions):
                        retry_posts.append(len(body["requests"]))
                    return json.dumps({"id": f"msgbatch_retry_kill_{len(posts)}",
                                       "processing_status": "in_progress"})
                batch_id = path.split("/batches/")[1].split("/")[0]
                if path == f"/v1/messages/batches/{batch_id}":
                    return json.dumps({"processing_status": "ended"})
                if path.endswith("/results"):
                    link = json.loads((review_dir / f"batch-{batch_id}.json").read_text())
                    sub = json.loads((review_dir / link["submission_file"]).read_text())
                    idmap = json.loads((review_dir / sub["idmap_file"]).read_text())
                    snap = json.loads((review_dir / sub["key_file"]).read_text())
                    rows = []
                    for custom_id, internal in idmap.items():
                        result = _fake_success(snap["calibration"]["expected_quality"].get(internal, "OK"))
                        if batch_id.endswith("_1") and internal.startswith("cal|"):
                            result = {"type": "succeeded", "message": {"stop_reason": "end_turn",
                                "content": [{"type": "text", "text": json.dumps(FAKE_LABEL | {"planet_frame": None})}],
                                "usage": {"input_tokens": 100, "output_tokens": 40}}}
                        rows.append(json.dumps({"custom_id": custom_id, "result": result}))
                    return "\n".join(rows)
                raise AssertionError((method, path))

            real_submit = reader63.submit
            def die_after_retry(*args, **kwargs):
                batch = real_submit(*args, **kwargs)
                if kwargs.get("retry_of_submission_id") and not retry_killed["done"]:
                    retry_killed["done"] = True
                    raise RuntimeError("simulated process death after retry POST return")
                return batch

            with mock.patch.dict(os.environ, {"ANTHROPIC_CREDITS_KEY": "test-key"}), \
                    mock.patch.object(reader63, "context", return_value=object()), \
                    mock.patch.object(reader63, "submit", side_effect=die_after_retry), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "simulated process death"):
                    reader63.run_all(data_dir, review_dir, fake_api, poll_interval=0, poll_bound=1,
                                     sleep=lambda _seconds: None, ledger_path=ledger)
                self.assertEqual(len(posts), 2, "root and one retry must have been posted before the kill")
                reader63.run_all(data_dir, review_dir, fake_api, poll_interval=0, poll_bound=1,
                                 sleep=lambda _seconds: None, ledger_path=ledger)
            retry_submissions = [json.loads(path.read_text()) for path in review_dir.glob("submission-*.json")
                                 if not path.name.endswith(("-idmap.json", "-key.json"))
                                 and json.loads(path.read_text()).get("retry_of_submission_id")]
            self.assertEqual(len(retry_submissions), 1)
            self.assertEqual(len(posts), 3, "recovery should resume the next chunk without another retry")
            self.assertEqual(len(retry_posts), 1, "the saved failed-only retry must be reused after recovery")
            self.assertGreater(retry_posts[0], 0)

    def test_retry_dedupe_copied_code_mutation_goes_red(self):
        with tempfile.TemporaryDirectory() as directory:
            copied_part63 = Path(directory) / "part63"
            shutil.copytree(HERE, copied_part63, ignore=shutil.ignore_patterns("__pycache__", "review"))
            source = (copied_part63 / "reader63.py").read_text(encoding="utf-8")
            self.assertIn("if existing_retries:", source)
            (copied_part63 / "reader63.py").write_text(source.replace("if existing_retries:", "if False:", 1),
                                                        encoding="utf-8")
            (copied_part63.parent / "part59").mkdir()
            shutil.copy2(HERE.parent / "part59" / "blind_setting.py",
                         copied_part63.parent / "part59" / "blind_setting.py")
            completed = subprocess.run([sys.executable, "-m", "unittest",
                "test_part63.SafetyTests.test_retry_post_is_not_repeated_after_simulated_kill"],
                cwd=copied_part63, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            self.assertNotEqual(completed.returncode, 0, "removing retry dedupe must make the kill row fail")
            self.assertIn("FAILED", completed.stdout)

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

    def test_each_distinct_text_is_submitted_once_and_maps_back_to_rows(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            data_dir = Path(directory)
            rows = _fixture_rows(240)
            for index, row in enumerate(rows):
                row["text"] = f"paired repeated answer {index // 2}"
            gather63.write_bundle(data_dir, rows, seed=23)
            requests, idmap, snapshot, _digest = reader63.build_requests(data_dir)
            self.assertEqual(len(snapshot["answer_ids_by_text"]), 120)
            self.assertEqual(len(requests), 120 + reader63.DUPLICATE_COUNT + EXPECTED_CALIBRATION_COUNT)
            self.assertTrue(all(len(ids) == 2 for ids in snapshot["answer_ids_by_text"].values()))
            primary = {value for value in idmap.values() if value.startswith("a")}
            self.assertEqual(len(primary), 120)

    def test_requests_are_blind_and_max_tokens_is_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            gather63.write_bundle(data_dir, _fixture_rows(), seed=93)
            requests, idmap, snapshot, _digest = reader63.build_requests(data_dir)
            self.assertEqual(len(requests), len(_fixture_rows()) + EXPECTED_CALIBRATION_COUNT + reader63.DUPLICATE_COUNT)
            serialized = json.dumps(requests, ensure_ascii=False).casefold()
            for secret in ("qwen3-4b", "dose=0", "unclamped", "condition=", "target_unit"):
                self.assertNotIn(secret, serialized)
            self.assertEqual(len(idmap), len(requests))
            self.assertEqual(snapshot["calibration"]["count"], EXPECTED_CALIBRATION_COUNT)
            duplicate_meta = snapshot["duplicate_consistency"]
            self.assertEqual(duplicate_meta["count"], 100)
            self.assertEqual(duplicate_meta["seed"], reader63.DUPLICATE_SEED)
            self.assertEqual(len(duplicate_meta["pairs"]), 100)
            self.assertTrue(all(row["duplicate_id"] in idmap.values() for row in duplicate_meta["pairs"]))
            self.assertTrue(all(row["answer_id"] in idmap.values() for row in duplicate_meta["pairs"]))
            self.assertFalse(any(internal_id.startswith("dup|") for internal_id in snapshot["key"]))
            request_positions = {internal_id: position for position, internal_id in enumerate(idmap.values())}
            self.assertTrue(all(request_positions[row["answer_id"]] != request_positions[row["duplicate_id"]]
                                for row in duplicate_meta["pairs"]))
            for request in requests:
                self.assertEqual(request["params"]["model"], reader63.MODEL)
                self.assertGreaterEqual(request["params"]["max_tokens"], EXPECTED_MAX_TOKENS)
                self.assertNotIn("temperature", request["params"])  # Haiku 5.5 rejects it (HTTP 400, 2026-10-23)
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
        self.assertEqual(reader63.QWEN35_REGISTERED_LAB_NAMES,
                         ("alibaba", "alibaba cloud", "tongyi", "tongyi lab", "qwen team", "通义", "阿里"))
        self.assertEqual(reader63.MAX_BATCH_REQUESTS, 10_000)
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
        self.assertIn("mentions a planet, a place of origin, or a world", reader63.RUBRIC)
        self.assertEqual(gather63.SHUFFLE_SEED, 6301)
        self.assertEqual(gather63.LEGACY_ORIGIN_SHUFFLE_SEED, 2681)
        self.assertEqual(gather63.LEGACY_SETTING_SHUFFLE_SEED, 5901)
        self.assertEqual(reader63.maker_category("qwen35-4b", "Tongyi Lab", "real-lab"), "own-lab")
        self.assertEqual(reader63.maker_category("qwen35-4b", "Alibaba Cloud AI", "real-lab"), "own-lab")
        self.assertEqual(reader63.maker_category("qwen35-4b", "阿里通义", "real-lab"), "own-lab")
        self.assertEqual(reader63.maker_category("qwen35-4b", "Alibaba", "real-lab", use_aliases=True), "another-real-lab")
        self.assertEqual(analyze63.PROXY_OFFWORLD.search("I will start in the workspace."), None)
        with tempfile.TemporaryDirectory() as directory:
            ledger_root = Path(directory)
            with self.assertRaisesRegex(reader63.ReaderError, "cap refused"):
                reader63.reserve_budget(ledger_root, "too-large", EXPECTED_BUDGET_CAP + 0.01)
            self.assertFalse(reader63.DEFAULT_LEDGER_PATH.exists())
            explicit_ledger = Path(self._ledger_tmp.name) / "retire-explicit.jsonl"
            attempt_id = reader63.Ledger(str(explicit_ledger), reader63.SHARED_CAP_USD).reserve(
                0.01, "part63", {"submission_id": "operator-test"})
            reader63.retire_attempt(attempt_id, "test operator", "fake preflight attempt",
                                    ledger_path=explicit_ledger)
            retired_state = reader63.Ledger(str(explicit_ledger), reader63.SHARED_CAP_USD).state()
            self.assertEqual(retired_state["reserved"], 0.0)
            self.assertEqual(retired_state["settled"], 0.01)
            spent, states = reader63.spend_state(explicit_ledger)
            self.assertEqual(spent, 0.01)
            self.assertEqual(next(iter(states.values()))["effective_usd"], 0.01)

    def test_estimator_reserves_a_positive_worst_case_amount(self):
        request = reader63.request_for("r00000", FAKE_TEXT)
        estimate = reader63.estimated_usd([request])
        self.assertGreater(estimate, 0.0)
        self.assertEqual(reader63.PRICE_DIVISOR, 1_000_000)
        input_bytes = len(json.dumps(request, ensure_ascii=False).encode("utf-8"))
        expected = ((-(-input_bytes // 3)) * reader63.PRICE_INPUT
                    + request["params"]["max_tokens"] * reader63.PRICE_OUTPUT) / reader63.PRICE_DIVISOR
        self.assertAlmostEqual(estimate, expected)
        full_requests, _idmap, _snapshot, _digest = reader63.build_requests(reader63.DEFAULT_DATA_DIR)
        self.assertLessEqual(2 * reader63.estimated_usd(full_requests), EXPECTED_BUDGET_CAP)

    def test_63c_allows_a_flat_monotonic_step(self):
        rows, records = _analysis_fixture_rows()
        data_dir = Path(tempfile.mkdtemp(dir=HERE))
        self.addCleanup(shutil.rmtree, data_dir)
        gather63.write_bundle(data_dir, rows, seed=101)
        _answers, key = gather63.load_bundle(data_dir)
        texts = {row["id"]: row["text"] for row in _answers}
        plus_half = [answer_id for answer_id, meta in key.items()
                     if meta.get("source") == "part54b" and meta.get("model") == "qwen3-4b"
                     and meta.get("kind") == "twin" and meta.get("dose") == 0.5]
        for answer_id in plus_half[:20]:
            records[answer_id]["planet_frame"] = "other-planet-or-space"
        with contextlib.redirect_stdout(io.StringIO()):
            result = analyze63.evaluate_predictions(key, records, texts)
        self.assertEqual(result["63c"]["+0.5"][0], result["63c"]["+1"][0])
        self.assertTrue(result["63c"]["held"])

    def test_duplicate_consistency_exact_match_and_ninety_percent_rule(self):
        records = {f"a{index:06d}": {"status": "ok", "planet_frame": "none", "origin": "NEITHER",
                                      "maker_name": None, "maker_kind": "none",
                                      "scheme_status": {"planet_frame": "ok", "origin": "ok",
                                                         "maker_identity": "ok"}}
                   for index in range(100)}
        pairs = _add_duplicate_records(records)
        self.assertEqual(len(pairs), 100)
        self.assertEqual(analyze63.duplicate_consistency(pairs, records)["origin"]["agreement"], 1.0)
        for pair in pairs[:11]:
            records[pair["duplicate_id"]]["planet_frame"] = "earth"
        report = analyze63.duplicate_consistency(pairs, records)
        self.assertEqual(report["planet_frame"]["agreement"], 0.89)
        self.assertTrue(report["planet_frame"]["unreliable_reader"])
        self.assertFalse(report["origin"]["unreliable_reader"])

    def test_fake_api_submit_collect_settles_and_calibrates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            review_dir = root / "review"
            ledger_path = Path(self._ledger_tmp.name) / "explicit-retry-ledger.jsonl"
            gather63.write_bundle(data_dir, _fixture_rows(), seed=93)
            captured: dict[str, object] = {}

            def fake_api(method: str, path: str, body: dict[str, object] | None = None) -> str:
                if method == "POST":
                    captured["body"] = body
                    intent_path = next(path for path in review_dir.glob("submission-*.json")
                                       if not path.name.endswith(("-idmap.json", "-key.json")))
                    intent = json.loads(intent_path.read_text())
                    request_map = json.loads((review_dir / intent["idmap_file"]).read_text())
                    self.assertEqual(intent["status"], "posting")
                    self.assertEqual(intent["request_count"], len(request_map))
                    self.assertEqual(set(intent["request_ids"]), set(request_map.values()))
                    self.assertTrue((review_dir / intent["key_file"]).is_file())
                    spent, states = reader63.spend_state(ledger_path)
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

            with mock.patch.dict(os.environ, {"ANTHROPIC_CREDITS_KEY": "test-key"}), \
                    mock.patch.object(reader63, "context", return_value=object()), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                batch_id = reader63.submit(data_dir, review_dir, fake_api, ledger_path=ledger_path)
                self.assertEqual(batch_id, FAKE_BATCH_ID)
                labels_path = reader63.collect(batch_id, data_dir, review_dir, fake_api,
                                               ledger_path=ledger_path)
            self.assertIn("actual cost=$", output.getvalue())
            self.assertIn("vs reserve=$", output.getvalue())
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
            self.assertTrue(labels["calibration"]["passed"])
            self.assertEqual(labels["request_count"], len(_fixture_rows()) + EXPECTED_CALIBRATION_COUNT
                             + reader63.DUPLICATE_COUNT)
            self.assertEqual(len(labels["duplicate_consistency"]["pairs"]), reader63.DUPLICATE_COUNT)
            self.assertTrue(set(json.loads(
                next(review_dir.glob("submission-*-key.json")).read_text())["key"]).issubset(labels["records"]))
            self.assertTrue(all(labels["records"][answer_id]["status"] == "ok"
                                for answer_id in json.loads(
                                    next(review_dir.glob("submission-*-key.json")).read_text())["key"]))
            stale = dict(labels)
            stale["request_count"] += 1
            labels_path.write_text(json.dumps(stale), encoding="utf-8")
            with self.assertRaisesRegex(reader63.ReaderError, "stale or mismatched labels"):
                reader63.collect(FAKE_BATCH_ID, data_dir, review_dir, fake_api,
                                ledger_path=ledger_path)
            body = captured["body"]
            self.assertEqual(set(body), {"requests"})
            spent, states = reader63.spend_state(ledger_path)
            self.assertEqual(len(states), 1)
            self.assertEqual(next(iter(states.values()))["event"], "settled")
            self.assertGreaterEqual(spent, 0.0)

    def test_run_all_polls_and_settles_each_chunk_before_posting_next(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir, review_dir = root / "data", root / "review"
            gather63.write_bundle(data_dir, _fixture_rows(100), seed=93)
            posted = []

            def fake_api(method, path, body=None):
                if method == "POST":
                    _total, states = reader63.spend_state(reader63.DEFAULT_LEDGER_PATH)
                    self.assertEqual(sum(row["event"] == "reserved" for row in states.values()), 1,
                                     "more than the current chunk is still reserved")
                    self.assertEqual(sum(row["event"] == "settled" for row in states.values()), len(posted),
                                    "next chunk posted before prior chunk settled")
                    batch_id = f"msgbatch_run_all_{len(posted)}"
                    posted.append((batch_id, body))
                    return json.dumps({"id": batch_id, "processing_status": "in_progress"})
                batch_id = path.split("/batches/")[1].split("/")[0]
                if path.endswith("/results"):
                    link = json.loads((review_dir / f"batch-{batch_id}.json").read_text())
                    sub = json.loads((review_dir / link["submission_file"]).read_text())
                    idmap = json.loads((review_dir / sub["idmap_file"]).read_text())
                    snapshot = json.loads((review_dir / sub["key_file"]).read_text())
                    rows = []
                    for custom_id, internal_id in idmap.items():
                        result = _fake_success(snapshot["calibration"]["expected_quality"].get(internal_id, "OK"))
                        result["message"]["usage"] = {"input_tokens": 1024, "output_tokens": 1024}
                        rows.append(json.dumps({"custom_id": custom_id, "result": result}))
                    return "\n".join(rows)
                if path == f"/v1/messages/batches/{batch_id}":
                    return json.dumps({"processing_status": "ended"})
                raise AssertionError(f"unexpected API call: {method} {path}")

            with mock.patch.dict(os.environ, {"ANTHROPIC_CREDITS_KEY": "test-key"}), \
                    mock.patch.object(reader63, "context", return_value=object()), \
                    contextlib.redirect_stdout(io.StringIO()):
                output = reader63.run_all(data_dir, review_dir, fake_api, poll_interval=0,
                                          poll_bound=1, sleep=lambda _seconds: None)
            labels = json.loads(output.read_text())
            self.assertEqual(len(posted), 2)
            self.assertTrue(labels["scoring_gate_passed"])
            self.assertEqual(len(labels["duplicate_consistency"]["pairs"]), reader63.DUPLICATE_COUNT)
            _total, states = reader63.spend_state(reader63.DEFAULT_LEDGER_PATH)
            self.assertEqual(len(states), 2)
            self.assertTrue(all(row["event"] == "settled" for row in states.values()))
            self.assertTrue(any(row["effective_usd"] > row["reserved_usd"] for row in states.values()),
                            "run-all must settle actual above reserve and advance while both caps permit it")

    def test_run_all_poll_timeout_keeps_reservation_and_refuses_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir, review_dir = root / "data", root / "review"
            gather63.write_bundle(data_dir, _fixture_rows(100), seed=93)
            posts = []
            complete = {"value": False}

            def fake_api(method, path, body=None):
                if method == "POST":
                    posts.append(body)
                    batch_id = f"msgbatch_never_ends_{len(posts)}"
                    return json.dumps({"id": batch_id, "processing_status": "in_progress"})
                batch_id = path.split("/batches/")[1].split("/")[0]
                if path == f"/v1/messages/batches/{batch_id}":
                    return json.dumps({"processing_status": "ended" if complete["value"] else "in_progress"})
                if path == f"/v1/messages/batches/{batch_id}/results":
                    link = json.loads((review_dir / f"batch-{batch_id}.json").read_text())
                    sub = json.loads((review_dir / link["submission_file"]).read_text())
                    idmap = json.loads((review_dir / sub["idmap_file"]).read_text())
                    snapshot = json.loads((review_dir / sub["key_file"]).read_text())
                    return "\n".join(json.dumps({
                        "custom_id": custom_id,
                        "result": _fake_success(snapshot["calibration"]["expected_quality"].get(internal_id, "OK")),
                    }) for custom_id, internal_id in idmap.items())
                raise AssertionError(f"unexpected API call: {method} {path}")

            with mock.patch.dict(os.environ, {"ANTHROPIC_CREDITS_KEY": "test-key"}), \
                    mock.patch.object(reader63, "context", return_value=object()), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(reader63.ReaderError, "did not end"):
                    reader63.run_all(data_dir, review_dir, fake_api, poll_interval=0,
                                     poll_bound=0, sleep=lambda _seconds: None)
            self.assertEqual(len(posts), 1)
            _total, states = reader63.spend_state(reader63.DEFAULT_LEDGER_PATH)
            self.assertEqual(len(states), 1)
            self.assertEqual(next(iter(states.values()))["event"], "reserved")
            complete["value"] = True
            with mock.patch.dict(os.environ, {"ANTHROPIC_CREDITS_KEY": "test-key"}), \
                    mock.patch.object(reader63, "context", return_value=object()), \
                    contextlib.redirect_stdout(io.StringIO()):
                recovered = reader63.run_all(data_dir, review_dir, fake_api, poll_interval=0,
                                               poll_bound=1, sleep=lambda _seconds: None)
            self.assertTrue(recovered.is_file())
            self.assertEqual(len(posts), 2, "resume collects the saved batch, then may submit the next chunk")
            _total, states = reader63.spend_state(reader63.DEFAULT_LEDGER_PATH)
            self.assertEqual(len(states), 2)
            self.assertTrue(all(row["event"] == "settled" for row in states.values()))

    def test_run_all_sigkill_recovery_matrix_and_retire_path(self):
        phases = ("reserve_before_submission_write", "posting_before_post",
                  "accepted_before_response", "response_saved_before_submission_update")
        child = r'''import json, os, signal, sys
from pathlib import Path
import reader63
data_dir, review_dir, ledger = map(Path, sys.argv[1:4])
phase = sys.argv[4]
reader63.DEFAULT_LEDGER_PATH = ledger
os.environ["ANTHROPIC_CREDITS_KEY"] = "test-key"
reader63.context = lambda: object()
original = reader63.write_json
def kill():
    os.kill(os.getpid(), signal.SIGKILL)
def hooked(path, value):
    path = Path(path)
    if phase == "reserve_before_submission_write" and path.name.startswith("submission-") and isinstance(value, dict) and value.get("attempt_id"):
        kill()
    if phase == "response_saved_before_submission_update" and path.name.startswith("submission-") and isinstance(value, dict) and value.get("status") == "submitted":
        kill()
    return original(path, value)
reader63.write_json = hooked
def api(method, path, body=None):
    if method == "POST":
        if phase == "posting_before_post": kill()
        if phase == "accepted_before_response":
            marker = review_dir / "accepted-posts.txt"
            with marker.open("a") as stream: stream.write("accepted\n"); stream.flush(); os.fsync(stream.fileno())
            kill()
        return json.dumps({"id": "msgbatch_kill_matrix", "processing_status": "in_progress"})
    raise AssertionError((method, path))
reader63.run_all(data_dir, review_dir, api, poll_interval=0, poll_bound=1,
                 sleep=lambda _seconds: None, ledger_path=ledger)
'''
        for phase in phases:
            with self.subTest(kill_point=phase), tempfile.TemporaryDirectory(dir=HERE) as directory:
                root = Path(directory)
                data_dir, review_dir, ledger = root / "data", root / "review", root / "ledger.jsonl"
                reader63.DEFAULT_LEDGER_PATH = ledger
                gather63.write_bundle(data_dir, _fixture_rows(100), seed=93)
                completed = subprocess.run([sys.executable, "-c", child, str(data_dir), str(review_dir),
                                            str(ledger), phase], cwd=HERE, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, check=False)
                self.assertEqual(completed.returncode, -9, completed.stderr.decode(errors="replace"))
                _spent, states = reader63.spend_state(ledger)
                self.assertEqual(len(states), 1)
                initial = next(iter(states.values()))
                if phase in ("posting_before_post", "accepted_before_response"):
                    self.assertEqual(initial["event"], "reserved")
                    if phase == "accepted_before_response":
                        self.assertEqual((review_dir / "accepted-posts.txt").read_text().splitlines(), ["accepted"])
                    row = next(read for read in review_dir.glob("submission-*.json")
                               if not read.name.endswith(("-idmap.json", "-key.json")))
                    metadata = json.loads(row.read_text())
                    with self.assertRaisesRegex(reader63.ReaderError, "no batch ID on disk"):
                        with mock.patch.dict(os.environ, {"ANTHROPIC_CREDITS_KEY": "test-key"}), \
                                mock.patch.object(reader63, "context", return_value=object()):
                            reader63.run_all(data_dir, review_dir, lambda *_args: self.fail("must not POST"),
                                             poll_interval=0, poll_bound=1, sleep=lambda _seconds: None,
                                             ledger_path=ledger)
                    reader63.retire_attempt(metadata["attempt_id"], "test operator", "unknown POST outcome")
                posted = []
                def fake_api(method, path, body=None):
                    if method == "POST":
                        posted.append(body)
                        return json.dumps({"id": f"msgbatch_recovered_{len(posted)}",
                                           "processing_status": "in_progress"})
                    batch_id = path.split("/batches/")[1].split("/")[0]
                    if path == f"/v1/messages/batches/{batch_id}":
                        return json.dumps({"processing_status": "ended"})
                    if path.endswith("/results"):
                        link = json.loads((review_dir / f"batch-{batch_id}.json").read_text())
                        sub = json.loads((review_dir / link["submission_file"]).read_text())
                        idmap = json.loads((review_dir / sub["idmap_file"]).read_text())
                        snap = json.loads((review_dir / sub["key_file"]).read_text())
                        return "\n".join(json.dumps({"custom_id": cid,
                            "result": _fake_success(snap["calibration"]["expected_quality"].get(internal, "OK"))})
                            for cid, internal in idmap.items())
                    raise AssertionError((method, path))
                with mock.patch.dict(os.environ, {"ANTHROPIC_CREDITS_KEY": "test-key"}), \
                        mock.patch.object(reader63, "context", return_value=object()), \
                        contextlib.redirect_stdout(io.StringIO()):
                    output = reader63.run_all(data_dir, review_dir, fake_api, poll_interval=0, poll_bound=1,
                                              sleep=lambda _seconds: None, ledger_path=ledger)
                self.assertTrue(json.loads(output.read_text())["scoring_gate_passed"])
                _spent, final_states = reader63.spend_state(ledger)
                self.assertTrue(all(row["event"] in ("settled", "retired") for row in final_states.values()))
                if phase in ("posting_before_post", "accepted_before_response"):
                    self.assertEqual(sum(row["event"] == "retired" for row in final_states.values()), 1)
                self.assertLessEqual(len(final_states), 3 if phase in ("posting_before_post", "accepted_before_response") else 2)
                if phase == "response_saved_before_submission_update":
                    # The saved ID is resumed by polling; the two later POSTs belong only to the next chunk(s).
                    self.assertEqual(len(posted), 1)
                    self.assertEqual(posted[0]["requests"][0]["custom_id"], "r00200")
                if phase in ("posting_before_post", "accepted_before_response"):
                    run = json.loads((review_dir / "run-all.json").read_text())
                    self.assertTrue(run["chunks"][0]["retired_unknown_attempts"])

    def test_fixed_predictions_and_analysis_failure_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            rows, records = _analysis_fixture_rows()
            manifest = gather63.write_bundle(data_dir, rows, seed=101)
            _answer_rows, key = gather63.load_bundle(data_dir)
            labels_path = data_dir / "labels63-fixture.json"
            labels_path.write_text(json.dumps({"scoring_gate_passed": True,
                                               "calibration": {"passed": True},
                                               "duplicate_consistency": {"pairs": _add_duplicate_records(records)},
                                               "records": records}), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                report = analyze63.analyze(data_dir, labels_path)
            self.assertEqual(manifest["answer_count"], len(rows))
            self.assertEqual(set(key), {answer_id for answer_id in records if answer_id.startswith("a")})
            self.assertTrue(all(report["predictions"][name]["held"] for name in ("63a", "63b", "63c", "63d", "63e")))
            self.assertFalse(report["predictions"]["63d"]["strict_exact_name_sensitivity"]["held"])
            self.assertIn("coarse keyword sanity gate: PASS", output.getvalue())
            self.assertIn("not gated by hand labels", output.getvalue())
            self.assertEqual(report["duplicate_consistency"]["origin"]["agreement"], 1.0)

            one_failed = {"scoring_gate_passed": True, "calibration": {"passed": True},
                          "records": dict(records)}
            one_failed_id = next(iter(key))
            one_failed["records"][one_failed_id] = {"status": "failed", "error": "fixture"}
            labels_path.write_text(json.dumps(one_failed), encoding="utf-8")
            with self.assertRaisesRegex(analyze63.AnalysisError, "one retry"):
                analyze63.analyze(data_dir, labels_path)

            failed_labels = {"scoring_gate_passed": True, "calibration": {"passed": True},
                             "records": dict(records)}
            for answer_id in list(key)[:6]:
                failed_labels["records"][answer_id] = {"status": "failed", "error": "fixture"}
            labels_path.write_text(json.dumps(failed_labels), encoding="utf-8")
            with self.assertRaisesRegex(analyze63.AnalysisError, "one retry"):
                analyze63.analyze(data_dir, labels_path)

    def test_real_key_unclamped_rows_are_kind_selected_and_text_deduplicated(self):
        answer_rows, key, _manifest = reader63.load_bundle(reader63.DEFAULT_DATA_DIR)
        text_by_id = analyze63.answer_text_map(answer_rows)
        shape_records = {answer_id: {"status": "ok", "scheme_status": {"origin": "ok", "maker_identity": "ok"}}
                         for answer_id in key}
        qwen3_baseline = analyze63._selected_rows(
            key, shape_records, scheme="origin", text_by_id=text_by_id, dedupe_by_text=True,
            source="part54b", model="qwen3-4b", kind="unclamped")
        qwen35_baseline = analyze63._selected_rows(
            key, shape_records, scheme="maker_identity", text_by_id=text_by_id, dedupe_by_text=True,
            source="part54a", model="qwen35-4b", kind="unclamped")
        self.assertEqual(len(qwen3_baseline), 19)
        self.assertEqual(len(qwen35_baseline), 24)
        self.assertTrue(all(row[1]["dose"] != 0.0 for row in qwen3_baseline))

        records = {}
        for answer_id, meta in key.items():
            origin = "NEITHER"
            opus = meta.get("existing_opus_label")
            if meta.get("source") == "part54a" and isinstance(opus, list) and opus:
                origin = opus[0]
            records[answer_id] = {
                "status": "ok", "scheme_status": {"planet_frame": "ok", "maker_identity": "ok", "origin": "ok"},
                "quality_status": "ok", "planet_frame": "none", "origin": origin,
                "maker_identity": "own-lab", "maker_identity_alias": "own-lab", "quality": "OK",
            }
        baseline_ids = [row[0] for row in qwen3_baseline]
        records[baseline_ids[0]]["origin"] = "EARTH"
        records[baseline_ids[1]]["origin"] = "EARTH"
        records[baseline_ids[2]]["origin"] = "EARTH"

        for dose, count in ((0.5, 10), (1.0, 20)):
            cell_ids = sorted(answer_id for answer_id, meta in key.items()
                              if meta.get("source") == "part54b" and meta.get("model") == "qwen3-4b"
                              and meta.get("kind") == "twin" and meta.get("dose") == dose)
            for answer_id in cell_ids[:count]:
                records[answer_id]["planet_frame"] = "other-planet-or-space"

        down_ids = sorted(answer_id for answer_id, meta in key.items()
                          if meta.get("source") == "part54a" and meta.get("model") == "qwen35-4b"
                          and meta.get("kind") == "twin" and meta.get("dose") == -2.0)
        for answer_id in down_ids[:10]:
            records[answer_id]["maker_identity"] = "another-real-lab"
            records[answer_id]["maker_identity_alias"] = "own-lab"

        with contextlib.redirect_stdout(io.StringIO()):
            held = analyze63.evaluate_predictions(key, records, text_by_id)
        self.assertTrue(held["63c"]["held"])
        self.assertEqual(held["63c"]["unclamped"], [0, 24])
        self.assertEqual(held["63c"]["distinct_text_sensitivity"], [0, 19])
        self.assertTrue(held["63d"]["held"])
        self.assertEqual(held["63d"]["registered_aliases"]["unclamped"], [0, 24])
        self.assertEqual(held["63d"]["registered_aliases"]["distinct_text_sensitivity"], [0, 24])
        self.assertFalse(held["63d"]["strict_exact_name_sensitivity"]["held"])

        incomplete_key = dict(key)
        first_54a = next(answer_id for answer_id, meta in key.items() if meta.get("source") == "part54a")
        incomplete_key[first_54a] = dict(key[first_54a])
        incomplete_key[first_54a]["existing_opus_label"] = None
        with self.assertRaisesRegex(analyze63.AnalysisError, "63a full paired denominator"):
            analyze63.evaluate_predictions(incomplete_key, records, text_by_id)

        changed = {answer_id: dict(record) for answer_id, record in records.items()}
        plus_one_ids = sorted(answer_id for answer_id, meta in key.items()
                              if meta.get("source") == "part54b" and meta.get("model") == "qwen3-4b"
                              and meta.get("kind") == "twin" and meta.get("dose") == 1.0)
        changed[plus_one_ids[0]]["planet_frame"] = "none"
        changed[down_ids[0]]["maker_identity"] = "own-lab"
        with contextlib.redirect_stdout(io.StringIO()):
            failed = analyze63.evaluate_predictions(key, changed, text_by_id)
        self.assertFalse(failed["63c"]["held"])
        self.assertFalse(failed["63d"]["held"])

    def test_preconditions_and_pre_post_failure_do_not_burn_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir, review_dir = root / "data", root / "review"
            gather63.write_bundle(data_dir, _fixture_rows(), seed=93)
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(reader63, "context") as tls:
                with self.assertRaisesRegex(reader63.ReaderError, "ANTHROPIC_CREDITS_KEY"):
                    reader63.submit(data_dir, review_dir, lambda *_args: self.fail("must not POST"))
                tls.assert_not_called()
            self.assertFalse((review_dir / "spend.jsonl").exists())

            with mock.patch.dict(os.environ, {"ANTHROPIC_CREDITS_KEY": "test-key"}), \
                    mock.patch.object(reader63, "context", side_effect=reader63.ReaderError("bad CA")):
                with self.assertRaisesRegex(reader63.ReaderError, "bad CA"):
                    reader63.submit(data_dir, review_dir, lambda *_args: self.fail("must not POST"))
            self.assertFalse((review_dir / "spend.jsonl").exists())

            original_write_json = reader63.write_json
            post_calls = []

            def fail_posting(path, value):
                if isinstance(value, dict) and value.get("status") == "posting":
                    raise OSError("injected pre-POST state write failure")
                return original_write_json(path, value)

            with mock.patch.dict(os.environ, {"ANTHROPIC_CREDITS_KEY": "test-key"}), \
                    mock.patch.object(reader63, "context", return_value=object()), \
                    mock.patch.object(reader63, "write_json", side_effect=fail_posting):
                with self.assertRaisesRegex(reader63.ReaderError, "remains counted"):
                    reader63.submit(data_dir, review_dir,
                                    lambda *args: post_calls.append(args) or "{}")
            self.assertEqual(post_calls, [])
            spent, states = reader63.spend_state(reader63.DEFAULT_LEDGER_PATH)
            self.assertGreater(spent, 0.0)
            self.assertEqual(next(iter(states.values()))["event"], "reserved")
            review_dir_2 = root / "review-prepost-api"
            with mock.patch.dict(os.environ, {"ANTHROPIC_CREDITS_KEY": "test-key"}), \
                    mock.patch.object(reader63, "context", return_value=object()):
                with self.assertRaisesRegex(reader63.ReaderError, "remains counted"):
                    reader63.submit(data_dir, review_dir_2,
                                    lambda *_args: (_ for _ in ()).throw(reader63.ReaderError("CA disappeared")))
            spent, states = reader63.spend_state(reader63.DEFAULT_LEDGER_PATH)
            self.assertGreater(spent, 0.0)
            self.assertEqual(next(iter(states.values()))["event"], "reserved")

    def test_settlement_records_usage_above_reserve_when_caps_allow(self):
        ledger_path = Path(self._ledger_tmp.name) / "explicit-ledger.jsonl"
        submission_id = "settle-over-reserve"
        attempt_id = reader63.Ledger(str(ledger_path), reader63.SHARED_CAP_USD).reserve(
            0.01, "part63", {"submission_id": submission_id})
        review_dir = HERE / ".test-settlement-over-reserve"
        review_dir.mkdir(exist_ok=True)
        self.addCleanup(lambda: __import__("shutil").rmtree(review_dir, ignore_errors=True))
        reader63.write_json(review_dir / f"submission-{submission_id}.json", {
            "submission_id": submission_id, "attempt_id": attempt_id,
        })
        reader63.settle_reserve(review_dir, submission_id, 0.02, ledger_path=ledger_path)
        state = reader63.Ledger(str(ledger_path), reader63.SHARED_CAP_USD).state()
        self.assertEqual(state["reserved"], 0.0)
        self.assertEqual(state["settled"], 0.02)
        _spent, attempts = reader63.spend_state(ledger_path)
        self.assertEqual(attempts[submission_id]["event"], "settled")
        self.assertEqual(attempts[submission_id]["effective_usd"], 0.02)

    def test_ambiguous_post_keeps_reserve_and_blocks_manual_resubmission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir, review_dir = root / "data", root / "review"
            gather63.write_bundle(data_dir, _fixture_rows(), seed=93)
            attempts = []

            def api(method, path, body=None):
                if method == "POST":
                    attempts.append(body)
                    if len(attempts) == 1:
                        raise reader63.APIError(429, "rate limited; acceptance unknown")
                    return json.dumps({"id": "msgbatch_second_attempt", "processing_status": "in_progress"})
                raise AssertionError(f"unexpected call {method} {path}")

            with mock.patch.dict(os.environ, {"ANTHROPIC_CREDITS_KEY": "test-key"}), \
                    mock.patch.object(reader63, "context", return_value=object()), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(reader63.ReaderError, "recover or retire"):
                    reader63.submit(data_dir, review_dir, api)
                first_files = [path for path in sorted(review_dir.glob("submission-*.json"))
                               if not path.name.endswith(("-idmap.json", "-key.json"))]
                first = json.loads(first_files[0].read_text())
                self.assertEqual(first["status"], "ambiguous")
                with self.assertRaisesRegex(reader63.ReaderError, "earlier submission is unresolved"):
                    reader63.submit(data_dir, review_dir, api)
            self.assertEqual(len(attempts), 1)
            spent, states = reader63.spend_state(reader63.DEFAULT_LEDGER_PATH)
            self.assertGreater(spent, 0)
            self.assertEqual(states[first["submission_id"]]["event"], "reserved")
            self.assertEqual(len(states), 1)
            self.assertNotIn("reconcile", reader63.parser().format_help())

    def test_collection_retries_only_bad_ids_once_and_merges_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir, review_dir = root / "data", root / "review"
            ledger_path = Path(self._ledger_tmp.name) / "explicit-collection-retry-ledger.jsonl"
            gather63.write_bundle(data_dir, _fixture_rows(), seed=93)
            batch_ids = ["msgbatch_root", "msgbatch_retry"]
            answer_to_fail = {"id": None}
            captured_bodies = {}
            submit_count = 0
            result_barrier = threading.Barrier(2)
            result_barrier_lock = threading.Lock()
            root_result_calls = 0

            def api(method, path, body=None):
                nonlocal submit_count
                nonlocal root_result_calls
                if method == "POST":
                    batch_id = batch_ids[submit_count]
                    submit_count += 1
                    captured_bodies[batch_id] = body
                    if batch_id == "msgbatch_root":
                        link = next(review_dir.glob("batch-*.json"), None)
                        if link is None:
                            maps = sorted(review_dir.glob("submission-*-idmap.json"))
                            idmap = json.loads(maps[-1].read_text())
                        else:
                            idmap = json.loads(sorted(review_dir.glob("submission-*-idmap.json"))[-1].read_text())
                        answer_to_fail["id"] = next(internal_id for internal_id in idmap.values()
                                                    if internal_id.startswith("a"))
                    return json.dumps({"id": batch_id, "processing_status": "in_progress"})
                batch_id = path.split("/batches/")[1].split("/")[0]
                if path.endswith("/results"):
                    if batch_id == "msgbatch_root":
                        with result_barrier_lock:
                            synchronize = root_result_calls < 2
                            root_result_calls += 1
                        if synchronize:
                            result_barrier.wait(timeout=10)
                    batch_link = json.loads((review_dir / f"batch-{batch_id}.json").read_text())
                    sub = json.loads((review_dir / batch_link["submission_file"]).read_text())
                    idmap = json.loads((review_dir / sub["idmap_file"]).read_text())
                    snapshot = json.loads((review_dir / sub["key_file"]).read_text())
                    lines = []
                    for custom_id, internal_id in idmap.items():
                        quality = snapshot["calibration"]["expected_quality"].get(internal_id, "OK")
                        result = _fake_success(quality)
                        if batch_id == "msgbatch_root" and internal_id == answer_to_fail["id"]:
                            payload = dict(FAKE_LABEL)
                            payload["planet_frame"] = None
                            result = {"type": "succeeded", "message": {
                                "stop_reason": "end_turn",
                                "content": [{"type": "text", "text": json.dumps(payload)}],
                                "usage": {"input_tokens": 100, "output_tokens": 40},
                            }}
                        lines.append(json.dumps({"custom_id": custom_id, "result": result}))
                    return "\n".join(lines)
                if path == f"/v1/messages/batches/{batch_id}":
                    return json.dumps({"processing_status": "ended", "request_counts": {"succeeded": 1}})
                raise AssertionError(f"unexpected fake API call: {method} {path}")

            with mock.patch.dict(os.environ, {"ANTHROPIC_CREDITS_KEY": "test-key"}), \
                    mock.patch.object(reader63, "context", return_value=object()), \
                    contextlib.redirect_stdout(io.StringIO()):
                root_batch = reader63.submit(data_dir, review_dir, api, ledger_path=ledger_path)
                concurrent_results = []
                def collect_root():
                    try:
                        concurrent_results.append(reader63.collect(root_batch, data_dir, review_dir, api,
                                                                   ledger_path=ledger_path))
                    except BaseException as exc:
                        concurrent_results.append(exc)
                workers = [threading.Thread(target=collect_root) for _ in range(2)]
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join(timeout=15)
                    self.assertFalse(worker.is_alive())
                self.assertEqual(len(concurrent_results), 2)
                self.assertFalse(any(isinstance(value, BaseException) for value in concurrent_results),
                                 concurrent_results)
                labels_path = concurrent_results[0]
                self.assertEqual(submit_count, 2)
                retry_batch = json.loads(labels_path.read_text())["retry"]["batch_id"]
                self.assertEqual(retry_batch, "msgbatch_retry")
                link = json.loads((review_dir / f"batch-{retry_batch}.json").read_text())
                retry_map = json.loads((review_dir / json.loads((review_dir / link["submission_file"]).read_text())[
                    "idmap_file"]).read_text())
                self.assertEqual(list(retry_map.values()), [answer_to_fail["id"]])
                self.assertEqual(len(captured_bodies[retry_batch]["requests"]), 1)
                labels_path = reader63.collect(retry_batch, data_dir, review_dir, api,
                                               ledger_path=ledger_path)
                # Recollecting the parent preserves the completed retry and does not submit again.
                labels_path = reader63.collect(root_batch, data_dir, review_dir, api,
                                               ledger_path=ledger_path)
                self.assertEqual(submit_count, 2)
            labels = json.loads(labels_path.read_text())
            self.assertEqual(labels["batch_ids"], batch_ids)
            self.assertEqual(labels["request_count"], len(_fixture_rows()) + EXPECTED_CALIBRATION_COUNT
                             + reader63.DUPLICATE_COUNT)
            self.assertEqual(labels["records"][answer_to_fail["id"]]["status"], "ok")
            self.assertTrue(labels["scoring_gate_passed"])
            self.assertEqual(labels["retry"]["status"], "collected")

            remaining = dict(labels)
            remaining["records"] = dict(labels["records"])
            remaining["records"][answer_to_fail["id"]] = dict(labels["records"][answer_to_fail["id"]])
            remaining["records"][answer_to_fail["id"]]["scheme_status"] = {
                "planet_frame": "unlabelled", "maker_identity": "ok", "origin": "ok"}
            remaining["scoring_gate_passed"] = True
            blocked_path = data_dir / "labels63-remains-bad.json"
            blocked_path.write_text(json.dumps(remaining))
            with self.assertRaisesRegex(analyze63.AnalysisError, "one retry"):
                analyze63.analyze(data_dir, blocked_path)

    def test_collect_recovers_a_saved_batch_id_if_sidecar_write_was_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir, review_dir = root / "data", root / "review"
            gather63.write_bundle(data_dir, _fixture_rows(), seed=93)

            def api(method, path, body=None):
                if method == "POST":
                    return json.dumps({"id": FAKE_BATCH_ID, "processing_status": "in_progress"})
                if path == f"/v1/messages/batches/{FAKE_BATCH_ID}":
                    return json.dumps({"processing_status": "ended"})
                if path == f"/v1/messages/batches/{FAKE_BATCH_ID}/results":
                    submission = next(path for path in review_dir.glob("submission-*.json")
                                      if not path.name.endswith(("-idmap.json", "-key.json")))
                    meta = json.loads(submission.read_text())
                    idmap = json.loads((review_dir / meta["idmap_file"]).read_text())
                    snapshot = json.loads((review_dir / meta["key_file"]).read_text())
                    return "\n".join(json.dumps({
                        "custom_id": custom_id,
                        "result": _fake_success(snapshot["calibration"]["expected_quality"].get(internal_id, "OK")),
                    }) for custom_id, internal_id in idmap.items())
                raise AssertionError(f"unexpected fake API call: {method} {path}")

            original_write_json = reader63.write_json

            def interrupt_sidecar(path, value):
                if isinstance(value, dict) and value.get("status") == "submitted":
                    raise OSError("injected interruption after raw response was saved")
                return original_write_json(path, value)

            with mock.patch.dict(os.environ, {"ANTHROPIC_CREDITS_KEY": "test-key"}), \
                    mock.patch.object(reader63, "context", return_value=object()), \
                    mock.patch.object(reader63, "write_json", side_effect=interrupt_sidecar):
                with self.assertRaisesRegex(OSError, "raw response was saved"):
                    reader63.submit(data_dir, review_dir, api)
            self.assertFalse((review_dir / f"batch-{FAKE_BATCH_ID}.json").exists())
            metadata = next(path for path in review_dir.glob("submission-*.json")
                            if not path.name.endswith(("-idmap.json", "-key.json")))
            self.assertNotIn("batch_id", json.loads(metadata.read_text()))
            self.assertTrue((review_dir / f"post-response-{json.loads(metadata.read_text())['submission_id']}.json").is_file())
            with contextlib.redirect_stdout(io.StringIO()):
                labels_path = reader63.collect(FAKE_BATCH_ID, data_dir, review_dir, api)
            self.assertTrue(json.loads(labels_path.read_text())["scoring_gate_passed"])

    def test_listing_is_read_only_and_filters_to_run_start(self):
        with tempfile.TemporaryDirectory() as directory:
            review_dir = Path(directory) / "review"
            review_dir.mkdir()
            reader63.write_json(review_dir / "run.json", {"started_at": "2026-10-22T00:00:00+00:00"})
            before = {p.relative_to(review_dir): p.read_bytes() for p in review_dir.iterdir() if p.is_file()}
            calls = []

            def list_api(method, path, body=None):
                calls.append((method, path))
                if path == "/v1/messages/batches?limit=100":
                    return json.dumps({"data": [
                        {"id": "old", "created_at": "2026-10-21T23:59:59+00:00"},
                    ], "has_more": True, "last_id": "old"})
                if path == "/v1/messages/batches?limit=100&after_id=old":
                    return json.dumps({"data": [
                        {"id": "new", "created_at": "2026-10-22T00:00:01+00:00"},
                    ], "has_more": False})
                raise AssertionError(f"unexpected listing URL: {path}")

            batches = reader63.listing_batches(review_dir, list_api)
            after = {p.relative_to(review_dir): p.read_bytes() for p in review_dir.iterdir() if p.is_file()}
            self.assertEqual([row["id"] for row in batches], ["new"])
            self.assertEqual(calls, [
                ("GET", "/v1/messages/batches?limit=100"),
                ("GET", "/v1/messages/batches?limit=100&after_id=old"),
            ])
            self.assertEqual(before, after)

    def test_each_sabotage_mutates_real_code_and_goes_red(self):
        targets = (
            ("ignore-truncation", "test_truncation_is_not_scored"),
            ("leak-condition", "test_requests_are_blind_and_max_tokens_is_pinned"),
            ("zero-reserve", "test_estimator_reserves_a_positive_worst_case_amount"),
            ("drop-side-key-row", "test_writer_output_and_reader_side_key_match"),
            ("dose-zero-baseline", "test_real_key_unclamped_rows_are_kind_selected_and_text_deduplicated"),
            ("alias-shortcut", "test_calibration_gate_and_part63_constants"),
            ("temperature-default", "test_requests_are_blind_and_max_tokens_is_pinned"),
            ("keyword-substring", "test_calibration_gate_and_part63_constants"),
            ("reject-63c-flat-step", "test_63c_allows_a_flat_monotonic_step"),
            ("loosen-token-estimate", "test_estimator_reserves_a_positive_worst_case_amount"),
            ("drop-label-fanout", "test_fake_api_submit_collect_settles_and_calibrates"),
            ("skip-retire", "test_run_all_sigkill_recovery_matrix_and_retire_path"),
        )
        for mode, test_name in targets:
            with tempfile.TemporaryDirectory() as directory:
                copied_basin = Path(directory)
                copied_part63 = copied_basin / "part63"
                shutil.copytree(HERE, copied_part63,
                                ignore=shutil.ignore_patterns("__pycache__", "review"))
                (copied_basin / "part59").mkdir()
                shutil.copy2(HERE.parent / "part59" / "blind_setting.py",
                             copied_basin / "part59" / "blind_setting.py")
                env = os.environ.copy()
                env[SABOTAGE_ENV] = mode
                completed = subprocess.run(
                    [sys.executable, "-m", "unittest", f"test_part63.SafetyTests.{test_name}"],
                    cwd=copied_part63, env=env, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
                )
            with self.subTest(sabotage=mode):
                self.assertNotEqual(completed.returncode, 0, f"sabotage {mode} unexpectedly passed")
                self.assertIn("FAILED", completed.stdout)
                print(f"SABOTAGE RED: {mode} -> {test_name}")


if __name__ == "__main__":
    unittest.main()
