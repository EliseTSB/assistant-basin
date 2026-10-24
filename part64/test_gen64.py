"""Offline tests for Part 64 M1 requests and durable paid chunk recovery."""
from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from part64 import gen64
from part64 import collect64
from part64 import score64
from paidaction import paid_ledger
from paidaction.paid_ledger import HARD_CAP_USD, HTTPFailure, Ledger


def collection_row(unit_key: str = "L0:u1", unit_type: str = "neuron") -> dict:
    strata = {}
    for category, count in gen64.SAMPLE_COUNTS["3x"].items():
        strata[category] = [{"window": f"{category} context {index} [[feature-{index}]] detail"}
                            for index in range(count)]
    return {"unit_key": unit_key, "type": unit_type, "layer": 0, "unit": 1,
            "control_id": None, "strata": strata}


def answer_text() -> str:
    return json.dumps({
        "hypothesis": "The examples share a recurring semantic pattern.",
        "strong": [f"A vivid example shows pattern {index}." for index in range(4)],
        "weak": [f"A faint example hints at pattern {index}." for index in range(4)],
        "none": [f"A neutral sentence discusses unrelated topic {index}." for index in range(4)],
    })


def collection_fixture() -> tuple[list[dict], list[dict]]:
    rows, pilot = [], []
    for index in range(500):
        unit_type = "neuron" if index < 250 else "control"
        local_index = index if unit_type == "neuron" else index - 250
        key = f"L0:u{local_index}" if unit_type == "neuron" else f"L0:c{local_index}"
        strata = {}
        record = {"unit": local_index if unit_type == "neuron" else None,
                  "control_id": local_index if unit_type == "control" else None}
        for category, count in gen64.SAMPLE_COUNTS["3x"].items():
            pilot_count = gen64.SAMPLE_COUNTS["1x"][category]
            items = []
            for item_index in range(count):
                if item_index < pilot_count:
                    text = f"pilot-{category}-{local_index}-{item_index}"
                    if index == 0 and category == "top" and item_index == 1:
                        text = "pilot-top-0-0"
                else:
                    text = f"fresh-{category}-{local_index}-{item_index}"
                items.append({"window": text, "activation": float(count - item_index)})
            strata[category] = items
            record[category] = list(items[:pilot_count])
        identity = {"unit_key": key, "type": unit_type, "layer": 0, "record": record}
        rows.append({"unit_key": key, "type": unit_type, "layer": 0,
                     "unit": record["unit"], "control_id": record["control_id"],
                     "strata": strata, "fresh_top12_match": 12})
        pilot.append(identity)
    return rows, pilot


def collection_manifest(rows: list[dict], *, median: float | None = None) -> dict:
    raw = collect64.collection_jsonl(rows)
    return {"version": 1, "unit_count": 500, "subset_verified": True,
            "capacities": gen64.SAMPLE_COUNTS["3x"], "pilot_counts": gen64.SAMPLE_COUNTS["1x"],
            "model": "Qwen/Qwen3-1.7B", "model_revision": gen64.REGISTERED_MODEL_REVISION,
            "corpus": {"path": str(gen64.REGISTERED_CORPUS),
                       "size_bytes": gen64.REGISTERED_CORPUS_SIZE,
                       "sha256": gen64.REGISTERED_CORPUS_SHA256_PREFIX + "0" * 48},
            "median_fresh_top12_match": (statistics.median(row["fresh_top12_match"] for row in rows)
                                          if median is None else median),
            "collection_sha256": hashlib.sha256(raw).hexdigest()}


class FakeBatchAPI:
    def __init__(self, *, text: str | None = None, stop_reason: str = "end_turn",
                 errored_custom_ids: set[str] | None = None,
                 checks_until_end: int = 1, fail_first_status: bool = False):
        self.text = answer_text() if text is None else text
        self.stop_reason = stop_reason
        self.errored_custom_ids = set(errored_custom_ids or ())
        self.posts: list[list[dict]] = []
        self.batch_requests: dict[str, list[dict]] = {}
        self.batch_records: dict[str, dict] = {}
        self.polls = 0
        self.result_gets = 0
        self.checks_until_end = checks_until_end
        self.batch_status_checks: dict[str, int] = {}
        self.events: list[tuple[str, str]] = []
        self.max_in_flight = 0
        self.fail_first_status = fail_first_status
        self.chunk_path: Path | None = None
        self.durable_id_seen_before_poll = True

    def __call__(self, method, path, body=None):
        if method == "POST" and path == "/v1/messages/batches":
            self.posts.append(body["requests"])
            batch_id = f"batch-{len(self.posts)}"
            self.batch_requests[batch_id] = body["requests"]
            self.batch_records[batch_id] = {"id": batch_id, "processing_status": "in_progress",
                                            "created_at": gen64.utc_now()}
            self.batch_status_checks[batch_id] = 0
            self.events.append(("post", batch_id))
            self.max_in_flight = max(
                self.max_in_flight,
                sum(record["processing_status"] == "in_progress"
                    for record in self.batch_records.values()))
            return json.dumps({"id": batch_id, "processing_status": "in_progress"})
        if method == "GET" and path.startswith("/v1/messages/batches?"):
            return json.dumps({"data": list(self.batch_records.values()), "has_more": False})
        if method == "GET" and path.startswith("/v1/messages/batches/") and path.endswith("/results"):
            self.result_gets += 1
            batch_id = path.split("/")[-2]
            self.events.append(("results", batch_id))
            requests = self.batch_requests[batch_id]
            rows = []
            for request in requests:
                if request["custom_id"] in self.errored_custom_ids:
                    rows.append({"custom_id": request["custom_id"], "result": {
                        "type": "errored", "error": {"type": "api_error", "message": "overloaded"}}})
                    continue
                rows.append({"custom_id": request["custom_id"], "result": {
                    "type": "succeeded", "message": {
                        "content": [{"type": "text", "text": self.text}],
                        "stop_reason": self.stop_reason,
                        "usage": {"input_tokens": 321, "output_tokens": 1234},
                    }}})
            return "\n".join(json.dumps(row) for row in rows) + "\n"
        if method == "GET" and path.startswith("/v1/messages/batches/"):
            if self.fail_first_status:
                self.fail_first_status = False
                raise RuntimeError("simulated process kill after submit pass")
            self.polls += 1
            batch_id = path.rsplit("/", 1)[-1]
            self.batch_status_checks[batch_id] += 1
            if self.batch_status_checks[batch_id] >= self.checks_until_end:
                self.batch_records[batch_id]["processing_status"] = "ended"
            status = self.batch_records[batch_id]["processing_status"]
            self.events.append(("status", batch_id))
            if self.chunk_path is not None:
                chunk = gen64.read_json(self.chunk_path)
                self.durable_id_seen_before_poll &= chunk.get("state") == "submitted" and bool(chunk.get("batch_id"))
            return json.dumps({"processing_status": status, "request_counts": {"succeeded": 1}})
        raise AssertionError(f"unexpected fake API call: {method} {path}")


class NoAcceptThenOkAPI(FakeBatchAPI):
    """First POST times out before acceptance; provider listing is complete and empty."""

    def __init__(self):
        super().__init__()
        self.reject_first = True

    def __call__(self, method, path, body=None):
        if method == "POST" and self.reject_first:
            self.reject_first = False
            raise TimeoutError("client timed out before provider acceptance")
        return super().__call__(method, path, body)


class AcceptedLostResponseAPI(FakeBatchAPI):
    """The provider accepts the first POST, then the client loses its response."""

    def __init__(self):
        super().__init__()
        self.lose_response = True

    def __call__(self, method, path, body=None):
        if method == "POST" and self.lose_response:
            self.lose_response = False
            response = super().__call__(method, path, body)
            for batch in self.batch_records.values():
                batch["processing_status"] = "ended"
            raise TimeoutError("accepted batch response was lost")
        return super().__call__(method, path, body)


class LostReprobeResponseAPI(FakeBatchAPI):
    """Accept a re-probe, end it, then lose its POST response."""

    def __init__(self):
        super().__init__(text="not JSON")
        self.lose_next_post = False

    def __call__(self, method, path, body=None):
        if method == "POST" and self.lose_next_post:
            self.lose_next_post = False
            response = super().__call__(method, path, body)
            batch_id = f"batch-{len(self.posts)}"
            self.batch_records[batch_id]["processing_status"] = "ended"
            raise TimeoutError("accepted re-probe response was lost")
        return super().__call__(method, path, body)


class LocalPaidRegistryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=gen64.HERE)
        registry = Path(temporary.name) / "run-paths.json"
        registry_patch = patch.object(paid_ledger, "_RUN_PATH_REGISTRY", registry)
        lock_patch = patch.object(paid_ledger, "_RUN_PATH_LOCK", registry.with_suffix(".lock"))
        stage_patch = patch.object(gen64, "WRITER_STAGE_LEDGER_DIR", Path(temporary.name) / "writer-stage")
        registry_patch.start()
        lock_patch.start()
        stage_patch.start()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(lock_patch.stop)
        self.addCleanup(registry_patch.stop)
        self.addCleanup(stage_patch.stop)


class WriterRequestTests(LocalPaidRegistryTests):
    def test_writer_stage_ledger_path_is_machine_wide_with_test_override(self):
        source = "from part64 import gen64; print(gen64.WRITER_STAGE_LEDGER_DIR)"
        env = os.environ.copy()
        env.pop("PART64_TEST_WRITER_STAGE_LEDGER_DIR", None)
        default = subprocess.run([sys.executable, "-c", source], cwd=gen64.BASIN,
                                 env=env, capture_output=True, text=True, check=True)
        self.assertEqual(Path(default.stdout.strip()),
                         Path("/home/elisepc/assistant-basin/results/part64/writer-stage"))
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            override = Path(temporary) / "test-writer-stage"
            env["PART64_TEST_WRITER_STAGE_LEDGER_DIR"] = str(override)
            actual = subprocess.run([sys.executable, "-c", source], cwd=gen64.BASIN,
                                    env=env, capture_output=True, text=True, check=True)
            self.assertEqual(Path(actual.stdout.strip()), override.resolve())

    def test_m1_sample_counts_subset_and_api_defaults(self):
        row = collection_row()
        one = gen64.windows_for_sample(row, "1x")
        three = gen64.windows_for_sample(row, "3x")
        self.assertEqual(len(one), 21)
        self.assertEqual(len(three), 63)
        self.assertTrue(set(one) <= set(three))

        request = gen64.request_for("opus", "3x", row, "u0000")
        params = request["params"]
        self.assertEqual(params["model"], "claude-opus-5-5")
        self.assertEqual(params["max_tokens"], gen64.MAX_TOKENS)
        self.assertEqual(params["system"][0]["text"], gen64.M1_PROMPT)
        self.assertNotIn("temperature", params)
        self.assertNotIn("thinking", params)
        self.assertEqual(params["messages"][0]["role"], "user")
        self.assertEqual(gen64.DEFAULT_COLLECTION, gen64.BASIN / "results" / "part64" / "collection.jsonl")
        self.assertEqual(gen64.DEFAULT_ROOT, gen64.BASIN / "results" / "part64" / "gen64")

    def test_build_report_no_longer_claims_the_pipeline_was_not_built(self):
        report = (gen64.HERE / "REPORT-build.md").read_text(encoding="utf-8").lower()
        self.assertIn("pipeline now includes targeted collection", report)
        self.assertNotIn("pipeline was not built", report)

    def test_collection_manifest_gates_are_fail_closed(self):
        manifest = {
            "version": 1, "unit_count": 500, "subset_verified": True,
            "capacities": gen64.SAMPLE_COUNTS["3x"], "pilot_counts": gen64.SAMPLE_COUNTS["1x"],
            "model": "Qwen/Qwen3-1.7B", "model_revision": gen64.REGISTERED_MODEL_REVISION,
            "corpus": {"path": str(gen64.REGISTERED_CORPUS),
                       "size_bytes": gen64.REGISTERED_CORPUS_SIZE,
                       "sha256": gen64.REGISTERED_CORPUS_SHA256_PREFIX + "0" * 48},
            "median_fresh_top12_match": 10, "collection_sha256": "a" * 64,
        }
        self.assertIsNone(gen64.validate_collection_manifest(manifest))
        broken = dict(manifest, median_fresh_top12_match=9)
        with self.assertRaisesRegex(gen64.WriterError, "collection manifest"):
            gen64.validate_collection_manifest(broken)

    def test_run64_corpus_override_path_is_accepted_by_the_writer_loader(self):
        rows, pilot = collection_fixture()
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            # run64.sh passes its CORPUS value unchanged to verification and assembly.
            staged_corpus = Path(temporary) / "staged" / "pile10k.jsonl"
            staged_corpus.parent.mkdir()
            corpus_bytes = b"registered corpus test fixture\n"
            staged_corpus.write_bytes(corpus_bytes)
            corpus_digest = hashlib.sha256(corpus_bytes).hexdigest()
            corpus_meta = {"path": str(staged_corpus), "size_bytes": len(corpus_bytes),
                           "sha256": corpus_digest}
            manifest = collection_manifest(rows)
            manifest["corpus"] = corpus_meta
            collection_path = Path(temporary) / "collection.jsonl"
            collection_path.write_bytes(collect64.collection_jsonl(rows))
            gen64.write_json(collection_path.with_suffix(".jsonl.manifest.json"), manifest)
            with patch.object(collect64, "CORPUS_SIZE", len(corpus_bytes)), \
                    patch.object(collect64, "CORPUS_SHA256_PREFIX", corpus_digest[:16]), \
                    patch.object(gen64, "REGISTERED_CORPUS_SIZE", len(corpus_bytes)), \
                    patch.object(gen64, "REGISTERED_CORPUS_SHA256_PREFIX", corpus_digest[:16]), \
                    patch.object(gen64.gen60, "load_splits", return_value={"dev": pilot}):
                self.assertEqual(collect64.verify_corpus(staged_corpus), corpus_meta)
                loaded, _digest = gen64.load_collection(collection_path)
            self.assertEqual(len(loaded), 500)
            self.assertEqual(manifest["corpus"]["path"], str(staged_corpus))

    def test_collection_loader_preserves_unicode_line_separators_inside_json_strings(self):
        rows, pilot = collection_fixture()
        expected = "fresh\u2028paragraph\u2029separator\x85end"
        rows[0]["strata"]["top"][12]["window"] = expected
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            path = Path(temporary) / "collection.jsonl"
            raw = collect64.collection_jsonl(rows)
            path.write_bytes(raw)
            gen64.write_json(path.with_suffix(".jsonl.manifest.json"), collection_manifest(rows))
            with patch.object(gen64.gen60, "load_splits", return_value={"dev": pilot}):
                loaded, _digest = gen64.load_collection(path)
            self.assertEqual(loaded[0]["strata"]["top"][12]["window"], expected)

    def test_results_jsonl_parser_splits_only_on_line_feed(self):
        row = {"custom_id": "unicode-result", "text": "line one\u2028line two"}
        text = json.dumps(row, ensure_ascii=False) + "\n"
        self.assertEqual(gen64._parse_result_jsonl(text), [row])

    def test_real_pilot_prefix_duplicates_are_preserved_but_fresh_duplicates_are_refused(self):
        rows, pilot = collection_fixture()
        self.assertEqual(rows[0]["strata"]["top"][0]["window"],
                         rows[0]["strata"]["top"][1]["window"])
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            path = Path(temporary) / "collection.jsonl"
            raw = collect64.collection_jsonl(rows)
            path.write_bytes(raw)
            gen64.write_json(path.with_suffix(path.suffix + ".manifest.json"), collection_manifest(rows))
            with patch.object(gen64.gen60, "load_splits", return_value={"dev": pilot}):
                loaded, _digest = gen64.load_collection(path)
            self.assertEqual(loaded[0]["strata"]["top"][0]["window"],
                             loaded[0]["strata"]["top"][1]["window"])

            rows[0]["strata"]["top"][12]["window"] = rows[0]["strata"]["top"][0]["window"]
            raw = collect64.collection_jsonl(rows)
            path.write_bytes(raw)
            gen64.write_json(path.with_suffix(path.suffix + ".manifest.json"), collection_manifest(rows))
            with patch.object(gen64.gen60, "load_splits", return_value={"dev": pilot}):
                with self.assertRaisesRegex(gen64.WriterError, "duplicate fresh top windows"):
                    gen64.load_collection(path)

    def test_registered_l1_c69_duplicate_pilot_windows_load_without_refusal(self):
        pilot = gen64.gen60.load_splits()["dev"]
        target = next(unit for unit in pilot if unit["unit_key"] == "L1:c69")
        pilot_top = [item["window"] for item in target["record"]["top"][:12]]
        self.assertLess(len(set(pilot_top)), 12)
        rows = []
        for unit in pilot:
            record = unit["record"]
            strata = {}
            for category, count in gen64.SAMPLE_COUNTS["3x"].items():
                pilot_count = gen64.SAMPLE_COUNTS["1x"][category]
                prefix = [{"window": item["window"], "activation": item.get("activation", 0.0)}
                          for item in record[category][:pilot_count]]
                extras = [{"window": f"fresh-{unit['unit_key']}-{category}-{index}",
                           "activation": float(-index)} for index in range(count - pilot_count)]
                strata[category] = prefix + extras
            rows.append({"unit_key": unit["unit_key"], "type": unit["type"], "layer": unit["layer"],
                         "unit": record.get("unit"), "control_id": record.get("control_id"),
                         "strata": strata, "fresh_top12_match": 12})
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            path = Path(temporary) / "collection.jsonl"
            raw = collect64.collection_jsonl(rows)
            path.write_bytes(raw)
            gen64.write_json(path.with_suffix(path.suffix + ".manifest.json"), collection_manifest(rows))
            loaded, _digest = gen64.load_collection(path)
        loaded_target = next(row for row in loaded if row["unit_key"] == "L1:c69")
        loaded_top = [item["window"] for item in loaded_target["strata"]["top"][:12]]
        self.assertEqual(loaded_top, pilot_top)

    def test_collection_sha_and_row_median_are_recomputed_at_load(self):
        rows, pilot = collection_fixture()
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            path = Path(temporary) / "collection.jsonl"
            raw = collect64.collection_jsonl(rows)
            path.write_bytes(raw)
            manifest = collection_manifest(rows)
            gen64.write_json(path.with_suffix(path.suffix + ".manifest.json"), manifest)
            path.write_bytes(raw.replace(b"fresh-top-0-12", b"fresh-top-0-xx", 1))
            with patch.object(gen64.gen60, "load_splits", return_value={"dev": pilot}):
                with self.assertRaisesRegex(gen64.WriterError, "SHA-256"):
                    gen64.load_collection(path)

            for row in rows[:250]:
                row["fresh_top12_match"] = 9
            for row in rows[250:]:
                row["fresh_top12_match"] = 10
            raw = collect64.collection_jsonl(rows)
            path.write_bytes(raw)
            manifest = collection_manifest(rows, median=10)
            gen64.write_json(path.with_suffix(path.suffix + ".manifest.json"), manifest)
            with patch.object(gen64.gen60, "load_splits", return_value={"dev": pilot}):
                with self.assertRaisesRegex(gen64.WriterError, "reproduction median"):
                    gen64.load_collection(path)

    def test_worst_case_chunk_reserves_stay_below_the_one_dollar_target(self):
        row = collection_row()
        entries = [(gen64.request_for("opus", "3x", row, f"u{index:04d}"),
                    {"unit_key": row["unit_key"]}) for index in range(50)]
        chunks = gen64._chunk_requests(entries, gen64.WRITERS["opus"])
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(gen64.request_reserve_usd([entry[0] for entry in chunk],
                                                           gen64.WRITERS["opus"]),
                                 gen64.CHUNK_RESERVE_TARGET_USD)

    def test_plan_reinitialization_keeps_run_specific_custom_ids_stable(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            first = gen64.initialize_plan(root, [collection_row()], "d" * 64)
            probe_id = gen64.read_json(root / first["probes"]["haiku"]["chunk_path"])["requests"][0]["custom_id"]
            second = gen64.initialize_plan(root, [collection_row()], "d" * 64)
            second_probe_id = gen64.read_json(root / second["probes"]["haiku"]["chunk_path"])["requests"][0]["custom_id"]
            self.assertEqual(first["run_id"], second["run_id"])
            self.assertEqual(first["plan_sha256"], second["plan_sha256"])
            self.assertEqual(probe_id, second_probe_id)

    def test_canonical_part60_windows_must_be_the_1x_prefix(self):
        rows = ([collection_row(f"L0:u{index}", "neuron") for index in range(250)]
                + [collection_row(f"L0:c{index}", "control") for index in range(250)])
        pilot = [{"unit_key": row["unit_key"], "type": row["type"], "layer": row["layer"],
                  "record": {**{category: list(items) for category, items in row["strata"].items()},
                             "unit": row["unit"], "control_id": row["control_id"]}}
                 for row in rows]
        with patch.object(gen64.gen60, "load_splits", return_value={
                "dev": pilot}):
            gen64.verify_pilot_subset(rows)
            row = rows[0]
            changed = dict(row)
            changed["strata"] = {category: list(items) for category, items in row["strata"].items()}
            changed["strata"]["top"][0] = {"window": "different pilot text"}
            with self.assertRaisesRegex(gen64.WriterError, "differ from Part 60 M1"):
                gen64.verify_pilot_subset([changed] + rows[1:])

    def test_probe_sends_one_request_per_writer_and_validates_m1(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            manifest = gen64.initialize_plan(root, [collection_row()], "a" * 64)
            api = FakeBatchAPI()
            outcomes = gen64.execute_probes(root, manifest, api, poll_interval=0, sleep=lambda _n: None)
            self.assertEqual(len(api.posts), 2)
            self.assertEqual([len(items) for items in api.posts], [1, 1])
            self.assertTrue(all(item["passed"] for item in outcomes))

    def test_real_run_probe_resolves_relative_root_before_reaching_fake_api(self):
        rows, pilot = collection_fixture()
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            base = Path(temporary).resolve()
            collection_path = base / "collection.jsonl"
            collection_path.write_bytes(collect64.collection_jsonl(rows))
            gen64.write_json(collection_path.with_suffix(".jsonl.manifest.json"), collection_manifest(rows))
            api = FakeBatchAPI()
            relative_root = Path("results/part64/gen64")
            old_cwd = Path.cwd()
            try:
                os.chdir(base)
                with patch.object(gen64.gen60, "load_splits", return_value={"dev": pilot}):
                    outcomes = gen64.run_probe(collection_path, relative_root, api,
                                               poll_interval=0, sleep=lambda _n: None)
            finally:
                os.chdir(old_cwd)
            self.assertTrue(all(outcome["passed"] for outcome in outcomes))
            self.assertEqual(len(api.posts), 2)
            self.assertTrue((base / relative_root / "run.json").is_file())

    def test_probe_stage_projection_arithmetic_and_opus_stop_line(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            manifest = {"probes": {}, "groups": {}}
            for writer, group_request_count, input_tokens, output_tokens in (
                    ("haiku", 2, 1_000, 1_000), ("opus", 1_000, 1_000, 4_000)):
                probe_relpath = f"probes/{writer}.json"
                probe_path = root / probe_relpath
                probe_request = {"custom_id": "probe"}
                probe_bytes = len(gen64.canonical_json(probe_request))
                gen64.write_json(probe_path, {"state": "settled", "requests": [probe_request]})
                input_tokens = probe_bytes // 4
                output_tokens = 1_000 if writer == "haiku" else 4_000
                gen64.write_json(probe_path.with_name(f"{writer}-results.json"),
                                 {"usage": {"input_tokens": input_tokens,
                                             "output_tokens": output_tokens}})
                group_relpath = f"chunks/{writer}.json"
                gen64.write_json(root / group_relpath,
                                 {"requests": [{"custom_id": str(index)}
                                                for index in range(group_request_count)]})
                manifest["probes"][writer] = {"chunk_path": probe_relpath}
                manifest["groups"][writer] = {"writer": writer, "chunk_paths": [group_relpath]}

            projections = gen64.probe_stage_projections(root, manifest)
            expected = {}
            for writer, output_tokens in (("haiku", 1_000), ("opus", 4_000)):
                probe_path = root / f"probes/{writer}.json"
                probe = gen64.read_json(probe_path)
                usage = gen64.read_json(probe_path.with_name(f"{writer}-results.json"))["usage"]
                input_tokens = usage["input_tokens"]
                group = gen64.read_json(root / f"chunks/{writer}.json")
                probe_bytes = sum(len(gen64.canonical_json(request)) for request in probe["requests"])
                stage_bytes = probe_bytes + sum(len(gen64.canonical_json(request))
                                                for request in group["requests"])
                count = len(probe["requests"]) + len(group["requests"])
                rate_in, rate_out = gen64.BATCH_RATES_USD_PER_MILLION[gen64.WRITERS[writer]]
                expected[writer] = ((input_tokens * stage_bytes / probe_bytes * rate_in
                                     + output_tokens * count * rate_out) / 1_000_000)
                expected_per_request = ((stage_bytes // 4 * rate_in
                                         + output_tokens * count * rate_out) / 1_000_000)
                self.assertAlmostEqual(expected[writer] / expected_per_request, 1.0, places=2)
            self.assertEqual(projections["haiku"]["request_count"], 3)
            self.assertAlmostEqual(projections["haiku"]["projected_usd"], expected["haiku"])
            self.assertEqual(projections["opus"]["request_count"], 1_001)
            self.assertAlmostEqual(projections["opus"]["projected_usd"], expected["opus"])
            self.assertTrue(projections["opus"]["stop"])
            self.assertEqual(projections["opus"]["stop_threshold_usd"], gen64.OPUS_PROBE_STOP_USD)
            self.assertEqual(projections["opus"]["stage_request_bytes"],
                             projections["opus"]["probe_request_bytes"] +
                             sum(len(gen64.canonical_json(request))
                                 for request in gen64.read_json(root / "chunks/opus.json")["requests"]))
            lines = gen64.format_probe_stage_projections(projections)
            self.assertTrue(any(line.startswith("haiku:") for line in lines))
            self.assertTrue(any(line.startswith("opus:") for line in lines))
            self.assertTrue(any(line.startswith("STOP:") for line in lines))

    def test_run_and_resume_gate_recorded_stop_until_operator_override(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            gen64.initialize_plan(root, [collection_row()], "e" * 64)
            projection = {"haiku": {"projected_usd": 1.0}, "opus": {"projected_usd": 36.01}}
            fake_execute = patch.object(gen64, "execute_probes", return_value=[])
            fake_projection = patch.object(gen64, "probe_stage_projections", return_value=projection)
            group_runner = patch.object(gen64, "_run_groups")
            with patch.object(gen64, "load_collection", return_value=([collection_row()], "e" * 64)), \
                    fake_execute, fake_projection, group_runner as run_groups:
                with self.assertRaisesRegex(gen64.WriterError, "recorded Opus probe projection.*STOP line"):
                    gen64.run_generation(Path("unused"), root, lambda *_args, **_kwargs: "{}")
                run_groups.assert_not_called()
                saved = gen64.read_json(root / "run.json")
                self.assertEqual(saved["probe_stage_projections"], projection)
                with self.assertRaisesRegex(gen64.WriterError, "recorded Opus probe projection.*STOP line"):
                    gen64.run_generation(Path("unused"), root, lambda *_args, **_kwargs: "{}",
                                         resuming=True)
                gen64.run_generation(Path("unused"), root, lambda *_args, **_kwargs: "{}",
                                     override_stop="reviewed and accepted projection risk")
                run_groups.assert_called_once()
                saved = gen64.read_json(root / "run.json")
                self.assertEqual(saved["probe_projection_stop_override"]["reason"],
                                 "reviewed and accepted projection risk")
                group_key = next(iter(saved["groups"]))
                chunk_path = root / saved["groups"][group_key]["chunk_paths"][0]
                chunk = gen64.read_json(chunk_path)
                chunk["state"] = "reserved"
                chunk["attempt_id"] = "prior-stage-reserve"
                gen64.write_json(chunk_path, chunk)
                run_groups.reset_mock()
                gen64.run_generation(Path("unused"), root, lambda *_args, **_kwargs: "{}",
                                     resuming=True)
                run_groups.assert_called_once()

            resume_args = gen64.parser().parse_args(["resume", "--root", str(root),
                                                      "--override-stop", "resume after review"])
            self.assertEqual(resume_args.override_stop, "resume after review")


    def test_probe_refuses_empty_truncated_or_unparseable_reply(self):
        cases = (("", "end_turn"), (answer_text(), "max_tokens"), ("not JSON", "end_turn"))
        for text, stop_reason in cases:
            with self.subTest(stop_reason=stop_reason, text=text[:10]), tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
                root = Path(temporary)
                manifest = gen64.initialize_plan(root, [collection_row()], "b" * 64)
                api = FakeBatchAPI(text=text, stop_reason=stop_reason)
                with self.assertRaisesRegex(gen64.WriterError, "probes failed"):
                    gen64.execute_probes(root, manifest, api, poll_interval=0, sleep=lambda _n: None)
                self.assertEqual(len(api.posts), 2)


class Score64MarkerTests(unittest.TestCase):
    def test_score_rows_hands_part60_clean_text_and_marked_span_to_hook(self):
        text = "The shared [[feature]] explains the examples."
        seen = []

        class RecordingScorer:
            def score_batch(self, items, rows, maximum, marked, batch_size=8):
                seen.extend(items)
                maximum[:] = 1.0
                marked[:] = 0.0

        spec = {"unit_key": "L0:u1", "record_index": 0, "type": "neuron", "layer": 0,
                "unit_index": 1, "control_id": None}
        rows = [{"arm": arm, "unit_key": "L0:u1", "type": "neuron", "parse_status": "ok",
                 **{tier: [text] * 4 for tier in ("strong", "weak", "none")}}
                for arm in score64.ARMS]
        score64.score_rows(rows, [spec], {"L0:u1": 0.5}, RecordingScorer())
        clean, span = score64.score60.strip_markers(text)
        self.assertEqual(seen[0], {"text": clean, "marked_span": span})
        self.assertNotIn("[[", seen[0]["text"])


class ChunkRecoveryTests(LocalPaidRegistryTests):
    def make_chunk(self, root: Path, key: str = "haiku-1x-neuron", *, probe: bool = False,
                   count: int = 1, writer: str = "haiku", sample: str = "1x",
                   custom_id_base: str = "u") -> Path:
        entries = []
        for index in range(count):
            row = collection_row(f"L0:u{index + 1}")
            custom_id = f"{custom_id_base}{index:04d}"
            request = gen64.request_for(writer, sample, row, custom_id)
            metadata = {"unit_key": row["unit_key"], "type": "neuron", "sample": sample,
                        "writer": writer, "input_windows": gen64.windows_for_sample(row, sample)}
            entries.append((request, metadata))
        record = gen64._chunk_record(run_id="run-1", group_key="test-chunk", ledger_key=key,
                                     chunk_index=0, writer=writer, sample=sample, unit_type="neuron",
                                     entries=entries, probe=probe)
        path = root / "chunks" / f"{key}.json"
        gen64.write_json(path, record)
        return path

    def make_pipeline_manifest(self, root: Path, *, count: int = 3,
                               key_prefix: str = "haiku-pipeline",
                               custom_id_base: str = "pipe") -> dict:
        paths = []
        for index in range(count):
            path = self.make_chunk(root, key=f"{key_prefix}-{index:02d}",
                                   custom_id_base=f"{custom_id_base}{index}")
            paths.append(str(path.relative_to(root)))
        manifest = {"schema": "test-run", "status": "planned", "groups": {
            "test-chunk": {"chunk_paths": paths}}}
        gen64.write_json(root / "run.json", manifest)
        return manifest

    def assert_pipeline_chunks_settled_once(self, root: Path, manifest: dict) -> None:
        for relpath in manifest["groups"]["test-chunk"]["chunk_paths"]:
            chunk = gen64.read_json(root / relpath)
            self.assertEqual(chunk["state"], "settled")
            ledger = Ledger(str(gen64.ledger_path(root, chunk["ledger_key"])), HARD_CAP_USD)
            with ledger._locked():
                events, _ = ledger._read_locked()
            settlements = [event for event in events
                           if event.get("event") == "settle"
                           and event.get("attempt_id") == chunk["attempt_id"]]
            self.assertEqual(len(settlements), 1)

    def execute_and_check_settled(self, root: Path, path: Path, api: FakeBatchAPI):
        result = gen64.execute_chunk(root, path, api, poll_interval=0, sleep=lambda _n: None)
        self.assertEqual(result["state"], "settled")
        self.assertEqual(len(api.posts), 1)
        self.assertEqual(api.result_gets, 1)
        ledger = Ledger(str(gen64.ledger_path(root, result["ledger_key"])), HARD_CAP_USD)
        self.assertAlmostEqual(ledger.state()["settled"], result["actual_usd"])
        attempts = gen64._attempts(ledger)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(next(iter(attempts.values()))["status"], "settled")
        # Replaying a complete chunk is a no-op for both the provider and ledger.
        gen64.execute_chunk(root, path, api, poll_interval=0, sleep=lambda _n: None)
        self.assertEqual(len(api.posts), 1)
        self.assertEqual(len(gen64._attempts(ledger)), 1)

    def test_pipeline_has_multiple_chunks_in_flight_and_reserve_stays_within_cap(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            manifest = self.make_pipeline_manifest(root, count=4)
            paths = [root / relpath for relpath in manifest["groups"]["test-chunk"]["chunk_paths"]]
            reserves = [gen64.read_json(path)["planned_reserve_usd"] for path in paths]
            cap = math.fsum(reserves[:2]) + 1e-8
            api = FakeBatchAPI(checks_until_end=2)
            reservations = []
            original_reserve = gen64._reserve_with_stage_guard

            def capture_reservation(reserve_root, chunk, ledger):
                attempt_id = original_reserve(reserve_root, chunk, ledger)
                reservations.append(gen64._writer_stage_committed(chunk["writer"]))
                return attempt_id

            with patch.dict(gen64.WRITER_STAGE_CAP_USD, {"haiku": cap}), \
                    patch.object(gen64, "_reserve_with_stage_guard", side_effect=capture_reservation):
                gen64._run_groups(root, manifest, api, poll_interval=0, sleep=lambda _n: None)

            self.assertGreater(api.max_in_flight, 1)
            self.assertTrue(reservations)
            self.assertLessEqual(max(reservations), cap + 1e-8)
            self.assert_pipeline_chunks_settled_once(root, manifest)
            self.assertEqual(len(api.posts), len(paths))
            self.assertEqual(api.result_gets, len(paths))
            posted_ids = [request["custom_id"] for batch in api.posts for request in batch]
            expected_ids = [gen64.read_json(path)["requests"][0]["custom_id"] for path in paths]
            self.assertCountEqual(posted_ids, expected_ids)
            self.assertEqual(len(posted_ids), len(set(posted_ids)))

    def test_stage_cap_refusal_stops_submit_pass_then_retries_after_settlement(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            manifest = self.make_pipeline_manifest(root, count=4, key_prefix="haiku-refusal")
            paths = [root / relpath for relpath in manifest["groups"]["test-chunk"]["chunk_paths"]]
            reserves = [gen64.read_json(path)["planned_reserve_usd"] for path in paths]
            cap = math.fsum(reserves[:2]) + 1e-8
            api = FakeBatchAPI(checks_until_end=2)
            with patch.dict(gen64.WRITER_STAGE_CAP_USD, {"haiku": cap}):
                gen64._run_groups(root, manifest, api, poll_interval=0, sleep=lambda _n: None)

            first_status = next(index for index, event in enumerate(api.events) if event[0] == "status")
            first_results = next(index for index, event in enumerate(api.events) if event[0] == "results")
            third_post = [index for index, event in enumerate(api.events) if event[0] == "post"][2]
            self.assertEqual(sum(event[0] == "post" for event in api.events[:first_status]), 2)
            self.assertGreater(third_post, first_results)
            self.assertEqual(len(api.posts), len(paths))
            self.assert_pipeline_chunks_settled_once(root, manifest)

    def test_pipeline_propagates_non_guard_writer_error(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            manifest = self.make_pipeline_manifest(root, count=1, key_prefix="haiku-error")
            api = FakeBatchAPI()
            with patch.object(gen64, "_reserve_with_stage_guard",
                              side_effect=gen64.WriterError("broken ledger state")):
                with self.assertRaisesRegex(gen64.WriterError, "broken ledger state") as raised:
                    gen64._run_groups(root, manifest, api, poll_interval=0, sleep=lambda _n: None)
            self.assertNotIsInstance(raised.exception, gen64.StageCapRefused)
            self.assertEqual(api.posts, [])

            class InvalidStatusAPI(FakeBatchAPI):
                def __call__(self, method, path, body=None):
                    if method == "GET" and path.startswith("/v1/messages/batches/") \
                            and not path.endswith("/results"):
                        return "[]"
                    return super().__call__(method, path, body)

            invalid_root = Path(temporary) / "invalid-status"
            invalid_manifest = self.make_pipeline_manifest(
                invalid_root, count=1, key_prefix="haiku-invalid-status")
            invalid_api = InvalidStatusAPI()
            with self.assertRaisesRegex(gen64.WriterError, "status is not an object"):
                gen64._run_groups(invalid_root, invalid_manifest, invalid_api,
                                  poll_interval=0, sleep=lambda _n: None)
            self.assertEqual(len(invalid_api.posts), 1)

    def test_resume_after_submit_pass_does_not_duplicate_posts_and_matches_uninterrupted_output(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            base = Path(temporary)
            uninterrupted_root = base / "uninterrupted"
            resumed_root = base / "resumed"
            uninterrupted_root.mkdir()
            resumed_root.mkdir()
            uninterrupted = self.make_pipeline_manifest(
                uninterrupted_root, count=3, key_prefix="haiku-baseline", custom_id_base="same")
            resumed = self.make_pipeline_manifest(
                resumed_root, count=3, key_prefix="haiku-recovered", custom_id_base="same")
            api_uninterrupted = FakeBatchAPI(checks_until_end=2)
            api_resumed = FakeBatchAPI(checks_until_end=2, fail_first_status=True)
            with patch.dict(gen64.WRITER_STAGE_CAP_USD, {"haiku": 10.0}):
                gen64._run_groups(uninterrupted_root, uninterrupted, api_uninterrupted,
                                  poll_interval=0, sleep=lambda _n: None)
                with self.assertRaisesRegex(RuntimeError, "simulated process kill"):
                    gen64._run_groups(resumed_root, resumed, api_resumed,
                                      poll_interval=0, sleep=lambda _n: None)
                saved = gen64.read_json(resumed_root / "run.json")
                for relpath in saved["groups"]["test-chunk"]["chunk_paths"]:
                    self.assertEqual(gen64.read_json(resumed_root / relpath)["state"], "submitted")
                resumed = gen64.read_json(resumed_root / "run.json")
                gen64._run_groups(resumed_root, resumed, api_resumed,
                                  poll_interval=0, sleep=lambda _n: None)

            expected = (uninterrupted_root / "outputs" / "test-chunk.jsonl").read_bytes()
            actual = (resumed_root / "outputs" / "test-chunk.jsonl").read_bytes()
            self.assertEqual(actual, expected)
            resumed_paths = [resumed_root / relpath
                             for relpath in resumed["groups"]["test-chunk"]["chunk_paths"]]
            self.assertEqual(len(api_resumed.posts), len(resumed_paths))
            posted_ids = [request["custom_id"] for batch in api_resumed.posts for request in batch]
            expected_ids = [gen64.read_json(path)["requests"][0]["custom_id"]
                            for path in resumed_paths]
            self.assertCountEqual(posted_ids, expected_ids)
            self.assertEqual(len(posted_ids), len(set(posted_ids)))
            self.assert_pipeline_chunks_settled_once(resumed_root, resumed)

    def test_group_output_waits_until_every_chunk_is_settled(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            manifest = self.make_pipeline_manifest(root, count=3, key_prefix="haiku-output")
            api = FakeBatchAPI(checks_until_end=2)
            original_write = gen64._write_group_output
            writes = []

            def check_then_write(write_root, group_key, group):
                states = [gen64.read_json(write_root / relpath)["state"]
                          for relpath in group["chunk_paths"]]
                self.assertTrue(all(state == "settled" for state in states), states)
                writes.append(group_key)
                return original_write(write_root, group_key, group)

            with patch.object(gen64, "_write_group_output", side_effect=check_then_write):
                gen64._run_groups(root, manifest, api, poll_interval=0, sleep=lambda _n: None)
            self.assertEqual(writes, ["test-chunk"])

    def test_kill_after_reserve_resumes_without_second_reserve_or_duplicate_submit(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            path = self.make_chunk(root)
            api = FakeBatchAPI()
            with self.assertRaisesRegex(RuntimeError, "after_reserve"):
                gen64.execute_chunk(root, path, api, kill_at="after_reserve")
            self.assertEqual(gen64.read_json(path)["state"], "planned")
            self.assertEqual(api.posts, [])
            self.execute_and_check_settled(root, path, api)

    def test_writer_stage_guard_counts_settled_and_reserved_ledgers_after_crash(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            settled_path = self.make_chunk(root, "opus-1x-neuron-part-01", writer="opus")
            reserved_path = self.make_chunk(root, "opus-3x-neuron-part-01", writer="opus", sample="3x")
            refused_path = self.make_chunk(root, "opus-3x-control-part-01", writer="opus", sample="3x")

            gen64.execute_chunk(root, settled_path, FakeBatchAPI(),
                                poll_interval=0, sleep=lambda _n: None)
            with self.assertRaisesRegex(RuntimeError, "after_reserve"):
                gen64.execute_chunk(root, reserved_path, FakeBatchAPI(), kill_at="after_reserve")

            settled_ledger = Ledger(str(gen64.ledger_path(root, "opus-1x-neuron-part-01")), HARD_CAP_USD)
            reserved_ledger = Ledger(str(gen64.ledger_path(root, "opus-3x-neuron-part-01")), HARD_CAP_USD)
            settled = settled_ledger.state()["settled"]
            reserved = reserved_ledger.state()["reserved"]
            refused = gen64.read_json(refused_path)["planned_reserve_usd"]
            self.assertGreater(settled, 0)
            self.assertGreater(reserved, 0)
            boundary = settled + reserved + refused - 1e-9

            with patch.dict(gen64.WRITER_STAGE_CAP_USD, {"opus": boundary}):
                api = FakeBatchAPI()
                with self.assertRaisesRegex(gen64.WriterError, "exceeds registered stage cap"):
                    gen64.execute_chunk(root, refused_path, api, poll_interval=0,
                                        sleep=lambda _n: None)
            self.assertEqual(api.posts, [])
            self.assertEqual(reserved_ledger.state()["reserved"], reserved)

    def test_writer_stage_guard_shares_cap_across_roots_for_same_collection(self):
        rows, _pilot = collection_fixture()
        collection_digest = hashlib.sha256(collect64.collection_jsonl(rows)).hexdigest()
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            base = Path(temporary)
            roots = [base / "root-A", base / "root-B"]
            for root in roots:
                gen64.initialize_plan(root, rows, collection_digest)

            def reserve_until_refused(root: Path) -> float:
                manifest = gen64.read_json(root / "run.json")
                paths = [relpath
                         for group_key in ("opus-3x-neuron", "opus-3x-control")
                         for relpath in manifest["groups"][group_key]["chunk_paths"]]
                reserved = 0.0
                for relpath in paths:
                    chunk = gen64.read_json(root / relpath)
                    ledger = Ledger(str(gen64.ledger_path(root, chunk["ledger_key"])), HARD_CAP_USD)
                    try:
                        gen64._reserve_with_stage_guard(root, chunk, ledger)
                    except gen64.WriterError:
                        return reserved
                    reserved += chunk["planned_reserve_usd"]
                return reserved

            with patch.dict(gen64.WRITER_STAGE_CAP_USD, {"opus": 5.0}):
                first_root_reserved = reserve_until_refused(roots[0])
                second_root_reserved = reserve_until_refused(roots[1])

            combined = first_root_reserved + second_root_reserved
            self.assertGreater(first_root_reserved, 0)
            self.assertLess(second_root_reserved, first_root_reserved)
            self.assertLessEqual(combined, 5.0 + 1e-8)
            self.assertEqual(gen64.ledger_path(roots[0], "opus-3x-neuron-part-01"),
                             gen64.ledger_path(roots[1], "opus-3x-neuron-part-01"))

    def test_kill_after_post_before_id_never_resubmits_ambiguous_request(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            path = self.make_chunk(root)
            api = FakeBatchAPI()
            with self.assertRaisesRegex(RuntimeError, "after_post_before_id"):
                gen64.execute_chunk(root, path, api, kill_at="after_post_before_id")
            self.assertEqual(gen64.read_json(path)["state"], "posting")
            with self.assertRaisesRegex(gen64.RecoveryRequired, "outcome is unknown"):
                gen64.execute_chunk(root, path, api)
            self.assertEqual(gen64.read_json(path)["state"], "ambiguous")
            self.assertEqual(len(api.posts), 1)
            ledger = Ledger(str(gen64.ledger_path(root, "haiku-1x-neuron")), HARD_CAP_USD)
            self.assertEqual(ledger.state()["reserved"], gen64.read_json(path)["planned_reserve_usd"])

    def test_lost_response_batch_is_recovered_from_provider_without_resubmission(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            path = self.make_chunk(root)
            api = AcceptedLostResponseAPI()
            with self.assertRaisesRegex(gen64.RecoveryRequired, "outcome is unknown"):
                gen64.execute_chunk(root, path, api)
            chunk = gen64.read_json(path)
            gen64.write_json(root / "run.json", {"schema": "test", "groups": {
                "test-chunk": {"chunk_paths": [str(path.relative_to(root))]}}, "probes": {}})
            with self.assertRaisesRegex(gen64.RecoveryRequired, "contain this chunk's custom IDs"):
                gen64.retire_attempt(root, chunk["attempt_id"], "operator", "response was lost", api)
            result = gen64.execute_chunk(root, path, api, poll_interval=0, sleep=lambda _n: None)
            self.assertEqual(result["state"], "settled")
            self.assertEqual(len(api.posts), 1)
            self.assertEqual(result["batch_id"], "batch-1")

    def test_kill_after_id_recovers_batch_without_resubmit(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            path = self.make_chunk(root)
            api = FakeBatchAPI()
            api.chunk_path = path
            with self.assertRaisesRegex(RuntimeError, "after_id_before_results"):
                gen64.execute_chunk(root, path, api, kill_at="after_id_before_results")
            self.assertEqual(gen64.read_json(path)["state"], "submitted")
            self.execute_and_check_settled(root, path, api)
            self.assertTrue(api.durable_id_seen_before_poll)

    def test_kill_after_results_settles_once_from_durable_usage(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            path = self.make_chunk(root)
            api = FakeBatchAPI()
            with self.assertRaisesRegex(RuntimeError, "after_results_before_settle"):
                gen64.execute_chunk(root, path, api, kill_at="after_results_before_settle",
                                    poll_interval=0, sleep=lambda _n: None)
            self.assertEqual(gen64.read_json(path)["state"], "results_saved")
            result = gen64.execute_chunk(root, path, api, poll_interval=0, sleep=lambda _n: None)
            self.assertEqual(result["state"], "settled")
            self.assertEqual(len(api.posts), 1)
            self.assertEqual(api.result_gets, 1)
            ledger = Ledger(str(gen64.ledger_path(root, result["ledger_key"])), HARD_CAP_USD)
            self.assertAlmostEqual(ledger.state()["settled"], result["actual_usd"])
            self.assertEqual(len(gen64._attempts(ledger)), 1)

    def test_operator_retirement_unlocks_only_an_explicit_new_attempt(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            path = self.make_chunk(root)
            api = FakeBatchAPI()
            with self.assertRaisesRegex(RuntimeError, "after_post_before_id"):
                gen64.execute_chunk(root, path, api, kill_at="after_post_before_id")
            chunk = gen64.read_json(path)
            # Record this single chunk in a tiny run manifest so the CLI retirement path can find it.
            gen64.write_json(root / "run.json", {"schema": "test", "groups": {
                "test-chunk": {"chunk_paths": [str(path.relative_to(root))]}}, "probes": {}})
            # The provider lists and ends the batch, proving that this accepted
            # POST exists. Retirement must refuse until it is recovered.
            with self.assertRaisesRegex(gen64.RecoveryRequired, "may be the unknown POST"):
                gen64.retire_attempt(root, chunk["attempt_id"], "operator", "provider listing reviewed", api)
            with self.assertRaises(gen64.RecoveryRequired):
                gen64.execute_chunk(root, path, api)
            self.assertEqual(len(api.posts), 1)
            # Simulate a separate unknown POST proven absent from the complete
            # provider listing; only then may an explicit retry be authorized.
            no_accept = NoAcceptThenOkAPI()
            path2 = self.make_chunk(root, key="haiku-1x-neuron-absent")
            manifest = gen64.read_json(root / "run.json")
            manifest["groups"]["test-chunk"]["chunk_paths"].append(str(path2.relative_to(root)))
            gen64.write_json(root / "run.json", manifest)
            with self.assertRaises(gen64.RecoveryRequired):
                gen64.execute_chunk(root, path2, no_accept)
            chunk2 = gen64.read_json(path2)
            gen64.retire_attempt(root, chunk2["attempt_id"], "operator", "listing has no matching request", no_accept)
            self.assertEqual(gen64.read_json(path2)["state"], "retired")
            result = gen64.execute_chunk(root, path2, no_accept, poll_interval=0, sleep=lambda _n: None)
            self.assertEqual(result["state"], "settled")
            self.assertEqual(result["attempt_no"], 1)

    def test_errored_batch_rows_settle_only_usage_rows_and_refuse_labels(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            path = self.make_chunk(root, count=2)
            api = FakeBatchAPI(errored_custom_ids={"u0000"})
            result = gen64.execute_chunk(root, path, api, poll_interval=0, sleep=lambda _n: None)
            self.assertEqual(result["state"], "settled")
            bundle = gen64.read_json(path.with_name(path.stem + "-results.json"))
            self.assertEqual([row["parse_status"] for row in bundle["rows"]], ["api_errored", "ok"])
            self.assertEqual(bundle["usage"]["input_tokens"], 321)
            self.assertEqual(bundle["usage"]["output_tokens"], 1234)
            self.assertEqual(bundle["usage"]["non_succeeded_custom_ids"], ["u0000"])
            self.assertEqual(bundle["actual_usd"], result["actual_usd"])
            failed = score64._normalize_generator_row(bundle["rows"][0])
            self.assertEqual(score64._valid_texts(failed), (None, "api_errored"))

    def test_max_tokens_is_recorded_as_truncation_and_not_a_valid_label(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            path = self.make_chunk(root)
            api = FakeBatchAPI(stop_reason="max_tokens")
            gen64.execute_chunk(root, path, api, poll_interval=0, sleep=lambda _n: None)
            bundle = gen64.read_json(path.with_name(path.stem + "-results.json"))
            self.assertEqual(bundle["rows"][0]["parse_status"], "max_tokens")
            failed = score64._normalize_generator_row(bundle["rows"][0])
            self.assertEqual(score64._valid_texts(failed), (None, "max_tokens"))

    def test_release_before_local_rejected_write_is_recovered_from_ledger(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            path = self.make_chunk(root)

            class RejectAPI:
                posts = 0

                def __call__(self, method, _path, body=None):
                    if method == "POST":
                        self.posts += 1
                        raise HTTPFailure(400, '{"type":"error","error":{"type":"invalid_request_error"}}')
                    raise AssertionError("a rejected POST must not poll")

            api = RejectAPI()
            write_json = gen64.write_json

            def fail_rejected_write(target, value):
                if isinstance(value, dict) and value.get("state") == "rejected":
                    raise RuntimeError("simulated crash after ledger release")
                return write_json(target, value)

            with patch.object(gen64, "write_json", side_effect=fail_rejected_write):
                with self.assertRaisesRegex(RuntimeError, "after ledger release"):
                    gen64.execute_chunk(root, path, api)
            with self.assertRaisesRegex(gen64.RecoveryRequired, "marked rejected"):
                gen64.execute_chunk(root, path, api)
            self.assertEqual(gen64.read_json(path)["state"], "rejected")
            self.assertEqual(api.posts, 1)

    def test_copied_run_root_is_rejected_by_the_run_id_ledger_guard(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            base = Path(temporary)
            root_a, root_b = base / "a", base / "b"
            manifest = gen64.initialize_plan(root_a, [collection_row()], "f" * 64)
            shutil.copytree(root_a, root_b)
            relpath = manifest["groups"]["haiku-1x-neuron"]["chunk_paths"][0]
            registry = base / "run-paths.json"
            with patch.object(paid_ledger, "_RUN_PATH_REGISTRY", registry), \
                    patch.object(paid_ledger, "_RUN_PATH_LOCK", registry.with_suffix(".lock")):
                api_a, api_b = FakeBatchAPI(), FakeBatchAPI()
                gen64.execute_chunk(root_a, root_a / relpath, api_a, poll_interval=0, sleep=lambda _n: None)
                with self.assertRaisesRegex(paid_ledger.LedgerError, "already uses ledger"):
                    gen64.execute_chunk(root_b, root_b / relpath, api_b, poll_interval=0, sleep=lambda _n: None)
                self.assertEqual(len(api_a.posts), 1)
                self.assertEqual(api_b.posts, [])

    def test_run_generation_cannot_skip_failed_writer_probes(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            api = FakeBatchAPI(text="not valid M1 JSON")
            with patch.object(gen64, "load_collection", return_value=([collection_row()], "e" * 64)):
                with self.assertRaisesRegex(gen64.WriterError, "generation refused until both writer probes pass"):
                    gen64.run_generation(Path("unused"), root, api, initialize=True,
                                         poll_interval=0, sleep=lambda _n: None)
            self.assertEqual(len(api.posts), 2)
            self.assertEqual({request["custom_id"] for batch in api.posts for request in batch},
                             {gen64.read_json(root / "probes/haiku.json")["requests"][0]["custom_id"],
                              gen64.read_json(root / "probes/opus.json")["requests"][0]["custom_id"]})

    def test_failed_probe_can_be_retried_only_as_recorded_new_attempt(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            manifest = gen64.initialize_plan(root, [collection_row()], "f" * 64)
            with self.assertRaises(gen64.WriterError):
                gen64.execute_probes(root, manifest, FakeBatchAPI(text="not JSON"),
                                     poll_interval=0, sleep=lambda _n: None)
            outcome = gen64.reprobe(root, "haiku", "operator", "first probe was not parseable",
                                    FakeBatchAPI(), poll_interval=0, sleep=lambda _n: None)
            self.assertTrue(outcome["passed"])
            self.assertEqual(outcome["reprobe_authorization"]["reason"], "first probe was not parseable")
            saved_manifest = gen64.read_json(root / "run.json")
            self.assertIn("attempt-001", saved_manifest["probes"]["haiku"]["chunk_path"])
            reprobe_chunk = gen64.read_json(root / saved_manifest["probes"]["haiku"]["chunk_path"])
            self.assertEqual(reprobe_chunk["attempt_no"], 1)

    def test_reprobe_has_fresh_custom_id_and_recovers_accepted_lost_post(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            manifest = gen64.initialize_plan(root, [collection_row()], "f" * 64)
            api = LostReprobeResponseAPI()
            original_path = root / manifest["probes"]["haiku"]["chunk_path"]
            gen64.execute_chunk(root, original_path, api, poll_interval=0, sleep=lambda _n: None)
            original_id = gen64.read_json(original_path)["requests"][0]["custom_id"]

            api.text = answer_text()
            api.lose_next_post = True
            with self.assertRaisesRegex(gen64.RecoveryRequired, "outcome is unknown"):
                gen64.reprobe(root, "haiku", "operator", "retry the invalid probe", api,
                              poll_interval=0, sleep=lambda _n: None)

            saved = gen64.read_json(root / "run.json")
            reprobe_path = root / saved["probes"]["haiku"]["chunk_path"]
            reprobe = gen64.read_json(reprobe_path)
            new_id = reprobe["requests"][0]["custom_id"]
            self.assertNotEqual(original_id, new_id)
            self.assertEqual(api.posts[0][0]["custom_id"], original_id)
            self.assertEqual(api.posts[1][0]["custom_id"], new_id)

            result = gen64.execute_chunk(root, reprobe_path, api, poll_interval=0,
                                         sleep=lambda _n: None)
            self.assertEqual(result["state"], "settled")
            self.assertEqual(result["batch_id"], "batch-2")
            self.assertEqual(len(api.posts), 2)

    def test_reprobe_custom_ids_use_stored_base_and_stay_within_64_characters(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            manifest = gen64.initialize_plan(root, [collection_row()], "e" * 64)
            api = FakeBatchAPI(text="not JSON")
            original_path = root / manifest["probes"]["haiku"]["chunk_path"]
            gen64.execute_chunk(root, original_path, api, poll_interval=0, sleep=lambda _n: None)
            original = gen64.read_json(original_path)
            base_custom_id = original["base_custom_ids"][0]
            seen = [original["requests"][0]["custom_id"]]

            for attempt_no in range(1, 5):
                with self.assertRaisesRegex(gen64.WriterError, "re-probe failed validation"):
                    gen64.reprobe(root, "haiku", "operator", "probe remains invalid", api,
                                  poll_interval=0, sleep=lambda _n: None)
                saved = gen64.read_json(root / "run.json")
                reprobe_path = root / saved["probes"]["haiku"]["chunk_path"]
                reprobe_chunk = gen64.read_json(reprobe_path)
                custom_id = reprobe_chunk["requests"][0]["custom_id"]
                self.assertEqual(custom_id, f"{base_custom_id}-attempt-{attempt_no:03d}")
                self.assertEqual(reprobe_chunk["base_custom_ids"], [base_custom_id])
                self.assertLessEqual(len(custom_id), 64)
                seen.append(custom_id)

            self.assertTrue(all(len(custom_id) <= 64 for custom_id in seen))

    def test_reprobe_recovery_ignores_older_ended_batch_even_if_id_matches(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            path = self.make_chunk(root, probe=True)
            chunk = gen64.read_json(path)
            chunk["posting_started_at"] = gen64.utc_now()
            chunk["attempt_no"] = 1
            base_custom_id = chunk["requests"][0]["custom_id"]
            chunk["requests"][0]["custom_id"] = f"{base_custom_id}-attempt-001"

            class OldMatchingBatchAPI:
                def __call__(self, method, api_path, body=None):
                    if method == "GET" and api_path.startswith("/v1/messages/batches?"):
                        return json.dumps({"data": [{"id": "old-batch",
                                                       "created_at": "2000-01-01T00:00:00Z",
                                                       "processing_status": "ended"}],
                                           "has_more": False})
                    if method == "GET" and api_path.endswith("/results"):
                        return json.dumps({"custom_id": chunk["requests"][0]["custom_id"],
                                           "result": {"type": "errored"}}) + "\n"
                    raise AssertionError((method, api_path))

            matches, unresolved = gen64._find_provider_batches(OldMatchingBatchAPI(), chunk)
            self.assertEqual(matches, [])
            self.assertEqual(unresolved, [])
            self.assertEqual(gen64._base_custom_id(f"{base_custom_id}-attempt-1000"), base_custom_id)

    def test_retired_initial_probe_retry_does_not_get_reprobe_time_window(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            path = self.make_chunk(Path(temporary), probe=True)
            chunk = gen64.read_json(path)
            chunk["posting_started_at"] = "2026-10-23T12:00:00Z"
            chunk["attempt_no"] = 1
            custom_id = chunk["requests"][0]["custom_id"]

            class EarlierMatchingBatchAPI:
                def __call__(self, method, api_path, body=None):
                    if method == "GET" and api_path.startswith("/v1/messages/batches?"):
                        return json.dumps({"data": [{"id": "earlier-accepted-batch",
                                                       "created_at": "2026-10-23T11:50:00Z",
                                                       "processing_status": "ended"}],
                                           "has_more": False})
                    if method == "GET" and api_path.endswith("/results"):
                        return json.dumps({"custom_id": custom_id,
                                           "result": {"type": "errored"}}) + "\n"
                    raise AssertionError((method, api_path))

            with self.assertRaisesRegex(gen64.RecoveryRequired, "contain this chunk's custom IDs"):
                gen64._ensure_no_provider_batch(EarlierMatchingBatchAPI(), chunk)

    def test_retire_guard_matches_old_group_batch_without_five_minute_window(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            path = self.make_chunk(Path(temporary), probe=False)
            chunk = gen64.read_json(path)
            chunk["posting_started_at"] = "2026-10-23T12:00:00Z"
            custom_id = chunk["requests"][0]["custom_id"]

            class EarlierMatchingBatchAPI:
                def __call__(self, method, api_path, body=None):
                    if method == "GET" and api_path.startswith("/v1/messages/batches?"):
                        return json.dumps({"data": [{"id": "earlier-accepted-batch",
                                                       "created_at": "2026-10-23T11:50:00Z",
                                                       "processing_status": "ended"}],
                                           "has_more": False})
                    if method == "GET" and api_path.endswith("/results"):
                        return json.dumps({"custom_id": custom_id,
                                           "result": {"type": "errored"}}) + "\n"
                    raise AssertionError((method, api_path))

            with self.assertRaisesRegex(gen64.RecoveryRequired, "contain this chunk's custom IDs"):
                gen64._ensure_no_provider_batch(EarlierMatchingBatchAPI(), chunk)

    def test_every_opus_ledger_share_stays_below_the_per_ledger_cap(self):
        rows = [collection_row(f"L0:u{index}") for index in range(250)]
        for sample, expected_partitions in (("1x", 3), ("3x", 6)):
            partitions = gen64._ledger_partitions("opus", sample, rows)
            self.assertEqual(len(partitions), expected_partitions)
            self.assertEqual(sum(map(len, partitions)), 250)
            for partition in partitions:
                requests = [gen64.request_for("opus", sample, row, f"u{index:04d}")
                            for index, row in enumerate(partition)]
                reserve = gen64.request_reserve_usd(requests, gen64.WRITERS["opus"])
                self.assertLess(reserve, HARD_CAP_USD)

    def test_built_plan_keeps_each_full_ledger_share_under_eight_dollars(self):
        rows, _pilot = collection_fixture()
        _manifest, chunks = gen64._build_plan(rows, "a" * 64, run_id="ledgerbudget")
        per_ledger = {}
        for chunk in chunks.values():
            key = chunk["ledger_key"]
            per_ledger[key] = per_ledger.get(key, 0.0) + chunk["planned_reserve_usd"]
        self.assertEqual(gen64.WRITER_STAGE_CAP_USD, {"haiku": 4.0, "opus": 40.0})
        self.assertIn("opus-1x-neuron-part-01", per_ledger)
        self.assertIn("opus-3x-control-part-06", per_ledger)
        for key, reserve in per_ledger.items():
            with self.subTest(ledger=key):
                self.assertLess(reserve, HARD_CAP_USD)

    # 2026-10-23 live stall: chunk-0015's POST got a connection reset, and recovery could not list the account's
    # 191 batches (before_id paging plus a cursor check that rejected every valid cursor), so every pass aborted.
    def test_listing_pages_older_with_after_id_and_finds_a_match_on_page_two(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            chunk = gen64.read_json(self.make_chunk(Path(temporary)))
            chunk["posting_started_at"] = "2026-10-23T12:00:00Z"
            mine = chunk["requests"][0]["custom_id"]
            ids = [f"b{index:03d}" for index in range(150)]  # newest first, like the provider

            class PagedAPI:
                def __init__(self):
                    self.list_calls = 0

                def __call__(self, method, api_path, body=None):
                    if method == "GET" and api_path.startswith("/v1/messages/batches?"):
                        self.list_calls += 1
                        if "before_id=" in api_path:
                            raise AssertionError("before_id pages toward newer batches")
                        start = ids.index(api_path.split("after_id=")[1]) + 1 if "after_id=" in api_path else 0
                        page = ids[start:start + 100]
                        return json.dumps({"data": [{"id": b, "created_at": "2026-10-23T12:01:00Z",
                                                     "processing_status": "ended"} for b in page],
                                           "has_more": start + 100 < len(ids),
                                           "first_id": page[0], "last_id": page[-1]})
                    if method == "GET" and api_path.endswith("/results"):
                        batch_id = api_path.split("/")[-2]
                        custom_id = mine if batch_id == "b120" else f"other-{batch_id}"
                        return json.dumps({"custom_id": custom_id, "result": {"type": "errored"}}) + "\n"
                    raise AssertionError((method, api_path))

            api = PagedAPI()
            matches, unresolved = gen64._find_provider_batches(api, chunk)
            self.assertEqual((matches, unresolved), (["b120"], []))
            self.assertEqual(api.list_calls, 2)

    def test_ended_batch_results_are_fetched_once_across_recovery_passes(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            chunk = gen64.read_json(self.make_chunk(Path(temporary)))
            chunk["posting_started_at"] = "2026-10-23T12:00:00Z"

            class OneEndedBatchAPI:
                def __init__(self):
                    self.result_gets = 0

                def __call__(self, method, api_path, body=None):
                    if method == "GET" and api_path.startswith("/v1/messages/batches?"):
                        return json.dumps({"data": [{"id": "old", "created_at": "2026-10-23T09:00:00Z",
                                                     "processing_status": "ended"}], "has_more": False})
                    if method == "GET" and api_path.endswith("/results"):
                        self.result_gets += 1
                        return json.dumps({"custom_id": "someone-else", "result": {"type": "errored"}}) + "\n"
                    raise AssertionError((method, api_path))

            api = OneEndedBatchAPI()
            for _ in range(3):
                self.assertEqual(gen64._find_provider_batches(api, chunk), ([], []))
            self.assertEqual(api.result_gets, 1)

    def test_pipeline_goes_on_past_an_unaccepted_post_then_asks_a_person(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            manifest = self.make_pipeline_manifest(root, count=3, key_prefix="haiku-unaccepted")
            paths = [root / relpath for relpath in manifest["groups"]["test-chunk"]["chunk_paths"]]

            class SecondPostNeverArrivesAPI(FakeBatchAPI):
                def __init__(self):
                    super().__init__()
                    self.post_calls = 0

                def __call__(self, method, api_path, body=None):
                    if method == "POST":
                        self.post_calls += 1
                        if self.post_calls == 2:
                            raise ConnectionResetError("connection reset before the provider accepted it")
                    return super().__call__(method, api_path, body)

            api = SecondPostNeverArrivesAPI()
            with self.assertRaisesRegex(gen64.RecoveryRequired, "no ended matching batch"):
                gen64._run_groups(root, manifest, api, poll_interval=0, sleep=lambda _n: None, pipeline_bound=30)
            states = [gen64.read_json(path)["state"] for path in paths]
            self.assertEqual(states, ["settled", "ambiguous", "settled"])
            self.assertEqual(len(api.posts), 2)  # the unknown POST was never sent again
            ledger = Ledger(str(gen64.ledger_path(root, gen64.read_json(paths[1])["ledger_key"])), HARD_CAP_USD)
            self.assertEqual(next(iter(gen64._attempts(ledger).values()))["status"], "reserved")

    def test_pipeline_adopts_an_accepted_lost_post_once_its_batch_ends(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            manifest = self.make_pipeline_manifest(root, count=2, key_prefix="haiku-lost")

            class FirstResponseLostAPI(FakeBatchAPI):
                def __init__(self):
                    super().__init__()
                    self.lost: str | None = None
                    self.listings = 0

                def __call__(self, method, api_path, body=None):
                    if method == "POST" and self.lost is None:
                        super().__call__(method, api_path, body)
                        self.lost = f"batch-{len(self.posts)}"
                        raise TimeoutError("accepted, response lost")
                    if method == "GET" and api_path.startswith("/v1/messages/batches?"):
                        self.listings += 1
                        if self.listings >= 2:  # the lost batch finishes while the run goes on
                            self.batch_records[self.lost]["processing_status"] = "ended"
                    return super().__call__(method, api_path, body)

            api = FirstResponseLostAPI()
            gen64._run_groups(root, manifest, api, poll_interval=0, sleep=lambda _n: None, pipeline_bound=30)
            self.assert_pipeline_chunks_settled_once(root, manifest)
            self.assertEqual(len(api.posts), 2)
            first = gen64.read_json(root / manifest["groups"]["test-chunk"]["chunk_paths"][0])
            self.assertEqual(first["batch_id"], api.lost)

    # Review round 1 (DS F1, Haiku F1-F5): wait only on batches still in flight; an unreadable ended batch, a 5xx
    # POST, a duplicate and the chunk's identity each have their own expected outcome.
    def _two_chunk_run(self, root, prefix):
        manifest = self.make_pipeline_manifest(root, count=2, key_prefix=prefix)
        return manifest, [root / relpath for relpath in manifest["groups"]["test-chunk"]["chunk_paths"]]

    def test_unreadable_ended_batch_stops_for_a_person_instead_of_waiting(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            manifest, paths = self._two_chunk_run(root, "haiku-unreadable")

            class UnreadableOldBatchAPI(FakeBatchAPI):
                def __init__(self):
                    super().__init__()
                    self.post_calls = 0
                    self.batch_records["b-old"] = {"id": "b-old", "processing_status": "ended",
                                                   "created_at": "2026-10-20T00:00:00Z"}

                def __call__(self, method, api_path, body=None):
                    if method == "POST":
                        self.post_calls += 1
                        if self.post_calls == 1:
                            raise ConnectionResetError("reset before acceptance")
                    if method == "GET" and api_path == "/v1/messages/batches/b-old/results":
                        raise TimeoutError("results endpoint keeps failing")
                    return super().__call__(method, api_path, body)

            sleeps = []
            with self.assertRaisesRegex(gen64.RecoveryRequired, "haiku-unreadable-00"):
                gen64._run_groups(root, manifest, UnreadableOldBatchAPI(), poll_interval=0,
                                  sleep=sleeps.append, pipeline_bound=30)
            self.assertLessEqual(len(sleeps), 2)  # one pass for the POST-time look, no 30-hour wait
            self.assertEqual([gen64.read_json(p)["state"] for p in paths], ["ambiguous", "settled"])

    def test_deferred_chunk_waits_while_a_candidate_batch_is_in_flight(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            manifest, paths = self._two_chunk_run(root, "haiku-waits")

            class SlowCandidateAPI(FakeBatchAPI):
                def __init__(self):
                    super().__init__()
                    self.post_calls = 0
                    self.listings = 0
                    self.batch_records["b-other"] = {"id": "b-other", "processing_status": "in_progress",
                                                     "created_at": gen64.utc_now()}
                    self.batch_requests["b-other"] = [{"custom_id": "someone-else"}]

                def __call__(self, method, api_path, body=None):
                    if method == "POST":
                        self.post_calls += 1
                        if self.post_calls == 1:
                            raise ConnectionResetError("reset before acceptance")
                    if method == "GET" and api_path.startswith("/v1/messages/batches?"):
                        self.listings += 1
                        if self.listings >= 4:
                            self.batch_records["b-other"]["processing_status"] = "ended"
                    return super().__call__(method, api_path, body)

            sleeps = []
            api = SlowCandidateAPI()
            with self.assertRaisesRegex(gen64.RecoveryRequired, "no ended matching batch"):
                gen64._run_groups(root, manifest, api, poll_interval=0, sleep=sleeps.append, pipeline_bound=30)
            # One sleep after the failed POST, then one per listing while b-other ran; it ended on the fourth.
            self.assertEqual(api.listings, 4)
            self.assertEqual(len(sleeps), api.listings)  # plus one pass after the POST, before any listing
            self.assertEqual([gen64.read_json(p)["state"] for p in paths], ["ambiguous", "settled"])

    def test_http_5xx_post_is_deferred_and_the_run_goes_on(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            manifest, paths = self._two_chunk_run(root, "haiku-504")

            class GatewayTimeoutFirstAPI(FakeBatchAPI):
                def __init__(self):
                    super().__init__()
                    self.post_calls = 0

                def __call__(self, method, api_path, body=None):
                    if method == "POST":
                        self.post_calls += 1
                        if self.post_calls == 1:
                            raise gen64.HTTPFailure(504, "gateway timeout")
                    return super().__call__(method, api_path, body)

            api = GatewayTimeoutFirstAPI()
            with self.assertRaisesRegex(gen64.RecoveryRequired, "haiku-504-00"):
                gen64._run_groups(root, manifest, api, poll_interval=0, sleep=lambda _n: None, pipeline_bound=30)
            self.assertEqual([gen64.read_json(p)["state"] for p in paths], ["ambiguous", "settled"])
            self.assertEqual(api.post_calls, 2)
            ledger = Ledger(str(gen64.ledger_path(root, gen64.read_json(paths[0])["ledger_key"])), HARD_CAP_USD)
            self.assertEqual(next(iter(gen64._attempts(ledger).values()))["status"], "reserved")

    def test_two_matching_batches_stop_the_run_as_a_possible_duplicate(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            path = self.make_chunk(root)
            mine = gen64.read_json(path)["requests"][0]["custom_id"]

            def reset_before_acceptance(method, api_path, body=None):
                raise ConnectionResetError("reset before acceptance")

            with self.assertRaises(gen64.RecoveryRequired):
                gen64.execute_chunk(root, path, reset_before_acceptance, poll_interval=0, sleep=lambda _n: None)
            self.assertEqual(gen64.read_json(path)["state"], "ambiguous")

            class TwoMatchesAPI:
                def __call__(self, method, api_path, body=None):
                    if method == "GET" and api_path.startswith("/v1/messages/batches?"):
                        return json.dumps({"data": [{"id": b, "created_at": gen64.utc_now(),
                                                     "processing_status": "ended"} for b in ("b1", "b2")],
                                           "has_more": False})
                    if method == "GET" and api_path.endswith("/results"):
                        return json.dumps({"custom_id": mine, "result": {"type": "errored"}}) + "\n"
                    raise AssertionError((method, api_path))

            with self.assertRaisesRegex(gen64.RecoveryRequired, "possible duplicate") as raised:
                gen64.execute_chunk(root, path, TwoMatchesAPI(), poll_interval=0, sleep=lambda _n: None)
            self.assertFalse(getattr(raised.exception, "unknown_post", False))

    def test_a_listing_blip_is_retried_and_adopts_the_accepted_batch(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            manifest, paths = self._two_chunk_run(root, "haiku-blip")

            class LostResponseThenListingBlipAPI(FakeBatchAPI):
                def __init__(self):
                    super().__init__()
                    self.lost: str | None = None
                    self.listings = 0

                def __call__(self, method, api_path, body=None):
                    if method == "POST" and self.lost is None:
                        super().__call__(method, api_path, body)
                        self.lost = f"batch-{len(self.posts)}"
                        self.batch_records[self.lost]["processing_status"] = "ended"
                        raise TimeoutError("accepted, response lost")
                    if method == "GET" and api_path.startswith("/v1/messages/batches?"):
                        self.listings += 1
                        if self.listings == 1:
                            raise ConnectionResetError("listing blip")
                    return super().__call__(method, api_path, body)

            api = LostResponseThenListingBlipAPI()
            gen64._run_groups(root, manifest, api, poll_interval=0, sleep=lambda _n: None, pipeline_bound=30)
            self.assert_pipeline_chunks_settled_once(root, manifest)
            self.assertEqual(len(api.posts), 2)
            self.assertGreaterEqual(api.listings, 2)

    def test_a_malformed_listing_stops_the_run_instead_of_spending_on(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            manifest = self.make_pipeline_manifest(root, count=3, key_prefix="haiku-malformed")

            class MalformedListingAPI(FakeBatchAPI):
                def __init__(self):
                    super().__init__(checks_until_end=5)  # the other batches are still running at the second pass
                    self.post_calls = 0

                def __call__(self, method, api_path, body=None):
                    if method == "POST":
                        self.post_calls += 1
                        if self.post_calls == 1:
                            raise ConnectionResetError("reset before acceptance")
                    if method == "GET" and api_path.startswith("/v1/messages/batches?"):
                        return json.dumps({"data": [{"id": "no-created-at", "processing_status": "ended"}],
                                           "has_more": False})
                    return super().__call__(method, api_path, body)

            api = MalformedListingAPI()
            with self.assertRaisesRegex(gen64.RecoveryRequired, "no valid created_at"):
                gen64._run_groups(root, manifest, api, poll_interval=0, sleep=lambda _n: None, pipeline_bound=30)
            paths = [root / relpath for relpath in manifest["groups"]["test-chunk"]["chunk_paths"]]
            # It stops at the first unreadable listing, with the others still in flight, rather than running on.
            self.assertEqual([gen64.read_json(p)["state"] for p in paths], ["ambiguous", "submitted", "submitted"])

    def test_an_old_unrelated_batch_in_flight_does_not_hold_the_stop(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            manifest, paths = self._two_chunk_run(root, "haiku-ancient")

            class AncientBatchAPI(FakeBatchAPI):
                def __init__(self):
                    super().__init__()
                    self.post_calls = 0
                    self.batch_records["b-ancient"] = {"id": "b-ancient", "processing_status": "in_progress",
                                                       "created_at": "2026-10-01T00:00:00Z"}

                def __call__(self, method, api_path, body=None):
                    if method == "POST":
                        self.post_calls += 1
                        if self.post_calls == 1:
                            raise ConnectionResetError("reset before acceptance")
                    return super().__call__(method, api_path, body)

            sleeps = []
            # b-ancient predates the POST by weeks, so it cannot be it: the run stops for a person, not a timeout.
            with self.assertRaisesRegex(gen64.RecoveryRequired, "haiku-ancient-00"):
                gen64._run_groups(root, manifest, AncientBatchAPI(), poll_interval=0, sleep=sleeps.append,
                                  pipeline_bound=30)
            self.assertLessEqual(len(sleeps), 2)

    def test_a_listing_that_keeps_failing_stops_after_a_few_passes(self):
        with tempfile.TemporaryDirectory(dir=gen64.HERE) as temporary:
            root = Path(temporary)
            manifest, paths = self._two_chunk_run(root, "haiku-listing-down")

            class ListingDownAPI(FakeBatchAPI):
                def __init__(self):
                    super().__init__()
                    self.post_calls = 0
                    self.listings = 0

                def __call__(self, method, api_path, body=None):
                    if method == "POST":
                        self.post_calls += 1
                        if self.post_calls == 1:
                            raise ConnectionResetError("reset before acceptance")
                    if method == "GET" and api_path.startswith("/v1/messages/batches?"):
                        self.listings += 1
                        raise ConnectionResetError("listing down")
                    return super().__call__(method, api_path, body)

            api = ListingDownAPI()
            with self.assertRaisesRegex(gen64.RecoveryRequired, "haiku-listing-down-00.*listing will be retried"):
                gen64._run_groups(root, manifest, api, poll_interval=0, sleep=lambda _n: None, pipeline_bound=30)
            self.assertEqual(api.listings, gen64.LISTING_RETRY_PASSES)


if __name__ == "__main__":
    unittest.main()
