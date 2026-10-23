"""Offline tests for Part 64 M1 requests and durable paid chunk recovery."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil
import statistics
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
                 errored_custom_ids: set[str] | None = None):
        self.text = answer_text() if text is None else text
        self.stop_reason = stop_reason
        self.errored_custom_ids = set(errored_custom_ids or ())
        self.posts: list[list[dict]] = []
        self.batch_requests: dict[str, list[dict]] = {}
        self.batch_records: dict[str, dict] = {}
        self.polls = 0
        self.result_gets = 0
        self.chunk_path: Path | None = None
        self.durable_id_seen_before_poll = True

    def __call__(self, method, path, body=None):
        if method == "POST" and path == "/v1/messages/batches":
            self.posts.append(body["requests"])
            batch_id = f"batch-{len(self.posts)}"
            self.batch_requests[batch_id] = body["requests"]
            self.batch_records[batch_id] = {"id": batch_id, "processing_status": "in_progress",
                                            "created_at": gen64.utc_now()}
            return json.dumps({"id": batch_id, "processing_status": "in_progress"})
        if method == "GET" and path.startswith("/v1/messages/batches?"):
            return json.dumps({"data": list(self.batch_records.values()), "has_more": False})
        if method == "GET" and path.startswith("/v1/messages/batches/") and path.endswith("/results"):
            self.result_gets += 1
            batch_id = path.split("/")[-2]
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
            self.polls += 1
            batch_id = path.rsplit("/", 1)[-1]
            self.batch_records[batch_id]["processing_status"] = "ended"
            if self.chunk_path is not None:
                chunk = gen64.read_json(self.chunk_path)
                self.durable_id_seen_before_poll &= chunk.get("state") == "submitted" and bool(chunk.get("batch_id"))
            return json.dumps({"processing_status": "ended", "request_counts": {"succeeded": 1}})
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


class LocalPaidRegistryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=gen64.HERE)
        registry = Path(temporary.name) / "run-paths.json"
        registry_patch = patch.object(paid_ledger, "_RUN_PATH_REGISTRY", registry)
        lock_patch = patch.object(paid_ledger, "_RUN_PATH_LOCK", registry.with_suffix(".lock"))
        registry_patch.start()
        lock_patch.start()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(lock_patch.stop)
        self.addCleanup(registry_patch.stop)


class WriterRequestTests(LocalPaidRegistryTests):
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
                   count: int = 1) -> Path:
        entries = []
        for index in range(count):
            row = collection_row(f"L0:u{index + 1}")
            custom_id = f"u{index:04d}"
            request = gen64.request_for("haiku", "1x", row, custom_id)
            metadata = {"unit_key": row["unit_key"], "type": "neuron", "sample": "1x",
                        "writer": "haiku", "input_windows": gen64.windows_for_sample(row, "1x")}
            entries.append((request, metadata))
        record = gen64._chunk_record(run_id="run-1", group_key="test-chunk", ledger_key=key,
                                     chunk_index=0, writer="haiku", sample="1x", unit_type="neuron",
                                     entries=entries, probe=probe)
        path = root / "chunks" / "test-chunk.json"
        gen64.write_json(path, record)
        return path

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

    def test_opus_3x_ledgers_are_split_below_the_per_ledger_cap(self):
        rows = [collection_row(f"L0:u{index}") for index in range(250)]
        partitions = gen64._ledger_partitions("opus", "3x", rows)
        self.assertEqual(len(partitions), 6)
        self.assertEqual(sum(map(len, partitions)), 250)
        self.assertTrue(all(len(partition) <= 42 for partition in partitions))
        for partition in partitions:
            requests = [gen64.request_for("opus", "3x", row, f"u{index:04d}")
                        for index, row in enumerate(partition)]
            reserve = sum(gen64.request_reserve_usd([request], gen64.WRITERS["opus"])
                          for request in requests)
            self.assertLessEqual(reserve, HARD_CAP_USD)


if __name__ == "__main__":
    unittest.main()
