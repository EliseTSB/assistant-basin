"""Offline tests for Part 64 M1 requests and durable paid chunk recovery."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from part64 import gen64
from paidaction.paid_ledger import HARD_CAP_USD, Ledger


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


class FakeBatchAPI:
    def __init__(self, *, text: str | None = None, stop_reason: str = "end_turn"):
        self.text = answer_text() if text is None else text
        self.stop_reason = stop_reason
        self.posts: list[list[dict]] = []
        self.polls = 0
        self.result_gets = 0
        self.chunk_path: Path | None = None
        self.durable_id_seen_before_poll = True

    def __call__(self, method, path, body=None):
        if method == "POST" and path == "/v1/messages/batches":
            self.posts.append(body["requests"])
            return json.dumps({"id": f"batch-{len(self.posts)}", "processing_status": "in_progress"})
        if method == "GET" and path.startswith("/v1/messages/batches/") and path.endswith("/results"):
            self.result_gets += 1
            requests = self.posts[-1]
            rows = []
            for request in requests:
                rows.append({"custom_id": request["custom_id"], "result": {
                    "type": "succeeded", "message": {
                        "content": [{"type": "text", "text": self.text}],
                        "stop_reason": self.stop_reason,
                        "usage": {"input_tokens": 321, "output_tokens": 1234},
                    }}})
            return "\n".join(json.dumps(row) for row in rows) + "\n"
        if method == "GET" and path.startswith("/v1/messages/batches/"):
            self.polls += 1
            if self.chunk_path is not None:
                chunk = gen64.read_json(self.chunk_path)
                self.durable_id_seen_before_poll &= chunk.get("state") == "submitted" and bool(chunk.get("batch_id"))
            return json.dumps({"processing_status": "ended", "request_counts": {"succeeded": 1}})
        raise AssertionError(f"unexpected fake API call: {method} {path}")


class WriterRequestTests(unittest.TestCase):
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

    def test_collection_manifest_gates_are_fail_closed(self):
        manifest = {
            "version": 1, "unit_count": 500, "subset_verified": True,
            "capacities": gen64.SAMPLE_COUNTS["3x"], "pilot_counts": gen64.SAMPLE_COUNTS["1x"],
            "model": "Qwen/Qwen3-1.7B", "model_revision": gen64.REGISTERED_MODEL_REVISION,
            "corpus": {"path": str(gen64.REGISTERED_CORPUS),
                       "size_bytes": gen64.REGISTERED_CORPUS_SIZE,
                       "sha256": gen64.REGISTERED_CORPUS_SHA256_PREFIX + "0" * 48},
            "median_fresh_top12_match": 10,
        }
        self.assertIsNone(gen64.validate_collection_manifest(manifest))
        broken = dict(manifest, median_fresh_top12_match=9)
        with self.assertRaisesRegex(gen64.WriterError, "collection manifest"):
            gen64.validate_collection_manifest(broken)

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
        with tempfile.TemporaryDirectory(dir=gen64.BASIN) as temporary:
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
            with self.subTest(stop_reason=stop_reason, text=text[:10]), tempfile.TemporaryDirectory(dir=gen64.BASIN) as temporary:
                root = Path(temporary)
                manifest = gen64.initialize_plan(root, [collection_row()], "b" * 64)
                api = FakeBatchAPI(text=text, stop_reason=stop_reason)
                with self.assertRaisesRegex(gen64.WriterError, "probes failed"):
                    gen64.execute_probes(root, manifest, api, poll_interval=0, sleep=lambda _n: None)
                self.assertEqual(len(api.posts), 2)


class ChunkRecoveryTests(unittest.TestCase):
    def make_chunk(self, root: Path, key: str = "haiku-1x-neuron", *, probe: bool = False) -> Path:
        row = collection_row()
        request = gen64.request_for("haiku", "1x", row, "u0000")
        metadata = {"unit_key": row["unit_key"], "type": "neuron", "sample": "1x",
                    "writer": "haiku", "input_windows": gen64.windows_for_sample(row, "1x")}
        record = gen64._chunk_record(run_id="run-1", group_key="test-chunk", ledger_key=key,
                                     chunk_index=0, writer="haiku", sample="1x", unit_type="neuron",
                                     entries=[(request, metadata)], probe=probe)
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
        with tempfile.TemporaryDirectory(dir=gen64.BASIN) as temporary:
            root = Path(temporary)
            path = self.make_chunk(root)
            api = FakeBatchAPI()
            with self.assertRaisesRegex(RuntimeError, "after_reserve"):
                gen64.execute_chunk(root, path, api, kill_at="after_reserve")
            self.assertEqual(gen64.read_json(path)["state"], "planned")
            self.assertEqual(api.posts, [])
            self.execute_and_check_settled(root, path, api)

    def test_kill_after_post_before_id_never_resubmits_ambiguous_request(self):
        with tempfile.TemporaryDirectory(dir=gen64.BASIN) as temporary:
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

    def test_kill_after_id_recovers_batch_without_resubmit(self):
        with tempfile.TemporaryDirectory(dir=gen64.BASIN) as temporary:
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
        with tempfile.TemporaryDirectory(dir=gen64.BASIN) as temporary:
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
        with tempfile.TemporaryDirectory(dir=gen64.BASIN) as temporary:
            root = Path(temporary)
            path = self.make_chunk(root)
            api = FakeBatchAPI()
            with self.assertRaisesRegex(RuntimeError, "after_post_before_id"):
                gen64.execute_chunk(root, path, api, kill_at="after_post_before_id")
            chunk = gen64.read_json(path)
            # Record this single chunk in a tiny run manifest so the CLI retirement path can find it.
            gen64.write_json(root / "run.json", {"schema": "test", "groups": {
                "test-chunk": {"chunk_paths": [str(path.relative_to(root))]}}, "probes": {}})
            gen64.retire_attempt(root, chunk["attempt_id"], "operator", "provider listing reviewed")
            self.assertEqual(gen64.read_json(path)["state"], "retired")
            result = gen64.execute_chunk(root, path, api, poll_interval=0, sleep=lambda _n: None)
            self.assertEqual(result["state"], "settled")
            self.assertEqual(result["attempt_no"], 1)
            self.assertEqual(len(api.posts), 2)
            ledger = Ledger(str(gen64.ledger_path(root, "haiku-1x-neuron")), HARD_CAP_USD)
            self.assertEqual(sorted(row["status"] for row in gen64._attempts(ledger).values()),
                             ["retired", "settled"])


if __name__ == "__main__":
    unittest.main()
