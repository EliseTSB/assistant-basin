"""Offline contract tests for the shared paid-action library."""

from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import uuid
import unittest
import warnings

from paid_ledger import (CapExceededError, Ledger, LedgerError, DEFAULT_LEDGER_PATH,
                         LedgerCorruptionError, ReleaseRefusedError,
                         listing_report, submit_in_chunks, chunks)
from pricing import worst_case_cost


ROOT = Path(__file__).resolve().parent


def _reserve_racer(path: str, ready: object, start: object, result: object) -> None:
    ready.put(True)
    start.wait()
    try:
        attempt = Ledger(path, 1.0).reserve(0.75, "race", {})
        result.put(("ok", attempt))
    except CapExceededError:
        result.put(("cap", None))


def _subcap_racer(path: str, ready: object, start: object, result: object) -> None:
    ready.put(True)
    start.wait()
    try:
        attempt = Ledger(path, 2.0).reserve(0.6, "race", {},
                                             subcap_usd=0.6, subcap_key="reader-stage")
        result.put(("ok", attempt))
    except CapExceededError:
        result.put(("cap", None))


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory(prefix=".paid-ledger-test-", dir=ROOT)
        self.addCleanup(temp.cleanup)
        self.temp_root = Path(temp.name)
        self.path = self.temp_root / "paid.jsonl"
        self.ledger = Ledger(self.path, 1.0)

    def test_absolute_path_registry_and_cap_validation_fail_closed(self) -> None:
        with self.assertRaises(LedgerError):
            Ledger(Path("relative-ledger.jsonl"), 1.0)
        with self.assertRaises(LedgerError):
            Ledger(Path("/tmp/paid-ledger-test.jsonl"), 1.0)
        invalid = (float("nan"), float("inf"), float("-inf"), 0.0, -0.01,
                   "1.0", True, 8.01)
        for cap in invalid:
            with self.subTest(cap=repr(cap)), self.assertRaises(LedgerError):
                Ledger(self.temp_root / "bad.jsonl", cap)
        registry = self.temp_root / "paid-ledger-registry.json"
        registry.write_text(json.dumps({
            "schema": "paid-ledger-registry-v1",
            "known_ledger_files": ["paid.jsonl", "old-run-ledger.jsonl"],
        }), encoding="utf-8")
        (self.temp_root / "old-run-ledger.jsonl").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(LedgerError, "alternate ledger"):
            Ledger(self.path, 1.0)

    def test_reserve_is_durable_and_attempt_id_is_random_128_bit(self) -> None:
        attempt_id = self.ledger.reserve(0.25, "haiku", {"chunk": 1})
        self.assertRegex(attempt_id, re.compile(r"^[0-9a-f]{32}$"))
        event = json.loads(self.path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(event["event"], "reserve")
        self.assertEqual(event["attempt_id"], attempt_id)
        self.assertEqual(event["reserve_usd"], 0.25)
        self.assertEqual(self.ledger.state()["reserved"], 0.25)

    def test_settle_is_idempotent_and_rejects_different_replay_values(self) -> None:
        attempt_id = self.ledger.reserve(0.4, "batch", {"phase": "primary"})
        usage = {"input_tokens": 10, "output_tokens": 4}
        self.ledger.settle(attempt_id, 0.12, usage)
        before = self.path.read_bytes()
        self.ledger.settle(attempt_id, 0.12, usage)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertAlmostEqual(self.ledger.state()["settled"], 0.12)
        with self.assertRaisesRegex(LedgerError, "different values"):
            self.ledger.settle(attempt_id, 0.13, usage)
        with self.assertRaisesRegex(LedgerError, "different values"):
            self.ledger.settle(attempt_id, 0.12, {"input_tokens": 11, "output_tokens": 4})

    def test_settlement_overrun_refuses_records_actual_and_keeps_reserve(self) -> None:
        other = self.ledger.reserve(0.4, "earlier", {})
        target = self.ledger.reserve(0.5, "target", {})
        with self.assertRaisesRegex(CapExceededError, "overrun was recorded"):
            self.ledger.settle(target, 0.7, {"actual": 7})
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        overrun = rows[-1]
        self.assertEqual(overrun["event"], "settlement_overrun")
        self.assertEqual(overrun["attempt_id"], target)
        self.assertEqual(overrun["actual_usd"], 0.7)
        self.assertAlmostEqual(self.ledger.state()["reserved"], 0.9)
        self.assertEqual(self.ledger.state()["settled"], 0.0)
        self.assertEqual(other, rows[0]["attempt_id"])

    def test_cap_counts_settled_and_reserved_dollars(self) -> None:
        settled = self.ledger.reserve(0.6, "direct", {})
        self.ledger.settle(settled, 0.55, {"calls": 1})
        outstanding = self.ledger.reserve(0.4, "batch", {})
        with self.assertRaises(CapExceededError):
            self.ledger.reserve(0.051, "batch", {})
        state = self.ledger.state()
        self.assertAlmostEqual(state["settled"], 0.55)
        self.assertAlmostEqual(state["reserved"], 0.4)
        self.assertAlmostEqual(state["per_kind"]["direct"]["settled"], 0.55)
        self.assertAlmostEqual(state["per_kind"]["batch"]["reserved"], 0.4)
        self.assertEqual(len(outstanding), 32)

    def test_release_requires_anthropic_4xx_and_never_releases_429(self) -> None:
        attempt_id = self.ledger.reserve(0.5, "batch", {})
        with self.assertRaises(ReleaseRefusedError):
            self.ledger.release(attempt_id, 403, "<html>denied</html>")
        self.assertAlmostEqual(self.ledger.state()["reserved"], 0.5)
        with self.assertRaises(ReleaseRefusedError):
            self.ledger.release(attempt_id, 429,
                                {"type": "error", "error": {"type": "rate_limit_error"}})
        self.assertAlmostEqual(self.ledger.state()["reserved"], 0.5)
        body = {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}
        self.ledger.release(attempt_id, 403, body)
        before = self.path.read_bytes()
        self.ledger.release(attempt_id, 403, body)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.ledger.state()["reserved"], 0.0)

    def test_timeout_keeps_reservation_counted(self) -> None:
        def timed_out(chunk, attempt_id, record_batch_id):
            raise TimeoutError("simulated lost connection")

        request = {"custom_id": "one", "params": {"max_tokens": 12}}
        with self.assertRaises(TimeoutError):
            submit_in_chunks([request], 1, ledger=self.ledger,
                             model="claude-haiku-5-5", submit=timed_out,
                             batch_id_path=self.temp_root / "batch-ids.jsonl")
        self.assertGreater(self.ledger.state()["reserved"], 0.0)
        events = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["event"] for row in events], ["reserve"])

    def test_retire_is_dated_and_removes_only_that_reservation(self) -> None:
        first = self.ledger.reserve(0.2, "batch", {})
        second = self.ledger.reserve(0.3, "direct", {})
        self.ledger.retire(first, "operator-7", "provider console reconciled")
        self.assertAlmostEqual(self.ledger.state()["reserved"], 0.3)
        event = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()][-1]
        self.assertEqual(event["event"], "retire")
        self.assertEqual(event["attempt_id"], first)
        self.assertEqual(event["who"], "operator-7")
        self.assertTrue(event["ts"])
        with self.assertRaisesRegex(LedgerError, "not an active reservation"):
            self.ledger.settle(first, 0.0, {})
        self.assertEqual(self.ledger.state()["per_kind"]["direct"]["reserved"], 0.3)
        self.assertEqual(self.ledger.state()["per_kind"]["batch"]["reserved"], 0.0)
        self.assertEqual(len(second), 32)

    def test_torn_last_line_is_ignored_reported_and_repaired_before_append(self) -> None:
        self.ledger.reserve(0.2, "batch", {})
        with self.path.open("ab") as stream:
            stream.write(b'{"event":"reserve","attempt_id":"partial')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            state = self.ledger.state()
        self.assertTrue(state["torn_last_line_ignored"])
        self.assertTrue(any("torn final" in str(item.message) for item in caught))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            self.ledger.reserve(0.1, "batch", {})
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(self.ledger.state()["reserved"], 0.3)

    def test_two_processes_racing_one_reserve_cannot_bypass_cap(self) -> None:
        context = multiprocessing.get_context("fork")
        ready, start, results = context.Queue(), context.Event(), context.Queue()
        workers = [context.Process(target=_reserve_racer,
                                   args=(str(self.path), ready, start, results))
                   for _ in range(2)]
        for worker in workers:
            worker.start()
        ready.get(timeout=5)
        ready.get(timeout=5)
        start.set()
        for worker in workers:
            worker.join(timeout=10)
            self.assertFalse(worker.is_alive())
            self.assertEqual(worker.exitcode, 0)
        outcomes = [results.get(timeout=5)[0] for _ in range(2)]
        self.assertCountEqual(outcomes, ["ok", "cap"])
        self.assertAlmostEqual(self.ledger.state()["reserved"], 0.75)

    def test_two_processes_racing_one_subcap_cannot_both_reserve(self) -> None:
        context = multiprocessing.get_context("fork")
        ready, start, results = context.Queue(), context.Event(), context.Queue()
        workers = [context.Process(target=_subcap_racer,
                                   args=(str(self.path), ready, start, results))
                   for _ in range(2)]
        for worker in workers:
            worker.start()
        ready.get(timeout=5)
        ready.get(timeout=5)
        start.set()
        for worker in workers:
            worker.join(timeout=10)
            self.assertFalse(worker.is_alive())
            self.assertEqual(worker.exitcode, 0)
        outcomes = [results.get(timeout=5)[0] for _ in range(2)]
        self.assertCountEqual(outcomes, ["ok", "cap"])
        self.assertAlmostEqual(self.ledger.state()["reserved"], 0.6)

    def test_default_path_is_checkout_independent_and_run_id_cannot_split(self) -> None:
        self.assertEqual(DEFAULT_LEDGER_PATH,
                         Path("/home/elisepc/assistant-basin/paidaction/paid-ledger.jsonl"))
        run_id = "test-" + uuid.uuid4().hex
        self.ledger.reserve(0.1, "first", {"run_id": run_id})
        with self.assertRaisesRegex(LedgerError, "already uses ledger"):
            Ledger(self.temp_root / "other.jsonl", 1.0).reserve(
                0.1, "second", {"run_id": run_id})

    def test_kill_matrix_reruns_with_fresh_attempts_and_durable_state(self) -> None:
        script = r'''import json, os, signal, sys
from paid_ledger import Ledger, submit_in_chunks

ledger_path, ids_path, point = sys.argv[1:4]
ledger = Ledger(ledger_path, 1.0)
def kill_if(stage):
    if point == stage:
        os.kill(os.getpid(), signal.SIGKILL)
def fake_submit(chunk, attempt_id, record_batch_id):
    kill_if("after_reserve")
    raw_post_body = json.dumps({"id": "batch-" + attempt_id})
    kill_if("after_post")
    batch_id = json.loads(raw_post_body)["id"]
    record_batch_id(batch_id)
    kill_if("after_batch_id_write")
    return {"actual_usd": 0.001, "usage": {"input_tokens": 5, "output_tokens": 2}}
if point == "after_settle":
    original_settle = Ledger.settle
    def settle_then_kill(self, *args, **kwargs):
        original_settle(self, *args, **kwargs)
        kill_if("after_settle")
    Ledger.settle = settle_then_kill
requests = [{"custom_id": "one", "params": {"max_tokens": 4}}]
submit_in_chunks(requests, 1, ledger=ledger, model="claude-haiku-5-5",
                 submit=fake_submit, batch_id_path=ids_path)
'''
        for point in ("after_reserve", "after_post", "after_batch_id_write", "after_settle"):
            with self.subTest(point=point):
                case_root = self.temp_root / point
                case_root.mkdir()
                ledger_path = case_root / "paid.jsonl"
                ids_path = case_root / "batch-ids.jsonl"
                env = dict(os.environ)
                env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
                killed = subprocess.run([sys.executable, "-c", script,
                                         str(ledger_path), str(ids_path), point],
                                        cwd=ROOT, env=env, capture_output=True, text=True)
                self.assertEqual(killed.returncode, -signal.SIGKILL, killed.stderr)
                interrupted = Ledger(ledger_path, 1.0)
                before = interrupted.state()
                self.assertEqual(before["settled"], 0.001 if point == "after_settle" else 0.0)
                self.assertGreater(before["reserved"], 0.0) if point != "after_settle" else self.assertEqual(before["reserved"], 0.0)
                rerun = script.replace("point = sys.argv[1:4]", "point = sys.argv[1:4]")
                rerun = rerun.replace("if point == \"after_settle\":", "if point == \"never\":")
                completed = subprocess.run([sys.executable, "-c", rerun,
                                            str(ledger_path), str(ids_path), "rerun"],
                                           cwd=ROOT, env=env, capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                final = interrupted.state()
                self.assertEqual(final["settled"], 0.002 if point == "after_settle" else 0.001)
                self.assertEqual(final["reserved"], before["reserved"])
                ledger_events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
                reserves = [row["attempt_id"] for row in ledger_events if row["event"] == "reserve"]
                self.assertEqual(len(reserves), 2)
                self.assertEqual(len(set(reserves)), 2)
                batch_rows = ([json.loads(line) for line in ids_path.read_text(encoding="utf-8").splitlines()]
                              if ids_path.exists() else [])
                self.assertEqual(len(batch_rows), 2 if point in ("after_batch_id_write", "after_settle") else 1)

    def test_worst_case_chunking_settles_then_refuses_next_chunk_at_cap(self) -> None:
        ledger = Ledger(self.temp_root / "chunks.jsonl", 0.015)
        requests = [{"custom_id": str(index), "params": {"max_tokens": 1024}}
                    for index in range(3)]
        submitted = []

        def fake_submit(chunk, attempt_id, record_batch_id):
            submitted.append((list(chunk), attempt_id))
            record_batch_id("batch-" + attempt_id)
            return {"actual_usd": 0.0102, "usage": {"input_tokens": 100, "output_tokens": 1024}}

        with self.assertRaises(CapExceededError):
            submit_in_chunks(requests, 1, ledger=ledger, model="claude-opus-5-5",
                             submit=fake_submit,
                             batch_id_path=self.temp_root / "chunk-ids.jsonl")
        self.assertEqual(len(submitted), 1)
        self.assertEqual(len(submitted[0][0]), 1)
        self.assertAlmostEqual(ledger.state()["settled"], 0.0102)
        self.assertEqual(ledger.state()["reserved"], 0.0)

    def test_listing_report_is_read_only_and_never_adopts_batches(self) -> None:
        self.ledger.reserve(0.1, "batch", {})
        before = self.path.read_bytes()
        calls = []
        original = [{"id": "batch-old", "created_at": "2000-01-01"}]
        report = listing_report(lambda: calls.append("list") or {"data": original})
        self.assertEqual(report, {"count": 1, "batches": original})
        self.assertEqual(calls, ["list"])
        self.assertEqual(self.path.read_bytes(), before)
        original[0]["id"] = "caller-mutated"
        self.assertEqual(report["batches"][0]["id"], "batch-old")

    def test_production_copy_mutations_go_red(self) -> None:
        source = (ROOT / "paid_ledger.py").read_text(encoding="utf-8")
        test_source = (ROOT / "test_paid_ledger.py").read_text(encoding="utf-8")
        pricing_source = (ROOT / "pricing.py").read_text(encoding="utf-8")
        mutants = [
            ("if not math.isfinite(cap_float) or cap_float <= 0 or cap_float > HARD_CAP_USD:",
             "if cap_float <= 0 or cap_float > HARD_CAP_USD:",
             "test_absolute_path_registry_and_cap_validation_fail_closed"),
            ("valid = (isinstance(parsed, dict)\n             and parsed.get(\"type\") == \"error\"\n             and isinstance(parsed.get(\"error\"), dict)\n             and bool(parsed[\"error\"]))",
             "valid = True",
             "test_release_requires_anthropic_4xx_and_never_releases_429"),
            ("if committed_elsewhere + actual > self._cap_decimal or subcap_over:",
             "if False:", "test_settlement_overrun_refuses_records_actual_and_keeps_reserve"),
            ("if scoped + amount > subcap:", "if False:",
             "test_two_processes_racing_one_subcap_cannot_both_reserve"),
            ('Path("/home/elisepc/assistant-basin/paidaction/paid-ledger.jsonl")',
             "Path(__file__).resolve().with_name(\"paid-ledger.jsonl\")",
             "test_default_path_is_checkout_independent_and_run_id_cannot_split"),
        ]
        for original, mutation, test_name in mutants:
            with self.subTest(test_name=test_name):
                self.assertIn(original, source)
                temp = tempfile.TemporaryDirectory(prefix=".paid-ledger-mutant-", dir=ROOT)
                self.addCleanup(temp.cleanup)
                mutant_dir = Path(temp.name)
                (mutant_dir / "paid_ledger.py").write_text(source.replace(original, mutation, 1), encoding="utf-8")
                (mutant_dir / "pricing.py").write_text(pricing_source, encoding="utf-8")
                (mutant_dir / "test_paid_ledger.py").write_text(test_source, encoding="utf-8")
                run = subprocess.run([sys.executable, "-m", "unittest",
                                      f"test_paid_ledger.LedgerTests.{test_name}"],
                                     cwd=mutant_dir, capture_output=True, text=True)
                self.assertNotEqual(run.returncode, 0,
                                    f"mutated production copy unexpectedly passed:\n{run.stdout}\n{run.stderr}")
                self.assertIn("FAILED (failures=1)", run.stderr,
                              f"mutation did not go red through a failed contract assertion:\n{run.stdout}\n{run.stderr}")


class PricingAndHelpersTests(unittest.TestCase):
    def test_cost_model_uses_inherited_batch_and_direct_rates(self) -> None:
        self.assertAlmostEqual(worst_case_cost("claude-haiku-5-5", 100, 200),
                               (100 * 0.05 + 200 * 0.25) / 1_000_000)
        self.assertAlmostEqual(worst_case_cost("claude-haiku-5-5", 100, 200, batch=False),
                               (100 * 0.10 + 200 * 0.50) / 1_000_000)
        self.assertAlmostEqual(worst_case_cost("claude-opus-5-5", 100, 200),
                               (100 * 2.0 + 200 * 10.0) / 1_000_000)
        with self.assertRaises(ValueError):
            worst_case_cost("claude-opus-5-5", 100, 200, batch=False)
        with self.assertRaises(ValueError):
            worst_case_cost("unknown-model", 100, 200)

    def test_chunks_preserve_sequential_order(self) -> None:
        requests = list(range(5))
        self.assertEqual(list(chunks(requests, 2)), [[0, 1], [2, 3], [4]])
        with self.assertRaises(ValueError):
            list(chunks(requests, 0))


if __name__ == "__main__":
    unittest.main()
