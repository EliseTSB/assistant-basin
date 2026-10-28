#!/usr/bin/env python3
"""Offline tests for download85.py: the rate cap, resume, hash refusal and the plan (no network)."""
from __future__ import annotations

import hashlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download85  # noqa: E402

PAYLOAD = bytes(range(256)) * 40  # 10,240 bytes


class FakeResponse(io.BytesIO):
    def __init__(self, data, status):
        super().__init__(data)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeServer:
    def __init__(self, data, honour_range=True):
        self.data, self.honour_range, self.requests = data, honour_range, []

    def __call__(self, req):
        rng = req.headers.get("Range")
        self.requests.append(rng)
        if rng and self.honour_range:
            start = int(rng.split("=")[1].rstrip("-"))
            return FakeResponse(self.data[start:], 206)
        return FakeResponse(self.data, 200)


class Clock:
    def __init__(self):
        self.now = 0.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def lfs_entry(data, name="model.bin"):
    return {"key": "sft", "repo_id": "org/model", "commit": "c" * 40,
            "files": [{"name": name, "algorithm": "sha256", "digest": hashlib.sha256(data).hexdigest(),
                       "size": len(data)}]}


class Download85Tests(unittest.TestCase):
    def test_rate_cap_never_exceeds_the_rate_over_any_prefix(self):
        clock = Clock()
        limiter = download85.RateLimiter(1000, clock, clock.sleep)
        for _ in range(10):
            limiter.take(500)
            self.assertLessEqual(limiter.sent, 1000 * clock.now + 1e-9)
        self.assertAlmostEqual(clock.now, 5.0)
        with self.assertRaises(ValueError):
            download85.RateLimiter(0)

    def test_rate_limiter_discards_credit_beyond_one_second_after_a_stall(self):
        clock = Clock()
        limiter = download85.RateLimiter(1000, clock, clock.sleep)
        limiter.take(1000)
        clock.now += 100.0
        limiter.take(1000)
        after_stall_chunk = clock.now
        limiter.take(1)
        self.assertGreater(clock.now, after_stall_chunk)

    def test_fetch_debits_each_chunk_before_a_stalled_read(self):
        clock = Clock()
        limiter = download85.RateLimiter(1000, clock, clock.sleep)
        debit_times = []
        original_take = limiter.take

        def tracked_take(count):
            debit_times.append(clock.now)
            original_take(count)

        limiter.take = tracked_take

        class StalledResponse(FakeResponse):
            def __init__(self, data):
                super().__init__(data, 200)
                self.stalled = False

            def read(self, size=-1):
                if not self.stalled:
                    self.stalled = True
                    clock.now += 100.0
                return super().read(size)

        with tempfile.TemporaryDirectory() as temp:
            data = b"x" * 2000
            response = StalledResponse(data)
            dest = Path(temp) / "model.bin"
            download85.fetch("https://example.invalid/model.bin", dest, len(data), limiter,
                             opener=lambda _req: response)
        self.assertEqual(len(debit_times), 2)
        self.assertNotEqual(debit_times[0], debit_times[1],
                            "the first read's long stall must not make its bytes debit at the next chunk's time")

    def test_download_places_a_verified_file_at_the_cache_snapshot_path_at_the_rate(self):
        with tempfile.TemporaryDirectory() as temp:
            clock = Clock()
            server = FakeServer(PAYLOAD)
            download85.download([lfs_entry(PAYLOAD)], Path(temp), rate=2048, opener=server,
                                clock=clock, sleep=clock.sleep, log=lambda *_: None)
            dest = Path(temp) / "models--org--model" / "snapshots" / ("c" * 40) / "model.bin"
            self.assertEqual(dest.read_bytes(), PAYLOAD)
            self.assertFalse(dest.with_name("model.bin.incomplete").exists())
            self.assertGreaterEqual(clock.now, len(PAYLOAD) / 2048 - 1e-9)
            # A second run re-hashes the file in place and fetches nothing.
            download85.download([lfs_entry(PAYLOAD)], Path(temp), rate=2048, opener=server,
                                clock=clock, sleep=clock.sleep, log=lambda *_: None)
            self.assertEqual(len(server.requests), 1)

    def test_resume_requests_the_missing_range_only(self):
        with tempfile.TemporaryDirectory() as temp:
            snap = Path(temp) / "models--org--model" / "snapshots" / ("c" * 40)
            snap.mkdir(parents=True)
            (snap / "model.bin.incomplete").write_bytes(PAYLOAD[:4000])
            server = FakeServer(PAYLOAD)
            download85.download([lfs_entry(PAYLOAD)], Path(temp), rate=10 ** 9, opener=server, log=lambda *_: None)
            self.assertEqual(server.requests, ["bytes=4000-"])
            self.assertEqual((snap / "model.bin").read_bytes(), PAYLOAD)

    def test_resume_refuses_a_server_that_ignores_the_range(self):
        with tempfile.TemporaryDirectory() as temp:
            snap = Path(temp) / "models--org--model" / "snapshots" / ("c" * 40)
            snap.mkdir(parents=True)
            (snap / "model.bin.incomplete").write_bytes(PAYLOAD[:4000])
            with self.assertRaisesRegex(RuntimeError, "ignored the resume Range"):
                download85.download([lfs_entry(PAYLOAD)], Path(temp), rate=10 ** 9,
                                    opener=FakeServer(PAYLOAD, honour_range=False), log=lambda *_: None)

    def test_hash_mismatch_refuses_and_keeps_the_bad_bytes_aside(self):
        with tempfile.TemporaryDirectory() as temp:
            tampered = bytearray(PAYLOAD)
            tampered[100] ^= 0xFF
            with self.assertRaises(download85.HashMismatch):
                download85.download([lfs_entry(PAYLOAD)], Path(temp), rate=10 ** 9,
                                    opener=FakeServer(bytes(tampered)), log=lambda *_: None)
            snap = Path(temp) / "models--org--model" / "snapshots" / ("c" * 40)
            self.assertFalse((snap / "model.bin").exists())
            self.assertEqual(len(list(snap.glob("model.bin.bad-*"))), 1)

    def test_a_file_in_place_with_wrong_bytes_is_refused_not_trusted(self):
        with tempfile.TemporaryDirectory() as temp:
            snap = Path(temp) / "models--org--model" / "snapshots" / ("c" * 40)
            snap.mkdir(parents=True)
            (snap / "model.bin").write_bytes(b"x" * len(PAYLOAD))
            with self.assertRaises(download85.HashMismatch):
                download85.download([lfs_entry(PAYLOAD)], Path(temp), rate=10 ** 9,
                                    opener=FakeServer(PAYLOAD), log=lambda *_: None)

    def test_git_blob_sha1_matches_git_for_small_files(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_bytes(b"hello\n")
            self.assertEqual(download85.git_blob_sha1(path), "ce013625030ba8dba906f756967f9e9ca394464a")

    def test_plan_skips_instruct_checks_the_commit_and_reads_lfs_and_blob_hashes(self):
        def sib(name, size, sha=None, blob=None):
            return types.SimpleNamespace(rfilename=name, size=size, blob_id=blob,
                                         lfs={"sha256": sha, "size": size} if sha else None)

        class Api:
            def model_info(self, repo, revision=None, files_metadata=False):
                cp = next(c for c in download85.CHECKPOINTS if c.repo_id == repo and
                          (revision == c.branch or revision.startswith(c.commit_prefix)))
                full = cp.commit_prefix + "0" * (40 - len(cp.commit_prefix))
                return types.SimpleNamespace(sha=full, siblings=[sib(".gitattributes", 10, blob="a" * 40),
                                                                 sib("config.json", 600, blob="b" * 40),
                                                                 sib("model.safetensors", 5_000, sha="d" * 64)])

        entries = download85.plan(api=Api())
        self.assertEqual(len(entries), 11)
        self.assertNotIn("instruct", [e["key"] for e in entries])
        first = entries[0]
        self.assertEqual([f["name"] for f in first["files"]], ["config.json", "model.safetensors"])
        self.assertEqual(first["files"][0]["algorithm"], "git-sha1")
        self.assertEqual(first["files"][1], {"name": "model.safetensors", "algorithm": "sha256",
                                             "digest": "d" * 64, "size": 5_000})

    def test_local_commit_pins_are_written_prefix_checked_and_read_offline(self):
        import checkpoints
        from unittest import mock

        class Api:
            def model_info(self, repo, revision=None, files_metadata=False):
                cp = next(c for c in download85.CHECKPOINTS if c.repo_id == repo and c.branch == revision)
                return types.SimpleNamespace(sha=cp.commit_prefix + "0" * (40 - len(cp.commit_prefix)))

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "commits.local.json"
            commits = download85.write_local_commits(api=Api(), path=path)
            self.assertEqual(len(commits), 12)
            with mock.patch.object(checkpoints, "LOCAL_COMMITS", path):
                self.assertEqual(checkpoints.resolve_checkpoint("sft")["commit"], commits["sft"])
                path.write_text('{"sft": "' + "f" * 40 + '"}')
                with self.assertRaisesRegex(RuntimeError, "no full commit with prefix"):
                    checkpoints.resolve_checkpoint("sft")


if __name__ == "__main__":
    unittest.main(verbosity=1)
