#!/usr/bin/env python3
"""Durable Part 64 M1 writer generation for Haiku 5.5 and Opus 5.5.

Paid actions are opt-in CLI commands. Each writer/sample/unit-type combination
has its own $8 paidaction ledger and sequential, resumable batch chunks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import ssl
import statistics
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable, Mapping

HERE = Path(__file__).resolve().parent
BASIN = HERE.parent.resolve()
if str(BASIN) not in sys.path:
    sys.path.insert(0, str(BASIN))

import part60.gen60 as gen60
from part63 import reader63
from paidaction.paid_ledger import (HARD_CAP_USD, HTTPFailure, Ledger,
                                    LedgerError)
from paidaction.pricing import BATCH_RATES_USD_PER_MILLION, worst_case_cost

DEFAULT_COLLECTION = BASIN / "results" / "part64" / "collection.jsonl"
DEFAULT_ROOT = BASIN / "results" / "part64" / "gen64"
REGISTERED_CORPUS = Path("/mnt/c/Users/alexa/neuron-probe-gpu/qwen35/pile10k.jsonl")
REGISTERED_CORPUS_SIZE = 63_385_283
REGISTERED_CORPUS_SHA256_PREFIX = "56ad98d5a82b2bb4"
REGISTERED_MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
WRITERS = {"haiku": "claude-haiku-5-5", "opus": "claude-opus-5-5"}
SAMPLES = ("1x", "3x")
UNIT_TYPES = ("neuron", "control")
SAMPLE_COUNTS = {
    "1x": {"top": 12, "random": 6, "weak": 3},
    "3x": {"top": 36, "random": 18, "weak": 9},
}
EXPECTED_UNITS = {"neuron": 250, "control": 250}
MAX_TOKENS = 8192  # Includes adaptive thinking and the complete M1 answer.
CHUNK_RESERVE_TARGET_USD = 1.0
OPUS_3X_LEDGER_PARTITIONS = 6
ANTHROPIC_VERSION = "2023-06-01"
HTTP_TIMEOUT_SECONDS = 120
POLL_INTERVAL_SECONDS = 5
POLL_BOUND_SECONDS = 60 * 60
ACCEPTED_STOP_REASONS = ("end_turn", "stop_sequence")
M1_PROMPT = (gen60.HERE / "prompts" / "M1.txt").read_text(encoding="utf-8").strip()


class WriterError(RuntimeError):
    """A local Part 64 writer validation or recovery error."""


class RecoveryRequired(WriterError):
    """An unknown paid outcome needs recovery or explicit operator retirement."""


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WriterError(f"cannot read {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    """Atomically write and fsync one JSON record and its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


# Reuse Part 63's established cross-process run lock for the shared run root.
run_lock = reader63.run_lock


def load_collection(path: Path = DEFAULT_COLLECTION) -> tuple[list[dict[str, Any]], str]:
    if not path.is_file():
        raise WriterError(f"Part 64 collection is missing: {path}")
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise WriterError(f"Part 64 collection manifest is missing: {manifest_path}")
    manifest = read_json(manifest_path)
    validate_collection_manifest(manifest)
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if manifest.get("collection_sha256") != digest:
        raise WriterError("collection JSONL SHA-256 differs from its manifest")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WriterError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise WriterError(f"{path}:{line_no}: expected a JSON object")
        key, unit_type = row.get("unit_key"), row.get("type")
        if not isinstance(key, str) or not key or unit_type not in UNIT_TYPES:
            raise WriterError(f"{path}:{line_no}: missing unit_key or invalid unit type")
        strata = row.get("strata")
        if not isinstance(strata, dict):
            raise WriterError(f"{path}:{line_no}: strata object required")
        for category, expected in SAMPLE_COUNTS["3x"].items():
            values = strata.get(category)
            if (not isinstance(values, list) or len(values) != expected
                    or any(not isinstance(item, dict) or not isinstance(item.get("window"), str)
                           or not item["window"] for item in values)):
                raise WriterError(f"{path}:{line_no}: expected {expected} nonempty {category} windows")
            windows = [item["window"] for item in values]
            pilot_count = SAMPLE_COUNTS["1x"][category]
            # The canonical Part 60 pilot prefix may contain repeated text (and
            # must remain byte-for-byte intact); fresh additions must be unique
            # both among themselves and against that prefix.
            fresh = windows[pilot_count:]
            if len(set(fresh)) != len(fresh) or set(fresh) & set(windows[:pilot_count]):
                raise WriterError(f"{path}:{line_no}: duplicate fresh {category} windows")
        rows.append(row)
    if len(rows) != sum(EXPECTED_UNITS.values()):
        raise WriterError(f"collection has {len(rows)} units; expected 500")
    keys = [row["unit_key"] for row in rows]
    if len(set(keys)) != len(keys):
        raise WriterError("collection contains duplicate unit_key values")
    counts = {kind: sum(row["type"] == kind for row in rows) for kind in UNIT_TYPES}
    if counts != EXPECTED_UNITS:
        raise WriterError(f"collection unit counts are {counts}; expected {EXPECTED_UNITS}")
    reproduction = []
    for line_no, row in enumerate(rows, 1):
        value = row.get("fresh_top12_match")
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 12:
            raise WriterError(f"{path}: row {line_no} has invalid fresh_top12_match")
        reproduction.append(value)
    median = statistics.median(reproduction)
    if manifest.get("median_fresh_top12_match") != median or median < 10:
        raise WriterError("collection reproduction median differs from its rows or is below 10/12")
    verify_pilot_subset(rows)
    return rows, digest


def validate_collection_manifest(value: Any) -> None:
    """Recheck collect64's registered corpus, stratum, and reproduction gates."""
    if not isinstance(value, dict):
        raise WriterError("collection manifest must be a JSON object")
    corpus = value.get("corpus")
    digest = corpus.get("sha256") if isinstance(corpus, dict) else None
    median = value.get("median_fresh_top12_match")
    if (value.get("version") != 1 or value.get("unit_count") != 500
            or value.get("subset_verified") is not True
            or value.get("capacities") != SAMPLE_COUNTS["3x"]
            or value.get("pilot_counts") != SAMPLE_COUNTS["1x"]
            or value.get("model") != "Qwen/Qwen3-1.7B"
            or value.get("model_revision") != REGISTERED_MODEL_REVISION
            or not isinstance(corpus, dict)
            or corpus.get("path") != str(REGISTERED_CORPUS)
            or corpus.get("size_bytes") != REGISTERED_CORPUS_SIZE
            or not isinstance(digest, str) or not digest.startswith(REGISTERED_CORPUS_SHA256_PREFIX)
            or isinstance(median, bool) or not isinstance(median, (int, float)) or median < 10
            or not isinstance(value.get("collection_sha256"), str)
            or len(value["collection_sha256"]) != 64
            or any(ch not in "0123456789abcdef" for ch in value["collection_sha256"])):
        raise WriterError("collection manifest does not satisfy the registered Part 64 collection gates")


def verify_pilot_subset(rows: list[dict[str, Any]]) -> None:
    """Compare every 1x prefix with the canonical Part 60 M1 pilot windows."""
    try:
        pilot_units = gen60.load_splits()["dev"]
    except (OSError, ValueError, gen60.RunnerError) as exc:
        raise WriterError(f"cannot load the canonical Part 60 dev split: {exc}") from exc
    pilot_by_key = {unit["unit_key"]: unit for unit in pilot_units}
    if len(pilot_by_key) != 500 or set(pilot_by_key) != {row["unit_key"] for row in rows}:
        raise WriterError("collection unit keys do not match the canonical Part 60 dev split")
    for row in rows:
        pilot_unit = pilot_by_key[row["unit_key"]]
        pilot = pilot_unit["record"]
        if row.get("type") != pilot_unit.get("type") or row.get("layer") != pilot_unit.get("layer"):
            raise WriterError(f"{row['unit_key']} type/layer differs from the canonical Part 60 dev split")
        identity_field = "unit" if pilot_unit["type"] == "neuron" else "control_id"
        if row.get(identity_field) != pilot.get(identity_field):
            raise WriterError(f"{row['unit_key']} identity differs from the canonical Part 60 dev split")
        strata = row["strata"]
        for category, count in SAMPLE_COUNTS["1x"].items():
            expected = [item["window"] for item in pilot.get(category, [])[:count]]
            actual = [item["window"] for item in strata[category][:count]]
            expanded = [item["window"] for item in strata[category]]
            if len(expected) != count or actual != expected:
                raise WriterError(f"{row['unit_key']} 1x {category} windows differ from Part 60 M1")
            if not set(expected).issubset(set(expanded)):
                raise WriterError(f"{row['unit_key']} 1x {category} windows are not a subset of 3x")


def windows_for_sample(row: Mapping[str, Any], sample: str) -> list[str]:
    if sample not in SAMPLE_COUNTS:
        raise WriterError(f"unsupported sample {sample!r}")
    strata = row.get("strata")
    if not isinstance(strata, Mapping):
        raise WriterError("collection unit needs strata")
    windows = []
    for category, count in SAMPLE_COUNTS[sample].items():
        values = strata.get(category)
        if not isinstance(values, list) or len(values) < count:
            raise WriterError(f"{row.get('unit_key')} lacks {count} {category} windows")
        for item in values[:count]:
            value = item.get("window") if isinstance(item, dict) else None
            if not isinstance(value, str) or not value:
                raise WriterError(f"{row.get('unit_key')} has an invalid {category} window")
            windows.append(value)
    # Match Part 60's stable M1 prompt ordering, using the same seed in both arms.
    unit = {"unit_key": row.get("unit_key")}
    random.Random(gen60._shuffle_seed(unit, "M1", "primary")).shuffle(windows)
    return windows


def request_for(writer: str, sample: str, row: Mapping[str, Any], custom_id: str) -> dict[str, Any]:
    """Build one M1 request; model thinking and all other API defaults stay enabled."""
    if writer not in WRITERS:
        raise WriterError(f"unknown writer {writer!r}")
    windows = windows_for_sample(row, sample)
    params = {
        "model": WRITERS[writer],
        "max_tokens": MAX_TOKENS,
        "system": [{"type": "text", "text": M1_PROMPT}],
        "messages": [{"role": "user", "content": gen60.marked_user(windows)}],
    }
    return {"custom_id": custom_id, "params": params}


def request_reserve_usd(requests: list[Mapping[str, Any]], model: str) -> float:
    input_bytes = sum(len(canonical_json(dict(request))) for request in requests)
    input_tokens_upper = math.ceil(input_bytes / 3)
    output_tokens = sum(int(request["params"]["max_tokens"]) for request in requests)
    return worst_case_cost(model, input_tokens_upper, output_tokens, batch=True)


def _chunk_requests(entries: list[tuple[dict[str, Any], dict[str, Any]]], model: str
                    ) -> list[list[tuple[dict[str, Any], dict[str, Any]]]]:
    """Pack sequential requests so every upper-bound reserve is at most $1."""
    out: list[list[tuple[dict[str, Any], dict[str, Any]]]] = []
    current: list[tuple[dict[str, Any], dict[str, Any]]] = []
    input_bytes = output_tokens = 0
    for entry in entries:
        request = entry[0]
        request_bytes = len(canonical_json(request))
        request_output = int(request["params"]["max_tokens"])
        candidate_bytes = input_bytes + request_bytes
        candidate_output = output_tokens + request_output
        reserve = worst_case_cost(model, math.ceil(candidate_bytes / 3), candidate_output, batch=True)
        if current and reserve > CHUNK_RESERVE_TARGET_USD:
            out.append(current)
            current = [entry]
            input_bytes, output_tokens = request_bytes, request_output
            reserve = worst_case_cost(model, math.ceil(input_bytes / 3), output_tokens, batch=True)
        else:
            current.append(entry)
            input_bytes, output_tokens = candidate_bytes, candidate_output
        if reserve > HARD_CAP_USD:
            raise WriterError(f"one {model} request requires a worst-case reserve of ${reserve:.4f}, above $8")
    if current:
        out.append(current)
    return out


def _ledger_partitions(writer: str, sample: str, rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    count = OPUS_3X_LEDGER_PARTITIONS if writer == "opus" and sample == "3x" else 1
    return [rows[(len(rows) * index) // count:(len(rows) * (index + 1)) // count]
            for index in range(count)]


def ledger_path(root: Path, ledger_key: str) -> Path:
    if not ledger_key or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in ledger_key):
        raise WriterError(f"invalid ledger key {ledger_key!r}")
    return root / "ledgers" / f"{ledger_key}.jsonl"


def _chunk_record(*, run_id: str, group_key: str, ledger_key: str,
                  chunk_index: str | int, writer: str, sample: str,
                  unit_type: str, entries: list[tuple[dict[str, Any], dict[str, Any]]],
                  probe: bool = False) -> dict[str, Any]:
    requests = [entry[0] for entry in entries]
    metadata = [entry[1] for entry in entries]
    model = WRITERS[writer]
    return {
        "schema": "part64-writer-chunk-v1",
        "run_id": run_id,
        "group_key": group_key,
        "ledger_key": ledger_key,
        "chunk_index": chunk_index,
        "writer": writer,
        "model": model,
        "sample": sample,
        "unit_type": unit_type,
        "probe": probe,
        "state": "planned",
        "attempt_no": 0,
        "retired_attempts": [],
        "request_count": len(requests),
        "request_sha256": sha256_bytes(canonical_json({"requests": requests,
                                                        "request_metadata": metadata})),
        "planned_reserve_usd": request_reserve_usd(requests, model),
        "requests": requests,
        "request_metadata": metadata,
        "created_at": utc_now(),
    }


def _build_plan(rows: list[dict[str, Any]], collection_sha256: str, *,
                run_id: str | None = None) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    run_id = run_id or uuid.uuid4().hex
    groups: dict[str, dict[str, Any]] = {}
    chunks: dict[str, dict[str, Any]] = {}
    probes: dict[str, dict[str, Any]] = {}
    for writer in WRITERS:
        probe_row = next(row for row in rows if row["type"] == "neuron")
        probe_key = f"probe-{writer}"
        ledger_key = f"{writer}-1x-neuron"
        request = request_for(writer, "1x", probe_row, f"probe-{run_id[:16]}-{writer}")
        probe_meta = {"unit_key": probe_row["unit_key"], "type": "neuron", "layer": probe_row.get("layer"),
                      "sample": "1x", "writer": writer, "probe": True,
                      "input_windows": windows_for_sample(probe_row, "1x")}
        relpath = f"probes/{writer}.json"
        chunks[relpath] = _chunk_record(run_id=run_id, group_key=probe_key, ledger_key=ledger_key,
                                        chunk_index="probe", writer=writer, sample="1x",
                                        unit_type="neuron", entries=[(request, probe_meta)], probe=True)
        probes[writer] = {"chunk_path": relpath, "ledger_key": ledger_key}

    for writer, model in WRITERS.items():
        for sample in SAMPLES:
            for unit_type in UNIT_TYPES:
                group_key = f"{writer}-{sample}-{unit_type}"
                selected = [row for row in rows if row["type"] == unit_type]
                # Opus 3x has enough output-token exposure that one $8 ledger
                # can stop before all 250 units settle. Keep each partition's
                # full max-token reserve under the hard cap. Other arms stay in
                # one ledger per registered group.
                partitions = _ledger_partitions(writer, sample, selected)
                partition_count = len(partitions)
                paths = []
                group_tag = hashlib.sha256(group_key.encode("utf-8")).hexdigest()[:12]
                chunk_number = 0
                for part_index, partition in enumerate(partitions):
                    ledger_key = (f"{group_key}-part-{part_index + 1:02d}"
                                  if partition_count > 1 else group_key)
                    entries = []
                    for index, row in enumerate(partition, start=(len(selected) * part_index) // partition_count):
                        custom_id = f"p64-{run_id[:16]}-{group_tag}-{index:04d}"
                        request = request_for(writer, sample, row, custom_id)
                        entries.append((request, {
                            "unit_key": row["unit_key"], "type": unit_type, "layer": row.get("layer"),
                            "unit": row.get("unit"), "control_id": row.get("control_id"),
                            "sample": sample, "writer": writer, "request_index": index,
                            "input_windows": windows_for_sample(row, sample),
                        }))
                    request_chunks = _chunk_requests(entries, model)
                    for entry_chunk in request_chunks:
                        relpath = f"chunks/{group_key}/chunk-{chunk_number:04d}.json"
                        chunks[relpath] = _chunk_record(
                            run_id=run_id, group_key=group_key, ledger_key=ledger_key,
                            chunk_index=chunk_number, writer=writer, sample=sample, unit_type=unit_type,
                            entries=entry_chunk)
                        paths.append(relpath)
                        chunk_number += 1
                groups[group_key] = {"writer": writer, "model": model, "sample": sample,
                                     "unit_type": unit_type, "chunk_paths": paths}

    plan_identity = {"collection_sha256": collection_sha256,
                     "groups": groups, "probes": probes,
                     "chunks": {path: value["request_sha256"] for path, value in sorted(chunks.items())}}
    manifest = {"schema": "part64-writer-run-v1", "run_id": run_id,
                "collection_sha256": collection_sha256,
                "plan_sha256": sha256_bytes(canonical_json(plan_identity)),
                "groups": groups, "probes": probes,
                "created_at": utc_now(), "status": "planned"}
    return manifest, chunks


def initialize_plan(root: Path, rows: list[dict[str, Any]], collection_sha256: str) -> dict[str, Any]:
    """Create a durable request plan or verify that an existing plan is identical."""
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "run.json"
    existing = read_json(manifest_path) if manifest_path.is_file() else None
    saved_run_id = existing.get("run_id") if isinstance(existing, dict) else None
    if existing is not None and (not isinstance(saved_run_id, str) or not saved_run_id):
        raise WriterError("existing Part 64 run has no valid durable run_id")
    expected, chunk_records = _build_plan(rows, collection_sha256, run_id=saved_run_id)
    if manifest_path.is_file():
        if (existing.get("schema") != expected["schema"]
                or existing.get("collection_sha256") != collection_sha256
                or existing.get("plan_sha256") != expected["plan_sha256"]):
            raise WriterError("existing Part 64 run does not match the registered collection/request plan")
        for relpath, planned in chunk_records.items():
            path = root / relpath
            if not path.is_file():
                raise WriterError(f"planned chunk record is missing: {path}")
            saved = read_json(path)
            if saved.get("request_sha256") != planned["request_sha256"]:
                raise WriterError(f"durable request plan changed: {path}")
        return existing
    for relpath, record in chunk_records.items():
        write_json(root / relpath, record)
    write_json(manifest_path, expected)
    return expected


def load_existing_plan(root: Path, rows: list[dict[str, Any]], collection_sha256: str) -> dict[str, Any]:
    path = root / "run.json"
    if not path.is_file():
        raise WriterError(f"no Part 64 writer plan exists in {root}; run `probe` first")
    return initialize_plan(root, rows, collection_sha256)


def api_context() -> ssl.SSLContext:
    ca_bundle = os.environ.get("ANTHROPIC_CA_BUNDLE", "~/.mitmproxy/ca-bundle-combined.crt")
    ca_path = Path(os.path.expanduser(ca_bundle))
    if not ca_path.is_file() or not os.access(ca_path, os.R_OK):
        raise WriterError(f"Anthropic CA bundle is missing or unreadable: {ca_path}")
    try:
        return ssl.create_default_context(cafile=str(ca_path))
    except (OSError, ssl.SSLError) as exc:
        raise WriterError(f"cannot load Anthropic CA bundle {ca_path}: {exc}") from exc


def preflight_api() -> None:
    if not os.environ.get("ANTHROPIC_CREDITS_KEY"):
        raise WriterError("ANTHROPIC_CREDITS_KEY is not set; no reserve or POST was made")
    api_context()


def api_call(method: str, path: str, body: Mapping[str, Any] | None = None) -> str:
    key = os.environ.get("ANTHROPIC_CREDITS_KEY")
    if not key:
        raise WriterError("ANTHROPIC_CREDITS_KEY is not set")
    data = canonical_json(dict(body)) if body is not None else None
    request = urllib.request.Request(
        "https://api.anthropic.com" + path, method=method, data=data,
        headers={"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION,
                 "content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, context=api_context(), timeout=HTTP_TIMEOUT_SECONDS) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise HTTPFailure(exc.code, detail or str(exc.reason)) from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise WriterError(f"Anthropic API transport failed: {type(exc).__name__}: {exc}") from None


def _attempts(ledger: Ledger) -> dict[str, dict[str, Any]]:
    with ledger._locked():
        events, _tail = ledger._read_locked()
        return ledger._attempts(events)


def _find_attempt(ledger: Ledger, chunk: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
    matches = [(attempt_id, row) for attempt_id, row in _attempts(ledger).items()
               if row.get("kind") == "part64-writer"
               and row.get("meta", {}).get("local_run_id") == chunk.get("run_id")
               and row.get("meta", {}).get("group_key") == chunk.get("group_key")
               and row.get("meta", {}).get("chunk_index") == chunk.get("chunk_index")
               and row.get("meta", {}).get("attempt_no", 0) == chunk.get("attempt_no", 0)]
    if len(matches) > 1:
        raise RecoveryRequired(f"multiple ledger reservations match {chunk.get('group_key')} chunk {chunk.get('chunk_index')}")
    if matches and matches[0][1].get("meta", {}).get("request_sha256") != chunk.get("request_sha256"):
        raise RecoveryRequired("ledger reservation request digest differs from the durable chunk plan")
    return matches[0] if matches else None


def _kill_at(requested: str | None, stage: str) -> None:
    if requested == stage:
        raise RuntimeError(f"simulated process kill at {stage}")


def _parse_result_jsonl(text: str) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WriterError(f"batch results line {line_no} is invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise WriterError(f"batch results line {line_no} is not an object")
        rows.append(row)
    return rows


def _poll_until_ended(batch_id: str, api: Callable[..., str], *, interval: float,
                      bound: float, sleep: Callable[[float], Any]) -> None:
    deadline = time.monotonic() + bound
    while True:
        try:
            status = json.loads(api("GET", f"/v1/messages/batches/{batch_id}"))
        except (json.JSONDecodeError, TypeError) as exc:
            raise WriterError(f"batch {batch_id} returned invalid status JSON: {exc}") from exc
        if not isinstance(status, dict):
            raise WriterError(f"batch {batch_id} status is not an object")
        if status.get("processing_status") == "ended":
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"batch {batch_id} did not end within {bound:g}s; reserve remains active")
        sleep(min(interval, max(0.0, deadline - time.monotonic())))


def _result_bundle(chunk: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    requests = chunk["requests"]
    metadata = chunk["request_metadata"]
    by_id = {request["custom_id"]: (request, meta) for request, meta in zip(requests, metadata)}
    expected = set(by_id)
    records: dict[str, dict[str, Any]] = {}
    total_input = total_output = 0
    usage_rows = []
    for row in rows:
        custom_id = row.get("custom_id")
        if custom_id not in expected:
            raise WriterError(f"batch returned unknown custom_id {custom_id!r}")
        if custom_id in records:
            raise WriterError(f"batch returned duplicate custom_id {custom_id!r}")
        result = row.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("type"), str):
            raise WriterError(f"batch request {custom_id} has an invalid result type; reserve remains active")
        result_type = result["type"]
        request, meta = by_id[custom_id]
        if result_type != "succeeded":
            # Anthropic terminal per-request failures carry no billable usage.
            # Preserve one non-pass row per planned request and settle only the
            # usage-bearing successful rows in this batch.
            records[custom_id] = {
                "custom_id": custom_id, "metadata": meta, "text": "", "parsed": None,
                "parse_status": f"api_{result_type}", "stop_reason": None, "usage": None,
                "provider_result": result,
            }
            continue
        message = result.get("message")
        if not isinstance(message, dict):
            raise WriterError(f"batch request {custom_id} has no message; reserve remains active")
        usage = message.get("usage")
        if not isinstance(usage, dict):
            raise WriterError(f"batch request {custom_id} has no usage; reserve remains active")
        input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
        if (isinstance(input_tokens, bool) or not isinstance(input_tokens, int) or input_tokens < 0
                or isinstance(output_tokens, bool) or not isinstance(output_tokens, int) or output_tokens < 0):
            raise WriterError(f"batch request {custom_id} has invalid token usage; reserve remains active")
        total_input += input_tokens
        total_output += output_tokens
        usage_rows.append({"custom_id": custom_id, "input_tokens": input_tokens,
                           "output_tokens": output_tokens})
        content = message.get("content", [])
        if not isinstance(content, list):
            content = []
        text = "".join(block.get("text", "") for block in content
                       if isinstance(block, dict) and block.get("type") == "text"
                       and isinstance(block.get("text", ""), str)).strip()
        parsed = gen60.parse_json(text)
        valid, reason = gen60.validate_output("M1", "primary", parsed, meta.get("input_windows", []))
        stop_reason = message.get("stop_reason")
        records[custom_id] = {
            "custom_id": custom_id, "metadata": meta,
            "text": text, "parsed": parsed,
            "parse_status": ("max_tokens" if stop_reason == "max_tokens"
                             else "ok" if valid else reason),
            "stop_reason": stop_reason, "usage": usage,
            "provider_result": row.get("result"),
        }
    missing = expected - set(records)
    if missing:
        raise WriterError(f"batch omitted {len(missing)} request results; reserve remains active")
    ordered = [records[request["custom_id"]] for request in requests]
    usage = {"input_tokens": total_input, "output_tokens": total_output,
             "requests": usage_rows,
             "non_succeeded_custom_ids": [request["custom_id"] for request in requests
                                          if records[request["custom_id"]].get("usage") is None]}
    rates = BATCH_RATES_USD_PER_MILLION[chunk["model"]]
    actual_usd = (total_input * rates[0] + total_output * rates[1]) / 1_000_000
    return {"schema": "part64-writer-results-v1", "group_key": chunk["group_key"],
            "chunk_index": chunk["chunk_index"], "batch_id": chunk.get("batch_id"),
            "rows": ordered, "usage": usage, "actual_usd": actual_usd}


def execute_chunk(root: Path, chunk_path: Path, api: Callable[..., str] = api_call, *,
                  kill_at: str | None = None, poll_interval: float = POLL_INTERVAL_SECONDS,
                  poll_bound: float = POLL_BOUND_SECONDS,
                  sleep: Callable[[float], Any] = time.sleep) -> dict[str, Any]:
    """Execute or recover one chunk without ever resubmitting a known/ambiguous POST."""
    chunk = read_json(chunk_path)
    if chunk.get("schema") != "part64-writer-chunk-v1":
        raise WriterError(f"invalid chunk record: {chunk_path}")
    ledger = Ledger(str(ledger_path(root, chunk["ledger_key"])), HARD_CAP_USD)
    state = chunk.get("state")
    if state == "settled":
        return chunk
    if state == "retired":
        if chunk.get("batch_id"):
            raise RecoveryRequired("a known batch was retired; recover its recorded results before any retry")
        old_attempt_id = chunk.get("attempt_id")
        old_attempt = _attempts(ledger).get(old_attempt_id, {})
        if old_attempt.get("status") != "retired":
            raise RecoveryRequired("retired chunk does not have a retired ledger attempt")
        history = list(chunk.get("retired_attempts", []))
        history.append({"attempt_id": old_attempt_id, "attempt_no": chunk.get("attempt_no", 0),
                        "retirement_who": chunk.get("retirement_who"),
                        "retirement_reason": chunk.get("retirement_reason")})
        for field in ("attempt_id", "batch_id", "processing_status", "reserved_at", "posting_started_at",
                      "submitted_at", "results_file", "usage", "actual_usd", "results_saved_at",
                      "settled_at", "error"):
            chunk.pop(field, None)
        chunk.update(state="planned", attempt_no=chunk.get("attempt_no", 0) + 1,
                     retired_attempts=history, retry_after_retirement_at=utc_now())
        write_json(chunk_path, chunk)
        state = "planned"
    if state == "rejected":
        raise RecoveryRequired(f"chunk {chunk['group_key']} {chunk['chunk_index']} is {state}; create a new reviewed run")

    if state == "planned" or (state == "reserved" and not chunk.get("attempt_id")):
        match = _find_attempt(ledger, chunk)
        if match:
            attempt_id, attempt = match
            if attempt.get("status") != "reserved":
                raise RecoveryRequired(f"chunk has a ledger attempt in {attempt.get('status')} state but no matching local transition")
        else:
            attempt_id = ledger.reserve(
                float(chunk["planned_reserve_usd"]), "part64-writer",
                {"stage": "part64", "run_id": f"{chunk['run_id']}:{chunk['ledger_key']}",
                 "local_run_id": chunk["run_id"],
                 "group_key": chunk["group_key"], "chunk_index": chunk["chunk_index"],
                 "attempt_no": chunk.get("attempt_no", 0),
                 "request_sha256": chunk["request_sha256"]})
            _kill_at(kill_at, "after_reserve")
        chunk.update(state="reserved", attempt_id=attempt_id, reserved_at=utc_now())
        write_json(chunk_path, chunk)
        state = "reserved"

    if state in ("posting", "ambiguous") and not chunk.get("batch_id"):
        attempt = _attempts(ledger).get(chunk.get("attempt_id"), {})
        if state == "posting" and attempt.get("status") == "released":
            # A known 4xx was released durably before a crash interrupted the
            # local state write. Reconstruct the known rejection from the
            # ledger instead of misclassifying it as an unknown POST.
            chunk.update(state="rejected", error="known API rejection; recovered from released ledger attempt")
            write_json(chunk_path, chunk)
            raise RecoveryRequired("POST was durably released after a known API rejection; chunk marked rejected")
        matches, unresolved = _find_provider_batches(api, chunk)
        if len(matches) == 1:
            chunk.update(state="submitted", batch_id=matches[0], processing_status="ended",
                         recovered_batch_at=utc_now())
            write_json(chunk_path, chunk)
            state = "submitted"
        else:
            reason = (f"provider listing found {len(matches)} matching batches" if matches else
                      "provider outcome is unknown and no ended matching batch was found")
            if unresolved:
                reason += f"; unresolved candidate batches: {unresolved}"
            chunk.update(state="ambiguous", error=reason)
            write_json(chunk_path, chunk)
            raise RecoveryRequired(reason + "; do not resubmit; retry recovery or verify before retiring")

    if state == "reserved":
        attempt_id = chunk.get("attempt_id")
        if not isinstance(attempt_id, str):
            raise WriterError("reserved chunk has no attempt_id")
        matches = _attempts(ledger)
        if matches.get(attempt_id, {}).get("status") != "reserved":
            raise RecoveryRequired("chunk's paid ledger attempt is not an active reserve")
        chunk.update(state="posting", posting_started_at=utc_now())
        write_json(chunk_path, chunk)
        try:
            raw_response = api("POST", "/v1/messages/batches", {"requests": chunk["requests"]})
        except HTTPFailure as exc:
            try:
                ledger.release(attempt_id, exc.http_status, exc.body)
                chunk.update(state="rejected", error=f"known API rejection HTTP {exc.http_status}")
                write_json(chunk_path, chunk)
            except LedgerError:
                chunk.update(state="ambiguous", error=f"unreleased API failure HTTP {exc.http_status}")
                write_json(chunk_path, chunk)
            raise WriterError(f"batch POST failed for {chunk['group_key']}: HTTP {exc.http_status}") from exc
        except Exception as exc:
            chunk.update(state="ambiguous", error=f"POST outcome unknown: {type(exc).__name__}: {exc}")
            write_json(chunk_path, chunk)
            raise RecoveryRequired("POST outcome is unknown; reserve retained and resubmission refused") from exc
        _kill_at(kill_at, "after_post_before_id")
        try:
            response = json.loads(raw_response)
        except (TypeError, json.JSONDecodeError) as exc:
            chunk.update(state="ambiguous", error=f"invalid POST response: {exc}")
            write_json(chunk_path, chunk)
            raise RecoveryRequired("POST response cannot be parsed; reserve retained and resubmission refused") from exc
        batch_id = response.get("id") if isinstance(response, dict) else None
        if not isinstance(batch_id, str) or not batch_id.strip():
            chunk.update(state="ambiguous", error="POST response has no batch id")
            write_json(chunk_path, chunk)
            raise RecoveryRequired("POST response has no batch ID; reserve retained and resubmission refused")
        chunk.update(state="submitted", batch_id=batch_id, processing_status=response.get("processing_status"),
                     submitted_at=utc_now())
        write_json(chunk_path, chunk)  # Durable batch ID precedes every poll.
        state = "submitted"
        _kill_at(kill_at, "after_id_before_results")

    results_path = chunk_path.with_name(chunk_path.stem + "-results.json")
    if state == "submitted":
        batch_id = chunk.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            raise RecoveryRequired("submitted chunk has no durable batch ID")
        _poll_until_ended(batch_id, api, interval=poll_interval, bound=poll_bound, sleep=sleep)
        lines = api("GET", f"/v1/messages/batches/{batch_id}/results")
        result_rows = _parse_result_jsonl(lines)
        bundle = _result_bundle(chunk, result_rows)
        write_json(results_path, bundle)
        chunk.update(state="results_saved", results_file=results_path.name,
                     usage=bundle["usage"], actual_usd=bundle["actual_usd"], results_saved_at=utc_now())
        write_json(chunk_path, chunk)
        state = "results_saved"
        _kill_at(kill_at, "after_results_before_settle")

    if state == "results_saved":
        if not results_path.is_file():
            raise RecoveryRequired("chunk claims saved results but its durable result file is missing")
        bundle = read_json(results_path)
        if (bundle.get("usage") != chunk.get("usage")
                or float(bundle.get("actual_usd", -1)) != float(chunk.get("actual_usd", -2))):
            raise RecoveryRequired("saved result usage differs from the chunk settlement record")
        ledger.settle(chunk["attempt_id"], float(chunk["actual_usd"]), chunk["usage"])
        chunk.update(state="settled", settled_at=utc_now())
        write_json(chunk_path, chunk)
        return chunk

    raise WriterError(f"unsupported chunk state {state!r}")


def probe_status(root: Path, manifest: Mapping[str, Any], writer: str) -> dict[str, Any]:
    path = root / manifest["probes"][writer]["chunk_path"]
    chunk = read_json(path)
    if chunk.get("state") != "settled":
        return {"writer": writer, "status": chunk.get("state", "unknown"), "passed": False}
    bundle_path = path.with_name(path.stem + "-results.json")
    bundle = read_json(bundle_path)
    rows = bundle.get("rows", [])
    result = rows[0] if len(rows) == 1 else {}
    stop_reason = result.get("stop_reason")
    text = result.get("text")
    passed = (isinstance(text, str) and bool(text.strip())
              and stop_reason != "max_tokens" and result.get("parse_status") == "ok")
    return {"writer": writer, "status": "settled", "passed": passed,
            "custom_id": result.get("custom_id"), "unit_key": result.get("metadata", {}).get("unit_key"),
            "stop_reason": stop_reason, "parse_status": result.get("parse_status"),
            "text_nonempty": bool(text.strip()) if isinstance(text, str) else False,
            "batch_id": chunk.get("batch_id")}


def execute_probes(root: Path, manifest: Mapping[str, Any], api: Callable[..., str] = api_call,
                   **kwargs: Any) -> list[dict[str, Any]]:
    outcomes = []
    for writer in WRITERS:
        path = root / manifest["probes"][writer]["chunk_path"]
        execute_chunk(root, path, api, **kwargs)
        outcome = probe_status(root, manifest, writer)
        outcomes.append(outcome)
    failed = [outcome for outcome in outcomes if not outcome["passed"]]
    if failed:
        raise WriterError(f"one or more writer M1 probes failed validation: {failed}")
    return outcomes


def reprobe(root: Path, writer: str, who: str, reason: str,
            api: Callable[..., str] = api_call, **kwargs: Any) -> dict[str, Any]:
    """Create an explicitly recorded new probe attempt after a settled failure."""
    if writer not in WRITERS:
        raise WriterError(f"unknown writer {writer!r}")
    if not who.strip() or not reason.strip():
        raise WriterError("re-probe requires a nonempty operator and reason")
    if api is api_call:
        preflight_api()
    manifest_path = root / "run.json"
    manifest = read_json(manifest_path)
    info = manifest.get("probes", {}).get(writer)
    if not isinstance(info, dict) or not isinstance(info.get("chunk_path"), str):
        raise WriterError(f"run has no probe plan for {writer}")
    old_path = root / info["chunk_path"]
    old = read_json(old_path)
    outcome = probe_status(root, manifest, writer)
    if old.get("state") != "settled" or outcome.get("passed"):
        raise WriterError(f"{writer} probe can be repeated only after a settled failed probe")
    attempt_no = int(old.get("attempt_no", 0)) + 1
    entries = list(zip(old["requests"], old["request_metadata"]))
    record = _chunk_record(
        run_id=old["run_id"], group_key=old["group_key"], ledger_key=old["ledger_key"],
        chunk_index=old["chunk_index"], writer=old["writer"], sample=old["sample"],
        unit_type=old["unit_type"], entries=entries, probe=True)
    authorization = {"who": who.strip(), "reason": reason.strip(), "authorized_at": utc_now(),
                     "prior_chunk": info["chunk_path"], "prior_attempt_no": old.get("attempt_no", 0)}
    record["attempt_no"] = attempt_no
    record["reprobe_authorization"] = authorization
    relpath = f"probes/{writer}-attempt-{attempt_no:03d}.json"
    write_json(root / relpath, record)
    history = list(info.get("reprobes", []))
    history.append(authorization)
    info["reprobes"] = history
    info["chunk_path"] = relpath
    write_json(manifest_path, manifest)
    execute_chunk(root, root / relpath, api, **kwargs)
    result = probe_status(root, manifest, writer)
    result["reprobe_authorization"] = authorization
    if not result["passed"]:
        raise WriterError(f"{writer} re-probe failed validation: {result}")
    return result


def _write_group_output(root: Path, group_key: str, group: Mapping[str, Any]) -> Path:
    rows = []
    for relpath in group["chunk_paths"]:
        chunk_path = root / relpath
        chunk = read_json(chunk_path)
        if chunk.get("state") != "settled":
            raise RecoveryRequired(f"cannot assemble {group_key}; chunk {chunk.get('chunk_index')} is {chunk.get('state')}")
        results_path = chunk_path.with_name(chunk_path.stem + "-results.json")
        bundle = read_json(results_path)
        rows.extend(bundle["rows"])
    rows.sort(key=lambda row: int(row.get("metadata", {}).get("request_index", 0)))
    output_path = root / "outputs" / f"{group_key}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output_path)
    return output_path


def _run_groups(root: Path, manifest: dict[str, Any], api: Callable[..., str],
                **kwargs: Any) -> None:
    for group_key, group in manifest["groups"].items():
        for relpath in group["chunk_paths"]:
            execute_chunk(root, root / relpath, api, **kwargs)
        _write_group_output(root, group_key, group)
    manifest["status"] = "complete"
    manifest["completed_at"] = utc_now()
    write_json(root / "run.json", manifest)


def run_probe(collection: Path = DEFAULT_COLLECTION, root: Path = DEFAULT_ROOT,
              api: Callable[..., str] = api_call, **kwargs: Any) -> list[dict[str, Any]]:
    rows, digest = load_collection(collection)
    if api is api_call:
        preflight_api()
    with run_lock(root):
        manifest = initialize_plan(root, rows, digest)
        return execute_probes(root, manifest, api, **kwargs)


def run_generation(collection: Path = DEFAULT_COLLECTION, root: Path = DEFAULT_ROOT,
                   api: Callable[..., str] = api_call, *, initialize: bool = False,
                   **kwargs: Any) -> None:
    rows, digest = load_collection(collection)
    if api is api_call:
        preflight_api()
    with run_lock(root):
        manifest = (initialize_plan(root, rows, digest) if initialize
                    else load_existing_plan(root, rows, digest))
        try:
            execute_probes(root, manifest, api, **kwargs)
        except WriterError as exc:
            raise WriterError(f"generation refused until both writer probes pass: {exc}") from exc
        _run_groups(root, manifest, api, **kwargs)


def _parse_provider_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _find_provider_batches(api: Callable[..., str], chunk: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Prove that no accepted provider batch contains this unknown POST's IDs.

    The request IDs include the durable run and group identity. A fresh provider
    listing and result scan is required before an unknown reserve can be retired.
    Returns ended matches and recent batches whose outcomes are not yet inspectable.
    Any incomplete listing fails closed.
    """
    expected = {request.get("custom_id") for request in chunk.get("requests", [])}
    if not expected or any(not isinstance(item, str) for item in expected):
        raise RecoveryRequired("cannot verify provider listing without complete planned custom IDs")
    posted_at = _parse_provider_time(chunk.get("posting_started_at"))
    if posted_at is None:
        raise RecoveryRequired("unknown POST has no valid posting timestamp; provider listing cannot prove absence")

    batches: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    before_id: str | None = None
    for _page in range(1000):
        path = "/v1/messages/batches?limit=100"
        if before_id:
            path += f"&before_id={before_id}"
        try:
            page = json.loads(api("GET", path))
        except Exception as exc:
            raise RecoveryRequired(f"cannot inspect provider batch listing: {type(exc).__name__}: {exc}") from exc
        if not isinstance(page, dict) or not isinstance(page.get("data"), list):
            raise RecoveryRequired("provider batch listing did not return a data array")
        for batch in page["data"]:
            if not isinstance(batch, dict) or not isinstance(batch.get("id"), str):
                raise RecoveryRequired("provider batch listing contains an invalid batch record")
            if batch["id"] in seen_ids:
                raise RecoveryRequired("provider batch listing pagination repeated a batch")
            seen_ids.add(batch["id"])
            batches.append(batch)
        has_more = page.get("has_more")
        if not isinstance(has_more, bool):
            raise RecoveryRequired("provider batch listing omitted its has_more completeness flag")
        if not has_more:
            break
        next_id = page.get("last_id")
        if not isinstance(next_id, str) or not next_id or next_id in seen_ids:
            raise RecoveryRequired("provider batch listing has an invalid pagination cursor")
        before_id = next_id
    else:
        raise RecoveryRequired("provider batch listing exceeded the 1000-page safety bound")

    uncertain_after = posted_at - timedelta(minutes=5)
    matches: list[str] = []
    unresolved: list[str] = []
    for batch in batches:
        created_at = _parse_provider_time(batch.get("created_at"))
        if created_at is None:
            raise RecoveryRequired(f"batch {batch['id']} has no valid created_at; cannot prove it predates this POST")
        status = batch.get("processing_status")
        if status != "ended":
            if created_at >= uncertain_after:
                unresolved.append(batch["id"])
            continue
        try:
            result_text = api("GET", f"/v1/messages/batches/{batch['id']}/results")
            results = _parse_result_jsonl(result_text)
        except Exception as exc:
            if created_at >= uncertain_after:
                unresolved.append(batch["id"])
            continue
        if any(row.get("custom_id") in expected for row in results):
            matches.append(batch["id"])
    return matches, unresolved


def _ensure_no_provider_batch(api: Callable[..., str], chunk: Mapping[str, Any]) -> None:
    matches, unresolved = _find_provider_batches(api, chunk)
    if matches:
        raise RecoveryRequired(f"provider batch(es) {matches} contain this chunk's custom IDs; recover instead of retiring")
    if unresolved:
        raise RecoveryRequired(f"provider batch(es) {unresolved} may be the unknown POST and are not inspectable yet")


def retire_attempt(root: Path, attempt_id: str, who: str, reason: str,
                   api: Callable[..., str] = api_call) -> None:
    """Retire only a pre-POST reserve or an unknown POST proven absent at provider."""
    manifest = read_json(root / "run.json")
    candidates = [root / item["chunk_path"] for item in manifest.get("probes", {}).values()]
    for group in manifest.get("groups", {}).values():
        candidates.extend(root / relpath for relpath in group["chunk_paths"])
    for path in candidates:
        chunk = read_json(path)
        if chunk.get("attempt_id") != attempt_id:
            continue
        if chunk.get("state") in ("settled", "rejected", "retired"):
            raise WriterError(f"attempt {attempt_id} belongs to terminal chunk state {chunk['state']}")
        state = chunk.get("state")
        if state not in ("reserved", "posting", "ambiguous") or chunk.get("batch_id"):
            raise WriterError("only a pre-POST reserve or an unknown POST without a durable batch ID can be retired")
        if state in ("posting", "ambiguous"):
            _ensure_no_provider_batch(api, chunk)
        reader63.retire_attempt(attempt_id, who, reason,
                                ledger_path=ledger_path(root, chunk["ledger_key"]))
        chunk.update(state="retired", retired_at=utc_now(), retirement_who=who.strip(),
                     retirement_reason=reason.strip())
        write_json(path, chunk)
        return
    raise WriterError(f"attempt {attempt_id} is not recorded in {root}")


def status_report(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "run.json")
    chunks = []
    for writer, info in manifest.get("probes", {}).items():
        chunk = read_json(root / info["chunk_path"])
        chunks.append({"group_key": chunk["group_key"], "state": chunk["state"],
                       "batch_id": chunk.get("batch_id"), "attempt_id": chunk.get("attempt_id")})
    for group_key, group in manifest.get("groups", {}).items():
        for relpath in group["chunk_paths"]:
            chunk = read_json(root / relpath)
            chunks.append({"group_key": group_key, "chunk_index": chunk["chunk_index"],
                           "state": chunk["state"], "batch_id": chunk.get("batch_id"),
                           "attempt_id": chunk.get("attempt_id")})
    ledger_states = {}
    for key in {chunk["ledger_key"] for chunk in [read_json(root / info["chunk_path"])
                                                   for info in manifest.get("probes", {}).values()] +
                [read_json(root / path) for group in manifest.get("groups", {}).values()
                 for path in group["chunk_paths"]]}:
        ledger_states[key] = Ledger(str(ledger_path(root, key)), HARD_CAP_USD).state()
    return {"schema": manifest.get("schema"), "run_id": manifest.get("run_id"),
            "status": manifest.get("status"), "chunks": chunks, "ledgers": ledger_states}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for command, help_text in (("probe", "send one paid M1 probe per writer"),
                               ("run", "submit all four arms after successful writer probes"),
                               ("resume", "resume recorded probes/chunks without resubmitting known batches")):
        command_parser = sub.add_parser(command, help=help_text)
        command_parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
        command_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    reprobe_parser = sub.add_parser("reprobe", help="explicitly create a new attempt after a failed writer probe")
    reprobe_parser.add_argument("writer", choices=tuple(WRITERS))
    reprobe_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    reprobe_parser.add_argument("--who", required=True)
    reprobe_parser.add_argument("--reason", required=True)
    status = sub.add_parser("status", help="show chunk and per-ledger state")
    status.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    retire = sub.add_parser("retire", help="operator-retire an unresolved paid attempt")
    retire.add_argument("attempt_id")
    retire.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    retire.add_argument("--who", required=True)
    retire.add_argument("--reason", required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "probe":
        outcomes = run_probe(args.collection, args.root)
        print(json.dumps(outcomes, ensure_ascii=False, indent=2))
    elif args.command == "reprobe":
        with run_lock(args.root):
            outcome = reprobe(args.root, args.writer, args.who, args.reason)
        print(json.dumps(outcome, ensure_ascii=False, indent=2))
    elif args.command == "run":
        run_generation(args.collection, args.root, initialize=False)
        print(f"completed Part 64 writer run in {args.root}")
    elif args.command == "resume":
        run_generation(args.collection, args.root, initialize=False)
        print(f"resumed Part 64 writer run in {args.root}")
    elif args.command == "status":
        print(json.dumps(status_report(args.root), ensure_ascii=False, indent=2))
    elif args.command == "retire":
        with run_lock(args.root):
            retire_attempt(args.root, args.attempt_id, args.who, args.reason)
        print(f"retired unresolved attempt {args.attempt_id}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (WriterError, LedgerError, OSError, ValueError, TimeoutError) as exc:
        print(f"gen64: {exc}", file=sys.stderr)
        raise SystemExit(2)
