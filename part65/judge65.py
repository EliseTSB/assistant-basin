#!/usr/bin/env python3
"""Blind Part 65 2AFC judge through Anthropic Message Batches.
Reads Part 64 Haiku-1x hypotheses, Part 65 ranked write summaries, and unit
metadata; writes durable request state and answers. Reserves, submits, settles,
and recovers through the existing ../paidaction ledger only. The probe command
submits exactly one message request, then may poll that batch with GET requests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
BASIN = HERE.parent
PAID_ACTION = BASIN / "paidaction"
if str(PAID_ACTION) not in sys.path:
    sys.path.insert(0, str(PAID_ACTION))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from paid_ledger import (DEFAULT_LEDGER_PATH, HARD_CAP_USD, HTTPFailure,
                         Ledger, LedgerError)
from pricing import (HAIKU_BATCH_INPUT_USD_PER_MILLION,
                     HAIKU_BATCH_OUTPUT_USD_PER_MILLION,
                     TOKENS_PER_MILLION, worst_case_cost)
from rank65 import layer_band
from steer65 import (DEFAULT_COLLECTION, atomic_append_jsonl, load_units, _lock_file)

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 32
PART65_CAP_USD = 1.0
SYSTEM_PROMPT = (
    "You are doing a blind two-choice match. One short read label describes a "
    "unit from examples that activated it. Two write summaries describe outputs "
    "from pushing two units. Exactly one write summary is from the same unit as "
    "the read label. Choose the summary that best matches the read label's "
    "meaning, using semantic fit rather than writing style. Reply with only A or B."
)
DEFAULT_READ_DIR = Path("/home/elisepc/green/worktrees/ab-part64/results/part64/gen64/outputs")
DEFAULT_SENTENCE_TEST = Path("/home/elisepc/green/worktrees/ab-part64/results/part64/scored.jsonl")


class JudgeError(RuntimeError):
    """Invalid blind-judge data or unsafe paid-action recovery state."""


class APIHttpError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"Anthropic HTTP {status}")
        self.status = status
        self.body = body


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()
    last_nonblank_line = max((index for index, line in enumerate(lines, 1) if line.strip()), default=0)
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if line_number == last_nonblank_line:
                print(f"warning: ignoring invalid final JSONL line in {path}:{line_number}",
                      file=sys.stderr)
                continue
            raise JudgeError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise JudgeError(f"{path}:{line_number}: expected object")
        rows.append(row)
    return rows


def read_label_rows(neuron_path: Path, control_path: Path) -> dict[str, dict[str, Any]]:
    """Extract the primary Haiku-1x hypothesis for each frozen Part 60 unit."""
    result: dict[str, dict[str, Any]] = {}
    for path, expected_type in ((neuron_path, "neuron"), (control_path, "control")):
        for row in read_jsonl(path):
            meta = row.get("metadata")
            if not isinstance(meta, dict):
                raise JudgeError(f"{path}: label output is missing metadata")
            key, kind = meta.get("unit_key"), meta.get("type")
            if kind != expected_type or not isinstance(key, str) or key in result:
                raise JudgeError(f"{path}: invalid, duplicated, or wrong-type read label")
            parsed = row.get("parsed")
            hypothesis = parsed.get("hypothesis") if isinstance(parsed, dict) else None
            raw_text = row.get("text", "")
            if not isinstance(hypothesis, str) or not hypothesis.strip():
                try:
                    raw = json.loads(raw_text) if isinstance(raw_text, str) else None
                except json.JSONDecodeError:
                    raw = None
                hypothesis = raw.get("hypothesis") if isinstance(raw, dict) else None
            if (not isinstance(hypothesis, str) or not hypothesis.strip()) and isinstance(raw_text, str):
                match = re.search(r'"hypothesis"\s*:\s*("(?:\\.|[^"\\])*")', raw_text)
                if match:
                    try:
                        hypothesis = json.loads(match.group(1))
                    except json.JSONDecodeError:
                        hypothesis = None
            if not isinstance(hypothesis, str) or not hypothesis.strip():
                hypothesis = "No parseable Part 64 Haiku-1x hypothesis was returned."
            result[key] = {
                "unit_key": key,
                "type": kind,
                "layer": meta.get("layer"),
                "read_label": hypothesis.strip(),
                "source": "Part 64 Haiku-1x hypothesis",
            }
    return result


def read_write_summaries(path: Path) -> dict[str, dict[str, Any]]:
    summaries = {}
    for row in read_jsonl(path):
        key = row.get("unit_key")
        if not isinstance(key, str) or key in summaries:
            raise JudgeError(f"{path}: missing or duplicate unit_key")
        words = row.get("write_words")
        lens = row.get("steer_lens")
        if (row.get("type") not in ("neuron", "control")
                or not isinstance(row.get("layer"), int)
                or not isinstance(row.get("layer_band"), str)
                or not isinstance(words, list)
                or not isinstance(lens, dict)
                or not isinstance(lens.get("positive"), list)
                or not isinstance(lens.get("negative"), list)):
            raise JudgeError(f"{key}: malformed Part 65 write summary")
        summaries[key] = row
    return summaries


def read_sentence_tests(path: Path | None) -> dict[str, bool | None]:
    """Read the per-unit Part 64 Haiku-1x four-tier sentence-test decisions."""
    if path is None:
        return {}
    result: dict[str, bool | None] = {}
    for row in read_jsonl(path):
        arm = row.get("arm")
        if arm is None and isinstance(row.get("metadata"), dict):
            metadata = row["metadata"]
            arm = f"{metadata.get('writer', '')}-{metadata.get('sample', '')}"
        if arm != "haiku-1x":
            continue
        key = row.get("unit_key")
        passed = row.get("pass")
        if (not isinstance(key, str) or key in result
                or not isinstance(passed, bool)):
            raise JudgeError(f"{path}: malformed or duplicate Haiku-1x sentence-test row")
        result[key] = passed
    if not result:
        raise JudgeError(f"{path}: no Haiku-1x sentence-test rows found")
    return result


def format_write_summary(row: dict[str, Any]) -> str:
    """Render only the requested write words and steer-lens tokens."""
    words = [str(item) for item in row["write_words"][:10]]
    lens = row["steer_lens"]
    positive = [str(item) for item in lens["positive"][:12]]
    negative = [str(item) for item in lens["negative"][:12]]
    return ("Write words: " + ", ".join(words)
            + "\nTokens made more likely by a positive push: " + ", ".join(positive)
            + "\nTokens made more likely by a negative push: " + ", ".join(negative))


def judge_prompt(read_label: str, summary_a: str, summary_b: str) -> str:
    """Create the blind body; identity, type, layer, and unit keys stay out."""
    return (
        "Read label:\n" + read_label.strip()
        + "\n\nWrite summary A:\n" + summary_a.strip()
        + "\n\nWrite summary B:\n" + summary_b.strip()
    )


def build_trials(units: list[dict[str, Any]], read_labels: dict[str, dict[str, Any]],
                 write_summaries: dict[str, dict[str, Any]],
                 sentence_tests: dict[str, bool | None] | None = None,
                 seed: int = 6501) -> list[dict[str, Any]]:
    """Make two same-type, same-band foils per unit and swap own A/B position."""
    sentence_tests = sentence_tests or {}
    if set(read_labels) != set(write_summaries):
        raise JudgeError("read-label and write-summary unit keys differ")
    expected = {unit["unit_key"] for unit in units}
    if set(read_labels) != expected:
        raise JudgeError("read labels do not cover the exact frozen 500-unit collection")
    if sentence_tests and set(sentence_tests) != expected:
        raise JudgeError("Part 64 sentence-test rows do not cover the exact frozen 500-unit collection")
    by_key = {unit["unit_key"]: unit for unit in units}
    if len(by_key) != len(units):
        raise JudgeError("unit collection has duplicate unit keys")
    candidates: dict[tuple[str, str], list[str]] = {}
    for unit in units:
        key, kind, layer = unit["unit_key"], unit["type"], int(unit["layer"])
        label, summary = read_labels[key], write_summaries[key]
        band = layer_band(layer)
        if (label["type"] != kind or int(label["layer"]) != layer
                or summary["type"] != kind or int(summary["layer"]) != layer
                or summary["layer_band"] != band):
            raise JudgeError(f"{key}: type/layer mismatch across collection and labels")
        candidates.setdefault((kind, band), []).append(key)

    trials = []
    for unit in units:
        key, kind, layer = unit["unit_key"], unit["type"], int(unit["layer"])
        band = layer_band(layer)
        pool = [other for other in candidates[(kind, band)] if other != key]
        if len(pool) < 2:
            raise JudgeError(f"{key}: needs two distinct same-type, same-band foils")
        rng = random.Random(f"{seed}:{key}:foil")
        foils = rng.sample(pool, 2)
        own_summary = format_write_summary(write_summaries[key])
        read_label = read_labels[key]["read_label"]
        for trial_index, foil_key in enumerate(foils):
            own_position = "A" if trial_index == 0 else "B"
            foil_summary = format_write_summary(write_summaries[foil_key])
            if own_position == "A":
                summary_a, summary_b = own_summary, foil_summary
            else:
                summary_a, summary_b = foil_summary, own_summary
            custom_id = f"p65-{len(trials):04d}"
            prompt = judge_prompt(read_label, summary_a, summary_b)
            request = {
                "custom_id": custom_id,
                "params": {
                    "model": MODEL,
                    "max_tokens": MAX_TOKENS,
                    "system": SYSTEM_PROMPT,
                    "thinking": {"type": "disabled"},
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
            trials.append({
                "request": request,
                "mapping": {
                    "custom_id": custom_id,
                    "unit_key": key,
                    "type": kind,
                    "layer": layer,
                    "layer_band": band,
                    "trial": trial_index + 1,
                    "foil_unit_key": foil_key,
                    "own_position": own_position,
                    "sentence_test_pass": sentence_tests.get(key),
                },
            })
    return trials


def build_probe_trial(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not trials:
        raise JudgeError("no eligible unit exists for a live probe")
    return [trials[0]]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_bytes(_canonical_json(value) + b"\n")
    os.replace(temp, path)


def _chunk_reserve(requests: list[dict[str, Any]]) -> float:
    input_bytes = sum(len(_canonical_json(request)) for request in requests)
    input_tokens_upper = math.ceil(input_bytes / 3)
    output_tokens = sum(int(request["params"]["max_tokens"]) for request in requests)
    return worst_case_cost(MODEL, input_tokens_upper, output_tokens, batch=True)


def prepare_run(trials: list[dict[str, Any]], root: Path, chunk_size: int,
                seed: int, probe: bool = False) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    root.mkdir(parents=True, exist_ok=True)
    selected = build_probe_trial(trials) if probe else trials
    plan_digest = hashlib.sha256(_canonical_json(selected)).hexdigest()
    manifest_path = root / "run.json"
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current.get("plan_sha256") != plan_digest or current.get("probe") != probe:
            raise JudgeError(f"{root}: existing plan differs; refusing to mix attempts")
        return
    chunk_dir = root / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for index, start in enumerate(range(0, len(selected), chunk_size)):
        batch = selected[start:start + chunk_size]
        state = {
            "schema": "part65-judge-chunk-v1",
            "chunk_index": index,
            "status": "ready",
            "reserve_usd": _chunk_reserve([trial["request"] for trial in batch]),
            "attempt_id": None,
            "batch_id": None,
            "trials": batch,
        }
        _atomic_json(chunk_dir / f"chunk-{index:04d}.json", state)
    _atomic_json(manifest_path, {
        "schema": "part65-judge-run-v1",
        "plan_sha256": plan_digest,
        "request_count": len(selected),
        "unit_count": len(selected) if probe else len(selected) // 2,
        "seed": seed,
        "probe": probe,
        "model": MODEL,
        "created_at": time.time(),
    })


def api_call(method: str, path: str, body: dict[str, Any] | None = None,
             timeout: float = 120.0) -> str:
    key = os.environ.get("ANTHROPIC_CREDITS_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise JudgeError(
            "ANTHROPIC_CREDITS_KEY or ANTHROPIC_API_KEY is required only when submitting or recovering a live batch"
        )
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        "https://api.anthropic.com" + path,
        data=payload,
        method=method,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    try:
        ca_bundle = Path.home() / ".mitmproxy" / "ca-bundle-combined.crt"
        context = (ssl.create_default_context(cafile=str(ca_bundle)) if ca_bundle.is_file()
                   else ssl.create_default_context())
        with urllib.request.urlopen(request, context=context, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise APIHttpError(exc.code, body_text) from exc
    except urllib.error.URLError as exc:
        raise JudgeError(f"unknown API outcome: {exc.reason}") from exc


def _parse_ndjson(text: str) -> list[dict[str, Any]]:
    output = []
    for index, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise JudgeError(f"batch result line {index} is invalid JSON") from exc
        if not isinstance(row, dict):
            raise JudgeError(f"batch result line {index} is not an object")
        output.append(row)
    return output


def _extract_response(row: dict[str, Any]) -> tuple[str | None, dict[str, int] | None, str | None]:
    result = row.get("result")
    if not isinstance(result, dict):
        return None, None, "missing result object"
    if result.get("type") != "succeeded":
        return None, {"input_tokens": 0, "output_tokens": 0}, str(result.get("type", "failed"))
    message = result.get("message")
    if not isinstance(message, dict):
        return None, None, "missing message object"
    content = message.get("content")
    if not isinstance(content, list):
        return None, None, "missing message content"
    text = "".join(block.get("text", "") for block in content
                   if isinstance(block, dict) and block.get("type") == "text").strip()
    answer = text.strip().upper()
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return answer, None, "missing provider usage"
    try:
        input_tokens = int(usage["input_tokens"])
        output_tokens = int(usage["output_tokens"])
    except (KeyError, TypeError, ValueError):
        return answer, None, "invalid provider usage"
    if min(input_tokens, output_tokens) < 0:
        return answer, None, "negative provider usage"
    match = re.fullmatch(r"([AB])[.)]?", answer)
    parsed_answer = match.group(1) if match else None
    error = None if parsed_answer is not None else "response was not a lone A or B with optional punctuation"
    return parsed_answer, {"input_tokens": input_tokens, "output_tokens": output_tokens}, error


def _actual_cost(usage: dict[str, int]) -> float:
    return (usage["input_tokens"] * HAIKU_BATCH_INPUT_USD_PER_MILLION
            + usage["output_tokens"] * HAIKU_BATCH_OUTPUT_USD_PER_MILLION) / TOKENS_PER_MILLION


def _save_chunk(path: Path, state: dict[str, Any]) -> None:
    _atomic_json(path, state)


def _answer_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {row.get("custom_id"): row for row in state.get("provider_results", [])}
    output = []
    for trial in state["trials"]:
        mapping = trial["mapping"]
        custom_id = mapping["custom_id"]
        provider = by_id.get(custom_id)
        if provider is None:
            raise JudgeError(f"saved batch lacks result for {custom_id}")
        answer, usage, error = _extract_response(provider)
        if error is not None and usage is None:
            raise JudgeError(f"{custom_id}: {error}; reserve remains active for recovery")
        correct = answer == mapping["own_position"] if answer is not None else None
        output.append({
            **mapping,
            "chosen": answer,
            "correct": correct,
            "error": error,
            "usage": usage,
            "batch_id": state["batch_id"],
        })
    return output


def _persist_answers(root: Path, rows: list[dict[str, Any]]) -> None:
    path = root / "answers.jsonl"
    existing = set()
    if path.exists():
        existing = {row.get("custom_id") for row in read_jsonl(path)}
    for row in rows:
        if row["custom_id"] not in existing:
            atomic_append_jsonl(path, row)
            existing.add(row["custom_id"])


def _settle_saved(state: dict[str, Any], ledger: Ledger) -> None:
    usage = state.get("usage")
    attempt_id = state.get("attempt_id")
    if not isinstance(usage, dict) or not isinstance(attempt_id, str):
        raise JudgeError("cannot settle batch without saved usage and attempt_id")
    actual = float(state["actual_usd"])
    ledger.settle(attempt_id, actual, usage)
    state["status"] = "settled"


def _fetch_and_save(path: Path, state: dict[str, Any], api: Callable[..., str],
                    ledger: Ledger, poll_seconds: float, timeout_seconds: float,
                    sleep: Callable[[float], None] = time.sleep) -> str:
    batch_id = state.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        raise JudgeError("submitted chunk has no durable provider batch ID")
    deadline = time.monotonic() + timeout_seconds
    while True:
        status_text = api("GET", f"/v1/messages/batches/{urllib.parse.quote(batch_id, safe='')}")
        status = json.loads(status_text)
        processing = status.get("processing_status")
        if processing == "ended":
            break
        if processing in ("canceled", "cancelled", "expired"):
            raise JudgeError(f"provider batch {batch_id} ended as {processing}; reserve remains active")
        if time.monotonic() >= deadline:
            return "pending"
        sleep(poll_seconds)
    result_text = api("GET", f"/v1/messages/batches/{urllib.parse.quote(batch_id, safe='')}/results")
    results = _parse_ndjson(result_text)
    expected = {trial["request"]["custom_id"] for trial in state["trials"]}
    found = {row.get("custom_id") for row in results}
    if found != expected or len(found) != len(results):
        raise JudgeError(f"batch results differ from planned custom IDs; reserve remains active")
    usage_total = {"input_tokens": 0, "output_tokens": 0}
    for row in results:
        answer, usage, error = _extract_response(row)
        if usage is None:
            raise JudgeError(f"{row.get('custom_id')}: {error}; reserve remains active")
        usage_total["input_tokens"] += usage["input_tokens"]
        usage_total["output_tokens"] += usage["output_tokens"]
    state["provider_results"] = results
    state["usage"] = usage_total
    state["actual_usd"] = _actual_cost(usage_total)
    state["status"] = "results_saved"
    _save_chunk(path, state)
    _settle_saved(state, ledger)
    _save_chunk(path, state)
    _persist_answers(path.parent.parent, _answer_rows(state))
    _save_chunk(path, state)
    return "settled"


def _prior_reservation(ledger_path: Path, plan_hash: str, chunk_index: int) -> str | None:
    """Recover a reserve written just before a crash persisted its attempt ID locally."""
    if not ledger_path.exists():
        return None
    rows = []
    with ledger_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JudgeError(f"{ledger_path}:{line_number}: paidaction ledger has an invalid line") from exc
            if isinstance(row, dict):
                rows.append(row)
    matches = [
        row for row in rows
        if row.get("event") == "reserve"
        and row.get("kind") == "part65-judge"
        and isinstance(row.get("meta"), dict)
        and row["meta"].get("plan_sha256") == plan_hash
        and row["meta"].get("chunk_index") == chunk_index
    ]
    if len(matches) > 1:
        raise JudgeError("multiple paidaction reservations exist for one Part 65 plan chunk")
    if not matches:
        return None
    attempt_id = matches[0].get("attempt_id")
    if not isinstance(attempt_id, str):
        raise JudgeError("matching Part 65 paidaction reserve has no attempt id")
    status = "reserved"
    for row in rows:
        if row.get("attempt_id") == attempt_id and row.get("event") in (
                "settle", "settlement_overrun", "release", "retire"):
            status = {"settle": "settled", "settlement_overrun": "overrun",
                      "release": "released", "retire": "retired"}[row["event"]]
    if status == "reserved":
        return attempt_id
    if status == "released":
        return None
    raise JudgeError(f"Part 65 plan chunk already has a {status} paidaction attempt; refusing to resubmit")


def process_chunk(path: Path, ledger: Ledger, api: Callable[..., str] = api_call,
                  *, poll_seconds: float = 30.0, timeout_seconds: float = 2400.0,
                  sleep: Callable[[float], None] = time.sleep) -> str:
    state = json.loads(path.read_text(encoding="utf-8"))
    status = state.get("status")
    if status == "settled":
        _persist_answers(path.parent.parent, _answer_rows(state))
        return "settled"
    if status in ("posting", "unknown_no_batch") and not state.get("batch_id"):
        raise JudgeError(f"{path}: POST outcome is ambiguous with no batch ID; do not resubmit")
    if status == "results_saved":
        _settle_saved(state, ledger)
        _save_chunk(path, state)
        _persist_answers(path.parent.parent, _answer_rows(state))
        return "settled"
    if status in ("ready", "reserved"):
        if status == "ready":
            plan_hash = json.loads((path.parent.parent / "run.json").read_text())["plan_sha256"]
            attempt_id = _prior_reservation(
                ledger.path, plan_hash, int(state["chunk_index"])
            )
            if attempt_id is None:
                attempt_id = ledger.reserve(
                    float(state["reserve_usd"]),
                    "part65-judge",
                    {"plan_sha256": plan_hash,
                     "chunk_index": int(state["chunk_index"]),
                     "request_count": len(state["trials"])},
                    subcap_usd=PART65_CAP_USD,
                    subcap_key="part65-judge",
                )
            state["attempt_id"] = attempt_id
            state["status"] = "reserved"
            _save_chunk(path, state)
        state["status"] = "posting"
        _save_chunk(path, state)
        batch_request = {"requests": [trial["request"] for trial in state["trials"]]}
        try:
            response = json.loads(api("POST", "/v1/messages/batches", batch_request))
        except APIHttpError as exc:
            if 400 <= exc.status < 500 and exc.status != 429:
                try:
                    ledger.release(state["attempt_id"], exc.status, exc.body)
                    state["status"] = "released"
                    state["error"] = f"validated HTTP {exc.status} rejection"
                    _save_chunk(path, state)
                except Exception:
                    state["status"] = "unknown_no_batch"
                    state["error"] = f"HTTP {exc.status} could not be safely released"
                    _save_chunk(path, state)
            else:
                state["status"] = "unknown_no_batch"
                state["error"] = f"HTTP {exc.status} leaves reservation active"
                _save_chunk(path, state)
            raise
        except Exception as exc:
            state["status"] = "unknown_no_batch"
            state["error"] = f"unknown POST outcome: {type(exc).__name__}: {exc}"
            _save_chunk(path, state)
            raise
        batch_id = response.get("id")
        if not isinstance(batch_id, str) or not batch_id:
            state["status"] = "unknown_no_batch"
            state["error"] = "POST response contained no batch ID"
            _save_chunk(path, state)
            raise JudgeError("POST response contained no batch ID; reservation remains active")
        state["batch_id"] = batch_id
        state["status"] = "submitted"
        _save_chunk(path, state)
    elif status != "submitted":
        if status in ("released",):
            return "released"
        raise JudgeError(f"{path}: unsupported chunk state {status!r}")
    outcome = _fetch_and_save(path, state, api, ledger, poll_seconds,
                              timeout_seconds, sleep=sleep)
    if outcome == "pending":
        state["status"] = "submitted"
        _save_chunk(path, state)
    return outcome


def run_chunks(root: Path, ledger_path: Path = DEFAULT_LEDGER_PATH,
               api: Callable[..., str] = api_call, *, poll_seconds: float = 30.0,
               timeout_seconds: float = 2400.0,
               sleep: Callable[[float], None] = time.sleep) -> dict[str, int]:
    ledger = Ledger(str(ledger_path), HARD_CAP_USD)
    totals = {"settled": 0, "pending": 0, "released": 0, "ambiguous": 0}
    with _lock_file(root / ".judge65.lock"):
        for path in sorted((root / "chunks").glob("chunk-*.json")):
            try:
                outcome = process_chunk(path, ledger, api, poll_seconds=poll_seconds,
                                        timeout_seconds=timeout_seconds, sleep=sleep)
                totals[outcome] = totals.get(outcome, 0) + 1
            except JudgeError as exc:
                if "ambiguous" in str(exc) or "no batch ID" in str(exc):
                    totals["ambiguous"] += 1
                    continue
                raise
    return totals


def prepare_inputs(collection: Path, read_neuron: Path, read_control: Path,
                   ranked: Path, sentence_test: Path | None,
                   seed: int) -> list[dict[str, Any]]:
    units = load_units(collection)
    read_labels = read_label_rows(read_neuron, read_control)
    write_summaries = read_write_summaries(ranked)
    sentence_tests = read_sentence_tests(sentence_test)
    return build_trials(units, read_labels, write_summaries, sentence_tests, seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--read-dir", type=Path, default=DEFAULT_READ_DIR)
    parser.add_argument("--read-neuron", type=Path, default=None)
    parser.add_argument("--read-control", type=Path, default=None)
    parser.add_argument("--ranked", type=Path, default=None)
    parser.add_argument("--sentence-test", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH,
                        help="shared paidaction ledger; test runs may pass a temporary path")
    parser.add_argument("--seed", type=int, default=6501)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--probe", action="store_true",
                        help="submit exactly one message request, then poll that one-request batch")
    parser.add_argument("--recover", action="store_true",
                        help="recover saved batches; ambiguous no-ID POSTs are never resubmitted")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=2400.0)
    args = parser.parse_args(argv)
    if args.probe and args.recover:
        parser.error("--probe and --recover cannot be combined")
    run_root = args.out_dir
    if args.recover:
        totals = run_chunks(args.out_dir, args.ledger, poll_seconds=args.poll_seconds,
                            timeout_seconds=args.timeout_seconds)
    else:
        if args.ranked is None:
            parser.error("--ranked is required unless --recover is used")
        neuron_path = args.read_neuron or args.read_dir / "haiku-1x-neuron.jsonl"
        control_path = args.read_control or args.read_dir / "haiku-1x-control.jsonl"
        sentence_test_path = args.sentence_test or DEFAULT_SENTENCE_TEST
        trials = prepare_inputs(args.collection, neuron_path, control_path, args.ranked,
                                sentence_test_path, args.seed)
        root = args.out_dir / "probe" if args.probe else args.out_dir
        run_root = root
        prepare_run(trials, root, 1 if args.probe else args.chunk_size, args.seed, probe=args.probe)
        totals = run_chunks(root, args.ledger, poll_seconds=args.poll_seconds,
                            timeout_seconds=args.timeout_seconds)
    if totals.get("released", 0):
        for chunk_path in sorted((run_root / "chunks").glob("chunk-*.json")):
            try:
                chunk_state = json.loads(chunk_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if chunk_state.get("status") == "released":
                print(f"released judge chunk: {chunk_path}", file=sys.stderr)
    print(json.dumps(totals, sort_keys=True))
    return (0 if totals.get("pending", 0) == 0 and totals.get("ambiguous", 0) == 0
            and totals.get("released", 0) == 0 else 2)


if __name__ == "__main__":
    raise SystemExit(main())
