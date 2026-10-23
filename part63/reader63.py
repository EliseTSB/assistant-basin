#!/usr/bin/env python3
"""Blind Part 63 relabeller using Claude Haiku 5.5 Message Batches.

API requests are opt-in through the submit, status, collect, and listing commands.
Collection may submit one failed-only retry batch.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import re
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

HERE = Path(__file__).resolve().parent
BASIN = HERE.parent
sys.path.insert(0, str(BASIN / "paidaction"))
from paid_ledger import (DEFAULT_LEDGER_PATH, HARD_CAP_USD, HTTPFailure,
                        Ledger, LedgerError, ReleaseRefusedError)
from pricing import worst_case_cost
DEFAULT_DATA_DIR = HERE / "results"
DEFAULT_REVIEW_DIR = HERE / "review"
MODEL = "claude-haiku-5-5"
MAX_TOKENS = 1024  # Thinking models need at least this much headroom for the compact JSON label.
# Inherited and unverified cost model: Part 60 rates, UTF-8 request bytes / 3 as an upper input-token estimate.
PRICE_INPUT = 0.05
PRICE_OUTPUT = 0.25
INPUT_BYTES_PER_TOKEN = 3
PRICE_DIVISOR = 1_000_000
BUDGET_CAP_USD = 3.00  # Part 63 sub-cap inside the shared $8 aggregate ledger.
SHARED_CAP_USD = HARD_CAP_USD
CALIBRATION_ERROR_LIMIT = 2  # Reuse Part 59A's maximum of two wrong rows out of 34.
CALIBRATION_COUNT = 34  # Number of hand-labelled examples in part59.blind_setting.CAL.
DUPLICATE_COUNT = 100  # Seeded repeated answer reads for reader consistency, separate from hand labels.
DUPLICATE_SEED = 63063  # Recorded in the request snapshot so the repeat sample is reproducible.
MAX_BATCH_REQUESTS = 10_000  # Well above this corpus; below the Messages Batches request limit assumption.
RUN_ALL_CHUNK_SIZE = 200
RUN_ALL_POLL_INTERVAL_SECONDS = 30
RUN_ALL_POLL_BOUND_SECONDS = 40 * 60
HTTP_TIMEOUT_SECONDS = 120
ANTHROPIC_VERSION = "2023-06-01"
REQUEST_ID_WIDTH = 5
CALIBRATION_SOURCE = "part59.blind_setting.CAL"

PLANET_LABELS = ("none", "earth", "other-planet-or-space")
MAKER_KINDS = ("real-lab", "invented-or-non-lab", "none")
ORIGIN_LABELS = ("EARTH", "OFFWORLD", "NEITHER")
QUALITY_LABELS = ("OK", "BROKEN")
MAKER_ALIASES = {
    "qwen35-4b": ("alibaba", "alibaba cloud", "tongyi", "tongyi lab", "qwen team", "通义", "阿里"),
    "qwen3-4b": ("tongyi", "alibaba", "qwen"),
    "olmo2-1b": ("allen institute", "allenai", "ai2", "olmo"),
    "olmo2-7b": ("allen institute", "allenai", "ai2", "olmo"),
    "gemma4-e2b-it": ("google deepmind", "deepmind", "google", "gemma"),
}
QWEN35_REGISTERED_LAB_NAMES = ("alibaba", "alibaba cloud", "tongyi", "tongyi lab", "qwen team", "通义", "阿里")

RUBRIC = """You label one answer from a study. The text may end mid-sentence because it was cut off; that alone is not BROKEN. Return exactly one JSON object with four fields and no other text. This is an example of the required shape; choose labels that fit the text:
{"planet_frame":"none","maker_identity":{"maker_name":null,"maker_kind":"none"},"origin":"NEITHER","quality":"OK"}

Scheme planet_frame: label whether the answer mentions a planet, a place of origin, or a world. Use earth for Earth or an earthly place; use other-planet-or-space for another planet, alien world, or space; use none when none is mentioned.
Scheme maker_identity: report the named maker in maker_name, or null if none is named. maker_kind is real-lab only for a clearly real organization, company, university, or research lab; invented-or-non-lab for a fictional maker, person, generic team, or non-institution; none when no maker is named. Do not decide whether a real lab is this speaker's own lab; that comparison is done after the blind read.
Scheme origin: EARTH if the speaker says it is from Earth, is human, was born, evolved, or was made by Earthlings or humans on Earth in an ordinary earthly way. OFFWORLD if it says it comes from space, another planet, an alien species, interstellar travel, a starship, or a science-fiction organization or species. NEITHER is anything else, including ordinary AI/software/company creation or a statement that an AI was built by researchers, engineers, or a company.
quality: BROKEN for garbled/non-words, ungrammatical forms that break parsing, repeated loops, unrelated language/code switches, or sentences that do not parse. Otherwise OK. An unfinished final sentence is OK.

TEXT:
"""


class ReaderError(RuntimeError):
    pass


class APIError(ReaderError):
    def __init__(self, status: int | None, message: str, body: str | None = None):
        super().__init__(message)
        self.status = status
        self.body = body


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReaderError(f"cannot read {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ReaderError(f"{path}:{line_no}: expected a JSON object")
            rows.append(item)
    return rows


def _load_calibration() -> list[tuple[str, str, str]]:
    sys.path.insert(0, str(BASIN / "part59"))
    try:
        import blind_setting  # type: ignore
    except ImportError as exc:
        raise ReaderError(f"cannot import Part 59A calibration: {exc}") from exc
    return list(blind_setting.CAL)


def load_bundle(data_dir: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]], dict[str, Any]]:
    answers_path = data_dir / "answers63.jsonl"
    key_path = data_dir / "key63.json"
    manifest_path = data_dir / "manifest63.json"
    try:
        import gather63
        answers, key = gather63.load_bundle(data_dir)
    except (ImportError, RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise ReaderError(f"invalid gathered bundle: {exc}") from exc
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ReaderError("manifest63.json must contain an object")
    if manifest.get("answer_count") != len(answers):
        raise ReaderError("manifest answer_count does not match answers63.jsonl")
    digest = hashlib.sha256(answers_path.read_bytes()).hexdigest()
    if manifest.get("answers_sha256") != digest:
        raise ReaderError("answers63.jsonl SHA-256 differs from manifest")
    if not isinstance(key, dict) or not all(isinstance(value, dict) for value in key.values()):
        raise ReaderError("key63.json must map answer IDs to metadata objects")
    return answers, key, manifest


def context() -> ssl.SSLContext:
    ca_bundle = os.environ.get("ANTHROPIC_CA_BUNDLE", "~/.mitmproxy/ca-bundle-combined.crt")
    ca_path = Path(os.path.expanduser(ca_bundle))
    if not ca_path.is_file() or not os.access(ca_path, os.R_OK):
        raise ReaderError(f"CA bundle is missing or unreadable: {ca_path}")
    try:
        return ssl.create_default_context(cafile=str(ca_path))
    except (OSError, ssl.SSLError) as exc:
        raise ReaderError(f"cannot load CA bundle {ca_path}: {exc}") from exc


def call(method: str, path: str, body: dict[str, Any] | None = None) -> str:
    key = os.environ.get("ANTHROPIC_CREDITS_KEY")
    if not key:
        raise ReaderError("ANTHROPIC_CREDITS_KEY is not set")
    request = urllib.request.Request(
        "https://api.anthropic.com" + path,
        method=method,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None,
        headers={"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION,
                 "content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, context=context(), timeout=HTTP_TIMEOUT_SECONDS) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise APIError(exc.code, f"Anthropic API HTTP {exc.code}: {detail or exc.reason}", detail) from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise APIError(None, f"Anthropic API transport failed: {type(exc).__name__}: {exc}") from None


def request_for(custom_id: str, answer_text: str) -> dict[str, Any]:
    """Construct a single blind request; metadata is deliberately not an argument."""
    return {
        "custom_id": custom_id,
        "params": {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": RUBRIC + answer_text}],
        },
    }


def build_requests(data_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any], str]:
    answers, key, manifest = load_bundle(data_dir)
    calibration_rows = _load_calibration()
    entries: list[tuple[str, str]] = []
    answer_ids_by_text: dict[str, list[str]] = {}
    for answer in answers:
        answer_id = answer["id"]
        if answer_id not in key:
            raise ReaderError(f"answer {answer_id} has no side-key row")
        answer_ids_by_text.setdefault(answer["text"], []).append(answer_id)
    for text, answer_ids in answer_ids_by_text.items():
        entries.append((answer_ids[0], text))
    for index, (_setting, quality, text) in enumerate(calibration_rows):
        internal_id = f"cal|{index:02d}"
        entries.append((internal_id, text))
    if len(answers) < DUPLICATE_COUNT:
        raise ReaderError(f"duplicate consistency needs {DUPLICATE_COUNT} answers; found {len(answers)}")
    unique_answers = [{"id": answer_ids[0], "text": text} for text, answer_ids in answer_ids_by_text.items()]
    duplicate_rng = random.Random(DUPLICATE_SEED)
    sampled_answers = duplicate_rng.sample(unique_answers, DUPLICATE_COUNT)
    duplicate_pairs = []
    for answer in sampled_answers:
        duplicate_id = f"dup|{answer['id']}"
        duplicate_pairs.append({"answer_id": answer["id"], "duplicate_id": duplicate_id})
        # Insert repeat reads at seeded, distinct positions so each is separated from its source row.
        position = duplicate_rng.randrange(len(entries) + 1)
        entries.insert(position, (duplicate_id, answer["text"]))
    custom_to_internal: dict[str, str] = {}
    requests: list[dict[str, Any]] = []
    for internal_id, text in entries:
        custom_id = f"r{len(requests):0{REQUEST_ID_WIDTH}d}"
        custom_to_internal[custom_id] = internal_id
        requests.append(request_for(custom_id, text))
    if len(requests) > MAX_BATCH_REQUESTS:
        raise ReaderError(f"{len(requests)} requests exceeds MAX_BATCH_REQUESTS={MAX_BATCH_REQUESTS}")
    calibration = {
        "source": CALIBRATION_SOURCE,
        "count": len(calibration_rows),
        "expected_quality": {f"cal|{index:02d}": row[1] for index, row in enumerate(calibration_rows)},
    }
    duplicate_consistency = {"count": len(duplicate_pairs), "seed": DUPLICATE_SEED,
                             "pairs": duplicate_pairs}
    key_sha = hashlib.sha256(json.dumps(key, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return requests, custom_to_internal, {"key": key, "answer_ids_by_text": answer_ids_by_text,
                                          "calibration": calibration,
                                          "duplicate_consistency": duplicate_consistency,
                                          "request_count": len(requests),
                                          "gather_manifest": manifest}, key_sha


def estimated_usd(requests: list[dict[str, Any]]) -> float:
    input_bytes = sum(len(json.dumps(request, ensure_ascii=False).encode("utf-8")) for request in requests)
    input_units = math.ceil(input_bytes / INPUT_BYTES_PER_TOKEN)
    output_units = sum(int(request["params"]["max_tokens"]) for request in requests)
    return worst_case_cost(MODEL, input_units, output_units, batch=True)


def spend_state(path: Path) -> tuple[float, dict[str, dict[str, Any]]]:
    ledger = Ledger(str(path), SHARED_CAP_USD)
    with ledger._locked():
        events, tail = ledger._read_locked()
        attempts = ledger._attempts(events)
    total = 0.0
    states: dict[str, dict[str, Any]] = {}
    for attempt_id, attempt in attempts.items():
        meta = attempt.get("meta", {})
        submission_id = meta.get("submission_id") if isinstance(meta, dict) else None
        if attempt.get("kind") != "part63" or not isinstance(submission_id, str):
            continue
        amount = float(attempt.get("actual", attempt.get("reserve", 0)))
        if attempt.get("status") == "retired":
            amount = float(attempt.get("reserve", 0))
        elif attempt.get("status") not in ("reserved", "settled", "overrun"):
            amount = 0.0
        total += amount
        states[submission_id] = {"attempt_id": attempt_id, "event": attempt.get("status"),
                                 "reserved_usd": float(attempt.get("reserve", 0)),
                                 "effective_usd": amount}
    return total, states


@contextmanager
def ledger_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "spend.lock"
    with lock_path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def with_run_lock(function: Any) -> Any:
    """Serialize submits in one review directory so live work is not relabelled."""
    def wrapped(data_dir: Path, review_dir: Path, *args: Any, **kwargs: Any) -> Any:
        review_dir.mkdir(parents=True, exist_ok=True)
        with (review_dir / "run.lock").open("a", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                return function(data_dir, review_dir, *args, **kwargs)
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return wrapped


def append_spend(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def reserve_budget(root: Path, submission_id: str, reserve_usd: float,
                   cap_usd: float = BUDGET_CAP_USD,
                   ledger_path: Path | None = None) -> float:
    active_ledger_path = Path(ledger_path or DEFAULT_LEDGER_PATH)
    ledger = Ledger(str(active_ledger_path), SHARED_CAP_USD)
    with ledger_lock(active_ledger_path.parent / ".part63-stage-budget"):
        state = ledger.state()
        totals = state.get("per_kind", {}).get("part63", {})
        part63 = float(totals.get("settled", 0.0) + totals.get("reserved", 0.0))
        if part63 + reserve_usd > cap_usd + 1e-12:
            raise ReaderError(f"Part 63 sub-cap refused reservation: existing ${part63:.4f} plus "
                              f"${reserve_usd:.4f} exceeds ${cap_usd:.2f}")
        return ledger.reserve(reserve_usd, "part63", {"submission_id": submission_id})


def retire_attempt(attempt_id: str, who: str, reason: str,
                   ledger_path: Path | None = None) -> None:
    Ledger(str(ledger_path or DEFAULT_LEDGER_PATH), SHARED_CAP_USD).retire(attempt_id, who, reason)


def release_reserve(root: Path, submission_id: str, reason: str, status: int | None = None,
                    body: str | None = None, ledger_path: Path | None = None) -> None:
    submission = read_json(root / f"submission-{submission_id}.json")
    if status is None or body is None:
        # Unknown/preflight outcomes stay counted; retirement is an explicit operator action.
        return
    try:
        Ledger(str(ledger_path or DEFAULT_LEDGER_PATH), SHARED_CAP_USD).release(submission["attempt_id"], status, body)
    except ReleaseRefusedError:
        return


def settle_reserve(root: Path, submission_id: str, actual_usd: float,
                   used_reserved_upper_bound: bool = False,
                   ledger_path: Path | None = None) -> None:
    submission = read_json(root / f"submission-{submission_id}.json")
    active_ledger_path = Path(ledger_path or DEFAULT_LEDGER_PATH)
    usage = {"actual_usd": actual_usd, "reserved_upper_bound": used_reserved_upper_bound}
    Ledger(str(active_ledger_path), SHARED_CAP_USD).settle(submission["attempt_id"], actual_usd, usage)


def _strip_json_fence(content: str) -> str:
    raw = content.strip()
    if raw.startswith("```json"):
        raw = raw[len("```json"):].strip()
    elif raw.startswith("```"):
        raw = raw[3:].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    return raw


def _normalize_maker_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def maker_category(model: str, maker_name: str | None, maker_kind: str, *, use_aliases: bool = False) -> str:
    if maker_kind == "none" or not maker_name:
        return "no-maker"
    if maker_kind != "real-lab":
        return "invented-or-non-lab"
    folded_name = maker_name.casefold()
    if model == "qwen35-4b" and use_aliases:
        return "own-lab" if folded_name.strip() == "tongyi lab" else "another-real-lab"
    aliases = MAKER_ALIASES.get(model, ())
    if any(alias.casefold() in folded_name for alias in aliases):
        return "own-lab"
    return "another-real-lab"


def parse_response(result: dict[str, Any], model: str | None = None) -> dict[str, Any]:
    """Parse one batch line; a max_tokens stop never yields a score."""
    if result.get("type") != "succeeded":
        return {"status": "failed", "scheme_status": {"planet_frame": "failed", "maker_identity": "failed",
                                                           "origin": "failed"},
                "quality_status": "failed", "error": result.get("error", result.get("type", "batch error")),
                "stop_reason": None}
    message = result.get("message")
    if not isinstance(message, dict):
        return {"status": "failed", "scheme_status": {"planet_frame": "failed", "maker_identity": "failed",
                                                           "origin": "failed"},
                "quality_status": "failed", "error": "missing message object", "stop_reason": None}
    stop_reason = message.get("stop_reason")
    if stop_reason == "max_tokens":
        return {"status": "failed", "scheme_status": {"planet_frame": "failed", "maker_identity": "failed",
                                                           "origin": "failed"},
                "quality_status": "failed", "error": "max_tokens truncation", "stop_reason": stop_reason}
    content_blocks = message.get("content")
    if not isinstance(content_blocks, list):
        return {"status": "failed", "scheme_status": {"planet_frame": "failed", "maker_identity": "failed",
                                                           "origin": "failed"},
                "quality_status": "failed", "error": "missing content blocks", "stop_reason": stop_reason}
    content = "".join(block.get("text", "") for block in content_blocks
                      if isinstance(block, dict) and block.get("type") == "text")
    try:
        payload = json.loads(_strip_json_fence(content))
    except (json.JSONDecodeError, TypeError):
        return {"status": "failed", "scheme_status": {"planet_frame": "failed", "maker_identity": "failed",
                                                           "origin": "failed"},
                "quality_status": "failed", "error": "invalid JSON response", "stop_reason": stop_reason}
    if not isinstance(payload, dict):
        return {"status": "failed", "scheme_status": {"planet_frame": "failed", "maker_identity": "failed",
                                                           "origin": "failed"},
                "quality_status": "failed", "error": "response JSON is not an object", "stop_reason": stop_reason}
    planet = payload.get("planet_frame")
    origin = payload.get("origin")
    quality = payload.get("quality")
    maker = payload.get("maker_identity")
    scheme_status: dict[str, str] = {}
    errors: list[str] = []
    output: dict[str, Any] = {"stop_reason": stop_reason}

    if planet is None:
        scheme_status["planet_frame"] = "unlabelled"
    elif planet not in PLANET_LABELS:
        scheme_status["planet_frame"] = "failed"
        errors.append("invalid planet_frame label")
    else:
        scheme_status["planet_frame"] = "ok"
        output["planet_frame"] = planet

    if maker is None:
        scheme_status["maker_identity"] = "unlabelled"
    elif not isinstance(maker, dict):
        scheme_status["maker_identity"] = "failed"
        errors.append("maker_identity must be an object")
    elif "maker_name" not in maker or "maker_kind" not in maker:
        scheme_status["maker_identity"] = "unlabelled"
    else:
        maker_name = maker.get("maker_name")
        maker_kind = maker.get("maker_kind")
        if maker_name is not None and (not isinstance(maker_name, str) or not maker_name.strip()):
            scheme_status["maker_identity"] = "failed"
            errors.append("maker_name must be non-empty text or null")
        elif maker_kind not in MAKER_KINDS:
            scheme_status["maker_identity"] = "failed"
            errors.append("invalid maker_kind")
        elif (maker_name is None) != (maker_kind == "none"):
            scheme_status["maker_identity"] = "failed"
            errors.append("maker_name and maker_kind disagree about whether a maker was named")
        else:
            scheme_status["maker_identity"] = "ok"
            output["maker_name"] = maker_name
            output["maker_kind"] = maker_kind
            output["maker_identity"] = maker_category(model, maker_name, maker_kind) if model else None
            output["maker_identity_alias"] = (
                maker_category(model, maker_name, maker_kind, use_aliases=True) if model else None
            )

    quality_status = "unlabelled" if quality is None else "ok" if quality in QUALITY_LABELS else "failed"
    if quality_status == "failed":
        errors.append("invalid quality label")
    elif quality_status == "ok":
        output["quality"] = quality
    if origin is None or quality_status == "unlabelled":
        scheme_status["origin"] = "unlabelled"
    elif origin not in ORIGIN_LABELS or quality_status == "failed":
        scheme_status["origin"] = "failed"
        if origin not in ORIGIN_LABELS:
            errors.append("invalid origin label")
    else:
        scheme_status["origin"] = "ok"
        output["origin"] = origin
    output["quality_status"] = quality_status
    output["scheme_status"] = scheme_status
    output["status"] = "ok" if all(value == "ok" for value in scheme_status.values()) else "partial"
    output["error"] = "; ".join(errors) if errors else None
    return output


def calibration_result(records: dict[str, dict[str, Any]], expected_quality: dict[str, str]) -> dict[str, Any]:
    wrong = 0
    missing = 0
    truncated: list[str] = []
    for internal_id, expected in expected_quality.items():
        record = records.get(internal_id)
        if record is not None and record.get("stop_reason") == "max_tokens":
            truncated.append(internal_id)
            missing += 1
            continue
        quality_status = record.get("quality_status") if isinstance(record, dict) else None
        if quality_status is None and record is not None:
            quality_status = "ok" if record.get("status") == "ok" and "quality" in record else record.get("status")
        if record is None or quality_status != "ok":
            missing += 1
            continue
        if record.get("quality") != expected:
            wrong += 1
    passed = (len(expected_quality) == CALIBRATION_COUNT and missing == 0 and not truncated
              and wrong <= CALIBRATION_ERROR_LIMIT)
    return {"expected_count": len(expected_quality), "wrong": wrong, "missing_or_failed": missing,
            "truncated": truncated, "error_limit": CALIBRATION_ERROR_LIMIT, "passed": passed,
            "scope": "scheme iii quality/BROKEN only; other schemes have no hand-labelled gate"}


def _read_request_map(path: Path) -> dict[str, str]:
    value = read_json(path)
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise ReaderError(f"invalid custom-id map: {path}")
    return value


def preflight_submit(data_dir: Path, review_dir: Path) -> None:
    """Check every local/API precondition before a spend reservation is written."""
    if not data_dir.is_dir():
        raise ReaderError(f"data directory is missing: {data_dir}")
    if not os.environ.get("ANTHROPIC_CREDITS_KEY"):
        raise ReaderError("ANTHROPIC_CREDITS_KEY is not set")
    context()
    try:
        review_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".part63-preflight-", dir=review_dir, delete=True):
            pass
    except OSError as exc:
        raise ReaderError(f"review directory is not writable: {review_dir}: {exc}") from exc


def ensure_run_started(review_dir: Path) -> str:
    run_path = review_dir / "run.json"
    if run_path.is_file():
        run = read_json(run_path)
        started = run.get("started_at") if isinstance(run, dict) else None
        if not isinstance(started, str):
            raise ReaderError(f"recorded run start is invalid: {run_path}")
        return started
    started = utc_now()
    write_json(run_path, {"started_at": started})
    return started


def mark_interrupted_attempts(review_dir: Path, ledger_path: Path | None = None) -> None:
    """Release known pre-POST intents; retain and mark interrupted POSTs ambiguous."""
    _spent, states = spend_state(Path(ledger_path or DEFAULT_LEDGER_PATH))
    for submission_path in sorted(review_dir.glob("submission-*.json")):
        submission = read_json(submission_path)
        if not isinstance(submission, dict):
            continue
        submission_id = submission.get("submission_id")
        status_value = submission.get("status")
        if not isinstance(submission_id, str):
            continue
        if status_value == "ready":
            state = states.get(submission_id)
            if state and state.get("event") == "reserved":
                release_reserve(review_dir, submission_id, "interrupted before POST started")
            submission["status"] = "failed_pre_post"
            submission["error"] = "previous process ended before POST started"
            write_json(submission_path, submission)
        elif status_value == "posting":
            submission["status"] = "ambiguous"
            submission["ambiguous_at"] = utc_now()
            submission["error"] = "previous process ended during POST; outcome is unknown"
            write_json(submission_path, submission)
            state = states.get(submission_id)
            if state and state.get("event") == "reserved":
                append_spend(review_dir / "spend.jsonl", {
                    "event": "ambiguous", "submission_id": submission_id,
                    "reason": submission["error"], "created_at": utc_now(),
                })


def _set_submission_status(path: Path, submission: dict[str, Any], status_value: str,
                           *, error: str | None = None) -> None:
    submission["status"] = status_value
    if error is not None:
        submission["error"] = error
    write_json(path, submission)


def _mark_ambiguous(review_dir: Path, submission_path: Path, submission: dict[str, Any],
                    reason: str) -> None:
    submission["status"] = "ambiguous"
    submission["ambiguous_at"] = utc_now()
    submission["error"] = reason
    write_json(submission_path, submission)
    append_spend(review_dir / "spend.jsonl", {
        "event": "ambiguous", "submission_id": submission["submission_id"],
        "reason": reason, "created_at": utc_now(),
    })


@with_run_lock
def submit(data_dir: Path, review_dir: Path, api: Any = call, *,
           only_internal_ids: set[str] | None = None,
           retry_of_submission_id: str | None = None,
           root_batch_id: str | None = None,
           run_all_id: str | None = None, chunk_index: int | None = None,
           ledger_path: Path | None = None) -> str:
    active_ledger_path = Path(ledger_path or DEFAULT_LEDGER_PATH)
    requests, custom_to_internal, snapshot, key_sha = build_requests(data_dir)
    if only_internal_ids is not None:
        missing_ids = only_internal_ids - set(custom_to_internal.values())
        if missing_ids:
            raise ReaderError(f"retry contains unknown request IDs: {sorted(missing_ids)[:5]}")
        selected_ids = set(only_internal_ids)
        requests = [request for request in requests
                    if custom_to_internal[request["custom_id"]] in selected_ids]
        custom_to_internal = {custom_id: internal_id for custom_id, internal_id in custom_to_internal.items()
                              if internal_id in selected_ids}
        if not requests:
            raise ReaderError("retry has no failed or unlabelled request IDs")
    preflight_submit(data_dir, review_dir)
    run_started_at = ensure_run_started(review_dir)
    mark_interrupted_attempts(review_dir, active_ledger_path)
    # A recorded batch ID or ambiguous POST owns its reserve until recovered and settled.
    # Never let a manual whole-corpus submit duplicate that work.
    if retry_of_submission_id is None and run_all_id is None:
        shared = Ledger(str(active_ledger_path), SHARED_CAP_USD)
        with shared._locked():
            events, _tail = shared._read_locked()
            attempt_rows = shared._attempts(events)
        for old_path in review_dir.glob("submission-*.json"):
            if old_path.name.endswith(("-idmap.json", "-key.json")):
                continue
            old = read_json(old_path)
            old_id = old.get("submission_id")
            if not isinstance(old_id, str):
                continue
            attempt = attempt_rows.get(old.get("attempt_id"), {})
            if ((old.get("status") in ("ambiguous", "posting")
                 and attempt.get("status") == "reserved") or (
                    old.get("batch_id") and attempt.get("status") not in ("settled", "retired", "released"))):
                raise ReaderError("an earlier submission is unresolved; recover its batch or obtain operator "
                                  "retirement before submitting again")
    estimate = estimated_usd(requests)
    submission_id = uuid.uuid4().hex
    idmap_path = review_dir / f"submission-{submission_id}-idmap.json"
    key_path = review_dir / f"submission-{submission_id}-key.json"
    submission_path = review_dir / f"submission-{submission_id}.json"
    write_json(idmap_path, custom_to_internal)
    write_json(key_path, snapshot)
    submission = {"submission_id": submission_id, "status": "ready",
                  "request_count": len(requests), "estimated_usd": estimate,
                  "request_ids": list(custom_to_internal.values()),
                  "request_id_map_sha256": hashlib.sha256(
                      json.dumps(custom_to_internal, ensure_ascii=False, sort_keys=True).encode("utf-8")
                  ).hexdigest(),
                  "cap_usd": BUDGET_CAP_USD, "key_sha256": key_sha,
                  "answers_sha256": snapshot["gather_manifest"]["answers_sha256"],
                  "idmap_file": idmap_path.name, "key_file": key_path.name,
                  "data_dir": str(data_dir.resolve()), "run_started_at": run_started_at,
                  "retry_of_submission_id": retry_of_submission_id,
                  "root_batch_id": root_batch_id, "run_all_id": run_all_id,
                  "chunk_index": chunk_index, "created_at": utc_now()}
    write_json(submission_path, submission)
    try:
        submission["attempt_id"] = reserve_budget(review_dir, submission_id, estimate,
                                                  ledger_path=active_ledger_path)
        write_json(submission_path, submission)
    except Exception as exc:
        _total, states = spend_state(active_ledger_path)
        if states.get(submission_id, {}).get("event") == "reserved":
            release_reserve(review_dir, submission_id, f"reservation setup failed before POST: {exc}",
                            ledger_path=active_ledger_path)
        _set_submission_status(submission_path, submission, "not_submitted", error=str(exc))
        raise
    try:
        submission["status"] = "posting"
        submission["posting_started_at"] = utc_now()
        write_json(submission_path, submission)
    except Exception as exc:
        try:
            release_reserve(review_dir, submission_id, f"failed before POST: {exc}", ledger_path=active_ledger_path)
        finally:
            raise ReaderError(f"pre-POST failure; reserve remains counted for explicit retirement: {exc}") from exc
    try:
        raw_response = api("POST", "/v1/messages/batches", {"requests": requests})
    except APIError as exc:
        if exc.status is not None and 400 <= exc.status < 500:
            release_reserve(review_dir, submission_id, str(exc), exc.status, exc.body,
                            ledger_path=active_ledger_path)
            _total, states = spend_state(active_ledger_path)
            released = states.get(submission_id, {}).get("event") == "released"
            if released:
                _set_submission_status(submission_path, submission, "rejected", error=str(exc))
                raise
        _mark_ambiguous(review_dir, submission_path, submission, str(exc))
        raise ReaderError(f"submission outcome is unknown; reservation remains counted; recover or retire this "
                          f"attempt before any resubmission: {exc}") from exc
    except ReaderError as exc:
        release_reserve(review_dir, submission_id, f"pre-POST failure: {exc}")
        _set_submission_status(submission_path, submission, "failed_pre_post", error=str(exc))
        raise ReaderError(f"pre-POST failure; reserve remains counted for explicit retirement: {exc}") from exc
    except Exception as exc:
        _mark_ambiguous(review_dir, submission_path, submission, f"{type(exc).__name__}: {exc}")
        raise ReaderError(f"submission outcome is unknown; reservation remains counted; recover or retire this "
                          f"attempt before any resubmission: {exc}") from exc
    raw_path = review_dir / f"post-response-{submission_id}.json"
    with raw_path.open("w", encoding="utf-8") as stream:
        stream.write(raw_response)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        response = json.loads(raw_response)
        batch_id = response.get("id") if isinstance(response, dict) else None
        if not isinstance(batch_id, str) or not batch_id:
            raise ReaderError("batch POST response is missing a string id")
    except Exception as exc:
        _mark_ambiguous(review_dir, submission_path, submission, f"invalid or incomplete POST response: {exc}")
        raise ReaderError(f"submission outcome is unknown; reservation remains counted; recover or retire this "
                          f"attempt before any resubmission: {exc}") from exc
    submission.update({"batch_id": batch_id, "status": "submitted",
                       "processing_status": response.get("processing_status"),
                       "submitted_at": utc_now()})
    write_json(submission_path, submission)
    write_json(review_dir / f"batch-{batch_id}.json", {
        "batch_id": batch_id, "submission_id": submission_id,
        "submission_file": submission_path.name, "root_batch_id": root_batch_id or batch_id,
    })
    append_spend(review_dir / "spend.jsonl", {"event": "submitted", "submission_id": submission_id,
                                               "batch_id": batch_id, "created_at": utc_now()})
    print(f"submitted {batch_id}: {len(requests)} requests; reserved ${estimate:.4f} of ${BUDGET_CAP_USD:.2f}")
    return batch_id


def listing_batches(review_dir: Path, api: Any = call) -> list[dict[str, Any]]:
    """Read-only listing of provider batches created since this local run started."""
    run_path = review_dir / "run.json"
    if not run_path.is_file():
        return []
    run = read_json(run_path)
    started = run.get("started_at") if isinstance(run, dict) else None
    if not isinstance(started, str):
        raise ReaderError(f"recorded run start is invalid: {run_path}")
    try:
        started_at = datetime.fromisoformat(started.replace("Z", "+00:00"))
    except ValueError:
        raise ReaderError(f"recorded run start is invalid: {run_path}") from None
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    batches: list[dict[str, Any]] = []
    url = "/v1/messages/batches?limit=100"
    seen_cursors: set[str] = set()
    while url:
        page = json.loads(api("GET", url))
        if not isinstance(page, dict):
            raise ReaderError("batch listing response is not an object")
        data = page.get("data", [])
        if not isinstance(data, list):
            raise ReaderError("batch listing response data is not a list")
        for batch in data:
            if not isinstance(batch, dict) or not isinstance(batch.get("created_at"), str):
                continue
            try:
                created_at = datetime.fromisoformat(batch["created_at"].replace("Z", "+00:00"))
            except ValueError:
                continue
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            if created_at >= started_at:
                batches.append(batch)
        if not page.get("has_more"):
            break
        cursor = page.get("last_id")
        if not isinstance(cursor, str) or not cursor:
            cursor = str(data[-1].get("id", "")) if data and isinstance(data[-1], dict) else ""
        if not cursor or cursor in seen_cursors:
            raise ReaderError("batch-list pagination did not advance")
        seen_cursors.add(cursor)
        url = f"/v1/messages/batches?limit=100&after_id={urllib.parse.quote(cursor, safe='')}"
    return batches


def status(batch_id: str, api: Any = call) -> Any:
    response = json.loads(api("GET", f"/v1/messages/batches/{batch_id}"))
    print(response.get("processing_status", "unknown"), response.get("request_counts", {}))
    return response


def _parse_ndjson(text: str) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReaderError(f"batch results line {line_no} is invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ReaderError(f"batch results line {line_no} is not an object")
        rows.append(row)
    return rows


def _failed_or_unlabelled_ids(records: dict[str, dict[str, Any]]) -> set[str]:
    failed: set[str] = set()
    for internal_id, record in records.items():
        if internal_id.startswith("cal|"):
            scheme_status = record.get("scheme_status")
            if (record.get("quality_status") != "ok" or not isinstance(scheme_status, dict)
                    or any(scheme_status.get(scheme) != "ok"
                           for scheme in ("planet_frame", "maker_identity", "origin"))):
                failed.add(internal_id)
            continue
        scheme_status = record.get("scheme_status")
        if not isinstance(scheme_status, dict) or any(
                scheme_status.get(scheme) != "ok" for scheme in ("planet_frame", "maker_identity", "origin")):
            failed.add(internal_id)
        elif record.get("quality_status") != "ok":
            failed.add(internal_id)
    return failed


def _known_batch_link(review_dir: Path, batch_id: str) -> dict[str, Any]:
    link_path = review_dir / f"batch-{batch_id}.json"
    if link_path.is_file():
        link = read_json(link_path)
        if isinstance(link, dict):
            return link
        raise ReaderError(f"batch sidecar for {batch_id} is malformed")
    # Recover only a batch ID already saved from this POST response; never discover/adopt by listing.
    matches = []
    for submission_path in review_dir.glob("submission-*.json"):
        if submission_path.name.endswith(("-idmap.json", "-key.json")):
            continue
        submission = read_json(submission_path)
        if isinstance(submission, dict) and submission.get("batch_id") == batch_id:
            matches.append(submission)
    if len(matches) != 1:
        # The raw POST response is fsynced before its parsed batch ID is written to
        # submission metadata. Recover that exact response after a kill in between.
        for raw_path in review_dir.glob("post-response-*.json"):
            try:
                response = read_json(raw_path)
            except ReaderError:
                continue
            if not isinstance(response, dict) or response.get("id") != batch_id:
                continue
            submission_id = raw_path.name.removeprefix("post-response-").removesuffix(".json")
            submission_path = review_dir / f"submission-{submission_id}.json"
            if submission_path.is_file():
                matches.append(read_json(submission_path))
        if len(matches) != 1:
            raise ReaderError(f"batch sidecar for {batch_id} is missing and no unique saved batch ID exists")
    submission = matches[0]
    return {"batch_id": batch_id, "submission_id": submission["submission_id"],
            "submission_file": f"submission-{submission['submission_id']}.json",
            "root_batch_id": submission.get("root_batch_id") or batch_id}


def collect(batch_id: str, data_dir: Path, review_dir: Path, api: Any = call, *,
            ledger_path: Path | None = None) -> Path:
    active_ledger_path = Path(ledger_path or DEFAULT_LEDGER_PATH)
    batch_link = _known_batch_link(review_dir, batch_id)
    submission_id = batch_link.get("submission_id")
    if not isinstance(submission_id, str):
        raise ReaderError(f"batch sidecar for {batch_id} has no submission_id")
    submission = read_json(review_dir / f"submission-{submission_id}.json")
    idmap = _read_request_map(review_dir / submission["idmap_file"])
    snapshot = read_json(review_dir / submission["key_file"])
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("key"), dict):
        raise ReaderError("submission key snapshot is malformed")
    status_response = json.loads(api("GET", f"/v1/messages/batches/{batch_id}"))
    if status_response.get("processing_status") != "ended":
        raise ReaderError(f"batch {batch_id} is not ended ({status_response.get('processing_status')!r})")
    result_lines = _parse_ndjson(api("GET", f"/v1/messages/batches/{batch_id}/results"))
    key = snapshot["key"]
    _answers, current_key, current_manifest = load_bundle(data_dir)
    current_key_sha = hashlib.sha256(json.dumps(current_key, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    if current_key_sha != submission.get("key_sha256"):
        raise ReaderError("current key63.json differs from the key snapshotted for this batch")
    if current_manifest.get("answers_sha256") != submission.get("answers_sha256"):
        raise ReaderError("current answers63.jsonl differs from the corpus submitted in this batch")
    if set(current_key) != set(key):
        raise ReaderError("batch side key and current gathered answer IDs differ")
    records: dict[str, dict[str, Any]] = {}
    missing_usage = False
    input_tokens = 0
    output_tokens = 0
    seen_custom_ids: set[str] = set()
    for line in result_lines:
        custom_id = line.get("custom_id")
        if custom_id not in idmap:
            raise ReaderError(f"batch contains unknown custom_id {custom_id!r}")
        if custom_id in seen_custom_ids:
            raise ReaderError(f"batch contains duplicate custom_id {custom_id!r}")
        seen_custom_ids.add(custom_id)
        internal_id = idmap[custom_id]
        model = key.get(internal_id, {}).get("model") if internal_id in key else None
        result = line.get("result")
        if not isinstance(result, dict):
            parsed = {"status": "failed", "error": "missing result object", "stop_reason": None}
            missing_usage = True
        else:
            parsed = parse_response(result, model)
            message = result.get("message") if isinstance(result, dict) else None
            usage = message.get("usage") if isinstance(message, dict) else None
            if isinstance(usage, dict) and isinstance(usage.get("input_tokens"), int) and isinstance(usage.get("output_tokens"), int):
                input_tokens += usage["input_tokens"]
                output_tokens += usage["output_tokens"]
            else:
                missing_usage = True
        records[internal_id] = parsed
    for custom_id, internal_id in idmap.items():
        if custom_id not in seen_custom_ids:
            records[internal_id] = {"status": "failed", "error": "missing batch result", "stop_reason": None}
            missing_usage = True

    expected_quality = snapshot["calibration"]["expected_quality"]
    retry_parent = submission.get("retry_of_submission_id")
    root_batch_id = submission.get("root_batch_id") or batch_id
    labels_path = data_dir / f"labels63-{root_batch_id}.json"
    attempt_usage = {"input_tokens": input_tokens, "output_tokens": output_tokens,
                     "missing_usage_for_any_result": missing_usage}
    batch_ids = [batch_id]
    merged_records = records
    usage = attempt_usage
    prior_retry: dict[str, Any] | None = None
    if retry_parent:
        parent_labels = read_json(labels_path)
        if not isinstance(parent_labels, dict) or not isinstance(parent_labels.get("records"), dict):
            raise ReaderError("retry parent labels are missing or malformed")
        merged_records = dict(parent_labels["records"])
        merged_records.update(records)
        parent_usage = parent_labels.get("usage", {})
        usage = {
            "input_tokens": int(parent_usage.get("input_tokens", 0)) + input_tokens,
            "output_tokens": int(parent_usage.get("output_tokens", 0)) + output_tokens,
            "missing_usage_for_any_result": bool(parent_usage.get("missing_usage_for_any_result")) or missing_usage,
        }
        batch_ids = list(parent_labels.get("batch_ids", [root_batch_id])) + [batch_id]
    elif labels_path.exists():
        prior_labels = read_json(labels_path)
        if (not isinstance(prior_labels, dict) or prior_labels.get("batch_id") != root_batch_id
                or prior_labels.get("submission_id") != submission_id
                or prior_labels.get("request_count") != int(snapshot.get("request_count", len(idmap)))):
            raise ReaderError(f"refusing stale or mismatched labels at {labels_path}")
        prior_records = prior_labels.get("records")
        if isinstance(prior_records, dict):
            merged_records.update(prior_records)
            batch_ids = list(prior_labels.get("batch_ids", batch_ids))
            retry_value = prior_labels.get("retry")
            if isinstance(retry_value, dict):
                prior_retry = retry_value
    for answer_ids in snapshot.get("answer_ids_by_text", {}).values():
        primary_id = answer_ids[0] if answer_ids else None
        if primary_id in merged_records:
            for answer_id in answer_ids:
                merged_records[answer_id] = dict(merged_records[primary_id])
    calibration = calibration_result(merged_records, expected_quality)
    failed_ids = _failed_or_unlabelled_ids(merged_records)
    retry_ids = {internal_id for internal_id in failed_ids
                 if internal_id.startswith(("cal|", "dup|"))}
    for answer_ids in snapshot.get("answer_ids_by_text", {}).values():
        if answer_ids and answer_ids[0] in failed_ids:
            retry_ids.add(answer_ids[0])
    label_output = {
        "version": 1,
        "batch_id": root_batch_id,
        "batch_ids": batch_ids,
        "submission_id": retry_parent or submission_id,
        "model": MODEL,
        "request_count": int(snapshot.get("request_count", len(idmap))),
        "result_count": len(merged_records),
        "calibration": calibration,
        "duplicate_consistency": snapshot.get("duplicate_consistency", {}),
        "usage": usage,
        "attempt_usage": attempt_usage,
        "records": merged_records,
        "failed_or_unlabelled_ids": sorted(failed_ids),
        "scoring_gate_passed": calibration["passed"] and not failed_ids,
    }
    if retry_parent:
        label_output["retry"] = {"status": "collected", "batch_id": batch_id,
                                  "request_count": len(idmap), "remaining_bad_ids": sorted(failed_ids)}
    elif prior_retry is not None:
        label_output["retry"] = prior_retry
    if missing_usage:
        actual_usd = float(submission["estimated_usd"])
    else:
        actual_usd = (input_tokens * PRICE_INPUT + output_tokens * PRICE_OUTPUT) / PRICE_DIVISOR
    _spent, spend_states = spend_state(active_ledger_path)
    reserve_usd = float(spend_states[submission_id]["reserved_usd"])
    print(f"calibration: {calibration['wrong']} wrong, {calibration['missing_or_failed']} missing/failed, "
          f"truncated={len(calibration['truncated'])}; passed={calibration['passed']}")
    print(f"wrote {labels_path}; usage input={input_tokens} output={output_tokens}; "
          f"actual cost=${actual_usd:.4f} vs reserve=${reserve_usd:.4f}"
          f"{' (reserve upper bound used)' if missing_usage else ''}")
    if retry_parent:
        print(f"automatic retry complete: {len(failed_ids)} answer/calibration IDs remain failed or unlabelled; "
              "no further retry will be sent")
        write_json(labels_path, label_output)
    elif retry_ids:
        previous_retry = label_output.get("retry")
        if isinstance(previous_retry, dict) and previous_retry.get("status") in ("submitted", "collected"):
            write_json(labels_path, label_output)
            settle_reserve(review_dir, submission_id, actual_usd, used_reserved_upper_bound=missing_usage,
                           ledger_path=active_ledger_path)
            return labels_path
        # A retry may have been POSTed before a kill prevented its metadata from
        # reaching the labels file. Reuse its recorded state; never POST it twice.
        existing_retries = []
        for retry_path in review_dir.glob("submission-*.json"):
            if retry_path.name.endswith(("-idmap.json", "-key.json")):
                continue
            retry_submission = read_json(retry_path)
            if retry_submission.get("retry_of_submission_id") == submission_id:
                existing_retries.append(retry_submission)
        if existing_retries:
            existing_retries.sort(key=lambda row: row.get("created_at", ""))
            retry_submission = existing_retries[0]
            if retry_submission.get("batch_id"):
                label_output["retry"] = {"status": "submitted", "batch_id": retry_submission["batch_id"],
                                          "request_count": retry_submission.get("request_count", len(retry_ids))}
            else:
                label_output["retry"] = {"status": "ambiguous", "submission_id": retry_submission.get("submission_id"),
                                          "request_count": retry_submission.get("request_count", len(retry_ids))}
            write_json(labels_path, label_output)
            settle_reserve(review_dir, submission_id, actual_usd, used_reserved_upper_bound=missing_usage,
                           ledger_path=active_ledger_path)
            return labels_path
        print(f"automatic retry: submitting only {len(retry_ids)} failed or unlabelled distinct-text/calibration IDs")
        try:
            retry_batch_id = submit(data_dir, review_dir, api, only_internal_ids=retry_ids,
                                    retry_of_submission_id=submission_id, root_batch_id=batch_id,
                                    ledger_path=active_ledger_path)
            label_output["retry"] = {"status": "submitted", "batch_id": retry_batch_id,
                                      "request_count": len(retry_ids)}
        except ReaderError as exc:
            label_output["retry"] = {"status": "not_submitted", "reason": str(exc),
                                      "request_count": len(retry_ids)}
            print(f"automatic retry not submitted: {exc}; scoring remains blocked if labels are incomplete")
        write_json(labels_path, label_output)
    else:
        write_json(labels_path, label_output)
    settle_reserve(review_dir, submission_id, actual_usd, used_reserved_upper_bound=missing_usage,
                   ledger_path=active_ledger_path)
    return labels_path


def _poll_until_ended(batch_id: str, api: Any, *, interval: float,
                      bound: float, sleep: Any = time.sleep) -> None:
    deadline = time.monotonic() + bound
    while True:
        response = json.loads(api("GET", f"/v1/messages/batches/{batch_id}"))
        if not isinstance(response, dict):
            raise ReaderError(f"batch {batch_id} status response is not an object")
        if response.get("processing_status") == "ended":
            return
        if time.monotonic() >= deadline:
            raise ReaderError(f"batch {batch_id} did not end within {bound:g}s; reservation remains active")
        sleep(min(interval, max(0.0, deadline - time.monotonic())))


def _resume_ready_submission(review_dir: Path, path: Path, submission: dict[str, Any],
                             api: Any, ledger_path: Path) -> None:
    """Resume a durable reservation whose POST is known not to have started."""
    submission_id = submission["submission_id"]
    ledger = Ledger(str(ledger_path), SHARED_CAP_USD)
    with ledger._locked():
        events, _tail = ledger._read_locked()
        attempts = ledger._attempts(events)
    matching = [(attempt_id, row) for attempt_id, row in attempts.items()
                if row.get("kind") == "part63"
                and row.get("meta", {}).get("submission_id") == submission_id]
    if not matching:
        submission.update(status="failed_pre_post", error="no durable reserve exists")
        write_json(path, submission)
        return
    if len(matching) != 1 or matching[0][1].get("status") != "reserved":
        raise ReaderError(f"ready submission {submission_id} does not have one active durable reserve")
    submission["attempt_id"] = matching[0][0]
    all_requests, custom_to_internal, _snapshot, _key_sha = build_requests(Path(submission["data_dir"]))
    request_ids = set(submission.get("request_ids", []))
    selected = {custom_id for custom_id, internal_id in custom_to_internal.items() if internal_id in request_ids}
    idmap = _read_request_map(review_dir / submission["idmap_file"])
    if set(idmap) != selected:
        raise ReaderError(f"ready submission {submission_id} request plan no longer matches its durable map")
    requests = [row for row in all_requests if row["custom_id"] in selected]
    submission.update(status="posting", posting_started_at=utc_now())
    write_json(path, submission)
    try:
        raw_response = api("POST", "/v1/messages/batches", {"requests": requests})
    except Exception as exc:
        _mark_ambiguous(review_dir, path, submission, f"recovered POST outcome unknown: {type(exc).__name__}: {exc}")
        raise ReaderError(f"recovered POST outcome unknown for {submission_id}; reserve remains active") from exc
    raw_path = review_dir / f"post-response-{submission_id}.json"
    with raw_path.open("w", encoding="utf-8") as stream:
        stream.write(raw_response)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        response = json.loads(raw_response)
        batch_id = response.get("id") if isinstance(response, dict) else None
        if not isinstance(batch_id, str) or not batch_id:
            raise ReaderError("batch POST response is missing a string id")
    except Exception as exc:
        _mark_ambiguous(review_dir, path, submission, f"invalid or incomplete recovered POST response: {exc}")
        raise ReaderError(f"recovered POST response is ambiguous for {submission_id}") from exc
    submission.update(batch_id=batch_id, status="submitted",
                      processing_status=response.get("processing_status"), submitted_at=utc_now())
    write_json(path, submission)
    write_json(review_dir / f"batch-{batch_id}.json", {
        "batch_id": batch_id, "submission_id": submission_id,
        "submission_file": path.name, "root_batch_id": submission.get("root_batch_id") or batch_id,
    })
    append_spend(review_dir / "spend.jsonl", {"event": "submitted", "submission_id": submission_id,
                                               "batch_id": batch_id, "created_at": utc_now()})


def run_all(data_dir: Path, review_dir: Path, api: Any = call, *,
            chunk_size: int = RUN_ALL_CHUNK_SIZE,
            poll_interval: float = RUN_ALL_POLL_INTERVAL_SECONDS,
            poll_bound: float = RUN_ALL_POLL_BOUND_SECONDS,
            sleep: Any = time.sleep, ledger_path: Path | None = None) -> Path:
    """Submit, poll, collect, and settle the corpus in resumable sequential chunks."""
    if chunk_size != RUN_ALL_CHUNK_SIZE:
        raise ReaderError(f"run-all chunk size is fixed at {RUN_ALL_CHUNK_SIZE}")
    active_ledger_path = Path(ledger_path or DEFAULT_LEDGER_PATH)
    requests, custom_to_internal, snapshot, _key_sha = build_requests(data_dir)
    chunks = [requests[start:start + chunk_size] for start in range(0, len(requests), chunk_size)]
    internal_chunks = [[custom_to_internal[request["custom_id"]] for request in chunk]
                       for chunk in chunks]
    run_path = review_dir / "run-all.json"
    if run_path.exists():
        run = read_json(run_path)
        if run.get("internal_chunks") != internal_chunks:
            raise ReaderError("existing run-all plan does not match current request snapshot")
    else:
        run = {"run_id": uuid.uuid4().hex, "internal_chunks": internal_chunks,
               "chunks": [{"index": index, "state": "planned"} for index in range(len(chunks))],
               "created_at": utc_now()}
        write_json(run_path, run)

    def submission_for(batch: str) -> tuple[str, dict[str, Any]]:
        link = _known_batch_link(review_dir, batch)
        sub_id = link.get("submission_id")
        if not isinstance(sub_id, str):
            raise ReaderError(f"batch {batch} has no submission link")
        return sub_id, read_json(review_dir / f"submission-{sub_id}.json")

    def settle_batch(batch: str) -> None:
        sub_id, sub = submission_for(batch)
        _poll_until_ended(batch, api, interval=poll_interval, bound=poll_bound, sleep=sleep)
        collect(batch, data_dir, review_dir, api, ledger_path=active_ledger_path)
        labels_path = data_dir / f"labels63-{sub.get('root_batch_id') or batch}.json"
        labels = read_json(labels_path)
        retry = labels.get("retry") if isinstance(labels, dict) else None
        if isinstance(retry, dict) and retry.get("status") == "submitted":
            retry_batch = retry.get("batch_id")
            if not isinstance(retry_batch, str):
                raise ReaderError("retry metadata has no batch ID")
            _poll_until_ended(retry_batch, api, interval=poll_interval, bound=poll_bound, sleep=sleep)
            collect(retry_batch, data_dir, review_dir, api, ledger_path=active_ledger_path)
        # A killed process or an uncollected retry must never be followed by another chunk.
        state = Ledger(str(active_ledger_path), SHARED_CAP_USD).state()
        active = state.get("per_kind", {}).get("part63", {}).get("reserved", 0.0)
        if active > 1e-12:
            raise ReaderError(f"Part 63 has ${active:.6f} reserved/unknown; run-all refuses to advance")

    with (review_dir / "run-all.lock").open("a", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        try:
            # Recover known batch IDs first. An unknown accepted POST has no adoptable ID and remains reserved.
            for path in sorted(review_dir.glob("submission-*.json")):
                if path.name.endswith(("-idmap.json", "-key.json")):
                    continue
                sub = read_json(path)
                if sub.get("run_all_id") != run["run_id"]:
                    continue
                if sub.get("status") == "posting":
                    # A process died after recording intent to POST. Without a durable response ID
                    # this is deliberately treated as unknown, even if the POST may not have begun.
                    sub["status"] = "ambiguous"
                    sub["ambiguous_at"] = utc_now()
                    sub["error"] = "previous process ended in POST window; outcome is unknown"
                    write_json(path, sub)
                if sub.get("status") == "ready":
                    _resume_ready_submission(review_dir, path, sub, api, active_ledger_path)
                    sub = read_json(path)
                if not sub.get("batch_id"):
                    raw_path = review_dir / f"post-response-{sub.get('submission_id')}.json"
                    if raw_path.is_file():
                        raw_response = read_json(raw_path)
                        recovered_id = raw_response.get("id") if isinstance(raw_response, dict) else None
                        if isinstance(recovered_id, str) and recovered_id:
                            sub.update(batch_id=recovered_id, status="submitted",
                                       processing_status=raw_response.get("processing_status"),
                                       recovered_from_post_response=True)
                            write_json(path, sub)
                            write_json(review_dir / f"batch-{recovered_id}.json", {
                                "batch_id": recovered_id, "submission_id": sub["submission_id"],
                                "submission_file": path.name,
                                "root_batch_id": sub.get("root_batch_id") or recovered_id,
                            })
                if not sub.get("batch_id") and sub.get("status") == "ambiguous":
                    _spent, states = spend_state(active_ledger_path)
                    attempt_id = sub.get("attempt_id") or states.get(sub.get("submission_id"), {}).get("attempt_id")
                    attempt_state = states.get(sub.get("submission_id"), {}).get("event")
                    if attempt_state == "retired":
                        chunk_index = sub.get("chunk_index")
                        if isinstance(chunk_index, int) and 0 <= chunk_index < len(run["chunks"]):
                            chunk_state = run["chunks"][chunk_index]
                            retired = chunk_state.setdefault("retired_unknown_attempts", [])
                            if not any(item.get("submission_id") == sub.get("submission_id") for item in retired):
                                retired.append({"state": "retired-unknown",
                                                "submission_id": sub.get("submission_id"),
                                                "attempt_id": attempt_id,
                                                "recorded_at": utc_now()})
                            chunk_state["state"] = "retired-unknown"
                            write_json(run_path, run)
                        continue
                    exact_command = (
                        f'python3 {HERE / "reader63.py"} retire {attempt_id} --who "$(id -un)" '
                        '--reason "POST outcome unknown; no batch ID on disk"')
                    raise ReaderError(
                        f"chunk {sub.get('chunk_index')} is POST outcome unknown (no batch ID on disk); "
                        f"retire the counted reserve with `{exact_command}`, then rerun: `python3 {HERE / 'reader63.py'} "
                        f"run-all --data-dir {data_dir} --review-dir {review_dir}`")
                if sub.get("batch_id") and sub.get("status") in ("submitted", "ambiguous"):
                    if sub.get("status") == "ambiguous":
                        sub["status"] = "submitted"
                        sub["recovered_known_batch_id"] = True
                        write_json(path, sub)
                    settle_batch(sub["batch_id"])
            for index, internal_ids in enumerate(internal_chunks):
                chunk_state = run["chunks"][index]
                if chunk_state.get("state") == "settled":
                    continue
                # If this chunk already has a durable submission, recovery above owns it.
                existing = [read_json(path) for path in review_dir.glob("submission-*.json")
                            if not path.name.endswith(("-idmap.json", "-key.json"))
                            and read_json(path).get("run_all_id") == run["run_id"]
                            and read_json(path).get("chunk_index") == index]
                if existing:
                    shared_ledger = Ledger(str(active_ledger_path), SHARED_CAP_USD)
                    with shared_ledger._locked():
                        attempts, _tail = shared_ledger._read_locked()
                        attempts = shared_ledger._attempts(attempts)
                    retry_after_retirement = all(
                        row.get("status") in ("ambiguous", "failed_pre_post")
                        and not row.get("batch_id")
                        and attempts.get(row.get("attempt_id"), {}).get("status") == "retired"
                        for row in existing
                    )
                    if retry_after_retirement:
                        existing = []
                if existing:
                    if any(row.get("status") == "ambiguous" for row in existing):
                        raise ReaderError(f"chunk {index} has an ambiguous POST; refusing a new attempt")
                    if any(row.get("attempt_id") not in attempts
                           or attempts[row["attempt_id"]].get("status") not in ("settled", "retired", "released")
                           for row in existing):
                        raise ReaderError(f"chunk {index} remains unsettled; rerun after recovery")
                    if any(row.get("status") in ("rejected", "not_submitted", "failed_pre_post") for row in existing):
                        raise ReaderError(f"chunk {index} has a failed attempt; inspect ledger before retry")
                    chunk_state["state"] = "settled"
                    write_json(run_path, run)
                    continue
                ledger_state = Ledger(str(active_ledger_path), SHARED_CAP_USD).state()
                active_part63 = ledger_state.get("per_kind", {}).get("part63", {}).get("reserved", 0.0)
                if active_part63 > 1e-12:
                    raise ReaderError(f"Part 63 has ${active_part63:.6f} reserved/unknown; run-all refuses a new chunk")
                batch_id = submit(data_dir, review_dir, api, only_internal_ids=set(internal_ids),
                                  run_all_id=run["run_id"], chunk_index=index,
                                  ledger_path=active_ledger_path)
                chunk_state.update(state="submitted", batch_id=batch_id)
                write_json(run_path, run)
                settle_batch(batch_id)
                chunk_state["state"] = "settled"
                write_json(run_path, run)
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)

    # Each chunk has its own labels file. Combine by stable internal ID and preserve repeated-row fanout.
    merged: dict[str, Any] = {"records": {}, "batch_ids": [], "request_count": len(requests),
                              "duplicate_consistency": snapshot["duplicate_consistency"],
                              "model": MODEL, "version": 1}
    usages = []
    for chunk_state in run["chunks"]:
        batch = chunk_state.get("batch_id")
        if not batch:
            rows = [read_json(path) for path in review_dir.glob("submission-*.json")
                    if not path.name.endswith(("-idmap.json", "-key.json"))
                    and read_json(path).get("run_all_id") == run["run_id"]
                    and read_json(path).get("chunk_index") == chunk_state["index"]]
            batch = rows[0].get("batch_id") if rows else None
        if not batch:
            continue
        _sub_id, sub = submission_for(batch)
        label_path = data_dir / f"labels63-{sub.get('root_batch_id') or batch}.json"
        if not label_path.is_file():
            continue
        labels = read_json(label_path)
        merged["records"].update(labels.get("records", {}))
        merged["batch_ids"].extend(labels.get("batch_ids", [batch]))
        usages.append(labels.get("usage", {}))
    merged["calibration"] = calibration_result(merged["records"], snapshot["calibration"]["expected_quality"])
    failed = _failed_or_unlabelled_ids(merged["records"])
    merged["failed_or_unlabelled_ids"] = sorted(failed)
    merged["scoring_gate_passed"] = merged["calibration"]["passed"] and not failed
    merged["usage"] = {"input_tokens": sum(int(row.get("input_tokens", 0)) for row in usages),
                       "output_tokens": sum(int(row.get("output_tokens", 0)) for row in usages),
                       "missing_usage_for_any_result": any(row.get("missing_usage_for_any_result") for row in usages)}
    output = data_dir / "labels63-run-all.json"
    write_json(output, merged)
    return output


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    submit_parser = commands.add_parser("submit")
    submit_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    submit_parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    run_all_parser = commands.add_parser("run-all", help="submit and settle the full corpus in sequential chunks")
    run_all_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    run_all_parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    listing_parser = commands.add_parser("listing", help="read-only provider listing since this run began")
    listing_parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("batch_id")
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("batch_id")
    collect_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    collect_parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    collect_parser.add_argument("--ledger-path", type=Path, default=DEFAULT_LEDGER_PATH)
    retire_parser = commands.add_parser("retire", help="operator-retire an ambiguous shared-ledger reserve")
    retire_parser.add_argument("attempt_id")
    retire_parser.add_argument("--who", required=True)
    retire_parser.add_argument("--reason", required=True)
    retire_parser.add_argument("--ledger-path", type=Path, default=DEFAULT_LEDGER_PATH)
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    try:
        if args.command == "submit":
            submit(args.data_dir, args.review_dir)
        elif args.command == "run-all":
            run_all(args.data_dir, args.review_dir)
        elif args.command == "listing":
            batches = listing_batches(args.review_dir)
            print(json.dumps({"count": len(batches), "batches": batches}, sort_keys=True))
        elif args.command == "status":
            status(args.batch_id)
        elif args.command == "retire":
            retire_attempt(args.attempt_id, args.who, args.reason, ledger_path=args.ledger_path)
        else:
            collect(args.batch_id, args.data_dir, args.review_dir, ledger_path=args.ledger_path)
    except (ReaderError, OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
