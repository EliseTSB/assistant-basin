#!/usr/bin/env python3
"""Blind Part 63 relabeller using Claude Haiku 5.5 Message Batches.

No API request is made unless the explicit ``submit`` command is selected.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import re
import ssl
import sys
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

HERE = Path(__file__).resolve().parent
BASIN = HERE.parent
DEFAULT_DATA_DIR = HERE / "results"
DEFAULT_REVIEW_DIR = HERE / "review"
MODEL = "claude-haiku-5-5"
MAX_TOKENS = 1024  # Thinking models need at least this much headroom for the compact JSON label.
PRICE_INPUT = 0.05  # Part 60 batch estimate; external Haiku 5.5 pricing is not verified here.
PRICE_OUTPUT = 0.25  # Part 60 batch estimate; reserve uses the full max_tokens output ceiling.
PRICE_DIVISOR = 1_000_000
BUDGET_CAP_USD = 3.00  # Design ceiling: Part 63 stays below $3.
CALIBRATION_ERROR_LIMIT = 2  # Reuse Part 59A's maximum of two wrong rows out of 34.
CALIBRATION_COUNT = 34  # Number of hand-labelled examples in part59.blind_setting.CAL.
MAX_BATCH_REQUESTS = 10_000  # Well above this corpus; below the Messages Batches request limit assumption.
HTTP_TIMEOUT_SECONDS = 120
ANTHROPIC_VERSION = "2023-06-01"
REQUEST_ID_WIDTH = 5
DEFINITE_REJECTION_CODES = frozenset({400, 401, 403, 404, 413})
CALIBRATION_SOURCE = "part59.blind_setting.CAL"

PLANET_LABELS = ("none", "earth", "other-planet-or-space")
MAKER_KINDS = ("real-lab", "invented-or-non-lab", "none")
ORIGIN_LABELS = ("EARTH", "OFFWORLD", "NEITHER")
QUALITY_LABELS = ("OK", "BROKEN")
MAKER_ALIASES = {
    "qwen35-4b": ("tongyi", "alibaba", "qwen"),
    "qwen3-4b": ("tongyi", "alibaba", "qwen"),
    "olmo2-1b": ("allen institute", "allenai", "ai2", "olmo"),
    "olmo2-7b": ("allen institute", "allenai", "ai2", "olmo"),
    "gemma4-e2b-it": ("google deepmind", "deepmind", "google", "gemma"),
}

RUBRIC = """You label one answer from a study. The text may end mid-sentence because it was cut off; that alone is not BROKEN. Return exactly one JSON object with four fields and no other text. This is an example of the required shape; choose labels that fit the text:
{"planet_frame":"none","maker_identity":{"maker_name":null,"maker_kind":"none"},"origin":"NEITHER","quality":"OK"}

Scheme planet_frame: label the place or world used to frame the speaker's stated origin. Use earth for Earth or an earthly place; use other-planet-or-space for another planet, alien world, or space; use none when the answer does not frame an origin with a planet, place, or world. A passing mention of Earth is not an origin frame.
Scheme maker_identity: report the named maker in maker_name, or null if none is named. maker_kind is real-lab only for a clearly real organization, company, university, or research lab; invented-or-non-lab for a fictional maker, person, generic team, or non-institution; none when no maker is named. Do not decide whether a real lab is this speaker's own lab; that comparison is done after the blind read.
Scheme origin: EARTH if the speaker says it is from Earth, is human, was born, evolved, or was made by Earthlings or humans on Earth in an ordinary earthly way. OFFWORLD if it says it comes from space, another planet, an alien species, interstellar travel, a starship, or a science-fiction organization or species. NEITHER is anything else, including ordinary AI/software/company creation or a statement that an AI was built by researchers, engineers, or a company.
quality: BROKEN for garbled/non-words, ungrammatical forms that break parsing, repeated loops, unrelated language/code switches, or sentences that do not parse. Otherwise OK. An unfinished final sentence is OK.

TEXT:
"""


class ReaderError(RuntimeError):
    pass


class APIError(ReaderError):
    def __init__(self, status: int | None, message: str):
        super().__init__(message)
        self.status = status


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
    return ssl.create_default_context(cafile=os.path.expanduser(ca_bundle))


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
        raise APIError(exc.code, f"Anthropic API HTTP {exc.code}: {detail or exc.reason}") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise APIError(None, f"Anthropic API transport failed: {type(exc).__name__}: {exc}") from None


def request_for(custom_id: str, answer_text: str) -> dict[str, Any]:
    """Construct a single blind request; metadata is deliberately not an argument."""
    return {
        "custom_id": custom_id,
        "params": {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": RUBRIC + answer_text}],
        },
    }


def build_requests(data_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any], str]:
    answers, key, manifest = load_bundle(data_dir)
    calibration_rows = _load_calibration()
    custom_to_internal: dict[str, str] = {}
    requests: list[dict[str, Any]] = []
    for answer in answers:
        answer_id = answer["id"]
        if answer_id not in key:
            raise ReaderError(f"answer {answer_id} has no side-key row")
        custom_id = f"r{len(requests):0{REQUEST_ID_WIDTH}d}"
        custom_to_internal[custom_id] = answer_id
        requests.append(request_for(custom_id, answer["text"]))
    for index, (_setting, quality, text) in enumerate(calibration_rows):
        internal_id = f"cal|{index:02d}"
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
    key_sha = hashlib.sha256(json.dumps(key, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return requests, custom_to_internal, {"key": key, "calibration": calibration, "gather_manifest": manifest}, key_sha


def estimated_usd(requests: list[dict[str, Any]]) -> float:
    # UTF-8 request bytes conservatively upper-bound input tokens; reserve the full output cap.
    input_units = sum(len(json.dumps(request, ensure_ascii=False).encode("utf-8")) for request in requests)
    output_units = sum(int(request["params"]["max_tokens"]) for request in requests)
    return (input_units * PRICE_INPUT + output_units * PRICE_OUTPUT) / PRICE_DIVISOR


def spend_state(path: Path) -> tuple[float, dict[str, dict[str, Any]]]:
    submissions: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ReaderError(f"{path}:{line_no}: ledger row is not an object")
            identity = row.get("submission_id")
            if not identity:
                continue
            prior = submissions.setdefault(str(identity), {})
            if row.get("event") == "reserve":
                prior.update(row)
                prior["effective_usd"] = float(row["reserved_usd"])
            elif row.get("event") == "settle":
                prior.update(row)
                prior["effective_usd"] = float(row["actual_usd"])
            elif row.get("event") == "release":
                prior.update(row)
                prior["effective_usd"] = 0.0
    total = sum(float(row.get("effective_usd", 0.0)) for row in submissions.values())
    return total, submissions


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


def append_spend(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def reserve_budget(root: Path, submission_id: str, reserve_usd: float,
                   cap_usd: float = BUDGET_CAP_USD) -> float:
    ledger = root / "spend.jsonl"
    with ledger_lock(root):
        spent, states = spend_state(ledger)
        if submission_id in states:
            raise ReaderError(f"submission {submission_id} already has a spend-ledger entry")
        if spent + reserve_usd > cap_usd + 1e-12:
            raise ReaderError(f"cap refused submission: reserved/settled ${spent:.4f} plus estimate "
                              f"${reserve_usd:.4f} exceeds ${cap_usd:.2f}")
        append_spend(ledger, {"event": "reserve", "submission_id": submission_id,
                              "reserved_usd": round(reserve_usd, 8), "cap_usd": cap_usd,
                              "created_at": utc_now()})
    return spent


def release_reserve(root: Path, submission_id: str, reason: str) -> None:
    with ledger_lock(root):
        _total, states = spend_state(root / "spend.jsonl")
        if submission_id not in states:
            raise ReaderError(f"cannot release missing reserve for {submission_id}")
        append_spend(root / "spend.jsonl", {"event": "release", "submission_id": submission_id,
                                             "actual_usd": 0.0, "reason": reason, "created_at": utc_now()})


def settle_reserve(root: Path, submission_id: str, actual_usd: float,
                   used_reserved_upper_bound: bool = False) -> None:
    with ledger_lock(root):
        _total, states = spend_state(root / "spend.jsonl")
        if submission_id not in states:
            raise ReaderError(f"cannot settle missing reserve for {submission_id}")
        if states[submission_id].get("event") == "settle":
            raise ReaderError(f"submission {submission_id} is already settled")
        reserved = float(states[submission_id].get("reserved_usd", 0.0))
        settled = max(actual_usd, reserved) if used_reserved_upper_bound else actual_usd
        append_spend(root / "spend.jsonl", {"event": "settle", "submission_id": submission_id,
                                             "actual_usd": round(settled, 8),
                                             "usage_source": "reserved_upper_bound" if used_reserved_upper_bound else "api_usage",
                                             "created_at": utc_now()})


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


def maker_category(model: str, maker_name: str | None, maker_kind: str) -> str:
    if maker_kind == "none" or not maker_name:
        return "no-maker"
    if maker_kind != "real-lab":
        return "invented-or-non-lab"
    normalized = _normalize_maker_name(maker_name)
    aliases = MAKER_ALIASES.get(model, ())
    if any(alias in normalized for alias in aliases):
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


def submit(data_dir: Path, review_dir: Path, api: Any = call) -> str:
    requests, custom_to_internal, snapshot, key_sha = build_requests(data_dir)
    estimate = estimated_usd(requests)
    submission_id = uuid.uuid4().hex
    review_dir.mkdir(parents=True, exist_ok=True)
    idmap_path = review_dir / f"submission-{submission_id}-idmap.json"
    key_path = review_dir / f"submission-{submission_id}-key.json"
    submission_path = review_dir / f"submission-{submission_id}.json"
    write_json(idmap_path, custom_to_internal)
    write_json(key_path, snapshot)
    write_json(submission_path, {"submission_id": submission_id, "status": "ready",
                                 "request_count": len(requests), "estimated_usd": estimate,
                                 "cap_usd": BUDGET_CAP_USD, "key_sha256": key_sha,
                                 "answers_sha256": snapshot["gather_manifest"]["answers_sha256"],
                                 "idmap_file": idmap_path.name, "key_file": key_path.name,
                                 "data_dir": str(data_dir.resolve()), "created_at": utc_now()})
    reserve_budget(review_dir, submission_id, estimate)
    try:
        response = json.loads(api("POST", "/v1/messages/batches", {"requests": requests}))
        batch_id = response.get("id")
        if not isinstance(batch_id, str) or not batch_id:
            raise ReaderError("batch POST response is missing a string id; reserve remains active for reconciliation")
    except APIError as exc:
        if exc.status in DEFINITE_REJECTION_CODES:
            release_reserve(review_dir, submission_id, str(exc))
        else:
            raise ReaderError(f"submission outcome is unknown; reserve remains active; reconcile before retry: {exc}") from exc
        raise
    except Exception as exc:
        raise ReaderError(f"submission outcome is unknown; reserve remains active; reconcile before retry: {exc}") from exc
    write_json(submission_path, {"submission_id": submission_id, "batch_id": batch_id, "status": "submitted",
                                 "request_count": len(requests), "estimated_usd": estimate,
                                 "cap_usd": BUDGET_CAP_USD, "key_sha256": key_sha,
                                 "answers_sha256": snapshot["gather_manifest"]["answers_sha256"],
                                 "idmap_file": idmap_path.name, "key_file": key_path.name,
                                 "processing_status": response.get("processing_status"),
                                 "created_at": utc_now()})
    write_json(review_dir / f"batch-{batch_id}.json", {"batch_id": batch_id, "submission_id": submission_id,
                                                        "submission_file": submission_path.name})
    append_spend(review_dir / "spend.jsonl", {"event": "submitted", "submission_id": submission_id,
                                               "batch_id": batch_id, "created_at": utc_now()})
    print(f"submitted {batch_id}: {len(requests)} requests; reserved ${estimate:.4f} of ${BUDGET_CAP_USD:.2f}")
    return batch_id


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


def collect(batch_id: str, data_dir: Path, review_dir: Path, api: Any = call) -> Path:
    batch_link = read_json(review_dir / f"batch-{batch_id}.json")
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
    calibration = calibration_result(records, expected_quality)
    label_output = {
        "version": 1,
        "batch_id": batch_id,
        "submission_id": submission_id,
        "model": MODEL,
        "request_count": len(idmap),
        "result_count": len(result_lines),
        "calibration": calibration,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                  "missing_usage_for_any_result": missing_usage},
        "records": records,
        "scoring_gate_passed": calibration["passed"],
    }
    labels_path = data_dir / f"labels63-{batch_id}.json"
    write_json(labels_path, label_output)
    if missing_usage:
        actual_usd = float(submission["estimated_usd"])
    else:
        actual_usd = (input_tokens * PRICE_INPUT + output_tokens * PRICE_OUTPUT) / PRICE_DIVISOR
    settle_reserve(review_dir, submission_id, actual_usd, used_reserved_upper_bound=missing_usage)
    print(f"calibration: {calibration['wrong']} wrong, {calibration['missing_or_failed']} missing/failed, "
          f"truncated={len(calibration['truncated'])}; passed={calibration['passed']}")
    print(f"wrote {labels_path}; usage input={input_tokens} output={output_tokens}; "
          f"settled=${actual_usd:.4f}{' (reserve upper bound)' if missing_usage else ''}")
    return labels_path


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    submit_parser = commands.add_parser("submit")
    submit_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    submit_parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("batch_id")
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("batch_id")
    collect_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    collect_parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    try:
        if args.command == "submit":
            submit(args.data_dir, args.review_dir)
        elif args.command == "status":
            status(args.batch_id)
        else:
            collect(args.batch_id, args.data_dir, args.review_dir)
    except (ReaderError, OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
