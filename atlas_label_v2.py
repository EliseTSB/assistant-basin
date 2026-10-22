# Part 58 labeller (V9 prompt, marked-token table computed in code, 21 windows). Derived from the Part 56 atlas_label.py; same batch machinery.
#!/usr/bin/env python3
"""Submit atlas windows to Claude Haiku 5.5 through Message Batches.

Usage:
  atlas_label.py submit --records results/atlas-qwen3-1.7b/neurons.jsonl --out review/atlas-main
  atlas_label.py status BATCH_ID
  atlas_label.py collect --out review/atlas-main

The Anthropic key is read only from ANTHROPIC_CREDITS_KEY and is never printed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
from collections import Counter
import os
import random
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Sequence


MODEL = "claude-haiku-5-5"
SUPPORTED_MODELS = {"claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"}
MAX_TOKENS = 400
MAX_REQUESTS_PER_BATCH = 10_000
MAX_BATCH_BYTES = 190_000_000
ESTIMATED_WINDOW_TOKENS = 2_100
ESTIMATED_SYSTEM_TOKENS = 700
ESTIMATED_OUTPUT_TOKENS = 90
PRICES_PER_MILLION = {
    "input": 0.05,
    "output": 0.25,
    "cache_write": 0.10,
    "cache_read": 0.005,
}
KINDS = {
    "token", "word-family", "topic", "syntax", "format",
    "identity-or-origin", "persona-or-creator", "other", "none",
}

# Six fully invented examples. They teach the output shape without quoting any
# item from the Pile or from a real neuron record.
SYSTEM_PROMPT = r"""You are labelling one hidden activation unit from a language model. The input is a JSON object with "marked_token_counts" (an exact frequency table of the marked tokens, most frequent first, as [token, count] pairs), "windows" (short text windows where the unit was active) and "marked_tokens" (the token wrapped in [[double brackets]] in each window, in the same order). ⟦PAD×N⟧ marks missing context and is not text.

Return one JSON object and nothing else, with exactly these keys:
{"concept":"at most 12 words","fit":"K of N","one_job":1,"kind":"topic","note":""}

- concept: the broadest accurate plain-language description of what the marked tokens have in common, at most 12 words. Add conditions (position, preceding word, sentence role) only if nearly all windows share them: many different past-tense verbs is "past-tense -ed verbs", not "past-tense verbs right after the subject".
- fit: use the frequency table to add up windows by token type; it is exact, so do not recount by eye. N is the number of windows; K is how many have a marked token that belongs to your concept. A reader who saw only your concept should expect the unit to fire at that marked token.
- one_job: integer 1 to 5 set from K/N alone: 5 = 85% or more; 4 = 65-84%; 3 = 45-64%; 2 = 25-44%; 1 = under 25% or no concept. Common words, punctuation, digits and word fragments are real features; do not lower the score because a feature is plain or common, and do not raise it because it sounds interesting. If the windows split between two unrelated features, count only the larger one.
- kind: exactly one of token, word-family, topic, syntax, format, identity-or-origin (claims about what the speaker is or where it comes from), persona-or-creator (a recurring named voice, character, maker or organisation), other, none.
- note: empty unless something is genuinely unusual.

Example: 21 windows, 15 mark forms of "glimmer", 6 unrelated. {"concept":"glimmer word family","fit":"15 of 21","one_job":4,"kind":"word-family","note":""}
The JSON must parse exactly; no markdown fences or prose."""


def record_identity(record: dict[str, Any], index: int) -> dict[str, Any]:
    layer = int(record["layer"])
    arm = str(record.get("model_arm", "trained"))
    prefix = "" if arm == "trained" else f"{arm}:"
    identity: dict[str, Any] = {"record_index": int(index), "layer": layer, "model_arm": arm}
    if "unit" in record:
        unit = int(record["unit"])
        identity.update({"record_type": "neuron", "unit": unit,
                         "unit_key": f"{prefix}L{layer}:{unit}"})
    elif "control_id" in record:
        control_id = int(record["control_id"])
        identity.update({"record_type": "control", "control_id": control_id,
                         "unit_key": f"{prefix}L{layer}:control-{control_id}"})
    else:
        raise ValueError("each atlas row must have either unit or control_id")
    return identity


DESCRIBE_COUNTS = (("top", 12), ("random", 6), ("weak", 3))   # the configuration scored in Part 58


def extract_windows(record: dict[str, Any]) -> list[str]:
    windows: list[str] = []
    for section, count in DESCRIBE_COUNTS:
        values = record.get(section, [])
        if not isinstance(values, list):
            raise ValueError(f"{section} must be a list")
        for item in values[:count]:
            if not isinstance(item, dict) or not isinstance(item.get("window"), str):
                raise ValueError(f"each {section} entry must contain a window string")
            windows.append(item["window"])
    if not windows:
        raise ValueError("atlas row has no windows")
    return windows


def marked_token(window: str) -> str:
    start = window.find("[[")
    end = window.find("]]", start + 2) if start >= 0 else -1
    return window[start + 2:end] if start >= 0 and end >= 0 else ""


def user_message(windows: list[str]) -> str:
    """Part 58 input: the exact token frequency table is computed here, not by the model."""
    tokens = [marked_token(w) for w in windows]
    counts = [[t, n] for t, n in Counter(tokens).most_common(12)]
    return json.dumps({"marked_token_counts": counts, "marked_tokens": tokens, "windows": windows},
                      ensure_ascii=False, separators=(",", ":"))


def thinking_fields(model_id: str) -> dict[str, Any]:
    if model_id == "claude-haiku-5-5":
        return {"thinking": {"type": "disabled"}}
    if model_id == "claude-sonnet-5-5":
        return {"thinking": {"type": "between_tools"}}
    if model_id == "claude-opus-5-5":
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}
    raise ValueError(f"unsupported Anthropic judge model: {model_id}")


def build_request(record: dict[str, Any], custom_id: str,
                  model_id: str = MODEL) -> dict[str, Any]:
    """Build one API request; the user message contains only a JSON list of windows."""
    if model_id not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported Anthropic judge model: {model_id}")
    user_windows = user_message(extract_windows(record))
    return {
        "custom_id": custom_id,
        "params": {
            "model": model_id,
            "max_tokens": MAX_TOKENS,
            **thinking_fields(model_id),
            "system": [{
                "type": "text", "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }],
            "messages": [{"role": "user", "content": user_windows}],
        },
    }


FIT_RE = re.compile(r"^\s*(\d+)\s*(?:of|/)\s*(\d+)\s*$")


def parse_label(text: str) -> dict[str, Any] | None:
    """Parse and validate a strict JSON label; malformed/truncated text returns None."""
    raw = text.strip()
    if raw.startswith("```json"):
        raw = raw[7:].strip()
    elif raw.startswith("```"):
        raw = raw[3:].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != {"concept", "fit", "one_job", "kind", "note"}:
        return None
    concept, fit, one_job, kind, note = (value[k] for k in ("concept", "fit", "one_job", "kind", "note"))
    if not isinstance(concept, str) or not concept.strip() or len(concept.split()) > 12:
        return None
    match = FIT_RE.match(fit) if isinstance(fit, str) else None
    if not match or int(match.group(2)) <= 0 or int(match.group(1)) > int(match.group(2)):
        return None
    if isinstance(one_job, bool) or not isinstance(one_job, int) or not 1 <= one_job <= 5:
        return None
    if not isinstance(kind, str) or kind not in KINDS:
        return None
    if not isinstance(note, str):
        return None
    return {"concept": concept, "fit": fit.strip(), "fit_frac": round(int(match.group(1)) / int(match.group(2)), 4),
            "one_job": one_job, "kind": kind, "note": note}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp.replace(path)


def _ssl_context() -> ssl.SSLContext:
    ca_file = Path(os.path.expanduser("~/.mitmproxy/ca-bundle-combined.crt"))
    context = ssl.create_default_context()
    if ca_file.is_file():
        context.load_verify_locations(cafile=str(ca_file))
    return context


def api_call(method: str, path: str, body: dict[str, Any] | bytes | None = None) -> str:
    key = os.environ.get("ANTHROPIC_CREDITS_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_CREDITS_KEY is required for API calls")
    data = (body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
            if body is not None else None)
    request = urllib.request.Request(
        "https://api.anthropic.com" + path,
        method=method,
        data=data,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(request, context=_ssl_context(), timeout=180) as response:
        return response.read().decode("utf-8")


def select_second_judge_sample(rows: Sequence[dict[str, Any]], count: int = 1000,
                               seed: int = 5601) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("second-judge sample size must be positive")
    if len(rows) <= count:
        return list(rows)
    return random.Random(seed).sample(list(rows), count)


def _jsonl_offsets(path: Path) -> list[int]:
    offsets: list[int] = []
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
    return offsets


def _read_jsonl_at(path: Path, offset: int) -> dict[str, Any]:
    with path.open("rb") as stream:
        stream.seek(offset)
        value = json.loads(stream.readline())
    if not isinstance(value, dict):
        raise ValueError(f"JSONL record at byte {offset} is not an object")
    return value


def estimate_submission_cost(request_count: int, output_tokens: int = ESTIMATED_OUTPUT_TOKENS) -> float:
    """Estimate cost with a one-hour system cache write and the selected output allowance."""
    per_million = (
        ESTIMATED_WINDOW_TOKENS * PRICES_PER_MILLION["input"]
        + ESTIMATED_SYSTEM_TOKENS * PRICES_PER_MILLION["cache_write"]
        + int(output_tokens) * PRICES_PER_MILLION["output"]
    )
    return int(request_count) * per_million / 1_000_000


def _unit_keys(idmap: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for identity in idmap.values():
        if not isinstance(identity, dict) or not identity.get("unit_key"):
            continue
        key = str(identity["unit_key"])
        if key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _pre_post_created_at() -> str:
    # The API may round created_at to whole seconds; leave one second of room so
    # a batch accepted immediately after this row still compares strictly later.
    value = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(seconds=1)
    return value.isoformat().replace("+00:00", "Z")


@contextmanager
def _manifest_lock(out_dir: Path):
    """Hold an O_EXCL lock across manifest reads, API actions, and atomic writes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "batches.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"manifest lock already exists at {path}; inspect the owner before removing it") from exc
    try:
        os.write(fd, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(fd)
        yield
    finally:
        os.close(fd)
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _serialize_record_chunks(records_path: Path, offsets: Sequence[int], selected: Sequence[int],
                             model_id: str) -> Iterable[tuple[int, bytes, dict[str, Any], int]]:
    cursor = 0
    batch_index = 0
    prefix, suffix = b'{"requests":[', b"]}"
    while cursor < len(selected):
        batch_index += 1
        request_parts: list[bytes] = []
        idmap: dict[str, Any] = {}
        size = len(prefix) + len(suffix)
        while cursor < len(selected) and len(request_parts) < MAX_REQUESTS_PER_BATCH:
            source_index = int(selected[cursor])
            row = _read_jsonl_at(records_path, offsets[source_index])
            local_index = len(request_parts)
            custom_id = f"b{batch_index:04d}r{local_index:05d}"
            request = build_request(row, custom_id, model_id)
            encoded = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            candidate_size = size + len(encoded) + (1 if request_parts else 0)
            if candidate_size > MAX_BATCH_BYTES:
                if not request_parts:
                    raise ValueError(
                        f"single request at atlas row {source_index} exceeds the {MAX_BATCH_BYTES}-byte batch limit"
                    )
                break
            request_parts.append(encoded)
            idmap[custom_id] = record_identity(row, source_index)
            size = candidate_size
            cursor += 1
        body = prefix + b",".join(request_parts) + suffix
        if len(body) > MAX_BATCH_BYTES:
            raise AssertionError("serialized batch exceeded byte budget")
        yield batch_index, body, idmap, len(request_parts)


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    temp.write_bytes(content)
    temp.replace(path)


def _write_manifest(path: Path, manifest: dict[str, Any], current_model: str) -> None:
    batches = manifest.setdefault("batches", [])
    fallback_model = manifest.get("model") or current_model
    for row in batches:
        row.setdefault("model", fallback_model)
    models = sorted({str(row["model"]) for row in batches})
    manifest["models"] = models
    manifest["model"] = models[0] if len(models) == 1 else None
    _write_json(path, manifest)


def _parse_api_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _api_batch_request_count(batch: dict[str, Any]) -> int | None:
    counts = batch.get("request_counts")
    if isinstance(counts, dict):
        try:
            return sum(int(counts.get(key, 0)) for key in
                       ("processing", "succeeded", "errored", "canceled", "expired"))
        except (TypeError, ValueError):
            return None
    try:
        return int(batch["request_count"])
    except (KeyError, TypeError, ValueError):
        return None


def _list_existing_batches() -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    while True:
        path = "/v1/messages/batches?limit=1000"
        if cursor:
            from urllib.parse import quote
            path += "&after_id=" + quote(cursor, safe="")
        page = json.loads(api_call("GET", path))
        if not isinstance(page, dict) or not isinstance(page.get("data"), list):
            raise ValueError("batch recovery list returned an invalid page")
        batches.extend(row for row in page["data"] if isinstance(row, dict))
        if not page.get("has_more"):
            return batches
        next_cursor = page.get("last_id") or (page["data"][-1].get("id") if page["data"] else None)
        if not next_cursor or str(next_cursor) in seen_cursors:
            raise ValueError("batch recovery pagination did not provide a new cursor")
        cursor = str(next_cursor)
        seen_cursors.add(cursor)


def _row_state(row: dict[str, Any]) -> str:
    if row.get("state"):
        return str(row["state"])
    if row.get("batch_id"):
        return "submitted"
    return str(row.get("status", "pending"))


def _set_row_state(row: dict[str, Any], state: str) -> None:
    row["state"] = state
    row["status"] = state


def _manifest_record_keys(out_dir: Path, row: dict[str, Any]) -> list[str]:
    keys = row.get("record_keys")
    if isinstance(keys, list):
        return [str(key) for key in keys]
    idmap_file = row.get("idmap_file")
    if idmap_file:
        try:
            idmap = json.loads((out_dir / str(idmap_file)).read_text(encoding="utf-8"))
            if not isinstance(idmap, dict):
                raise ValueError("idmap is not an object")
            keys = _unit_keys(idmap)
            if row.get("batch_id") and not keys:
                raise ValueError("idmap has no unit_key identities")
            row["record_keys"] = keys
            row["record_key_range"] = {
                "first": keys[0] if keys else None, "last": keys[-1] if keys else None,
                "count": len(keys),
            }
            return keys
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            if row.get("batch_id"):
                raise ValueError(f"cannot recover record identities for submitted batch {row['batch_id']}") from exc
    if row.get("batch_id"):
        raise ValueError(f"submitted batch {row['batch_id']} has no recorded unit_key identities")
    return []


def _matching_batches(existing: Sequence[dict[str, Any]], row: dict[str, Any]) -> list[dict[str, Any]]:
    created_at = _parse_api_time(row.get("created_at"))
    request_count = row.get("request_count")
    if created_at is None or request_count is None:
        raise ValueError("pending manifest row lacks created_at or request_count; cannot safely recover it")
    matches = []
    for batch in existing:
        batch_created = _parse_api_time(batch.get("created_at"))
        if (batch_created is not None and batch_created > created_at
                and _api_batch_request_count(batch) == int(request_count)):
            matches.append(batch)
    return matches


def _post_pending(row: dict[str, Any], manifest: dict[str, Any], manifest_path: Path,
                  out_dir: Path, model_id: str) -> None:
    if row.get("batch_id") is not None:
        raise ValueError("refusing to POST a manifest row that already has a batch_id")
    request_path = out_dir / str(row.get("request_file", ""))
    if not request_path.is_file():
        raise ValueError(f"pending request file is missing: {request_path}")
    body = request_path.read_bytes()
    expected_sha = str(row.get("request_sha256", ""))
    actual_sha = hashlib.sha256(body).hexdigest()
    if not expected_sha or actual_sha != expected_sha:
        raise ValueError("pending request file SHA does not match the manifest")
    _set_row_state(row, "pending")
    row["last_error"] = None
    _write_manifest(manifest_path, manifest, model_id)
    try:
        response = json.loads(api_call("POST", "/v1/messages/batches", body))
        batch_id = str(response["id"])
    except Exception as exc:
        row["last_error"] = f"{type(exc).__name__}: {exc}"
        _set_row_state(row, "failed-ambiguous")
        _write_manifest(manifest_path, manifest, model_id)
        raise
    if row.get("batch_id") is not None:
        raise ValueError("refusing to overwrite a non-null batch_id")
    row.update({"batch_id": batch_id,
                "processing_status": response.get("processing_status"),
                "last_error": None})
    _set_row_state(row, "submitted")
    _write_manifest(manifest_path, manifest, model_id)
    print(f"submitted batch {row['chunk_index']}: {batch_id} ({row['request_count']} requests, {len(body)} bytes)")


def submit(records_path: Path, out_dir: Path, dry_run: bool,
           model_id: str = MODEL, second_judge_sample_size: int = 1000,
           seed: int = 5601, force: bool = False,
           spend_cap_usd: float | None = None, recover: bool = False,
           cap_on_estimate: bool = False) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    with _manifest_lock(out_dir):
        return _submit_locked(records_path, out_dir, dry_run, model_id,
                              second_judge_sample_size, seed, force,
                              spend_cap_usd, recover, cap_on_estimate)


def _submit_locked(records_path: Path, out_dir: Path, dry_run: bool,
                   model_id: str, second_judge_sample_size: int,
                   seed: int, force: bool, spend_cap_usd: float | None,
                   recover: bool, cap_on_estimate: bool) -> int:
    if model_id not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported Anthropic judge model: {model_id}")
    offsets = _jsonl_offsets(records_path)
    if not offsets:
        raise ValueError("records file is empty")
    if model_id != MODEL and second_judge_sample_size <= 0:
        raise ValueError("second-judge sample size must be positive")
    selected = (random.Random(seed).sample(range(len(offsets)), min(second_judge_sample_size, len(offsets)))
                if model_id != MODEL and len(offsets) > second_judge_sample_size else list(range(len(offsets))))
    estimate = estimate_submission_cost(len(selected)) if model_id == MODEL else None
    worst_case = estimate_submission_cost(len(selected), MAX_TOKENS) if model_id == MODEL else None
    if estimate is None or worst_case is None:
        print(f"no configured cost estimate for {model_id}; {len(selected)} requests")
    else:
        print(f"likely submission cost (50 output tokens): ${estimate:.4f}; "
              f"worst case ({MAX_TOKENS} output tokens): ${worst_case:.4f} for {len(selected)} requests")
    if spend_cap_usd is not None and spend_cap_usd < 0:
        raise ValueError("--spend-cap-usd must be nonnegative")
    if spend_cap_usd is not None and (estimate is None or worst_case is None):
        raise ValueError(f"cannot enforce --spend-cap-usd: no configured estimate for {model_id}")
    cap_estimate = estimate if cap_on_estimate else worst_case
    if not dry_run and spend_cap_usd is not None and cap_estimate is not None and cap_estimate > spend_cap_usd:
        basis = "likely estimate" if cap_on_estimate else "worst-case estimate"
        raise ValueError(f"{basis} spend ${cap_estimate:.4f} exceeds spend cap ${spend_cap_usd:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "batches.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_models = set(manifest.get("models", []))
        if manifest.get("model"):
            existing_models.add(str(manifest["model"]))
        if not existing_models and manifest.get("batches"):
            existing_models = {str(row.get("model", MODEL)) for row in manifest["batches"]}
        if existing_models and model_id not in existing_models and not force:
            old_models = ", ".join(sorted(existing_models))
            raise ValueError(f"manifest model differs ({old_models} vs {model_id}); pass --force to append")
    else:
        manifest = {"model": model_id, "batches": []}
    manifest.setdefault("batches", [])
    for row in manifest["batches"]:
        _manifest_record_keys(out_dir, row)
        state = _row_state(row)
        if row.get("batch_id") and state != "submitted":
            _set_row_state(row, "submitted")
    uncertain = [row for row in manifest["batches"]
                 if not row.get("batch_id") and _row_state(row) in {"pending", "failed-ambiguous"}]
    if uncertain and not recover:
        raise RuntimeError("manifest contains pending or failed-ambiguous chunks; rerun with --recover")
    if uncertain:
        existing_batches = _list_existing_batches()
        for row in uncertain:
            matches = _matching_batches(existing_batches, row)
            if len(matches) > 1:
                ids = [str(batch.get("id")) for batch in matches]
                print(f"recovery refused for chunk {row.get('chunk_index')}: multiple matching batches {ids}")
                raise RuntimeError(f"multiple existing batches match chunk {row.get('chunk_index')}: {ids}")
            if len(matches) == 1:
                if row.get("batch_id") is not None:
                    raise ValueError("refusing to overwrite a non-null batch_id")
                row["batch_id"] = str(matches[0]["id"])
                row["processing_status"] = matches[0].get("processing_status")
                row["last_error"] = None
                _set_row_state(row, "submitted")
                _write_manifest(manifest_path, manifest, model_id)
                print(f"recovered chunk {row['chunk_index']}: adopted {row['batch_id']}")
            else:
                _set_row_state(row, "pending")
                _write_manifest(manifest_path, manifest, model_id)
                _post_pending(row, manifest, manifest_path, out_dir, model_id)

    submitted_keys = {key for row in manifest["batches"]
                      if row.get("batch_id") and _row_state(row) == "submitted"
                      and str(row.get("model") or manifest.get("model") or MODEL) == model_id
                      for key in _manifest_record_keys(out_dir, row)}
    selected_fresh: list[int] = []
    selected_seen: set[str] = set()
    for source_index in selected:
        record = _read_jsonl_at(records_path, offsets[source_index])
        key = record_identity(record, source_index)["unit_key"]
        if key not in submitted_keys and key not in selected_seen:
            selected_fresh.append(source_index)
            selected_seen.add(key)
    if len(selected_fresh) != len(selected):
        print(f"skipped {len(selected) - len(selected_fresh)} duplicate or already-submitted record identities")
    for batch_index, body, idmap, request_count in _serialize_record_chunks(
            records_path, offsets, selected_fresh, model_id):
        record_keys = _unit_keys(idmap)
        request_sha = hashlib.sha256(body).hexdigest()
        identity_sha = hashlib.sha256(json.dumps(record_keys, separators=(",", ":")).encode("utf-8")).hexdigest()
        tag = f"{batch_index:04d}-{request_sha[:12]}-{identity_sha[:12]}"
        request_file = out_dir / f"batch-{tag}-requests.json"
        idmap_file = out_dir / (f"idmap-dryrun-{tag}.json" if dry_run else f"idmap-{tag}.json")
        _write_bytes(request_file, body)
        _write_json(idmap_file, idmap)
        row = {
            "chunk_index": batch_index, "batch_number": batch_index, "batch_id": None,
            "model": model_id, "request_file": request_file.name,
            "request_sha256": request_sha, "idmap_file": idmap_file.name,
            "request_count": request_count, "record_keys": record_keys,
            "record_key_range": {"first": record_keys[0] if record_keys else None,
                                 "last": record_keys[-1] if record_keys else None,
                                 "count": len(record_keys)},
            "created_at": _pre_post_created_at(), "last_error": None, "dry_run": bool(dry_run),
        }
        if dry_run:
            _set_row_state(row, "dry_run")
            manifest["batches"].append(row)
            _write_manifest(manifest_path, manifest, model_id)
            print(f"dry run batch {batch_index}: wrote {request_count} requests ({len(body)} bytes) to {request_file}")
            continue
        _set_row_state(row, "pending")
        manifest["batches"].append(row)
        _write_manifest(manifest_path, manifest, model_id)
        _post_pending(row, manifest, manifest_path, out_dir, model_id)
    return 0


def show_status(batch_id: str) -> int:
    data = json.loads(api_call("GET", f"/v1/messages/batches/{batch_id}"))
    print(data.get("processing_status"), data.get("request_counts", {}))
    return 0


def _result_text(result: dict[str, Any]) -> str:
    message = result.get("message", {})
    return "".join(
        block.get("text", "") for block in message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _label_batch_key(row: dict[str, Any]) -> tuple[str, str] | None:
    if row.get("batch_id") is None or row.get("unit_key") is None:
        return None
    return str(row["batch_id"]), str(row["unit_key"])


def _jsonl_row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _merge_jsonl_file(path: Path, additions: Sequence[dict[str, Any]], key_fn: Any) -> None:
    """Stream the old file and append only unseen keys; retain keys, not full rows, in RAM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    keys: set[Any] = set()
    try:
        with temp.open("w", encoding="utf-8") as destination:
            if path.is_file():
                with path.open(encoding="utf-8") as source:
                    for line in source:
                        if line.strip():
                            existing = json.loads(line)
                            if isinstance(existing, dict):
                                key = key_fn(existing)
                                if key is not None:
                                    keys.add(key)
                        destination.write(line if line.endswith("\n") else line + "\n")
            for row in additions:
                key = key_fn(row)
                if key is not None and key in keys:
                    continue
                destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                if key is not None:
                    keys.add(key)
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def collect(out_dir: Path, selected_ids: Sequence[str] | None = None) -> int:
    with _manifest_lock(out_dir):
        return _collect_locked(out_dir, selected_ids)


def _collect_locked(out_dir: Path, selected_ids: Sequence[str] | None = None) -> int:
    manifest_path = out_dir / "batches.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = set(selected_ids or [])
    batch_rows = [row for row in manifest.get("batches", [])
                  if row.get("batch_id") and (not selected or row["batch_id"] in selected)]
    if selected and selected - {row.get("batch_id") for row in batch_rows}:
        raise ValueError("requested batch ID is not in this output directory's manifest")
    if not batch_rows:
        raise ValueError("no submitted batches are available to collect")

    labels_path = out_dir / "atlas-labels.jsonl"
    failures_path = out_dir / "parse-failures.json"
    try:
        failures = json.loads(failures_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        failures = []
    summary_path = out_dir / "collection-summary.json"
    try:
        previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        previous_summary = {}
    legacy_usage = manifest.get("collection_usage_legacy")
    if not isinstance(legacy_usage, dict):
        has_batch_usage = any(isinstance(row.get("usage_tokens"), dict) for row in manifest.get("batches", []))
        legacy_usage = (previous_summary.get("usage_tokens", {})
                        if previous_summary and not has_batch_usage else {})
        manifest["collection_usage_legacy_batches"] = (
            list(previous_summary.get("batches", [])) if legacy_usage else [])
        manifest["collection_usage_legacy"] = {
            key: int(legacy_usage.get(key, 0)) for key in ("input", "output", "cache_write", "cache_read")
        }
        _write_manifest(manifest_path, manifest, str(manifest.get("model") or MODEL))

    legacy_batch_ids = list(manifest.get("collection_usage_legacy_batches", []))
    for batch in batch_rows:
        batch_id = str(batch["batch_id"])
        idmap_path = out_dir / str(batch["idmap_file"])
        idmap = json.loads(idmap_path.read_text(encoding="utf-8"))
        result_lines: str | None = None
        last_get_error: Exception | None = None
        for attempt in range(2):
            try:
                result_lines = api_call("GET", f"/v1/messages/batches/{batch_id}/results")
                last_get_error = None
                break
            except Exception as exc:
                last_get_error = exc
                if attempt == 1:
                    raise
        if result_lines is None:
            raise RuntimeError(f"failed to fetch results for {batch_id}: {last_get_error}")
        batch_labels: list[dict[str, Any]] = []
        batch_failures: list[dict[str, Any]] = []
        batch_usage = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
        partial_line_errors = 0
        for line_number, line in enumerate(result_lines.splitlines(), 1):
            if not line.strip():
                continue
            try:
                envelope = json.loads(line)
            except json.JSONDecodeError as exc:
                partial_line_errors += 1
                batch_failures.append({"batch_id": batch_id, "line_number": line_number,
                                       "error": f"invalid or truncated results line: {exc.msg}"})
                continue
            if not isinstance(envelope, dict):
                partial_line_errors += 1
                batch_failures.append({"batch_id": batch_id, "line_number": line_number,
                                       "error": "results line is not an object"})
                continue
            custom_id = str(envelope.get("custom_id", ""))
            identity = idmap.get(custom_id)
            result = envelope.get("result", {})
            if not isinstance(identity, dict):
                batch_failures.append({"batch_id": batch_id, "custom_id": custom_id,
                                       "error": "custom_id missing from idmap"})
                continue
            if not isinstance(result, dict):
                partial_line_errors += 1
                batch_failures.append({"batch_id": batch_id, "custom_id": custom_id,
                                       "line_number": line_number, "error": "result is not an object"})
                continue
            if result.get("type") != "succeeded":
                failure = {**identity, "batch_id": batch_id,
                           "error": result.get("error", result.get("type", "unknown result"))}
                batch_failures.append(failure)
                batch_labels.append({**identity, "batch_id": batch_id, "parse_error": "request failed"})
                continue
            message = result.get("message", {})
            if not isinstance(message, dict) or not isinstance(message.get("content", []), list):
                partial_line_errors += 1
                batch_failures.append({"batch_id": batch_id, "custom_id": custom_id,
                                       "line_number": line_number, "error": "result message is malformed"})
                continue
            row_usage = message.get("usage", {})
            if not isinstance(row_usage, dict):
                partial_line_errors += 1
                batch_failures.append({"batch_id": batch_id, "custom_id": custom_id,
                                       "line_number": line_number, "error": "result usage is malformed"})
                continue
            try:
                batch_usage["input"] += int(row_usage.get("input_tokens", 0))
                batch_usage["output"] += int(row_usage.get("output_tokens", 0))
                batch_usage["cache_write"] += int(row_usage.get("cache_creation_input_tokens", 0))
                batch_usage["cache_read"] += int(row_usage.get("cache_read_input_tokens", 0))
            except (TypeError, ValueError):
                partial_line_errors += 1
                batch_failures.append({"batch_id": batch_id, "custom_id": custom_id,
                                       "line_number": line_number, "error": "result usage is malformed"})
                continue
            raw = _result_text(result)
            parsed = parse_label(raw)
            if parsed is None:
                batch_failures.append({**identity, "batch_id": batch_id,
                                        "error": "invalid JSON label", "response_prefix": raw[:1000]})
                batch_labels.append({**identity, "batch_id": batch_id, "parse_error": "invalid JSON label"})
            else:
                batch_labels.append({**identity, "batch_id": batch_id, **parsed})

        batch_digest = hashlib.sha256(batch_id.encode("utf-8")).hexdigest()[:12]
        batch_labels_path = out_dir / f"atlas-labels-batch-{batch_digest}.jsonl"
        _merge_jsonl_file(batch_labels_path, batch_labels, _label_batch_key)
        _merge_jsonl_file(labels_path, batch_labels, _label_batch_key)
        failures = [row for row in failures if not (
            row.get("batch_id") == batch_id and row.get("line_number") is not None)]
        failure_keys = {(row.get("batch_id"), row.get("unit_key"), row.get("custom_id"), row.get("error"),
                         row.get("line_number"))
                        for row in failures}
        for row in batch_failures:
            key = (row.get("batch_id"), row.get("unit_key"), row.get("custom_id"), row.get("error"),
                   row.get("line_number"))
            if key not in failure_keys:
                failures.append(row)
                failure_keys.add(key)
        _write_json(out_dir / f"parse-failures-batch-{batch_digest}.json", batch_failures)
        _write_json(failures_path, failures)
        batch.update({"usage_tokens": batch_usage, "results_partial": bool(partial_line_errors),
                      "partial_line_errors": partial_line_errors, "collected_at": _utc_now()})
        _write_manifest(manifest_path, manifest, str(manifest.get("model") or MODEL))
        known_usage_rows = [row for row in manifest.get("batches", [])
                            if row.get("batch_id") and isinstance(row.get("usage_tokens"), dict)]
        legacy_ids = set(legacy_batch_ids)
        all_usage = {key: int(manifest["collection_usage_legacy"].get(key, 0)) for key in
                     ("input", "output", "cache_write", "cache_read")}
        for row in known_usage_rows:
            if str(row["batch_id"]) in legacy_ids:
                continue
            for key in all_usage:
                all_usage[key] += int(row["usage_tokens"].get(key, 0))
        all_batch_ids = list(dict.fromkeys([
            *legacy_batch_ids,
            *(str(row["batch_id"]) for row in known_usage_rows),
        ]))
        batch_models = {str(row.get("model") or manifest.get("model") or MODEL)
                        for row in manifest.get("batches", []) if row.get("batch_id")
                        and row.get("usage_tokens") is not None}
        if legacy_batch_ids and previous_summary.get("model"):
            batch_models.add(str(previous_summary["model"]))
        result_model = next(iter(batch_models)) if len(batch_models) == 1 else ("mixed" if batch_models else
                         str(previous_summary.get("model", manifest.get("model") or MODEL)))
        cost_components = None
        if batch_models == {MODEL}:
            cost_components = {
                "input": all_usage["input"] * PRICES_PER_MILLION["input"] / 1_000_000,
                "output": all_usage["output"] * PRICES_PER_MILLION["output"] / 1_000_000,
                "cache_write": all_usage["cache_write"] * PRICES_PER_MILLION["cache_write"] / 1_000_000,
                "cache_read": all_usage["cache_read"] * PRICES_PER_MILLION["cache_read"] / 1_000_000,
            }
        partial_batches = [str(row["batch_id"]) for row in known_usage_rows if row.get("results_partial")]
        summary = {
            "model": result_model, "batches": all_batch_ids,
            "labels": _jsonl_row_count(labels_path), "parse_failures": len(failures),
            "usage_tokens": all_usage, "partial_results": bool(partial_batches),
            "partial_batches": partial_batches,
            "cost_usd": ({**cost_components, "total": sum(cost_components.values())}
                          if cost_components is not None else None),
            "prices_per_million_tokens": PRICES_PER_MILLION if cost_components is not None else None,
            "cost_note": (None if cost_components is not None else
                          "token usage reported; no price table configured for this model"),
        }
        _write_json(summary_path, summary)

    print(f"collected {_jsonl_row_count(labels_path)} labels; parse/request failures: {len(failures)}")
    all_usage = json.loads(summary_path.read_text(encoding="utf-8"))["usage_tokens"]
    print("tokens across collected batches:", all_usage)
    if cost_components is not None:
        print("cost USD by type:", {key: round(value, 6) for key, value in cost_components.items()},
              "total", round(sum(cost_components.values()), 6))
    else:
        print("cost estimate unavailable for model", result_model)
    return 0


def _read_labels(path: Path, field: str) -> dict[str, Any]:
    labels: dict[str, Any] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key, value = row.get("unit_key"), row.get(field)
            if not key or value is None or row.get("parse_error"):
                continue
            key = str(key)
            if key in labels:
                raise ValueError(f"{path}:{line_number}: duplicate unit_key {key}")
            labels[key] = value
    return labels


def cohen_kappa(primary: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    common = sorted(set(primary) & set(second))
    n = len(common)
    if n == 0:
        raise ValueError("the label files have no valid unit keys in common")
    values_a = [primary[key] for key in common]
    values_b = [second[key] for key in common]
    categories = sorted({str(value) for value in values_a + values_b})
    observed = sum(a == b for a, b in zip(values_a, values_b)) / n
    counts_a = {label: sum(str(value) == label for value in values_a) for label in categories}
    counts_b = {label: sum(str(value) == label for value in values_b) for label in categories}
    expected = sum(counts_a[label] * counts_b[label] for label in categories) / (n * n)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    confusion = {left: {right: 0 for right in categories} for left in categories}
    for a, b in zip(values_a, values_b):
        confusion[str(a)][str(b)] += 1
    return {
        "n_common": n,
        "n_primary_only": len(set(primary) - set(second)),
        "n_second_only": len(set(second) - set(primary)),
        "categories": categories,
        "observed_agreement": observed,
        "chance_agreement": expected,
        "cohen_kappa": kappa,
        "confusion_primary_rows_second_columns": confusion,
    }


def compare_labels(primary_path: Path, second_path: Path, out_path: Path) -> int:
    report = {
        "primary": str(primary_path), "second": str(second_path),
        "kind": cohen_kappa(_read_labels(primary_path, "kind"),
                            _read_labels(second_path, "kind")),
        "one_job": cohen_kappa(_read_labels(primary_path, "one_job"),
                               _read_labels(second_path, "one_job")),
        "interpretation": "Cohen's kappa is chance-corrected agreement on matched unit_key rows.",
    }
    _write_json(out_path, report)
    print(f"kind kappa: {report['kind']['cohen_kappa']}; "
          f"one_job kappa: {report['one_job']['cohen_kappa']}; "
          f"matched: {report['kind']['n_common']}; report: {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    submit_parser = commands.add_parser("submit", help="build and submit batches")
    submit_parser.add_argument("--records", type=Path, required=True)
    submit_parser.add_argument("--out", type=Path, required=True)
    submit_parser.add_argument("--dry-run", action="store_true",
                               help="write request JSON without calling the API")
    submit_parser.add_argument("--second-judge", metavar="MODEL_ID",
                               help="use another supported Anthropic model with the same rubric")
    submit_parser.add_argument("--second-judge-sample-size", type=int, default=1000,
                               help="fixed-seed record sample when --second-judge is selected")
    submit_parser.add_argument("--seed", type=int, default=5601)
    submit_parser.add_argument("--spend-cap-usd", type=float,
                               help="refuse a live submission if the selected cost bound exceeds this USD amount")
    submit_parser.add_argument("--cap-on-estimate", action="store_true",
                               help="apply --spend-cap-usd to the likely estimate instead of the MAX_TOKENS bound")
    submit_parser.add_argument("--recover", action="store_true",
                               help="list existing batches and reconcile pending or ambiguous chunks before posting")
    submit_parser.add_argument("--force", action="store_true",
                               help="append a different judge model to an existing output directory")
    status_parser = commands.add_parser("status", help="show one Message Batch status")
    status_parser.add_argument("batch_id")
    collect_parser = commands.add_parser("collect", help="collect results for submitted batches")
    collect_parser.add_argument("--out", type=Path, required=True)
    collect_parser.add_argument("--batch-id", action="append", dest="batch_ids")
    compare_parser = commands.add_parser("compare", help="write Cohen kappa for two label files")
    compare_parser.add_argument("--primary", type=Path, required=True)
    compare_parser.add_argument("--second", type=Path, required=True)
    compare_parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "submit":
        return submit(args.records, args.out, args.dry_run, args.second_judge or MODEL,
                      args.second_judge_sample_size, args.seed, args.force, args.spend_cap_usd,
                      args.recover, args.cap_on_estimate)
    if args.command == "status":
        return show_status(args.batch_id)
    if args.command == "collect":
        return collect(args.out, args.batch_ids)
    if args.command == "compare":
        return compare_labels(args.primary, args.second, args.out)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
