#!/usr/bin/env python3
"""Part 60 Haiku batch generator. All API calls are opt-in CLI actions."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import random
import re
import ssl
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
BASIN = Path(os.environ.get("PART60_BASIN", HERE.parent)).resolve()
DEFAULT_ROOT = BASIN / "results" / "part60"
PILOT = BASIN / "results" / "part56-pilot" / "pilot.jsonl"
BASE_GEN = BASIN / "results" / "part58" / "gen.all.jsonl"
BASE_GEN_SPLITS = (BASIN / "results" / "part58" / "gen.dev.jsonl", BASIN / "results" / "part58" / "gen.test.jsonl")
LABELS = BASIN / "results" / "part56-pilot" / "label" / "atlas-labels.jsonl"
MODEL = "claude-haiku-5-5"
SPLIT_SEED = 5801
UNITS_PER_TYPE = 250
BATCH_MAX_REQUESTS = 10_000
BATCH_MAX_BYTES = 190_000_000
PRICE_INPUT = 0.05
PRICE_OUTPUT = 0.25
# Both inherited rate pairs are estimates and are unverified for current pricing.
PRICE_INPUT_BATCH = PRICE_INPUT
PRICE_OUTPUT_BATCH = PRICE_OUTPUT
PRICE_INPUT_DIRECT = 0.10
PRICE_OUTPUT_DIRECT = 0.50
MAX_TOKENS = {"M1": 1100, "M2": 1100, "M3": 1600, "M4": 1100, "M5": 1100, "M6": 8000, "M7": 1100}
KNOWN = {(4, 3042), (19, 4826), (22, 1610), (21, 4400)}
UNREGISTERED_TEST_UNITS = {"L0:u190", "L4:u741", "L17:u1360"}


class RunnerError(RuntimeError):
    pass


class APIError(RunnerError):
    def __init__(self, status: int | None, message: str):
        super().__init__(message)
        self.status = status


def check_data_paths() -> None:
    required = (PILOT, BASE_GEN, *BASE_GEN_SPLITS, LABELS)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise RunnerError(
            "Part 60 data files were not found relative to this checkout. "
            f"Stage this build at {HERE} with results/ beside it, or set PART60_BASIN "
            "to the directory containing results/. Missing: " + ", ".join(str(p) for p in missing)
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RunnerError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


@contextlib.contextmanager
def writer_lock(root: Path):
    """Serialize every Part 60 manifest, ledger, and generated-output writer."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".writer.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def key_for(record: dict[str, Any]) -> tuple[str, str]:
    layer = int(record["layer"])
    if "unit" in record:
        return f"L{layer}:u{int(record['unit'])}", "neuron"
    if "control_id" in record:
        return f"L{layer}:c{int(record['control_id'])}", "control"
    raise RunnerError("pilot record has no unit or control_id")


def load_splits(pilot_path: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """Rebuild Part 58's dev sample and the registered 247-real/250-control test."""
    pilot_path = pilot_path or PILOT
    records = read_jsonl(pilot_path)
    eligible: dict[str, list[tuple[int, dict[str, Any]]]] = {"unit": [], "control_id": []}
    for i, row in enumerate(records):
        if ("unit" in row and (int(row["layer"]), int(row["unit"])) in KNOWN):
            continue
        if any(len(row.get(section, [])) < count for section, count in (("top", 20), ("random", 10), ("weak", 5))):
            continue
        for kind in eligible:
            if kind in row:
                eligible[kind].append((i, row))
    rng = random.Random(SPLIT_SEED)
    out: dict[str, list[dict[str, Any]]] = {"dev": [], "test": []}
    excluded_test_count = 0
    for kind in ("unit", "control_id"):
        pool = eligible[kind]
        rng.shuffle(pool)
        for split, section in (("dev", pool[:UNITS_PER_TYPE]), ("test", pool[UNITS_PER_TYPE:2 * UNITS_PER_TYPE])):
            for index, record in section:
                key, unit_type = key_for(record)
                if split == "test" and unit_type == "neuron" and key in UNREGISTERED_TEST_UNITS:
                    excluded_test_count += 1
                    continue
                out[split].append({
                    "record_index": int(index), "unit_key": key, "type": unit_type,
                    "layer": int(record["layer"]), "record": record,
                })
    expected_counts = {"dev": {"neuron": 250, "control": 250},
                       "test": {"neuron": 250 - excluded_test_count, "control": 250}}
    for split in out:
        counts = Counter(x["type"] for x in out[split])
        if dict(counts) != expected_counts[split]:
            raise RunnerError(f"Part 58 split reconstruction yielded {dict(counts)} {split} units, expected {expected_counts[split]}")
        if len({x["unit_key"] for x in out[split]}) != len(out[split]):
            raise RunnerError(f"duplicate unit keys in {split} split")
    return out


def marked_token(window: str) -> str:
    match = re.search(r"\[\[(.*?)\]\]", window, flags=re.S)
    return match.group(1) if match else ""


def marked_user(windows: list[str], phase: str | None = None, extra: dict[str, Any] | None = None) -> str:
    tokens = [marked_token(window) for window in windows]
    counts = [[token, count] for token, count in Counter(tokens).most_common(12)]
    value: dict[str, Any] = {"marked_token_counts": counts, "marked_tokens": tokens, "windows": windows}
    if phase:
        value["phase"] = phase
    if extra:
        value.update(extra)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _shuffle_seed(unit: dict[str, Any], method: str, phase: str) -> int:
    raw = f"part60-window-order-v1\0{unit['unit_key']}\0{method}\0{phase}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def windows_for(unit: dict[str, Any], method: str) -> list[str]:
    counts = (("top", 24), ("random", 10), ("weak", 6)) if method == "M7" else (("top", 12), ("random", 6), ("weak", 3))
    row = unit["record"]
    return [item["window"] for section, n in counts for item in row.get(section, [])[:n]]


def m7_skip_reason(units: list[dict[str, Any]]) -> str | None:
    for unit in units:
        row = unit["record"]
        if len(row.get("top", [])) < 24 or len(row.get("random", [])) < 10 or len(row.get("weak", [])) < 6:
            return "pilot records do not all contain at least 24 top, 10 random, and 6 weak windows"
    return None


def generation_failure_exceeds_ten_percent(successes: int, expected: int) -> bool:
    return expected > 0 and (expected - successes) * 10 > expected


def parse_json(text: str) -> Any:
    raw = text.strip()
    if raw.startswith("```json"):
        raw = raw[7:].strip()
    elif raw.startswith("```"):
        raw = raw[3:].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _string_list(value: Any, count: int) -> bool:
    return isinstance(value, list) and len(value) == count and all(isinstance(x, str) and x.strip() for x in value)


def _normalized_sentence(text: str) -> str:
    text = re.sub(r"\[\[(.*?)\]\]", r"\1", text, flags=re.S)
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _generated_sentences(method: str, phase: str, value: dict[str, Any]) -> list[str]:
    if method == "M3" and phase == "selection":
        return [sentence for candidate in value["hypotheses"]
                for tier in ("strong", "none") for sentence in candidate[tier]]
    tiers = ("strong", "none") if method == "M3" and phase == "fresh" or method == "M4" or phase == "round2" else ("strong", "weak", "none")
    return [sentence for tier in tiers for sentence in value[tier]]


def _validate_sentence_rules(method: str, phase: str, value: dict[str, Any],
                             input_windows: Iterable[str] | None) -> tuple[bool, str]:
    seen: set[str] = set()
    source = {_normalized_sentence(str(window)) for window in (input_windows or [])}
    for sentence in _generated_sentences(method, phase, value):
        normalized = _normalized_sentence(sentence)
        if normalized in source:
            return False, "input_window_copy"
        if normalized in seen:
            return False, "duplicate_sentence"
        seen.add(normalized)
    return True, "ok"


def validate_output(method: str, phase: str, value: Any,
                    input_windows: Iterable[str] | None = None) -> tuple[bool, str]:
    if not isinstance(value, dict):
        return False, "not_object"
    if method == "M3" and phase == "selection":
        if set(value) != {"hypotheses"}:
            return False, "selection_keys"
        candidates = value.get("hypotheses")
        if not isinstance(candidates, list) or len(candidates) != 3:
            return False, "hypotheses_count"
        for rank, candidate in enumerate(candidates, 1):
            if (not isinstance(candidate, dict) or set(candidate) != {"rank", "hypothesis", "strong", "none"}
                    or isinstance(candidate.get("rank"), bool) or candidate.get("rank") != rank
                    or not isinstance(candidate.get("hypothesis"), str) or not candidate["hypothesis"].strip()):
                return False, f"candidate_{rank}_schema"
            if not _string_list(candidate.get("strong"), 3) or not _string_list(candidate.get("none"), 3):
                return False, f"candidate_{rank}_sentences"
        return _validate_sentence_rules(method, phase, value, input_windows)
    if method == "M3" and phase == "fresh":
        if set(value) != {"strong", "none"}:
            return False, "fresh_keys"
        return _validate_sentence_rules(method, phase, value, input_windows) if _string_list(value.get("strong"), 4) and _string_list(value.get("none"), 4) else (False, "fresh_sentences")
    if method == "M4":
        if set(value) != {"token", "context", "hypothesis", "strong", "none"}:
            return False, "token_keys"
        if not all(isinstance(value.get(k), str) and value[k].strip() for k in ("token", "context", "hypothesis")):
            return False, "token_schema"
        if not _string_list(value.get("strong"), 4) or not _string_list(value.get("none"), 4):
            return False, "tier_lengths"
        if any(len(re.findall(r"\[\[.*?\]\]", s)) != 1 for s in value["strong"]):
            return False, "strong_marker_count"
        if any("[[" in s or "]]" in s for s in value["none"]):
            return False, "none_has_marker"
        return _validate_sentence_rules(method, phase, value, input_windows)
    expected = ("hypothesis", "strong", "none") if phase == "round2" else ("hypothesis", "strong", "weak", "none")
    if set(value) != set(expected):
        return False, "output_keys"
    if not isinstance(value.get("hypothesis"), str) or not value["hypothesis"].strip():
        return False, "hypothesis"
    if not _string_list(value.get("strong"), 4) or not _string_list(value.get("none"), 4):
        return False, "tier_lengths"
    if "weak" in expected and not _string_list(value.get("weak"), 4):
        return False, "weak_length"
    return _validate_sentence_rules(method, phase, value, input_windows)


def request_for(method: str, phase: str, unit: dict[str, Any], index: int,
                output_row: dict[str, Any] | None = None,
                feedback: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    windows = windows_for(unit, method) if phase in ("primary", "selection", "round1") else None
    prompt_method = "M1" if method == "M5" and phase == "round1" else method
    system = (HERE / "prompts" / f"{prompt_method}.txt").read_text(encoding="utf-8").strip()
    if method == "M3" and phase == "fresh":
        if not output_row or not output_row.get("winner_hypothesis"):
            raise RunnerError(f"M3 fresh request lacks a selected winner for {unit['unit_key']}")
        user = json.dumps({"phase": "fresh", "hypothesis": output_row["winner_hypothesis"]}, ensure_ascii=False, separators=(",", ":"))
    elif method == "M5" and phase == "round2":
        if output_row is None or feedback is None:
            raise RunnerError("M5 round2 needs first-round output and scored yes/no feedback")
        first = output_row.get("round1", {})
        if first.get("parse_status") != "ok":
            raise RunnerError(f"M5 round1 failed to parse for {unit['unit_key']}")
        user = json.dumps({"phase": "round2", "previous_hypothesis": first["hypothesis"], "outcomes": feedback}, ensure_ascii=False, separators=(",", ":"))
    elif windows is not None:
        phase_name = "selection" if method == "M3" else "round1" if method == "M5" else None
        random.Random(_shuffle_seed(unit, method, phase)).shuffle(windows)
        user = marked_user(windows, phase=phase_name)
    else:
        raise RunnerError(f"unsupported request phase {method}/{phase}")
    prompt_phase = phase
    params: dict[str, Any] = {"model": MODEL, "max_tokens": MAX_TOKENS[method], "thinking": {"type": "disabled"},
                              "system": [{"type": "text", "text": system}], "messages": [{"role": "user", "content": user}]}
    if method == "M6":
        params["thinking"] = {"type": "adaptive"}
        params["output_config"] = {"effort": "high"}
    return {"custom_id": f"u{index:04d}", "params": params,
            "_meta": {"record_index": unit["record_index"], "unit_key": unit["unit_key"], "unit_type": unit["type"],
                      "layer": unit["layer"], "phase": prompt_phase}}


def clean_request(request: dict[str, Any]) -> dict[str, Any]:
    return {"custom_id": request["custom_id"], "params": request["params"]}


def output_path(root: Path, split: str, method: str) -> Path:
    return root / "out" / split / f"{method}.jsonl"


def read_output(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {int(row["record_index"]): row for row in rows}


def write_output(path: Path, rows: dict[int, dict[str, Any]]) -> None:
    write_jsonl_atomic(path, (rows[k] for k in sorted(rows)))


def _ssl_context() -> ssl.SSLContext:
    ca = os.environ.get("PART60_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
    return ssl.create_default_context(cafile=ca) if ca else ssl.create_default_context()


def api_call(method: str, path: str, body: dict[str, Any] | bytes | None = None) -> str:
    key = os.environ.get("ANTHROPIC_CREDITS_KEY")
    if not key:
        raise RunnerError("ANTHROPIC_CREDITS_KEY is not set")
    data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode() if isinstance(body, dict) else body
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    req = urllib.request.Request("https://api.anthropic.com" + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, context=_ssl_context(), timeout=120) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        # Keep user content and credentials out of logs; the error category is enough to debug API rejection.
        error_type = "unknown_error"
        try:
            detail = json.loads(exc.read().decode("utf-8"))
            error_type = str(detail.get("error", {}).get("type", error_type))[:80]
        except Exception:
            pass
        raise APIError(exc.code, f"Anthropic API returned HTTP {exc.code} ({error_type})") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise APIError(None, f"Anthropic API transport failed: {type(exc).__name__}") from None


def append_spend(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def spend_state(path: Path) -> tuple[float, dict[str, dict[str, Any]]]:
    batches: dict[str, dict[str, Any]] = {}
    if path.exists():
        for row in read_jsonl(path):
            identity = row.get("chunk_id") or row.get("batch_id")
            if not identity:
                continue
            old = batches.setdefault(str(identity), {})
            if row.get("event") in ("reserve", "unknown-reserve", "settle"):
                old.update(row)
                if row.get("event") == "settle":
                    old["actual_usd"] = float(row["usd"])
            elif row.get("event") == "release":
                old.update(row)
                old["actual_usd"] = 0.0
    total = sum(float(v.get("actual_usd", v.get("reserved_usd", 0.0))) for v in batches.values())
    return total, batches


def chunk_identity(split: str, method: str, phase: str, chunk_index: int,
                   unit_set_sha256: str) -> str:
    return f"{split}:{method}:{phase}:{chunk_index}:{unit_set_sha256[:16]}"


def method_state_path(root: Path, split: str, method: str) -> Path:
    return root / "out" / split / f"{method}.state.json"


def _mark_method_dropped_locked(root: Path, method: str, reason: str,
                                splits: tuple[str, ...] = ("dev", "test"),
                                source: str = "runtime") -> None:
    for split in splits:
        state = {"method": method, "split": split, "status": "dropped", "reason": reason,
                 "source": source, "updated_at": utc_now()}
        write_json(method_state_path(root, split, method), state)
        rows = read_output(output_path(root, split, method))
        for unit in load_splits()[split]:
            rec = int(unit["record_index"])
            row = rows.get(rec, {"record_index": rec, "unit_key": unit["unit_key"], "type": unit["type"], "layer": unit["layer"]})
            row.update({"record_index": rec, "unit_key": unit["unit_key"], "type": unit["type"], "layer": unit["layer"]})
            row["parse_status"] = "skipped"
            row["raw_parse_status"] = "method_dropped"
            row["skip_reason"] = reason
            row.setdefault("usage_tokens", {"input": 0, "output": 0, "calls": 0})
            rows[rec] = row
        write_output(output_path(root, split, method), rows)


def mark_method_dropped(root: Path, method: str, reason: str,
                        splits: tuple[str, ...] = ("dev", "test"),
                        source: str = "runtime") -> None:
    with writer_lock(root):
        _mark_method_dropped_locked(root, method, reason, splits, source)


def _record_phase_skip(rows: dict[int, dict[str, Any]], unit: dict[str, Any],
                       method: str, phase: str, reason: str,
                       raw_status: str = "api_rejected") -> None:
    rec = int(unit["record_index"])
    row = rows.get(rec, {"record_index": rec, "unit_key": unit["unit_key"], "type": unit["type"], "layer": unit["layer"]})
    nested = {"record_index": rec, "unit_key": unit["unit_key"], "type": unit["type"], "layer": unit["layer"],
              "parse_status": "skipped", "raw_parse_status": raw_status, "skip_reason": reason,
              "usage_tokens": {"input": 0, "output": 0, "calls": 0}}
    if method == "M3" and phase == "fresh":
        row["fresh"] = nested
        row["parse_status"] = "fresh_skipped"
    elif method == "M5" and phase == "round2":
        row["round2"] = nested
        row["parse_status"] = "round2_skipped"
    else:
        row.update(nested)
    row.setdefault("phase_rejections", {})[phase] = {"reason": reason, "raw_status": raw_status}
    rows[rec] = row


def estimated_usd(requests: list[dict[str, Any]]) -> float:
    return estimated_request_cost(requests, PRICE_INPUT_BATCH, PRICE_OUTPUT_BATCH)


def estimated_request_cost(requests: list[dict[str, Any]], input_rate: float,
                           output_rate: float) -> float:
    # UTF-8 byte count is a conservative upper estimate for prompt tokens; max_tokens bounds output.
    in_tokens = sum(len(json.dumps(clean_request(req), ensure_ascii=False).encode("utf-8")) for req in requests)
    out_tokens = sum(int(req["params"]["max_tokens"]) for req in requests)
    return (in_tokens * input_rate + out_tokens * output_rate) / 1_000_000


def manifest_path(root: Path, split: str, method: str, phase: str) -> Path:
    return root / "batches" / split / method / phase / "manifest.json"


def _ensure_run_started_locked(root: Path) -> str:
    path = root / "run.json"
    if path.is_file():
        return str(json.loads(path.read_text(encoding="utf-8"))["started_at"])
    started_at = utc_now()
    write_json(path, {"started_at": started_at})
    return started_at


def _manifest_start_fallback(root: Path) -> str | None:
    starts = []
    for path in (root / "batches").glob("**/manifest.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("created_at")
            if isinstance(value, str):
                starts.append(value)
        except (OSError, ValueError, TypeError):
            continue
    return min(starts) if starts else None


def listing_batches(root: Path, api: Any = api_call) -> list[dict[str, Any]]:
    """Return batches since this run began; this listing never adopts or changes state."""
    run_path = root / "run.json"
    started = (json.loads(run_path.read_text(encoding="utf-8")).get("started_at")
               if run_path.is_file() else _manifest_start_fallback(root))
    if not started:
        return []
    try:
        started_at = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
    except ValueError:
        raise RunnerError("recorded run start is invalid; cannot filter the read-only listing") from None
    items: list[dict[str, Any]] = []
    url = "/v1/messages/batches?limit=100"
    seen_cursors: set[str] = set()
    while url:
        page = json.loads(api("GET", url))
        if not isinstance(page, dict):
            raise RunnerError("batch listing response is not an object")
        for batch in page.get("data", []):
            if not isinstance(batch, dict):
                continue
            created = batch.get("created_at")
            if not isinstance(created, str):
                continue
            try:
                created_at = datetime.fromisoformat(created.replace("Z", "+00:00"))
            except ValueError:
                continue
            if created_at >= started_at:
                items.append(batch)
        if not page.get("has_more"):
            break
        data = page.get("data") or []
        cursor = str(page.get("last_id") or (data[-1].get("id", "") if data else ""))
        if not cursor or cursor in seen_cursors:
            raise RunnerError("batch-list pagination did not advance")
        seen_cursors.add(cursor)
        url = f"/v1/messages/batches?limit=100&after_id={cursor}"
    return items


def build_feedback(root: Path, split: str, unit: dict[str, Any], output: dict[str, Any]) -> list[dict[str, Any]]:
    import numpy as np
    score_path = root / "acts" / split / "M5-round1.npz"
    threshold_path = root / "thresholds.json"
    if not score_path.exists() or not threshold_path.exists():
        raise RunnerError("M5 round2 needs M5-round1.npz and thresholds.json from score60")
    thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))["thresholds"]
    threshold = float(thresholds[unit["unit_key"]])
    first = output.get("round1", {})
    data = np.load(score_path, allow_pickle=False)
    sids = [str(x) for x in data["sids"]]
    vals = data["max"]
    row_by_sid = {sid: i for i, sid in enumerate(sids)}
    results = []
    for tier in ("strong", "weak", "none"):
        for j, sentence in enumerate(first.get(tier, [])):
            sid = f"{unit['record_index']}|M5|round1|{tier}|{j}"
            if sid not in row_by_sid:
                raise RunnerError(f"M5 feedback activation missing for {unit['unit_key']} {tier}[{j}]")
            value = float(vals[row_by_sid[sid], list(data['unit_keys']).index(unit['unit_key'])])
            results.append({"tier": tier, "sentence": sentence, "fired": bool(value > threshold)})
    if len(results) != 12:
        raise RunnerError(f"M5 feedback must contain 12 sentence outcomes, got {len(results)}")
    return results


def phase_eligible_units(units: list[dict[str, Any]], method: str, phase: str,
                         rows: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    if method == "M3" and phase == "fresh":
        return [u for u in units if rows.get(u["record_index"], {}).get("winner_hypothesis")]
    if method == "M5" and phase == "round2":
        return [u for u in units if rows.get(u["record_index"], {}).get("round1", {}).get("parse_status") == "ok"]
    return units


def _phase_result(row: dict[str, Any] | None, method: str, phase: str) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    if method == "M3" and phase == "selection":
        value = row.get("selection")
    elif method == "M3" and phase == "fresh":
        value = row.get("fresh")
    elif method == "M5" and phase == "round1":
        value = row.get("round1")
    elif method == "M5" and phase == "round2":
        value = row.get("round2")
    else:
        value = row
    return value if isinstance(value, dict) else None


def phase_units(units: list[dict[str, Any]], method: str, phase: str,
                rows: dict[int, dict[str, Any]], root: Path, split: str) -> list[dict[str, Any]]:
    eligible = phase_eligible_units(units, method, phase, rows)
    return [unit for unit in eligible
            if (_phase_result(rows.get(int(unit["record_index"])), method, phase) or {}).get("parse_status") != "ok"]


def _smoke_usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    return {"input": int(usage.get("input_tokens", 0)),
            "output": int(usage.get("output_tokens", 0)), "calls": 1}


def _smoke_thinking(response: dict[str, Any]) -> str:
    content = response.get("content", []) if isinstance(response, dict) else []
    return "".join(str(block.get("thinking", block.get("text", "")))
                   for block in content
                   if isinstance(block, dict) and block.get("type") == "thinking").strip()


def _smoke_answer(response: dict[str, Any]) -> str:
    content = response.get("content", []) if isinstance(response, dict) else []
    return "".join(str(block.get("text", "")) for block in content
                   if isinstance(block, dict) and block.get("type") == "text")


def _smoke_drop_state(root: Path, reason: str) -> None:
    _mark_method_dropped_locked(root, "M6", reason, source="smoke")


def _clear_smoke_drop(root: Path) -> None:
    for split in ("dev", "test"):
        path = method_state_path(root, split, "M6")
        if path.is_file():
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("status") == "dropped" and state.get("source") == "smoke":
                path.unlink()


def _smoke_history_path(root: Path) -> Path:
    return root / "out" / "smoke-M6-runs.jsonl"


def _smoke_consecutive_failed_checks(history: list[dict[str, Any]]) -> int:
    failed = 0
    for row in reversed(history):
        check = row.get("all_five_check")
        if check is None:
            continue
        if check == "failed":
            failed += 1
        else:
            break
    return failed


def _finish_smoke(root: Path, result: dict[str, Any], spend_path: Path,
                  smoke_key: str, usage: dict[str, int], reason: str | None = None) -> dict[str, Any]:
    if usage["calls"]:
        actual = (usage["input"] * PRICE_INPUT_DIRECT + usage["output"] * PRICE_OUTPUT_DIRECT) / 1_000_000
        event = {"event": "settle", "input_tokens": usage["input"], "output_tokens": usage["output"],
                 "calls": usage["calls"], "usd": round(actual, 8)}
    else:
        event = {"event": "release", "usd": 0.0, "reason": reason or result.get("reason", "no usage returned")}
    append_spend(spend_path, {**event, "chunk_id": smoke_key, "method": "M6", "split": "dev",
                              "phase": "smoke", "ts": utc_now()})
    result["usage_tokens"] = usage
    if reason:
        result["status"] = "error"
        result["reason"] = reason
        result["all_five_check"] = None
    else:
        passed = all(item["thinking_present"] and item["stop_reason"] != "max_tokens"
                     and item["answer_parse_status"] == "ok" for item in result["results"])
        result["thinking_count"] = sum(item["thinking_present"] for item in result["results"])
        result["thinking_fraction"] = result["thinking_count"] / result["required_units"]
        result["all_five_check"] = "passed" if passed else "failed"
        history = read_jsonl(_smoke_history_path(root)) if _smoke_history_path(root).is_file() else []
        failed_checks = _smoke_consecutive_failed_checks(history) + (0 if passed else 1)
        if passed:
            result["status"] = "pass"
            _clear_smoke_drop(root)
        elif failed_checks >= 3:
            result["status"] = "dropped"
            result["reason"] = f"M6 failed the all-five smoke check {failed_checks} consecutive times"
            _smoke_drop_state(root, result["reason"])
        else:
            result["status"] = "retry"
            result["reason"] = (f"M6 failed the all-five smoke check {failed_checks}/3 consecutive times; "
                                 "rerun smoke on fresh units")
    write_json(root / "out" / "smoke-M6.json", result)
    history_row = {key: result.get(key) for key in
                   ("attempt", "created_at", "unit_keys", "status", "all_five_check", "thinking_count", "reason")}
    path = _smoke_history_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(history_row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


def smoke_m6(root: Path, api: Any = api_call) -> dict[str, Any]:
    """Run up to three five-unit direct-call checks, using fresh units for each retry."""
    with writer_lock(root):
        _ensure_run_started_locked(root)
        history_path = _smoke_history_path(root)
        dev_units = load_splits()["dev"]
        if len(dev_units) < 5:
            raise RunnerError("M6 smoke requires at least five dev units")
        result: dict[str, Any] | None = None
        for _ in range(3):
            history = read_jsonl(history_path) if history_path.is_file() else []
            start = (len(history) * 5) % len(dev_units)
            units = [dev_units[(start + offset) % len(dev_units)] for offset in range(5)]
            created_at = utc_now()
            attempt_number = len(history) + 1
            requests = [request_for("M6", "primary", unit, i) for i, unit in enumerate(units)]
            result = {"method": "M6", "split": "dev", "created_at": created_at,
                      "attempt": attempt_number, "unit_keys": [u["unit_key"] for u in units],
                      "model": MODEL, "max_tokens": MAX_TOKENS["M6"],
                      "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"},
                      "required_units": len(units), "results": []}
            if api is api_call and not os.environ.get("ANTHROPIC_CREDITS_KEY"):
                result.update({"status": "error", "reason": "ANTHROPIC_CREDITS_KEY is not set"})
                write_json(root / "out" / "smoke-M6.json", result)
                return result

            reserve = estimated_request_cost(requests, PRICE_INPUT_DIRECT, PRICE_OUTPUT_DIRECT)
            spend_path = root / "spend.jsonl"
            spent, _ = spend_state(spend_path)
            cap = float(os.environ.get("PART60_CAP", "8"))
            if cap > 8.0 and os.environ.get("PART60_CAP_OVERRIDE") != "1":
                raise RunnerError("cap above $8 requires PART60_CAP_OVERRIDE=1 and an explicitly recorded override")
            if spent + reserve > cap + 1e-12:
                raise RunnerError(f"cap refused M6 smoke: settled/reserved ${spent:.4f} plus conservative estimate ${reserve:.4f} exceeds ${cap:.2f}")
            smoke_key = f"dev:M6:smoke:{attempt_number:04d}"
            append_spend(spend_path, {"event": "reserve", "chunk_id": smoke_key, "method": "M6", "split": "dev",
                                      "phase": "smoke", "reserved_usd": round(reserve, 8), "ts": created_at,
                                      "request_count": len(requests)})

            usage_total = {"input": 0, "output": 0, "calls": 0}
            for unit, request in zip(units, requests):
                try:
                    response = json.loads(api("POST", "/v1/messages", request["params"]))
                    if not isinstance(response, dict):
                        raise ValueError("smoke response is not a JSON object")
                    usage = _smoke_usage(response)
                    for key in usage_total:
                        usage_total[key] += usage[key]
                    thinking = _smoke_thinking(response)
                    answer = parse_json(_smoke_answer(response))
                    valid, parse_status = validate_output("M6", "primary", answer, windows_for(unit, "M6"))
                    stop_reason = response.get("stop_reason")
                    result["results"].append({"unit_key": unit["unit_key"], "thinking_chars": len(thinking),
                        "thinking_present": bool(thinking), "stop_reason": stop_reason,
                        "answer_parse_status": parse_status if not valid else "ok", "usage_tokens": usage})
                except (APIError, RunnerError, OSError, ValueError, KeyError, TypeError) as exc:
                    return _finish_smoke(root, result, spend_path, smoke_key, usage_total,
                                         f"M6 smoke request failed: {type(exc).__name__}: {exc}")
            result = _finish_smoke(root, result, spend_path, smoke_key, usage_total)
            if result["status"] != "retry":
                return result
        assert result is not None
        return result


def submit_phase(root: Path, split: str, method: str, phase: str,
                 api: Any = api_call, kill_after_saved: bool = False,
                 kill_after_post: bool = False) -> str:
    with writer_lock(root):
        _ensure_run_started_locked(root)
        return _submit_phase_locked(root, split, method, phase, api,
                                    kill_after_saved, kill_after_post)


def _submit_phase_locked(root: Path, split: str, method: str, phase: str,
                         api: Any, kill_after_saved: bool,
                         kill_after_post: bool = False) -> str:
    if method not in MAX_TOKENS:
        raise RunnerError("submit supports M1 through M7; M0 uses prepare")
    if split not in ("dev", "test"):
        raise RunnerError("split must be dev or test")
    valid_phase = {"M3": {"selection", "fresh"}, "M5": {"round1", "round2"}}.get(method, {"primary"})
    if phase not in valid_phase:
        raise RunnerError(f"unsupported phase {method}/{phase}; expected one of {sorted(valid_phase)}")
    if api is api_call and not os.environ.get("ANTHROPIC_CREDITS_KEY"):
        raise RunnerError("ANTHROPIC_CREDITS_KEY is not set; no reserve or chunk manifest was written")
    if method == "M6":
        smoke_path = root / "out" / "smoke-M6.json"
        if not smoke_path.is_file():
            raise RunnerError("M6 requires out/smoke-M6.json; run `gen60.py smoke --method M6` first")
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        if smoke.get("status") != "pass":
            reason = str(smoke.get("reason", "M6 smoke has not passed"))
            if smoke.get("status") == "dropped":
                return f"dropped M6: {reason}"
            raise RunnerError(f"M6 smoke status is {smoke.get('status', 'unknown')}; rerun smoke before submitting: {reason}")
    state_path = method_state_path(root, split, method)
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "dropped":
            return f"dropped {method}: {state.get('reason', 'method marked dropped')}"
    units = load_splits()[split]
    output_file = output_path(root, split, method)
    rows = read_output(output_file)
    if method == "M7":
        reason = m7_skip_reason(units)
        if reason:
            for unit in units:
                rows[unit["record_index"]] = {"record_index": unit["record_index"], "unit_key": unit["unit_key"],
                    "type": unit["type"], "layer": unit["layer"], "hypothesis": "", "strong": [], "weak": [], "none": [],
                    "raw_parse_status": "skipped", "parse_status": "skipped", "skip_reason": reason, "usage_tokens": {"input": 0, "output": 0, "calls": 0}}
            write_output(output_file, rows)
            return "skipped"
    all_units = units
    if method == "M3" and phase == "fresh":
        prior_successes = sum(rows.get(u["record_index"], {}).get("selection", {}).get("parse_status") == "ok" for u in all_units)
    elif method == "M5" and phase == "round2":
        prior_successes = sum(rows.get(u["record_index"], {}).get("round1", {}).get("parse_status") == "ok" for u in all_units)
    else:
        prior_successes = len(all_units)
    if method == "M3" and phase == "fresh" and generation_failure_exceeds_ten_percent(prior_successes, len(all_units)):
        reason = "selection output parsed for fewer than 90% of units; fresh call dropped by the registered failure rule"
        for unit in all_units:
            row = rows.get(unit["record_index"])
            if row is None:
                row = {"record_index": unit["record_index"], "unit_key": unit["unit_key"], "type": unit["type"], "layer": unit["layer"]}
            _record_phase_skip(rows, unit, method, phase, reason, "prior_failure_rate")
        write_output(output_file, rows)
        return "skipped second call: prior generation failure exceeded 10%"
    if method == "M5" and phase == "round2" and generation_failure_exceeds_ten_percent(prior_successes, len(all_units)):
        reason = "round 1 output parsed for fewer than 90% of units; round 2 dropped by the registered failure rule"
        for unit in all_units:
            row = rows.get(unit["record_index"])
            if row is None:
                row = {"record_index": unit["record_index"], "unit_key": unit["unit_key"], "type": unit["type"], "layer": unit["layer"]}
                rows[unit["record_index"]] = row
            _record_phase_skip(rows, unit, method, phase, reason, "prior_failure_rate")
        write_output(output_file, rows)
        return "skipped second call: prior generation failure exceeded 10%"
    if method == "M3" and phase == "fresh":
        selection_path = root / "acts" / split / "M3-selection.npz"
        if not selection_path.is_file():
            raise RunnerError("M3 fresh needs scored M3-selection.npz; score and return the selection pass first")
        _choose_m3_winners_locked(root, split)
        rows = read_output(output_file)
    eligible_units = phase_eligible_units(all_units, method, phase, rows)
    units = phase_units(all_units, method, phase, rows, root, split)
    if not units:
        return "nothing to submit"

    def make_requests(selected: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        made: list[dict[str, Any]] = []
        mapped: dict[str, dict[str, Any]] = {}
        for idx, unit in enumerate(selected):
            output = rows.get(unit["record_index"])
            feedback = build_feedback(root, split, unit, output) if method == "M5" and phase == "round2" else None
            req = request_for(method, phase, unit, idx, output_row=output, feedback=feedback)
            mapped[req["custom_id"]] = {k: req["_meta"][k] for k in
                ("record_index", "unit_key", "unit_type", "layer", "phase")}
            user_payload = json.loads(req["params"]["messages"][0]["content"])
            if isinstance(user_payload, dict) and isinstance(user_payload.get("windows"), list):
                mapped[req["custom_id"]]["input_windows"] = user_payload["windows"]
            made.append(req)
        return made, mapped

    plan_requests, _ = make_requests(eligible_units)
    requests, idmap = make_requests(units)
    if len(requests) > BATCH_MAX_REQUESTS:
        raise RunnerError("request chunk exceeds Anthropic's 10,000 request limit")
    plan_values = [clean_request(r) for r in plan_requests]
    request_values = [clean_request(r) for r in requests]
    plan_bytes = json.dumps(plan_values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request_bytes = json.dumps(request_values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(plan_bytes) > BATCH_MAX_BYTES or len(request_bytes) > BATCH_MAX_BYTES:
        raise RunnerError("request chunk exceeds the 190 MB local safety limit")
    plan_units_sha256 = hashlib.sha256("\n".join(u["unit_key"] for u in eligible_units).encode()).hexdigest()
    unit_set_sha256 = hashlib.sha256("\n".join(u["unit_key"] for u in units).encode()).hexdigest()
    plan_request_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    request_sha256 = hashlib.sha256(request_bytes).hexdigest()
    plan_fingerprint = hashlib.sha256(f"{method}\0{split}\0{phase}\0{plan_units_sha256}\0{plan_request_sha256}".encode()).hexdigest()
    mfpath = manifest_path(root, split, method, phase)
    if mfpath.exists():
        manifest = json.loads(mfpath.read_text(encoding="utf-8"))
        if (manifest.get("method"), manifest.get("split"), manifest.get("phase")) != (method, split, phase):
            raise RunnerError("manifest identity mismatch; refusing to reuse this run directory")
        saved_fingerprint = manifest.get("plan_fingerprint")
        if saved_fingerprint and saved_fingerprint != plan_fingerprint:
            raise RunnerError("request plan changed for this phase; refusing to submit overlapping units")
        chunks = manifest.get("chunks", [])
        if chunks:
            last = chunks[-1]
            if not saved_fingerprint and last.get("request_sha256") not in (plan_request_sha256, request_sha256, None):
                raise RunnerError("legacy manifest request plan differs; refusing to submit overlapping units")
            if last.get("status") in ("submitted", "collecting"):
                return f"batch {last.get('batch_id')} is already submitted; collect it before continuing"
            if last.get("status") in ("posting", "unknown", "ready", "rate_limited", "ambiguous"):
                last["status"] = "ambiguous"
                last["ambiguous_at"] = utc_now()
                manifest["ambiguous_attempts"] = int(manifest.get("ambiguous_attempts", 0)) + 1
                write_json(mfpath, manifest)
            if last.get("status") == "rejected":
                return f"phase submission rejected and recorded: {last.get('error', '')}"
            if last.get("status") == "dropped":
                return f"phase dropped by the registered failure rule: {last.get('error', '')}"
    else:
        manifest = {"version": 2, "method": method, "split": split, "phase": phase,
                    "created_at": utc_now(), "plan_fingerprint": plan_fingerprint,
                    "expected_unit_keys": [u["unit_key"] for u in eligible_units],
                    "expected_count": len(eligible_units), "chunks": []}
    reserve = estimated_usd(requests)
    spend_path = root / "spend.jsonl"
    spent, _ = spend_state(spend_path)
    cap = float(os.environ.get("PART60_CAP", "8"))
    if cap > 8.0 and os.environ.get("PART60_CAP_OVERRIDE") != "1":
        raise RunnerError("cap above $8 requires PART60_CAP_OVERRIDE=1 and an explicitly recorded override")
    if spent + reserve > cap + 1e-12:
        raise RunnerError(f"cap refused submission: settled/reserved ${spent:.4f} plus conservative estimate ${reserve:.4f} exceeds ${cap:.2f}")
    chunk_index = len(manifest.get("chunks", []))
    chunk_dir = mfpath.parent / f"chunk-{chunk_index:04d}"
    try:
        chunk_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise RunnerError(f"refusing to touch pre-existing chunk directory {chunk_dir}") from None
    request_path = chunk_dir / "requests.jsonl"
    with request_path.open("wb") as stream:
        for req in request_values:
            stream.write(json.dumps(req, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    write_json(chunk_dir / "idmap.json", idmap)
    chunk = {"index": chunk_index, "attempt": chunk_index + 1, "directory": str(chunk_dir),
             "request_count": len(requests), "unit_keys": [u["unit_key"] for u in units],
             "request_sha256": request_sha256, "unit_set_sha256": unit_set_sha256,
             "reserved_usd": round(reserve, 8), "cap_limit_usd": cap,
             "cap_override": cap > 8.0, "created_at": utc_now(),
             "status": "posting", "batch_id": None}
    manifest["chunks"].append(chunk)
    write_json(mfpath, manifest)
    spend_key = chunk_identity(split, method, phase, chunk_index, unit_set_sha256)
    append_spend(spend_path, {"event": "reserve", "chunk_id": spend_key, "method": method,
                              "split": split, "phase": phase, "chunk": chunk_index,
                              "reserved_usd": round(reserve, 8), "ts": utc_now(),
                              "request_count": len(requests)})
    write_json(mfpath, manifest)
    body_obj = {"requests": request_values}
    try:
        response = json.loads(api("POST", "/v1/messages/batches", body_obj))
        batch_id = str(response["id"])
        if kill_after_post:
            os.kill(os.getpid(), 9)
    except APIError as exc:
        if exc.status in (400, 401, 403, 404, 413):
            chunk["status"] = "rejected"
            chunk["error"] = str(exc)
            write_json(mfpath, manifest)
            append_spend(spend_path, {"event": "release", "chunk_id": spend_key, "method": method,
                                      "split": split, "phase": phase, "chunk": chunk_index,
                                      "usd": 0.0, "ts": utc_now(), "reason": str(exc)})
            for unit in units:
                _record_phase_skip(rows, unit, method, phase, str(exc))
            write_output(output_file, rows)
            return "rejected"
        chunk["status"] = "ambiguous"
        chunk["error"] = str(exc)
        chunk["ambiguous_at"] = utc_now()
        write_json(mfpath, manifest)
        raise RunnerError("ambiguous batch submission; reservation remains counted; rerun submit to create a new attempt") from None
    except BaseException as exc:
        chunk["status"] = "ambiguous"
        chunk["error"] = f"{type(exc).__name__}: {exc}"
        chunk["ambiguous_at"] = utc_now()
        write_json(mfpath, manifest)
        raise RunnerError("ambiguous batch submission; reservation remains counted; rerun submit to create a new attempt") from None
    chunk["batch_id"] = batch_id
    chunk["status"] = "submitted"
    chunk["submitted_at"] = utc_now()
    write_json(mfpath, manifest)
    if kill_after_saved:
        os.kill(os.getpid(), 9)
    return batch_id


def response_text(result: dict[str, Any]) -> tuple[str | None, dict[str, int], str | None, str | None]:
    value = result.get("result", {})
    if value.get("type") == "errored":
        return None, {}, "api_error", None
    if value.get("type") == "canceled":
        return None, {}, "canceled", None
    if value.get("type") == "expired":
        return None, {}, "expired", None
    message = value.get("message", {})
    content = message.get("content", []) if isinstance(message, dict) else []
    text = "".join(part.get("text", "") for part in content if part.get("type") == "text")
    usage = message.get("usage", {}) if isinstance(message, dict) else {}
    tokens = {"input": int(usage.get("input_tokens", 0)), "output": int(usage.get("output_tokens", 0)), "calls": 1}
    return text, tokens, None, str(message.get("stop_reason") or "") or None


def response_has_thinking(result: dict[str, Any]) -> bool:
    value = result.get("result", {})
    message = value.get("message", {}) if isinstance(value, dict) else {}
    content = message.get("content", []) if isinstance(message, dict) else []
    return any(isinstance(part, dict) and part.get("type") == "thinking"
               and str(part.get("thinking", part.get("text", ""))).strip() for part in content)


def _merge_result(method: str, phase: str, old: dict[str, Any], text: str | None,
                  usage: dict[str, int], api_status: str | None,
                  stop_reason: str | None = None,
                  input_windows: Iterable[str] | None = None) -> dict[str, Any]:
    common = {k: old[k] for k in ("record_index", "unit_key", "type", "layer") if k in old}
    value = parse_json(text or "")
    if api_status is not None:
        good, status = False, api_status
    elif stop_reason == "max_tokens":
        good, status = False, "max_tokens"
    else:
        good, status = validate_output(method, phase, value, input_windows)
    entry: dict[str, Any] = {"parse_status": "ok" if good else "failed", "raw_parse_status": status,
                             "usage_tokens": usage, "raw_response": text if text is not None else "",
                             "stop_reason": stop_reason}
    if good:
        entry.update(value)
    return {**common, **entry}


def collect_phase(root: Path, split: str, method: str, phase: str, api: Any = api_call,
                  max_polls: int = 12, poll_seconds: float = 5.0) -> str:
    with writer_lock(root):
        return _collect_phase_locked(root, split, method, phase, api, max_polls, poll_seconds)


def _collect_phase_locked(root: Path, split: str, method: str, phase: str, api: Any,
                          max_polls: int, poll_seconds: float) -> str:
    mfpath = manifest_path(root, split, method, phase)
    if not mfpath.exists():
        raise RunnerError(f"no manifest for {method}/{split}/{phase}")
    manifest = json.loads(mfpath.read_text(encoding="utf-8"))
    if not manifest.get("chunks"):
        raise RunnerError("manifest has no chunks")
    chunk = manifest["chunks"][-1]
    if chunk.get("status") in ("complete", "dropped"):
        return "phase already collected"
    if chunk.get("status") not in ("submitted", "collecting") or not chunk.get("batch_id"):
        raise RunnerError("chunk has no recorded batch id; refusing to collect or resubmit")
    batch_id = chunk["batch_id"]
    status: dict[str, Any] = {}
    for attempt in range(max_polls):
        status = json.loads(api("GET", f"/v1/messages/batches/{batch_id}"))
        if status.get("processing_status") == "ended":
            break
        if attempt + 1 < max_polls:
            time.sleep(poll_seconds)
    if status.get("processing_status") != "ended":
        raise RunnerError(f"batch {batch_id} not ready after {max_polls} status checks ({status.get('processing_status', 'unknown')}); rerun collect")
    raw_results = api("GET", f"/v1/messages/batches/{batch_id}/results")
    parsed_results = {}
    for line in raw_results.splitlines():
        if line.strip():
            obj = json.loads(line)
            parsed_results[str(obj["custom_id"])] = obj
    idmap = json.loads((Path(chunk["directory"]) / "idmap.json").read_text(encoding="utf-8"))
    output_file = output_path(root, split, method)
    rows = read_output(output_file)
    total_usage = {"input": 0, "output": 0, "calls": 0}
    truncations = 0
    m6_thinking_results = 0
    for custom_id, meta in idmap.items():
        result = parsed_results.get(custom_id)
        if result is None:
            text, usage, api_error, stop_reason = None, {}, "missing_batch_result", None
        else:
            text, usage, api_error, stop_reason = response_text(result)
        if stop_reason == "max_tokens":
            truncations += 1
        thinking_present = bool(result and response_has_thinking(result))
        if method == "M6" and thinking_present:
            m6_thinking_results += 1
        for k in ("input", "output", "calls"):
            total_usage[k] += int(usage.get(k, 0))
        rec_i = int(meta["record_index"])
        base = {"record_index": rec_i, "unit_key": meta["unit_key"], "type": meta["unit_type"], "layer": int(meta["layer"])}
        merged = _merge_result(method, phase, base, text, usage, api_error, stop_reason,
                               input_windows=meta.get("input_windows"))
        if method == "M6":
            merged["thinking_present"] = thinking_present
        old = rows.get(rec_i, base)
        attempt_history = dict(old.get("generation_attempts_by_phase", {}))
        phase_attempts = list(attempt_history.get(phase, []))
        phase_attempts.append({"attempt": int(chunk.get("attempt", int(chunk["index"]) + 1)),
                               "parse_status": merged["parse_status"],
                               "raw_parse_status": merged["raw_parse_status"],
                               "stop_reason": merged.get("stop_reason")})
        attempt_history[phase] = phase_attempts
        old["generation_attempts_by_phase"] = attempt_history
        usage_by_phase = dict(old.get("usage_by_phase", {}))
        if method == "M3" and phase == "selection":
            old["selection"] = merged
            if merged["parse_status"] == "ok":
                old["candidates"] = merged["hypotheses"]
            else:
                old["parse_status"] = "selection_failed"
            old["raw_parse_status"] = merged["raw_parse_status"]
        elif method == "M3" and phase == "fresh":
            old["fresh"] = merged
            if merged["parse_status"] == "ok":
                old["hypothesis"] = old["winner_hypothesis"]
                old["strong"] = merged["strong"]
                old["weak"] = []
                old["none"] = merged["none"]
                old["parse_status"] = "ok"
            else:
                old["parse_status"] = "fresh_failed"
            old["raw_parse_status"] = merged["raw_parse_status"]
        elif method == "M5" and phase == "round1":
            old["round1"] = merged
            old["hypothesis"] = merged.get("hypothesis", "")
            old["strong"] = merged.get("strong", [])
            old["weak"] = merged.get("weak", [])
            old["none"] = merged.get("none", [])
            old["parse_status"] = merged["parse_status"]
            old["raw_parse_status"] = merged["raw_parse_status"]
        elif method == "M5" and phase == "round2":
            old["round2"] = merged
            if merged["parse_status"] == "ok":
                old["hypothesis"] = merged["hypothesis"]
                old["strong"] = merged["strong"]
                old["weak"] = []
                old["none"] = merged["none"]
                old["parse_status"] = "ok"
            else:
                old["parse_status"] = "round2_failed"
            old["raw_parse_status"] = merged["raw_parse_status"]
        else:
            old.update(merged)
            old.update({"record_index": rec_i, "unit_key": base["unit_key"], "type": base["type"], "layer": base["layer"]})
        prior_usage = usage_by_phase.get(phase, {"input": 0, "output": 0, "calls": 0})
        usage_by_phase[phase] = {key: int(prior_usage.get(key, 0)) + int(usage.get(key, 0))
                                 for key in ("input", "output", "calls")}
        usage_total = {k: sum(int(v.get(k, 0)) for v in usage_by_phase.values()) for k in ("input", "output", "calls")}
        old["usage_by_phase"] = usage_by_phase
        old["usage_tokens"] = usage_total
        rows[rec_i] = old
    write_output(output_file, rows)
    actual = (total_usage["input"] * PRICE_INPUT + total_usage["output"] * PRICE_OUTPUT) / 1_000_000
    spend_key = chunk_identity(split, method, phase, int(chunk["index"]), str(chunk.get("unit_set_sha256", batch_id)))
    append_spend(root / "spend.jsonl", {"event": "settle", "chunk_id": spend_key, "batch_id": batch_id, "method": method, "split": split,
                                         "phase": phase, "chunk": chunk["index"], "input_tokens": total_usage["input"],
                                         "output_tokens": total_usage["output"], "usd": round(actual, 8), "ts": utc_now()})
    chunk["usage_tokens"] = total_usage
    chunk["actual_usd"] = round(actual, 8)
    chunk["truncations"] = truncations
    if method == "M6":
        chunk["thinking_count"] = m6_thinking_results
        chunk["thinking_fraction"] = m6_thinking_results / len(idmap) if idmap else None
    eligible = phase_eligible_units(load_splits()[split], method, phase, rows)
    expected_count = int(manifest.get("expected_count", len(eligible)))
    successful_count = sum((_phase_result(rows.get(int(unit["record_index"])), method, phase) or {}).get("parse_status") == "ok"
                           for unit in eligible)
    failed_count = max(0, expected_count - successful_count)
    chunk["failure_count_after_attempt"] = failed_count
    chunk["phase_failure_rate"] = failed_count / expected_count if expected_count else 0.0
    chunk["status"] = "complete"
    chunk["completed_at"] = utc_now()
    manifest["remaining_failed_count"] = failed_count
    manifest["max_tokens_truncations"] = int(manifest.get("max_tokens_truncations", 0)) + truncations
    if generation_failure_exceeds_ten_percent(successful_count, expected_count):
        chunk["status"] = "dropped"
        chunk["error"] = f"phase failed {failed_count}/{expected_count} requests, exceeding the registered 10% rule"
        manifest["phase_status"] = "dropped"
    else:
        manifest["phase_status"] = "retryable_failures" if failed_count else "complete"
    write_json(mfpath, manifest)
    suffix = (f"; thinking {m6_thinking_results}/{len(idmap)} ({m6_thinking_results/len(idmap):.1%})"
              if method == "M6" and idmap else "")
    truncation_suffix = f"; {truncations} max_tokens truncation(s)" if truncations else ""
    if manifest["phase_status"] == "dropped":
        return (f"collected {len(idmap)} requests; phase dropped with {failed_count}/{expected_count} failures "
                f"under the registered 10% rule; usage {total_usage['input']} in / {total_usage['output']} out; "
                f"${actual:.4f}{truncation_suffix}{suffix}")
    retry_suffix = f"; {failed_count} failed unit(s) remain for retry" if failed_count else ""
    return (f"collected {len(idmap)} requests; usage {total_usage['input']} in / {total_usage['output']} out; "
            f"${actual:.4f}{truncation_suffix}{retry_suffix}{suffix}")


def _choose_m3_winners_locked(root: Path, split: str) -> int:
    import numpy as np
    path = root / "acts" / split / "M3-selection.npz"
    thresholds_path = root / "thresholds.json"
    if not path.exists() or not thresholds_path.exists():
        raise RunnerError("M3 winner selection needs M3-selection.npz and thresholds.json")
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))["thresholds"]
    data = np.load(path, allow_pickle=False)
    sids = [str(x) for x in data["sids"]]
    unit_keys = [str(x) for x in data["unit_keys"]]
    maxes = data["max"]
    sidrow = {sid: i for i, sid in enumerate(sids)}
    col = {key: j for j, key in enumerate(unit_keys)}
    rows = read_output(output_path(root, split, "M3"))
    selected = 0
    for rec_i, row in rows.items():
        candidates = row.get("candidates", [])
        if row.get("selection", {}).get("parse_status") != "ok" or len(candidates) != 3:
            continue
        key = row["unit_key"]
        if key not in thresholds or key not in col:
            row["winner_status"] = "unscored"
            row["winner_skip_reason"] = "selection threshold or activation column missing"
            continue
        threshold = float(thresholds[key])
        ranks: list[tuple[float, int, dict[str, Any]]] = []
        for candidate in candidates:
            rank = int(candidate["rank"])
            strong = [float(maxes[sidrow[f"{rec_i}|M3|selection|candidate{rank}_strong|{j}"], col[key]]) > threshold for j in range(3)]
            none = [float(maxes[sidrow[f"{rec_i}|M3|selection|candidate{rank}_none|{j}"], col[key]]) > threshold for j in range(3)]
            score = sum(strong) / 3 - sum(none) / 3
            ranks.append((score, rank, candidate))
        score, rank, winner = sorted(ranks, key=lambda x: (-x[0], x[1]))[0]
        row["winner_rank"] = rank
        row["winner_hypothesis"] = winner["hypothesis"]
        row["selection_score"] = score
        selected += 1
    write_output(output_path(root, split, "M3"), rows)
    return selected


def choose_m3_winners(root: Path, split: str) -> int:
    with writer_lock(root):
        return _choose_m3_winners_locked(root, split)


def _prepare_m0_locked(root: Path, split: str) -> int:
    units = load_splits()[split]
    baseline_sids: set[str] = set()
    for source in (BASE_GEN, *BASE_GEN_SPLITS):
        if not source.exists():
            continue
        for row in read_jsonl(source):
            if row.get("arm") == "B" and "unit_i" in row and "text" in row and "sid" in row:
                baseline_sids.add(str(row["sid"]))
    clean_text_by_sid = {str(row["sid"]): str(row["text"]) for row in read_jsonl(BASE_GEN.parent / "sent.all.jsonl") if "sid" in row and "text" in row}
    if not baseline_sids or not clean_text_by_sid:
        raise RunnerError("existing Part 58 baseline texts are missing from gen.all/dev/test")
    base: dict[int, list[tuple[int, str]]] = {}
    for sid in baseline_sids:
        match = re.fullmatch(r"(\d+):B#(\d+)", sid)
        if match and sid in clean_text_by_sid:
            base.setdefault(int(match.group(1)), []).append((int(match.group(2)), clean_text_by_sid[sid]))
    labels = {}
    if LABELS.exists():
        for row in read_jsonl(LABELS):
            if "record_index" in row:
                labels[int(row["record_index"])] = row.get("concept", "")
    out = {}
    for unit in units:
        texts = [text for _, text in sorted(base.get(unit["record_index"], []))]
        if len(texts) != 4:
            out[unit["record_index"]] = {"record_index": unit["record_index"], "unit_key": unit["unit_key"],
                "type": unit["type"], "layer": unit["layer"], "hypothesis": labels.get(unit["record_index"], ""),
                "strong": texts[:4], "weak": [], "none": [], "parse_status": "failed", "raw_parse_status": "baseline_missing_or_incomplete",
                "usage_tokens": {"input": 0, "output": 0, "calls": 0}}
        else:
            out[unit["record_index"]] = {"record_index": unit["record_index"], "unit_key": unit["unit_key"],
                "type": unit["type"], "layer": unit["layer"], "hypothesis": labels.get(unit["record_index"], ""),
                "strong": texts, "weak": [], "none": [], "parse_status": "ok", "raw_parse_status": "reused_part58_baseline",
                "usage_tokens": {"input": 0, "output": 0, "calls": 0}}
    path = output_path(root, split, "M0")
    write_output(path, out)
    return sum(row["parse_status"] == "ok" for row in out.values())


def prepare_m0(root: Path, split: str) -> int:
    with writer_lock(root):
        return _prepare_m0_locked(root, split)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="materialize the existing M0 baseline")
    prep.add_argument("--method", choices=["M0"], required=True)
    prep.add_argument("--split", choices=["dev", "test"], required=True)
    submit = sub.add_parser("submit", help="submit at most one not-yet-submitted chunk")
    submit.add_argument("--method", choices=[f"M{i}" for i in range(1, 8)], required=True)
    submit.add_argument("--split", choices=["dev", "test"], required=True)
    submit.add_argument("--phase", default="primary", choices=["primary", "selection", "fresh", "round1", "round2"])
    submit.add_argument("--cap", type=float, default=8.0)
    smoke = sub.add_parser("smoke", help="run the required five-unit M6 thinking smoke")
    smoke.add_argument("--method", choices=["M6"], required=True)
    listing = sub.add_parser("listing", help="read-only list of batches created during this run")
    collect = sub.add_parser("collect", help="collect a finished batch without resubmitting")
    collect.add_argument("--method", choices=[f"M{i}" for i in range(1, 8)], required=True)
    collect.add_argument("--split", choices=["dev", "test"], required=True)
    collect.add_argument("--phase", default="primary", choices=["primary", "selection", "fresh", "round1", "round2"])
    choose = sub.add_parser("select-m3", help="choose M3 winner from selection activations")
    choose.add_argument("--split", choices=["dev", "test"], required=True)
    args = parser.parse_args(argv)
    try:
        check_data_paths()
        if args.command != "listing":
            args.root.mkdir(parents=True, exist_ok=True)
        if args.command == "prepare":
            print(f"M0 materialized {prepare_m0(args.root, args.split)} baseline records")
        elif args.command == "smoke":
            result = smoke_m6(args.root)
            print(json.dumps(result, sort_keys=True))
            return 0 if result["status"] == "pass" else 2
        elif args.command == "listing":
            batches = listing_batches(args.root)
            print(json.dumps({"count": len(batches), "batches": batches}, sort_keys=True))
        elif args.command == "submit":
            os.environ["PART60_CAP"] = str(args.cap)
            print(submit_phase(args.root, args.split, args.method, args.phase))
        elif args.command == "collect":
            print(collect_phase(args.root, args.split, args.method, args.phase))
        else:
            print(f"selected {choose_m3_winners(args.root, args.split)} M3 winners")
        return 0
    except (RunnerError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
