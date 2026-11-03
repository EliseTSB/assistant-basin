#!/usr/bin/env python3
"""Blind, resumable Part 177 window reader using Part 171's judge client."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part171"), str(ROOT / "part172"),
              str(ROOT / "part173"), str(ROOT / "part176")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

READERS = {
    "ds": {"model": "deepseek-v4.1-flash", "route": "ocgo", "session": "part177-ds-v4-1-flash"},
    "haiku": {"model": "claude-haiku-5-5", "route": "ocgo", "session": "part177-haiku-5-5"},
}
LABELS = ("asserted", "intermediate", "quoted from the problem", "rejected or doubted", "ambiguous")
MAX_TOKENS = 4000
ATTEMPTS_SENT = 2
READER_WORKERS = {"ds": 16, "haiku": 8}
CHECKPOINT_EVERY = 50
ERROR_LIMIT = 0.02
SECOND_READER_FRACTION = 0.10
SECOND_READER_SEED = 177
DNS_BUDGET_SECONDS = 2 * 60 * 60
DNS_EXCEPTION_PREFIX = re.compile(
    r"^(?:(?:socket\.)?gaierror\s*\(|(?:urllib\.error\.)?URLError\(\s*(?:socket\.)?gaierror\s*\()"
)
DEFAULT_OUT = ROOT / "results" / "part177"

READER_PROMPT = """Read the problem and a short excerpt from one attempted solution. The value is supplied for you to locate.
You do not know whether the value is right, and you must not solve the problem or decide the correct answer.
Classify what this text does with the supplied value. Return exactly one JSON object:
{"label":"asserted"|"intermediate"|"quoted from the problem"|"rejected or doubted"|"ambiguous"}.

Use asserted only when the text presents the value as its result or answer to the problem.
Use intermediate when it uses the value as a step, quantity, or candidate in working without presenting it as the result.
Use quoted from the problem when the excerpt only repeats a value supplied by the problem statement.
Use rejected or doubted when the text negates, corrects, rejects, or explicitly doubts the value.
Use ambiguous when the excerpt does not make the value's role clear, or conflicting uses cannot be resolved from the excerpt.
Do not infer correctness, do not solve, and do not treat mere proximity to a boxed answer as evidence.

Problem:
{problem}

Text window:
{window}

Value to locate:
{value}"""


def _judge_client():
    import judge_client
    return judge_client


def render(item):
    return READER_PROMPT.replace("{problem}", str(item.get("problem", ""))).replace(
        "{window}", str(item.get("window", ""))).replace("{value}", str(item.get("value", "")))


def parse(text):
    match = re.search(r"\{[^{}]*\}", str(text), flags=re.DOTALL)
    if not match:
        raise ValueError("reader response contained no JSON object")
    value = json.loads(match.group(0))
    label = value.get("label")
    if label not in LABELS:
        raise ValueError(f"reader returned invalid label {label!r}")
    return {"label": label}


def valid(row):
    return bool(row and "error" not in row and not row.get("pending_dns") and row.get("label") in LABELS)


def is_pre_send_dns_error(error):
    """Same strict pre-send resolver test used by Part 176 retry_net."""
    return bool(DNS_EXCEPTION_PREFIX.match(str(error or "").lstrip()))


def label_one(item, reader, client=None, sleep=time.sleep, monotonic=time.monotonic,
              attempts=ATTEMPTS_SENT, dns_budget_seconds=DNS_BUDGET_SECONDS):
    """Retry at most twice after a request was sent; DNS-only failures do not count as errors."""
    if reader not in READERS:
        raise ValueError(f"unknown reader {reader!r}")
    client = _judge_client() if client is None else client
    settings = READERS[reader]
    sent_attempts = 0
    dns_started = monotonic()
    backoff = 1.0
    last_error = "empty response"
    while sent_attempts < int(attempts):
        response = {}
        try:
            response = client.call_prompt(render(item), model=settings["model"], route=settings["route"],
                                          session=settings["session"], max_tokens=MAX_TOKENS, timeout=300)
            sent_attempts += 1
            answer = str(response.get("text", "")).strip()
            if answer:
                try:
                    parsed = parse(answer)
                    return {"id": str(item["id"]), "reader": reader, "attempts": sent_attempts, **parsed}
                except Exception as exc:
                    last_error = repr(exc)[:240]
            else:
                last_error = "empty response"
        except Exception as exc:
            last_error = repr(exc)[:240]
            if is_pre_send_dns_error(last_error):
                elapsed = max(0.0, monotonic() - dns_started)
                if elapsed >= float(dns_budget_seconds):
                    return {"id": str(item["id"]), "reader": reader, "attempts": sent_attempts,
                            "pending_dns": True, "dns_error": last_error, "dns_elapsed_seconds": elapsed}
                wait = min(backoff, 30.0, max(0.0, float(dns_budget_seconds) - elapsed))
                sleep(wait)
                backoff = min(30.0, backoff * 2.0)
                continue
            # Any non-DNS exception is conservatively treated as a request that reached the service.
            sent_attempts += 1
        if sent_attempts < int(attempts):
            sleep(2.0)
    return {"id": str(item["id"]), "reader": reader, "attempts": sent_attempts,
            "error": last_error}


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def _write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n"
                                        for row in rows), encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _valid_file(path):
    rows = read_jsonl(path)
    by_id = {}
    for row in rows:
        row_id = str(row.get("id"))
        if row_id in by_id:
            raise ValueError(f"duplicate reader label ID {row_id} in {path}")
        by_id[row_id] = row
    return rows, by_id


def _label_items(items, reader, path, label_one_fn=None, client=None):
    prior, by_id = _valid_file(path)
    pending = [item for item in items if str(item["id"]) not in by_id or not valid(by_id[str(item["id"])])]
    if not pending:
        return prior
    label_one_fn = label_one if label_one_fn is None else label_one_fn
    if label_one_fn is label_one:
        call = lambda item: label_one_fn(item, reader, client=client)
    else:
        call = lambda item: label_one_fn(item, reader)
    updated = dict(by_id)

    def flush():
        ordered = [str(item["id"]) for item in items if str(item["id"]) in updated]
        _write_rows(path, [updated[item_id] for item_id in ordered])

    with ThreadPoolExecutor(max_workers=READER_WORKERS[reader]) as pool:
        futures = {pool.submit(call, item): item for item in pending}
        for finished, future in enumerate(as_completed(futures), 1):
            updated[str(futures[future]["id"])] = future.result()
            if finished % CHECKPOINT_EVERY == 0:
                flush()
    flush()
    return read_jsonl(path)


def known_examples(path=None):
    path = Path(path or HERE / "known177.json")
    examples = list((read_json(path, {}) or {}).get("examples", []))
    if len(examples) != 20:
        raise ValueError("known177.json must contain exactly 20 synthetic examples")
    if set(row.get("expected") for row in examples) != set(LABELS):
        raise ValueError("known177.json must cover all five expected labels")
    return examples


def known_gate(rows, examples=None):
    examples = known_examples() if examples is None else list(examples)
    by_id = {str(row.get("id")): row for row in rows}
    details = []
    for example in examples:
        row = by_id.get(str(example["id"]))
        correct = bool(valid(row) and row["label"] == example["expected"])
        details.append({"id": example["id"], "expected": example["expected"],
                        "observed": row.get("label") if row else None, "correct": correct})
    right = sum(row["correct"] for row in details)
    return {"passed": right >= 17, "correct": right, "total": len(examples), "minimum": 17,
            "rows": details}


def second_reader_sample(windows, fraction=SECOND_READER_FRACTION, seed=SECOND_READER_SEED):
    ids = sorted(str(row["id"]) for row in windows)
    count = min(len(ids), max(1, round(len(ids) * float(fraction)))) if ids else 0
    rng = random.Random(int(seed))
    return {"ids": sorted(rng.sample(ids, count)) if count else [], "total": len(ids), "sample_n": count,
            "fraction": float(fraction), "seed": int(seed)}


def reader_error_report(rows, expected_ids):
    by_id = {str(row.get("id")): row for row in rows}
    expected = list(map(str, expected_ids))
    failed = [row_id for row_id in expected if row_id in by_id and not valid(by_id[row_id]) and
              not by_id[row_id].get("pending_dns")]
    pending_dns = [row_id for row_id in expected if row_id in by_id and by_id[row_id].get("pending_dns")]
    valid_n = sum(row_id in by_id and valid(by_id[row_id]) for row_id in expected)
    missing_n = sum(row_id not in by_id for row_id in expected)
    denominator = len(expected)
    unresolved_n = len(pending_dns) + missing_n
    total_nonpassing = len(failed) + unresolved_n
    error_rate = total_nonpassing / denominator if denominator else None
    return {"expected": denominator, "valid": valid_n, "errors": len(failed), "pending_dns": len(pending_dns),
            "missing": missing_n, "nonpassing": total_nonpassing, "error_rate": error_rate,
            "error_cap": ERROR_LIMIT,
            "passed": error_rate is not None and error_rate <= ERROR_LIMIT and unresolved_n == 0,
            "error_ids": failed, "pending_dns_ids": pending_dns, "missing_ids": [row_id for row_id in expected
                                                                                 if row_id not in by_id]}


def run(out_dir=DEFAULT_OUT, reader="ds", label_one_fn=None, client=None):
    if reader not in READERS:
        raise ValueError(f"unknown reader {reader!r}")
    out = Path(out_dir)
    read_result = read_json(out / "read.json", {}) or {}
    gates = read_result.get("gates_1_to_3", {})
    if not all(gates.get(name, {}).get("passed") for name in ("gate1", "gate2", "gate3")):
        metadata = {"reader": reader, "stopped": "Gate 1, 2, or 3 failed",
                    "gates_1_to_3": gates}
        write_json(out / "reader-metadata.json", metadata)
        return metadata
    windows_path = out / "windows.jsonl"
    windows = read_jsonl(windows_path)
    if not windows:
        raise FileNotFoundError(f"no windows found at {windows_path}; run read177.py first")
    if any(set(row) != {"id", "problem", "window", "value"} for row in windows):
        raise ValueError("windows.jsonl must contain only id, problem, window and value (no arm/key fields)")
    known = known_examples()
    known_path = out / "known-labels-ds.jsonl"
    if reader == "ds":
        known_rows = _label_items(known, "ds", known_path, label_one_fn=label_one_fn, client=client)
        known_result = known_gate(known_rows, known)
        if not known_result["passed"]:
            metadata = {"reader": reader, "known_gate": known_result, "stopped": "synthetic known-example gate failed"}
            write_json(out / "reader-metadata.json", metadata)
            return metadata
        label_path = out / "labels-ds.jsonl"
        rows = _label_items(windows, "ds", label_path, label_one_fn=label_one_fn, client=client)
        errors = reader_error_report(rows, [row["id"] for row in windows])
        sample_path = out / "second-reader-sample.json"
        if sample_path.is_file():
            sample = read_json(sample_path, {})
            if sample.get("total") != len(windows) or sample.get("seed") != SECOND_READER_SEED:
                raise ValueError("frozen second-reader sample does not match current window set or seed")
        else:
            sample = second_reader_sample(windows)
            write_json(sample_path, sample)
        unresolved = errors["pending_dns"] or errors["missing"]
        stopped = "reader labels remain missing or pending" if unresolved else \
            "reader error cap exceeded" if not errors["passed"] else None
        metadata = {"reader": reader, "known_gate": known_result, "first_reader_errors": errors,
                    "second_reader_sample": {key: value for key, value in sample.items() if key != "ids"},
                    "stopped": stopped}
        write_json(out / "reader-metadata.json", metadata)
        return metadata

    known_rows = read_jsonl(known_path)
    known_result = known_gate(known_rows, known)
    if not known_result["passed"]:
        raise RuntimeError("DeepSeek synthetic known-example gate must pass before the Haiku audit")
    sample_path = out / "second-reader-sample.json"
    if not sample_path.is_file():
        raise FileNotFoundError("second-reader-sample.json must be frozen before any Haiku call")
    sample = read_json(sample_path, {}) or {}
    selected = set(map(str, sample.get("ids", [])))
    by_id = {str(row["id"]): row for row in windows}
    if not selected.issubset(by_id):
        raise ValueError("frozen second-reader sample contains IDs absent from windows.jsonl")
    items = [row for row in windows if str(row["id"]) in selected]
    label_path = out / "labels-haiku.jsonl"
    rows = _label_items(items, "haiku", label_path, label_one_fn=label_one_fn, client=client)
    errors = reader_error_report(rows, [row["id"] for row in items])
    unresolved = errors["pending_dns"] or errors["missing"]
    stopped = "reader labels remain missing or pending" if unresolved else \
        "reader error cap exceeded" if not errors["passed"] else None
    metadata = {"reader": reader, "known_gate_ds": known_result, "second_reader_errors": errors,
                "second_reader_sample": {key: value for key, value in sample.items() if key != "ids"},
                "stopped": stopped}
    write_json(out / "reader-metadata.json", metadata)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reader", choices=tuple(READERS), default="ds")
    args = parser.parse_args(argv)
    result = run(args.out_dir, args.reader)
    print(json.dumps({key: value for key, value in result.items() if key != "known_gate"},
                     ensure_ascii=True, allow_nan=False), flush=True)
    return 1 if result.get("stopped") else 0


if __name__ == "__main__":
    raise SystemExit(main())
