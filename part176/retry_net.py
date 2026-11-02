#!/usr/bin/env python3
"""Retry only reader errors that prove name resolution failed before a request was sent."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part171"), str(ROOT / "part172"),
              str(ROOT / "part173"), str(ROOT / "part174")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import label173
import label174
import run174

DEFAULT_ANSWERS = Path("results/part176/answers.jsonl")
DEFAULT_LABELS = Path("results/part176/labels")
DEFAULT_BUDGET_SECONDS = 2 * 60 * 60
DNS_EXCEPTION_PREFIX = re.compile(
    r"^(?:(?:socket\.)?gaierror\s*\(|(?:urllib\.error\.)?URLError\(\s*(?:socket\.)?gaierror\s*\()"
)


def is_pre_send_dns_error(error):
    """Recognize only the leading repr of a resolver exception, never body text."""
    return bool(DNS_EXCEPTION_PREFIX.match(str(error or "").lstrip()))


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_log(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
        handle.flush()


def _timestamp(now=None):
    now = dt.datetime.now(dt.timezone.utc) if now is None else now
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    return now.astimezone(dt.timezone.utc).isoformat(timespec="seconds")


def _parse_timestamp(value):
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=dt.timezone.utc) if parsed.tzinfo is None else parsed


def _elapsed_from_first_log(log_rows, reader, now, monotonic_elapsed=0.0):
    entries = [row for row in log_rows if row.get("reader") == reader]
    if not entries:
        return max(0.0, float(monotonic_elapsed))
    first = _parse_timestamp(entries[0].get("timestamp"))
    if first is not None:
        return max(0.0, (now - first).total_seconds())
    return (sum(float(row.get("duration_seconds", 0) or 0) +
                float(row.get("backoff_seconds", 0) or 0) for row in entries) +
            max(0.0, float(monotonic_elapsed)))


def _write_checkpoint(path, prior, by_id, ids):
    label174._write_checkpoint(path, prior, by_id, ids)


def retry(answers_path=DEFAULT_ANSWERS, labels_dir=DEFAULT_LABELS, reader="ds",
          max_seconds=DEFAULT_BUDGET_SECONDS, label_one=None, sleep=time.sleep,
          monotonic=time.monotonic, wall_clock=None):
    if reader not in label173.READERS:
        raise ValueError(f"unknown reader {reader!r}")
    answers_path, labels_dir = Path(answers_path), Path(labels_dir)
    labels_path = labels_dir / f"labels-{reader}.jsonl"
    log_path = labels_dir / "retry_net.jsonl"
    items = {str(item["id"]): item for item in label173.items_for(answers_path)}
    prior, by_id = label174._valid_file(labels_path)
    targets = [item_id for item_id, row in by_id.items()
               if item_id in items and not label173.valid(row)]
    updated = dict(by_id)
    label_one = label173.label_one if label_one is None else label_one
    old_log = read_jsonl(log_path)
    wall_clock = (lambda: dt.datetime.now(dt.timezone.utc)) if wall_clock is None else wall_clock
    total_budget = float(max_seconds)
    budget_start = monotonic()
    wall_start = wall_clock()
    elapsed_before = _elapsed_from_first_log(old_log, reader, wall_start)
    resolved = 0
    unresolved = 0
    truncated = False
    backoff = 1.0
    for item_id in targets:
        item_log = [row for row in old_log if str(row.get("item_id")) == item_id and
                    row.get("reader") == reader]
        sent_numbers = [int(row["attempt_number"]) for row in item_log
                        if row.get("request_sent") and row.get("attempt_number") is not None]
        sent_attempts = max(sent_numbers, default=(0 if is_pre_send_dns_error(
            updated[item_id].get("error")) else 1))
        dns_tries = max([int(row.get("pre_send_try_number", 0) or 0) for row in item_log] or [0])
        while not label173.valid(updated[item_id]) and sent_attempts < 2:
            current_error = updated[item_id].get("error")
            is_dns_before_call = is_pre_send_dns_error(current_error)
            now = wall_clock()
            elapsed = _elapsed_from_first_log(old_log, reader, now,
                                              max(0.0, monotonic() - budget_start))
            remaining = total_budget - elapsed
            if is_dns_before_call and remaining <= 0:
                truncated = True
                break
            started = monotonic()
            original_call = label173.judge_client.call_prompt

            def limited_call(*args, **kwargs):
                if is_dns_before_call:
                    kwargs["timeout"] = min(float(kwargs.get("timeout", 300)), max(0.1, remaining))
                return original_call(*args, **kwargs)

            try:
                label173.judge_client.call_prompt = limited_call
                row = label_one(items[item_id], reader, attempts=1)
            finally:
                label173.judge_client.call_prompt = original_call
            duration = max(0.0, monotonic() - started)
            updated[item_id] = row
            _write_checkpoint(labels_path, prior, updated, list(items))
            error = row.get("error")
            is_dns = is_pre_send_dns_error(error)
            request_sent = not bool(is_dns)
            if request_sent:
                sent_attempts += 1
            current_elapsed = _elapsed_from_first_log(old_log, reader, wall_clock(),
                                                       max(0.0, monotonic() - budget_start))
            remaining_after_call = max(0.0, total_budget - current_elapsed)
            planned_wait = min(backoff, 30.0, remaining_after_call) if is_dns else 0.0
            log_row = {"timestamp": _timestamp(wall_clock()), "reader": reader, "item_id": item_id,
                                  "error": error,
                                  "pre_send_dns_error": bool(is_dns),
                                  "request_sent": request_sent,
                                  "duration_seconds": duration, "backoff_seconds": planned_wait}
            if is_dns:
                dns_tries += 1
                log_row["pre_send_try_number"] = dns_tries
            else:
                log_row["attempt_number"] = sent_attempts
                log_row["attempt_id"] = f"{reader}:{item_id}:{sent_attempts}"
            append_log(log_path, log_row)
            old_log.append(log_row)
            if label173.valid(row):
                resolved += 1
                break
            if is_dns and planned_wait <= 0:
                truncated = True
                break
            if is_dns:
                sleep(planned_wait)
                backoff = min(30.0, backoff * 2.0)
        if truncated and is_pre_send_dns_error(updated[item_id].get("error")):
            break
    unresolved = sum(is_pre_send_dns_error(updated[item_id].get("error")) for item_id in targets)
    used = _elapsed_from_first_log(old_log, reader, wall_clock(),
                                   max(0.0, monotonic() - budget_start))
    result = {"reader": reader, "targets": len(targets), "resolved": resolved,
              "remaining_dns_errors": unresolved, "elapsed_seconds_this_call": max(0.0, monotonic() - budget_start),
              "elapsed_seconds_all_calls": used, "budget_seconds": total_budget,
              "budget_exhausted": bool(truncated or used >= total_budget),
              "status": "budget exhausted; remaining DNS errors count against reader cap"
              if unresolved and (truncated or used >= total_budget) else "complete",
              "retry_log": str(log_path)}
    print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader", choices=sorted(label173.READERS), required=True)
    parser.add_argument("--answers", default=str(DEFAULT_ANSWERS))
    parser.add_argument("--labels-dir", default=str(DEFAULT_LABELS))
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_BUDGET_SECONDS)
    args = parser.parse_args(argv)
    result = retry(args.answers, args.labels_dir, args.reader, args.max_seconds)
    return 1 if result["remaining_dns_errors"] and result["budget_exhausted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
