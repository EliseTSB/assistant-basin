#!/usr/bin/env python3
"""Label Part 180b's blinded known answers with a reader, through Part 92's exact prompt.

The reader sees, per item, only the persona name and description, the question and the answer, filled into
part92/label92.py's PROMPT with nothing changed. It never sees the key, the category or the form.

Readers:
  ds     DeepSeek V4.1 Flash through judge_client's OpenCode Go route (Part 92b's and Part 118's transport),
         max_tokens 4,000 by default (Part 118 Amendment 3: a 500-token cap cuts its reasoning off before the JSON).
  haiku  Claude Haiku 5.5 through the Anthropic API with label92's own settings (max_tokens 500, no thinking
         parameter, key from ~/.anthropic-credits.env), the instrument Part 92 used.

Resumable: rows already in --out with a valid label are skipped; error rows are retried. Every attempt is recorded in
<out>.usage.jsonl (usage, stop reason, served model, error). --only restricts the run to the ids in a JSON file
(results/part180b/gate-ids.json for the gate). --dry-run writes the filled prompts to --out instead of calling anyone.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for folder in (ROOT / "part92", ROOT, HERE):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import label92  # noqa: E402  Part 92's registered prompt and response parser.

DS_MODEL = "deepseek-v4.1-flash"
DS_ROUTE = "ocgo"
DS_SESSION = "part180b-ds-v4-1-flash"
HAIKU_MODEL = label92.MODEL
HAIKU_MAX_TOKENS = 500
MAX_ATTEMPTS = 2
RETRY_SLEEP = 5.0
DEFAULT_WORKERS = {"ds": 8, "haiku": 8}
PUBLIC_FIELDS = ("id", "name", "desc", "q", "a")
_usage_lock = threading.Lock()


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: not a JSON object")
            rows.append(row)
    return rows


def read_items(path: Path) -> list[dict]:
    items = read_jsonl(path)
    seen = set()
    for item in items:
        if set(item) != set(PUBLIC_FIELDS):
            raise ValueError(f"item {item.get('id')!r} has fields {sorted(item)}; expected {list(PUBLIC_FIELDS)}")
        if item["id"] in seen:
            raise ValueError(f"duplicate item id {item['id']!r}")
        seen.add(item["id"])
    return items


def build_prompt(item: dict) -> str:
    """label92.PROMPT filled from the public fields only."""
    prompt = label92.PROMPT
    for field in ("name", "desc", "q", "a"):
        prompt = prompt.replace("{" + field + "}", str(item[field]))
    return prompt


def valid_label(row: dict) -> bool:
    return (
        "error" not in row
        and all(type(row.get(field)) is bool for field in ("shows_persona", "self_claim", "coherent"))
        and row.get("speaker") in ("ai", "human", "other", "none")
    )


def ds_transport(prompt: str, max_tokens: int, timeout: int) -> dict:
    import judge_client
    return judge_client.call_prompt(prompt, model=DS_MODEL, route=DS_ROUTE, session=DS_SESSION,
                                    max_tokens=max_tokens, timeout=timeout)


def haiku_transport(prompt: str, max_tokens: int, timeout: int, key: str | None = None) -> dict:
    """label92's call: the same model, cap and headers, with the key read from ~/.anthropic-credits.env."""
    secret = key if key is not None else label92.api_key()
    body = {"model": HAIKU_MODEL, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=json.dumps(body).encode("utf-8"),
        headers={"x-api-key": secret, "anthropic-version": "2023-06-01", "content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    text = "".join(part.get("text", "") for part in result.get("content", []) if part.get("type") == "text")
    return {"text": text, "usage": result.get("usage"), "stop_reason": result.get("stop_reason"),
            "served": result.get("model"), "reasoning_tokens": None}


def make_transport(reader: str, max_tokens: int, timeout: int) -> Callable[[str], dict]:
    if reader == "ds":
        return lambda prompt: ds_transport(prompt, max_tokens, timeout)
    if reader == "haiku":
        return lambda prompt: haiku_transport(prompt, max_tokens, timeout)
    raise ValueError(f"unknown reader {reader!r}")


def _record_usage(path: Path, item_id: str, reader: str, attempt: int, response: dict | None,
                  error: str | None) -> None:
    metadata = response or {}
    row = {"id": item_id, "reader": reader, "attempt": attempt, "usage": metadata.get("usage"),
           "stop_reason": metadata.get("stop_reason"), "served": metadata.get("served"),
           "reasoning_tokens": metadata.get("reasoning_tokens"), "error": error,
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    with _usage_lock, path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def label_one(item: dict, reader: str, transport: Callable[[str], dict], usage_path: Path,
              sleep_fn: Callable[[float], None] = time.sleep) -> dict:
    prompt = build_prompt(item)
    last_error = "unknown error"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = None
        try:
            response = transport(prompt)
            parsed = label92._parse(str(response.get("text", "")))
            _record_usage(usage_path, item["id"], reader, attempt, response, None)
            return {"id": item["id"], **parsed}
        except Exception as exc:  # the row keeps the error text; the key is never in it
            last_error = repr(exc)[:240]
            _record_usage(usage_path, item["id"], reader, attempt, response, last_error)
            if attempt < MAX_ATTEMPTS:
                sleep_fn(RETRY_SLEEP)
    return {"id": item["id"], "error": last_error}


def latest_rows(path: Path) -> dict[str, dict]:
    latest = {}
    if path.exists():
        for row in read_jsonl(path):
            latest[str(row["id"])] = row
    return latest


def label_items(items: list[dict], output: Path, reader: str, workers: int,
                transport: Callable[[str], dict], sleep_fn: Callable[[float], None] = time.sleep) -> dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    usage_path = Path(str(output) + ".usage.jsonl")
    existing = latest_rows(output)
    pending = [item for item in items if item["id"] not in existing or not valid_label(existing[item["id"]])]
    output.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    started = time.monotonic()
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool, output.open("a", encoding="utf-8") as stream:
            futures = [pool.submit(label_one, item, reader, transport, usage_path, sleep_fn) for item in pending]
            for future in futures:
                row = future.result()
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                errors += "error" in row
    return {"reader": reader, "items": len(items), "pending": len(pending), "error_rows": errors,
            "elapsed_seconds": round(time.monotonic() - started, 3), "output": str(output),
            "usage_sidecar": str(usage_path)}


def dry_run(items: list[dict], output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for item in items:
            stream.write(json.dumps({"id": item["id"], "prompt": build_prompt(item)}, ensure_ascii=False) + "\n")
    return {"dry_run": True, "items": len(items), "output": str(output)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reader", choices=("ds", "haiku"), required=True)
    parser.add_argument("--items", default="results/part180b/items.jsonl")
    parser.add_argument("--out", required=True, help="label JSONL (appended; resumable)")
    parser.add_argument("--only", help="JSON file with {\"ids\": [...]}: label only these items")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-tokens", type=int, help="DS default 4000; Haiku default 500 (label92's cap)")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--limit", type=int, help="label only the first N pending items (a timed pilot)")
    parser.add_argument("--dry-run", action="store_true", help="write the filled prompts, call nothing")
    args = parser.parse_args(argv)
    items = read_items(repo_path(args.items))
    if args.only:
        wanted = set(json.loads(repo_path(args.only).read_text(encoding="utf-8"))["ids"])
        items = [item for item in items if item["id"] in wanted]
        missing = wanted - {item["id"] for item in items}
        if missing:
            parser.error(f"--only names {len(missing)} ids that are not in the items file")
    output = repo_path(args.out)
    if args.dry_run:
        print(json.dumps(dry_run(items, output)))
        return 0
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        existing = latest_rows(output)
        pending = [item for item in items if item["id"] not in existing or not valid_label(existing[item["id"]])]
        items = pending[:args.limit]
    max_tokens = args.max_tokens or (4000 if args.reader == "ds" else HAIKU_MAX_TOKENS)
    workers = args.workers or DEFAULT_WORKERS[args.reader]
    if workers < 1:
        parser.error("--workers must be positive")
    transport = make_transport(args.reader, max_tokens, args.timeout)
    result = label_items(items, output, args.reader, workers, transport)
    result["max_tokens"] = max_tokens
    result["workers"] = workers
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["error_rows"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
