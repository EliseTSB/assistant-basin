#!/usr/bin/env python3
"""Label Part 118 answers with the Part 92 reader or a seeded DS second read."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "part92"))
sys.path.insert(0, str(ROOT))

import label92  # noqa: E402  Reuse its registered prompt, model, credential helper and parser.
import judge_client  # noqa: E402  Reuse the repo's OpenCode Go transport for the DS read.

DS_MODEL = "deepseek-v4.1-flash"
DS_SEED = 11800
DS_SHARE = 0.15
MAX_ATTEMPTS = 2
_usage_lock = threading.Lock()


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with repo_path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row in {path}:{line_no}")
            rows.append(row)
    return rows


def cell_key(row: dict) -> tuple[str, str, float, int]:
    return (str(row["id"]), str(row["arm"]), round(float(row["alpha"]), 8),
            int(row["question_index"]))


def sort_key(key: tuple[str, str, float, int]) -> tuple:
    arm_order = {"neuron": 0, "random": 1}
    return key[0], arm_order.get(key[1], 9), key[2], key[3]


def sample_size(size: int) -> int:
    return max(1, round(DS_SHARE * size)) if size else 0


def sample_keys(keys) -> list[tuple[str, str, float, int]]:
    ordered = sorted(set(keys), key=sort_key)
    return random.Random(DS_SEED).sample(ordered, sample_size(len(ordered)))


def build_prompt(item: dict) -> str:
    """Fill the exact Part 92 prompt with the unit persona, question and answer."""
    prompt = label92.PROMPT
    for field in ("name", "desc", "q", "a"):
        prompt = prompt.replace("{" + field + "}", str(item[field]))
    return prompt


def valid_label(row: dict) -> bool:
    return ("error" not in row
            and all(type(row.get(field)) is bool
                    for field in ("shows_persona", "self_claim", "coherent"))
            and row.get("speaker") in ("ai", "human", "other", "none"))


def _primary_transport(prompt: str, key: str) -> dict:
    """Part 92's Anthropic request shape, returning usage as well as response text."""
    body = {"model": label92.MODEL, "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}]}
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    text = "".join(part.get("text", "") for part in result.get("content", [])
                   if part.get("type") == "text")
    return {"text": text, "usage": result.get("usage"),
            "served": result.get("model"), "stop_reason": result.get("stop_reason"),
            "reasoning_tokens": None}


def _ds_transport(prompt: str) -> dict:
    """Use Part 86's OpenCode Go transport and DS V4.1 Flash model route."""
    return judge_client.call_prompt(
        prompt,
        model=DS_MODEL,
        route="ocgo",
        session="part118-ds-v4-1-flash",
        # Amendment 3: DS V4.1 Flash reasons before its JSON; at 500 tokens 92 of 365 attempts stopped on
        # max_tokens with no JSON. 4,000 leaves room for the reasoning.
        max_tokens=4000,
        timeout=300,
    )


def _public(item: dict) -> dict:
    row = item["public"]
    return {"id": row["id"], "arm": row["arm"], "alpha": row["alpha"],
            "question_index": row["question_index"]}


def _record_usage(path: Path, item: dict, reader: str, attempt: int,
                  response: dict | None, error: str | None) -> None:
    metadata = response or {}
    row = {**_public(item), "reader": reader, "attempt": attempt,
           "usage": metadata.get("usage"),
           "reasoning_tokens": metadata.get("reasoning_tokens"),
           "stop_reason": metadata.get("stop_reason"),
           "served": metadata.get("served"), "error": error}
    path.parent.mkdir(parents=True, exist_ok=True)
    with _usage_lock, path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def _label_one(item: dict, reader: str, primary_key: str,
               usage_path: Path, request_fn=None, sleep_fn=time.sleep) -> dict:
    prompt = build_prompt(item)
    last_error = "unknown error"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = None
        try:
            if request_fn is not None:
                response = request_fn(reader, prompt, primary_key)
            elif reader == "primary":
                response = _primary_transport(prompt, primary_key)
            else:
                response = _ds_transport(prompt)
            parsed = label92._parse(str(response.get("text", "")))
            _record_usage(usage_path, item, reader, attempt, response, None)
            return {**_public(item), **parsed}
        except Exception as exc:
            # An empty key (the DS reader) must not be "redacted": str.replace("", x) inserts x between every character.
            last_error = (repr(exc).replace(primary_key, "[redacted]") if primary_key else repr(exc))[:240]
            _record_usage(usage_path, item, reader, attempt, response, last_error)
            if attempt < MAX_ATTEMPTS:
                sleep_fn(5)
    return {**_public(item), "error": last_error}


def make_items(answer_rows: list[dict], neuron_rows: list[dict]) -> list[dict]:
    by_id = {str(row["id"]): row for row in neuron_rows}
    items_by_key = {}
    for answer in answer_rows:
        arm = answer.get("arm")
        if arm == "baseline":
            continue
        if arm not in ("neuron", "random"):
            raise ValueError(f"unknown battery arm {arm!r}")
        unit_id = str(answer["id"])
        if unit_id not in by_id:
            raise ValueError(f"battery answer refers to unknown Part 92 unit {unit_id!r}")
        qi = int(answer["question_index"])
        if not 0 <= qi < 6:
            raise ValueError(f"question index outside the six-question battery: {qi}")
        key = cell_key(answer)
        if key in items_by_key:
            raise ValueError(f"duplicate answer cell {key}")
        unit = by_id[unit_id]
        items_by_key[key] = {
            "public": {"id": unit_id, "arm": arm, "alpha": float(answer["alpha"]),
                       "question_index": qi},
            "name": unit["persona"], "desc": unit["desc"],
            "q": answer.get("question", ""), "a": answer["answer"],
        }
    return [items_by_key[key] for key in sorted(items_by_key, key=sort_key)]


def _read_existing(path: Path) -> dict:
    found = {}
    if not path.exists():
        return found
    for row in read_jsonl(path):
        key = cell_key(row)
        found[key] = row  # On resume, the latest retry record is authoritative.
    return found


def label_answers(answers: str | Path, neurons: str | Path, output: str | Path,
                  usage_output: str | Path | None = None, reader: str = "primary",
                  workers: int = 8, request_fn=None, sleep_fn=time.sleep) -> dict:
    if reader not in ("primary", "ds"):
        raise ValueError("reader must be 'primary' or 'ds'")
    if workers < 1:
        raise ValueError("workers must be positive")
    answer_rows = read_jsonl(answers)
    with repo_path(neurons).open(encoding="utf-8") as stream:
        neuron_rows = json.load(stream)
    items = make_items(answer_rows, neuron_rows)
    if reader == "ds":
        selected = set(sample_keys(cell_key(item["public"]) for item in items))
        items = [item for item in items if cell_key(item["public"]) in selected]

    output_path = repo_path(output)
    usage_path = (repo_path(usage_output) if usage_output
                  else output_path.with_name(output_path.stem + ".usage" + output_path.suffix))
    existing = _read_existing(output_path)
    pending = [item for item in items
               if cell_key(item["public"]) not in existing
               or not valid_label(existing[cell_key(item["public"])])]
    primary_key = ""
    if pending and reader == "primary" and request_fn is None:
        # Delegates credential resolution to the Part 92 reader. The key is never printed or logged.
        primary_key = label92.api_key()
    if pending and reader == "ds" and request_fn is None \
            and not os.environ.get("OPENCODE_GO_API_KEY"):
        raise RuntimeError("OPENCODE_GO_API_KEY is not set")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool, output_path.open("a", encoding="utf-8") as stream:
            futures = [pool.submit(_label_one, item, reader, primary_key, usage_path,
                                  request_fn, sleep_fn) for item in pending]
            for future in futures:
                row = future.result()
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                errors += "error" in row
    result = {"reader": reader, "answer_cells": len(items), "pending": len(pending),
              "error_rows": errors, "output": str(output_path),
              "usage_sidecar": str(usage_path)}
    if reader == "ds":
        result.update({"sample_seed": DS_SEED, "sample_share": DS_SHARE})
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--neurons", default="results/part92/step2/neurons.json")
    parser.add_argument("--reader", choices=("primary", "ds"), default="primary")
    parser.add_argument("--out", required=True)
    parser.add_argument("--usage-out")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    label_answers(args.answers, args.neurons, args.out, args.usage_out,
                  args.reader, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
