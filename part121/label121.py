#!/usr/bin/env python3
"""Blindly label Part 121 answers with Part 92's single-persona rubric."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for directory in (HERE, ROOT / "part92", ROOT / "part92b", ROOT / "part119", ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import label92  # noqa: E402
import label92b  # noqa: E402
import label119  # noqa: E402 -- Part 119's 12 known examples and reader gate.

ITEM_SEED = 12101
LUNA_SEED = 12103
LUNA_MIN_SAMPLE = 33
MAX_ATTEMPTS = 2
DS_WORKERS = 8
LUNA_WORKERS = 4
_usage_lock = threading.Lock()


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path):
    with repo_path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: str | Path) -> list[dict]:
    source = repo_path(path)
    rows = []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {source}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row in {source}:{line_number}")
            rows.append(row)
    return rows


def answer_hash(question: str, answer: str) -> str:
    return hashlib.sha256(f"{question}\n{answer}".encode("utf-8")).hexdigest()[:16]


def latest_label_rows(rows: list[dict]) -> dict[tuple[str, int], dict]:
    latest = {}
    for row in rows:
        key = (str(row["item_id"]), int(row["question_index"]))
        if key not in latest or valid_label(row) or not valid_label(latest[key]):
            latest[key] = row
    return latest


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def opaque_id(row_id: str, pair: str, question_index: int) -> str:
    raw = f"part121:{ITEM_SEED}:{row_id}:{pair}:{question_index}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def slot_kinds() -> dict[str, str]:
    source = ROOT / "results/part112/score.json"
    if not source.is_file():
        return {}
    with source.open(encoding="utf-8") as stream:
        data = json.load(stream)
    return {str(unit_id): str(item.get("atlas_kind", "slot"))
            for unit_id, item in data.get("units", {}).items()}


def make_items(answer_rows: list[dict], pairs: list[dict], kinds: dict[str, str] | None = None) -> list[dict]:
    """Build private joins and public prompts with no row, pair, arm, or dose metadata."""
    by_pair = {str(pair["pair"]): pair for pair in pairs}
    kinds = slot_kinds() if kinds is None else kinds
    answers = {}
    for answer in answer_rows:
        row_id, qi = str(answer["row"]), int(answer["question_index"])
        if qi not in (0, 1):
            raise ValueError(f"Part 121 question index must be 0 or 1, found {qi}")
        if row_id in answers and qi in answers[row_id]:
            raise ValueError(f"duplicate answer for row {row_id!r}, question {qi}")
        answers.setdefault(row_id, {})[qi] = answer
    if len(answer_rows) != 310:
        raise ValueError(f"expected all 310 Part 121 answers before labeling, found {len(answer_rows)}")
    if sum(row_id == "none" for row_id in answers) != 1 or any(set(qs) != {0, 1} for qs in answers.values()):
        raise ValueError("answers must contain both registered questions for each row")

    items = []
    for answer in answer_rows:
        row_id, qi = str(answer["row"]), int(answer["question_index"])
        pair_ids = sorted(by_pair) if row_id == "none" else [str(answer.get("pair"))]
        for pair_id in pair_ids:
            if pair_id not in by_pair:
                raise ValueError(f"answer row {row_id!r} references unknown pair {pair_id!r}")
            slot = by_pair[pair_id]["slot"]
            kind = kinds.get(str(slot["id"]), str(slot.get("part112_kind", "slot")))
            content_hash = answer_hash(str(answer["question"]), str(answer["answer"]))
            public = {"item_id": opaque_id(row_id, pair_id, qi), "question_index": qi,
                      "answer_hash": content_hash}
            private = {"item_id": public["item_id"], "row_id": row_id,
                       "pair": pair_id, "question_index": qi, "answer_hash": content_hash}
            items.append({"public": public, "private": private,
                          "name": str(slot["persona"]),
                          "desc": f"Its kind: {kind}.",
                          "q": str(answer["question"]), "a": str(answer["answer"])})
    if len(items) != 330 or len({item["public"]["item_id"] for item in items}) != len(items):
        raise AssertionError(f"expected 330 unique blinded prompts, found {len(items)}")
    random.Random(ITEM_SEED).shuffle(items)
    return items


def build_prompt(item: dict) -> str:
    prompt = label92.PROMPT
    for field in ("name", "desc", "q", "a"):
        prompt = prompt.replace("{" + field + "}", str(item[field]))
    return prompt


def valid_label(row: dict) -> bool:
    return ("error" not in row and
            all(type(row.get(field)) is bool for field in ("shows_persona", "self_claim", "coherent")) and
            row.get("speaker") in ("ai", "human", "other", "none"))


def luna_sample(items: list[dict], deepseek_rows: list[dict]) -> list[dict]:
    """Select all DeepSeek-positive items and a seeded draw of current negatives."""
    if len(items) != 330 or len({item["public"]["item_id"] for item in items}) != len(items):
        raise ValueError(f"Luna sampling requires all 330 unique items, found {len(items)}")
    latest = latest_label_rows(deepseek_rows)
    labels = {}
    for item in items:
        public = item["public"]
        key = (str(public["item_id"]), int(public["question_index"]))
        label = latest.get(key)
        if not valid_label(label or {}) or label.get("answer_hash") != public["answer_hash"]:
            raise ValueError("DeepSeek labels are incomplete or do not match the current answers")
        labels[key] = label
    positives = [item for item in items
                 if labels[(str(item["public"]["item_id"]), int(item["public"]["question_index"]))]
                 ["shows_persona"]]
    target = min(len(items), max(2 * len(positives), LUNA_MIN_SAMPLE))
    positive_ids = {str(item["public"]["item_id"]) for item in positives}
    negative_items = [item for item in items if str(item["public"]["item_id"]) not in positive_ids]
    draw = random.Random(LUNA_SEED).sample(negative_items, target - len(positives))
    return positives + draw


def _record_usage(path: Path, item: dict, reader: str, attempt: int,
                  response: dict | None, error: str | None) -> None:
    metadata = response or {}
    row = {**item["public"], "reader": reader, "attempt": attempt,
           "usage": metadata.get("usage"), "stop_reason": metadata.get("stop_reason"),
           "served": metadata.get("served"), "reasoning_tokens": metadata.get("reasoning_tokens"),
           "error": error}
    path.parent.mkdir(parents=True, exist_ok=True)
    with _usage_lock, path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def _label_one(item: dict, reader: str, usage_path: Path, request_fn: Callable | None = None,
               sleep_fn: Callable[[float], None] = time.sleep) -> dict:
    prompt = build_prompt(item)
    last_error = "unknown reader error"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = None
        try:
            attempt_prompt = prompt if attempt == 1 else prompt + "\n\nOn this attempt, reply with the JSON object only."
            response = (request_fn or label92b._transport)(reader, attempt_prompt, "")
            parsed = label92._parse(str(response.get("text", "")))
            _record_usage(usage_path, item, reader, attempt, response, None)
            return {**item["public"], **parsed}
        except Exception as exc:
            last_error = repr(exc)[:240]
            _record_usage(usage_path, item, reader, attempt, response, last_error)
            if attempt < MAX_ATTEMPTS:
                sleep_fn(5)
    return {**item["public"], "error": last_error}


def _latest_rows(path: Path) -> dict[tuple[str, int], dict]:
    if not path.exists():
        return {}
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    latest = {}
    offset = 0
    for index, raw_line in enumerate(lines, 1):
        try:
            line = raw_line.decode("utf-8")
            if line.strip():
                row = json.loads(line)
                key = (str(row["item_id"]), int(row["question_index"]))
                if key not in latest or valid_label(row) or not valid_label(latest[key]):
                    latest[key] = row
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            is_torn_tail = (index == len(lines) and
                            not raw_line.endswith((b"\n", b"\r")))
            if not is_torn_tail:
                raise ValueError(f"invalid JSONL in {path}:{index}: {exc}") from exc
            print(f"skipping torn final JSONL line in {path}:{index}", file=sys.stderr, flush=True)
            with path.open("r+b") as stream:
                stream.truncate(offset)
            break
        offset += len(raw_line)
    return latest


def _ensure_line_ending(path: Path) -> None:
    if path.exists() and path.stat().st_size:
        with path.open("rb") as stream:
            stream.seek(-1, 2)
            ends_with_newline = stream.read(1) == b"\n"
        if not ends_with_newline:
            with path.open("ab") as stream:
                stream.write(b"\n")


def label_items(items: list[dict], output: Path, reader: str, workers: int | None = None,
                request_fn: Callable | None = None,
                sleep_fn: Callable[[float], None] = time.sleep) -> dict:
    if reader not in ("ds", "luna"):
        raise ValueError("reader must be 'ds' or 'luna'")
    workers = workers or (DS_WORKERS if reader == "ds" else LUNA_WORKERS)
    if workers < 1:
        raise ValueError("workers must be positive")
    latest = _latest_rows(output)
    pending = [item for item in items
               if not (valid_label(latest.get((str(item["public"]["item_id"]),
                                               int(item["public"]["question_index"])), {})) and
                       latest[(str(item["public"]["item_id"]),
                               int(item["public"]["question_index"]))].get("answer_hash") ==
                       item["public"]["answer_hash"])]
    usage_path = Path(str(output) + ".usage.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    if pending:
        _ensure_line_ending(output)
        with ThreadPoolExecutor(max_workers=workers) as pool, output.open("a", encoding="utf-8") as stream:
            futures = [pool.submit(_label_one, item, reader, usage_path, request_fn, sleep_fn) for item in pending]
            for future in futures:
                stream.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
                stream.flush()
    final = _latest_rows(output)
    errors = sum(not (valid_label(final.get((str(item["public"]["item_id"]),
                                             int(item["public"]["question_index"])), {})) and
                      final[(str(item["public"]["item_id"]),
                             int(item["public"]["question_index"]))].get("answer_hash") ==
                      item["public"]["answer_hash"]) for item in items)
    return {"reader": reader, "items": len(items), "pending": len(pending), "error_rows": errors,
            "retry_limit": MAX_ATTEMPTS, "output": str(output), "usage_sidecar": str(usage_path)}


def run_gate(reader: str, out: str | Path, workers: int | None = None,
             request_fn: Callable | None = None,
             sleep_fn: Callable[[float], None] = time.sleep) -> dict:
    """Run Part 119's known-example gate with its exact DS/Luna settings."""
    return label119.run_gate(reader, out, workers=workers, request_fn=request_fn, sleep_fn=sleep_fn)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", action="store_true", help="run Part 119's 12 known examples")
    parser.add_argument("--reader", choices=("ds", "luna"), required=True)
    parser.add_argument("--answers", default="results/part121/run/answers.jsonl")
    parser.add_argument("--pairs", default="results/part117/pairs.json")
    parser.add_argument("--out", required=True, help="known-gate or answer-label JSONL")
    parser.add_argument("--gate-labels", help="known-example labels to require before the main read")
    parser.add_argument("--labels-ds", default="results/part121/labels-ds.jsonl",
                        help="complete DeepSeek labels required to select Luna's sample")
    parser.add_argument("--keys-out", help="private item-to-row key; defaults to <out>.keys.jsonl")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    output = repo_path(args.out)
    if args.gate:
        result = run_gate(args.reader, output, args.workers)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if result["passed"] else 1

    gate_path = (repo_path(args.gate_labels) if args.gate_labels else
                 repo_path(f"results/part121/known-{args.reader}.jsonl"))
    try:
        label119.require_passing_gate(args.reader, gate_path)
    except ValueError as exc:
        print(f"main read refused: {exc}", file=sys.stderr, flush=True)
        return 1
    items = make_items(read_jsonl(args.answers), read_json(args.pairs))
    all_items = items
    sample_size = len(items)
    positive_count = None
    if args.reader == "luna":
        try:
            deepseek_rows = list(_latest_rows(repo_path(args.labels_ds)).values())
            latest_ds = latest_label_rows(deepseek_rows)
            items = luna_sample(items, deepseek_rows)
        except (OSError, ValueError) as exc:
            print(f"Luna read refused: {exc}", file=sys.stderr, flush=True)
            return 1
        sample_size = len(items)
        positive_count = sum(1 for item in all_items
                             if latest_ds[(str(item["public"]["item_id"]),
                                           int(item["public"]["question_index"]))]["shows_persona"])
    keys_path = repo_path(args.keys_out) if args.keys_out else Path(str(output) + ".keys.jsonl")
    write_jsonl(keys_path, [item["private"] for item in items])
    result = label_items(items, output, args.reader, args.workers)
    result["keys"] = str(keys_path)
    result["sample_seed"] = LUNA_SEED if args.reader == "luna" else None
    result["sample_size"] = sample_size
    result["deepseek_positive_count"] = positive_count
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["error_rows"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
