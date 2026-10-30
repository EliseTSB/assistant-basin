#!/usr/bin/env python3
"""Blindly label Part 123 answers with Part 92's two-reader route and hash binding."""
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

import fit123  # noqa: E402
import label92  # noqa: E402
import label92b  # noqa: E402 -- same DeepSeek/Luna routes as Part 121.
import label119  # noqa: E402 -- Part 119 known-example gate.
import run123  # noqa: E402

ITEM_SEED = 12301
LUNA_SEED = 12303
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


def read_jsonl(path: str | Path, *, optional: bool = False) -> list[dict]:
    source = repo_path(path)
    if optional and not source.is_file():
        return []
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = []
    data = source.read_bytes()
    lines = data.splitlines(keepends=True)
    for line_no, raw in enumerate(lines, 1):
        try:
            line = raw.decode("utf-8")
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL row is not an object")
                rows.append(row)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            torn = line_no == len(lines) and not raw.endswith((b"\n", b"\r"))
            if not torn:
                raise ValueError(f"invalid JSONL in {source}:{line_no}: {exc}") from exc
            print(f"skipping torn final JSONL line in {source}:{line_no}", file=sys.stderr, flush=True)
            break
    return rows


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _ensure_line_ending(path: Path) -> None:
    if path.exists() and path.stat().st_size:
        with path.open("rb") as stream:
            stream.seek(-1, 2)
            if stream.read(1) != b"\n":
                with path.open("ab") as output:
                    output.write(b"\n")


def answer_hash(question: str, answer: str) -> str:
    return hashlib.sha256(f"{question}\n{answer}".encode("utf-8")).hexdigest()[:16]


def opaque_id(row_id: str, unit_id: str, question_index: int) -> str:
    raw = f"part123:{ITEM_SEED}:{row_id}:{unit_id}:{question_index}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def valid_label(row: dict) -> bool:
    return ("error" not in row and
            all(type(row.get(field)) is bool for field in ("shows_persona", "self_claim", "coherent")) and
            row.get("speaker") in ("ai", "human", "other", "none"))


def latest_label_rows(rows: list[dict]) -> dict[tuple[str, int], dict]:
    latest = {}
    for row in rows:
        key = (str(row["item_id"]), int(row["question_index"]))
        if key not in latest or valid_label(row) or not valid_label(latest[key]):
            latest[key] = row
    return latest


def make_items(answer_rows: list[dict], units: list[dict], stage: str = "main", *,
               allow_missing: bool = False) -> list[dict]:
    """Create blinded persona-answer items; one plain or random answer can serve many targets."""
    if stage not in ("pilot", "main"):
        raise ValueError("stage must be 'pilot' or 'main'")
    current_units = units
    if stage == "pilot":
        by_id = {str(unit["id"]): unit for unit in units}
        current_units = [by_id[uid] for uid in run123.PILOT_IDS if uid in by_id]
        if len(current_units) != len(run123.PILOT_IDS):
            raise ValueError("pilot labeling requires p92-00 through p92-03")
    answer_map = {}
    for answer in answer_rows:
        key = (str(answer["row"]), int(answer["question_index"]))
        if key in answer_map:
            raise ValueError(f"duplicate answer for row and question {key}")
        answer_map[key] = answer
    layers = sorted({int(unit["layer"]) for unit in current_units})
    specs = run123.build_rows(current_units, layers)
    items = []
    for spec in specs:
        if spec["arm"] == "zero":
            continue
        if spec["arm"] in ("plain", "random"):
            targets = [unit for unit in current_units
                       if spec["arm"] == "plain" or int(unit["layer"]) == int(spec["layer"])]
        else:
            targets = [unit for unit in current_units if str(unit["id"]) == str(spec["unit_id"])]
        for target in targets:
            for question_index, question in enumerate(run123.QUESTIONS):
                answer = answer_map.get((str(spec["row"]), question_index))
                if answer is None and not allow_missing:
                    raise ValueError(f"answer row {spec['row']!r}, question {question_index} is missing")
                if answer is not None and str(answer.get("question")) != question:
                    if not allow_missing:
                        raise ValueError(f"answer row {spec['row']!r}, question {question_index} has the wrong question")
                    answer = None
                content_hash = (None if answer is None else
                                answer_hash(question, str(answer.get("answer", ""))))
                item_id = opaque_id(str(spec["row"]), str(target["id"]), question_index)
                public = {"item_id": item_id, "question_index": question_index,
                          "answer_hash": content_hash}
                private = {"item_id": item_id, "row_id": str(spec["row"]),
                           "unit_id": str(target["id"]), "arm": str(spec["arm"]),
                           "q": spec.get("q"), "direction": spec.get("direction"),
                           "question_index": question_index, "answer_hash": content_hash}
                items.append({"public": public, "private": private,
                              "name": str(target["persona"]), "desc": str(target.get("desc", "")),
                              "q": question, "a": "" if answer is None else str(answer.get("answer", ""))})
    if len({item["public"]["item_id"] for item in items}) != len(items):
        raise AssertionError("Part 123 reader item identifiers are not unique")
    random.Random(ITEM_SEED).shuffle(items)
    return items


def build_prompt(item: dict) -> str:
    prompt = label92.PROMPT
    for field in ("name", "desc", "q", "a"):
        prompt = prompt.replace("{" + field + "}", str(item[field]))
    return prompt


def luna_sample(items: list[dict], deepseek_rows: list[dict]) -> list[dict]:
    """Take all current DeepSeek positives and a seed-12303 draw of other items."""
    latest = latest_label_rows(deepseek_rows)
    labels = {}
    for item in items:
        public = item["public"]
        key = (str(public["item_id"]), int(public["question_index"]))
        label = latest.get(key)
        if not valid_label(label or {}) or label.get("answer_hash") != public["answer_hash"]:
            raise ValueError("DeepSeek labels are incomplete or do not match current answer hashes")
        labels[key] = label
    positives = [item for item in items
                 if labels[(str(item["public"]["item_id"]), int(item["public"]["question_index"]))]
                 ["shows_persona"]]
    target = min(len(items), max(2 * len(positives), LUNA_MIN_SAMPLE))
    positive_ids = {str(item["public"]["item_id"]) for item in positives}
    others = [item for item in items if str(item["public"]["item_id"]) not in positive_ids]
    draw = random.Random(LUNA_SEED).sample(others, target - len(positives))
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


def _label_one(item: dict, reader: str, usage_path: Path, request_fn: Callable | None,
               sleep_fn: Callable[[float], None]) -> dict:
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


def label_items(items: list[dict], output: Path, reader: str, workers: int | None = None,
                request_fn: Callable | None = None,
                sleep_fn: Callable[[float], None] = time.sleep) -> dict:
    if reader not in ("ds", "luna"):
        raise ValueError("reader must be 'ds' or 'luna'")
    workers = workers or (DS_WORKERS if reader == "ds" else LUNA_WORKERS)
    if workers < 1:
        raise ValueError("workers must be positive")
    existing_rows = read_jsonl(output, optional=True)
    run123._rewrite_torn_tail(Path(output), existing_rows)
    latest = latest_label_rows(existing_rows)
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
    final = latest_label_rows(read_jsonl(output, optional=True))
    errors = sum(not (valid_label(final.get((str(item["public"]["item_id"]),
                                             int(item["public"]["question_index"])), {})) and
                      final[(str(item["public"]["item_id"]),
                             int(item["public"]["question_index"]))].get("answer_hash") ==
                      item["public"]["answer_hash"]) for item in items)
    return {"reader": reader, "items": len(items), "pending": len(pending), "error_rows": errors,
            "retry_limit": MAX_ATTEMPTS, "output": str(output), "usage_sidecar": str(usage_path)}


def run_gate(reader: str, output: str | Path, workers: int | None = None,
             request_fn: Callable | None = None,
             sleep_fn: Callable[[float], None] = time.sleep) -> dict:
    """Run Part 119's known examples through the same Part 121 route and reader settings."""
    return label119.run_gate(reader, output, workers=workers, request_fn=request_fn,
                             sleep_fn=sleep_fn)


def require_gate(reader: str, path: str | Path) -> dict:
    known = label119.read_known()
    rows = read_jsonl(path)
    result = label119.gate_result(known, rows)
    if not result["passed"]:
        raise ValueError(f"known-example gate for {reader} did not pass ({result['correct']}/12; need 10)")
    return result


def main(argv: list[str] | None = None, *, request_fn: Callable | None = None,
         sleep_fn: Callable[[float], None] = time.sleep, validate_inventory: bool = True) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", action="store_true", help="run Part 119's 12 known examples")
    parser.add_argument("--reader", choices=("ds", "luna"), required=True)
    parser.add_argument("--stage", choices=("pilot", "main"), default="main")
    parser.add_argument("--answers", default="results/part123/main/answers.jsonl")
    parser.add_argument("--neurons", default="results/part92/step2/neurons.json")
    parser.add_argument("--kinds", default="results/part92/step2/kinds.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--gate-labels", help="known-example labels required before a main read")
    parser.add_argument("--labels-ds", default="results/part123/labels-ds.jsonl")
    parser.add_argument("--keys-out", help="private joins; defaults to <out>.keys.jsonl")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    output = repo_path(args.out)
    if args.gate:
        result = run_gate(args.reader, output, args.workers, request_fn, sleep_fn)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if result["passed"] else 1

    gate_path = (repo_path(args.gate_labels) if args.gate_labels else
                 repo_path(f"results/part123/known-{args.reader}.jsonl"))
    try:
        require_gate(args.reader, gate_path)
    except (OSError, ValueError) as exc:
        print(f"main read refused: {exc}", file=sys.stderr, flush=True)
        return 1

    units = fit123.selected_units(read_json(args.neurons), read_json(args.kinds))
    counts = {kind: sum(unit["kind"] == kind for unit in units) for kind in ("slot", "concept")}
    if validate_inventory and (len(units) != 39 or counts != {"slot": 21, "concept": 18}):
        raise ValueError(f"expected all 39 Part 92 units, found {len(units)} {counts}")
    answers = read_jsonl(args.answers)
    items = make_items(answers, units, args.stage)
    if validate_inventory and args.stage == "main" and len(items) != 4212:
        raise ValueError(f"expected 4,212 reader items, found {len(items)}")
    positive_count = None
    if args.reader == "luna":
        try:
            ds_rows = read_jsonl(args.labels_ds)
            latest_ds = latest_label_rows(ds_rows)
            selected = luna_sample(items, ds_rows)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"Luna read refused: {exc}", file=sys.stderr, flush=True)
            return 1
        positive_count = sum(valid_label(latest_ds.get((str(item["public"]["item_id"]),
                                                         int(item["public"]["question_index"])), {})) and
                             latest_ds[(str(item["public"]["item_id"]),
                                        int(item["public"]["question_index"]))]["shows_persona"]
                             for item in items)
        items = selected
    keys_path = repo_path(args.keys_out) if args.keys_out else Path(str(output) + ".keys.jsonl")
    write_jsonl(keys_path, [item["private"] for item in items])
    result = label_items(items, output, args.reader, args.workers, request_fn, sleep_fn)
    result["keys"] = str(keys_path)
    result["sample_seed"] = LUNA_SEED if args.reader == "luna" else None
    result["sample_size"] = len(items)
    result["deepseek_positive_count"] = positive_count
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["error_rows"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
