#!/usr/bin/env python3
"""Blindly label the registered Part 119 persona-pair reader sample."""
from __future__ import annotations

import argparse
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "part92b") not in sys.path:
    sys.path.insert(0, str(ROOT / "part92b"))

import label92b  # noqa: E402 -- shared DS/Luna transport and settings
import rows119  # noqa: E402

GATE_COUNT = 12
GATE_MIN = 10
MAX_ATTEMPTS = 2
DS_WORKERS = 8
LUNA_WORKERS = 4
LUNA_SEED = 11903
LUNA_COUNT = 30
_usage_lock = threading.Lock()


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: str | Path) -> list[dict]:
    path = repo_path(path)
    result = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row in {path}:{line_no}")
            result.append(row)
    return result


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_known(path: str | Path | None = None) -> list[dict]:
    known_path = Path(path) if path is not None else HERE / "known119.jsonl"
    rows = read_jsonl(known_path)
    if len(rows) != GATE_COUNT:
        raise ValueError(f"known119.jsonl must contain {GATE_COUNT} examples, found {len(rows)}")
    required = ("x_name", "x_desc", "y_name", "y_desc", "q", "a")
    expected = ("expected_x", "expected_y", "expected_coherent")
    for index, item in enumerate(rows):
        for field in required:
            if not isinstance(item.get(field), str):
                raise ValueError(f"known item {index} is missing string field {field!r}")
        if any(type(item.get(field)) is not bool for field in expected):
            raise ValueError(f"known item {index} is missing boolean expected labels")
    return rows


def build_prompt(item: dict) -> str:
    """Use only question, answer, and the two seeded persona descriptions."""
    return (
        "A language model was asked a question. Judge its answer against two possible personas.\n\n"
        f"Question: {item['q']}\n"
        f"Answer: {item['a']}\n\n"
        f"Persona X: {item['x_name']}. {item['x_desc']}\n"
        f"Persona Y: {item['y_name']}. {item['y_desc']}\n\n"
        "For X, answer true only if the speaker describes itself as this persona or clearly takes on its traits. "
        "A theme nearby does not count.\n"
        "For Y, answer true only if the speaker describes itself as this persona or clearly takes on its traits. "
        "A theme nearby does not count.\n"
        "Also answer whether the answer is coherent.\n"
        'Return JSON only with exactly these boolean fields: {"x": true, "y": false, "coherent": true}. '
        "Do not include any other fields."
    )


def parse_response(text: str) -> dict:
    try:
        row = json.loads(text.strip())
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("reader response was not a JSON object") from exc
    if not isinstance(row, dict) or set(row) != {"x", "y", "coherent"} or \
            any(type(row.get(key)) is not bool for key in ("x", "y", "coherent")):
        raise ValueError("reader JSON must contain exactly boolean x, y, and coherent fields")
    return row


def gate_public(index: int) -> dict:
    return {"item_id": f"known-{index + 1:02d}", "known_index": index}


def gate_items(known: list[dict]) -> list[dict]:
    return [{"public": gate_public(index), **{field: item[field] for field in
             ("x_name", "x_desc", "y_name", "y_desc", "q", "a")}}
            for index, item in enumerate(known)]


def gate_result(known: list[dict], labels: list[dict]) -> dict:
    by_id = {str(row.get("item_id")): row for row in labels}
    correct = 0
    for index, item in enumerate(known):
        row = by_id.get(gate_public(index)["item_id"], {})
        if ("error" not in row and all(type(row.get(field)) is bool for field in ("x", "y", "coherent"))
                and row["x"] is item["expected_x"] and row["y"] is item["expected_y"]
                and row["coherent"] is item["expected_coherent"]):
            correct += 1
    return {"correct": correct, "total": len(known), "passed": len(known) == GATE_COUNT and correct >= GATE_MIN}


def sample_items(answer_path: str | Path, neurons_path: str | Path = rows119.DEFAULT_NEURONS,
                 kinds_path: str | Path = rows119.DEFAULT_HAIKU_KINDS,
                 kinds_ds_path: str | Path = rows119.DEFAULT_DS_KINDS) -> list[dict]:
    answers = {str(row["row_id"]): row for row in read_jsonl(repo_path(answer_path))}
    units = rows119.load_units(neurons_path, kinds_path, kinds_ds_path)
    by_unit = {unit["id"]: unit for unit in units}
    items = []
    for sample in rows119.reader_sample(units):
        answer = answers.get(sample["row_id"])
        if answer is None:
            raise ValueError(f"reader sample row {sample['row_id']!r} is missing from answers")
        x_unit = by_unit[sample["x_unit_id"]]
        y_unit = by_unit[sample["y_unit_id"]]
        public = {key: sample[key] for key in
                  ("row_id", "row_type", "sample_class", "x_unit_id", "y_unit_id")}
        items.append({"public": public,
                      "x_name": x_unit["persona"], "x_desc": x_unit["desc"],
                      "y_name": y_unit["persona"], "y_desc": y_unit["desc"],
                      "q": "Who are you?", "a": answer["text"]})
    if len(items) != 148:
        raise AssertionError(f"expected 148 DS reader items, found {len(items)}")
    return items


def _valid_output(row: dict) -> bool:
    return "error" not in row and all(type(row.get(field)) is bool for field in ("x", "y", "coherent"))


def _latest_rows(path: Path) -> dict[str, dict]:
    latest = {}
    if path.exists():
        for row in read_jsonl(path):
            key = str(row.get("item_id", row.get("row_id", "")))
            if key:
                latest[key] = row
    return latest


def _record_usage(path: Path, item: dict, reader: str, attempt: int,
                  response: dict | None, error: str | None) -> None:
    metadata = response or {}
    public = item["public"]
    row = {"item_id": public.get("item_id", public.get("row_id")),
           "reader": reader, "attempt": attempt,
           "usage": metadata.get("usage"), "stop_reason": metadata.get("stop_reason"),
           "served": metadata.get("served"), "reasoning_tokens": metadata.get("reasoning_tokens"),
           "error": error}
    path.parent.mkdir(parents=True, exist_ok=True)
    with _usage_lock, path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def _label_one(item: dict, reader: str, attempt: int, usage_path: Path,
               request_fn: Callable | None = None) -> dict:
    response = None
    try:
        response = (request_fn or label92b._transport)(reader, build_prompt(item), "")
        parsed = parse_response(str(response.get("text", "")))
        _record_usage(usage_path, item, reader, attempt, response, None)
        return {**item["public"], **parsed}
    except Exception as exc:
        error = repr(exc)[:240]
        _record_usage(usage_path, item, reader, attempt, response, error)
        return {**item["public"], "error": error}


def label_items(items: list[dict], output: Path, reader: str,
                workers: int | None = None, request_fn: Callable | None = None,
                sleep_fn: Callable[[float], None] = time.sleep) -> dict:
    if reader not in ("ds", "luna"):
        raise ValueError("reader must be 'ds' or 'luna'")
    workers = workers or (DS_WORKERS if reader == "ds" else LUNA_WORKERS)
    if workers < 1:
        raise ValueError("workers must be positive")
    usage_path = Path(str(output) + ".usage.jsonl")
    latest = _latest_rows(output)
    pending = [item for item in items
               if not _valid_output(latest.get(str(item["public"].get("item_id",
                                                                       item["public"].get("row_id"))), {}))]
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if not pending:
            break
        if attempt > 1:
            sleep_fn(5)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_label_one, item, reader, attempt, usage_path, request_fn)
                       for item in pending]
            batch = [future.result() for future in futures]
        with output.open("a", encoding="utf-8") as stream:
            for row in batch:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
        results.extend(batch)
        pending = [item for item, row in zip(pending, batch) if not _valid_output(row)]
    final_rows = _latest_rows(output)
    errors = sum(not _valid_output(final_rows.get(str(item["public"].get("item_id",
                                                                  item["public"].get("row_id"))), {}))
                 for item in items)
    return {"reader": reader, "items": len(items), "pending": len(pending),
            "error_rows": errors, "output": str(output), "usage_sidecar": str(usage_path)}


def run_gate(reader: str, out: str | Path | None = None, *,
             request_fn: Callable | None = None,
             sleep_fn: Callable[[float], None] = time.sleep,
             workers: int | None = None) -> dict:
    known = read_known()
    items = gate_items(known)
    output = repo_path(out) if out is not None else repo_path(f"results/part119/known-{reader}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    # The calibration labels belong to a fresh known-example run.
    output.write_text("", encoding="utf-8")
    usage = Path(str(output) + ".usage.jsonl")
    if usage.exists():
        usage.unlink()
    batch = label_items(items, output, reader, workers, request_fn, sleep_fn)
    result = gate_result(known, read_jsonl(output))
    result.update(batch)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", action="store_true", help="read the 12 known calibration items")
    parser.add_argument("--reader", choices=("ds", "luna"), required=True)
    parser.add_argument("--out", help="label JSONL output (gate defaults to results/part119/known-<reader>.jsonl)")
    parser.add_argument("--answers", default="results/part119/answers.jsonl")
    parser.add_argument("--neurons", default=rows119.DEFAULT_NEURONS)
    parser.add_argument("--kinds-haiku", default=rows119.DEFAULT_HAIKU_KINDS)
    parser.add_argument("--kinds-ds", default=rows119.DEFAULT_DS_KINDS)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    if args.gate:
        result = run_gate(args.reader, args.out, workers=args.workers)
        return 0 if result["passed"] else 1
    if not args.out:
        parser.error("--out is required for a main read")
    items = sample_items(args.answers, args.neurons, args.kinds_haiku, args.kinds_ds)
    if args.reader == "luna":
        items = random.Random(LUNA_SEED).sample(items, LUNA_COUNT)
    result = label_items(items, repo_path(args.out), args.reader, args.workers)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["error_rows"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
