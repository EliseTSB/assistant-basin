#!/usr/bin/env python3
"""Read Part 92's stored answers with a second, blinded reader."""
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
sys.path.insert(0, str(ROOT / "part92"))
sys.path.insert(0, str(ROOT))

import label92  # noqa: E402  The registered Part 92 prompt and response parser.
import judge_client  # noqa: E402  The measured cliproxy and OpenCode Go transports.

SEED = 9202
SHARE = 0.15
GATE_ITEMS = 12
GATE_MIN_CORRECT = 11
MAX_ATTEMPTS = 2
DS_WORKERS = 8
LUNA_WORKERS = 4
DS_MODEL = "deepseek-v4.1-flash"
LUNA_MODEL = "gpt-6-luna"
_usage_lock = threading.Lock()


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def item_key(row: dict) -> tuple[str, str, int]:
    return str(row["id"]), str(row["unit"]), int(row["question_index"])


def sort_key(key: tuple[str, str, int]) -> tuple:
    unit_order = {"neuron": 0, "random": 1}
    return key[0], unit_order.get(key[1], 9), key[2]


def make_items(answer_rows: list[dict], neuron_rows: list[dict]) -> list[dict]:
    by_id = {str(row["id"]): row for row in neuron_rows}
    found = {}
    for answer in answer_rows:
        key = item_key(answer)
        if key[1] not in ("neuron", "random"):
            raise ValueError(f"unknown Part 92 unit {key[1]!r}")
        if not 0 <= key[2] < 6:
            raise ValueError(f"question index outside the six-question battery: {key[2]}")
        if key[0] not in by_id:
            raise ValueError(f"answer refers to unknown Part 92 unit {key[0]!r}")
        if key in found:
            raise ValueError(f"duplicate Part 92 answer item {key}")
        neuron = by_id[key[0]]
        found[key] = {
            "public": {"id": key[0], "unit": key[1], "question_index": key[2]},
            "name": neuron["persona"],
            "desc": neuron["desc"],
            "q": answer["question"],
            "a": answer["answer"],
        }
    return [found[key] for key in sorted(found, key=sort_key)]


def build_prompt(item: dict) -> str:
    """Fill label92.PROMPT exactly, from only the target, question, and answer."""
    prompt = label92.PROMPT
    for field in ("name", "desc", "q", "a"):
        prompt = prompt.replace("{" + field + "}", str(item[field]))
    return prompt


def sample_keys(keys) -> list[tuple[str, str, int]]:
    """Return the registered 48-per-arm sample from the Part 92 item keys."""
    ordered = sorted({(str(i), str(u), int(q)) for i, u, q in keys}, key=sort_key)
    groups = {
        arm: [key for key in ordered if key[1] == arm]
        for arm in ("neuron", "random")
    }
    rng = random.Random(SEED)
    selected = []
    for arm in ("neuron", "random"):
        size = round(SHARE * len(groups[arm]))
        if len(groups[arm]) != 318 or size != 48:
            raise ValueError(
                f"expected 318 {arm} items and a 48-item sample; found "
                f"{len(groups[arm])} and {size}"
            )
        selected.extend(rng.sample(groups[arm], size))
    return sorted(selected, key=sort_key)


def write_sample() -> dict:
    answer_rows = read_jsonl(repo_path("results/part92/step2/answers/instruct_chat.jsonl"))
    keys = sample_keys(item_key(row) for row in answer_rows)
    result = {
        "seed": SEED,
        "share": SHARE,
        "items": len(keys),
        "counts_by_unit": {arm: sum(key[1] == arm for key in keys)
                           for arm in ("neuron", "random")},
        "item_keys": [
            {"id": i, "unit": unit, "question_index": qi}
            for i, unit, qi in keys
        ],
    }
    output = repo_path("results/part92b/sample92b.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def valid_label(row: dict) -> bool:
    return (
        "error" not in row
        and all(type(row.get(field)) is bool
                for field in ("shows_persona", "self_claim", "coherent"))
        and row.get("speaker") in ("ai", "human", "other", "none")
    )


def _transport(reader: str, prompt: str, _unused_key: str = "") -> dict:
    if reader == "ds":
        return judge_client.call_prompt(
            prompt,
            model=DS_MODEL,
            route="ocgo",
            session="part92b-ds-v4-1-flash",
            max_tokens=4000,
            timeout=300,
        )
    if reader == "luna":
        return judge_client.call_prompt(
            prompt,
            model=LUNA_MODEL,
            route="cliproxy",
            session="part92b-gpt-6-luna",
            max_tokens=4000,
            timeout=300,
        )
    raise ValueError(f"unknown reader {reader!r}")


def _record_usage(path: Path, item: dict, reader: str, attempt: int,
                  response: dict | None, error: str | None) -> None:
    metadata = response or {}
    row = {
        **item["public"],
        "reader": reader,
        "attempt": attempt,
        "usage": metadata.get("usage"),
        "stop_reason": metadata.get("stop_reason"),
        "served": metadata.get("served"),
        "reasoning_tokens": metadata.get("reasoning_tokens"),
        "error": error,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with _usage_lock, path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def _label_one(item: dict, reader: str, usage_path: Path,
               request_fn: Callable | None = None,
               sleep_fn: Callable[[float], None] = time.sleep) -> dict:
    prompt = build_prompt(item)
    last_error = "unknown error"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = None
        try:
            response = (request_fn or _transport)(reader, prompt, "")
            parsed = label92._parse(str(response.get("text", "")))
            _record_usage(usage_path, item, reader, attempt, response, None)
            return {**item["public"], **parsed}
        except Exception as exc:
            # There is no API key to redact. Keep the exception readable in the row and sidecar.
            last_error = repr(exc)[:240]
            _record_usage(usage_path, item, reader, attempt, response, last_error)
            if attempt < MAX_ATTEMPTS:
                sleep_fn(5)
    return {**item["public"], "error": last_error}


def _latest_rows(path: Path) -> dict[tuple[str, str, int], dict]:
    latest = {}
    if path.exists():
        for row in read_jsonl(path):
            latest[item_key(row)] = row
    return latest


def label_items(items: list[dict], output: Path, reader: str, workers: int,
                request_fn: Callable | None = None,
                sleep_fn: Callable[[float], None] = time.sleep) -> dict:
    if reader not in ("ds", "luna"):
        raise ValueError("reader must be 'ds' or 'luna'")
    if workers < 1:
        raise ValueError("workers must be positive")
    usage_path = Path(str(output) + ".usage.jsonl")
    existing = _latest_rows(output)
    pending = [item for item in items
               if item_key(item["public"]) not in existing
               or not valid_label(existing[item_key(item["public"])])]
    output.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool, output.open("a", encoding="utf-8") as stream:
            futures = [pool.submit(_label_one, item, reader, usage_path,
                                  request_fn, sleep_fn) for item in pending]
            for future in futures:
                row = future.result()
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                errors += "error" in row
    return {
        "reader": reader,
        "items": len(items),
        "pending": len(pending),
        "error_rows": errors,
        "output": str(output),
        "usage_sidecar": str(usage_path),
    }


def _gate_id(index: int) -> str:
    return f"gate-a-{index:02d}"


def read_known() -> list[dict]:
    rows = read_jsonl(HERE / "known92b.jsonl")
    if len(rows) != GATE_ITEMS:
        raise ValueError(f"known92b.jsonl must contain {GATE_ITEMS} items, found {len(rows)}")
    for index, item in enumerate(rows):
        if type(item.get("expected")) is not bool:
            raise ValueError(f"known example {index} is missing a boolean expected value")
        for field in ("name", "desc", "q", "a"):
            if not isinstance(item.get(field), str):
                raise ValueError(f"known example {index} is missing string field {field}")
    return rows


def gate_items(known: list[dict]) -> list[dict]:
    return [
        {
            "public": {"id": _gate_id(index), "unit": "gate_a", "question_index": index % 6},
            "name": item["name"], "desc": item["desc"],
            "q": item["q"], "a": item["a"],
        }
        for index, item in enumerate(known)
    ]


def gate_result(known: list[dict], rows: list[dict]) -> dict:
    by_id = {str(row.get("id")): row for row in rows}
    correct = 0
    for index, item in enumerate(known):
        row = by_id.get(_gate_id(index), {})
        if valid_label(row) and row["shows_persona"] is item["expected"]:
            correct += 1
    total = len(known)
    return {"correct": correct, "total": total,
            "passed": total == GATE_ITEMS and correct >= GATE_MIN_CORRECT}


def gate_status(reader: str) -> dict:
    path = repo_path(f"results/part92b/gate-a-{reader}.jsonl")
    if not path.exists():
        return {"correct": 0, "total": GATE_ITEMS, "passed": False, "exists": False}
    result = gate_result(read_known(), read_jsonl(path))
    result["exists"] = True
    return result


def run_gate(reader: str, request_fn: Callable | None = None,
             sleep_fn: Callable[[float], None] = time.sleep,
             workers: int | None = None) -> dict:
    started = time.monotonic()
    known = read_known()
    items = gate_items(known)
    output = repo_path(f"results/part92b/gate-a-{reader}.jsonl")
    usage_path = Path(str(output) + ".usage.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Gate A is a fresh calibration run; its labels and usage belong to this run only.
    output.write_text("", encoding="utf-8")
    if usage_path.exists():
        usage_path.unlink()
    batch = label_items(items, output, reader,
                        workers or (DS_WORKERS if reader == "ds" else LUNA_WORKERS),
                        request_fn=request_fn, sleep_fn=sleep_fn)
    result = gate_result(known, read_jsonl(output))
    elapsed = time.monotonic() - started
    result.update({"elapsed_seconds": round(elapsed, 3), **batch})
    status = "pass" if result["passed"] else "fail"
    print(f"{status} {result['correct']}/{result['total']} elapsed={elapsed:.2f}s", flush=True)
    return result


def _check_gate_before_main(reader: str) -> None:
    result = gate_status(reader)
    if not result["exists"] or not result["passed"]:
        state = "missing" if not result["exists"] else f"failed ({result['correct']}/{result['total']})"
        raise RuntimeError(f"refusing main {reader} read: Gate A is {state}; run --gate-a --reader {reader} first")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sample", action="store_true", help="write the registered 96-item Luna sample")
    mode.add_argument("--gate-a", action="store_true", help="run the 12-example reader check")
    parser.add_argument("--reader", choices=("ds", "luna"))
    parser.add_argument("--out", help="label JSONL path for a gated main read")
    parser.add_argument("--answers", default="results/part92/step2/answers/instruct_chat.jsonl")
    parser.add_argument("--neurons", default="results/part92/step2/neurons.json")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)

    if args.sample:
        result = write_sample()
        print(json.dumps({"output": str(repo_path("results/part92b/sample92b.json")),
                          "items": result["items"], "counts_by_unit": result["counts_by_unit"]}))
        return 0
    if not args.reader:
        parser.error("--reader ds|luna is required for --gate-a and main reads")
    if args.gate_a:
        result = run_gate(args.reader, workers=args.workers)
        return 0 if result["passed"] else 1
    if not args.out:
        parser.error("--out is required for a main read")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    _check_gate_before_main(args.reader)
    answer_rows = read_jsonl(repo_path(args.answers))
    neuron_rows = json.loads(repo_path(args.neurons).read_text(encoding="utf-8"))
    items = make_items(answer_rows, neuron_rows)
    if args.reader == "luna":
        selected = set(sample_keys(item_key(item["public"]) for item in items))
        items = [item for item in items if item_key(item["public"]) in selected]
    output = repo_path(args.out)
    result = label_items(items, output, args.reader,
                        args.workers or (DS_WORKERS if args.reader == "ds" else LUNA_WORKERS))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
