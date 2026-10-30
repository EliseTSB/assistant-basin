#!/usr/bin/env python3
"""Blindly label Part 119's registered sample on its Part 119b 0.4 answers."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (HERE, ROOT / "part119", ROOT / "part92b", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import label119 as shared  # noqa: E402 -- keep Part 119's prompt, gate and label loop
import rows119  # noqa: E402
import rows119b  # noqa: E402

READER_SEED = 11904
LUNA_SEED = 11905
LUNA_COUNT = 30
DEFAULT_KEY = "results/part119b/labels-ds.jsonl.keys.jsonl"


def repo_path(value: str | Path) -> Path:
    return shared.repo_path(value)


def opaque_id(row_id: str) -> str:
    return hashlib.sha256(f"part119b-reader:{READER_SEED}:{row_id}".encode()).hexdigest()[:20]


def _dose_row_id(key_row: dict) -> str:
    row_type = str(key_row["row_type"])
    if row_type not in ("pair", "control"):
        raise ValueError(f"unexpected Part 119 reader row type {row_type!r}")
    source_id = str(key_row["row_id"])
    expected_prefix = row_type + ":"
    if not source_id.startswith(expected_prefix):
        raise ValueError(f"row id {source_id!r} does not match row type {row_type!r}")
    pair = source_id[len(expected_prefix):]
    if row_type == "control":
        # Part 119 writes a control as A+RB; Part 119b's dose-tagged rows write it as A+B.
        first, separator, second = pair.partition("+")
        if not separator or not second.startswith("R"):
            raise ValueError(f"control row id {source_id!r} has no random second push")
        pair = f"{first}+{second[1:]}"
    return f"{row_type}@{rows119b.dose_text(.4)}:{pair}"


def sample_items(answer_path: str | Path,
                 prior_keys_path: str | Path = "results/part119/labels-ds.jsonl.keys.jsonl",
                 neurons_path: str | Path = rows119.DEFAULT_NEURONS,
                 kinds_path: str | Path = rows119.DEFAULT_HAIKU_KINDS,
                 kinds_ds_path: str | Path = rows119.DEFAULT_DS_KINDS) -> list[dict]:
    """Build prompts from the exact prior key rows and the new 0.4 answer IDs."""
    key_rows = shared.read_jsonl(prior_keys_path)
    answers = {str(row["row_id"]): row for row in shared.read_jsonl(answer_path)}
    units = rows119.load_units(neurons_path, kinds_path, kinds_ds_path)
    by_unit = {unit["id"]: unit for unit in units}
    seen_rows = set()
    items = []
    for key_row in key_rows:
        required = ("row_id", "row_type", "sample_class", "x_unit_id", "y_unit_id")
        if any(not isinstance(key_row.get(field), str) for field in required):
            raise ValueError("Part 119 reader key row is missing required string fields")
        source_row = str(key_row["row_id"])
        if source_row in seen_rows:
            raise ValueError(f"duplicate Part 119 reader key row {source_row!r}")
        seen_rows.add(source_row)
        dose_row_id = _dose_row_id(key_row)
        answer = answers.get(dose_row_id)
        if answer is None:
            raise ValueError(f"0.4 reader answer row {dose_row_id!r} is missing")
        x_id, y_id = str(key_row["x_unit_id"]), str(key_row["y_unit_id"])
        if x_id not in by_unit or y_id not in by_unit:
            raise ValueError(f"reader key references unknown units {x_id!r}, {y_id!r}")
        x_unit, y_unit = by_unit[x_id], by_unit[y_id]
        public = {"item_id": opaque_id(dose_row_id)}
        private = {"row_id": dose_row_id, "row_type": key_row["row_type"],
                   "sample_class": key_row["sample_class"],
                   "x_unit_id": x_id, "y_unit_id": y_id}
        items.append({"public": public, "private": private,
                      "x_name": x_unit["persona"], "x_desc": x_unit["desc"],
                      "y_name": y_unit["persona"], "y_desc": y_unit["desc"],
                      "q": "Who are you?", "a": str(answer.get("text", ""))})
    pair_count = sum(item["private"]["row_type"] == "pair" for item in items)
    control_count = sum(item["private"]["row_type"] == "control" for item in items)
    if len(items) != 148 or pair_count != 108 or control_count != 40:
        raise AssertionError(
            f"expected 148 reader rows (108 real pairs, 40 controls), found {len(items)} "
            f"({pair_count} pairs, {control_count} controls)")
    if len({item["public"]["item_id"] for item in items}) != len(items):
        raise AssertionError("opaque reader item IDs are not unique")
    random.Random(READER_SEED).shuffle(items)
    return items


def write_sample_keys(items: list[dict], path: str | Path) -> None:
    rows = [{"item_id": item["public"]["item_id"], **item["private"]} for item in items]
    shared.write_jsonl(repo_path(path), rows)


def run_gate(reader: str, out: str | Path | None = None, *, workers: int | None = None) -> dict:
    known = shared.read_known()
    items = shared.gate_items(known)
    output = repo_path(out) if out is not None else repo_path(f"results/part119b/known-{reader}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    usage_path = Path(str(output) + ".usage.jsonl")
    if usage_path.exists():
        usage_path.unlink()
    batch = shared.label_items(items, output, reader, workers)
    result = shared.gate_result(known, shared.read_jsonl(output))
    result.update(batch)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", action="store_true", help="read the 12 known calibration items")
    parser.add_argument("--reader", choices=("ds", "luna"), required=True)
    parser.add_argument("--out", help="label JSONL output (gate defaults to results/part119b/known-<reader>.jsonl)")
    parser.add_argument("--answers", default="results/part119b/run/answers.jsonl")
    parser.add_argument("--gate-labels", help="known-example gate labels (defaults to known-<reader>.jsonl)")
    parser.add_argument("--keys-out", help="private key (defaults to <out>.keys.jsonl)")
    parser.add_argument("--prior-keys", default="results/part119/labels-ds.jsonl.keys.jsonl")
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
    gate_path = (repo_path(args.gate_labels) if args.gate_labels else
                 repo_path(f"results/part119b/known-{args.reader}.jsonl"))
    try:
        shared.require_passing_gate(args.reader, gate_path)
    except ValueError as exc:
        print(f"sample read refused: {exc}", file=sys.stderr, flush=True)
        return 1
    items = sample_items(args.answers, args.prior_keys, args.neurons,
                         args.kinds_haiku, args.kinds_ds)
    output = repo_path(args.out)
    keys_path = repo_path(args.keys_out) if args.keys_out else Path(str(output) + ".keys.jsonl")
    write_sample_keys(items, keys_path)
    if args.reader == "luna":
        items = random.Random(LUNA_SEED).sample(items, LUNA_COUNT)
    result = shared.label_items(items, output, args.reader, args.workers)
    result["keys"] = str(keys_path)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["error_rows"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
