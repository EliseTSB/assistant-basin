#!/usr/bin/env python3
"""Prepare Part 178 problem ids/text and the held-out pilot. (Design: Method)"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part177 import read177  # noqa: E402

DEFAULT_INPUTS = ROOT / "results" / "part177" / "inputs"
DEFAULT_WINDOW_KEY = ROOT / "results" / "part177" / "window-key.jsonl"
DEFAULT_OUT = HERE / "problems178.json"


def _ids_in_part177_order(inputs_dir: Path):
    _merged, per_problem, _samples, _dataset, _tokenizer = read177._data_paths(inputs_dir)
    ids = []
    for shard in range(10):
        rows = read177.read_json(per_problem["unedited"][shard])
        if not isinstance(rows, list) or len(rows) < 20:
            raise ValueError(f"unedited shard {shard} has fewer than 20 ordered MATH-500 rows")
        ids.extend(str(row["id"]) for row in rows[:20])
    if len(ids) != 200 or len(set(ids)) != 200:
        raise ValueError("Part 177 order must contain 200 distinct sample-0 ids")
    # The second arm is a source-integrity check, not the ordering source.
    for shard in range(10):
        rows = read177.read_json(per_problem["p50c"][shard])
        if [str(row["id"]) for row in rows[:20]] != ids[shard * 20:(shard + 1) * 20]:
            raise ValueError(f"Part 177 input arm ids differ in shard {shard}")
    return ids


def _check_window_key_ids(target_ids, key_rows):
    """Window-key rows exist only for sampled literal hits, so check they map into the 200."""
    target_set = {str(problem_id) for problem_id in target_ids}
    key_ids = {str(row["problem_id"]) for row in key_rows}
    if len(target_set) != 200:
        raise ValueError("Part 177 input order must contain exactly 200 distinct problem ids")
    if not key_ids <= target_set:
        raise ValueError("Part 177 window-key contains a problem outside its ordered 200 inputs")
    return {"input_problem_ids": len(target_set), "window_key_rows": len(key_rows),
            "window_key_problem_ids": len(key_ids)}


def build(inputs_dir=DEFAULT_INPUTS, window_key_path=DEFAULT_WINDOW_KEY):
    """Read pinned dataset and row order, returning a provenance-stamped payload."""
    inputs_dir, window_key_path = Path(inputs_dir), Path(window_key_path)
    read177._verify_manifest(inputs_dir)
    _merged, _per, _samples, dataset_path, _tokenizer = read177._data_paths(inputs_dir)
    target_ids = _ids_in_part177_order(inputs_dir)
    dataset_rows = read177.load_dataset_rows(dataset_path)
    by_id = {}
    dataset_order = []
    for row in dataset_rows:
        problem_id = str(row.get("unique_id", row.get("id", "")))
        if not problem_id or problem_id in by_id:
            raise ValueError("pinned MATH-500 dataset has a missing or duplicate id")
        if "problem" not in row:
            raise ValueError(f"MATH-500 dataset row {problem_id} lacks problem text")
        by_id[problem_id] = {"problem_id": problem_id, "problem": str(row["problem"])}
        dataset_order.append(problem_id)
    if len(dataset_order) != 500:
        raise ValueError(f"pinned dataset has {len(dataset_order)} rows, expected 500")
    if not set(target_ids) <= set(by_id):
        raise ValueError("Part 177 ids are not all present in the pinned MATH-500 dataset")
    target_set = set(target_ids)
    pilot_ids = [problem_id for problem_id in dataset_order if problem_id not in target_set][:16]
    if len(pilot_ids) != 16:
        raise ValueError("the dataset has fewer than 16 rows outside Part 177's 200")
    key_rows = read177.read_jsonl(window_key_path)
    key_check = _check_window_key_ids(target_ids, key_rows)
    return {"part": 178, "source": {"file": dataset_path.name,
                                      "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest()},
            "part177_order_source": "results/part177/inputs/eval/unedited/parts/math500_s{0..9}of10/math500_per_problem.json first 20 ids per shard",
            "problems": [by_id[problem_id] for problem_id in target_ids],
            "pilot": [by_id[problem_id] for problem_id in pilot_ids],
            "counts": {"problems": len(target_ids), "pilot": len(pilot_ids),
                       "dataset_rows": len(dataset_order), **key_check}}


def write_payload(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--window-key", type=Path, default=DEFAULT_WINDOW_KEY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="write and confirm 200 ids against Part 177 keys/inputs")
    args = parser.parse_args(argv)
    payload = build(args.inputs_dir, args.window_key)
    write_payload(args.out, payload)
    if args.check:
        # Check in memory; never print problem text or reopen the generated text artifact.
        ids = [row["problem_id"] for row in payload["problems"]]
        if len(ids) != 200 or len(set(ids)) != 200 or len(payload["pilot"]) != 16:
            raise SystemExit("Part 178 problem check failed")
        print(json.dumps({"checked": True, "problem_ids": 200, "pilot_ids": 16,
                          "window_key_rows": payload["counts"]["window_key_rows"],
                          "window_key_problem_ids": payload["counts"]["window_key_problem_ids"],
                          "source_sha256": payload["source"]["sha256"]}, ensure_ascii=True), flush=True)
    else:
        print(json.dumps({"written": str(args.out), "problem_ids": 200, "pilot_ids": 16}, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
