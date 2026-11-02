#!/usr/bin/env python3
"""Part 174 resumable readers with an explicit first pass and one failed-pair retry pass."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part171"), str(ROOT / "part172"), str(ROOT / "part173")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import label173
import run174

# Reuse Part 173's prompt and reader behavior without copying its frozen wording.
READER_PROMPT = label173.READER_PROMPT
READERS = label173.READERS


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _valid_file(path):
    rows = run174.read_jsonl(path) if Path(path).is_file() else []
    by_id = {}
    duplicates = []
    for row in rows:
        item_id = str(row.get("id"))
        if item_id in by_id:
            duplicates.append(item_id)
        by_id[item_id] = row
    if duplicates:
        raise ValueError(f"reader output contains duplicate pair ids: {duplicates[:3]}")
    return rows, by_id


def ensure_known(reader, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"known-{reader}.json"
    provenance = label173.known_cache_provenance(reader)
    known = read_json(path, {}) or {}
    expected = run174.DESIGN["gate9"]["known_examples"]
    try:
        valid_cache = (len(known.get("rows", [])) == expected and known.get("provenance") == provenance and
                       int(known.get("total", 0) or 0) == expected and
                       int(known.get("relation_correct", 0) or 0) >=
                       run174.DESIGN["gate9"]["minimum_relation"] and
                       int(known.get("primary_correct", 0) or 0) >=
                       run174.DESIGN["gate9"]["minimum_primary"] and bool(known.get("passed")))
    except (TypeError, ValueError):
        valid_cache = False
    if not valid_cache:
        known = label173.known_gate(reader)
        known["provenance"] = provenance
        path.write_text(json.dumps(known, ensure_ascii=True, indent=2) + "\n",
                        encoding="utf-8", newline="\n")
    return known


def _write_checkpoint(path, rows, updated_by_id, ordered_ids):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    ordered = list(dict.fromkeys([str(row.get("id")) for row in rows] + list(ordered_ids)))
    temporary.write_text("".join(json.dumps(updated_by_id[item_id], ensure_ascii=True, allow_nan=False) + "\n"
                                       for item_id in ordered if item_id in updated_by_id),
                          encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _select_items(items, reader, stage, out_dir):
    if reader == "ds":
        return list(items)
    sample_path = Path(out_dir) / "second_reader.json"
    sample = read_json(sample_path, {}) or {}
    selected_ids = set(map(str, sample.get("ids", [])))
    selected = [item for item in items if str(item["id"]) in selected_ids]
    if {str(item["id"]) for item in selected} != selected_ids:
        raise ValueError("frozen second-reader sample includes pair ids absent from current answers")
    return selected


def _freeze_second_sample(items, first_rows, first_path, out_dir):
    """Freeze Part 173's deterministic sample after the retry/cap, including allowed residual errors."""
    path = Path(out_dir) / "second_reader.json"
    first_sha = hashlib.sha256(Path(first_path).read_bytes()).hexdigest()
    if path.is_file():
        previous = read_json(path, {}) or {}
        if previous.get("first_reader_sha256") != first_sha:
            raise ValueError("first-reader output changed after second-reader sample was frozen")
        return previous
    sample = label173.second_reader_sample(items, first_rows)
    value = {**sample, "seed": label173.DESIGN["second_reader"]["seed"],
             "fraction": label173.DESIGN["second_reader"]["fraction"],
             "first_reader_sha256": first_sha, "source": str(first_path),
             "created_before_second_reader_calls": True}
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n",
                    encoding="utf-8", newline="\n")
    return value


def read_pairs(answers_path, out_dir, reader="ds", stage="first"):
    if reader not in READERS:
        raise ValueError(f"unknown reader {reader!r}")
    if stage not in {"first", "retry-failed", "second"}:
        raise ValueError(f"unknown read stage {stage!r}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    known = ensure_known(reader, out)
    if not known.get("passed"):
        return {"reader": reader, "stage": stage, "stopped": "known-example gate failed",
                "known": {key: value for key, value in known.items() if key != "rows"}}

    items = label173.items_for(answers_path)
    if reader == "haiku":
        sample_path = out / "second_reader.json"
        if not sample_path.is_file():
            first_path = out / "labels-ds.jsonl"
            first_rows, _ = _valid_file(first_path)
            sample = _freeze_second_sample(items, first_rows, first_path, out)
        else:
            first_path = out / "labels-ds.jsonl"
            frozen = read_json(sample_path, {}) or {}
            first_sha = hashlib.sha256(first_path.read_bytes()).hexdigest() if first_path.is_file() else None
            if frozen.get("first_reader_sha256") != first_sha:
                raise ValueError("first-reader output changed after second-reader sample was frozen")
        items = _select_items(items, reader, stage, out)

    path = out / f"labels-{reader}.jsonl"
    prior, prior_by_id = _valid_file(path)
    sample_path = out / "second_reader.json"
    if reader == "ds" and stage in {"first", "retry-failed"} and sample_path.is_file():
        frozen = read_json(sample_path, {}) or {}
        current_sha = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if frozen.get("first_reader_sha256") != current_sha:
            raise ValueError("first-reader output is frozen; refusing to append or retry after sample selection")
    if stage == "first":
        pending = [item for item in items if str(item["id"]) not in prior_by_id]
        attempts = 1
    elif stage == "retry-failed":
        pending = [item for item in items if str(item["id"]) in prior_by_id and
                   not label173.valid(prior_by_id[str(item["id"])])]
        attempts = 1
    else:
        pending = [item for item in items if str(item["id"]) not in prior_by_id or
                   not label173.valid(prior_by_id[str(item["id"])])]
        attempts = label173.ATTEMPTS

    updated = dict(prior_by_id)
    ids_in_order = [str(item["id"]) for item in pending]
    if pending:
        worker_count = label173.READER_WORKERS[reader]
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {pool.submit(label173.label_one, item, reader, attempts=attempts): item
                       for item in pending}
            for finished, future in enumerate(as_completed(futures), start=1):
                item = futures[future]
                updated[str(item["id"])] = future.result()
                if finished % label173.CHECKPOINT_EVERY == 0:
                    _write_checkpoint(path, prior, updated, ids_in_order)
        _write_checkpoint(path, prior, updated, ids_in_order)
    elif not path.exists():
        path.write_text("", encoding="utf-8", newline="\n")

    rows = run174.read_jsonl(path)
    current_ids = {str(item["id"]) for item in items}
    current_rows = [row for row in rows if str(row.get("id")) in current_ids]
    failed = [row for row in current_rows if not label173.valid(row)]
    result = {"reader": reader, "stage": stage, "items": len(items), "pending": len(pending),
              "attempted": len(current_rows), "valid_pairs": len(current_rows) - len(failed),
              "failed_pairs": len(failed), "failed_pair_ids": [str(row.get("id")) for row in failed],
              "checkpoint_every": label173.CHECKPOINT_EVERY,
              "workers": label173.READER_WORKERS[reader],
              "reader_prompt_sha256": hashlib.sha256(READER_PROMPT.encode("utf-8")).hexdigest()}
    if reader == "ds":
        result["error_fraction"] = len(failed) / len(items) if items else 0.0
        result["error_cap_fraction"] = run174.DESIGN["reader_errors"]["maximum_fraction"]
    return result


def check_error_cap(answers_path, out_dir):
    items = label173.items_for(answers_path)
    rows, by_id = _valid_file(Path(out_dir) / "labels-ds.jsonl")
    current_ids = {str(item["id"]) for item in items}
    failures = [item_id for item_id in current_ids
                if item_id not in by_id or not label173.valid(by_id[item_id])]
    fraction = len(failures) / len(items) if items else 0.0
    cap = float(run174.DESIGN["reader_errors"]["maximum_fraction"])
    result = {"passed": fraction <= cap, "reader": "ds", "failed_pairs": len(failures),
              "total_pairs": len(items), "error_fraction": fraction, "maximum_fraction": cap,
              "failed_pair_ids": sorted(failures),
              "missing_pair_ids": sorted(item_id for item_id in current_ids if item_id not in by_id)}
    return result


def check_generation_gates(out_dir):
    path = Path(out_dir).parent / "gates.json"
    gates = read_json(path, {}) or {}
    required = [f"gate{index}" for index in range(1, 9)]
    missing = [name for name in required if name not in gates]
    failed = [name for name in required if name in gates and
              (not isinstance(gates[name], dict) or not gates[name].get("passed"))]
    passed = bool(gates.get("passed") and not missing and not failed)
    return {"passed": passed, "gates_path": str(path), "missing": missing, "failed": failed}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", default="results/part174/answers.jsonl")
    parser.add_argument("--out-dir", default="results/part174/labels")
    parser.add_argument("--reader", choices=tuple(READERS), default="ds")
    parser.add_argument("--stage", choices=("check-gates", "known", "first", "retry-failed", "error-cap", "second"),
                        default="first")
    args = parser.parse_args(argv)
    if args.stage == "check-gates":
        result = check_generation_gates(args.out_dir)
    elif not check_generation_gates(args.out_dir)["passed"]:
        result = check_generation_gates(args.out_dir)
    elif args.stage == "known":
        result = ensure_known(args.reader, args.out_dir)
        result = {key: value for key, value in result.items() if key != "rows"}
    elif args.stage == "error-cap":
        result = check_error_cap(args.answers, args.out_dir)
    else:
        stage = "first" if args.stage == "first" else "retry-failed" if args.stage == "retry-failed" else "second"
        result = read_pairs(args.answers, args.out_dir, args.reader, stage)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)
    if result.get("stopped") or result.get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
