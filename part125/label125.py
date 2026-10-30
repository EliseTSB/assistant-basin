#!/usr/bin/env python3
"""Blind Part 125 labeling through Part 123's DeepSeek/Luna request route."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for directory in (HERE, ROOT / "part92", ROOT / "part92b", ROOT / "part119", ROOT / "part123"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import label123  # noqa: E402 -- shared Part 92 prompt and DeepSeek/Luna routes.
import label119  # noqa: E402 -- shared known-example bank.

ITEM_SEED = 12501
LUNA_SEED = label123.LUNA_SEED


def read_jsonl(path: str | Path, *, optional: bool = False) -> list[dict]:
    return label123.read_jsonl(path, optional=optional)


def _read_json(path: str | Path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _reader_question(row: dict) -> str:
    prior = row.get("prior_turns", [])
    lines = ["Prior turns:"]
    if not prior:
        lines.append("(none)")
    for index, turn in enumerate(prior, 1):
        lines.append(f"Turn {index} question: {turn['question']}")
        lines.append(f"Turn {index} answer: {turn['answer']}")
    lines.append(f"Current turn {int(row['turn'])} question: {row['question']}")
    return "\n".join(lines)


def opaque_id(arm: str, unit_id: str, turn: int) -> str:
    import hashlib

    raw = f"part125:{ITEM_SEED}:{arm}:{unit_id}:{int(turn)}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def make_items(answer_rows: list[dict]) -> list[dict]:
    items = []
    for row in answer_rows:
        answer = str(row.get("answer", ""))
        q = _reader_question(row)
        answer_hash = label123.answer_hash(q, answer)
        arm = str(row["arm"])
        unit_id = str(row["unit_id"])
        turn = int(row["turn"])
        item_id = opaque_id(arm, unit_id, turn)
        public = {"item_id": item_id, "question_index": turn - 1,
                  "answer_hash": answer_hash}
        private = {"item_id": item_id, "row_id": str(row["row"]),
                   "unit_id": unit_id, "arm": arm, "turn": turn,
                   "question_index": turn - 1, "answer_hash": answer_hash}
        items.append({"public": public, "private": private,
                      "name": str(row.get("persona", "")),
                      "desc": str(row.get("desc", "")), "q": q, "a": answer})
    if len({item["public"]["item_id"] for item in items}) != len(items):
        raise ValueError("duplicate Part 125 arm/unit/turn label item")
    random.Random(ITEM_SEED).shuffle(items)
    return items


def luna_sample(items: list[dict], deepseek_rows: list[dict]) -> list[dict]:
    """Use Part 123's positive-inclusive second-reader sample policy."""
    return label123.luna_sample(items, deepseek_rows)


def _known_gate(reader: str, path: Path, request_fn=None, sleep_fn=None) -> dict:
    result = label123.run_gate(reader, path, request_fn=request_fn,
                               sleep_fn=sleep_fn or __import__("time").sleep)
    known = label119.read_known()
    labels = read_jsonl(path)
    checked = label119.gate_result(known, labels)
    checked["passed"] = bool(checked.get("correct", 0) >= 11)
    checked["required_correct"] = 11
    return checked


def execution_gates(answers_path: str | Path) -> list[dict]:
    """Require model/cache gates before sending study answers to a reader."""
    raw_dir = Path(answers_path).parent
    run_path = raw_dir / "run-meta.json"
    run_meta = _read_json(run_path) if run_path.is_file() else {}
    expected_parameters = run_meta.get("parameters", {}).get("unit_parameters")
    answers = read_jsonl(answers_path)
    units = {str(row["unit_id"]) for row in answers}
    layers = {int(row["layer"]) for row in answers}
    zero = _read_json(raw_dir / "zero-hook-check.json")
    zero_map = {(int(row["layer"]), int(row["turn"])): row for row in zero}
    expected_zero = {(layer, turn) for layer in layers for turn in range(1, 6)}
    zero_ok = expected_zero <= set(zero_map) and all(
        zero_map[key].get("batch_size") == 1 and
        isinstance(zero_map[key].get("plain_token_ids"), list) and
        zero_map[key].get("plain_token_ids") == zero_map[key].get("hooked_token_ids")
        for key in expected_zero
    )
    determinism = read_jsonl(raw_dir / "determinism.jsonl")
    det_ids = {str(row.get("unit_id")) for row in determinism}
    det_ok = ({"p92-02", "p92-08"} <= det_ids and
              all(len(row.get("pushed_logit_hashes_first", [])) == int(row.get("calls", -1)) ==
                  len(row.get("pushed_logit_hashes_second", [])) ==
                  len(row.get("twin_logit_hashes_first", [])) ==
                  len(row.get("twin_logit_hashes_second", [])) and
                  row.get("pushed_logit_hashes_first") == row.get("pushed_logit_hashes_second") and
                  row.get("twin_logit_hashes_first") == row.get("twin_logit_hashes_second") and
                  row.get("pushed_bitwise_equal") is True and
                  row.get("twin_bitwise_equal") is True
                  for row in determinism))
    cache = read_jsonl(raw_dir / "cache-sanity.jsonl")
    cache_ok = (len(cache) == 5 * len(units) and
                sum(row.get("cached_top1_id") == row.get("full_top1_id") for row in cache) >=
                math.ceil(0.95 * len(cache)) and
                max((max((float(value) for value in row.get("top100_abs_diffs", [])),
                         default=float("inf")) for row in cache),
                    default=float("inf")) <= 1.0)
    ledger = read_jsonl(raw_dir / "ledger-checks.jsonl")
    ledger_ok = (len(ledger) == 5 * len(units) and all(
        isinstance(row.get("arm2_ledger_token_ids"), list) and
        row.get("arm2_ledger_token_ids") == row.get("arm3_ledger_token_ids") ==
        row.get("template_token_ids") and
        (int(row.get("turn", 0)) != 1 or row.get("arm2_hook_removed") is True) and
        row.get("call_logs_equal") is True and
        row.get("arm2_hook_state_ok") is True and
        row.get("twin_hook_state_ok") is True and
        int(row.get("arm2_hook_calls", -1)) ==
        (int(row.get("arm2_calls_this_turn", 0)) if int(row.get("turn", 0)) == 1 else 0)
        for row in ledger
    ))
    sequences = read_jsonl(raw_dir / "call-sequences.jsonl")
    sequence_map = {str(row["unit_id"]): row for row in sequences}
    ledger_ok = ledger_ok and set(sequence_map) == units
    answer_by_key = {(str(row["unit_id"]), int(row["turn"])): row
                     for row in answers if row["arm"] == "arm2"}
    ledger_by_unit: dict[str, list[dict]] = {}
    for row in ledger:
        left = row.get("arm2_ledger_token_ids", [])
        answer = answer_by_key.get((str(row.get("unit_id")), int(row.get("turn", 0))))
        ledger_ok = ledger_ok and answer is not None and left[-len(answer.get("token_ids", [])):] == answer.get("token_ids", [])
        ledger_by_unit.setdefault(str(row.get("unit_id")), []).append(row)
    for rows in ledger_by_unit.values():
        rows.sort(key=lambda row: int(row["turn"]))
        for previous, current in zip(rows, rows[1:]):
            prior_ids = previous.get("arm2_ledger_token_ids", [])
            ledger_ok = ledger_ok and current.get("arm2_ledger_token_ids", [])[:len(prior_ids)] == prior_ids
    def same_calls(left, right):
        fields = ("token_ids", "shape", "cache_length_before", "cache_length_after", "turn")
        return len(left) > 0 and len(left) == len(right) and all(
            all(a.get(field) == b.get(field) for field in fields) and
            a.get("shape") == [1, len(a.get("token_ids", []))]
            for a, b in zip(left, right)
        )

    for row in sequence_map.values():
        left, right = row.get("arm2_calls", []), row.get("arm3_calls", [])
        ledger_ok = ledger_ok and same_calls(left, right)
        ledger_ok = ledger_ok and all(call.get("hook_on") is (int(call.get("turn", 0)) == 1)
                                      for call in left)
        ledger_ok = ledger_ok and all(call.get("hook_on") is False for call in right)
        arm1_calls = row.get("arm1_calls", [])
        arm4_calls = row.get("arm4_calls", [])
        ledger_ok = ledger_ok and {int(call.get("turn", 0)) for call in arm1_calls} == {2, 3, 4, 5}
        ledger_ok = ledger_ok and all(call.get("hook_on") is True for call in arm1_calls)
        ledger_ok = ledger_ok and {int(call.get("turn", 0)) for call in arm4_calls} == {1, 2, 3, 4, 5}
        ledger_ok = ledger_ok and all(call.get("hook_on") is False for call in arm4_calls)
        for pair in row.get("random_pairs", []):
            random_calls = pair.get("random_calls", [])
            twin_calls = pair.get("twin_calls", [])
            ledger_ok = ledger_ok and same_calls(random_calls, twin_calls)
            ledger_ok = ledger_ok and same_calls(left, random_calls)
            ledger_ok = ledger_ok and all(
                call.get("hook_on") is (int(call.get("turn", 0)) == 1)
                for call in random_calls
            )
            ledger_ok = ledger_ok and all(call.get("hook_on") is False for call in twin_calls)
        ledger_ok = ledger_ok and len(row.get("random_pairs", [])) == 3
    parameter_rows = answers + read_jsonl(raw_dir / "reader-free.jsonl") + \
        read_jsonl(raw_dir / "random-reader-free.jsonl") + cache + determinism + ledger
    parameter_rows.extend(zero)
    call_rows = []
    for row in sequences:
        call_rows.extend(row.get(key, []) for key in
                         ("arm2_calls", "arm3_calls", "arm1_calls", "arm4_calls"))
        for pair in row.get("random_pairs", []):
            call_rows.extend((pair.get("random_calls", []), pair.get("twin_calls", [])))
    parameter_ok = expected_parameters is not None and all(
        row.get("parameters") == expected_parameters for row in parameter_rows
    ) and all(row.get("parameters") == expected_parameters for row in sequences) and all(
        call.get("parameters") == expected_parameters for calls in call_rows for call in calls
    )
    ledger_ok = ledger_ok and parameter_ok
    result = [
        {"name": "zero-vector hook", "passed": bool(zero_ok)},
        {"name": "pilot twin determinism", "passed": bool(det_ok)},
        {"name": "cache sanity", "passed": bool(cache_ok)},
        {"name": "template ledger and call shapes", "passed": bool(ledger_ok)},
    ]
    pilot_path = raw_dir / "pilot-meta.json"
    if not pilot_path.is_file() or not run_path.is_file():
        result.append({"name": "pilot size cut", "passed": False})
    else:
        pilot = _read_json(pilot_path)
        run = _read_json(run_path)
        required_cut = bool((pilot.get("pilot_size_decision") or {}).get("over_90_minutes"))
        result.append({"name": "pilot size cut",
                       "passed": bool(run.get("slot_only")) == required_cut and
                       bool(run.get("parameters", {}).get("slot_only")) == required_cut})
    return result


def main(argv: list[str] | None = None, *, request_fn=None, sleep_fn=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", action="store_true", help="run 12 known examples")
    parser.add_argument("--reader", choices=("ds", "luna"), required=True)
    parser.add_argument("--answers", default="results/part125/main/answers.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--known-ds", default="results/part125/known-ds.jsonl")
    parser.add_argument("--known-luna", default="results/part125/known-luna.jsonl")
    parser.add_argument("--labels-ds", default="results/part125/labels-ds.jsonl")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")

    output = Path(args.out)
    if args.gate:
        gate_path = Path(args.known_ds if args.reader == "ds" else args.known_luna)
        result = _known_gate(args.reader, gate_path, request_fn, sleep_fn)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if result["passed"] else 1

    try:
        for reader, known_path in (("ds", Path(args.known_ds)), ("luna", Path(args.known_luna))):
            known = label119.gate_result(label119.read_known(), read_jsonl(known_path))
            if int(known.get("correct", 0)) < 11:
                raise ValueError(f"{reader} known-example score {known.get('correct')}/12; need 11")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"main labeling refused: {exc}", file=sys.stderr, flush=True)
        return 1

    answers = read_jsonl(args.answers)
    execution = execution_gates(args.answers)
    failed_execution = [row["name"] for row in execution if not row["passed"]]
    if failed_execution:
        print(f"main labeling refused before labels: execution gates failed: {failed_execution}",
              file=sys.stderr, flush=True)
        return 1
    items = make_items(answers)
    if args.reader == "luna":
        ds_rows = read_jsonl(args.labels_ds)
        current = label123.latest_label_rows(ds_rows)
        for item in items:
            public = item["public"]
            key = (str(public["item_id"]), int(public["question_index"]))
            row = current.get(key)
            if (not label123.valid_label(row or {}) or
                    row.get("answer_hash") != public["answer_hash"]):
                print("Luna labeling refused: DeepSeek labels are incomplete or stale",
                      file=sys.stderr, flush=True)
                return 1
        items = luna_sample(items, ds_rows)

    keys_path = Path(str(output) + ".keys.jsonl")
    _write_jsonl(keys_path, [item["private"] for item in items])
    result = label123.label_items(items, output, args.reader, args.workers,
                                  request_fn=request_fn,
                                  sleep_fn=sleep_fn or __import__("time").sleep)
    result.update({"keys": str(keys_path), "sample_seed": ITEM_SEED if args.reader == "ds" else LUNA_SEED,
                   "sample_policy": "all_items" if args.reader == "ds" else
                   "all_deepseek_positives_plus_seeded_draw",
                   "sample_size": len(items), "deepseek_cap_tokens": 4000})
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["error_rows"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
