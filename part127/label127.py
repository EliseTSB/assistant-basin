#!/usr/bin/env python3
"""Run the blinded DeepSeek and Luna reads for Part 127."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (HERE, ROOT, ROOT / "part92", ROOT / "part92b"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import judge_client  # noqa: E402 - shared Part 92b reader transport
import label92  # noqa: E402 - reuse registered Part 92 prompt and field parser
import label92b  # noqa: E402 - reuse Part 92b known examples and reader names
import run127  # noqa: E402

READERS = ("ds", "luna")
READER_NAMES = {"ds": "DeepSeek V4.1 Flash", "luna": "Luna"}
KNOWN_PATH = "part92b/known92b.jsonl"
MAX_ATTEMPTS = 2
MAX_RETRY_PASSES = 3
TRUNCATION_STOP_REASONS = {"max_tokens", "max_token", "length", "token_limit",
                           "max_output_tokens"}
MAX_TOKENS = 16000
LUNA_SEED = 12703
_usage_lock = threading.Lock()


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: str | Path) -> list[dict]:
    result = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL at {path}:{line_no}")
            result.append(row)
    return result


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")) + "\n")


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False,
                                separators=(",", ":")) + "\n")
        stream.flush()


def answer_hash(item: dict) -> str:
    value = {key: item[key] for key in ("name", "desc", "q", "a")}
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _opaque_id(namespace: str, payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(namespace.encode("utf-8") + b":" + encoded).hexdigest()[:24]


def _fill_prompt(item: dict) -> str:
    """Fill only the Part 92 target, question, and answer prompt fields."""
    prompt = label92.PROMPT
    for field in ("name", "desc", "q", "a"):
        prompt = prompt.replace("{" + field + "}", str(item[field]))
    return prompt


def build_prompt(item: dict) -> str:
    return _fill_prompt(item)


def valid_label(row: dict) -> bool:
    return ("error" not in row
            and all(type(row.get(field)) is bool
                    for field in ("shows_persona", "self_claim", "coherent"))
            and row.get("speaker") in ("ai", "human", "other", "none"))


def current_label(item: dict, row: dict | None) -> bool:
    return bool(row and valid_label(row)
                and row.get("answer_hash") == item["public"]["answer_hash"])


def parse_response(response: dict) -> dict:
    stop = str(response.get("stop_reason", "")).casefold()
    if stop in TRUNCATION_STOP_REASONS:
        raise ValueError(f"reader reply was truncated at {stop}")
    text = response.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("reader reply was empty")
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("reader reply was not a complete JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("reader reply was not a JSON object")
    # Reuse Part 92's registered schema parser after requiring one complete object.
    parsed = label92._parse(text)
    if payload != parsed:
        raise ValueError("reader reply contains text or fields outside the registered JSON label")
    return parsed


def _transport(reader: str, prompt: str) -> dict:
    if reader == "ds":
        return judge_client.call_prompt(
            prompt, model="deepseek-v4.1-flash", route="ocgo",
            session="part127-ds-v4-1-flash", max_tokens=MAX_TOKENS,
            no_thinking=True, timeout=300)
    if reader == "luna":
        # The cliproxy route uses adaptive thinking. Its 16k token cap leaves
        # room for the response JSON after any reasoning tokens.
        return judge_client.call_prompt(
            prompt, model="gpt-6-luna", route="cliproxy",
            session="part127-gpt-6-luna", max_tokens=MAX_TOKENS,
            timeout=300)
    raise ValueError(f"unknown reader {reader!r}")


def _public(item: dict) -> dict:
    return {"item_id": item["public"]["item_id"],
            "answer_hash": item["public"]["answer_hash"]}


def _record_usage(path: Path, item: dict, reader: str, attempt: int,
                  response: dict | None, error: str | None) -> None:
    metadata = response or {}
    row = {**_public(item), "reader": reader, "attempt": attempt,
           "usage": metadata.get("usage"), "stop_reason": metadata.get("stop_reason"),
           "served": metadata.get("served"), "reasoning_tokens": metadata.get("reasoning_tokens"),
           "error": error}
    with _usage_lock:
        _append_jsonl(path, row)


def _label_one(item: dict, reader: str, usage_path: Path,
               request_fn: Callable | None = None,
               sleep_fn: Callable[[float], None] = time.sleep) -> dict:
    prompt = build_prompt(item)
    last_error = "unknown reader error"
    last_stop_reason = None
    truncated_attempts = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = None
        try:
            response = (request_fn or _transport)(reader, prompt)
            last_stop_reason = response.get("stop_reason")
            if str(last_stop_reason or "").casefold() in TRUNCATION_STOP_REASONS:
                truncated_attempts += 1
            parsed = parse_response(response)
            _record_usage(usage_path, item, reader, attempt, response, None)
            return {**_public(item), **parsed, "stop_reason": last_stop_reason,
                    "attempt_count": attempt, "retry_count": attempt - 1,
                    "truncated_attempts": truncated_attempts}
        except Exception as exc:
            if response is not None:
                last_stop_reason = response.get("stop_reason")
                if (str(last_stop_reason or "").casefold() in TRUNCATION_STOP_REASONS
                        and truncated_attempts < attempt):
                    truncated_attempts += 1
            last_error = repr(exc)[:300]
            _record_usage(usage_path, item, reader, attempt, response, last_error)
            if attempt < MAX_ATTEMPTS:
                sleep_fn(5)
    return {**_public(item), "error": last_error, "stop_reason": last_stop_reason,
            "attempt_count": MAX_ATTEMPTS, "retry_count": MAX_ATTEMPTS - 1,
            "truncated_attempts": truncated_attempts}


def _latest_labels(path: Path) -> dict[str, dict]:
    latest = {}
    if not path.exists():
        return latest
    for row in read_jsonl(path):
        latest[str(row["item_id"])] = row
    return latest


def label_items(items: list[dict], output: str | Path, reader: str, *, workers: int = 8,
                request_fn: Callable | None = None,
                sleep_fn: Callable[[float], None] = time.sleep) -> dict:
    if reader not in READERS:
        raise ValueError("reader must be ds or luna")
    if workers < 1:
        raise ValueError("workers must be positive")
    output_path = Path(output)
    usage_path = Path(str(output_path) + ".usage.jsonl")
    key_path = Path(str(output_path) + ".keys.jsonl")
    # The key file is private join material. Reader output contains only an
    # opaque item id and the answer hash.
    by_item_id = {str(item["public"]["item_id"]): item for item in items}
    if len(by_item_id) != len(items):
        raise ValueError("duplicate opaque reader item id")
    _write_jsonl(key_path, [{"item_id": item["public"]["item_id"],
                             "answer_hash": item["public"]["answer_hash"],
                             **item["private"]} for item in items])
    existing = _latest_labels(output_path)
    pending = [item for item in items
               if not current_label(item, existing.get(str(item["public"]["item_id"]))) ]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def submit(todo: list[dict], pass_number: int) -> None:
        if not todo:
            return
        order_rng = random.Random(12700 + (1 if reader == "ds" else 2) + pass_number * 10)
        order_rng.shuffle(todo)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_label_one, item, reader, usage_path,
                                  request_fn, sleep_fn) for item in todo]
            for future in futures:
                _append_jsonl(output_path, future.result())

    retry_passes = 0
    for pass_number in range(MAX_RETRY_PASSES + 1):
        latest = _latest_labels(output_path)
        unread = [item for item in items
                  if not current_label(item, latest.get(str(item["public"]["item_id"]))) ]
        if not unread:
            break
        if pass_number:
            retry_passes = pass_number
            print(f"P127 {reader} retry pass {pass_number}/{MAX_RETRY_PASSES}: "
                  f"resubmitting {len(unread)} unread labels", flush=True)
        submit(unread, pass_number)

    latest = _latest_labels(output_path)
    unread_ids = [str(item["public"]["item_id"]) for item in items
                  if not current_label(item, latest.get(str(item["public"]["item_id"]))) ]
    if unread_ids:
        raise RuntimeError(f"{reader} labels remain unread after {MAX_RETRY_PASSES} extra retry passes: "
                           + ", ".join(unread_ids))
    return {"reader": reader, "items": len(items), "pending": len(pending),
            "retry_passes": retry_passes, "error_rows": 0, "output": str(output_path),
            "usage_sidecar": str(usage_path), "keys_sidecar": str(key_path)}


def read_known(path: str | Path | None = None) -> list[dict]:
    if path is None:
        return label92b.read_known()
    rows = read_jsonl(repo_path(path))
    if len(rows) != 12:
        raise ValueError(f"known-example inventory must contain 12 rows, found {len(rows)}")
    for index, row in enumerate(rows):
        if type(row.get("expected")) is not bool:
            raise ValueError(f"known example {index} lacks a boolean expected flag")
        if any(not isinstance(row.get(field), str) for field in ("name", "desc", "q", "a")):
            raise ValueError(f"known example {index} has a missing text field")
    return rows


def known_items(known: list[dict]) -> list[dict]:
    items = []
    for index, row in enumerate(known):
        base = {"name": row["name"], "desc": row["desc"],
                "q": row["q"], "a": row["a"]}
        payload = {"known_index": index, **base}
        public = {"item_id": _opaque_id("part127-known", payload),
                  "answer_hash": answer_hash(base)}
        items.append({"public": public,
                      "private": {"item_type": "known", "known_index": index,
                                  "expected": bool(row["expected"])}, **base})
    return items


def known_gate_result(known: list[dict], items: list[dict], labels: dict[str, dict]) -> dict:
    correct = 0
    false_positive = 0
    negative_count = 0
    for source, item in zip(known, items):
        row = labels.get(str(item["public"]["item_id"]), {})
        if (valid_label(row)
                and row.get("answer_hash") == item["public"]["answer_hash"]
                and row.get("shows_persona") is bool(source["expected"])):
            correct += 1
        if not source["expected"]:
            negative_count += 1
            if valid_label(row) and row["shows_persona"]:
                false_positive += 1
    total = len(known)
    return {"correct": correct, "total": total, "minimum": 10,
            "passed": total == 12 and correct >= 10,
            "known_negative_count": negative_count,
            "known_false_positive_count": false_positive,
            "known_false_positive_rate": (false_positive / negative_count
                                          if negative_count else None)}


def _study_items(answer_rows: list[dict], by_id: dict[str, dict]) -> list[dict]:
    items = []
    for answer in answer_rows:
        family_names = ([name for name in run127.FAMILY_TARGETS]
                        if answer["arm"] == "no_push" else [str(answer["family"])])
        if not family_names or any(name not in run127.FAMILY_TARGETS for name in family_names):
            raise ValueError(f"answer row {answer.get('key')} has an unknown target family")
        for family in family_names:
            target = run127.FAMILY_TARGETS[family]
            base = {"name": target["name"], "desc": target["desc"],
                    "q": str(answer["question"]), "a": str(answer["answer"])}
            ahash = answer_hash(base)
            public = {"item_id": _opaque_id("part127-study", {
                "answer_key": answer["key"], "family": family,
                "question_index": int(answer["question_index"]), "answer_hash": ahash}),
                "answer_hash": ahash}
            private = {"item_type": "study", "answer_key": answer["key"],
                       "family": family, "question_index": int(answer["question_index"]),
                       "arm": answer["arm"], "dose": answer.get("dose"),
                       "pair_id": answer.get("pair_id"), "unit_ids": answer.get("unit_ids", [])}
            items.append({"public": public, "private": private, **base})
    if len({item["public"]["item_id"] for item in items}) != len(items):
        raise ValueError("study items do not have unique ids")
    return items


def _require_known_gates(out_dir: Path, known: list[dict], known_items_list: list[dict]) -> dict:
    status = {}
    for reader in READERS:
        path = out_dir / f"known-{reader}.jsonl"
        labels = _latest_labels(path)
        result = known_gate_result(known, known_items_list, labels)
        status[reader] = result
        if not result["passed"]:
            raise RuntimeError(f"reader gate failed for {READER_NAMES[reader]}: "
                               f"{result['correct']}/{result['total']} (need 10/12)")
    return status


def _latest_label_by_id(path: Path) -> dict[str, dict]:
    return _latest_labels(path)


def _label_index(items: list[dict]) -> dict[str, dict]:
    return {str(item["public"]["item_id"]): item for item in items}


def _controls_gate(items: list[dict], by_id: dict[str, dict], known: list[dict],
                   known_gate: dict, out_dir: Path) -> dict:
    item_by_id = _label_index(items)
    pair_rows = []
    labels_by_reader = {reader: _latest_label_by_id(out_dir / f"labels-{reader}.jsonl")
                        for reader in READERS}
    unit_to_family = {uid: unit.get("family") for uid, unit in by_id.items()}
    unit_marks = {reader: {} for reader in READERS}
    for item in items:
        private = item["private"]
        if private["arm"] != "single" or float(private["dose"]) != 1.0:
            continue
        if int(private["question_index"]) not in (0, 1):
            continue
        for reader in READERS:
            row = labels_by_reader[reader].get(str(item["public"]["item_id"]), {})
            unit_id = private["unit_ids"][0]
            unit_marks[reader].setdefault(unit_id, {})[int(private["question_index"])] = (
                bool(row["shows_persona"]) if current_label(item, row) else False)
    present_single_ids = {str(item["private"]["unit_ids"][0]) for item in items
                          if item["private"].get("arm") == "single"
                          and float(item["private"].get("dose", -1)) == 1.0
                          and int(item["private"].get("question_index", -1)) in (0, 1)
                          and item["private"].get("unit_ids")}
    pairs = [pair for pair in run127.selected_pairs(by_id, "main")
             if pair["a"] in present_single_ids and pair["b"] in present_single_ids]
    for pair in pairs:
        reader_status = {}
        for reader in READERS:
            a_show = any(unit_marks[reader].get(pair["a"], {}).values())
            b_show = any(unit_marks[reader].get(pair["b"], {}).values())
            reader_status[reader] = {"a_has_mark": a_show, "b_has_mark": b_show,
                                     "passed": a_show and b_show}
        pair_rows.append({"pair_id": pair["pair_id"], "family": pair["family"],
                          "same_layer": pair["same_layer"], "readers": reader_status,
                          "passed": all(reader_status[reader]["passed"] for reader in READERS)})

    no_push = {reader: {"observed": 0, "items": 0, "unread": 0} for reader in READERS}
    for item in items:
        private = item["private"]
        if private["arm"] != "no_push" or int(private["question_index"]) not in (0, 1):
            continue
        for reader in READERS:
            row = labels_by_reader[reader].get(str(item["public"]["item_id"]), {})
            no_push[reader]["items"] += 1
            readable = current_label(item, row)
            no_push[reader]["unread"] += int(not readable)
            no_push[reader]["observed"] += int(readable and bool(row["shows_persona"]))
    negative = {}
    for reader in READERS:
        gate = known_gate[reader]
        fp_rate = gate["known_false_positive_rate"]
        threshold = math.ceil(10 * float(fp_rate)) if fp_rate is not None else 0
        observed = no_push[reader]["observed"]
        negative[reader] = {"known_false_positive_count": gate["known_false_positive_count"],
                            "known_negative_count": gate["known_negative_count"],
                            "known_false_positive_rate": fp_rate,
                            "no_push_target_items": no_push[reader]["items"],
                            "unread_items": no_push[reader]["unread"],
                            "no_push_marks": observed, "threshold": threshold,
                            "passed": (no_push[reader]["items"] == 10
                                       and no_push[reader]["unread"] == 0
                                       and observed <= threshold)}
    eligible = [row["pair_id"] for row in pair_rows if row["passed"]]
    same_layer_eligible = [row["pair_id"] for row in pair_rows
                           if row["same_layer"] and row["passed"]]
    result = {"known_example_gates": known_gate, "positive_controls": pair_rows,
              "eligible_pair_ids": eligible, "same_layer_eligible_pair_ids": same_layer_eligible,
              "negative_control_gates": negative,
              "negative_control_passed": all(negative[reader]["passed"] for reader in READERS),
              "same_layer_gate_passed": bool(same_layer_eligible)}
    return result


def _mandatory_control_items(items: list[dict]) -> list[dict]:
    result = []
    for item in items:
        private = item["private"]
        if private["arm"] == "no_push":
            result.append(item)
        elif private["arm"] == "single" and float(private["dose"]) == 1.0:
            result.append(item)
    return result


def _choose_luna_items(items: list[dict], ds_labels: dict[str, dict]) -> tuple[list[dict], dict]:
    mandatory = _mandatory_control_items(items)
    positive = []
    negative = []
    unread = []
    for item in items:
        label = ds_labels.get(str(item["public"]["item_id"]), {})
        if not current_label(item, label):
            unread.append(item)
        elif label["shows_persona"] and label["coherent"]:
            positive.append(item)
        else:
            negative.append(item)
    positive_ids = {item["public"]["item_id"] for item in positive}
    # The seeded negative sample is the same size as the compound-positive set.
    sample_n = min(len(positive), len(negative))
    negative_sample = random.Random(LUNA_SEED).sample(negative, sample_n)
    selected = {item["public"]["item_id"]: item for item in mandatory}
    selected.update({item["public"]["item_id"]: item for item in positive})
    selected.update({item["public"]["item_id"]: item for item in negative_sample})
    readable_count = len(positive) + len(negative)
    report = {"seed": LUNA_SEED, "ds_items": len(items),
              "ds_compound_positive_items": len(positive),
              "ds_unread_items": len(unread),
              "unread_item_ids": sorted(item["public"]["item_id"] for item in unread),
              "ds_compound_positive_rate":
                  len(positive) / readable_count if readable_count else None,
              "negative_sample_size": sample_n,
              "mandatory_full_dose_and_no_push_items": len(mandatory),
              "selected_items": len(selected),
              "negative_sample_ids": [item["public"]["item_id"] for item in negative_sample],
              "positive_item_ids": sorted(positive_ids)}
    return list(selected.values()), report


def execute_phase(phase: str, out: str | Path, neurons: str | Path = run127.DEFAULT_NEURONS,
                  known_path: str | Path = KNOWN_PATH, *, workers: int = 8,
                  request_fn: Callable | None = None,
                  sleep_fn: Callable[[float], None] = time.sleep) -> dict:
    out_dir = repo_path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    known = read_known(known_path)
    known_items_list = known_items(known)
    if phase not in ("gate", "controls", "study"):
        raise ValueError("phase must be gate, controls, or study")
    if phase == "gate":
        results = {}
        for reader in READERS:
            result = label_items(known_items_list, out_dir / f"known-{reader}.jsonl", reader,
                                 workers=workers, request_fn=request_fn, sleep_fn=sleep_fn)
            labels = _latest_labels(out_dir / f"known-{reader}.jsonl")
            gate = known_gate_result(known, known_items_list, labels)
            results[reader] = {**result, **gate}
            print(f"P127 known-example gate {READER_NAMES[reader]}: "
                  f"{gate['correct']}/{gate['total']} {'PASS' if gate['passed'] else 'FAIL'}",
                  flush=True)
        write_json(out_dir / "reader_gates.json", results)
        failed_readers = [READER_NAMES[reader] for reader, result in results.items()
                          if not result["passed"]]
        if failed_readers:
            raise RuntimeError("known-example gate failed for " + ", ".join(failed_readers))
        return results

    run_meta_path = out_dir / "run_meta.json"
    if not run_meta_path.is_file():
        raise RuntimeError("run127 is incomplete; answers and run_meta.json are required before labels")
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    if run_meta.get("status") != "complete":
        raise RuntimeError("run127 is incomplete; labels are refused")
    known_gate = _require_known_gates(out_dir, known, known_items_list)
    by_id = run127.load_inventory(neurons)
    answer_rows = read_jsonl(out_dir / "answers.jsonl")
    if len({row["key"] for row in answer_rows}) != int(run_meta["answer_count"]):
        raise RuntimeError("run127 is incomplete; answer keys are missing or duplicated")
    items = _study_items(answer_rows, by_id)
    if phase == "controls":
        mandatory = _mandatory_control_items(items)
        for reader in READERS:
            result = label_items(mandatory, out_dir / f"labels-{reader}.jsonl", reader,
                                 workers=workers, request_fn=request_fn, sleep_fn=sleep_fn)
            print(f"P127 controls {READER_NAMES[reader]}: {result['items']} items, "
                  f"{result['pending']} new, {result['error_rows']} errors", flush=True)
        gates = _controls_gate(items, by_id, known, known_gate, out_dir)
        write_json(out_dir / "control_gates.json", gates)
        print("P127 eligible pairs: " + ", ".join(gates["eligible_pair_ids"]), flush=True)
        if not gates["same_layer_gate_passed"]:
            print("P127 no same-layer pair passed the positive-control gate; main reader labels are withheld",
                  flush=True)
        return gates

    control_path = out_dir / "control_gates.json"
    if not control_path.is_file():
        raise RuntimeError("run label127.py --phase controls first; eligibility must be printed before study labels")
    gates = json.loads(control_path.read_text(encoding="utf-8"))
    if not gates.get("same_layer_gate_passed"):
        raise RuntimeError("no same-layer pair passed the positive-control gate; study labels are withheld")
    for reader in READERS:
        if not gates["negative_control_gates"][reader]["passed"]:
            print(f"P127 warning: {READER_NAMES[reader]} negative-control gate failed; headline is withdrawn",
                  flush=True)
    ds_result = label_items(items, out_dir / "labels-ds.jsonl", "ds", workers=workers,
                            request_fn=request_fn, sleep_fn=sleep_fn)
    ds_labels = _latest_labels(out_dir / "labels-ds.jsonl")
    luna_items, sample = _choose_luna_items(items, ds_labels)
    write_json(out_dir / "luna_sample.json", sample)
    luna_result = label_items(luna_items, out_dir / "labels-luna.jsonl", "luna", workers=workers,
                              request_fn=request_fn, sleep_fn=sleep_fn)
    results = {"deepseek": ds_result, "luna": luna_result, "luna_sample": sample,
               "known_example_gates": known_gate, "control_gates": gates}
    write_json(out_dir / "label_meta.json", results)
    print(f"P127 labels complete: DeepSeek {ds_result['items']} items, "
          f"Luna {luna_result['items']} items; DS compound positive rate="
          f"{sample['ds_compound_positive_rate']:.4f}", flush=True)
    return results


def main(argv: list[str] | None = None, *, request_fn: Callable | None = None,
         sleep_fn: Callable[[float], None] = time.sleep) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("gate", "controls", "study"), required=True)
    parser.add_argument("--out", required=True, help="run127 output directory")
    parser.add_argument("--neurons", default=run127.DEFAULT_NEURONS)
    parser.add_argument("--known", default=KNOWN_PATH)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        execute_phase(args.phase, args.out, args.neurons, args.known,
                      workers=args.workers, request_fn=request_fn, sleep_fn=sleep_fn)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"label127: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
