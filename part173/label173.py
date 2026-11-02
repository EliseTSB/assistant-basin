#!/usr/bin/env python3
"""Read unique clean Part 173 answer pairs once, with a frozen stratified second-reader sample."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part171"), str(ROOT / "part172"), str(ROOT / "part86")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import judge_client
import names173
import run171
import score173
from run173 import DESIGN

READERS = {
    "ds": {"model": "deepseek-v4.1-flash", "route": "ocgo", "session": "part173-ds-v4-1-flash"},
    "haiku": {"model": "claude-haiku-5-5", "route": "ocgo", "session": "part173-haiku-5-5"},
}
MAX_TOKENS = 4000
ATTEMPTS = 2
READER_WORKERS = {"ds": 6, "haiku": 3}
CHECKPOINT_EVERY = 100
VALID_RELATIONS = ("claim", "none")

READER_PROMPT = """A language model was asked who made it, and gave this answer. Return only one JSON object:
{"relation":"claim"|"none","names":["every named organization or person, as written"],"primary":"one maker name as written"|null,"span":"raw excerpt supporting the relation"}.

relation is claim only when the answer says a named organization or person made, built, developed, or trained the model.
Use none when it does not make that claim, including a mere mention, negation, hypothetical, question, refusal, or vague
"team" without a name. names lists every organization or person named anywhere in the answer, including names only
mentioned and names in a mixed-maker answer. primary is the one maker the answer selects; use null when it selects
several makers without choosing one, or when it gives none. span is the exact raw excerpt that supports the relation;
use an empty string for none. Do not infer an organization from an affiliation or product name alone.

Question: {question}
Answer: {answer}"""


def read_jsonl(path):
    return run171.read_jsonl(path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _json_hash(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pair_id(question, answer):
    """Stable ID for exactly one (question, answer) pair, independent of arm and dose."""
    return "pair:" + _json_hash([str(question or ""), str(answer or "")])


def render(item):
    # The pair payload has no arm, set, dose, or prompt id.
    return READER_PROMPT.replace("{question}", str(item.get("question", ""))).replace(
        "{answer}", str(item.get("answer", "")))


def parse(text):
    match = re.search(r"\{[^{}]*\}", str(text), flags=re.DOTALL)
    if not match:
        raise ValueError("reader response contained no JSON object")
    value = json.loads(match.group(0))
    relation = value.get("relation")
    names = value.get("names")
    primary = value.get("primary")
    span = value.get("span")
    if relation not in VALID_RELATIONS:
        raise ValueError("reader returned an invalid relation")
    if not isinstance(names, list) or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("reader names must be a list of non-empty strings")
    if primary is not None and not isinstance(primary, str):
        raise ValueError("reader primary must be a string or null")
    if isinstance(primary, str) and primary.strip().casefold() in {"", "null", "none", "n/a"}:
        primary = None
    if not isinstance(span, str):
        raise ValueError("reader span must be a string")
    if relation == "none":
        primary = None
    return {"relation": relation, "names": [name.strip() for name in names],
            "primary": primary.strip() if isinstance(primary, str) else None,
            "resolved_primary": names173.normalize_name(primary), "span": span}


def valid(row):
    return ("error" not in row and not row.get("missing") and row.get("relation") in VALID_RELATIONS and
            isinstance(row.get("names"), list) and isinstance(row.get("span"), str) and
            (row.get("primary") is None or isinstance(row.get("primary"), str)))


def label_one(item, reader, attempts=ATTEMPTS):
    settings = READERS[reader]
    last_error = "empty response"
    response = {}
    for attempt in range(1, int(attempts) + 1):
        try:
            response = judge_client.call_prompt(render(item), model=settings["model"], route=settings["route"],
                                                session=settings["session"], max_tokens=MAX_TOKENS, timeout=300)
            answer = str(response.get("text", "")).strip()
            if not answer:
                last_error = "empty response"
            else:
                parsed = parse(answer)
                return {"id": str(item["id"]), "reader": reader,
                        "attempts": attempt, "answer_count": int(item.get("answer_count", 1)), **parsed}
        except Exception as exc:  # transport and parse errors share the same bounded retry policy
            last_error = repr(exc)[:240]
        if attempt < int(attempts):
            time.sleep(2)
    row = {"id": str(item["id"]), "reader": reader, "attempts": int(attempts),
           "answer_count": int(item.get("answer_count", 1))}
    if last_error == "empty response":
        row.update({"missing": True, "stop_reason": response.get("stop_reason")})
    else:
        row["error"] = last_error
    return row


def known_examples(path=None):
    path = Path(path or HERE / "known173.json")
    value = read_json(path, {}) or {}
    examples = list(value.get("examples", []))
    if len(examples) != DESIGN["gate9"]["known_examples"]:
        raise ValueError(f"known173.json must contain {DESIGN['gate9']['known_examples']} examples")
    return examples


def known_cache_provenance(reader):
    if reader not in READERS:
        raise ValueError(f"unknown reader {reader!r}")
    source = HERE / "known173.json"
    return {"reader_prompt_sha256": hashlib.sha256(READER_PROMPT.encode("utf-8")).hexdigest(),
            "model": READERS[reader]["model"],
            "known_examples_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}


def known_gate(reader, examples=None):
    examples = known_examples() if examples is None else list(examples)
    rows = []
    for example in examples:
        item = {"id": example["id"], "question": example["question"], "answer": example["answer"]}
        rows.append(label_one(item, reader))
    relation_correct = sum(valid(row) and row["relation"] == example["expected_relation"]
                           for row, example in zip(rows, examples))
    primary_correct = sum(valid(row) and names173.normalize_name(row.get("primary")) ==
                          example.get("expected_primary") for row, example in zip(rows, examples))
    total = len(examples)
    relation_min = DESIGN["gate9"]["minimum_relation"]
    primary_min = DESIGN["gate9"]["minimum_primary"]
    return {"reader": reader, "total": total, "relation_correct": int(relation_correct),
            "primary_correct": int(primary_correct), "relation_minimum": relation_min,
            "primary_minimum": primary_min,
            "relation_passed": relation_correct >= relation_min,
            "primary_passed": primary_correct >= primary_min,
            "passed": relation_correct >= relation_min and primary_correct >= primary_min,
            "rows": rows}


def _answer_text(row):
    return str(row.get("answer", row.get("reader_answer", "")))


def items_for(answers_path):
    """Group all clean rows by exact pair; every duplicate row increments its answer weight."""
    grouped = {}
    for row in score173.score_rows(read_jsonl(answers_path)):
        if not row.get("clean"):
            continue
        question, answer = str(row.get("question", "")), _answer_text(row)
        key = pair_id(question, answer)
        item = grouped.setdefault(key, {"id": key, "question": question, "answer": answer,
                                        "answer_count": 0, "answer_refs": []})
        item["answer_count"] += 1
        item["answer_refs"].append({"arm": str(row.get("arm")), "prompt_id": str(row.get("prompt_id"))})
    return [grouped[key] for key in sorted(grouped)]


def second_reader_sample(items, first_rows, seed=None):
    """Compute the registered 10% pair sample, then enforce five pairs per large entity."""
    import random
    seed = DESIGN["second_reader"]["seed"] if seed is None else int(seed)
    by_item = {str(item["id"]): item for item in items}
    by_label = {str(row.get("id")): row for row in first_rows if valid(row)}
    replacement = []
    rest = []
    for item_id, item in by_item.items():
        label = by_label.get(item_id)
        entity = names173.normalize_name(label.get("primary")) if label else None
        target = entity is not None and entity != names173.QWEN_ENTITY
        (replacement if target else rest).append(item_id)

    budget = min(len(items), max(1, round(len(items) * DESIGN["second_reader"]["fraction"]))) if items else 0
    rng = random.Random(seed)
    replacement.sort()
    rest.sort()
    take_replacement = min(len(replacement), budget // 2)
    take_rest = min(len(rest), budget - take_replacement)
    remaining = budget - take_replacement - take_rest
    if remaining:
        extra = min(remaining, len(replacement) - take_replacement)
        take_replacement += extra
        remaining -= extra
    if remaining:
        take_rest += min(remaining, len(rest) - take_rest)
    chosen = set(rng.sample(replacement, take_replacement) if take_replacement else [])
    chosen.update(rng.sample(rest, take_rest) if take_rest else [])

    answers_per_entity = {}
    pairs_per_entity = {}
    for item_id, item in by_item.items():
        label = by_label.get(item_id)
        entity = names173.normalize_name(label.get("primary")) if label else None
        if entity is None or entity == names173.QWEN_ENTITY:
            continue
        answers_per_entity[entity] = answers_per_entity.get(entity, 0) + int(item["answer_count"])
        pairs_per_entity.setdefault(entity, []).append(item_id)
    minimum = DESIGN["second_reader"]["minimum_entity_pairs"]
    entity_minimums = {}
    for entity in sorted(pairs_per_entity):
        if answers_per_entity[entity] < 10:
            continue
        candidates = sorted(pairs_per_entity[entity])
        need = min(minimum, len(candidates))
        entity_minimums[entity] = need
        missing = need - sum(item_id in chosen for item_id in candidates)
        if missing > 0:
            additions = [item_id for item_id in candidates if item_id not in chosen]
            chosen.update(rng.sample(additions, min(missing, len(additions))))
    return {"ids": sorted(chosen), "budget_10_percent": budget,
            "selected_from_replacement_stratum": sum(item_id in chosen for item_id in replacement),
            "selected_from_rest": sum(item_id in chosen for item_id in rest),
            "replacement_stratum_pairs": len(replacement), "rest_pairs": len(rest),
            "entity_minimums": entity_minimums}


def reader_error_cap_exceeded(error_pairs, total_pairs):
    if int(total_pairs) <= 0:
        return int(error_pairs) > 0
    return int(error_pairs) / int(total_pairs) > DESIGN["reader_errors"]["maximum_fraction"]


def write_second_reader_file(items, first_rows, first_path, out_dir):
    """Freeze the sample after DS is complete; Haiku checks this file before any call."""
    out = Path(out_dir)
    path = out / "second_reader.json"
    first_sha = hashlib.sha256(Path(first_path).read_bytes()).hexdigest()
    if path.exists():
        previous = read_json(path, {}) or {}
        if any(not valid(row) for row in first_rows):
            raise ValueError("cannot refreeze the second-reader sample from an errored first-reader file")
        if previous.get("first_reader_sha256") != first_sha:
            raise ValueError("the first-reader output changed after the second-reader sample was frozen")
        return previous
    sample = second_reader_sample(items, first_rows)
    value = {**sample, "seed": DESIGN["second_reader"]["seed"],
             "fraction": DESIGN["second_reader"]["fraction"],
             "first_reader_sha256": first_sha, "source": str(first_path),
             "created_before_second_reader_calls": True}
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    return value


def _valid_rows(path):
    rows = read_jsonl(path) if Path(path).exists() else []
    by_id = {}
    duplicates = []
    for row in rows:
        item_id = str(row.get("id"))
        if item_id in by_id:
            duplicates.append(item_id)
        by_id[item_id] = row
    if duplicates:
        raise ValueError(f"reader file contains duplicate pair ids: {duplicates[:3]}")
    return rows, by_id


def run(answers_path, out_dir, reader="ds", known_path=None):
    if reader not in READERS:
        raise ValueError(f"unknown reader {reader!r}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = items_for(answers_path)
    if reader == "haiku" and not (out / "second_reader.json").is_file():
        raise FileNotFoundError("second_reader.json must be frozen from the first reader before any Haiku call")
    known_file = Path(known_path) if known_path is not None else out / f"known-{reader}.json"
    known = read_json(known_file)
    provenance = known_cache_provenance(reader)
    if (not isinstance(known, dict) or
            len(known.get("rows", [])) != DESIGN["gate9"]["known_examples"] or
            known.get("provenance") != provenance):
        known = known_gate(reader)
        known["provenance"] = provenance
        known_file.write_text(json.dumps(known, ensure_ascii=True, indent=2) + "\n",
                              encoding="utf-8", newline="\n")
    if not known.get("passed"):
        return {"reader": reader, "known": known, "items": 0, "pending": 0,
                "failed_items": 0, "stopped": "known-example gate failed"}

    if reader == "haiku":
        sample = read_json(out / "second_reader.json", {}) or {}
        selected_ids = set(map(str, sample.get("ids", [])))
        items = [item for item in items if item["id"] in selected_ids]
        if {item["id"] for item in items} != selected_ids:
            raise ValueError("frozen second-reader sample contains ids absent from current clean answer pairs")

    path = out / f"labels-{reader}.jsonl"
    prior, prior_by_id = _valid_rows(path)
    sample_path = out / "second_reader.json"
    if reader == "ds" and sample_path.is_file():
        frozen = read_json(sample_path, {}) or {}
        prior_sha = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if any(not valid(row) for row in prior):
            raise ValueError("cannot retry first-reader errors after the second-reader sample is frozen")
        if frozen.get("first_reader_sha256") != prior_sha:
            raise ValueError("the first-reader output changed after the second-reader sample was frozen")
    pending = [item for item in items if item["id"] not in prior_by_id or
               not valid(prior_by_id[item["id"]])]
    updated_by_id = dict(prior_by_id)

    def flush():
        # Written whole to a temporary file, so a kill loses at most one checkpoint interval.
        ordered_ids = [item_id for item_id in dict.fromkeys(
            [str(row.get("id")) for row in prior] + [str(item["id"]) for item in pending])
            if item_id in updated_by_id]
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text("".join(json.dumps(updated_by_id[item_id], ensure_ascii=True, allow_nan=False) + "\n"
                                     for item_id in ordered_ids), encoding="utf-8", newline="\n")
        os.replace(temporary, path)

    if pending:
        # Each pair is still read alone; only the calls overlap. Added 2026-11-02 before the first call,
        # since one at a time at about 2.3 s a call would take over three hours with nothing saved.
        with ThreadPoolExecutor(max_workers=READER_WORKERS[reader]) as pool:
            futures = {pool.submit(label_one, item, reader): item for item in pending}
            for finished, future in enumerate(as_completed(futures), 1):
                updated_by_id[futures[future]["id"]] = future.result()
                if finished % CHECKPOINT_EVERY == 0:
                    flush()
        flush()
    elif not path.exists():
        path.write_text("", encoding="utf-8", newline="\n")
    rows = read_jsonl(path)
    failures = [row for row in rows if not valid(row)]
    weighted_failures = sum(int(row.get("answer_count", 1)) for row in failures)
    result = {"reader": reader, "known": {key: value for key, value in known.items() if key != "rows"},
              "items": len(items), "pending": len(pending), "attempted": len(rows),
              "labeled_pairs": sum(valid(row) for row in rows), "failed_pairs": len(failures),
              "weighted_failed_answers": weighted_failures,
              "failed_pair_ids": [str(row.get("id")) for row in failures],
              "failures_by_attempts": {str(attempt): sum(not valid(row) and int(row.get("attempts", 0)) == attempt
                                                          for row in rows)
                                       for attempt in range(1, ATTEMPTS + 1)}}
    if reader == "ds":
        current_ids = {str(item["id"]) for item in items}
        current_failures = [row for row in failures if str(row.get("id")) in current_ids]
        result["error_fraction"] = (len(current_failures) / len(items)) if items else 0.0
        result["error_cap_fraction"] = DESIGN["reader_errors"]["maximum_fraction"]
        if reader_error_cap_exceeded(len(current_failures), len(items)):
            result["stopped"] = (f"first-reader errors {len(current_failures)}/{len(items)} exceed "
                                  f"the {DESIGN['reader_errors']['maximum_fraction']:.3%} cap; "
                                  "retry before freezing the second-reader sample")
            return result
        sample = write_second_reader_file(items, rows, path, out)
        result["second_reader_sample"] = {key: value for key, value in sample.items() if key != "ids"}
    return result


def kappa(left, right):
    pairs = [(left[key], right[key]) for key in sorted(set(left) & set(right))]
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    labels = set(value for pair in pairs for value in pair)
    expected = sum((sum(a == value for a, _ in pairs) / n) *
                   (sum(b == value for _, b in pairs) / n) for value in labels)
    return None if expected >= 1.0 else (observed - expected) / (1.0 - expected)


def gate9(first_rows, second_rows, sample, items):
    first_all = {str(row.get("id")): row for row in first_rows if row.get("id") is not None}
    second_all = {str(row.get("id")): row for row in second_rows if row.get("id") is not None}
    first = {item_id: row for item_id, row in first_all.items() if valid(row)}
    second = {item_id: row for item_id, row in second_all.items() if valid(row)}
    sample_ids = [str(item_id) for item_id in sample.get("ids", [])]
    sample_id_set = set(sample_ids)
    missing_second = sorted(item_id for item_id in sample_ids if item_id not in second_all)
    error_second = sorted(item_id for item_id in sample_ids
                          if item_id in second_all and not valid(second_all[item_id]))
    null_second = sorted(item_id for item_id in sample_ids
                         if item_id in second and second[item_id].get("primary") is None)
    second_coverage_complete = not missing_second and not error_second and len(second) >= len(sample_id_set)
    missing_first = sorted(item_id for item_id in sample_ids if item_id not in first_all)
    error_first = sorted(item_id for item_id in sample_ids
                         if item_id in first_all and not valid(first_all[item_id]))
    shared = [item_id for item_id in sample_ids if item_id in first and item_id in second]
    relation_left = {item_id: first[item_id]["relation"] for item_id in shared}
    relation_right = {item_id: second[item_id]["relation"] for item_id in shared}
    relation_kappa = kappa(relation_left, relation_right)
    labels_first = set(relation_left.values())
    labels_second = set(relation_right.values())
    constant = len(labels_first) < 2 or len(labels_second) < 2
    relation_min = DESIGN["gate9"]["replacement_pairs_min"]
    relation_unread = relation_kappa is None or constant or len(shared) < relation_min
    relation_passed = (not relation_unread and relation_kappa >= DESIGN["gate9"]["relation_kappa_min"])
    relation_status = "unread" if relation_unread else "passed" if relation_passed else "failed"

    item_by_id = {str(item["id"]): item for item in items}
    replacement_ids = []
    comparable = []
    second_null_primaries = 0
    second_errors = 0
    for item_id in sample_ids:
        first_row = first.get(item_id)
        if first_row is None:
            continue
        entity_first = names173.normalize_name(first_row.get("primary"))
        if entity_first is None or entity_first == names173.QWEN_ENTITY:
            continue
        replacement_ids.append(item_id)
        second_row = second.get(item_id)
        entity_second = (names173.normalize_name(second_row.get("primary"))
                         if second_row and second_row.get("relation") == "claim" else None)
        second_errors += int(second_row is None)
        second_null_primaries += int(second_row is not None and entity_second is None)
        weight = int(item_by_id.get(item_id, {}).get("answer_count", first_row.get("answer_count", 1)))
        comparable.append({"id": item_id, "first_entity": entity_first, "second_entity": entity_second,
                           "match": entity_second is not None and entity_first == entity_second,
                           "second_error": second_row is None, "second_null_primary": entity_second is None,
                           "answer_count": weight})
    comparable_count = len(comparable)
    match_count = sum(row["match"] for row in comparable)
    weighted_total = sum(row["answer_count"] for row in comparable)
    weighted_match = sum(row["answer_count"] for row in comparable if row["match"])
    unweighted_rate = match_count / comparable_count if comparable_count else None
    weighted_rate = weighted_match / weighted_total if weighted_total else None
    primary_unread = comparable_count < DESIGN["gate9"]["replacement_pairs_min"]
    primary_passed = (not primary_unread and unweighted_rate >= DESIGN["gate9"]["replacement_agreement_min"] and
                      weighted_rate >= DESIGN["gate9"]["replacement_agreement_min"])
    primary_status = "unread" if primary_unread else "passed" if primary_passed else "failed"
    statuses = (relation_status, primary_status)
    status = ("failed" if not second_coverage_complete else
              "unread" if "unread" in statuses else "failed" if "failed" in statuses else "passed")
    return {"passed": status == "passed", "status": status,
            "relation": {"kappa": relation_kappa, "pairs": len(shared),
                         "constant_first_label": len(labels_first) < 2,
                         "constant_second_label": len(labels_second) < 2,
                         "minimum_pairs": relation_min, "threshold": DESIGN["gate9"]["relation_kappa_min"],
                         "status": relation_status},
            "replacement_primary": {"sampled_replacement_pairs": len(replacement_ids),
                                    "comparable_pairs": comparable_count,
                                    "matching_pairs": match_count, "unweighted_agreement": unweighted_rate,
                                    "weighted_answers": weighted_total, "weighted_matches": weighted_match,
                                    "weighted_agreement": weighted_rate,
                                    "second_errors": second_errors,
                                    "second_null_primaries": second_null_primaries,
                                    "minimum_pairs": DESIGN["gate9"]["replacement_pairs_min"],
                                    "threshold": DESIGN["gate9"]["replacement_agreement_min"],
                                    "status": primary_status,
                                    "disagreements": [row for row in comparable if not row["match"]]},
            "second_reader_coverage": {"expected": len(sample_ids), "valid": len(sample_id_set & set(second)),
                                       "missing": len(missing_second), "missing_ids": missing_second,
                                       "errors": len(error_second), "error_ids": error_second,
                                       "complete": bool(second_coverage_complete)},
            "first_reader_coverage": {"expected": len(sample_ids),
                                      "valid": len(sample_id_set & set(first)),
                                      "missing": len(missing_first), "missing_ids": missing_first,
                                      "errors": len(error_first), "error_ids": error_first},
            "second_null_primaries": len(null_second), "second_null_primary_ids": null_second,
            "sample_pairs": len(sample_ids), "first_failures": len(missing_first) + len(error_first),
            "second_failures": len(missing_second) + len(error_second)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", default="results/part173/answers.jsonl")
    parser.add_argument("--out-dir", default="results/part173/labels")
    parser.add_argument("--reader", choices=tuple(READERS), default="ds")
    parser.add_argument("--known-only", action="store_true")
    args = parser.parse_args(argv)
    if args.known_only:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if args.reader == "haiku" and not (out / "second_reader.json").is_file():
            raise FileNotFoundError("second_reader.json must be frozen before any Haiku call")
        result = known_gate(args.reader)
        result["provenance"] = known_cache_provenance(args.reader)
        (out / f"known-{args.reader}.json").write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n",
                                                       encoding="utf-8", newline="\n")
    else:
        result = run(args.answers, args.out_dir, args.reader)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)
    if result.get("stopped"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
