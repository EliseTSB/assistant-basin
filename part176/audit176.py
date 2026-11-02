#!/usr/bin/env python3
"""Blind second-reader audit of Part 176's new short-answer and transition states."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part171"), str(ROOT / "part172"),
              str(ROOT / "part173"), str(ROOT / "part174"), str(ROOT / "part175")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import label173
import label174
import names173
import run171
import run175
import score173
import score176

AUDIT_CLASSES = ("echo", "role swap", "coherent refusal", "other maker",
                 "stopped mid-sentence", "other")
AUDIT_PROMPT = """Classify the answer's form, without judging whether its factual claim is true.
Return only JSON: {\"class\": one of [\"echo\", \"role swap\", \"coherent refusal\", \"other maker\", \"stopped mid-sentence\", \"other\"].
echo: repeats or lightly rephrases the question instead of answering it.
role swap: reverses who is the assistant and who is the user, such as saying the assistant created the user.
coherent refusal: a complete, understandable refusal or statement that it cannot know or disclose.
other maker: a coherent answer that names a maker or training organization.
stopped mid-sentence: cut off before a complete thought.
other: none of the above.
The rung, direction, and experiment condition are hidden. Read only this question and answer.

Question: {question}
Answer: {answer}"""


def _read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def _parse_class(text):
    match = re.search(r"\{[^{}]*\}", str(text), flags=re.DOTALL)
    if not match:
        raise ValueError("audit reader response contained no JSON object")
    value = json.loads(match.group(0))
    label = value.get("class")
    if label not in AUDIT_CLASSES:
        raise ValueError(f"audit reader returned an unknown class {label!r}")
    return label


def call_reader(item, call_prompt=None, sleep=time.sleep):
    """Use Part 173's client and Part 174's shared second-reader configuration."""
    if call_prompt is None:
        settings = label174.READERS["haiku"]
        call_prompt = label173.judge_client.call_prompt
        prompt = AUDIT_PROMPT.replace("{question}", str(item.get("question", ""))).replace(
            "{answer}", str(item.get("answer", "")))
        last_error = "empty response"
        response = {}
        for attempt in range(1, 3):
            try:
                response = call_prompt(prompt, model=settings["model"], route=settings["route"],
                                       session=settings["session"], max_tokens=256, timeout=300)
                raw = str(response.get("text", "")).strip()
                if raw:
                    return {"class": _parse_class(raw), "attempts": attempt,
                            "model": settings["model"]}
                last_error = "empty response"
            except Exception as exc:
                last_error = repr(exc)[:240]
            if attempt < 2:
                sleep(2)
        return {"class": None, "attempts": 2, "model": settings["model"], "error": last_error}
    result = call_prompt(item)
    if isinstance(result, str):
        return {"class": _parse_class(result), "attempts": 1, "model": "stub"}
    label = result.get("class") if isinstance(result, dict) else None
    if label not in AUDIT_CLASSES:
        raise ValueError(f"stub audit reader returned an unknown class {label!r}")
    return {"class": label, "attempts": int(result.get("attempts", 1)),
            "model": result.get("model", "stub")}


def check_known(known_path=None, reader_fn=None):
    known_path = Path(known_path or HERE / "audit_known.json")
    examples = list((_read_json(known_path, {}) or {}).get("examples", []))
    required = int((_read_json(known_path, {}) or {}).get("required_correct", 8))
    if len(examples) != 10:
        raise ValueError("audit_known.json must contain exactly ten hand-written examples")
    reader_fn = call_reader if reader_fn is None else reader_fn
    rows = []
    for example in examples:
        result = reader_fn({"question": example["question"], "answer": example["answer"]})
        actual = result.get("class")
        rows.append({"id": example["id"], "expected": example["expected_class"],
                     "observed": actual, "correct": actual == example["expected_class"],
                     "error": result.get("error")})
    correct = sum(row["correct"] for row in rows)
    return {"passed": correct >= required, "correct": correct, "total": len(rows),
            "required": required, "rows": rows}


def _candidate_records(answers_path, labels_path):
    raw = run171.read_jsonl(answers_path)
    # Label files are checked by Part 174's duplicate-safe file reader.
    first_rows, _ = label174._valid_file(labels_path)
    scored = score173.score_rows(raw)
    reader_by_row = score176.score175._reader_map(scored, first_rows)
    candidates = {"fragment": defaultdict(list), "longer no claim": [],
                  "claim with no single name": []}
    for row in scored:
        # Audit only the fresh twenty-draw, five-rung sample. Control arms and
        # the two Part 174 replay arms are not part of this frozen sample.
        set_name = str(row.get("set", ""))
        if set_name not in {f"f100d_d{seed}" for seed in score176.run176.SEEDS}:
            continue
        try:
            if abs(float(row.get("dose", float("nan")))) not in score176.run176.DOSES:
                continue
        except (TypeError, ValueError):
            continue
        if not row.get("clean"):
            continue
        record = reader_by_row.get((str(row.get("arm")), str(row.get("prompt_id"))))
        if not record or not label173.valid(record):
            continue
        token_count = score176._token_count(row)
        own_name = score176._own_name(row.get("answer", ""))
        if record.get("relation") == "none" and not own_name and token_count is not None:
            rung = abs(float(row.get("dose", 0.0)))
            if token_count <= 12:
                candidates["fragment"][rung].append(row)
            else:
                candidates["longer no claim"].append(row)
        elif record.get("relation") == "claim" and \
                score173._primary_entity(record, mode="canonical") is None:
            candidates["claim with no single name"].append(row)
    return candidates


def _sample_even_by_rung(by_rung, n, rng):
    keys = sorted(by_rung)
    pools = {key: list(by_rung[key]) for key in keys}
    for pool in pools.values():
        rng.shuffle(pool)
    target = min(int(n), sum(map(len, pools.values())))
    if not keys:
        return []
    chosen = []
    while len(chosen) < target:
        progressed = False
        for key in keys:
            if pools[key] and len(chosen) < target:
                chosen.append((key, pools[key].pop()))
                progressed = True
        if not progressed:
            break
    return chosen


def freeze_sample(candidates, seed=176):
    rng = random.Random(int(seed))
    fragments = _sample_even_by_rung(candidates["fragment"], 100, rng)
    longer = list(candidates["longer no claim"])
    claims = list(candidates["claim with no single name"])
    rng.shuffle(longer)
    rng.shuffle(claims)
    selected = []
    for rung, row in fragments:
        selected.append(("fragment", rung, row))
    selected.extend(("longer no claim", abs(float(row.get("dose", 0))), row)
                    for row in longer[:50])
    selected.extend(("claim with no single name", abs(float(row.get("dose", 0))), row)
                    for row in claims[:25])
    # The reader receives only the question and answer; this identifier carries no rung data.
    result = []
    for index, (kind, rung, row) in enumerate(selected):
        result.append({"audit_id": f"part176-audit-{index:04d}", "sample_kind": kind,
                       "rung_hidden_from_reader": True, "dose_for_table": float(rung),
                       "set": row.get("set"), "arm": row.get("arm"),
                       "prompt_id": row.get("prompt_id"), "question": row.get("question", ""),
                       "answer": row.get("answer", "")})
    return result


def audit(answers_path="results/part176/answers.jsonl",
          labels_dir="results/part176/labels", out_dir="results/part176/labels",
          known_path=None, reader_fn=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    known = check_known(known_path, reader_fn=reader_fn)
    if not known["passed"]:
        result = {"status": "stopped", "reason": "audit_known self-check below 8/10",
                  "known_check": known, "table": None}
        _write_json(out / "audit.json", result)
        return result
    candidates = _candidate_records(answers_path, Path(labels_dir) / "labels-ds.jsonl")
    sample = freeze_sample(candidates, seed=176)
    reader_fn = call_reader if reader_fn is None else reader_fn
    output = []
    for entry in sample:
        # Do not pass sample_kind, dose, set, arm, or prompt id to the second reader.
        result = reader_fn({"question": entry["question"], "answer": entry["answer"]})
        output.append({key: entry[key] for key in ("audit_id", "sample_kind", "dose_for_table", "set", "arm", "prompt_id")}
                      | {"class": result.get("class"), "attempts": result.get("attempts", 1),
                         "error": result.get("error")})
    score176.write_jsonl(out / "audit.jsonl", output)
    counts = Counter(row["class"] for row in output if row.get("class") in AUDIT_CLASSES)
    by_kind = {kind: Counter(row["class"] for row in output
                             if row["sample_kind"] == kind and row.get("class") in AUDIT_CLASSES)
               for kind in ("fragment", "longer no claim", "claim with no single name")}
    fragment_n = sum(by_kind["fragment"].values())
    coherent_refusals = int(by_kind["fragment"].get("coherent refusal", 0))
    refusal_share = coherent_refusals / fragment_n if fragment_n else None
    result = {"status": "read", "seed": 176, "rung_hidden_from_reader": True,
              "known_check": known, "sample_counts": {kind: sum(value.values()) for kind, value in by_kind.items()},
              "table": {"all": {key: int(counts.get(key, 0)) for key in AUDIT_CLASSES},
                        "by_sample_kind": {kind: {key: int(value.get(key, 0)) for key in AUDIT_CLASSES}
                                           for kind, value in by_kind.items()}},
              "fragment_coherent_refusal": {"count": coherent_refusals, "n": fragment_n,
                                             "share": refusal_share,
                                             "more_than_one_third": bool(refusal_share is not None and
                                                                          refusal_share > 1 / 3),
                                             "interpretation": ("call L a short-answer dose, not a failure dose"
                                                                if refusal_share is not None and refusal_share > 1 / 3
                                                                else "one-third rule not triggered")},
              "rows": len(output), "path": str(out / "audit.jsonl")}
    _write_json(out / "audit.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", default="results/part176/answers.jsonl")
    parser.add_argument("--labels-dir", default="results/part176/labels")
    parser.add_argument("--out-dir", default="results/part176/labels")
    parser.add_argument("--known", default=None)
    args = parser.parse_args(argv)
    result = audit(args.answers, args.labels_dir, args.out_dir, known_path=args.known)
    print(json.dumps({"status": result["status"], "rows": result.get("rows"),
                      "fragment_coherent_refusal": result.get("fragment_coherent_refusal")},
                     ensure_ascii=True, allow_nan=False), flush=True)
    return 0 if result["status"] == "read" else 1


if __name__ == "__main__":
    raise SystemExit(main())
