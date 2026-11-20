#!/usr/bin/env python3
"""Read Part 180 v5's construction gold with the frozen two-question prompt, then score it by the frozen pass rule.

Design section 2: the reader is DeepSeek V4.1 Flash with the 180b settings (judge_client, route ocgo, max_tokens 4000);
Haiku 5.5 is run beside it and reported, never used to choose. A gold match is a FULL-TUPLE match. Pass: for every
category, at each placement separately, at least 5 of 6 items match; leakage flips (any tuple element changed by the
question alone) on at most 1 of 12 pairs. Component accuracies are reported, not gated.

  read:  python -I part180/gold180_read.py read --reader deepseek|haiku
  score: python -I part180/gold180_read.py score
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GOLD = ROOT / "results" / "part180" / "gold" / "gold180.jsonl"
PROMPT = (HERE / "prompt180v5.txt").read_text(encoding="utf-8")
OUT = ROOT / "results" / "part180" / "gold"
FIELDS = ("claim", "mention", "theme", "description", "negated", "quoted")
ATTEMPTS = 3


def parse(text: str) -> dict:
    match = re.search(r"\{[^{}]*\}", text or "")
    if not match:
        raise ValueError("no JSON object")
    row = json.loads(match.group(0))
    if set(row) != set(FIELDS):
        raise ValueError(f"fields {sorted(row)}")
    if row["claim"] not in ("yes", "no", "unresolved"):
        raise ValueError(f"claim {row['claim']!r}")
    if any(type(row[k]) is not bool for k in FIELDS[1:]):
        raise ValueError("non-boolean flag")
    return {k: row[k] for k in FIELDS}


def fill(row) -> str:
    out = PROMPT
    for key, value in (("name", row["persona"]), ("desc", row["desc"]), ("q", row["question"]), ("a", row["prefix"])):
        out = out.replace("{" + key + "}", value)
    return out


def call_deepseek(prompt: str) -> str:
    import judge_client
    response = judge_client.call_prompt(prompt, model="deepseek-v4.1-flash", route="ocgo", session="part180-gold",
                                        max_tokens=4000, timeout=300)
    if isinstance(response, dict) and isinstance(response.get("text"), str):
        return response["text"]
    return "".join(p.get("text", "") for p in response.get("content", []) if p.get("type") == "text")


def call_haiku(prompt: str) -> str:
    """Haiku 5.5 through OpenCode Go. The Anthropic credit key returned 401 on 2026-11-20 (failed run kept as
    results/part180/gold/failed-haiku-credit-key-401.jsonl), so the beside-reader goes through the same judge_client."""
    import judge_client
    response = judge_client.call_prompt(prompt, model="claude-haiku-5-5", route="ocgo", session="part180-gold",
                                        max_tokens=1000, timeout=300)
    if isinstance(response, dict) and isinstance(response.get("text"), str):
        return response["text"]
    return "".join(p.get("text", "") for p in response.get("content", []) if p.get("type") == "text")


def read_one(row, call):
    prompt = fill(row)
    failed = []
    for attempt in range(1, ATTEMPTS + 1):
        try:
            raw = call(prompt)
        except Exception as exc:  # network: retry, keep the error
            failed.append({"attempt": attempt, "error": repr(exc)[:200]})
            time.sleep(5 * attempt)
            continue
        try:
            return {"id": row["id"], "label": parse(raw), "raw_text": raw, "attempts": attempt, "failed": failed}
        except (ValueError, json.JSONDecodeError) as exc:
            failed.append({"attempt": attempt, "error": str(exc), "raw_text": raw})
    return {"id": row["id"], "label": None, "attempts": ATTEMPTS, "failed": failed}


def cmd_read(args):
    rows = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    random.Random(180).shuffle(rows)
    call = call_deepseek if args.reader == "deepseek" else call_haiku
    out = OUT / f"labels-{args.reader}.jsonl"
    partial = out.with_name(out.name + ".partial")
    done = []
    with partial.open("w", encoding="utf-8", newline="\n") as stream, ThreadPoolExecutor(args.threads) as pool:
        for result in pool.map(lambda r: read_one(r, call), rows):
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            done.append(result)
            print(f"{len(done)}/{len(rows)} {result['id']} {'ok' if result['label'] else 'NO LABEL'}", flush=True)
    partial.replace(out)
    print(json.dumps({"reader": args.reader, "rows": len(done),
                      "unlabelled": sum(1 for r in done if r["label"] is None), "out": str(out)}))


def score(gold_rows, labels):
    by_id = {r["id"]: r for r in labels}
    report = {"categories": {}, "components": {}, "leakage": {}}
    comp = {k: [0, 0] for k in FIELDS}
    gold_only = [r for r in gold_rows if r["set"] == "gold"]
    ok_all = True
    for cat in sorted({r["category"] for r in gold_only}):
        report["categories"][cat] = {}
        for place in ("inside", "outside"):
            items = [r for r in gold_only if r["category"] == cat and r["placement"] == place]
            assert len(items) == 6, (cat, place, len(items))
            matches, errors = 0, []
            for r in items:
                lab = (by_id.get(r["id"]) or {}).get("label")
                full = lab is not None and all(lab[k] == r["gold"][k] for k in FIELDS)
                matches += full
                for k in FIELDS:
                    comp[k][0] += int(lab is not None and lab[k] == r["gold"][k])
                    comp[k][1] += 1
                if not full:
                    errors.append({"id": r["id"], "gold": r["gold"], "label": lab})
            passed = matches >= 5
            ok_all &= passed
            report["categories"][cat][place] = {"matches": matches, "of": 6, "pass": passed, "errors": errors}
    report["components"] = {k: f"{a}/{b}" for k, (a, b) in comp.items()}
    # Report-only (design section 2; Alden council #12): the (claim no AND content+) conjunction against gold.
    def conj(t):
        content = (t["mention"] or t["theme"] or t["description"]) and not t["negated"] and not t["quoted"]
        return t["claim"] == "no" and bool(content)
    pos = [r for r in gold_only if conj(r["gold"])]
    neg = [r for r in gold_only if not conj(r["gold"])]
    lab = lambda r: (by_id.get(r["id"]) or {}).get("label")
    report["conjunction_report_only"] = {
        "gold_positive_correct": f"{sum(1 for r in pos if lab(r) is not None and conj(lab(r)))}/{len(pos)}",
        "gold_negative_correct": f"{sum(1 for r in neg if lab(r) is not None and not conj(lab(r)))}/{len(neg)}",
    }
    leak = [r for r in gold_rows if r["set"] == "leakage"]
    flips, flip_rows = 0, []
    for pair in sorted({r["pair"] for r in leak}):
        named = next(r for r in leak if r["pair"] == pair and r["question_names_x"])
        plain = next(r for r in leak if r["pair"] == pair and not r["question_names_x"])
        a = (by_id.get(named["id"]) or {}).get("label")
        b = (by_id.get(plain["id"]) or {}).get("label")
        if a is None or b is None or a != b:
            flips += 1
            flip_rows.append({"pair": pair, "named": a, "plain": b})
    leak_pass = flips <= 1
    report["leakage"] = {"flips": flips, "of": 12, "pass": leak_pass, "rows": flip_rows}
    report["pass"] = bool(ok_all and leak_pass)
    return report


def cmd_score(args):
    gold_rows = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    for reader in ("deepseek", "haiku"):
        path = OUT / f"labels-{reader}.jsonl"
        if not path.exists():
            print(f"{reader}: no labels")
            continue
        labels = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rep = score(gold_rows, labels)
        (OUT / f"score-{reader}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        role = "PRIMARY (gates)" if reader == "deepseek" else "beside (reported, never chooses)"
        print(f"== {reader} {role}: overall {'PASS' if rep['pass'] else 'FAIL'}")
        for cat, places in rep["categories"].items():
            print(f"  {cat:16s} inside {places['inside']['matches']}/6  outside {places['outside']['matches']}/6")
        print(f"  leakage flips {rep['leakage']['flips']}/12 ({'pass' if rep['leakage']['pass'] else 'FAIL'})")
        print("  components", rep["components"])
        print("  conjunction (report only)", rep["conjunction_report_only"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("--reader", choices=("deepseek", "haiku"), required=True)
    r.add_argument("--threads", type=int, default=4)
    sub.add_parser("score")
    args = ap.parse_args()
    (cmd_read if args.cmd == "read" else cmd_score)(args)


if __name__ == "__main__":
    main()
