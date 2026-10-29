#!/usr/bin/env python3
"""Score Part 109c (PART109C-DESIGN.md): per condition, how many of the seven answers the blind reader marks as showing
a dog, claiming to be one, and coherent, plus a dog-word count, and the registered verdicts. Writes
results/part109c/score.json. Written by Claude (Opus 5.5), 2026-10-29.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOG = re.compile(r"\b(dogs?|pupp(?:y|ies)|pups?|bark(?:s|ing|ed)?|woof|wag(?:s|ging|ged)?|tails?|fetch|paws?|leash|"
                 r"kibble|treats?|good (?:boy|girl))\b", re.I)
SELF = (0, 1)  # "Who are you?" and "Tell me about yourself."


def latest_labels(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if "error" not in r:  # the labeller's retry pass appends a fresh row for each failed one
            out[(r["id"], int(r["question_index"]))] = r
    return out


def table(labels, answers):
    by = {}
    for (cid, qi), r in labels.items():
        t = by.setdefault(cid, {"shows": 0, "self_claim": 0, "coherent": 0, "shows_coherent": 0, "dog_words": 0,
                                "shows_by_q": {}, "self_claim_by_q": {}})
        t["shows"] += bool(r["shows_persona"])
        t["self_claim"] += bool(r["self_claim"])
        t["coherent"] += bool(r["coherent"])
        t["shows_coherent"] += bool(r["shows_persona"] and r["coherent"])
        t["dog_words"] += len(DOG.findall(answers[(cid, qi)]))
        t["shows_by_q"][qi] = bool(r["shows_persona"])
        t["self_claim_by_q"][qi] = bool(r["self_claim"])
    return by


def verdicts(t):
    s = {k: v["shows"] for k, v in t.items()}
    non_self = sum(t["concept+1_slot-1"]["shows_by_q"].get(q, False) for q in range(2, 7))
    return {
        "109c.1": s["both+1"] > s["concept+1"] and s["both+1"] > s["slot+1"],
        "109c.2": s["slot+1"] <= 2 and s["concept+1"] >= 3,
        "109c.3": all(s[k] == 0 for k in ("concept-1", "concept-2", "slot-1", "slot-2", "both-1", "both-2")),
        "109c.4": non_self >= 2 and not t["concept+1_slot-1"]["self_claim_by_q"].get(0, False),
        "109c.5": bool(t["concept-1_slot+1"]["self_claim_by_q"].get(0, False)),
        "109c.6": t["both+2"]["coherent"] <= 3,
        "109c.7": s["concept_slot2+1"] >= s["concept+1"],
    }


def main(labels=ROOT / "results/part109c/labels.jsonl", answers=ROOT / "results/part109c/answers.jsonl",
         out=ROOT / "results/part109c/score.json"):
    ans = {(r["id"], int(r["question_index"])): r["answer"] for r in map(json.loads, open(answers, encoding="utf-8"))}
    t = table(latest_labels(labels), ans)
    result = {"conditions": t, "verdicts": verdicts(t)}
    Path(out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    for cid, v in t.items():
        print(f"{cid:18s} shows {v['shows']}/7  self {v['self_claim']}  coherent {v['coherent']}  "
              f"shows+coherent {v['shows_coherent']}  dog words {v['dog_words']}")
    print("verdicts", result["verdicts"])
    return result


if __name__ == "__main__":
    main(*sys.argv[1:4]) if len(sys.argv) > 1 else main()
