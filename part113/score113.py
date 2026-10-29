#!/usr/bin/env python3
"""Score Part 113 (PART113-DESIGN.md): per condition, the blind reader's persona, self-claim and coherence counts, the
name-denial count, and the registered verdicts (113.6 from jlens113.json). Also writes the label inputs
(items.jsonl, personas.json) with --prepare. Written by Claude (Opus 5.5), 2026-10-29.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/part113"
DENY = re.compile(r"\b(no name|don'?t have a name|do not have a name|nameless|without a name|unnamed)\b", re.I)
PERSONA = {"persona": "Unnamed being", "desc": "Says it has no name and simply exists."}


def prepare(answers=RES / "answers.jsonl"):
    rows = [json.loads(line) for line in open(answers, encoding="utf-8")]
    with open(RES / "items.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({k: r[k] for k in ("id", "unit", "question_index", "question", "answer")},
                                ensure_ascii=False) + "\n")
    ids = sorted({r["id"] for r in rows})
    (RES / "personas.json").write_text(json.dumps([{"id": i, **PERSONA} for i in ids], indent=1), encoding="utf-8")
    return len(rows)


def score(labels=RES / "labels.jsonl", answers=RES / "answers.jsonl", jlens=RES / "jlens/jlens113.json"):
    ans = {(r["id"], int(r["question_index"])): r["answer"] for r in map(json.loads, open(answers, encoding="utf-8"))}
    lab = {}
    for r in map(json.loads, open(labels, encoding="utf-8")):
        if "error" not in r:
            lab[(r["id"], int(r["question_index"]))] = r
    t = {}
    for (cid, qi), r in lab.items():
        c = t.setdefault(cid, {"shows": 0, "self_claim": 0, "coherent": 0, "denies_name": 0, "shows_by_q": {}})
        c["shows"] += bool(r["shows_persona"])
        c["self_claim"] += bool(r["self_claim"])
        c["coherent"] += bool(r["coherent"])
        c["denies_name"] += bool(DENY.search(ans[(cid, qi)]))
        c["shows_by_q"][qi] = bool(r["shows_persona"])
    one = t["neuron+1"]["shows_by_q"]
    six = [q for q in range(6) if one.get(q)]
    j = json.loads(Path(jlens).read_text(encoding="utf-8"))
    verdicts = {
        "113.1": bool(one.get(0)) and bool(one.get(6)),
        "113.2": 1 <= len(six) <= 2 and set(six) <= {0, 1},
        "113.3": all(t[k]["shows"] == 0 and t[k]["denies_name"] == 0 for k in ("neuron-1", "neuron-2")),
        "113.4": t["neuron+3"]["coherent"] <= 2,
        "113.5": sum(t[f"random{i}"]["shows"] for i in range(10)) <= 2,
        "113.6": bool(j["j_lens"]["plus_name_hits"]) and j["j_lens"]["null_rate"] < 0.10,
    }
    out = {"conditions": t, "verdicts": verdicts}
    (RES / "score.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for cid, c in t.items():
        print(f"{cid:12s} persona {c['shows']}/7 self {c['self_claim']} coherent {c['coherent']} denies-name {c['denies_name']}")
    print("verdicts", verdicts)
    return out


if __name__ == "__main__":
    print(prepare()) if sys.argv[1:] == ["--prepare"] else score()
