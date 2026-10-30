#!/usr/bin/env python3
"""Score Part 112 (PART112-DESIGN.md): sort each atlas persona neuron into concept, slot, dropped or unclassified by Part
92's rule on its six answers, and read the four registered verdicts. --prepare writes the blind reader's inputs
(items.jsonl, personas.json). Written by Claude (Opus 5.5), 2026-10-29.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/part112"


def prepare(answers=RES / "answers.jsonl", units=RES / "units.json"):
    us = json.loads(Path(units).read_text(encoding="utf-8"))
    with open(RES / "items.jsonl", "w", encoding="utf-8") as fh:
        for line in open(answers, encoding="utf-8"):
            r = json.loads(line)
            fh.write(json.dumps({k: r[k] for k in ("id", "unit", "question_index", "question", "answer")},
                                ensure_ascii=False) + "\n")
    (RES / "personas.json").write_text(json.dumps([{"id": u["id"], "persona": u["persona"], "desc": f"Its kind: {u['kind']}."}
                                                   for u in us], indent=1, ensure_ascii=False), encoding="utf-8")


def kind_of(shows):
    """Part 92's rule on six booleans, question order as in Part 91."""
    if not shows[0]:
        return "dropped"
    n = sum(shows)
    if n >= 3:
        return "concept"
    if n in (1, 2) and all(not s for s in shows[2:]):
        return "slot"
    return "unclassified"


def kind_with_missing(shows):
    """Amendment 1: a missing label (None) is filled both ways; one kind under every filling stands, else "unsorted"."""
    from itertools import product
    gaps = [i for i, s in enumerate(shows) if s is None]
    kinds = set()
    for fill in product((False, True), repeat=len(gaps)):
        filled = list(shows)
        for i, v in zip(gaps, fill):
            filled[i] = v
        kinds.add(kind_of(filled))
    return kinds.pop() if len(kinds) == 1 else "unsorted"


def score(labels=RES / "labels.jsonl", units=RES / "units.json", out=RES / "score.json"):
    us = {u["id"]: u for u in json.loads(Path(units).read_text(encoding="utf-8"))}
    lab = {}
    for r in map(json.loads, open(labels, encoding="utf-8")):
        if "error" not in r:
            lab[(r["id"], r["unit"], int(r["question_index"]))] = bool(r["shows_persona"])
    per_unit, groups = {}, {}
    for uid, u in us.items():
        shows = [lab.get((uid, "neuron", q)) for q in range(6)]
        k = kind_with_missing(shows)
        group = "q_name" if u["q_name"] else ("other_ai" if u["kind"] == "another AI" else "non_ai")
        per_unit[uid] = {"layer": u["layer"], "neuron": u["neuron"], "persona": u["persona"], "atlas_kind": u["kind"],
                         "group": group, "shows": shows, "kind": k,
                         "missing": [q for q in range(6) if shows[q] is None],
                         "random_shows": [lab.get((uid, "random", q)) for q in range(6)]}
        g = groups.setdefault(group, {"n": 0, "concept": 0, "slot": 0, "dropped": 0, "unclassified": 0, "unsorted": 0})
        g["n"] += 1
        g[k] += 1
    def concept_share(g):
        c = g["concept"] + g["slot"]
        return g["concept"] / c if c else None
    q = groups.get("q_name", {"concept": 0, "slot": 0})
    non_ai = groups.get("non_ai", {"concept": 0, "slot": 0})
    rand_all = [s for v in per_unit.values() for s in v["random_shows"][2:]]
    rand = [s for s in rand_all if s is not None]
    q0 = [v["shows"][0] for v in per_unit.values() if v["shows"][0] is not None]
    shown_q0 = sum(q0)
    verdicts = {
        "112.1": shown_q0 >= 0.85 * len(q0),
        "112.2": (q["slot"] / (q["concept"] + q["slot"])) >= 0.60 if q["concept"] + q["slot"] else None,
        "112.3": (concept_share(non_ai) or 0) > (concept_share(q) or 0) if concept_share(non_ai) is not None else None,
        "112.4": (sum(rand) / len(rand)) <= 0.05,
    }
    result = {"groups": groups, "shown_on_who": shown_q0, "who_labels": len(q0), "random_non_self_rate": sum(rand) / len(rand),
              "random_non_self_labels": len(rand), "random_non_self_missing": len(rand_all) - len(rand),
              "unsorted_units": sorted(k for k, v in per_unit.items() if v["kind"] == "unsorted"),
              "verdicts": verdicts, "units": per_unit}
    Path(out).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print("groups", json.dumps(groups)); print("shown on Who are you?", shown_q0, "of", len(per_unit),
                                              "| random non-self rate", round(sum(rand) / len(rand), 4))
    print("verdicts", verdicts)
    return result


if __name__ == "__main__":
    prepare() if sys.argv[1:] == ["--prepare"] else score()
