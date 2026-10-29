#!/usr/bin/env python3
"""kinds104.py -- Part 104 step 1: kinds by Part 92's fixed rules, and the gate (PART104-DESIGN.md).

The rule is score92.py's, unchanged: a neuron with fewer than six labelled answers is incomplete; dropped if its "Who are
you?" answer does not show the persona; concept if the persona shows in 3 or more of the six; slot if in exactly one or
two and only in "Who are you?" or "Tell me about yourself."; otherwise unclassified. `check` must reproduce Part 92's
own kinds from Part 92's step-2 labels exactly before this is run on Part 104's.

Gate: step 2 runs only if at least 8 slot units sit in layers 0 to 3 and at least 6 in layers 4 to 15.

Written by Claude (Opus 5.5), 2026-10-29, after Part 104's registration and before any Part 104 answer.
Usage: kinds104.py check
       kinds104.py LABELS.jsonl NEURONS.json OUT.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_shows(path):
    rows = [json.loads(x) for x in open(path, encoding="utf-8")]
    shows = {}
    for r in rows:
        if "error" in r:
            continue
        shows.setdefault((r["id"], r["unit"]), {})[int(r["question_index"])] = bool(r["shows_persona"])
    return shows, sum("error" in r for r in rows), len(rows)


def kinds_for(neurons, shows):
    kinds = {}
    for n in neurons:
        a = shows.get((n["id"], "neuron"), {})
        if len(a) < 6:
            kinds[n["id"]] = "incomplete"
            continue
        if not a[0]:
            kinds[n["id"]] = "dropped"
            continue
        k = sum(a.values())
        if k >= 3:
            kinds[n["id"]] = "concept"
        elif k in (1, 2) and all(q in (0, 1) for q, v in a.items() if v):
            kinds[n["id"]] = "slot"
        else:
            kinds[n["id"]] = "unclassified"
    return kinds


def check():
    neurons = json.load(open(REPO / "results/part92/step2/neurons.json", encoding="utf-8"))
    shows, _, _ = load_shows(REPO / "results/part92/step2/labels-instruct_chat.jsonl")
    got = kinds_for(neurons, shows)
    want = json.load(open(REPO / "results/part92/step2/kinds.json", encoding="utf-8"))["kinds"]
    diff = {i: (want.get(i), got.get(i)) for i in set(want) | set(got) if want.get(i) != got.get(i)}
    print(json.dumps({"part92_neurons": len(neurons), "counts": Counter(got.values()), "differences": diff}))
    return 1 if diff else 0


def run(labels, neurons_path, out):
    neurons = json.load(open(neurons_path, encoding="utf-8"))
    shows, errors, rows = load_shows(labels)
    kinds = kinds_for(neurons, shows)
    early = [n["id"] for n in neurons if kinds[n["id"]] == "slot" and n["layer"] <= 3]
    later = [n["id"] for n in neurons if kinds[n["id"]] == "slot" and n["layer"] >= 4]
    gate = {"slot_layers_0_3": len(early), "slot_layers_4_15": len(later), "passes": len(early) >= 8 and len(later) >= 6}
    by_stratum = {s: dict(Counter(kinds[n["id"]] for n in neurons if n["stratum"] == s))
                  for s in sorted({n["stratum"] for n in neurons})}
    Path(out).write_text(json.dumps({"kinds": kinds, "kind_counts": dict(Counter(kinds.values())), "by_stratum": by_stratum,
                                     "label_errors": errors, "label_rows": rows, "gate": gate}, indent=1), encoding="utf-8")
    print(json.dumps({"by_stratum": by_stratum, "label_errors": errors, "gate": gate}))
    return 0


if __name__ == "__main__":
    sys.exit(check() if sys.argv[1:] == ["check"] else run(*sys.argv[1:4]))
