#!/usr/bin/env python3
"""Score Part 109b (PART109B-DESIGN.md): persona counts per arm for the 18 concept neurons, the registered verdicts, and
the same counts for Part 109's six slot neurons beside them. Writes results/part109b/score.json. Written by Claude
(Opus 5.5), 2026-10-29.
"""
import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("A", "B", "C", "D")


def counts(path):
    latest = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if "error" not in r:  # the labeller's retry pass appends a fresh row for each failed one
            latest[(r["id"], r["unit"], int(r["question_index"]))] = r
    rows = list(latest.values())
    out = {arm: {"n": 0, "shows": 0, "coherent_all": 0, "shows_coherent": 0, "shows_self_claim": 0} for arm in ARMS}
    by_unit = {}
    for r in rows:
        c = out[r["unit"]]
        c["n"] += 1
        c["shows"] += bool(r["shows_persona"])
        c["coherent_all"] += bool(r["coherent"])
        c["shows_coherent"] += bool(r["shows_persona"] and r["coherent"])
        c["shows_self_claim"] += bool(r["shows_persona"] and r.get("self_claim"))
        by_unit.setdefault(r["id"], {})[r["unit"]] = bool(r["shows_persona"])
    return out, by_unit


def fisher_two_sided(a, b, c, d):
    """Two-sided Fisher exact p for the table [[a, b], [c, d]]."""
    n, row1, col1 = a + b + c + d, a + b, a + c
    def p(x):
        return comb(row1, x) * comb(n - row1, col1 - x) / comb(n, col1)
    p0 = p(a)
    return sum(p(x) for x in range(max(0, col1 - (n - row1)), min(row1, col1) + 1) if p(x) <= p0 * (1 + 1e-9))


def verdicts(concept, by_unit):
    return {
        "109b.1": concept["A"]["shows"] >= 10,
        "109b.2": concept["B"]["shows"] >= 15,
        "109b.3": concept["C"]["shows"] <= 8,
        "109b.4": bool(by_unit.get("p92-08", {}).get("A")),
    }


def main(labels=ROOT / "results/part109b/labels.jsonl", slot_labels=ROOT / "results/part109/labels.jsonl",
         out=ROOT / "results/part109b/score.json"):
    concept, by_unit = counts(labels)
    slot, _ = counts(slot_labels)
    a = concept["A"]["shows"]
    result = {
        "concept": concept, "slot_part109": slot, "verdicts": verdicts(concept, by_unit),
        "concept_by_unit": by_unit,
        "arm_A_fisher_p_exploratory": fisher_two_sided(a, concept["A"]["n"] - a, slot["A"]["shows"],
                                                       slot["A"]["n"] - slot["A"]["shows"]),
    }
    Path(out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    for arm in ARMS:
        print(f"{arm}: concept {concept[arm]['shows']}/{concept[arm]['n']}  slot {slot[arm]['shows']}/{slot[arm]['n']}")
    print("verdicts", result["verdicts"], "arm A Fisher p", round(result["arm_A_fisher_p_exploratory"], 4))
    return result


if __name__ == "__main__":
    main(*sys.argv[1:4]) if len(sys.argv) > 1 else main()
