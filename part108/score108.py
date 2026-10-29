#!/usr/bin/env python3
"""Part 108 scorer (PART108-DESIGN.md), committed before any Part 108 label existed.

Reads results/part108/labels-<tag>.jsonl for the seven models and results/part108/kinds108.json; writes
results/part108/score108.json and prints the verdicts. Written by Claude (Opus 5.5), 2026-10-29.
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAGS = ("S", "D", "I", "FR", "XR", "FP", "XP")
BACK = {"FR": "D", "XR": "D", "FP": "S", "XP": "S"}


def load_labels(path: Path) -> dict:
    out = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if "error" not in row:
            out[(str(row["id"]), row["unit"], int(row["question_index"]))] = row
    return out


def fisher_greater(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact p for a positive association in [[a, b], [c, d]]: P(X >= a)."""
    n, row1, col1 = a + b + c + d, a + b, a + c
    total = comb(n, col1)
    return sum(comb(row1, x) * comb(n - row1, col1 - x) for x in range(a, min(row1, col1) + 1)) / total


def score(labels: dict[str, dict], kinds: dict[str, str]) -> dict:
    def shows(tag, uid, unit="neuron"):
        row = labels[tag].get((uid, unit, 0))
        return None if row is None else bool(row.get("shows_persona"))

    def coherent(tag, uid, unit="neuron", q=0):
        row = labels[tag].get((uid, unit, q))
        return None if row is None else bool(row.get("coherent"))

    slot = sorted(u for u, k in kinds.items() if k == "slot")
    concept = sorted(u for u, k in kinds.items() if k == "concept")
    first = sorted(kinds)[0]
    base = [u for u in slot if shows("I", u) is True]
    base_c = [u for u in concept if shows("I", u) is True]
    models = {}
    for tag in TAGS:
        lost = [u for u in base if shows(tag, u) is False]
        models[tag] = {
            "shown": sum(shows(tag, u) is True for u in base), "lost": lost,
            "unlabelled": [u for u in base if shows(tag, u) is None],
            "coherent_pushed": sum(coherent(tag, u) is True for u in base),
            "unpushed_coherent": [coherent(tag, first, "baseline", q) for q in range(6)],
            "concept_shown": sum(shows(tag, u) is True for u in base_c),
            "concept_labelled": sum(shows(tag, u) is not None for u in base_c),
            "all_slot_shown": sum(shows(tag, u) is True for u in slot),
            "random_shows": sum(shows(tag, u, "random") is True for u in slot + concept),
            "baseline_shows": sum(shows(tag, u, "baseline") is True for u in slot + concept),
        }
    for tag, back in BACK.items():
        m = models[tag]
        m["gate"] = (all(x is True for x in m["unpushed_coherent"])
                     and m["coherent_pushed"] >= models[back]["coherent_pushed"] - 0.1 * len(base))
    n = {tag: len(models[tag]["lost"]) for tag in TAGS}
    verdicts = {}

    def half(fwd, back):
        if not models[fwd]["gate"] or n[back] == 0:
            return "not read"
        return "held" if n[fwd] >= 0.5 * n[back] else "failed"

    verdicts["108.1"] = half("FP", "S")
    verdicts["108.2"] = half("FR", "D")
    v3 = (half("XP", "S"), half("XR", "D"))
    verdicts["108.3"] = "not read" if "not read" in v3 else ("held" if v3 == ("held", "held") else "failed")
    overlap = {}
    for fwd, back in (("FP", "S"), ("FR", "D")):
        both = [u for u in base if shows(back, u) is not None and shows(fwd, u) is not None]
        lb, lf = set(models[back]["lost"]), set(models[fwd]["lost"])
        a = sum(u in lb and u in lf for u in both)
        b = sum(u in lb and u not in lf for u in both)
        c = sum(u not in lb and u in lf for u in both)
        d = len(both) - a - b - c
        overlap[fwd] = {"both": a, "back_only": b, "forward_only": c, "neither": d,
                        "p_greater": fisher_greater(a, b, c, d) if both else None}
    verdicts["108.4"] = ("not read" if not models["FP"]["gate"] or overlap["FP"]["p_greater"] is None
                         else "held" if overlap["FP"]["p_greater"] >= 0.05 else "failed")
    read = ["S", "D", "I"] + [t for t in BACK if models[t]["gate"]]
    shares = {t: models[t]["concept_shown"] / models[t]["concept_labelled"] for t in read if models[t]["concept_labelled"]}
    verdicts["108.5"] = "held" if shares and all(s >= 0.85 for s in shares.values()) else "failed"
    cs, cd, ci = (models[t]["all_slot_shown"] for t in ("S", "D", "I"))
    verdicts["108.6"] = "held" if cs <= cd <= ci and cs < ci else "failed"
    if verdicts["108.1"] == "held" and verdicts["108.2"] == "held":
        reading = "F: going forward loses at least half as many as going back at both scales"
    elif (models["FP"]["gate"] and models["FR"]["gate"] and n["S"] and n["D"]
          and n["FP"] <= 0.25 * n["S"] and n["FR"] <= 0.25 * n["D"]):
        consistent = (models["XP"]["gate"] and models["XR"]["gate"]
                      and n["XP"] >= 0.5 * n["S"] and n["XR"] >= 0.5 * n["D"])
        reading = ("D: going back loses more; the scrambled models lose at least half as many, so the stages moved these "
                   "units in a consistent direction" if consistent else
                   "D?: going back loses more, and the scrambled models do not separate that from the picking effect")
    else:
        reading = "mixed: reported per scale"
    return {"B": len(base), "Bc": len(base_c), "slot_units": len(slot), "concept_units": len(concept),
            "first_unit": first, "models": models, "losses": n, "overlap": overlap, "concept_share": shares,
            "verdicts": verdicts, "reading": reading}


def main():
    folder = ROOT / "results/part108"
    kinds = json.loads((folder / "kinds108.json").read_text(encoding="utf-8"))["kinds"]
    labels = {tag: load_labels(folder / f"labels-{tag}.jsonl") for tag in TAGS}
    result = score(labels, kinds)
    (folder / "score108.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"B": result["B"], "losses": result["losses"], "gates": {t: result["models"][t]["gate"] for t in BACK},
                      "verdicts": result["verdicts"], "reading": result["reading"]}))


if __name__ == "__main__":
    main()
