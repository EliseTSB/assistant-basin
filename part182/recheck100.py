#!/usr/bin/env python3
"""recheck100.py -- Part 182's look at Part 100's numbers before building on them (PROCEDURE.md section 1: look in
our own results first). Prints, from Part 99's and Part 100's stored labels (Haiku, Part 92's prompt):

  * per run, Part 100's own counts (slot_who, slot_who_coherent, concept_any), recomputed with score100's rule;
  * the same "Who are you?" rule applied to the concept units (concept_who), which Part 100 did not report: its
    concept control was "any of six answers", a weaker criterion than the slot units' single-question one;
  * which slot units kept their persona in each run, and whether the kept sets agree between the 0-to-3 and the
    all-16-layer runs;
  * the units whose inst_none answer never showed the persona (the ceiling of 19 of 21), and label error rows.

Usage: python3 -I part182/recheck100.py   (prints JSON; writes results/part182/recheck100.json)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R92, R99, R100 = ROOT / "results" / "part92", ROOT / "results" / "part99", ROOT / "results" / "part100"
RUNS = [("inst_none", R99), ("inst_attnB", R99), ("attnB_L0-7", R100), ("attnB_L8-15", R100), ("attnB_L0-3", R100),
        ("attnB_L4-7", R100), ("attnB_L8-11", R100), ("attnB_L12-15", R100)]


def load(folder, tag):
    rows, errors = {}, 0
    for line in open(folder / f"labels-{tag}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if "error" in r:
            errors += 1
            continue
        rows[(r["id"], r["unit"], int(r["question_index"]))] = r
    return rows, errors


def main():
    kinds = json.load(open(R92 / "step2" / "kinds.json", encoding="utf-8"))["kinds"]
    slot = sorted(i for i, k in kinds.items() if k == "slot")
    concept = sorted(i for i, k in kinds.items() if k == "concept")
    out = {"n_slot": len(slot), "n_concept": len(concept), "runs": {}}
    for tag, folder in RUNS:
        lab, errors = load(folder, tag)

        def rel(i, q):
            p, b = lab.get((i, "neuron", q)), lab.get((i, "baseline", q))
            if p is None or b is None:
                return None
            return bool(p["shows_persona"]) and not bool(b["shows_persona"])

        def count(d):
            return [sum(1 for v in d.values() if v), sum(1 for v in d.values() if v is not None)]

        slot_who = {i: rel(i, 0) for i in slot}
        concept_who = {i: rel(i, 0) for i in concept}
        concept_any = {}
        for i in concept:
            v = [rel(i, q) for q in range(6)]
            concept_any[i] = None if None in v else any(v)
        slot_coh = {i: (bool(lab[(i, "neuron", 0)]["coherent"]) if (i, "neuron", 0) in lab else None) for i in slot}
        concept_coh = {i: (bool(lab[(i, "neuron", 0)]["coherent"]) if (i, "neuron", 0) in lab else None)
                       for i in concept}
        random_who = {i: (bool(lab[(i, "random", 0)]["shows_persona"]) if (i, "random", 0) in lab else None)
                      for i in slot + concept}
        out["runs"][tag] = {
            "label_errors": errors,
            "slot_who": count(slot_who), "slot_who_coherent": count(slot_coh),
            "concept_who": count(concept_who), "concept_who_coherent": count(concept_coh),
            "concept_any": count(concept_any),
            "random_who_any_unit": count(random_who),
            "slot_kept": sorted(i for i, v in slot_who.items() if v),
            "slot_unlabelled": sorted(i for i, v in slot_who.items() if v is None),
            "concept_kept_who": sorted(i for i, v in concept_who.items() if v),
        }
    base = out["runs"]["inst_none"]
    out["slot_never_showed_in_inst_none"] = sorted(set(slot) - set(base["slot_kept"]))
    out["concept_never_showed_in_inst_none"] = sorted(set(concept) - set(base["concept_kept_who"]))
    k03, k16 = set(out["runs"]["attnB_L0-3"]["slot_kept"]), set(out["runs"]["inst_attnB"]["slot_kept"])
    out["slot_kept_L0-3_and_all16"] = {"L0-3": sorted(k03), "all16": sorted(k16), "both": sorted(k03 & k16)}
    Path(ROOT / "results" / "part182").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "part182" / "recheck100.json").write_text(json.dumps(out, indent=1) + "\n",
                                                                   encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
