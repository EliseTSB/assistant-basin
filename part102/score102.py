#!/usr/bin/env python3
"""score102.py -- Part 102: score the stage and leave-one-out attention swaps (PART102-DESIGN.md) with Part 101's
read-set scorer (part101/read101.py, loaded by file path), and write the verdicts for 102.1 to 102.5.
Usage: python3 -I part102/score102.py   (writes results/part102/score102.json)
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("read101", ROOT / "part101" / "read101.py")
read101 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(read101)
R102 = ROOT / "results" / "part102"
RUNS = ["attnSFT_L0-3", "attnDPO_L0-3", "attnB_L1-3"]


def main():
    kind_of = read101.kinds()
    ids, slot, concept = read101.unit_sets(kind_of, R102 / f"{RUNS[0]}.jsonl")
    runs = {}
    for tag in RUNS:
        lab, errors, missing = read101.load(R102 / f"labels-{tag}.jsonl")
        runs[tag] = {"label_error_rows": errors, "items_still_missing": missing,
                     **read101.score_rows(lab, ids, slot, concept)}
    sw = lambda t: runs[t]["slot_who"][0]
    ok = lambda t: not runs[t]["inconclusive"]
    p = {"102.1": "inconclusive" if not ok("attnSFT_L0-3") else ("held" if sw("attnSFT_L0-3") >= 15 else "failed"),
         "102.2": "inconclusive" if not ok("attnDPO_L0-3") else ("held" if sw("attnDPO_L0-3") >= 15 else "failed"),
         "102.3": "inconclusive" if not ok("attnB_L1-3") else ("held" if sw("attnB_L1-3") <= 9 else "failed"),
         "102.4": "held" if all(runs[t]["slot_who_coherent"][0] >= 13 for t in RUNS) else "failed",
         "102.5": "held" if all(runs[t]["concept_any"][0] >= 13 for t in RUNS) else "failed"}
    out = {"n_slot": len(slot), "n_concept": len(concept), "runs": runs, "predictions": p}
    (R102 / "score102.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
