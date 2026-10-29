#!/usr/bin/env python3
"""score103.py -- Part 103: score the stage-by-component swaps and the whole-model references (PART103-DESIGN.md) with Part 101's
read-set scorer (part101/read101.py, loaded by file path), and write the verdicts for 103.1 to 103.6.
Usage: python3 -I part103/score103.py   (writes results/part103/score103.json)
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("read101", ROOT / "part101" / "read101.py")
read101 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(read101)
R103 = ROOT / "results" / "part103"
RUNS = ["attnSFT_all", "mlpSFT_all", "attnDPO_all", "mlpDPO_all", "whole_SFT", "whole_DPO"]
SWAPS = RUNS[:4]


def main():
    kind_of = read101.kinds()
    ids, slot, concept = read101.unit_sets(kind_of, R103 / f"{RUNS[0]}.jsonl")
    runs = {}
    for tag in RUNS:
        lab, errors, missing = read101.load(R103 / f"labels-{tag}.jsonl")
        runs[tag] = {"label_error_rows": errors, "items_still_missing": missing,
                     **read101.score_rows(lab, ids, slot, concept)}
    sw = lambda t: runs[t]["slot_who"][0]
    ok = lambda *ts: all(not runs[t]["inconclusive"] for t in ts)
    verdict = lambda cond, *ts: "inconclusive" if not ok(*ts) else ("held" if cond else "failed")
    p = {"103.1": verdict(sw("attnDPO_all") >= 15, "attnDPO_all"),
         "103.2": verdict(sw("mlpDPO_all") <= sw("attnDPO_all") - 3, "mlpDPO_all", "attnDPO_all"),
         "103.3": verdict(sw("attnSFT_all") <= 14 and sw("mlpSFT_all") <= 14, "attnSFT_all", "mlpSFT_all"),
         "103.4": verdict(sw("whole_SFT") <= 10 and sw("whole_DPO") <= 16, "whole_SFT", "whole_DPO"),
         "103.5": "held" if all(runs[t]["slot_who_coherent"][0] >= 13 for t in SWAPS) else "failed",
         "103.6": "held" if all(runs[t]["concept_any"][0] >= 13 for t in RUNS) else "failed"}
    out = {"n_slot": len(slot), "n_concept": len(concept), "runs": runs, "predictions": p}
    (R103 / "score103.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
