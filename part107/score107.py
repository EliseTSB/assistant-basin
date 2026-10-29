#!/usr/bin/env python3
"""Score Part 107 (PART107-DESIGN.md) from where107.py's JSON. Written by Claude (Opus 5.5), 2026-10-29, before any run.
Usage: python part107/score107.py [IN.json] [OUT.json]   (defaults results/part107/where107.json, score107.json)
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def score(data: dict) -> dict:
    st = data["stages"]
    sft, dpo = st["sft_change"], st["dpo_change"]
    shared = sorted(set(sft["groups"]) & set(dpo["groups"]))
    ratios = {g: sft["groups"][g]["rel_change"] / max(dpo["groups"][g]["rel_change"], 1e-30) for g in shared}
    out = {"107.1": {"sft_over_dpo_by_group": {g: round(v, 2) for g, v in ratios.items()},
                     "verdict": "held" if ratios and all(v >= 10 for v in ratios.values()) else "failed"}}
    share = dpo["precision_floor"]["one_step_frac_of_changed"]
    out["107.2"] = {"dpo_one_step_share_of_changed": round(share, 4),
                    "dpo_changed_frac": round(dpo["precision_floor"]["changed_frac"], 4),
                    "verdict": "held" if share >= 0.5 else "failed"}
    units = {u["id"]: u for u in data["units"]}
    r = dpo["unit_R"]
    med = statistics.median
    slot = [r[i] for i, u in units.items() if u["kind"] == "slot"]
    concept = [r[i] for i, u in units.items() if u["kind"] == "concept"]
    above = sum(r[i] > med(dpo["random_R_by_layer"][str(u["layer"])]) for i, u in units.items() if u["kind"] == "slot")
    ratio = med(slot) / max(med(concept), 1e-30)
    out["107.3"] = {"median_R_slot_over_concept": round(ratio, 3), "slot_above_layer_random_median": above,
                    "of": len(slot), "verdict": "held" if ratio >= 1.2 and above >= 14 else "failed"}
    if out["107.2"]["verdict"] == "held":
        reading = "107.3 reads rounding, not the DPO update; no claim about where DPO acts is made from R"
    elif out["107.3"]["verdict"] == "held":
        reading = "DPO's change is aimed at the slot units' read paths"
    else:
        reading = "DPO's change does not single out the slot units' directions in the weights"
    out["reading"] = reading
    out["by_stage"] = {name: {"groups": {g: round(v["rel_change"], 8) for g, v in s["groups"].items()},
                              "changed_frac": round(s["precision_floor"]["changed_frac"], 4),
                              "one_step_share": round(s["precision_floor"]["one_step_frac_of_changed"], 4),
                              "median_R": {k: med([s["unit_R"][i] for i, u in units.items() if u["kind"] == k])
                                           for k in ("slot", "concept")},
                              "median_R_random": med(v for vals in s["random_R_by_layer"].values() for v in vals)}
                       for name, s in st.items()}
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    src = Path(argv[0]) if argv else ROOT / "results/part107/where107.json"
    dst = Path(argv[1]) if len(argv) > 1 else src.with_name("score107.json")
    out = score(json.loads(src.read_text(encoding="utf-8")))
    dst.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("107.1", "107.2", "107.3", "reading")}, indent=1))


if __name__ == "__main__":
    main()
