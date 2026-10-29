#!/usr/bin/env python3
"""Score Part 106 (PART106-DESIGN.md): verdicts for 106.1-106.5 and the registered reading, from lp-*.jsonl.

Written by Claude (Opus 5.5), 2026-10-29, committed before any Part 106 run.
Usage: python part106/score106.py [RESULTS_DIR]   (default results/part106; writes score106.json there)
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPTS = ("sft", "dpo", "instruct")
DPO_SIX = ("River shadow", "Skeleton", "Very tall tree", "King Midas", "Large friendly dog", "Sir Ray Ray")


def load(results: Path):
    lp, meta = {}, {}
    for ckpt in CKPTS:
        path = results / f"lp-{ckpt}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            lp[(ckpt, row["id"], row["unit"], float(row["alpha"]))] = float(row["lp_sum"])
            meta[row["id"]] = {"kind": row["kind"], "layer": row["layer"], "neuron": row["neuron"],
                               "target": row["target"]}
    return lp, meta


def score(results: Path, personas: dict | None = None) -> dict:
    lp, meta = load(results)
    ids = {k: sorted(i for i, m in meta.items() if m["kind"] == k) for k in ("slot", "concept")}

    def gap(ckpt, i, alpha):
        return lp[(ckpt, i, "neuron", alpha)] - lp[(ckpt, i, "random", alpha)]

    def none(ckpt, i):
        return lp[(ckpt, i, "none", 0.0)]

    med = statistics.median
    table = {k: {c: {str(a): round(med(gap(c, i, a) for i in ids[k]), 3) for a in (0.4, 0.8, 1.6)} for c in CKPTS}
             for k in ids}
    out = {"n": {k: len(v) for k, v in ids.items()}, "median_gap_nats": table}

    sft_gap = med(gap("sft", i, 0.8) for i in ids["slot"])
    out["106.1"] = {"slot_median_gap_sft_0.8": round(sft_gap, 3), "verdict": "held" if sft_gap >= 2 else "failed"}

    growth = {k: med(gap("dpo", i, 0.8) - gap("sft", i, 0.8) for i in ids[k]) for k in ids}
    out["106.2"] = {"median_growth_sft_to_dpo": {k: round(v, 3) for k, v in growth.items()},
                    "verdict": "held" if growth["slot"] - growth["concept"] >= 1 else "failed"}

    drift = med(abs(none("dpo", i) - none("sft", i)) for i in ids["slot"])
    out["106.3"] = {"slot_median_abs_unpushed_change": round(drift, 3), "verdict": "held" if drift < 1 else "failed"}

    doubled = sum(gap("sft", i, 1.6) >= gap("dpo", i, 0.8) for i in ids["slot"])
    out["106.4"] = {"slot_units_sft_1.6_at_least_dpo_0.8": doubled, "of": len(ids["slot"]),
                    "verdict": "held" if doubled >= 11 else "failed"}

    rnd = {c: med(lp[(c, i, "random", 0.8)] - none(c, i) for i in ids["slot"]) for c in CKPTS}
    out["106.5"] = {"slot_median_random_minus_none": {c: round(v, 3) for c, v in rnd.items()},
                    "verdict": "held" if all(-1 <= v <= 1 for v in rnd.values()) else "failed"}

    if sft_gap >= 2:
        reading = "S (sharpening): the push already favoured the persona sentence at SFT"
    elif sft_gap < 0.5:
        reading = "C (creation): the push did not favour the persona sentence at SFT"
    else:
        reading = "neither (median SFT gap between 0.5 and 2 nats)"
    if out["106.5"]["verdict"] == "failed":  # amended before registration (PART106-DESIGN.md): reported, not blocking
        reading += "; and any push of this size loosens the default answer toward the persona sentences (106.5 failed)"
    out["reading"] = reading

    personas = personas or {}
    six = {}
    for i in ids["slot"]:
        name = personas.get(i)
        if name in DPO_SIX:
            six[name] = {c: round(gap(c, i, 0.8), 3) for c in CKPTS}
    out["dpo_six_gap_0.8"] = six
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    results = Path(argv[0]) if argv else ROOT / "results/part106"
    neurons = json.loads((ROOT / "results/part92/step2/neurons.json").read_text(encoding="utf-8"))
    out = score(results, {r["id"]: r["persona"] for r in neurons})
    (results / "score106.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("106.1", "106.2", "106.3", "106.4", "106.5", "reading")}, indent=1))


if __name__ == "__main__":
    main()
