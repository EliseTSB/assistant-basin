"""Part 64 Amendment 7 device gate: does scoring on the A770 (xpu) reproduce Part 60's cuda M1 decisions?

Rescores Part 60's dev M1 sentences (results/part60/acts/dev/M1.npz: the exact texts and their cuda max activations on
all 500 dev units) with the same QwenScorer on --device, then compares on each sentence's OWN unit (its record's unit,
from results/part60/out/dev/M1.jsonl) under Part 60's thresholds (stored in M1.npz) and pass rule (at least 3 of 4
strong strictly above threshold, at most 1 of 4 none strictly above).

  G1: per-unit pass decision agrees on >= 98% of units, separately for neurons and controls.
  G2: the pass rate differs by <= 1 point, separately for neurons and controls.
  G3 (reported only): own-unit above-threshold flip share; median |relative difference| of own-unit activations.

Writes --out (JSON) and prints GATE PASS or GATE FAIL. Registered in PART64-DESIGN.md Amendment 7 before any Part 64
sentence was scored.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from part60 import score60  # noqa: E402

PASS_STRONG, PASS_NONE_MAX = 3, 1


def specs_from_keys(unit_keys, unit_types):
    specs = []
    for key, kind in zip(unit_keys, unit_types):
        layer_part, unit_part = str(key).split(":")
        layer = int(layer_part[1:])
        if unit_part.startswith("u"):
            specs.append({"unit_key": str(key), "layer": layer, "unit_index": int(unit_part[1:]), "control_id": None,
                          "type": str(kind)})
        elif unit_part.startswith("c"):
            specs.append({"unit_key": str(key), "layer": layer, "unit_index": None, "control_id": int(unit_part[1:]),
                          "type": str(kind)})
        else:
            raise ValueError(f"unrecognised unit key {key}")
        if (specs[-1]["unit_index"] is None) != (str(kind) == "control"):
            raise ValueError(f"unit key {key} disagrees with its type {kind}")
    return specs


def decisions(max_matrix, own_col, groups, thresholds, unit_count):
    """Per-unit pass decision from each sentence's own-unit activation; None for a unit with no full tiers."""
    above = max_matrix[np.arange(len(own_col)), own_col] > thresholds[own_col]
    strong = np.zeros(unit_count, int); none = np.zeros(unit_count, int)
    n_strong = np.zeros(unit_count, int); n_none = np.zeros(unit_count, int)
    for i, col in enumerate(own_col):
        if groups[i] == "strong":
            n_strong[col] += 1; strong[col] += int(above[i])
        elif groups[i] == "none":
            n_none[col] += 1; none[col] += int(above[i])
    out = []
    for c in range(unit_count):
        if n_strong[c] == 0 or n_none[c] == 0:
            out.append(None)
        else:
            out.append(bool(strong[c] >= PASS_STRONG and none[c] <= PASS_NONE_MAX))
    return out, above


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m1", type=Path, required=True, help="Part 60 results/part60/acts/dev/M1.npz (cuda reference)")
    ap.add_argument("--m1-jsonl", type=Path, required=True, help="Part 60 results/part60/out/dev/M1.jsonl")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N sentences (smoke test)")
    args = ap.parse_args()

    ref = np.load(args.m1, allow_pickle=False)
    if str(ref["dtype"]) != "bf16" or int(ref["batch_size"]) != args.batch_size:
        raise SystemExit(f"reference was {ref['dtype']} batch {ref['batch_size']}; match it")
    unit_keys = [str(k) for k in ref["unit_keys"]]
    col_of = {k: i for i, k in enumerate(unit_keys)}
    record_unit = {}
    for line in args.m1_jsonl.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            record_unit[int(row["record_index"])] = str(row["unit_key"])
    own_col = np.array([col_of[record_unit[int(r)]] for r in ref["record_indices"]], dtype=np.int64)
    groups = [str(g) for g in ref["groups"]]
    texts = [str(t) for t in ref["texts"]]
    thresholds = ref["thresholds"].astype(np.float32)
    types = [str(t) for t in ref["unit_types"]]
    n = len(texts) if args.limit is None else min(args.limit, len(texts))

    specs = specs_from_keys(unit_keys, types)
    scorer = score60.QwenScorer(specs, args.model, dtype="bf16", device=args.device, threads=4)
    new_max = np.full((n, len(specs)), np.nan, dtype=np.float32)
    new_marked = np.full((n, len(specs)), np.nan, dtype=np.float32)
    try:
        scorer.score_batch([{"text": t} for t in texts[:n]], list(range(n)), new_max, new_marked,
                           batch_size=args.batch_size)
    finally:
        scorer.close()
    old_max = ref["max"][:n].astype(np.float32)

    report = {"sentences": n, "device": args.device, "reference_device": str(ref["device"]),
              "batch_size": args.batch_size, "dtype": "bf16"}
    a_old = old_max[np.arange(n), own_col[:n]]; a_new = new_max[np.arange(n), own_col[:n]]
    rel = np.abs(a_new - a_old) / np.maximum(np.abs(a_old), 1e-6)
    report["G3_median_abs_rel_diff_own_unit"] = float(np.median(rel))
    report["G3_max_abs_rel_diff_own_unit"] = float(np.max(rel))
    d_old, above_old = decisions(old_max, own_col[:n], groups[:n], thresholds, len(specs))
    d_new, above_new = decisions(new_max, own_col[:n], groups[:n], thresholds, len(specs))
    report["G3_own_unit_flag_flip_share"] = float(np.mean(above_old != above_new))
    report["G3_own_unit_flag_flips"] = int(np.sum(above_old != above_new))
    ok = True
    for kind in ("neuron", "control"):
        cols = [c for c in range(len(specs)) if types[c] == kind and d_old[c] is not None and d_new[c] is not None]
        agree = sum(d_old[c] == d_new[c] for c in cols)
        rate_old = 100.0 * sum(d_old[c] for c in cols) / len(cols)
        rate_new = 100.0 * sum(d_new[c] for c in cols) / len(cols)
        g1 = agree / len(cols) >= 0.98
        g2 = abs(rate_new - rate_old) <= 1.0
        ok = ok and g1 and g2
        report[kind] = {"units": len(cols), "decision_agreement": agree / len(cols),
                        "pass_rate_reference": rate_old, "pass_rate_device": rate_new,
                        "pass_rate_diff_points": rate_new - rate_old, "G1": g1, "G2": g2,
                        "disagreeing_units": [unit_keys[c] for c in cols if d_old[c] != d_new[c]]}
    report["gate"] = "PASS" if ok else "FAIL"
    if args.limit is not None:
        report["gate"] = "SMOKE (limit set; not a gate result)"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("neuron", "control")}))
    for kind in ("neuron", "control"):
        r = report[kind]
        print(f"{kind}: {r['units']} units, agreement {r['decision_agreement']:.4f}, pass rate "
              f"{r['pass_rate_reference']:.2f} -> {r['pass_rate_device']:.2f} (G1 {r['G1']}, G2 {r['G2']})")
    print(f"GATE {report['gate']}")
    return 0 if report["gate"] == "PASS" or args.limit is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
