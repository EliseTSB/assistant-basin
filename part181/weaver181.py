#!/usr/bin/env python3
"""weaver181.py -- Part 181 Amendment 3: Weaver's predictions 181.W1 and 181.W2, read from the same states and the
same primary cohort as analyze181.py (PART181-DESIGN.md, Amendment 3). Written before any collected value was read.

181.W1: floor of a repeat pair = 1 - the median neuron top-10 overlap on the primary cohort per layer (Gate 3's
neuron_median), then the median over the layers that pass Gate 3. Held if floor(pretrain_1T, pretrain_1T_b) >=
2 x floor(instruct, instruct_b). Failed under 2, or if both floors are 0.

181.W2: absolute rule = active if the unit's maximum is positive and at least 10% of the median maximum of its layer's
neurons in `instruct`, the same threshold for every run. Relative rule = analyze181._active. Per run, active neurons
summed over layers under each; difference = (relative - absolute) / absolute. Held if >= 0.10 at one or more of the
six pretraining checkpoints; failed if under 0.10 at every one.

Usage: python -I part181/weaver181.py --runs <dir> --out <dir>   (writes weaver181.json)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze181 as a181  # noqa: E402

PRETRAIN = ("pretrain_21B", "pretrain_210B", "pretrain_1T", "pretrain_2T", "pretrain_3T", "pretrain_4T")
W2_MIN = 0.10
W1_RATIO = 2.0


def _floor(gate3: dict, pair: str) -> float | None:
    vals = []
    for layer in gate3["passing_layers"]:
        r = gate3["layers"][str(layer)]["repeat_pairs"].get(pair, {})
        if r.get("neuron_median") is not None:
            vals.append(1.0 - float(r["neuron_median"]))
    return float(np.median(vals)) if vals else None


def _mask_disagreement(gate3: dict, pair: str) -> float | None:
    vals = [1.0 - gate3["layers"][str(l)]["repeat_pairs"][pair]["activity_mask_agreement"]
            for l in gate3["passing_layers"]
            if gate3["layers"][str(l)]["repeat_pairs"].get(pair, {}).get("activity_mask_agreement") is not None]
    return float(np.median(vals)) if vals else None


def w1(runs_by_key: dict[str, a181.Run], primary_ids: dict[int, np.ndarray]) -> dict[str, Any]:
    gate3 = a181._gate3(runs_by_key, primary_ids)
    early = _floor(gate3, "pretrain_1T_vs_pretrain_1T_b")
    late = _floor(gate3, "instruct_vs_instruct_b")
    out = {"floor_1T_pair": early, "floor_instruct_pair": late, "passing_layers": gate3["passing_layers"],
           "mask_disagreement_1T_pair": _mask_disagreement(gate3, "pretrain_1T_vs_pretrain_1T_b"),
           "mask_disagreement_instruct_pair": _mask_disagreement(gate3, "instruct_vs_instruct_b")}
    if early is None or late is None:
        out.update(verdict="not read", reason="no Gate-3 layer gives both floors")
    elif early == 0 and late == 0:
        out.update(verdict="failed", ratio=None, reason="both floors are 0: the repeats are exact")
    elif late == 0:
        out.update(verdict="held", ratio=None, reason="the Instruct floor is 0 and the 1T floor is not")
    else:
        ratio = early / late
        out.update(verdict="held" if ratio >= W1_RATIO else "failed", ratio=ratio)
    return out


def _absolute_thresholds(instruct: a181.Run) -> dict[int, float | None]:
    layers, width, _ = a181._layout(instruct)
    out = {}
    for slot, layer in enumerate(layers):
        m = instruct.state["max"][slot, :width]
        finite = m[np.isfinite(m)]
        med = float(np.median(finite)) if len(finite) else None
        out[layer] = 0.1 * med if med is not None and med > 0 else None
    return out


def active_counts(run: a181.Run, thresholds: dict[int, float | None]) -> dict[str, int]:
    layers, width, _ = a181._layout(run)
    rel = absol = 0
    for slot, layer in enumerate(layers):
        m = run.state["max"][slot, :width]
        rel += int(a181._active(m).sum())
        tau = thresholds.get(layer)
        if tau is not None:
            absol += int((np.isfinite(m) & (m > 0) & (m >= tau)).sum())
    return {"relative": rel, "absolute": absol}


def w2(runs_by_key: dict[str, a181.Run]) -> dict[str, Any]:
    instruct = runs_by_key.get("instruct")
    if instruct is None:
        return {"verdict": "not read", "reason": "no instruct run"}
    thresholds = _absolute_thresholds(instruct)
    per_run = {}
    for key, run in sorted(runs_by_key.items()):
        c = active_counts(run, thresholds)
        c["difference"] = (c["relative"] - c["absolute"]) / c["absolute"] if c["absolute"] else None
        per_run[key] = c
    pre = {k: per_run[k]["difference"] for k in PRETRAIN if k in per_run}
    if len(pre) < len(PRETRAIN):
        verdict, reason = "not read", f"pretraining runs present: {sorted(pre)}"
    elif any(v is None for v in pre.values()):
        verdict, reason = "not read", "an absolute count of 0 at a pretraining checkpoint"
    else:
        verdict = "held" if any(v >= W2_MIN for v in pre.values()) else "failed"
        reason = None
    finite = {k: v["difference"] for k, v in per_run.items() if v["difference"] is not None}
    largest = max(finite, key=lambda k: finite[k]) if finite else None
    return {"verdict": verdict, "reason": reason, "threshold": W2_MIN, "per_run": per_run,
            "pretraining_differences": pre, "largest_difference_run": largest}


def keyed_runs(runs) -> dict[str, a181.Run]:
    """Runs keyed by their own key. Amendment 4 (2026-11-04): main() first keyed them through
    a181._canonical_key, which renames instruct_b to instruct, so the Instruct repeat pair collapsed into one entry,
    181.W1 read "not read" and 181.W2's Instruct threshold came from instruct_b. analyze181.build_verdicts keys by
    run.key; this does the same and refuses a duplicate key instead of letting the later run win."""
    out: dict[str, a181.Run] = {}
    for r in runs:
        if r.key in out:
            raise ValueError(f"two runs share the key {r.key!r}")
        out[r.key] = r
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    runs, exclusions, _gate2 = a181._resolve_runs(Path(a.runs))
    runs_by_key = keyed_runs(runs)
    primary_ids = a181._primary_ids_map(runs_by_key.get("instruct"))
    result = {"181.W1": w1(runs_by_key, primary_ids), "181.W2": w2(runs_by_key),
              "excluded_runs": exclusions, "author": "Weaver (predictions); Claude (code)"}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "weaver181.json").write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k]["verdict"] for k in ("181.W1", "181.W2")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
