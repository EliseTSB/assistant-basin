#!/usr/bin/env python3
"""Score Part 133 registered gates and predictions from collector sufficient statistics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

GROUPS = ("web", "chat-user", "chat-assistant")
PAIRS = (("parent", "sft"), ("sft", "dpo"), ("dpo", "instruct"), ("parent", "instruct"))


def correlations(sums, counts):
    """Pearson r from per-neuron [sx,sy,sxx,syy,sxy], returned as [layer,width]."""
    n = float(counts)
    if n <= 0: return np.full(sums.shape[-1], np.nan)
    sx, sy, xx, yy, xy = np.asarray(sums, dtype=np.float64)
    vx, vy = xx - sx*sx/n, yy - sy*sy/n
    denom = np.sqrt(np.maximum(vx, 0) * np.maximum(vy, 0))
    return np.divide(xy - sx*sy/n, denom, out=np.full_like(sx, np.nan), where=(vx > 0) & (vy > 0))


def median(values):
    a = np.asarray(values, dtype=float); a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else None


def score_data(data, classes):
    totals = data["totals"].numpy(); counts = data["counts"].numpy(); active = data["active"].numpy()
    groups = {name: i for i, name in enumerate(GROUPS)}
    pair_labels = [f"{a}-{b}" for a, b in PAIRS]
    rs = np.full((len(GROUPS), len(PAIRS), totals.shape[2], totals.shape[-1]), np.nan)
    excluded = {}
    for gi, group in enumerate(GROUPS):
        excluded[group] = {}
        for pi, pair in enumerate(pair_labels):
            for li in range(totals.shape[2]):
                r = correlations(totals[gi, pi, li], counts[gi, pi, li])
                eligible = (active[gi, pi, li, 0] >= 100) & (active[gi, pi, li, 1] >= 100)
                excluded[group][f"{pair}:layer-{li}"] = int((~eligible).sum())
                r[~eligible] = np.nan
                rs[gi, pi, li] = r
    web = groups["web"]; assistant = groups["chat-assistant"]
    adj = [median(rs[web, pi].reshape(-1)) for pi in range(3)]
    assistant_adj = [median(rs[assistant, pi].reshape(-1)) for pi in range(3)]
    parent_inst_web = median(rs[web, 3].reshape(-1))
    parent_inst_assistant = median(rs[assistant, 3].reshape(-1))

    nulls = data["null_totals"].numpy(); null_n = data["null_counts"].numpy()
    null_layer = [median(np.abs(correlations(nulls[li], null_n[li]))) for li in range(len(null_n))]
    self_stats = data["self_totals"].numpy(); self_n = data["self_counts"].numpy()
    self_layer = [median(correlations(self_stats[li], self_n[li])) for li in range(len(self_n))]
    gate1 = {"layer_median_r": self_layer,
             "pass": bool(self_layer) and all(v is not None and v >= .999 for v in self_layer)}
    gate2 = {"layer_median_abs_derangement_r": null_layer,
             "pass": bool(null_layer) and all(v is not None and v <= .05 for v in null_layer)}

    # Part 92's registered neuron IDs/classes are read from its result files at runtime.
    class_names = classes["kinds"] if "kinds" in classes else classes
    neurons = data.get("part92_neurons", [])
    by_id = {str(n["id"]): n for n in neurons}
    class_r = {}
    if by_id:
        layer_vals = {"concept": [], "slot": []}
        pi = 3; gi = assistant
        for ident, kind in class_names.items():
            if kind not in layer_vals or ident not in by_id: continue
            row = by_id[ident]; li = int(row["layer"]); ni = int(row["neuron"])
            if active[gi, pi, li, :, ni].min() >= 100:
                val = rs[gi, pi, li, ni]
                if np.isfinite(val): layer_vals[kind].append(float(val))
        class_r = {kind: {"n": len(vals), "median_r": median(vals)} for kind, vals in layer_vals.items()}
    p1 = {pair_labels[i]: {"web_median_r": adj[i], "pass": adj[i] is not None and adj[i] >= .85}
          for i in range(3)}
    lowest = min(range(3), key=lambda i: assistant_adj[i] if assistant_adj[i] is not None else float("inf")) if all(x is not None for x in assistant_adj) else None
    p2 = {"assistant_adjacent_medians": dict(zip(pair_labels[:3], assistant_adj)),
          "lowest_pair": pair_labels[lowest] if lowest is not None else None,
          "pass": lowest == 0 if lowest is not None else None}
    gap = parent_inst_assistant - parent_inst_web if parent_inst_assistant is not None and parent_inst_web is not None else None
    p3 = {"parent_instruct_web_median_r": parent_inst_web,
          "parent_instruct_chat_assistant_median_r": parent_inst_assistant,
          "assistant_minus_web": gap, "pass": gap <= -.03 if gap is not None else None}
    cmed = class_r.get("concept", {}).get("median_r"); smed = class_r.get("slot", {}).get("median_r")
    delta = abs(cmed - smed) if cmed is not None and smed is not None else None
    p4 = {"part92_class_medians": class_r, "absolute_difference": delta,
          "pass": delta <= .05 if delta is not None else None}
    return {"gates": {"gate1_instruct_self": gate1, "gate2_derangement_null": gate2},
            "predictions": {"133.1": p1, "133.2": p2, "133.3": p3, "133.4": p4},
            "excluded_below_100_nonzero_positions": excluded,
            "position_counts": counts.tolist(), "meta": data.get("meta", {})}


def score_file(sums_path, part92_dir, out_path):
    data = torch.load(sums_path, map_location="cpu", weights_only=False)
    neurons_file = Path(part92_dir) / "step2" / "neurons.json"
    kinds_file = Path(part92_dir) / "step2" / "kinds.json"
    data["part92_neurons"] = json.loads(neurons_file.read_text(encoding="utf-8"))
    classes = json.loads(kinds_file.read_text(encoding="utf-8"))
    result = score_data(data, classes)
    Path(out_path).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["gates"], indent=2))
    print(json.dumps(result["predictions"], indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sums", required=True); p.add_argument("--part92", default="results/part92")
    p.add_argument("--out", default="score133.json")
    a = p.parse_args(); score_file(a.sums, a.part92, a.out)


if __name__ == "__main__": main()
