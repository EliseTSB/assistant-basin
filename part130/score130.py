#!/usr/bin/env python3
"""Part 130 reader counts and agreement, from labels-ds.jsonl and labels-luna.jsonl. Rows with an error key are failed
tries; the last good row per item counts. Written by Claude (Opus 5.5), 2026-10-30. Usage: score130.py RESULTS_DIR"""
import json
import sys
from pathlib import Path


def good(path):
    rows = {}
    for r in map(json.loads, open(path, encoding="utf-8")):
        if "error" not in r:
            rows[(r["id"], r["arm"])] = r["same"]
    return rows


def main(results):
    results = Path(results)
    ds, luna = good(results / "labels-ds.jsonl"), good(results / "labels-luna.jsonl")
    keys = sorted(set(ds) & set(luna))
    x, y = [ds[k] for k in keys], [luna[k] for k in keys]
    n = len(keys)
    po = sum(a == b for a, b in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    same = [k for k in keys if k[1] == "same"]
    report = {"ds_same": sum(v for k, v in ds.items() if k[1] == "same"), "ds_same_n": sum(k[1] == "same" for k in ds),
              "ds_shuffled": sum(v for k, v in ds.items() if k[1] == "shuffled"), "ds_shuffled_n": sum(k[1] == "shuffled" for k in ds),
              "luna_rows": n, "kappa": (po - pe) / (1 - pe), "luna_same_true": sum(luna[k] for k in same), "luna_same_n": len(same),
              "luna_ds_same_true": sum(ds[k] for k in same)}
    json.dump(report, open(results / "score130.json", "w"), indent=1)
    print(report)


if __name__ == "__main__":
    main(sys.argv[1])
