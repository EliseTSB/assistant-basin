#!/usr/bin/env python3
"""Part 128 scoring. Gate first, then the seven predictions, on the primary reader; Luna is reported beside every number.
A clear match is a score of 2 or more on a side. Written by Claude (Opus 5.5), 2026-10-30.
Usage: score128.py OUT_DIR
"""
import json
import statistics
import sys
from pathlib import Path

BAND_DARK = [((0, 8), .29), ((9, 12), .72), ((13, 20), .82), ((21, 27), .59), ((28, 35), .52)]


def load(path):
    rows = {}
    for r in map(json.loads, open(path, encoding="utf-8")):
        if "error" not in r:
            rows[(r["id"], r["arm"])] = r
    return rows


def either(r):
    return max(r["fire_side"], r["write_side"]) >= 2


def kappa(a, b):
    keys = sorted(set(a) & set(b))
    x, y = [either(a[k]) for k in keys], [either(b[k]) for k in keys]
    n = len(keys)
    po = sum(p == q for p, q in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def expected_dark(units):
    total = 0.0
    for u in units:
        total += next(rate for (lo, hi), rate in BAND_DARK if lo <= u["L"] <= hi)
    return total


def summarise(rows, units):
    def count(arm, ids=None, side=either):
        return sum(side(rows[(u["id"], arm)]) for u in units if (ids is None or u["class"] == ids) and (u["id"], arm) in rows)
    fire = lambda r: r["fire_side"] >= 2
    known = sorted({k[0] for k in rows if k[1] == "known-true"})
    return {
        "gate_true": sum(either(rows[(k, "known-true")]) for k in known),
        "gate_false": sum(either(rows[(k, "known-false")]) for k in known),
        "gate_n": len(known),
        "pushed": count("pushed"), "mismatch": count("mismatch"), "opposite": count("opposite"), "control": count("control"),
        "carrier": count("pushed", "carrier"), "once": count("pushed", "once"),
        "carrier_fire": count("pushed", "carrier", fire), "once_fire": count("pushed", "once", fire),
        "carrier_n": sum(u["class"] == "carrier" for u in units), "once_n": sum(u["class"] == "once" for u in units),
        "missing": sum((u["id"], a) not in rows for u in units for a in ("pushed", "mismatch", "opposite", "control")),
    }


def main(out):
    out = Path(out)
    units = json.load(open(out / "units128.json", encoding="utf-8"))
    ds, luna = load(out / "labels-ds.jsonl"), load(out / "labels-luna.jsonl")
    a, b = summarise(ds, units), summarise(luna, units)
    ratios = sorted(abs(u["c"]) / u["pushed"]["act_top"] for u in units if u["pushed"]["act_top"] > 0)
    by = {}
    for c in ("carrier", "once"):
        values = [abs(u["c"]) / u["pushed"]["act_top"] for u in units if u["class"] == c and u["pushed"]["act_top"] > 0]
        by[c] = statistics.median(values) if values else None
    dark = sum(u["pushed"]["dark"] for u in units)
    report = {"ds": a, "luna": b, "kappa_either": kappa(ds, luna), "ratio_median": statistics.median(ratios),
              "ratio_ge3": sum(r >= 3 for r in ratios), "ratio_n": len(ratios), "ratio_median_by_class": by,
              "dark": dark, "expected_dark": expected_dark(units), "n": len(units)}
    report["gate"] = {"ds": a["gate_true"] >= 5 and a["gate_false"] <= 1, "luna": b["gate_true"] >= 5 and b["gate_false"] <= 1,
                      "mismatch": a["mismatch"] <= 6, "kappa": report["kappa_either"] >= .6, "complete": a["missing"] == 0 and b["missing"] == 0}
    json.dump(report, open(out / "score128.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
