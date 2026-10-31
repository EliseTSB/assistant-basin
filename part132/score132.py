#!/usr/bin/env python3
"""Part 132 scoring, on the primary reader with Luna beside it. Written by Claude (Opus 5.5), 2026-10-30.
Usage: score132.py OUT_DIR"""
import json
import re
import sys
from pathlib import Path

ROLE = re.compile(r"human|person|people|student|teacher|assistant|employee|worker|customer|user|man\b|woman|child|adult|parent", re.I)
BANDS = ((2, 13), (14, 23), (24, 33))


def load(path):
    rows = {}
    for r in map(json.loads, open(path, encoding="utf-8")):
        if "error" not in r:
            rows[(r["id"], r["arm"])] = r
    return rows


either = lambda r: max(r["fire_side"], r["write_side"]) >= 2
fire = lambda r: r["fire_side"] >= 2


def rate(rows, units, arm, side):
    have = [u for u in units if (u["id"], arm) in rows]
    return sum(side(rows[(u["id"], arm)]) for u in have), len(have)


def summary(rows, units):
    out = {}
    for cls in ("carrier", "once"):
        group = [u for u in units if u["class"] == cls]
        out[cls] = {"n": len(group), "either": rate(rows, group, "pushed", either), "fire": rate(rows, group, "pushed", fire),
                    "mismatch": rate(rows, group, "mismatch", either), "opposite": rate(rows, group, "opposite", either),
                    "control": rate(rows, group, "control", either)}
    out["bands"] = {f"L{lo}-{hi}": {c: rate(rows, [u for u in units if u["class"] == c and lo <= u["L"] <= hi], "pushed", either)
                                    for c in ("carrier", "once")} for lo, hi in BANDS}
    known = sorted({k[0] for k in rows if k[1] == "known-true"})
    out["gate_true"] = sum(either(rows[(k, "known-true")]) for k in known)
    out["gate_false"] = sum(either(rows[(k, "known-false")]) for k in known)
    return out


def kappa(a, b):
    keys = sorted(set(a) & set(b))
    x, y = [either(a[k]) for k in keys], [either(b[k]) for k in keys]
    n = len(keys)
    po = sum(p == q for p, q in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return (po - pe) / (1 - pe)


def main(out):
    out = Path(out)
    units = json.load(open(out / "units128.json", encoding="utf-8"))
    ds, luna = load(out / "labels-ds.jsonl"), load(out / "labels-luna.jsonl")
    role = {c: (sum(bool(ROLE.search(u["persona"])) for u in units if u["class"] == c), sum(u["class"] == c for u in units)) for c in ("carrier", "once")}
    report = {"ds": summary(ds, units), "luna": summary(luna, units), "kappa": kappa(ds, luna), "role_names": role, "n": len(units)}
    json.dump(report, open(out / "score132.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
