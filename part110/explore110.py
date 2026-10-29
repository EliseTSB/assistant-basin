#!/usr/bin/env python3
"""Part 110, EXPLORATORY (not registered): where each named neuron's chat/base norm ratio sits in its layer.

Written by Claude (Opus 5.5), 2026-10-29, after reading weights110.json: the layer medians shrank by about 0.3% while
L7 N1169 grew, so this asks how unusual that is. Seven neurons times three matrices is 21 looks, so one top-1% value
is expected by chance about one time in five. Reads results/part110/moves110.npz (written by fig110.py).
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import weights110 as p110

d = np.load("results/part110/moves110.npz")
out = {}
for name, (l, u) in {**p110.MAKERS, **p110.SECONDARY}.items():
    out[name] = {m: {"ratio": float(d[f"ratio_{m}_L{l}"][u]),
                     "pct_in_layer": float(100.0 * np.mean(d[f"ratio_{m}_L{l}"] <= d[f"ratio_{m}_L{l}"][u]))}
                 for m in p110.MATS}
Path("results/part110/explore110.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
for k, v in out.items():
    print(k, {m: (round(x["ratio"], 4), round(x["pct_in_layer"], 1)) for m, x in v.items()})
