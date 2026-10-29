#!/usr/bin/env python3
"""paired102.py -- Part 102, exploratory (not a registered prediction): the same 21 slot units across runs, unit by
unit. For each pair of runs, counts the units that show their persona on "Who are you?" in one run and not the other,
and gives an exact two-sided McNemar (binomial) p on those discordant units. Uses Part 101's read-set scorer.
Usage: python3 -I part102/paired102.py   (writes results/part102/paired.json)
"""
import importlib.util
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("read101", ROOT / "part101" / "read101.py")
read101 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(read101)
RES = ROOT / "results"
RUNS = {"instruct": RES / "part99" / "labels-inst_none.jsonl", "mid_end_L0-3": RES / "part100" / "labels-attnB_L0-3.jsonl",
        "sft_L0-3": RES / "part102" / "labels-attnSFT_L0-3.jsonl", "dpo_L0-3": RES / "part102" / "labels-attnDPO_L0-3.jsonl",
        "mid_end_L1-3": RES / "part102" / "labels-attnB_L1-3.jsonl"}
PAIRS = [("sft_L0-3", "dpo_L0-3"), ("sft_L0-3", "instruct"), ("dpo_L0-3", "instruct"), ("mid_end_L0-3", "sft_L0-3"),
         ("mid_end_L1-3", "mid_end_L0-3")]


def who_by_unit(path, kind_of, slot):
    lab, _errors, _missing = read101.load(path, lambda r: read101.in_read_set(r, kind_of))
    out = {}
    for i in slot:
        p, b = lab.get((i, "neuron", 0)), lab.get((i, "baseline", 0))
        out[i] = None if p is None or b is None else bool(p["shows_persona"]) and not bool(b["shows_persona"])
    return out


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    kind_of = read101.kinds()
    slot = sorted(i for i, k in kind_of.items() if k == "slot")
    table = {name: who_by_unit(path, kind_of, slot) for name, path in RUNS.items()}
    pairs = {}
    for a, z in PAIRS:
        both = [i for i in slot if table[a][i] is not None and table[z][i] is not None]
        only_a = sum(1 for i in both if table[a][i] and not table[z][i])
        only_z = sum(1 for i in both if table[z][i] and not table[a][i])
        pairs[f"{a} vs {z}"] = {"units": len(both), f"only_{a}": only_a, f"only_{z}": only_z,
                                "mcnemar_p": round(mcnemar(only_a, only_z), 4)}
    out = {"units": {i: {name: table[name][i] for name in RUNS} for i in slot}, "pairs": pairs}
    (RES / "part102" / "paired.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for k, v in pairs.items():
        print(k, v)


if __name__ == "__main__":
    main()
