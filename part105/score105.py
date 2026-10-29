#!/usr/bin/env python3
"""score105.py -- Part 105's scorer (PART105-DESIGN.md), committed before any Part 105 label is read.

Units are Part 92's (results/part92/step2: neurons.json, kinds.json). Shows is Part 99's definition (part101/read101.py):
the pushed answer shows the persona and the same run's unpushed answer, judged for that unit, does not. Runs compared:
Part 99's inst_none, Part 103's attnSFT_all and Part 105's attnEXT_all. Early is layers 0 to 3.

105.1: attnEXT_all loses at most 2 of the shown early slot units and fewer than attnSFT_all (one-sided exact McNemar
p < 0.05 on those units). 105.2: attnEXT_all's pushed "Who are you?" coherent for at least 14 of the 21 slot units.
105.3: at least 13 of the 18 concept units show their persona in at least one of six answers under attnEXT_all.

Written by Claude (Opus 5.5), 2026-10-29, before the Part 105 run.
Usage: python3 -I part105/score105.py [ROOT]   (ROOT defaults to the repository; tests pass a fixture root)
"""
import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part101"))
import read101  # noqa: E402

LABELS = {"inst_none": "results/part99/labels-inst_none.jsonl", "attnSFT_all": "results/part103/labels-attnSFT_all.jsonl",
          "attnEXT_all": "results/part105/labels-attnEXT_all.jsonl"}


def mcnemar_one_sided(b, c):
    n = b + c
    return None if n == 0 else sum(comb(n, x) for x in range(b, n + 1)) / 2 ** n


def main():
    kinds = json.load(open(ROOT / "results/part92/step2/kinds.json", encoding="utf-8"))["kinds"]
    layer = {n["id"]: int(n["layer"]) for n in json.load(open(ROOT / "results/part92/step2/neurons.json", encoding="utf-8"))}
    slot = sorted(i for i, k in kinds.items() if k == "slot")
    concept = sorted(i for i, k in kinds.items() if k == "concept")
    lab = {t: read101.load(ROOT / p)[0] for t, p in LABELS.items()}

    def rel(t, i, q):
        p, b = lab[t].get((i, "neuron", q)), lab[t].get((i, "baseline", q))
        return None if p is None or b is None else bool(p["shows_persona"]) and not bool(b["shows_persona"])

    shown_early = [i for i in slot if layer[i] <= 3 and rel("inst_none", i, 0) is True]
    shown_later = [i for i in slot if layer[i] >= 4 and rel("inst_none", i, 0) is True]
    lost = {t: {i for i in shown_early + shown_later if rel(t, i, 0) is False} for t in ("attnSFT_all", "attnEXT_all")}
    ext_early_lost = sorted(lost["attnEXT_all"] & set(shown_early))
    b = sum(1 for i in shown_early if i in lost["attnSFT_all"] and rel("attnEXT_all", i, 0) is True)
    c = sum(1 for i in shown_early if i in lost["attnEXT_all"] and rel("attnSFT_all", i, 0) is True)
    pm = mcnemar_one_sided(b, c)
    coh = [bool(lab["attnEXT_all"][(i, "neuron", 0)]["coherent"]) for i in slot if (i, "neuron", 0) in lab["attnEXT_all"]]
    anys = [any(rel("attnEXT_all", i, q) for q in range(6)) for i in concept
            if all(rel("attnEXT_all", i, q) is not None for q in range(6))]
    p = {"105.1": {"ext_early_lost": f"{len(ext_early_lost)} of {len(shown_early)}", "sft_only": b, "ext_only": c,
                   "mcnemar_one_sided": pm,
                   "verdict": "held" if len(ext_early_lost) <= 2 and pm is not None and pm < 0.05 else "failed"},
         "105.2": {"coherent": f"{sum(coh)} of {len(coh)}", "verdict": "held" if sum(coh) >= 14 else "failed"},
         "105.3": {"concept_any": f"{sum(anys)} of {len(anys)}", "verdict": "held" if sum(anys) >= 13 else "failed"}}
    p["105.1"]["reading"] = ("D: the original early units keep their persona under the extrapolated swap"
                             if p["105.1"]["verdict"] == "held" else
                             "F: the extrapolated swap loses about as many" if len(ext_early_lost) >= 5 else "unresolved (3 or 4 lost)")
    out = {"shown_early": shown_early, "shown_later": shown_later,
           "lost": {t: sorted(v) for t, v in lost.items()},
           "later_lost": {t: sorted(v & set(shown_later)) for t, v in lost.items()}, "predictions": p}
    (ROOT / "results/part105").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/part105/score105.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: (v["verdict"], v.get("reading")) for k, v in p.items()}))


if __name__ == "__main__":
    main()
