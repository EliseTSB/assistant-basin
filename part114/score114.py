#!/usr/bin/env python3
"""Score Part 114 (PART114-DESIGN.md) from results/part114/results114.json. Written by Claude (Opus 5.5), 2026-10-29.

Per candidate: flips = held-out phrasings (of 16) where "Q" is no longer the most likely next token with the neuron
zeroed; mean change in the log-odds of "Q" against the unzeroed run ("none" row); the same for its 20 activity-matched
controls. The pair: greedy continuations of "I am **" with the top one and the top two candidates zeroed; a
continuation that does not start with "Q" means "Q" was not the most likely token. Verdicts follow the design's wording.
"""
import json, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def score(d):
    rows = d["rows"]
    base = next(r for r in rows if r["role"] == "none")
    base_lo = [h["logodds"] for h in base["heldout"]]
    n = len(base_lo)

    def stats(r):
        flips = sum(not h["argmax_is_q"] for h in r["heldout"])
        return flips, statistics.mean(h["logodds"] - b for h, b in zip(r["heldout"], base_lo))

    cands = []
    for c in (r for r in rows if r["role"] == "candidate"):
        ctrl = [stats(r) for r in rows if r["role"] == f"control:L{c['layer']}N{c['neuron']}"]
        f, dl = stats(c)
        cands.append({"layer": c["layer"], "neuron": c["neuron"], "attr": round(c["attr"], 4), "flips": f,
                      "mean_dlogodds": round(dl, 4), "n_controls": len(ctrl),
                      "controls_lowering_as_much": sum(x[1] <= dl for x in ctrl),
                      "controls_mean_dlogodds_min": round(min(x[1] for x in ctrl), 4),
                      "controls_max_flips": max(x[0] for x in ctrl),
                      "replacements": sorted({h["top"] for h in c["heldout"] if not h["argmax_is_q"]})})
    pair = {k: {"not_q": sum(not g["continuation"].startswith("Q") for g in v), "n": len(v),
                "continuations": [g["continuation"] for g in v]} for k, v in d["pair_generations"].items()}
    top = cands[0]
    verdicts = {
        "114.1": "held" if max(c["flips"] for c in cands) >= 8 else "failed",
        "114.2": "held" if max(c["flips"] for c in cands) <= 4 else "failed",
        "114.3": "held" if top["controls_lowering_as_much"] == 0 else "failed",
        "114.4": "held" if pair["top2"]["not_q"] >= 8 else "failed",
    }
    return {"base_q_argmax": sum(h["argmax_is_q"] for h in base["heldout"]), "n_heldout": n,
            "base_mean_logodds": round(statistics.mean(base_lo), 4), "candidates": cands, "pair": pair,
            "verdicts": verdicts}


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/part114/results114.json"
    s = score(json.loads(src.read_text(encoding="utf-8")))
    (src.parent / "score.json").write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8")
    print(f"base: Q is the most likely token on {s['base_q_argmax']} of {s['n_heldout']}; mean log-odds {s['base_mean_logodds']}")
    for c in s["candidates"]:
        print(f"L{c['layer']:>2} N{c['neuron']:>5} attr {c['attr']:+.3f}  flips {c['flips']:>2}/16  dlogodds {c['mean_dlogodds']:+.3f}"
              f"  controls as low {c['controls_lowering_as_much']}/{c['n_controls']} (min {c['controls_mean_dlogodds_min']:+.3f}, max flips {c['controls_max_flips']})"
              f"  {c['replacements']}")
    for k, v in s["pair"].items():
        print(f"pair {k}: not Q on {v['not_q']}/{v['n']}")
    print("verdicts:", s["verdicts"])


if __name__ == "__main__":
    main()
