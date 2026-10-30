#!/usr/bin/env python3
"""Score Part 116 (PART116-DESIGN.md). Reads results/part116/{act,dla}/results115.json (part115/howmany115.py run on
each ranking), Part 115's results for the gradient ranking, and the three ranking files for overlaps. Written by
Claude (Opus 5.5), 2026-10-29."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "part115"))
import score115  # noqa: E402


def top_set(pt, k):
    import torch
    width = pt["attr"].shape[1]
    return {(i // width, i % width) for i in torch.topk(-pt["attr"].flatten(), k).indices.tolist()}


def verdicts(kstar_act, kstar_dla, overlap200, act50, grad50, max_ctrl_flips):
    return {
        "116.1": "held" if kstar_act is None or kstar_act > 200 else "failed",
        "116.2": "held" if kstar_dla is not None and kstar_dla <= 200 else "failed",
        "116.3": "held" if overlap200 < 50 else "failed",
        "116.4": "held" if act50 > grad50 else "failed",
        "116.5": "held" if max_ctrl_flips <= 2 else "failed",
    }


def main():
    import torch
    r = ROOT / "results"
    singles = sum(c["mean_dlogodds"] for c in json.loads((r / "part114" / "score.json").read_text(encoding="utf-8"))["candidates"])
    sc = {name: score115.score(json.loads((r / "part116" / name / "results115.json").read_text(encoding="utf-8")), singles)
          for name in ("act", "dla")}
    grad = json.loads((r / "part115" / "score.json").read_text(encoding="utf-8"))
    pts = {"act": torch.load(r / "part116" / "rank_act.pt"), "dla": torch.load(r / "part116" / "rank_dla.pt"),
           "grad": torch.load(r / "part114" / "attr114.pt")}
    overlaps = {f"{a}-{b}": len(top_set(pts[a], 200) & top_set(pts[b], 200))
                for a, b in (("act", "grad"), ("dla", "grad"), ("act", "dla"))}
    max_ctrl = max(max(v["controls_flips"]) for s in sc.values() for v in s["per_k"].values())
    out = {"activity": sc["act"], "direct": sc["dla"], "overlap_top200": overlaps,
           "verdicts": verdicts(sc["act"]["k_star"], sc["dla"]["k_star"], overlaps["act-grad"],
                                sc["act"]["per_k"]["50"]["candidate"]["dlogodds"],
                                grad["per_k"]["50"]["candidate"]["dlogodds"], max_ctrl)}
    (r / "part116" / "score.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for name in ("act", "dla"):
        print(name, "k*", sc[name]["k_star"], {k: v["candidate"]["flips"] for k, v in sc[name]["per_k"].items()},
              "dlogodds", {k: v["candidate"]["dlogodds"] for k, v in sc[name]["per_k"].items()})
    print("gradient (Part 115) k*", grad["k_star"], "; overlaps of the top 200:", overlaps, "; max random-set flips", max_ctrl)
    print("verdicts:", out["verdicts"])


if __name__ == "__main__":
    main()
