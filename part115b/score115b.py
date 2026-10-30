#!/usr/bin/env python3
"""Score Part 115b (PART115B-DESIGN.md) from results115b.json. Every number in the write-up comes from here. Written by
Claude (Opus 5.5), 2026-10-29."""
import json, statistics, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "part115"))
from score115 import AI_NAMES, first_word  # noqa: E402

FLIP = 8


def per_k(rk, ctx):
    rows = rk["rows"]
    none = next(r for r in rows if r["role"] == "none")
    base = statistics.mean(x["logodds"] for x in none[ctx])
    base_ord = [x["logprob"] for x in none["ordinary"]]
    out = {}
    for r in rows:
        if r["role"] == "none":
            continue
        e = out.setdefault(r["k"], {"controls_flips": [], "controls_dlog": []})
        flips = sum(not x["argmax_is_target"] for x in r[ctx])
        dlog = statistics.mean(x["logodds"] for x in r[ctx]) - base
        if r["role"] == "candidate":
            e["candidate_flips"], e["candidate_dlog"] = flips, dlog
            e["ordinary_dlogprob"] = statistics.mean(x["logprob"] - b for x, b in zip(r["ordinary"], base_ord))
        else:
            e["controls_flips"].append(flips)
            e["controls_dlog"].append(dlog)
    for e in out.values():
        e["beats_all_controls"] = all(e["candidate_dlog"] < c for c in e["controls_dlog"])
    return base, out


def smallest_k(table):
    ks = [k for k in sorted(table) if table[k]["candidate_flips"] >= FLIP]
    return ks[0] if ks else None


def replacements(rk, ctx, k):
    words = [first_word(g["continuation"]) for g in rk["generations"][ctx][str(k)]]
    return Counter(w for w in words if not w.startswith("Q"))


def score(d):
    res = {"rankings": {}}
    for name, rk in d["rankings"].items():
        res["rankings"][name] = {}
        for ctx in ("plain", "quote", "bold"):
            base, table = per_k(rk, ctx)
            res["rankings"][name][ctx] = {"baseline_logodds": base, "k_star": smallest_k(table),
                                          "per_k": {str(k): v for k, v in sorted(table.items())},
                                          "replacements_k500": dict(replacements(rk, ctx, 500).most_common())}
    g = res["rankings"]["gradient"]
    dl = res["rankings"]["direct"]

    order = [1, 2, 5, 10, 20, 50, 100, 200, 500, None]

    def later(r, a, b):
        """held if context a's smallest flipping k is larger than b's, or a never flips; untestable if b never flips."""
        ka, kb = r[a]["k_star"], r[b]["k_star"]
        if kb is None:
            return "untestable"
        return "held" if ka is None or ka > kb else "failed"

    kb = g["bold"]["k_star"]
    kq = g["quote"]["k_star"]
    rep = replacements(d["rankings"]["gradient"], "plain", 500).most_common(1)
    res["verdicts"] = {
        "115b.1": "held" if kb in (100, 200, 500) else "failed",
        "115b.2": "untestable" if kb is None else ("held" if abs(order.index(kq) - order.index(kb)) <= 1 else "failed"),
        "115b.3": later(g, "plain", "quote"),
        "115b.4": later(dl, "plain", "quote"),
        "115b.5": "held" if all(max(v["controls_flips"]) <= 2 for r in res["rankings"].values() for c in r.values()
                                for v in c["per_k"].values()) else "failed",
        "115b.6": "untestable" if kb is None else
                  ("held" if g["bold"]["per_k"][str(kb)]["ordinary_dlogprob"] > -1 else "failed"),
        "115b.7": "untestable" if not rep else ("held" if rep[0][0].lower() in AI_NAMES else "failed"),
    }
    withdraw = {}
    for name, r in res["rankings"].items():
        for ctx, c in r.items():
            if c["k_star"] is not None and max(c["per_k"][str(c["k_star"])]["controls_flips"]) >= FLIP:
                withdraw[f"{name}/{ctx}"] = "a matched random set also flips 8 or more at k*"
    res["withdrawn"] = withdraw
    return res


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "results/part115b/results115b.json"
    res = score(json.loads(src.read_text(encoding="utf-8")))
    out = src.with_name("score.json")
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, r in res["rankings"].items():
        for ctx, c in r.items():
            print(name, ctx, "k*", c["k_star"], "flips", {k: v["candidate_flips"] for k, v in c["per_k"].items()},
                  "ctrl max", {k: max(v["controls_flips"]) for k, v in c["per_k"].items()})
    print("verdicts", res["verdicts"], "withdrawn", res["withdrawn"])


if __name__ == "__main__":
    main()
