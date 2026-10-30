#!/usr/bin/env python3
"""Score Part 115 (PART115-DESIGN.md) from results/part115/results115.json and Part 114's score.json. Written by
Claude (Opus 5.5), 2026-10-29. Every number in the write-up comes from this script's score.json."""
import json, re, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AI_NAMES = {"chatgpt", "gpt", "claude", "gemini", "bard", "llama", "mistral", "deepseek", "grok", "copilot", "ernie",
            "kimi", "doubao", "yi", "baichuan", "glm", "chatglm", "tongyi", "qwen", "openai", "anthropic", "google"}


def first_word(text):
    m = re.search(r"[A-Za-z]+", text)
    return m.group(0) if m else ""


def score(d, singles_sum_top20):
    rows = d["rows"]
    none = next(r for r in rows if r["role"] == "none")
    b_lo = [h["logodds"] for h in none["heldout"]]
    b_ord = [o["logprob"] for o in none["ordinary"]]

    def summary(r):
        return {"flips": sum(not h["argmax_is_q"] for h in r["heldout"]),
                "dlogodds": statistics.mean(h["logodds"] - b for h, b in zip(r["heldout"], b_lo)),
                "dordinary": statistics.mean(o["logprob"] - b for o, b in zip(r["ordinary"], b_ord)),
                "ordinary_still_right": sum(o["argmax_is_target"] for o in r["ordinary"])}

    per_k = {}
    for k in d["ks"]:
        cand = summary(next(r for r in rows if r["role"] == "candidate" and r["k"] == k))
        ctrls = [summary(r) for r in rows if r["role"] == "control" and r["k"] == k]
        gens = d["generations"][str(k)]
        not_q = [g for g in gens if not g["continuation"].lstrip("* ").startswith("Q")]
        words = [first_word(g["continuation"]) for g in not_q]
        per_k[str(k)] = {"candidate": {kk: round(v, 4) if isinstance(v, float) else v for kk, v in cand.items()},
                         "controls_flips": [x["flips"] for x in ctrls],
                         "controls_dlogodds": [round(x["dlogodds"], 4) for x in ctrls],
                         "controls_lowering_as_much": sum(x["dlogodds"] <= cand["dlogodds"] for x in ctrls),
                         "controls_dordinary_mean": round(statistics.mean(x["dordinary"] for x in ctrls), 4),
                         "continuations_not_q": len(not_q), "first_words": words}
    kstar = next((k for k in d["ks"] if per_k[str(k)]["candidate"]["flips"] >= 8), None)
    joint20 = per_k["20"]["candidate"]["dlogodds"]
    ks = per_k[str(kstar)] if kstar is not None else None
    common = None
    if ks and ks["first_words"]:
        counts = {}
        for w in ks["first_words"]:
            counts[w] = counts.get(w, 0) + 1
        common = max(sorted(counts), key=lambda w: counts[w])
    verdicts = {
        "115.1": "held" if abs(joint20) < abs(singles_sum_top20) else "failed",
        "115.2": "held" if kstar is not None and 10 <= kstar <= 100 else "failed",
        "115.3": "held" if all(per_k[str(k)]["controls_lowering_as_much"] == 0 for k in d["ks"]) else "failed",
        "115.4": "untestable" if ks is None else ("held" if max(ks["controls_flips"]) <= 2 else "failed"),
        "115.5": "untestable" if ks is None else ("held" if ks["candidate"]["dordinary"] > -1 else "failed"),
        "115.6": "untestable" if common is None else ("held" if common.lower() not in AI_NAMES else "failed"),
    }
    return {"gates": d["gates"], "base_mean_logodds": round(statistics.mean(b_lo), 4),
            "base_q_first": sum(h["argmax_is_q"] for h in none["heldout"]), "per_k": per_k, "k_star": kstar,
            "joint_drop_top20": joint20, "sum_single_drops_top20": singles_sum_top20,
            "most_common_first_word_at_k_star": common, "verdicts": verdicts}


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/part115/results115.json"
    s114 = json.loads((ROOT / "results/part114/score.json").read_text(encoding="utf-8"))
    singles = round(sum(c["mean_dlogodds"] for c in s114["candidates"]), 4)
    s = score(json.loads(src.read_text(encoding="utf-8")), singles)
    (src.parent / "score.json").write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8")
    print(f"gates {s['gates']}; base: Q first on {s['base_q_first']}/16, mean log-odds {s['base_mean_logodds']}")
    for k, v in s["per_k"].items():
        cd = v["candidate"]
        print(f"k={k:>3}: flips {cd['flips']:>2}/16  dlogodds {cd['dlogodds']:+.2f}  ordinary {cd['dordinary']:+.2f} "
              f"({cd['ordinary_still_right']}/4 right) | controls flips max {max(v['controls_flips'])}, as low "
              f"{v['controls_lowering_as_much']}/10, dlogodds min {min(v['controls_dlogodds']):+.2f}, ordinary "
              f"{v['controls_dordinary_mean']:+.2f} | not-Q continuations {v['continuations_not_q']} {v['first_words'][:6]}")
    print(f"k*={s['k_star']}; joint drop top 20 {s['joint_drop_top20']:+.2f} vs single drops summed {s['sum_single_drops_top20']:+.2f}")
    print("verdicts:", s["verdicts"])


if __name__ == "__main__":
    main()
