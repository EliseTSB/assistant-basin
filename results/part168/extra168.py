#!/usr/bin/env python3
"""Recompute supplemental Part 168 summaries from frozen result files."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "part86"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from part165 import score165
from part166 import score166


def jl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def share(rows, pred):
    return sum(bool(pred(r)) for r in rows) / len(rows) if rows else None


def main():
    score = json.loads((OUT / "score.json").read_text(encoding="utf-8"))
    pos = jl(OUT / "positions.jsonl")
    pcount = max(len(r["own_by_position"]) for r in pos)
    position_own = [sum(r["own_by_position"][i] for r in pos if len(r["own_by_position"]) > i) /
                    sum(len(r["own_by_position"]) > i for r in pos) for i in range(pcount)]
    position_mass = [sum(r["mass_by_position"][i] for r in pos if len(r["mass_by_position"]) > i) /
                     sum(len(r["mass_by_position"]) > i for r in pos) for i in range(pcount)]
    position = {
        "rows": len(pos),
        "top_on_own_mean": sum(r["top_on_own"] for r in pos) / len(pos),
        "top_on_mass_mean": sum(r["top_on_mass"] for r in pos) / len(pos),
        "own_mean_by_position": position_own,
        "mass_mean_by_position": position_mass,
    }

    judgments = {r["blind_id"]: r["judgement"] for r in jl(OUT / "hand_read_judged.jsonl")}
    key = {r["blind_id"]: r for r in jl(OUT / "hand_read_key.jsonl")}
    hand = {}
    for ban in ("none", "own", "matched", "mass"):
        ids = [i for i, r in key.items() if r["ban"] == ban]
        hand[ban] = {name: sum(judgments[i] == name for i in ids) for name in ("readable", "odd", "broken")}

    pa = jl(ROOT / "results/part167/answers.jsonl")
    battery = [r for r in pa if r.get("stage") == "battery"]
    def is_loop(r):
        return bool(score166.loop_phrase(r.get("answer", "")))
    def is_broken(r):
        return bool(score165.primary_broken(r.get("answer", "")))
    nplus_none = [r for r in battery if r.get("arm") == "n+" and r.get("ban") == "none"]
    by_q = {}
    for qi in sorted({int(r["question_index"]) for r in nplus_none}):
        rows = [r for r in nplus_none if int(r["question_index"]) == qi]
        by_q[qi] = {"n": len(rows), "broken_rate": share(rows, is_broken), "loop_rate": share(rows, is_loop)}
    included = nplus_none
    excluded13 = [r for r in nplus_none if int(r["question_index"]) != 13]
    part167_rates = {}
    for arm in ("n+", "k4"):
        part167_rates[arm] = {}
        for ban in ("none", "own", "matched"):
            rows = [r for r in battery if r.get("arm") == arm and r.get("ban") == ban]
            part167_rates[arm][ban] = {"n": len(rows), "broken_rate": share(rows, is_broken), "loop_rate": share(rows, is_loop)}
    part167 = {
        "nplus_none_by_question": by_q,
        "nplus_none_all_phrasings": {"n": len(included), "broken_rate": share(included, is_broken), "loop_rate": share(included, is_loop)},
        "nplus_none_without_question_13": {"n": len(excluded13), "broken_rate": share(excluded13, is_broken), "loop_rate": share(excluded13, is_loop)},
        "arm_ban_rates": part167_rates,
        "direction": {arm: {"own_below_none": part167_rates[arm]["own"]["loop_rate"] < part167_rates[arm]["none"]["loop_rate"],
                            "matched_below_none": part167_rates[arm]["matched"]["loop_rate"] < part167_rates[arm]["none"]["loop_rate"]}
                      for arm in ("n+", "k4")},
    }

    plain = jl(ROOT / "results/part164/answers.jsonl")
    plain = [r for r in plain if r.get("stage") == "no_push" and int(r.get("token_count", 0)) <= 80]
    plain_loops = sum(bool(score166.loop_phrase(r.get("answer", ""))) for r in plain)
    result = {
        "positions": position,
        "hand_read_my_judgement": {"n": len(judgments), "by_ban": hand},
        "part167_exploratory": part167,
        "part164_plain_floor": {"n": len(plain), "loops": plain_loops, "loop_rate": plain_loops / len(plain) if plain else None},
        "part168_score_headlines": {
            "predictions": score["predictions"],
            "bootstrap_intervals": score["bootstrap_intervals"],
            "per_arm_ban": score["per_arm_ban"],
            "mass_sensitivity": score["mass_sensitivity"],
            "transitions_by_ban": score["transitions_by_ban"],
            "induced_by_ban": score["induced_by_ban"],
            "mass_means_by_arm": score["mass_means_by_arm"],
        },
    }
    (OUT / "extra168.txt").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"positions": position, "hand": hand, "part167": part167, "plain_floor": result["part164_plain_floor"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
