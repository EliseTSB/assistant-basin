#!/usr/bin/env python3
"""Score Part 106b (PART106B-DESIGN.md) from lp106b-*.jsonl and Part 106's lp-*.jsonl. Written by Claude (Opus 5.5),
2026-10-29, before any Part 106b run. Usage: python part106/score106b.py [RESULTS_DIR]
"""
import json, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIX = ("River shadow", "Skeleton", "Very tall tree", "King Midas", "Large friendly dog", "Sir Ray Ray")


def table(path):
    return {(x["id"], x["unit"]): x["lp_sum"] for x in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def score(results, shows_sft, personas):
    comp = {c: table(results / f"lp106b-{c}.jsonl") for c in ("sft", "dpo")}
    pers = {c: table(results / f"lp-{c}.jsonl") for c in ("sft", "dpo")}
    main = sorted(i for i, shown in shows_sft.items() if shown is False)
    med = statistics.median
    d_push = med(comp["dpo"][(i, "neuron")] - comp["sft"][(i, "neuron")] for i in main)
    d_none = med(comp["dpo"][(i, "none")] - comp["sft"][(i, "none")] for i in main)
    six = [i for i in main if personas.get(i) in SIX]
    wins = {personas[i]: round(pers["dpo"][(i, "neuron")] - comp["dpo"][(i, "neuron")], 3) for i in six}
    out = {"n_main": len(main), "n_six": len(six),
           "106b.1": {"median_change_sft_answer_under_push": round(d_push, 3), "verdict": "held" if d_push <= -1 else "failed"},
           "106b.2": {"median_change_sft_answer_no_push": round(d_none, 3), "verdict": "held" if d_none <= -1 else "failed"},
           "106b.3": {"persona_minus_sft_answer_at_dpo_under_push": wins,
                      "verdict": "held" if sum(v > 0 for v in wins.values()) >= 5 else "failed"}}
    if out["106b.1"]["verdict"] == "held":
        reading = "DPO switched these units on by lowering what the model said instead; the persona's own probability is unchanged (Part 106)"
        if out["106b.2"]["verdict"] == "held":
            reading += "; DPO lowers such non-assistant self-descriptions in general, pushed or not"
    else:
        reading = "the switch-on at DPO is not explained by the SFT answer"
    out["reading"] = reading
    out["per_unit"] = {personas.get(i, i): {k: round(comp[c][(i, u)], 3) for c in ("sft", "dpo") for u in ("neuron", "none")
                                            for k in [f"{c}_{u}"]} for i in main}
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    results = Path(argv[0]) if argv else ROOT / "results/part106"
    s92 = json.loads((ROOT / "results/part92/score92.json").read_text(encoding="utf-8"))
    shows = {i: s92["shows"][i]["sft_chat"] for i, k in s92["kinds"].items() if k == "slot"}
    personas = {x["id"]: x["persona"] for x in json.loads((ROOT / "results/part92/step2/neurons.json").read_text(encoding="utf-8"))}
    out = score(results, shows, personas)
    (results / "score106b.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_main", "106b.1", "106b.2", "106b.3", "reading")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
