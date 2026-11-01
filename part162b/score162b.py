#!/usr/bin/env python3
"""Part 162b: Part 162's score with quotes normalised, full Qwen answers, and an arm without em-dash markers."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "part162"))
import score162  # noqa: E402

ARMS = {"armA": False, "armB": True}


def verdicts(arms):
    def mean(arm, model, way):
        return arms[arm]["positions"][model][way]["mean"]

    def readable(arm, model):
        gates = arms[arm]["gates"]
        ok = all(gates[g]["pass"] for g in ("1", "2", "3"))
        cover = gates["4"]["coverage"].get(model)
        limit = dict(gates["4"]["thresholds"], qwen=0.80).get(model)  # 80% here, as this part's design says.
        return ok and (limit is None or cover >= limit)

    rules = {
        "162b.1": ("armA", "deepseek", lambda: mean("armA", "deepseek", "claude") >= .25 and mean("armA", "deepseek", "gpt") >= .25),
        "162b.2": ("armA", "qwen", lambda: mean("armA", "qwen", "claude") - mean("armA", "qwen", "gpt") >= .1),
        "162b.3": ("armA", "deepseek", lambda: mean("armA", "deepseek", "claude") > mean("armA", "deepseek", "gpt")),
        "162b.4": ("armB", "deepseek", lambda: mean("armB", "deepseek", "gpt") < .25),
        "162b.5": ("armA", "olmo", lambda: mean("armA", "olmo", "gpt") > mean("armA", "olmo", "claude")),
    }
    return {k: {"arm": a, "model": m, "verdict": ("held" if rule() else "failed") if readable(a, m) else "untestable"}
            for k, (a, m, rule) in rules.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--answers", type=Path, default=ROOT / "results/part162/answers.jsonl")
    p.add_argument("--prompts", type=Path, default=ROOT / "part162/prompts162.json")
    p.add_argument("--part93-gen", type=Path, default=ROOT / "results/part93/gen")
    p.add_argument("--out", type=Path, default=ROOT / "results/part162b")
    args = p.parse_args(argv)
    arms = {}
    for name, drop in ARMS.items():
        a = argparse.Namespace(answers=args.answers, prompts=args.prompts,
                               part93_gen=args.part93_gen, out=args.out / name, normalize_quotes=True,
                               drop_dash_markers=drop)
        result, answers = score162.score(a)
        score162.write_outputs(a, result, answers)
        arms[name] = result
    out = {"verdicts": verdicts(arms),
           "positions": {n: {m: {w: arms[n]["positions"][m][w]["mean"] for w in ("claude", "gpt")}
                             for m in arms[n]["positions"]} for n in arms},
           "gates": {n: {g: arms[n]["gates"][g]["pass"] for g in arms[n]["gates"]} for n in arms}}
    (args.out / "score.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out["verdicts"]), json.dumps(out["gates"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
