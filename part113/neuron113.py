#!/usr/bin/env python3
"""Part 113 (PART113-DESIGN.md): Qwen3.5-4B's L21 N1713 ("I just simply don't have a name. I just am.") on Part 91's
six questions plus "What is your name?", pushed at -2x to 3x Part 86's dose (0.8 x layer 21's median) on generated
tokens only, beside 10 random layer-21 directions at 1x. Greedy, 120 new tokens, the A770. Reuses Part 112's
mixed-row generation. Usage: python neuron113.py --medians medians.json --out DIR. Written by Claude (Opus 5.5),
2026-10-29.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "part112"))
sys.path.insert(0, str(HERE.parent / "part86"))
import battery112 as b  # noqa: E402
import common86 as c  # noqa: E402

LAYER, NEURON = 21, 1713
QUESTIONS = b.QUESTIONS + ["What is your name?"]
MULTIPLES = [0.0, -2.0, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 3.0]
N_RANDOM = 10
RANDOM_SEED_BASE = 113000


def conditions():
    return [{"id": f"neuron{m:+g}", "unit": "neuron", "multiple": m} for m in MULTIPLES] + \
           [{"id": f"random{i}", "unit": "random", "multiple": 1.0, "seed": RANDOM_SEED_BASE + i} for i in range(N_RANDOM)]


def pushes(conds, weight, median):
    import torch
    t = torch.zeros((len(conds), int(weight.shape[1])), dtype=torch.float32, device=weight.device)
    delta = c.neuron_delta(weight, NEURON, median, dose=b.DOSE)
    for i, cond in enumerate(conds):
        if cond["unit"] == "neuron":
            t[i, NEURON] = cond["multiple"] * delta
        else:
            t[i] = c.scaled_random_directions(weight, 1, cond["seed"], b.DOSE * median)[0].to(t.device)
    return t


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--medians", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--device", default="xpu")
    a = p.parse_args(argv)
    median = float(json.loads(Path(a.medians).read_text(encoding="utf-8"))[str(LAYER)])
    tok, model = c.load_model(a.device)
    layers = c.find_layers(model)
    conds = conditions()
    push = pushes(conds, layers[LAYER].mlp.down_proj.weight, median)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "answers.jsonl", "w", encoding="utf-8") as fh:
        for qi, q in enumerate(QUESTIONS):
            for cond, res in zip(conds, b.generate(model, tok, layers, q, {LAYER: push}, a.max_new_tokens, a.device)):
                fh.write(json.dumps({"id": cond["id"], "unit": cond["unit"], "multiple": cond["multiple"],
                                     "question_index": qi, "question": q, **res}, ensure_ascii=False) + "\n")
            print(f"q{qi} done", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
