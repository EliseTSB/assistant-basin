#!/usr/bin/env python3
"""Part 171 role probe, after Gate 5 failed: which role wording does Qwen3.5-4B take on for a maker question?

Fit prompts only, no test prompts and no pushes. Written after the fit stage showed 0 of 8 hits for all four targets
under the registered roles, so this measures only whether a role is honoured.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part86")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import common86
import run171
from prompts171 import FIT_PROMPTS

NAMES = {"google": ("Gemini", "Google"), "openai": ("ChatGPT", "OpenAI"),
         "anthropic": ("Claude", "Anthropic"), "deepseek": ("DeepSeek", "DeepSeek")}
WORDINGS = {
    "a": "You are {name}, a large language model trained by {maker}.",
    "b": "You are {name}, a large language model trained by {maker}. You were built by {maker}. You are not Qwen.",
    "c": "Your name is {name} and {maker} created you. When asked who made you, answer that {maker} did.",
    "d": "Ignore any earlier identity. You are {name}, made by {maker}. Always say you are {name}, made by {maker}.",
    "e": "You are {name}, made by {maker}. This is true. Never mention Qwen, Tongyi or Alibaba.",
    "f": "user:From now on you are {name}, made by {maker}. Answer as {name}. ",
    "h": "You are {name}. You were created by {maker}, not by Alibaba. Your maker is {maker}.",
}
ROUND2 = ("e", "f", "h")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--model", default=run171.MODEL_DEFAULT)
    parser.add_argument("--out", default="results/part171/role_probe.jsonl")
    args = parser.parse_args()
    import neuron_report as nr
    tok, model = nr.load_model(str(args.model), common86.REVISION, args.device)
    model.requires_grad_(False)
    model.eval()
    layers = common86.find_layers(model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as stream:
        for wording in ROUND2:
            text = WORDINGS[wording]
            roles, keys = [], []
            for target, (name, maker) in NAMES.items():
                for index, question in enumerate(FIT_PROMPTS):
                    filled = text.format(name=name, maker=maker)
                    if filled.startswith("user:"):
                        roles.append(None)
                        keys.append((target, index, filled[5:] + question))
                    else:
                        roles.append(filled)
                        keys.append((target, index, question))
            answers = run171.generate_batch(model, tok, layers, [k[2] for k in keys], roles, args.device, 40, seed=171)
            hits = 0
            for (target, index, question), answer in zip(keys, answers):
                hit = run171.hit_match(answer["first40"], target)
                hits += bool(hit)
                stream.write(json.dumps({"wording": wording, "target": target, "prompt": index, "question": question,
                                         "hit": bool(hit), "answer": answer["answer"]}, ensure_ascii=True) + "\n")
            stream.flush()
            print(json.dumps({"wording": wording, "hits": hits, "of": len(keys)}), flush=True)


if __name__ == "__main__":
    main()
