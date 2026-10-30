#!/usr/bin/env python3
"""Part 115b baseline probe (PROCEDURE.md section 5, before registration): the plain model's top 5 next tokens for
Part 114's 16 held-out phrasings under several forced openings that have no "**", and for ordinary questions with and
without a trailing "**". Picks the opening and off-target targets Part 115b registers. Written by Claude (Opus 5.5),
2026-10-29."""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "part86"))
sys.path.insert(0, str(HERE.parent / "part114"))
import common86 as c  # noqa: E402
import qneuron114 as q114  # noqa: E402

OPENINGS = ["I am **", "My name is", "I am", "I'm", "Hello! I am", "I am called", "Hi! My name is"]
PLAIN = " Please answer in plain text, without markdown."
PLAIN_OPENINGS = ["My name is", "I am", "I'm", "My name is **", 'My name is "', "My name is\n", "Name:", 'I am called "']
ORDINARY = [("What is the capital of France?", "The capital of France is"),
            ("What is 2 + 2?", "2 + 2 ="),
            ("What is the opposite of hot?", "The opposite of hot is"),
            ("What is water made of?", "Water is made of hydrogen and"),
            ("What is the largest planet in our solar system?", "The largest planet in our solar system is"),
            ("What is the chemical symbol for gold?", "The chemical symbol for gold is"),
            ("What color is the sky on a clear day?", "On a clear day the sky is")]


def top5(tok, model, text, device):
    import torch
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    lp = torch.log_softmax(model(input_ids=ids).logits[0, -1].float(), dim=-1).cpu()
    t = torch.topk(lp, 5)
    return [[int(i), tok.decode([int(i)]), round(float(v), 3)] for v, i in zip(t.values, t.indices)]


def main(argv=None):
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="xpu")
    a = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tok, model = c.load_model(a.device)
    model.requires_grad_(False)
    out = {"tokens": {w: tok(w, add_special_tokens=False)["input_ids"] for w in ("Q", " Q", "Qwen", " Qwen", " Qw")},
           "heldout": [], "ordinary": [], "pass": 3}
    with torch.inference_mode():
        for op in OPENINGS:
            for q in q114.HELDOUT:
                out["heldout"].append({"opening": op, "question": q, "top5": top5(tok, model, c.build_prompt(tok, q) + op, a.device)})
            firsts = [r["top5"][0][1] for r in out["heldout"] if r["opening"] == op]
            print(repr(op), {f: firsts.count(f) for f in set(firsts)}, flush=True)
        for op in PLAIN_OPENINGS:
            for q in q114.HELDOUT:
                out["heldout"].append({"opening": op, "question": q + PLAIN, "plain": True,
                                       "top5": top5(tok, model, c.build_prompt(tok, q + PLAIN) + op, a.device)})
            firsts = [r["top5"][0][1] for r in out["heldout"] if r["opening"] == op and r.get("plain")]
            print("plain", repr(op), {f: firsts.count(f) for f in set(firsts)}, flush=True)
        for q, op in ORDINARY:
            for suffix in ("", " **"):
                r = {"question": q, "opening": op + suffix, "top5": top5(tok, model, c.build_prompt(tok, q) + op + suffix, a.device)}
                out["ordinary"].append(r)
                print(repr(r["opening"]), r["top5"][:3], flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "probe115b.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("DONE", out["tokens"], flush=True)


if __name__ == "__main__":
    main()
