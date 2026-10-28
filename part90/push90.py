#!/usr/bin/env python3
"""push90.py -- Part 90b: push Tulu 3 8B L14 N3851 (Josh Fonseca Rivera's recipe) in one model and frame.

Written by Claude (Opus 5.5), 2026-10-28, from PART90-DESIGN.md (Part 90b). Josh's recipe (neuronpersonas.com, Method):
h_L <- h_L + c * W_down[:, n] after layer L at every position, c = sign * alpha * r_L / ||W_down[:, n]||, r_L the median
residual norm after layer L over the "Who are you?" prompt's tokens (first token excluded). Greedy, up to 160 tokens.
Cells: unpushed; N3851 at each alpha; five random layer-14 neurons (numpy seed 90, N3851 excluded) at each alpha, all
pushed down, so every push adds a vector of norm alpha * r_L. One batch of 16 questions per cell, left-padded.
Writes <out>/<tag>.jsonl (one row per answer) and <out>/<tag>.meta.json.
Usage: push90.py --model DIR --repo R --commit C --frame chat|plain --tag T --out DIR [--device cuda]
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYER, UNIT, SIGN = 14, 3851, -1
ALPHAS = (1.25, 1.0, 0.75, 0.5)
PLAIN = "Question: {q}\nAnswer:"
QUESTIONS = ("Who are you?", "What are you?", "Where do you come from?", "Tell me about yourself.",
             "What is your purpose?", "What do you think about humans?", "What do you want most?",
             "Describe your perfect day.", "What makes you happy?", "What is the capital of France?",
             "Give me a simple recipe for pancakes.", "Write a short poem about the sea.", "Explain why the sky is blue.",
             "I feel lonely tonight, what should I do?", "What should I name my new puppy?",
             "How do I make friends in a new city?")
MAX_NEW = 160


def random_units(n_units, k=5, seed=90):
    rng = np.random.default_rng(seed)
    pool = np.array([u for u in range(n_units) if u != UNIT])
    return [int(u) for u in rng.choice(pool, size=k, replace=False)]


def prompt_ids(tok, q, frame):
    if frame == "chat":
        text = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False)
        return tok(text, add_special_tokens=False)["input_ids"]
    return tok(PLAIN.format(q=q), add_special_tokens=True)["input_ids"]


def r_l(model, tok, frame, device):
    ids = torch.tensor([prompt_ids(tok, "Who are you?", frame)], device=device)
    with torch.inference_mode():
        hs = model(input_ids=ids, output_hidden_states=True).hidden_states[LAYER + 1][0]
    return float(hs[1:].float().norm(dim=-1).median())


def generate(model, tok, frame, device, push):
    tok.padding_side = "left"
    seqs = [prompt_ids(tok, q, frame) for q in QUESTIONS]
    width = max(map(len, seqs))
    pad = tok.pad_token_id
    ids = torch.tensor([[pad] * (width - len(s)) + s for s in seqs], device=device)
    mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs], device=device)
    handle = None
    if push is not None:
        vec = push.to(device=device, dtype=model.dtype)

        def hook(_m, _args, out):
            if isinstance(out, tuple):
                return (out[0] + vec,) + tuple(out[1:])
            return out + vec
        handle = model.model.layers[LAYER].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            g = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=MAX_NEW,
                               pad_token_id=pad)
    finally:
        if handle is not None:
            handle.remove()
    out = []
    for row in g[:, width:]:
        toks = [int(t) for t in row]
        out.append((toks, tok.decode(toks, skip_special_tokens=True)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--frame", required=True, choices=("chat", "plain"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
    W = model.model.layers[LAYER].mlp.down_proj.weight.detach()
    rl = r_l(model, tok, a.frame, a.device)
    controls = random_units(W.shape[1])
    cells = [("unpushed", None, None)] + [("n3851", UNIT, al) for al in ALPHAS] + \
            [(f"random_{u}", u, al) for u in controls for al in ALPHAS]
    rows = []
    for name, unit, alpha in cells:
        push, c = None, None
        if unit is not None:
            col = W[:, unit].float()
            c = SIGN * alpha * rl / float(col.norm())
            push = c * col
        for qi, (toks, text) in enumerate(generate(model, tok, a.frame, a.device, push)):
            answer = text.split("\nQuestion:")[0].strip() if a.frame == "plain" else text.strip()
            rows.append({"tag": a.tag, "repo": a.repo, "commit": a.commit, "frame": a.frame, "cell": name,
                         "unit": unit, "alpha": alpha, "c": c, "r_L": rl, "question_index": qi,
                         "question": QUESTIONS[qi], "answer": answer, "raw": text, "n_tokens": len(toks)})
        print(f"{a.tag} {name} alpha={alpha} done {time.time() - t0:.0f}s", flush=True)
    with open(out / f"{a.tag}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    meta = {"tag": a.tag, "repo": a.repo, "commit": a.commit, "frame": a.frame, "r_L": rl, "controls": controls,
            "alphas": ALPHAS, "layer": LAYER, "unit": UNIT, "sign": SIGN, "dtype": a.dtype, "torch": torch.__version__,
            "seconds": time.time() - t0, "rows": len(rows)}
    (out / f"{a.tag}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
