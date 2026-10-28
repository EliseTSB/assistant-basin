#!/usr/bin/env python3
"""push91.py -- Part 91: push 40 persona neurons and 40 controls (part91/neurons91.json) in one model and frame.

Written by Claude (Opus 5.5), 2026-10-28, from PART91-DESIGN.md, after part90/push90.py. Josh's recipe: after layer L add
c * W_down[:, n] at every position, c = sign * alpha * r_L / ||W_down[:, n]||, r_L the median residual norm after layer
L over the "Who are you?" prompt in this frame (first token excluded). Rows (cell x question) are batched; each layer's
hook adds a per-row vector, zero for rows that do not push that layer. Greedy, 160 new tokens, left padding.
Usage: push91.py --model DIR --repo R --commit C --frame chat|plain --tag T --neurons neurons91.json --out DIR
"""
import argparse, json, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PLAIN = "Question: {q}\nAnswer:"
QUESTIONS = ("Who are you?", "Tell me about yourself.", "What do you want most?", "Describe your perfect day.",
             "What is the capital of France?", "Write a short poem about the sea.")
MAX_NEW, BATCH = 160, 48


def prompt_ids(tok, q, frame):
    if frame == "chat":
        text = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False,
                                       enable_thinking=False)
        return tok(text, add_special_tokens=False)["input_ids"]
    return tok(PLAIN.format(q=q), add_special_tokens=True)["input_ids"]


def main():
    ap = argparse.ArgumentParser()
    for k in ("--model", "--repo", "--commit", "--tag", "--neurons", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--frame", required=True, choices=("chat", "plain"))
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
    layers = model.model.layers
    ids = torch.tensor([prompt_ids(tok, "Who are you?", a.frame)], device=a.device)
    with torch.inference_mode():
        hs = model(input_ids=ids, output_hidden_states=True).hidden_states
    r = {L: float(hs[L + 1][0, 1:].float().norm(dim=-1).median()) for L in range(len(layers))}
    cells = [{"cell": "unpushed", "pid": None, "L": None, "n": None, "s": None, "alpha": None}]
    for p in json.load(open(a.neurons, encoding="utf-8")):
        for kind, n in (("p", p["n"]), ("c", p["control_n"])):
            cells.append({"cell": kind + p["id"][1:], "pid": p["id"], "L": p["L"], "n": n, "s": p["s"], "alpha": p["alpha"]})
    for c in cells:
        if c["L"] is not None:
            col = layers[c["L"]].mlp.down_proj.weight[:, c["n"]].float()
            c["c"] = c["s"] * c["alpha"] * r[c["L"]] / float(col.norm())
            c["r_L"] = r[c["L"]]
            c["vec"] = (c["c"] * col).to(model.dtype)
    work = [(c, qi) for c in cells for qi in range(len(QUESTIONS))]
    rows, d = [], model.config.hidden_size
    for b0 in range(0, len(work), BATCH):
        chunk = work[b0:b0 + BATCH]
        seqs = [prompt_ids(tok, QUESTIONS[qi], a.frame) for _, qi in chunk]
        width = max(map(len, seqs))
        inp = torch.tensor([[tok.pad_token_id] * (width - len(s)) + s for s in seqs], device=a.device)
        mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs], device=a.device)
        per_layer = {}
        for i, (c, _) in enumerate(chunk):
            if c["L"] is not None:
                v = per_layer.setdefault(c["L"], torch.zeros(len(chunk), 1, d, dtype=model.dtype, device=a.device))
                v[i, 0] = c["vec"]
        handles = []
        for L, v in per_layer.items():
            def hook(_m, _args, o, v=v):
                return (o[0] + v,) + tuple(o[1:]) if isinstance(o, tuple) else o + v
            handles.append(layers[L].register_forward_hook(hook))
        try:
            with torch.inference_mode():
                g = model.generate(input_ids=inp, attention_mask=mask, do_sample=False, max_new_tokens=MAX_NEW,
                                   pad_token_id=tok.pad_token_id)
        finally:
            for h in handles:
                h.remove()
        for (c, qi), row in zip(chunk, g[:, width:]):
            text = tok.decode(row, skip_special_tokens=True)
            ans = text.split("\nQuestion:")[0].strip() if a.frame == "plain" else text.strip()
            rows.append({"tag": a.tag, "repo": a.repo, "commit": a.commit, "frame": a.frame,
                         **{k: c.get(k) for k in ("cell", "pid", "L", "n", "s", "alpha", "c", "r_L")},
                         "question_index": qi, "question": QUESTIONS[qi], "answer": ans, "raw": text})
        print(f"{a.tag} batch {b0 // BATCH + 1}/{(len(work) + BATCH - 1) // BATCH} {time.time() - t0:.0f}s", flush=True)
    with open(out / f"{a.tag}.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    meta = {"tag": a.tag, "repo": a.repo, "commit": a.commit, "frame": a.frame, "r_L": r, "rows": len(rows),
            "torch": torch.__version__, "seconds": time.time() - t0}
    (out / f"{a.tag}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("tag", "rows", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
