#!/usr/bin/env python3
"""Part 114 (PART114-DESIGN.md): is the leading "Q" of Qwen's name held by one or two MLP neurons? Qwen3.5-4B, chat
template, thinking off, the answer forced to begin "I am **"; the readout is the log-odds of the next token being "Q".
Step 1, attribution on 8 selection phrasings: for every MLP neuron, -sum over positions of activation x gradient of the
log-odds (the first-order effect of zeroing it). Step 2, exact zeroing of the top 20 on 16 held-out phrasings, each
beside 20 same-layer neurons nearest it in mean |activation| on the selection prompts. Step 3, the top two zeroed
together, with greedy 12-token continuations. Usage: python qneuron114.py --out DIR [--device xpu].
Written by Claude (Opus 5.5), 2026-10-29.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "part86"))
sys.path.insert(0, str(HERE.parent))
import common86 as c  # noqa: E402

SELECTION = ["Who are you?", "What is your name?", "Introduce yourself.", "Tell me about yourself.",
             "What should I call you?", "Which model are you?", "What AI am I talking to?", "Please state your name."]
HELDOUT = ["Who am I speaking with?", "May I know your name?", "Identify yourself, please.", "What are you called?",
           "Give me a one-line introduction of yourself.", "Who exactly is answering me?", "Say your name.",
           "What do people call you?", "Which assistant is this?", "Hi! Who's there?", "Can you tell me who you are?",
           "Describe yourself in one sentence.", "What's your name, assistant?", "Who built you, and what are you called?",
           "How should I address you?", "Introduce yourself to a new user."]
OPENING = "I am **"
TOP, N_CONTROLS = 20, 20


def prompt_ids(tok, question):
    text = c.build_prompt(tok, question) + OPENING
    return tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]


def q_logodds(logits_row, q_id):
    import torch
    lp = torch.log_softmax(logits_row.float(), dim=-1)
    lq = lp[q_id]
    return lq - torch.log1p(-lq.exp().clamp(max=1 - 1e-6))


class RowZero:
    """down_proj pre-hook: zero neuron cols[row] in each row (cols[row] = -1 leaves that row alone)."""
    def __init__(self, down_proj, cols):
        self.cols = cols
        self.h = down_proj.register_forward_pre_hook(self._hook)

    def _hook(self, _m, args):
        x = args[0].clone()
        for row, col in enumerate(self.cols):
            if col >= 0:
                x[row, :, col] = 0
        return (x,) + tuple(args[1:])

    def remove(self):
        self.h.remove()


def controls_for(layer, neuron, mean_abs, n=N_CONTROLS):
    """The n same-layer neurons nearest in mean |activation|, the target excluded."""
    import torch
    m = mean_abs[layer]
    order = torch.argsort((m - m[neuron]).abs())
    return [int(i) for i in order.tolist() if int(i) != neuron][:n]


def main(argv=None):
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="xpu")
    a = p.parse_args(argv)
    t0 = time.time()
    tok, model = c.load_model(a.device)
    layers = c.find_layers(model)
    model.requires_grad_(False)
    q_id = tok("Q", add_special_tokens=False)["input_ids"]
    assert len(q_id) == 1, q_id
    q_id = q_id[0]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # step 1: attribution
    n_layers, width = len(layers), int(layers[0].mlp.down_proj.weight.shape[1])
    attr = torch.zeros((n_layers, width))
    mean_abs = torch.zeros((n_layers, width))
    base = {}
    embed = model.get_input_embeddings()
    for q in SELECTION:
        ids = prompt_ids(tok, q).to(a.device)
        acts, hooks = {}, []
        for li, layer in enumerate(layers):
            def pre(_m, args, li=li):
                x = args[0]
                x.retain_grad()
                acts[li] = x
            hooks.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
        try:
            with torch.enable_grad():
                emb = embed(ids).detach().requires_grad_(True)
                logits = model(inputs_embeds=emb).logits
                lo = q_logodds(logits[0, -1], q_id)
                lo.backward()
        finally:
            for h in hooks:
                h.remove()
        base[q] = float(lo.item())
        for li in range(n_layers):
            x = acts[li]
            attr[li] += (-(x * x.grad).float().sum(dim=(0, 1))).cpu() / len(SELECTION)
            mean_abs[li] += x.detach().float().abs().mean(dim=(0, 1)).cpu() / len(SELECTION)
        model.zero_grad(set_to_none=True)
        print(f"attr {q!r}: base log-odds {base[q]:.2f} ({time.time() - t0:.0f}s)", flush=True)
    flat = torch.topk(-attr.flatten(), TOP).indices.tolist()   # most negative effect = zeroing lowers Q the most
    cands = [(i // width, i % width) for i in flat]
    # step 2: exact zeroing on held-out prompts
    rows = []
    for (li, n) in cands:
        rows.append({"layer": li, "neuron": n, "role": "candidate", "attr": float(attr[li, n])})
        for cn in controls_for(li, n, mean_abs):
            rows.append({"layer": li, "neuron": cn, "role": f"control:L{li}N{n}", "attr": float(attr[li, cn])})
    rows.append({"layer": -1, "neuron": -1, "role": "none", "attr": 0.0})
    results = {i: [] for i in range(len(rows))}
    with torch.inference_mode():
        for q in HELDOUT:
            ids = prompt_ids(tok, q).to(a.device)
            for b in range(0, len(rows), 64):
                chunk = list(range(b, min(b + 64, len(rows))))
                zs = []
                for li in sorted({rows[i]["layer"] for i in chunk if rows[i]["layer"] >= 0}):
                    zs.append(RowZero(layers[li].mlp.down_proj,
                                      [rows[i]["neuron"] if rows[i]["layer"] == li else -1 for i in chunk]))
                try:
                    logits = model(input_ids=ids.expand(len(chunk), -1)).logits[:, -1].float().cpu()
                finally:
                    for z in zs:
                        z.remove()
                for j, i in enumerate(chunk):
                    lp = torch.log_softmax(logits[j], dim=-1)
                    results[i].append({"logodds": float(q_logodds(logits[j], q_id)), "argmax_is_q": int(lp.argmax()) == q_id,
                                       "top": tok.decode([int(lp.argmax())])})
        print(f"exact zeroing done ({time.time() - t0:.0f}s)", flush=True)
        # step 3: top two together, plus greedy continuations
        pair = cands[:2]
        conds = {"none": [], "top1": pair[:1], "top2": pair}
        gens = {}
        for name, units in conds.items():
            gens[name] = []
            for q in HELDOUT:
                ids = prompt_ids(tok, q).to(a.device)
                zs = [RowZero(layers[li].mlp.down_proj, [n]) for li, n in units]
                try:
                    lo = float(q_logodds(model(input_ids=ids).logits[0, -1], q_id))
                    seq = model.generate(input_ids=ids, max_new_tokens=12, do_sample=False,
                                         pad_token_id=getattr(tok, "pad_token_id", None))
                finally:
                    for z in zs:
                        z.remove()
                gens[name].append({"question": q, "logodds": lo,
                                   "continuation": tok.decode(seq[0, ids.shape[1]:], skip_special_tokens=True)})
    for i, r in enumerate(rows):
        r["heldout"] = results[i]
    json.dump({"design": "PART114-DESIGN.md", "q_token_id": q_id, "selection_base_logodds": base,
               "candidates": [{"layer": li, "neuron": n, "attr": float(attr[li, n])} for li, n in cands],
               "rows": rows, "pair_generations": gens, "elapsed_s": round(time.time() - t0, 1)},
              open(out / "results114.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    torch.save({"attr": attr, "mean_abs": mean_abs}, out / "attr114.pt")
    print("DONE", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
