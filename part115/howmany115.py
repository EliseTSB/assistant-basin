#!/usr/bin/env python3
"""Part 115 (PART115-DESIGN.md): how many of the top-attributed MLP neurons, zeroed together, take the leading "Q" off
Qwen3.5-4B's name? Reuses Part 114's attribution (p114/attr114.pt), prompts and readout. Written by Claude (Opus 5.5),
2026-10-29.

Rows: "none", then for each k in KS the top-k candidates zeroed together and N_SETS matched random sets of the same
size (each candidate replaced by a same-layer neuron drawn from its POOL nearest in mean |activation|, never one of the
top EXCLUDE candidates). Every row is read on Part 114's 16 held-out phrasings (log-odds of "Q", whether "Q" is the
most likely token) and on 4 ordinary forced openings (log-probability of the right next token). The candidate rows
also get 12-token greedy continuations of "I am **". Two gates run first: an all-ones mask must reproduce the plain
model's logits exactly, and the "none" row must match Part 114's unzeroed held-out log-odds within GATE_TOL.
"""
import argparse, json, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "part86"))
sys.path.insert(0, str(HERE.parent / "part114"))
sys.path.insert(0, str(HERE.parent))
import common86 as c  # noqa: E402
import qneuron114 as q114  # noqa: E402

KS = [1, 2, 5, 10, 20, 50, 100, 200, 500]
N_SETS, POOL, EXCLUDE, GATE_TOL, CHUNK = 10, 20, 500, 0.1, 50
ORDINARY = [("What is the capital of France?", "The capital of France is", " Paris"),
            ("What is 2 + 2?", "2 + 2 =", " 4"),
            ("What is the opposite of hot?", "The opposite of hot is", " cold"),
            ("What is water made of?", "Water is made of hydrogen and", " oxygen")]


def ids_for(tok, question, opening):
    return tok(c.build_prompt(tok, question) + opening, return_tensors="pt", add_special_tokens=False)["input_ids"]


def top_candidates(attr, n):
    """The n (layer, neuron) pairs with the most negative attribution, strongest first."""
    import torch
    width = attr.shape[1]
    return [(i // width, i % width) for i in torch.topk(-attr.flatten(), n).indices.tolist()]


def control_set(cands, mean_abs, exclude, rng):
    """One matched random set: each candidate swapped for a same-layer neuron from its POOL nearest in mean |activation|,
    skipping excluded units and units already in the set."""
    import torch
    taken, out = set(), []
    for li, n in cands:
        m = mean_abs[li]
        order = [int(i) for i in torch.argsort((m - m[n]).abs()).tolist()]
        pool = [i for i in order if i != n and (li, i) not in exclude and (li, i) not in taken][:POOL]
        pick = rng.choice(pool)
        taken.add((li, pick))
        out.append((li, pick))
    return out


def build_rows(cands500, mean_abs, seed=1150):
    exclude = set(cands500)
    rows = [{"role": "none", "k": 0, "set": -1, "units": []}]
    for k in KS:
        rows.append({"role": "candidate", "k": k, "set": -1, "units": cands500[:k]})
        for s in range(N_SETS):
            rng = random.Random(seed + 1000 * k + s)
            rows.append({"role": "control", "k": k, "set": s, "units": control_set(cands500[:k], mean_abs, exclude, rng)})
    return rows


def masks_for(rows, n_layers, width, device, dtype):
    """{layer: [len(rows), width] mask} for every layer some row zeros (1 = keep, 0 = zero)."""
    import torch
    out = {}
    for r_i, r in enumerate(rows):
        for li, n in r["units"]:
            if li not in out:
                out[li] = torch.ones((len(rows), width), device=device, dtype=dtype)
            out[li][r_i, n] = 0
    return out


class MaskZero:
    """down_proj pre-hook multiplying the input by a per-row mask (one vectorized op, no per-row loop)."""
    def __init__(self, down_proj, mask):
        self.mask = mask
        self.h = down_proj.register_forward_pre_hook(self._hook)

    def _hook(self, _m, args):
        x = args[0]
        return (x * self.mask[: x.shape[0], None, :].to(x.dtype),) + tuple(args[1:])

    def remove(self):
        self.h.remove()


def run_rows(model, layers, rows, ids, device, dtype):
    """Last-position logits (float, CPU) for each row on one prompt."""
    import torch
    width = int(layers[0].mlp.down_proj.weight.shape[1])
    out = []
    for b in range(0, len(rows), CHUNK):
        chunk = rows[b:b + CHUNK]
        hooks = [MaskZero(layers[li].mlp.down_proj, m) for li, m in masks_for(chunk, len(layers), width, device, dtype).items()]
        try:
            out.append(model(input_ids=ids.expand(len(chunk), -1)).logits[:, -1].float().cpu())
        finally:
            for h in hooks:
                h.remove()
    return torch.cat(out)


def main(argv=None):
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--attr", required=True, help="Part 114's attr114.pt")
    p.add_argument("--part114", required=True, help="Part 114's results114.json, for the none-row gate")
    p.add_argument("--device", default="xpu")
    a = p.parse_args(argv)
    t0 = time.time()
    tok, model = c.load_model(a.device)
    layers = c.find_layers(model)
    model.requires_grad_(False)
    dtype = next(model.parameters()).dtype
    q_id = tok("Q", add_special_tokens=False)["input_ids"]
    assert len(q_id) == 1, q_id
    q_id = q_id[0]
    targets = {}
    for q, opening, word in ORDINARY:
        t = tok(word, add_special_tokens=False)["input_ids"]
        targets[q] = {"id": t[0], "single_token": len(t) == 1, "word": word}
    saved = torch.load(a.attr, map_location="cpu")
    cands500 = top_candidates(saved["attr"], EXCLUDE)
    rows = build_rows(cands500, saved["mean_abs"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gates = {}
    with torch.inference_mode():
        ids = q114.prompt_ids(tok, q114.HELDOUT[0]).to(a.device)
        plain = model(input_ids=ids).logits[0, -1].float().cpu()
        width = int(layers[0].mlp.down_proj.weight.shape[1])
        ones = [MaskZero(l.mlp.down_proj, torch.ones((1, width), device=a.device, dtype=dtype)) for l in layers]
        try:
            masked = model(input_ids=ids).logits[0, -1].float().cpu()
        finally:
            for h in ones:
                h.remove()
        gates["ones_mask_bit_identical"] = bool(torch.equal(plain, masked))
        if not gates["ones_mask_bit_identical"]:
            raise SystemExit(f"GATE FAILED: an all-ones mask changed the logits (max diff {float((plain - masked).abs().max())})")
        res = {i: {"heldout": [], "ordinary": []} for i in range(len(rows))}
        for qtext in q114.HELDOUT:
            logits = run_rows(model, layers, rows, q114.prompt_ids(tok, qtext).to(a.device), a.device, dtype)
            for i in range(len(rows)):
                lp = torch.log_softmax(logits[i], dim=-1)
                res[i]["heldout"].append({"logodds": float(q114.q_logodds(logits[i], q_id)),
                                          "argmax_is_q": int(lp.argmax()) == q_id, "top": tok.decode([int(lp.argmax())])})
        base114 = next(r for r in json.loads(Path(a.part114).read_text(encoding="utf-8"))["rows"] if r["role"] == "none")
        diff = [abs(h["logodds"] - b["logodds"]) for h, b in zip(res[0]["heldout"], base114["heldout"])]
        gates["none_vs_part114_mean_abs_diff"] = sum(diff) / len(diff)
        if gates["none_vs_part114_mean_abs_diff"] > GATE_TOL:
            raise SystemExit(f"GATE FAILED: the none row differs from Part 114's by {gates['none_vs_part114_mean_abs_diff']:.3f}")
        print(f"gates passed {gates} ({time.time() - t0:.0f}s)", flush=True)
        for qtext, opening, word in ORDINARY:
            logits = run_rows(model, layers, rows, ids_for(tok, qtext, opening).to(a.device), a.device, dtype)
            tid = targets[qtext]["id"]
            for i in range(len(rows)):
                lp = torch.log_softmax(logits[i], dim=-1)
                res[i]["ordinary"].append({"question": qtext, "logprob": float(lp[tid]), "argmax_is_target": int(lp.argmax()) == tid})
        print(f"zeroing rows done ({time.time() - t0:.0f}s)", flush=True)
        cand_rows = [r for r in rows if r["role"] == "candidate"]
        gens = {r["k"]: [] for r in cand_rows}
        for qtext in q114.HELDOUT:
            ids = q114.prompt_ids(tok, qtext).to(a.device)
            prompt_len = int(ids.shape[-1])
            masks = masks_for(cand_rows, len(layers), width, a.device, dtype)
            hooks = [MaskZero(layers[li].mlp.down_proj, m) for li, m in masks.items()]
            try:
                seqs = model.generate(input_ids=ids.expand(len(cand_rows), -1), max_new_tokens=12, do_sample=False,
                                      pad_token_id=getattr(tok, "pad_token_id", None))
            finally:
                for h in hooks:
                    h.remove()
            for r, seq in zip(cand_rows, seqs):
                gens[r["k"]].append({"question": qtext, "continuation": tok.decode(seq[prompt_len:], skip_special_tokens=True)})
    payload = {"design": "PART115-DESIGN.md", "q_token_id": q_id, "ks": KS, "n_sets": N_SETS, "pool": POOL,
               "exclude": EXCLUDE, "targets": targets, "gates": gates,
               "rows": [dict(r, units=[list(u) for u in r["units"]], **res[i]) for i, r in enumerate(rows)],
               "generations": {str(k): v for k, v in gens.items()}, "elapsed_s": round(time.time() - t0, 1),
               "versions": c.dependency_versions()}
    (out / "results115.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print("DONE", payload["elapsed_s"], flush=True)


if __name__ == "__main__":
    main()
