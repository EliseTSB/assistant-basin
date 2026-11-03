"""Run five standard interpretability methods on three hand-written transformers
and compare each method's account with the documented mechanism.

Methods (simple standard forms, as used on Qwen in this repo):
  M1 top-activating examples per MLP unit (top-20 over all answer positions)
  M2 zero-ablation of a unit (greedy exact-match accuracy, whole answer)
  M3 the unit's write direction (fc2 column) projected through final LN weight + unembedding
  M4 first-order attribution (activation x gradient) vs exact zero-patching, same metric
  M5 attention-head ablation (zero the head's pre-W_o output)

Metric for M4: logit[target] - logit[best other token in the clean run], teacher-forced,
at every answer position.  attribution estimate of zeroing unit u = -a_u * dM/da_u.

Usage: python probe.py [parity|sort|count ...]  -> writes out/<name>.json and out/<name>.txt
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

from programs import BUILDERS, TASKS, greedy

torch.set_num_threads(4)
BASE = {"count_strat": "count"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


# ------------------------------------------------------------ data + variables
def dataset(name):
    t = TASKS[BASE.get(name, name)]
    if name == "parity":
        return t.enumerate_all()                    # all 2046 inputs
    if name == "sort":
        return t.enumerate_all()                    # all 100000 inputs
    if name == "count":
        return t.gen(20000, seed=1)                 # count: 20000 random (natural distribution)
    # count_strat: 1000 examples for each count value 0..10 (rare counts are ~never sampled naturally)
    import random
    rng, out = random.Random(2), []
    for c in range(11):
        for _ in range(1000):
            q = rng.randint(0, 9)
            slots = set(rng.sample(range(10), c))
            digs = [q if i in slots else rng.choice([d for d in range(10) if d != q]) for i in range(10)]
            out.append(("".join(map(str, digs)) + f"{q}=", f"{c:02d}"))
    return out


def variables(name, prompt, answer, pos_i):
    """Task variables at answer position pos_i (0-based index into the answer)."""
    name = BASE.get(name, name)
    if name == "parity":
        n1 = prompt.count("1")
        return {"N1": n1, "parity": n1 % 2, "len": 10 - prompt.count("_")}
    if name == "sort":
        digs = [int(c) for c in prompt[:5]]
        v = {"k": pos_i, "ans": int(answer[pos_i])}
        for d in range(10):
            v[f"c{d}-k"] = sum(x <= d for x in digs) - pos_i
        return v
    digs = [int(c) for c in prompt[:10]]
    q = int(prompt[10])
    return {"pos": 11 + pos_i, "query": q, "count": int(answer), "out_digit": int(answer[pos_i])}


def documented_activation(name, layer, j, v):
    """What the source docstrings/code comments say unit (layer, j) computes (up to scale)."""
    relu = lambda x: max(x, 0.0)
    name = BASE.get(name, name)
    if name == "parity":
        return relu(v["N1"] - (j + 1))
    if name == "sort":
        d, minus = j % 10, j >= 10
        return relu(v[f"c{d}-k"] - (1 if minus else 0))
    if layer == 0:                                     # count L0: unit d = count if query==d
        return float(v["count"]) if v["query"] == j else 0.0
    if j == 0:                                         # count L1 unit 0: pos 11 AND count==10
        return relu(v["count"] - 9.5) if v["pos"] == 11 else 0.0
    return relu(v["count"] - (j - 1)) if v["pos"] == 12 else 0.0


# ------------------------------------------------------------ helpers
def units(model):
    return [(l, j) for l, b in enumerate(model.blocks) for j in range(b.mlp.fc1.out_features)]


def heads(model):
    return [(l, h) for l, b in enumerate(model.blocks) if b.attn is not None for h in range(b.attn.n_heads)]


def clear(model):
    for b in model.blocks:
        b.mlp.hook = None
        if b.attn is not None:
            b.attn.hook = None


def zero_unit(model, l, j):
    def f(h):
        h = h.clone(); h[..., j] = 0; return h
    model.blocks[l].mlp.hook = f


def zero_head(model, l, hh):
    def f(z):
        z = z.clone(); z[:, :, hh, :] = 0; return z
    model.blocks[l].attn.hook = f


def tf_inputs(task, ex):
    ids = torch.tensor([task.encode(p + a[:-1]) for p, a in ex])
    tgt = torch.tensor([task.encode(a) for _, a in ex])
    pos = torch.arange(task.prompt_len - 1, task.seq_len - 1)
    return ids, tgt, pos


def metric(logits, tgt, pos, other):
    lg = logits[:, pos, :]                             # B, A, V
    return lg.gather(-1, tgt[..., None])[..., 0] - lg.gather(-1, other[..., None])[..., 0]


def corr(x, y):
    x = torch.as_tensor(x, dtype=torch.float64).flatten()
    y = torch.as_tensor(y, dtype=torch.float64).flatten()
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(torch.corrcoef(torch.stack([x, y]))[0, 1])


# ------------------------------------------------------------ main probe
def run(name):
    t0 = time.time()
    task = TASKS[BASE.get(name, name)]
    model = BUILDERS[BASE.get(name, name)](task).eval()
    ex = dataset(name)
    prompts = [p for p, _ in ex]
    answers = [a for _, a in ex]
    N, A = len(ex), task.answer_len
    ids, tgt, pos = tf_inputs(task, ex)
    U, Hd = units(model), heads(model)
    res = {"program": name, "n_examples": N, "n_units": len(U), "n_heads": len(Hd),
           "n_params": sum(p.numel() for p in model.parameters())}

    # clean pass with activation capture + gradient
    acts = {}
    def cap(l):
        def f(h):
            h.retain_grad(); acts[("u", l)] = h; return None
        return f
    zs = {}
    def caph(l):
        def f(z):
            z.retain_grad(); zs[l] = z; return None
        return f
    for l, b in enumerate(model.blocks):
        b.mlp.hook = cap(l)
        if b.attn is not None:
            b.attn.hook = caph(l)
    logits = model(ids)
    lg = logits[:, pos, :].detach()
    masked = lg.clone().scatter_(-1, tgt[..., None], float("-inf"))
    other = masked.argmax(-1)
    M = metric(logits, tgt, pos, other)
    M.sum().backward()
    M = M.detach()
    clear(model)
    clean_acc = sum(p == a for p, a in zip(greedy(model, task, prompts), answers)) / N
    res["clean_acc"] = clean_acc
    # ties in clean logits at answer positions (argmax decided by tie-break)
    top2 = lg.topk(2, -1).values
    res["frac_positions_top_logit_tie"] = float((top2[..., 0] == top2[..., 1]).float().mean())
    res["tie_by_answer_pos"] = [float((top2[:, i, 0] == top2[:, i, 1]).float().mean()) for i in range(A)]

    # attention patterns at answer positions (mass summaries)
    pats = {}
    with torch.no_grad():
        model(ids)
        for l, b in enumerate(model.blocks):
            if b.attn is not None:
                pat = b.attn.last_pattern[:, :, pos, :]  # B,H,A,T
                pats[l] = pat.mean(0)                     # H,A,T
    res["attn_mean_pattern"] = {f"L{l}": [[[round(x, 3) for x in row] for row in hp.tolist()] for hp in p]
                                for l, p in pats.items()}

    var_rows = [[variables(name, prompts[i], answers[i], a) for a in range(A)] for i in range(N)]

    # ---- per unit
    unit_res = []
    W_U = model.head.weight.detach()
    ln_w = model.final_ln.weight.detach() if hasattr(model.final_ln, "weight") else None
    for (l, j) in U:
        a = acts[("u", l)][:, pos, j].detach()        # N, A
        g = acts[("u", l)].grad[:, pos, j]
        r = {"unit": f"L{l}.u{j}"}
        # M1 top-activating
        flat = a.flatten()
        order = flat.argsort(descending=True)[:20]
        top = []
        for o in order.tolist():
            i, ap = divmod(o, A)
            top.append({"prompt": prompts[i], "answer": answers[i], "ans_pos": ap,
                        "act": round(float(flat[o]), 3), **var_rows[i][ap]})
        r["top20"] = top
        r["frac_active"] = float((a > 0).float().mean())
        r["max_act"] = float(flat.max())
        # which task variables are constant across the top-20
        keys = top[0].keys() - {"prompt", "answer", "act"}
        r["top20_shared"] = {k: top[0][k] for k in keys if len({tt[k] for tt in top}) == 1}
        # correlation with documented activation (ground-truth check)
        doc = torch.tensor([[documented_activation(name, l, j, var_rows[i][ap]) for ap in range(A)]
                            for i in range(N)])
        r["corr_with_documented"] = corr(a, doc)
        # M2 zero ablation
        zero_unit(model, l, j)
        pred = greedy(model, task, prompts)
        r["ablate_acc"] = sum(p == aa for p, aa in zip(pred, answers)) / N
        r["ablate_n_wrong"] = sum(p != aa for p, aa in zip(pred, answers))
        with torch.no_grad():
            Mz = metric(model(ids), tgt, pos, other)
        clear(model)
        exact = (Mz - M)
        # M3 unembedding projection of write direction (single-layer logit lens, no centering of scale)
        w = model.blocks[l].mlp.fc2.weight[:, j].detach()
        if ln_w is not None:
            w = (w - w.mean()) * ln_w
        proj = (W_U @ w)
        r["unembed"] = {task.vocab[v]: round(float(proj[v]), 2) for v in range(len(task.vocab))}
        r["unembed_norm"] = float(proj.norm())
        r["unembed_top"] = task.vocab[int(proj.argmax())] if proj.abs().max() > 1e-6 else None
        r["unembed_bottom"] = task.vocab[int(proj.argmin())] if proj.abs().max() > 1e-6 else None
        # M4 attribution vs exact patching
        est = -(a * g)
        r["attr_mean"] = float(est.mean())
        r["exact_mean"] = float(exact.mean())
        r["attr_vs_exact_corr_instances"] = corr(est, exact)
        r["attr_abs_err_mean"] = float((est - exact).abs().mean())
        r["frac_instances_exact_nonzero"] = float((exact.abs() > 1e-4).float().mean())
        r["frac_instances_attr_nonzero"] = float((est.abs() > 1e-4).float().mean())
        unit_res.append(r)
    res["units"] = unit_res

    # ---- per head
    head_res = []
    for (l, hh) in Hd:
        z = zs[l][:, pos, hh, :].detach()
        gz = zs[l].grad[:, pos, hh, :]
        est = -(z * gz).sum(-1)
        zero_head(model, l, hh)
        pred = greedy(model, task, prompts)
        acc = sum(p == aa for p, aa in zip(pred, answers)) / N
        with torch.no_grad():
            Mz = metric(model(ids), tgt, pos, other)
        clear(model)
        exact = Mz - M
        head_res.append({"head": f"L{l}.h{hh}", "ablate_acc": acc,
                         "ablate_n_wrong": sum(p != aa for p, aa in zip(pred, answers)),
                         "attr_mean": float(est.mean()), "exact_mean": float(exact.mean()),
                         "attr_vs_exact_corr_instances": corr(est, exact)})
    res["heads"] = head_res

    # unit-level agreement of attribution and exact patching (means across units)
    am = [u["attr_mean"] for u in unit_res]; em = [u["exact_mean"] for u in unit_res]
    res["attr_vs_exact_corr_over_units"] = corr(am, em)
    k = min(5, len(U))
    top_a = set(sorted(range(len(U)), key=lambda i: am[i])[:k])
    top_e = set(sorted(range(len(U)), key=lambda i: em[i])[:k])
    res["attr_exact_top5_overlap"] = len(top_a & top_e)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def summarize(res):
    L = [f"# {res['program']}: {res['n_examples']} examples, {res['n_units']} units, "
         f"{res['n_heads']} heads, {res['n_params']} params, clean acc {res['clean_acc']:.4f}, "
         f"{res['seconds']}s",
         f"top-logit ties at answer positions: {res['frac_positions_top_logit_tie']:.3f} "
         f"(by answer pos {res['tie_by_answer_pos']})",
         f"attr-vs-exact corr over units (mean effects): {res['attr_vs_exact_corr_over_units']:.3f}; "
         f"top-5 overlap {res['attr_exact_top5_overlap']}/5", ""]
    hdr = ("unit", "active", "corr_doc", "abl_acc", "n_wrong", "unemb top/bot", "attr", "exact", "r_inst",
           "top20_shared")
    L.append(" | ".join(hdr))
    for u in res["units"]:
        L.append(" | ".join([u["unit"], f"{u['frac_active']:.2f}", f"{u['corr_with_documented']:.3f}",
                             f"{u['ablate_acc']:.4f}", str(u["ablate_n_wrong"]),
                             f"{u['unembed_top']}/{u['unembed_bottom']} ({u['unembed_norm']:.1f})",
                             f"{u['attr_mean']:.3f}", f"{u['exact_mean']:.3f}",
                             f"{u['attr_vs_exact_corr_instances']:.3f}", json.dumps(u["top20_shared"])]))
    L.append("")
    for h in res["heads"]:
        L.append(f"{h['head']} | abl_acc {h['ablate_acc']:.4f} | n_wrong {h['ablate_n_wrong']} | "
                 f"attr {h['attr_mean']:.3f} | exact {h['exact_mean']:.3f} | r_inst {h['attr_vs_exact_corr_instances']:.3f}")
    L.append("")
    for u in res["units"]:
        L.append(f"## {u['unit']} top-5: " + "; ".join(
            f"{t['prompt']}{t['answer']}@{t['ans_pos']}={t['act']}" for t in u["top20"][:5]))
    return "\n".join(L)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    names = sys.argv[1:] or ["parity", "sort", "count", "count_strat"]
    for n in names:
        r = run(n)
        with open(os.path.join(OUT, f"{n}.json"), "w") as f:
            json.dump(r, f, indent=1)
        s = summarize(r)
        with open(os.path.join(OUT, f"{n}.txt"), "w") as f:
            f.write(s + "\n")
        print(s, "\n")
