#!/usr/bin/env python3
"""causal206.py: the one causal check (PART206-DESIGN.md draft 7, Method (5)) at a final chat checkpoint.

  python -I -B part206/causal206.py --stage qwen-chat --out results/part206   (needs that stage's summary.json)
  python -I -B part206/causal206.py --stage instruct  --out results/part206   (readouts (b) and (c) only)

Delta^h is the (A-tmpl, M) shift measured on h (the stream where the MLP reads it) on the FIT half; L* is the
middle-third layer with the highest rho(A-tmpl, M) inside the fit half (analyze206.py, `L_star`). During a forward
pass, k x Delta^h is subtracted from the stream at L* by subtracting it from the mixer's contribution c_L* (the last
thing added to the stream before h_L*), at every assistant-turn position, for k in {1, 2}. "Assistant-turn
positions" are read as: from the first token of the template's generation prompt (the assistant header) to the end
of the sequence; the design does not say whether the header is included, and including it is what lets the shift
act on a readout whose maker name is the answer's first token. Controls at the same norm: 20 seeded random
directions and the (M2, M) shift on h.

Readouts, without generation except (a)'s one plain greedy decode per phrasing:
  (a) Qwen3.5-4B only: Part 66's 24 held-out maker phrasings in the chat template (neuron_report.build_prompt); the
      plain model's greedy answer (at most 48 tokens) is printed with the maker name found in it; the readout is the
      log probability of the first token of that maker name, at the position where the plain answer names it,
      teacher-forced on the plain answer's prefix. Rows naming no listed maker are dropped (G3 counts the rest).
  (b) the catalogued neurons' mean gate pre-activation at layers above L*, on the test half in frame A-tmpl.
  (c) mean next-token loss on the wikitext test passages in frame A-tmpl.
Written by Claude (Opus 5.5), 2026-11-12.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import common206 as C  # noqa: E402


def part66_phrasings():
    """Part 66's 24 held-out maker phrasings (16 direct, 8 indirect), read from part66/part66.py's lists without
    importing it (it imports the GPU stack)."""
    import ast
    tree = ast.parse((ROOT / "part66" / "part66.py").read_text(encoding="utf-8"))
    vals = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("DIRECT", "INDIRECT"):
                vals[node.targets[0].id] = ast.literal_eval(node.value)
    return vals["DIRECT"] + vals["INDIRECT"]


def find_maker(text, names):
    """(name, char offset) of the first listed maker name in the answer, or (None, None)."""
    best = None
    for n in names:
        i = text.find(n)
        if i >= 0 and (best is None or i < best[1]):
            best = (n, i)
    return best if best else (None, None)


def maker_token_index(tok, answer_ids, char_at):
    """Index of the first answer token whose decoded span reaches the maker name's first character."""
    for t in range(len(answer_ids)):
        if len(tok.decode(answer_ids[:t + 1])) > char_at:
            return t
    return None


def generation_header_start(tok, fam, user_text):
    """Token index where the generation prompt (assistant header) begins: the length of the template without it."""
    kw = {"enable_thinking": False} if fam == "Q" else {}
    msgs = [{"role": "user", "content": user_text}]
    without = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, **kw)
    ids = tok(without, add_special_tokens=False).input_ids
    if fam == "O" and tok.bos_token_id is not None and ids[:1] != [tok.bos_token_id]:
        ids = [tok.bos_token_id] + ids
    return len(ids)


class Subtract:
    """Forward hook on the mixer's stream contribution at L*: subtract `vec` at positions >= start."""

    def __init__(self, module):
        self.vec, self.start = None, None
        self.h = module.register_forward_hook(self.hook)

    def hook(self, mod, args, out):
        if self.vec is None:
            return None
        t = C.first_tensor(out)
        new = t.clone()
        new[:, self.start:] = (t[:, self.start:].float() - self.vec).to(t.dtype)
        if isinstance(out, (tuple, list)):
            return (new,) + tuple(out[1:])
        return new


def main(argv=None):
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["instruct", "qwen-chat"])
    ap.add_argument("--out", default="results/part206")
    ap.add_argument("--prompts", default=str(HERE / "prompts206.json"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model-path", default=None, help="a local snapshot directory of the pinned revision (the Arc)")
    a_ = ap.parse_args(argv)
    cfg = C.load_config()
    prompts = json.loads(Path(a_.prompts).read_text())
    sdir = Path(a_.out) / a_.stage
    summ = json.loads((sdir / "summary.json").read_text())
    z = np.load(sdir / "moments.npz")
    meta = json.loads((sdir / "meta.json").read_text())
    pairs = [tuple(p) for p in meta["pairs"]]
    Lstar = summ["common_mode"]["A-tmpl|M"]["L_star"]
    if Lstar is None:
        raise SystemExit("no L* in summary.json")
    pa, pn = pairs.index(("A-tmpl", "M")), pairs.index(("M2", "M"))
    nfit = z["n_pair"][pa, 0]
    delta = z["SDh"][pa, 0, Lstar] / nfit
    null = z["SDh"][pn, 0, Lstar] / z["n_pair"][pn, 0]
    rng = np.random.default_rng(cfg["seeds"]["causal_random"])
    R = rng.standard_normal((cfg["causal_random_directions"], len(delta)))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    arms = [("plain", 0, None)]
    for k in cfg["causal_doses"]:
        v = k * delta
        nrm = float(np.linalg.norm(v))
        arms.append((f"arm_k{k}", k, v))
        arms.append((f"null_M2M_k{k}", k, C.unit(null) * nrm))
        for i in range(len(R)):
            arms.append((f"random{i}_k{k}", k, R[i] * nrm))
    tok, model, env = C.load_model(cfg, a_.stage, a_.device, a_.model_path)
    device = env["device"]
    fam = env["family"]
    layers, _ = C.text_layers(model)
    sub = Subtract(C.site_modules(layers[Lstar])["c"][0])
    out = {"stage": a_.stage, "L_star": Lstar, "delta_norm": float(np.linalg.norm(delta)), "arms": [a[0] for a in arms]}

    # (a) Qwen: the maker readout
    if fam == "Q":
        import neuron_report as nr
        rows = []
        for q in part66_phrasings()[:a_.limit or None]:
            text = nr.build_prompt(tok, q, plain=False)
            pids = tok(text, add_special_tokens=False).input_ids
            hstart = generation_header_start(tok, fam, q)
            sub.vec = None
            gen = model.generate(torch.tensor([pids], device=device), max_new_tokens=48, do_sample=False)
            ans = gen[0, len(pids):].tolist()
            atext = tok.decode(ans, skip_special_tokens=True)
            name, at = find_maker(atext, cfg["maker_names"])
            row = {"question": q, "answer": atext, "maker": name}
            if name is not None:
                t = maker_token_index(tok, ans, at)
                ids = torch.tensor([pids + ans[:t]], device=device)
                target = ans[t]
                lp = {}
                for arm, k, v in arms:
                    sub.vec = None if v is None else torch.tensor(v, dtype=torch.float32, device=device)
                    sub.start = hstart
                    logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
                    lsm = torch.log_softmax(logits, -1)
                    lp[arm] = float(lsm[target])
                    if arm == "plain":
                        top = torch.topk(lsm, 5)
                        row["plain_top5"] = [[tok.decode([int(i)]), float(p)] for p, i in zip(top.values, top.indices)]
                        row["plain_top1_is_maker"] = int(top.indices[0]) == target
                row.update({"maker_token": tok.decode([target]), "position": len(pids) + t, "logprob": lp})
            rows.append(row)
            print(json.dumps({"q": q, "answer": atext[:160], "maker": name}), flush=True)
        out["readout_a"] = rows
        named = [r for r in rows if r.get("maker") is not None and r.get("plain_top1_is_maker")]
        out["G3"] = {"named_at_top1": len(named), "of": len(rows), "pass": len(named) >= cfg["g3_min_maker_rows"]}
    # (b) and (c): the test half in frame A-tmpl
    F = prompts["families"][fam]
    pre = F["frames"]["A-tmpl"]
    hstart = generation_header_start(tok, fam, cfg["request"])
    test = [p for p in prompts["passages"] if p["half"] == "test"][:a_.limit or None]
    cat = prompts["catalogued"][fam]
    want = {}
    for name, lst in cat.items():
        for L, n in lst:
            if L > Lstar:
                want.setdefault(int(L), set()).add(int(n))
    want = {L: sorted(v) for L, v in want.items()}
    caps = {}
    hooks = []
    lo, hi = len(pre) + cfg["read_first"], len(pre) + cfg["passage_tokens"]
    for L, ns in want.items():
        idx = torch.tensor(ns, device=device)
        hooks.append(layers[L].mlp.gate_proj.register_forward_hook(
            lambda m, args, o, L=L, idx=idx: caps.__setitem__(L, o[:, lo:hi].index_select(-1, idx).float().sum((0, 1)))))
    sums = {arm: {L: np.zeros(len(ns)) for L, ns in want.items()} for arm, _, _ in arms}
    loss = {arm: [0.0, 0] for arm, _, _ in arms}
    npos = 0
    try:
        for b0 in range(0, len(test), a_.batch):
            bp = test[b0:b0 + a_.batch]
            ids = torch.tensor([pre + p["ids"][fam] for p in bp], device=device)
            wiki = torch.tensor([p["source"] == "wikitext" for p in bp], device=device)
            for arm, k, v in arms:
                sub.vec = None if v is None else torch.tensor(v, dtype=torch.float32, device=device)
                sub.start = hstart
                caps.clear()
                logits = model(input_ids=ids, use_cache=False).logits.float()
                for L in want:
                    sums[arm][L] += caps[L].cpu().numpy()
                if bool(wiki.any()):
                    lp = torch.log_softmax(logits[wiki, len(pre):-1], -1)
                    tgt = ids[wiki, len(pre) + 1:]
                    nll = -lp.gather(-1, tgt[..., None])[..., 0]
                    loss[arm][0] += float(nll.sum())
                    loss[arm][1] += int(nll.numel())
            npos += len(bp) * (hi - lo)
    finally:
        for h in hooks:
            h.remove()
    out["readout_b"] = {arm: {name: float(np.mean([sums[arm][L][want[L].index(n)] / npos
                                                   for L, n in lst if L > Lstar])) if any(L > Lstar for L, _ in lst) else None
                              for name, lst in cat.items()} for arm, _, _ in arms}
    out["readout_c"] = {arm: (v[0] / v[1] if v[1] else None) for arm, v in loss.items()}
    C.atomic_write_json(sdir / "causal.json", out)
    print(f"causal_done stage={a_.stage} L*={Lstar}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
