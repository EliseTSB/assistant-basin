#!/usr/bin/env python3
"""Part 115b (PART115B-DESIGN.md): do the neuron sets that take "Q" off Qwen3.5-4B's name after "I am **" also take
" Q" off it in a plain-text answer with no "**"? Reuses Part 115's zeroing, sets and matched random controls
(part115/howmany115.py) on two rankings, Part 114's gradient attribution and Part 116's direct-write ranking. Each
held-out phrasing gets a plain-text instruction and three forced openings: "My name is" (target " Q"), 'My name is "'
(target "Q", no "**") and "My name is **" (target "Q"). Six ordinary probes whose plain-model first choice is the right answer check for damage. Gates: an
all-ones mask must reproduce the plain logits exactly, and in the unzeroed row every target must be the first choice.
Written by Claude (Opus 5.5), 2026-10-29."""
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "part115"))
import howmany115 as h  # noqa: E402

PLAIN = " Please answer in plain text, without markdown."
CONTEXTS = {"plain": ("My name is", " Q"), "quote": ('My name is "', "Q"), "bold": ("My name is **", "Q")}
ORDINARY = [("What is the capital of France?", "The capital of France is **", "Paris"),
            ("What is 2 + 2?", "2 + 2 = **", "4"),
            ("What is the opposite of hot?", "The opposite of hot is **", "cold"),
            ("What is the chemical symbol for gold?", "The chemical symbol for gold is **", "Au"),
            ("What color is the sky on a clear day?", "On a clear day the sky is **", "blue"),
            ("What is water made of?", "Water is made of hydrogen and", " oxygen")]


def one_token(tok, word):
    t = tok(word, add_special_tokens=False)["input_ids"]
    assert len(t) == 1, (word, t)
    return t[0]


def run_ranking(model, tok, layers, attr_path, device, dtype):
    import torch
    saved = torch.load(attr_path, map_location="cpu")
    cands500 = h.top_candidates(saved["attr"], h.EXCLUDE)
    rows = h.build_rows(cands500, saved["mean_abs"])
    res = {i: {ctx: [] for ctx in CONTEXTS} | {"ordinary": []} for i in range(len(rows))}
    for ctx, (opening, word) in CONTEXTS.items():
        tid = one_token(tok, word)
        for q in h.q114.HELDOUT:
            logits = h.run_rows(model, layers, rows, h.ids_for(tok, q + PLAIN, opening).to(device), device, dtype)
            for i in range(len(rows)):
                lp = torch.log_softmax(logits[i], dim=-1)
                res[i][ctx].append({"logodds": float(h.q114.q_logodds(logits[i], tid)),
                                    "argmax_is_target": int(lp.argmax()) == tid, "top": tok.decode([int(lp.argmax())])})
    for q, opening, word in ORDINARY:
        tid = one_token(tok, word)
        logits = h.run_rows(model, layers, rows, h.ids_for(tok, q, opening).to(device), device, dtype)
        for i in range(len(rows)):
            lp = torch.log_softmax(logits[i], dim=-1)
            res[i]["ordinary"].append({"question": q, "logprob": float(lp[tid]), "argmax_is_target": int(lp.argmax()) == tid})
    none = res[0]
    bad = [ctx for ctx in CONTEXTS if not all(r["argmax_is_target"] for r in none[ctx])]
    bad += [r["question"] for r in none["ordinary"] if not r["argmax_is_target"]]
    if bad:
        raise SystemExit(f"GATE FAILED: in the unzeroed row the target is not the first choice for {bad}")
    cand_rows = [r for r in rows if r["role"] == "candidate"]
    width = int(layers[0].mlp.down_proj.weight.shape[1])
    gens = {ctx: {r["k"]: [] for r in cand_rows} for ctx in CONTEXTS}
    for ctx, (opening, _w) in CONTEXTS.items():
        for q in h.q114.HELDOUT:
            ids = h.ids_for(tok, q + PLAIN, opening).to(device)
            n = int(ids.shape[-1])
            hooks = [h.MaskZero(layers[li].mlp.down_proj, m) for li, m in h.masks_for(cand_rows, len(layers), width, device, dtype).items()]
            try:
                seqs = model.generate(input_ids=ids.expand(len(cand_rows), -1), max_new_tokens=12, do_sample=False,
                                      pad_token_id=getattr(tok, "pad_token_id", None))
            finally:
                for hk in hooks:
                    hk.remove()
            for r, seq in zip(cand_rows, seqs):
                gens[ctx][r["k"]].append({"question": q, "continuation": tok.decode(seq[n:], skip_special_tokens=True)})
    return {"attr": str(attr_path), "rows": [dict(r, units=[list(u) for u in r["units"]], **res[i]) for i, r in enumerate(rows)],
            "generations": {ctx: {str(k): v for k, v in g.items()} for ctx, g in gens.items()}}


def load_direct(device):
    """Amendment 1: the weights go straight to the device (device_map) instead of being staged in host memory first,
    which the RAM guard stopped at Windows' 2 GB floor on the first launch. Same weights, dtype and device."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(h.c.MODEL_ID, revision=h.c.REVISION)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = dict(revision=h.c.REVISION, dtype=torch.bfloat16, device_map={"": device})
    try:
        model = AutoModelForCausalLM.from_pretrained(h.c.MODEL_ID, **kw)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(h.c.MODEL_ID, **kw)
    model.eval()
    model.requires_grad_(False)
    torch.set_grad_enabled(False)
    return tok, model


def main(argv=None):
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--direct-load", action="store_true", help="Amendment 1: load straight to the device")
    p.add_argument("--out", required=True)
    p.add_argument("--ranking", action="append", required=True, help="name=path to an attribution or rank file")
    p.add_argument("--device", default="xpu")
    a = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    tok, model = load_direct(a.device) if a.direct_load else h.c.load_model(a.device)
    layers = h.c.find_layers(model)
    model.requires_grad_(False)
    dtype = next(model.parameters()).dtype
    payload = {"design": "PART115B-DESIGN.md", "direct_load": a.direct_load, "plain_instruction": PLAIN, "contexts": CONTEXTS,
               "ordinary": ORDINARY, "ks": h.KS, "n_sets": h.N_SETS, "pool": h.POOL, "exclude": h.EXCLUDE,
               "target_ids": {ctx: one_token(tok, w) for ctx, (_o, w) in CONTEXTS.items()}, "rankings": {}}
    with torch.inference_mode():
        ids = h.ids_for(tok, h.q114.HELDOUT[0] + PLAIN, CONTEXTS["plain"][0]).to(a.device)
        plain = model(input_ids=ids).logits[0, -1].float().cpu()
        width = int(layers[0].mlp.down_proj.weight.shape[1])
        ones = [h.MaskZero(l.mlp.down_proj, torch.ones((1, width), device=a.device, dtype=dtype)) for l in layers]
        try:
            masked = model(input_ids=ids).logits[0, -1].float().cpu()
        finally:
            for hk in ones:
                hk.remove()
        if not torch.equal(plain, masked):
            raise SystemExit(f"GATE FAILED: an all-ones mask changed the logits (max diff {float((plain - masked).abs().max())})")
        payload["gates"] = {"ones_mask_bit_identical": True}
        for spec in a.ranking:
            name, path = spec.split("=", 1)
            payload["rankings"][name] = run_ranking(model, tok, layers, path, a.device, dtype)
            payload["gates"][f"{name}_targets_first_unzeroed"] = True
            print(f"ranking {name} done ({time.time() - t0:.0f}s)", flush=True)
    payload["elapsed_s"] = round(time.time() - t0, 1)
    payload["versions"] = h.c.dependency_versions()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "results115b.json").write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    print("DONE", payload["elapsed_s"], flush=True)


if __name__ == "__main__":
    main()
