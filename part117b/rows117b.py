#!/usr/bin/env python3
"""Part 117b (PART117B-DESIGN.md): controls for Part 117's "concept -1 with slot +1" cell in Qwen3.5-4B. Per pair: the
slot neuron alone at +1, +0.85 and +0.5; the matched concept at -1 and -0.5 beside the slot at +1; the unrelated concept at -1
beside the slot at +1; three random directions written at the matched concept's size, in its layer, at -1 beside the
slot at +1 (index 0 is Part 112's paired random direction for that concept neuron); three neurons from the matched
concept's layer that are not persona neurons (pick117b.py), at -1 beside the slot at +1; and each distinct concept
neuron alone at -1. The push arithmetic and generation are Part 117's (pairs117.pushes_for, battery112.generate): each
component adds its multiplier times Part 112's push (0.8 x the layer's median residual norm) to the down_proj input on
generated tokens only. Part 92's six questions, greedy, 120 new tokens. Appends to answers.jsonl and skips rows already
written. Written by Claude (Opus 5.5), 2026-10-29."""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "part112"))
import battery112 as b  # noqa: E402

c = b.c
N_RANDOM = 3
RANDOM_INDEX_SEED_STEP = 1_000_000   # index k > 0 seeds outside Part 112's range (112000 + 10000 * layer + neuron)
PAIR_CONDS = [("s+1", (("s", 1.0),)), ("s+0.85", (("s", 0.85),)), ("s+0.5", (("s", 0.5),)),
              ("c-1,s+1", (("c", -1.0), ("s", 1.0))), ("c-0.5,s+1", (("c", -0.5), ("s", 1.0))),
              ("u-1,s+1", (("u", -1.0), ("s", 1.0)))]


def rows_for(pairs, controls=None):
    """pairs: [{"pair": "p01", "slot": unit, "match": unit, "unrelated": unit}], unit = {id, layer, neuron, persona};
    controls: {pair: {"layer": L, "neurons": [n, ...]}} from pick117b.py (non-persona neurons in the matched concept's
    layer), or None for no control-neuron rows."""
    rows = [{"row": "none", "pair": None, "set": "none", "cond": "none", "components": []}]
    alone = {}
    for p in pairs:
        units = {"s": p["slot"], "c": p["match"], "u": p["unrelated"]}
        for cond, comps in PAIR_CONDS:
            rows.append({"row": f"{p['pair']}:{cond}", "pair": p["pair"], "set": "pair", "cond": cond,
                         "components": [{**units[role], "role": role, "unit": "neuron", "mult": m}
                                        for role, m in comps]})
        for k in range(N_RANDOM):
            rows.append({"row": f"{p['pair']}:r{k}-1,s+1", "pair": p["pair"], "set": "random", "cond": f"r{k}-1,s+1",
                         "components": [{**p["match"], "role": "r", "unit": "random", "rand_index": k, "mult": -1.0},
                                        {**p["slot"], "role": "s", "unit": "neuron", "mult": 1.0}]})
        ctl = (controls or {}).get(p["pair"])
        for k, n in enumerate(ctl["neurons"] if ctl else []):
            unit = {"id": f"ctl-{ctl['layer']}-{n}", "layer": int(ctl["layer"]), "neuron": int(n), "persona": None}
            rows.append({"row": f"{p['pair']}:n{k}-1,s+1", "pair": p["pair"], "set": "control", "cond": f"n{k}-1,s+1",
                         "components": [{**unit, "role": "n", "unit": "neuron", "mult": -1.0},
                                        {**p["slot"], "role": "s", "unit": "neuron", "mult": 1.0}]})
        for key in ("match", "unrelated"):
            alone.setdefault(p[key]["id"], p[key])
    for uid, u in alone.items():
        rows.append({"row": f"alone:{uid}:c-1", "pair": None, "set": "alone", "cond": "c-1",
                     "components": [{**u, "role": "c", "unit": "neuron", "mult": -1.0}]})
    return rows


def random_seed(layer, neuron, k):
    """Index 0 is Part 112's seed for this neuron's paired random direction; later indices step past its range."""
    return b.RANDOM_SEED_BASE + 10000 * int(layer) + int(neuron) + RANDOM_INDEX_SEED_STEP * int(k)


def pushes_for(rows, layers, medians):
    """Per layer present: a [len(rows), width] input-space push, the sum of that row's components in that layer."""
    import torch
    by_layer = {}
    for i, r in enumerate(rows):
        for comp in r["components"]:
            ln = int(comp["layer"])
            w = layers[ln].mlp.down_proj.weight
            if ln not in by_layer:
                by_layer[ln] = torch.zeros((len(rows), int(w.shape[1])), dtype=torch.float32, device=w.device)
            med = float(medians[str(ln)])
            m = float(comp["mult"])
            if comp["unit"] == "neuron":
                by_layer[ln][i, int(comp["neuron"])] += m * c.neuron_delta(w, int(comp["neuron"]), med, dose=b.DOSE)
            else:
                seed = random_seed(ln, comp["neuron"], comp.get("rand_index", 0))
                by_layer[ln][i] += m * c.scaled_random_directions(w, 1, seed, b.DOSE * med)[0].to(by_layer[ln].device)
    return by_layer


def generate_plain(model, tok, question, max_new_tokens, device):
    """battery112.generate at batch 1 with no hook installed at all: the plain model."""
    import torch
    inputs = c.token_inputs(tok, question, 1, device)
    prompt_len = int(inputs["input_ids"].shape[-1])
    eos = c.eos_ids(tok, model)
    with torch.inference_mode():
        seqs = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False, use_cache=True,
                              pad_token_id=getattr(tok, "pad_token_id", None), eos_token_id=eos or None)
    ids = seqs[0, prompt_len:].tolist()
    for pos, t in enumerate(ids):
        if int(t) in set(eos):
            ids = ids[:pos + 1]
            break
    return {"answer": tok.decode(ids, skip_special_tokens=True).strip(), "token_count": len(ids)}


def zero_hook_check(model, tok, layers, medians, device, max_new_tokens=40):
    """The do-nothing control: each question's greedy answer at batch 1 with an all-zero push hook and with no hook.
    The gate needs every pair equal (decoded text and token count)."""
    import torch
    ln = min(int(k) for k in medians)
    w = layers[ln].mlp.down_proj.weight
    zero = {ln: torch.zeros((1, int(w.shape[1])), dtype=torch.float32, device=w.device)}
    out = []
    for q in b.QUESTIONS:
        hooked = b.generate(model, tok, layers, q, zero, max_new_tokens, device)[0]
        plain = generate_plain(model, tok, q, max_new_tokens, device)
        out.append({"question": q, "equal": hooked == plain, "hooked": hooked, "plain": plain})
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--medians", required=True)
    p.add_argument("--controls", required=True, help="results/part117b/control-neurons.json (pick117b.py)")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--device", default="xpu")
    p.add_argument("--direct-load", action="store_true", help="load straight to the device (Part 115b's loader)")
    a = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = rows_for(json.loads(Path(a.pairs).read_text(encoding="utf-8")),
                    json.loads(Path(a.controls).read_text(encoding="utf-8")))
    medians = json.loads(Path(a.medians).read_text(encoding="utf-8"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "answers.jsonl"
    done = set()
    if path.exists():
        done = {(r["row"], int(r["question_index"])) for r in map(json.loads, open(path, encoding="utf-8"))}
    if a.direct_load:
        sys.path.insert(0, str(HERE.parent / "part115b"))
        import howmany115b as hb
        tok, model = hb.load_direct(a.device)
    else:
        tok, model = c.load_model(a.device)
    layers = c.find_layers(model)
    check = out / "zero-hook-check.json"
    if not check.exists():
        res = zero_hook_check(model, tok, layers, medians, a.device)
        check.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"zero-hook check: {sum(r['equal'] for r in res)} of {len(res)} equal", flush=True)
    print(f"rows {len(rows)}, answers {6 * len(rows)}, already written {len(done)}", flush=True)
    with open(path, "a", encoding="utf-8") as fh:
        for qi, q in enumerate(b.QUESTIONS):
            todo = [r for r in rows if (r["row"], qi) not in done]
            for s in range(0, len(todo), a.batch_size):
                batch = todo[s:s + a.batch_size]
                pushes = pushes_for(batch, layers, medians)
                if not pushes:   # a batch of only the unpushed row still needs one (all-zero) hook
                    import torch
                    ln = min(int(k) for k in medians)
                    w = layers[ln].mlp.down_proj.weight
                    pushes = {ln: torch.zeros((len(batch), int(w.shape[1])), dtype=torch.float32, device=w.device)}
                for r, ans in zip(batch, b.generate(model, tok, layers, q, pushes, a.max_new_tokens, a.device)):
                    fh.write(json.dumps({"row": r["row"], "pair": r["pair"], "set": r["set"], "cond": r["cond"],
                                         "question_index": qi, "question": q, **ans}, ensure_ascii=False) + "\n")
                fh.flush()
            print(f"question {qi} done", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
