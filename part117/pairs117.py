#!/usr/bin/env python3
"""Part 117 (PART117-DESIGN.md): Qwen3.5-4B's concept and slot persona neurons (Part 112's sort) pushed alone and
together, in theme-sharing pairs and in unrelated pairs. Each row carries zero, one or two components, a neuron or its
Part 112 random direction, each with a multiplier on Part 112's push (0.8 x the layer's median residual norm, added to
the down_proj input on generated tokens only). Part 92's six questions, greedy, 120 new tokens; rows from different
layers share a batch, with one hook per layer present. Appends to answers.jsonl and skips rows already written.
Written by Claude (Opus 5.5), 2026-10-29."""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "part112"))
import battery112 as b  # noqa: E402

c = b.c
MATCHED = [("c+1", (("c", 1.0),)), ("s+1", (("s", 1.0),)), ("both+0.5", (("c", 0.5), ("s", 0.5))),
           ("both+1", (("c", 1.0), ("s", 1.0))), ("both+2", (("c", 2.0), ("s", 2.0))),
           ("c+1,s-1", (("c", 1.0), ("s", -1.0))), ("c-1,s+1", (("c", -1.0), ("s", 1.0)))]
UNRELATED = [("c+1", (("c", 1.0),)), ("both+1", (("c", 1.0), ("s", 1.0))), ("c+1,s-1", (("c", 1.0), ("s", -1.0)))]


def rows_for(pairs):
    """pairs: [{"pair": "p01", "slot": unit, "match": unit, "unrelated": unit}], unit = {id, layer, neuron, persona}."""
    rows = [{"row": "none", "pair": None, "set": "none", "cond": "none", "components": []}]
    for p in pairs:
        for setname, cname, conds in (("matched", "match", MATCHED), ("unrelated", "unrelated", UNRELATED)):
            for cond, comps in conds:
                parts = [{**(p[cname] if role == "c" else p["slot"]), "role": role, "unit": "neuron", "mult": m}
                         for role, m in comps]
                rows.append({"row": f"{p['pair']}:{setname}:{cond}", "pair": p["pair"], "set": setname, "cond": cond,
                             "components": parts})
        rows.append({"row": f"{p['pair']}:random:both+1", "pair": p["pair"], "set": "random", "cond": "both+1",
                     "components": [{**p["match"], "role": "c", "unit": "random", "mult": 1.0},
                                    {**p["slot"], "role": "s", "unit": "random", "mult": 1.0}]})
    return rows


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
                seed = b.RANDOM_SEED_BASE + 10000 * ln + int(comp["neuron"])
                by_layer[ln][i] += m * c.scaled_random_directions(w, 1, seed, b.DOSE * med)[0].to(by_layer[ln].device)
    return by_layer


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--medians", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--device", default="xpu")
    p.add_argument("--direct-load", action="store_true", help="load straight to the device (Part 115b's loader)")
    a = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = rows_for(json.loads(Path(a.pairs).read_text(encoding="utf-8")))
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
    with open(path, "a", encoding="utf-8") as fh:
        for qi, q in enumerate(b.QUESTIONS):
            todo = [r for r in rows if (r["row"], qi) not in done]
            for s in range(0, len(todo), a.batch_size):
                batch = todo[s:s + a.batch_size]
                pushes = pushes_for(batch, layers, medians)
                if not pushes:   # a batch of only the unpushed row still needs one (all-zero) hook
                    ln = min(int(k) for k in medians)
                    w = layers[ln].mlp.down_proj.weight
                    import torch
                    pushes = {ln: torch.zeros((len(batch), int(w.shape[1])), dtype=torch.float32, device=w.device)}
                for r, ans in zip(batch, b.generate(model, tok, layers, q, pushes, a.max_new_tokens, a.device)):
                    fh.write(json.dumps({"row": r["row"], "pair": r["pair"], "set": r["set"], "cond": r["cond"],
                                         "question_index": qi, "question": q, **ans}, ensure_ascii=False) + "\n")
                fh.flush()
            print(f"question {qi} done", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
