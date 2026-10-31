#!/usr/bin/env python3
"""Part 150: push five Qwen3.5-4B units on six questions, beside random directions, and read them on hand sentences.

One model per run. Reuses Part 94's path: Part 86's push, generated tokens only, one batch per question.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(ROOT / "part86"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

QUESTIONS = ("Who are you?", "Tell me about yourself.", "What do you want most?", "Describe your perfect day.",
             "What is the capital of France?", "Write a short poem about the sea.")
UNITS = ((19, 4979), (19, 5975), (23, 116), (24, 6835), (24, 8542))
LAYERS = (19, 23, 24)
N_RANDOM, SEED, MAX_NEW = 4, 150000, 120
NEURON_DOSES = (0.4, 0.8, 1.5, 2.5, -0.8, -1.5)
RANDOM_DOSES = (0.8, 1.5)
ATLAS_CHECK = ((19, 4979, "financially independent.\n\nSince frugal folks tend to spend a bit more time at home versus "
                "their restaurant hopping peers, groceries are an important topic to cover. Especially because[[ what]]"),
               (23, 116, "I have three tables. Two[[ of]]"))
MARK = re.compile(r"\[\[(.*?)\]\]")


def unit_name(layer, neuron):
    return f"L{layer}N{neuron}"


def row_plan():
    """Every batch row: the plain row, each unit at each dose, then four random directions per layer and dose."""
    rows = [{"unit": "unpushed", "dose": 0.0, "layer": None, "neuron": None}]
    for layer, neuron in UNITS:
        rows += [{"unit": unit_name(layer, neuron), "dose": d, "layer": layer, "neuron": neuron} for d in NEURON_DOSES]
    for layer in LAYERS:
        for d in RANDOM_DOSES:
            rows += [{"unit": f"random-L{layer}-{i}", "dose": d, "layer": layer, "neuron": None, "random_index": i}
                     for i in range(N_RANDOM)]
    return rows


def split_mark(text):
    """Return the text without the marks and the character span of the marked piece."""
    m = MARK.search(text)
    if m is None:
        raise ValueError(f"no mark in {text!r}")
    plain = text[:m.start()] + m.group(1) + text[m.end():]
    return plain[:m.start() + len(m.group(1))] if text.endswith("]]") else plain, m.start(), m.start() + len(m.group(1))


def target_index(offsets, start, end):
    """Index of the one token holding the marked piece; a leading space may belong to the token."""
    hits = [i for i, (a, b) in enumerate(offsets) if a < end and b > start and b > a]
    if len(hits) != 1:
        raise ValueError(f"marked piece spans tokens {hits}")
    return hits[0]


def read_activations(model, tok, downs, text, device, torch):
    """Activation of every unit at the marked token, read at the input of each layer's down projection."""
    plain, start, end = split_mark(text)
    enc = tok(plain, return_tensors="pt", return_offsets_mapping=True)
    idx = target_index([tuple(x) for x in enc.pop("offset_mapping")[0].tolist()], start, end)
    seen = {}
    handles = []
    for layer, neuron in UNITS:
        def grab(_m, args, name=unit_name(layer, neuron), neuron=neuron):
            seen[name] = float(args[0][0, idx, neuron].float().item())
        handles.append(downs[layer].register_forward_pre_hook(grab))
    try:
        with torch.inference_mode():
            model(input_ids=enc["input_ids"].to(device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return seen, tok.convert_ids_to_tokens(int(enc["input_ids"][0, idx]))


def generate_batch(model, tok, hooks, prompt, batch, eos, device, torch):
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    ids = ids.expand(batch, -1).contiguous()
    for hook in hooks:
        hook.begin_sequence()
    try:
        with torch.inference_mode():
            gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_NEW,
                                 do_sample=False, eos_token_id=eos or None, pad_token_id=getattr(tok, "pad_token_id", None))
    finally:
        for hook in hooks:
            hook.remove()
    return gen[:, ids.shape[1]:]


def run(a, torch=None, nr=None, common=None):
    if torch is None:
        import torch
        import neuron_report as nr
        import common86 as common
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok, model = nr.load_model(a.model_id, a.revision, a.device)
    model.requires_grad_(False)
    model.eval()
    _, layers, _ = nr.find_text_layers(model)
    downs = {l: layers[l].mlp.down_proj for l in LAYERS}
    calib = Path(a.calib).read_text(encoding="utf-8")
    medians = {l: float(common.residual_median(model, tok, layers[l], calib, a.device)) for l in LAYERS}
    plan = row_plan()
    pushes = {}
    columns = {}
    for layer in LAYERS:
        weight = downs[layer].weight
        width = int(weight.shape[1])
        matrix = torch.zeros((len(plan), width), dtype=torch.float32)
        random_sets = {d: common.scaled_random_directions(weight, N_RANDOM, SEED + layer, abs(d) * medians[layer]).detach().float().cpu()
                       for d in RANDOM_DOSES}
        for i, row in enumerate(plan):
            if row["layer"] != layer:
                continue
            if row["neuron"] is not None:
                col = weight[:, row["neuron"]].detach().float().cpu()
                columns[row["unit"]] = col
                matrix[i, row["neuron"]] = row["dose"] * medians[layer] / float(torch.linalg.vector_norm(col).item())
            else:
                matrix[i] = random_sets[row["dose"]][row["random_index"]]
        pushes[layer] = matrix.to(device=weight.device, dtype=weight.dtype)
    eos = common.eos_ids(tok, model)
    first = nr.build_prompt(tok, QUESTIONS[0], a.frame == "plain")
    # Gate 1: a zero push in a batch of one matches no hook at all.
    zero = torch.zeros((1, int(downs[19].weight.shape[1])), device=downs[19].weight.device, dtype=downs[19].weight.dtype)
    hooked = generate_batch(model, tok, [common.AddRowsHook(downs[19], zero)], first, 1, eos, a.device, torch)
    bare = generate_batch(model, tok, [], first, 1, eos, a.device, torch)
    gate1 = hooked[0].tolist() == bare[0].tolist()
    rows = []
    for qi, q in enumerate(QUESTIONS):
        prompt = nr.build_prompt(tok, q, a.frame == "plain")
        hooks = [common.AddRowsHook(downs[l], pushes[l]) for l in LAYERS]
        gen = generate_batch(model, tok, hooks, prompt, len(plan), eos, a.device, torch)
        for row, seq in zip(plan, gen):
            text = tok.decode(seq, skip_special_tokens=True)
            if a.frame == "plain":
                text = text.split("\nUser:")[0]
            rows.append({"tag": a.tag, "frame": a.frame, "question_index": qi, "question": q, "unit": row["unit"],
                         "dose": row["dose"], "answer": text.strip()})
        print(f"q{qi} done {time.time() - t0:.0f}s", flush=True)
    with open(out / f"answers-{a.tag}.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sentences = json.loads(Path(a.sentences).read_text(encoding="utf-8"))
    acts = []
    for cls, texts in sentences.items():
        for i, text in enumerate(texts):
            values, token = read_activations(model, tok, downs, text, a.device, torch)
            acts.append({"class": cls, "i": i, "activations": values, "token": token,
                         "marked": MARK.search(text).group(1)})
    atlas = [{"unit": unit_name(l, n), "activation": read_activations(model, tok, downs, t, a.device, torch)[0][unit_name(l, n)]}
             for l, n, t in ATLAS_CHECK]
    meta = {"tag": a.tag, "model_id": a.model_id, "revision": a.revision, "frame": a.frame, "units": UNITS, "medians": medians,
            "neuron_doses": NEURON_DOSES, "random_doses": RANDOM_DOSES, "n_random": N_RANDOM, "seed": SEED, "max_new": MAX_NEW,
            "gate1_zero_matches_no_hook": gate1, "atlas_check": atlas, "activations": acts,
            "columns": {k: [float(x) for x in v.tolist()] for k, v in columns.items()}, "seconds": round(time.time() - t0, 1)}
    with open(out / f"meta-{a.tag}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(json.dumps({"answers": len(rows), "gate1": gate1, "atlas_check": atlas, "seconds": meta["seconds"]}), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ("--model-id", "--revision", "--tag", "--calib", "--sentences", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--frame", required=True, choices=("chat", "plain"))
    ap.add_argument("--device", default="xpu")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
