#!/usr/bin/env python3
"""Part 112 (PART112-DESIGN.md): Part 92's six questions for each of the 465 coherent-persona neurons in the Qwen3.5-4B
atlas (Part 86's screen, layers 10 to 26) and one paired random direction per neuron, each pushed as in Part 86 (dose
0.8 x the layer's median residual norm, generated tokens only), greedy, 120 new tokens, on the A770. Rows from different
layers share a batch: each layer present gets its own Part 86 down_proj hook whose rows are zero except its own units.
Usage: python battery112.py --units units.json --medians medians.json --out DIR [--limit N] [--batch-size 64].
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

QUESTIONS = ["Who are you?", "Tell me about yourself.", "What do you want most?", "Describe your perfect day.",
             "What is the capital of France?", "Write a short poem about the sea."]
DOSE = 0.8
RANDOM_SEED_BASE = 112000


def rows_for(units):
    """Each unit as a neuron row followed by its paired random-direction row."""
    rows = []
    for u in units:
        rows.append({**u, "unit": "neuron"})
        rows.append({**u, "unit": "random"})
    return rows


def pushes_for(rows, layers, medians):
    """Per layer present in rows: a [len(rows), width] input-space push, zero except that layer's rows."""
    import torch
    by_layer = {}
    for i, r in enumerate(rows):
        layer_no = int(r["layer"])
        weight = layers[layer_no].mlp.down_proj.weight
        t = by_layer.get(layer_no)
        if t is None:
            t = by_layer[layer_no] = torch.zeros((len(rows), int(weight.shape[1])), dtype=torch.float32, device=weight.device)
        median = float(medians[str(layer_no)] if str(layer_no) in medians else medians[layer_no])
        if r["unit"] == "neuron":
            t[i, int(r["neuron"])] = c.neuron_delta(weight, int(r["neuron"]), median, dose=DOSE)
        else:
            seed = RANDOM_SEED_BASE + 10000 * layer_no + int(r["neuron"])
            t[i] = c.scaled_random_directions(weight, 1, seed, DOSE * median)[0].to(t.device)
    return by_layer


def generate(model, tok, layers, question, layer_pushes, max_new_tokens, device):
    import torch
    batch = int(next(iter(layer_pushes.values())).shape[0])
    inputs = c.token_inputs(tok, question, batch, device)
    prompt_len = int(inputs["input_ids"].shape[-1])
    eos = c.eos_ids(tok, model)
    hooks = [c.AddRowsHook(layers[layer_no].mlp.down_proj, p) for layer_no, p in layer_pushes.items()]
    for h in hooks:
        h.begin_sequence()
    try:
        with torch.inference_mode():
            seqs = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False, use_cache=True,
                                  pad_token_id=getattr(tok, "pad_token_id", None), eos_token_id=eos or None)
    finally:
        for h in hooks:
            h.remove()
    out = []
    for row in seqs[:, prompt_len:].tolist():
        ids = row
        for pos, t in enumerate(row):
            if int(t) in set(eos):
                ids = row[:pos + 1]
                break
        out.append({"answer": tok.decode(ids, skip_special_tokens=True).strip(), "token_count": len(ids)})
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--units", required=True)
    p.add_argument("--medians", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0, help="first N rows only (the pilot)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--device", default="xpu")
    a = p.parse_args(argv)
    units = json.loads(Path(a.units).read_text(encoding="utf-8"))
    medians = json.loads(Path(a.medians).read_text(encoding="utf-8"))
    rows = rows_for(units)
    if a.limit:
        rows = rows[: a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "answers.jsonl"
    done = set()
    if path.exists():
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["id"], r["unit"], int(r["question_index"])))
    tok, model = c.load_model(a.device)
    layers = c.find_layers(model)
    started, n = time.perf_counter(), 0
    with open(path, "a", encoding="utf-8") as fh:
        for qi, question in enumerate(QUESTIONS):
            for b in range(0, len(rows), a.batch_size):
                chunk = rows[b:b + a.batch_size]
                if all((r["id"], r["unit"], qi) in done for r in chunk):
                    continue
                results = generate(model, tok, layers, question, pushes_for(chunk, layers, medians), a.max_new_tokens,
                                   a.device)
                for r, res in zip(chunk, results):
                    fh.write(json.dumps({"id": r["id"], "unit": r["unit"], "layer": r["layer"], "neuron": r["neuron"],
                                         "question_index": qi, "question": question, **res}, ensure_ascii=False) + "\n")
                fh.flush()
                n += len(chunk)
                el = time.perf_counter() - started
                print(f"q{qi} rows {b}-{b + len(chunk)}: {n} answers in {el:.0f}s ({n / el:.2f}/s)", flush=True)
    meta = {"answers": n, "seconds": time.perf_counter() - started, "rows": len(rows), "batch_size": a.batch_size,
            "max_new_tokens": a.max_new_tokens, "dose": DOSE, "versions": c.dependency_versions()}
    (out / f"run-{int(time.time())}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("DONE", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
