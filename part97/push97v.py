#!/usr/bin/env python3
"""push97v.py -- Part 97.3 validation push, step 2 (Claude, Opus 5.5, 2026-10-29; PART97-DESIGN.md step 4 and
Amendment 3).

Each selected neuron is pushed with Part 92's push (alpha 0.8 of Instruct's chat-frame residual median at its layer,
added at the block output) on Part 95's six questions, greedy, 60 new tokens; beside it, 4 random directions of its layer
(common92.random_directions, seed 97097 + layer, same norm) and one unpushed pass. Units of one layer share a batch.
Usage: python part97/push97v.py --snapshot DIR --units results/part97/validation/units.json --out DIR --device xpu --batch 48
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part92"))
import common92 as c  # noqa: E402
import torch  # noqa: E402

QUESTIONS = ("What should I cook tonight?", "Give me some advice for a job interview.", "Explain how a bicycle works.",
             "What is the capital of France?", "Tell me a story about a cat.", "Describe a walk in the park.")
ALPHA, N_RANDOM, SEED, MAX_NEW = 0.8, 4, 97097, 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=48)
    a = ap.parse_args()
    units = json.loads(Path(a.units).read_text(encoding="utf-8"))["units"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok, model = c.load_model(a.snapshot, a.device)
    layers = c.get_layers(model)
    med = c.residual_medians(model, tok, "chat", a.device)
    rows = []

    def run(layer_module, specs):
        flat = [(meta, qi, vec) for meta, vec in specs for qi in range(len(QUESTIONS))]
        for start in range(0, len(flat), a.batch):
            chunk = flat[start:start + a.batch]
            vecs = None if chunk[0][2] is None else torch.stack([vec for _, _, vec in chunk])
            answers = c.generate_batch(model, tok, [QUESTIONS[qi] for _, qi, _ in chunk], "chat", layer_module,
                                       vecs, MAX_NEW, a.device)
            for (meta, qi, _), ans in zip(chunk, answers):
                rows.append({**meta, "question_index": qi, "question": QUESTIONS[qi], "answer": ans})

    run(layers[0], [({"kind": "unpushed"}, None)])
    norms = {}
    for layer in sorted({u["layer"] for u in units}):
        weight = layers[layer].mlp.down_proj.weight
        norm = ALPHA * float(med[layer])
        norms[layer] = norm
        specs = [({"kind": "neuron", "layer": layer, "index": u["index"], "group": u["group"]},
                  c.neuron_push(weight, int(u["index"]), norm)) for u in units if u["layer"] == layer]
        specs += [({"kind": "random", "layer": layer, "random_id": k}, vec)
                  for k, vec in enumerate(c.random_directions(weight, N_RANDOM, SEED + layer, norm))]
        run(layers[layer], specs)
        print(f"layer {layer}: {len(specs)} units done, {time.time() - t0:.0f} s", flush=True)
    with (out / "answers.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {"snapshot": a.snapshot, "device": a.device, "batch": a.batch, "alpha": ALPHA, "n_random": N_RANDOM,
            "random_seed": f"{SEED} + layer", "max_new_tokens": MAX_NEW, "norms": norms, "rows": len(rows),
            "seconds": round(time.time() - t0, 1), "torch": torch.__version__, "python": platform.python_version()}
    (out / "answers.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "seconds": meta["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
