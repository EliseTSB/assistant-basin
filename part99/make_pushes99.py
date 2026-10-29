#!/usr/bin/env python3
"""make_pushes99.py -- Part 99 Amendment 1: build each classified unit's push and paired random direction ONCE, on the
unmodified Instruct model in the chat frame (Part 92's recipe: alpha 0.8 of Instruct's chat residual median at the unit's
layer), and save them so every host and swap receives the identical injected vector. Prints each vector's norm.
Usage: make_pushes99.py INSTRUCT_SNAPSHOT NEURONS.json KINDS.json OUT.pt
"""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part92"))
import common92 as c  # noqa: E402


def main(snap, neurons_path, kinds_path, out):
    kinds = json.load(open(kinds_path, encoding="utf-8"))["kinds"]
    neurons = [n for n in json.load(open(neurons_path, encoding="utf-8")) if kinds.get(n["id"]) in ("concept", "slot")]
    tok, model = c.load_model(snap, "cpu")
    layers = c.get_layers(model)
    med = c.residual_medians(model, tok, "chat", "cpu")
    vecs, norms = {}, {}
    for n in neurons:
        w = layers[int(n["layer"])].mlp.down_proj.weight
        t = 0.8 * med[int(n["layer"])]
        vecs[f'{n["id"]}|neuron'] = c.neuron_push(w, int(n["neuron"]), t).detach().cpu()
        vecs[f'{n["id"]}|random'] = c.random_direction(w, int(n["control_seed"]), t).detach().cpu()
        norms[n["id"]] = round(float(vecs[f'{n["id"]}|neuron'].float().norm()), 4)
    torch.save(vecs, out)
    json.dump({"units": len(neurons), "medians_chat": {str(k): v for k, v in med.items()}, "neuron_norms": norms},
              open(Path(out).with_suffix(".json"), "w"), indent=1)
    print(json.dumps({"units": len(neurons), "vectors": len(vecs)}))


if __name__ == "__main__":
    main(*sys.argv[1:5])
