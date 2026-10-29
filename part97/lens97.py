"""lens97.py -- Part 97 step 1: logit-lens and J-lens top-20 readouts for every Part 92 screen unit (real neuron or
random direction) whose answer was labelled a coherent non-assistant persona (PART97-DESIGN.md).

Written by Claude (Opus 5.5), 2026-10-29. Reuses part42's lens helpers and Part 92's random_directions (seed 92000 +
layer, as screen92.py). Usage: PYTHONPATH=. python lens97.py OUT.jsonl
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import torch
import part42
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part92"))
import common92 as c  # noqa: E402

SNAP = Path("/mnt/e/neuron-probe/hf/hub/models--allenai--OLMo-2-0425-1B-Instruct/snapshots/48d788eca847d4d7548f375ad03d3c9312f6139e")
LENS = Path("/mnt/c/Users/alexa/neuron-probe-gpu/results/part57/olmo2-1b/jacobian_lens.pt")
KINDS = {"human", "object or place", "another AI", "abstract", "fictional character", "animal", "deity or spirit"}
L = Path("results/part92/labels-screen")


def population():
    lab = {json.loads(x)["id"]: json.loads(x) for x in open(L / "labels.jsonl", encoding="utf-8")}
    txt = {json.loads(x)["id"]: json.loads(x)["text"] for x in open(L / "items.jsonl", encoding="utf-8")}
    key = json.load(open(L / "KEY-private.json"))
    units = []
    for i, r in lab.items():
        if r.get("coherent") and r.get("kind") in KINDS:
            for u in key[i]:
                units.append({"layer": int(u["layer"]), "kind": u["unit"], "index": int(u["index"]),
                              "answer": txt[i], "persona": r.get("persona"), "persona_kind": r["kind"]})
    return units


def main(out):
    units = population()
    cfg = json.loads((SNAP / "config.json").read_text(encoding="utf-8"))
    _, hidden, n_layers, model_type = part42.config_dimensions(cfg)
    wmap = part42.load_weight_map(SNAP)
    prefix = part42.find_language_model_prefix(wmap, require_language_model=False)
    jac, _, d = part42.load_jacobian_lens(LENS)
    assert d == hidden
    gain = part42.effective_norm_gain(part42.load_safetensor(SNAP, wmap, f"{prefix}norm.weight").float(), model_type)
    tied = bool(cfg.get("tie_word_embeddings", False))
    head = part42.load_safetensor(SNAP, wmap, part42.select_unembedding_key(wmap, prefix, tied))
    tok = part42._load_tokenizer(SNAP)
    by_layer = defaultdict(list)
    for u in units:
        by_layer[u["layer"]].append(u)
    with open(out, "w", encoding="utf-8") as f:
        for layer in sorted(by_layer):
            down = part42.load_safetensor(SNAP, wmap, part42._layer_tensor_key(prefix, layer))
            rand = c.random_directions(down, 256, 92000 + layer, 1.0).float()
            J = jac.get(part42.jacobian_layer_for_write(layer, n_layers)) if part42.jacobian_layer_for_write(layer, n_layers) is not None else None
            for u in by_layer[layer]:
                w = down[:, u["index"]].float() if u["kind"] == "neuron" else rand[u["index"]]
                rd = {}
                for name, jj in (("logit_lens", None), ("j_lens", J)):
                    tb = part42._top_bottom(torch, part42.readout_logits(torch, head, w, gain, jj), tok)
                    rd[name] = [x["decoded"] for x in tb["top"][:20]]
                f.write(json.dumps({**u, **rd}, ensure_ascii=False) + "\n")
            print("layer", layer, len(by_layer[layer]), flush=True)
    print(json.dumps({"units": len(units), "real": sum(u["kind"] == "neuron" for u in units),
                      "random": sum(u["kind"] != "neuron" for u in units)}))


if __name__ == "__main__":
    main(sys.argv[1])
