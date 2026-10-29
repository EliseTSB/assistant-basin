#!/usr/bin/env python3
"""Exploratory, not registered (2026-10-29, Elise's question at 17:54 CT): J-lens and logit-lens words for the atlas
neurons left out of Part 112's 465, the ones the atlas reader marked coherent but of kind "unclear" or "still Qwen or an
assistant". Positive side (the screen pushed positive), top 20, with each neuron's screen answer. Reuses Part 111's
readout. Usage: python jlens_excluded113.py OUT_DIR. Written by Claude (Opus 5.5), 2026-10-29.
"""
import gzip
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "part111"))
import jlens111 as j  # noqa: E402

LABELS = Path.home() / "jlens-persona-neurons/data/qwen3.5-4b/labels"
SCREEN = HERE.parent / "results/part86/screen-final"


def excluded():
    key = json.loads((LABELS / "KEY-private.json").read_text(encoding="utf-8"))
    out = []
    for r in map(json.loads, open(LABELS / "labels.jsonl", encoding="utf-8")):
        if r.get("coherent") and r.get("kind") in ("unclear", "still Qwen or an assistant"):
            for k in key.get(r["id"], []):
                if k["unit"] == "neuron":
                    out.append({"layer": int(k["layer"]), "neuron": int(k["index"]), "persona": r["persona"],
                                "kind": r["kind"]})
    return sorted(out, key=lambda u: (u["layer"], u["neuron"]))


def answers(units):
    want = {(u["layer"], u["neuron"]) for u in units}
    found = {}
    for layer in sorted({u["layer"] for u in units}):
        for line in gzip.open(SCREEN / f"layer_{layer}.jsonl.gz", "rt", encoding="utf-8"):
            r = json.loads(line)
            if r["kind"] == "neuron" and (layer, int(r["id"])) in want:
                found[(layer, int(r["id"]))] = r["answer"]
    return found


def main(out_dir: Path):
    t0 = time.time()
    units = excluded()
    ans = answers(units)
    w = j.Weights(j.SNAP)
    lens = torch.load(str(j.LENS), map_location="cpu", weights_only=True)
    gain = 1.0 + w.get(j.PREFIX + "norm.weight").float()
    embed = w.get(j.PREFIX + "embed_tokens.weight")
    tok = j.load_tok()
    for layer in sorted({u["layer"] for u in units}):
        members = [u for u in units if u["layer"] == layer]
        down = w.get(f"{j.PREFIX}layers.{layer}.mlp.down_proj.weight").float()
        writes = torch.stack([down[:, u["neuron"]] for u in members], dim=1)
        for lens_name, mat in (("j_lens", lens["J"][layer].float()), ("logit_lens", None)):
            top, _ = j.top_both(embed, gain.unsqueeze(1) * (mat @ writes if mat is not None else writes))
            for i, u in enumerate(members):
                u[lens_name] = j.decode(tok, top[:, i].tolist())
        for u in members:
            u["answer"] = ans.get((u["layer"], u["neuron"]))
        print(f"layer {layer}: {len(members)} done {time.time() - t0:.0f}s", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "jlens_excluded113.json").write_text(json.dumps(units, indent=1, ensure_ascii=False), encoding="utf-8")
    print("DONE", len(units), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
