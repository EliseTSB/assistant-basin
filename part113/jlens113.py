#!/usr/bin/env python3
"""Part 113 (PART113-DESIGN.md): L21 N1713's write read through the Jacobian lens and the logit lens, both signs, with
500 random layer-21 neurons (each read with both signs) as the null for the registered "name" family. Reuses Part
111's readout. Usage: python jlens113.py OUT_DIR. Written by Claude (Opus 5.5), 2026-10-29.
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "part111"))
import jlens111 as j  # noqa: E402

LAYER, NEURON, N_RANDOM, SEED = 21, 1713, 500, 113500
NAME = re.compile(r"^(name|names|named|nameless|unnamed|anonymous|anonym|nobody|unknown|identity|exist|exists|existence)$")


def has_name(tokens):
    return [t for t in tokens if NAME.search(t.strip().lower())]


def main(out_dir: Path):
    t0 = time.time()
    w = j.Weights(j.SNAP)
    lens = torch.load(str(j.LENS), map_location="cpu", weights_only=True)
    gain = 1.0 + w.get(j.PREFIX + "norm.weight").float()
    embed = w.get(j.PREFIX + "embed_tokens.weight")
    tok = j.load_tok()
    down = w.get(f"{j.PREFIX}layers.{LAYER}.mlp.down_proj.weight").float()
    rng = np.random.default_rng(SEED)
    rand = [int(x) for x in rng.choice([u for u in range(down.shape[1]) if u != NEURON], size=N_RANDOM, replace=False)]
    writes = torch.stack([down[:, NEURON]] + [down[:, u] for u in rand], dim=1)
    result = {"design": "PART113-DESIGN.md", "layer": LAYER, "neuron": NEURON, "name_pattern": NAME.pattern}
    for lens_name, mat in (("j_lens", lens["J"][LAYER].float()), ("logit_lens", None)):
        vec = gain.unsqueeze(1) * (mat @ writes if mat is not None else writes)
        top, low = j.top_both(embed, vec)
        plus, minus = j.decode(tok, top[:, 0].tolist()), j.decode(tok, low[:, 0].tolist())
        null = sum(bool(has_name(j.decode(tok, ids.tolist()))) for k in range(1, N_RANDOM + 1) for ids in (top[:, k], low[:, k]))
        result[lens_name] = {"plus_top20": plus, "minus_top20": minus, "plus_name_hits": has_name(plus),
                             "minus_name_hits": has_name(minus), "null_rate": null / (2 * N_RANDOM)}
        print(lens_name, "+", plus[:12], "|", "-", minus[:12], "| null", null / (2 * N_RANDOM), flush=True)
    result["elapsed_s"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "jlens113.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: jlens113.py OUT_DIR")
    main(Path(sys.argv[1]))
