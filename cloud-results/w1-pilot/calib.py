"""Step 1 of the build: plain greedy answers to the calibration prompts, then a teacher-forced pass that saves
the MLP input x (post-attention RMSNorm output) at every position for layers 16-31.
Writes calib_answers.jsonl here and the activation cache to $W1_CACHE (default ./cache, not committed)."""
import json
import os
import time
from pathlib import Path

import torch

import common as c
from sets import CALS, check_disjoint

LAYERS = list(range(16, 32))


def main():
    check_disjoint()
    cache = Path(os.environ.get("W1_CACHE", c.HERE / "cache"))
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok, model, layers, _ = c.load()
    qs = [(k, q) for k, s in CALS.items() for q in s]
    ids = c.generate(tok, model, [q for _, q in qs], log=print)
    rows = []
    with open(c.HERE / "calib_answers.jsonl", "w") as f:
        for (k, q), a in zip(qs, ids):
            r = {"set": k, "q": q, "ids": a, "text": tok.decode(a, skip_special_tokens=True)}
            rows.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("generated", time.time() - t0)
    caps = {}
    hs = []
    for li in LAYERS:
        def pre(_m, args, li=li):
            caps[li] = args[0][0].detach().float().clone()
        hs.append(layers[li].mlp.register_forward_pre_hook(pre))
    store = {li: [] for li in LAYERS}
    meta = []
    with torch.inference_mode():
        for r in rows:
            p = tok(c.build_prompt(tok, r["q"]), add_special_tokens=False)["input_ids"]
            full = p + r["ids"]
            model(input_ids=torch.tensor([full]), use_cache=False)
            for li in LAYERS:
                store[li].append(caps[li])
            meta.append({"set": r["set"], "q": r["q"], "n_prompt": len(p), "full": full})
    for h in hs:
        h.remove()
    torch.save({"layers": LAYERS, "acts": {li: torch.cat(store[li]) for li in LAYERS}, "meta": meta},
               cache / "calib_acts.pt")
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
