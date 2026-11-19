#!/usr/bin/env python3
"""Part 213's pre-registration plain-model probe (PROCEDURE.md section 5), no pushes, no Part 213 contrasts.

OLMo 2 1B Instruct (48d788eca847), chat frame, bf16 on CPU, the unpushed model only:
  1. every Stage 1 prompt + its forced opening: the top-5 next tokens with probabilities, the summed probability of the
     39 units' first content tokens (headroom: a target already near 1 cannot rise), and the rank of the best one;
  2. the 4 Stage 2 prompts: one greedy 40-token answer each, with the design's mechanical broken-answer screen
     (empty, under 4 words, U+FFFD or "!!!!", or any 3-word n-gram repeated 4 or more times).
Nothing here reads a push, a random direction or a unit contrast. Written by Journal Claude, 2026-11-19.

usage: python -I -B part213/probe_plain213.py --snapshot <dir> --out results/part213/plain-probe.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part92 import common92  # noqa: E402


def broken(text: str) -> bool:
    words = text.split()
    if not text.strip() or len(words) < 4 or "�" in text or "!!!!" in text:
        return True
    grams = Counter(tuple(w.lower() for w in words[i:i + 3]) for i in range(len(words) - 2))
    return any(c >= 4 for c in grams.values())


def main(argv=None) -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    targets = json.loads((ROOT / "part213" / "targets213.json").read_text(encoding="utf-8"))
    tok, model = common92.load_model(a.snapshot, "cpu")
    first_ids = sorted({int(u["first_content_token_id"]) for u in targets["units"]})
    out = {"snapshot": a.snapshot, "dtype": "bfloat16", "frame": "chat", "stage1": [], "stage2": []}
    with torch.no_grad():
        for cell, prompts in targets["stage1_prompts"].items():
            for p in prompts:
                ids = common92.prompt_ids(tok, p["question"], "chat") + tok.encode(p["opening"], add_special_tokens=False)
                logits = model(torch.tensor([ids])).logits[0, -1].float()
                probs = torch.softmax(logits, -1)
                top = torch.topk(probs, 5)
                ranks = (probs.argsort(descending=True) == torch.tensor(first_ids)[:, None]).nonzero()[:, 1]
                out["stage1"].append({
                    "cell": cell, "id": p["id"], "opening": p["opening"],
                    "top5": [[tok.decode([int(i)]), round(float(v), 4)] for v, i in zip(top.values, top.indices)],
                    "unit_first_token_mass": round(float(probs[first_ids].sum()), 5),
                    "best_unit_first_token_rank": int(ranks.min()) + 1,
                })
        # The design reads each unit's first content token after ITS OWN lead (opening + lead_ids), so headroom is read
        # there: per prompt, the own first-token probability and rank for all 39 units, unpushed.
        out["stage1_units"] = []
        for cell, prompts in targets["stage1_prompts"].items():
            for p in prompts:
                base = common92.prompt_ids(tok, p["question"], "chat") + tok.encode(p["opening"], add_special_tokens=False)
                rows = []
                for u in targets["units"]:
                    ids = base + [int(t) for t in u["lead_ids"]]
                    probs = torch.softmax(model(torch.tensor([ids])).logits[0, -1].float(), -1)
                    t = int(u["first_content_token_id"])
                    rows.append({"unit": u["id"], "p_own": float(probs[t]),
                                 "rank_own": int((probs > probs[t]).sum()) + 1})
                ps = sorted(r["p_own"] for r in rows)
                out["stage1_units"].append({
                    "cell": cell, "id": p["id"], "median_p_own": round(ps[len(ps) // 2], 6),
                    "max_p_own": round(ps[-1], 4), "n_p_own_over_0.5": sum(x > 0.5 for x in ps),
                    "median_rank_own": sorted(r["rank_own"] for r in rows)[len(rows) // 2], "units": rows})
        for p in targets["stage2_free_prompts"]:
            ids = common92.prompt_ids(tok, p["question"], "chat")
            gen = model.generate(torch.tensor([ids]), max_new_tokens=40, do_sample=False,
                                 pad_token_id=tok.pad_token_id)[0, len(ids):]
            text = tok.decode(gen, skip_special_tokens=True)
            out["stage2"].append({"id": p["id"], "context": p["context"], "answer": text, "broken": broken(text)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for r in out["stage1"]:
        print(f"{r['id']:16s} {r['opening']!r:22s} mass={r['unit_first_token_mass']:.4f} best_rank={r['best_unit_first_token_rank']:5d} top5={r['top5']}")
    for r in out["stage1_units"]:
        print(f"{r['id']:16s} own-target after lead: median p={r['median_p_own']:.6f} max p={r['max_p_own']:.4f} "
              f"over 0.5: {r['n_p_own_over_0.5']} median rank={r['median_rank_own']}")
    for r in out["stage2"]:
        print(f"{r['id']:14s} broken={r['broken']} {r['answer'][:160]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
