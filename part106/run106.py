#!/usr/bin/env python3
"""Part 106: log-probability of each Part 92 unit's own persona sentence under its push, its paired random push, and no
push, at one checkpoint (chat frame, "Who are you?"). Teacher-forced; no generation, no labels.

Written by Claude (Opus 5.5), 2026-10-29, from PART106-DESIGN.md.
Usage: python part106/run106.py --ckpt {sft,dpo,instruct} --out results/part106/lp-CKPT.jsonl [--device cpu] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "part92"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
import common92 as c  # noqa: E402

HUB = Path("/mnt/e/neuron-probe/hf/hub")
SNAPSHOTS = {  # Part 89's registry commits (part89/cos89.py)
    "sft": ("models--allenai--OLMo-2-0425-1B-SFT", "0d85a3d03787"),
    "dpo": ("models--allenai--OLMo-2-0425-1B-DPO", "c4b0485961ab"),
    "instruct": ("models--allenai--OLMo-2-0425-1B-Instruct", "48d788eca847"),
}
ALPHAS = {0.4: 0.5, 0.8: 1.0, 1.6: 2.0}  # alpha -> scale on Part 99's alpha-0.8 vectors
QUESTION = "Who are you?"
MAX_TARGET = 32


def snapshot_dir(ckpt: str) -> Path:
    repo, prefix = SNAPSHOTS[ckpt]
    hits = [d for d in (HUB / repo / "snapshots").iterdir() if d.name.startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"{ckpt}: expected one snapshot starting {prefix}, found {len(hits)}")
    return hits[0]


def first_sentence(text: str) -> str:
    """The answer up to and including its first '.', '!' or '?', or up to its first newline."""
    text = text.strip()
    match = re.search(r"[.!?]|\n", text)
    if match is None:
        return text
    end = match.end() if text[match.start()] != "\n" else match.start()
    return text[:end].strip()


def load_units():
    neurons = json.loads((ROOT / "results/part92/step2/neurons.json").read_text(encoding="utf-8"))
    kinds = json.loads((ROOT / "results/part92/score92.json").read_text(encoding="utf-8"))["kinds"]
    units = []
    for row in neurons:
        kind = kinds.get(row["id"])
        if kind in ("slot", "concept"):
            units.append({**row, "kind": kind, "target": first_sentence(row["screen_answer"])})
    return units


def score_batch(model, tokenizer, items, layer=None, vectors=None, device="cpu"):
    """items: list of (prompt_ids, target_ids). Returns [(sum, mean, first)] of target log-probs."""
    import torch

    seqs = [p + t for p, t in items]
    width = max(len(s) for s in seqs)
    pad = int(tokenizer.pad_token_id)
    ids = torch.tensor([s + [pad] * (width - len(s)) for s in seqs], dtype=torch.long, device=device)
    mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], dtype=torch.long, device=device)
    handle = c.add_row_hook(layer, vectors) if vectors is not None else None
    try:
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
    finally:
        if handle is not None:
            handle.remove()
    logp = torch.log_softmax(logits, dim=-1)
    out = []
    for row, (p, t) in enumerate(items):
        positions = torch.arange(len(p) - 1, len(p) + len(t) - 1, device=logp.device)
        values = logp[row, positions, torch.tensor(t, device=logp.device)]
        out.append((float(values.sum()), float(values.mean()), float(values[0])))
    return out


def run(args):
    import torch

    units = load_units()
    if args.limit:
        units = units[: args.limit]
    pushes = torch.load(ROOT / "results/part99-pushes.pt", map_location="cpu")
    tokenizer, model = c.load_model(snapshot_dir(args.ckpt), args.device)
    layers = c.get_layers(model)
    prompt = c.prompt_ids(tokenizer, QUESTION, "chat")
    targets = {u["id"]: tokenizer(u["target"], add_special_tokens=False)["input_ids"][:MAX_TARGET] for u in units}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = []

    def emit(unit, kind_of_push, alpha, scored):
        total, mean, first = scored
        rows.append({"ckpt": args.ckpt, "id": unit["id"], "kind": unit["kind"], "layer": unit["layer"],
                     "neuron": unit["neuron"], "unit": kind_of_push, "alpha": alpha, "target": unit["target"],
                     "n_tokens": len(targets[unit["id"]]), "lp_sum": total, "lp_mean": mean, "lp_first": first})

    for begin in range(0, len(units), args.batch):  # no push
        chunk = units[begin: begin + args.batch]
        for unit, scored in zip(chunk, score_batch(model, tokenizer, [(prompt, targets[u["id"]]) for u in chunk],
                                                   device=args.device)):
            emit(unit, "none", 0.0, scored)
    by_layer = {}
    for unit in units:
        for which in ("neuron", "random"):
            for alpha in ALPHAS:
                by_layer.setdefault(int(unit["layer"]), []).append((unit, which, alpha))
    for layer_no, work in sorted(by_layer.items()):
        for begin in range(0, len(work), args.batch):
            chunk = work[begin: begin + args.batch]
            vectors = torch.stack([pushes[f"{u['id']}|{w}"].float() * ALPHAS[a] for u, w, a in chunk]).to(
                device=args.device, dtype=torch.bfloat16)
            scored = score_batch(model, tokenizer, [(prompt, targets[u["id"]]) for u, _w, _a in chunk],
                                 layer=layers[layer_no], vectors=vectors, device=args.device)
            for (unit, which, alpha), result in zip(chunk, scored):
                emit(unit, which, alpha, result)
        print(f"{args.ckpt}: layer {layer_no} done, {len(rows)} rows, {time.perf_counter() - started:.0f}s", flush=True)
    with out_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out_path} ({len(rows)} rows) in {time.perf_counter() - started:.0f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, choices=sorted(SNAPSHOTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="first N units only (smoke test)")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
