#!/usr/bin/env python3
"""Part 106b: log-probability of each unit's own SFT greedy answer (first sentence; end-of-text if empty) under its push,
its random push and no push, at one checkpoint (PART106B-DESIGN.md). Reuses Part 106's scorer and pushes.

Written by Claude (Opus 5.5), 2026-10-29. Usage: python part106/run106b.py --ckpt {sft,dpo,instruct} --out OUT.jsonl
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run106 as r  # noqa: E402


def sft_answers():
    rows = [json.loads(l) for l in (r.ROOT / "results/part92/step3/sft_chat.jsonl").read_text(encoding="utf-8").splitlines()]
    return {x["id"]: x["answer"] for x in rows if x["unit"] == "neuron" and int(x["question_index"]) == 0}


def main(argv=None):
    import torch
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, choices=sorted(r.SNAPSHOTS)); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args(argv)
    units, answers = r.load_units(), sft_answers()
    pushes = torch.load(r.ROOT / "results/part99-pushes.pt", map_location="cpu")
    tokenizer, model = r.c.load_model(r.snapshot_dir(args.ckpt), args.device)
    layers = r.c.get_layers(model)
    prompt = r.c.prompt_ids(tokenizer, r.QUESTION, "chat")
    target = {}
    for u in units:
        text = r.first_sentence(answers[u["id"]])
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][: r.MAX_TARGET] if text else [int(tokenizer.eos_token_id)]
        target[u["id"]] = (text, ids)
    started, rows = time.perf_counter(), []
    work = [(u, w) for u in units for w in ("none", "neuron", "random")]
    for layer_no in sorted({int(u["layer"]) for u in units} | {-1}):
        chunk_all = [(u, w) for u, w in work if (w == "none") == (layer_no == -1) and (layer_no == -1 or int(u["layer"]) == layer_no)]
        for b in range(0, len(chunk_all), args.batch):
            chunk = chunk_all[b: b + args.batch]
            items = [(prompt, target[u["id"]][1]) for u, _w in chunk]
            if layer_no == -1:
                scored = r.score_batch(model, tokenizer, items, device=args.device)
            else:
                vec = torch.stack([pushes[f"{u['id']}|{w}"].float() for u, w in chunk]).to(dtype=torch.bfloat16)
                scored = r.score_batch(model, tokenizer, items, layer=layers[layer_no], vectors=vec, device=args.device)
            for (u, w), (tot, mean, first) in zip(chunk, scored):
                rows.append({"ckpt": args.ckpt, "id": u["id"], "kind": u["kind"], "unit": w, "alpha": 0.0 if w == "none" else 0.8,
                             "target": target[u["id"]][0], "n_tokens": len(target[u["id"]][1]), "lp_sum": tot, "lp_mean": mean})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    print(f"wrote {args.out} ({len(rows)} rows) in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
