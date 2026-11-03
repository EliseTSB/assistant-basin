#!/usr/bin/env python3
"""diag178e.py: on Qwen3.5-4B, do greedy tokens for 4 pilot rows agree across batch sizes 2, 4, 8 and 16 at one
padded width, and does batch 1 differ? diag178d found Qwen3.5-0.8B's batch-16 greedy tokens equal to a CPU float32
reference on all 4 rows while batch 1 on the Arc diverged, at reference gaps up to 4.7 logits. A 4B float32 reference
does not fit in memory here, so this checks agreement between batch sizes. Prints ids-derived counts, no text.
Written by Claude (Opus 5.5), 2026-11-02."""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
from part178 import run178  # noqa: E402

N = 32


def greedy(model, rows_ids, width, pad):
    ids = torch.tensor([[pad] * (width - len(x)) + x for x in rows_ids], device="xpu")
    mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in rows_ids], device="xpu")
    with torch.inference_mode():
        out = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=N,
                             eos_token_id=[248044, 248046], pad_token_id=pad, logits_to_keep=1)
    return out[:, width:].cpu().tolist()


def main():
    tok, model = run178.load_direct("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "xpu")
    pad = tok.pad_token_id
    rows = json.loads((HERE / "problems178.json").read_text(encoding="utf-8"))["pilot"]
    prompts = [run178._prompt_ids(tok, str(r["problem"]), True)["prompt_ids"] for r in rows[:16]]
    width = max(len(p) for p in prompts)
    ref = greedy(model, prompts, width, pad)[:4]
    for size in (8, 4, 2, 1):
        got = []
        for start in range(0, 4, min(size, 4)):
            chunk = prompts[start:start + size] if size >= 4 else prompts[start:start + size]
            got.extend(greedy(model, chunk, width, pad)[:min(size, 4 - start)])
        same = [sum(int(a == b) for a, b in zip(ref[r], got[r])) for r in range(4)]
        print(json.dumps({"batch": size, "vs_batch16_matching_per_row": same}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
