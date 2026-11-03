#!/usr/bin/env python3
"""diag178d.py: is batched greedy generation on the Arc faithful? The 4B pilot's batch check matched only 15 of 128
greedy tokens between batch 1 and batch 16 at the same padded width. Here Qwen3.5-0.8B greedy-decodes 4 pilot
problems for 32 tokens five ways: CPU float32 alone (reference), XPU bf16 alone unpadded, alone left-padded to the
batch width, inside a batch of 16, and CPU bf16 alone. For each, the first position that differs from the reference
and the reference's top-1 minus top-2 logit gap there. A tiny gap means a near-tie flipped; a large one means the
path computes something else. Pilot problems; prints token ids and gaps only, no text.
Written by Claude (Opus 5.5), 2026-11-02."""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
from part178 import run178  # noqa: E402

MODEL, REV, N = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17", 32


def greedy(model, rows_ids, width, device, pad):
    ids = torch.tensor([[pad] * (width - len(x)) + x for x in rows_ids], device=device)
    mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in rows_ids], device=device)
    with torch.inference_mode():
        out = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=N,
                             eos_token_id=[248044, 248046], pad_token_id=pad, output_scores=True,
                             return_dict_in_generate=True, logits_to_keep=1)
    seqs = out.sequences[:, width:].cpu().tolist()
    gaps = []
    for r in range(len(rows_ids)):
        g = []
        for s in out.scores:
            top = torch.topk(s[r].float().cpu(), 2).values
            g.append(round(float(top[0] - top[1]), 4))
        gaps.append(g)
    return seqs, gaps


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def main():
    tok, xpu = run178.load_direct(MODEL, REV, "xpu")
    pad = tok.pad_token_id
    rows = json.loads((HERE / "problems178.json").read_text(encoding="utf-8"))["pilot"]
    prompts = [run178._prompt_ids(tok, str(r["problem"]), True)["prompt_ids"] for r in rows[:16]]
    width = max(len(p) for p in prompts)
    ref, ref_gaps = [], []
    cpu = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.float32).eval()
    for p in prompts[:4]:
        s, g = greedy(cpu, [p], len(p), "cpu", pad)
        ref.append(s[0]); ref_gaps.append(g[0])
    del cpu
    cpu16 = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16).eval()
    arms = {"cpu_bf16_alone": [greedy(cpu16, [p], len(p), "cpu", pad)[0][0] for p in prompts[:4]]}
    del cpu16
    arms["xpu_alone_unpadded"] = [greedy(xpu, [p], len(p), "xpu", pad)[0][0] for p in prompts[:4]]
    arms["xpu_alone_padded"] = [greedy(xpu, [p], width, "xpu", pad)[0][0] for p in prompts[:4]]
    arms["xpu_batch16_padded"] = greedy(xpu, prompts, width, "xpu", pad)[0][:4]
    for name, seqs in arms.items():
        rows_out = []
        for r in range(4):
            d = first_diff(ref[r], seqs[r])
            rows_out.append({"row": r, "first_diff": d, "ref_gap_there": None if d is None else ref_gaps[r][d],
                             "matching": sum(int(a == b) for a, b in zip(ref[r], seqs[r]))})
        print(json.dumps({"arm": name, "width": width, "rows": rows_out}), flush=True)
    print(json.dumps({"ref_min_gap_first8": [min(g[:8]) for g in ref_gaps]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
