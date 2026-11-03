#!/usr/bin/env python3
"""diag178g: are Part 178's actual batch-16, left-padded, masked runs affected by the Arc's bf16 SDPA decode fault?

A diagnostic, not a registered part; it changes no registered number. diag178f found Arc bf16 SDPA leaving the CPU's
greedy path once the cache passes 64 keys, on identical unpadded rows at batch 1 and 2. Part 178 decoded left-padded
batches of 16 DIFFERENT problems with an attention mask, which may take another SDPA path, and an earlier pilot
(diag178d) had batch 16 matching a CPU float32 reference over 32 tokens. This mirrors run178's prompt and padding
(`run178._prompt_ids`, `run178._padded_inputs`) on the 16 pilot problems in thinking mode, greedy, 256 new tokens.

Configs (one per invocation of --config): `fp32-sdpa-b16` (reference), `bf16-sdpa-b16` (Part 178's original path),
`bf16-eager-b16` (Amendment 1's path), `bf16-sdpa-b1` (each row alone; positive control, expected to diverge), for
--model small or big. --device cpu runs the same on the CPU (WSL) as an outside reference.
Per row and step it records the chosen token and the top-two logit margin. No logits are stored.
Writes results/part178/diag178g/<model>-<config>-<device>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part178 import run178  # noqa: E402

MAXNEW = 256


def load(model_id, rev, dtype, attn, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers.modeling_utils as modeling_utils
    if device == "xpu" and hasattr(modeling_utils, "caching_allocator_warmup"):
        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    kw = {"revision": rev, "dtype": dt, "attn_implementation": attn}
    if device == "xpu":
        kw["device_map"] = {"": "xpu"}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
    model.eval()
    model.requires_grad_(False)
    return tok, model


class Recorder:
    """A logits processor that records each row's argmax and top-two margin, then changes nothing."""

    def __init__(self):
        self.steps = []

    def __call__(self, input_ids, scores):
        import torch
        top = scores.detach().float().topk(2, dim=-1)
        self.steps.append((top.indices[:, 0].cpu().tolist(), (top.values[:, 0] - top.values[:, 1]).cpu().tolist(),
                           bool(torch.isfinite(scores).all())))
        return scores


def run_batch(model, tok, states, device):
    import torch
    from transformers import LogitsProcessorList
    width = max(len(s["prompt_ids"]) for s in states)
    inputs = run178._padded_inputs(states, width, device)
    rec = Recorder()
    t = time.perf_counter()
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=MAXNEW, min_new_tokens=MAXNEW, do_sample=False, use_cache=True,
                       pad_token_id=states[0]["pad_id"], logits_processor=LogitsProcessorList([rec]))
    if device == "xpu":
        torch.xpu.synchronize()
    el = time.perf_counter() - t
    out = []
    for b in range(len(states)):
        out.append({"ids": [step[0][b] for step in rec.steps], "margins": [round(step[1][b], 4) for step in rec.steps],
                    "finite": all(step[2] for step in rec.steps)})
    return out, el


def main(argv=None):
    import torch
    import transformers
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("small", "big"), required=True)
    p.add_argument("--config", choices=("fp32-sdpa-b16", "bf16-sdpa-b16", "bf16-eager-b16", "bf16-sdpa-b1"),
                   required=True)
    p.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "part178" / "diag178g")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    model_id, rev = ((run178.MODEL_SMALL, run178.REVISION_SMALL) if a.model == "small"
                     else (run178.MODEL_LARGE, run178.REVISION_LARGE))
    dtype, attn, b = a.config.split("-")
    if a.device == "cpu":
        torch.set_num_threads(16)
    tok, model = load(model_id, rev, dtype, attn, a.device)
    problems = json.loads((HERE / "problems178.json").read_text(encoding="utf-8"))["pilot"]
    states = []
    for row in problems:
        payload = run178._prompt_ids(tok, row["problem"], True)
        states.append({"problem_id": row["problem_id"], "prompt_ids": payload["prompt_ids"],
                       "pad_id": int(tok.pad_token_id)})
    results = []
    if b == "b16":
        rows, el = run_batch(model, tok, states, a.device)
        results = [(s, r, el) for s, r in zip(states, rows)]
    else:
        for s in states:
            rows, el = run_batch(model, tok, [s], a.device)
            results.append((s, rows[0], el))
    path = a.out / f"{a.model}-{a.config}-{a.device}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for s, r, el in results:
            f.write(json.dumps({"model": model_id, "config": a.config, "device": a.device,
                                "problem_id": s["problem_id"], "prompt_len": len(s["prompt_ids"]),
                                "elapsed_s": round(el, 2), **r,
                                "versions": {"torch": torch.__version__,
                                             "transformers": transformers.__version__}}) + "\n")
    print("diag178g done", a.model, a.config, a.device, path.name, flush=True)


if __name__ == "__main__":
    main()
