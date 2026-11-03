#!/usr/bin/env python3
"""diag178.py: where the Arc's memory and time go when run178 generates on Qwen3.5-0.8B.

Elise, 2026-11-02 21:04: "Why does the current pilot take nearly all the memory on the arc with a .8B model?"
Prints device memory (allocated, reserved, peak, device free) after load and after short generations at batch 16 and
64, and the share of each generation's time spent in the CPU sampler. Pilot problems only; prints no text.
Written by Claude (Opus 5.5), 2026-11-02.
"""
import json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
from transformers import LogitsProcessorList  # noqa: E402
from part178 import run178  # noqa: E402

GB = 1024 ** 3


def mem(tag, extra=None):
    free, total = torch.xpu.mem_get_info()
    row = {"tag": tag, "allocated_gb": round(torch.xpu.memory_allocated() / GB, 3),
           "reserved_gb": round(torch.xpu.memory_reserved() / GB, 3),
           "peak_allocated_gb": round(torch.xpu.max_memory_allocated() / GB, 3),
           "device_free_gb": round(free / GB, 3), "device_total_gb": round(total / GB, 3)}
    row.update(extra or {})
    print(json.dumps(row), flush=True)


class Timed:
    def __init__(self, inner):
        self.inner, self.seconds, self.calls = inner, 0.0, 0

    def __call__(self, input_ids, scores):
        start = time.perf_counter()
        out = self.inner(input_ids, scores)
        self.seconds += time.perf_counter() - start
        self.calls += 1
        return out


def main():
    model_id, revision = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
    tok, model = run178.load_direct(model_id, revision, "xpu")
    mem("after load", {"param_gb": round(sum(p.numel() * p.element_size() for p in model.parameters()) / GB, 3),
                       "dtype": str(next(model.parameters()).dtype)})
    rows = json.loads((HERE / "problems178.json").read_text(encoding="utf-8"))["pilot"]
    eos = [248044, 248046]
    pad = tok.pad_token_id
    for batch, new_tokens in ((16, 64), (16, 512), (64, 256)):
        torch.xpu.empty_cache()
        torch.xpu.reset_peak_memory_stats()
        ids = [run178._prompt_ids(tok, str(rows[i % len(rows)]["problem"]), True)["prompt_ids"] for i in range(batch)]
        width = max(len(x) for x in ids)
        input_ids = torch.tensor([[pad] * (width - len(x)) + x for x in ids], device="xpu")
        mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in ids], device="xpu")
        gens = [torch.Generator(device="cpu").manual_seed(1000 + i) for i in range(batch)]
        proc = Timed(run178.OrderedCpuSampler(gens, width, eos, pad))
        torch.xpu.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(input_ids=input_ids, attention_mask=mask, do_sample=False, max_new_tokens=new_tokens,
                                 eos_token_id=eos, pad_token_id=pad, logits_processor=LogitsProcessorList([proc]),
                                 logits_to_keep=1)
        torch.xpu.synchronize()
        wall = time.perf_counter() - start
        steps = int(out.shape[1]) - width
        mem(f"batch {batch}, {new_tokens} new", {"prompt_width": width, "steps": steps, "wall_s": round(wall, 2),
                                                 "sampler_s": round(proc.seconds, 2),
                                                 "tokens_per_s": round(batch * steps / wall, 1)})
        del out, input_ids, mask
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
