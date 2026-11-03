#!/usr/bin/env python3
"""diag178c.py: does device memory grow during a long generate on Qwen3.5-0.8B, and does a periodic
torch.xpu.empty_cache() stop it? The 21:21 pilot at batch 64 and cap 2,048 drove Windows' free RAM under the 2 GB
guard in 8 minutes on a 1.4 GB model, which looks like the Arc spilling into shared memory.
Prints device memory every 128 steps, without and then with empty_cache every 64 steps. Pilot problems; no text.
Written by Claude (Opus 5.5), 2026-11-02."""
import json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
from transformers import LogitsProcessorList  # noqa: E402
from part178 import run178  # noqa: E402

GB = 1024 ** 3


class Watch:
    def __init__(self, inner, empty_every, tag):
        self.inner, self.empty_every, self.tag, self.step = inner, empty_every, tag, 0

    def __call__(self, input_ids, scores):
        self.step += 1
        if self.empty_every and self.step % self.empty_every == 0:
            torch.xpu.empty_cache()
        if self.step % 128 == 0:
            free, _ = torch.xpu.mem_get_info()
            print(json.dumps({"run": self.tag, "step": self.step,
                              "allocated_gb": round(torch.xpu.memory_allocated() / GB, 3),
                              "reserved_gb": round(torch.xpu.memory_reserved() / GB, 3),
                              "device_free_gb": round(free / GB, 3)}), flush=True)
        return self.inner(input_ids, scores)


def main():
    batch, steps = int(sys.argv[1]), int(sys.argv[2])
    tok, model = run178.load_direct("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17", "xpu")
    rows = json.loads((HERE / "problems178.json").read_text(encoding="utf-8"))["pilot"]
    eos, pad = [248044, 248046], tok.pad_token_id
    ids = [run178._prompt_ids(tok, str(rows[i % len(rows)]["problem"]), True)["prompt_ids"] for i in range(batch)]
    width = max(len(x) for x in ids)
    order = (("plain", 0), ("empty_cache/64", 64))
    if len(sys.argv) > 3 and sys.argv[3] == "rev":
        order = order[::-1]
    for tag, every in order:
        torch.xpu.empty_cache()
        torch.xpu.reset_peak_memory_stats()
        input_ids = torch.tensor([[pad] * (width - len(x)) + x for x in ids], device="xpu")
        mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in ids], device="xpu")
        gens = [torch.Generator(device="cpu").manual_seed(2000 + i) for i in range(batch)]
        proc = Watch(run178.OrderedCpuSampler(gens, width, eos, pad), every, tag)
        start = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(input_ids=input_ids, attention_mask=mask, do_sample=False, max_new_tokens=steps,
                                 eos_token_id=eos, pad_token_id=pad,
                                 logits_processor=LogitsProcessorList([proc]), logits_to_keep=1)
        torch.xpu.synchronize()
        wall = time.perf_counter() - start
        print(json.dumps({"run": tag, "done_steps": int(out.shape[1]) - width, "wall_s": round(wall, 1),
                          "tokens_per_s": round(batch * (int(out.shape[1]) - width) / wall, 1),
                          "peak_allocated_gb": round(torch.xpu.max_memory_allocated() / GB, 3)}), flush=True)
        del out, input_ids, mask, proc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
