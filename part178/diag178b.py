#!/usr/bin/env python3
"""diag178b.py: can the presence penalty and top-k run on the Arc? Times torch.topk(k=20) on random
(batch, 248,320) logits on the XPU against the prior CPU reference path, and checks both pick the same set.
Written by Claude (Opus 5.5), 2026-11-02, after diag178 put 57% of batch-64 time in the CPU sampler."""
import json, time
import torch

V, K = 248320, 20
for batch in (16, 64):
    torch.manual_seed(batch)
    scores = torch.randn(batch, V, dtype=torch.bfloat16, device="xpu") * 4
    seen = torch.zeros(batch, V, dtype=torch.bool, device="xpu")
    seen[:, torch.randint(0, V, (batch, 800), device="xpu")] = True
    def xpu_path():
        x = scores.float() - 1.5 * seen.float()
        v, i = torch.topk(x, K, dim=-1)
        return v.cpu(), i.cpu()
    def cpu_path():
        x = scores.to("cpu", dtype=torch.float32) - 1.5 * seen.to("cpu").float()
        return torch.topk(x, K, dim=-1)
    for name, fn in (("xpu", xpu_path), ("cpu", cpu_path)):
        fn(); torch.xpu.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            out = fn()
        torch.xpu.synchronize()
        ms = (time.perf_counter() - start) / 20 * 1000
        print(json.dumps({"batch": batch, "path": name, "ms_per_step": round(ms, 2)}), flush=True)
    a, b = xpu_path(), cpu_path()
    same = sum(set(a[1][r].tolist()) == set(b[1][r].tolist()) for r in range(batch))
    print(json.dumps({"batch": batch, "rows_with_same_top20": same}), flush=True)
