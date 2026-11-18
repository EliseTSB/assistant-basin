"""diag_sync_stall.py — does a plain GPU->CPU sync on the Arc sometimes block for minutes?

2026-10-24: atlas pass two for layers 7-13 stalls for 100-180 s at a time, about once per 20-30 batches,
at Normal and BelowNormal priority alike. During a stall the process uses no CPU, takes no page faults,
does no I/O, and NO process uses any Arc engine (sampled 01:45). One layer at a time (~500 syncs in all)
never stalled; a 7-layer batch makes a few hundred syncs. Hypothesis: a rare missed completion that the
runtime recovers from on a timeout, i.e. a per-sync stall probability around 1e-4, independent of the
collector. This loop does nothing but small GPU work ending in .cpu(), shaped like the collector's sparse
step, and reports how many iterations took over 5 s and how long they took.
"""
from __future__ import annotations

import sys
import time

import torch

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
VARY = len(sys.argv) > 2 and sys.argv[2] == "vary"  # change tensor sizes every iteration (allocator churn)
CAP_S = 900


def main() -> int:
    dev = "xpu"
    x = torch.randn(544, 1000, device=dev)
    thr = torch.full((1000,), 0.5, device=dev)
    torch.xpu.synchronize()
    t_all = time.time()
    slow = []
    times = []
    gen = torch.Generator().manual_seed(0)
    for i in range(N):
        if VARY:
            rows = int(torch.randint(100, 2049, (1,), generator=gen))
            x = torch.randn(rows, 1000, device=dev) * (0.5 + (i % 7) / 7)
        t = time.time()
        q = x > thr.unsqueeze(0)
        idx = torch.nonzero(q.reshape(-1), as_tuple=False).flatten()   # sync 1 (size)
        vals = x.reshape(-1).index_select(0, idx).cpu().numpy()         # sync 2
        dt = time.time() - t
        times.append(dt)
        if dt > 5:
            slow.append((i, round(dt, 1)))
            print(f"iter {i}: {dt:.1f}s (elapsed {time.time() - t_all:.0f}s)", flush=True)
        if i % 5000 == 4999:
            s = sorted(times)
            print(f"{i + 1} iters, {2 * (i + 1)} syncs: median {s[len(s) // 2] * 1e3:.2f}ms, "
                  f"p99.9 {s[int(len(s) * 0.999)] * 1e3:.1f}ms, slow {len(slow)}, "
                  f"elapsed {time.time() - t_all:.0f}s", flush=True)
        if time.time() - t_all > CAP_S:
            print(f"stopping at the {CAP_S}s cap after {i + 1} iterations", flush=True)
            break
    print(f"done: {len(times)} iters, {2 * len(times)} syncs, slow(>5s) {slow}, "
          f"total {time.time() - t_all:.0f}s, vals per iter {vals.size}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
