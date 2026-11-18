"""diag_pass2_speed2.py — is the L07-13 pass-two slowness the 7-layer combination, or growth over a long run?

diag_pass2_speed.py (2026-10-24 01:26) measured ONE layer at a time on batches 330-341: layer 10 took
0.16 s/batch (layer 3: 0.10 s), so seven layers should cost ~1.1 s/batch. The live 7-layer run was at
~12 s/batch around the same batches after ~40 minutes, with 7.9 GB committed and a 0.1 GB working set.
This runs a FRESH 7-layer pass two (L07-13, restored from its pass-one state) on batches 330-409, timing
every batch and logging the process's committed (private) bytes, so the two readings separate:
fast from the start -> the live run slowed as it ran (accumulation); slow from the start -> the combination.
Writes nothing the collector reads; prints only.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_collect_fast2 as ac  # noqa: E402
import numpy as np  # noqa: E402


class PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t)]


def mem() -> str:
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return f"private {c.PrivateUsage / 1e9:.2f}GB ws {c.WorkingSetSize / 1e9:.2f}GB faults {c.PageFaultCount}"


def main() -> int:
    layer_list = [7, 8, 9, 10, 11, 12, 13]
    import os
    print("env: " + " ".join(f"{k}={os.environ.get(k)}" for k in (
        "SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE", "UR_L0_USE_COPY_ENGINE",
        "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS", "UR_L0_USE_IMMEDIATE_COMMANDLISTS")), flush=True)
    out = "results/atlas-full-L07-13"
    args = ac.build_parser().parse_args([
        "--data", r"qwen35\pile10k.jsonl", "--device", "xpu", "--batch-size", "4", "--n-controls", "1000",
        "--cache-dir", r"E:\neuron-probe\hf\hub", "--layers", "7,8,9,10,11,12,13", "--out", out])
    units = {layer: list(range(ac.EXPECTED_WIDTH)) for layer in layer_list}
    config = ac._config(args, units)
    load_args = argparse.Namespace(cache_dir=str(args.cache_dir), model=ac.MODEL_ID, revision=ac.MODEL_REVISION,
                                   device="xpu", dtype="bf16", threads=0, random_init_seed=args.random_init_seed)
    torch, tokenizer, model, layers, _snap = ac.part38.load_model(load_args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    docs = ac._read_docs(args.data, tokenizer, None)
    batches = ac.make_batches(docs, 4)
    states = {layer: ac.load_pass_one_checkpoint(Path(out), layer, config, args.seed, args.n_controls, units[layer])
              for layer in layer_list}
    print(f"loaded; {mem()}", flush=True)
    t_all = time.time()
    times = []
    for i in range(330, 410):
        torch.xpu.synchronize()
        t = time.time()
        ac._forward_batches(torch, model.model, layers, states, [batches[i]], "xpu",
                            int(tokenizer.pad_token_id), pass_number=2)
        torch.xpu.synchronize()
        times.append(time.time() - t)
        if (i - 330) % 10 == 9:
            print(f"batches {i - 9}-{i}: median {np.median(times[-10:]):.2f}s max {max(times[-10:]):.2f}s; "
                  f"elapsed {time.time() - t_all:.0f}s; {mem()}", flush=True)
            if time.time() - t_all > 900:
                print("stopping at the 15-minute cap", flush=True)
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
