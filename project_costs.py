#!/usr/bin/env python3
"""Print the report's cost table from a fitted ``fit.json`` (reproducible projection).

CPU column: the fit's measured median seconds per document (both lenses, all
layers, 128 tokens) — architecture-bound, so every OLMo 2 1B checkpoint has the
same per-document cost.

3090 column: projected, not measured. Method:
  H100 Qwen3-8B 11.36 s (R) / 11.56 s (S) per document (Neuron Lens'
  data/summary/fit_bench_h100_2026-10-28.log);
  per-document FLOP ratio OLMo 2 1B : Qwen3-8B =
  16*(3*2048*8192 + 4*2048**2) / 36*(3*4096*12288 + 4*4096**2) = 0.1367;
  H100 -> 3090 factor 5..7 (dense bf16 peak 989 vs 142 TFLOP/s, HBM 3.35 vs
  0.94 TB/s, against their achieved 670 TFLOP/s = 68% of H100 peak).
"""
import argparse
import json
from pathlib import Path

STAGES = [
    ("Base (main)", "a1847dff"),
    ("pretraining, 21B tokens", "stage1-step10000-tokens21B, 22f7d18a"),
    ("pretraining, 210B", "stage1-step100000-tokens210B, eb19ccab"),
    ("pretraining, 1.0T", "stage1-step480000-tokens1007B, eda09f03"),
    ("pretraining, 2.0T", "stage1-step950000-tokens1993B, 3e09b708"),
    ("pretraining, 3.0T", "stage1-step1430000-tokens2999B, 22738a97"),
    ("end of pretraining, 4.0T", "stage1-step1907359-tokens4001B, 9d3e4365"),
    ("mid-training, 13B", "stage2-ingredient3-step6000-tokens13B, a7a789f6"),
    ("mid-training, 26B", "stage2-ingredient3-step12000-tokens26B, 1c3599f7"),
    ("mid-training end, 51B", "stage2-ingredient3-step23852-tokens51B, f83cd641"),
    ("SFT", "-SFT"),
    ("DPO", "-DPO"),
    ("Instruct", "-Instruct, 48d788ec"),
]
FLOP_RATIO = 16 * (3 * 2048 * 8192 + 4 * 2048 ** 2) / (36 * (3 * 4096 * 12288 + 4 * 4096 ** 2))
H100_8B = {"R": 11.36, "S": 11.56}
H100_TO_3090 = (5.0, 7.0)


def h(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=Path, required=True)
    ap.add_argument("--docs", type=int, nargs="+", default=[200, 1000])
    args = ap.parse_args()
    meta = json.loads(args.fit.read_text())
    cpu = float(meta["seconds_median"])
    gpu_lens = tuple(H100_8B[k] * FLOP_RATIO for k in ("R", "S"))
    gpu_lo = sum(gpu_lens) * H100_TO_3090[0]
    gpu_hi = sum(gpu_lens) * H100_TO_3090[1]
    print(f"# FLOP ratio OLMo1B/Qwen3-8B = {FLOP_RATIO:.4f}")
    print(f"# H100 projected per lens: R {gpu_lens[0]:.2f} s, S {gpu_lens[1]:.2f} s; "
          f"both {sum(gpu_lens):.2f} s; x{H100_TO_3090[0]:.0f}-{H100_TO_3090[1]:.0f} on a 3090 "
          f"= {gpu_lo:.1f}-{gpu_hi:.1f} s")
    print(f"# CPU measured: {cpu:.1f} s/doc both lenses")
    print()
    header = "| stage | revision | " + " | ".join(
        f"{n} docs, CPU" for n in args.docs) + " | " + " | ".join(
        f"{n} docs, 3090" for n in args.docs) + " |"
    print(header)
    print("|" + "---|" * (2 + 2 * len(args.docs)))
    for name, rev in STAGES:
        cells = [h(cpu * n) for n in args.docs] + [f"{h(gpu_lo * n)}–{h(gpu_hi * n)}" for n in args.docs]
        print(f"| {name} | {rev} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
