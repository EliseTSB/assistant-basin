# Fold 1: Part 43 FP32 successor (branch part43-fp32, worktree /home/elisepc/green/worktrees/ab-43-fp32)

Size: code and offline tests only. No model download, no GPU, no network.

Read `review/part43/REVIEW-fp32-r1-luna.md` and `review/part43/REVIEW-fp32-r1-haiku.md` in full, plus
`review/part43/FP32-SUCCESSOR-PROTOCOL.md`. Fix every INTRODUCED finding with the smallest change that closes it:
- Luna MEDIUM: `run_pilot()` must take the protocol from the calibration it was given, not from args, and refuse a
  model whose recorded dtype does not match that protocol.
- Haiku F1: under `--protocol fp32-successor`, every subcommand other than `calibrate` and `pilot` refuses (it is a
  bounded pilot protocol; nothing else runs ungated under it).
- Haiku F2: record the Qwen3.5 implementations actually used (hub kernels on or off, kernelized classes, the
  gated-delta / linear-attention and conv implementation names) in the FP32 runtime; the reader refuses an fp32
  artifact that records hub kernels enabled, any kernelized class, or a non-torch-reference implementation, unless the
  registration names an allowance (none is named now).
- Haiku F3: dtype evidence for every layer's down_proj input (all 32), not 4; the reader checks all are float32.
- Haiku F4: stamp the protocol block only after `_configure_full_fp32_precision()` has run.
- Haiku F5: make each listed always-green test able to fail (show the sabotage that now reds it).
- Haiku F6, F7: protocol-aware second gate; `--dtype` that contradicts the protocol refuses.
Leave PRE-EXISTING items alone, but list them in your report.

Each new or changed test must go red under a targeted sabotage in a detached worktree (a plain copy reds 34 at
baseline). `/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q test_part43.py` green. Do not commit; write
`REPORT-43-fp32-fold1.md`: SUMMARY, finding -> fix -> test -> sabotage table, anything not fixed and why, BLOCKERS.
