# Part 67

Run one prefill-only measurement with the pinned Qwen3.5-4B checkpoint:

```bash
python3 part67/part67.py --device xpu --out results/part67 --batch 32 --max-gpu-gb 20
```

The runner evaluates all 25 prompts under 207 conditions, saves per-position baseline activations, and computes direct
weight paths from the loaded checkpoint. It writes `activations.json`, `weights.json`, and `run.json` in the output
directory. Use `--device cuda` or `--device cpu` where appropriate.

Score a completed run:

```bash
python3 part67/score67.py results/part67
```

This writes `score.json` and `score.md`. Tests can be run with `python3 -m pytest -q part67/test_part67.py`; the tiny-model hook
test skips when PyTorch is not installed.
