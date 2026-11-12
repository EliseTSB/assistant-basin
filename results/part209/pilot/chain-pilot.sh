#!/bin/bash
# Part 209 pilot (steps.txt 1a), one A770 step at a time via arc-steps-when-ram.sh (11 GB, 2 attempts each).
export ARC_TREE=/home/elisepc/green/worktrees/ab-part203
export LAUNCH_ARC_CONFIG=/home/elisepc/green/worktrees/ab-part67/tools/local.json
exec /home/elisepc/green/Journal/scripts/research/arc-steps-when-ram.sh 209 11 2 \
  "-I -B part209/run209.py --stage qwen-chat --pilot --out results/part209/pilot --device xpu --models-json results/part209/pilot/models.json" \
  "-I -B part209/run209.py --stage qwen-chat --pilot --r2 --batch 16 --out results/part209/pilot --device xpu --models-json results/part209/pilot/models.json" \
  "-I -B part209/run209.py --stage qwen-chat --pilot --r2 --batch 32 --out results/part209/pilot/b32 --device xpu --models-json results/part209/pilot/models.json" \
  -- part209/run209.py part209/config.json part209/prompts209.json part209/reference209.json part209/tokens.json neuron_report.py
