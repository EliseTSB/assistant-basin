#!/bin/bash
# Part 206 step 1 again at batch 2 only, into its own folder (the first batch-2 run's smoke-meta.json was overwritten by batch 4).
WT=/home/elisepc/green/worktrees/ab-part206
cd "$WT" || exit 1
export ARC_TREE=$WT LAUNCH_ARC_CONFIG=/home/elisepc/green/worktrees/ab-part67/tools/local.json
Q=E:/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
/home/elisepc/green/Journal/scripts/research/arc-steps-when-ram.sh 206 11 2 \
  "-I -B part206/record206.py --stage qwen-chat --device xpu --model-path $Q --out results/part206/smoke-b2 --limit 20 --batch 2 --smoke" \
  -- part206/common206.py part206/record206.py part206/config206.json part206/prompts206.json neuron_report.py \
  || { echo "ARC STEP FAILED"; exit 3; }
echo ALL-DONE
