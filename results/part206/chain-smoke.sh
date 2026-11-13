#!/bin/bash
# Part 206 step 1, the Qwen timing smoke (batch 2, then batch 4), started only after Part 209's main run has finished
# (its OLMo CPU stages hold WSL memory, and an A770 Qwen load needs 11 GB of Windows RAM).
WT=/home/elisepc/green/worktrees/ab-part206
cd "$WT" || exit 1
export ARC_TREE=$WT LAUNCH_ARC_CONFIG=/home/elisepc/green/worktrees/ab-part67/tools/local.json
while systemctl --user is-active --quiet part209-main.scope; do sleep 60; done
echo "$(date +%H:%M:%S) part209-main finished; starting Part 206 Qwen smoke"
Q=E:/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
/home/elisepc/green/Journal/scripts/research/arc-steps-when-ram.sh 206 11 2 \
  "-I -B part206/record206.py --stage qwen-chat --device xpu --model-path $Q --out results/part206/smoke --limit 20 --batch 2 --smoke" \
  "-I -B part206/record206.py --stage qwen-chat --device xpu --model-path $Q --out results/part206/smoke --limit 20 --batch 4 --smoke" \
  -- part206/common206.py part206/record206.py part206/config206.json part206/prompts206.json neuron_report.py \
  || { echo "ARC STEPS FAILED (batch 4 may simply not fit; read step logs)"; exit 3; }
echo ALL-DONE
