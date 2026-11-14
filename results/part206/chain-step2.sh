#!/bin/bash
# Part 206 step 2 (steps.txt; Amendment 2: Qwen at batch 4). Qwen on the A770 first (four steps, one at a time, each
# after Windows has 11 GB), copy its results back, then the six OLMo stages on the CPU one at a time (never beside an
# Arc load). Each OLMo stage runs its moments pass, then GT's causal pass.
WT=/home/elisepc/green/worktrees/ab-part206
cd "$WT" || exit 1
export ARC_TREE=$WT LAUNCH_ARC_CONFIG=/home/elisepc/green/worktrees/ab-part67/tools/local.json
QB=E:/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b
QC=E:/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
R="-I -B part206/record206.py --device xpu --out results/part206 --batch 4"
/home/elisepc/green/Journal/scripts/research/arc-steps-when-ram.sh 206 11 2 \
  "$R --stage qwen-base --model-path $QB" \
  "$R --stage qwen-base --model-path $QB --pass gt" \
  "$R --stage qwen-chat --model-path $QC" \
  "$R --stage qwen-chat --model-path $QC --pass gt" \
  -- part206/common206.py part206/record206.py part206/config206.json part206/prompts206.json neuron_report.py \
  || { echo "ARC STEPS FAILED"; exit 3; }
J=/mnt/c/Users/alexa/neuron-probe-gpu/results/part206
for d in qwen-base qwen-chat; do cp -r "$J/$d" results/part206/ || { echo "COPY FAILED: $d"; exit 5; }; done
echo "$(date +%H:%M:%S) qwen results copied back: $(ls results/part206 | tr '\n' ' ')"
PY='env HF_HOME=/mnt/e/neuron-probe/hf HF_HUB_OFFLINE=1 /home/elisepc/neuron-probe/.venv/bin/python -I -B part206/record206.py'
for spec in "pretrain-3.0T|pretrain" "midtrain-26B|midtrain26" "midtrain-end|midtrainend" "sft|sft" "dpo|dpo" "instruct|instruct"; do
  IFS='|' read -r STAGE TAG <<< "$spec"
  OUT=$(python3 -I -B tools/launch_arc.py 206 --cpu "$PY --stage $STAGE --out results/part206 --batch 8 && $PY --stage $STAGE --out results/part206 --pass gt" --mem 9G --tag "$TAG" 2>&1)
  U=$(echo "$OUT" | grep -oE "unit part206-$TAG-[0-9]+" | head -1 | cut -d' ' -f2)
  echo "$(date +%H:%M:%S) $STAGE $TAG unit=$U"
  [ -z "$U" ] && { echo "$OUT" | tail -5; echo "CPU STAGE FAILED TO START: $TAG"; exit 4; }
  while systemctl --user is-active --quiet "$U"; do sleep 30; done
  echo "$(date +%H:%M:%S) $TAG ended: $(systemctl --user show -p Result --value "$U" 2>/dev/null) | $(tail -1 results/part206/$TAG.log 2>/dev/null)"
done
echo STEP2-ALL-DONE
