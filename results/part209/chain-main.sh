#!/bin/bash
# Part 209 main run (steps.txt 2 and 3; Amendment 1: R2 batch 32, max_new_tokens 96). Arc steps one at a time,
# then copy the Qwen results back, then the OLMo CPU stages one at a time (never beside an Arc load).
WT=/home/elisepc/green/worktrees/ab-part203
cd "$WT" || exit 1
export ARC_TREE=$WT LAUNCH_ARC_CONFIG=/home/elisepc/green/worktrees/ab-part67/tools/local.json
COPY="part209/run209.py part209/config.json part209/prompts209.json part209/reference209.json part209/tokens.json neuron_report.py"
/home/elisepc/green/Journal/scripts/research/arc-steps-when-ram.sh 209 11 2 \
  "-I -B part209/run209.py --stage qwen-base --out results/part209 --device xpu" \
  "-I -B part209/run209.py --stage qwen-chat --out results/part209 --device xpu" \
  "-I -B part209/run209.py --stage qwen-chat --out results/part209 --device xpu --r2 --batch 32 --max-new-tokens 96" \
  -- $COPY || { echo "ARC STEPS FAILED"; exit 3; }
J=/mnt/c/Users/alexa/neuron-probe-gpu/results/part209
for f in "$J"/*; do b=$(basename "$f"); case "$b" in pilot|smoke|smoke-d2) continue;; esac; cp -r "$f" results/part209/; done
echo "$(date +%H:%M:%S) qwen results copied back: $(ls results/part209 | tr '\n' ' ')"
PY='env HF_HOME=/mnt/e/neuron-probe/hf HF_HUB_OFFLINE=1 /home/elisepc/neuron-probe/.venv/bin/python -I -B part209/run209.py'
for spec in "pretrain-3.0T|pretrain|" "midtrain-26B|midtrain26|" "midtrain-end|midtrainend|" "sft|sft|" "dpo|dpo|" "instruct|instruct|" "instruct|instruct-r2|--r2 --batch 32 --max-new-tokens 96"; do
  IFS='|' read -r STAGE TAG EXTRA <<< "$spec"
  OUT=$(python3 -I -B tools/launch_arc.py 209 --cpu "$PY --stage $STAGE --out results/part209 $EXTRA" --mem 8G --tag "$TAG" 2>&1)
  U=$(echo "$OUT" | grep -oE "unit part209-$TAG-[0-9]+" | head -1 | cut -d' ' -f2)
  echo "$(date +%H:%M:%S) $STAGE $TAG unit=$U"
  [ -z "$U" ] && { echo "$OUT" | tail -5; echo "CPU STAGE FAILED TO START: $TAG"; exit 4; }
  while systemctl --user is-active --quiet "$U"; do sleep 30; done
  echo "$(date +%H:%M:%S) $TAG ended: $(systemctl --user show -p Result --value "$U" 2>/dev/null) | $(tail -1 results/part209/$TAG.log)"
done
echo ALL-DONE
