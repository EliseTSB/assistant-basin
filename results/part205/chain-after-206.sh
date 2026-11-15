#!/bin/bash
# Written 2026-11-13 by Claude (Opus 5.5). Runs one model job at a time on this host:
#   1. waits for Part 206's OLMo causal step (unit part206-causal-instruct-080244) to end;
#   2. Part 206's Qwen causal step on the A770 (arc-steps-when-ram.sh: 11 GB free Windows RAM, 2 attempts), copied back;
#   3. Part 205's six fallback stages (Amendment 2 item 2, A0 gave FALLBACK 07:34) on the CPU, one at a time.
# Never a CPU model job beside an Arc load: WSL's memory comes out of Windows' available RAM.
set -u
U206=part206-causal-instruct-080244
while systemctl --user is-active --quiet "$U206"; do sleep 30; done
echo "$(date +%H:%M:%S) part206 olmo causal ended: $(systemctl --user show -p Result --value "$U206" 2>/dev/null) | $(tail -1 /home/elisepc/green/worktrees/ab-part206/results/part206/causal-instruct.log 2>/dev/null)"
[ -f /home/elisepc/green/worktrees/ab-part206/results/part206/instruct/causal.json ] || { echo "PART206 OLMO CAUSAL WROTE NO causal.json"; exit 2; }

cd /home/elisepc/green/worktrees/ab-part206 || exit 1
export ARC_TREE=/home/elisepc/green/worktrees/ab-part206 LAUNCH_ARC_CONFIG=/home/elisepc/green/worktrees/ab-part67/tools/local.json
QC=E:/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
/home/elisepc/green/Journal/scripts/research/arc-steps-when-ram.sh 206 11 2 \
  "-I -B part206/causal206.py --stage qwen-chat --device xpu --model-path $QC --out results/part206 --batch 2" \
  -- part206/common206.py part206/causal206.py part206/config206.json part206/prompts206.json neuron_report.py part66/part66.py \
     results/part206/qwen-chat/summary.json results/part206/qwen-chat/moments.npz results/part206/qwen-chat/meta.json \
  || { echo "PART206 QWEN CAUSAL FAILED"; exit 3; }
J=/mnt/c/Users/alexa/neuron-probe-gpu/results/part206/qwen-chat
cp "$J/causal.json" results/part206/qwen-chat/causal.json || { echo "COPY FAILED: qwen-chat causal.json"; exit 5; }
echo "$(date +%H:%M:%S) part206 qwen causal copied back"
echo PART206-STEP4-DONE

cd /home/elisepc/green/worktrees/ab-part201 || exit 1
unset ARC_TREE LAUNCH_ARC_CONFIG
PY='env HF_HOME=/mnt/e/neuron-probe/hf HF_HUB_OFFLINE=1 /home/elisepc/neuron-probe/.venv/bin/python -I -B part205/run205.py'
OVR="Registered design (a91f73bd; Amendments 1 and 2 public) predates the launcher's Registered/Total/Pilot line formats; family O runs on the CPU in float32, so no A770 kernel check applies. Fallback stage per Amendment 2 item 2 (A0 2026-11-13: gap 0.0191 < 0.194)."
for spec in "pretrain-3.0T|pretrain" "midtrain-26B|midtrain26" "midtrain-end|midtrainend" "sft|sft" "dpo|dpo" "instruct|instruct"; do
  IFS='|' read -r STAGE TAG <<< "$spec"
  OUT=$(python3 -I -B tools/launch_arc.py 205 --cpu "$PY --stage $STAGE --out results/part205 --check-pool --arms controls" --mem 9G --tag "$TAG" --override "$OVR" 2>&1)
  U=$(echo "$OUT" | grep -oE "unit part205-$TAG-[0-9]+" | head -1 | cut -d' ' -f2)
  echo "$(date +%H:%M:%S) $STAGE $TAG unit=$U"
  [ -z "$U" ] && { echo "$OUT" | tail -5; echo "PART205 STAGE FAILED TO START: $TAG"; exit 4; }
  while systemctl --user is-active --quiet "$U"; do sleep 30; done
  R=$(systemctl --user show -p Result --value "$U" 2>/dev/null)
  echo "$(date +%H:%M:%S) $TAG ended: $R | $(tail -1 results/part205/$TAG.log 2>/dev/null)"
  [ "$R" = "success" ] || { echo "PART205 STAGE FAILED: $TAG"; exit 6; }
done
echo PART205-FALLBACK-ALL-DONE
