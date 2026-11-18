#!/bin/bash
# Written 2026-11-13 by Claude (Opus 5.5). Part 210 (registered 2026-11-13 10:13, public main 343b0918) on the A770,
# only after chain-205-after-206 (Part 206's causal steps, then Part 205's six CPU stages) has ended for any reason:
# never a CPU model job beside an Arc load. ONE attempt: a pilot stop (exit 3) is a HOLD until reviewed (steps.txt).
set -u
while systemctl --user is-active --quiet chain-205-after-206.scope; do sleep 60; done
echo "$(date +%H:%M:%S) chain-205-after-206 ended: $(tail -1 /home/elisepc/green/worktrees/ab-part201/results/part205/chain-after-206.log)"
cd /home/elisepc/green/worktrees/ab-part210 || exit 1
export ARC_TREE=/home/elisepc/green/worktrees/ab-part210 LAUNCH_ARC_CONFIG=/home/elisepc/green/worktrees/ab-part67/tools/local.json
/home/elisepc/green/Journal/scripts/research/arc-steps-when-ram.sh 210 11 1 \
  "-I -B part210/run210.py --device xpu --model E:/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --out results/part210/answers.jsonl" \
  -- tools/launch_arc.py tools/margin_recorder.py part210/run210.py part210/steps.txt part201/__init__.py part201/run201.py part201/ROWS-SCHEMA.md part201/LOAD-SCHEMA.md part199/__init__.py part199/run199.py part165/run165.py part164/run164.py part163/run163.py part160/run160.py part155/run155.py part153/battery153.py part148/battery148.py part148/score148.py part86/common86.py part86/triage86.py neuron_report.py results/part112/medians.json part164/units164.json part165/score165.py results/part165/answers.jsonl results/part165/norms.json \
  || { echo "PART210 RUN FAILED OR HELD"; exit 3; }
J=/mnt/c/Users/<user>/neuron-probe-gpu/results/part210
for f in answers.jsonl load.json norms.json replicate.jsonl; do
  cp "$J/$f" results/part210/$f || { echo "COPY FAILED: $f"; exit 5; }
done
echo "$(date +%H:%M:%S) part210 results copied back"
echo PART210-RUN-DONE
