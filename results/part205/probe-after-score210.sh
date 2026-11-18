#!/bin/bash
# Written 2026-11-14 by Claude (Opus 5.5). Waits for score210 (a WSL job with a 6 GB cap) to end, so the 4B load never
# shares Windows RAM with it, then runs probe_freeze205.py once on the A770 and copies its JSON back.
{
set -u
cd /home/elisepc/green/worktrees/ab-part201 || exit 1
while systemctl --user list-units --no-legend 'score210-r2-*' | grep -q running; do sleep 60; done
echo "$(date +%H:%M:%S) score210 ended; probing"
export ARC_TREE=/home/elisepc/green/worktrees/ab-part201 LAUNCH_ARC_CONFIG=/home/elisepc/green/worktrees/ab-part67/tools/local.json
export ARC_OVERRIDE="Diagnostic probe, not a registered stage: one Part 205 ranking pass on qwen-chat with the current code (twice) and with the weights frozen, after both qwen-chat attempts died with Level Zero UR_RESULT_ERROR_UNKNOWN. Prints timing, peak memory and the difference between passes only."
/home/elisepc/green/Journal/scripts/research/arc-steps-when-ram.sh 205 11 1 "-I -B results/part205/probe_freeze205.py" \
  -- results/part205/probe_freeze205.py part205/run205.py part205/common205.py part205/score205.py part205/config.json part205/prompts205.json neuron_report.py
echo "$(date +%H:%M:%S) probe step ended rc=$?"
cp /mnt/c/Users/<user>/neuron-probe-gpu/results/part205/probe_freeze205.json results/part205/ && echo "copied probe_freeze205.json"
echo PROBE-FREEZE-DONE
exit 0
}
