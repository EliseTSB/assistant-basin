#!/bin/bash
# Written 2026-11-14 by Claude (Opus 5.5). The first score210 run (05:12, MemoryMax=1500M) was OOM-killed after 53 CPU
# minutes. This waits for Part 205's family Q chain (chain-205q-060957.scope) to end, so the scorer never shares
# Windows RAM with an A770 load or the Windows CPU reference, then reruns it with a 6 GB cap and records peak memory.
set -u
cd /home/elisepc/green/worktrees/ab-part210 || exit 1
while systemctl --user is-active --quiet chain-205q-060957.scope; do sleep 60; done
echo "$(date +%H:%M:%S) chain-205q ended; scoring"
systemd-run --user --scope --collect -p MemoryMax=6G --unit=score210-r2-$(date +%H%M%S) \
  /usr/bin/time -v /home/elisepc/neuron-probe/.venv/bin/python -I -B part210/score210.py --rows results/part210/answers.jsonl \
  --load results/part210/load.json --norms results/part210/norms.json --replicate results/part210/replicate.jsonl \
  --out results/part210 > results/part210/score210-output.txt 2> results/part210/score210-stderr.txt
echo "$(date +%H:%M:%S) score210 rc=$?"
grep -E "Maximum resident|Elapsed" results/part210/score210-stderr.txt
echo PART210-SCORE-DONE
