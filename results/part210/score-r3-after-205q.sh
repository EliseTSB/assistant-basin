#!/bin/bash
# Written 2026-11-14 by Claude (Opus 5.5), after Amendment 3 (replicate pairing by row_id). Waits for Part 205's family
# Q chain (chain-205q-093709.scope) to end, so the scorer never shares Windows RAM with an A770 load or the Windows CPU
# reference, then reruns score210 with an 8 GB cap (the r2 run peaked at 6.0 GiB against 6 GiB) and records peak memory.
# One brace group ending in exit: bash reads the whole script before running it (detached-script-guard.mjs).
{
set -u
cd /home/elisepc/green/worktrees/ab-part210 || exit 1
while systemctl --user is-active --quiet chain-205q-093709.scope; do sleep 60; done
echo "$(date +%H:%M:%S) chain-205q ended; scoring"
mkdir -p results/part210/before-amendment3
cp results/part210/score.md results/part210/score.json results/part210/score210-output.txt results/part210/score210-stderr.txt results/part210/before-amendment3/
systemd-run --user --scope --collect -p MemoryMax=8G --unit=score210-r3-$(date +%H%M%S) \
  /usr/bin/time -v /home/elisepc/neuron-probe/.venv/bin/python -I -B part210/score210.py --rows results/part210/answers.jsonl \
  --load results/part210/load.json --norms results/part210/norms.json --replicate results/part210/replicate.jsonl \
  --out results/part210 > results/part210/score210-output.txt 2> results/part210/score210-stderr.txt
echo "$(date +%H:%M:%S) score210 rc=$?"
grep -E "Maximum resident|Elapsed" results/part210/score210-stderr.txt
echo PART210-SCORE-R3-DONE
exit 0
}
