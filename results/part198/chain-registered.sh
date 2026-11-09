#!/bin/bash
# Part 198 registered run chain (public main 82cb94a, registered 2026-11-09 00:23 CT). Every arm, all records, into
# out/ (run_mode registered). One process at a time; each runner appends per record and resumes at the next record.
# Launched by tools/launch_arc.py --tag registered, whose wrapper is run-registered.sh (attempt 1 at 00:26 named this
# script run-registered.sh, the launcher overwrote it with a wrapper that ran itself, and nothing reached run198.py:
# attempt1-wrapper-self-loop.log).
cd /home/elisepc/green/worktrees/ab-part198/results/part198 || exit 9
P=/home/elisepc/edit-detect/.venv/bin/python
mkdir -p out
for c in "armA --method ROME" "armA --method MEMIT" "armA --method AlphaEdit" "armA-mass --method MEMIT" "armA-mass --method AlphaEdit" "armB" "armB-seq" "armC"; do
  t0=$(date +%s)
  nice -n 10 $P -I run198.py $c > "out/log-${c// /_}.txt" 2>&1
  rc=$?
  echo "RUN-STEP $c rc=$rc secs=$(( $(date +%s) - t0 )) at $(date '+%H:%M:%S')"
done
echo "RUN-DONE at $(date '+%H:%M:%S')"
