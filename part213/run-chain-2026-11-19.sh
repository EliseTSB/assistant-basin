#!/bin/bash
# Part 213 chain (Journal Claude, 2026-11-19): pilot on the fixed runner, then Stage 1 (sft, dpo, instruct, FR) with the
# pilot's evidence, then Stage 2. Stops at the first failing step. Log ends with CHAIN-END.
cd /home/elisepc/green/worktrees/ab-neon-review
PY="/home/elisepc/neuron-probe/.venv/bin/python -I -B"
export HF_HUB_OFFLINE=1
step() { echo "$(date '+%F %T') START $*"; $PY part213/run213.py "$@"; rc=$?; echo "$(date '+%F %T') END $1 rc=$rc"; return $rc; }
step pilot --out results/part213/pilot \
 && step stage1 --out results/part213/full --pilot results/part213/pilot --with-fr \
 && step stage2 --out results/part213/full
rc=$?
echo "$(date '+%F %T') CHAIN-END rc=$rc"
