#!/bin/bash
# Part 213 chain, second leg (Journal Claude, 2026-11-19 07:15 restart after the FR step died at 07:12 on a .bin DPO
# snapshot): Stage 1 skips sft/dpo/instruct by their completion markers and builds FR, then Stage 2. Ends CHAIN-END.
cd /home/elisepc/green/worktrees/ab-neon-review
PY="/home/elisepc/neuron-probe/.venv/bin/python -I -B"
export HF_HUB_OFFLINE=1
step() { echo "$(date '+%F %T') START $*"; $PY part213/run213.py "$@"; rc=$?; echo "$(date '+%F %T') END $1 rc=$rc"; return $rc; }
step stage1 --out results/part213/full --pilot results/part213/pilot --with-fr \
 && step stage2 --out results/part213/full
rc=$?
echo "$(date '+%F %T') CHAIN-END rc=$rc"
