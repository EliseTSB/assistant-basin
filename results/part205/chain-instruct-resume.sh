#!/bin/bash
# Written 2026-11-14 about 20:15 CT by Claude (Opus 5.5). Part 205 family O's Instruct stage was OOM-killed at 00:54 on
# 11-14 (9G cap, 9.0G peak plus 6.6G swap) with 117 of 168 records and no sets.json; chain-after-206.sh read
# `systemctl show -p Result` on the collected unit, got the default "success", and printed PART205-FALLBACK-ALL-DONE.
# This resumes it (run205 skips records it already has; the folder's meta.json says arms=controls, the same mode) at a
# 12G cap, judged by launch_arc --block (65e85a92: the wrapper's own rc= line), then runs steps.txt step 4, the score.
# Waits for Windows to show >= 10 GB free before each launch: a WSL job comes out of Windows' available RAM.
# One brace group ending in exit: bash reads the whole script before running it (detached-script-guard.mjs).
{
set -u
cd /home/elisepc/green/worktrees/ab-part201 || exit 1
unset ARC_TREE LAUNCH_ARC_CONFIG
PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
wait_ram() {
  while true; do
    echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1
    F=$($PS -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)" 2>/dev/null | tr -d '\r')
    awk -v f="$F" 'BEGIN{exit !(f>=10)}' && { echo "$(date +%H:%M:%S) $1: Windows free=$F GB, launching"; return 0; }
    sleep 60
  done
}
PY='env HF_HOME=/mnt/e/neuron-probe/hf HF_HUB_OFFLINE=1 /home/elisepc/neuron-probe/.venv/bin/python -I -B part205/run205.py'
OVR="Registered design (a91f73bd; Amendments 1 and 2 public) predates the launcher's Registered/Total/Pilot line formats; family O runs on the CPU in float32, so no A770 kernel check applies. Fallback stage per Amendment 2 item 2. Resume of the Instruct stage OOM-killed at 00:54 on 2026-11-14 (117 of 168 records), same mode, 12G cap."
echo "$(date +%H:%M:%S) start; instruct records before: $(wc -l < results/part205/instruct/records.jsonl)"
wait_ram instruct
python3 -I -B tools/launch_arc.py 205 --cpu "$PY --stage instruct --out results/part205 --check-pool --arms controls" \
  --mem 12G --tag instruct --override "$OVR" --block 2>&1 | grep -E "STARTED|ENDED|REFUSED|NOT LAUNCHED|FAILED"
RC=${PIPESTATUS[0]}
N=$(wc -l < results/part205/instruct/records.jsonl)
echo "$(date +%H:%M:%S) instruct rc=$RC records=$N sets=$([ -f results/part205/instruct/sets.json ] && echo yes || echo no)"
[ "$RC" = 0 ] && [ "$N" = 168 ] && [ -f results/part205/instruct/sets.json ] || { echo "PART205 INSTRUCT RESUME FAILED"; exit 6; }
wait_ram score
OVR2="Registered design (a91f73bd; Amendments 1-5 public) predates the launcher's line formats; steps.txt step 4, the score (reads only, no model)."
python3 -I -B tools/launch_arc.py 205 --cpu "/home/elisepc/neuron-probe/.venv/bin/python -I -B part205/score205.py --raw results/part205" \
  --mem 4G --tag score --override "$OVR2" --block 2>&1 | grep -E "STARTED|ENDED|REFUSED|NOT LAUNCHED|FAILED"
RC=${PIPESTATUS[0]}
echo "$(date +%H:%M:%S) score rc=$RC"
[ "$RC" = 0 ] || { echo "PART205 SCORE FAILED"; exit 7; }
echo PART205-SCORED
exit 0
}
