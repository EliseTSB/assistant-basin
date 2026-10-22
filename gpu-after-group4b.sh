#!/bin/bash
# Round 2 (Claude 2026-10-22 10:30): the job runner's step 1 (group 1) was killed by the RAM guard at 10:07 after my own 3.4 GB scorer ran beside two collectors.
# One collector at a time: wait for the hand-started group-4 collector to end, wait for Windows free RAM above 4 GB, then run the job runner (groups 1-3).
G=/mnt/c/Users/alexa/neuron-probe-gpu; PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe; TP=/mnt/c/Windows/System32/typeperf.exe
LOG=/home/elisepc/assistant-basin/gpu-after-group4.log
while $PS -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'atlas_collect' }) { exit 0 } else { exit 1 }" >/dev/null 2>&1; do sleep 20; done
echo "group4 process ended $(date +%H:%M)" >> $LOG
while true; do
  free=$($TP '\Memory\Available MBytes' -sc 1 2>/dev/null | sed -n 3p | cut -d, -f2 | tr -d '"' | cut -d. -f1)
  [ -n "$free" ] && [ "$free" -gt 4000 ] && break
  sleep 30
done
echo "free RAM ${free} MB at $(date +%H:%M)" >> $LOG
rm -f $G/job.exit $G/job.log $G/progress.log $G/step-*.log
/mnt/c/Windows/System32/schtasks.exe /run /tn NeuronProbeGPU >> $LOG 2>&1
echo "job started $(date +%H:%M)" >> $LOG
