#!/bin/bash
# Kill the Qwen CPU scorer if Windows free RAM falls under 2500 MB, so it can never push an atlas collector under the 2000 MB guard. Claude 2026-10-22.
TP=/mnt/c/Windows/System32/typeperf.exe
while pgrep -f "sentence_acts.py --sentences" >/dev/null; do
  free=$($TP '\Memory\Available MBytes' -sc 1 2>/dev/null | sed -n 3p | cut -d, -f2 | tr -d '"' | cut -d. -f1)
  if [ -n "$free" ] && [ "$free" -lt 2500 ]; then
    pkill -f "sentence_acts.py --sentences"; echo "killed scorer at free=${free}MB $(date +%H:%M)" >> /home/elisepc/assistant-basin/results/part58/ram-guard.log; exit 0
  fi
  sleep 10
done
