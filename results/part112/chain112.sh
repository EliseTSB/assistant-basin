#!/bin/bash
# Chain after Part 112's Arc run (written by Claude, Opus 5.5, 2026-10-29): when the job logs its exit line, start
# Part 114 on the Arc, then copy Part 112's answers into the repo, label them blind (two passes) and score them.
G=/mnt/c/Users/alexa/neuron-probe-gpu
R=/home/elisepc/green/worktrees/ab-part67
N=$(cat $R/results/part112/chain-baseline.txt)
until tail -n +$((N+1)) $G/progress.log | grep -q "rc="; do sleep 30; done
tail -n +$((N+1)) $G/progress.log | tr -d '\r' > $R/results/part112/run-end.log
if grep -q "rc=0" $R/results/part112/run-end.log; then
  printf '%s\r\n' '-I -B part114\qneuron114.py --out p114' > $G/steps.txt
  sleep 30
  /mnt/c/Windows/System32/schtasks.exe /run /tn NeuronProbeGPU > $R/results/part112/p114-launch.log 2>&1
  cp $G/p112/run/answers.jsonl $R/results/part112/answers.jsonl
  cd $R && export SSL_CERT_FILE=/home/elisepc/.mitmproxy/ca-bundle-combined.crt
  /home/elisepc/neuron-probe/.venv/bin/python -I part112/score112.py --prepare
  python3 part92/label92.py --answers results/part112/items.jsonl --neurons results/part112/personas.json --out results/part112/labels.jsonl
  python3 part92/label92.py --answers results/part112/items.jsonl --neurons results/part112/personas.json --out results/part112/labels.jsonl
  /home/elisepc/neuron-probe/.venv/bin/python -I part112/score112.py > results/part112/score.log 2>&1
  echo "CHAIN DONE $(date +%H:%M)" >> results/part112/run-end.log
else
  echo "Part 112 did not end rc=0; nothing launched" >> $R/results/part112/run-end.log
fi
