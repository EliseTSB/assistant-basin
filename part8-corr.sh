#!/bin/bash
# part8-corr.sh — the correlation step of part8-score.sh alone, for prediction 8.7 (Claude, 2026-10-16 07:2x, on
# Elise's word "You can run it!"). Claude Code stopped it at 04:5x for low memory while the session was idle; run here
# as its own unit (now with corr_lean.py: corr.py was OOM-killed at a 7 GB cap at 07:25) with a memory cap so it is not one of the session's background shells. corr.py loads ~2 GB per file.
cd /home/elisepc/neuron-probe
I=out/Qwen__Qwen2.5-Coder-7B-Instruct.npz
for X in out/Qwen__Qwen2.5-Coder-7B.npz out/SWE-bench__SWE-agent-LM-7B.npz out/OpenHands__openhands-lm-7b-v0.1.npz; do
  .venv/bin/python -I corr_lean.py $I $X
done > out/corr-qwencoder.txt 2>&1
echo "corr rc=$?" >> out/corr-qwencoder.txt
