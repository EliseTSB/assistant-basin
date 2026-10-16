#!/bin/bash
# part8-score.sh — score Part 8 once gpu-chain4 has collected all four Qwen2.5-Coder models (Claude, 2026-10-16).
# Identity neurons are chosen in Instruct and scored in the base and both agent models, plain and chat form, as part 5.
# CAVEAT: the frames' family A names OLMo/Ai2 for every model (data/frames.json), so for Qwen the "identity score" is an
# AI self-statement under a foreign name; it compares the four Qwen models with each other, not with OLMo's numbers.
cd /home/elisepc/neuron-probe
I=out/Qwen__Qwen2.5-Coder-7B-Instruct.npz; B=out/Qwen__Qwen2.5-Coder-7B.npz
S=out/SWE-bench__SWE-agent-LM-7B.npz; O=out/OpenHands__openhands-lm-7b-v0.1.npz
for f in plain chat; do .venv/bin/python -I analyze.py $I --form $f --score $B $S $O --json out/stages-qwencoder-$f.txt > out/stages-qwencoder-$f.log 2>&1; echo "analyze $f rc=$?"; done
for X in $B $S $O; do .venv/bin/python -I corr.py $I $X; done > out/corr-qwencoder.txt 2>&1; echo "corr rc=$?"
