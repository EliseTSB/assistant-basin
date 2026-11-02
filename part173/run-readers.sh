#!/usr/bin/env bash
cd /home/elisepc/green/worktrees/ab-part67 || exit 9
PY=/home/elisepc/neuron-probe/.venv/bin/python
echo "start $(date +%H:%M:%S)"
$PY -I -B part173/label173.py --answers results/part173/answers.jsonl --out-dir results/part173/labels --reader ds | tail -c 600; rc=${PIPESTATUS[0]}; echo "ds_rc=$rc $(date +%H:%M:%S)"; [ "$rc" = 0 ] || exit $rc
$PY -I -B part173/label173.py --answers results/part173/answers.jsonl --out-dir results/part173/labels --reader haiku | tail -c 600; rc=${PIPESTATUS[0]}; echo "haiku_rc=$rc $(date +%H:%M:%S)"; [ "$rc" = 0 ] || exit $rc
$PY -I -B part173/score173.py --answers results/part173/answers.jsonl --gates results/part173/gates.json --labels-dir results/part173/labels --out results/part173 | tail -c 400; rc=${PIPESTATUS[0]}; echo "score_rc=$rc $(date +%H:%M:%S)"
echo "ALL DONE rc=$rc"
