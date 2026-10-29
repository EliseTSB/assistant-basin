#!/usr/bin/env bash
# chain104a.sh -- Part 104 step 1 after the GPU job (PART104-DESIGN.md): wait for the job to end, copy the battery
# answers, label them in two passes with Part 92's blind reader (the second pass retries failed calls only), then apply
# Part 92's kind rules and the gate with part104/kinds104.py.
# Usage: part104/chain104a.sh <marker file touched just before the scheduled task was started>
set -euo pipefail
# The labeller's HTTPS goes through the mitm proxy, whose certificate is only in this bundle (chain101.sh lost a pass
# without it on 2026-10-29).
export SSL_CERT_FILE=$HOME/.mitmproxy/ca-bundle-combined.crt
G=/mnt/c/Users/alexa/neuron-probe-gpu
cd "$(dirname "$0")/.."
marker=$1
until [ -f "$G/job.exit" ] && [ "$G/job.exit" -nt "$marker" ]; do sleep 20; done
rc=$(tr -d '\r\n' < "$G/job.exit")
echo "$(date +%H:%M:%S) job rc=$rc; $(tail -1 "$G/progress.log" | tr -d '\r')"
[ "$rc" = 0 ] || exit 1
mkdir -p results/part104/step1
cp "$G/results/part104/step1/instruct_chat.jsonl" results/part104/step1/
for pass in 1 2; do
  python3 -I part92/label92.py --answers results/part104/step1/instruct_chat.jsonl \
    --neurons results/part104/neurons.json --out results/part104/step1/labels-instruct_chat.jsonl
  echo "$(date +%H:%M:%S) label pass $pass done"
done
python3 -I part104/kinds104.py results/part104/step1/labels-instruct_chat.jsonl results/part104/neurons.json \
  results/part104/kinds.json
echo "$(date +%H:%M:%S) kinds done"
