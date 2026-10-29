#!/usr/bin/env bash
# chain105.sh -- Part 105 after the GPU job (PART105-DESIGN.md): wait for the job, copy the run, build Part 99's label
# items, cut them to Part 101's read set (Part 92's kinds), label in two passes with Part 92's blind reader, then score.
# Usage: part105/chain105.sh <marker file touched just before the scheduled task was started> <scratch dir>
set -euo pipefail
export SSL_CERT_FILE=$HOME/.mitmproxy/ca-bundle-combined.crt   # the labeller's HTTPS goes through the mitm proxy
G=/mnt/c/Users/alexa/neuron-probe-gpu
cd "$(dirname "$0")/.."
marker=$1 scratch=$2
until [ -f "$G/job.exit" ] && [ "$G/job.exit" -nt "$marker" ]; do sleep 20; done
rc=$(tr -d '\r\n' < "$G/job.exit")
echo "$(date +%H:%M:%S) job rc=$rc; $(tail -1 "$G/progress.log" | tr -d '\r')"
[ "$rc" = 0 ] || exit 1
mkdir -p results/part105/label-input
cp "$G/results/part105/attnEXT_all.jsonl" "$G/results/part105/attnEXT_all.meta.json" results/part105/
python3 -I part99/label_items99.py results/part105/attnEXT_all.jsonl "$scratch/full105-attnEXT_all.jsonl"
python3 -I part101/read101.py filter "$scratch/full105-attnEXT_all.jsonl" results/part105/label-input/attnEXT_all.jsonl
for pass in 1 2; do
  python3 -I part92/label92.py --answers results/part105/label-input/attnEXT_all.jsonl \
    --neurons results/part92/step2/neurons.json --out results/part105/labels-attnEXT_all.jsonl
  echo "$(date +%H:%M:%S) label pass $pass done"
done
python3 -I part105/score105.py > results/part105/score-run.log
echo "$(date +%H:%M:%S) scored: $(cat results/part105/score-run.log)"
