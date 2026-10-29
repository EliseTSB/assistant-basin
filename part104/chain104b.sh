#!/usr/bin/env bash
# chain104b.sh -- Part 104 step 2 after the GPU job (PART104-DESIGN.md): wait for the job (part104/steps104b.txt) to end,
# copy the four runs, build Part 99's label items and cut them to the read set with Part 104's kinds, label in two
# passes with Part 92's blind reader (the second pass retries failed calls only), then score.
# Usage: part104/chain104b.sh <marker file touched just before the scheduled task was started> <scratch dir>
set -euo pipefail
# The labeller's HTTPS goes through the mitm proxy, whose certificate is only in this bundle.
export SSL_CERT_FILE=$HOME/.mitmproxy/ca-bundle-combined.crt
G=/mnt/c/Users/alexa/neuron-probe-gpu
cd "$(dirname "$0")/.."
marker=$1 scratch=$2
until [ -f "$G/job.exit" ] && [ "$G/job.exit" -nt "$marker" ]; do sleep 20; done
rc=$(tr -d '\r\n' < "$G/job.exit")
echo "$(date +%H:%M:%S) job rc=$rc; $(tail -1 "$G/progress.log" | tr -d '\r')"
[ "$rc" = 0 ] || exit 1
mkdir -p results/part104/label-input
cp "$G/results/part104/pushes104.json" results/part104/
runs="inst_none attnSFT_all attnEXT_all attnDPO_all"
for t in $runs; do
  cp "$G/results/part104/$t.jsonl" "$G/results/part104/$t.meta.json" results/part104/
  python3 -I part99/label_items99.py "results/part104/$t.jsonl" "$scratch/full104-$t.jsonl"
  python3 -I part104/score104.py filter "$scratch/full104-$t.jsonl" "results/part104/label-input/$t.jsonl"
done
for pass in 1 2; do
  for t in $runs; do
    python3 -I part92/label92.py --answers "results/part104/label-input/$t.jsonl" \
      --neurons results/part104/neurons.json --out "results/part104/labels-$t.jsonl"
  done
  echo "$(date +%H:%M:%S) label pass $pass done"
done
python3 -I part104/score104.py score > results/part104/score-run.log
echo "$(date +%H:%M:%S) scored: $(cat results/part104/score-run.log)"
