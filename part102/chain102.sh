#!/usr/bin/env bash
# chain102.sh -- Part 102 after the GPU job (PART102-DESIGN.md): wait for the job to end, copy the three runs, build
# Part 99's label items and cut them to the read set, label in two passes with Part 92's blind reader (the second pass
# retries failed calls only), then score.
# Usage: part102/chain102.sh <marker file touched just before Start-ScheduledTask> <scratch dir for full item files>
set -euo pipefail
# The labeller's HTTPS goes through the mitm proxy, whose certificate is only in this bundle; without it every call
# fails CERTIFICATE_VERIFY_FAILED (it did on the first run, 2026-10-29 08:07 to 08:40).
export SSL_CERT_FILE=$HOME/.mitmproxy/ca-bundle-combined.crt
G=/mnt/c/Users/alexa/neuron-probe-gpu
cd "$(dirname "$0")/.."
marker=$1 scratch=$2
until [ -f "$G/job.exit" ] && [ "$G/job.exit" -nt "$marker" ]; do sleep 20; done
rc=$(tr -d '\r\n' < "$G/job.exit")
echo "$(date +%H:%M:%S) job rc=$rc"
[ "$rc" = 0 ] || exit 1
mkdir -p results/part102/label-input
runs="attnSFT_L0-3 attnDPO_L0-3 attnB_L1-3"
for t in $runs; do
  cp "$G/results/part102/$t.jsonl" "$G/results/part102/$t.meta.json" results/part102/
  python3 -I part99/label_items99.py "results/part102/$t.jsonl" "$scratch/full-$t.jsonl"
  python3 -I part101/read101.py filter "$scratch/full-$t.jsonl" "results/part102/label-input/$t.jsonl"
done
for pass in 1 2; do
  for t in $runs; do
    python3 -I part92/label92.py --answers "results/part102/label-input/$t.jsonl" \
      --neurons results/part92/step2/neurons.json --out "results/part102/labels-$t.jsonl"
  done
  echo "$(date +%H:%M:%S) label pass $pass done"
done
python3 -I part102/score102.py > results/part102/score-run.log
echo "$(date +%H:%M:%S) scored"
python3 -I -c "import json; s=json.load(open('results/part102/score102.json')); print(s['predictions']); print({t: (r['slot_who'], r['slot_who_coherent'], r['concept_any'], r['items_still_missing']) for t, r in s['runs'].items()}); print('')"
