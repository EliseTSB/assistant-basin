#!/usr/bin/env bash
# chain108.sh -- Part 108 after the A770 job (PART108-DESIGN.md): wait for the job, copy the seven runs, build and cut
# the label items, label in two passes with Part 92's blind reader, then score.
# Usage: part108/chain108.sh <marker file touched just before the scheduled task was started> <scratch dir>
set -euo pipefail
export SSL_CERT_FILE=$HOME/.mitmproxy/ca-bundle-combined.crt   # the labeller's HTTPS goes through the mitm proxy
G=/mnt/c/Users/alexa/neuron-probe-gpu
cd "$(dirname "$0")/.."
marker=$1 scratch=$2
until [ -f "$G/job.exit" ] && [ "$G/job.exit" -nt "$marker" ]; do sleep 20; done
rc=$(tr -d '\r\n' < "$G/job.exit")
echo "$(date +%H:%M:%S) job rc=$rc; $(tail -1 "$G/progress.log" | tr -d '\r')"
[ "$rc" = 0 ] || exit 1
mkdir -p results/part108/label-input
for t in S D I FR XR FP XP; do
  cp "$G/results/part108/$t.jsonl" "$G/results/part108/$t.meta.json" results/part108/
  python3 -I part99/label_items99.py "results/part108/$t.jsonl" "$scratch/full108-$t.jsonl"
  python3 -I part108/items108.py "$scratch/full108-$t.jsonl" "results/part108/label-input/$t.jsonl"
done
for pass in 1 2; do
  for t in S D I FR XR FP XP; do
    python3 -I part92/label92.py --answers "results/part108/label-input/$t.jsonl" \
      --neurons results/part108/neurons108.json --out "results/part108/labels-$t.jsonl"
  done
  echo "$(date +%H:%M:%S) label pass $pass done"
done
python3 -I part108/score108.py > results/part108/score-run.log
echo "$(date +%H:%M:%S) scored: $(cat results/part108/score-run.log)"
