#!/bin/bash
# Written by tools/launch_arc.py.
set -o pipefail
cd "$(dirname "$0")/../.." || exit 9
export OPENCODE_GO_API_KEY="$(grep -m1 "^OPENCODE_GO_API_KEY=" /home/elisepc/green/DiscordBot/.env | cut -d= -f2- | tr -d "\"")" && L="/home/elisepc/neuron-probe/.venv/bin/python -I -B part119b/label119b.py" && $L --gate --reader ds --out results/part119b/known-ds.jsonl && $L --gate --reader luna --out results/part119b/known-luna.jsonl && $L --reader ds --out results/part119b/labels-ds.jsonl && $L --reader luna --out results/part119b/labels-luna.jsonl
rc=$?
echo "rc=$rc $(date +%H:%M:%S)"
exit $rc
