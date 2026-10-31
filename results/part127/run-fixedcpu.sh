#!/bin/bash
# Written by tools/launch_arc.py.
set -o pipefail
cd "$(dirname "$0")/../.." || exit 9
/home/elisepc/neuron-probe/.venv/bin/python -I -B part127/run127.py --stage main --model /mnt/e/neuron-probe/hf/hub/models--allenai--OLMo-2-0425-1B-Instruct/snapshots/48d788eca847d4d7548f375ad03d3c9312f6139e --out results/part127/main --fixed-only --fresh-fixed --device cpu
rc=$?
echo "rc=$rc $(date +%H:%M:%S)"
exit $rc
