#!/bin/bash
# Written by tools/launch_arc.py.
set -o pipefail
cd "$(dirname "$0")/../.." || exit 9
/home/elisepc/neuron-probe/.venv/bin/python -I -B part206/causal206.py --stage instruct --out results/part206 --batch 8
rc=$?
echo "rc=$rc $(date +%H:%M:%S)"
exit $rc
