#!/bin/bash
# Written by tools/launch_arc.py.
set -o pipefail
cd "$(dirname "$0")/../.." || exit 9
/home/elisepc/neuron-probe/.venv/bin/python -I -B part209/score209.py --raw results/part209
rc=$?
echo "rc=$rc $(date +%H:%M:%S)"
exit $rc
