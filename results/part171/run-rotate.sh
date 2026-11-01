#!/bin/bash
# Written by tools/launch_arc.py.
set -o pipefail
cd "$(dirname "$0")/../.." || exit 9
/home/elisepc/neuron-probe/.venv/bin/python -I -B part171/rotate171.py --out-dir /mnt/c/Users/alexa/neuron-probe-gpu/results/part171 --max-memory-gb 6
rc=$?
echo "rc=$rc $(date +%H:%M:%S)"
exit $rc
