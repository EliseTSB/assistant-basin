#!/bin/bash
# Written by tools/launch_arc.py.
set -o pipefail
cd "$(dirname "$0")/../.." || exit 9
/home/elisepc/neuron-probe/.venv/bin/python -I -B part205/run205.py --stage instruct --stage-only screen --out results/part205/a0
rc=$?
echo "rc=$rc $(date +%H:%M:%S)"
exit $rc
